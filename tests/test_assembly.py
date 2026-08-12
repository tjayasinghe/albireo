"""Tests for direct band assembly (``albireo.assembly``) and the custom-VJP solve stage.

The band path assembles the posterior precision per epoch from the static rebin pair
tables (``docs/math.md`` §4.2 read backwards: the *same* matrix, summed in a different
order). Its reference here is the original global comb probing (``assembly="probe"``),
which ``test_likelihood.py`` in turn pins to dense brute-force linear algebra — so the
chain reaches ground truth without ever assembling a dense matrix at this size. The two
assemblies must agree to floating-point summation order for every model variant, under
``jax.jit`` with traced velocities and LSF widths, and in reverse mode.

The second half checks the pieces the band path leans on: the analytic prior diagonals,
the block Takahashi selected inverse (against a dense inverse), the rebin pair tables
(against a dense ``R^T diag(w) R``), and the closed-form reverse pass of
``likelihood._solve_stage`` against plain autodiff through the Cholesky/solve scans.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import albireo as ab
from albireo.assembly import _prior_diagonals, band_block_tridiagonal, prior_block_tridiagonal
from albireo.forward import build_problem, rhs, weighted_data_terms, with_lsf, with_velocities
from albireo.likelihood import _pack, marginal_loglikelihood
from albireo.operators import rebin_operator, rebin_pair_tables
from albireo.priors import SmoothnessPrior
from albireo.solver import (
    block_cholesky,
    dense_from_block_tridiagonal,
    logdet,
    probe_block_tridiagonal,
    selected_inverse_blocks,
    solve_lower,
    solve_upper,
)

GRID = ab.LogGrid.from_wavelength_range(5000.0, 5040.0, dv_kms=5.5)
N_EP = 8

PRIOR1 = SmoothnessPrior(tau=[200.0], eta=[4.0])
PRIOR2 = SmoothnessPrior(tau=[200.0, 200.0], eta=[4.0, 4.0])
PRIOR3 = SmoothnessPrior(tau=[200.0, 200.0, 150.0], eta=[4.0, 4.0, 4.0])


def simulate(frame="barycentric", n_comp=2, two_inst=False):
    """A small multi-epoch dataset; the band/probe equivalence fixtures share it."""
    rng = np.random.default_rng(5)
    bjd = np.sort(rng.uniform(0.0, 14.0, N_EP))
    comps = [
        ab.synthetic_deviation_spectrum(
            GRID, n_lines=12, depth_range=(0.1, 0.6), sigma_v_range=(9.0, 18.0), seed=s
        )
        for s in range(1, n_comp + 1)
    ]
    orbit = ab.OrbitParams(period=6.31, t_peri=2.0, ecc=0.25, omega=0.7, k=(34.0, 51.0)[:n_comp])
    ell = np.array([0.65, 0.35])[:n_comp]
    ell = ell / ell.sum()
    instruments = {
        "a": ab.InstrumentSpec(wave=np.arange(5002.0, 5038.0, 0.105), sigma_v_lsf=7.0, snr=90.0)
    }
    if two_inst:
        instruments["b"] = ab.InstrumentSpec(
            wave=np.arange(5003.0, 5037.0, 0.145), sigma_v_lsf=11.0, snr=60.0
        )
    ds, truth = ab.simulate_dataset(
        GRID,
        comps,
        bjd=bjd,
        instruments=instruments,
        light_fractions=ell,
        orbit=orbit,
        frame=frame,
        seed=2,
    )
    return ds, truth, ell


# The plain SB2 problem is reused by the jit/gradient and custom-VJP tests, so it (and
# the other variants) are built once at import time rather than per test.
DS, TRUTH, ELL = simulate()
PROBLEM = build_problem(
    GRID, DS, velocities=TRUTH.velocities, light_fractions=ELL, lsf_sigma_v={"a": 7.0}
)
VEL = jnp.asarray(TRUTH.velocities)
B_NAT = PROBLEM.half_bandwidth_bound(120.0)


def _equivalence_cases():
    """(id, problem, prior) for every model variant the band path must reproduce."""
    ds_two, truth_two, ell_two = simulate(two_inst=True)
    ds_topo, truth_topo, ell_topo = simulate(frame="topocentric")
    ds_one, truth_one, _ = simulate(n_comp=1)

    rng = np.random.default_rng(11)
    ell_epoch = np.abs(rng.normal(0.6, 0.05, (2, N_EP)))
    ell_epoch = ell_epoch / ell_epoch.sum(axis=0)
    response = [np.array([0.02, -0.01]) for _ in range(N_EP)]

    return [
        ("sb2_barycentric", PROBLEM, PRIOR2),
        (
            "two_instruments",
            build_problem(
                GRID,
                ds_two,
                velocities=truth_two.velocities,
                light_fractions=ell_two,
                lsf_sigma_v={"a": 7.0, "b": 11.0},
            ),
            PRIOR2,
        ),
        (
            "per_epoch_light_response",
            build_problem(
                GRID,
                DS,
                velocities=TRUTH.velocities,
                light_fractions=ell_epoch,
                lsf_sigma_v={"a": 7.0},
                response_coeffs=response,
            ),
            PRIOR2,
        ),
        (
            "topocentric_telluric",
            build_problem(
                GRID,
                ds_topo,
                velocities=truth_topo.velocities,
                light_fractions=ell_topo,
                lsf_sigma_v={"a": 7.0},
                telluric=True,
            ),
            PRIOR3,
        ),
        (
            "single_component",
            build_problem(
                GRID,
                ds_one,
                velocities=truth_one.velocities,
                light_fractions=np.array([1.0]),
                lsf_sigma_v={"a": 7.0},
            ),
            PRIOR1,
        ),
    ]


CASES = _equivalence_cases()


# ---------------------------------------------------------------------------
# Band assembly reproduces comb probing exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("problem", "prior"), [c[1:] for c in CASES], ids=[c[0] for c in CASES])
def test_band_matches_probe(problem, prior):
    band = marginal_loglikelihood(problem, prior, assembly="band", validate=True)
    probe = marginal_loglikelihood(problem, prior, assembly="probe")

    scale = abs(float(probe.log_likelihood)) + 1.0
    rel = abs(float(band.log_likelihood) - float(probe.log_likelihood)) / scale
    assert rel < 1e-11, f"log-likelihood relative difference {rel:.2e}"

    d_diff = float(jnp.max(jnp.abs(band.d_hat - probe.d_hat)))
    assert d_diff < 1e-8, f"max |d_hat difference| {d_diff:.2e}"


def test_band_jitted_traced_theta():
    """Traced velocities + traced LSF width: same value and same gradients as probing."""

    def ll(problem, vel, sigma, mode):
        p = with_lsf(with_velocities(problem, vel), {"a": sigma})
        return marginal_loglikelihood(p, PRIOR2, half_bandwidth=B_NAT, assembly=mode).log_likelihood

    v_band = float(jax.jit(lambda p, v, s: ll(p, v, s, "band"))(PROBLEM, VEL, 6.5))
    v_probe = float(jax.jit(lambda p, v, s: ll(p, v, s, "probe"))(PROBLEM, VEL, 6.5))
    assert abs(v_band - v_probe) / (abs(v_probe) + 1.0) < 1e-11

    g_band = jax.jit(jax.grad(lambda v, s: ll(PROBLEM, v, s, "band"), argnums=(0, 1)))(VEL, 6.5)
    g_probe = jax.jit(jax.grad(lambda v, s: ll(PROBLEM, v, s, "probe"), argnums=(0, 1)))(VEL, 6.5)

    d_vel = float(jnp.max(jnp.abs(g_band[0] - g_probe[0])) / (jnp.max(jnp.abs(g_probe[0])) + 1.0))
    d_sigma = float(jnp.abs(g_band[1] - g_probe[1]) / (jnp.abs(g_probe[1]) + 1.0))
    assert d_vel < 1e-9, f"velocity gradient relative difference {d_vel:.2e}"
    assert d_sigma < 1e-9, f"LSF gradient relative difference {d_sigma:.2e}"
    assert float(jnp.max(jnp.abs(g_probe[0]))) > 0.0  # the comparison is not against zero


# ---------------------------------------------------------------------------
# Closed-form reverse pass of the solve stage
# ---------------------------------------------------------------------------


def test_solve_stage_vjp_matches_autodiff():
    """``_solve_stage``'s custom VJP vs plain autodiff through the same computation."""
    nc, n_pix = PROBLEM.n_components, GRID.n
    n = nc * n_pix

    def ll_custom(v, log_tau, log_eta):
        prior = SmoothnessPrior(jnp.exp(log_tau), jnp.exp(log_eta))
        res = marginal_loglikelihood(with_velocities(PROBLEM, v), prior, half_bandwidth=B_NAT)
        return res.log_likelihood, res.d_hat

    def ll_ref(v, log_tau, log_eta):
        """The same arithmetic with the Cholesky/solve tail written out inline, so
        reverse mode differentiates through the scans instead of the custom rule."""
        prior = SmoothnessPrior(jnp.exp(log_tau), jnp.exp(log_eta))
        problem = with_velocities(PROBLEM, v)
        chol = block_cholesky(band_block_tridiagonal(problem, prior, B_NAT))
        chol_prior = block_cholesky(prior_block_tridiagonal(prior, n_pix, nc, max(2 * nc, 64)))
        n_pad = chol.num_blocks * chol.block_size
        b_pad = jnp.pad(_pack(rhs(problem), nc, n_pix), (0, n_pad - n))
        y = solve_lower(chol, b_pad)
        d_pad = solve_upper(chol, y)
        zwz, logw, n_good = weighted_data_terms(problem)
        logp = (
            -0.5 * (zwz - jnp.sum(y * y))
            - 0.5 * logdet(chol)
            + 0.5 * logdet(chol_prior)
            + 0.5 * logw
            - 0.5 * n_good * jnp.log(2.0 * jnp.pi)
        )
        return logp, d_pad[:n].reshape(n_pix, nc).T

    log_tau = jnp.log(jnp.asarray([200.0, 200.0]))
    log_eta = jnp.log(jnp.asarray([4.0, 4.0]))

    # The custom VJP must not perturb the primal at all: same operations, same order.
    assert float(ll_custom(VEL, log_tau, log_eta)[0]) == float(ll_ref(VEL, log_tau, log_eta)[0])

    g_custom = jax.grad(lambda v, t, e: ll_custom(v, t, e)[0], argnums=(0, 1, 2))(
        VEL, log_tau, log_eta
    )
    g_ref = jax.grad(lambda v, t, e: ll_ref(v, t, e)[0], argnums=(0, 1, 2))(VEL, log_tau, log_eta)
    for name, a, b in zip(("velocities", "log_tau", "log_eta"), g_custom, g_ref, strict=True):
        assert float(jnp.max(jnp.abs(b))) > 0.0, name
        rel = float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))
        assert rel < 1e-9, f"{name} gradient relative difference {rel:.2e}"

    # Cotangents arriving through d_hat (the u-solve branch of the reverse rule).
    gd_custom = jax.grad(lambda v: jnp.sum(jnp.sin(ll_custom(v, log_tau, log_eta)[1])))(VEL)
    gd_ref = jax.grad(lambda v: jnp.sum(jnp.sin(ll_ref(v, log_tau, log_eta)[1])))(VEL)
    rel = float(jnp.max(jnp.abs(gd_custom - gd_ref)) / jnp.max(jnp.abs(gd_ref)))
    assert rel < 1e-9, f"d_hat-cotangent gradient relative difference {rel:.2e}"

    # Independent spot check: central differences on one velocity entry.
    eps = 1e-4
    step = jnp.zeros_like(VEL).at[0, 3].set(eps)
    fd = (
        ll_custom(VEL + step, log_tau, log_eta)[0] - ll_custom(VEL - step, log_tau, log_eta)[0]
    ) / (2 * eps)
    assert abs(float(fd - g_custom[0][0, 3])) / (abs(float(fd)) + 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Building blocks: prior band, selected inverse, rebin pair tables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_pix", [5, 8, 30])
def test_prior_diagonals_vs_dense(n_pix):
    """Analytic ``tau D2^T D2 + eta I`` diagonals vs :meth:`SmoothnessPrior.dense`."""
    prior = SmoothnessPrior(tau=[2.5, 0.75], eta=[1e-3, 7.0])
    d0, d1, d2 = _prior_diagonals(prior, n_pix)
    dense = prior.dense(n_pix)
    for i in range(prior.n_components):
        block = dense[i * n_pix : (i + 1) * n_pix, i * n_pix : (i + 1) * n_pix]
        np.testing.assert_allclose(np.asarray(d0[i]), np.diag(block), rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(d1[i]), np.diag(block, 1), rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(d2[i]), np.diag(block, 2), rtol=0, atol=1e-12)
        # nothing outside half-bandwidth 2
        np.testing.assert_allclose(np.diag(block, 3), 0.0, rtol=0, atol=1e-12)


def test_prior_block_tridiagonal_vs_dense():
    """The prior-only block-tridiagonal, densified, is the dense prior re-interleaved."""
    n_pix, nc = 12, 2
    prior = SmoothnessPrior(tau=[3.0, 0.4], eta=[0.5, 2.0])
    bt = prior_block_tridiagonal(prior, n_pix, nc, 8)
    assert bt.n == nc * n_pix

    # component-major (i * n_pix + q) -> pixel-major (q * nc + i), the stacking used
    # throughout the likelihood
    perm = np.array([i * n_pix + q for q in range(n_pix) for i in range(nc)])
    expected = prior.dense(n_pix)[np.ix_(perm, perm)]
    np.testing.assert_allclose(dense_from_block_tridiagonal(bt), expected, rtol=0, atol=1e-12)


def _random_banded_spd(n, p, rng):
    """Symmetric banded matrix made SPD by diagonal dominance (as in test_solver)."""
    m = np.zeros((n, n))
    for o in range(1, p + 1):
        vals = rng.standard_normal(n - o)
        m[np.arange(n - o), np.arange(o, n)] = vals
        m[np.arange(o, n), np.arange(n - o)] = vals
    m[np.diag_indices(n)] = np.abs(m).sum(axis=1) + 1.0 + rng.uniform(0, 1, n)
    return m


@pytest.mark.parametrize("block_size", [24, 8, 7, 6])
def test_selected_inverse_blocks_vs_dense(block_size):
    """Block Takahashi returns exactly the block-tridiagonal part of the inverse."""
    n, p = 24, 4
    m = _random_banded_spd(n, p, np.random.default_rng(7))
    bt = probe_block_tridiagonal(lambda v: jnp.asarray(m) @ v, n, p, block_size)
    chol = block_cholesky(bt)
    s_diag, s_sub = selected_inverse_blocks(chol)

    k, b = bt.num_blocks, bt.block_size
    assert s_diag.shape == (k, b, b)
    assert s_sub.shape == (k - 1, b, b)

    # the pad coordinates are identity-decoupled, so the padded inverse is blkdiag(M^-1, I)
    padded = np.eye(k * b)
    padded[:n, :n] = m
    inv = np.linalg.inv(padded)
    for i in range(k):
        np.testing.assert_allclose(
            np.asarray(s_diag[i]),
            inv[i * b : (i + 1) * b, i * b : (i + 1) * b],
            rtol=1e-10,
            atol=0,
        )
    for i in range(k - 1):
        np.testing.assert_allclose(
            np.asarray(s_sub[i]),
            inv[(i + 1) * b : (i + 2) * b, i * b : (i + 1) * b],
            rtol=1e-10,
            atol=0,
        )


def test_rebin_pair_tables():
    """One segment-sum over the pair tables reproduces ``R^T diag(w) R`` exactly."""
    x_in = np.arange(5000.0, 5001.0, 0.01)
    x_out = np.arange(5000.03, 5000.99, 0.05)
    reb = rebin_operator(x_in=x_in, x_out=x_out)
    pair_val, pair_sid, pair_row, h = rebin_pair_tables(reb)
    n_in, n_out = reb.n_in, reb.n_out
    assert h > 2  # rows genuinely overlap, so the band is wider than the diagonal

    rng = np.random.default_rng(3)
    w = rng.uniform(0.2, 3.0, n_out)
    upper = np.asarray(
        jax.ops.segment_sum(
            pair_val * jnp.asarray(w)[pair_row], pair_sid, num_segments=n_in * h
        ).reshape(n_in, h)
    )

    got = np.zeros((n_in, n_in))
    for o in range(h):
        idx = np.arange(n_in - o)
        got[idx, idx + o] = upper[idx, o]
        got[idx + o, idx] = upper[idx, o]
        if o > 0:  # entries that would fall off the grid must be empty segments
            np.testing.assert_allclose(upper[n_in - o :, o], 0.0, rtol=0, atol=0)

    r_dense = np.zeros((n_out, n_in))
    r_dense[np.asarray(reb.rows), np.asarray(reb.cols)] = np.asarray(reb.vals)
    expected = r_dense.T @ np.diag(w) @ r_dense
    assert np.count_nonzero(expected) > 500  # the comparison is not against a zero matrix
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-13)


def test_second_order_reverse_matches_plain_autodiff():
    """``jacrev(jacrev(...))`` through the custom VJP equals the plain-autodiff Hessian.

    Regression for the forward-rule re-entry defect: with ``_solve_stage_fwd`` calling
    the custom function itself, the outer reverse pass re-entered the custom boundary,
    hit the dropped Cholesky cotangent, and lost the chol-mediated second-order terms
    (8e-3 relative, measured). The forward rule now recomputes its primal inline, and
    Hessians agree to machine precision. Forward mode stays unsupported by
    construction (`custom_vjp`), so ``laplace_inverse_mass`` uses rev-over-rev.
    """
    nc, n_pix = PROBLEM.n_components, GRID.n
    n = nc * n_pix
    v0 = VEL[:, 0]  # wiggle epoch-0 velocities only: a 2x2 Hessian keeps this fast

    def f_custom(v0_):
        v = VEL.at[:, 0].set(v0_)
        res = marginal_loglikelihood(with_velocities(PROBLEM, v), PRIOR2, half_bandwidth=B_NAT)
        return res.log_likelihood

    def f_ref(v0_):
        v = VEL.at[:, 0].set(v0_)
        problem = with_velocities(PROBLEM, v)
        chol = block_cholesky(band_block_tridiagonal(problem, PRIOR2, B_NAT))
        chol_prior = block_cholesky(prior_block_tridiagonal(PRIOR2, n_pix, nc, max(2 * nc, 64)))
        n_pad = chol.num_blocks * chol.block_size
        b_pad = jnp.pad(_pack(rhs(problem), nc, n_pix), (0, n_pad - n))
        y = solve_lower(chol, b_pad)
        zwz, logw, n_good = weighted_data_terms(problem)
        return (
            -0.5 * (zwz - jnp.sum(y * y))
            - 0.5 * logdet(chol)
            + 0.5 * logdet(chol_prior)
            + 0.5 * logw
            - 0.5 * n_good * jnp.log(2.0 * jnp.pi)
        )

    h_custom = np.asarray(jax.jacrev(jax.jacrev(f_custom))(v0))
    h_ref = np.asarray(jax.jacrev(jax.jacrev(f_ref))(v0))
    assert np.max(np.abs(h_ref)) > 0.0
    np.testing.assert_allclose(h_custom, h_ref, rtol=1e-12, atol=0)
    np.testing.assert_allclose(h_custom, h_custom.T, rtol=1e-12, atol=0)  # symmetric

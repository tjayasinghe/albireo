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
from albireo.assembly import (
    _prior_diagonals,
    band_block_tridiagonal,
    prior_block_tridiagonal,
    prior_logdet,
)
from albireo.forward import (
    build_problem,
    normal_matvec,
    rhs,
    weighted_data_terms,
    with_lsf,
    with_velocities,
)
from albireo.likelihood import _pack, marginal_loglikelihood, spectra_std
from albireo.operators import rebin_operator, rebin_pair_tables
from albireo.priors import SmoothnessPrior
from albireo.solver import (
    block_cholesky,
    dense_from_block_tridiagonal,
    logdet,
    probe_block_tridiagonal,
    selected_inverse_blocks,
    selected_inverse_cotangent,
    solve_lower,
    solve_upper,
)

GRID = ab.LogGrid.from_wavelength_range(5000.0, 5040.0, dv_kms=5.5)
N_EP = 8

PRIOR1 = SmoothnessPrior(tau=[200.0], eta=[4.0])
PRIOR2 = SmoothnessPrior(tau=[200.0, 200.0], eta=[4.0, 4.0])
PRIOR3 = SmoothnessPrior(tau=[200.0, 200.0, 150.0], eta=[4.0, 4.0, 4.0])


def simulate(frame="barycentric", n_comp=2, two_inst=False, native=(5002.0, 5038.0)):
    """A small multi-epoch dataset; the band/probe equivalence fixtures share it.

    ``native`` is the wavelength span of the instrument grid. The default leaves a
    margin inside the model grid; ``edge_covered`` below closes it, which is the only
    configuration that exercises the model-grid boundary of the band assembly (with a
    margin the weights vanish there and boundary defects are multiplied by zero).
    """
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
        "a": ab.InstrumentSpec(
            wave=np.arange(native[0], native[1], 0.105), sigma_v_lsf=7.0, snr=90.0
        )
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
        # Without this every epoch is labelled with the first instrument, so the
        # second one is built and never used and the "two instruments" fixture
        # degenerates into a duplicate of the single-instrument one.
        epoch_instruments=(["a", "b"] * (N_EP // 2)) if two_inst else None,
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
    ds_edge, truth_edge, ell_edge = simulate(native=(5000.4, 5039.6))

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
        (
            "edge_covered",
            build_problem(
                GRID,
                ds_edge,
                velocities=truth_edge.velocities,
                light_fractions=ell_edge,
                lsf_sigma_v={"a": 7.0},
            ),
            PRIOR2,
        ),
    ]


CASES = _equivalence_cases()


def test_equivalence_cases_have_the_topologies_they_claim():
    """Guard the fixtures themselves — a degenerate one tests nothing, silently.

    ``two_instruments`` in particular is only meaningful if the epochs actually carry
    both labels: ``simulate_dataset`` defaults every epoch to the first instrument, so
    omitting ``epoch_instruments`` builds the second instrument's operators and never
    uses them, leaving a numerically identical copy of the plain SB2 case. Groups carry
    per-instrument rebin support and kernel radii, and the band layout is sized from a
    single global bandwidth, so multi-group really is a distinct code path.
    """
    by_id = {c[0]: c[1] for c in CASES}
    assert len(by_id["two_instruments"].groups) == 2
    two = by_id["two_instruments"]
    supports = {g.instrument: (g.row_support, g.kernel.shape[-1]) for g in two.groups}
    assert supports["a"] != supports["b"], f"groups are not actually distinct: {supports}"
    assert by_id["sb2_barycentric"].n_components == 2
    assert by_id["single_component"].n_components == 1
    assert by_id["topocentric_telluric"].n_components == 3
    assert by_id["topocentric_telluric"].telluric


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


@pytest.mark.parametrize(("problem", "prior"), [c[1:] for c in CASES], ids=[c[0] for c in CASES])
def test_band_matches_probe_entrywise(problem, prior):
    """Every *entry* of the assembled precision, not just the scalars it feeds.

    ``test_band_matches_probe`` compares a log-determinant and a solve, both of which
    average over the matrix; a defect confined to a few dozen rows at the model-grid
    boundary moves them by ~1e-7 relative and can slip under a loose threshold. This
    compares the dense matrices directly, which is only affordable at fixture size.
    """
    nc, n_pix = problem.n_components, problem.grid.n
    n = nc * n_pix
    b_nat = max(problem.natural_half_bandwidth, prior.half_bandwidth)

    def full_matvec(v):
        d = jnp.asarray(v).reshape(n_pix, nc).T
        return _pack(normal_matvec(problem, d) + prior.apply(d), nc, n_pix)

    band = dense_from_block_tridiagonal(band_block_tridiagonal(problem, prior, b_nat))
    probe = dense_from_block_tridiagonal(
        probe_block_tridiagonal(full_matvec, n, nc * b_nat + nc - 1)
    )
    scale = float(np.abs(probe).max())
    err = float(np.abs(band - probe).max()) / scale
    assert err < 1e-12, f"max entrywise relative difference {err:.2e}"


def test_band_matches_probe_with_model_grid_inside_the_data():
    """The worst boundary case: zero margin between data coverage and grid edge.

    Choosing a model grid *narrower* than the observed range is the documented way to
    fit a sub-region, and it drives the data-coverage margin to zero. The band image of
    ``G`` is nonzero for a strip of ``kernel_radius`` columns past each grid edge (the
    LSF smears in-grid mass outward), and only the *row* index of the T-sandwich is
    zero-filled, so before the column mask this configuration was wrong by tens of nats
    — and asymmetric, which the symmetric probe reference is not. The trigger is
    ``coverage margin < kernel radius``, so it is also reachable at a fixed grid simply
    by fitting a wider LSF.
    """
    ds, truth, ell = simulate(native=(5001.0, 5039.0))
    inner = ab.LogGrid.from_wavelength_range(5004.0, 5036.0, dv_kms=5.5)
    # build_problem warns about exactly this configuration (weighted native pixels off the
    # grid); the point of the test is that the assembly is right anyway.
    with pytest.warns(RuntimeWarning, match="outside the model grid"):
        problem = build_problem(
            inner, ds, velocities=truth.velocities, light_fractions=ell, lsf_sigma_v={"a": 7.0}
        )
    nc, n_pix = problem.n_components, inner.n
    n = nc * n_pix
    b_nat = max(problem.natural_half_bandwidth, PRIOR2.half_bandwidth)

    def full_matvec(v):
        d = jnp.asarray(v).reshape(n_pix, nc).T
        return _pack(normal_matvec(problem, d) + PRIOR2.apply(d), nc, n_pix)

    band = dense_from_block_tridiagonal(band_block_tridiagonal(problem, PRIOR2, b_nat))
    probe = dense_from_block_tridiagonal(
        probe_block_tridiagonal(full_matvec, n, nc * b_nat + nc - 1)
    )
    scale = float(np.abs(probe).max())
    assert float(np.abs(band - probe).max()) / scale < 1e-12
    # only the column side ever leaked, so asymmetry is the sharpest discriminator
    assert float(np.abs(band - band.T).max()) / scale < 1e-12

    ll_band = marginal_loglikelihood(problem, PRIOR2, assembly="band").log_likelihood
    ll_probe = marginal_loglikelihood(problem, PRIOR2, assembly="probe").log_likelihood
    assert abs(float(ll_band) - float(ll_probe)) < 1e-6, "log-likelihood differs in nats"


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
        n_pad = chol.num_blocks * chol.block_size
        b_pad = jnp.pad(_pack(rhs(problem), nc, n_pix), (0, n_pad - n))
        y = solve_lower(chol, b_pad)
        d_pad = solve_upper(chol, y)
        zwz, logw, n_good = weighted_data_terms(problem)
        logp = (
            -0.5 * (zwz - jnp.sum(y * y))
            - 0.5 * logdet(chol)
            + 0.5 * prior_logdet(prior, n_pix)
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


def test_factor_derived_gradients_are_not_silently_zero():
    """Gradients through the Cholesky factor must be real, not the custom rule's zero.

    ``_solve_stage``'s reverse rule cannot carry a cotangent on the factorization, so
    if ``MarginalResult`` handed back *that* factor, every gradient of a factor-derived
    quantity — the posterior spectral uncertainties, the draws — would come out
    identically zero with no error. The factor is therefore rebuilt outside the custom
    boundary; this checks the resulting gradient against central differences.
    """

    def total_std(v):
        res = marginal_loglikelihood(with_velocities(PROBLEM, v), PRIOR2, half_bandwidth=B_NAT)
        return jnp.sum(spectra_std(res))

    g = jax.grad(total_std)(VEL)
    assert jnp.all(jnp.isfinite(g))
    assert float(jnp.max(jnp.abs(g))) > 0.0, "factor-derived gradient is identically zero"

    eps = 1e-3
    step = jnp.zeros_like(VEL).at[0, 2].set(eps)
    fd = (total_std(VEL + step) - total_std(VEL - step)) / (2 * eps)
    assert abs(float(fd - g[0, 2])) / (abs(float(fd)) + 1e-6) < 1e-4, f"{float(fd)} vs {g[0, 2]}"


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


@pytest.mark.parametrize("n_pix", [5, 9, 64, 257])
def test_prior_logdet_matches_blocked_cholesky(n_pix):
    """The scalar pentadiagonal recursion reproduces the blocked factorization."""
    prior = SmoothnessPrior(tau=[2.5, 0.75, 40.0], eta=[1e-3, 7.0, 0.2])
    nc = prior.n_components
    got = float(prior_logdet(prior, n_pix))
    want = float(logdet(block_cholesky(prior_block_tridiagonal(prior, n_pix, nc, max(2 * nc, 16)))))
    assert abs(got - want) / (abs(want) + 1.0) < 1e-12, f"{got!r} vs {want!r}"

    # ... and the dense determinant, so both routes are pinned to ground truth.
    sign, dense_ld = np.linalg.slogdet(prior.dense(n_pix))
    assert sign > 0
    assert abs(got - float(dense_ld)) / (abs(float(dense_ld)) + 1.0) < 1e-10


def test_prior_logdet_differentiable_in_hyperparameters():
    """tau/eta gradients agree with the blocked route (ML-II differentiates this)."""
    n_pix = 40

    def via_scan(log_tau, log_eta):
        return prior_logdet(SmoothnessPrior(jnp.exp(log_tau), jnp.exp(log_eta)), n_pix)

    def via_blocks(log_tau, log_eta):
        prior = SmoothnessPrior(jnp.exp(log_tau), jnp.exp(log_eta))
        return logdet(block_cholesky(prior_block_tridiagonal(prior, n_pix, 2, 16)))

    lt = jnp.log(jnp.asarray([3.0, 0.4]))
    le = jnp.log(jnp.asarray([0.5, 2.0]))
    for a, b in zip(
        jax.grad(via_scan, argnums=(0, 1))(lt, le),
        jax.grad(via_blocks, argnums=(0, 1))(lt, le),
        strict=True,
    ):
        assert float(jnp.max(jnp.abs(b))) > 0.0
        assert float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b))) < 1e-10


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


@pytest.mark.parametrize("block_size", [24, 8, 7, 6])
def test_selected_inverse_cotangent_matches_unfused(block_size):
    """Fusing the cotangent into the Takahashi sweep changes nothing numerically.

    The unfused form materializes ``Sigma`` and then contracts it; the fused form
    forms each block as the recursion produces it. Same arithmetic, 2K-1 fewer live
    blocks — this pins them together, including the k == 1 single-block branch.
    """
    n, p = 24, 4
    rng = np.random.default_rng(19)
    m = _random_banded_spd(n, p, rng)
    bt = probe_block_tridiagonal(lambda v: jnp.asarray(m) @ v, n, p, block_size)
    chol = block_cholesky(bt)
    k, b = bt.num_blocks, bt.block_size

    d = jnp.asarray(rng.standard_normal((k, b)))
    u = jnp.asarray(rng.standard_normal((k, b)))
    g_ld, g_quad = 0.37, -1.9

    s_diag, s_sub = selected_inverse_blocks(chol)

    def outer(x, y):
        return x[:, :, None] * y[:, None, :]

    want_diag = g_ld * s_diag - g_quad * outer(d, d) - outer(u, d)
    want_sub = (
        2.0 * g_ld * s_sub
        - 2.0 * g_quad * outer(d[1:], d[:-1])
        - (outer(u[1:], d[:-1]) + outer(d[1:], u[:-1]))
    )
    got_diag, got_sub = selected_inverse_cotangent(chol, d, u, g_ld, g_quad)

    assert got_diag.shape == want_diag.shape
    assert got_sub.shape == want_sub.shape
    np.testing.assert_allclose(np.asarray(got_diag), np.asarray(want_diag), rtol=1e-11, atol=0)
    if k > 1:
        np.testing.assert_allclose(np.asarray(got_sub), np.asarray(want_sub), rtol=1e-11, atol=0)


@pytest.mark.parametrize("chunk", [1, 3, N_EP])
def test_band_epoch_chunk_invariance(chunk):
    """Batching the velocity-independent G pre-pass must not move any number.

    ``epoch_chunk`` only decides how many epochs' worth of G is live at once (and
    whether the backward recomputes it), so the assembled matrix, the likelihood and
    the gradient are all invariant — including when the batch size does not divide
    the epoch count and the last batch is zero-weight padded.
    """

    def ll(v, epoch_chunk):
        problem = with_velocities(PROBLEM, v)
        bt = band_block_tridiagonal(problem, PRIOR2, B_NAT, epoch_chunk=epoch_chunk)
        return logdet(block_cholesky(bt)), bt

    ref_ld, ref_bt = ll(VEL, N_EP)
    got_ld, got_bt = ll(VEL, chunk)
    np.testing.assert_allclose(np.asarray(got_bt.diag), np.asarray(ref_bt.diag), rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        np.asarray(got_bt.lower), np.asarray(ref_bt.lower), rtol=0, atol=1e-9
    )
    assert abs(float(got_ld - ref_ld)) / (abs(float(ref_ld)) + 1.0) < 1e-12

    g_ref = jax.grad(lambda v: ll(v, N_EP)[0])(VEL)
    g_got = jax.grad(lambda v: ll(v, chunk)[0])(VEL)
    assert float(jnp.max(jnp.abs(g_ref))) > 0.0
    rel = float(jnp.max(jnp.abs(g_got - g_ref)) / jnp.max(jnp.abs(g_ref)))
    assert rel < 1e-10, f"chunk={chunk} gradient relative difference {rel:.2e}"


def test_band_rejects_too_small_half_bandwidth():
    """A bandwidth below the kernel+rebin floor is refused, not silently mis-assembled.

    The per-epoch block is written into the band with a clamped ``dynamic_update_slice``,
    so a bandwidth that cannot hold one block would land the window at the wrong offset.
    """
    g = PROBLEM.groups[0]
    floor = g.row_support + 2 * ((g.kernel.shape[-1] - 1) // 2)  # zero-shift containment
    with pytest.raises(ValueError, match="half_bandwidth too small"):
        band_block_tridiagonal(PROBLEM, PRIOR2, floor - 1)
    band_block_tridiagonal(PROBLEM, PRIOR2, floor)  # the floor itself fits


def test_deleted_native_samples_warn_about_bandwidth():
    """Deleting samples (rather than masking them) inflates the solver bandwidth.

    Edges sit at midpoints, so removing a block of native samples makes the two
    bracketing pixels each absorb half the gap. ``row_support`` is a max, so those two
    pixels set the bandwidth for the entire run — cost grows quadratically. Real spectra
    hit this constantly (telluric windows, order gaps), hence a warning that names the
    pixel and the remedy.
    """
    wave = np.arange(5002.0, 5038.0, 0.105)
    gap = (wave > 5015.0) & (wave < 5019.0)  # a "removed" telluric window
    ds, truth = ab.simulate_dataset(
        GRID,
        [ab.synthetic_deviation_spectrum(GRID, n_lines=10, seed=s) for s in (1, 2)],
        bjd=np.linspace(0.0, 12.0, 4),
        instruments={"a": ab.InstrumentSpec(wave=wave[~gap], sigma_v_lsf=7.0, snr=90.0)},
        light_fractions=np.array([0.6, 0.4]),
        orbit=ab.OrbitParams(period=6.31, t_peri=2.0, ecc=0.25, omega=0.7, k=(34.0, 51.0)),
        frame="barycentric",
        seed=2,
    )
    with pytest.warns(RuntimeWarning, match="native pixels span far more model pixels"):
        gappy = build_problem(
            GRID,
            ds,
            velocities=truth.velocities,
            light_fractions=np.array([0.6, 0.4]),
            lsf_sigma_v={"a": 7.0},
        )
    # and the warning is about something real: the bandwidth actually did blow up
    assert gappy.groups[0].row_support > 4 * PROBLEM.groups[0].row_support


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
        n_pad = chol.num_blocks * chol.block_size
        b_pad = jnp.pad(_pack(rhs(problem), nc, n_pix), (0, n_pad - n))
        y = solve_lower(chol, b_pad)
        zwz, logw, n_good = weighted_data_terms(problem)
        return (
            -0.5 * (zwz - jnp.sum(y * y))
            - 0.5 * logdet(chol)
            + 0.5 * prior_logdet(PRIOR2, n_pix)
            + 0.5 * logw
            - 0.5 * n_good * jnp.log(2.0 * jnp.pi)
        )

    h_custom = np.asarray(jax.jacrev(jax.jacrev(f_custom))(v0))
    h_ref = np.asarray(jax.jacrev(jax.jacrev(f_ref))(v0))
    assert np.max(np.abs(h_ref)) > 0.0
    np.testing.assert_allclose(h_custom, h_ref, rtol=1e-12, atol=0)
    np.testing.assert_allclose(h_custom, h_custom.T, rtol=1e-12, atol=0)  # symmetric

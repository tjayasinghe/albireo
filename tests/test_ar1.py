"""Tests for the AR(1) correlated-noise model (docs/design.md D34, math.md §1.4a).

D31 measured why this exists: a rescaled diagonal noise model whitens the residual
*scale* while the *structure* — here, adjacent-pixel correlation from pipeline
resampling — keeps selecting a biased optimum. What is pinned here: the closed-form
tridiagonal chain precision must equal a dense reference built independently from the
chain correlation matrix (masked gaps included, where links carry ``phi**gap``); the
whole marginal must match dense brute force under the correlated covariance; ``phi = 0``
must reproduce the diagonal model; the chain whitener must remove the lag-1
autocorrelation a diagonal whitener provably cannot; and the marginal must *recover* an
injected ``phi`` and noise-scale error jointly with the orbit.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

import albireo as ab
from albireo.assembly import band_block_tridiagonal
from albireo.data import Dataset, EpochData
from albireo.forward import build_problem, data_residual_zscores, with_ar1, with_jitter
from albireo.inference import MarginalOrbitModel, run_map
from albireo.kepler import t_peri_from_t_conj
from albireo.likelihood import marginal_loglikelihood
from albireo.priors import SmoothnessPrior
from albireo.simulate import InstrumentSpec, OrbitParams, simulate_dataset
from albireo.simulate import synthetic_deviation_spectrum as synth
from tests.test_likelihood import dense_design_matrix

SMALL_GRID = ab.LogGrid.from_wavelength_range(5000.0, 5003.0, dv_kms=3.0)
SMALL_VEL = np.array([[9.0, -12.0, 3.0], [-14.0, 18.0, -5.0]])
SMALL_KW = dict(velocities=SMALL_VEL, light_fractions=[0.6, 0.4], lsf_sigma_v={"A": 4.0, "B": 9.0})


def small_problem(cosmic_fraction=0.02):
    """Three epochs, two instruments, response, and enough cosmics to make gaps."""
    comps = [synth(SMALL_GRID, n_lines=6, seed=s, margin=0.15) for s in (1, 2)]
    ds, truth = simulate_dataset(
        SMALL_GRID,
        comps,
        bjd=np.arange(3.0),
        velocities=SMALL_VEL,
        light_fractions=[0.6, 0.4],
        instruments={
            "A": InstrumentSpec(wave=np.arange(5000.5, 5002.4, 0.055), sigma_v_lsf=4.0, snr=50.0),
            "B": InstrumentSpec(wave=np.arange(5000.6, 5002.3, 0.09), sigma_v_lsf=9.0, snr=30.0),
        },
        epoch_instruments=["A", "B", "A"],
        v_bary=np.array([8.0, -20.0, 3.0]),
        response_order=1,
        response_amplitude=0.03,
        cosmic_fraction=cosmic_fraction,
        seed=9,
    )
    problem = build_problem(SMALL_GRID, ds, response_coeffs=list(truth.response_coeffs), **SMALL_KW)
    prior = SmoothnessPrior(tau=[2.0, 0.7], eta=[1e-3, 2e-3])
    return ds, truth, problem, prior


# ------------------------------------------------------------------ dense reference


def _dense_epoch_precision(w_row, gap_row, phi, alpha):
    """Dense ``W_e`` built independently: chain correlation matrix, then inverted.

    The chain correlation between observed pixels is the product of the stored link
    correlations along the path (``phi**gap`` per link, 0 where the cap restarted the
    chain) — a Markov chain's correlation factorizes over links, which is the fact the
    production code's tridiagonal closed form relies on. Building R densely and
    inverting it with LAPACK shares nothing with that closed form.
    """
    n = w_row.size
    good = w_row > 0
    gs = np.flatnonzero(good)
    m = gs.size
    rho_row = np.where(gap_row > 0, np.sign(phi) ** gap_row * np.abs(phi) ** gap_row, 0.0)
    link = np.zeros(m)
    for k in range(1, m):
        link[k] = rho_row[gs[k]]
    corr = np.eye(m)
    for k in range(m):
        acc = 1.0
        for ell in range(k + 1, m):
            acc *= link[ell]
            if acc == 0.0:
                break
            corr[k, ell] = corr[ell, k] = acc
    w_full = np.zeros((n, n))
    s = np.sqrt(w_row[gs])
    w_full[np.ix_(gs, gs)] = (s[:, None] * np.linalg.inv(corr) * s[None, :]) / alpha**2
    return w_full


def dense_noise_precision(problem):
    """Block-diagonal dense noise precision in the design-matrix row ordering."""
    blocks = []
    for g in problem.groups:
        for e in range(g.n_epochs):
            blocks.append(
                _dense_epoch_precision(
                    np.asarray(g.w)[e],
                    np.asarray(g.ar_gap)[e],
                    float(np.asarray(g.ar_phi)[e]),
                    float(np.asarray(g.jitter)[e]),
                )
            )
    n = sum(b.shape[0] for b in blocks)
    out = np.zeros((n, n))
    at = 0
    for b in blocks:
        out[at : at + b.shape[0], at : at + b.shape[0]] = b
        at += b.shape[0]
    return out


def dense_marginal_correlated(problem, prior):
    a = dense_design_matrix(problem)
    z = np.concatenate([np.asarray(g.z).ravel() for g in problem.groups])
    w = np.concatenate([np.asarray(g.w).ravel() for g in problem.groups])
    big_w = dense_noise_precision(problem)
    lam_p = prior.dense(problem.grid.n)
    lam = lam_p + a.T @ big_w @ a
    b = a.T @ (big_w @ z)
    x = np.linalg.solve(lam, b)
    good = w > 0
    logdet_w_good = np.linalg.slogdet(big_w[np.ix_(good.nonzero()[0], good.nonzero()[0])])[1]
    logp = (
        -0.5 * (z @ (big_w @ z) - b @ x)
        - 0.5 * np.linalg.slogdet(lam)[1]
        + 0.5 * np.linalg.slogdet(lam_p)[1]
        + 0.5 * logdet_w_good
        - 0.5 * good.sum() * np.log(2 * np.pi)
    )
    return logp, x.reshape(problem.n_components, problem.grid.n)


# ------------------------------------------------------------------ exactness


def test_marginal_matches_dense_brute_force_with_correlation_and_gaps():
    """The gold test: chain closed forms == dense LAPACK, gaps and jitter included."""
    _, _, problem, prior = small_problem()
    correlated = with_jitter(with_ar1(problem, jnp.asarray([0.4, -0.3, 0.55])), [1.3, 0.8, 1.0])
    assert any(int(np.max(np.asarray(g.ar_gap))) > 1 for g in correlated.groups), (
        "fixture must contain at least one multi-pixel gap link or the phi**gap "
        "treatment goes untested — raise cosmic_fraction"
    )
    res = marginal_loglikelihood(correlated, prior, validate=True)
    logp_dense, d_hat_dense = dense_marginal_correlated(correlated, prior)
    np.testing.assert_allclose(float(res.log_likelihood), logp_dense, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(res.d_hat), d_hat_dense, rtol=1e-7, atol=1e-9)


def test_zero_phi_reproduces_the_diagonal_model():
    _, _, problem, prior = small_problem()
    base = marginal_loglikelihood(problem, prior, assembly="probe")
    same = marginal_loglikelihood(with_ar1(problem, 0.0), prior)
    np.testing.assert_allclose(float(same.log_likelihood), float(base.log_likelihood), rtol=1e-12)
    # Since D35 this is also a cross-path comparison (the correlated problem runs the
    # band assembly at a widened bandwidth, the base runs probing), so d_hat carries
    # float-reordering noise amplified by solver conditioning — same tolerance story
    # as the response swap's d_hat check (tests/test_response.py).
    np.testing.assert_allclose(
        np.asarray(same.d_hat), np.asarray(base.d_hat), rtol=1e-8, atol=1e-11
    )


def test_band_assembly_matches_probe_and_is_the_default():
    """D35: the correlated marginal runs the band assembly; probing is the oracle."""
    _, _, problem, prior = small_problem()
    correlated = with_jitter(with_ar1(problem, jnp.asarray([0.4, -0.3, 0.55])), [1.3, 0.8, 1.0])
    band = marginal_loglikelihood(correlated, prior, assembly="band")
    probe = marginal_loglikelihood(correlated, prior, assembly="probe")
    auto = marginal_loglikelihood(correlated, prior)
    np.testing.assert_allclose(float(band.log_likelihood), float(probe.log_likelihood), rtol=1e-12)
    assert float(auto.log_likelihood) == float(band.log_likelihood)
    assert correlated.natural_half_bandwidth >= problem.natural_half_bandwidth


def test_correlated_band_epoch_chunk_invariance():
    """The batched G pre-pass must thread the link weights exactly like the hoisted one.

    ``epoch_chunk`` batching pads the trailing chunk with zero-weight epochs; the AR
    weight tuple (diagonal, link, gap table) must pad and slice together or a batched
    run silently drops link terms. Mirrors the diagonal-path invariance test in
    tests/test_assembly.py.
    """
    _, _, problem, prior = small_problem()
    correlated = with_jitter(with_ar1(problem, jnp.asarray([0.4, -0.3, 0.55])), [1.3, 0.8, 1.0])
    b_nat = correlated.natural_half_bandwidth
    ref = band_block_tridiagonal(correlated, prior, b_nat)
    got = band_block_tridiagonal(correlated, prior, b_nat, epoch_chunk=1)
    np.testing.assert_allclose(np.asarray(got.diag), np.asarray(ref.diag), rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.asarray(got.lower), np.asarray(ref.lower), rtol=0, atol=1e-9)


def test_band_and_probe_gradients_agree_in_phi():
    """The traced link weights must carry d/dphi through the band assembly exactly."""
    _, _, problem, prior = small_problem()
    bandwidth = with_ar1(problem, 0.0).natural_half_bandwidth
    phi0 = jnp.asarray([0.4, -0.3, 0.55])

    def loglike(assembly, phi):
        return marginal_loglikelihood(
            with_ar1(problem, phi), prior, half_bandwidth=bandwidth, assembly=assembly
        ).log_likelihood

    g_band = jax.grad(partial(loglike, "band"))(phi0)
    g_probe = jax.grad(partial(loglike, "probe"))(phi0)
    np.testing.assert_allclose(np.asarray(g_band), np.asarray(g_probe), rtol=1e-9)


def test_gradient_matches_finite_differences():
    _, _, problem, prior = small_problem()
    bandwidth = with_ar1(problem, 0.0).natural_half_bandwidth
    phi0 = jnp.asarray([0.4, -0.3, 0.55])

    @jax.jit
    def loglike(pb, phi):
        return marginal_loglikelihood(
            with_ar1(pb, phi), prior, half_bandwidth=bandwidth
        ).log_likelihood

    grad = jax.grad(loglike, argnums=1)(problem, phi0)
    assert bool(jnp.all(jnp.isfinite(grad)))
    for j in range(3):
        h = 1e-6
        pp, pm = np.asarray(phi0).copy(), np.asarray(phi0).copy()
        pp[j] += h
        pm[j] -= h
        fd = (
            float(loglike(problem, jnp.asarray(pp))) - float(loglike(problem, jnp.asarray(pm)))
        ) / (2 * h)
        np.testing.assert_allclose(float(grad[j]), fd, rtol=1e-5)


def test_gradient_is_finite_and_exact_at_phi_zero():
    """The gap-1 branch must carry the exact d/dphi at phi = 0 (pow's nan-grad trap)."""
    _, _, problem, prior = small_problem()
    bandwidth = with_ar1(problem, 0.0).natural_half_bandwidth

    def loglike(phi):
        return marginal_loglikelihood(
            with_ar1(problem, phi), prior, half_bandwidth=bandwidth
        ).log_likelihood

    g = float(jax.grad(loglike)(0.0))
    assert np.isfinite(g)
    h = 1e-5
    fd = (float(loglike(h)) - float(loglike(-h))) / (2 * h)
    np.testing.assert_allclose(g, fd, rtol=1e-6)


# ------------------------------------------------------------------ whitening


def test_chain_whitener_removes_lag1_autocorrelation():
    """The discriminator a diagonal model cannot see.

    AR(1) noise has unit marginal variance, so diagonally-whitened residuals still
    have sd 1 — the scale diagnostic is blind to it. The lag-1 autocorrelation is
    not: it reads ~phi under diagonal whitening and ~0 under the chain whitener.
    """
    phi_true = 0.5
    comps = [synth(SMALL_GRID, n_lines=6, seed=s, margin=0.15) for s in (1, 2)]
    ds, truth = simulate_dataset(
        SMALL_GRID,
        comps,
        bjd=np.arange(6.0),
        velocities=np.tile(SMALL_VEL, (1, 2)),
        light_fractions=[0.6, 0.4],
        instruments={
            "A": InstrumentSpec(wave=np.arange(5000.5, 5002.4, 0.055), sigma_v_lsf=4.0, snr=80.0)
        },
        epoch_instruments=["A"] * 6,
        v_bary=np.zeros(6),
        ar1_phi=phi_true,
        seed=21,
    )
    problem = build_problem(
        SMALL_GRID,
        ds,
        velocities=np.tile(SMALL_VEL, (1, 2)),
        light_fractions=[0.6, 0.4],
        lsf_sigma_v={"A": 4.0},
    )
    d_true = jnp.asarray(np.stack(truth.components))

    def lag1(x):
        return float(np.corrcoef(x[:-1], x[1:])[0, 1])

    z_diag = data_residual_zscores(problem, d_true)
    z_chain = data_residual_zscores(with_ar1(problem, phi_true), d_true)
    assert abs(np.std(z_diag) - 1.0) < 0.1, "marginal scale should look fine either way"
    assert abs(np.std(z_chain) - 1.0) < 0.1
    assert lag1(z_diag) > 0.35, "diagonal whitening must expose the correlation"
    assert abs(lag1(z_chain)) < 0.1, "the chain whitener must remove it"


# ------------------------------------------------------------------ closed loop

P_TRUE, TCONJ_TRUE, ECC_TRUE, OMEGA_TRUE = 6.31, 2.05, 0.2, 0.7
K_TRUE = np.array([30.0, 22.0])
ELL = np.array([0.62, 0.38])
GATE_GRID = ab.LogGrid.from_wavelength_range(5000.0, 5045.0, dv_kms=5.5)
N_EP = 10
PHI_TRUE = 0.45
ALPHA_TRUE = 1.5  # the supplied ivar will overstate precision by this factor


def test_closed_loop_recovers_phi_alpha_and_orbit():
    """The D34 gate: injected correlation and scale error inferred jointly with the orbit."""
    rng = np.random.default_rng(42)
    comps = [
        synth(GATE_GRID, n_lines=30, depth_range=(0.1, 0.7), sigma_v_range=(9.0, 20.0), seed=1),
        synth(GATE_GRID, n_lines=25, depth_range=(0.1, 0.7), sigma_v_range=(9.0, 20.0), seed=2),
    ]
    tperi = float(t_peri_from_t_conj(TCONJ_TRUE, period=P_TRUE, ecc=ECC_TRUE, omega=OMEGA_TRUE))
    orbit = OrbitParams(
        period=P_TRUE, t_peri=tperi, ecc=ECC_TRUE, omega=OMEGA_TRUE, k=tuple(K_TRUE)
    )
    bjd = np.sort(rng.uniform(0.0, 2.4 * P_TRUE, N_EP))
    spec = InstrumentSpec(wave=np.arange(5003.0, 5042.0, 0.11), sigma_v_lsf=7.0, snr=130.0)
    ds, _ = simulate_dataset(
        GATE_GRID,
        comps,
        bjd=bjd,
        instruments={"inst": spec},
        light_fractions=ELL,
        orbit=orbit,
        v_bary=rng.uniform(-25.0, 25.0, N_EP),
        frame="topocentric",
        ar1_phi=PHI_TRUE,
        seed=11,
    )
    # Overstate the supplied inverse variances so the jitter has something to find.
    epochs = [
        EpochData(
            wave=ep.wave,
            flux=ep.flux,
            ivar=ep.ivar * ALPHA_TRUE**2,
            bjd=ep.bjd,
            v_bary=ep.v_bary,
            instrument=ep.instrument,
        )
        for ep in ds
    ]
    ds = Dataset(epochs=tuple(epochs), frame=ds.frame)

    model = MarginalOrbitModel(
        GATE_GRID,
        ds,
        light_fractions=ELL,
        lsf_sigma_v={"inst": 7.0},
        v_rel_max_kms=float(K_TRUE.sum()) * (1 + ECC_TRUE) * 1.35,
        ar1=True,
    )
    priors = {
        "period": dist.Normal(P_TRUE + 0.001, 0.003),
        "t_conj": dist.Normal(TCONJ_TRUE + 0.005, 0.02),
        "secosw": dist.Uniform(-1.0, 1.0),
        "sesinw": dist.Uniform(-1.0, 1.0),
        "k": dist.Uniform(jnp.array([10.0, 5.0]), jnp.array([45.0, 40.0])),
        "log_tau": dist.Normal(jnp.full(2, np.log(300.0)), 3.0),
        "log_eta": dist.Normal(jnp.full(2, np.log(5.0)), 3.0),
        "log_jitter": dist.Normal(0.0, 1.0),
        "ar1_phi": dist.Uniform(-0.9, 0.9),
    }
    init = {
        "period": P_TRUE + 0.001,
        "t_conj": TCONJ_TRUE + 0.005,
        "secosw": np.sqrt(0.15) * np.cos(0.5),
        "sesinw": np.sqrt(0.15) * np.sin(0.5),
        "k": jnp.array([27.0, 25.0]),
        "log_tau": jnp.full(2, np.log(300.0)),
        "log_eta": jnp.full(2, np.log(5.0)),
        "log_jitter": jnp.asarray(0.1),
        "ar1_phi": jnp.asarray(0.1),
    }
    fit = run_map(model.model(priors), init=init, max_steps=250)

    np.testing.assert_allclose(np.asarray(fit.params["k"]), K_TRUE, rtol=1e-2)
    phi_hat = float(fit.params["ar1_phi"])
    alpha_hat = float(np.exp(fit.params["log_jitter"]))
    assert abs(phi_hat - PHI_TRUE) < 0.05, f"phi {phi_hat:.3f} vs injected {PHI_TRUE}"
    np.testing.assert_allclose(alpha_hat, ALPHA_TRUE, rtol=0.05)

    theta_hat = {
        s: jnp.asarray(fit.params[s])
        for s in (
            "period",
            "t_conj",
            "secosw",
            "sesinw",
            "k",
            "log_tau",
            "log_eta",
            "log_jitter",
            "ar1_phi",
        )
    }
    z = data_residual_zscores(model.problem_at(theta_hat), model.marginal(theta_hat).d_hat)
    # Residuals are taken about the *fitted* spectra, so their sd reads low by
    # sqrt(1 - p_eff/N) even at a perfect noise model (math.md §3.2a; measured 0.944
    # here, i.e. p_eff/N ~ 0.11 at gate scale) — that shortfall is the D31 dof effect,
    # not a miscalibration, and the marginal's own alpha-hat above is unbiased anyway.
    assert 0.88 < float(np.std(z)) < 1.02
    assert abs(float(np.corrcoef(z[:-1], z[1:])[0, 1])) < 0.1


# ------------------------------------------------------------------ guards


def test_site_requires_ar1_opt_in():
    ds, _, _, _ = small_problem()
    model = MarginalOrbitModel(
        SMALL_GRID,
        ds,
        light_fractions=[0.6, 0.4],
        lsf_sigma_v={"A": 4.0, "B": 9.0},
        v_rel_max_kms=40.0,
    )
    theta = {
        "period": jnp.asarray(6.0),
        "t_conj": jnp.asarray(1.0),
        "secosw": jnp.asarray(0.1),
        "sesinw": jnp.asarray(0.1),
        "k": jnp.asarray([9.0, 14.0]),
        "log_tau": jnp.log(jnp.asarray([2.0, 0.7])),
        "log_eta": jnp.log(jnp.asarray([1e-3, 2e-3])),
        "ar1_phi": jnp.asarray(0.3),
    }
    with pytest.raises(ValueError, match="ar1=True"):
        model.problem_at(theta)
    ar_model = MarginalOrbitModel(
        SMALL_GRID,
        ds,
        light_fractions=[0.6, 0.4],
        lsf_sigma_v={"A": 4.0, "B": 9.0},
        v_rel_max_kms=40.0,
        ar1=True,
    )
    assert ar_model.half_bandwidth >= model.half_bandwidth
    assert np.isfinite(float(ar_model.log_likelihood(theta)))


def test_bad_phi_shape_is_rejected():
    _, _, problem, _ = small_problem()
    with pytest.raises(ValueError, match="scalar or"):
        with_ar1(problem, jnp.zeros(5))

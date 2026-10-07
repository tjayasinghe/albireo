"""Tests for the marginalized likelihood: dense brute-force equivalence, the closed-loop
acceptance test, mask invariance, and the Fourier degeneracy theory.

The dense reference implements ``docs/math.md`` §3.1 with plain NumPy linear algebra on
an explicitly assembled design matrix. It is an independent path that shares only the
elementary operators with the production code.
"""

import jax
import jax.numpy as jnp
import numpy as np

import albireo as ab
import albireo.likelihood as likelihood
from albireo.assembly import band_block_tridiagonal, prior_logdet
from albireo.data import Dataset, EpochData
from albireo.forward import (
    apply_model,
    build_problem,
    data_residual_zscores,
    rhs,
    weighted_data_terms,
)
from albireo.likelihood import _pack, draw_spectra, marginal_loglikelihood, spectra_std
from albireo.priors import SmoothnessPrior
from albireo.simulate import InstrumentSpec, OrbitParams, simulate_dataset
from albireo.simulate import synthetic_deviation_spectrum as synth

RNG = np.random.default_rng(31)

# Tolerance within which the band assembly and the probe assembly must agree on the
# marginal log-likelihood. They are two different algorithms for the same number (a
# banded Cholesky and operator probing), so they sum the same terms in different orders
# and the last digits can differ. The bound therefore has to allow for float64
# accumulation on any platform.
#
# A bound of 1e-12 was met on the Windows development machine and exceeded on
# ubuntu-latest, where test_arbitrary_asymmetric_bank_band_matches_probe[True] gave
# 1.204e-12 relative on a log-likelihood of -27.15, 20% above the bound. Differences in
# BLAS, SIMD width and XLA fusion decisions are enough to change a reduction by that
# much, and the random-asymmetric-kernel case is the worst-conditioned one in the suite.
#
# 1e-10 is the tolerance these same tests already use for band-against-dense, and it is
# still ten significant figures. The errors the assertion exists to detect (a transposed
# tap, a reversed kernel, a mis-ordered component block) change the result in the first
# digits, so the wider tolerance still detects them.
BAND_PROBE_RTOL = 1e-10


# ---------------------------------------------------------------------------
# Dense brute-force reference (component-major ordering)
# ---------------------------------------------------------------------------


def dense_design_matrix(problem):
    """Assemble the weighted design matrix column-by-column via the forward operators."""
    n_c, n_p = problem.n_components, problem.grid.n
    cols = []
    for i in range(n_c):
        for q in range(n_p):
            d = np.zeros((n_c, n_p))
            d[i, q] = 1.0
            per_group = apply_model(problem, jnp.asarray(d))
            cols.append(
                np.concatenate(
                    [
                        np.asarray(g.r * m).ravel()
                        for g, m in zip(problem.groups, per_group, strict=True)
                    ]
                )
            )
    return np.stack(cols, axis=1)


def dense_marginal(problem, prior):
    a = dense_design_matrix(problem)
    z = np.concatenate([np.asarray(g.z).ravel() for g in problem.groups])
    w = np.concatenate([np.asarray(g.w).ravel() for g in problem.groups])
    lam_p = prior.dense(problem.grid.n)
    lam = lam_p + (a.T * w) @ a
    b = a.T @ (w * z)
    x = np.linalg.solve(lam, b)
    good = w > 0
    logp = (
        -0.5 * (z @ (w * z) - b @ x)
        - 0.5 * np.linalg.slogdet(lam)[1]
        + 0.5 * np.linalg.slogdet(lam_p)[1]
        + 0.5 * np.log(w[good]).sum()
        - 0.5 * good.sum() * np.log(2 * np.pi)
    )
    d_hat = x.reshape(problem.n_components, problem.grid.n)
    var = np.diag(np.linalg.inv(lam)).reshape(problem.n_components, problem.grid.n)
    return logp, d_hat, var


# ---------------------------------------------------------------------------
# Small mixed-instrument problem for exactness tests
# ---------------------------------------------------------------------------

SMALL_GRID = ab.LogGrid.from_wavelength_range(5000.0, 5003.0, dv_kms=3.0)
SMALL_VEL = np.array([[9.0, -12.0, 3.0], [-14.0, 18.0, -5.0]])


def small_problem(cosmic_fraction=0.01):
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
    problem = build_problem(
        SMALL_GRID,
        ds,
        velocities=SMALL_VEL,
        light_fractions=[0.6, 0.4],
        lsf_sigma_v={"A": 4.0, "B": 9.0},
        response_coeffs=list(truth.response_coeffs),
    )
    prior = SmoothnessPrior(tau=[2.0, 0.7], eta=[1e-3, 2e-3])
    return ds, truth, problem, prior


def test_marginal_matches_dense_brute_force():
    _, _, problem, prior = small_problem()
    res = marginal_loglikelihood(problem, prior, validate=True)
    logp_dense, d_hat_dense, var_dense = dense_marginal(problem, prior)

    np.testing.assert_allclose(float(res.log_likelihood), logp_dense, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(res.d_hat), d_hat_dense, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(np.asarray(spectra_std(res)) ** 2, var_dense, rtol=1e-7, atol=1e-12)


def test_masked_pixel_values_do_not_affect_anything():
    ds, truth, problem, prior = small_problem()
    res = marginal_loglikelihood(problem, prior)

    # overwrite the flux at masked pixels and rebuild the dataset and the problem
    epochs = []
    for ep in ds:
        flux = np.where(ep.good, ep.flux, 1234.5)
        epochs.append(
            EpochData(
                wave=ep.wave,
                flux=flux,
                ivar=ep.ivar,
                bjd=ep.bjd,
                v_bary=ep.v_bary,
                instrument=ep.instrument,
            )
        )
    ds2 = Dataset(epochs=tuple(epochs), frame=ds.frame)
    problem2 = build_problem(
        SMALL_GRID,
        ds2,
        velocities=SMALL_VEL,
        light_fractions=[0.6, 0.4],
        lsf_sigma_v={"A": 4.0, "B": 9.0},
        response_coeffs=list(truth.response_coeffs),
    )
    res2 = marginal_loglikelihood(problem2, prior)
    np.testing.assert_allclose(float(res2.log_likelihood), float(res.log_likelihood), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(res2.d_hat), np.asarray(res.d_hat), rtol=1e-10)


# ---------------------------------------------------------------------------
# The chi-square sign guard
# ---------------------------------------------------------------------------


def unguarded_marginal(problem, prior):
    """The marginal log-likelihood without the sign guard.

    The operations, their inputs and their order are those of
    :func:`marginal_loglikelihood`, and both functions run eagerly here. A healthy value
    and its gradient are therefore bit-identical to the guarded ones unless the guard has
    changed the arithmetic.
    """
    n_comp, n_pix = problem.n_components, problem.grid.n
    b_nat = max(problem.natural_half_bandwidth, prior.half_bandwidth)
    bt = band_block_tridiagonal(problem, prior, b_nat, None)
    ld_prior = prior_logdet(prior, n_pix)
    b_vec = _pack(rhs(problem), n_comp, n_pix)
    b_pad = jnp.pad(b_vec, (0, bt.num_blocks * bt.block_size - n_comp * n_pix))
    ld, quad, _ = likelihood._solve_stage(bt, b_pad)
    zwz, logw, n_good = weighted_data_terms(problem)
    return (
        -0.5 * (zwz - quad)
        - 0.5 * ld
        + 0.5 * ld_prior
        + 0.5 * logw
        - 0.5 * n_good * jnp.log(2.0 * jnp.pi)
    )


def _loglike_at_log_tau(problem, prior, log_tau, fun=marginal_loglikelihood):
    p = SmoothnessPrior(tau=jnp.exp(jnp.asarray(log_tau)), eta=jnp.asarray(prior.eta))
    out = fun(problem, p)
    return out if fun is unguarded_marginal else out.log_likelihood


def test_sign_guard_leaves_a_healthy_evaluation_and_its_gradient_untouched():
    _, _, problem, prior = small_problem()
    log_tau = jnp.log(jnp.asarray(prior.tau))

    guarded = float(_loglike_at_log_tau(problem, prior, log_tau))
    plain = float(_loglike_at_log_tau(problem, prior, log_tau, fun=unguarded_marginal))
    assert guarded == plain, f"guard changed the value: {guarded!r} vs {plain!r}"

    g_guarded = jax.grad(lambda x: _loglike_at_log_tau(problem, prior, x))(log_tau)
    g_plain = jax.grad(lambda x: _loglike_at_log_tau(problem, prior, x, fun=unguarded_marginal))(
        log_tau
    )
    np.testing.assert_array_equal(np.asarray(g_guarded), np.asarray(g_plain))
    assert np.all(np.isfinite(np.asarray(g_guarded)))


def test_negative_chi_square_is_rejected_with_a_finite_gradient(monkeypatch):
    """``zwz - quad`` is ``z^T (W^-1 + A Lambda_p^-1 A^T)^-1 z``, positive definite.

    A negative value therefore indicates a failed evaluation, and the failure increases
    the reported log-likelihood. The diverging ML-II run that motivated the guard
    accepted a trial whose quadratic form exceeded ``zwz`` by 417,000, which the
    unguarded expression reported as 216,000 nats of improvement. Here the same error is
    imposed on an otherwise healthy problem by inflating ``quad``, so every intermediate
    stays finite and the gradient at the rejected point can be checked. The gradient
    returned to the line search must be a number, not a nan.
    """
    _, _, problem, prior = small_problem()
    log_tau = jnp.log(jnp.asarray(prior.tau))
    zwz = float(weighted_data_terms(problem)[0])
    real = likelihood._solve_stage

    def inflated(bt, b_pad):
        ld, quad, d_pad = real(bt, b_pad)
        return ld, quad + (zwz + 1.0), d_pad  # chi-square = -1 by construction

    assert np.isfinite(float(_loglike_at_log_tau(problem, prior, log_tau)))
    monkeypatch.setattr(likelihood, "_solve_stage", inflated)
    assert float(_loglike_at_log_tau(problem, prior, log_tau)) == -np.inf
    grad = np.asarray(jax.grad(lambda x: _loglike_at_log_tau(problem, prior, x))(log_tau))
    assert np.all(np.isfinite(grad)), grad
    np.testing.assert_array_equal(grad, np.zeros_like(grad))


def test_absurd_tau_is_rejected_rather_than_reported_as_an_improvement():
    """Beyond a stiffness ratio of about 1e13 the assembled arithmetic loses every digit.

    ``tau/eta`` here is ``exp(log_tau) / 1e-3``, so every entry of the sweep is past
    :data:`albireo.inference._LOG_TAU_ETA_MAX`, the bound the model applies to keep the
    optimizer out of this region. The guard must ensure that no evaluation in this region
    exceeds the healthy one: each is ``-inf`` or a finite number no larger.
    """
    _, _, problem, prior = small_problem()
    healthy = float(_loglike_at_log_tau(problem, prior, jnp.log(jnp.asarray(prior.tau))))
    values = [
        float(_loglike_at_log_tau(problem, prior, jnp.full(2, lt)))
        for lt in (40.0, 50.0, 60.0, 80.0, 100.0, 120.0)
    ]
    assert not any(np.isnan(v) for v in values), values
    assert all(v <= healthy for v in values), values
    assert any(v == -np.inf for v in values), values


# ---------------------------------------------------------------------------
# M2 closed-loop acceptance test (design.md §8)
# ---------------------------------------------------------------------------


def test_closed_loop_recovery_snr100_30_epochs():
    # The injected lines are kept broader than the LSF (8 vs 4 km/s), because pixel-level
    # recovery of the deconvolved spectrum below the instrument resolution is ill-posed.
    # The LSF transfer function suppresses the data precision at sub-LSF modes below any
    # weak prior, and the posterior reports that as large flat variance. The prior must
    # encode the true smoothness of the spectra. It is set manually here, and by ML-II
    # optimization of the marginal likelihood in joint inference.
    grid = ab.LogGrid.from_wavelength_range(4500.0, 4580.0, dv_kms=2.5)
    comps = [synth(grid, seed=s, margin=0.08, sigma_v_range=(8.0, 20.0)) for s in (1, 2)]
    orbit = OrbitParams(period=11.3, t_peri=2.0, ecc=0.2, omega=0.7, k=(45.0, 70.0))

    # Per-epoch light fractions with four eclipse epochs. Without them the k = 0
    # difference of the mean depressions is unconstrained in principle (the classic
    # additive indeterminacy of disentangling, math.md §5.2). The components differ in
    # mean absorption depression, the data constrain only the light-weighted sum, and the
    # continuum anchor shrinks the unconstrained difference to zero, which is a ~1.5%
    # systematic that no method can avoid. Eclipse epochs remove the indeterminacy
    # (design.md D13), and with them the recovery becomes noise-limited.
    n_ep = 30
    ell = np.tile(np.array([[0.6], [0.4]]), (1, n_ep))
    ell[:, [3, 10, 17, 24]] = np.array([[0.75], [0.25]])

    ds, truth = simulate_dataset(
        grid,
        comps,
        bjd=np.linspace(0.0, 22.0, n_ep),
        orbit=orbit,
        light_fractions=ell,
        instruments={
            "H": InstrumentSpec(wave=np.arange(4505.0, 4575.0, 0.04), sigma_v_lsf=4.0, snr=100.0)
        },
        gap_fraction=0.05,
        cosmic_fraction=0.002,
        seed=42,
    )
    problem = build_problem(
        grid,
        ds,
        velocities=truth.velocities,
        light_fractions=ell,
        lsf_sigma_v={"H": 4.0},
    )
    # tau = 1e3: prior curvature scale 1/sqrt(tau) ~ 0.03/px^2, matching the true line
    # smoothness, so the sub-LSF band has neither signal nor posterior variance.
    # eta = 20 is the continuum anchor (docs/math.md §2, §5.1). The exact k=0 difference
    # mode is unconstrained by the data and its posterior std per pixel is
    # ~ sqrt(1/eta)/sqrt(2n) ~ 0.3% here. The line-depth bias from both terms is <~1e-3
    # because the data precision at line scales is ~1e4-1e5.
    prior = SmoothnessPrior(tau=[1e3, 1e3], eta=[20.0, 20.0])
    res = marginal_loglikelihood(problem, prior, validate=True)

    d_hat = np.asarray(res.d_hat)
    std = np.asarray(spectra_std(res))

    # <1% RMS recovery in line regions (the acceptance criterion)
    for i, d_true in enumerate(truth.components):
        lines = np.abs(d_true) > 0.05
        assert lines.sum() > 100
        rms = np.sqrt(np.mean((d_hat[i] - d_true)[lines] ** 2))
        assert rms < 0.01, f"component {i}: line-region RMS {rms:.4f}"

    # The reported uncertainties must be consistent or conservative. The injected spectra
    # are a fixed draw (smoother than the prior at high k), so the whitened errors may
    # have var < 1 but must not have var >> 1 (overconfident). Strict prior-drawn
    # calibration (SBC) is the acceptance criterion for joint inference.
    zspec = (d_hat - np.stack(truth.components)) / std
    assert abs(zspec.mean()) < 0.1
    assert zspec.var() < 1.3
    assert (np.abs(zspec) > 3).mean() < 0.01

    # whitened data residuals ~ N(0, 1) up to the fitted degrees of freedom
    zdata = data_residual_zscores(problem, res.d_hat)
    assert abs(zdata.mean()) < 0.02
    assert 0.8 < zdata.var() < 1.02

    # posterior draws agree with the Takahashi pointwise variances
    draws = np.asarray(draw_spectra(res, jax.random.PRNGKey(1), 40))
    ratio = draws.std(axis=0) / std
    assert 0.85 < np.median(ratio) < 1.15


# ---------------------------------------------------------------------------
# Degeneracy theory verification (docs/math.md §5.1)
# ---------------------------------------------------------------------------


def test_low_frequency_degeneracy_matches_theory():
    """Posterior variance of the per-mode difference direction follows
    1 / (w l^2 (J - |g(k)|) + tau (2 - 2 cos k)^2 + eta) with g(k) = sum_j e^{ik dDelta_j}."""
    n = 256
    grid = ab.LogGrid(x0=float(np.log(5000.0)), dx=1.0007e-5, n=n)
    amp = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    vel = np.stack([amp, -amp])
    comps = [synth(grid, n_lines=8, seed=s, margin=0.1) for s in (3, 4)]
    ds, _ = simulate_dataset(
        grid,
        comps,
        bjd=np.arange(6.0),
        velocities=vel,
        light_fractions=[0.5, 0.5],
        instruments={"A": InstrumentSpec(wave=grid.wave[10:-10], sigma_v_lsf=0.6, snr=20.0)},
        v_bary=np.zeros(6),
        seed=13,
    )
    tau, eta = 1.0, 1e-3
    problem = build_problem(
        grid, ds, velocities=vel, light_fractions=[0.5, 0.5], lsf_sigma_v={"A": 0.6}
    )
    prior = SmoothnessPrior(tau=[tau, tau], eta=[eta, eta])

    a = dense_design_matrix(problem)
    w = np.concatenate([np.asarray(g.w).ravel() for g in problem.groups])
    lam = prior.dense(n) + (a.T * w) @ a
    cov = np.linalg.inv(lam)

    shifts = np.asarray(problem.groups[0].shifts)  # (J, 2) pixels
    delta = shifts[:, 0] - shifts[:, 1]
    w_ell2, j_ep = 400.0 * 0.25, 6.0  # w * l^2 with snr=20, l=0.5; J epochs

    def minor_mode(m):
        """Predicted minor-eigen variance and phase for DFT mode m (math.md §5.1)."""
        k = 2 * np.pi * m / n
        g = np.exp(1j * k * delta).sum()
        lam_prior = tau * (2 - 2 * np.cos(k)) ** 2 + eta
        return 1.0 / (w_ell2 * (j_ep - np.abs(g)) + lam_prior), np.angle(g)

    # Hann-windowed modes: the discrete Hann window's DFT has support {-1, 0, +1}
    # exactly, so a windowed mode m mixes only modes m-1, m, m+1 with power weights
    # (1, 4, 1)/6. There is no leakage into the near-singular k ~ 0 directions, and the
    # window suppresses the non-periodic edge effects that the theory neglects.
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
    measured, predicted = [], []
    for m in range(4, 25):
        k = 2 * np.pi * m / n
        _, phase = minor_mode(m)
        mode = hann * np.exp(1j * k * np.arange(n))
        mode = mode / np.linalg.norm(mode)
        # minor eigenvector of the per-mode 2x2 information matrix: phase -arg g
        f = np.concatenate([mode, -np.exp(-1j * phase) * mode]) / np.sqrt(2)
        measured.append(np.real(np.conj(f) @ cov @ f))
        v = [minor_mode(mm)[0] for mm in (m - 1, m, m + 1)]
        predicted.append((v[0] + 4.0 * v[1] + v[2]) / 6.0)
    measured, predicted = np.array(measured), np.array(predicted)

    ratio = measured / predicted
    assert 0.4 < ratio.min() and ratio.max() < 2.5, ratio
    assert 0.65 < np.median(ratio) < 1.55, np.median(ratio)
    # the scaling signature: low-k separation variance is strongly inflated
    assert measured[0] / measured[-5] > 4.0

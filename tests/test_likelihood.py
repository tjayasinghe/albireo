"""Tests for the marginalized likelihood: dense brute-force equivalence, the M2
closed-loop acceptance gate, mask invariance, and the Fourier degeneracy theory.

The dense reference implements ``docs/math.md`` §3.1 with plain NumPy linear algebra on
an explicitly assembled design matrix — an independent path that shares only the
elementary operators with the production code.
"""

import jax
import jax.numpy as jnp
import numpy as np

import albireo as ab
from albireo.data import Dataset, EpochData
from albireo.forward import apply_model, build_problem, data_residual_zscores
from albireo.likelihood import draw_spectra, marginal_loglikelihood, spectra_std
from albireo.priors import SmoothnessPrior
from albireo.simulate import InstrumentSpec, OrbitParams, simulate_dataset
from albireo.simulate import synthetic_deviation_spectrum as synth

RNG = np.random.default_rng(31)

# Tolerance for "the band assembly and the probe assembly agree on the marginal
# log-likelihood". These are two *different algorithms* for the same number — a banded
# Cholesky against operator probing — so they sum the same terms in different orders and
# the last digits are free to disagree. The bound therefore has to be a statement about
# float64 accumulation, not about the platform it was first measured on.
#
# It was 1e-12, which held on the Windows development machine and failed on ubuntu-latest
# the first time CI ran: test_arbitrary_asymmetric_bank_band_matches_probe[True] came back
# at 1.204e-12 relative on a log-likelihood of -27.15, i.e. 20% over the line. Different
# BLAS, different SIMD width and different XLA fusion decisions are enough to move a
# reduction by that much, and the random-asymmetric-kernel case is deliberately the
# worst-conditioned one in the suite.
#
# 1e-10 is the tolerance these same tests already use for band-against-dense, and it is
# still ten significant figures. What the assertion exists to catch — a transposed tap, a
# reversed kernel, a mis-ordered component block — moves the answer in the first digits,
# not the eleventh, so nothing is given up by quoting a bound that is about arithmetic
# rather than about a machine.
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

    # rewrite the garbage at masked pixels and rebuild everything from scratch
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
# M2 closed-loop acceptance gate (design.md §8)
# ---------------------------------------------------------------------------


def test_closed_loop_recovery_snr100_30_epochs():
    # Truth lines are kept broader than the LSF (8 vs 4 km/s): pixel-level recovery of
    # the DECONVOLVED spectrum below the instrument resolution is genuinely ill-posed
    # (the LSF transfer function suppresses the data precision at sub-LSF modes below
    # any weak prior), and the posterior honestly reports that as large flat variance.
    # The prior must encode the spectra's true smoothness — here set by hand, in M3 by
    # ML-II optimization of the marginal likelihood.
    grid = ab.LogGrid.from_wavelength_range(4500.0, 4580.0, dv_kms=2.5)
    comps = [synth(grid, seed=s, margin=0.08, sigma_v_range=(8.0, 20.0)) for s in (1, 2)]
    orbit = OrbitParams(period=11.3, t_peri=2.0, ecc=0.2, omega=0.7, k=(45.0, 70.0))

    # Per-epoch light fractions with four eclipse epochs. Without them the k = 0
    # "difference of mean depressions" direction is invisible IN PRINCIPLE (the classic
    # additive indeterminacy of disentangling, math.md §5.2): each component's mean
    # absorption depression differs, the data only see the light-weighted sum, and the
    # continuum anchor shrinks the invisible difference to zero — a ~1.5% systematic
    # no method can avoid. Eclipse epochs are the documented breaker (design.md D13),
    # and with them the recovery becomes noise-limited.
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
    # smoothness, so the sub-LSF band carries neither signal nor posterior variance.
    # eta = 20 is the continuum anchor (docs/math.md §2, §5.1): the exact k=0 difference
    # mode is invisible to the data and its posterior std per pixel is
    # ~ sqrt(1/eta)/sqrt(2n) ~ 0.3% here; line-depth bias from both terms is <~1e-3
    # because the data precision at line scales is ~1e4-1e5.
    prior = SmoothnessPrior(tau=[1e3, 1e3], eta=[20.0, 20.0])
    res = marginal_loglikelihood(problem, prior, validate=True)

    d_hat = np.asarray(res.d_hat)
    std = np.asarray(spectra_std(res))

    # <1% RMS recovery in line regions (the M2 gate)
    for i, d_true in enumerate(truth.components):
        lines = np.abs(d_true) > 0.05
        assert lines.sum() > 100
        rms = np.sqrt(np.mean((d_hat[i] - d_true)[lines] ** 2))
        assert rms < 0.01, f"component {i}: line-region RMS {rms:.4f}"

    # reported uncertainties are conservative-consistent: the truth is a fixed draw
    # (smoother than the prior at high k), so whitened errors must not be OVERconfident
    # (var >> 1) but may be < 1. Strict prior-drawn calibration (SBC) is the M3 gate.
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
    """Posterior variance of the per-mode 'difference' direction follows
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
    # (1, 4, 1)/6 — no leakage into the near-singular k ~ 0 directions, and the
    # window suppresses the non-periodic edge effects the theory ignores.
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

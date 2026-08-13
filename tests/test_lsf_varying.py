"""Tests for the wavelength-dependent (tabulated) LSF (docs/design.md D8 v2, D37).

D8 fixed a Gaussian constant-resolving-power LSF and reserved a seam: "tabulated LSF
is v2 (banded matrix, no structural change)". D37 opens that seam — each model pixel
may apply its own kernel row, realized from per-anchor kernels through static
interpolation tables. What is pinned here: the row-varying convolution must *be* the
banded matrix it claims (dense reconstruction), form an exact adjoint pair, and
reduce to the stationary operator when every row is equal; an anchored problem's
marginal must agree between the band assembly, comb probing, and a dense LAPACK
reference — under diagonal and AR(1) noise, for Gaussian *and* arbitrary asymmetric
banks (asymmetry is what flushes out a hidden kernel-flip assumption), and under
gradients in the per-anchor widths; and a constant-width anchored build must
reproduce the stationary marginal, because it is the same matrix realized through
the other code path.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import albireo as ab
from albireo.forward import build_problem, with_ar1, with_jitter, with_lsf
from albireo.likelihood import marginal_loglikelihood
from albireo.operators import (
    convolve_spectrum,
    convolve_varying,
    convolve_varying_adjoint,
    gaussian_kernel,
    lsf_anchor_tables,
)
from albireo.priors import SmoothnessPrior
from albireo.simulate import InstrumentSpec, simulate_dataset
from albireo.simulate import synthetic_deviation_spectrum as synth
from tests.test_likelihood import dense_marginal

RNG = np.random.default_rng(7)

SMALL_GRID = ab.LogGrid.from_wavelength_range(5000.0, 5003.0, dv_kms=3.0)
SMALL_VEL = np.array([[9.0, -12.0, 3.0], [-14.0, 18.0, -5.0]])
ANCHORS = {"A": (5000.0, 5001.2, 5003.0), "B": (5000.0, 5003.0)}
SIGMAS = {"A": [3.0, 4.0, 3.5], "B": [9.0, 7.0]}
PRIOR = SmoothnessPrior(tau=[2.0, 0.7], eta=[1e-3, 2e-3])


def small_dataset(cosmic_fraction=0.02):
    comps = [synth(SMALL_GRID, n_lines=6, seed=s, margin=0.15) for s in (1, 2)]
    return simulate_dataset(
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


def anchored_problem(lsf_sigma_v=None):
    ds, truth = small_dataset()
    return build_problem(
        SMALL_GRID,
        ds,
        velocities=SMALL_VEL,
        light_fractions=[0.6, 0.4],
        lsf_sigma_v=SIGMAS if lsf_sigma_v is None else lsf_sigma_v,
        lsf_anchors_angstrom=ANCHORS,
        response_coeffs=list(truth.response_coeffs),
    )


# ------------------------------------------------------------------ the operator


def test_convolve_varying_is_the_banded_matrix_it_claims():
    """Dense reconstruction from the documented convention: K[m, c] = P[m, m - c + r]."""
    n, w = 41, 7
    r = (w - 1) // 2
    profiles = RNG.standard_normal((n, w))
    k = np.zeros((n, n))
    for m in range(n):
        for c in range(max(0, m - r), min(n, m + r + 1)):
            k[m, c] = profiles[m, m - c + r]
    x = RNG.standard_normal(n)
    y = RNG.standard_normal(n)
    np.testing.assert_allclose(np.asarray(convolve_varying(x, profiles)), k @ x, atol=1e-13)
    np.testing.assert_allclose(
        np.asarray(convolve_varying_adjoint(y, profiles)), k.T @ y, atol=1e-13
    )


def test_convolve_varying_adjoint_pair():
    n, w = 57, 9
    profiles = jnp.asarray(RNG.standard_normal((n, w)))
    u = jnp.asarray(RNG.standard_normal(n))
    v = jnp.asarray(RNG.standard_normal(n))
    (ct,) = jax.linear_transpose(lambda f: convolve_varying(f, profiles), u)(v)
    np.testing.assert_allclose(
        np.asarray(ct), np.asarray(convolve_varying_adjoint(v, profiles)), atol=1e-13
    )


def test_constant_rows_reduce_to_stationary_convolution():
    kernel = np.asarray(gaussian_kernel(1.3))
    x = RNG.standard_normal(64)
    np.testing.assert_allclose(
        np.asarray(convolve_varying(x, np.tile(kernel, (64, 1)))),
        np.asarray(convolve_spectrum(x, kernel)),
        atol=1e-15,
    )


def test_lsf_anchor_tables_interpolate_and_clamp():
    grid = ab.LogGrid.from_wavelength_range(5000.0, 5010.0, dv_kms=10.0)
    # Anchors strictly inside the grid: pixels outside are clamped to the end pairs.
    idx, t = lsf_anchor_tables((5002.0, 5005.0, 5008.0), grid.wave)
    assert idx.shape == t.shape == (grid.n,)
    assert idx.dtype == np.int32
    assert np.all((t >= 0.0) & (t <= 1.0))
    w = np.asarray(grid.wave)
    assert np.all(idx[w <= 5002.0] == 0) and np.all(t[w < 5002.0] == 0.0)
    assert np.all(idx[w >= 5008.0] == 1) and np.all(t[w > 5008.0] == 1.0)
    # Interior: reconstructing the anchor wavelength from (idx, t) inverts the tables.
    la = np.log(np.array([5002.0, 5005.0, 5008.0]))
    inside = (w > 5002.0) & (w < 5008.0)
    rec = la[idx[inside]] * (1.0 - t[inside]) + la[idx[inside] + 1] * t[inside]
    np.testing.assert_allclose(rec, np.log(w[inside]), rtol=1e-14)
    with pytest.raises(ValueError, match="strictly increasing"):
        lsf_anchor_tables((5005.0, 5002.0), grid.wave)
    with pytest.raises(ValueError, match="at least 2"):
        lsf_anchor_tables((5005.0,), grid.wave)


# ------------------------------------------------------------------ the marginal


def test_anchored_constant_width_matches_stationary_marginal():
    """Equal anchor widths realize the stationary matrix through the varying path."""
    ds, truth = small_dataset()
    kw = dict(
        velocities=SMALL_VEL,
        light_fractions=[0.6, 0.4],
        response_coeffs=list(truth.response_coeffs),
    )
    stationary = build_problem(SMALL_GRID, ds, lsf_sigma_v={"A": 4.0, "B": 9.0}, **kw)
    anchored = build_problem(
        SMALL_GRID,
        ds,
        lsf_sigma_v={"A": 4.0, "B": 9.0},
        lsf_anchors_angstrom=ANCHORS,
        **kw,
    )
    for g in anchored.groups:
        assert g.kernel.shape[0] == SMALL_GRID.n  # really on the varying path
    ref = marginal_loglikelihood(stationary, PRIOR, assembly="band")
    out = marginal_loglikelihood(anchored, PRIOR, assembly="band")
    np.testing.assert_allclose(float(out.log_likelihood), float(ref.log_likelihood), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(out.d_hat), np.asarray(ref.d_hat), rtol=1e-8, atol=1e-10)


def test_band_matches_probe_and_dense_with_varying_lsf():
    """The gold identity on a genuinely wavelength-dependent problem (diagonal noise)."""
    problem = anchored_problem()
    band = marginal_loglikelihood(problem, PRIOR, assembly="band")
    probe = marginal_loglikelihood(problem, PRIOR, assembly="probe")
    logp, d_hat, _ = dense_marginal(problem, PRIOR)
    np.testing.assert_allclose(float(band.log_likelihood), float(probe.log_likelihood), rtol=1e-12)
    np.testing.assert_allclose(float(band.log_likelihood), logp, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(band.d_hat), np.asarray(probe.d_hat), atol=1e-9)
    np.testing.assert_allclose(np.asarray(band.d_hat), d_hat, rtol=1e-7, atol=1e-9)


def test_band_matches_probe_with_varying_lsf_and_ar1():
    """Varying LSF x correlated noise: the two bandwidth-widening features compose."""
    problem = with_ar1(with_jitter(anchored_problem(), np.array([1.2, 0.9, 1.4])), 0.45)
    band = marginal_loglikelihood(problem, PRIOR, assembly="band", validate=True)
    probe = marginal_loglikelihood(problem, PRIOR, assembly="probe")
    np.testing.assert_allclose(float(band.log_likelihood), float(probe.log_likelihood), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(band.d_hat), np.asarray(probe.d_hat), atol=1e-9)


@pytest.mark.parametrize("correlated", [False, True])
def test_arbitrary_asymmetric_bank_band_matches_probe(correlated):
    """Random asymmetric profiles, varying and stationary: no hidden flip survives this.

    The Gaussian rows every fixture builds are symmetric, so a transposed tap or a
    reversed kernel would cancel silently there. Random banks — a full per-pixel one
    on the anchored groups and a (1, w) one on a stationary build — are the
    adversarial case.
    """
    rng = np.random.default_rng(11)
    problem = anchored_problem()
    groups = tuple(
        replace(g, kernel=jnp.asarray(rng.standard_normal(g.kernel.shape) * 0.3 + 0.1))
        for g in problem.groups
    )
    problem = replace(problem, groups=groups)

    ds, truth = small_dataset()
    stationary = build_problem(
        SMALL_GRID,
        ds,
        velocities=SMALL_VEL,
        light_fractions=[0.6, 0.4],
        lsf_sigma_v={"A": 4.0, "B": 9.0},
        response_coeffs=list(truth.response_coeffs),
    )
    stat_groups = tuple(
        replace(g, kernel=jnp.asarray(rng.standard_normal(g.kernel.shape) * 0.3 + 0.1))
        for g in stationary.groups
    )
    cases = [("varying", problem), ("stationary", replace(stationary, groups=stat_groups))]
    for label, prob in cases:
        if correlated:
            prob = with_ar1(with_jitter(prob, np.array([1.2, 0.9, 1.4])), 0.45)
        band = marginal_loglikelihood(prob, PRIOR, assembly="band")
        probe = marginal_loglikelihood(prob, PRIOR, assembly="probe")
        np.testing.assert_allclose(
            float(band.log_likelihood),
            float(probe.log_likelihood),
            rtol=1e-12,
            err_msg=label,
        )
        np.testing.assert_allclose(
            np.asarray(band.d_hat), np.asarray(probe.d_hat), atol=1e-9, err_msg=label
        )


def test_gradient_in_anchor_widths_band_matches_probe():
    """d(marginal)/d(per-anchor widths) through with_lsf, band vs probe."""
    problem = anchored_problem()

    def loglike(sig_a, assembly):
        p = with_lsf(problem, {"A": sig_a, "B": jnp.asarray(SIGMAS["B"])})
        return marginal_loglikelihood(p, PRIOR, assembly=assembly).log_likelihood

    sig = jnp.asarray(SIGMAS["A"])
    g_band = jax.grad(lambda s: loglike(s, "band"))(sig)
    g_probe = jax.grad(lambda s: loglike(s, "probe"))(sig)
    assert np.all(np.isfinite(np.asarray(g_band)))
    np.testing.assert_allclose(np.asarray(g_band), np.asarray(g_probe), rtol=1e-9)


# ------------------------------------------------------------------ validation


def test_kernel_radius_follows_the_widest_anchor():
    problem = anchored_problem()
    sigma_px = max(max(SIGMAS["A"]), max(SIGMAS["B"])) / SMALL_GRID.dv_kms
    assert problem.kernel_radius == int(np.ceil(4.0 * sigma_px))


def test_build_and_with_lsf_reject_mismatched_widths():
    ds, truth = small_dataset()
    kw = dict(
        velocities=SMALL_VEL,
        light_fractions=[0.6, 0.4],
        response_coeffs=list(truth.response_coeffs),
    )
    with pytest.raises(ValueError, match="per-anchor widths need lsf_anchors_angstrom"):
        build_problem(SMALL_GRID, ds, lsf_sigma_v={"A": [3.0, 4.0], "B": 9.0}, **kw)
    with pytest.raises(ValueError, match="one per anchor"):
        build_problem(
            SMALL_GRID,
            ds,
            lsf_sigma_v={"A": [3.0, 4.0], "B": 9.0},
            lsf_anchors_angstrom=ANCHORS,
            **kw,
        )
    problem = anchored_problem()
    with pytest.raises(ValueError, match="one per anchor"):
        with_lsf(problem, {"A": jnp.asarray([3.0, 4.0]), "B": 9.0})
    stationary = build_problem(SMALL_GRID, ds, lsf_sigma_v={"A": 4.0, "B": 9.0}, **kw)
    with pytest.raises(ValueError, match="built without LSF anchors"):
        with_lsf(stationary, {"A": jnp.asarray([3.0, 4.0, 3.5]), "B": 9.0})


def test_with_lsf_scalar_broadcasts_over_anchors():
    """A scalar width on an anchored group reproduces the constant-width bank."""
    problem = anchored_problem()
    a = with_lsf(problem, {"A": 4.0, "B": 9.0})
    b = with_lsf(problem, {"A": jnp.full(3, 4.0), "B": jnp.full(2, 9.0)})
    for ga, gb in zip(a.groups, b.groups, strict=True):
        np.testing.assert_allclose(np.asarray(ga.kernel), np.asarray(gb.kernel), atol=1e-15)


def test_site_layout_concatenates_anchored_and_stationary_instruments():
    """lsf_sigma = anchored instrument's per-anchor widths ++ stationary scalars."""
    from albireo.inference import MarginalOrbitModel

    ds, truth = small_dataset()
    model = MarginalOrbitModel(
        SMALL_GRID,
        ds,
        light_fractions=[0.6, 0.4],
        lsf_sigma_v={"A": SIGMAS["A"], "B": 9.0},
        lsf_anchors_angstrom={"A": ANCHORS["A"]},
        v_rel_max_kms=45.0,
        prior=PRIOR,
        response_coeffs=list(truth.response_coeffs),
    )
    theta = {
        "period": jnp.asarray(10.0),
        "t_conj": jnp.asarray(0.0),
        "secosw": jnp.asarray(0.05),
        "sesinw": jnp.asarray(0.05),
        "k": jnp.asarray([5.0, 4.0]),
        "lsf_sigma": jnp.asarray([3.0, 4.0, 3.5, 9.0]),
    }
    assert np.isfinite(float(model.marginal(theta).log_likelihood))
    with pytest.raises(ValueError, match="lsf_sigma must have 4 entries"):
        model.marginal({**theta, "lsf_sigma": jnp.asarray([3.0, 9.0])})


# ------------------------------------------------------------------ closed loop

P_TRUE, TCONJ_TRUE, ECC_TRUE, OMEGA_TRUE = 6.31, 2.05, 0.2, 0.7
K_TRUE = np.array([30.0, 22.0])
ELL = np.array([0.62, 0.38])
GATE_GRID = ab.LogGrid.from_wavelength_range(5000.0, 5045.0, dv_kms=5.5)
N_EP = 10
GATE_ANCHORS = (5000.0, 5022.5, 5045.0)
SIG_TRUE = np.array([5.0, 7.0, 9.5])
GATE_PRIOR = SmoothnessPrior(tau=[300.0, 300.0], eta=[5.0, 5.0])


def gate_dataset():
    from albireo.kepler import t_peri_from_t_conj
    from albireo.simulate import OrbitParams

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
    spec = InstrumentSpec(
        wave=np.arange(5003.0, 5042.0, 0.11),
        sigma_v_lsf=tuple(SIG_TRUE),
        snr=130.0,
        lsf_anchors_angstrom=GATE_ANCHORS,
    )
    return simulate_dataset(
        GATE_GRID,
        comps,
        bjd=bjd,
        instruments={"inst": spec},
        light_fractions=ELL,
        orbit=orbit,
        v_bary=rng.uniform(-25.0, 25.0, N_EP),
        frame="topocentric",
        seed=11,
    )


def gate_model():
    from albireo.inference import MarginalOrbitModel

    ds, _ = gate_dataset()
    return MarginalOrbitModel(
        GATE_GRID,
        ds,
        light_fractions=ELL,
        lsf_sigma_v={"inst": 10.5},
        lsf_anchors_angstrom={"inst": GATE_ANCHORS},
        v_rel_max_kms=float(K_TRUE.sum()) * (1 + ECC_TRUE) * 1.35,
        prior=GATE_PRIOR,
    )


def test_data_term_prefers_the_injected_width_profile_at_fixed_spectra():
    """The injected wavelength dependence is in the data and the forward model sees it.

    At the *true spectra* — absorption switched off — the weighted residuals under
    the true width ramp must beat a flat width at the ramp's mean: the varying
    kernel matters and the forward model realizes it correctly end-to-end.

    Deliberately NOT asserted: that the *marginal* (free spectra) prefers the truth.
    Measured here, it does not — the flat width beats the true ramp by ~3 nats, and
    the ML profile beats the truth by ~8 while sitting ~3 km/s off one anchor. A
    stationary width change commutes with the shifts, so the free spectra absorb it
    and the marginal's width preference is dominated by the smoothness prior's taste
    for smoother spectra, not by the instrument. Fitted anchor widths are therefore
    diagnostics, not measurements — the recovery test below asserts exactly the
    identified content (the orbit, and the ramp's direction) and no more.
    """
    from albireo.forward import data_residual_zscores

    ds, truth = gate_dataset()
    d_true = jnp.asarray(np.stack(truth.components))
    vel = truth.velocities

    def chi2(widths):
        problem = build_problem(
            GATE_GRID,
            ds,
            velocities=vel,
            light_fractions=ELL,
            lsf_sigma_v={"inst": widths},
            lsf_anchors_angstrom={"inst": GATE_ANCHORS},
        )
        z = data_residual_zscores(problem, d_true)
        return float(np.sum(np.square(z)))

    c_ramp = chi2(list(SIG_TRUE))
    c_flat = chi2(float(SIG_TRUE.mean()))
    assert c_ramp < c_flat - 100.0, f"ramp chi2 {c_ramp:.1f} vs flat {c_flat:.1f}"


@pytest.mark.slow
def test_closed_loop_joint_fit_recovers_orbit_and_ramp_direction():
    """The D37 gate: per-anchor widths free, and the orbit must not be corrupted.

    Asserted: the orbit (the quantity the LSF could plausibly bias), the widths
    staying interior to their bounds, and the injected ramp's *direction* across the
    data span. Absolute width levels are deliberately not asserted — they sit along
    the absorption-degenerate direction (measured here: ~2 nats between the fitted
    3-width profile and one shared width), so a tight tolerance would test the
    optimizer's wandering, not the physics.
    """
    import numpyro.distributions as dist

    from albireo.inference import run_map

    model = gate_model()
    priors = {
        "period": dist.Normal(P_TRUE + 0.001, 0.003),
        "t_conj": dist.Normal(TCONJ_TRUE + 0.005, 0.02),
        "secosw": dist.Uniform(-1.0, 1.0),
        "sesinw": dist.Uniform(-1.0, 1.0),
        "k": dist.Uniform(jnp.array([10.0, 5.0]), jnp.array([45.0, 40.0])),
        "lsf_sigma": dist.Uniform(jnp.full(3, 2.0), jnp.full(3, 10.5)).to_event(1),
    }
    init = {
        "period": P_TRUE + 0.001,
        "t_conj": TCONJ_TRUE + 0.005,
        "secosw": np.sqrt(0.15) * np.cos(0.5),
        "sesinw": np.sqrt(0.15) * np.sin(0.5),
        "k": jnp.array([27.0, 25.0]),
        "lsf_sigma": jnp.full(3, 7.0),
    }
    fit = run_map(model.model(priors), init=init, max_steps=300)
    np.testing.assert_allclose(np.asarray(fit.params["k"]), K_TRUE, rtol=1e-2)
    np.testing.assert_allclose(float(fit.params["period"]), P_TRUE, atol=2e-3)
    sig = np.asarray(fit.params["lsf_sigma"])
    assert np.all((sig > 2.0) & (sig < 10.5))
    assert sig[-1] > sig[0] + 0.5, f"injected ramp direction lost: {sig}"

"""Tests for the LSF asymmetry lever: per-anchor Gauss-Hermite h3 (D38).

D37 opened the tabulated-LSF seam and its record narrowed the surviving LSF suspect
to profile *asymmetry* — the first-order centroid channel a symmetric width can
never produce. The operator already takes arbitrary asymmetric banks; D38 adds the
θ-parameterization: per-anchor Gauss-Hermite ``h3``. What is pinned here: ``h3 = 0``
must reproduce the pure Gaussian machinery exactly (bit-for-bit, so D37 problems are
untouched); the kernel's centroid must move as the series says (``~ sqrt(3) h3
sigma``, the physical content of the site); the marginal must agree between band
assembly, probing, and dense LAPACK on h3-anchored problems under diagonal and AR(1)
noise, and under gradients in h3; injected asymmetry must be seen by the
fixed-spectra data term; and a joint fit with the site free must leave the orbit
unbiased. The injected profile's *recovery* is deliberately not asserted — the free
spectra absorb even the wavelength-varying part of an asymmetry (a static
centroid-warp field is representable by a free spectrum outright), so fitted h3
profiles are diagnostics more thoroughly than fitted widths; see the closed-loop
test's docstring for the measured numbers.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from albireo.forward import build_problem, with_ar1, with_jitter, with_lsf
from albireo.likelihood import marginal_loglikelihood
from albireo.operators import (
    gauss_hermite_kernel_traced,
    gaussian_kernel_traced,
    gaussian_lsf_profiles,
)
from albireo.simulate import InstrumentSpec, simulate_dataset
from albireo.simulate import synthetic_deviation_spectrum as synth
from tests.test_likelihood import dense_marginal
from tests.test_lsf_varying import (
    ANCHORS,
    ELL,
    GATE_ANCHORS,
    GATE_GRID,
    GATE_PRIOR,
    K_TRUE,
    N_EP,
    PRIOR,
    SIGMAS,
    SMALL_GRID,
    SMALL_VEL,
    small_dataset,
)

H3S = {"A": [-0.10, 0.05, 0.15], "B": [0.10, -0.05]}


def h3_problem():
    ds, truth = small_dataset()
    return build_problem(
        SMALL_GRID,
        ds,
        velocities=SMALL_VEL,
        light_fractions=[0.6, 0.4],
        lsf_sigma_v=SIGMAS,
        lsf_anchors_angstrom=ANCHORS,
        lsf_h3=H3S,
        response_coeffs=list(truth.response_coeffs),
    )


# ------------------------------------------------------------------ the kernel


def test_h3_zero_is_exactly_gaussian():
    k_gh = np.asarray(gauss_hermite_kernel_traced(jnp.asarray(1.7), jnp.asarray(0.0), 8))
    k_g = np.asarray(gaussian_kernel_traced(jnp.asarray(1.7), 8))
    np.testing.assert_array_equal(k_gh, k_g)
    grid_wave = np.asarray(SMALL_GRID.wave)
    p_none = gaussian_lsf_profiles([1.5, 2.0], (5000.0, 5003.0), grid_wave)
    p_zero = gaussian_lsf_profiles([1.5, 2.0], (5000.0, 5003.0), grid_wave, h3=[0.0, 0.0])
    np.testing.assert_array_equal(p_none, p_zero)


def test_h3_moves_the_centroid_as_the_series_says():
    """The physical content of the site: centroid shift ~ sqrt(3) * h3 * sigma."""
    sigma, radius = 3.0, 14
    off = np.arange(-radius, radius + 1, dtype=np.float64)
    for h3 in (-0.05, 0.02, 0.1):
        k = np.asarray(gauss_hermite_kernel_traced(jnp.asarray(sigma), jnp.asarray(h3), radius))
        np.testing.assert_allclose(k.sum(), 1.0, rtol=1e-13)
        centroid = float(np.sum(off * k))
        np.testing.assert_allclose(centroid, np.sqrt(3.0) * h3 * sigma, rtol=0.02)


# ------------------------------------------------------------------ the marginal


def test_band_matches_probe_and_dense_with_h3():
    problem = h3_problem()
    band = marginal_loglikelihood(problem, PRIOR, assembly="band")
    probe = marginal_loglikelihood(problem, PRIOR, assembly="probe")
    logp, d_hat, _ = dense_marginal(problem, PRIOR)
    np.testing.assert_allclose(float(band.log_likelihood), float(probe.log_likelihood), rtol=1e-12)
    np.testing.assert_allclose(float(band.log_likelihood), logp, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(band.d_hat), np.asarray(probe.d_hat), atol=1e-9)
    np.testing.assert_allclose(np.asarray(band.d_hat), d_hat, rtol=1e-7, atol=1e-9)


def test_band_matches_probe_with_h3_and_ar1():
    problem = with_ar1(with_jitter(h3_problem(), np.array([1.2, 0.9, 1.4])), 0.45)
    band = marginal_loglikelihood(problem, PRIOR, assembly="band", validate=True)
    probe = marginal_loglikelihood(problem, PRIOR, assembly="probe")
    np.testing.assert_allclose(float(band.log_likelihood), float(probe.log_likelihood), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(band.d_hat), np.asarray(probe.d_hat), atol=1e-9)


def test_gradient_in_h3_band_matches_probe():
    problem = h3_problem()

    def loglike(h3_a, assembly):
        p = with_lsf(
            problem,
            {"A": jnp.asarray(SIGMAS["A"]), "B": jnp.asarray(SIGMAS["B"])},
            {"A": h3_a, "B": jnp.asarray(H3S["B"])},
        )
        return marginal_loglikelihood(p, PRIOR, assembly=assembly).log_likelihood

    h3 = jnp.asarray(H3S["A"])
    g_band = jax.grad(lambda h: loglike(h, "band"))(h3)
    g_probe = jax.grad(lambda h: loglike(h, "probe"))(h3)
    assert np.all(np.isfinite(np.asarray(g_band)))
    np.testing.assert_allclose(np.asarray(g_band), np.asarray(g_probe), rtol=1e-9)


# ------------------------------------------------------------------ validation


def test_h3_requires_anchors():
    ds, truth = small_dataset()
    kw = dict(
        velocities=SMALL_VEL,
        light_fractions=[0.6, 0.4],
        response_coeffs=list(truth.response_coeffs),
    )
    with pytest.raises(ValueError, match="lsf_h3 needs lsf_anchors_angstrom"):
        build_problem(SMALL_GRID, ds, lsf_sigma_v={"A": 4.0, "B": 9.0}, lsf_h3={"A": 0.1}, **kw)
    stationary = build_problem(SMALL_GRID, ds, lsf_sigma_v={"A": 4.0, "B": 9.0}, **kw)
    with pytest.raises(ValueError, match="lsf_h3 needs LSF anchors"):
        with_lsf(stationary, {"A": 4.0, "B": 9.0}, {"A": jnp.asarray(0.1)})
    problem = h3_problem()
    with pytest.raises(ValueError, match="h3 values for"):
        with_lsf(problem, {"A": SIGMAS["A"], "B": SIGMAS["B"]}, {"A": jnp.asarray([0.1, 0.2])})


def test_h3_site_layout_skips_unanchored_instruments():
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
        "lsf_h3": jnp.asarray([0.05, -0.05, 0.1]),  # anchored instrument A only
    }
    assert np.isfinite(float(model.marginal(theta).log_likelihood))
    with pytest.raises(ValueError, match="lsf_h3 must have 3 entries"):
        model.marginal({**theta, "lsf_h3": jnp.asarray([0.05, -0.05, 0.1, 0.0])})


# ------------------------------------------------------------------ closed loop

P_TRUE, TCONJ_TRUE, ECC_TRUE, OMEGA_TRUE = 6.31, 2.05, 0.2, 0.7
H3_TRUE = np.array([-0.12, 0.0, 0.12])
SIG_GATE = 7.0


def h3_gate_dataset():
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
        sigma_v_lsf=SIG_GATE,
        snr=130.0,
        lsf_anchors_angstrom=GATE_ANCHORS,
        lsf_h3=tuple(H3_TRUE),
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


def test_data_term_prefers_the_injected_h3_profile_at_fixed_spectra():
    """The injected asymmetry gradient is in the data and the forward model sees it."""
    from albireo.forward import data_residual_zscores

    ds, truth = h3_gate_dataset()
    d_true = jnp.asarray(np.stack(truth.components))

    def chi2(h3):
        problem = build_problem(
            GATE_GRID,
            ds,
            velocities=truth.velocities,
            light_fractions=ELL,
            lsf_sigma_v={"inst": SIG_GATE},
            lsf_anchors_angstrom={"inst": GATE_ANCHORS},
            lsf_h3=None if h3 is None else {"inst": h3},
        )
        z = data_residual_zscores(problem, d_true)
        return float(np.sum(np.square(z)))

    c_true = chi2(list(H3_TRUE))
    c_none = chi2(None)
    assert c_true < c_none - 100.0, f"true h3 chi2 {c_true:.1f} vs symmetric {c_none:.1f}"


def test_closed_loop_h3_joint_fit_leaves_orbit_unbiased():
    """The D38 gate: h3 anchors free against injected asymmetry — the orbit holds.

    Asserted: the orbit, and the fitted h3 staying interior to its bound. The
    injected profile's recovery is deliberately NOT asserted — measured here, the
    free spectra absorb the injected gradient entirely (fitted h3 came back at the
    [-0.03, +0.03] level against an injected [-0.12, 0, +0.12] ramp, with the orbit
    unharmed). This is *stronger* absorption than D37's widths: an asymmetry
    profile imprints a static centroid-warp field c(lambda) ~ sqrt(3) h3 sigma,
    and a free spectrum can represent any static warp outright — the data-identified
    remainder is the epoch-coupled sampling term ~ c'(lambda) * lambda * (v - v_b)/c,
    tens of m/s at this scale, far below the noise. Fitted h3 profiles are therefore
    diagnostics even more thoroughly than fitted widths; the orbit's response is the
    only readout that matters (the fixed-spectra data-term test above pins that the
    injection itself is real and seen).
    """
    import numpyro.distributions as dist

    from albireo.inference import MarginalOrbitModel, run_map

    ds, _ = h3_gate_dataset()
    model = MarginalOrbitModel(
        GATE_GRID,
        ds,
        light_fractions=ELL,
        lsf_sigma_v={"inst": 10.5},
        lsf_anchors_angstrom={"inst": GATE_ANCHORS},
        v_rel_max_kms=float(K_TRUE.sum()) * (1 + ECC_TRUE) * 1.35,
        prior=GATE_PRIOR,
    )
    priors = {
        "period": dist.Normal(P_TRUE + 0.001, 0.003),
        "t_conj": dist.Normal(TCONJ_TRUE + 0.005, 0.02),
        "secosw": dist.Uniform(-1.0, 1.0),
        "sesinw": dist.Uniform(-1.0, 1.0),
        "k": dist.Uniform(jnp.array([10.0, 5.0]), jnp.array([45.0, 40.0])),
        "lsf_sigma": dist.Uniform(jnp.full(3, 2.0), jnp.full(3, 10.5)).to_event(1),
        "lsf_h3": dist.Uniform(jnp.full(3, -0.2), jnp.full(3, 0.2)).to_event(1),
    }
    init = {
        "period": P_TRUE + 0.001,
        "t_conj": TCONJ_TRUE + 0.005,
        "secosw": np.sqrt(0.15) * np.cos(0.5),
        "sesinw": np.sqrt(0.15) * np.sin(0.5),
        "k": jnp.array([27.0, 25.0]),
        "lsf_sigma": jnp.full(3, 7.0),
        "lsf_h3": jnp.zeros(3),
    }
    fit = run_map(model.model(priors), init=init, max_steps=300)
    np.testing.assert_allclose(np.asarray(fit.params["k"]), K_TRUE, rtol=1e-2)
    np.testing.assert_allclose(float(fit.params["period"]), P_TRUE, atol=2e-3)
    h3 = np.asarray(fit.params["lsf_h3"])
    assert np.all(np.abs(h3) < 0.15), f"h3 should stay interior (absorbed, not pegged): {h3}"

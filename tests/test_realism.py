"""Closed-loop tests for the M4 realism features.

One closed loop per feature (the M4 acceptance gate, internal/design.md §8):
hierarchical SB3 orbits, per-epoch light-fraction inference (the eclipse mode of
math.md §5.2, with the breaker *inferred* rather than fixed), and multi-instrument
LSF-width inference. Exactness/unit tests for the new θ-paths ride along.

The telluric closed loop lives in test_telluric.py; the K2 scan in test_scan.py.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro.infer.util import log_density

import albireo as ab
from albireo.inference import MarginalOrbitModel, orbit_velocities, run_map
from albireo.kepler import radial_velocity, t_peri_from_t_conj
from albireo.priors import SmoothnessPrior
from albireo.simulate import InstrumentSpec, OrbitParams, simulate_dataset
from albireo.simulate import synthetic_deviation_spectrum as synth_spectrum

# ---------------------------------------------------------------------------
# Shared gate-scale configuration (inner orbit identical to test_inference.py)
# ---------------------------------------------------------------------------

GRID = ab.LogGrid.from_wavelength_range(5000.0, 5045.0, dv_kms=5.5)
P_IN, TCONJ_IN, ECC_IN, OMEGA_IN = 6.31, 2.05, 0.2, 0.7
K_IN = np.array([30.0, 22.0])
P_OUT, TCONJ_OUT, ECC_OUT, OMEGA_OUT = 47.3, 15.0, 0.12, -1.1
K_OUT = np.array([14.0, 26.0])
WAVE_NATIVE = np.arange(5003.0, 5042.0, 0.11)


def _se(ecc, omega):
    return np.sqrt(ecc) * np.cos(omega), np.sqrt(ecc) * np.sin(omega)


def _theta_inner(k=K_IN):
    h, s = _se(ECC_IN, OMEGA_IN)
    return {
        "period": jnp.asarray(P_IN),
        "t_conj": jnp.asarray(TCONJ_IN),
        "secosw": jnp.asarray(h),
        "sesinw": jnp.asarray(s),
        "k": jnp.asarray(k),
    }


def _theta_sb3():
    h, s = _se(ECC_OUT, OMEGA_OUT)
    return {
        **_theta_inner(),
        "period_out": jnp.asarray(P_OUT),
        "t_conj_out": jnp.asarray(TCONJ_OUT),
        "secosw_out": jnp.asarray(h),
        "sesinw_out": jnp.asarray(s),
        "k_out": jnp.asarray(K_OUT),
    }


def _gate_spectra(seeds):
    return [
        synth_spectrum(GRID, n_lines=26, depth_range=(0.1, 0.7), sigma_v_range=(9.0, 20.0), seed=s)
        for s in seeds
    ]


ORBIT_PRIORS = {
    "period": dist.Normal(P_IN + 0.001, 0.003),
    "t_conj": dist.Normal(TCONJ_IN + 0.005, 0.02),
    "secosw": dist.Uniform(-1.0, 1.0),
    "sesinw": dist.Uniform(-1.0, 1.0),
    "k": dist.Uniform(jnp.array([10.0, 5.0]), jnp.array([45.0, 40.0])),
}
ORBIT_INIT = {
    "period": P_IN + 0.001,
    "t_conj": TCONJ_IN + 0.005,
    "secosw": np.sqrt(0.15) * np.cos(0.5),
    "sesinw": np.sqrt(0.15) * np.sin(0.5),
    "k": jnp.array([28.0, 24.0]),
}


def _hyper_priors(n_comp):
    return {
        "log_tau": dist.Normal(jnp.full(n_comp, np.log(300.0)), 3.0),
        "log_eta": dist.Normal(jnp.full(n_comp, np.log(5.0)), 3.0),
    }


def _hyper_init(n_comp):
    return {"log_tau": jnp.full(n_comp, np.log(300.0)), "log_eta": jnp.full(n_comp, np.log(5.0))}


# ---------------------------------------------------------------------------
# Hierarchical SB3: parameterization units
# ---------------------------------------------------------------------------


def test_sb3_velocities_match_hand_composed():
    bjd = jnp.asarray(np.linspace(0.0, 2.0 * P_OUT, 33))
    got = np.asarray(orbit_velocities(_theta_sb3(), bjd))
    tp_in = t_peri_from_t_conj(TCONJ_IN, period=P_IN, ecc=ECC_IN, omega=OMEGA_IN)
    tp_out = t_peri_from_t_conj(TCONJ_OUT, period=P_OUT, ecc=ECC_OUT, omega=OMEGA_OUT)

    def rv(period, t_peri, ecc, omega, k):
        return np.asarray(
            radial_velocity(bjd, period=period, t_peri=t_peri, ecc=ecc, omega=omega, k=k)
        )

    v1 = rv(P_IN, tp_in, ECC_IN, OMEGA_IN, K_IN[0])
    v2 = rv(P_IN, tp_in, ECC_IN, OMEGA_IN + np.pi, K_IN[1])
    v_com = rv(P_OUT, tp_out, ECC_OUT, OMEGA_OUT, K_OUT[0])
    v3 = rv(P_OUT, tp_out, ECC_OUT, OMEGA_OUT + np.pi, K_OUT[1])
    np.testing.assert_allclose(got, np.stack([v1 + v_com, v2 + v_com, v3]), atol=1e-12)


def test_outer_sites_all_or_none():
    theta = dict(_theta_sb3())
    del theta["sesinw_out"]
    with pytest.raises(ValueError, match="all of"):
        orbit_velocities(theta, np.arange(4.0))


def test_k_out_needs_two_entries():
    theta = {**_theta_sb3(), "k_out": jnp.asarray([14.0])}
    with pytest.raises(ValueError, match="two entries"):
        orbit_velocities(theta, np.arange(4.0))


# ---------------------------------------------------------------------------
# θ-path exactness and gradients for the light / lsf_sigma sites (small problem)
# ---------------------------------------------------------------------------

SMALL_GRID = ab.LogGrid.from_wavelength_range(4500.0, 4512.0, dv_kms=6.0)


@pytest.fixture(scope="module")
def small_data():
    comps = [synth_spectrum(SMALL_GRID, n_lines=6, seed=s, margin=0.1) for s in (1, 2)]
    orbit = OrbitParams(
        period=P_IN,
        t_peri=float(t_peri_from_t_conj(TCONJ_IN, period=P_IN, ecc=ECC_IN, omega=OMEGA_IN)),
        ecc=ECC_IN,
        omega=OMEGA_IN,
        k=tuple(K_IN),
    )
    ds, _ = simulate_dataset(
        SMALL_GRID,
        comps,
        bjd=np.array([0.7, 2.2, 4.9, 6.0]),
        instruments={
            "A": InstrumentSpec(wave=np.arange(4502.0, 4510.0, 0.08), sigma_v_lsf=8.25, snr=60.0)
        },
        light_fractions=[0.6, 0.4],
        orbit=orbit,
        v_bary=np.array([10.0, -15.0, 20.0, 0.0]),
        frame="topocentric",
        seed=4,
    )
    return ds


def _small_model(ds, sigma_v, prior=None):
    return MarginalOrbitModel(
        SMALL_GRID,
        ds,
        light_fractions=(0.6, 0.4),
        lsf_sigma_v={"A": sigma_v},
        v_rel_max_kms=float(K_IN.sum()) * (1 + ECC_IN) * 1.35,
        prior=prior,
    )


def test_light_site_matches_fresh_model(small_data):
    prior = SmoothnessPrior(tau=jnp.array([300.0, 300.0]), eta=jnp.array([5.0, 5.0]))
    model = _small_model(small_data, 8.25, prior)
    ell1 = np.array([0.52, 0.66, 0.58, 0.61])
    theta = {**_theta_inner(), "light": jnp.stack([ell1, 1.0 - ell1], axis=1)}
    via_site = float(model.log_likelihood(theta))
    fresh = MarginalOrbitModel(
        SMALL_GRID,
        small_data,
        light_fractions=np.stack([ell1, 1.0 - ell1]),
        lsf_sigma_v={"A": 8.25},
        v_rel_max_kms=float(K_IN.sum()) * (1 + ECC_IN) * 1.35,
        prior=prior,
    )
    np.testing.assert_allclose(via_site, float(fresh.log_likelihood(_theta_inner())), rtol=1e-12)


def test_lsf_site_matches_fresh_model_at_same_radius(small_data):
    # 7.7 km/s at dv = 6 has the same natural kernel radius as the 8.25 build bound,
    # so the θ-path and a fresh build must agree to machine precision.
    prior = SmoothnessPrior(tau=jnp.array([300.0, 300.0]), eta=jnp.array([5.0, 5.0]))
    model = _small_model(small_data, 8.25, prior)
    theta = {**_theta_inner(), "lsf_sigma": jnp.asarray([7.7])}
    via_site = float(model.log_likelihood(theta))
    fresh = _small_model(small_data, 7.7, prior)
    assert fresh.problem.kernel_radius == model.problem.kernel_radius
    np.testing.assert_allclose(via_site, float(fresh.log_likelihood(_theta_inner())), rtol=1e-12)


def test_gradients_light_lsf_match_finite_differences(small_data):
    prior = SmoothnessPrior(tau=jnp.array([300.0, 300.0]), eta=jnp.array([5.0, 5.0]))
    model = _small_model(small_data, 8.25, prior)
    ell1 = np.array([0.52, 0.66, 0.58, 0.61])
    theta = {
        **_theta_inner(),
        "light": jnp.stack([ell1, 1.0 - ell1], axis=1),
        "lsf_sigma": jnp.asarray([7.0]),
    }
    fun = jax.jit(jax.value_and_grad(lambda th: model._marginal(th).log_likelihood))
    _, grad = fun(theta)

    def loglike_at(site, index, value):
        th = dict(theta)
        th[site] = th[site].at[index].set(value)
        return float(fun(th)[0])

    for site, index, h in [("light", (1, 0), 1e-6), ("lsf_sigma", (0,), 1e-4)]:
        x0 = float(theta[site][index])
        fd = (loglike_at(site, index, x0 + h) - loglike_at(site, index, x0 - h)) / (2 * h)
        np.testing.assert_allclose(float(grad[site][index]), fd, rtol=1e-4)


def test_guards_reject_wide_lsf_and_outer_disk(small_data):
    model = _small_model(small_data, 8.25)
    priors = {
        **ORBIT_PRIORS,
        **_hyper_priors(2),
        "lsf_sigma": dist.Uniform(jnp.asarray([3.0]), jnp.asarray([9.5])),
    }
    init = {**ORBIT_INIT, **_hyper_init(2), "lsf_sigma": jnp.asarray([7.0])}
    ld, _ = log_density(model.model(priors), (), {}, init)
    assert np.isfinite(float(ld))
    # LSF width above the build bound: kernel would be truncated -> rejected
    ld, _ = log_density(model.model(priors), (), {}, {**init, "lsf_sigma": jnp.asarray([8.6])})
    assert not np.isfinite(float(ld))
    # outer eccentricity off the unit disk -> rejected (outer sites imply a
    # tertiary, so the model must be built with three light fractions)
    model3 = MarginalOrbitModel(
        SMALL_GRID,
        small_data,
        light_fractions=(0.5, 0.3, 0.2),
        lsf_sigma_v={"A": 8.25},
        v_rel_max_kms=110.0,
    )
    priors_out = {
        **ORBIT_PRIORS,
        **_hyper_priors(3),
        "period_out": dist.Normal(P_OUT, 0.5),
        "t_conj_out": dist.Normal(TCONJ_OUT, 1.0),
        "secosw_out": dist.Uniform(-1.0, 1.0),
        "sesinw_out": dist.Uniform(-1.0, 1.0),
        "k_out": dist.Uniform(jnp.array([3.0, 5.0]), jnp.array([30.0, 45.0])),
    }
    init_out = {
        **ORBIT_INIT,
        **_hyper_init(3),
        "period_out": P_OUT,
        "t_conj_out": TCONJ_OUT,
        "secosw_out": 0.99,
        "sesinw_out": 0.2,
        "k_out": jnp.array([5.0, 8.0]),
    }
    ld, _ = log_density(model3.model(priors_out), (), {}, init_out)
    assert not np.isfinite(float(ld))
    # a component-count mismatch is a clear error, not a shape crash deep inside
    with pytest.raises(ValueError, match="light fractions"):
        model.log_likelihood({**_theta_sb3(), "log_tau": jnp.zeros(2), "log_eta": jnp.zeros(2)})


# ---------------------------------------------------------------------------
# Closed loop 1: hierarchical SB3 (MAP)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sb3_fit():
    rng = np.random.default_rng(44)
    n_ep = 14
    bjd = np.sort(rng.uniform(0.0, 2.2 * P_OUT, n_ep))
    v_bary = rng.uniform(-25.0, 25.0, n_ep)
    vel_true = np.asarray(orbit_velocities(_theta_sb3(), bjd))
    comps = _gate_spectra((1, 2, 4))
    ell = np.array([0.5, 0.3, 0.2])
    spec = InstrumentSpec(wave=WAVE_NATIVE, sigma_v_lsf=7.0, snr=130.0)
    ds, _ = simulate_dataset(
        GRID,
        comps,
        bjd=bjd,
        instruments={"inst": spec},
        light_fractions=ell,
        velocities=vel_true,
        v_bary=v_bary,
        frame="topocentric",
        gap_fraction=0.01,
        cosmic_fraction=0.002,
        seed=13,
    )
    model = MarginalOrbitModel(
        GRID, ds, light_fractions=ell, lsf_sigma_v={"inst": 7.0}, v_rel_max_kms=95.0
    )
    priors = {
        **ORBIT_PRIORS,
        **_hyper_priors(3),
        "period_out": dist.Normal(P_OUT + 0.05, 0.5),
        "t_conj_out": dist.Normal(TCONJ_OUT + 0.1, 1.0),
        "secosw_out": dist.Uniform(-1.0, 1.0),
        "sesinw_out": dist.Uniform(-1.0, 1.0),
        "k_out": dist.Uniform(jnp.array([3.0, 5.0]), jnp.array([30.0, 45.0])),
    }
    init = {
        **ORBIT_INIT,
        **_hyper_init(3),
        "period_out": P_OUT + 0.05,
        "t_conj_out": TCONJ_OUT + 0.1,
        "secosw_out": np.sqrt(0.08) * np.cos(-0.8),
        "sesinw_out": np.sqrt(0.08) * np.sin(-0.8),
        "k_out": jnp.array([12.0, 28.0]),
    }
    fit = run_map(model.model(priors), init=init, max_steps=250)
    return model, comps, ell, fit


@pytest.mark.slow
def test_sb3_map_recovers_inner_and_outer(sb3_fit):
    _, _, _, fit = sb3_fit
    np.testing.assert_allclose(np.asarray(fit.params["k"]), K_IN, rtol=0.02)
    np.testing.assert_allclose(np.asarray(fit.params["k_out"]), K_OUT, rtol=0.02)
    np.testing.assert_allclose(float(fit.params["ecc"]), ECC_IN, atol=0.03)
    np.testing.assert_allclose(float(fit.params["ecc_out"]), ECC_OUT, atol=0.05)
    np.testing.assert_allclose(float(fit.params["period_out"]), P_OUT, rtol=2e-3)


@pytest.mark.slow
def test_sb3_spectra_recovered(sb3_fit):
    model, comps, ell, fit = sb3_fit
    theta = {
        s: jnp.asarray(fit.params[s])
        for s in (
            "period",
            "t_conj",
            "secosw",
            "sesinw",
            "k",
            "period_out",
            "t_conj_out",
            "secosw_out",
            "sesinw_out",
            "k_out",
            "log_tau",
            "log_eta",
        )
    }
    d_hat = np.asarray(model.marginal(theta).d_hat)
    td = np.stack([np.asarray(c) for c in comps])
    core = (td < -0.15).any(axis=0)
    obs_err = ell @ d_hat - ell @ td
    assert np.sqrt(np.mean(obs_err[core] ** 2)) < 0.02


# ---------------------------------------------------------------------------
# Closed loop 2: per-epoch light fractions (the inferred eclipse breaker)
# ---------------------------------------------------------------------------

ECLIPSE_EPOCHS = [2, 5, 9]
ECLIPSE_ELL1 = [0.45, 0.35, 0.50]


@pytest.fixture(scope="module")
def light_fit():
    rng = np.random.default_rng(42)
    n_ep = 12
    bjd = np.sort(rng.uniform(0.0, 2.4 * P_IN, n_ep))
    v_bary = rng.uniform(-25.0, 25.0, n_ep)
    ell1 = np.full(n_ep, 0.62)
    ell1[ECLIPSE_EPOCHS] = ECLIPSE_ELL1
    ell = np.stack([ell1, 1.0 - ell1])
    comps = _gate_spectra((1, 2))
    orbit = OrbitParams(
        period=P_IN,
        t_peri=float(t_peri_from_t_conj(TCONJ_IN, period=P_IN, ecc=ECC_IN, omega=OMEGA_IN)),
        ecc=ECC_IN,
        omega=OMEGA_IN,
        k=tuple(K_IN),
    )
    spec = InstrumentSpec(wave=WAVE_NATIVE, sigma_v_lsf=7.0, snr=130.0)
    ds, _ = simulate_dataset(
        GRID,
        comps,
        bjd=bjd,
        instruments={"inst": spec},
        light_fractions=ell,
        orbit=orbit,
        v_bary=v_bary,
        frame="topocentric",
        gap_fraction=0.01,
        cosmic_fraction=0.002,
        seed=11,
    )
    model = MarginalOrbitModel(
        GRID,
        ds,
        light_fractions=(0.62, 0.38),
        lsf_sigma_v={"inst": 7.0},
        v_rel_max_kms=float(K_IN.sum()) * (1 + ECC_IN) * 1.35,
    )
    priors = {
        **ORBIT_PRIORS,
        **_hyper_priors(2),
        "light": dist.Dirichlet(jnp.ones(2)).expand([n_ep]),
    }
    init = {**ORBIT_INIT, **_hyper_init(2), "light": jnp.full((n_ep, 2), 0.5)}
    fit = run_map(model.model(priors), init=init, max_steps=250)
    return model, comps, ell1, fit


@pytest.mark.slow
def test_per_epoch_light_recovered(light_fit):
    _, _, ell1_true, fit = light_fit
    ell1_hat = np.asarray(fit.params["light"])[:, 0]
    assert np.sqrt(np.mean((ell1_hat - ell1_true) ** 2)) < 0.01
    np.testing.assert_allclose(ell1_hat[ECLIPSE_EPOCHS], ECLIPSE_ELL1, atol=0.02)
    np.testing.assert_allclose(np.asarray(fit.params["k"]), K_IN, rtol=0.015)


@pytest.mark.slow
def test_eclipse_epochs_break_additive_indeterminacy(light_fit):
    # With per-epoch light fractions *inferred*, the k = 0 invisible direction of
    # the constant-light case (math.md §5.2) becomes observable: each component is
    # recovered individually, not just the light-weighted sum.
    model, comps, _, fit = light_fit
    theta = {
        s: jnp.asarray(fit.params[s])
        for s in ("period", "t_conj", "secosw", "sesinw", "k", "light", "log_tau", "log_eta")
    }
    d_hat = np.asarray(model.marginal(theta).d_hat)
    td = np.stack([np.asarray(c) for c in comps])
    core = (td < -0.15).any(axis=0)
    for i in range(2):
        assert np.sqrt(np.mean((d_hat[i] - td[i])[core] ** 2)) < 0.03


# ---------------------------------------------------------------------------
# Closed loop 3: multi-instrument LSF widths (ML-II)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lsf_fit():
    rng = np.random.default_rng(43)
    n_ep = 12
    bjd = np.sort(rng.uniform(0.0, 2.4 * P_IN, n_ep))
    v_bary = rng.uniform(-25.0, 25.0, n_ep)
    comps = _gate_spectra((1, 2))
    orbit = OrbitParams(
        period=P_IN,
        t_peri=float(t_peri_from_t_conj(TCONJ_IN, period=P_IN, ecc=ECC_IN, omega=OMEGA_IN)),
        ecc=ECC_IN,
        omega=OMEGA_IN,
        k=tuple(K_IN),
    )
    inst = {
        "A": InstrumentSpec(wave=WAVE_NATIVE, sigma_v_lsf=6.0, snr=130.0),
        "B": InstrumentSpec(wave=np.arange(5003.5, 5041.5, 0.13), sigma_v_lsf=10.0, snr=110.0),
    }
    ds, _ = simulate_dataset(
        GRID,
        comps,
        bjd=bjd,
        instruments=inst,
        epoch_instruments=["A", "B"] * (n_ep // 2),
        light_fractions=(0.62, 0.38),
        orbit=orbit,
        v_bary=v_bary,
        frame="topocentric",
        seed=12,
    )
    model = MarginalOrbitModel(
        GRID,
        ds,
        light_fractions=(0.62, 0.38),
        lsf_sigma_v={"A": 9.0, "B": 14.0},
        v_rel_max_kms=float(K_IN.sum()) * (1 + ECC_IN) * 1.35,
    )
    # The absolute LSF width is degenerate with the intrinsic line widths in a
    # template-free model (verified: both-free ML-II inflates both widths by tens
    # of percent while K's stay sub-0.2%). The supported workflow anchors one
    # *reference* instrument with a tight prior; cross-instrument spectrum sharing
    # then identifies the others. Mirrors the light-ratio policy (D13).
    order = list(model.instruments)
    lo = {"A": 5.999, "B": 6.0}
    hi = {"A": 6.001, "B": 14.0}
    priors = {
        **ORBIT_PRIORS,
        **_hyper_priors(2),
        "lsf_sigma": dist.Uniform(
            jnp.asarray([lo[i] for i in order]), jnp.asarray([hi[i] for i in order])
        ),
    }
    init = {
        **ORBIT_INIT,
        **_hyper_init(2),
        "lsf_sigma": jnp.asarray([6.0 if i == "A" else 0.5 * (lo[i] + hi[i]) for i in order]),
    }
    fit = run_map(model.model(priors), init=init, max_steps=250)
    return model, fit


@pytest.mark.slow
def test_lsf_width_recovered_against_reference_instrument(lsf_fit):
    model, fit = lsf_fit
    true_sigma = {"A": 6.0, "B": 10.0}
    sig_hat = np.asarray(fit.params["lsf_sigma"])
    for j, name in enumerate(model.instruments):
        np.testing.assert_allclose(sig_hat[j], true_sigma[name], rtol=0.03)
    np.testing.assert_allclose(np.asarray(fit.params["k"]), K_IN, rtol=0.015)

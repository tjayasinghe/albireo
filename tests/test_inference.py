"""Tests for joint inference (M3): θ-path exactness, gradients, MAP/ML-II, NUTS gate.

The NUTS closed-loop test is the M3 acceptance gate: posterior means of K_1, K_2
within 1% of truth with a converged, divergence-free chain (``internal/design.md`` §8).
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro.infer import init_to_value
from numpyro.infer.util import initialize_model, log_density

import albireo as ab
from albireo.forward import build_problem, with_velocities
from albireo.inference import (
    MarginalOrbitModel,
    laplace_inverse_mass,
    orbit_parameters,
    orbit_velocities,
    posterior_spectra,
    run_map,
    run_nuts,
)
from albireo.kepler import t_peri_from_t_conj
from albireo.likelihood import marginal_loglikelihood
from albireo.priors import SmoothnessPrior
from albireo.simulate import InstrumentSpec, OrbitParams, simulate_dataset
from albireo.simulate import synthetic_deviation_spectrum as synth_spectrum

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Gate-scale closed-loop configuration (shared by the MAP and NUTS tests)
# ---------------------------------------------------------------------------

GRID = ab.LogGrid.from_wavelength_range(5000.0, 5045.0, dv_kms=5.5)
P_TRUE, TCONJ_TRUE, ECC_TRUE, OMEGA_TRUE = 6.31, 2.05, 0.2, 0.7
K_TRUE = np.array([30.0, 22.0])
ELL = np.array([0.62, 0.38])
LSF = {"inst": 7.0}
N_EP = 12


def _theta(period=P_TRUE, t_conj=TCONJ_TRUE, ecc=ECC_TRUE, omega=OMEGA_TRUE, k=K_TRUE):
    return {
        "period": jnp.asarray(period),
        "t_conj": jnp.asarray(t_conj),
        "secosw": jnp.asarray(np.sqrt(ecc) * np.cos(omega)),
        "sesinw": jnp.asarray(np.sqrt(ecc) * np.sin(omega)),
        "k": jnp.asarray(k),
    }


def _gate_orbit():
    tperi = float(t_peri_from_t_conj(TCONJ_TRUE, period=P_TRUE, ecc=ECC_TRUE, omega=OMEGA_TRUE))
    return OrbitParams(period=P_TRUE, t_peri=tperi, ecc=ECC_TRUE, omega=OMEGA_TRUE, k=tuple(K_TRUE))


@pytest.fixture(scope="module")
def gate_data():
    rng = np.random.default_rng(42)
    comps = [
        synth_spectrum(GRID, n_lines=30, depth_range=(0.1, 0.7), sigma_v_range=(9.0, 20.0), seed=1),
        synth_spectrum(GRID, n_lines=25, depth_range=(0.1, 0.7), sigma_v_range=(9.0, 20.0), seed=2),
    ]
    bjd = np.sort(rng.uniform(0.0, 2.4 * P_TRUE, N_EP))
    v_bary = rng.uniform(-25.0, 25.0, N_EP)
    spec = InstrumentSpec(wave=np.arange(5003.0, 5042.0, 0.11), sigma_v_lsf=7.0, snr=130.0)
    ds, truth = simulate_dataset(
        GRID,
        comps,
        bjd=bjd,
        instruments={"inst": spec},
        light_fractions=ELL,
        orbit=_gate_orbit(),
        v_bary=v_bary,
        frame="topocentric",
        gap_fraction=0.01,
        cosmic_fraction=0.002,
        seed=11,
    )
    model = MarginalOrbitModel(
        GRID,
        ds,
        light_fractions=ELL,
        lsf_sigma_v=LSF,
        v_rel_max_kms=float(K_TRUE.sum()) * (1 + ECC_TRUE) * 1.35,
    )
    return ds, truth, model


PRIORS = {
    "period": dist.Normal(P_TRUE + 0.001, 0.003),
    "t_conj": dist.Normal(TCONJ_TRUE + 0.005, 0.02),
    "secosw": dist.Uniform(-1.0, 1.0),
    "sesinw": dist.Uniform(-1.0, 1.0),
    "k": dist.Uniform(jnp.array([10.0, 5.0]), jnp.array([45.0, 40.0])),
    "log_tau": dist.Normal(jnp.full(2, np.log(300.0)), 3.0),
    "log_eta": dist.Normal(jnp.full(2, np.log(5.0)), 3.0),
}
INIT = {
    "period": P_TRUE + 0.001,
    "t_conj": TCONJ_TRUE + 0.005,
    "secosw": np.sqrt(0.15) * np.cos(0.5),
    "sesinw": np.sqrt(0.15) * np.sin(0.5),
    "k": jnp.array([27.0, 25.0]),
    "log_tau": jnp.full(2, np.log(300.0)),
    "log_eta": jnp.full(2, np.log(5.0)),
}


@pytest.fixture(scope="module")
def map_fit(gate_data):
    _, _, model = gate_data
    return run_map(model.model(PRIORS), init=INIT)


@pytest.fixture(scope="module")
def nuts_fit(gate_data, map_fit):
    _, _, model = gate_data
    hyper = {"log_tau": map_fit.params["log_tau"], "log_eta": map_fit.params["log_eta"]}
    orbit_priors = {k: v for k, v in PRIORS.items() if k not in hyper}
    nuts_model = model.model(orbit_priors, fixed=hyper)
    mcmc = run_nuts(
        nuts_model,
        rng_key=jax.random.PRNGKey(3),
        init=map_fit.params,
        inverse_mass_matrix=laplace_inverse_mass(nuts_model, map_fit.params),
        num_warmup=150,
        num_samples=250,
        num_chains=1,
    )
    return mcmc, hyper


# ---------------------------------------------------------------------------
# Parameterization and θ-path exactness
# ---------------------------------------------------------------------------


def test_orbit_parameters_roundtrip():
    par = orbit_parameters(_theta())
    np.testing.assert_allclose(float(par["ecc"]), ECC_TRUE, rtol=1e-14)
    np.testing.assert_allclose(float(par["omega"]), OMEGA_TRUE, rtol=1e-14)
    # eccentricity is clipped at ecc_max before the Kepler solve
    par = orbit_parameters({**_theta(), "secosw": jnp.asarray(0.999), "sesinw": jnp.asarray(0.1)})
    assert float(par["ecc"]) == 0.95


def test_orbit_velocities_match_simulator():
    bjd = np.linspace(0.0, 3.0 * P_TRUE, 40)
    v_theta = np.asarray(orbit_velocities(_theta(), bjd))
    v_sim = _gate_orbit().component_velocities(bjd)
    np.testing.assert_allclose(v_theta, v_sim, atol=1e-10)


@pytest.mark.parametrize("frame", ["topocentric", "barycentric"])
@pytest.mark.parametrize("telluric", [False, True])
def test_with_velocities_matches_build_problem(frame, telluric):
    grid = ab.LogGrid.from_wavelength_range(4500.0, 4512.0, dv_kms=6.0)
    comps = [synth_spectrum(grid, n_lines=6, seed=s, margin=0.1) for s in (1, 2)]
    vel_a = np.array([[20.0, -35.0, 5.0], [-30.0, 50.0, -8.0]])
    vel_b = np.array([[-12.0, 25.0, 0.0], [18.0, -37.5, 1.0]])
    tell = ab.synthetic_telluric_spectrum(grid, seed=3) if telluric else None
    ds, _ = simulate_dataset(
        grid,
        comps,
        bjd=np.arange(3.0),
        velocities=vel_a,
        light_fractions=[0.6, 0.4],
        instruments={
            "A": InstrumentSpec(wave=np.arange(4502.0, 4510.0, 0.08), sigma_v_lsf=6.0, snr=50.0)
        },
        v_bary=np.array([10.0, -15.0, 20.0]),
        frame=frame,
        telluric=tell,
        seed=4,
    )
    kwargs = dict(light_fractions=[0.6, 0.4], lsf_sigma_v={"A": 6.0}, telluric=telluric)
    updated = with_velocities(build_problem(grid, ds, velocities=vel_a, **kwargs), vel_b)
    rebuilt = build_problem(grid, ds, velocities=vel_b, **kwargs)
    for g_up, g_re in zip(updated.groups, rebuilt.groups, strict=True):
        np.testing.assert_allclose(np.asarray(g_up.shifts), np.asarray(g_re.shifts), atol=1e-13)


def test_model_loglike_matches_m2_path(gate_data):
    ds, truth, _ = gate_data
    prior = SmoothnessPrior(tau=jnp.array([1e3, 1e3]), eta=jnp.array([20.0, 20.0]))
    model_fixed = MarginalOrbitModel(
        GRID,
        ds,
        light_fractions=ELL,
        lsf_sigma_v=LSF,
        v_rel_max_kms=float(K_TRUE.sum()) * (1 + ECC_TRUE) * 1.35,
        prior=prior,
    )
    got = model_fixed.marginal(_theta())
    prob_ref = build_problem(
        GRID, ds, velocities=truth.velocities, light_fractions=ELL, lsf_sigma_v=LSF
    )
    ref = marginal_loglikelihood(prob_ref, prior, validate=True)
    # different (but both sufficient) bandwidths: identical answers
    np.testing.assert_allclose(float(got.log_likelihood), float(ref.log_likelihood), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(got.d_hat), np.asarray(ref.d_hat), atol=1e-8)


def test_hyperparameter_theta_matches_fixed_prior(gate_data):
    ds, _, model = gate_data
    tau, eta = np.array([400.0, 900.0]), np.array([7.0, 30.0])
    theta_h = {
        **_theta(),
        "log_tau": jnp.log(jnp.asarray(tau)),
        "log_eta": jnp.log(jnp.asarray(eta)),
    }
    via_theta = float(model.log_likelihood(theta_h))
    model_fixed = MarginalOrbitModel(
        GRID,
        ds,
        light_fractions=ELL,
        lsf_sigma_v=LSF,
        v_rel_max_kms=float(K_TRUE.sum()) * (1 + ECC_TRUE) * 1.35,
        prior=SmoothnessPrior(tau=jnp.asarray(tau), eta=jnp.asarray(eta)),
    )
    np.testing.assert_allclose(via_theta, float(model_fixed.log_likelihood(_theta())), rtol=1e-12)


def test_gradient_matches_finite_differences(gate_data):
    _, _, model = gate_data
    prior_theta = {
        **_theta(),
        "log_tau": jnp.log(jnp.array([1e3, 1e3])),
        "log_eta": jnp.log(jnp.array([20.0, 20.0])),
    }
    fun = jax.jit(jax.value_and_grad(lambda th: model._marginal(th).log_likelihood))
    _, grad = fun(prior_theta)

    def loglike_at(site, value, index=None):
        th = dict(prior_theta)
        th[site] = th[site].at[index].set(value) if index is not None else jnp.asarray(value)
        return float(fun(th)[0])

    for site, index, h in [
        ("k", 0, 1e-4),
        ("t_conj", None, 1e-6),
        ("secosw", None, 1e-6),
        ("log_tau", 1, 1e-4),
    ]:
        x0 = float(prior_theta[site][index] if index is not None else prior_theta[site])
        fd = (loglike_at(site, x0 + h, index) - loglike_at(site, x0 - h, index)) / (2 * h)
        an = float(grad[site][index] if index is not None else grad[site])
        np.testing.assert_allclose(an, fd, rtol=1e-4)


def test_bandwidth_guard_rejects_out_of_bound_orbits(gate_data):
    ds, _, model = gate_data
    numpyro_model = model.model(PRIORS)
    ld, _ = log_density(numpyro_model, (), {}, {**INIT})
    assert np.isfinite(float(ld))
    # The guard fires on the *realized* max relative shift at the observed epochs,
    # not on a (K, e) envelope — so build a configuration whose relative velocity
    # actually exceeds the shift budget at some epoch: max K's, e = 0.6, and omega
    # scanned so a periastron passage lands on an epoch.
    budget_kms = model._shift_bound * GRID.dv_kms
    outside = None
    for omega in np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False):
        cand = {
            **INIT,
            "k": jnp.array([45.0, 40.0]),
            "secosw": jnp.asarray(np.sqrt(0.6) * np.cos(omega)),
            "sesinw": jnp.asarray(np.sqrt(0.6) * np.sin(omega)),
        }
        vel = np.asarray(orbit_velocities(cand, ds.bjd))
        if np.max(np.abs(vel[0] - vel[1])) > budget_kms + 15.0:
            outside = cand
            break
    assert outside is not None, "no epoch-realized violation found; loosen the scan"
    ld, _ = log_density(numpyro_model, (), {}, outside)
    assert not np.isfinite(float(ld))


# ---------------------------------------------------------------------------
# MAP / ML-II and the NUTS gate
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_map_recovers_orbit_and_hyperparameters(map_fit):
    k_map = np.asarray(map_fit.params["k"])
    np.testing.assert_allclose(k_map, K_TRUE, rtol=5e-3)
    np.testing.assert_allclose(float(map_fit.params["ecc"]), ECC_TRUE, atol=0.02)
    np.testing.assert_allclose(float(map_fit.params["omega"]), OMEGA_TRUE, atol=0.05)
    assert np.isfinite(map_fit.potential)
    # ML-II hyperparameters stay in a sane range (truth-ish scales: tau 1e3, eta 20)
    assert np.all(np.asarray(map_fit.params["log_tau"]) > np.log(10.0))
    assert np.all(np.asarray(map_fit.params["log_eta"]) > np.log(0.1))


@pytest.mark.slow
def test_run_map_callback_reports_progress_and_can_stop_early(gate_data):
    """`run_map` is otherwise silent, and on real data it runs for hours.

    Its `tol` is an *absolute* gradient-norm threshold on a potential whose scale grows
    with the good-pixel count, so at survey sizes it is unreachable and `converged` says
    nothing. Watching the parameters is the only way to know what the optimizer is doing.
    """
    _, _, model = gate_data
    seen = []

    def record(step, potential, grad_norm, params):
        seen.append((step, potential, grad_norm, float(np.asarray(params["k"])[0])))
        return step >= 3  # early stop

    fit = run_map(model.model(PRIORS), init=INIT, max_steps=50, tol=0.0, callback=record)
    assert [row[0] for row in seen] == [1, 2, 3]
    assert fit.num_steps == 3, "returning True from the callback must stop the loop"
    assert all(np.isfinite(row[1]) and np.isfinite(row[2]) for row in seen)
    assert all(row[3] > 0.0 for row in seen), "callback receives constrained values"


@pytest.mark.slow
def test_problem_at_exposes_the_problem_for_residuals(gate_data, map_fit):
    """Residual z-scores are the only check there is on estimated inverse variances.

    Getting at them used to need a private method. The scatter is not expected to be
    exactly 1 even on simulated data — the smoothness prior pulls the conditional
    spectra away from the pixels — but it is expected to be *of order* 1, and that is
    what makes it a usable diagnostic for a real dataset whose ivar was estimated.
    """
    from albireo.forward import data_residual_zscores

    ds, _, model = gate_data
    theta = {
        k: v
        for k, v in map_fit.params.items()
        if k in ("period", "t_conj", "secosw", "sesinw", "k", "log_tau", "log_eta")
    }
    problem = model.problem_at(theta)
    assert problem.n_epochs == ds.n_epochs
    z = data_residual_zscores(problem, model.marginal(theta).d_hat)
    assert z.size > 0 and np.all(np.isfinite(z))
    assert 0.5 < float(np.std(z)) < 2.0
    assert abs(float(np.mean(z))) < 0.2


@pytest.mark.slow
def test_nuts_gate_k_within_one_percent(nuts_fit):
    """M3 acceptance gate: K_1, K_2 posterior means within 1%, healthy chain."""
    mcmc, _ = nuts_fit
    samples = mcmc.get_samples()
    extra = mcmc.get_extra_fields()
    k_samples = np.asarray(samples["k"])
    for i in range(2):
        rel_err = abs(k_samples[:, i].mean() - K_TRUE[i]) / K_TRUE[i]
        assert rel_err < 0.01, f"K_{i + 1} off by {100 * rel_err:.2f}% (gate: < 1%)"
        # truth inside the central 95% interval
        lo, hi = np.percentile(k_samples[:, i], [2.5, 97.5])
        assert lo < K_TRUE[i] < hi
    assert int(np.sum(np.asarray(extra["diverging"]))) == 0
    # period and t_conj come along for free
    assert abs(np.asarray(samples["period"]).mean() - P_TRUE) / P_TRUE < 1e-4
    assert abs(np.asarray(samples["t_conj"]).mean() - TCONJ_TRUE) < 0.01


@pytest.mark.slow
def test_posterior_spectra_from_samples(gate_data, nuts_fit):
    _, truth, model = gate_data
    mcmc, hyper = nuts_fit
    draws = posterior_spectra(
        model, mcmc.get_samples(), jax.random.PRNGKey(9), num_draws=8, extra=hyper
    )
    assert draws.shape == (8, 2, GRID.n)
    mean = np.asarray(draws).mean(axis=0)
    truth_d = np.stack([np.asarray(c) for c in truth.components])
    core = (truth_d[0] < -0.15) | (truth_d[1] < -0.15)
    # With constant light fractions the k=0 additive indeterminacy (math.md §5.2,
    # benchmarks.md M2) legitimately inflates each *component* in the invisible
    # direction — the *observable* light-weighted combination must still be tight.
    visible_err = ELL @ mean - ELL @ truth_d
    assert np.sqrt(np.mean(visible_err[core] ** 2)) < 0.02
    for i in range(2):
        assert np.sqrt(np.mean((mean[i][core] - truth_d[i][core]) ** 2)) < 0.15


# ---------------------------------------------------------------------------
# Problem-as-argument through the numpyro path (D32)
# ---------------------------------------------------------------------------


def _largest_const_nbytes(closed_jaxpr) -> int:
    return max((np.asarray(c).nbytes for c in closed_jaxpr.consts), default=0)


def test_model_advertises_its_problem_as_model_args(gate_data):
    _, _, model = gate_data
    numpyro_model = model.model(PRIORS)
    assert len(numpyro_model.model_args) == 1
    assert numpyro_model.model_args[0] is model.problem


def test_potential_with_model_args_embeds_no_problem_constants(gate_data):
    """The numpyro potential must receive the problem as a traced argument.

    Closure-captured, the problem's arrays become jaxpr constants, which jit bakes
    into the HLO as literals — and at survey scale XLA's compile-time folding of
    those literals allocates the multi-GB temporaries D27 documents for the direct
    `marginal` path. The consts are directly observable on the jaxpr, so assert on
    them: with the problem passed as an argument nothing problem-sized may remain,
    and the closure build must show the leak (proving the probe can see it).
    """
    _, _, model = gate_data
    numpyro_model = model.model(PRIORS)
    info = initialize_model(
        jax.random.PRNGKey(0),
        numpyro_model,
        init_strategy=init_to_value(values=INIT),
        dynamic_args=True,
        model_args=(model.problem,),
    )
    z = jax.tree.map(jnp.asarray, info.param_info.z)
    data_nbytes = max(np.asarray(leaf).nbytes for leaf in jax.tree.leaves(model.problem))

    traced = jax.make_jaxpr(lambda zz, pb: info.potential_fn(pb)(zz))(z, model.problem)
    baked = jax.make_jaxpr(lambda zz: info.potential_fn(model.problem)(zz))(z)
    assert _largest_const_nbytes(baked) >= data_nbytes, "the probe cannot see the leak"
    assert _largest_const_nbytes(traced) < data_nbytes


@pytest.mark.slow
def test_run_map_closure_and_argument_paths_agree(gate_data):
    """model_args=() forces the closure path; the traced default must match it.

    The two builds compile different graphs (embedded constants vs. runtime
    parameters), so agreement is to float tolerance, not bitwise.
    """
    _, _, model = gate_data
    via_args = run_map(model.model(PRIORS), init=INIT, max_steps=3, tol=0.0)
    via_closure = run_map(model.model(PRIORS), init=INIT, max_steps=3, tol=0.0, model_args=())
    assert via_args.num_steps == via_closure.num_steps == 3
    np.testing.assert_allclose(via_args.potential, via_closure.potential, rtol=1e-8)
    for site, v in via_args.unconstrained.items():
        np.testing.assert_allclose(
            v, via_closure.unconstrained[site], rtol=1e-6, atol=1e-9, err_msg=site
        )


@pytest.mark.slow
def test_laplace_closure_and_argument_paths_agree(gate_data, map_fit):
    """Both paths run the same eager ops on the same arrays — exact agreement."""
    _, _, model = gate_data
    numpyro_model = model.model(PRIORS)
    via_args = laplace_inverse_mass(numpyro_model, map_fit.params)
    via_closure = laplace_inverse_mass(numpyro_model, map_fit.params, model_args=())
    np.testing.assert_array_equal(via_args, via_closure)

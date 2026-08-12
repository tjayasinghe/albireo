"""Tests for the multiplicative per-epoch response θ-swap (D7's deferral, closed by D33).

The response enters the *targets* ``z = y - r (R 1)`` and the sandwich weights
``w r^2``, not just the forward operator — which is why D7 deferred the swap. What has
to hold, and is pinned here: the swap must equal having built the problem with those
coefficients in the first place (to float precision — ``z`` is updated in place via the
stored response-independent ``base = R 1``); it must replace rather than compound;
masked pixels must stay inert (the D30 ``0 * nan`` trap); gradients must flow; and
maximizing the marginal likelihood over the site must recover injected per-epoch
response perturbations without corrupting the orbit. The epoch-*shared* part of a
low-order response is only weakly identified (it trades against the components' broad
features, design.md §5), so the closed loop asserts sharply on the epoch-to-epoch
*differences* — the thing a per-epoch continuum treatment exists to absorb — and
loosely on the common mode.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

import albireo as ab
from albireo.data import Dataset, EpochData
from albireo.forward import build_problem, with_response, with_velocities
from albireo.inference import MarginalOrbitModel, orbit_velocities, run_map
from albireo.kepler import t_peri_from_t_conj
from albireo.likelihood import marginal_loglikelihood
from albireo.priors import SmoothnessPrior
from albireo.simulate import InstrumentSpec, OrbitParams, chebyshev_response, simulate_dataset
from albireo.simulate import synthetic_deviation_spectrum as synth

SMALL_GRID = ab.LogGrid.from_wavelength_range(5000.0, 5003.0, dv_kms=3.0)
SMALL_VEL = np.array([[9.0, -12.0, 3.0], [-14.0, 18.0, -5.0]])
SMALL_LSF = {"A": 4.0, "B": 9.0}
SMALL_KW = dict(velocities=SMALL_VEL, light_fractions=[0.6, 0.4], lsf_sigma_v=SMALL_LSF)


def small_problem():
    """Three epochs, two instruments, an injected response and some masked pixels."""
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
        cosmic_fraction=0.01,
        seed=9,
    )
    prior = SmoothnessPrior(tau=[2.0, 0.7], eta=[1e-3, 2e-3])
    return ds, truth, prior


# ------------------------------------------------------------------ swap exactness


def test_with_response_matches_fresh_build():
    """The defining identity: the swap == build_problem with those coefficients."""
    ds, truth, prior = small_problem()
    fresh = build_problem(SMALL_GRID, ds, response_coeffs=list(truth.response_coeffs), **SMALL_KW)
    swapped = with_response(
        build_problem(SMALL_GRID, ds, **SMALL_KW), jnp.asarray(np.stack(truth.response_coeffs))
    )
    for gf, gs in zip(fresh.groups, swapped.groups, strict=True):
        np.testing.assert_array_equal(np.asarray(gf.r), np.asarray(gs.r))
        np.testing.assert_allclose(np.asarray(gf.z), np.asarray(gs.z), atol=1e-14)
    want = marginal_loglikelihood(fresh, prior)
    got = marginal_loglikelihood(swapped, prior)
    np.testing.assert_allclose(float(got.log_likelihood), float(want.log_likelihood), rtol=1e-12)
    # z agrees to 1e-16 but re-associated float sums pass through (Lambda + A^T W A)^{-1},
    # whose conditioning amplifies them; observed max 1.5e-9 relative on d_hat.
    np.testing.assert_allclose(np.asarray(got.d_hat), np.asarray(want.d_hat), rtol=1e-8, atol=1e-11)


def test_zero_coefficients_are_the_unit_response():
    """All-zero coefficients must be the identity to the last bit."""
    ds, _, prior = small_problem()
    problem = build_problem(SMALL_GRID, ds, **SMALL_KW)
    same = with_response(problem, jnp.zeros((ds.n_epochs, 3)))
    for g0, g1 in zip(problem.groups, same.groups, strict=True):
        np.testing.assert_array_equal(np.asarray(g0.r), np.asarray(g1.r))
        np.testing.assert_array_equal(np.asarray(g0.z), np.asarray(g1.z))
    base = marginal_loglikelihood(problem, prior)
    got = marginal_loglikelihood(same, prior)
    assert float(got.log_likelihood) == float(base.log_likelihood)


def test_with_response_replaces_rather_than_compounds():
    ds, truth, prior = small_problem()
    problem = build_problem(SMALL_GRID, ds, **SMALL_KW)
    c1 = jnp.asarray([[0.05, -0.02], [0.01, 0.03], [-0.04, 0.02]])
    c2 = jnp.asarray(np.stack(truth.response_coeffs))
    direct = with_response(problem, c2)
    twice = with_response(with_response(problem, c1), c2)
    for gd, gt in zip(direct.groups, twice.groups, strict=True):
        np.testing.assert_array_equal(np.asarray(gd.r), np.asarray(gt.r))
        np.testing.assert_allclose(np.asarray(gd.z), np.asarray(gt.z), atol=1e-14)
    np.testing.assert_allclose(
        float(marginal_loglikelihood(twice, prior).log_likelihood),
        float(marginal_loglikelihood(direct, prior).log_likelihood),
        rtol=1e-12,
    )


def test_shared_coefficients_broadcast_per_epoch():
    ds, _, _ = small_problem()
    problem = build_problem(SMALL_GRID, ds, **SMALL_KW)
    c = jnp.asarray([0.04, -0.01])
    shared = with_response(problem, c)
    tiled = with_response(problem, jnp.tile(c, (ds.n_epochs, 1)))
    for gs, gt in zip(shared.groups, tiled.groups, strict=True):
        np.testing.assert_array_equal(np.asarray(gs.r), np.asarray(gt.r))
        np.testing.assert_array_equal(np.asarray(gs.z), np.asarray(gt.z))


def test_traced_chebyshev_matches_numpy_convention():
    """The swap's Clenshaw must reproduce simulate.chebyshev_response exactly."""
    ds, _, _ = small_problem()
    problem = build_problem(SMALL_GRID, ds, **SMALL_KW)
    rng = np.random.default_rng(3)
    for n_coef in (1, 2, 3, 5):
        c = rng.normal(0.0, 0.05, size=(ds.n_epochs, n_coef))
        swapped = with_response(problem, jnp.asarray(c))
        for g in swapped.groups:
            wave = ds[g.epoch_indices[0]].wave
            for row, j in enumerate(g.epoch_indices):
                np.testing.assert_allclose(
                    np.asarray(g.r)[row],
                    chebyshev_response(wave, c[j]),
                    rtol=0.0,
                    atol=1e-15,
                )


def test_bad_shapes_are_rejected():
    ds, _, _ = small_problem()
    problem = build_problem(SMALL_GRID, ds, **SMALL_KW)
    for bad in (jnp.zeros((ds.n_epochs + 1, 2)), jnp.zeros((ds.n_epochs, 0))):
        try:
            with_response(problem, bad)
        except ValueError:
            continue
        raise AssertionError(f"shape {bad.shape} should have been rejected")


# ------------------------------------------------------------------ masks and gradients


def test_masked_pixel_values_stay_inert_through_the_swap():
    """Garbage (including nan) at zero-weight pixels must not reach the marginal.

    data.py documents masked flux as never read, and the swap rebuilds z — so this
    pins that the rebuild cannot resurrect the D30 ``0 * nan`` trap.
    """
    ds, _, prior = small_problem()
    poisoned_epochs = []
    for ep in ds:
        flux = ep.flux.copy()
        flux[ep.effective_ivar == 0] = np.nan
        poisoned_epochs.append(
            EpochData(
                wave=ep.wave,
                flux=flux,
                ivar=ep.ivar,
                bjd=ep.bjd,
                v_bary=ep.v_bary,
                instrument=ep.instrument,
                mask=ep.mask,
            )
        )
    poisoned = Dataset(epochs=tuple(poisoned_epochs), frame=ds.frame)
    c = jnp.asarray([[0.05, -0.02], [0.01, 0.03], [-0.04, 0.02]])
    clean_ll = marginal_loglikelihood(
        with_response(build_problem(SMALL_GRID, ds, **SMALL_KW), c), prior
    ).log_likelihood
    poisoned_ll = marginal_loglikelihood(
        with_response(build_problem(SMALL_GRID, poisoned, **SMALL_KW), c), prior
    ).log_likelihood
    assert np.isfinite(float(poisoned_ll))
    np.testing.assert_allclose(float(poisoned_ll), float(clean_ll), rtol=1e-12)


def test_gradient_matches_finite_differences():
    ds, truth, prior = small_problem()
    problem = build_problem(SMALL_GRID, ds, **SMALL_KW)
    bandwidth = problem.natural_half_bandwidth
    coeffs = jnp.asarray(np.stack(truth.response_coeffs))

    @jax.jit
    def loglike(pb, c):
        return marginal_loglikelihood(
            with_response(pb, c), prior, half_bandwidth=bandwidth
        ).log_likelihood

    grad = jax.grad(loglike, argnums=1)(problem, coeffs)
    assert bool(jnp.all(jnp.isfinite(grad)))
    c0 = np.asarray(coeffs)
    for j, m in [(0, 0), (1, 1), (2, 0)]:
        h = 1e-6
        cp, cm = c0.copy(), c0.copy()
        cp[j, m] += h
        cm[j, m] -= h
        fd = (
            float(loglike(problem, jnp.asarray(cp))) - float(loglike(problem, jnp.asarray(cm)))
        ) / (2 * h)
        np.testing.assert_allclose(float(grad[j, m]), fd, rtol=1e-4)


# ------------------------------------------------------------------ θ site and closed loop

P_TRUE, TCONJ_TRUE, ECC_TRUE, OMEGA_TRUE = 6.31, 2.05, 0.2, 0.7
K_TRUE = np.array([30.0, 22.0])
ELL = np.array([0.62, 0.38])
GATE_GRID = ab.LogGrid.from_wavelength_range(5000.0, 5045.0, dv_kms=5.5)
N_EP = 10
RESPONSE_ORDER = 2


def _gate_data(response_amplitude):
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
    ds, truth = simulate_dataset(
        GATE_GRID,
        comps,
        bjd=bjd,
        instruments={"inst": spec},
        light_fractions=ELL,
        orbit=orbit,
        v_bary=rng.uniform(-25.0, 25.0, N_EP),
        frame="topocentric",
        response_order=RESPONSE_ORDER,
        response_amplitude=response_amplitude,
        seed=11,
    )
    model = MarginalOrbitModel(
        GATE_GRID,
        ds,
        light_fractions=ELL,
        lsf_sigma_v={"inst": 7.0},
        v_rel_max_kms=float(K_TRUE.sum()) * (1 + ECC_TRUE) * 1.35,
    )
    return ds, truth, model


def _theta(truth_coeffs=None):
    theta = {
        "period": jnp.asarray(P_TRUE),
        "t_conj": jnp.asarray(TCONJ_TRUE),
        "secosw": jnp.asarray(np.sqrt(ECC_TRUE) * np.cos(OMEGA_TRUE)),
        "sesinw": jnp.asarray(np.sqrt(ECC_TRUE) * np.sin(OMEGA_TRUE)),
        "k": jnp.asarray(K_TRUE),
        "log_tau": jnp.full(2, np.log(300.0)),
        "log_eta": jnp.full(2, np.log(5.0)),
    }
    if truth_coeffs is not None:
        theta["response"] = jnp.asarray(truth_coeffs)
    return theta


def test_theta_site_matches_the_direct_route():
    """model.log_likelihood with a response site == with_velocities + with_response."""
    ds, truth, model = _gate_data(response_amplitude=0.03)
    coeffs = np.stack(truth.response_coeffs)
    theta = _theta(coeffs)
    via_site = float(model.log_likelihood(theta))
    direct_problem = with_response(
        with_velocities(model.problem, orbit_velocities(theta, ds.bjd)), jnp.asarray(coeffs)
    )
    direct = float(
        marginal_loglikelihood(
            direct_problem,
            SmoothnessPrior(jnp.exp(theta["log_tau"]), jnp.exp(theta["log_eta"])),
            half_bandwidth=model.half_bandwidth,
        ).log_likelihood
    )
    np.testing.assert_allclose(via_site, direct, rtol=1e-12)


def test_closed_loop_recovers_per_epoch_response_and_orbit():
    """The D33 gate: injected per-epoch response perturbations are inferred jointly.

    Sharp assertion on the epoch-to-epoch *differences* of the coefficients (the
    identifiable direction, and the point of the site); loose on the common mode,
    which legitimately trades against the components' broad features (design.md §5)
    and is pinned only by its zero-centered prior. The orbit must come out at gate
    accuracy alongside, and the fitted response must beat the unit response by a
    decisive margin at the same orbit.
    """
    _, truth, model = _gate_data(response_amplitude=0.03)
    c_true = np.stack(truth.response_coeffs)

    priors = {
        "period": dist.Normal(P_TRUE + 0.001, 0.003),
        "t_conj": dist.Normal(TCONJ_TRUE + 0.005, 0.02),
        "secosw": dist.Uniform(-1.0, 1.0),
        "sesinw": dist.Uniform(-1.0, 1.0),
        "k": dist.Uniform(jnp.array([10.0, 5.0]), jnp.array([45.0, 40.0])),
        "log_tau": dist.Normal(jnp.full(2, np.log(300.0)), 3.0),
        "log_eta": dist.Normal(jnp.full(2, np.log(5.0)), 3.0),
        "response": dist.Normal(jnp.zeros((N_EP, RESPONSE_ORDER + 1)), 0.05),
    }
    init = {
        "period": P_TRUE + 0.001,
        "t_conj": TCONJ_TRUE + 0.005,
        "secosw": np.sqrt(0.15) * np.cos(0.5),
        "sesinw": np.sqrt(0.15) * np.sin(0.5),
        "k": jnp.array([27.0, 25.0]),
        "log_tau": jnp.full(2, np.log(300.0)),
        "log_eta": jnp.full(2, np.log(5.0)),
        "response": jnp.zeros((N_EP, RESPONSE_ORDER + 1)),
    }
    fit = run_map(model.model(priors), init=init, max_steps=250)

    np.testing.assert_allclose(np.asarray(fit.params["k"]), K_TRUE, rtol=1e-2)

    c_hat = np.asarray(fit.params["response"])
    assert c_hat.shape == (N_EP, RESPONSE_ORDER + 1)
    diff_err = (c_hat - c_hat.mean(axis=0)) - (c_true - c_true.mean(axis=0))
    assert float(np.sqrt(np.mean(diff_err**2))) < 5e-3, (
        f"difference-mode response error {np.sqrt(np.mean(diff_err**2)):.2e} "
        f"(injected rms {np.sqrt(np.mean(c_true**2)):.2e})"
    )
    # The common mode is the §5-degenerate direction: nearly flat in the likelihood
    # (the components' broad features absorb it, at ML-II hyperparameters happily), it
    # lands within the prior scale of zero rather than at truth — measured ~0.04 rms
    # against a 0.05 prior. Asserted at prior scale to pin that it cannot run away;
    # anyone tightening this below ~2 prior sigmas is testing the prior, not the data.
    common_err = c_hat.mean(axis=0) - c_true.mean(axis=0)
    assert np.all(np.abs(common_err) < 0.1)

    theta_sites = ("period", "t_conj", "secosw", "sesinw", "k", "log_tau", "log_eta", "response")
    theta_hat = {s: jnp.asarray(fit.params[s]) for s in theta_sites}
    with_fit = float(model.log_likelihood(theta_hat))
    without = float(
        model.log_likelihood({**theta_hat, "response": jnp.zeros((N_EP, RESPONSE_ORDER + 1))})
    )
    assert with_fit - without > 100.0, f"response site only bought {with_fit - without:.1f} nats"

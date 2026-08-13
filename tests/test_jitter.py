"""Tests for the per-epoch noise-inflation factor (docs/design.md D15, math.md §1.4).

The point of a jitter site is that archival inverse variances are usually *estimated*
rather than measured, so their overall scale is unknown. Two properties have to hold for
it to be worth having, and both are pinned here: it must be exactly equivalent to having
been handed rescaled inverse variances in the first place, and maximizing the marginal
likelihood over it must recover an injected scale error — including the effective-degrees-
of-freedom correction that the naive "set chi-square per pixel to one" estimator misses.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from scipy.optimize import minimize_scalar

import albireo as ab
from albireo.data import Dataset, EpochData
from albireo.forward import build_problem, data_residual_zscores, with_jitter
from albireo.likelihood import marginal_loglikelihood
from albireo.priors import SmoothnessPrior
from albireo.simulate import InstrumentSpec, simulate_dataset
from albireo.simulate import synthetic_deviation_spectrum as synth

RNG = np.random.default_rng(41)

SMALL_GRID = ab.LogGrid.from_wavelength_range(5000.0, 5003.0, dv_kms=3.0)
SMALL_VEL = np.array([[9.0, -12.0, 3.0], [-14.0, 18.0, -5.0]])
SMALL_LSF = {"A": 4.0, "B": 9.0}


def small_problem():
    """Three epochs, two instruments, a response and some masked pixels."""
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
    problem = build_problem(
        SMALL_GRID,
        ds,
        velocities=SMALL_VEL,
        light_fractions=[0.6, 0.4],
        lsf_sigma_v=SMALL_LSF,
        response_coeffs=list(truth.response_coeffs),
    )
    return ds, truth, problem, SmoothnessPrior(tau=[2.0, 0.7], eta=[1e-3, 2e-3])


def scaled_ivar_dataset(ds: Dataset, factor) -> Dataset:
    """``ds`` with every epoch's inverse variance multiplied by ``factor`` (per epoch)."""
    factor = np.broadcast_to(np.asarray(factor, dtype=float), (ds.n_epochs,))
    epochs = [
        EpochData(
            wave=ep.wave,
            flux=ep.flux,
            ivar=ep.ivar * f,
            bjd=ep.bjd,
            v_bary=ep.v_bary,
            instrument=ep.instrument,
            mask=ep.mask,
        )
        for ep, f in zip(ds, factor, strict=True)
    ]
    return Dataset(epochs=tuple(epochs), frame=ds.frame)


# --------------------------------------------------------------------------- equivalence


def test_unit_jitter_changes_nothing():
    """alpha = 1 must be the identity to the last bit, not merely to a tolerance."""
    _, _, problem, prior = small_problem()
    base = marginal_loglikelihood(problem, prior)
    same = marginal_loglikelihood(with_jitter(problem, 1.0), prior)
    assert float(same.log_likelihood) == float(base.log_likelihood)
    np.testing.assert_array_equal(np.asarray(same.d_hat), np.asarray(base.d_hat))


@pytest.mark.parametrize("alpha", [0.5, 1.7, [1.0, 2.5, 0.8]])
def test_jitter_equals_rescaling_the_inverse_variances(alpha):
    """The defining identity: jitter alpha_j == being handed ivar_j / alpha_j^2.

    Everything in the likelihood that touches the weights has to agree, including the
    ``sum log w`` term — which is computed by a different route with jitter (a per-epoch
    scalar correction) than without (a sum over pixels).
    """
    ds, truth, problem, prior = small_problem()
    a = np.broadcast_to(np.asarray(alpha, dtype=float), (ds.n_epochs,))
    rescaled = build_problem(
        SMALL_GRID,
        scaled_ivar_dataset(ds, 1.0 / a**2),
        velocities=SMALL_VEL,
        light_fractions=[0.6, 0.4],
        lsf_sigma_v=SMALL_LSF,
        response_coeffs=list(truth.response_coeffs),
    )
    want = marginal_loglikelihood(rescaled, prior)
    got = marginal_loglikelihood(with_jitter(problem, alpha), prior)
    np.testing.assert_allclose(float(got.log_likelihood), float(want.log_likelihood), rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(got.d_hat), np.asarray(want.d_hat), rtol=1e-10, atol=1e-12
    )


def test_jitter_leaves_the_raw_weights_alone_and_is_idempotent():
    """`w` stays the measurement; applying a jitter twice replaces, never compounds."""
    _, _, problem, prior = small_problem()
    once = with_jitter(problem, 2.0)
    twice = with_jitter(once, 2.0)
    for g0, g1 in zip(problem.groups, once.groups, strict=True):
        np.testing.assert_array_equal(np.asarray(g1.w), np.asarray(g0.w))
        np.testing.assert_allclose(np.asarray(g1.effective_w), np.asarray(g0.w) / 4.0, rtol=1e-14)
    assert float(marginal_loglikelihood(twice, prior).log_likelihood) == float(
        marginal_loglikelihood(once, prior).log_likelihood
    )


def test_band_and_probe_assembly_agree_under_jitter():
    """The jitter enters band assembly through its own path (`wprime`); check both."""
    _, _, problem, prior = small_problem()
    jittered = with_jitter(problem, [1.3, 0.7, 2.1])
    band = marginal_loglikelihood(jittered, prior, assembly="band", validate=True)
    probe = marginal_loglikelihood(jittered, prior, assembly="probe")
    np.testing.assert_allclose(float(band.log_likelihood), float(probe.log_likelihood), rtol=1e-10)


def test_jitter_rejects_the_wrong_shape():
    _, _, problem, _ = small_problem()
    with pytest.raises(ValueError, match="scalar or have shape"):
        with_jitter(problem, [1.0, 2.0])


def test_jitter_is_differentiable_under_jit():
    """Autodiff vs central differences of d logp / d log alpha, jit-compiled."""
    _, _, problem, prior = small_problem()

    @jax.jit
    def logp(log_alpha, prob):
        jittered = with_jitter(prob, jnp.exp(log_alpha))
        return marginal_loglikelihood(
            jittered, prior, half_bandwidth=problem.natural_half_bandwidth
        ).log_likelihood

    x = jnp.asarray([0.15, -0.3, 0.05])
    grad = np.asarray(jax.grad(logp)(x, problem))
    h = 1e-5
    fd = np.array(
        [
            (float(logp(x.at[i].add(h), problem)) - float(logp(x.at[i].add(-h), problem))) / (2 * h)
            for i in range(x.size)
        ]
    )
    np.testing.assert_allclose(grad, fd, rtol=2e-6, atol=1e-6)


# --------------------------------------------------------- what ML-II actually estimates

# A deliberately over-determined problem: 2400 weighted pixels against 2 x ~121 model
# pixels, with a weak prior so most of those are data-determined. The residuals are then
# close to the injected noise, and the effective parameter count is a visible ~9% of the
# data — enough to separate the corrected estimator from the naive one (which lands 4.6%
# low here) without making either of them meaningless.
BIG_GRID = ab.LogGrid.from_wavelength_range(5000.0, 5010.0, dv_kms=5.0)
BIG_SNR = 60.0
N_BIG_EPOCHS = 12


def big_problem(ivar_inflation: float):
    """Data at SNR 60, but told its inverse variances are ``ivar_inflation`` times larger.

    Claiming ``ivar * f^2`` is claiming errors smaller by ``f``, so the jitter that
    restores the truth is ``alpha = f``.
    """
    comps = [synth(BIG_GRID, n_lines=25, seed=s, margin=0.1) for s in (3, 4)]
    rng = np.random.default_rng(5)
    vel = np.stack(
        [60.0 * np.sin(np.arange(N_BIG_EPOCHS)), -90.0 * np.sin(np.arange(N_BIG_EPOCHS))]
    )
    ds, truth = simulate_dataset(
        BIG_GRID,
        comps,
        bjd=np.arange(float(N_BIG_EPOCHS)),
        velocities=vel,
        light_fractions=[0.55, 0.45],
        instruments={
            "A": InstrumentSpec(wave=np.arange(5001.0, 5009.0, 0.04), sigma_v_lsf=6.0, snr=BIG_SNR)
        },
        epoch_instruments=["A"] * N_BIG_EPOCHS,
        v_bary=rng.uniform(-25.0, 25.0, N_BIG_EPOCHS),
        seed=11,
    )
    problem = build_problem(
        BIG_GRID,
        scaled_ivar_dataset(ds, ivar_inflation**2),
        velocities=vel,
        light_fractions=[0.55, 0.45],
        lsf_sigma_v={"A": 6.0},
    )
    return problem, truth, SmoothnessPrior(tau=[3e-3, 3e-3], eta=[1e-6, 1e-6])


def profile_jitter(problem, prior, bracket=(-1.5, 1.5)):
    """argmax over a single shared log-jitter, plus the value of the marginal there."""

    def negative(log_alpha):
        jittered = with_jitter(problem, float(np.exp(log_alpha)))
        return -float(marginal_loglikelihood(jittered, prior).log_likelihood)

    out = minimize_scalar(negative, bounds=bracket, method="bounded", options={"xatol": 1e-5})
    return float(np.exp(out.x))


@pytest.mark.parametrize("injected", [1.0, 2.0])
def test_ml2_recovers_an_injected_noise_inflation(injected):
    problem, _, prior = big_problem(injected)
    alpha_hat = profile_jitter(problem, prior)
    np.testing.assert_allclose(alpha_hat, injected, rtol=0.04)


def test_the_marginal_supplies_the_effective_dof_correction():
    """The maximizing alpha is *not* the residual standard deviation.

    In the data-dominated limit the marginal's log-determinant terms contribute
    ``+p_eff log alpha`` against the weight term's ``-N log alpha``, so profiling gives
    ``alpha^2 = chi2 / (N - p_eff)`` — the classical dof-corrected variance estimate —
    where whitening the residuals by hand would give ``chi2 / N`` and land low by
    ``sqrt(1 - p_eff/N)``. This test only passes if that correction is really there.
    """
    injected = 2.0
    problem, _, prior = big_problem(injected)
    alpha_hat = profile_jitter(problem, prior)

    fitted = with_jitter(problem, alpha_hat)
    d_hat = marginal_loglikelihood(fitted, prior).d_hat
    # Residuals whitened by the *supplied* (optimistic) weights: their standard deviation
    # is the naive estimator of the same quantity.
    naive = float(np.std(data_residual_zscores(problem, d_hat)))

    # The naive estimator is biased low, and the profiled one is several times closer to
    # the truth — so this is a real comparison, not two estimators tying.
    assert naive < alpha_hat, (naive, alpha_hat)
    assert abs(alpha_hat - injected) < 0.5 * abs(naive - injected), (alpha_hat, naive)

    # Read the implied effective parameter count back out of the two estimators; it should
    # land near the model dimension (2 components x n_pix), since this prior is weak.
    n_data = int(sum(int(np.sum(np.asarray(g.w) > 0)) for g in problem.groups))
    p_eff = n_data * (1.0 - (naive / alpha_hat) ** 2)
    assert 0.5 * 2 * BIG_GRID.n < p_eff < 1.5 * 2 * BIG_GRID.n, (p_eff, n_data, BIG_GRID.n)

    # And the residuals under the fitted weights are calibrated, which is the whole point.
    z = data_residual_zscores(fitted, d_hat)
    assert 0.9 < float(np.std(z)) < 1.02


# ------------------------------------------------------------------- the inference site


def orbit_model_and_theta():
    comps = [synth(SMALL_GRID, n_lines=6, seed=s, margin=0.15) for s in (1, 2)]
    ds, _ = simulate_dataset(
        SMALL_GRID,
        comps,
        bjd=np.arange(4.0),
        velocities=np.array([[10.0, -14.0, 4.0, 8.0], [-15.0, 21.0, -6.0, -12.0]]),
        light_fractions=[0.6, 0.4],
        instruments={
            "A": InstrumentSpec(wave=np.arange(5000.5, 5002.4, 0.055), sigma_v_lsf=4.0, snr=50.0)
        },
        epoch_instruments=["A"] * 4,
        v_bary=np.array([8.0, -20.0, 3.0, 11.0]),
        seed=9,
    )
    model = ab.MarginalOrbitModel(
        SMALL_GRID,
        ds,
        light_fractions=[0.6, 0.4],
        lsf_sigma_v={"A": 4.0},
        v_rel_max_kms=90.0,
        prior=SmoothnessPrior(tau=[2.0, 0.7], eta=[1e-3, 2e-3]),
    )
    theta = {
        "period": jnp.asarray(3.1),
        "t_conj": jnp.asarray(0.4),
        "secosw": jnp.asarray(0.05),
        "sesinw": jnp.asarray(0.02),
        "k": jnp.asarray([18.0, 26.0]),
    }
    return model, theta


def test_log_jitter_site_is_accepted_scalar_and_per_epoch():
    model, theta = orbit_model_and_theta()
    base = float(model.log_likelihood(theta))
    scalar = float(model.log_likelihood({**theta, "log_jitter": jnp.asarray(0.0)}))
    per_epoch = float(model.log_likelihood({**theta, "log_jitter": jnp.zeros(4)}))
    assert scalar == base
    assert per_epoch == base
    assert float(model.log_likelihood({**theta, "log_jitter": jnp.asarray(0.4)})) != base


def test_problem_at_reflects_the_jitter():
    model, theta = orbit_model_and_theta()
    problem = model.problem_at({**theta, "log_jitter": jnp.log(jnp.asarray(2.0))})
    for g in problem.groups:
        np.testing.assert_allclose(np.asarray(g.jitter), 2.0, rtol=1e-12)
        np.testing.assert_allclose(np.asarray(g.effective_w), np.asarray(g.w) / 4.0, rtol=1e-14)


@pytest.mark.slow
def test_log_jitter_is_a_sampleable_site():
    """It has to survive numpyro's site-name validation and one MAP step."""
    model, theta = orbit_model_and_theta()
    priors = {
        "period": dist.Normal(3.1, 0.05),
        "t_conj": dist.Normal(0.4, 0.05),
        "secosw": dist.Normal(0.05, 0.05),
        "sesinw": dist.Normal(0.02, 0.05),
        "k": dist.Normal(jnp.asarray([18.0, 26.0]), 3.0).to_event(1),
        "log_jitter": dist.Normal(0.0, 1.0),
    }
    result = ab.run_map(
        model.model(priors), init={**theta, "log_jitter": jnp.asarray(0.3)}, max_steps=3
    )
    assert np.isfinite(result.potential)
    assert np.isfinite(float(result.params["log_jitter"]))

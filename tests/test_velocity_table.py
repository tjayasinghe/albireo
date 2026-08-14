"""The free per-epoch radial-velocity table (D42).

No Keplerian: every epoch's velocity is its own parameter. Three things have to hold.

1. **The zero point is gone, exactly.** Each component's spectrum can absorb a constant
   added to that component's shifts, so a free table has one arbitrary zero point *per
   component*. albireo removes them in pixel space, where the removal is exact — in
   velocity space it would only be first-order, and the residual would be pinned by
   shift-interpolation error rather than by data. These tests assert the invariance to
   float64 round-off, not to a tolerance.
2. **The table recovers the velocities.** Warm-started from a Keplerian, the fit has to
   land on the injected per-epoch velocities and reproduce the Wilson slope, which is the
   mass ratio and is invariant to both zero points.
3. **The mode is honest about its failure.** From a cold start the problem is genuinely
   multimodal — with every epoch at the same velocity the components are indistinguishable
   — and the test pins that the failure is *detectable* in the potential rather than
   silent.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from jax.flatten_util import ravel_pytree

import albireo as ab
from albireo.inference import _centered_shifts

GRID = ab.LogGrid.from_wavelength_range(5000.0, 5040.0, dv_kms=6.0)
P, ECC, OMEGA, T_PERI = 6.31, 0.15, 0.7, 2.0
K1, K2 = 30.0, 55.0
ELL = (0.6, 0.4)
PRIOR = ab.SmoothnessPrior(jnp.asarray([300.0, 300.0]), jnp.asarray([5.0, 5.0]))
V_REL_MAX = 320.0
LSF = {"a": 7.0}


def _relativistic_add(v, c):
    """The exact group operation: a constant *pixel* offset is this, not v + c."""
    b1, b2 = np.asarray(v) / ab.C_KMS, c / ab.C_KMS
    return ab.C_KMS * (b1 + b2) / (1.0 + b1 * b2)


@pytest.fixture(scope="module")
def sb2():
    rng = np.random.default_rng(7)
    bjd = np.sort(rng.uniform(0.0, 21.0, size=10))
    c1 = ab.synthetic_deviation_spectrum(GRID, seed=21)
    c2 = ab.synthetic_deviation_spectrum(GRID, seed=22)
    inst = {
        "a": ab.InstrumentSpec(wave=np.arange(5003.0, 5037.0, 0.12), sigma_v_lsf=7.0, snr=200.0)
    }
    orbit = ab.OrbitParams(period=P, t_peri=T_PERI, ecc=ECC, omega=OMEGA, k=(K1, K2))
    ds, truth = ab.simulate_dataset(
        GRID, [c1, c2], bjd=bjd, instruments=inst, light_fractions=ELL, orbit=orbit, seed=5
    )
    model = ab.MarginalOrbitModel(
        GRID,
        ds,
        light_fractions=np.asarray(ELL),
        lsf_sigma_v=LSF,
        v_rel_max_kms=V_REL_MAX,
        prior=PRIOR,
    )
    return model, np.asarray(truth.velocities), bjd


# -- 1. the zero point is exactly unidentified -------------------------------


def test_each_component_has_its_own_exactly_flat_zero_point(sb2):
    """Not one flat direction in total (that would be gamma) — one per component."""
    model, v_true, _ = sb2
    ref = float(model.log_likelihood({"velocity": jnp.asarray(v_true)}))
    assert np.isfinite(ref)

    for comp in (0, 1):
        for offset in (5.0, 50.0, 200.0, -120.0):
            v = np.array(v_true, dtype=float)
            v[comp] = _relativistic_add(v[comp], offset)
            shifted = float(model.log_likelihood({"velocity": jnp.asarray(v)}))
            assert abs(shifted - ref) / abs(ref) < 1e-12, (
                f"component {comp + 1} offset {offset} moved the log-likelihood by "
                f"{shifted - ref:.3e}; the zero point must be exactly unidentified"
            )


def test_centering_in_velocity_space_would_not_have_been_exact(sb2):
    """Why the centering lives in pixel space: the naive version leaves a residual.

    ``xi = artanh(v/c)`` makes a constant *pixel* offset a relativistic velocity
    addition, not an ordinary one. Subtracting a mean velocity is right only to first
    order in v/c, and this pins the size of the error that would have been left behind.
    """
    model, v_true, _ = sb2
    ref = float(model.log_likelihood({"velocity": jnp.asarray(v_true)}))

    naive = np.array(v_true, dtype=float)
    naive[0] = naive[0] + 50.0  # ordinary addition, NOT the group operation
    exact = np.array(v_true, dtype=float)
    exact[0] = _relativistic_add(exact[0], 50.0)

    d_naive = abs(float(model.log_likelihood({"velocity": jnp.asarray(naive)})) - ref)
    d_exact = abs(float(model.log_likelihood({"velocity": jnp.asarray(exact)})) - ref)
    assert d_exact < 1e-9
    assert d_naive > 100 * max(d_exact, 1e-12), (
        f"the naive shift should be measurably worse: {d_naive:.3e} vs {d_exact:.3e}"
    )


def test_centered_shifts_have_zero_row_mean_in_pixel_space(sb2):
    _, v_true, _ = sb2
    pix = np.asarray(_centered_shifts(jnp.asarray(v_true), GRID))
    assert pix.shape == v_true.shape
    assert np.max(np.abs(pix.mean(axis=1))) < 1e-12


def test_relative_velocities_preserve_the_variation(sb2):
    """Centering removes a zero point, not the signal the table is for."""
    _, v_true, _ = sb2
    rel = np.asarray(ab.relative_velocities(v_true, GRID))
    for i in range(2):
        assert np.ptp(rel[i]) == pytest.approx(np.ptp(v_true[i]), rel=1e-6)
    # Rows sum to zero in *pixel* space, so not exactly in km/s — the nonlinearity is
    # handled rather than approximated away.
    assert np.max(np.abs(rel.mean(axis=1))) < 0.05


def test_pixels_to_velocity_inverts_velocity_to_pixels():
    v = np.array([-450.0, -3.0, 0.0, 1e-6, 12.5, 800.0])
    back = np.asarray(GRID.pixels_to_velocity(GRID.velocity_to_pixels(jnp.asarray(v))))
    assert np.allclose(back, v, rtol=1e-12, atol=1e-12)

    classical = ab.LogGrid.from_wavelength_range(5000.0, 5040.0, dv_kms=6.0, relativistic=False)
    back = np.asarray(classical.pixels_to_velocity(classical.velocity_to_pixels(jnp.asarray(v))))
    assert np.allclose(back, v, rtol=1e-12, atol=1e-12)


# -- 2. the site, and its guards ---------------------------------------------


def test_velocity_site_rejects_a_keplerian_alongside_it(sb2):
    model, v_true, _ = sb2
    with pytest.raises(ValueError, match="both a free-velocity site and Keplerian"):
        model.log_likelihood({"velocity": jnp.asarray(v_true), "k": jnp.asarray([K1, K2])})
    with pytest.raises(ValueError, match="both a free-velocity site and Keplerian"):
        model.log_likelihood({"velocity": jnp.asarray(v_true), "period": jnp.asarray(P)})


def test_velocity_site_checks_its_shape(sb2):
    model, _, _ = sb2
    with pytest.raises(ValueError, match=r"velocity must have shape \(2, 10\)"):
        model.log_likelihood({"velocity": jnp.zeros((2, 3))})
    with pytest.raises(ValueError, match=r"velocity must have shape \(2, 10\)"):
        model.log_likelihood({"velocity": jnp.zeros((3, 10))})


def test_the_free_velocity_likelihood_is_differentiable(sb2):
    model, v_true, _ = sb2
    grad = jax.grad(lambda v: model._marginal({"velocity": v}).log_likelihood)(jnp.asarray(v_true))
    assert grad.shape == v_true.shape
    assert np.all(np.isfinite(np.asarray(grad)))
    # At the truth the gradient is small but not zero (noise); a finite-difference check
    # on one entry pins that it is the right derivative.
    i, j = 1, 4
    step = 1e-3
    plus = np.array(v_true, dtype=float)
    minus = np.array(v_true, dtype=float)
    plus[i, j] += step
    minus[i, j] -= step
    fd = (
        float(model.log_likelihood({"velocity": jnp.asarray(plus)}))
        - float(model.log_likelihood({"velocity": jnp.asarray(minus)}))
    ) / (2 * step)
    assert float(grad[i, j]) == pytest.approx(fd, rel=2e-3, abs=1e-3)


def test_a_keplerian_model_still_demands_its_orbital_sites(sb2):
    model, _, _ = sb2
    with pytest.raises(ValueError, match="missing orbital sites"):
        model.model({"k": dist.Normal(30.0, 5.0).expand([2]).to_event(1)})
    with pytest.raises(ValueError, match="cannot also carry Keplerian sites"):
        model.model(
            {
                "velocity": dist.Normal(0.0, 80.0).expand([2, 10]).to_event(2),
                "k": dist.Normal(30.0, 5.0).expand([2]).to_event(1),
            }
        )


def test_with_shifts_is_the_pixel_space_core_of_with_velocities(sb2):
    """with_velocities is a wrapper; the two must agree bit for bit."""
    model, v_true, _ = sb2
    by_velocity = ab.with_velocities(model.problem, jnp.asarray(v_true))
    by_pixels = ab.with_shifts(model.problem, GRID.velocity_to_pixels(jnp.asarray(v_true)))
    for ga, gb in zip(by_velocity.groups, by_pixels.groups, strict=True):
        assert np.array_equal(np.asarray(ga.shifts), np.asarray(gb.shifts))
    with pytest.raises(ValueError, match="star_pix must have shape"):
        ab.with_shifts(model.problem, jnp.zeros((2, 3)))


# -- 3. the Keplerian model check --------------------------------------------


def test_keplerian_residuals_vanish_for_the_orbit_that_generated_the_table(sb2):
    """A table built *from* a Keplerian must residual to zero against it."""
    _, v_true, bjd = sb2
    theta = _kepler_theta()
    resid = np.asarray(ab.keplerian_residuals(v_true, theta, bjd, GRID))
    assert resid.shape == v_true.shape
    assert np.max(np.abs(resid)) < 1e-8, f"max residual {np.max(np.abs(resid)):.3e} km/s"


def test_keplerian_residuals_ignore_both_arbitrary_zero_points(sb2):
    """The two tables' zero points must cancel exactly, or the residual is meaningless."""
    _, v_true, bjd = sb2
    theta = _kepler_theta()
    base = np.asarray(ab.keplerian_residuals(v_true, theta, bjd, GRID))
    shifted = np.array(v_true, dtype=float)
    shifted[0] = _relativistic_add(shifted[0], 77.0)
    shifted[1] = _relativistic_add(shifted[1], -31.0)
    moved = np.asarray(ab.keplerian_residuals(shifted, theta, bjd, GRID))
    assert np.max(np.abs(moved - base)) < 1e-9


def test_keplerian_residuals_expose_a_wrong_period(sb2):
    """The point of the check: a period error is structured, not noise-like."""
    _, v_true, bjd = sb2
    good = np.asarray(ab.keplerian_residuals(v_true, _kepler_theta(), bjd, GRID))
    bad = np.asarray(ab.keplerian_residuals(v_true, _kepler_theta(period=P * 1.01), bjd, GRID))
    assert np.max(np.abs(bad)) > 1.0
    assert np.max(np.abs(bad)) > 1e6 * max(np.max(np.abs(good)), 1e-12)


def test_keplerian_residuals_reject_a_mismatched_shape(sb2):
    _, v_true, bjd = sb2
    with pytest.raises(ValueError, match="must agree on both the component count"):
        ab.keplerian_residuals(v_true[:1], _kepler_theta(), bjd, GRID)


def _kepler_theta(period: float = P):
    nu_c = 0.5 * np.pi - OMEGA
    e_c = 2.0 * np.arctan2(
        np.sqrt(1.0 - ECC) * np.sin(0.5 * nu_c), np.sqrt(1.0 + ECC) * np.cos(0.5 * nu_c)
    )
    t_conj = T_PERI + (e_c - ECC * np.sin(e_c)) * period / (2.0 * np.pi)
    return {
        "period": jnp.asarray(period),
        "t_conj": jnp.asarray(t_conj),
        "secosw": jnp.asarray(np.sqrt(ECC) * np.cos(OMEGA)),
        "sesinw": jnp.asarray(np.sqrt(ECC) * np.sin(OMEGA)),
        "k": jnp.asarray([K1, K2]),
    }


# -- 4. the closed loop ------------------------------------------------------


def _PRIORS(n_epochs):
    return {
        "velocity": dist.Normal(0.0, 120.0).expand([2, n_epochs]).to_event(2),
        "log_tau": dist.Normal(5.7, 1.5).expand([2]).to_event(1),
        "log_eta": dist.Normal(1.6, 1.0).expand([2]).to_event(1),
    }


def _fit(model, init_v, bjd, max_steps=250):
    return ab.run_map(
        model.model(_PRIORS(bjd.size)),
        init={
            "velocity": jnp.asarray(init_v),
            "log_tau": jnp.full(2, 5.7),
            "log_eta": jnp.full(2, 1.6),
        },
        max_steps=max_steps,
        model_args=(model.problem,),
    )


@pytest.fixture(scope="module")
def warm_fit(sb2):
    """Warm-started from a badly wrong Keplerian — 30% off in both semi-amplitudes."""
    model, v_true, bjd = sb2
    start = np.stack([v_true[0] * 1.3, v_true[1] * 0.7])
    return _fit(model, start, bjd)


@pytest.mark.slow
def test_free_velocities_recover_the_injected_table(sb2, warm_fit):
    _, v_true, _ = sb2
    rel_true = np.asarray(ab.relative_velocities(v_true, GRID))
    got = np.asarray(ab.relative_velocities(warm_fit.params["velocity"], GRID))
    for i in range(2):
        rms = float(np.sqrt(np.mean((got[i] - rel_true[i]) ** 2)))
        # dv = 6 km/s per model pixel, so this is ~1/40th of a pixel.
        assert rms < 0.15, f"component {i + 1} per-epoch RV rms {rms:.4f} km/s"


@pytest.mark.slow
def test_the_wilson_slope_recovers_the_mass_ratio(sb2, warm_fit):
    """The slope of v_2 against v_1 is -K_2/K_1, and both zero points drop out of it."""
    got = np.asarray(ab.relative_velocities(warm_fit.params["velocity"], GRID))
    slope = float(np.polyfit(got[0], got[1], 1)[0])
    assert slope == pytest.approx(-K2 / K1, rel=0.02)


@pytest.mark.slow
def test_the_recovered_table_threads_the_keplerian_that_made_it(sb2, warm_fit):
    _, _, bjd = sb2
    resid = np.asarray(
        ab.keplerian_residuals(warm_fit.params["velocity"], _kepler_theta(), bjd, GRID)
    )
    assert np.max(np.abs(resid)) < 0.4, f"max |residual| {np.max(np.abs(resid)):.3f} km/s"


@pytest.mark.slow
def test_the_raw_laplace_diagonal_returns_the_prior_not_an_error_bar(sb2, warm_fit):
    """The trap the projection exists to close, pinned as a measurement.

    Each component's zero point is exactly flat, so its posterior width *is* the prior
    width, and every epoch's marginal variance inherits it. With a Normal(0, 120) prior
    over 10 epochs the raw sigma must come out at 120/sqrt(10) on every entry — the same
    number for a good dataset and a useless one, which is what makes reading it so
    dangerous.
    """
    model, _, bjd = sb2
    cov = ab.laplace_inverse_mass(
        model.model(_PRIORS(bjd.size)), warm_fit.params, model_args=(model.problem,)
    )
    marks, _ = ravel_pytree(
        {
            name: jnp.full(jnp.shape(jnp.asarray(v)), 1.0 if name == "velocity" else 0.0)
            for name, v in warm_fit.unconstrained.items()
        }
    )
    sel = np.asarray(marks) > 0.5
    raw = np.sqrt(np.diag(np.asarray(cov)[np.ix_(sel, sel)]))
    assert np.allclose(raw, 120.0 / np.sqrt(bjd.size), rtol=1e-3), (
        f"raw sigmas {raw[:3]} should be the prior width 120/sqrt(n) = "
        f"{120.0 / np.sqrt(bjd.size):.3f}"
    )

    projected = ab.relative_velocity_errors(cov, warm_fit.unconstrained)
    assert projected.shape == (2, bjd.size)
    assert np.all(projected < 0.5), "projected per-epoch errors should be well under a km/s"
    assert raw.mean() > 100 * projected.mean(), (
        f"the projection must change the answer by orders of magnitude: "
        f"{raw.mean():.3f} vs {projected.mean():.4f} km/s"
    )


@pytest.mark.slow
def test_projected_errors_are_consistent_with_the_realized_ones(sb2, warm_fit):
    """Honest bars, within the slack a MAP-fixed Laplace approximation earns."""
    model, v_true, bjd = sb2
    cov = ab.laplace_inverse_mass(
        model.model(_PRIORS(bjd.size)), warm_fit.params, model_args=(model.problem,)
    )
    sigma = ab.relative_velocity_errors(cov, warm_fit.unconstrained)
    err = np.asarray(ab.relative_velocities(warm_fit.params["velocity"], GRID)) - np.asarray(
        ab.relative_velocities(v_true, GRID)
    )
    ratio = float(np.sqrt(np.mean((err / sigma) ** 2)))
    # Laplace holds the hyperparameters at their MAP values, so the bars are expected to
    # run slightly optimistic; an order of magnitude either way would be a defect.
    assert 0.3 < ratio < 3.0, f"error/sigma rms {ratio:.3f}"


def test_relative_velocity_errors_validates_its_inputs(sb2):
    _, v_true, _ = sb2
    unconstrained = {"velocity": jnp.asarray(v_true), "log_tau": jnp.zeros(2)}
    n = v_true.size + 2
    with pytest.raises(ValueError, match="no 'nope' site"):
        ab.relative_velocity_errors(np.eye(n), unconstrained, site="nope")
    with pytest.raises(ValueError, match="must come from the same model"):
        ab.relative_velocity_errors(np.eye(n + 1), unconstrained)
    with pytest.raises(ValueError, match=r"must be \(n_stellar, n_epochs\)"):
        ab.relative_velocity_errors(np.eye(4), {"velocity": jnp.zeros(4)})


def test_relative_velocity_errors_kills_exactly_one_direction_per_component(sb2):
    """The count is the claim: n_stellar flat directions, not one and not n_epochs."""
    _, v_true, _ = sb2
    n_stellar, n_epochs = v_true.shape
    unconstrained = {"velocity": jnp.asarray(v_true)}
    identity = np.eye(v_true.size)
    sigma = ab.relative_velocity_errors(identity, unconstrained)
    # Projecting an identity covariance leaves variance (n-1)/n on every entry.
    assert np.allclose(sigma, np.sqrt((n_epochs - 1) / n_epochs))
    # And the projector itself has exactly n_stellar null directions.
    centre = np.eye(n_epochs) - np.ones((n_epochs, n_epochs)) / n_epochs
    proj = np.kron(np.eye(n_stellar), centre)
    assert np.sum(np.abs(np.linalg.eigvalsh(proj)) < 1e-10) == n_stellar


@pytest.mark.slow
def test_a_cold_start_fails_loudly_rather_than_quietly(sb2, warm_fit):
    """Every epoch at one velocity makes the components indistinguishable.

    This is a real limitation of the free-velocity mode, and the test exists to pin that
    it is *detectable*: the cold start ends at a potential enormously worse than the warm
    one, so a user comparing them cannot mistake the failure for a fit.
    """
    model, _, bjd = sb2
    cold = _fit(model, np.zeros((2, bjd.size)), bjd)
    assert cold.potential > warm_fit.potential + 1e4, (
        f"cold {cold.potential:.1f} vs warm {warm_fit.potential:.1f} — the cold-start "
        "failure must stay obvious in the potential"
    )

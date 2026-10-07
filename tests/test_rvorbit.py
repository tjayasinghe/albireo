"""Tests for the Keplerian fit to a velocity table, and the period search."""

from __future__ import annotations

import math

import numpy as np
import pytest

import albireo as ab
from albireo.rvorbit import find_period, fit_rv_orbit, reassign_by_orbit
from albireo.todcor import VelocityTable

P, T_PERI, ECC, OMEGA, K1, K2, GAMMA = 6.31, 2.0, 0.15, 0.7, 30.0, 55.0, 12.0

# The transit-time seeds were fixed after a scan over the first twenty. The cadence
# determines whether an alias outranks the injected period, and these two draws do what
# their tests describe. With the classical periodogram the injected period is rank 10 at
# the first and rank 1 at fourteen of the twenty. With one harmonic the injected period of
# the eccentric orbit is rank 15 at the second and rank 1 at five of them.
CLUMPED_SEED = 7
HARMONIC_SEED = 7


def rng_epochs(seed, *, n, span):
    return np.sort(np.random.default_rng(seed).uniform(0.0, span, size=n))


def _table_from(bjd, velocity, sigma, absolute=(True, True)):
    """A two-component table over arbitrary epochs, with per-epoch errors."""
    bjd = np.asarray(bjd, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), velocity.shape).copy()
    n = bjd.size
    return VelocityTable(
        names=("A", "B"),
        bjd=bjd,
        instrument=("a",) * n,
        velocity=velocity,
        sigma=sigma,
        sigma_ivar=sigma,
        covariance=np.stack([np.diag(sigma[:, i] ** 2) for i in range(n)]),
        light=np.repeat(np.array([[0.6], [0.4]]), n, axis=1),
        light_mode="fixed",
        chi2=np.full(n, 1000.0),
        chi2_null=np.full(n, 1e5),
        n_pixels=np.full(n, 1000),
        delta_chi2=np.full(velocity.shape, 1e4),
        blended=np.zeros(n, dtype=bool),
        at_edge=np.zeros(velocity.shape, dtype=bool),
        refined=np.ones(n, dtype=bool),
        absolute=tuple(absolute),
        frame="barycentric",
        settings={"n_parameters": 3},
    )


def _injected(bjd, *, ecc=ECC, period=P, sigma=0.5, seed=0):
    """The table of a two-component orbit sampled at ``bjd``, with Gaussian errors."""
    orbit = ab.OrbitParams(
        period=period, t_peri=T_PERI, ecc=ecc, omega=OMEGA, k=(K1, K2), gamma=GAMMA
    )
    truth = orbit.component_velocities(np.asarray(bjd, dtype=float))
    velocity = truth + np.random.default_rng(seed).normal(0.0, sigma, truth.shape)
    return _table_from(bjd, velocity, sigma)


def _greedy_peaks(freqs, power, n_peaks=20, tol=0.02):
    """The exhaustive form of the peak loop: every grid point, in descending power."""
    peaks: list[float] = []
    for k in np.argsort(power)[::-1]:
        period = float(1.0 / freqs[k])
        if all(abs(period / other - 1.0) > tol for other in peaks):
            peaks.append(period)
        if len(peaks) >= n_peaks:
            break
    return peaks


def _rank(peaks, target, tol=0.02):
    """One-based rank of the first peak within ``tol`` of ``target``; 0 if there is none."""
    for i, period in enumerate(peaks):
        if abs(period / target - 1.0) < tol:
            return i + 1
    return 0


def make_table(n_epochs=14, sigma=0.05, seed=0, absolute=(True, True), offsets=(0.0, 0.0), ecc=ECC):
    rng = np.random.default_rng(seed)
    bjd = np.sort(rng.uniform(0.0, 40.0, size=n_epochs))
    orbit = ab.OrbitParams(period=P, t_peri=T_PERI, ecc=ecc, omega=OMEGA, k=(K1, K2), gamma=GAMMA)
    truth = orbit.component_velocities(bjd)
    velocity = truth + np.asarray(offsets)[:, None] + rng.normal(0.0, sigma, truth.shape)
    n = bjd.size
    return (
        VelocityTable(
            names=("A", "B"),
            bjd=bjd,
            instrument=("a",) * n,
            velocity=velocity,
            sigma=np.full(truth.shape, sigma),
            sigma_ivar=np.full(truth.shape, sigma),
            covariance=np.repeat(np.diag([sigma**2, sigma**2])[None], n, axis=0),
            light=np.repeat(np.array([[0.6], [0.4]]), n, axis=1),
            light_mode="fixed",
            chi2=np.full(n, 1000.0),
            chi2_null=np.full(n, 1e5),
            n_pixels=np.full(n, 1000),
            delta_chi2=np.full(truth.shape, 1e4),
            blended=np.zeros(n, dtype=bool),
            at_edge=np.zeros(truth.shape, dtype=bool),
            refined=np.ones(n, dtype=bool),
            absolute=tuple(absolute),
            frame="barycentric",
            settings={"n_parameters": 3},
        ),
        truth,
        orbit,
    )


def test_the_orbit_is_recovered_from_a_noisy_table():
    table, _, _ = make_table()
    fit = fit_rv_orbit(table, period=P * 1.01)
    assert abs(fit.period - P) < 3.0 * fit.errors["period"] + 1e-4
    assert abs(fit.ecc - ECC) < 3.0 * fit.errors["ecc"] + 1e-3
    assert abs(fit.omega - OMEGA) < 3.0 * fit.errors["omega"] + 1e-2
    np.testing.assert_allclose(fit.k, [K1, K2], atol=0.1)
    np.testing.assert_allclose(fit.gamma, [GAMMA, GAMMA], atol=0.1)
    assert fit.gamma_mode == "shared"
    assert fit.n_points == 28 and fit.n_parameters == 7
    # The reduced chi-square is near one, since the errors were injected at the declared level.
    assert 0.4 < fit.chi2 / (fit.n_points - fit.n_parameters) < 2.0
    assert np.all(fit.rms < 0.1)
    assert fit.mass_ratio == pytest.approx(K1 / K2, abs=0.005)
    masses = fit.minimum_masses()
    factor = 1.0361e-7 * (1 - fit.ecc**2) ** 1.5 * (fit.k[0] + fit.k[1]) ** 2 * fit.period
    assert masses["A"] == pytest.approx(factor * fit.k[1])
    assert masses["B"] == pytest.approx(factor * fit.k[0])
    assert "K_A" in fit.summary() and "q = K_A/K_B" in fit.summary()


def test_predict_and_to_theta_agree_with_the_package_conventions():
    table, _, _ = make_table()
    fit = fit_rv_orbit(table, period=P)
    from_theta = np.asarray(ab.orbit_velocities(fit.to_theta(), table.bjd)) + fit.gamma[:, None]
    np.testing.assert_allclose(fit.predict(table.bjd), from_theta, atol=1e-8)
    np.testing.assert_allclose(
        fit.t_peri,
        float(ab.t_peri_from_t_conj(fit.t_conj, period=fit.period, ecc=fit.ecc, omega=fit.omega)),
    )
    resid = table.velocity - fit.predict(table.bjd)
    np.testing.assert_allclose(fit.residuals, resid, atol=1e-8)


def test_differential_components_get_their_own_gamma():
    table, _, _ = make_table(absolute=(False, False), offsets=(35.0, -80.0))
    fit = fit_rv_orbit(table, period=P)
    assert fit.gamma_mode == "one per component"
    np.testing.assert_allclose(fit.gamma, [GAMMA + 35.0, GAMMA - 80.0], atol=0.1)
    np.testing.assert_allclose(fit.k, [K1, K2], atol=0.1)
    with pytest.warns(UserWarning, match="shared systemic velocity"):
        shared = fit_rv_orbit(table, period=P, gamma="shared")
    assert shared.gamma_mode == "shared"
    # Forcing one gamma onto two different zero points biases the semi-amplitudes.
    assert np.max(np.abs(shared.k - np.array([K1, K2]))) > 1.0


def test_a_circular_fit_holds_the_eccentricity_at_zero():
    table, _, _ = make_table(ecc=0.0)
    fit = fit_rv_orbit(table, period=P, circular=True)
    assert fit.ecc == 0.0 and fit.errors["ecc"] == 0.0
    np.testing.assert_allclose(fit.k, [K1, K2], atol=0.1)
    assert fit.n_parameters == 5
    assert "held at zero" in fit.summary()
    theta = fit.to_theta()
    assert float(theta["secosw"]) == 0.0 and float(theta["sesinw"]) == 0.0


def test_the_period_search_finds_the_orbit():
    table, _, _ = make_table(n_epochs=30, seed=3)
    found = find_period(table, period_range=(2.0, 20.0))
    assert abs(found["period"] / P - 1.0) < 0.01, found["period"]
    assert len(found["aliases"]) >= 1
    assert found["power"].shape == found["periods"].shape
    fit = fit_rv_orbit(table, period=found["period"])
    assert abs(fit.period - P) < 0.01


def test_the_period_search_resolves_a_survey_baseline():
    """With forty epochs over five years the default grid must resolve the 6.31 d peak."""
    rng = np.random.default_rng(5)
    bjd = np.sort(rng.uniform(0.0, 1900.0, size=40))
    orbit = ab.OrbitParams(period=P, t_peri=T_PERI, ecc=ECC, omega=OMEGA, k=(K1, K2), gamma=GAMMA)
    truth = orbit.component_velocities(bjd)
    sigma = 0.3
    velocity = truth + rng.normal(0.0, sigma, truth.shape)
    n = bjd.size
    table = VelocityTable(
        names=("A", "B"),
        bjd=bjd,
        instrument=("a",) * n,
        velocity=velocity,
        sigma=np.full(truth.shape, sigma),
        sigma_ivar=np.full(truth.shape, sigma),
        covariance=np.repeat(np.diag([sigma**2, sigma**2])[None], n, axis=0),
        light=np.repeat(np.array([[0.6], [0.4]]), n, axis=1),
        light_mode="fixed",
        chi2=np.full(n, 1000.0),
        chi2_null=np.full(n, 1e5),
        n_pixels=np.full(n, 1000),
        delta_chi2=np.full(truth.shape, 1e4),
        blended=np.zeros(n, dtype=bool),
        at_edge=np.zeros(truth.shape, dtype=bool),
        refined=np.ones(n, dtype=bool),
        absolute=(True, True),
        frame="barycentric",
        settings={"n_parameters": 3},
    )
    found = find_period(table, period_range=(2.0, 20.0))
    # Ten samples per baseline-limited peak width over 1/2 - 1/20 per day.
    assert found["periods"].size >= int(10.0 * 1900.0 * (0.5 - 0.05))
    assert abs(found["period"] - P) < 0.01 * P, found["period"]
    coarse = find_period(table, period_range=(2.0, 20.0), n_frequencies=500)
    assert coarse["periods"].size == 500


def test_swapped_epochs_are_reassigned_by_the_orbit():
    """A twin table with the components exchanged at some epochs is returned in order."""
    table, _truth, orbit = make_table(
        n_epochs=16, sigma=0.2, seed=2, absolute=(False, False), offsets=(3.0, -3.0)
    )
    swapped_in = np.zeros(16, dtype=bool)
    swapped_in[[1, 4, 9, 13]] = True
    velocity = np.array(table.velocity)
    velocity[:, swapped_in] = velocity[::-1, swapped_in]
    from dataclasses import replace

    broken = replace(table, velocity=velocity)
    predicted = orbit.component_velocities(table.bjd) - GAMMA  # no systemic velocity
    fixed, swapped_out = reassign_by_orbit(broken, predicted)
    np.testing.assert_array_equal(swapped_out, swapped_in)
    np.testing.assert_allclose(fixed.velocity, table.velocity)
    assert fixed.settings["reassigned_by_orbit"] == 4
    # An untouched table is returned as it is, and a one-component table is not changed.
    same, none = reassign_by_orbit(table, predicted)
    assert not none.any() and same is table
    # The orbit fitted to the repaired table recovers the semi-amplitudes; the broken one
    # does not.
    good = fit_rv_orbit(fixed, period=P)
    bad = fit_rv_orbit(broken, period=P)
    np.testing.assert_allclose(good.k, [K1, K2], atol=0.3)
    assert bad.chi2 > 10.0 * good.chi2


def test_the_swap_invariant_search_survives_exchanged_epochs():
    """With half the epochs exchanged the plain search fails and the invariant one does not."""
    table, _truth, _orbit = make_table(n_epochs=30, sigma=0.2, seed=4, absolute=(False, False))
    rng = np.random.default_rng(1)
    swapped = rng.uniform(size=30) < 0.5
    velocity = np.array(table.velocity)
    velocity[:, swapped] = velocity[::-1, swapped]
    from dataclasses import replace

    broken = replace(table, velocity=velocity)
    invariant = find_period(broken, period_range=(2.0, 20.0), swap_invariant=True)
    peaks = [invariant["period"], *invariant["aliases"]]
    doubled = [p for peak in peaks for p in (peak, 2.0 * peak)]
    assert any(abs(p / P - 1.0) < 0.02 for p in doubled), doubled
    with pytest.raises(ValueError, match="two components"):
        find_period(broken, components=["A"], swap_invariant=True)


def test_the_periodogram_is_the_floating_mean_generalized_one():
    """The power is the weighted variance a free constant and a sinusoid remove together.

    It is checked against a direct weighted least-squares fit of ``c + a cos wt + b sin wt``
    at four frequencies of the returned grid, which is the definition the closed form of
    Zechmeister & Kürster (2009) evaluates.
    """
    rng = np.random.default_rng(7)
    table = _injected(np.sort(rng.uniform(0.0, 40.0, size=18)), sigma=0.5, seed=7)
    found = find_period(table, period_range=(2.0, 20.0))
    y = table.velocity[0] - table.velocity[1]
    w = 1.0 / (table.sigma[0] ** 2 + table.sigma[1] ** 2)
    chi2_null = float(w @ (y - np.average(y, weights=w)) ** 2)
    freqs = 1.0 / found["periods"]
    for k in (0, 137, int(np.argmax(found["power"])), found["power"].size - 1):
        omega = 2.0 * np.pi * freqs[k]
        design = np.stack(
            [np.ones_like(table.bjd), np.cos(omega * table.bjd), np.sin(omega * table.bjd)],
            axis=1,
        )
        root = np.sqrt(w)
        coefficients, *_ = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)
        chi2 = float(w @ (y - design @ coefficients) ** 2)
        assert found["power"][k] == pytest.approx(1.0 - chi2 / chi2_null, abs=1e-9), k


def test_the_floating_mean_survives_a_clumped_survey_cadence():
    """On a Gaia-like cadence the classical periodogram ranks an alias first and this one
    ranks the injected period first.

    The cadence has sixteen transits in about eleven visibility periods over the DR4 span.
    The sampling window has a large mean at most frequencies, so the constant a zero-offset
    model cannot fit is absorbed into the sinusoid. The seed is fixed at one that shows it.
    """
    from scipy.signal import lombscargle

    from albireo.gaia import rvs_transit_times

    table = _injected(rvs_transit_times(16, seed=CLUMPED_SEED), ecc=0.0, sigma=0.5, seed=3)
    found = find_period(table)
    assert _rank([found["period"], *found["aliases"]], P) == 1, found["period"]
    y = table.velocity[0] - table.velocity[1]
    w = 1.0 / (table.sigma[0] ** 2 + table.sigma[1] ** 2)
    freqs = 1.0 / found["periods"]
    classical = lombscargle(
        table.bjd,
        (y - np.average(y, weights=w)) * np.sqrt(w),
        2.0 * np.pi * freqs,
        normalize=True,
    )
    classical_rank = _rank(_greedy_peaks(freqs, classical), P)
    assert classical_rank != 1, classical_rank


def test_the_second_harmonic_ranks_an_eccentric_period_higher():
    """At e = 0.5 the velocity curve is not a sinusoid and the extra harmonic finds the period."""
    from albireo.gaia import rvs_transit_times

    table = _injected(rvs_transit_times(18, seed=HARMONIC_SEED), ecc=0.5, sigma=0.5, seed=2)
    one = find_period(table)
    two = find_period(table, n_harmonics=2)
    rank_one = _rank([one["period"], *one["aliases"]], P)
    rank_two = _rank([two["period"], *two["aliases"]], P)
    assert rank_two == 1 and rank_one != 1, (rank_one, rank_two)
    # The models are nested on the same grid, so the second harmonic cannot explain less.
    assert np.all(two["power"] >= one["power"] - 1e-8)
    with pytest.raises(ValueError, match="n_harmonics"):
        find_period(table, n_harmonics=0)


def test_the_number_of_peaks_reported_is_a_parameter():
    from albireo.gaia import rvs_transit_times

    table, _, _ = make_table(n_epochs=30, seed=3)
    default = find_period(table, period_range=(2.0, 20.0))
    few = find_period(table, period_range=(2.0, 20.0), n_peaks=4)
    assert [few["period"], *few["aliases"]] == [default["period"], *default["aliases"][:3]]
    # A short list is a prefix of a long one, and the list is as long as the periodogram
    # allows: a 40 d baseline resolves only sixteen distinct peaks between 2 and 20 d,
    # which is fewer than the twenty requested. A survey baseline resolves all of them.
    assert len(default["aliases"]) == 15
    survey = _injected(rvs_transit_times(20, seed=1), sigma=0.5, seed=1)
    assert len(find_period(survey)["aliases"]) == 19
    with pytest.raises(ValueError, match="n_peaks"):
        find_period(table, n_peaks=0)


def test_the_peak_loop_walks_only_the_local_maxima():
    """On a grid that resolves the 2% window it returns what the exhaustive loop returns."""
    from albireo.rvorbit import _distinct_peaks

    rng = np.random.default_rng(11)
    freqs = np.linspace(1.0 / 20.0, 1.0, 20_000)  # the step in period is under 0.1% throughout
    kernel = np.exp(-0.5 * (np.arange(-40, 41) / 8.0) ** 2)
    power = np.convolve(rng.normal(size=freqs.size), kernel / kernel.sum(), mode="same")
    power = (power - power.min()) / np.ptp(power)
    assert _distinct_peaks(freqs, power, 20) == _greedy_peaks(freqs, power, 20)


def test_the_weights_enter_the_statistic_and_not_the_data():
    """Errors varying by four change the power; scaling every error leaves it unchanged."""
    from dataclasses import replace

    table = _injected(rng_epochs(5, n=20, span=40.0), sigma=0.5, seed=5)
    before = np.array(table.velocity)
    uneven_sigma = np.array(table.sigma)
    uneven_sigma[:, ::2] *= 4.0
    even = find_period(table, period_range=(2.0, 20.0))
    uneven = find_period(
        replace(table, sigma=uneven_sigma, sigma_ivar=uneven_sigma), period_range=(2.0, 20.0)
    )
    scaled = find_period(
        replace(table, sigma=2.0 * table.sigma, sigma_ivar=2.0 * table.sigma),
        period_range=(2.0, 20.0),
    )
    np.testing.assert_allclose(scaled["power"], even["power"], atol=1e-12)
    assert np.max(np.abs(uneven["power"] - even["power"])) > 1e-3
    np.testing.assert_array_equal(table.velocity, before)


def _without(table, component, epochs, *, at_edge=True):
    """``table`` with one component unmeasured at ``epochs``, as the correlation leaves it."""
    from dataclasses import replace

    velocity = np.array(table.velocity)
    sigma = np.array(table.sigma)
    edge = np.array(table.at_edge)
    velocity[component, epochs] = np.nan
    sigma[component, epochs] = np.nan
    edge[component, epochs] = at_edge
    return replace(table, velocity=velocity, sigma=sigma, sigma_ivar=sigma, at_edge=edge)


def test_a_velocity_counts_wherever_its_own_component_was_measured():
    """A primary measured at an epoch whose secondary was at the search edge is kept (D65).

    ``VelocityTable.good`` requires every component to be finite and off the edge. If the
    search on the first component alone and the fit were restricted to those epochs, that
    search, which is meant for a companion the templates cannot follow, would lose the
    primary epochs at which the companion was not measured. A velocity is therefore used
    where its own component was measured and the epoch is not blended. The relative
    velocity still needs both.
    """
    from dataclasses import replace

    table, _, _ = make_table(n_epochs=14, sigma=0.05, seed=0)
    lost = np.zeros(14, dtype=bool)
    lost[[2, 5, 9]] = True
    edge = _without(table, 1, lost)
    assert not edge.good[lost].any()  # the table's own epoch flag drops them

    # Where every component is valid nothing changes: the searched epochs are table.good.
    np.testing.assert_array_equal(find_period(table, period_range=(2.0, 20.0))["used"], table.good)
    # Searched alone, the primary keeps every epoch it was measured at, and its periodogram
    # is the one of the complete table; the relative velocity needs both components.
    alone = find_period(edge, components=["A"], period_range=(2.0, 20.0))
    full = find_period(table, components=["A"], period_range=(2.0, 20.0))
    assert alone["used"].all()
    np.testing.assert_array_equal(alone["power"], full["power"])
    np.testing.assert_array_equal(find_period(edge, period_range=(2.0, 20.0))["used"], ~lost)

    # The fit takes each velocity where its own component was measured.
    fit = fit_rv_orbit(edge, period=P)
    assert fit.used.all() and fit.n_points == 2 * 14 - 3
    np.testing.assert_array_equal(np.isfinite(fit.residuals[0]), np.ones(14, dtype=bool))
    np.testing.assert_array_equal(np.isfinite(fit.residuals[1]), ~lost)
    np.testing.assert_allclose(fit.k, [K1, K2], atol=0.2)
    primary = fit_rv_orbit(edge, period=P, components=["A"])
    assert primary.n_points == 14
    assert primary.chi2 == pytest.approx(fit_rv_orbit(table, period=P, components=["A"]).chi2)

    # A blended epoch is dropped for every component.
    blended = np.array(edge.blended)
    blended[4] = True
    merged = replace(edge, blended=blended)
    assert not find_period(merged, components=["A"], period_range=(2.0, 20.0))["used"][4]
    assert fit_rv_orbit(merged, period=P).n_points == 2 * 14 - 3 - 2


def test_a_component_with_no_usable_velocity_leaves_the_other_fitted():
    """A companion with no usable velocity is held, and the primary's orbit is still fitted.

    The semi-amplitude of a component no velocity constrains is not a measurement, and the
    optimizer does not keep an unconstrained parameter at its start. On a benchmark table
    whose secondary was below the detection threshold at every epoch the semi-amplitude
    reached 1.5e7 km/s, and the ranking then rejected the true period as outside the
    declared ranges. The semi-amplitude is held at 1e-3 km/s, with the component's own
    systemic velocity where each component has one. Neither counts as a parameter, and
    nothing derived from it is reported. The parameters the primary constrains keep finite
    errors.
    """
    from dataclasses import replace

    table, _, _ = make_table(n_epochs=14, sigma=0.05, seed=0)
    blind = _without(table, 1, np.ones(14, dtype=bool), at_edge=False)
    fit = fit_rv_orbit(blind, period=P)
    assert fit.n_points == 14 and fit.used.all() and fit.n_parameters == 6
    assert fit.held == ("B",)
    assert abs(fit.k[0] - K1) < 0.2 and abs(fit.period - P) < 0.01
    assert fit.k[1] == 1e-3
    assert np.isnan(fit.errors["k"][1])
    assert np.isfinite(fit.errors["k"][0]) and np.isfinite(fit.errors["period"])
    assert np.isnan(fit.residuals[1]).all() and np.isnan(fit.rms[1]) and fit.rms[0] < 0.2
    assert fit.mass_ratio is None and fit.minimum_masses() == {}
    assert set(fit.projected_semiaxes()) == {"A"}
    assert "held, not fitted" in fit.summary() and "q = " not in fit.summary()
    # One systemic velocity per component: the second is held at the first's mean as well.
    differential = fit_rv_orbit(replace(blind, absolute=(False, False)), period=P)
    assert differential.gamma_mode == "one per component" and differential.n_parameters == 6
    assert differential.k[1] == 1e-3 and np.isnan(differential.errors["gamma"][1])
    assert differential.gamma[1] == pytest.approx(float(np.mean(blind.velocity[0])))
    assert abs(differential.k[0] - K1) < 0.2
    # A held semi-amplitude stays at the value the caller gives.
    assert fit_rv_orbit(blind, period=P, k=[25.0, 40.0]).k[1] == 40.0
    # Fewer usable velocities than parameters raises an error stated in terms of velocities.
    with pytest.raises(ValueError, match="usable velocities"):
        fit_rv_orbit(_without(blind, 0, np.arange(8), at_edge=False), period=P)
    # With every component valid, nothing is held.
    assert fit_rv_orbit(table, period=P).held == ()


@pytest.mark.parametrize(
    ("absolute", "n_usable", "held"),
    [
        ((True, True), 1, True),  # shared gamma: one parameter of its own, one velocity
        ((True, True), 2, False),
        ((False, False), 1, True),  # its own gamma too: two parameters of its own
        ((False, False), 2, True),
        ((False, False), 3, False),
    ],
)
def test_a_companion_with_no_more_velocities_than_its_own_parameters_is_held(
    absolute, n_usable, held
):
    """The hold threshold is the component's own parameter count, not zero (D65).

    One velocity fixes a free semi-amplitude exactly, and two fix a semi-amplitude and a
    systemic velocity of the component's own, so neither is a measurement. Such a component
    is held, its velocities have no weight, and nothing derived from it is reported.
    """
    from dataclasses import replace

    table, _, _ = make_table(n_epochs=14, sigma=0.05, seed=0)
    table = replace(table, absolute=absolute)
    lost = np.ones(14, dtype=bool)
    lost[[1, 6, 11][:n_usable]] = False
    sparse = _without(table, 1, lost, at_edge=False)
    fit = fit_rv_orbit(sparse, period=P)
    own = 1 if all(absolute) else 2
    if held:
        assert fit.held == ("B",) and fit.k[1] == 1e-3
        assert fit.n_points == 14 and fit.n_parameters == 6
        assert np.isnan(fit.residuals[1]).all() and np.isnan(fit.errors["k"][1])
        assert fit.mass_ratio is None and fit.minimum_masses() == {}
    else:
        assert fit.held == () and n_usable > own
        assert fit.n_points == 14 + n_usable and fit.n_parameters == 4 + 2 + (1 if own == 1 else 2)
        assert np.isfinite(fit.residuals[1]).sum() == n_usable
        assert fit.mass_ratio is not None
    assert abs(fit.k[0] - K1) < 0.2


def test_the_semi_amplitude_start_is_half_the_range_of_the_usable_velocities():
    """The start is half the range of each component's own usable velocities (D65).

    Half the range is ``K`` at any eccentricity once the phases are covered, and the start
    is unchanged wherever every component is usable. A ``sqrt(2)`` times weighted standard
    deviation is not used: sampled evenly in time it is 0.34 to 0.51 of ``K`` at
    ``e = 0.9``. Velocities the caller set to nan (the pipeline's detection threshold) do
    not enter the start. ``max_iterations=1`` returns the start.
    """
    from dataclasses import replace

    from albireo.rvorbit import _semi_amplitude_start

    # At e = 0.9 with the phases covered, half the range is K and the weighted spread is not.
    orbit = ab.OrbitParams(period=P, t_peri=T_PERI, ecc=0.9, omega=OMEGA, k=(K1, K2), gamma=GAMMA)
    dense = orbit.component_velocities(np.linspace(0.0, P, 4000, endpoint=False))
    assert _semi_amplitude_start(dense[0]) == pytest.approx(K1, rel=5e-3)
    assert np.sqrt(2.0) * dense[0].std() < 0.6 * K1
    assert _semi_amplitude_start([5.0]) == 0.0 and _semi_amplitude_start([]) == 0.0

    table, _, _ = make_table(n_epochs=20, sigma=0.5, seed=1, ecc=0.0)
    np.testing.assert_allclose(
        fit_rv_orbit(table, period=P, max_iterations=1).k, 0.5 * np.ptp(table.velocity, axis=1)
    )
    rng = np.random.default_rng(15)
    noise = np.zeros(20, dtype=bool)
    noise[rng.choice(20, size=8, replace=False)] = True
    velocity = np.array(table.velocity)
    velocity[1, noise] = rng.uniform(-300.0, 300.0, size=8)
    noisy = replace(table, velocity=velocity)
    # A companion's undetected draws, if left in the table, set its start.
    assert fit_rv_orbit(noisy, period=P, max_iterations=1).k[1] > 250.0
    # Once they are removed, as the pipeline's detection threshold removes them, they do not.
    gated = _without(noisy, 1, noise, at_edge=False)
    start = fit_rv_orbit(gated, period=P, max_iterations=1).k
    assert start[1] == pytest.approx(0.5 * np.ptp(noisy.velocity[1, ~noise]))
    assert start[0] == pytest.approx(0.5 * np.ptp(noisy.velocity[0]))


def test_a_rounding_shift_of_one_epoch_time_moves_neither_the_grid_nor_the_peaks():
    """Rounding an epoch time to six decimals must not change the period search (D65).

    The high end of the default grid is twice the smallest gap between epochs. On a grid
    spaced evenly between its two ends, a shift of 1e-6 d in the closest pair moves every
    upper frequency by a sizeable fraction of a step and reorders near-degenerate peaks.
    The written tables have six decimals of the epoch time. The grid is therefore anchored
    at its low end with a step that does not depend on the high end.
    """
    from dataclasses import replace

    from albireo.gaia import rvs_transit_times

    bjd = rvs_transit_times(14, seed=2)
    order = np.argsort(bjd)
    closest = int(np.argmin(np.diff(bjd[order])))
    later = int(order[closest + 1])
    assert later not in (int(order[0]), int(order[-1]))
    moved = np.array(bjd)
    moved[later] -= 1e-6  # the closest gap shrinks, so the shortest period does
    table = _injected(bjd, ecc=0.3, sigma=5.0, seed=2)
    shifted = replace(table, bjd=moved)

    before, after = find_period(table), find_period(shifted)
    f_before, f_after = 1.0 / before["periods"], 1.0 / after["periods"]
    step = f_before[1] - f_before[0]
    assert f_after[-1] - f_before[-1] > step  # the high end moved by more than a step
    np.testing.assert_array_equal(f_after[: f_before.size - 1], f_before[:-1])
    for harmonics in (1, 2):
        one = find_period(table, n_harmonics=harmonics)
        two = find_period(shifted, n_harmonics=harmonics)
        assert [one["period"], *one["aliases"]] == [two["period"], *two["aliases"]]

    # The grid spaced evenly between the same two ends, as the default was, moves under the shift.
    def evenly(t):
        gaps = np.diff(np.sort(t))
        lo, hi = 2.0 * float(gaps.min()), 2.0 * float(np.ptp(t))
        return {
            "period_range": (lo, hi),
            "n_frequencies": max(20_000, int(np.ceil(10.0 * np.ptp(t) * (1.0 / lo - 1.0 / hi)))),
        }

    old_before = 1.0 / find_period(table, **evenly(bjd))["periods"]
    old_after = 1.0 / find_period(shifted, **evenly(moved))["periods"]
    common = min(old_before.size, old_after.size)
    assert np.max(np.abs(old_after[:common] - old_before[:common])) > 0.3 * step


def test_the_anchored_grid_on_a_short_baseline_where_more_samples_are_needed():
    """Where ten samples per 1/T give fewer than 20000 frequencies, N is raised and stays fixed.

    Twelve epochs over 40 d with a closest pair 0.074 d apart need about 75 samples per 1/T
    to reach 20000 frequencies. ``N`` is computed from the span rounded down to two
    significant figures, so the same 1e-6 d shift of the closest pair leaves ``N`` and every
    point below both high ends unchanged. The last point is the high end itself.
    """
    from dataclasses import replace

    from albireo.rvorbit import _frequency_grid

    bjd = np.array([0.0, 3.1, 7.4, 11.0, 15.2, 17.3, 17.374, 21.9, 26.5, 30.8, 35.2, 40.0])
    moved = np.array(bjd)
    later = int(np.flatnonzero(np.isclose(bjd, 17.374))[0])
    moved[later] -= 1e-6
    before = _frequency_grid(bjd, None, None)
    after = _frequency_grid(moved, None, None)
    baseline = float(np.ptp(bjd))
    step = before[1] - before[0]
    oversampling = 1.0 / (step * baseline)
    assert oversampling > 10.0 and round(oversampling) == pytest.approx(oversampling)
    assert before.size >= 20_000 and after.size >= 20_000
    assert before[-1] == 1.0 / (2.0 * float(np.min(np.diff(bjd))))
    assert after[-1] == 1.0 / (2.0 * float(np.min(np.diff(moved))))
    assert after[1] - after[0] == step  # N did not change
    below = int(np.sum(before < min(before[-1], after[-1])))
    np.testing.assert_array_equal(after[:below], before[:below])
    table = _injected(bjd, sigma=2.0, seed=4)
    one, two = find_period(table), find_period(replace(table, bjd=moved))
    assert [one["period"], *one["aliases"]] == [two["period"], *two["aliases"]]


# Reference values for an all-components-valid table, computed with the code of the base
# commit 001c07f (before D65) from a separate export of its sources; the D65 code reproduced
# every one of them bit for bit when they were recorded.
_REF_ECCENTRIC_CHI2 = 27.61742727604286
_REF_ECCENTRIC_PERIOD = 6.308626877857585
_REF_ECCENTRIC_K = [29.705874852032885, 54.99453389180863]
_REF_ECCENTRIC_COVARIANCE = [
    [4.3611943163049894e-06, -1.1124333729880686e-05, -2.485453365889922e-06,
     4.821642412368967e-07, -3.241018383999791e-05, -5.993495883374985e-05,
     8.751060671533285e-08],
    [-1.1124333729880679e-05, 7.80159137069242e-05, -3.120147254054062e-05,
     2.0887828285974e-05, 0.00022633995146627753, 0.0004052845110394827,
     -1.8198424052654937e-05],
    [-2.4854533658899287e-06, -3.120147254054059e-05, 4.716258565754293e-05,
     -4.964983402444409e-05, -0.00012401741920227934, -0.00022568719913273455,
     5.1744071441018045e-06],
    [4.821642412369044e-07, 2.0887828285973963e-05, -4.9649834024444076e-05,
     0.000103677140175676, 0.0002837468297946317, 0.0004866188190264264,
     -5.1237628937420784e-05],
    [-3.2410183839997875e-05, 0.00022633995146627747, -0.00012401741920227944,
     0.00028374682979463175, 0.05492910479276298, 0.0016996038896037226,
     -0.003537002464321398],
    [-5.9934958833749776e-05, 0.00040528451103948265, -0.0002256871991327348,
     0.0004866188190264267, 0.0016996038896037226, 0.05774784150287048,
     0.0026924705628849537],
    [8.751060671533381e-08, -1.8198424052654923e-05, 5.174407144101781e-06,
     -5.1237628937420764e-05, -0.003537002464321398, 0.0026924705628849537,
     0.01223982827073064],
]  # fmt: skip
_REF_CIRCULAR_CHI2 = 2504.9806295423596
_REF_CIRCULAR_K = [29.28552980872998, 54.35851733717239]
_REF_CIRCULAR_COVARIANCE = [
    [0.00030409918165089046, -0.0009671191102587181, -0.0005675475587329748,
     -0.0014129304546225939, -0.0004219790677445696],
    [-0.0009671191102587181, 0.004039208401222893, 0.0073253272088457735,
     0.012775294671924321, -0.0009645270300695788],
    [-0.0005675475587329749, 0.007325327208845772, 4.478815407916433,
     -0.03999673109852521, -0.3144601948204636],
    [-0.001412930454622594, 0.012775294671924323, -0.039996731098525196,
     4.523878268848356, 0.2841673117291289],
    [-0.0004219790677445701, -0.0009645270300695775, -0.3144601948204636,
     0.2841673117291289, 1.0187577388752167],
]  # fmt: skip
_REF_POWER = {
    1: ([0, 777, 1500, 960, 3999],
        [0.021125983096137335, 0.14718068148109076, 0.3337971685049737,
         0.9742340448482779, 0.18252440129307962],
        807.8944282823498,
        [6.328032281034892, 2.162030654448139, 11.765225066195939, 2.523665278303673,
         4.610861293670011, 2.840703249866809]),
    2: ([0, 777, 1500, 965, 3999],
        [0.3561309385631847, 0.20180156391936843, 0.39002893536829586,
         0.9995247220685054, 0.23496409018208886],
        1419.4561640329634,
        [6.30558183538316, 12.635071090047393, 2.548269929267826, 9.55213185238266,
         2.293136074316188, 2.8425205245761807]),
}  # fmt: skip


def test_an_all_valid_table_is_fitted_and_searched_as_before_d65():
    """Where every component is valid, D65 changes neither the fit nor the periodogram.

    The per-component validity, the hold, the start and the compiled objective must leave
    the results for such a table unchanged. With the start, conjunction and grid given
    explicitly, the chi-square, the parameter count, the covariance and the periodogram
    powers are compared with the base commit's values.
    """
    table, _, _ = make_table(n_epochs=14, sigma=0.5, seed=1)
    eccentric = fit_rv_orbit(table, period=P * 1.01, k=[28.0, 52.0], t_conj=2.5, ecc=0.1, omega=0.5)
    assert eccentric.n_parameters == 7 and eccentric.n_points == 28 and eccentric.held == ()
    assert eccentric.chi2 == pytest.approx(_REF_ECCENTRIC_CHI2, rel=1e-10)
    assert eccentric.period == pytest.approx(_REF_ECCENTRIC_PERIOD, rel=1e-12)
    np.testing.assert_allclose(eccentric.k, _REF_ECCENTRIC_K, rtol=1e-10)
    np.testing.assert_allclose(eccentric.covariance, _REF_ECCENTRIC_COVARIANCE, rtol=1e-8)
    circular = fit_rv_orbit(table, period=P * 1.01, k=[28.0, 52.0], t_conj=2.5, circular=True)
    assert circular.n_parameters == 5 and circular.n_points == 28
    assert circular.chi2 == pytest.approx(_REF_CIRCULAR_CHI2, rel=1e-10)
    np.testing.assert_allclose(circular.k, _REF_CIRCULAR_K, rtol=1e-10)
    np.testing.assert_allclose(circular.covariance, _REF_CIRCULAR_COVARIANCE, rtol=1e-8)
    for harmonics, (index, power, total, peaks) in _REF_POWER.items():
        found = find_period(
            table, period_range=(2.0, 20.0), n_frequencies=4000, n_harmonics=harmonics
        )
        assert int(np.argmax(found["power"])) == index[3]
        np.testing.assert_allclose(found["power"][index], power, rtol=1e-12)
        assert float(found["power"].sum()) == pytest.approx(total, rel=1e-12)
        np.testing.assert_allclose([found["period"], *found["aliases"][:5]], peaks, rtol=1e-12)


def test_a_single_component_table_is_fitted_too():
    table, _, _ = make_table()
    fit = fit_rv_orbit(table, period=P, components=["A"])
    assert fit.names == ("A",)
    assert abs(fit.k[0] - K1) < 0.15
    assert fit.mass_ratio is None and fit.minimum_masses() == {}
    assert fit.projected_semiaxes()["A"] == pytest.approx(
        86400.0 / (2 * math.pi) / 695_700.0 * fit.k[0] * fit.period * math.sqrt(1 - fit.ecc**2)
    )


def test_argument_validation():
    table, _, _ = make_table()
    with pytest.raises(ValueError, match="no component"):
        fit_rv_orbit(table, period=P, components=["C"])
    with pytest.raises(ValueError, match="gamma must be"):
        fit_rv_orbit(table, period=P, gamma="none")
    with pytest.raises(ValueError, match="period must be positive"):
        fit_rv_orbit(table, period=-1.0)
    with pytest.raises(ValueError, match="period_range"):
        find_period(table, period_range=(5.0, 1.0))

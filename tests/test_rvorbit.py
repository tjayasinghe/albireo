"""Tests for the Keplerian fit to a velocity table, and the period search."""

from __future__ import annotations

import math

import numpy as np
import pytest

import albireo as ab
from albireo.rvorbit import find_period, fit_rv_orbit, reassign_by_orbit
from albireo.todcor import VelocityTable

P, T_PERI, ECC, OMEGA, K1, K2, GAMMA = 6.31, 2.0, 0.15, 0.7, 30.0, 55.0, 12.0

# Transit-time seeds fixed after a scan over the first twenty: the cadence decides whether
# an alias outranks the truth, and these two draws do what their tests describe (with the
# classical periodogram the truth is rank 10 at the first, rank 1 at fourteen of the twenty;
# with one harmonic the eccentric truth is rank 15 at the second, rank 1 at five of them).
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
    # Reduced chi-square near one, since the errors were injected at the declared level.
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
    # Forcing one gamma onto two different zero points corrupts the semi-amplitudes.
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
    """Forty epochs over five years: the default grid must resolve the 6.31 d peak."""
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
    """A twin table with the components exchanged at some epochs comes back in order."""
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
    # An untouched table is returned as it is, and a one-component table is left alone.
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

    Checked against a direct weighted least-squares fit of ``c + a cos wt + b sin wt`` at
    four frequencies of the returned grid, which is the definition the closed form of
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
    """A Gaia-like cadence: the classical periodogram ranks an alias first, this one the truth.

    Sixteen transits in about eleven visibility periods over the DR4 span. The sampling
    window has a large mean at most frequencies, so the constant a zero-offset model cannot
    fit is absorbed into the sinusoid; the seed is fixed at one that shows it.
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
    """At e = 0.5 the velocity curve is not a sinusoid and the extra harmonic finds it."""
    from albireo.gaia import rvs_transit_times

    table = _injected(rvs_transit_times(18, seed=HARMONIC_SEED), ecc=0.5, sigma=0.5, seed=2)
    one = find_period(table)
    two = find_period(table, n_harmonics=2)
    rank_one = _rank([one["period"], *one["aliases"]], P)
    rank_two = _rank([two["period"], *two["aliases"]], P)
    assert rank_two == 1 and rank_one != 1, (rank_one, rank_two)
    # Nested models on the same grid: the second harmonic cannot explain less.
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
    # which is fewer than the twenty asked for. A survey baseline has room for all of them.
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
    """Errors varying by four change the power; scaling every error leaves it alone."""
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

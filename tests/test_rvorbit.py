"""Tests for the Keplerian fit to a velocity table, and the period search."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

import albireo as ab
from albireo.kepler import t_conj_from_t_peri
from albireo.rvorbit import (
    Assignment,
    assign_by_ephemeris,
    assign_by_orbit,
    assign_components,
    find_period,
    fit_rv_ephemeris,
    fit_rv_orbit,
    reassign_by_orbit,
)
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


TWIN_K = (61.0, 64.0)


def _twin_table(seed, *, n_epochs=12, ecc=0.3):
    """Two alike stars at random times, the two velocities of each epoch in a random order.

    The tables are those of ``scripts/assignment_bench.py``: the argument and the time of
    periastron are drawn per table, the errors are 0.5 km/s, and the light fractions are
    equal. Returns the table, the injected velocities and the epochs that were exchanged.
    """
    rng = np.random.default_rng(seed)
    bjd = np.sort(rng.uniform(0.0, 40.0, size=n_epochs))
    orbit = ab.OrbitParams(
        period=P,
        t_peri=rng.uniform(0.0, P),
        ecc=ecc,
        omega=rng.uniform(0.0, 2.0 * np.pi),
        k=TWIN_K,
        gamma=GAMMA,
    )
    truth = orbit.component_velocities(bjd)
    velocity = truth + rng.normal(0.0, 0.5, truth.shape)
    exchanged = rng.uniform(size=n_epochs) < 0.5
    velocity[:, exchanged] = velocity[::-1, exchanged]
    table = replace(_table_from(bjd, velocity, 0.5), light=np.full(truth.shape, 0.5))
    return table, truth, exchanged


def _twins_recovered(orbit) -> bool:
    """Both semi-amplitudes within 2 percent of the injected ones, in either order."""
    k = np.asarray(orbit.k)
    return min(np.max(np.abs(order / np.array(TWIN_K) - 1.0)) for order in (k, k[::-1])) < 0.02


def _same_assignment(mask, injected, truth) -> bool:
    """``mask`` undoes the injected exchange, in one of the two orders of the stars.

    Only the epochs with the two velocities more than ten errors apart are compared: an
    epoch at conjunction fits in either order and is left as it is.
    """
    clear = np.abs(truth[0] - truth[1]) > 5.0
    return bool(
        np.array_equal(mask[clear], injected[clear])
        or np.array_equal(mask[clear], ~injected[clear])
    )


@pytest.mark.parametrize("ecc", [0.0, 0.3])
def test_alike_stars_in_a_random_order_are_assigned_at_a_known_period(ecc):
    """Twelve epochs of two alike stars, each in a random order, give the orbit.

    One exchange by the first fit recovers 1 of these 20 tables for the circular orbit and
    none at an eccentricity of 0.3, because the first fit is poor. ``assign_components``
    recovers all 40 (``scripts/assignment_bench.py`` has the rates at other eccentricities
    and numbers of epochs).
    """
    n_tables = 20
    once = assigned = 0
    for seed in range(n_tables):
        table, truth, injected = _twin_table(seed, ecc=ecc)
        first = fit_rv_orbit(table, period=P)
        exchanged, moved = reassign_by_orbit(table, first.predict(table.bjd))
        once += _twins_recovered(fit_rv_orbit(exchanged, period=P) if moved.any() else first)
        result, orbit, decided = assign_components(table, period=P)
        assigned += _twins_recovered(orbit)
        assert _same_assignment(decided.exchanged, injected, truth)
        assert not decided.alternative.any() and not decided.resolved.any()
        # In one of the two orders of the stars every velocity is the injected one.
        errors = [np.abs(result.velocity[order] - truth).max() for order in ([0, 1], [1, 0])]
        assert min(errors) < 3.0 * 0.5
        assert orbit.chi2 < 2.0 * (orbit.n_points - orbit.n_parameters)
    assert assigned == n_tables
    assert once <= 2


def test_the_assignment_returns_the_table_the_orbit_and_the_exchanged_epochs():
    table, _, _ = _twin_table(3)
    assigned, orbit, decided = assign_components(table, period=P)
    mask = decided.exchanged
    assert mask.dtype == bool and mask.shape == (table.n_epochs,) and mask.any()
    assert decided.changes and decided.apply(table).velocity.tolist() == assigned.velocity.tolist()
    np.testing.assert_array_equal(assigned.velocity[:, mask], table.velocity[::-1][:, mask])
    np.testing.assert_array_equal(assigned.velocity[:, ~mask], table.velocity[:, ~mask])
    assert assigned.settings["reassigned_by_orbit"] == int(mask.sum())
    refit = fit_rv_orbit(assigned, period=P)
    assert orbit.chi2 == pytest.approx(refit.chi2, rel=1e-6)
    np.testing.assert_allclose(orbit.k, refit.k, rtol=1e-5)
    assert orbit.chi2 < 0.01 * fit_rv_orbit(table, period=P).chi2
    # The options of the fit are passed on.
    _, circular, _ = assign_components(table, period=P, circular=True, gamma="per-component")
    assert circular.ecc == 0.0 and circular.gamma_mode == "one per component"
    # A table that is in order is returned as it is, with the orbit of the plain fit.
    ordered, _, _ = make_table()
    same, plain, none = assign_components(ordered, period=P)
    assert same is ordered and not none.changes and none.apply(ordered) is ordered
    assert plain.chi2 == pytest.approx(fit_rv_orbit(ordered, period=P).chi2, rel=1e-9)
    # A table that cannot support an orbit raises as the fit does.
    few = replace(table, blended=np.arange(table.n_epochs) > 1)
    with pytest.raises(ValueError, match="not enough to fit"):
        assign_components(few, period=P)


def test_the_assignment_is_not_made_between_unalike_light_fractions():
    """At light fractions more than a factor of three apart no epoch is exchanged."""
    table, _, _ = _twin_table(3)
    n = table.n_epochs
    unalike = replace(table, light=np.repeat(np.array([[0.9], [0.1]]), n, axis=1))
    same, orbit, decided = assign_components(unalike, period=P)
    assert same is unalike and not decided.changes
    assert orbit.chi2 == pytest.approx(fit_rv_orbit(unalike, period=P).chi2, rel=1e-9)
    _, exchanged, moved = assign_components(unalike, period=P, light_ratio_max=None)
    assert moved.exchanged.any() and _twins_recovered(exchanged)
    assert assign_components(unalike, period=P, light_ratio_max=10.0)[2].exchanged.any()
    # The exchange can be refused outright, whatever the light fractions.
    kept, _, refused = assign_components(table, period=P, exchange=False)
    assert kept is table and not refused.changes
    # Fractions that are not finite and positive do not apply the rule.
    unknown = replace(table, light=np.full(table.velocity.shape, np.nan))
    assert assign_components(unknown, period=P)[2].exchanged.any()
    # A table with another number of components is fitted as it is.
    one = replace(
        table,
        names=("A",),
        velocity=table.velocity[:1],
        sigma=table.sigma[:1],
        sigma_ivar=table.sigma_ivar[:1],
        covariance=table.covariance[:, :1, :1],
        light=table.light[:1],
        delta_chi2=table.delta_chi2[:1],
        at_edge=table.at_edge[:1],
        absolute=(True,),
    )
    single, alone, untouched = assign_components(one, period=P)
    assert single is one and alone.names == ("A",) and not untouched.changes


def test_the_starting_assignments_come_from_a_grid_of_keplerian_curves():
    """The curves against the Kepler solver of the package, and the starts they give."""
    from albireo.kepler import radial_velocity
    from albireo.rvorbit import (
        _ASSIGNMENT_ECCENTRICITIES,
        _ASSIGNMENT_OMEGAS,
        _ASSIGNMENT_PHASES,
        _assignments_by_period,
        _relative_curves,
    )

    t = np.sort(np.random.default_rng(5).uniform(2457000.0, 2457900.0, 40))
    curves = _relative_curves(t, P)
    n_phase, n_omega = _ASSIGNMENT_PHASES, _ASSIGNMENT_OMEGAS
    assert curves.shape == (n_phase * (1 + len(_ASSIGNMENT_ECCENTRICITIES) * n_omega), t.size)
    for k in (0, 17, n_phase - 1):  # circular curves, at phases over half a period
        expected = radial_velocity(
            t, period=P, t_peri=k / (2 * n_phase) * P, ecc=0.0, omega=0.0, k=1.0
        )
        np.testing.assert_allclose(curves[k], expected, atol=1e-6)
    for i, j, k in ((0, 0, 0), (1, 3, 7), (3, 7, n_phase - 1), (3, 2, 31)):
        expected = radial_velocity(
            t,
            period=P,
            t_peri=k / n_phase * P,
            ecc=_ASSIGNMENT_ECCENTRICITIES[i],
            omega=j * 2.0 * np.pi / n_omega,
            k=1.0,
        )
        row = n_phase * (1 + i * n_omega + j) + k
        np.testing.assert_allclose(curves[row], expected, atol=1e-6)

    table, truth, injected = _twin_table(3, ecc=0.0)
    starts = _assignments_by_period(table, P, 3.0)
    assert 1 <= len(starts) <= 3
    assert all(start.any() and period == P for start, period in starts)
    assert len({start.tobytes() for start, _ in starts}) == len(starts)
    # The orbit is circular, so the best curve of the grid is the orbit.
    assert _same_assignment(starts[0][0], injected, truth)
    few = replace(table, blended=np.arange(table.n_epochs) > 1)
    assert _assignments_by_period(few, P, 3.0) == []


@pytest.mark.parametrize("ecc", [0.0, 0.3])
def test_the_assignments_are_also_made_over_a_window_of_period(ecc):
    """A period off by half the frequency resolution is repaired by the window (D68).

    The period given is ``0.5 / T`` from the injected one in frequency, ``T`` being the
    time span, so the phase is off by a quarter of a cycle at either end of the span. The
    assignment made at that period is wrong, and the fit that follows does not leave it:
    none of these six circular tables is recovered and one of the six eccentric ones.
    With a window of ``1 / T`` all twelve are (``scripts/assignment_bench.py`` has the
    rates over more tables and offsets).
    """
    from albireo.rvorbit import _ASSIGNMENT_WINDOW_STEPS, _window_periods

    t = _twin_table(0, ecc=ecc)[0].bjd
    periods = _window_periods(t, P, 1.0)
    assert periods[0] == P and len(periods) == 2 * _ASSIGNMENT_WINDOW_STEPS + 1
    steps = (1.0 / np.array(periods) - 1.0 / P) * np.ptp(t) * _ASSIGNMENT_WINDOW_STEPS
    np.testing.assert_allclose(np.sort(steps), np.arange(-8, 9), atol=1e-9)
    assert np.all(np.diff(np.abs(steps)) >= -1e-9), "ordered by the distance from the period"
    assert _window_periods(t, P, 0.0) == [P] and _window_periods(t[:1], P, 1.0) == [P]

    at_the_period, in_the_window = 0, 0
    for seed in range(6):
        table, _, _ = _twin_table(seed, ecc=ecc)
        off = 1.0 / (1.0 / P + 0.5 / np.ptp(table.bjd))
        at_the_period += _twins_recovered(assign_components(table, period=off)[1])
        _, orbit, _ = assign_components(table, period=off, period_window=1.0)
        in_the_window += _twins_recovered(orbit)
        assert abs(orbit.period / P - 1.0) < 2e-3
    assert at_the_period <= 1 and in_the_window == 6


BLEND_LIGHT = np.array([0.6, 0.4])
BLEND_SIGMA, BLEND_CORRELATION = 0.5, -0.6


def _blended_table(seed, *, n_epochs=30, interchanged=0.0):
    """An unequal pair whose blended epochs are flagged for a second minimum.

    At the epochs with the two velocities 8 to 40 km/s apart the table holds two minima:
    the injected pair, and the pair with the same light-weighted mean and the opposite
    difference, each with its own noise at a correlation of -0.6. Which of the two is the
    one returned is drawn per epoch. With ``interchanged`` that fraction of all the epochs
    also has its two components interchanged. Returns the table, the injected velocities,
    the flagged epochs, those at which the pair returned is the wrong one, and those
    interchanged.
    """
    from albireo.rvorbit import _exchanged

    rng = np.random.default_rng(seed)
    bjd = np.sort(rng.uniform(0.0, 40.0, size=n_epochs))
    orbit = ab.OrbitParams(period=P, t_peri=T_PERI, ecc=ECC, omega=OMEGA, k=(K1, K2), gamma=GAMMA)
    truth = orbit.component_velocities(bjd)
    cov = BLEND_SIGMA**2 * np.array([[1.0, BLEND_CORRELATION], [BLEND_CORRELATION, 1.0]])
    noise = rng.multivariate_normal(np.zeros(2), cov, size=(2, n_epochs))
    difference = truth[0] - truth[1]
    mean = BLEND_LIGHT @ truth
    exchanged = np.stack([mean - BLEND_LIGHT[1] * difference, mean + BLEND_LIGHT[0] * difference])
    right, other = truth + noise[0].T, exchanged + noise[1].T
    flagged = (np.abs(difference) > 8.0) & (np.abs(difference) < 40.0)
    wrong = flagged & (rng.uniform(size=n_epochs) < 0.5)
    covariance = np.repeat(cov[None], n_epochs, axis=0)
    table = replace(
        _table_from(bjd, np.where(wrong, other, right), BLEND_SIGMA),
        covariance=covariance,
        blended=flagged.copy(),
        margin=np.where(flagged, rng.uniform(0.0, 8.0, n_epochs), np.inf),
        alternative=np.where(flagged, np.where(wrong, right, other), np.nan),
        alternative_covariance=np.where(flagged[:, None, None], covariance, np.nan),
    )
    swapped = rng.uniform(size=n_epochs) < interchanged
    if swapped.any():
        table = replace(_exchanged(table, swapped), settings=table.settings)
    return table, truth, flagged, wrong, swapped


@pytest.mark.parametrize("seed", [0, 1, 3])
def test_the_orbit_decides_between_the_two_minima_of_a_flagged_epoch(seed):
    """Every flagged epoch takes the minimum the orbit fits, and is flagged no longer."""
    table, truth, flagged, wrong, _ = _blended_table(seed)
    assert flagged.sum() >= 6 and wrong.any() and not wrong[flagged].all()
    np.testing.assert_array_equal(table.second_minimum, flagged)
    np.testing.assert_array_equal(table.good, ~flagged)
    plain = fit_rv_orbit(table, period=P)
    assert plain.n_points == 2 * int((~flagged).sum())

    assigned, orbit, decided = assign_components(table, period=P, exchange=False)
    np.testing.assert_array_equal(decided.resolved, flagged)
    np.testing.assert_array_equal(decided.alternative, wrong)
    assert not decided.exchanged.any()
    assert assigned.good.all() and orbit.n_points == 2 * table.n_epochs
    assert np.abs(assigned.velocity - truth).max() < 4.0 * BLEND_SIGMA
    np.testing.assert_allclose(orbit.k, [K1, K2], atol=0.3)
    assert orbit.chi2 < 1.5 * (orbit.n_points - orbit.n_parameters)
    # The minimum the correlation returned stays in the table, as the alternative.
    np.testing.assert_array_equal(assigned.alternative[:, wrong], table.velocity[:, wrong])
    np.testing.assert_array_equal(assigned.velocity[:, wrong], table.alternative[:, wrong])
    np.testing.assert_allclose(assigned.margin[wrong], -table.margin[wrong])
    np.testing.assert_allclose(assigned.chi2[wrong], table.chi2[wrong] + table.margin[wrong])
    kept = flagged & ~wrong
    np.testing.assert_array_equal(assigned.velocity[:, kept], table.velocity[:, kept])
    np.testing.assert_array_equal(assigned.margin[kept], table.margin[kept])
    assert assigned.settings["resolved_by_orbit"] == int(flagged.sum())
    assert assigned.settings["second_minimum_by_orbit"] == int(wrong.sum())
    assert assigned.settings["reassigned_by_orbit"] == 0
    np.testing.assert_array_equal(decided.apply(table).velocity, assigned.velocity)

    # One round with given velocities decides the same, whatever their zero points.
    predicted = truth - np.array([[5.0], [-3.0]])
    once, by_prediction = assign_by_orbit(table, predicted, exchange=False)
    assert by_prediction.same_as(decided)
    np.testing.assert_array_equal(once.velocity, assigned.velocity)

    # A velocity removed by the caller closes its epoch: it stays flagged.
    closed = np.flatnonzero(flagged)[0]
    velocity, sigma = np.array(table.velocity), np.array(table.sigma)
    velocity[1, closed] = sigma[1, closed] = np.nan
    gated, _, partial = assign_components(
        replace(table, velocity=velocity, sigma=sigma), period=P, exchange=False
    )
    assert not partial.resolved[closed] and gated.blended[closed]
    assert partial.resolved.sum() == flagged.sum() - 1


@pytest.mark.parametrize("seed", [0, 3])
def test_the_order_and_the_minimum_are_decided_together(seed):
    """With three epochs in ten interchanged as well, both decisions are recovered."""
    table, truth, flagged, wrong, swapped = _blended_table(seed, interchanged=0.3)
    assert swapped.sum() >= 5 and (swapped & flagged).any()
    assigned, orbit, decided = assign_components(table, period=P)
    np.testing.assert_array_equal(decided.exchanged, swapped)
    np.testing.assert_array_equal(decided.alternative, wrong)
    np.testing.assert_array_equal(decided.resolved, flagged)
    assert assigned.good.all()
    assert np.abs(assigned.velocity - truth).max() < 4.0 * BLEND_SIGMA
    np.testing.assert_allclose(orbit.k, [K1, K2], atol=0.3)
    # At an epoch both interchanged and returned at the wrong minimum, the alternative in
    # the table is the wrong minimum in the order of the stars.
    both = swapped & wrong
    if both.any():
        np.testing.assert_array_equal(assigned.alternative[:, both], table.velocity[::-1][:, both])
    # Without the exchange the interchanged epochs stay as they are, and the fit is poor.
    _, held, without = assign_components(table, period=P, exchange=False)
    assert not without.exchanged.any() and held.chi2 > 50.0 * orbit.chi2


def test_an_assignment_that_decides_nothing_returns_the_table():
    table, _, _ = make_table()
    nothing = Assignment.none(table.n_epochs)
    assert not nothing.changes and nothing.apply(table) is table
    assert nothing.same_as(Assignment.none(table.n_epochs))
    same, decided = assign_by_orbit(table, np.zeros((3, table.n_epochs)))
    assert same is table and not decided.changes


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


# ---------------------------------------------------------------------------
# The fit at a held ephemeris
# ---------------------------------------------------------------------------


def _conjunction(orbit) -> float:
    return float(
        t_conj_from_t_peri(orbit.t_peri, period=orbit.period, ecc=orbit.ecc, omega=orbit.omega)
    )


def test_the_fit_at_a_held_ephemeris_is_linear_and_agrees_with_the_free_fit():
    table, _, orbit = make_table(n_epochs=20, sigma=0.5, ecc=0.0)
    held = fit_rv_ephemeris(table, period=P, t_conj=_conjunction(orbit))
    free = fit_rv_orbit(table, period=P, circular=True)
    # Three parameters where the free fit has five, and the ephemeris as given.
    assert held.n_parameters == 3 and held.parameter_names == ("k_A", "k_B", "gamma")
    assert held.period == P and held.errors["period"] == 0.0 and held.errors["t_conj"] == 0.0
    assert held.ecc == 0.0 and held.errors["ecc"] == 0.0
    np.testing.assert_allclose(held.k, [K1, K2], atol=4 * 0.2)
    np.testing.assert_allclose(held.k, free.k, atol=3 * float(np.max(free.errors["k"])))
    assert held.gamma[0] == pytest.approx(GAMMA, abs=0.5) and held.gamma_mode == "shared"
    # The errors are those of the linear model, scaled by its own reduced chi-square.
    np.testing.assert_allclose(held.errors["k"], free.errors["k"], rtol=0.15)
    assert held.chi2 >= free.chi2 - 1e-6
    assert held.covariance.shape == (3, 3)
    # The orbit predicts the velocities it was fitted to, through the compiled predictor.
    residual = table.velocity - held.predict(table.bjd)
    np.testing.assert_allclose(residual, held.residuals, atol=1e-8)
    assert "K_A" in held.summary()


def test_a_held_eccentric_shape_and_a_fitted_one():
    table, _, orbit = make_table(n_epochs=30, sigma=0.3, ecc=0.4)
    t_conj = _conjunction(orbit)
    held = fit_rv_ephemeris(table, period=P, t_conj=t_conj, ecc=0.4, omega=OMEGA)
    np.testing.assert_allclose(held.k, [K1, K2], atol=0.5)
    assert held.ecc == 0.4 and held.omega == OMEGA and held.n_parameters == 3
    fitted = fit_rv_ephemeris(table, period=P, t_conj=t_conj, fit_eccentricity=True)
    assert fitted.n_parameters == 5
    assert fitted.parameter_names[:2] == ("secosw", "sesinw")
    assert fitted.ecc == pytest.approx(0.4, abs=5 * max(fitted.errors["ecc"], 1e-3))
    assert math.remainder(fitted.omega - OMEGA, 2 * math.pi) == pytest.approx(0.0, abs=0.05)
    np.testing.assert_allclose(fitted.k, [K1, K2], atol=0.5)
    assert 0.0 < fitted.errors["ecc"] < 0.02 and fitted.covariance.shape == (5, 5)
    # A circular model of the same velocities fits far worse.
    assert fit_rv_ephemeris(table, period=P, t_conj=t_conj).chi2 > 50.0 * fitted.chi2


def test_the_held_fit_gives_differential_components_their_own_gamma():
    table, _, orbit = make_table(n_epochs=20, absolute=(False, False), offsets=(3.0, -2.0), ecc=0.0)
    fit = fit_rv_ephemeris(table, period=P, t_conj=_conjunction(orbit))
    assert fit.gamma_mode == "one per component" and fit.n_parameters == 4
    np.testing.assert_allclose(fit.gamma, [GAMMA + 3.0, GAMMA - 2.0], atol=0.1)
    np.testing.assert_allclose(fit.k, [K1, K2], atol=0.1)


def test_the_held_fit_holds_a_component_without_velocities_and_refuses_too_few():
    table, _, orbit = make_table(n_epochs=12, ecc=0.0)
    t_conj = _conjunction(orbit)
    velocity = np.array(table.velocity)
    velocity[1] = np.nan
    single = fit_rv_ephemeris(replace(table, velocity=velocity), period=P, t_conj=t_conj)
    assert single.held == ("B",) and single.n_parameters == 2
    assert single.k[0] == pytest.approx(K1, abs=0.1) and np.isnan(single.errors["k"][1])
    assert single.mass_ratio is None
    velocity[0, 1:] = np.nan
    with pytest.raises(ValueError, match="not enough"):
        fit_rv_ephemeris(replace(table, velocity=velocity), period=P, t_conj=t_conj)
    with pytest.raises(ValueError, match="period"):
        fit_rv_ephemeris(table, period=0.0, t_conj=t_conj)
    with pytest.raises(ValueError, match="ecc"):
        fit_rv_ephemeris(table, period=P, t_conj=t_conj, ecc=1.2)


def _twin_ephemeris(seed, n_epochs, ecc):
    """The periastron time and argument that :func:`_twin_table` drew, as a conjunction."""
    rng = np.random.default_rng(seed)
    rng.uniform(0.0, 40.0, size=n_epochs)
    t_peri, omega = rng.uniform(0.0, P), rng.uniform(0.0, 2.0 * np.pi)
    return float(t_conj_from_t_peri(t_peri, period=P, ecc=ecc, omega=omega)), omega


@pytest.mark.parametrize("ecc", [0.0, 0.3])
def test_the_ephemeris_orders_the_velocities_of_alike_stars(ecc):
    for seed in range(8):
        table, truth, exchanged = _twin_table(seed, ecc=ecc)
        t_conj, omega = _twin_ephemeris(seed, table.n_epochs, ecc)
        assigned, fitted, decided = assign_by_ephemeris(
            table, period=P, t_conj=t_conj, ecc=ecc, omega=omega
        )
        # The ephemeris names the stars: the semi-amplitudes come out in the injected
        # order, and not in either order as from the period alone.
        np.testing.assert_allclose(fitted.k, TWIN_K, rtol=0.02)
        np.testing.assert_allclose(assigned.velocity, truth, atol=4.0)
        # The exchanges are the injected ones wherever the two velocities differ by more
        # than their errors. Near a conjunction either order is the same measurement.
        resolved = np.abs(truth[0] - truth[1]) > 5.0
        assert np.array_equal(decided.exchanged[resolved], exchanged[resolved])


def test_the_ephemeris_does_not_exchange_unalike_light_fractions():
    table, _, _ = _twin_table(0, ecc=0.0)
    t_conj, _ = _twin_ephemeris(0, table.n_epochs, 0.0)
    unalike = replace(table, light=np.repeat(np.array([[0.9], [0.1]]), table.n_epochs, axis=1))
    same, _, decided = assign_by_ephemeris(unalike, period=P, t_conj=t_conj)
    assert same is unalike and not decided.changes
    _, _, forced = assign_by_ephemeris(unalike, period=P, t_conj=t_conj, light_ratio_max=None)
    assert forced.exchanged.any()


@pytest.mark.parametrize("ecc", [0.4, 0.6])
def test_the_order_on_an_eccentric_orbit_does_not_need_its_shape(ecc):
    """With the eccentricity fitted, the pairs are ordered by the shape that fits.

    The sign of ``v_1 - v_2`` is that of the velocity curve, and at the conjunction of an
    eccentric orbit the curve is ``e cos(omega)``, not zero. Ordering the pairs by the
    circular curve exchanged epochs that were measured in the right order, and the fit
    kept them: 16 of 20 such tables were recovered at ``e = 0.4`` and 12 of 20 at 0.6.
    """
    for seed in range(8):
        table, truth, exchanged = _twin_table(seed, n_epochs=24, ecc=ecc)
        t_conj, _ = _twin_ephemeris(seed, table.n_epochs, ecc)
        _, fitted, decided = assign_by_ephemeris(
            table, period=P, t_conj=t_conj, fit_eccentricity=True
        )
        np.testing.assert_allclose(fitted.k, TWIN_K, rtol=0.05)
        assert fitted.ecc == pytest.approx(ecc, abs=0.05)
        resolved = np.abs(truth[0] - truth[1]) > 5.0
        assert np.array_equal(decided.exchanged[resolved], exchanged[resolved])
        # The same table with every pair already in the injected order is left alone.
        ordered = replace(table, velocity=np.where(exchanged, table.velocity[::-1], table.velocity))
        _, again, kept = assign_by_ephemeris(
            ordered, period=P, t_conj=t_conj, fit_eccentricity=True
        )
        assert not kept.exchanged[resolved].any()
        # The two fits differ only in the order of the pairs too close to tell apart.
        np.testing.assert_allclose(again.k, fitted.k, rtol=0.01)

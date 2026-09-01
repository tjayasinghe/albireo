"""Tests for the Keplerian fit to a velocity table, and the period search."""

from __future__ import annotations

import math

import numpy as np
import pytest

import albireo as ab
from albireo.rvorbit import find_period, fit_rv_orbit
from albireo.todcor import VelocityTable

P, T_PERI, ECC, OMEGA, K1, K2, GAMMA = 6.31, 2.0, 0.15, 0.7, 30.0, 55.0, 12.0


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

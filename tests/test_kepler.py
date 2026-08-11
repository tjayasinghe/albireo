"""Tests for the differentiable Kepler solver and radial-velocity law."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from albireo.kepler import radial_velocity, solve_kepler, t_peri_from_t_conj, true_anomaly

ECCS = [0.0, 0.1, 0.3, 0.6, 0.9, 0.95]


@pytest.mark.parametrize("ecc", ECCS)
def test_kepler_equation_residual(ecc):
    m = jnp.linspace(-3 * jnp.pi, 3 * jnp.pi, 301)
    e_anom = solve_kepler(m, ecc)
    resid = e_anom - ecc * jnp.sin(e_anom) - m
    assert float(jnp.max(jnp.abs(resid))) < 1e-12


def test_branch_consistency():
    m = jnp.linspace(-jnp.pi, jnp.pi, 101)
    e1 = solve_kepler(m, 0.5)
    e2 = solve_kepler(m + 2 * jnp.pi, 0.5)
    np.testing.assert_allclose(np.asarray(e2 - e1), 2 * np.pi, rtol=0, atol=1e-12)


def test_circular_orbit():
    m = jnp.linspace(-3.0, 3.0, 41)
    np.testing.assert_allclose(np.asarray(solve_kepler(m, 0.0)), np.asarray(m), atol=1e-13)
    np.testing.assert_allclose(
        np.asarray(true_anomaly(m, 0.0)), np.asarray(m), atol=1e-13
    )  # nu == E == M at e = 0


def test_solver_gradients_match_implicit_analytic():
    ecc = 0.4
    m = jnp.linspace(-3.0, 3.0, 25)
    e_anom = solve_kepler(m, ecc)
    denom = 1.0 - ecc * jnp.cos(e_anom)

    de_dm = jax.vmap(jax.grad(lambda mm: solve_kepler(mm, ecc)))(m)
    np.testing.assert_allclose(np.asarray(de_dm), np.asarray(1.0 / denom), rtol=1e-10)

    de_de = jax.vmap(jax.grad(lambda mm, ee: solve_kepler(mm, ee), argnums=1))(
        m, jnp.full_like(m, ecc)
    )
    np.testing.assert_allclose(np.asarray(de_de), np.asarray(jnp.sin(e_anom) / denom), rtol=1e-10)


def test_solver_gradients_match_fd():
    m0, e0 = 1.234, 0.55
    h = 1e-6
    g_m = float(jax.grad(lambda m: solve_kepler(m, e0))(m0))
    fd_m = float((solve_kepler(m0 + h, e0) - solve_kepler(m0 - h, e0)) / (2 * h))
    np.testing.assert_allclose(g_m, fd_m, rtol=1e-6)
    g_e = float(jax.grad(lambda e: solve_kepler(m0, e))(e0))
    fd_e = float((solve_kepler(m0, e0 + h) - solve_kepler(m0, e0 - h)) / (2 * h))
    np.testing.assert_allclose(g_e, fd_e, rtol=1e-6)


ORBIT = dict(period=10.0, t_peri=2.3, ecc=0.55, omega=1.1, k=42.0, gamma=-7.5)


def test_rv_peak_to_peak_is_2k():
    t = jnp.linspace(0.0, ORBIT["period"], 100_001)
    v = radial_velocity(t, **ORBIT)
    ptp = float(jnp.max(v) - jnp.min(v))
    np.testing.assert_allclose(ptp, 2 * ORBIT["k"], rtol=1e-6)


def test_rv_periodicity():
    t = jnp.asarray(np.random.default_rng(3).uniform(0.0, 50.0, size=64))
    v1 = radial_velocity(t, **ORBIT)
    v2 = radial_velocity(t + ORBIT["period"], **ORBIT)
    np.testing.assert_allclose(np.asarray(v1), np.asarray(v2), atol=1e-10)


def test_rv_at_periastron():
    v = float(radial_velocity(jnp.asarray(ORBIT["t_peri"]), **ORBIT))
    expected = ORBIT["gamma"] + ORBIT["k"] * (1 + ORBIT["ecc"]) * np.cos(ORBIT["omega"])
    np.testing.assert_allclose(v, expected, rtol=1e-12)


def test_sb2_components_are_in_antiphase():
    k1, k2 = 60.0, 95.0
    t = jnp.linspace(0.0, 10.0, 57)
    base = dict(period=10.0, t_peri=2.3, ecc=0.4, gamma=12.0)
    v1 = radial_velocity(t, omega=1.1, k=k1, **base)
    v2 = radial_velocity(t, omega=1.1 + jnp.pi, k=k2, **base)
    np.testing.assert_allclose(
        np.asarray((v1 - base["gamma"]) * k2), np.asarray(-(v2 - base["gamma"]) * k1), rtol=1e-10
    )


def test_t_conj_convention():
    period, ecc, omega = 10.0, 0.55, 1.1
    t_conj = 4.2
    t_peri = float(t_peri_from_t_conj(t_conj, period=period, ecc=ecc, omega=omega))
    m_conj = 2 * np.pi * (t_conj - t_peri) / period
    nu = float(true_anomaly(solve_kepler(jnp.asarray(m_conj), ecc), ecc))
    # nu + omega = pi/2 (mod 2 pi)
    resid = (nu + omega - np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
    np.testing.assert_allclose(resid, 0.0, atol=1e-10)


def test_rv_gradients_vs_fd():
    t0 = 5.7

    def rv(params):
        period, t_peri, ecc, omega, k = params
        return radial_velocity(
            jnp.asarray(t0), period=period, t_peri=t_peri, ecc=ecc, omega=omega, k=k, gamma=0.0
        )

    p0 = jnp.asarray([10.0, 2.3, 0.55, 1.1, 42.0])
    grad = np.asarray(jax.grad(rv)(p0))
    for i in range(5):
        h = 1e-6 * max(1.0, abs(float(p0[i])))
        dp = np.zeros(5)
        dp[i] = h
        fd = float((rv(p0 + dp) - rv(p0 - dp)) / (2 * h))
        np.testing.assert_allclose(grad[i], fd, rtol=2e-5, err_msg=f"param index {i}")

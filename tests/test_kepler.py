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


@pytest.mark.parametrize("ecc", [0.0, 0.15, 0.6, 0.9])
@pytest.mark.parametrize("omega", [0.0, 0.7, 2.5, -1.2, 3.1])
def test_t_conj_and_t_peri_are_inverses(ecc, omega):
    from albireo.kepler import t_conj_from_t_peri, t_peri_from_t_conj

    period = 6.31
    t_peri = 2457001.234
    t_conj = float(t_conj_from_t_peri(t_peri, period=period, ecc=ecc, omega=omega))
    assert t_peri <= t_conj < t_peri + period
    back = float(t_peri_from_t_conj(t_conj, period=period, ecc=ecc, omega=omega))
    assert abs((back - t_peri + 0.5 * period) % period - 0.5 * period) < 1e-9
    # And the velocity there is the systemic one: nu + omega = pi/2 puts cos(nu + omega) = 0.
    v = float(radial_velocity(t_conj, period=period, t_peri=t_peri, ecc=ecc, omega=omega, k=50.0))
    assert abs(v - 50.0 * ecc * np.cos(omega)) < 1e-6


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


def test_the_solver_is_compiled_once_per_shape_outside_jit(compilations):
    # The Newton step is a function of the module with the eccentricity and the mean anomaly
    # in the loop state. As a closure made on every call it had the loop compiled on every
    # call: 29 to 40 ms and 1.9 MB of retained memory per call of radial_velocity.
    t = jnp.linspace(0.0, 30.0, 23)

    def call(seed):
        rng = np.random.default_rng(seed)
        return radial_velocity(
            t,
            period=float(rng.uniform(1.0, 20.0)),
            t_peri=float(rng.uniform(0.0, 5.0)),
            ecc=float(rng.uniform(0.0, 0.8)),
            omega=float(rng.uniform(0.0, 6.0)),
            k=float(rng.uniform(10.0, 150.0)),
            gamma=float(rng.normal(0.0, 30.0)),
        )

    call(0)
    assert compilations(lambda: [call(seed) for seed in range(1, 21)]) == 0
    # an array eccentricity is another shape: compiled once, then reused
    eccs = jnp.linspace(0.0, 0.7, 23)
    solve_kepler(t, eccs)
    assert compilations(lambda: [solve_kepler(t + shift, eccs) for shift in (0.1, 0.2, 0.3)]) == 0


def test_the_loop_state_form_returns_the_values_of_the_closure_form():
    # The solver as it was written before, with the step a closure over e and M. On the
    # development machine the two forms agree bit for bit; the tolerance is a few units
    # in the last place, for a platform that rounds one operation of the loop differently.
    def same(got, expected):
        np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=2e-15, atol=2e-15)

    def closure_form(mean_anomaly, ecc):
        m = jnp.asarray(mean_anomaly, dtype=jnp.result_type(float, mean_anomaly))
        e = jnp.asarray(ecc)
        m_wrapped = jnp.mod(m + jnp.pi, 2.0 * jnp.pi) - jnp.pi
        e_curr = m_wrapped + e * jnp.sin(m_wrapped) + 0.5 * e**2 * jnp.sin(2.0 * m_wrapped)

        def newton_step(_, e_curr):
            f = e_curr - e * jnp.sin(e_curr) - m_wrapped
            fp = 1.0 - e * jnp.cos(e_curr)
            return e_curr - f / fp

        return jax.lax.fori_loop(0, 15, newton_step, e_curr) + (m - m_wrapped)

    rng = np.random.default_rng(3)
    for size in (1, 7, 33, 200):
        m = jnp.asarray(rng.uniform(-20.0, 20.0, size))
        for ecc in (0.0, 0.1, 0.63, 0.95, jnp.asarray(rng.uniform(0.0, 0.9, size))):
            same(solve_kepler(m, ecc), closure_form(m, ecc))
    m = jnp.asarray(rng.uniform(-20.0, 20.0, 64))
    same(jax.jit(solve_kepler)(m, 0.4), jax.jit(closure_form)(m, 0.4))
    eccs = jnp.asarray(rng.uniform(0.0, 0.9, 5))
    same(
        jax.vmap(lambda e: solve_kepler(m, e))(eccs),
        jax.vmap(lambda e: closure_form(m, e))(eccs),
    )


def test_the_numpy_true_anomaly_is_that_of_the_jax_solver():
    from albireo.kepler import (
        _t_peri_from_t_conj_numpy,
        solve_kepler,
        t_peri_from_t_conj,
        true_anomaly,
        true_anomaly_numpy,
    )

    t = np.linspace(-7.0, 31.0, 1501)
    for ecc in (0.0, 0.2, 0.79, 0.81, 0.95):
        mean = 2.0 * np.pi * (t - 1.3) / 3.7
        expected = np.asarray(true_anomaly(solve_kepler(mean, ecc), ecc))
        got = true_anomaly_numpy(t, period=3.7, t_peri=1.3, ecc=ecc)
        assert got.shape == t.shape and np.all(np.abs(got) <= np.pi + 1e-12)
        # The same angle: the two differ by whole turns only.
        np.testing.assert_allclose(np.angle(np.exp(1j * (got - expected))), 0.0, atol=1e-11)
    # At periastron the true anomaly is zero, and half a period later it is pi.
    np.testing.assert_allclose(
        np.abs(true_anomaly_numpy([1.3, 1.3 + 1.85], period=3.7, t_peri=1.3, ecc=0.4)),
        [0.0, np.pi],
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="ecc"):
        true_anomaly_numpy(t, period=3.7, t_peri=1.3, ecc=1.0)
    for ecc, omega in ((0.0, 0.0), (0.4, 1.1), (0.7, 5.0)):
        expected = float(t_peri_from_t_conj(10.0, period=3.3, ecc=ecc, omega=omega))
        got = _t_peri_from_t_conj_numpy(10.0, period=3.3, ecc=ecc, omega=omega)
        assert got == pytest.approx(expected, abs=1e-12)

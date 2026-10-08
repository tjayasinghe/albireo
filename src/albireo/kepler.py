"""Differentiable Keplerian orbits and radial-velocity laws.

Kepler's equation, the true anomaly, the Keplerian radial-velocity law of
``docs/math.md`` §1.2, and the conversion between times of conjunction and periastron.
All functions are vectorized over time and differentiable in every parameter.

The eccentric-anomaly solver is a fixed-count Newton iteration wrapped in
``jax.custom_jvp`` with the implicit-function tangent rule (``docs/math.md`` §1.2), so
gradients are exact at the converged solution regardless of the iteration count.

Angle convention: ``omega`` is the argument of periastron of the component whose radial
velocity is being computed, in radians. Component 2 of a binary uses ``omega + pi``.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "radial_velocity",
    "solve_kepler",
    "t_conj_from_t_peri",
    "t_peri_from_t_conj",
    "true_anomaly",
    "true_anomaly_numpy",
]

_NEWTON_ITERATIONS = 15


@jax.custom_jvp
def solve_kepler(mean_anomaly, ecc):
    """Solve Kepler's equation ``E - e sin(E) = M`` for the eccentric anomaly.

    Parameters
    ----------
    mean_anomaly
        Mean anomaly ``M`` in radians (any branch; scalar or array).
    ecc
        Eccentricity, ``0 <= e < 1``. Accuracy is verified in the tests up to
        ``e = 0.95``; the solver is not intended for near-parabolic orbits.

    Returns
    -------
    jax.Array
        Eccentric anomaly ``E`` on the same branch as ``M`` (i.e. ``E - M`` is
        2π-periodic in ``M``).

    Notes
    -----
    The iteration starts from the third-order series in ``e`` and takes a fixed number of
    Newton steps on the principal branch of ``M``. The tangent is the implicit-function
    rule ``dE = (dM + sin(E) de) / (1 - e cos(E))`` (``docs/math.md`` §1.2), evaluated at
    the returned ``E``, so the gradient does not depend on the iteration count.
    """
    m = jnp.asarray(mean_anomaly, dtype=jnp.result_type(float, mean_anomaly))
    e = jnp.asarray(ecc)
    # Iterate on the principal branch, restore the branch offset afterwards.
    m_wrapped = jnp.mod(m + jnp.pi, 2.0 * jnp.pi) - jnp.pi
    # Third-order series starter, then fixed-count Newton (quadratic convergence).
    e_curr = m_wrapped + e * jnp.sin(m_wrapped) + 0.5 * e**2 * jnp.sin(2.0 * m_wrapped)
    e_final, _, _ = jax.lax.fori_loop(0, _NEWTON_ITERATIONS, _newton_step, (e_curr, e, m_wrapped))
    return e_final + (m - m_wrapped)


def _newton_step(_, state):
    """One Newton step of :func:`solve_kepler` on ``(E, e, M)``, which returns ``e`` and ``M``.

    The eccentricity and the mean anomaly are carried in the loop state and the step is a
    function of the module. Outside ``jit``, JAX compiles a loop once per body function and
    argument shapes, and keeps the program. A step defined inside :func:`solve_kepler`, as
    a closure over the two, is a new body function on every call, and the loop was compiled
    on every call: over 1500 calls on arrays of one shape, 29 to 40 ms and 1.9 MB of
    retained memory per call. With this step the same calls take 0.5 ms each and the memory
    does not grow. The values returned are the same bit for bit, outside ``jit``, under
    ``jit`` and ``vmap``, and through the derivative rule.
    """
    e_curr, e, m_wrapped = state
    f = e_curr - e * jnp.sin(e_curr) - m_wrapped
    fp = 1.0 - e * jnp.cos(e_curr)
    return e_curr - f / fp, e, m_wrapped


@solve_kepler.defjvp
def _solve_kepler_jvp(primals, tangents):
    # Implicit differentiation of E - e sin(E) = M (docs/math.md section 1.2):
    #   dE = (dM + sin(E) de) / (1 - e cos(E))
    m, e = primals
    dm, de = tangents
    e_anom = solve_kepler(m, e)
    denom = 1.0 - jnp.asarray(e) * jnp.cos(e_anom)
    tangent = (jnp.broadcast_to(dm, e_anom.shape) + jnp.sin(e_anom) * de) / denom
    return e_anom, tangent


def true_anomaly(ecc_anomaly, ecc):
    """True anomaly ``nu`` from eccentric anomaly, via the half-angle identity."""
    e_anom = jnp.asarray(ecc_anomaly)
    ecc = jnp.asarray(ecc)
    return 2.0 * jnp.arctan2(
        jnp.sqrt(1.0 + ecc) * jnp.sin(0.5 * e_anom),
        jnp.sqrt(1.0 - ecc) * jnp.cos(0.5 * e_anom),
    )


def radial_velocity(t, *, period, t_peri, ecc, omega, k, gamma=0.0):
    """Keplerian radial velocity ``v(t) = gamma + K [cos(nu + omega) + e cos(omega)]``.

    The radial-velocity law of ``docs/math.md`` §1.2, with the true anomaly ``nu`` from
    :func:`solve_kepler` and :func:`true_anomaly`.

    Parameters
    ----------
    t
        Times (same unit and zero-point as ``t_peri``; BJD_TDB in practice).
    period, t_peri, ecc, omega, k, gamma
        Orbital period, time of periastron passage, eccentricity, argument of
        periastron [rad], RV semi-amplitude, and systemic velocity. ``k`` and ``gamma``
        set the output unit (km/s in practice). Component 2 of a binary takes
        ``omega + pi`` and ``K_2``.

    Returns
    -------
    jax.Array
        Radial velocity at ``t``; positive = receding.
    """
    mean_anomaly = 2.0 * jnp.pi * (jnp.asarray(t) - t_peri) / period
    nu = true_anomaly(solve_kepler(mean_anomaly, ecc), ecc)
    return gamma + k * (jnp.cos(nu + omega) + ecc * jnp.cos(omega))


def t_peri_from_t_conj(t_conj, *, period, ecc, omega):
    """Time of periastron from a time of conjunction.

    The conjunction convention is ``nu(t_conj) + omega = pi/2``, the superior conjunction
    of the component whose ``omega`` is given. The line-of-sight displacement is
    ``r sin(i) sin(nu + omega)`` measured away from the observer, so at ``nu + omega =
    pi/2`` that component is farthest behind the plane of the sky and, in an eclipsing
    system, is the one eclipsed. With the primary's ``omega``, ``t_conj`` is the time of
    primary eclipse. The inverse mapping is ``nu -> E -> M -> t``.
    """
    nu_conj = 0.5 * jnp.pi - omega
    e_conj = 2.0 * jnp.arctan2(
        jnp.sqrt(1.0 - ecc) * jnp.sin(0.5 * nu_conj),
        jnp.sqrt(1.0 + ecc) * jnp.cos(0.5 * nu_conj),
    )
    m_conj = e_conj - ecc * jnp.sin(e_conj)
    return t_conj - m_conj * period / (2.0 * jnp.pi)


def t_conj_from_t_peri(t_peri, *, period, ecc, omega):
    """Time of conjunction from a time of periastron: the inverse of :func:`t_peri_from_t_conj`.

    Same convention, ``nu(t_conj) + omega = pi/2``, the superior conjunction of the
    component whose ``omega`` is given (its eclipse in an eclipsing system). Returns the
    conjunction in the same cycle as ``t_peri``, at or after it by less than one period.
    """
    nu_conj = 0.5 * jnp.pi - omega
    e_conj = 2.0 * jnp.arctan2(
        jnp.sqrt(1.0 - ecc) * jnp.sin(0.5 * nu_conj),
        jnp.sqrt(1.0 + ecc) * jnp.cos(0.5 * nu_conj),
    )
    m_conj = e_conj - ecc * jnp.sin(e_conj)
    return t_peri + jnp.mod(m_conj, 2.0 * jnp.pi) * period / (2.0 * jnp.pi)


def true_anomaly_numpy(t, *, period: float, t_peri: float, ecc: float) -> np.ndarray:
    """True anomaly at times ``t`` for a Keplerian orbit, in NumPy.

    Kepler's equation is solved by Newton's method on the mean anomaly folded to
    ``[0, pi]``, from ``E = pi`` where ``e > 0.8`` and from ``E = M + e sin M`` elsewhere,
    until no step exceeds 1e-13. The values are those of :func:`solve_kepler` and
    :func:`true_anomaly` to 1e-12 up to ``e = 0.95`` and to 6e-12 at ``e = 0.99``. Outside
    ``jit`` the JAX solver is compiled once for every array length. A caller that evaluates
    many orbits at different numbers of epochs and needs no derivative uses this function.

    Parameters
    ----------
    t
        Times, in the unit and zero point of ``t_peri``.
    period, t_peri, ecc
        Period, time of periastron passage and eccentricity, scalars.

    Returns
    -------
    numpy.ndarray
        True anomaly in ``(-pi, pi]``, of the shape of ``t``.
    """
    t = np.asarray(t, dtype=np.float64)
    e = float(ecc)
    if not 0.0 <= e < 1.0:
        raise ValueError(f"ecc must lie in [0, 1); got {ecc}")
    mean = 2.0 * math.pi * (((t - float(t_peri)) / float(period)) % 1.0)
    mean = np.where(mean > math.pi, mean - 2.0 * math.pi, mean)
    if e == 0.0:
        return mean
    sign, m = np.sign(mean), np.abs(mean)
    big = np.full_like(m, math.pi) if e > 0.8 else m + e * np.sin(m)
    for _ in range(100):
        step = (big - e * np.sin(big) - m) / (1.0 - e * np.cos(big))
        big = big - step
        if not np.any(np.abs(step) > 1e-13):
            break
    nu = 2.0 * np.arctan2(
        math.sqrt(1.0 + e) * np.sin(0.5 * big), math.sqrt(1.0 - e) * np.cos(0.5 * big)
    )
    return sign * nu


def _t_peri_from_t_conj_numpy(t_conj: float, *, period: float, ecc: float, omega: float) -> float:
    """:func:`t_peri_from_t_conj` for scalars, in NumPy."""
    nu_conj = 0.5 * math.pi - omega
    e_conj = 2.0 * math.atan2(
        math.sqrt(1.0 - ecc) * math.sin(0.5 * nu_conj),
        math.sqrt(1.0 + ecc) * math.cos(0.5 * nu_conj),
    )
    return t_conj - (e_conj - ecc * math.sin(e_conj)) * period / (2.0 * math.pi)

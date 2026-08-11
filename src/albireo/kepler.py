"""Differentiable Keplerian orbits and radial-velocity laws.

The eccentric-anomaly solver uses a fixed-count Newton iteration wrapped in
``jax.custom_jvp`` with the implicit-function-theorem tangent rule, so gradients are
exact at the converged solution regardless of the iteration count (``docs/math.md``
§1.2). All functions are vectorized over time and differentiable in every parameter.

Angle convention: ``omega`` is the argument of periastron of the component whose radial
velocity is being computed, in radians; component 2 of a binary uses ``omega + pi``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = [
    "radial_velocity",
    "solve_kepler",
    "t_peri_from_t_conj",
    "true_anomaly",
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
    """
    m = jnp.asarray(mean_anomaly, dtype=jnp.result_type(float, mean_anomaly))
    e = jnp.asarray(ecc)
    # Iterate on the principal branch, restore the branch offset afterwards.
    m_wrapped = jnp.mod(m + jnp.pi, 2.0 * jnp.pi) - jnp.pi
    # Third-order series starter, then fixed-count Newton (quadratic convergence).
    e_curr = m_wrapped + e * jnp.sin(m_wrapped) + 0.5 * e**2 * jnp.sin(2.0 * m_wrapped)

    def newton_step(_, e_curr):
        f = e_curr - e * jnp.sin(e_curr) - m_wrapped
        fp = 1.0 - e * jnp.cos(e_curr)
        return e_curr - f / fp

    e_final = jax.lax.fori_loop(0, _NEWTON_ITERATIONS, newton_step, e_curr)
    return e_final + (m - m_wrapped)


@solve_kepler.defjvp
def _solve_kepler_jvp(primals, tangents):
    # Implicit differentiation of E - e sin(E) = M:
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

    Parameters
    ----------
    t
        Times (same unit and zero-point as ``t_peri``; BJD_TDB in practice).
    period, t_peri, ecc, omega, k, gamma
        Orbital period, time of periastron passage, eccentricity, argument of
        periastron [rad], RV semi-amplitude, and systemic velocity. ``k`` and ``gamma``
        set the output unit (km/s in practice). For component 2 of a binary pass
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

    The conjunction convention is ``nu(t_conj) + omega = pi/2`` — the inferior
    conjunction of the component whose ``omega`` is given (for eclipsing systems: the
    time this component passes in front). Inverse mapping: ``nu -> E -> M -> t``.
    """
    nu_conj = 0.5 * jnp.pi - omega
    e_conj = 2.0 * jnp.arctan2(
        jnp.sqrt(1.0 - ecc) * jnp.sin(0.5 * nu_conj),
        jnp.sqrt(1.0 + ecc) * jnp.cos(0.5 * nu_conj),
    )
    m_conj = e_conj - ecc * jnp.sin(e_conj)
    return t_conj - m_conj * period / (2.0 * jnp.pi)

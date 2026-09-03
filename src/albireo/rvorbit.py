"""Keplerian orbits fitted to a velocity table.

**Experimental.** The names and return shapes defined here may change; the joint
path of ``docs/math.md`` §7 fits the same Keplerian from the spectra directly.

The main path of albireo infers the orbit from the spectra directly (``docs/math.md`` §7)
and does not use this module. :mod:`albireo.todcor` produces one velocity per component
per epoch, and this module fits a Keplerian to such a table with the same Kepler solver
and angle conventions as the joint model (:mod:`albireo.kepler`,
:func:`albireo.orbit_velocities`), so that the two routes can be compared element by
element (``docs/math.md`` §10.6).

The fit is weighted nonlinear least squares over the sampled elements: period, time of
conjunction, ``(sqrt(e) cos w, sqrt(e) sin w)``, one semi-amplitude per component and a
systemic velocity, with the Jacobian computed by JAX. The systemic velocity is a single
parameter when every component's velocities are absolute and one parameter per component
otherwise: a table built from disentangled templates carries one unidentified zero point
per component (``docs/math.md`` §7.6), and a shared gamma would then absorb two different
constants and bias both semi-amplitudes. :class:`RVOrbit` records which was used in
``gamma_mode``.

Uncertainties are the curvature errors at the optimum scaled by the reduced chi-square. The
per-epoch errors from :func:`albireo.todcor` are curvature errors of a template fit and
exclude template mismatch, line-profile variability and any third body, so the scatter of a
table about a Keplerian is generally larger than they imply.

Minimum masses and projected semi-axes follow Hilditch (2001), eqs. 3.17 and 3.18, with the
IAU 2015 nominal constants. :func:`find_period` is a Lomb-Scargle periodogram (Lomb 1976;
Scargle 1982; VanderPlas 2018).

References
----------
Hilditch, R. W. 2001, An Introduction to Close Binary Stars (Cambridge University Press)
Lomb, N. R. 1976, Ap&SS, 39, 447
Scargle, J. D. 1982, ApJ, 263, 835
VanderPlas, J. T. 2018, ApJS, 236, 16
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from albireo.grids import C_KMS
from albireo.kepler import radial_velocity, t_peri_from_t_conj

__all__ = ["RVOrbit", "find_period", "fit_rv_orbit"]

# Minimum masses and projected semi-axes in solar units from km/s and days
# (Hilditch 2001, eqs. 3.17 and 3.18, with the IAU 2015 nominal constants).
_MSIN3I_COEFF = 1.0361e-7  # M sin^3 i [M_sun] = coeff (1-e^2)^{3/2} (K_1 + K_2)^2 K_j P
_ASINI_COEFF = 86400.0 / (2.0 * math.pi) / 695_700.0  # a sin i [R_sun] = coeff K P sqrt(1-e^2)


def _table_arrays(table, components):
    names = list(table.names)
    if components is None:
        components = names
    for name in components:
        if name not in names:
            raise ValueError(f"no component {name!r}; the table has {names}")
    idx = [names.index(name) for name in components]
    v = np.asarray(table.velocity, dtype=np.float64)[idx]
    s = np.asarray(table.sigma, dtype=np.float64)[idx]
    absolute = [bool(table.absolute[i]) for i in idx]
    return list(components), v, s, absolute


def find_period(
    table,
    *,
    period_range: tuple[float, float] | None = None,
    n_frequencies: int = 20_000,
    components=None,
) -> dict:
    """Lomb-Scargle search for the orbital period of a velocity table.

    For two or more components the periodogram is computed on the difference of the first
    two components' velocities, which is free of both systemic velocities and both
    template zero points and has amplitude ``K_1 + K_2``. A single component is searched
    as it is. In either case the weighted mean is removed, the series is scaled by the
    square root of its weights, and the normalized periodogram of
    ``scipy.signal.lombscargle`` is evaluated on a grid uniform in frequency.

    Parameters
    ----------
    table
        A :class:`~albireo.todcor.VelocityTable`.
    period_range
        ``(shortest, longest)`` period in days. Default: twice the shortest epoch gap to
        twice the baseline.
    n_frequencies
        Size of the frequency grid, uniform in frequency.
    components
        Which components to use (names). Default: all, in table order.

    Returns
    -------
    dict
        ``period`` (best, days), ``periods`` and ``power`` (the periodogram), and
        ``aliases``, the five next-best peaks whose periods differ from each other and
        from the best by more than 2%. The periodogram of a sparsely sampled table is
        rarely unambiguous, and the aliases should be inspected.

    References
    ----------
    Lomb, N. R. 1976, Ap&SS, 39, 447
    Scargle, J. D. 1982, ApJ, 263, 835
    VanderPlas, J. T. 2018, ApJS, 236, 16
    """
    from scipy.signal import lombscargle

    _, v, s, _ = _table_arrays(table, components)
    good = np.all(np.isfinite(v), axis=0) & np.all(np.isfinite(s), axis=0) & table.good
    if int(good.sum()) < 4:
        raise ValueError(f"only {int(good.sum())} usable epochs; a period search needs at least 4")
    t = np.asarray(table.bjd, dtype=np.float64)[good]
    if v.shape[0] >= 2:
        y = v[0, good] - v[1, good]
        w = 1.0 / (s[0, good] ** 2 + s[1, good] ** 2)
    else:
        y = v[0, good]
        w = 1.0 / s[0, good] ** 2
    y = y - np.sum(w * y) / np.sum(w)
    if period_range is None:
        gaps = np.diff(np.sort(t))
        shortest = 2.0 * float(np.min(gaps[gaps > 0]))
        longest = 2.0 * float(t.max() - t.min())
        period_range = (shortest, longest)
    lo, hi = period_range
    if not (0.0 < lo < hi):
        raise ValueError(f"period_range must satisfy 0 < shortest < longest; got {period_range}")
    freqs = np.linspace(1.0 / hi, 1.0 / lo, int(n_frequencies))
    power = lombscargle(t, y * np.sqrt(w), 2.0 * np.pi * freqs, normalize=True)
    order = np.argsort(power)[::-1]
    peaks: list[float] = []
    for k in order:
        p = 1.0 / freqs[k]
        if all(abs(p / q - 1.0) > 0.02 for q in peaks):
            peaks.append(float(p))
        if len(peaks) >= 6:
            break
    return {
        "period": peaks[0],
        "periods": 1.0 / freqs,
        "power": power,
        "aliases": peaks[1:],
    }


@dataclass(frozen=True)
class RVOrbit:
    """A Keplerian fitted to a velocity table.

    Attributes
    ----------
    names
        Components, in the order of ``k`` and ``gamma``.
    period, t_conj, ecc, omega
        The elements: period [d], time of conjunction (``nu + omega = pi/2`` for the first
        component, its superior conjunction, so the eclipse of that component in an
        eclipsing system), eccentricity, and the first component's argument of periastron
        [rad]. The second component uses ``omega + pi``.
    k
        Semi-amplitudes [km/s], one per component.
    gamma
        Systemic velocity [km/s]: one value repeated when it was shared, one per component
        when each carried its own zero point.
    gamma_mode
        ``"shared"`` or ``"one per component"``.
    errors
        Standard errors for every fitted quantity, keyed like the attributes (``k`` and
        ``gamma`` hold arrays), after the reduced-chi-square rescaling.
    chi2, n_points, n_parameters
        The weighted chi-square at the optimum and its degrees of freedom.
    residuals
        ``(n_comp, n_epochs)`` observed minus model [km/s], NaN where an epoch was unused.
    rms
        Per-component RMS of the residuals over the used epochs.
    used
        Per epoch: whether the epoch entered the fit.
    """

    names: tuple[str, ...]
    period: float
    t_conj: float
    ecc: float
    omega: float
    k: np.ndarray
    gamma: np.ndarray
    gamma_mode: str
    errors: dict
    chi2: float
    n_points: int
    n_parameters: int
    residuals: np.ndarray
    rms: np.ndarray
    used: np.ndarray
    covariance: np.ndarray = field(repr=False)
    parameter_names: tuple[str, ...] = field(repr=False, default=())

    @property
    def t_peri(self) -> float:
        """Time of periastron passage."""
        return float(
            t_peri_from_t_conj(self.t_conj, period=self.period, ecc=self.ecc, omega=self.omega)
        )

    @property
    def mass_ratio(self) -> float | None:
        """``q = M_2 / M_1 = K_1 / K_2`` for a double-lined table; ``None`` otherwise."""
        if self.k.size < 2:
            return None
        return float(self.k[0] / self.k[1])

    def minimum_masses(self) -> dict[str, float]:
        """``M_i sin^3 i`` in solar masses for the first two components (double-lined only).

        ``M_{1,2} sin^3 i = 1.0361e-7 (1 - e^2)^{3/2} (K_1 + K_2)^2 K_{2,1} P`` with ``K``
        in km/s and ``P`` in days (``docs/math.md`` §10.6).

        References
        ----------
        Hilditch, R. W. 2001, An Introduction to Close Binary Stars (Cambridge University Press)
        """
        if self.k.size < 2:
            return {}
        k1, k2 = float(self.k[0]), float(self.k[1])
        factor = _MSIN3I_COEFF * (1.0 - self.ecc**2) ** 1.5 * (k1 + k2) ** 2 * self.period
        return {self.names[0]: factor * k2, self.names[1]: factor * k1}

    def projected_semiaxes(self) -> dict[str, float]:
        """``a_i sin i`` in solar radii for every component.

        ``a_i sin i = K_i P sqrt(1 - e^2) / (2 pi)`` with ``K`` in km/s and ``P`` in days,
        converted to solar radii.

        References
        ----------
        Hilditch, R. W. 2001, An Introduction to Close Binary Stars (Cambridge University Press)
        """
        return {
            name: _ASINI_COEFF * float(k) * self.period * math.sqrt(1.0 - self.ecc**2)
            for name, k in zip(self.names, self.k, strict=True)
        }

    def predict(self, t) -> np.ndarray:
        """Model velocities ``(n_comp, len(t))`` at times ``t``, including ``gamma``."""
        t = jnp.asarray(t, dtype=jnp.float64)
        rows = []
        for i in range(self.k.size):
            rows.append(
                np.asarray(
                    radial_velocity(
                        t,
                        period=self.period,
                        t_peri=self.t_peri,
                        ecc=self.ecc,
                        omega=self.omega + (i % 2) * math.pi,
                        k=float(self.k[i]),
                        gamma=float(self.gamma[i]),
                    )
                )
            )
        return np.stack(rows)

    def to_theta(self) -> dict:
        """The elements as the ``theta`` dictionary :func:`albireo.orbit_velocities` takes.

        The result can be passed to ``Disentangler(orbit=...)`` or to a low-level prior as
        a starting point for the joint fit.
        """
        return {
            "period": jnp.asarray(self.period),
            "t_conj": jnp.asarray(self.t_conj),
            "secosw": jnp.asarray(math.sqrt(self.ecc) * math.cos(self.omega)),
            "sesinw": jnp.asarray(math.sqrt(self.ecc) * math.sin(self.omega)),
            "k": jnp.asarray(self.k),
        }

    def summary(self) -> str:
        """A text report: the elements with their errors, and the derived quantities."""
        e = self.errors
        rescale = self.chi2 / max(self.n_points - self.n_parameters, 1)
        lines = [
            f"Keplerian fit to {self.n_points} velocities of {len(self.names)} component(s): "
            f"chi2 {self.chi2:.2f} for {self.n_points - self.n_parameters} dof "
            f"(errors rescaled by sqrt({rescale:.3f}))",
            f"  P      = {self.period:.6f} +- {e['period']:.6f} d",
            f"  T_conj = {self.t_conj:.5f} +- {e['t_conj']:.5f}",
            f"  e      = {self.ecc:.4f} +- {e['ecc']:.4f}"
            + ("   (held at zero)" if "ecc" in e and e["ecc"] == 0.0 else ""),
            f"  omega  = {math.degrees(self.omega):.2f} +- {math.degrees(e['omega']):.2f} deg",
        ]
        for i, name in enumerate(self.names):
            lines.append(
                f"  K_{name} = {self.k[i]:.4f} +- {e['k'][i]:.4f} km/s   "
                f"gamma_{name} = {self.gamma[i]:+.4f} +- {e['gamma'][i]:.4f} km/s   "
                f"rms {self.rms[i]:.4f} km/s"
            )
        lines.append(f"  systemic velocity: {self.gamma_mode}")
        if self.mass_ratio is not None:
            masses = self.minimum_masses()
            lines.append(
                f"  q = K_{self.names[0]}/K_{self.names[1]} = {self.mass_ratio:.4f};  "
                + ", ".join(f"M_{n} sin^3 i = {m:.4f} Msun" for n, m in masses.items())
            )
        return "\n".join(lines)


def fit_rv_orbit(
    table,
    *,
    period: float,
    t_conj: float | None = None,
    ecc: float | None = None,
    omega: float | None = None,
    k=None,
    gamma: str | None = None,
    circular: bool = False,
    components=None,
    max_iterations: int = 200,
) -> RVOrbit:
    """Fit a Keplerian to a velocity table by weighted least squares.

    The parameters are the period, the time of conjunction, ``(sqrt(e) cos w,
    sqrt(e) sin w)``, one semi-amplitude per component and the systemic velocity or
    velocities (``docs/math.md`` §10.6). The Jacobian is computed by JAX; the optimizer is
    ``scipy.optimize.least_squares`` with the bounds ``0.5 P_0 <= P <= 2 P_0``,
    ``|sqrt(e) cos w| <= 0.95``, ``|sqrt(e) sin w| <= 0.95`` and ``K_i >= 0``, where
    ``P_0`` is the starting period.

    Parameters
    ----------
    table
        A :class:`~albireo.todcor.VelocityTable`. Epochs flagged unusable (``table.good``)
        or with non-finite velocities are left out.
    period
        Starting period [d]. Required: a least-squares fit finds the nearest local optimum,
        so the period must be known to within a few percent, from the literature, from
        :func:`find_period`, or from an eclipse ephemeris.
    t_conj, ecc, omega, k
        Optional starting values; defaults are derived from the table (``k`` from the
        velocity ranges, ``t_conj`` from a scan over phase, ``ecc`` = 0.1, ``omega`` = 0).
    gamma
        ``"shared"`` fits one systemic velocity; ``"per-component"`` fits one per
        component. Default: shared when every component's velocities are absolute, per
        component otherwise (see the module docstring).
    circular
        Hold ``e = 0`` and ``omega = 0``. The two eccentricity parameters are removed from
        the fit rather than held at the singular origin of the ``sqrt(e)``
        parameterization.
    components
        Component names to fit (default all).
    max_iterations
        Cap on the number of least-squares function evaluations.

    Returns
    -------
    RVOrbit
    """
    from scipy.optimize import least_squares

    names, v, s, absolute = _table_arrays(table, components)
    n_comp = len(names)
    if gamma is None:
        gamma = "shared" if all(absolute) else "per-component"
    if gamma not in ("shared", "per-component"):
        raise ValueError("gamma must be 'shared' or 'per-component'")
    if gamma == "shared" and not all(absolute):
        import warnings

        warnings.warn(
            "a shared systemic velocity is being fitted to components whose zero points are "
            "not all absolute; each differential component carries its own unidentified "
            "constant, so a shared gamma biases the semi-amplitudes. Use "
            "gamma='per-component' unless you know the zero points agree.",
            stacklevel=2,
        )
    if not period > 0.0:
        raise ValueError("period must be positive")

    good = table.good & np.all(np.isfinite(v), axis=0) & np.all(np.isfinite(s), axis=0)
    good &= np.all(s > 0.0, axis=0)
    t = np.asarray(table.bjd, dtype=np.float64)[good]
    y = v[:, good]
    w = 1.0 / s[:, good] ** 2
    n_gamma = 1 if gamma == "shared" else n_comp
    n_par = 2 + (0 if circular else 2) + n_comp + n_gamma
    if t.size * n_comp <= n_par:
        raise ValueError(
            f"{t.size} usable epochs x {n_comp} components is not enough to fit {n_par} parameters"
        )

    # Starting values.
    k0 = np.asarray(k, dtype=np.float64) if k is not None else 0.5 * np.ptp(y, axis=1)
    k0 = np.maximum(k0, 1e-3)
    gamma0 = np.array([np.average(y[i], weights=w[i]) for i in range(n_comp)])
    if gamma == "shared":
        gamma0 = np.array([np.average(y, weights=w)])
    ecc0 = 0.0 if circular else (0.1 if ecc is None else float(ecc))
    omega0 = 0.0 if omega is None else float(omega)

    def model(params, t_eval):
        p, tc = params[0], params[1]
        if circular:
            e_val, om, rest = 0.0, 0.0, params[2:]
        else:
            h, g = params[2], params[3]
            e_val, om, rest = h * h + g * g, jnp.arctan2(g, h), params[4:]
        ks = rest[:n_comp]
        gs = rest[n_comp:]
        t_peri = t_peri_from_t_conj(tc, period=p, ecc=e_val, omega=om)
        rows = []
        for i in range(n_comp):
            g_i = gs[0] if n_gamma == 1 else gs[i]
            rows.append(
                radial_velocity(
                    t_eval,
                    period=p,
                    t_peri=t_peri,
                    ecc=e_val,
                    omega=om + (i % 2) * jnp.pi,
                    k=ks[i],
                    gamma=g_i,
                )
            )
        return jnp.stack(rows)

    sqrt_w = jnp.asarray(np.sqrt(w))
    y_j = jnp.asarray(y)
    t_j = jnp.asarray(t)

    def residuals(params):
        return ((y_j - model(params, t_j)) * sqrt_w).reshape(-1)

    residuals_jit = jax.jit(residuals)
    jacobian_jit = jax.jit(jax.jacfwd(residuals))

    def pack(tc):
        head = [period, tc]
        if not circular:
            head += [math.sqrt(ecc0) * math.cos(omega0), math.sqrt(ecc0) * math.sin(omega0)]
        return np.array([*head, *k0, *gamma0])

    if t_conj is None:
        # A scan over conjunction phase: the least-squares problem is multimodal in it.
        trial_phases = np.linspace(0.0, 1.0, 24, endpoint=False)
        chi2_trials = []
        for phase in trial_phases:
            params = pack(t.min() + phase * period)
            r = np.asarray(residuals_jit(jnp.asarray(params)))
            chi2_trials.append(float(r @ r))
        t_conj = t.min() + trial_phases[int(np.argmin(chi2_trials))] * period
    x0 = pack(float(t_conj))

    lower = np.full(x0.size, -np.inf)
    upper = np.full(x0.size, np.inf)
    lower[0] = 0.5 * period
    upper[0] = 2.0 * period
    if not circular:
        lower[2:4] = -0.95
        upper[2:4] = 0.95
    k_slice = slice(4 if not circular else 2, (4 if not circular else 2) + n_comp)
    lower[k_slice] = 0.0

    result = least_squares(
        lambda x: np.asarray(residuals_jit(jnp.asarray(x))),
        x0,
        jac=lambda x: np.asarray(jacobian_jit(jnp.asarray(x))),
        bounds=(lower, upper),
        max_nfev=max_iterations,
        x_scale="jac",
    )
    x = result.x
    jac = np.asarray(jacobian_jit(jnp.asarray(x)))
    n_points = int(t.size * n_comp)
    chi2 = float(result.fun @ result.fun)
    dof = max(n_points - n_par, 1)
    scale = chi2 / dof
    try:
        cov = np.linalg.inv(jac.T @ jac) * scale
    except np.linalg.LinAlgError:
        cov = np.full((x.size, x.size), np.nan)

    # Unpack, with errors propagated to (e, omega) by the delta method.
    p, tc = float(x[0]), float(x[1])
    if circular:
        ecc_fit, omega_fit = 0.0, 0.0
        offset = 2
        ecc_err, omega_err = 0.0, 0.0
    else:
        h, g = float(x[2]), float(x[3])
        ecc_fit = h * h + g * g
        omega_fit = math.atan2(g, h)
        offset = 4
        jac_e = np.array([2.0 * h, 2.0 * g])
        jac_w = np.array([-g, h]) / max(h * h + g * g, 1e-30)
        sub = cov[2:4, 2:4]
        ecc_err = math.sqrt(max(float(jac_e @ sub @ jac_e), 0.0))
        omega_err = math.sqrt(max(float(jac_w @ sub @ jac_w), 0.0))
    k_fit = np.asarray(x[offset : offset + n_comp], dtype=np.float64)
    gam = np.asarray(x[offset + n_comp :], dtype=np.float64)
    diag = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    k_err = diag[offset : offset + n_comp]
    g_err = diag[offset + n_comp :]
    if n_gamma == 1:
        gam = np.repeat(gam, n_comp)
        g_err = np.repeat(g_err, n_comp)
    errors = {
        "period": float(diag[0]),
        "t_conj": float(diag[1]),
        "ecc": ecc_err,
        "omega": omega_err,
        "k": k_err,
        "gamma": g_err,
    }
    model_all = np.asarray(model(jnp.asarray(x), jnp.asarray(table.bjd, dtype=jnp.float64)))
    resid = np.where(good[None, :], v - model_all, np.nan)
    rms = np.sqrt(np.nanmean(resid**2, axis=1))
    par_names = ["period", "t_conj"] + ([] if circular else ["secosw", "sesinw"])
    par_names += [f"k_{n}" for n in names]
    par_names += ["gamma"] if n_gamma == 1 else [f"gamma_{n}" for n in names]
    return RVOrbit(
        names=tuple(names),
        period=p,
        t_conj=tc,
        ecc=float(ecc_fit),
        omega=float(omega_fit),
        k=k_fit,
        gamma=gam,
        gamma_mode="shared" if n_gamma == 1 else "one per component",
        errors=errors,
        chi2=chi2,
        n_points=n_points,
        n_parameters=int(n_par),
        residuals=resid,
        rms=rms,
        used=good,
        covariance=cov,
        parameter_names=tuple(par_names),
    )


def relativistic_add(v_kms, u_kms):
    """Relativistic addition of two velocities, the composition law of log-wavelength shifts.

    See ``docs/math.md`` §7.6.
    """
    b1 = np.asarray(v_kms, dtype=np.float64) / C_KMS
    b2 = np.asarray(u_kms, dtype=np.float64) / C_KMS
    return C_KMS * (b1 + b2) / (1.0 + b1 * b2)

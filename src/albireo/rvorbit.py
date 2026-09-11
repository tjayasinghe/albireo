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
IAU 2015 nominal constants. :func:`find_period` is the floating-mean, weighted generalized
Lomb-Scargle periodogram of Zechmeister and Kürster (2009) (Lomb 1976; Scargle 1982;
VanderPlas 2018).

References
----------
Hilditch, R. W. 2001, An Introduction to Close Binary Stars (Cambridge University Press)
Lomb, N. R. 1976, Ap&SS, 39, 447
Scargle, J. D. 1982, ApJ, 263, 835
VanderPlas, J. T. 2018, ApJS, 236, 16
Zechmeister, M. & Kürster, M. 2009, A&A, 496, 577
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from albireo.grids import C_KMS
from albireo.kepler import radial_velocity, t_peri_from_t_conj

__all__ = ["RVOrbit", "find_period", "fit_rv_orbit", "reassign_by_orbit"]

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


def _gls_power(t, y, w, freqs) -> np.ndarray:
    """Floating-mean weighted periodogram of Zechmeister and Kürster (2009), normalized.

    With ``W = w / sum(w)``, ``Y = sum(W y)``, ``YY = sum(W (y - Y)^2)`` and, at each
    angular frequency, ``C = sum(W cos)``, ``S = sum(W sin)``, ``YC = sum(W y cos) - Y C``,
    ``YS = sum(W y sin) - Y S``, ``CC = sum(W cos^2) - C^2``, ``SS = sum(W sin^2) - S^2``
    and ``CS = sum(W cos sin) - C S``, the power is

    ``p = (SS YC^2 + CC YS^2 - 2 CS YC YS) / (YY (CC SS - CS^2))``,

    the fraction of the weighted variance the sinusoid plus a free constant removes. The
    weights enter through ``W`` alone; the data are never rescaled. A constant series, or
    a frequency at which the design is singular, gives zero.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weight = np.asarray(w, dtype=np.float64)
    weight = weight / float(weight.sum())
    y_bar = float(weight @ y)
    yy = float(weight @ (y - y_bar) ** 2)
    weighted_y = weight * y
    freqs = np.asarray(freqs, dtype=np.float64)
    power = np.empty(freqs.size, dtype=np.float64)
    chunk = max(1, int(2_000_000 // max(t.size, 1)))
    for start in range(0, freqs.size, chunk):
        block = freqs[start : start + chunk]
        angle = (2.0 * np.pi * block)[:, None] * t[None, :]
        cos, sin = np.cos(angle), np.sin(angle)
        c, s = cos @ weight, sin @ weight
        yc = cos @ weighted_y - y_bar * c
        ys = sin @ weighted_y - y_bar * s
        cc = (cos * cos) @ weight - c * c
        ss = (sin * sin) @ weight - s * s
        cs = (cos * sin) @ weight - c * s
        with np.errstate(divide="ignore", invalid="ignore"):
            block_power = (ss * yc**2 + cc * ys**2 - 2.0 * cs * yc * ys) / (
                yy * (cc * ss - cs * cs)
            )
        power[start : start + chunk] = np.nan_to_num(block_power, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(power, 0.0, 1.0)


def _harmonic_power(t, y, w, freqs, n_harmonics: int) -> np.ndarray:
    """``1 - chi2 / chi2_null`` for a floating mean plus ``n_harmonics`` harmonics.

    The model ``c + sum_h (a_h cos(h w t) + b_h sin(h w t))`` is fitted by weighted least
    squares at every frequency, as a batched solve of the ``(1 + 2 n_harmonics)`` square
    normal equations over the grid, in chunks. ``chi2_null`` is the weighted variance
    about the weighted mean, so the power is on the same scale as :func:`_gls_power` and
    equals it for one harmonic.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weight = np.asarray(w, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    n_par = 1 + 2 * int(n_harmonics)
    total = float(weight.sum())
    y_bar = float(weight @ y) / total
    chi2_null = float(weight @ (y - y_bar) ** 2)
    power = np.zeros(freqs.size, dtype=np.float64)
    if not chi2_null > 0.0:
        return power
    diagonal = np.arange(n_par)
    chunk = max(1, int(1_000_000 // (max(t.size, 1) * n_par)))
    for start in range(0, freqs.size, chunk):
        block = freqs[start : start + chunk]
        design = np.empty((block.size, t.size, n_par), dtype=np.float64)
        design[:, :, 0] = 1.0
        for h in range(1, int(n_harmonics) + 1):
            angle = (2.0 * np.pi * h * block)[:, None] * t[None, :]
            design[:, :, 2 * h - 1] = np.cos(angle)
            design[:, :, 2 * h] = np.sin(angle)
        weighted = design * weight[None, :, None]
        normal = np.einsum("fnp,fnq->fpq", weighted, design)
        rhs = np.einsum("fnp,n->fp", weighted, y)
        # The design is singular where the harmonics degenerate (the long-period corner of
        # the grid, and any frequency the epochs alias exactly); ridge it rather than fail.
        normal[:, diagonal, diagonal] *= 1.0 + 1e-10
        normal[:, diagonal, diagonal] += 1e-12 * total
        try:
            coefficients = np.linalg.solve(normal, rhs[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            coefficients = np.einsum("fpq,fq->fp", np.linalg.pinv(normal), rhs)
        model = np.einsum("fnp,fp->fn", design, coefficients)
        chi2 = np.einsum("n,fn->f", weight, (y[None, :] - model) ** 2)
        power[start : start + chunk] = 1.0 - chi2 / chi2_null
    return np.clip(np.nan_to_num(power, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)


def _distinct_peaks(freqs, power, n_peaks: int, tol: float = 0.02) -> list[float]:
    """The ``n_peaks`` highest grid periods that differ from each other by more than ``tol``.

    Only the local maxima of the power array are walked. An accepted peak is the highest
    point of its own exclusion window and so a local maximum, wherever the grid resolves
    that window: the step in period is ``dP/P = P / (N_os T)``, below ``tol`` up to about
    ``P = 0.2 T`` at ten samples per ``1/T``, and the list is then the one an exhaustive
    loop over every grid point returns, at a few thousand comparisons instead of a few
    hundred thousand. At longer periods the two lists differ, and in the direction that
    helps: two adjacent grid points on the flank of one broad peak are already more than
    ``tol`` apart in period, and the exhaustive loop reports both as separate candidates.
    """
    power = np.asarray(power)
    interior = np.flatnonzero((power[1:-1] > power[:-2]) & (power[1:-1] >= power[2:])) + 1
    candidates = np.concatenate(([0], interior, [power.size - 1]))
    peaks: list[float] = []
    for k in candidates[np.argsort(power[candidates])[::-1]]:
        period = float(1.0 / freqs[k])
        if all(abs(period / other - 1.0) > tol for other in peaks):
            peaks.append(period)
            if len(peaks) >= n_peaks:
                break
    return peaks


def find_period(
    table,
    *,
    period_range: tuple[float, float] | None = None,
    n_frequencies: int | None = None,
    components=None,
    swap_invariant: bool = False,
    n_peaks: int = 20,
    n_harmonics: int = 1,
) -> dict:
    """Generalized Lomb-Scargle search for the orbital period of a velocity table.

    For two or more components the periodogram is computed on the difference of the first
    two components' velocities, which is free of both systemic velocities and both
    template zero points and has amplitude ``K_1 + K_2``. A single component is searched
    as it is. The statistic is the floating-mean, weighted generalized periodogram of
    Zechmeister and Kürster (2009): a constant is fitted alongside the sinusoid at every
    frequency, and the returned power is the fraction of the weighted variance the pair
    removes. The weights ``1 / sigma^2`` enter the fit, not the data.

    The floating mean is what makes the search usable on clumped sampling. A classical
    periodogram fits ``a cos(wt) + b sin(wt)`` with the offset held at zero, which is
    harmless when the epochs are spread evenly enough that the sampling window has no mean
    at the frequencies of interest. A survey cadence is not like that: on the D62
    benchmark's Gaia-like tables, 10 to 25 epochs falling into about eleven visibility
    windows separated by hundreds of days, the constant the classical model cannot fit is
    absorbed into the sinusoid, and the spurious power buries the true period. Over those
    tables the true period is the highest peak of the classical periodogram in 5 of 13
    systems and of this one in 9 of 13.

    The frequency grid is uniform in frequency, so the step in period is
    ``dP/P = P / (N_os T)`` with ``T`` the span of the epochs and ``N_os = 10`` samples per
    ``1/T``. That is 0.1% at a hundredth of the baseline and 4.7% at 0.47 of it, so a peak
    reported near a third of the baseline is good to a few per cent only, and it is the
    Keplerian fit started from it, not the grid, that pins the period down. Refining the
    grid was measured and changes no outcome for the single-sinusoid search.

    With ``n_harmonics = 2`` the model is ``c + a1 cos(wt) + b1 sin(wt) + a2 cos(2wt) +
    b2 sin(2wt)``, fitted by weighted least squares at every frequency, and the power is
    ``1 - chi2 / chi2_null`` on the same scale. An eccentric orbit's velocity curve is not
    a sinusoid, and the second harmonic ranks its period higher: over the same benchmark it
    moved two systems at e = 0.41 and 0.47 from rank 10 to ranks 2 and 1, and one at
    e = 0.67 from beyond the six hundredth peak to rank 1. It is worse on circular orbits,
    where the extra freedom is spent on noise, so it belongs beside the one-harmonic search
    as a second source of candidates rather than in place of it.

    With ``swap_invariant`` the series is the magnitude of the difference instead. Two
    alike components at similar light fractions can be exchanged between epochs by the
    correlation that measured them (:func:`reassign_by_orbit`), which flips the sign of
    the difference at random epochs and destroys its periodogram; the magnitude is the
    same under the exchange. Its dominant peak sits at half the period for a circular
    orbit, so the caller should try both each peak and its double, and then re-assign the
    epochs by the orbit fitted at each candidate.

    A periodogram peak is a starting point, not a period. The Keplerian fitted at a
    candidate uses the shape of the curve and every velocity at once, and on the same
    benchmark it separated the truth from the best alias by hundreds in chi-square wherever
    the truth was reachable at all; the search should therefore propose many candidates and
    the fit decide among them. There is a floor to this. A table of ten or eleven epochs
    whose true Keplerian already leaves a reduced chi-square above about five cannot be
    searched by any statistic on that table: an alias then fits better than the truth, and
    the honest output is a failure rather than a period.

    Parameters
    ----------
    table
        A :class:`~albireo.todcor.VelocityTable`.
    period_range
        ``(shortest, longest)`` period in days. Default: twice the shortest epoch gap to
        twice the baseline.
    n_frequencies
        Size of the frequency grid, uniform in frequency. Default: enough to resolve the
        baseline ten times over, ``max(20000, 10 T (f_max - f_min))`` with ``T`` the
        span of the epochs, so that a multi-year survey cadence (Gaia's, for instance)
        is not searched on a grid coarser than its own peak width.
    components
        Which components to use (names). Default: all, in table order.
    swap_invariant
        Search the magnitude of the relative velocity, which the exchange of two alike
        components leaves unchanged. Two or more components only.
    n_peaks
        How many distinct peaks to report: the best and ``n_peaks - 1`` aliases.
    n_harmonics
        Harmonics in the model. One is the generalized periodogram in closed form; two is
        the eccentric-orbit search described above. ``1 + 2 n_harmonics`` parameters need
        that many usable epochs and then some.

    Returns
    -------
    dict
        ``period`` (best, days), ``periods`` and ``power`` (the periodogram), and
        ``aliases``, the ``n_peaks - 1`` next-best peaks whose periods differ from each
        other and from the best by more than 2%. The periodogram of a sparsely sampled
        table is rarely unambiguous, and the aliases should be inspected.

    References
    ----------
    Lomb, N. R. 1976, Ap&SS, 39, 447
    Scargle, J. D. 1982, ApJ, 263, 835
    VanderPlas, J. T. 2018, ApJS, 236, 16
    Zechmeister, M. & Kürster, M. 2009, A&A, 496, 577
    """
    _, v, s, _ = _table_arrays(table, components)
    good = np.all(np.isfinite(v), axis=0) & np.all(np.isfinite(s), axis=0) & table.good
    if int(good.sum()) < 4:
        raise ValueError(f"only {int(good.sum())} usable epochs; a period search needs at least 4")
    n_peaks = int(n_peaks)
    n_harmonics = int(n_harmonics)
    if n_peaks < 1:
        raise ValueError("n_peaks must be at least 1")
    if n_harmonics < 1:
        raise ValueError("n_harmonics must be at least 1")
    n_par = 1 + 2 * n_harmonics
    if int(good.sum()) <= n_par:
        raise ValueError(
            f"{int(good.sum())} usable epochs cannot support the {n_par} parameters of a "
            f"search with {n_harmonics} harmonic(s)"
        )
    t = np.asarray(table.bjd, dtype=np.float64)[good]
    if v.shape[0] >= 2:
        y = v[0, good] - v[1, good]
        if swap_invariant:
            y = np.abs(y)
        w = 1.0 / (s[0, good] ** 2 + s[1, good] ** 2)
    else:
        if swap_invariant:
            raise ValueError("swap_invariant needs at least two components")
        y = v[0, good]
        w = 1.0 / s[0, good] ** 2
    if period_range is None:
        gaps = np.diff(np.sort(t))
        shortest = 2.0 * float(np.min(gaps[gaps > 0]))
        longest = 2.0 * float(t.max() - t.min())
        period_range = (shortest, longest)
    lo, hi = period_range
    if not (0.0 < lo < hi):
        raise ValueError(f"period_range must satisfy 0 < shortest < longest; got {period_range}")
    if n_frequencies is None:
        baseline = float(t.max() - t.min())
        n_frequencies = max(20_000, int(np.ceil(10.0 * baseline * (1.0 / lo - 1.0 / hi))))
    freqs = np.linspace(1.0 / hi, 1.0 / lo, int(n_frequencies))
    if n_harmonics == 1:
        power = _gls_power(t, y, w, freqs)
    else:
        power = _harmonic_power(t, y, w, freqs, n_harmonics)
    peaks = _distinct_peaks(freqs, power, n_peaks)
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


def reassign_by_orbit(table, predicted, *, threshold: float = 3.0):
    """Swap the two components at epochs where the swap fits the orbit far better.

    Two similar spectra at similar light fractions give a correlation surface that is
    nearly symmetric under the exchange of the two shifts, and a per-epoch measurement
    then lands in one of the two equivalent minima at random. Nothing in a single epoch
    can decide which star is which; the orbit can, and a disentangling supplies one. The
    decision is made on the relative velocity ``v_1 - v_2``, which is free of a shared
    zero point, after removing the constant offset between the table and the prediction
    (each component of a differential table carries its own zero point).

    Parameters
    ----------
    table
        A two-component :class:`~albireo.todcor.VelocityTable`.
    predicted
        Predicted velocities, ``(2, n_epochs)`` km/s, from the disentangling's orbit
        (:meth:`albireo.Fit.velocities`). Their zero point need not match the table's.
    threshold
        A swap is made where it reduces the residual of the relative velocity by more
        than ``threshold`` times its error.

    Returns
    -------
    (VelocityTable, numpy.ndarray)
        The table with the swapped epochs exchanged (velocities, errors, covariances,
        light fractions and detection statistics alike), and the boolean mask of the
        epochs that were swapped. A table with other than two components is returned
        unchanged with an all-false mask.
    """
    from dataclasses import replace

    pred = np.asarray(predicted, dtype=np.float64)
    if table.n_components != 2 or pred.shape != (2, table.n_epochs):
        return table, np.zeros(table.n_epochs, dtype=bool)
    v = np.asarray(table.velocity, dtype=np.float64)
    s = np.asarray(table.sigma, dtype=np.float64)
    good = table.good & np.all(np.isfinite(v), axis=0) & np.all(np.isfinite(s), axis=0)
    if int(good.sum()) < 3:
        return table, np.zeros(table.n_epochs, dtype=bool)
    r_pred = pred[0] - pred[1]
    sigma_r = np.sqrt(s[0] ** 2 + s[1] ** 2)
    swap = np.zeros(table.n_epochs, dtype=bool)
    for _ in range(3):
        r_obs = np.where(swap, v[1] - v[0], v[0] - v[1])
        offset = float(np.median((r_obs - r_pred)[good]))
        keep_res = np.abs((v[0] - v[1]) - r_pred - offset) / sigma_r
        swap_res = np.abs((v[1] - v[0]) - r_pred - offset) / sigma_r
        new = good & (keep_res - swap_res > threshold)
        if np.array_equal(new, swap):
            break
        swap = new
    if not swap.any():
        return table, swap

    def exchange(array):
        array = np.asarray(array)
        out = np.array(array)
        out[0] = np.where(swap, array[1], array[0])
        out[1] = np.where(swap, array[0], array[1])
        return out

    covariance = np.array(table.covariance)
    covariance[swap] = covariance[swap][:, ::-1, :][:, :, ::-1]
    return replace(
        table,
        velocity=exchange(table.velocity),
        sigma=exchange(table.sigma),
        sigma_ivar=exchange(table.sigma_ivar),
        covariance=covariance,
        light=exchange(table.light),
        delta_chi2=exchange(table.delta_chi2),
        at_edge=exchange(table.at_edge),
        settings={**dict(table.settings), "reassigned_by_orbit": int(swap.sum())},
    ), swap


def relativistic_add(v_kms, u_kms):
    """Relativistic addition of two velocities, the composition law of log-wavelength shifts.

    See ``docs/math.md`` §7.6.
    """
    b1 = np.asarray(v_kms, dtype=np.float64) / C_KMS
    b2 = np.asarray(u_kms, dtype=np.float64) / C_KMS
    return C_KMS * (b1 + b2) / (1.0 + b1 * b2)

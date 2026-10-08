"""Keplerian orbits fitted to a velocity table.

**Experimental.** The names and return shapes defined here may change; the joint
path of ``docs/math.md`` §7 fits the same Keplerian from the spectra directly.

The joint path does not use this module. :mod:`albireo.todcor` produces one velocity per
component per epoch, and this module fits a Keplerian to such a table with the same
Kepler solver and angle conventions as the joint model (:mod:`albireo.kepler`,
:func:`albireo.orbit_velocities`). The two routes can then be compared element by
element (``docs/math.md`` §10.6).

The fit is weighted nonlinear least squares over the sampled elements: period, time of
conjunction, ``(sqrt(e) cos w, sqrt(e) sin w)``, one semi-amplitude per component and a
systemic velocity, with the Jacobian computed by JAX. The systemic velocity is a single
parameter when every component's velocities are absolute and one parameter per component
otherwise. A table built from disentangled templates has one unidentified zero point per
component (``docs/math.md`` §7.6), and a shared gamma would then absorb two different
constants and bias both semi-amplitudes. :class:`RVOrbit` records which was used in
``gamma_mode``.

Uncertainties are the curvature errors at the optimum scaled by the reduced chi-square. The
per-epoch errors from :func:`albireo.todcor` are curvature errors of a template fit and
exclude template mismatch, line-profile variability and any third body, so the scatter of a
table about a Keplerian is generally larger than they imply.

An orbit also decides what one epoch of a correlation leaves open: the order of the
velocities of two alike components (:func:`reassign_by_orbit`), and the minimum at an epoch
that the table flags for a second one (:func:`assign_by_orbit`). :func:`assign_components`
makes both decisions while it fits the orbit at a known period (``docs/math.md`` §10.6).

An eclipsing binary has a photometric ephemeris, which is far more precise than any
element a velocity table of tens of epochs determines. :func:`fit_rv_ephemeris` holds the
period and the time of conjunction at it, and the model is then linear in the
semi-amplitudes and the systemic velocity for a circular orbit. :func:`assign_by_ephemeris`
makes the decisions above with that fit, starting from the order of the pair that the
ephemeris gives: it states which star recedes after the primary eclipse.

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

import contextlib
import functools
import math
import warnings
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from albireo.grids import _EPOCH_BLOCK, C_KMS, _in_epoch_blocks
from albireo.kepler import (
    _NEWTON_ITERATIONS,
    _t_peri_from_t_conj_numpy,
    radial_velocity,
    t_peri_from_t_conj,
    true_anomaly_numpy,
)

__all__ = [
    "Assignment",
    "RVOrbit",
    "assign_by_ephemeris",
    "assign_by_orbit",
    "assign_components",
    "find_period",
    "fit_rv_ephemeris",
    "fit_rv_orbit",
    "reassign_by_orbit",
]

# Minimum masses and projected semi-axes in solar units from km/s and days
# (Hilditch 2001, eqs. 3.17 and 3.18, with the IAU 2015 nominal constants).
_MSIN3I_COEFF = 1.0361e-7  # M sin^3 i [M_sun] = coeff (1-e^2)^{3/2} (K_1 + K_2)^2 K_j P
_ASINI_COEFF = 86400.0 / (2.0 * math.pi) / 695_700.0  # a sin i [R_sun] = coeff K P sqrt(1-e^2)

# The default frequency grid of `find_period`: this many samples per 1/T at least, and at least
# this many frequencies (D65; see `_frequency_grid`).
_GRID_OVERSAMPLING = 10
_GRID_MIN_FREQUENCIES = 20_000

# The largest ratio of the two light fractions at which `assign_components` exchanges the
# components of an epoch, and the grid of orbits from which it takes its starting
# assignments (`_relative_curves`): the phases, the eccentricities above zero, the arguments
# of periastron, the number of best curves examined and the number of starts kept.
_EXCHANGE_LIGHT_RATIO = 3.0
_ASSIGNMENT_PHASES = 60
_ASSIGNMENT_ECCENTRICITIES = (0.2, 0.4, 0.6, 0.8)
_ASSIGNMENT_OMEGAS = 8
_ASSIGNMENT_CURVES = 200
_ASSIGNMENT_STARTS = 3
_ASSIGNMENT_WINDOW_STEPS = 8


def _valid_velocities(table) -> np.ndarray:
    """``(n_comp, n_epochs)``: whether each velocity can enter a period search or a fit.

    A velocity is valid where its own component was measured: a finite velocity and a
    finite error off that component's search edge, at an epoch that is not blended. The
    other components do not matter: a primary measured at an epoch whose secondary was at
    the search edge, or was removed by the caller, is still a measurement of the primary.
    Where every component is valid this is ``table.good`` repeated per component, with the
    errors' finiteness added.
    """
    v = np.asarray(table.velocity, dtype=np.float64)
    s = np.asarray(table.sigma, dtype=np.float64)
    at_edge = np.broadcast_to(np.asarray(table.at_edge, dtype=bool), v.shape)
    blended = np.asarray(table.blended, dtype=bool)
    return np.isfinite(v) & np.isfinite(s) & ~at_edge & ~blended[None, :]


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
    return list(components), v, s, absolute, _valid_velocities(table)[idx]


def _frequency_grid(t, period_range, n_frequencies) -> np.ndarray:
    """The frequencies ``find_period`` evaluates, in ascending order.

    With ``n_frequencies`` given the grid is that many points spaced evenly from
    ``1 / longest`` to ``1 / shortest``. By default it is anchored at the low end. The
    step is ``1 / (N T)`` with ``T`` the span of the epochs, the points are
    ``1 / longest + k / (N T)`` below the highest frequency, and the highest frequency is
    appended as the last point. ``N`` is ``_GRID_OVERSAMPLING`` samples per ``1/T``, raised
    where needed to the smallest integer that gives at least ``_GRID_MIN_FREQUENCIES``
    points. It is computed with the frequency span rounded down to two significant
    figures, so that the count still reaches the minimum and ``N`` is unaffected by small
    movements of the high end.

    The anchoring makes a run reproducible (D65). The default shortest period is twice the
    smallest gap between two epochs, so on a grid spaced evenly between the two ends every
    upper frequency moves when one epoch time moves. A shift of 1e-6 d in the closest pair
    of a Gaia-like table (gaps of 0.074 d, about 1500 d of baseline) moves the top of the
    grid by more than one step, and near-degenerate short-period peaks change order. Epoch
    times written to six decimals shift by that much. On an evenly spaced grid the
    recorded peaks of five benchmark tables were reproduced only after moving the shortest
    period by 0.5 to 4e-6 d.

    When an epoch other than the first and the last moves, ``T`` and the low end do not.
    Unless ``N`` changes, every point below both the old and the new highest frequency is
    then unchanged, bit for bit, the last point moves with the high end, and points are
    added or removed only between the two. ``N`` is 10 on any table whose ``10 T`` times
    the span reaches the minimum (every multi-year survey table), and there it does not
    depend on the high end. Below that it changes only when the shift moves the span
    across a two-significant-figure boundary or moves ``20000 / (T span)`` across an
    integer. A 1e-6 d shift moves a span of about 6.8 per day by about 1e-4, so that
    boundary case is rare but possible, and the whole grid is then respaced. Moving the
    first or the last epoch changes ``T`` and respaces every point by a relative
    ``dT / T``.

    Against the evenly spaced grid, over the 33 blind tables of the benchmark's third run,
    the anchored grid keeps at 20 the number of systems whose true period ranks first. It
    reorders near-degenerate candidates: one Gaia system from rank 39 to 11, one from 23
    to 24, one from 3 to 4, and field systems from 40 to 41 and from 2 to 6.
    """
    if period_range is None:
        gaps = np.diff(np.sort(t))
        shortest = 2.0 * float(np.min(gaps[gaps > 0]))
        longest = 2.0 * float(t.max() - t.min())
        period_range = (shortest, longest)
    lo, hi = period_range
    if not (0.0 < lo < hi):
        raise ValueError(f"period_range must satisfy 0 < shortest < longest; got {period_range}")
    f_lo, f_hi = 1.0 / hi, 1.0 / lo
    if n_frequencies is not None:
        return np.linspace(f_lo, f_hi, int(n_frequencies))
    baseline = float(t.max() - t.min())
    span = f_hi - f_lo
    if not baseline > 0.0:
        return np.linspace(f_lo, f_hi, _GRID_MIN_FREQUENCIES)
    scale = 10.0 ** (math.floor(math.log10(span)) - 1)
    span_down = math.floor(span / scale) * scale  # two significant figures, rounded down
    oversampling = max(
        _GRID_OVERSAMPLING, math.ceil(_GRID_MIN_FREQUENCIES / (baseline * span_down))
    )
    step = 1.0 / (oversampling * baseline)
    below = f_lo + step * np.arange(math.ceil(span / step))
    # A point within a millionth of a step of the end would duplicate it.
    below = below[below < f_hi - 1e-6 * step]
    return np.append(below, f_hi)


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
        # The design is singular where the harmonics degenerate (the long-period end of the
        # grid, and any frequency the epochs alias exactly). A ridge prevents a failed solve.
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

    Only the local maxima of the power array are examined. An accepted peak is the highest
    point of its own exclusion window and therefore a local maximum, wherever the grid
    resolves that window. The step in period is ``dP/P = P / (N_os T)``, below ``tol`` up
    to about ``P = 0.2 T`` at ten samples per ``1/T``. The list is then the one an
    exhaustive loop over every grid point returns, at a few thousand comparisons instead
    of a few hundred thousand. At longer periods the two lists differ, and this one is
    preferable: two adjacent grid points on the flank of one broad peak are already more
    than ``tol`` apart in period, and the exhaustive loop reports both as separate
    candidates.
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
    Zechmeister and Kürster (2009). A constant is fitted alongside the sinusoid at every
    frequency, and the returned power is the fraction of the weighted variance the pair
    removes. The weights ``1 / sigma^2`` enter the fit, and the data are not rescaled.

    The floating mean makes the search usable on clumped sampling. A classical
    periodogram fits ``a cos(wt) + b sin(wt)`` with the offset held at zero. This is
    harmless when the epochs are spread evenly enough that the sampling window has no mean
    at the frequencies of interest. A survey cadence is not. The D62 benchmark's Gaia-like
    tables have 10 to 25 epochs in about eleven visibility windows separated by hundreds
    of days. On them the constant the classical model cannot fit is absorbed into the
    sinusoid, and the spurious power can exceed that of the true period. Over those tables
    the true period is the highest peak of the classical periodogram in 5 of 13 systems
    and of this one in 9 of 13.

    The frequency grid is uniform in frequency, so the step in period is
    ``dP/P = P / (N_os T)`` with ``T`` the span of the epochs and ``N_os = 10`` samples per
    ``1/T``. That is 0.1% at a hundredth of the baseline and 4.7% at 0.47 of it, so a peak
    reported near a third of the baseline is located to a few per cent only. The Keplerian
    fit started from it, not the grid, determines the period. A finer grid was tested and
    changed no outcome for the single-sinusoid search. The default grid starts at the
    longest period and keeps its step whatever the shortest period is, so that moving one
    epoch time by a rounding error does not move the grid relative to the peaks
    (:func:`_frequency_grid`).

    Each velocity enters where its own component was measured: a finite velocity and error,
    off that component's search edge, at an epoch that is not blended. The relative velocity
    of two components is defined where both are. One component searched alone
    (``components=[name]``) keeps every epoch at which it was measured, regardless of the
    other components.

    With ``n_harmonics = 2`` the model is ``c + a1 cos(wt) + b1 sin(wt) + a2 cos(2wt) +
    b2 sin(2wt)``, fitted by weighted least squares at every frequency, and the power is
    ``1 - chi2 / chi2_null`` on the same scale. An eccentric orbit's velocity curve is not
    a sinusoid, and the second harmonic ranks its period higher. Over the same benchmark it
    moved two systems at e = 0.41 and 0.47 from rank 10 to ranks 2 and 1, and one at
    e = 0.67 from beyond the six hundredth peak to rank 1. It is worse on circular orbits,
    where the additional parameters fit noise, so it is a second source of candidates
    beside the one-harmonic search and does not replace it.

    With ``swap_invariant`` the series is the magnitude of the difference instead. Two
    alike components at similar light fractions can be exchanged between epochs by the
    correlation that measured them (:func:`reassign_by_orbit`), which flips the sign of
    the difference at random epochs and destroys its periodogram. The magnitude is
    unchanged by the exchange. Its dominant peak is at half the period for a circular
    orbit, so the caller should try each peak and its double, and then re-assign the
    epochs by the orbit fitted at each candidate.

    A periodogram peak is only a starting point. The Keplerian fitted at a candidate uses
    the shape of the curve and every velocity at once. On the same benchmark it separated
    the true period from the best alias by hundreds in chi-square wherever the true period
    was reachable. The search should propose many candidates and the fit decide among
    them. A table of ten or eleven epochs whose true Keplerian already leaves a reduced
    chi-square above about five cannot be searched by any statistic on that table. An
    alias then fits better than the true period, and the correct output is a failure
    rather than a period.

    Parameters
    ----------
    table
        A :class:`~albireo.todcor.VelocityTable`.
    period_range
        ``(shortest, longest)`` period in days. Default: twice the shortest epoch gap to
        twice the baseline.
    n_frequencies
        Size of the frequency grid, spaced evenly between the two ends. Default: a step of
        ``1 / (N T)`` from the lowest frequency, and the highest frequency as the last
        point. ``T`` is the span of the epochs and ``N`` ten samples per ``1/T``, raised
        to the smallest integer that gives at least 20000 frequencies (with the span
        rounded down to two significant figures). A multi-year survey cadence (Gaia's, for
        instance) is then not searched on a grid coarser than its own peak width, and the
        grid points do not move with the shortest period.
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
        more than that many usable epochs.

    Returns
    -------
    dict
        ``period`` (best, days), ``periods`` and ``power`` (the periodogram), ``aliases``,
        the ``n_peaks - 1`` next-best peaks whose periods differ from each other and from
        the best by more than 2%, and ``used``, per epoch whether it entered the searched
        series. The periodogram of a sparsely sampled table is rarely unambiguous, and the
        aliases should be inspected.

    References
    ----------
    Lomb, N. R. 1976, Ap&SS, 39, 447
    Scargle, J. D. 1982, ApJ, 263, 835
    VanderPlas, J. T. 2018, ApJS, 236, 16
    Zechmeister, M. & Kürster, M. 2009, A&A, 496, 577
    """
    _, v, s, _, valid = _table_arrays(table, components)
    # The series is the first two components' difference, or the one component searched.
    good = valid[0] & valid[1] if v.shape[0] >= 2 else valid[0]
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
    freqs = _frequency_grid(t, period_range, n_frequencies)
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
        "used": good,
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
        when each had its own zero point.
    gamma_mode
        ``"shared"`` or ``"one per component"``.
    errors
        Standard errors for every fitted quantity, keyed like the attributes (``k`` and
        ``gamma`` hold arrays), after the reduced-chi-square rescaling.
    chi2, n_points, n_parameters
        The weighted chi-square at the optimum, the number of velocities that entered it
        and the number of parameters fitted (a held parameter is not counted).
    residuals
        ``(n_comp, n_epochs)`` observed minus model [km/s], NaN wherever that component's
        velocity did not enter the fit: not measured, at its search edge, blended, or
        belonging to a component whose semi-amplitude was held.
    rms
        Per-component RMS of the residuals over the velocities that entered; NaN for a held
        component.
    used
        Per epoch: whether any component's velocity at that epoch entered the fit.
    held
        Components whose semi-amplitude was held at its starting value rather than fitted,
        because they had no more usable velocities than parameters of their own (one, or two
        with a systemic velocity of their own). Their semi-amplitude is not a measurement, and
        :attr:`mass_ratio`, :meth:`minimum_masses` and :meth:`projected_semiaxes` do not
        report numbers for them.
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
    held: tuple[str, ...] = ()

    @property
    def t_peri(self) -> float:
        """Time of periastron passage."""
        return float(
            t_peri_from_t_conj(self.t_conj, period=self.period, ecc=self.ecc, omega=self.omega)
        )

    def _measured_pair(self) -> bool:
        return self.k.size >= 2 and not ({self.names[0], self.names[1]} & set(self.held))

    @property
    def mass_ratio(self) -> float | None:
        """``q = M_2 / M_1 = K_1 / K_2`` for a double-lined table; ``None`` otherwise.

        Also ``None`` when either semi-amplitude was held rather than fitted (:attr:`held`).
        """
        if not self._measured_pair():
            return None
        return float(self.k[0] / self.k[1])

    def minimum_masses(self) -> dict[str, float]:
        """``M_i sin^3 i`` in solar masses for the first two components (double-lined only).

        ``M_{1,2} sin^3 i = 1.0361e-7 (1 - e^2)^{3/2} (K_1 + K_2)^2 K_{2,1} P`` with ``K``
        in km/s and ``P`` in days (``docs/math.md`` §10.6). Empty when either
        semi-amplitude was held rather than fitted (:attr:`held`).

        References
        ----------
        Hilditch, R. W. 2001, An Introduction to Close Binary Stars (Cambridge University Press)
        """
        if not self._measured_pair():
            return {}
        k1, k2 = float(self.k[0]), float(self.k[1])
        factor = _MSIN3I_COEFF * (1.0 - self.ecc**2) ** 1.5 * (k1 + k2) ** 2 * self.period
        return {self.names[0]: factor * k2, self.names[1]: factor * k1}

    def projected_semiaxes(self) -> dict[str, float]:
        """``a_i sin i`` in solar radii for every component whose semi-amplitude was fitted.

        ``a_i sin i = K_i P sqrt(1 - e^2) / (2 pi)`` with ``K`` in km/s and ``P`` in days,
        converted to solar radii. A component in :attr:`held` is left out.

        References
        ----------
        Hilditch, R. W. 2001, An Introduction to Close Binary Stars (Cambridge University Press)
        """
        return {
            name: _ASINI_COEFF * float(k) * self.period * math.sqrt(1.0 - self.ecc**2)
            for name, k in zip(self.names, self.k, strict=True)
            if name not in self.held
        }

    def predict(self, t) -> np.ndarray:
        """Model velocities ``(n_comp, len(t))`` at times ``t``, including ``gamma``.

        Evaluated by a function compiled once per component count and array shape
        (:func:`_predictor`). Eager evaluation would recompile the Kepler solver's
        fixed-count Newton loop at every call, and the period search calls this once per
        candidate.
        """
        predictor = _predictor(int(self.k.size))
        elements = (
            float(self.period),
            float(self.t_peri),
            float(self.ecc),
            float(self.omega),
            jnp.asarray(self.k, dtype=jnp.float64),
            jnp.asarray(self.gamma, dtype=jnp.float64),
        )
        # In blocks of epochs: the predictor is compiled once per array length, and the
        # tables of a population have every length.
        return _in_epoch_blocks(lambda times: predictor(times, *elements), t)

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
                + ("   (held, not fitted: too few usable velocities)" if name in self.held else "")
            )
        lines.append(f"  systemic velocity: {self.gamma_mode}")
        if self.mass_ratio is not None:
            masses = self.minimum_masses()
            lines.append(
                f"  q = K_{self.names[0]}/K_{self.names[1]} = {self.mass_ratio:.4f};  "
                + ", ".join(f"M_{n} sin^3 i = {m:.4f} Msun" for n, m in masses.items())
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class _Objective:
    """The Keplerian model and the weighted residuals of :func:`fit_rv_orbit`, compiled once.

    ``residuals(params, t, y, sqrt_w)`` and ``jacobian`` take the data as arguments, and the
    ``_free`` pair takes ``(fitted, start, free_index, t, y, sqrt_w)`` with the held
    parameters taken from ``start``. ``model(params, t)`` is compiled too.
    """

    model: object
    residuals: object
    jacobian: object
    residuals_free: object
    jacobian_free: object


@functools.cache
def _objective(circular: bool, n_comp: int, n_gamma: int) -> _Objective:
    """The compiled objective for one static configuration of :func:`fit_rv_orbit`.

    The functions take the data as arguments and are cached on the configuration (circular
    or not, the number of components, one systemic velocity or one per component). JAX
    then compiles once per configuration and array shape for all candidate periods of one
    table. Closures over each call's data compile per call, and with them
    benchmark-harness processes fitting thousands of starting periods grew to 2.7 to 3.8
    GB each (D65). With this function and :func:`_predictor`, three tables through six
    search configurations peak at 446 MB against 5061 MB with this alone, and the full
    33-table rerun at 697 MB.
    """

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

    def residuals(params, t, y, sqrt_w):
        return ((y - model(params, t)) * sqrt_w).reshape(-1)

    def residuals_free(fitted, start, free_index, t, y, sqrt_w):
        return residuals(start.at[free_index].set(fitted), t, y, sqrt_w)

    return _Objective(
        model=jax.jit(model),
        residuals=jax.jit(residuals),
        jacobian=jax.jit(jax.jacfwd(residuals)),
        residuals_free=jax.jit(residuals_free),
        jacobian_free=jax.jit(jax.jacfwd(residuals_free)),
    )


@functools.cache
def _predictor(n_comp: int):
    """``predict(t, period, t_peri, ecc, omega, k, gamma)`` for ``n_comp`` components, compiled.

    The second component uses ``omega + pi``, as everywhere in this module.
    """

    def predict(t, period, t_peri, ecc, omega, k, gamma):
        return jnp.stack(
            [
                radial_velocity(
                    t,
                    period=period,
                    t_peri=t_peri,
                    ecc=ecc,
                    omega=omega + (i % 2) * jnp.pi,
                    k=k[i],
                    gamma=gamma[i],
                )
                for i in range(n_comp)
            ]
        )

    return jax.jit(predict)


_PAD_EPOCHS = False


@contextlib.contextmanager
def _epoch_blocks():
    """Fit orbits on epochs padded to whole blocks, for the duration of the context.

    :func:`fit_rv_orbit` compiles its residuals and their Jacobian once for every number
    of usable epochs, and the rounds of :func:`assign_components` change that number
    within one table. A process that fits the tables of a population holds hundreds of
    such programs: on simulated Gaia RVS tables it grew by 16 MB per system. Inside this
    context the epochs are padded to a multiple of 32 with zero weight and the padded rows
    are cropped before the optimizer sees them, so every table shares a few programs.

    Each row depends on its own epoch alone, but a compiled program does not evaluate the
    elements of arrays of different lengths to the same last bit, and the optimizer
    carries the difference: of 40 tables fitted both ways, 5 differed, by at most 1.4e-10
    of the fitted values. The padding is therefore not the default, and results recorded
    without it are reproduced bit for bit.
    """
    global _PAD_EPOCHS
    previous = _PAD_EPOCHS
    _PAD_EPOCHS = True
    try:
        yield
    finally:
        _PAD_EPOCHS = previous


def _semi_amplitude_start(y) -> float:
    """Half the range of one component's usable velocities ``y``; zero with none.

    Used by :func:`fit_rv_orbit` as the starting semi-amplitude: ``K`` for a Keplerian of
    any eccentricity whose phases are covered, since the curve ranges from
    ``gamma - K (1 - e cos w)`` to ``gamma + K (1 + e cos w)``.
    """
    y = np.asarray(y, dtype=np.float64)
    return 0.5 * float(np.ptp(y)) if y.size else 0.0


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
    velocities (``docs/math.md`` §10.6). The Jacobian is computed by JAX. The optimizer is
    ``scipy.optimize.least_squares`` with the bounds ``0.5 P_0 <= P <= 2 P_0``,
    ``|sqrt(e) cos w| <= 0.95``, ``|sqrt(e) sin w| <= 0.95`` and ``K_i >= 0``, where
    ``P_0`` is the starting period.

    Parameters
    ----------
    table
        A :class:`~albireo.todcor.VelocityTable`. Each velocity enters where its own
        component was measured: a finite velocity and a positive finite error, off that
        component's search edge, at an epoch that is not blended. A primary measured at an
        epoch whose secondary was at the search edge, or was set to ``nan`` by the caller,
        still enters.

        A component with no more usable velocities than parameters of its own (one, its
        semi-amplitude, or two where each component has its own systemic velocity) cannot
        constrain them, and the optimizer does not hold an unconstrained parameter in place.
        On a benchmark table whose secondary was excluded at every epoch by the detection
        threshold, the semi-amplitude reached 1.5e7 km/s. Such a component is held
        (:attr:`RVOrbit.held`). Its semi-amplitude stays at 1e-3 km/s, or at the caller's
        ``k``, and its own systemic velocity at its starting value. Neither is fitted or
        counted in ``n_parameters``, both have ``nan`` errors, and its velocities have no
        weight.
    period
        Starting period [d]. Required. A least-squares fit finds the nearest local optimum,
        so the period must be known to within a few percent, from the literature, from
        :func:`find_period`, or from an eclipse ephemeris.
    t_conj, ecc, omega, k
        Optional starting values; defaults are derived from the table (``t_conj`` from a
        scan over phase, ``ecc`` = 0.1, ``omega`` = 0, and ``k`` as described below).

        The default starting semi-amplitude of a component is half the range of its own
        usable velocities, which is ``K`` at any eccentricity once the phases are covered.
        It does not exclude the velocities of an undetected companion: 321 km/s on a
        benchmark system whose 7% secondary the templates never detected, above the
        declared upper limit of 250 km/s. Removing those velocities is the caller's
        decision. The pipeline applies its detection threshold
        (``Analysis.detection_min``). ``sqrt(2)`` times the weighted standard deviation is
        not used. Sampled evenly in time it is 0.34 to 0.51 of ``K`` at ``e = 0.9`` (0.65 to
        0.77 at 0.7), and it changed the chi-square ranking of six benchmark tables whose
        components are all usable. The only bound the fit declares on ``K`` is zero, and
        the starting value is at least 1e-3 km/s.
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

    names, v, s, absolute, valid = _table_arrays(table, components)
    n_comp = len(names)
    if gamma is None:
        gamma = "shared" if all(absolute) else "per-component"
    if gamma not in ("shared", "per-component"):
        raise ValueError("gamma must be 'shared' or 'per-component'")
    if gamma == "shared" and not all(absolute):
        warnings.warn(
            "a shared systemic velocity is being fitted to components whose zero points are "
            "not all absolute; each differential component carries its own unidentified "
            "constant, so a shared gamma biases the semi-amplitudes. Use "
            "gamma='per-component' unless you know the zero points agree.",
            stacklevel=2,
        )
    if not period > 0.0:
        raise ValueError("period must be positive")

    # Per component and epoch: the velocities that enter. A component with no more usable
    # velocities than parameters of its own is held (see the docstring) and its velocities
    # have no weight. An epoch enters where any velocity does, and a velocity that does not
    # is held at zero with zero weight.
    valid = valid & (np.where(np.isfinite(s), s, 0.0) > 0.0)
    n_gamma = 1 if gamma == "shared" else n_comp
    own_parameters = 1 if n_gamma == 1 else 2
    held_components = valid.sum(axis=1) <= own_parameters
    valid = valid & ~held_components[:, None]
    good = np.any(valid, axis=0)
    mask = valid[:, good]
    t = np.asarray(table.bjd, dtype=np.float64)[good]
    y = np.where(mask, v[:, good], 0.0)
    w = np.where(mask, 1.0 / np.where(mask, s[:, good], 1.0) ** 2, 0.0)
    head = 2 if circular else 4
    held = np.zeros(head + n_comp + n_gamma, dtype=bool)
    for i in np.flatnonzero(held_components):
        held[head + i] = True
        if n_gamma > 1:
            held[head + n_comp + i] = True
    free = ~held
    n_par = int(free.sum())
    n_points = int(mask.sum())
    if n_points <= n_par:
        if bool(mask.all()):
            raise ValueError(
                f"{t.size} usable epochs x {n_comp} components is not enough to fit "
                f"{n_par} parameters"
            )
        raise ValueError(
            f"{n_points} usable velocities of {n_comp} components over {t.size} epochs is not "
            f"enough to fit {n_par} parameters"
        )

    # Starting values.
    if k is not None:
        k0 = np.asarray(k, dtype=np.float64)
    else:
        k0 = np.array([_semi_amplitude_start(y[i, mask[i]]) for i in range(n_comp)])
    k0 = np.maximum(k0, 1e-3)
    overall = float(np.average(y, weights=w))
    gamma0 = np.array(
        [np.average(y[i], weights=w[i]) if w[i].sum() > 0.0 else overall for i in range(n_comp)]
    )
    if gamma == "shared":
        gamma0 = np.array([overall])
    ecc0 = 0.0 if circular else (0.1 if ecc is None else float(ecc))
    omega0 = 0.0 if omega is None else float(omega)

    objective = _objective(bool(circular), int(n_comp), int(n_gamma))
    # Inside `_epoch_blocks()` the compiled residuals and their Jacobian are evaluated on
    # the epochs padded to a whole number of blocks, with zero weight on the padding, and
    # cropped before SciPy sees them (see `_epoch_blocks`).
    n_fit = int(t.size)
    pad = -n_fit % _EPOCH_BLOCK if _PAD_EPOCHS and n_fit >= 2 else 0
    data = (
        jnp.asarray(np.pad(t, (0, pad), mode="edge")),
        jnp.asarray(np.pad(y, ((0, 0), (0, pad)))),
        jnp.asarray(np.pad(np.sqrt(w), ((0, 0), (0, pad)))),
    )

    def crop(values):
        values = np.asarray(values)
        if pad == 0:
            return values
        shaped = values.reshape(n_comp, n_fit + pad, *values.shape[1:])
        return shaped[:, :n_fit].reshape(n_comp * n_fit, *values.shape[1:])

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
            r = crop(objective.residuals(jnp.asarray(params), *data))
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

    if held.any():
        extra = (jnp.asarray(x0), jnp.asarray(np.flatnonzero(free)), *data)
        fun_jit, jac_jit = objective.residuals_free, objective.jacobian_free
    else:
        extra = data
        fun_jit, jac_jit = objective.residuals, objective.jacobian

    result = least_squares(
        lambda x: crop(fun_jit(jnp.asarray(x), *extra)),
        x0[free],
        jac=lambda x: crop(jac_jit(jnp.asarray(x), *extra)),
        bounds=(lower[free], upper[free]),
        max_nfev=max_iterations,
        x_scale="jac",
    )
    x = np.array(x0)
    x[free] = result.x
    jac = crop(jac_jit(jnp.asarray(result.x), *extra))
    chi2 = float(result.fun @ result.fun)
    dof = max(n_points - n_par, 1)
    scale = chi2 / dof
    # A held parameter has a nan error.
    cov = np.full((x.size, x.size), np.nan)
    try:
        cov[np.ix_(free, free)] = np.linalg.inv(jac.T @ jac) * scale
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
    fitted_parameters = jnp.asarray(x)
    model_all = _in_epoch_blocks(
        lambda times: objective.model(fitted_parameters, times),
        np.asarray(table.bjd, dtype=np.float64),
    )
    resid = np.where(valid, v - model_all, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a component with no usable velocity
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
        held=tuple(name for name, h in zip(names, held_components, strict=True) if h),
    )


def _ephemeris_curve(t, period: float, t_conj: float, ecc: float, omega: float) -> np.ndarray:
    """``cos(nu + omega) + e cos(omega)`` of the first component at the times ``t``, in NumPy.

    The velocity of the first component is its systemic velocity plus its semi-amplitude
    times this curve, and that of the second its own with the opposite sign. For a circular
    orbit the curve is ``-sin[2 pi (t - t_conj) / P]``: the first component approaches
    after its eclipse.
    """
    t_peri = _t_peri_from_t_conj_numpy(float(t_conj), period=period, ecc=ecc, omega=omega)
    nu = true_anomaly_numpy(t, period=period, t_peri=t_peri, ecc=ecc)
    return np.cos(nu + omega) + ecc * math.cos(omega)


@dataclass(frozen=True)
class _EphemerisOrbit(RVOrbit):
    """An :class:`RVOrbit` whose :meth:`predict` is evaluated in NumPy.

    :meth:`RVOrbit.predict` is compiled once for every number of epochs. The rounds of
    :func:`assign_by_ephemeris` call it on tables of any length, for thousands of systems
    in one process.
    """

    def predict(self, t) -> np.ndarray:
        curve = _ephemeris_curve(t, self.period, self.t_conj, self.ecc, self.omega)
        signs = np.where(np.arange(self.k.size) % 2 == 0, 1.0, -1.0)
        return self.gamma[:, None] + (self.k * signs)[:, None] * curve[None, :]


def _fit_rv_ephemeris(
    table,
    *,
    period: float,
    t_conj: float,
    ecc: float = 0.0,
    omega: float = 0.0,
    fit_eccentricity: bool = False,
    gamma: str | None = None,
    components=None,
) -> _EphemerisOrbit:
    from scipy.optimize import least_squares, lsq_linear

    names, v, s, absolute, valid = _table_arrays(table, components)
    n_comp = len(names)
    if gamma is None:
        gamma = "shared" if all(absolute) else "per-component"
    if gamma not in ("shared", "per-component"):
        raise ValueError("gamma must be 'shared' or 'per-component'")
    if not period > 0.0:
        raise ValueError("period must be positive")
    if not 0.0 <= ecc < 1.0:
        raise ValueError(f"ecc must lie in [0, 1); got {ecc}")

    # The velocities that enter, and the components held, as in `fit_rv_orbit`.
    valid = valid & (np.where(np.isfinite(s), s, 0.0) > 0.0)
    n_gamma = 1 if gamma == "shared" else n_comp
    own_parameters = 1 if n_gamma == 1 else 2
    held_components = valid.sum(axis=1) <= own_parameters
    valid = valid & ~held_components[:, None]
    good = np.any(valid, axis=0)
    mask = valid[:, good]
    t = np.asarray(table.bjd, dtype=np.float64)[good]
    y = np.where(mask, v[:, good], 0.0)
    sqrt_w = np.where(mask, 1.0 / np.where(mask, s[:, good], 1.0), 0.0)
    signs = np.where(np.arange(n_comp) % 2 == 0, 1.0, -1.0)

    # Linear parameters: one semi-amplitude per component, then the systemic velocity or
    # velocities. A held component keeps its starting values and is not fitted.
    held = np.zeros(n_comp + n_gamma, dtype=bool)
    held[:n_comp] = held_components
    if n_gamma > 1:
        held[n_comp:] = held_components
    free = ~held
    n_linear = int(free.sum())
    n_shape = 2 if fit_eccentricity else 0
    n_par = n_linear + n_shape
    n_points = int(mask.sum())
    if n_points <= n_par:
        if bool(mask.all()):
            raise ValueError(
                f"{t.size} usable epochs x {n_comp} components is not enough to fit "
                f"{n_par} parameters"
            )
        raise ValueError(
            f"{n_points} usable velocities of {n_comp} components over {t.size} epochs is not "
            f"enough to fit {n_par} parameters"
        )
    start = np.zeros(n_comp + n_gamma)
    start[:n_comp] = 1e-3
    lower = np.full(n_comp + n_gamma, -np.inf)
    lower[:n_comp] = 0.0
    rows = mask.reshape(-1)

    def design(e_val: float, om: float) -> np.ndarray:
        curve = _ephemeris_curve(t, period, t_conj, e_val, om)
        matrix = np.zeros((n_comp, t.size, n_comp + n_gamma))
        for i in range(n_comp):
            matrix[i, :, i] = signs[i] * curve
            matrix[i, :, n_comp + (0 if n_gamma == 1 else i)] = 1.0
        return (matrix * sqrt_w[:, :, None]).reshape(-1, n_comp + n_gamma)[rows]

    target = (y * sqrt_w).reshape(-1)[rows]

    def solve(e_val: float, om: float):
        matrix = design(e_val, om)
        rhs = target - matrix[:, held] @ start[held]
        result = lsq_linear(matrix[:, free], rhs, bounds=(lower[free], np.inf))
        x = np.array(start)
        x[free] = result.x
        return x, rhs - matrix[:, free] @ result.x, matrix

    def unpack(shape):
        e_val = min(float(shape[0] ** 2 + shape[1] ** 2), 0.95)
        return e_val, (math.atan2(shape[1], shape[0]) if e_val > 0.0 else 0.0)

    shape = np.zeros(2)
    if fit_eccentricity:
        # The chi-square has several minima in (e, omega): a scan first, then a
        # least-squares fit from each of the three best points, with the linear
        # parameters solved for at every trial.
        trials = _shape_trials()
        chi2_trials = [float(np.sum(solve(e_val, om)[1] ** 2)) for e_val, om in trials]
        best = None
        for index in np.argsort(chi2_trials)[:3]:
            e_val, om = trials[int(index)]
            x0 = math.sqrt(e_val) * np.array([math.cos(om), math.sin(om)])
            result = least_squares(
                lambda p: solve(*unpack(p))[1], x0, bounds=(-0.95, 0.95), max_nfev=200
            )
            if best is None or result.cost < best.cost:
                best = result
        shape = np.asarray(best.x, dtype=np.float64)
        ecc_fit, omega_fit = unpack(shape)
    else:
        ecc_fit, omega_fit = float(ecc), float(omega)
    x, residual, matrix = solve(ecc_fit, omega_fit)
    chi2 = float(residual @ residual)
    dof = max(n_points - n_par, 1)

    # Covariance: the curvature at the optimum scaled by the reduced chi-square. The
    # columns of the shape parameters are central differences at the fitted linear ones.
    columns = [matrix[:, free]]
    if fit_eccentricity:
        steps = []
        for axis in range(2):
            h = np.zeros(2)
            h[axis] = 1e-4
            up = design(*unpack(shape + h)) @ x
            down = design(*unpack(shape - h)) @ x
            steps.append((up - down) / 2e-4)
        columns.insert(0, np.stack(steps, axis=1))
    jac = np.hstack(columns)
    size = n_shape + n_comp + n_gamma
    index = np.concatenate([np.ones(n_shape, dtype=bool), free])
    cov = np.full((size, size), np.nan)
    try:
        cov[np.ix_(index, index)] = np.linalg.inv(jac.T @ jac) * (chi2 / dof)
    except np.linalg.LinAlgError:
        cov = np.full((size, size), np.nan)
    diag = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    ecc_err, omega_err = 0.0, 0.0
    if fit_eccentricity:
        h, g = float(shape[0]), float(shape[1])
        jac_e = np.array([2.0 * h, 2.0 * g])
        jac_w = np.array([-g, h]) / max(h * h + g * g, 1e-30)
        ecc_err = math.sqrt(max(float(jac_e @ cov[:2, :2] @ jac_e), 0.0))
        omega_err = math.sqrt(max(float(jac_w @ cov[:2, :2] @ jac_w), 0.0))
    k_fit = np.asarray(x[:n_comp], dtype=np.float64)
    gam = np.asarray(x[n_comp:], dtype=np.float64)
    k_err = diag[n_shape : n_shape + n_comp]
    g_err = diag[n_shape + n_comp :]
    if n_gamma == 1:
        gam, g_err = np.repeat(gam, n_comp), np.repeat(g_err, n_comp)
    curve_all = _ephemeris_curve(
        np.asarray(table.bjd, dtype=np.float64), period, t_conj, ecc_fit, omega_fit
    )
    model_all = gam[:, None] + (k_fit * signs)[:, None] * curve_all[None, :]
    resid = np.where(valid, v - model_all, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a component with no usable velocity
        rms = np.sqrt(np.nanmean(resid**2, axis=1))
    par_names = ["secosw", "sesinw"] if fit_eccentricity else []
    par_names += [f"k_{n}" for n in names]
    par_names += ["gamma"] if n_gamma == 1 else [f"gamma_{n}" for n in names]
    return _EphemerisOrbit(
        names=tuple(names),
        period=float(period),
        t_conj=float(t_conj),
        ecc=float(ecc_fit),
        omega=float(omega_fit),
        k=k_fit,
        gamma=gam,
        gamma_mode="shared" if n_gamma == 1 else "one per component",
        errors={
            "period": 0.0,
            "t_conj": 0.0,
            "ecc": ecc_err,
            "omega": omega_err,
            "k": k_err,
            "gamma": g_err,
        },
        chi2=chi2,
        n_points=n_points,
        n_parameters=int(n_par),
        residuals=resid,
        rms=rms,
        used=good,
        covariance=cov,
        parameter_names=tuple(par_names),
        held=tuple(name for name, h in zip(names, held_components, strict=True) if h),
    )


def _shape_trials() -> list[tuple[float, float]]:
    """The 73 shapes ``(e, omega)`` from which a fit of the eccentricity starts: the circular
    orbit, and six eccentricities at twelve arguments of periastron each."""
    trials = [(0.0, 0.0)]
    for e_val in (0.03, 0.1, 0.2, 0.35, 0.5, 0.7):
        trials += [(e_val, w) for w in np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)]
    return trials


def _plain_orbit(orbit: RVOrbit) -> RVOrbit:
    from dataclasses import fields

    return RVOrbit(**{f.name: getattr(orbit, f.name) for f in fields(RVOrbit)})


def fit_rv_ephemeris(
    table,
    *,
    period: float,
    t_conj: float,
    ecc: float = 0.0,
    omega: float = 0.0,
    fit_eccentricity: bool = False,
    gamma: str | None = None,
    components=None,
) -> RVOrbit:
    """Fit the semi-amplitudes and the systemic velocity with the ephemeris held.

    The period and the time of conjunction are taken as known, as the light curve of an
    eclipsing binary gives them, and are not fitted. With the eccentricity and the argument
    of periastron held as well, the velocity of component ``i`` is

        v_i(t) = gamma_i + s_i K_i c(t),    c(t) = cos(nu(t) + omega) + e cos(omega),

    with ``s_i = +1`` for the first component and ``-1`` for the second, so the model is
    linear in the semi-amplitudes and the systemic velocity or velocities. They are found
    by weighted linear least squares with ``K_i >= 0``, and no starting value is needed.
    With ``fit_eccentricity`` the two shape parameters ``(sqrt(e) cos w, sqrt(e) sin w)``
    are fitted around that linear solution, from the three best points of a scan over 73
    shapes.

    A fit that holds the ephemeris has two parameters fewer than :func:`fit_rv_orbit`,
    and it names the stars: the first component is the one eclipsed at ``t_conj``, where
    a fit that knows the period alone returns the semi-amplitudes of two alike stars in
    either order. It does not recover more orbits. On 7,800 simulated Gaia RVS
    tables of eclipsing binaries the share with both semi-amplitudes within 10 percent
    was within 2 percentage points of that of :func:`assign_components`
    at the known period, in every half magnitude of ``G_RVS`` from 6.0 to 13.5, once the
    latter was compared in the order that fits (``docs/benchmarks.md``).

    Parameters
    ----------
    table
        A :class:`~albireo.todcor.VelocityTable`. The velocities that enter, and the
        components that are held for want of velocities, are those of
        :func:`fit_rv_orbit`.
    period
        The period [d], held.
    t_conj
        The time of conjunction, held: the superior conjunction of the first component,
        ``nu + omega = pi / 2``, which is the time of its eclipse.
    ecc, omega
        The eccentricity and the argument of periastron of the first component [rad]:
        held, or the reference of nothing where ``fit_eccentricity`` is set. The default
        is a circular orbit.
    fit_eccentricity
        Fit ``(sqrt(e) cos w, sqrt(e) sin w)``, each within 0.95 and with ``e`` at most
        0.95, in place of holding ``ecc`` and ``omega``.
    gamma
        ``"shared"`` or ``"per-component"``, as in :func:`fit_rv_orbit`.
    components
        Component names to fit (default all).

    Returns
    -------
    RVOrbit
        With ``period`` and ``t_conj`` as given and zero errors on both, and on ``ecc``
        and ``omega`` where they were held. ``covariance`` covers the parameters in
        ``parameter_names``: the two shape parameters where they were fitted, the
        semi-amplitudes and the systemic velocity or velocities.

    Raises
    ------
    ValueError
        If the table has no more usable velocities than parameters.
    """
    return _plain_orbit(
        _fit_rv_ephemeris(
            table,
            period=period,
            t_conj=t_conj,
            ecc=ecc,
            omega=omega,
            fit_eccentricity=fit_eccentricity,
            gamma=gamma,
            components=components,
        )
    )


def reassign_by_orbit(table, predicted, *, threshold: float = 3.0):
    """Swap the two components at epochs where the swap fits the orbit far better.

    Two similar spectra at similar light fractions give a correlation surface that is
    nearly symmetric under the exchange of the two shifts, and a per-epoch measurement
    then returns one of the two equivalent minima at random. No single epoch determines
    which star is which. The orbit does, and a disentangling provides one. The decision
    is made on the relative velocity ``v_1 - v_2``, which is free of a shared zero point,
    after removing the constant offset between the table and the prediction (each
    component of a differential table has its own zero point).

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
    return _exchanged(table, swap), swap


@dataclass(frozen=True, eq=False)
class Assignment:
    """What an orbit decided for the epochs of a two-component velocity table.

    Attributes
    ----------
    exchanged
        Per epoch: the two components were interchanged.
    alternative
        Per epoch: the other minimum of the correlation was taken
        (``VelocityTable.alternative``).
    resolved
        Per epoch: the epoch was flagged for a second minimum and the orbit decided between
        the two minima, so the epoch is no longer flagged ``blended``.
    """

    exchanged: np.ndarray
    alternative: np.ndarray
    resolved: np.ndarray

    @classmethod
    def none(cls, n_epochs: int) -> Assignment:
        """The assignment that changes nothing."""
        return cls(*(np.zeros(int(n_epochs), dtype=bool) for _ in range(3)))

    @property
    def changes(self) -> bool:
        """Whether any epoch is exchanged, takes the other minimum or loses its flag."""
        return bool(self.exchanged.any() or self.alternative.any() or self.resolved.any())

    def same_as(self, other: Assignment) -> bool:
        """Whether the two make the same decisions at every epoch."""
        return (
            np.array_equal(self.exchanged, other.exchanged)
            and np.array_equal(self.alternative, other.alternative)
            and np.array_equal(self.resolved, other.resolved)
        )

    def apply(self, table):
        """The table with these decisions made, or ``table`` itself where there are none.

        The other minimum is taken first: its velocities and covariance are interchanged
        with those the correlation returned, which stay recorded as the alternative,
        ``chi2`` rises by ``margin`` and ``margin`` changes sign. The resolved epochs lose
        the ``blended`` flag. The components are then interchanged at the exchanged
        epochs (:func:`reassign_by_orbit` lists what is exchanged). ``settings`` records
        the three counts as ``reassigned_by_orbit``, ``second_minimum_by_orbit`` and
        ``resolved_by_orbit``. The decisions can be made on one copy of a table and
        applied to another with the same epochs.
        """
        from dataclasses import replace

        if not self.changes:
            return table
        out = _with_alternative(table, self.alternative)
        if self.resolved.any():
            out = replace(out, blended=np.asarray(out.blended, dtype=bool) & ~self.resolved)
        if self.exchanged.any():
            out = _exchanged(out, self.exchanged)
        counts = {
            "reassigned_by_orbit": int(self.exchanged.sum()),
            "second_minimum_by_orbit": int(self.alternative.sum()),
            "resolved_by_orbit": int(self.resolved.sum()),
        }
        return replace(out, settings={**dict(out.settings), **counts})


def assign_by_orbit(table, predicted, *, threshold: float = 3.0, exchange: bool = True):
    """Decide by predicted velocities what the epochs of a table leave open.

    Two decisions are made. The order of the pair is decided at the usable epochs by
    :func:`reassign_by_orbit`, where ``exchange`` allows it. The minimum is decided at the
    epochs the table flags for a second minimum (``VelocityTable.second_minimum``), where
    it records the velocities of that minimum and their covariance
    (``VelocityTable.alternative``). At such an epoch the minimum returned and the other
    one are each compared with the prediction, in the order the first rule gives each,
    by

        chi2 = e . C^-1 . e,    e = v - v_predicted - offset,

    with ``C`` the covariance of that minimum's velocities and ``offset`` the median
    difference between the table and the prediction over the usable epochs, one value per
    component. The other minimum also carries its rise in the chi-square of the
    correlation, ``margin`` divided by the reduced chi-square, so that the sum is the
    chi-square of the spectrum and of the orbit together, and it is taken where its sum is
    the lower. Every such epoch is decided and loses the ``blended`` flag, with the
    velocities of the minimum taken.

    The covariance is used whole, because the two velocities of a blended epoch are
    correlated (to -0.8 at the smallest separations of the simulated Gaia RVS epochs of
    ``docs/benchmarks.md``), and the errors alone do not describe the direction along which
    the two minima differ.

    Parameters
    ----------
    table
        A two-component :class:`~albireo.todcor.VelocityTable`.
    predicted
        Predicted velocities, ``(2, n_epochs)`` km/s. Their zero points need not match
        the table's.
    threshold
        Passed to :func:`reassign_by_orbit`, and used in the same way for the order of
        each minimum at a flagged epoch.
    exchange
        Whether the two components may be interchanged. Without it only the minimum is
        decided.

    Returns
    -------
    (VelocityTable, Assignment)
        The table with the decisions made (:meth:`Assignment.apply`) and the decisions. A
        table with other than two components, or with fewer than three usable epochs, is
        returned unchanged.
    """
    pred = np.asarray(predicted, dtype=np.float64)
    assignment = Assignment.none(table.n_epochs)
    if table.n_components != 2 or pred.shape != (2, table.n_epochs):
        return table, assignment
    assignment = _decided(table, assignment, pred, threshold, exchange, _open_epochs(table))
    return assignment.apply(table), assignment


def assign_components(
    table,
    *,
    period: float,
    circular: bool = False,
    gamma: str | None = None,
    threshold: float = 3.0,
    max_rounds: int = 6,
    exchange: bool = True,
    light_ratio_max: float | None = _EXCHANGE_LIGHT_RATIO,
    period_window: float = 0.0,
):
    """Fit a Keplerian at a known period and decide by it what the epochs leave open.

    One epoch of a correlation can leave two things open. Two alike spectra at alike
    light fractions give a surface that is nearly symmetric under the exchange of the two
    shifts, and each epoch returns the pair of velocities in one of the two orders
    (:func:`reassign_by_orbit`). And where the lines of the two stars overlap, the surface
    has a second minimum that noise can make the lower. The table flags such an epoch and
    records the velocities of the other minimum (``VelocityTable.alternative``). With the
    period known, the orbit and these decisions are found together:

    1. A Keplerian is fitted to the usable epochs (:func:`fit_rv_orbit`). The epochs whose
       order it contradicts are exchanged, every epoch flagged for a second minimum takes
       the minimum the orbit fits better (:func:`assign_by_orbit`), and the fit is
       repeated with those epochs, until nothing changes or ``max_rounds`` rounds have
       been made.
    2. Where the two components may be exchanged, the same is done from up to three
       further assignments, made from the period alone. The magnitude of the velocity
       difference does not depend on the assignment. It is fitted with the magnitude of a
       Keplerian's relative velocity over a grid of phase, eccentricity and argument of
       periastron, and the sign of each of the best curves assigns the epochs. With
       ``period_window`` the grid also covers the periods of that window. Each of these
       fits starts from the period of its curve, and the table as measured is fitted
       from the best of those periods as well.
    3. The assignment reached from the table as measured is returned, unless the best of
       the others lowers the chi-square of the orbit by more than ``threshold**2`` times
       its reduced chi-square.

    One exchange from one fit is not enough where the first fit is poor, and a poor first
    fit is the rule for two alike stars, whose epochs come out in either order at random.
    On simulated tables of such a pair (semi-amplitudes of 61 and 64 km/s, errors of
    0.5 km/s, twelve epochs, each in a random order) one exchange gave both
    semi-amplitudes within 2 percent for 2, 1, 2 and 0 of 60 tables at eccentricities of
    0, 0.3, 0.6 and 0.8. With one further start, from the circular curves alone, the
    counts were 60, 45, 24 and 3, and with the grid they are 60, 60, 48 and 12
    (``scripts/assignment_bench.py``). Twelve epochs rarely sample the periastron passage
    of an orbit of eccentricity 0.8, and with forty epochs 42 of 60 are recovered.

    Parameters
    ----------
    table
        A two-component :class:`~albireo.todcor.VelocityTable`. A table with another
        number of components is fitted as it is.
    period
        The period [d], as :func:`fit_rv_orbit` takes it. Every fit starts from it, except
        those of ``period_window``.
    circular, gamma
        Passed to :func:`fit_rv_orbit`.
    threshold
        The significance an exchange needs. An epoch is exchanged where that reduces the
        residual of its relative velocity by more than this many times its error
        (:func:`reassign_by_orbit`), and an assignment made from the period alone replaces
        the one reached from the table as measured where it lowers the chi-square by more
        than the square of this number times the reduced chi-square.
    max_rounds
        The largest number of rounds of decisions followed by a new fit, from each
        starting assignment.
    exchange
        Whether the two components may be interchanged. Without it only the minimum is
        decided, at the epochs flagged for a second one.
    light_ratio_max
        No epoch is exchanged where the two light fractions, the medians of
        ``table.light`` over the epochs, differ by more than this factor. The exchange
        assumes that the two orders of a pair are equivalent solutions of the correlation,
        and at light fractions a factor of several apart they are not: a 95/5 pair went
        from 5 to 56 percent off in the primary, with 19 of its 80 epochs exchanged on
        noise in the faint component's velocity. ``None`` exchanges at any ratio, and
        fractions that are not finite and positive do not apply the rule. The rule does
        not concern the second minimum, which is decided at any ratio.
    period_window
        The half-width of the range of frequency around ``1 / period`` in which the
        assignments of step 2 are also made, in units of ``1 / T``, with ``T`` the time
        span of the usable epochs. It is sampled at eight frequencies per ``1 / T``. Zero,
        the default, is for a period that is known. A period that is off by ``x / T``
        puts the phase off by ``x / 2`` of a cycle at either end of the span, the epochs
        near a conjunction are then assigned wrongly, and the fit that follows does not
        leave that assignment. The peak of a periodogram of two alike components is known
        to a fraction of ``1 / T``. On a benchmark table of 12 usable epochs over 716 d
        the candidate nearest the injected period of 7.675 d was 7.650 d, ``0.30 / T``
        away. The assignment made at that period gave an orbit at a chi-square of 175
        with semi-amplitudes eleven times the injected ones, and with a window of 1 the
        same candidate gave the injected orbit at 14.9. The pipeline's candidate fits
        pass 1.

    Returns
    -------
    (VelocityTable, RVOrbit, Assignment)
        The table with the decisions made (:meth:`Assignment.apply`), the orbit fitted to
        it, and the decisions: the epochs exchanged, those at which the other minimum was
        taken, and those that lost the ``blended`` flag. Where nothing is decided the
        table is ``table`` itself.

    Notes
    -----
    Which of two alike stars is the first component is not determined by a table. The
    same velocities fit equally well with the two components exchanged at every epoch,
    the two semi-amplitudes interchanged and ``omega`` advanced by pi. The starting
    assignments keep the order of most of the weight of the table as measured.

    The lowest chi-square is not taken as it stands, because the assignments are many
    and a table of few epochs can be fitted by a wrong one. On a benchmark table of 10
    usable epochs (an eccentricity of 0.58, a S/N of 17), the assignment as measured gave
    semi-amplitudes 9 and 12 percent below the injected ones at a chi-square of 23.1 for
    13 degrees of freedom. With two epochs exchanged, an orbit of eccentricity 0.88 and
    semi-amplitudes eight times the injected ones fitted at 18.8.

    Every epoch flagged for a second minimum is decided, and none is left out, so that
    the orbits of different assignments and of different periods are fitted to the same
    velocities and their chi-squares can be compared.

    A :class:`ValueError` from the first fit is raised, as :func:`fit_rv_orbit` raises it
    for a table that cannot support an orbit. An assignment whose table cannot be fitted
    is left out.
    """

    def fit_from(start_period):
        def fit(candidate):
            return fit_rv_orbit(candidate, period=start_period, circular=circular, gamma=gamma)

        return fit

    fit = fit_from(period)
    nothing = Assignment.none(table.n_epochs)
    if table.n_components != 2:
        return table, fit(table), nothing
    exchange = bool(exchange) and _alike_light(table, light_ratio_max)
    if not exchange and not _open_epochs(table).any():
        return table, fit(table), nothing
    search = (threshold, int(max_rounds), exchange)
    measured = _settled_assignment(table, nothing.exchanged, fit, *search)
    if not exchange:
        return measured
    other = None
    starts = _assignments_by_period(table, float(period), threshold, float(period_window))
    for start, start_period in starts:
        try:
            reached = _settled_assignment(table, start, fit_from(start_period), *search)
        except ValueError:
            continue
        if other is None or reached[1].chi2 < other[1].chi2:
            other = reached
    if other is None:
        return measured
    reduced = other[1].chi2 / max(other[1].n_points - other[1].n_parameters, 1)
    gain = measured[1].chi2 - other[1].chi2
    return other if gain > float(threshold) ** 2 * reduced else measured


def assign_by_ephemeris(
    table,
    *,
    period: float,
    t_conj: float,
    ecc: float = 0.0,
    omega: float = 0.0,
    fit_eccentricity: bool = False,
    gamma: str | None = None,
    threshold: float = 3.0,
    max_rounds: int = 6,
    exchange: bool = True,
    light_ratio_max: float | None = _EXCHANGE_LIGHT_RATIO,
):
    """Fit an orbit at a held ephemeris and decide by it what the epochs leave open.

    The decisions are those of :func:`assign_components`: the order of the two velocities
    of an epoch, and the minimum at an epoch flagged for a second one. The search over
    starting assignments that :func:`assign_components` makes from the period alone is not
    needed here. The ephemeris gives the sign of ``v_1 - v_2`` at every epoch, since the
    first component approaches after its eclipse, and the pairs measured in the other
    order by more than ``threshold`` times the error of their difference are exchanged
    before the first fit. The rounds of decisions and fits then follow, with
    :func:`fit_rv_ephemeris` as the fit.

    The sign is that of the velocity curve, which depends on the eccentricity and the
    argument of periastron: at the conjunction of an eccentric orbit the curve is
    ``e cos(omega)`` and not zero. With ``fit_eccentricity`` the shape is not known, so
    every shape of the scan of :func:`fit_rv_ephemeris` orders the pairs in its own way and
    is fitted with the shape held, and the rounds start from each of the three orders whose
    fit is best. The assignment with the lowest chi-square is returned. Taking the order
    from a circular curve instead exchanged correctly measured epochs: on tables of two
    alike stars with every epoch in the injected order it recovered both semi-amplitudes
    within 5 percent for 16 of 20 tables at ``e = 0.4`` and 12 of 20 at ``e = 0.6``.

    Parameters
    ----------
    table
        A two-component :class:`~albireo.todcor.VelocityTable`. A table with another
        number of components is fitted as it is.
    period, t_conj, ecc, omega, fit_eccentricity, gamma
        Passed to :func:`fit_rv_ephemeris`. With ``fit_eccentricity`` the shape at ``ecc``
        and ``omega`` is one of those from which a starting order is taken.
    threshold, max_rounds, exchange, light_ratio_max
        As in :func:`assign_components`. The starting order by the sign is taken only
        with one systemic velocity for both components, since two zero points move the
        sign of the difference.

    Returns
    -------
    (VelocityTable, RVOrbit, Assignment)
        As :func:`assign_components` returns them.

    Raises
    ------
    ValueError
        From the first fit, for a table that cannot support it.
    """

    def fit(candidate):
        return _fit_rv_ephemeris(
            candidate,
            period=period,
            t_conj=t_conj,
            ecc=ecc,
            omega=omega,
            fit_eccentricity=fit_eccentricity,
            gamma=gamma,
        )

    nothing = Assignment.none(table.n_epochs)
    if table.n_components != 2:
        return table, _plain_orbit(fit(table)), nothing
    exchange = bool(exchange) and _alike_light(table, light_ratio_max)
    open_epochs = _open_epochs(table)
    if not exchange and not open_epochs.any():
        return table, _plain_orbit(fit(table)), nothing
    starts = [nothing.exchanged]
    shared = gamma == "shared" or (gamma is None and all(table.absolute))
    if exchange and shared:
        v = np.asarray(table.velocity, dtype=np.float64)
        s = np.asarray(table.sigma, dtype=np.float64)
        times = np.asarray(table.bjd, dtype=np.float64)
        orderable = np.asarray(table.good, dtype=bool) & ~open_epochs
        with np.errstate(invalid="ignore"):
            difference = (v[0] - v[1]) / np.hypot(s[0], s[1])

        def order_by(e_val: float, om: float) -> np.ndarray:
            """The epochs measured against the sign of the curve of one shape."""
            curve = _ephemeris_curve(times, period, t_conj, e_val, om)
            with np.errstate(invalid="ignore"):
                wrong = difference * np.sign(curve) < -float(threshold)
            return orderable & np.nan_to_num(wrong, nan=0.0).astype(bool)

        if not fit_eccentricity:
            starts = [order_by(float(ecc), float(omega))]
        else:
            scored = []
            for e_val, om in [(float(ecc), float(omega)), *_shape_trials()]:
                start = order_by(e_val, om)
                candidate = _exchanged(table, start) if start.any() else table
                try:
                    held = _fit_rv_ephemeris(
                        candidate, period=period, t_conj=t_conj, ecc=e_val, omega=om, gamma=gamma
                    )
                except ValueError:
                    continue
                scored.append((held.chi2, start))
            starts = []
            for _, start in sorted(scored, key=lambda item: item[0]):
                if not any(np.array_equal(start, other) for other in starts):
                    starts.append(start)
                if len(starts) == 3:
                    break
            starts = starts or [nothing.exchanged]
    best, error = None, None
    for start in starts:
        try:
            reached = _settled_assignment(table, start, fit, threshold, int(max_rounds), exchange)
        except ValueError as raised:
            error = raised
            continue
        if best is None or reached[1].chi2 < best[1].chi2:
            best = reached
    if best is None:
        raise error
    assigned, orbit, decided = best
    return assigned, _plain_orbit(orbit), decided


def _alike_light(table, ratio_max: float | None) -> bool:
    """Whether the median light fractions of a two-component table are within ``ratio_max``.

    True as well where the rule does not apply: no ``ratio_max``, or fractions that are
    not finite and positive.
    """
    if ratio_max is None:
        return True
    light = np.asarray(table.light, dtype=np.float64)
    if light.ndim != 2 or not np.all(np.any(np.isfinite(light), axis=1)):
        return True
    fractions = np.nanmedian(light, axis=1)
    if np.any(fractions <= 0.0):
        return True
    return bool(fractions.max() / fractions.min() <= float(ratio_max))


def _ridge(covariance) -> np.ndarray:
    """Per epoch: a correlation above 0.9 between two of the velocities (the blend flag's)."""
    cov = np.asarray(covariance, dtype=np.float64)
    variance = np.diagonal(cov, axis1=1, axis2=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.abs(cov) / np.sqrt(variance[:, :, None] * variance[:, None, :])
    off_diagonal = ~np.eye(cov.shape[-1], dtype=bool)
    return np.any(np.nan_to_num(correlation[:, off_diagonal], nan=0.0) > 0.9, axis=1)


def _open_epochs(table) -> np.ndarray:
    """Per epoch: flagged for a second minimum alone, with both minima usable.

    The epoch is flagged ``blended`` with ``second_minimum`` true, every velocity and
    error of the minimum returned and of the other one is finite, no component is at its
    search edge, and neither minimum lies on a ridge. An orbit decides such an epoch
    (:func:`assign_by_orbit`). A velocity removed by the caller closes its epoch.
    """
    v = np.asarray(table.velocity, dtype=np.float64)
    s = np.asarray(table.sigma, dtype=np.float64)
    other = np.asarray(table.alternative, dtype=np.float64)
    other_s = np.asarray(table.alternative_sigma, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        usable = np.all(np.isfinite(v) & np.isfinite(s) & (s > 0.0), axis=0)
        usable &= np.all(np.isfinite(other) & np.isfinite(other_s) & (other_s > 0.0), axis=0)
    at_edge = np.any(np.broadcast_to(np.asarray(table.at_edge, dtype=bool), v.shape), axis=0)
    flagged = np.asarray(table.blended, dtype=bool) & np.asarray(table.second_minimum, dtype=bool)
    ridge = _ridge(table.covariance) | _ridge(table.alternative_covariance)
    return flagged & usable & ~at_edge & ~ridge


def _with_alternative(table, taken):
    """The two-component ``table`` with the other minimum taken at the epochs ``taken``.

    The velocities and their covariance are interchanged with ``alternative`` and
    ``alternative_covariance``, and the errors follow. ``chi2`` rises by ``margin`` and
    ``margin`` changes sign, so the minimum the correlation returned stays recorded as the
    alternative, lower by that amount. ``light`` and ``delta_chi2`` are kept.
    """
    from dataclasses import replace

    taken = np.asarray(taken, dtype=bool)
    if not taken.any():
        return table
    velocity, other = np.array(table.velocity), np.array(table.alternative)
    velocity[:, taken], other[:, taken] = table.alternative[:, taken], table.velocity[:, taken]
    covariance, other_covariance = (
        np.array(table.covariance),
        np.array(table.alternative_covariance),
    )
    covariance[taken] = table.alternative_covariance[taken]
    other_covariance[taken] = table.covariance[taken]
    chi2, margin = np.array(table.chi2, dtype=np.float64), np.array(table.margin, dtype=np.float64)
    chi2[taken] = chi2[taken] + margin[taken]
    margin[taken] = -margin[taken]
    sigma, sigma_ivar = np.array(table.sigma), np.array(table.sigma_ivar)
    sigma[:, taken] = table.alternative_sigma[:, taken]
    # The errors with the weights as given are the quoted ones without the reduced
    # chi-square of that minimum, which the quoted errors include unless errors="ivar".
    scale = np.ones(int(taken.sum()))
    if table.settings.get("errors") != "ivar":
        n_parameters = table.settings.get("n_parameters", 0)
        dof = np.maximum(np.asarray(table.n_pixels) - n_parameters, 1)
        scale = chi2[taken] / dof[taken]
    sigma_ivar[:, taken] = sigma[:, taken] / np.sqrt(scale)
    return replace(
        table,
        velocity=velocity,
        sigma=sigma,
        sigma_ivar=sigma_ivar,
        covariance=covariance,
        chi2=chi2,
        margin=margin,
        alternative=other,
        alternative_covariance=other_covariance,
    )


def _chosen_minima(table, ordered, predicted, threshold: float, exchange: bool, open_epochs):
    """Which minimum each open epoch takes, and in which order of its components.

    ``ordered`` is the table with the order of the pair already decided at its usable
    epochs, from which the offsets between the table and the prediction are taken.
    Returns the epochs at which the other minimum is taken, those at which the minimum
    taken is interchanged, and those decided: every open epoch, or none where fewer than
    three usable epochs give no offset (:func:`assign_by_orbit` has the statistic).
    """
    n = table.n_epochs
    taken, flipped = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    v = np.asarray(ordered.velocity, dtype=np.float64)
    s = np.asarray(ordered.sigma, dtype=np.float64)
    good = ordered.good & np.all(np.isfinite(v), axis=0) & np.all(np.isfinite(s), axis=0)
    if not open_epochs.any() or int(good.sum()) < 3:
        return taken, flipped, np.zeros(n, dtype=bool)
    target = predicted + np.median((v - predicted)[:, good], axis=1)[:, None]
    scale = np.ones(n)
    if table.settings.get("errors") != "ivar":
        scale = np.asarray(table.reduced_chi2, dtype=np.float64)
    for j in np.flatnonzero(open_epochs):
        minima = (
            (table.velocity[:, j], table.covariance[j], 0.0),
            (table.alternative[:, j], table.alternative_covariance[j], table.margin[j] / scale[j]),
        )
        best = None
        for is_other, (u, cov, rise) in enumerate(minima):
            u, cov = np.asarray(u, dtype=np.float64), np.asarray(cov, dtype=np.float64)
            flip = False
            if exchange:
                # The rule of `reassign_by_orbit` on this minimum's own pair, with the
                # variance of the difference taken from the covariance.
                sigma_r = math.sqrt(max(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1], 1e-30))
                r, r_target = u[0] - u[1], target[0, j] - target[1, j]
                flip = (abs(r - r_target) - abs(-r - r_target)) / sigma_r > threshold
            if flip:
                u, cov = u[::-1], cov[::-1, ::-1]
            e = u - target[:, j]
            try:
                total = float(e @ np.linalg.solve(cov, e)) + float(rise)
            except np.linalg.LinAlgError:
                total = np.inf
            if best is None or total < best[0]:
                best = (total, bool(is_other), bool(flip))
        taken[j], flipped[j] = best[1], best[2]
    return taken, flipped, np.array(open_epochs, dtype=bool)


def _decided(table, state: Assignment, predicted, threshold: float, exchange: bool, open_epochs):
    """One round of decisions by ``predicted``, from the assignment ``state``.

    The order of the pair is decided at the epochs usable as measured, starting from the
    order ``state`` gives them, and then the minimum and its order at the open epochs,
    afresh from the table as measured.
    """
    ordered = state.exchanged & ~open_epochs
    base = _exchanged(table, ordered) if ordered.any() else table
    if exchange:
        ordered = ordered ^ reassign_by_orbit(base, predicted, threshold=threshold)[1]
        base = _exchanged(table, ordered) if ordered.any() else table
    taken, flipped, resolved = _chosen_minima(
        table, base, predicted, threshold, exchange, open_epochs
    )
    return Assignment(ordered | flipped, taken, resolved)


def _settled_assignment(table, start, fit, threshold: float, max_rounds: int, exchange: bool):
    """The best assignment reached from ``start`` by rounds of decisions and fits.

    ``start`` marks the epochs of ``table`` that are exchanged before the first fit, which
    leaves out the epochs flagged for a second minimum. Each round decides the order of
    the pairs and the minima by the current orbit (:func:`_decided`) and fits again, until
    nothing changes. Returns the table, the orbit and the assignment of the round with
    the lowest chi-square, which is the last one unless the decisions cycle. The first
    fit is not compared with the others where a second minimum is open, since it has
    fewer velocities.
    """
    open_epochs = _open_epochs(table)
    nothing = Assignment.none(table.n_epochs)
    state = Assignment(np.array(start, dtype=bool), nothing.alternative, nothing.resolved)
    current = state.apply(table)
    orbit = fit(current)
    best = (current, orbit, state)
    comparable = not open_epochs.any()
    for _ in range(max_rounds):
        decided = _decided(table, state, orbit.predict(table.bjd), threshold, exchange, open_epochs)
        if decided.same_as(state):
            break
        state = decided
        current = state.apply(table)
        try:
            orbit = fit(current)
        except ValueError:
            break
        if not comparable or orbit.chi2 < best[1].chi2:
            best = (current, orbit, state)
            comparable = True
    return best


def _relative_curves(t, period: float) -> np.ndarray:
    """Relative-velocity curves of unit amplitude on a grid of Keplerian orbits at ``period``.

    One row per orbit: ``cos(nu + omega) + e cos(omega)`` at the times ``t``, the relative
    velocity of two components divided by ``K_1 + K_2``. The circular curves come first,
    at ``_ASSIGNMENT_PHASES`` phases over half a period, since the magnitude of a circular
    curve repeats after half a period. Then, for every eccentricity of
    ``_ASSIGNMENT_ECCENTRICITIES`` and ``_ASSIGNMENT_OMEGAS`` arguments of periastron over
    the circle, the curves at ``_ASSIGNMENT_PHASES`` phases of periastron over a period.

    Kepler's equation is solved here with NumPy, by the starter and the Newton count of
    :func:`albireo.kepler.solve_kepler`, so that the scan compiles nothing. It is solved
    once per phase and eccentricity, since the argument of periastron does not enter it.
    A test checks the curves against :func:`albireo.kepler.radial_velocity`.
    """
    t = np.asarray(t, dtype=np.float64)
    half = np.arange(_ASSIGNMENT_PHASES) / (2.0 * _ASSIGNMENT_PHASES)
    full = np.arange(_ASSIGNMENT_PHASES) / float(_ASSIGNMENT_PHASES)
    omegas = np.arange(_ASSIGNMENT_OMEGAS) * 2.0 * np.pi / _ASSIGNMENT_OMEGAS

    def true_anomaly(phase, ecc):
        mean_anomaly = 2.0 * np.pi * (t[None, :] / period - phase[:, None])
        m = np.mod(mean_anomaly + np.pi, 2.0 * np.pi) - np.pi
        ecc_anomaly = m + ecc * np.sin(m) + 0.5 * ecc**2 * np.sin(2.0 * m)
        for _ in range(_NEWTON_ITERATIONS):
            ecc_anomaly = ecc_anomaly - (ecc_anomaly - ecc * np.sin(ecc_anomaly) - m) / (
                1.0 - ecc * np.cos(ecc_anomaly)
            )
        return 2.0 * np.arctan2(
            np.sqrt(1.0 + ecc) * np.sin(0.5 * ecc_anomaly),
            np.sqrt(1.0 - ecc) * np.cos(0.5 * ecc_anomaly),
        )

    curves = [np.cos(true_anomaly(half, 0.0))]
    for e in _ASSIGNMENT_ECCENTRICITIES:
        nu = true_anomaly(full, e)
        curves.extend(np.cos(nu + w) + e * np.cos(w) for w in omegas)
    return np.concatenate(curves)


def _window_periods(t, period: float, window: float) -> list[float]:
    """The periods of the assignment scan: ``period`` first, then those of the window.

    The window is ``window / T`` in frequency on either side of ``1 / period``, with ``T``
    the time span of the epochs ``t``, sampled at ``_ASSIGNMENT_WINDOW_STEPS`` frequencies
    per ``1 / T``. A change of ``1 / T`` in frequency moves the phase by half a cycle at
    either end of the span, so this spacing leaves a sixteenth of that between a period
    and the nearest one sampled. The periods are ordered by their distance from ``period``.
    """
    span = float(np.ptp(t)) if np.size(t) else 0.0
    if not window > 0.0 or not span > 0.0:
        return [period]
    steps = math.ceil(window * _ASSIGNMENT_WINDOW_STEPS)
    periods = [period]
    for k in range(1, steps + 1):
        for sign in (1.0, -1.0):
            frequency = 1.0 / period + sign * k / (_ASSIGNMENT_WINDOW_STEPS * span)
            if frequency > 0.0:
                periods.append(1.0 / frequency)
    return periods


def _assignments_by_period(
    table, period: float, threshold: float, window: float = 0.0
) -> list[tuple[np.ndarray, float]]:
    """Starting assignments from the period alone: the epochs each would exchange.

    The magnitude of the difference of the two velocities is the same in either order.
    Over the usable epochs it is fitted with ``A |c(t)|`` for every curve ``c`` of
    :func:`_relative_curves`, with ``A`` solved by weighted least squares. A curve, with
    the sign that agrees with most of the weight of the table as measured, is a predicted
    relative velocity, and :func:`reassign_by_orbit` gives the epochs that contradict it.
    The curves are taken in order of increasing chi-square, and the first
    ``_ASSIGNMENT_STARTS`` distinct sets of exchanged epochs are returned, each with the
    period of its curve. They are starting points for the Keplerian fit, which is not
    restricted to the grid.

    With ``window`` zero the curves are those at ``period``, and the empty set, the table
    as measured, is left out. With a window the curves of every period of
    :func:`_window_periods` are ranked together, and the empty set is returned as well,
    once, where its best curve is at another period than ``period``: the table as
    measured is then also fitted from that period.
    """
    v = np.asarray(table.velocity, dtype=np.float64)
    s = np.asarray(table.sigma, dtype=np.float64)
    good = table.good & np.all(np.isfinite(v), axis=0) & np.all(np.isfinite(s), axis=0)
    variance = np.where(good, s[0] ** 2 + s[1] ** 2, 1.0)
    good = good & (variance > 0.0)
    if int(good.sum()) < 3:
        return []
    difference = np.where(good, v[0] - v[1], 0.0)
    weight = np.where(good, 1.0 / variance, 0.0)
    t = np.asarray(table.bjd, dtype=np.float64)
    ranked = []  # (chi-square, period, curve), the best curves of every period
    for trial in _window_periods(t[good], period, window):
        curves = _relative_curves(t, trial)
        norm = (weight * curves**2).sum(axis=1)
        usable = norm > 0.0
        amplitude = (weight * np.abs(difference) * np.abs(curves)).sum(axis=1) / np.where(
            usable, norm, 1.0
        )
        residual = np.abs(difference)[None, :] - amplitude[:, None] * np.abs(curves)
        chi2 = np.where(usable, (weight * residual**2).sum(axis=1), np.inf)
        for index in np.argsort(chi2, kind="stable")[:_ASSIGNMENT_CURVES]:
            if np.isfinite(chi2[index]):
                ranked.append((float(chi2[index]), trial, amplitude[index] * curves[index]))
    ranked.sort(key=lambda entry: entry[0])  # stable: the nearer period first on a tie
    starts: list[tuple[np.ndarray, float]] = []
    seen = [np.zeros(table.n_epochs, dtype=bool)]
    as_measured = False  # whether the empty set has been met
    n_distinct = 0
    for _, trial, curve in ranked[:_ASSIGNMENT_CURVES]:
        if float(np.sum(weight * np.sign(difference * curve))) < 0.0:
            curve = -curve
        predicted = np.stack([0.5 * curve, -0.5 * curve])
        exchanged = reassign_by_orbit(table, predicted, threshold=threshold)[1]
        if not exchanged.any():
            if not as_measured and trial != period:
                starts.append((exchanged, trial))
            as_measured = True
        elif not any(np.array_equal(exchanged, other) for other in seen):
            seen.append(exchanged)
            starts.append((exchanged, trial))
            n_distinct += 1
            if n_distinct >= _ASSIGNMENT_STARTS:
                break
    return starts


def _exchanged(table, swap):
    """The two-component ``table`` with its components exchanged at the epochs ``swap``.

    Velocities, errors, covariances, light fractions, detection statistics and edge flags
    are exchanged together, with the velocities and the covariance of the other minimum
    (``alternative``), and ``settings["reassigned_by_orbit"]`` records the count. Used by
    :func:`reassign_by_orbit` and :meth:`Assignment.apply`.
    """
    from dataclasses import replace

    swap = np.asarray(swap, dtype=bool)

    def exchange(array):
        array = np.asarray(array)
        out = np.array(array)
        out[0] = np.where(swap, array[1], array[0])
        out[1] = np.where(swap, array[0], array[1])
        return out

    def exchange_covariance(array):
        out = np.array(array)
        out[swap] = out[swap][:, ::-1, :][:, :, ::-1]
        return out

    return replace(
        table,
        velocity=exchange(table.velocity),
        sigma=exchange(table.sigma),
        sigma_ivar=exchange(table.sigma_ivar),
        covariance=exchange_covariance(table.covariance),
        light=exchange(table.light),
        delta_chi2=exchange(table.delta_chi2),
        at_edge=exchange(table.at_edge),
        alternative=exchange(table.alternative),
        alternative_covariance=exchange_covariance(table.alternative_covariance),
        settings={**dict(table.settings), "reassigned_by_orbit": int(swap.sum())},
    )


def relativistic_add(v_kms, u_kms):
    """Relativistic addition of two velocities, the composition law of log-wavelength shifts.

    See ``docs/math.md`` §7.6.
    """
    b1 = np.asarray(v_kms, dtype=np.float64) / C_KMS
    b2 = np.asarray(u_kms, dtype=np.float64) / C_KMS
    return C_KMS * (b1 + b2) / (1.0 + b1 * b2)

"""Detached eclipsing binaries as a magnitude-limited survey lists them.

**Experimental.** The names, the draw parameters and the record's ``meta`` keys may change.

:func:`albireo.population.draw_population` draws double-lined binaries from the field
distributions on a mean dwarf sequence. This module draws the detached eclipsing binaries
of a catalogue such as Gaia's ``vari_eclipsing_binary``, for which four further things
matter: the stars have evolved, a system is listed in proportion to its eclipse
probability and to the volume its light reaches, an eclipse changes the light fractions of
the epochs that fall in it, and the two stars can lie in different boxes of a spectral
library.

- :class:`StellarTracks` reads the table of MIST evolutionary tracks kept in the package
  (``scripts/build_mist_tracks.py``) and gives the radius, temperature and luminosity of a
  star of given mass and age.
- :func:`roche_radius`, :func:`pseudo_synchronous_ratio`, :func:`hidden_fraction`,
  :func:`limb_darkening` and :func:`eclipse_light` are the geometry of two limb-darkened
  spheres on a Keplerian orbit.
- :class:`LibrarySet` joins library boxes that cover different temperature ranges into one
  label space, with one continuum scale.
- :func:`draw_eclipsing_population` draws a design sample stratified in G_RVS and in the
  temperature class of the primary, and :func:`catalogue_weights` maps it onto the counts of
  a catalogue.

The prescriptions and their sources are listed in the docstring of
:func:`draw_eclipsing_population`.

References
----------
Choi, J., Dotter, A., Conroy, C., et al. 2016, ApJ, 823, 102
Claret, A. 2019, RNAAS, 3, 17
Claret, A. & Bloemen, S. 2011, A&A, 529, A75
Dotter, A. 2016, ApJS, 222, 8
Eggleton, P. P. 1983, ApJ, 268, 368
Hut, P. 1981, A&A, 99, 126
IJspeert, L. W., Tkachenko, A., Johnston, C., et al. 2024, A&A, 691, A242
Lurie, J. C., Vyhmeister, K., Hawley, S. L., et al. 2017, AJ, 154, 250
Moe, M. & Di Stefano, R. 2017, ApJS, 230, 15
Mowlavi, N., Holl, B., Lecoeur-Taibi, I., et al. 2023, A&A, 674, A16
Tokovinin, A., Thomas, S., Sterzik, M. & Udry, S. 2006, A&A, 450, 681
Winn, J. N. 2010, in Exoplanets, ed. S. Seager (Tucson: University of Arizona Press), 55
Wright, N. J., Drake, J. J., Mamajek, E. E. & Henry, G. W. 2011, ApJ, 743, 48
"""

from __future__ import annotations

import dataclasses
import functools
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from albireo.gaia import RVS_BAND
from albireo.kepler import true_anomaly_numpy
from albireo.population import (
    _LOGG_SUN,
    _MAMAJEK_ROWS,
    _SYNCHRONOUS_KMS,
    BinarySystem,
    _f_logp,
    _f_twin,
    _gamma_largeq,
    _gamma_smallq,
    eclipse_probability,
    ecliptic_to_icrs,
    semi_amplitudes,
    semimajor_axis_rsun,
)

__all__ = [
    "GRVS_EDGES",
    "TEMPERATURE_CLASSES",
    "LibrarySet",
    "StellarTracks",
    "catalogue_weights",
    "design_cells",
    "draw_eclipsing_population",
    "eclipse_light",
    "hidden_fraction",
    "limb_darkening",
    "population_columns",
    "pseudo_synchronous_ratio",
    "roche_radius",
]

TEMPERATURE_CLASSES: dict[str, tuple[float, float]] = {
    "K": (4000.0, 5300.0),
    "G": (5300.0, 6000.0),
    "F": (6000.0, 7250.0),
    "A": (7250.0, 10000.0),
}
"""The classes of the primary's effective temperature, in K: lower limit included, upper
limit excluded."""

GRVS_EDGES: tuple[float, ...] = tuple(float(x) for x in np.arange(6.0, 13.5 + 1e-9, 0.5))
"""Edges of the G_RVS bins of the design sample: fifteen bins of 0.5 mag from 6.0 to 13.5."""

DEFAULT_LIBRARIES = ("bosz2024-cool-rvs", "bosz2024-fgk-rvs", "bosz2024-hot-rvs")
"""Registry names of the three BOSZ boxes in the RVS band, 3200 to 10,000 K together."""

ECLIPTIC_LATITUDE_SHARES = (0.119, 0.124, 0.129, 0.147, 0.180, 0.170, 0.088, 0.032, 0.010)
"""Shares of the Gaia DR3 eclipsing-binary candidates brighter than G_RVS = 12 in bins of
10 degrees of absolute ecliptic latitude, from 0 to 90 degrees (aggregate archive query of
2026-10-07, ``internal/research/2026-10-07-rvs-eclipsing-todcor``)."""

_TRACKS_FILE = Path(__file__).parent / "data_files" / "mist_v1.2_solar_tracks.npz"
_MBOL_SUN = 4.74
_DR4_SPAN_DAYS = 2005.0  # 2014-07-25 to 2020-01-20
_DR4_START_BJD = 2456863.94

# Linear limb-darkening coefficients at log g = 4.5 and solar metallicity. "rp": Gaia G_RP
# (Claret 2019). "rvs": interpolated at 858 nm between Cousins I and Sloan z' (Claret &
# Bloemen 2011), since no table has a G_RVS column.
_LD_TEFF = np.array(
    [4000.0, 4500.0, 5000.0, 5500.0, 5750.0, 6000.0, 6500.0, 7000.0, 7500.0, 8000.0, 9000.0, 1e4]
)
_LD = {
    "rvs": np.array([0.55, 0.58, 0.54, 0.50, 0.47, 0.45, 0.41, 0.38, 0.36, 0.34, 0.31, 0.29]),
    "rp": np.array(
        [0.599, 0.622, 0.580, 0.532, 0.508, 0.485, 0.445, 0.415, 0.391, 0.375, 0.351, 0.325]
    ),
}

# Share of orbits with e > 0.1 against period, for primaries cooler and hotter than
# 7000 K. The nodes reproduce the shares of the eclipsing samples: 4, 6, 10, 17 and 33
# percent at 1-1.8, 1.8-3.2, 3.2-5.6, 5.6-10 and 10-18 d for all temperatures together, none
# below 1.5 d, and at 1.5-4 d 16 percent of hot pairs against 1 to 2 percent of cool ones
# (Torres et al. 2010; Van Eylen et al. 2016; IJspeert et al. 2024). Beyond 18 d the share
# rises to that of the field.
_ECCENTRIC_TEFF = 7000.0
_ECCENTRIC_COOL = (
    np.log10([1.5, 2.4, 4.2, 7.5, 13.4, 30.0, 100.0]),
    np.array([0.0, 0.015, 0.06, 0.15, 0.33, 0.55, 0.85]),
)
_ECCENTRIC_HOT = (
    np.log10([1.2, 1.6, 2.4, 4.2, 7.5, 13.4, 30.0, 100.0]),
    np.array([0.0, 0.10, 0.16, 0.20, 0.25, 0.35, 0.55, 0.85]),
)

_ROCHE_FILLING = 0.75
_KRAFT_TEFF = 6250.0
_ACTIVITY_TEFF = 6500.0
_SATURATION_ROSSBY = 0.13
_UNRESOLVED_TERTIARY = 0.68

EB_TRANSITS_DR3 = (10.0, 20.0, 33.0)
"""10th, 50th and 90th percentiles of ``rv_nb_transits`` of the Gaia DR3 eclipsing-binary
candidates brighter than G_RVS = 12 that have a published velocity (aggregate archive query
of 2026-10-07). The all-sky values are 8, 18 and 30."""


# ---------------------------------------------------------------------------
# Evolutionary tracks
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class StellarTracks:
    """Radius, temperature and luminosity of a star of given mass and age.

    The table holds evolutionary tracks at a list of masses, each sampled at the same
    equivalent evolutionary points, which mark the same stage of evolution on every track
    (Dotter 2016). A quantity at a mass between two tracks is interpolated linearly in log
    mass at a fixed point, the age included, and then along the resulting track in log
    age. The packaged table is MIST v1.2 at solar composition (Choi et al. 2016), from
    the start of the pre-main-sequence contraction to the tip of the red giant branch, for
    0.10 to 6.0 solar masses (``scripts/build_mist_tracks.py``).

    Attributes
    ----------
    mass
        Track masses in solar masses, increasing, ``(n_mass,)``.
    eep
        The equivalent evolutionary points kept, ``(n_point,)``.
    log_age, log_l, log_teff, log_r
        ``(n_mass, n_point)``: log10 of the age in years, of the luminosity in solar
        units, of the effective temperature in K and of the radius in solar radii. ``nan``
        beyond the last point of a track.
    meta
        Provenance: source, checksum, citation, composition, and the points of the
        zero-age and terminal-age main sequence.
    """

    mass: np.ndarray
    eep: np.ndarray
    log_age: np.ndarray
    log_l: np.ndarray
    log_teff: np.ndarray
    log_r: np.ndarray
    meta: dict = dataclasses.field(default_factory=dict)

    @classmethod
    def load(cls, path=None) -> StellarTracks:
        """The packaged table, or the table at ``path`` in the same format."""
        return _load_tracks(str(_TRACKS_FILE if path is None else path))

    @property
    def zams(self) -> int:
        """The equivalent evolutionary point of the zero-age main sequence."""
        return int(self.meta.get("points", {}).get("zams", 202))

    @property
    def tams(self) -> int:
        """The equivalent evolutionary point of the terminal-age main sequence."""
        return int(self.meta.get("points", {}).get("tams", 454))

    def at(self, mass, age_yr) -> dict[str, np.ndarray]:
        """The state of stars of the given masses at the given ages.

        Parameters
        ----------
        mass
            Masses in solar masses.
        age_yr
            Ages in years, broadcast against ``mass``. An age below the first point of a
            track takes the first point.

        Returns
        -------
        dict
            ``radius`` [solar radii], ``teff`` [K], ``log_l`` [log10 solar luminosities],
            ``logg`` [log10 cm/s^2] and ``eep`` (the interpolated evolutionary point), each
            of the broadcast shape. Every entry is ``nan`` for a mass outside the table and
            for an age beyond the last point of the star's track.
        """
        mass, age = np.broadcast_arrays(
            np.asarray(mass, dtype=np.float64), np.asarray(age_yr, dtype=np.float64)
        )
        shape = mass.shape
        m, a = mass.ravel(), age.ravel()
        out = {key: np.full(m.size, np.nan) for key in ("log_r", "log_teff", "log_l", "eep")}
        with np.errstate(invalid="ignore"):
            inside = np.isfinite(m) & (m >= self.mass[0]) & (m <= self.mass[-1]) & (a > 0.0)
        log_mass = np.log10(self.mass)
        index = np.flatnonzero(inside)
        for chunk in np.array_split(index, max(1, index.size // 20_000 + 1)):
            if chunk.size == 0:
                continue
            lm, la = np.log10(m[chunk]), np.log10(a[chunk])
            hi = np.clip(np.searchsorted(log_mass, lm, side="right"), 1, log_mass.size - 1)
            lo = hi - 1
            w = ((lm - log_mass[lo]) / (log_mass[hi] - log_mass[lo]))[:, None]
            track_age = (1.0 - w) * self.log_age[lo] + w * self.log_age[hi]
            n_valid = np.isfinite(track_age).sum(axis=1)
            rows = np.arange(chunk.size)
            # The last point not after the age. `nan` compares false, so the points beyond
            # the end of a track are not counted.
            k = (track_age <= la[:, None]).sum(axis=1) - 1
            k = np.clip(k, 0, np.maximum(n_valid - 2, 0))
            x0, x1 = track_age[rows, k], track_age[rows, k + 1]
            f = np.clip((la - x0) / np.maximum(x1 - x0, 1e-9), 0.0, 1.0)
            alive = (n_valid >= 2) & (la <= track_age[rows, np.maximum(n_valid - 1, 0)])
            for key in ("log_r", "log_teff", "log_l"):
                table = getattr(self, key)
                column = (1.0 - w) * table[lo] + w * table[hi]
                value = (1.0 - f) * column[rows, k] + f * column[rows, k + 1]
                out[key][chunk] = np.where(alive, value, np.nan)
            point = (1.0 - f) * self.eep[k] + f * self.eep[k + 1]
            out["eep"][chunk] = np.where(alive, point, np.nan)
        radius = 10.0 ** out["log_r"]
        with np.errstate(invalid="ignore", divide="ignore"):
            logg = _LOGG_SUN + np.log10(m) - 2.0 * out["log_r"]
        return {
            "radius": radius.reshape(shape),
            "teff": (10.0 ** out["log_teff"]).reshape(shape),
            "log_l": out["log_l"].reshape(shape),
            "logg": logg.reshape(shape),
            "eep": out["eep"].reshape(shape),
        }

    def age_at(self, mass, eep) -> np.ndarray:
        """The age in years at which stars of the given masses reach a point.

        ``nan`` for a mass outside the table and for a track that ends before the point.
        """
        m = np.atleast_1d(np.asarray(mass, dtype=np.float64))
        log_mass = np.log10(self.mass)
        with np.errstate(invalid="ignore"):
            inside = (m >= self.mass[0]) & (m <= self.mass[-1])
        lm = np.log10(np.where(inside, m, self.mass[0]))
        hi = np.clip(np.searchsorted(log_mass, lm, side="right"), 1, log_mass.size - 1)
        lo = hi - 1
        w = ((lm - log_mass[lo]) / (log_mass[hi] - log_mass[lo]))[:, None]
        track_age = (1.0 - w) * self.log_age[lo] + w * self.log_age[hi]
        points = self.eep.astype(np.float64)
        ages = np.array([np.interp(float(eep), points, row, right=np.nan) for row in track_age])
        ages = np.where(inside, 10.0**ages, np.nan)
        return ages.reshape(np.shape(mass))


@functools.cache
def _load_tracks(path: str) -> StellarTracks:
    with np.load(path) as data:
        meta = json.loads(str(data["meta"])) if "meta" in data.files else {}
        arrays = {
            key: np.asarray(data[key], dtype=np.float64)
            for key in ("mass", "log_age", "log_l", "log_teff", "log_r")
        }
        eep = np.asarray(data["eep"], dtype=np.int64)
    for value in arrays.values():
        value.setflags(write=False)
    return StellarTracks(eep=eep, meta=meta, **arrays)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def roche_radius(mass_ratio):
    """Volume-equivalent Roche-lobe radius in units of the separation (Eggleton 1983).

    ``R_L / a = 0.49 q^(2/3) / [0.6 q^(2/3) + ln(1 + q^(1/3))]`` with ``q`` the mass of the
    star divided by that of its companion, accurate to 1 percent at every mass ratio.
    """
    q = np.asarray(mass_ratio, dtype=np.float64)
    q23 = q ** (2.0 / 3.0)
    return 0.49 * q23 / (0.6 * q23 + np.log1p(np.cbrt(q)))


def pseudo_synchronous_ratio(ecc):
    """Rotation rate over mean orbital motion at pseudo-synchronisation (Hut 1981, eq. 42).

    ``Omega_ps / n = (1 + 7.5 e^2 + 5.625 e^4 + 0.3125 e^6)
    / [(1 + 3 e^2 + 0.375 e^4) (1 - e^2)^(3/2)]``, the rate at which the tidal torque
    averaged over an eccentric orbit vanishes. It is one for a circular orbit and 2.05 at
    ``e = 0.4``.
    """
    e2 = np.asarray(ecc, dtype=np.float64) ** 2
    numerator = 1.0 + 7.5 * e2 + 5.625 * e2**2 + 0.3125 * e2**3
    return numerator / ((1.0 + 3.0 * e2 + 0.375 * e2**2) * (1.0 - e2) ** 1.5)


def limb_darkening(teff, band: str = "rvs"):
    """Linear limb-darkening coefficient of a dwarf at effective temperature ``teff``.

    Tabulated from 4000 to 10,000 K at ``log g = 4.5`` and solar metallicity and held
    constant outside. ``band="rvs"`` is the value at 858 nm, interpolated between the
    Cousins I and Sloan z' coefficients of Claret & Bloemen (2011), since no published
    table has a G_RVS column: 0.55 at 4000 K, 0.47 at 5750 K and 0.29 at 10,000 K.
    ``band="rp"`` is Gaia G_RP (Claret 2019). They are continuum values, and the cores of
    the Ca II and Paschen lines darken differently.
    """
    if band not in _LD:
        raise ValueError(f"band must be one of {sorted(_LD)}; got {band!r}")
    return np.interp(np.asarray(teff, dtype=np.float64), _LD_TEFF, _LD[band])


_RING_NODES, _RING_WEIGHTS = np.polynomial.legendre.leggauss(48)


def _disc_light(x, u):
    """Light of a limb-darkened disc of unit radius inside the fractional radius ``x``,
    divided by ``2 pi``: the integral of ``[1 - u (1 - sqrt(1 - x^2))] x``."""
    return 0.5 * (1.0 - u) * x**2 + u / 3.0 * (1.0 - (1.0 - x**2) ** 1.5)


def hidden_fraction(separation, r_back, r_front, u=0.0):
    """Fraction of the light of the farther disc that the nearer disc hides.

    The farther star is a disc of radius ``r_back`` with the linear limb-darkening law
    ``I(mu) = 1 - u (1 - mu)``. A ring of radius ``r`` about its centre lies inside the
    nearer disc, of radius ``r_front`` at the projected ``separation`` ``d``, over the
    half-angle ``alpha`` with ``cos(alpha) = (r^2 + d^2 - r_front^2) / (2 r d)``. The rings
    with ``r <= r_front - d`` are hidden whole, and their light has a closed form. Those
    with ``|d - r_front| < r < d + r_front`` are hidden in part, and the integral of
    ``I r alpha / pi`` over them is taken by Gauss-Legendre quadrature in ``phi``, with
    ``r = c + h sin(phi)`` between the two limits. The substitution removes the square-root
    behaviour of ``alpha`` at the contact radii and of ``I`` at the limb.

    Parameters
    ----------
    separation
        Projected separation of the two centres, in the unit of the radii.
    r_back, r_front
        Radii of the farther and of the nearer disc.
    u
        Linear limb-darkening coefficient of the farther disc.

    Returns
    -------
    numpy.ndarray
        The hidden fraction, in ``[0, 1]``, of the broadcast shape: exactly one in a total
        eclipse, and within 1e-12 of the closed form of a uniform disc on 4000 random
        configurations.
    """
    d, rb, rf, u = np.broadcast_arrays(
        *(np.asarray(x, dtype=np.float64) for x in (separation, r_back, r_front, u))
    )
    d = np.abs(d)
    whole = np.clip((rf - d) / rb, 0.0, 1.0)
    lower = np.clip(np.abs(d - rf) / rb, 0.0, 1.0)
    upper = np.clip((d + rf) / rb, 0.0, 1.0)
    centre = 0.5 * (upper + lower)[..., None]
    half = 0.5 * np.clip(upper - lower, 0.0, None)[..., None]
    phi = 0.5 * math.pi * _RING_NODES
    x = centre + half * np.sin(phi)  # fractional radius of the ring
    dd, ff = (d / rb)[..., None], (rf / rb)[..., None]
    numerator = x**2 + dd**2 - ff**2
    denominator = 2.0 * x * dd
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_alpha = np.where(
            denominator > 0.0,
            numerator / np.where(denominator > 0.0, denominator, 1.0),
            np.where(numerator < 0.0, -1.0, 1.0),
        )
    alpha = np.arccos(np.clip(cos_alpha, -1.0, 1.0))
    intensity = 1.0 - u[..., None] * (1.0 - np.sqrt(np.clip(1.0 - x**2, 0.0, None)))
    weights = 0.5 * math.pi * _RING_WEIGHTS * np.cos(phi)
    partial = np.sum(intensity * x * alpha / math.pi * weights, axis=-1) * half[..., 0]
    hidden = (_disc_light(whole, u) + partial) / _disc_light(1.0, u)
    return np.clip(hidden, 0.0, 1.0)


def eclipse_light(system: BinarySystem, bjd, *, band: str = "rvs", third_light: float = 0.0):
    """Light fractions of the two stars at the epochs, with the eclipses.

    The projected separation of the centres at true anomaly ``nu`` is
    ``d = r sqrt(1 - sin^2(i) sin^2(nu + omega))`` with ``r = a (1 - e^2) / (1 + e cos nu)``,
    and the primary is the farther star where ``sin(nu + omega) > 0``: its superior
    conjunction, ``nu + omega = pi / 2``, is the conjunction of
    :attr:`albireo.population.BinarySystem.t_conj`. Where ``d < R_1 + R_2`` the farther star
    loses the fraction :func:`hidden_fraction` of its light, with the limb darkening of
    :func:`limb_darkening` at its temperature.

    Parameters
    ----------
    system
        The binary. Its ``light_ratio`` is the flux ratio out of eclipse.
    bjd
        Epoch times.
    band
        The band of the limb darkening, ``"rvs"`` or ``"rp"``.
    third_light
        Fraction of the total light out of eclipse that a third, uneclipsed source
        contributes.

    Returns
    -------
    dict
        ``visible``: ``(2, n)`` visible fraction of each star's light. ``fractions``:
        ``(2, n)`` light fractions of the two stars, or ``(3, n)`` with the third light as
        the last row; they sum to one at every epoch. ``flux``: ``(n,)`` total light
        relative to that out of eclipse. ``in_eclipse``: ``(n,)``, either star partly
        hidden. ``separation``: ``(n,)`` projected separation in units of ``R_1 + R_2``.
        ``primary_behind``: ``(n,)``, the primary is the farther star.
    """
    if not 0.0 <= third_light < 1.0:
        raise ValueError(f"third_light must lie in [0, 1); got {third_light}")
    bjd = np.asarray(bjd, dtype=np.float64)
    nu = true_anomaly_numpy(bjd, period=system.period, t_peri=system.t_peri, ecc=system.ecc)
    angle = nu + system.omega
    radius = system.a_rsun * (1.0 - system.ecc**2) / (1.0 + system.ecc * np.cos(nu))
    separation = radius * np.sqrt(1.0 - (math.sin(system.incl) * np.sin(angle)) ** 2)
    primary_behind = np.sin(angle) > 0.0
    overlap = separation < system.r1 + system.r2
    u1 = float(limb_darkening(system.teff1, band))
    u2 = float(limb_darkening(system.teff2, band))
    hidden1 = np.where(
        overlap & primary_behind, hidden_fraction(separation, system.r1, system.r2, u1), 0.0
    )
    hidden2 = np.where(
        overlap & ~primary_behind, hidden_fraction(separation, system.r2, system.r1, u2), 0.0
    )
    visible = np.stack([1.0 - hidden1, 1.0 - hidden2])
    pair = np.asarray(system.light_fractions, dtype=np.float64) * (1.0 - third_light)
    light = pair[:, None] * visible
    if third_light > 0.0:
        light = np.vstack([light, np.full((1, bjd.size), third_light)])
    flux = light.sum(axis=0)
    return {
        "visible": visible,
        "fractions": light / flux,
        "flux": flux,
        "in_eclipse": (hidden1 > 0.0) | (hidden2 > 0.0),
        "separation": separation / (system.r1 + system.r2),
        "primary_behind": primary_behind,
    }


def _eclipse_duration(period, rho, cos_i, ecc, e_sin_omega):
    """Duration from first to last contact of the eclipse at one conjunction, in days.

    With the separation ``r_c = a (1 - e^2) / (1 + e sin w)`` and the angular rate both
    held at their values at the conjunction, the centres are closer than ``R_1 + R_2``
    over the half-angle ``asin[sqrt(rho^2 (a / r_c)^2 - cos^2 i) / sin i]`` about it, with
    ``rho = (R_1 + R_2) / a``, which the star sweeps at
    ``(2 pi / P) sqrt(1 - e^2) (a / r_c)^2``. For a circular orbit this is equation 14 of
    Winn (2010) with the sum of the radii. ``e_sin_omega`` is ``e sin w`` for the eclipse
    of the primary and ``-e sin w`` for that of the secondary. Zero where there is no
    eclipse.
    """
    scale = (1.0 - ecc**2) / (1.0 + e_sin_omega)  # r_c / a
    sin_i = np.sqrt(np.clip(1.0 - cos_i**2, 1e-12, None))
    inside = np.clip((rho / scale) ** 2 - cos_i**2, 0.0, None)
    half = np.arcsin(np.clip(np.sqrt(inside) / sin_i, 0.0, 1.0))
    return period / math.pi * half * scale**2 / np.sqrt(1.0 - ecc**2)


# ---------------------------------------------------------------------------
# Photometry of a star from its temperature and luminosity
# ---------------------------------------------------------------------------

_PHOT_TEFF = np.array([row[1] for row in _MAMAJEK_ROWS], dtype=np.float64)[::-1]
_PHOT_BC_G = np.array(
    [row[7] - (_MBOL_SUN - 2.5 * row[2]) for row in _MAMAJEK_ROWS], dtype=np.float64
)[::-1]
_PHOT_G_RP = np.array([row[6] for row in _MAMAJEK_ROWS], dtype=np.float64)[::-1]


def _absolute_g(teff, log_l):
    """Absolute G of a star: the bolometric magnitude plus the correction of the dwarf
    sequence of Pecaut & Mamajek at the star's temperature, held constant outside it."""
    return _MBOL_SUN - 2.5 * np.asarray(log_l) + np.interp(teff, _PHOT_TEFF, _PHOT_BC_G)


def _g_minus_rp(teff):
    return np.interp(teff, _PHOT_TEFF, _PHOT_G_RP)


def _teff_from_colour(g_rp):
    """The temperature of the dwarf sequence at a G - G_RP colour."""
    colour = np.maximum.accumulate(_PHOT_G_RP[::-1]) + 1e-9 * np.arange(_PHOT_G_RP.size)
    return np.interp(g_rp, colour, _PHOT_TEFF[::-1])


# ---------------------------------------------------------------------------
# Library boxes as one label space
# ---------------------------------------------------------------------------


class LibrarySet:
    """Library boxes that cover adjoining temperature ranges, used as one label space.

    A pair of stars of an eclipsing binary can lie in different boxes of a library that
    is published, or downloaded, in pieces: a 9000 K primary with a 5500 K secondary. Each
    star is rendered from the box that contains its temperature, and the flux ratio of the
    two is the ratio of the continua of the two boxes times the square of the radius
    ratio. The continua must therefore be on one scale. The three BOSZ boxes of the
    registry are (:data:`DEFAULT_LIBRARIES`): at the nodes they share, 4000 and 7000 K,
    their continua are equal.

    A label outside its box is moved to the box's edge and the star is flagged. A
    secondary of 0.15 solar masses, at 3000 K, takes the spectrum of the coolest node.

    Parameters
    ----------
    libraries
        :class:`~albireo.library.SpectralLibrary` instances with the axes ``teff``,
        ``logg`` and ``mh``, on complete boxes, in any order. A star is assigned to the
        first box, in increasing temperature, whose upper temperature limit exceeds its
        own, and to the hottest box otherwise.
    band
        The wavelength range over which the continuum is averaged, in Angstrom.
    """

    def __init__(self, libraries: Sequence[Any], band=RVS_BAND) -> None:
        from scipy.interpolate import RegularGridInterpolator

        libraries = sorted(libraries, key=lambda lib: lib.bounds["teff"][0])
        if not libraries:
            raise ValueError("a LibrarySet needs at least one library")
        self.libraries = tuple(libraries)
        self.bounds = tuple(dict(lib.bounds) for lib in libraries)
        self.band = (float(band[0]), float(band[1]))
        self._continuum = []
        for lib in libraries:
            names = list(lib.label_names)
            if sorted(names) != ["logg", "mh", "teff"]:
                raise ValueError(f"a LibrarySet needs the axes teff, logg and mh; got {names}")
            nodes = np.asarray(lib.nodes, dtype=np.float64)
            columns = [nodes[:, names.index(axis)] for axis in ("teff", "logg", "mh")]
            axes = [np.unique(column) for column in columns]
            if nodes.shape[0] != int(np.prod([axis.size for axis in axes])):
                raise ValueError("a LibrarySet needs libraries on complete boxes")
            keep = (lib.wave >= self.band[0] - 5.0) & (lib.wave <= self.band[1] + 5.0)
            if not keep.any():
                raise ValueError("a library of the set does not cover the band")
            values = np.empty([axis.size for axis in axes])
            mean = np.mean(np.asarray(lib.log_continuum)[:, keep], axis=1)
            index = tuple(
                np.searchsorted(axis, col) for axis, col in zip(axes, columns, strict=True)
            )
            values[index] = mean
            self._continuum.append(RegularGridInterpolator(axes, values, method="linear"))

    @classmethod
    def fetch(cls, names: Sequence[str] = DEFAULT_LIBRARIES) -> LibrarySet:
        """The set of the named registry libraries (:func:`albireo.fetch_library`)."""
        from albireo.library import fetch_library

        return cls([fetch_library(name, progress=False) for name in names])

    @property
    def teff_range(self) -> tuple[float, float]:
        """The lowest and the highest temperature of the set."""
        return self.bounds[0]["teff"][0], self.bounds[-1]["teff"][1]

    def box(self, teff) -> np.ndarray:
        """Index of the box a star of temperature ``teff`` is rendered from."""
        upper = np.array([b["teff"][1] for b in self.bounds[:-1]])
        return np.searchsorted(upper, np.asarray(teff, dtype=np.float64), side="right")

    def clamp(self, teff, logg, mh):
        """The labels moved into their box, and whether any was moved.

        Returns
        -------
        (box, teff, logg, mh, moved)
            Arrays of the broadcast shape: the box index, the labels inside it, and the
            flag.
        """
        teff, logg, mh = np.broadcast_arrays(
            *(np.asarray(x, dtype=np.float64) for x in (teff, logg, mh))
        )
        box = self.box(teff)
        out = []
        for values, axis in ((teff, "teff"), (logg, "logg"), (mh, "mh")):
            lo = np.array([b[axis][0] for b in self.bounds])[box]
            hi = np.array([b[axis][1] for b in self.bounds])[box]
            out.append(np.clip(values, lo, hi))
        moved = (out[0] != teff) | (out[1] != logg) | (out[2] != mh)
        return box, out[0], out[1], out[2], moved

    def labels(self, teff: float, logg: float, mh: float):
        """One star: its library, its labels inside the box, and whether any was moved."""
        box, t, g, m, moved = self.clamp(teff, logg, mh)
        labels = {"teff": float(t), "logg": float(g), "mh": float(m)}
        return self.libraries[int(box)], labels, bool(moved)

    def log_continuum(self, teff, logg, mh) -> np.ndarray:
        """Natural log of the continuum, averaged over the band, at the clamped labels.

        Linear in the labels between the nodes. The interpolator of the label fit
        (:func:`albireo.library_interpolator`) is cubic where the box allows it, and the
        two differ by less than a percent of the continuum inside the boxes.
        """
        box, t, g, m, _ = self.clamp(teff, logg, mh)
        out = np.empty(box.shape)
        for i, interpolator in enumerate(self._continuum):
            sel = box == i
            if sel.any():
                out[sel] = interpolator(np.stack([t[sel], g[sel], m[sel]], axis=-1))
        return out

    def flux_ratio(self, labels1: Mapping[str, float], labels2: Mapping[str, float], radius_ratio):
        """``F_2 / F_1`` in the band: the ratio of the continua times ``(R_2 / R_1)^2``."""
        c1 = self.log_continuum(labels1["teff"], labels1["logg"], labels1["mh"])
        c2 = self.log_continuum(labels2["teff"], labels2["logg"], labels2["mh"])
        return float(radius_ratio) ** 2 * float(np.exp(c2 - c1))


# ---------------------------------------------------------------------------
# Moe & Di Stefano (2017), tabulated for a vectorised draw
# ---------------------------------------------------------------------------


def _mass_ratio_density(m1: float, logp: float, q, twins=(1.0, 0.95, "uniform")):
    """Mass-ratio density of Moe & Di Stefano (2017) on ``q``, with its twin excess adjustable.

    The broken power law is theirs (equations 5 to 16). Their excess of twins is a
    fraction ``F`` of the companions above ``q = 0.3``, placed uniformly on ``[0.95, 1]``.
    ``twins = (scale, start, shape)`` multiplies ``F`` by ``scale`` and places the excess
    on ``[start, 1]``, uniformly or, with ``shape = "ramp"``, with a density that rises
    linearly from zero at ``start``. ``(1, 0.95, "uniform")`` is the published form. The
    density is normalised to one over ``q >= 0.3``, so that the companion frequency per
    decade of period, which counts those companions, keeps its meaning.
    """
    scale, start, shape = twins
    q = np.asarray(q, dtype=np.float64)
    gs, gl = _gamma_smallq(m1, logp), _gamma_largeq(m1, logp)
    power = np.where(q < 0.3, (q / 0.3) ** gs, (q / 0.3) ** gl)
    large = q >= 0.3
    n_large = float(np.trapezoid(power[large], q[large]))
    excess = min(float(scale) * _f_twin(m1, logp), 0.95)
    twin = np.where(q >= start, (q - start) if shape == "ramp" else 1.0, 0.0)
    norm = float(np.trapezoid(twin, q))
    if excess > 0.0 and norm > 0.0:
        power = power + twin / norm * n_large * excess / (1.0 - excess)
    return power * (1.0 - excess) / n_large


class _CompanionDistributions:
    """The period and mass-ratio distributions of Moe & Di Stefano (2017) on tables.

    :mod:`albireo.population` evaluates the published functions one system at a time. A
    draw that rejects most of its candidates needs them for 10^5 to 10^6 primaries, so they
    are tabulated on 36 primary masses and 28 periods, logarithmically spaced, and a
    candidate takes the nearest node. The companion frequency of the publication counts
    mass ratios above 0.3, and the frequency of companions above ``q_min`` follows from the
    mass-ratio density at each node (:func:`_mass_ratio_density`).
    """

    def __init__(self, mass_range, period_range, q_min: float, twins) -> None:
        self.log_mass = np.linspace(math.log10(mass_range[0]), math.log10(mass_range[1]), 36)
        lo, hi = math.log10(period_range[0]), math.log10(period_range[1])
        self.log_period = np.linspace(lo, hi, 28)
        self.fine = np.linspace(lo, hi, 341)
        full = np.linspace(0.1, 1.0, 901)
        keep = full >= q_min - 1e-12
        self.q = full[keep]
        boost = np.empty((self.log_mass.size, self.log_period.size))
        self.q_cdf = np.empty((self.log_mass.size, self.log_period.size, self.q.size))
        for i, lm in enumerate(self.log_mass):
            for j, lp in enumerate(self.log_period):
                density = _mass_ratio_density(10.0**lm, float(lp), full, twins)
                boost[i, j] = float(np.trapezoid(density[keep], full[keep]))
                cdf = np.concatenate(
                    [
                        [0.0],
                        np.cumsum(0.5 * (density[keep][1:] + density[keep][:-1]) * np.diff(self.q)),
                    ]
                )
                self.q_cdf[i, j] = cdf / cdf[-1]
        self.period_cdf = np.empty((self.log_mass.size, self.fine.size))
        self.frequency = np.empty(self.log_mass.size)
        for i, lm in enumerate(self.log_mass):
            density = np.array([_f_logp(10.0**lm, float(lp)) for lp in self.fine])
            density = density * np.interp(self.fine, self.log_period, boost[i])
            cdf = np.concatenate(
                [[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(self.fine))]
            )
            self.frequency[i] = cdf[-1]
            self.period_cdf[i] = cdf / cdf[-1]

    def _node(self, grid: np.ndarray, values) -> np.ndarray:
        step = grid[1] - grid[0]
        return np.clip(np.rint((values - grid[0]) / step).astype(int), 0, grid.size - 1)

    def companion_frequency(self, m1) -> np.ndarray:
        """Companions in the period and mass-ratio range per primary of mass ``m1``."""
        return np.interp(np.log10(m1), self.log_mass, self.frequency)

    def draw_period(self, rng, m1) -> np.ndarray:
        node = self._node(self.log_mass, np.log10(m1))
        u = rng.uniform(size=node.size)
        out = np.empty(node.size)
        for i in np.unique(node):
            sel = node == i
            out[sel] = np.interp(u[sel], self.period_cdf[i], self.fine)
        return 10.0**out

    def draw_mass_ratio(self, rng, m1, period) -> np.ndarray:
        i = self._node(self.log_mass, np.log10(m1))
        j = self._node(self.log_period, np.log10(period))
        key = i * self.log_period.size + j
        u = rng.uniform(size=key.size)
        out = np.empty(key.size)
        for k in np.unique(key):
            sel = key == k
            out[sel] = np.interp(
                u[sel], self.q_cdf[k // self.log_period.size, k % self.log_period.size], self.q
            )
        return out


# ---------------------------------------------------------------------------
# The draw
# ---------------------------------------------------------------------------


def design_cells(
    classes: Mapping[str, tuple[float, float]] = TEMPERATURE_CLASSES,
    grvs_edges: Sequence[float] = GRVS_EDGES,
) -> list[tuple[str, int, float, float]]:
    """The cells of the design sample: ``(class, bin index, G_RVS low, G_RVS high)``."""
    edges = [float(x) for x in grvs_edges]
    return [(name, k, edges[k], edges[k + 1]) for name in classes for k in range(len(edges) - 1)]


def _eccentric_share(period, teff1):
    """Share of orbits with ``e > 0.1`` at a period, for the temperature of the primary.

    Linear in log period between the nodes of the two tables above.
    """
    lp = np.log10(period)
    cool = np.interp(lp, *_ECCENTRIC_COOL)
    hot = np.interp(lp, *_ECCENTRIC_HOT)
    return np.where(np.asarray(teff1) < _ECCENTRIC_TEFF, cool, hot)


def _eccentricity_envelope(period):
    """Upper envelope of the eccentricities of close binaries (Mazeh 2008, eq. 4.4).

    ``e < 0.98 - 3.25 exp[-(6.3 P)^0.23]`` with ``P`` in days: zero at 0.35 d, 0.27 at 1 d,
    0.44 at 2 d and 0.62 at 5 d.
    """
    return np.clip(0.98 - 3.25 * np.exp(-((6.3 * np.asarray(period)) ** 0.23)), 0.0, 0.95)


def _draw_eccentricity(rng, period, teff1):
    """Eccentricities of systems of the given periods and primary temperatures.

    A share :func:`_eccentric_share` has ``e`` uniform between 0.1 and the envelope
    :func:`_eccentricity_envelope`. Half as many again have ``e`` uniform below 0.1, and
    the rest are circular.
    """
    share = _eccentric_share(period, teff1)
    top = _eccentricity_envelope(period)
    u, v = rng.uniform(size=np.shape(period)), rng.uniform(size=np.shape(period))
    large = (u < share) & (top > 0.1)
    small = ~large & (u < 1.5 * share)
    ecc = np.where(large, 0.1 + v * (top - 0.1), np.where(small, 0.1 * v, 0.0))
    return np.where(ecc >= 1e-3, ecc, 0.0)


def _field_rotation(rng, teff, radius):
    """Equatorial velocity in km/s of a star that tides have not synchronised.

    Cooler than 6250 K: a rotation period uniform between 5 and 40 d. Hotter: log-normal
    with a dispersion of 0.5 about a median that rises from 40 km/s at 6700 K to 130 km/s
    at 8000 K, at most 300 km/s.
    """
    teff = np.asarray(teff, dtype=np.float64)
    median = np.interp(teff, [6700.0, 8000.0], [40.0, 130.0])
    hot = np.minimum(np.exp(rng.normal(np.log(median), 0.5)), 300.0)
    cool = _SYNCHRONOUS_KMS * np.asarray(radius) / rng.uniform(5.0, 40.0, size=teff.shape)
    return np.where(teff >= _KRAFT_TEFF, hot, cool)


def _turnover_days(mass):
    """Convective turnover time of Wright et al. (2011), with the mass held inside
    their range of 0.09 to 1.36 solar masses."""
    lm = np.log10(np.clip(mass, 0.09, 1.36))
    return 10.0 ** (1.16 - 1.49 * lm - 0.54 * lm**2)


def _class_window(tracks: StellarTracks, teff_range, logg_min: float, max_age_yr: float):
    """Masses and ages at which a track lies in a temperature class above a gravity."""
    teff = 10.0**tracks.log_teff
    logg = _LOGG_SUN + np.log10(tracks.mass)[:, None] - 2.0 * tracks.log_r
    with np.errstate(invalid="ignore"):
        inside = (
            (teff >= teff_range[0])
            & (teff < teff_range[1])
            & (logg >= logg_min)
            & (tracks.log_age <= math.log10(max_age_yr))
            & (tracks.eep[None, :] >= tracks.zams)
        )
    rows = np.flatnonzero(inside.any(axis=1))
    if rows.size == 0:
        raise ValueError(f"no track reaches the temperatures {teff_range}")
    lo, hi = max(rows[0] - 1, 0), min(rows[-1] + 1, tracks.mass.size - 1)
    oldest = float(np.nanmax(np.where(inside, tracks.log_age, np.nan)))
    return float(tracks.mass[lo]), float(tracks.mass[hi]), min(max_age_yr, 1.3 * 10.0**oldest)


def _draw_candidates(rng, n, window, teff_range, config, tracks, companions, libraries):
    """One batch of candidate systems of a class, with their weights.

    Returns a dict of arrays over the candidates that pass every cut. ``weight`` is the
    product of the companion frequency, the eclipse probability, the probability that the
    photometry samples an eclipse, and the 3/2 power of the combined light in the band.
    """
    m_lo, m_hi, age_max = window
    slope = -2.3 + 1.0
    m1 = (m_lo**slope + rng.uniform(size=n) * (m_hi**slope - m_lo**slope)) ** (1.0 / slope)
    age = rng.uniform(0.0, age_max, size=n)
    star1 = tracks.at(m1, age)
    with np.errstate(invalid="ignore"):
        keep = (
            np.isfinite(star1["teff"])
            & (star1["teff"] >= teff_range[0])
            & (star1["teff"] < teff_range[1])
            & (star1["logg"] >= config["logg_min"])
            & (star1["eep"] >= tracks.zams)
        )
    m1, age = m1[keep], age[keep]
    star1 = {key: value[keep] for key, value in star1.items()}
    period = companions.draw_period(rng, m1)
    q = companions.draw_mass_ratio(rng, m1, period)
    m2 = q * m1
    star2 = tracks.at(m2, age)
    ecc = _draw_eccentricity(rng, period, star1["teff"])
    omega = rng.uniform(0.0, 2.0 * math.pi, size=m1.size)
    a = semimajor_axis_rsun(m1 + m2, period)
    r1, r2 = star1["radius"], star2["radius"]
    with np.errstate(invalid="ignore"):
        detached = (r1 < _ROCHE_FILLING * roche_radius(1.0 / q) * a * (1.0 - ecc)) & (
            r2 < _ROCHE_FILLING * roche_radius(q) * a * (1.0 - ecc)
        )
        keep = np.isfinite(r2) & detached
    p_ecl = eclipse_probability(r1, r2, a, ecc, omega)
    cos_i = rng.uniform(size=m1.size) * p_ecl
    mh = np.clip(rng.normal(config["mh"][0], config["mh"][1], size=m1.size), -0.9, 0.45)
    u_accept = rng.uniform(size=m1.size)

    # The eclipses at the two conjunctions, in G.
    e_sin = ecc * np.sin(omega)
    rho = (r1 + r2) / a
    depth, duration = [], []
    flux_g = [10.0 ** (-0.4 * _absolute_g(s["teff"], s["log_l"])) for s in (star1, star2)]
    for behind, front, sign in ((0, 1, 1.0), (1, 0, -1.0)):
        radii = (r1, r2)
        separation = a * (1.0 - ecc**2) / (1.0 + sign * e_sin) * cos_i
        teff = (star1, star2)[behind]["teff"]
        with np.errstate(invalid="ignore"):
            hidden = hidden_fraction(
                separation, radii[behind], radii[front], limb_darkening(teff, "rp")
            )
            share = flux_g[behind] / (flux_g[0] + flux_g[1])
            depth.append(-2.5 * np.log10(np.clip(1.0 - hidden * share, 1e-6, None)))
            duration.append(_eclipse_duration(period, rho, cos_i, ecc, sign * e_sin))
    missed = np.ones(m1.size)
    for k in range(2):
        mean = config["n_photometric_transits"] * duration[k] / period
        sampled = 1.0 - np.exp(-mean) * (1.0 + mean + 0.5 * mean**2)
        with np.errstate(invalid="ignore"):
            missed = missed * np.where(depth[k] >= config["min_depth_mag"], 1.0 - sampled, 1.0)
    with np.errstate(invalid="ignore"):
        keep &= np.maximum(depth[0], depth[1]) >= config["min_depth_mag"]

    cont1 = libraries.log_continuum(np.nan_to_num(star1["teff"], nan=5000.0), star1["logg"], mh)
    cont2 = libraries.log_continuum(
        np.nan_to_num(star2["teff"], nan=5000.0), np.nan_to_num(star2["logg"], nan=4.5), mh
    )
    with np.errstate(invalid="ignore"):
        light1, light2 = r1**2 * np.exp(cont1), r2**2 * np.exp(cont2)
        weight = (
            companions.companion_frequency(m1) * p_ecl * (1.0 - missed) * (light1 + light2) ** 1.5
        )
        keep &= np.isfinite(weight) & (weight > 0.0)
    out = {
        "m1": m1,
        "m2": m2,
        "age": age,
        "period": period,
        "ecc": ecc,
        "omega": omega,
        "cos_i": cos_i,
        "mh": mh,
        "a": a,
        "light_ratio": light2 / light1,
        "depth1": depth[0],
        "depth2": depth[1],
        "duration1": duration[0],
        "duration2": duration[1],
        "weight": weight,
        "u": u_accept,
        "p_ecl": p_ecl,
    }
    for label, star in (("1", star1), ("2", star2)):
        for key in ("radius", "teff", "log_l", "logg", "eep"):
            out[key + label] = star[key]
    return {key: value[keep] for key, value in out.items()}


def _draw_class(rng, n, teff_range, config, tracks, companions, libraries, *, batch=200_000):
    """``n`` systems of one temperature class, by rejection on the weights.

    Candidates are drawn in batches and each carries its own uniform deviate. After every
    batch the bound is the largest weight of the pool so far, a candidate is accepted
    where its deviate is below its weight over the bound, and the draw ends when ``n`` are
    accepted. The first ``n`` in the order drawn are returned.
    """
    window = _class_window(tracks, teff_range, config["logg_min"], config["max_age_yr"])
    pool: dict[str, list[np.ndarray]] = {}
    for _ in range(400):
        candidates = _draw_candidates(
            rng, batch, window, teff_range, config, tracks, companions, libraries
        )
        for key, value in candidates.items():
            pool.setdefault(key, []).append(value)
        weight = np.concatenate(pool["weight"])
        accepted = np.concatenate(pool["u"]) < weight / weight.max()
        if int(accepted.sum()) >= n:
            index = np.flatnonzero(accepted)[:n]
            merged = {key: np.concatenate(value)[index] for key, value in pool.items()}
            merged["pool_size"] = int(weight.size)
            merged["acceptance"] = float(accepted.mean())
            return merged
    raise RuntimeError(f"fewer than {n} systems accepted for the temperatures {teff_range}")


def draw_eclipsing_population(
    n_per_cell: int = 130,
    *,
    seed: int = 0,
    libraries: LibrarySet | None = None,
    tracks: StellarTracks | None = None,
    classes: Mapping[str, tuple[float, float]] | None = None,
    grvs_edges: Sequence[float] = GRVS_EDGES,
    period_range: tuple[float, float] = (0.2, 500.0),
    q_min: float = 0.1,
    twin_excess: float = 0.3,
    twin_from: float = 0.85,
    twin_shape: str = "ramp",
    logg_min: float = 3.5,
    max_age_gyr: float = 10.0,
    min_depth_mag: float = 0.04,
    n_photometric_transits: float = 93.0,
    mh: tuple[float, float] = (-0.1, 0.2),
    gamma_sigma_kms: float = 30.0,
    start_bjd: float = _DR4_START_BJD,
) -> list[BinarySystem]:
    """Draw detached eclipsing binaries, stratified in G_RVS and in temperature class.

    The sample has ``n_per_cell`` systems in every cell of a grid of G_RVS bins by classes
    of the primary's effective temperature. Within a class the systems follow the model
    below with no further selection, so the cell is a realistic mixture of mass ratios,
    periods, radii and inclinations. G_RVS is assigned uniformly within the bin and is not
    derived from a distance: the S/N of a transit depends on nothing else. A catalogue
    enters afterwards as weights (:func:`catalogue_weights`).

    **The model.** Each row is a published prescription.

    - *Primary.* The mass follows the initial mass function ``dN/dm ~ m^-2.3`` and the age
      is uniform up to ``max_age_gyr``, a constant star-formation rate. The star is kept
      where its track (:class:`StellarTracks`) puts it in the class, on or beyond the
      zero-age main sequence, with ``log g >= logg_min``. A massive star is then listed
      for the time it lives and a subgiant for the time it takes to cross the class.
    - *Companion.* The period follows the companion frequency per decade of Moe & Di
      Stefano (2017) and the mass ratio their broken power law, from ``q_min``, so that
      single-lined systems are present. Their excess of twins is taken at ``twin_excess``
      of its published fraction and spread from ``twin_from`` to one (see Notes). The
      secondary has the age of the primary and its own track.
    - *Eccentricity.* The share of orbits with ``e > 0.1`` rises with the period as in
      the eclipsing samples, earlier for primaries hotter than 7000 K: none below 1.5 d
      (1.2 d for hot primaries), 6 and 20 percent at 4.2 d, 33 and 35 percent at 13.4 d
      (Torres et al. 2010; Van Eylen et al. 2016; IJspeert et al. 2024). Those
      eccentricities are uniform up to the envelope of Mazeh (2008). The distribution of
      Moe & Di Stefano, which :func:`albireo.population.draw_population` uses, puts 3 and 7
      percent above 0.1 at 3.2 to 5.6 d and 5.6 to 10 d, where the eclipsing samples have
      10 and 17.
    - *Detached.* Each radius below 0.75 of its Roche lobe (:func:`roche_radius`) at
      periastron.
    - *Eclipses.* The orientation is isotropic and the system is kept where it eclipses,
      which lists it in proportion to its eclipse probability. The deeper eclipse must be
      ``min_depth_mag`` deep in G for limb-darkened spheres, the depth at which the Gaia
      DR3 candidates set in (Mowlavi et al. 2023), and the system is kept with the
      probability that three of ``n_photometric_transits`` transits fall in an eclipse of
      that depth, a Poisson count of mean ``N D / P`` for the duration ``D``.
    - *Brightness.* The system is kept in proportion to the volume its combined light in
      the band reaches, ``(F_1 + F_2)^(3/2)``, with ``F = R^2 C`` and ``C`` the library
      continuum (:class:`LibrarySet`).
    - *Rotation.* Aligned with the orbit. Pseudo-synchronous
      (:func:`pseudo_synchronous_ratio`) below 10 d for nine tenths of the stars cooler than
      6250 K and for hotter stars with ``R / a > 0.1``, for half of the systems from 10 to
      30 d, and that of a field star otherwise (Lurie et al. 2017).
    - *Sky and cadence.* The ecliptic latitude follows the Gaia DR3 candidates
      (:data:`ECLIPTIC_LATITUDE_SHARES`). The transit count follows the ``rv_nb_transits``
      of those candidates (:data:`EB_TRANSITS_DR3`), scaled to the 2005 d of DR4, and is
      drawn independently of the latitude.
    - *Composition and velocity.* ``[M/H]`` is normal with the mean and dispersion ``mh``,
      within the library, and applies to the spectra only: the tracks are of solar
      composition. The systemic velocity is normal with ``gamma_sigma_kms``.

    Two further draws are recorded in ``meta`` and used only by an analysis that asks for
    them. ``tertiary``: a third star with probability ``0.30 + 0.80 exp(-P / 7 d)``, at
    most 0.96, of mass uniform between 0.1 and 1.2 of the primary's and of the same age,
    unresolved in 68 percent of cases (Tokovinin et al. 2006). ``activity``: for a star
    cooler than 6500 K with a Rossby number below 0.13 (Wright et al. 2011), the excess
    equivalent width of the emission in the core of Ca II 8542, uniform between 0.3 and
    1.0 Angstrom.

    Parameters
    ----------
    n_per_cell
        Systems per cell.
    seed
        Seed. A class has its own generator, so the systems of one class do not depend on
        the other classes or on ``n_per_cell`` of another call beyond their number.
    libraries
        The :class:`LibrarySet` of the spectra. Default: the three BOSZ boxes of the
        registry, which must be in the cache or are downloaded.
    tracks
        The :class:`StellarTracks`. Default: the packaged table.
    classes
        ``{name: (teff_low, teff_high)}``; default :data:`TEMPERATURE_CLASSES`.
    grvs_edges
        Edges of the G_RVS bins; default :data:`GRVS_EDGES`.
    period_range
        Periods in days.
    q_min
        Smallest mass ratio, at least 0.1.
    twin_excess, twin_from, twin_shape
        The excess of twins: the factor on the published excess fraction, the mass ratio
        it starts at, and its form, ``"ramp"`` (rising linearly from zero) or
        ``"uniform"``. ``(1.0, 0.95, "uniform")`` is the published form.
    logg_min
        Smallest surface gravity of the primary, which bounds its evolution.
    max_age_gyr
        Largest age.
    min_depth_mag, n_photometric_transits
        The two conditions of the photometric detection.
    mh
        Mean and dispersion of the metallicity.
    gamma_sigma_kms
        Dispersion of the systemic velocity.
    start_bjd
        Start of the window in which the periastron time is drawn.

    Notes
    -----
    Moe & Di Stefano place an excess of twins on ``0.95 < q < 1`` that holds 0.30 of the
    companions above ``q = 0.3`` of a solar-type primary below 10 d. A magnitude-limited
    eclipsing sample multiplies it: a twin is seen to 2.8 times the volume of its primary
    alone and eclipses for a wider range of inclinations. With the published excess, 53
    percent of a drawn sample has ``q > 0.95``, and 67 percent of its systems with a flux
    ratio above 0.1. Two samples of eclipsing binaries disagree. Of the 293 detached
    double-lined systems of Eker et al. (2018), 30 percent have ``q > 0.95``, 49 percent
    ``q > 0.9`` and 83 percent ``q > 0.7``, with counts that rise from ``q = 0.85``. Of the
    Gaia DR3 candidates of the detached light-curve classes brighter than G_RVS = 12 whose
    primary eclipse is 0.2 mag deep, 14 percent have the two depths in the same bin of
    0.1 mag, and at most 28 percent if every single-eclipse model is a twin at half its
    period; the drawn sample has 45 percent. With 0.3 of the published excess, rising
    linearly from ``q = 0.85``, a sample of 7,800 systems weighted to the catalogue
    has 31, 50 and 86 percent above 0.95, 0.9 and 0.7 among its systems with
    a flux ratio above 0.1, and 32 percent of equal depths among those with an
    eclipse of 0.2 mag. That is the default. It corresponds to
    an excess of 0.09, near the 0.11 +/- 0.04 that Moe & Di Stefano give for early-type
    primaries.

    Returns
    -------
    list of BinarySystem
        ``n_per_cell`` systems per cell, ordered by class and, within a class, cycling
        through the G_RVS bins, so that the first ``k`` systems of every cell are a random
        subset of the class. The name is ``<class>-<bin>-<index>``. ``meta`` holds the
        class, the bin, the age, the evolutionary points, the fractional radii, the
        depths and durations of the two eclipses, the limb-darkening coefficients, the
        rotation periods and their kind, whether a label was moved into its library box,
        the tertiary, the activity and the simulation seed of the system.
    """
    if n_per_cell < 1:
        raise ValueError("n_per_cell must be positive")
    if not 0.1 <= q_min <= 1.0:
        raise ValueError("q_min must lie in [0.1, 1]")
    libraries = LibrarySet.fetch() if libraries is None else libraries
    tracks = StellarTracks.load() if tracks is None else tracks
    classes = dict(TEMPERATURE_CLASSES if classes is None else classes)
    edges = [float(x) for x in grvs_edges]
    n_bins = len(edges) - 1
    config = {
        "logg_min": float(logg_min),
        "max_age_yr": float(max_age_gyr) * 1e9,
        "min_depth_mag": float(min_depth_mag),
        "n_photometric_transits": float(n_photometric_transits),
        "mh": (float(mh[0]), float(mh[1])),
    }
    if twin_shape not in ("uniform", "ramp"):
        raise ValueError(f"twin_shape must be 'uniform' or 'ramp'; got {twin_shape!r}")
    companions = _CompanionDistributions(
        (max(0.3, float(tracks.mass[0])), float(tracks.mass[-1])),
        period_range,
        float(q_min),
        (float(twin_excess), float(twin_from), twin_shape),
    )
    n_class = n_per_cell * n_bins
    seeds = np.random.SeedSequence(seed).spawn(len(classes) * n_class)
    systems: list[BinarySystem] = []
    for c, (name, teff_range) in enumerate(classes.items()):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), c]))
        drawn = _draw_class(rng, n_class, teff_range, config, tracks, companions, libraries)
        extra = _finish_class(rng, drawn, tracks, libraries, gamma_sigma_kms, start_bjd)
        for k in range(n_class):
            cell = k % n_bins
            grvs = float(rng.uniform(edges[cell], edges[cell + 1]))
            system_seed = int(seeds[c * n_class + k].generate_state(1)[0] % (2**31 - 1))
            systems.append(
                _system(
                    f"{name}-{cell:02d}-{k // n_bins:03d}",
                    name,
                    cell,
                    grvs,
                    system_seed,
                    {key: value[k] for key, value in drawn.items() if np.ndim(value) == 1},
                    {key: value[k] for key, value in extra.items()},
                )
            )
    return systems


def _finish_class(rng, drawn, tracks, libraries, gamma_sigma_kms, start_bjd):
    """The quantities of the accepted systems that do not enter the selection."""
    n = drawn["m1"].size
    period, ecc, a = drawn["period"], drawn["ecc"], drawn["a"]
    sin_i = np.sqrt(1.0 - drawn["cos_i"] ** 2)

    # Rotation.
    system_draw = rng.uniform(size=n)
    out: dict[str, np.ndarray] = {}
    for label in ("1", "2"):
        teff, radius = drawn["teff" + label], drawn["radius" + label]
        cool = teff < _KRAFT_TEFF
        short = np.where(cool, rng.uniform(size=n) < 0.9, radius / a > 0.1)
        tidal = np.where(period < 10.0, short, (period < 30.0) & (system_draw < 0.5))
        synchronous = _SYNCHRONOUS_KMS * radius / period * pseudo_synchronous_ratio(ecc)
        v_eq = np.where(tidal, synchronous, _field_rotation(rng, teff, radius))
        out["vsini" + label] = v_eq * sin_i
        out["p_rot" + label] = _SYNCHRONOUS_KMS * radius / v_eq
        out["tidal" + label] = tidal
        # Emission in the cores of the Ca II triplet, for the activity perturbation.
        mass = drawn["m" + label]
        saturated = (teff < _ACTIVITY_TEFF) & (
            out["p_rot" + label] / _turnover_days(mass) < _SATURATION_ROSSBY
        )
        out["activity" + label] = np.where(saturated, rng.uniform(0.3, 1.0, size=n), 0.0)
        _, _, _, _, moved = libraries.clamp(teff, drawn["logg" + label], drawn["mh"])
        out["moved" + label] = moved

    # The tertiary, for the third-light perturbation.
    present = rng.uniform(size=n) < np.minimum(0.30 + 0.80 * np.exp(-period / 7.0), 0.96)
    unresolved = rng.uniform(size=n) < _UNRESOLVED_TERTIARY
    m3 = np.clip(rng.uniform(0.1, 1.2, size=n) * drawn["m1"], tracks.mass[0], tracks.mass[-1])
    star3 = tracks.at(m3, drawn["age"])
    with np.errstate(invalid="ignore"):
        usable = np.isfinite(star3["teff"]) & (star3["logg"] >= 3.5)
    teff3 = np.nan_to_num(star3["teff"], nan=5000.0)
    logg3 = np.nan_to_num(star3["logg"], nan=4.5)
    radius3 = np.nan_to_num(star3["radius"], nan=1.0)
    cont = [
        libraries.log_continuum(drawn["teff1"], drawn["logg1"], drawn["mh"]),
        libraries.log_continuum(drawn["teff2"], drawn["logg2"], drawn["mh"]),
        libraries.log_continuum(teff3, logg3, drawn["mh"]),
    ]
    light = [
        drawn["radius1"] ** 2 * np.exp(cont[0]),
        drawn["radius2"] ** 2 * np.exp(cont[1]),
        radius3**2 * np.exp(cont[2]),
    ]
    out["tertiary"] = present & unresolved & usable
    out["tertiary_present"] = present
    out["m3"], out["teff3"], out["logg3"], out["radius3"] = m3, teff3, logg3, radius3
    out["light3"] = light[2] / (light[0] + light[1] + light[2])
    spin = np.sqrt(1.0 - rng.uniform(size=n) ** 2)
    out["vsini3"] = _field_rotation(rng, teff3, radius3) * spin
    out["dv3"] = rng.normal(0.0, 3.0, size=n)

    # The colour of the pair in G - G_RP, from the dwarf sequence at each temperature.
    flux_g = [
        10.0 ** (-0.4 * _absolute_g(drawn["teff" + s], drawn["log_l" + s])) for s in ("1", "2")
    ]
    flux_rp = [
        f * 10.0 ** (0.4 * _g_minus_rp(drawn["teff" + s]))
        for f, s in zip(flux_g, ("1", "2"), strict=True)
    ]
    out["g_rp"] = 2.5 * np.log10((flux_rp[0] + flux_rp[1]) / (flux_g[0] + flux_g[1]))

    # Sky, cadence, phase and systemic velocity.
    shares = np.asarray(ECLIPTIC_LATITUDE_SHARES) / np.sum(ECLIPTIC_LATITUDE_SHARES)
    band = rng.choice(shares.size, size=n, p=shares)
    latitude = (10.0 * band + rng.uniform(0.0, 10.0, size=n)) * rng.choice([-1.0, 1.0], size=n)
    out["ecl_lat"] = latitude
    out["ecl_lon"] = rng.uniform(0.0, 360.0, size=n)
    # The transit count of the catalogue's own eclipsing binaries, a split log-normal
    # through its three quantiles, scaled from the 1038 d of DR3 to the span of DR4.
    low, median, high = EB_TRANSITS_DR3
    deviate = rng.normal(size=n)
    width = np.where(deviate > 0.0, math.log(high / median), math.log(median / low)) / 1.2816
    counts = median * np.exp(width * deviate) * _DR4_SPAN_DAYS / 1038.0
    out["n_transits"] = np.maximum(np.rint(counts).astype(int), 2)
    out["t_peri"] = start_bjd + rng.uniform(size=n) * period
    out["gamma"] = rng.normal(0.0, gamma_sigma_kms, size=n)
    return out


def _system(name, class_name, cell, grvs, seed, d, x) -> BinarySystem:
    incl = math.acos(float(d["cos_i"]))
    k1, k2 = semi_amplitudes(d["m1"], d["m2"], d["period"], d["ecc"], incl)
    g_rp = float(x["g_rp"])
    blue = -0.0397 - 0.2852 * g_rp - 0.0330 * g_rp**2 - 0.0867 * g_rp**3
    red = -4.0618 + 10.0187 * g_rp - 9.0532 * g_rp**2 + 2.6089 * g_rp**3
    g_mag = grvs + g_rp - (blue if g_rp <= 1.2 else red)  # the inverse of predict_grvs
    ra, dec = ecliptic_to_icrs(float(x["ecl_lon"]), float(x["ecl_lat"]))
    tertiary = None
    if bool(x["tertiary"]):
        tertiary = {
            "m": float(x["m3"]),
            "teff": float(x["teff3"]),
            "logg": float(x["logg3"]),
            "radius": float(x["radius3"]),
            "vsini": float(x["vsini3"]),
            "light": float(x["light3"]),
            "dv": float(x["dv3"]),
        }
    meta = {
        "class": class_name,
        "grvs_bin": int(cell),
        "seed": int(seed),
        "age_gyr": float(d["age"]) / 1e9,
        "eep": [float(d["eep1"]), float(d["eep2"])],
        "log_l": [float(d["log_l1"]), float(d["log_l2"])],
        "rho": float((d["radius1"] + d["radius2"]) / d["a"]),
        "depth_mag": [float(d["depth1"]), float(d["depth2"])],
        "duration_days": [float(d["duration1"]), float(d["duration2"])],
        "eclipse_probability": float(d["p_ecl"]),
        "limb_darkening": [
            float(limb_darkening(d["teff1"], "rvs")),
            float(limb_darkening(d["teff2"], "rvs")),
        ],
        "p_rot_days": [float(x["p_rot1"]), float(x["p_rot2"])],
        "tidal_rotation": [bool(x["tidal1"]), bool(x["tidal2"])],
        "moved_into_box": [bool(x["moved1"]), bool(x["moved2"])],
        "activity_ew": [float(x["activity1"]), float(x["activity2"])],
        "tertiary": tertiary,
        "tertiary_present": bool(x["tertiary_present"]),
        "ecl_lon": float(x["ecl_lon"]),
    }
    return BinarySystem(
        name=name,
        m1=float(d["m1"]),
        m2=float(d["m2"]),
        r1=float(d["radius1"]),
        r2=float(d["radius2"]),
        teff1=float(d["teff1"]),
        teff2=float(d["teff2"]),
        logg1=float(d["logg1"]),
        logg2=float(d["logg2"]),
        mh=float(d["mh"]),
        vsini1=float(x["vsini1"]),
        vsini2=float(x["vsini2"]),
        period=float(d["period"]),
        ecc=float(d["ecc"]),
        omega=float(d["omega"]),
        t_peri=float(x["t_peri"]),
        incl=incl,
        gamma=float(x["gamma"]),
        k1=float(k1),
        k2=float(k2),
        eclipsing=True,
        light_ratio=float(d["light_ratio"]),
        g_mag=float(g_mag),
        g_rp=g_rp,
        grvs=float(grvs),
        ecl_lat_deg=float(x["ecl_lat"]),
        n_transits=int(x["n_transits"]),
        ra_deg=float(ra),
        dec_deg=float(dec),
        source="eclipsing-design",
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Columns and catalogue weights
# ---------------------------------------------------------------------------


def population_columns(systems: Sequence[BinarySystem]) -> dict[str, np.ndarray]:
    """The systems as arrays, one entry per system, for tables and figures.

    The fields of the record, and from ``meta``: ``klass``, ``grvs_bin``, ``age_gyr``,
    ``eep1``, ``rho``, the depth of the deeper eclipse ``depth_mag``, ``tertiary_light``
    (zero without an unresolved tertiary) and ``weight`` (one where
    :func:`catalogue_weights` has not been applied).
    """
    fields = (
        "m1 m2 r1 r2 teff1 teff2 logg1 logg2 mh vsini1 vsini2 period ecc omega incl gamma "
        "k1 k2 light_ratio g_mag g_rp grvs ecl_lat_deg n_transits"
    ).split()
    out: dict[str, np.ndarray] = {
        name: np.array([getattr(s, name) for s in systems], dtype=np.float64) for name in fields
    }
    out["name"] = np.array([s.name for s in systems])
    out["klass"] = np.array([s.meta.get("class", "") for s in systems])
    out["grvs_bin"] = np.array([s.meta.get("grvs_bin", -1) for s in systems], dtype=int)
    for key in ("age_gyr", "rho", "weight", "weight_cell"):
        default = 1.0 if key.startswith("weight") else np.nan
        out[key] = np.array([s.meta.get(key, default) for s in systems], dtype=np.float64)
    out["eep1"] = np.array([s.meta.get("eep", [np.nan])[0] for s in systems], dtype=np.float64)
    out["depth_mag"] = np.array(
        [max(s.meta.get("depth_mag", [np.nan, np.nan])) for s in systems], dtype=np.float64
    )
    out["tertiary_light"] = np.array(
        [(s.meta.get("tertiary") or {}).get("light", 0.0) for s in systems], dtype=np.float64
    )
    out["q"] = out["m2"] / out["m1"]
    return out


def catalogue_weights(
    systems: Sequence[BinarySystem],
    counts: Mapping[tuple[int, str, int], float],
    *,
    period_bin_dex: float = 0.2,
    max_factor: float = 4.0,
    min_systems: int = 20,
    iterations: int = 50,
) -> tuple[list[BinarySystem], dict[str, Any]]:
    """Weights that map a design sample onto the counts of a catalogue.

    The design sample has the same number of systems in every cell, and a catalogue does
    not. Each system receives two weights in its ``meta``:

    - ``weight_cell``: the catalogue count of its cell (G_RVS bin and temperature class)
      divided by the number of simulated systems in the cell. The weighted sample has the
      magnitudes and classes of the catalogue and the periods of the model.
    - ``weight``: the same, raked to the catalogue's periods. The weights are multiplied
      in turn by the factors that make the weighted counts equal the catalogue's in every
      bin of ``period_bin_dex`` in log period of every class and in every cell, until they
      settle. A system's factor relative to its weight by cell, scaled by the share of
      its class that the raking covers, is held within ``1 / max_factor`` and
      ``max_factor`` before the totals of the cells are restored.

    A period bin of a class is raked only where the sample has ``min_systems`` systems in
    it. The catalogue entries of the other bins are outside the support of the model: a
    catalogue of light-curve classes lists near-contact pairs at periods at which no
    detached pair of main-sequence stars exists. Their share is reported, and the raked
    weights of a class sum to its supported count. A system of a cell that the catalogue
    does not list has no weight of either kind.

    Parameters
    ----------
    systems
        A design sample of :func:`draw_eclipsing_population`.
    counts
        ``{(grvs_bin, class, period_bin): count}`` with ``period_bin =
        floor(log10(P / d) / period_bin_dex)``.
    period_bin_dex, max_factor, min_systems, iterations
        The raking.

    Returns
    -------
    (list of BinarySystem, dict)
        The systems with the two weights, and a summary: the catalogue total, the share
        outside the supported period bins, the largest relative departure of a raked
        period margin from the catalogue's, and the effective sample size
        ``(sum w)^2 / sum w^2`` of each weight.
    """
    klass = np.array([s.meta["class"] for s in systems])
    mbin = np.array([int(s.meta["grvs_bin"]) for s in systems])
    pbin = np.array([math.floor(math.log10(s.period) / period_bin_dex) for s in systems])
    cell_total: dict[tuple[int, str], float] = {}
    period_total: dict[tuple[str, int], float] = {}
    for (m, c, p), value in counts.items():
        cell_total[(int(m), c)] = cell_total.get((int(m), c), 0.0) + float(value)
        period_total[(c, int(p))] = period_total.get((c, int(p)), 0.0) + float(value)
    weight_cell = np.zeros(len(systems))
    for (m, c), total in cell_total.items():
        sel = (mbin == m) & (klass == c)
        if sel.any():
            weight_cell[sel] = total / sel.sum()
    catalogue_total = float(sum(cell_total.values()))

    supported = {
        key: total
        for key, total in period_total.items()
        if int(np.sum((klass == key[0]) & (pbin == key[1]))) >= int(min_systems)
    }
    in_support = np.array(
        [(c, int(p)) in supported for c, p in zip(klass, pbin, strict=True)], dtype=bool
    )
    outside = 1.0 - sum(supported.values()) / max(sum(period_total.values()), 1e-300)
    class_share = {
        c: sum(v for (cc, _), v in supported.items() if cc == c)
        / max(sum(v for (cc, _), v in period_total.items() if cc == c), 1e-300)
        for c in set(klass)
    }
    # The weight of a system to which no period factor applies: its weight by cell times
    # the share of its class that the supported bins hold. The bound is taken about it.
    listed = in_support & (weight_cell > 0.0)
    neutral = weight_cell * np.array([class_share[c] for c in klass])
    weight = np.where(listed, neutral, 0.0)
    departure = np.inf
    for _ in range(int(iterations)):
        departure = 0.0
        for (c, p), target in supported.items():
            sel = (klass == c) & (pbin == p)
            have = weight[sel].sum()
            if have > 0.0:
                weight[sel] *= target / have
                departure = max(departure, abs(target / have - 1.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            factor = np.where(listed, weight / np.where(listed, neutral, 1.0), 0.0)
        weight = np.where(listed, neutral * np.clip(factor, 1.0 / max_factor, max_factor), 0.0)
        for (m, c), total in cell_total.items():
            sel = (mbin == m) & (klass == c)
            have = weight[sel].sum()
            if have > 0.0:
                weight[sel] *= total * class_share[c] / have
        if departure < 1e-3:
            break
    margin = 0.0
    for (c, p), target in supported.items():
        have = weight[(klass == c) & (pbin == p)].sum()
        margin = max(margin, abs(have / target - 1.0))

    def effective(w):
        return float(w.sum() ** 2 / np.sum(w**2)) if np.any(w > 0.0) else 0.0

    out = []
    for s, w_cell, w in zip(systems, weight_cell, weight, strict=True):
        meta = {**s.meta, "weight_cell": float(w_cell), "weight": float(w)}
        out.append(dataclasses.replace(s, meta=meta))
    summary = {
        "catalogue_total": catalogue_total,
        "share_outside_period_support": float(outside),
        "largest_period_margin_departure": float(margin),
        "effective_size_cell": effective(weight_cell),
        "effective_size_raked": effective(weight),
    }
    return out, summary

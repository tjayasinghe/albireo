"""Populations of double-lined binaries: the truths a Gaia RVS benchmark is drawn from.

**Experimental.** The record and the draw parameters may change.

A :class:`BinarySystem` is one complete truth: two main-sequence stars with masses, radii,
temperatures, gravities and rotation; a Keplerian orbit with its inclination; the
semi-amplitudes that follow; whether the pair eclipses; the light ratio in the RVS band;
the system's Gaia photometry and G_RVS; its sky position, its ecliptic latitude and the
number of RVS transits Gaia records for it. :func:`draw_population` builds such systems from the
empirical distributions of the field, and the catalogue adapters build them from real
binaries (Gaia DR3 double-lined orbits, DEBCat), filling only what the catalogue lacks.

The parametric arm composes published prescriptions with no free choices left. Primary
masses follow a Salpeter power law over the requested range. Periods are weighted by the
companion frequency per decade of Moe & Di Stefano (2017, their equations 20-23), mass
ratios by their broken power law with the twin excess (equations 5-16), and
eccentricities by their ``p(e) ~ e^eta`` with the tidal ceiling ``e_max(P)`` (equations 3,
17 and 18); orbits with periods of two days or less are circular. Mass maps to the other
stellar quantities through Pecaut & Mamajek's (2013) mean dwarf sequence, a table carried
here, or through the mass-luminosity, mass-radius and mass-temperature relations of Eker
et al. (2018), which are calibrated on detached eclipsing binaries. Eclipses follow from
the geometry with the eccentric-orbit factor (Winn 2010), rotation is synchronised for
periods below 15 days (Meibom, Mathieu & Stassun 2006), and the RVS light ratio is the
ratio of the library continua at the two stars' labels scaled by the radius ratio.

References
----------
Eker, Z., Bakis, V., Bilir, S., et al. 2018, MNRAS, 479, 5491
Gaia Collaboration, Arenou, F., Babusiaux, C., et al. 2023, A&A, 674, A34
Meibom, S., Mathieu, R. D. & Stassun, K. G. 2006, ApJ, 653, 621
Moe, M. & Di Stefano, R. 2017, ApJS, 230, 15
Pecaut, M. J. & Mamajek, E. E. 2013, ApJS, 208, 9
Southworth, J. 2015, ASP Conf. Ser., 496, 164
Winn, J. N. 2010, in Exoplanets, ed. S. Seager (Tucson: University of Arizona Press), 55
"""

from __future__ import annotations

import dataclasses
import json
import math
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from albireo.gaia import RVS_BAND, predict_grvs, rvs_transit_count
from albireo.kepler import t_conj_from_t_peri
from albireo.simulate import OrbitParams

__all__ = [
    "BinarySystem",
    "MainSequence",
    "draw_population",
    "eclipse_probability",
    "ecliptic_to_icrs",
    "from_debcat",
    "from_gaia_sb2",
    "population_summary",
    "query_gaia_sb2",
    "read_population",
    "rvs_light_ratio",
    "semi_amplitudes",
    "semimajor_axis_rsun",
    "write_population",
]

_G_SI = 6.67430e-11
_MSUN_KG = 1.988409870698051e30
_RSUN_M = 6.957e8
_DAY_S = 86400.0
_KEPLER_A_RSUN = 4.208  # a / Rsun = 4.208 (M / Msun)^(1/3) (P / d)^(2/3), IAU 2015 constants
_SYNCHRONOUS_KMS = 2.0 * math.pi * _RSUN_M / _DAY_S / 1e3  # 50.593 km/s per (R/Rsun)/(P/d)
_LOGG_SUN = 4.438  # log10 of the solar surface gravity in cgs
_TEFF_SUN = 5772.0


# ---------------------------------------------------------------------------
# Kepler's law and the geometry
# ---------------------------------------------------------------------------


def semimajor_axis_rsun(m_total_msun, period_days):
    """Semi-major axis in solar radii from Kepler's third law."""
    return (
        _KEPLER_A_RSUN
        * np.cbrt(np.asarray(m_total_msun, dtype=np.float64))
        * np.asarray(period_days, dtype=np.float64) ** (2.0 / 3.0)
    )


def semi_amplitudes(m1, m2, period, ecc, incl):
    """``(K1, K2)`` in km/s from the masses, the period, the eccentricity and the inclination.

    ``K_1 = (2 pi G / P)^(1/3) m_2 sin i / ((m_1 + m_2)^(2/3) sqrt(1 - e^2))`` and the
    same with the masses exchanged, as the reference implementation computes them; the
    notebook's 1.30 + 1.20 Msun pair at P = 4 d, e = 0.1, i = 87 degrees gives 87.7 and
    95.0 km/s.

    Parameters
    ----------
    m1, m2
        Masses in solar masses.
    period
        Orbital period in days.
    ecc
        Eccentricity.
    incl
        Inclination in radians.
    """
    m1 = np.asarray(m1, dtype=np.float64) * _MSUN_KG
    m2 = np.asarray(m2, dtype=np.float64) * _MSUN_KG
    total = m1 + m2
    coeff = (2.0 * np.pi * _G_SI / (np.asarray(period, dtype=np.float64) * _DAY_S)) ** (1.0 / 3.0)
    coeff = coeff / np.sqrt(1.0 - np.asarray(ecc, dtype=np.float64) ** 2) * np.sin(incl)
    k1 = coeff * m2 / total ** (2.0 / 3.0) / 1e3
    k2 = coeff * m1 / total ** (2.0 / 3.0) / 1e3
    return k1, k2


_OBLIQUITY_J2000_DEG = 23.4392911
"""Mean obliquity of the ecliptic at J2000.0 in degrees, 84381.448 arcsec (IAU 1976; the
IAU 2006 value is 84381.406 arcsec, 42 mas below it)."""


def ecliptic_to_icrs(lon_deg, lat_deg):
    """ICRS right ascension and declination from ecliptic longitude and latitude.

    One rotation about the equinox by the J2000 obliquity ``eps = 23.4392911`` degrees,
    written out here rather than delegated to a coordinates package because albireo takes
    no such dependency. With ``lam`` the longitude and ``beta`` the latitude,

    ``x = cos(beta) cos(lam)``, ``y = cos(beta) sin(lam)``, ``z = sin(beta)``;

    ``x' = x``, ``y' = y cos(eps) - z sin(eps)``, ``z' = y sin(eps) + z cos(eps)``;

    ``alpha = atan2(y', x') mod 360``, ``delta = asin(z')``.

    The frame is the mean equinox and ecliptic of J2000. Measured over 500 positions
    against astropy 8.0, the result sits 0.04 arcsec from ``BarycentricMeanEcliptic``,
    which is the same frame and leaves only the ICRS frame bias and the rounding of the
    obliquity; 14 arcsec at most from ``BarycentricTrueEcliptic``, which adds the nutation
    of the equinox and of the obliquity; and 35 arcsec at most from
    ``GeocentricTrueEcliptic``, which places the origin at the Earth. All three are under
    an arcminute, and a sky position enters this module only through the ecliptic latitude
    that sets the transit count and through the position a transit forecast is requested
    for, where an arcminute is immaterial.

    Parameters
    ----------
    lon_deg, lat_deg
        Ecliptic longitude and latitude in degrees; scalars or arrays.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Right ascension in [0, 360) and declination in [-90, 90], in degrees.

    Examples
    --------
    >>> ra, dec = ecliptic_to_icrs(0.0, 90.0)  # the north ecliptic pole
    >>> float(np.round(ra, 3)), float(np.round(dec, 3))
    (270.0, 66.561)
    """
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    eps = math.radians(_OBLIQUITY_J2000_DEG)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    y_eq = y * math.cos(eps) - z * math.sin(eps)
    z_eq = y * math.sin(eps) + z * math.cos(eps)
    ra = np.degrees(np.arctan2(y_eq, x)) % 360.0
    dec = np.degrees(np.arcsin(np.clip(z_eq, -1.0, 1.0)))
    return ra, dec


def eclipse_probability(r1_rsun, r2_rsun, a_rsun, ecc=0.0, omega=0.0):
    """Largest ``cos i`` at which either star eclipses, i.e. the eclipse probability.

    For a circular orbit this is ``(R_1 + R_2) / a``, exact because ``cos i`` is uniform
    for isotropic orbit normals. An eccentric orbit brings the stars closer at one
    conjunction than the other: the primary eclipse takes the factor
    ``(1 + e sin omega) / (1 - e^2)`` and the secondary ``(1 - e sin omega) / (1 - e^2)``
    (Winn 2010, equations 9 and 10); the larger of the two is returned, which is the
    condition for the system to eclipse at all.
    """
    base = (np.asarray(r1_rsun, dtype=np.float64) + np.asarray(r2_rsun, dtype=np.float64)) / (
        np.asarray(a_rsun, dtype=np.float64)
    )
    e = np.asarray(ecc, dtype=np.float64)
    s = np.sin(np.asarray(omega, dtype=np.float64))
    return np.minimum(base * (1.0 + np.abs(e * s)) / (1.0 - e**2), 1.0)


# ---------------------------------------------------------------------------
# The main sequence
# ---------------------------------------------------------------------------

# Pecaut & Mamajek's mean dwarf sequence, version 2022.04.16 of the table at
# http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt, from B9V to M6V
# (the rows that carry Gaia colours). Columns: spectral type, Teff [K], log L/Lsun, R/Rsun,
# M_V, Bp-Rp, G-Rp, M_G, M/Msun. R_Rsun is back-computed there from M_bol and Teff, and Msun
# from a polynomial in M_V, so the table carries mass and two independent quantities.
_MAMAJEK_ROWS = (
    ("B9V", 10700, 1.86, 2.49, 0.50, -0.120, -0.036, 0.515, 2.75),
    ("B9.5V", 10400, 1.80, 2.45, 0.60, -0.087, -0.016, 0.615, 2.68),
    ("A0V", 9700, 1.58, 2.193, 0.99, -0.037, -0.020, 1.00, 2.18),
    ("A1V", 9300, 1.49, 2.136, 1.16, 0.005, 0.032, 1.16, 2.05),
    ("A2V", 8800, 1.38, 2.117, 1.35, 0.068, 0.050, 1.345, 1.98),
    ("A3V", 8600, 1.23, 1.861, 1.70, 0.110, 0.092, 1.69, 1.86),
    ("A4V", 8250, 1.13, 1.794, 1.94, 0.166, 0.113, 1.92, 1.93),
    ("A5V", 8100, 1.09, 1.785, 2.01, 0.194, 0.130, 1.98, 1.88),
    ("A6V", 7910, 1.05, 1.775, 2.12, 0.222, 0.152, 2.09, 1.83),
    ("A7V", 7760, 1.00, 1.750, 2.23, 0.263, 0.173, 2.19, 1.77),
    ("A8V", 7590, 0.96, 1.747, 2.32, 0.320, 0.204, 2.27, 1.81),
    ("A9V", 7400, 0.92, 1.747, 2.43, 0.327, 0.207, 2.37, 1.75),
    ("F0V", 7220, 0.86, 1.728, 2.57, 0.377, 0.230, 2.51, 1.61),
    ("F1V", 7020, 0.79, 1.679, 2.76, 0.434, 0.252, 2.69, 1.50),
    ("F2V", 6820, 0.71, 1.622, 2.97, 0.490, 0.279, 2.89, 1.46),
    ("F3V", 6750, 0.67, 1.578, 3.08, 0.518, 0.293, 2.99, 1.44),
    ("F4V", 6670, 0.62, 1.533, 3.20, 0.546, 0.307, 3.10, 1.38),
    ("F5V", 6550, 0.56, 1.473, 3.37, 0.587, 0.329, 3.26, 1.33),
    ("F6V", 6350, 0.43, 1.359, 3.69, 0.640, 0.356, 3.56, 1.25),
    ("F7V", 6280, 0.39, 1.324, 3.80, 0.670, 0.372, 3.66, 1.21),
    ("F8V", 6180, 0.29, 1.221, 4.05, 0.694, 0.385, 3.90, 1.18),
    ("F9V", 6050, 0.22, 1.167, 4.25, 0.719, 0.399, 4.105, 1.13),
    ("F9.5V", 5990, 0.18, 1.142, 4.35, 0.767, 0.431, 4.195, 1.08),
    ("G0V", 5930, 0.13, 1.100, 4.48, 0.784, 0.439, 4.325, 1.06),
    ("G1V", 5860, 0.08, 1.060, 4.62, 0.803, 0.448, 4.462, 1.03),
    ("G2V", 5770, 0.01, 1.012, 4.80, 0.823, 0.459, 4.635, 1.00),
    ("G3V", 5720, -0.01, 1.002, 4.87, 0.832, 0.464, 4.703, 0.99),
    ("G4V", 5680, -0.04, 0.991, 4.93, 0.841, 0.468, 4.757, 0.985),
    ("G5V", 5660, -0.05, 0.977, 4.98, 0.850, 0.473, 4.801, 0.98),
    ("G6V", 5600, -0.10, 0.949, 5.10, 0.869, 0.483, 4.914, 0.97),
    ("G7V", 5550, -0.13, 0.927, 5.20, 0.880, 0.489, 5.006, 0.95),
    ("G8V", 5480, -0.17, 0.914, 5.30, 0.900, 0.499, 5.098, 0.94),
    ("G9V", 5380, -0.26, 0.853, 5.55, 0.950, 0.524, 5.34, 0.90),
    ("K0V", 5270, -0.34, 0.813, 5.78, 0.983, 0.56, 5.553, 0.88),
    ("K1V", 5170, -0.39, 0.797, 5.95, 1.01, 0.56, 5.65, 0.86),
    ("K2V", 5100, -0.43, 0.783, 6.07, 1.10, 0.62, 5.83, 0.82),
    ("K3V", 4830, -0.55, 0.755, 6.50, 1.21, 0.66, 6.20, 0.78),
    ("K4V", 4600, -0.69, 0.713, 6.98, 1.34, 0.70, 6.53, 0.73),
    ("K5V", 4440, -0.76, 0.701, 7.28, 1.43, 0.74, 6.83, 0.70),
    ("K6V", 4300, -0.86, 0.669, 7.64, 1.53, 0.79, 7.02, 0.69),
    ("K7V", 4100, -1.00, 0.630, 8.16, 1.70, 0.86, 7.57, 0.64),
    ("K8V", 3990, -1.06, 0.615, 8.43, 1.73, 0.88, 7.74, 0.62),
    ("K9V", 3930, -1.10, 0.608, 8.56, 1.79, 0.90, 8.03, 0.59),
    ("M0V", 3850, -1.16, 0.588, 8.80, 1.84, 0.92, 8.16, 0.57),
    ("M0.5V", 3770, -1.27, 0.544, 9.20, 1.97, 0.97, 8.44, 0.54),
    ("M1V", 3660, -1.39, 0.501, 9.64, 2.09, 1.00, 8.82, 0.50),
    ("M1.5V", 3620, -1.44, 0.482, 9.85, 2.13, 1.02, 8.98, 0.47),
    ("M2V", 3560, -1.54, 0.446, 10.21, 2.23, 1.06, 9.29, 0.44),
    ("M2.5V", 3470, -1.64, 0.421, 10.61, 2.39, 1.10, 9.67, 0.40),
    ("M3V", 3430, -1.79, 0.361, 11.15, 2.50, 1.13, 10.05, 0.37),
    ("M3.5V", 3270, -2.03, 0.300, 12.10, 2.78, 1.19, 10.87, 0.27),
    ("M4V", 3210, -2.14, 0.274, 12.61, 2.94, 1.24, 11.21, 0.23),
    ("M4.5V", 3110, -2.40, 0.217, 13.58, 3.16, 1.28, 12.04, 0.184),
    ("M5V", 3060, -2.52, 0.196, 14.15, 3.35, 1.33, 12.45, 0.162),
    ("M5.5V", 2930, -2.79, 0.156, 15.30, 3.71, 1.38, 13.35, 0.123),
    ("M6V", 2810, -2.98, 0.137, 16.32, 4.16, 1.43, 14.26, 0.102),
)


def _isotonic_rows(rows) -> np.ndarray:
    """The table sorted by mass, keeping the rows on which Teff rises and M_G falls with mass.

    The published sequence is a mean relation and its mass column is not strictly
    monotonic between neighbouring subtypes (A3V/A4V, A7V/A8V), which would make an
    interpolation on mass double-valued; the few rows that step backwards are dropped.
    """
    table = np.array([row[1:] for row in rows], dtype=np.float64)
    order = np.argsort(table[:, 7])
    kept = []
    for index in order:
        row = table[index]
        if kept and not (row[7] > kept[-1][7] and row[0] > kept[-1][0] and row[6] < kept[-1][6]):
            continue
        kept.append(row)
    return np.array(kept)


_MAMAJEK_BY_MASS = _isotonic_rows(_MAMAJEK_ROWS)  # increasing in mass
_MAMAJEK_LOGM = np.log10(_MAMAJEK_BY_MASS[:, 7])
_MAMAJEK_LOGT = np.log10(_MAMAJEK_BY_MASS[:, 0])

# Eker et al. (2018) Table 4: log L = a log M + b over six mass domains, and Table 5: the
# mass-radius relation below 1.5 Msun and the mass-temperature relation above it; the third
# relation follows from L = 4 pi R^2 sigma T^4 in each range.
_EKER_MLR = (
    (0.179, 0.45, 2.028, -0.976),
    (0.45, 0.72, 4.572, -0.102),
    (0.72, 1.05, 5.743, -0.007),
    (1.05, 2.40, 4.329, 0.010),
    (2.40, 7.0, 3.967, 0.093),
    (7.0, 31.0, 2.865, 1.105),
)


def _eker_log_luminosity(mass: float) -> float:
    for lo, hi, a, b in _EKER_MLR:
        if lo < mass <= hi or (mass <= lo and lo == _EKER_MLR[0][0]):
            return a * math.log10(mass) + b
    lo, hi, a, b = _EKER_MLR[-1]
    return a * math.log10(mass) + b


@dataclasses.dataclass(frozen=True)
class MainSequence:
    """Mass to the other stellar quantities, along the main sequence.

    Parameters
    ----------
    relation
        ``"mamajek"`` (default): Pecaut & Mamajek's mean dwarf sequence, interpolated in
        log mass, which gives Teff, radius, log g and the Gaia photometry from one table.
        ``"eker"``: the mass-luminosity, mass-radius and mass-temperature relations of Eker
        et al. (2018), calibrated on detached eclipsing binaries, for Teff, radius and log
        g, with the photometry taken from the Mamajek table at the resulting Teff.

    Notes
    -----
    Both relations describe field dwarfs of solar composition and no age; a real detached
    binary of the same mass can sit above the sequence by the evolution of its primary,
    which is one reason the catalogue adapters take the radii and temperatures from the
    catalogue where they exist.
    """

    relation: str = "mamajek"

    def __post_init__(self) -> None:
        if self.relation not in ("mamajek", "eker"):
            raise ValueError(f"relation must be 'mamajek' or 'eker'; got {self.relation!r}")

    @property
    def mass_range(self) -> tuple[float, float]:
        """Masses the relation is tabulated for, in solar masses."""
        if self.relation == "mamajek":
            return float(10 ** _MAMAJEK_LOGM[0]), float(10 ** _MAMAJEK_LOGM[-1])
        return 0.179, 31.0

    def teff(self, mass) -> np.ndarray:
        """Effective temperature in K."""
        m = np.asarray(mass, dtype=np.float64)
        if self.relation == "mamajek":
            return 10 ** np.interp(np.log10(m), _MAMAJEK_LOGM, _MAMAJEK_LOGT)
        return np.vectorize(self._eker_teff)(m)

    def radius(self, mass) -> np.ndarray:
        """Radius in solar radii."""
        m = np.asarray(mass, dtype=np.float64)
        if self.relation == "mamajek":
            return np.interp(np.log10(m), _MAMAJEK_LOGM, _MAMAJEK_BY_MASS[:, 2])
        return np.vectorize(self._eker_radius)(m)

    def logg(self, mass) -> np.ndarray:
        """Surface gravity, log10 of cm/s^2."""
        m = np.asarray(mass, dtype=np.float64)
        return _LOGG_SUN + np.log10(m) - 2.0 * np.log10(self.radius(m))

    def absolute_g(self, mass) -> np.ndarray:
        """Absolute Gaia G magnitude, from the Mamajek table."""
        return self._photometry(mass, 6)

    def g_minus_rp(self, mass) -> np.ndarray:
        """Gaia G - G_RP colour, from the Mamajek table."""
        return self._photometry(mass, 5)

    def bp_minus_rp(self, mass) -> np.ndarray:
        """Gaia G_BP - G_RP colour, from the Mamajek table."""
        return self._photometry(mass, 4)

    def mass_from_teff(self, teff) -> np.ndarray:
        """The mass whose sequence temperature is ``teff``: the inverse of :meth:`teff`."""
        t = np.log10(np.asarray(teff, dtype=np.float64))
        if self.relation == "mamajek":
            return 10 ** np.interp(t, _MAMAJEK_LOGT, _MAMAJEK_LOGM)
        masses = np.geomspace(0.18, 31.0, 600)
        return 10 ** np.interp(t, np.log10(self.teff(masses)), np.log10(masses))

    def _photometry(self, mass, column: int) -> np.ndarray:
        m = np.asarray(mass, dtype=np.float64)
        if self.relation == "mamajek":
            return np.interp(np.log10(m), _MAMAJEK_LOGM, _MAMAJEK_BY_MASS[:, column])
        t = np.log10(self.teff(m))
        return np.interp(t, _MAMAJEK_LOGT, _MAMAJEK_BY_MASS[:, column])

    @staticmethod
    def _eker_radius(mass: float) -> float:
        if mass < 1.5:
            return 0.438 * mass**2 + 0.479 * mass + 0.075
        log_l = _eker_log_luminosity(mass)
        log_t = -0.170 * math.log10(mass) ** 2 + 0.888 * math.log10(mass) + 3.671
        return math.sqrt(10**log_l) * (_TEFF_SUN / 10**log_t) ** 2

    @staticmethod
    def _eker_teff(mass: float) -> float:
        if mass < 1.5:
            log_l = _eker_log_luminosity(mass)
            radius = MainSequence._eker_radius(mass)
            return _TEFF_SUN * (10**log_l / radius**2) ** 0.25
        return 10 ** (-0.170 * math.log10(mass) ** 2 + 0.888 * math.log10(mass) + 3.671)


# ---------------------------------------------------------------------------
# Moe & Di Stefano (2017): periods, mass ratios, eccentricities
# ---------------------------------------------------------------------------


def _f_logp(m1: float, logp: float) -> float:
    """Companion frequency per decade of period with q > 0.3 (MDS17 equations 20-23)."""
    lm = math.log10(m1)
    a = 0.020 + 0.04 * lm + 0.07 * lm**2  # log P < 1
    b = 0.039 + 0.07 * lm + 0.01 * lm**2  # log P = 2.7
    c = 0.078 - 0.05 * lm + 0.04 * lm**2  # log P = 5.5
    alpha, dlogp = 0.018, 0.7
    if logp < 1.0:
        return a
    if logp < 2.7 - dlogp:
        return a + (logp - 1.0) / (1.7 - dlogp) * (b - a - alpha * dlogp)
    if logp < 2.7 + dlogp:
        return b + alpha * (logp - 2.7)
    if logp < 5.5:
        return b + alpha * dlogp + (logp - 2.7 - dlogp) / (2.8 - dlogp) * (c - b - alpha * dlogp)
    return c * math.exp(-0.3 * (logp - 5.5))


def _gamma_largeq(m1: float, logp: float) -> float:
    """Power-law slope of the mass-ratio distribution over 0.3 < q < 1 (equations 9-11)."""

    def solar(lp):
        return -0.5 if lp < 5.0 else -0.5 - 0.3 * (lp - 5.0)

    def mid(lp):  # 3.5 Msun
        if lp < 1.0:
            return -0.5
        if lp < 4.5:
            return -0.5 - 0.2 * (lp - 1.0)
        if lp < 6.5:
            return -1.2 - 0.4 * (lp - 4.5)
        return -2.0

    def high(lp):  # > 6 Msun
        if lp < 1.0:
            return -0.5
        if lp < 2.0:
            return -0.5 - 0.9 * (lp - 1.0)
        if lp < 4.0:
            return -1.4 - 0.3 * (lp - 2.0)
        return -2.0

    return _interpolate_in_mass(m1, solar(logp), mid(logp), high(logp))


def _gamma_smallq(m1: float, logp: float) -> float:
    """Power-law slope over 0.1 < q < 0.3 (equations 13-15)."""

    def mid(lp):
        if lp < 2.5:
            return 0.2
        if lp < 5.5:
            return 0.2 - 0.3 * (lp - 2.5)
        return -0.7 - 0.2 * (lp - 5.5)

    def high(lp):
        if lp < 1.0:
            return 0.1
        if lp < 3.0:
            return 0.1 - 0.15 * (lp - 1.0)
        if lp < 5.6:
            return -0.2 - 0.5 * (lp - 3.0)
        return -1.5

    return _interpolate_in_mass(m1, 0.3, mid(logp), high(logp))


def _interpolate_in_mass(m1: float, solar: float, mid: float, high: float) -> float:
    """Linear interpolation in mass between the 1, 3.5 and 6 Msun prescriptions."""
    if m1 <= 1.2:
        return solar
    if m1 <= 3.5:
        return solar + (mid - solar) * (m1 - 1.2) / (3.5 - 1.2)
    if m1 <= 6.0:
        return mid + (high - mid) * (m1 - 3.5) / (6.0 - 3.5)
    return high


def _f_twin(m1: float, logp: float) -> float:
    """Excess twin fraction over 0.95 < q < 1 (equations 5-7)."""
    f0 = 0.30 - 0.15 * math.log10(m1)
    logp_twin = 8.0 - m1 if m1 <= 6.5 else 1.5
    if logp < 1.0:
        return max(f0, 0.0)
    if logp < logp_twin:
        return max(f0 * (1.0 - (logp - 1.0) / (logp_twin - 1.0)), 0.0)
    return 0.0


def _eta(m1: float, logp: float) -> float:
    """Exponent of p(e) ~ e^eta (equations 17 and 18), interpolated over 3-7 Msun."""
    lp = max(logp, 0.55)
    low = 0.6 - 0.7 / (lp - 0.5)
    high = 0.9 - 0.2 / (lp - 0.5)
    if m1 <= 3.0:
        eta = low
    elif m1 >= 7.0:
        eta = high
    else:
        eta = low + (high - low) * (m1 - 3.0) / 4.0
    return max(eta, -0.95)


def _e_max(period: float) -> float:
    """Tidal ceiling on the eccentricity (equation 3): circular at and below two days."""
    if period <= 2.0:
        return 0.0
    return 1.0 - (period / 2.0) ** (-2.0 / 3.0)


def _mass_ratio_pdf(m1: float, logp: float, q_grid: np.ndarray) -> np.ndarray:
    """The MDS17 mass-ratio density on ``q_grid``, twin excess included.

    The excess twin fraction of MDS17 is the fraction of the companions with ``q > 0.3``
    that lie above the power law in ``0.95 < q < 1``, so the density is built on the full
    ``0.1 < q < 1`` interval and only then read on ``q_grid``: a grid that starts above
    0.3 sees the conditional density, and the twin fraction is not inflated by the cut.
    """
    gs, gl = _gamma_smallq(m1, logp), _gamma_largeq(m1, logp)
    full = np.linspace(0.1, 1.0, 1801)
    p = np.where(full < 0.3, (full / 0.3) ** gs, (full / 0.3) ** gl)
    large = full >= 0.3
    n_large = float(np.trapezoid(p[large], full[large]))
    f_twin = min(_f_twin(m1, logp), 0.95)
    twin = ((full >= 0.95) & (full <= 1.0)).astype(np.float64)
    twin = twin / float(np.trapezoid(twin, full)) * n_large * f_twin / (1.0 - f_twin)
    density = np.interp(q_grid, full, p + twin)
    return density / float(np.trapezoid(density, q_grid))


def _sample_from_pdf(rng, grid: np.ndarray, pdf: np.ndarray) -> float:
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    if cdf[-1] <= 0.0:
        return float(grid[0])
    cdf = cdf / cdf[-1]
    return float(np.interp(rng.uniform(), cdf, grid))


def _sample_eccentricity(rng, m1: float, period: float) -> float:
    e_max = _e_max(period)
    if e_max <= 0.0:
        return 0.0
    eta = _eta(m1, math.log10(period))
    u = rng.uniform()
    # p(e) ~ e^eta on (0, e_max): the inverse CDF is closed-form. Below 1e-3 the orbit is
    # circular for every purpose here, and an exact zero is what the fits hold exactly.
    e = float(e_max * u ** (1.0 / (eta + 1.0)))
    return e if e >= 1e-3 else 0.0


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class BinarySystem:
    """One double-lined binary, completely specified: the truth a simulation is built on.

    Attributes
    ----------
    name
        Identifier, unique in a population.
    m1, m2
        Masses in solar masses; component 1 is the more massive star.
    r1, r2
        Radii in solar radii.
    teff1, teff2, logg1, logg2, mh
        Atmospheric labels: temperatures in K, gravities in log cm/s^2, and one
        metallicity in dex shared by both stars.
    vsini1, vsini2
        Projected rotational velocities in km/s.
    period, ecc, omega, t_peri, incl, gamma
        The orbit: period [d], eccentricity, argument of periastron of the primary [rad],
        time of periastron [BJD], inclination [rad], systemic velocity [km/s].
    k1, k2
        Semi-amplitudes in km/s, consistent with the masses and the orbit.
    eclipsing
        Whether the geometry produces an eclipse.
    light_ratio
        ``F_2 / F_1`` in the RVS band.
    g_mag, g_rp, grvs
        The system's Gaia G, its G - G_RP, and G_RVS.
    ecl_lat_deg, n_transits
        Ecliptic latitude and the number of RVS transits Gaia records.
    ra_deg, dec_deg
        ICRS position in degrees, or ``None`` when the system has none. A drawn system
        gets one from its ecliptic latitude and a longitude drawn with it
        (:func:`ecliptic_to_icrs`), and a catalogue system gets the catalogue's. It is
        what a real transit forecast is requested for
        (:func:`albireo.gaia.gost_transits`); the ecliptic latitude alone sets the
        transit count.
    source
        Where the system came from: ``"parametric"`` or a catalogue name.
    meta
        Free-form provenance.
    """

    name: str
    m1: float
    m2: float
    r1: float
    r2: float
    teff1: float
    teff2: float
    logg1: float
    logg2: float
    mh: float
    vsini1: float
    vsini2: float
    period: float
    ecc: float
    omega: float
    t_peri: float
    incl: float
    gamma: float
    k1: float
    k2: float
    eclipsing: bool
    light_ratio: float
    g_mag: float
    g_rp: float
    grvs: float
    ecl_lat_deg: float
    n_transits: int
    ra_deg: float | None = None
    dec_deg: float | None = None
    source: str = "parametric"
    meta: dict = dataclasses.field(default_factory=dict)

    @property
    def q(self) -> float:
        """Mass ratio ``m_2 / m_1``."""
        return self.m2 / self.m1

    @property
    def light_fractions(self) -> tuple[float, float]:
        """``(l_1, l_2)`` summing to one, from the light ratio."""
        return 1.0 / (1.0 + self.light_ratio), self.light_ratio / (1.0 + self.light_ratio)

    @property
    def labels(self) -> tuple[dict[str, float], dict[str, float]]:
        """The library labels of the two stars, ``{"teff", "logg", "mh"}`` each."""
        return (
            {"teff": self.teff1, "logg": self.logg1, "mh": self.mh},
            {"teff": self.teff2, "logg": self.logg2, "mh": self.mh},
        )

    @property
    def a_rsun(self) -> float:
        """Semi-major axis in solar radii."""
        return float(semimajor_axis_rsun(self.m1 + self.m2, self.period))

    @property
    def t_conj(self) -> float:
        """Time of the primary's superior conjunction (its eclipse), from ``t_peri``."""
        return float(
            t_conj_from_t_peri(self.t_peri, period=self.period, ecc=self.ecc, omega=self.omega)
        )

    @property
    def max_separation_kms(self) -> float:
        """The largest relative velocity of the two stars, ``(K_1 + K_2)(1 + e)``."""
        return (self.k1 + self.k2) * (1.0 + self.ecc)

    def orbit(self) -> OrbitParams:
        """The orbit as :class:`albireo.simulate.OrbitParams`."""
        return OrbitParams(
            period=self.period,
            t_peri=self.t_peri,
            ecc=self.ecc,
            omega=self.omega,
            k=(self.k1, self.k2),
            gamma=self.gamma,
        )

    def to_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["meta"] = dict(self.meta)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BinarySystem:
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in fields})


def write_population(path, systems: Sequence[BinarySystem]) -> Path:
    """Write systems to a JSON file (one record per system)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([s.to_dict() for s in systems], indent=1, default=_jsonable), encoding="utf-8"
    )
    return path


def read_population(path) -> list[BinarySystem]:
    """Read systems written by :func:`write_population`."""
    with open(path, encoding="utf-8") as handle:
        return [BinarySystem.from_dict(d) for d in json.load(handle)]


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


# ---------------------------------------------------------------------------
# Light ratios
# ---------------------------------------------------------------------------


class _ContinuumRatio:
    """The RVS-band continuum of a library at a label point, cached per library."""

    def __init__(self, library, band=RVS_BAND):
        from albireo.library import library_interpolator

        self.library = library.sliced(float(band[0]) - 5.0, float(band[1]) + 5.0)
        self.interpolator = library_interpolator(self.library)
        self.bounds = self.library.bounds

    def log_continuum(self, labels: Mapping[str, float]) -> float:
        import jax.numpy as jnp

        point = jnp.asarray([float(labels[axis]) for axis in self.library.label_names])
        _, log_c = self.interpolator(point)
        return float(np.mean(np.asarray(log_c)))

    def inside(self, labels: Mapping[str, float]) -> bool:
        return all(lo <= float(labels[k]) <= hi for k, (lo, hi) in self.bounds.items())


def rvs_light_ratio(library, labels1, labels2, radius_ratio: float) -> float:
    """``F_2 / F_1`` in the RVS band from the library's continua and the radius ratio.

    The light ratio of two stars is the ratio of their surface fluxes times the ratio of
    their projected areas, ``(R_2 / R_1)^2 C_2 / C_1``, with the continua taken from the
    library at each star's labels and averaged over the band. This is the quantity the
    label fit's :class:`albireo.RadiusRatio` dilution model parameterizes, so a population
    drawn with it and a fit that measures the radius ratio are speaking the same language.

    Parameters
    ----------
    library
        A :class:`~albireo.library.SpectralLibrary` covering the band.
    labels1, labels2
        Label mappings for the two stars.
    radius_ratio
        ``R_2 / R_1``.
    """
    ratio = _ContinuumRatio(library)
    log_c1, log_c2 = ratio.log_continuum(labels1), ratio.log_continuum(labels2)
    return float(radius_ratio**2 * math.exp(log_c2 - log_c1))


# ---------------------------------------------------------------------------
# The parametric draw
# ---------------------------------------------------------------------------


def _draw_mass(rng, lo: float, hi: float, sequence=None, slope: float = -2.3) -> float:
    """A primary mass on ``[lo, hi]`` from the IMF ``dN/dm ~ m^slope``.

    With a ``sequence`` the mass function is weighted for a magnitude-limited sample: a
    survey to a fixed apparent magnitude reaches a star of absolute magnitude ``M_G`` to a
    distance proportional to ``10^(-0.2 M_G)``, so it contains such stars in proportion
    to ``10^(-0.6 M_G)``, and the field's K dwarfs give way to the F and G dwarfs such a
    sample is made of.
    """
    if sequence is None:
        p = slope + 1.0
        u = rng.uniform()
        return float((lo**p + u * (hi**p - lo**p)) ** (1.0 / p))
    grid = np.linspace(lo, hi, 400)
    absolute_g = np.asarray(sequence.absolute_g(grid), dtype=np.float64)
    return _sample_from_pdf(rng, grid, grid**slope * 10.0 ** (-0.6 * absolute_g))


def _draw_period(rng, m1: float, lo: float, hi: float) -> float:
    grid = np.linspace(math.log10(lo), math.log10(hi), 400)
    pdf = np.array([_f_logp(m1, lp) for lp in grid])
    return float(10 ** _sample_from_pdf(rng, grid, pdf))


def _draw_mass_ratio(rng, m1: float, logp: float, q_min: float) -> float:
    grid = np.linspace(max(q_min, 0.1), 1.0, 600)
    return _sample_from_pdf(rng, grid, _mass_ratio_pdf(m1, logp, grid))


def _draw_vsini(rng, radius: float, period: float, incl: float, teff: float) -> float:
    """Synchronised rotation for short periods; a field rotator otherwise.

    Below 15 days the star is taken as tidally synchronised, ``v = 2 pi R / P`` projected
    by the orbital inclination (Meibom, Mathieu & Stassun 2006 find synchronisation well
    before circularisation). Above it, stars hotter than the Kraft break rotate fast (a
    log-normal around 40 km/s) and cooler ones spin with periods of 5-40 days, projected
    by an isotropic spin axis.
    """
    if period < 15.0:
        return float(_SYNCHRONOUS_KMS * radius / period * math.sin(incl))
    sin_i = math.sqrt(1.0 - rng.uniform() ** 2)
    if teff > 6200.0:
        return float(np.exp(rng.normal(math.log(40.0), 0.5)) * sin_i)
    p_rot = rng.uniform(5.0, 40.0)
    return float(_SYNCHRONOUS_KMS * radius / p_rot * sin_i)


def draw_population(
    n: int,
    *,
    kind: str = "mixed",
    seed: int = 0,
    library=None,
    relation: MainSequence | str = "mamajek",
    mass_range: tuple[float, float] | None = None,
    period_range: tuple[float, float] = (0.5, 1000.0),
    q_min: float = 0.4,
    mass_weighting: str = "magnitude",
    min_separation_kms: float = 0.0,
    mh_range: tuple[float, float] = (-0.5, 0.3),
    g_mag: tuple[float, float] = (9.8, 1.3),
    g_mag_range: tuple[float, float] = (5.0, 12.2),
    gamma_sigma_kms: float = 30.0,
    release: str = "dr4",
    start_bjd: float = 2456863.94,
    max_tries: int = 100_000,
) -> list[BinarySystem]:
    """Draw ``n`` double-lined binaries from the field's distributions.

    The sky position is drawn with the rest: an ecliptic latitude from an isotropic
    distribution, which is what the transit count depends on, and an ecliptic longitude
    uniform over [0, 360), converted to ICRS through :func:`ecliptic_to_icrs`. The system
    then carries a position a real transit forecast can be requested for
    (:func:`albireo.gaia.gost_transits`), and the longitude is kept in ``meta``. The
    longitude comes from a generator spawned from ``seed`` rather than from the main
    stream, so that a population drawn before positions existed still reproduces.

    Parameters
    ----------
    n
        Number of systems.
    kind
        ``"eclipsing"``: every system eclipses, the inclination drawn from the eclipsing
        part of an isotropic distribution. ``"spectroscopic"``: none eclipses.
        ``"mixed"``: isotropic inclinations, eclipsing or not as the geometry falls.
    seed
        Seed.
    library
        A :class:`~albireo.library.SpectralLibrary` covering the RVS band. When given, the
        light ratios come from its continua (:func:`rvs_light_ratio`) and every star is
        kept inside its label box; otherwise the light ratio is the G-band flux ratio of
        the sequence and the mass range alone bounds the stars.
    relation
        The main-sequence relation (:class:`MainSequence` or its ``relation`` string).
    mass_range
        Primary and secondary masses in solar masses. Default: the masses whose sequence
        temperatures span the library's Teff box, or the tabulated range without one. Both
        stars must lie inside it, so the mass-ratio draw is conditioned on the secondary
        being in range: with the FGK box and a primary near a solar mass that favours
        ratios above 0.7, which is also what a double-lined selection favours.
    period_range
        Periods in days; the Gaia double-lined sample is two thirds below ten days, and
        the draw follows the field's period distribution within the range.
    q_min
        Smallest mass ratio drawn: a faint companion is not double-lined.
    mass_weighting
        ``"magnitude"``: the primary mass follows the initial mass function weighted by
        the volume a magnitude-limited survey reaches, ``10^(-0.6 M_G)``, and a system is
        kept with the probability its combined light adds over the primary alone (a twin
        is seen through 2.8 times the volume of a single star). ``"volume"``: the initial
        mass function alone. The Gaia double-lined sample is magnitude limited.
    min_separation_kms
        Smallest largest separation ``(K_1 + K_2)(1 + e)`` kept, the velocity selection of
        a double-lined sample; an RVS resolution element is 26 km/s. Zero keeps every
        orbit.
    mh_range
        Metallicity range, sampled uniformly and shared by both stars.
    g_mag, g_mag_range
        Mean and standard deviation of the system's G, and the clip. The defaults follow
        the Gaia DR3 double-lined sample (G from 3.6 to 12.2, mean 9.8).
    gamma_sigma_kms
        Standard deviation of the systemic velocity.
    release
        ``"dr3"`` or ``"dr4"``: the span the transit count refers to.
    start_bjd
        Start of the window the periastron time is drawn in.
    max_tries
        Cap on rejected draws.

    Returns
    -------
    list of BinarySystem
    """
    if kind not in ("eclipsing", "spectroscopic", "mixed"):
        raise ValueError(f"kind must be 'eclipsing', 'spectroscopic' or 'mixed'; got {kind!r}")
    if not 0.0 < q_min <= 1.0:
        raise ValueError("q_min must lie in (0, 1]")
    if mass_weighting not in ("magnitude", "volume"):
        raise ValueError(f"mass_weighting must be 'magnitude' or 'volume'; got {mass_weighting!r}")
    rng = np.random.default_rng(seed)
    # The sky longitude is drawn from a generator of its own, spawned from the same seed.
    # Nothing else depends on it, and keeping it out of the main stream means that adding
    # it left every other draw where it was: a population drawn before systems carried a
    # position reproduces unchanged.
    lon_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    sequence = relation if isinstance(relation, MainSequence) else MainSequence(relation)
    ratio = _ContinuumRatio(library) if library is not None else None
    if mass_range is None:
        if ratio is not None:
            t_lo, t_hi = ratio.bounds["teff"]
            m_lo = float(sequence.mass_from_teff(t_lo))
            m_hi = float(sequence.mass_from_teff(t_hi))
            lo, hi = sequence.mass_range
            mass_range = (max(m_lo, lo), min(m_hi, hi))
        else:
            mass_range = sequence.mass_range
    m_lo, m_hi = float(mass_range[0]), float(mass_range[1])
    if not 0.0 < m_lo < m_hi:
        raise ValueError(f"mass_range must satisfy 0 < lo < hi; got {mass_range}")
    p_lo, p_hi = float(period_range[0]), float(period_range[1])
    if not 0.0 < p_lo < p_hi:
        raise ValueError(f"period_range must satisfy 0 < lo < hi; got {period_range}")

    systems: list[BinarySystem] = []
    tries = 0
    while len(systems) < n:
        tries += 1
        if tries > max_tries:
            raise RuntimeError(
                f"only {len(systems)} of {n} systems drawn after {max_tries} tries; widen "
                "mass_range or period_range, or lower q_min"
            )
        m1 = _draw_mass(rng, m_lo, m_hi, sequence if mass_weighting == "magnitude" else None)
        period = _draw_period(rng, m1, p_lo, p_hi)
        q = _draw_mass_ratio(rng, m1, math.log10(period), q_min)
        m2 = q * m1
        if m2 < m_lo:
            continue
        if mass_weighting == "magnitude":
            # The pair is seen through the volume its combined light reaches, relative to
            # the twin that reaches furthest.
            delta = float(sequence.absolute_g(m2)) - float(sequence.absolute_g(m1))
            if rng.uniform() > ((1.0 + 10.0 ** (-0.4 * delta)) / 2.0) ** 1.5:
                continue
        ecc = _sample_eccentricity(rng, m1, period)
        omega = float(rng.uniform(0.0, 2.0 * math.pi))
        r1, r2 = float(sequence.radius(m1)), float(sequence.radius(m2))
        a = float(semimajor_axis_rsun(m1 + m2, period))
        if r1 + r2 > 0.6 * a:
            continue  # a near-contact pair is not the detached binary this describes
        p_ecl = float(eclipse_probability(r1, r2, a, ecc, omega))
        if kind == "eclipsing":
            cos_i = rng.uniform(0.0, p_ecl)
        elif kind == "spectroscopic":
            cos_i = rng.uniform(p_ecl, 1.0)
        else:
            cos_i = rng.uniform(0.0, 1.0)
        incl = float(math.acos(cos_i))
        eclipsing = bool(cos_i <= p_ecl)
        teff1, teff2 = float(sequence.teff(m1)), float(sequence.teff(m2))
        logg1, logg2 = float(sequence.logg(m1)), float(sequence.logg(m2))
        mh = float(rng.uniform(*mh_range))
        labels1 = {"teff": teff1, "logg": logg1, "mh": mh}
        labels2 = {"teff": teff2, "logg": logg2, "mh": mh}
        if ratio is not None and not (ratio.inside(labels1) and ratio.inside(labels2)):
            continue
        k1, k2 = semi_amplitudes(m1, m2, period, ecc, incl)
        if (float(k1) + float(k2)) * (1.0 + ecc) < min_separation_kms:
            continue
        if ratio is not None:
            light_ratio = float(
                (r2 / r1) ** 2
                * math.exp(ratio.log_continuum(labels2) - ratio.log_continuum(labels1))
            )
        else:
            light_ratio = float(10 ** (-0.4 * (sequence.absolute_g(m2) - sequence.absolute_g(m1))))
        g_sys = float(np.clip(rng.normal(g_mag[0], g_mag[1]), g_mag_range[0], g_mag_range[1]))
        # The system colour: flux-weighted in G and RP, from the G-band ratio of the sequence.
        gr1, gr2 = float(sequence.g_minus_rp(m1)), float(sequence.g_minus_rp(m2))
        f_g2 = 10 ** (-0.4 * (float(sequence.absolute_g(m2)) - float(sequence.absolute_g(m1))))
        f_rp1, f_rp2 = 10 ** (0.4 * gr1), f_g2 * 10 ** (0.4 * gr2)
        g_rp = float(2.5 * math.log10((f_rp1 + f_rp2) / (1.0 + f_g2)))
        grvs = float(predict_grvs(g_sys, g_sys - g_rp))
        if not np.isfinite(grvs):
            continue
        ecl_lat = float(math.degrees(math.asin(rng.uniform(-1.0, 1.0))))
        ecl_lon = float(lon_rng.uniform(0.0, 360.0))
        ra_deg, dec_deg = ecliptic_to_icrs(ecl_lon, ecl_lat)
        n_transits = int(
            rvs_transit_count(
                1, ecl_lat_deg=ecl_lat, release=release, seed=int(rng.integers(0, 2**31))
            )[0]
        )
        systems.append(
            BinarySystem(
                name=f"{kind}-{len(systems):04d}",
                m1=m1,
                m2=m2,
                r1=r1,
                r2=r2,
                teff1=teff1,
                teff2=teff2,
                logg1=logg1,
                logg2=logg2,
                mh=mh,
                vsini1=_draw_vsini(rng, r1, period, incl, teff1),
                vsini2=_draw_vsini(rng, r2, period, incl, teff2),
                period=period,
                ecc=ecc,
                omega=omega,
                t_peri=float(start_bjd + rng.uniform(0.0, period)),
                incl=incl,
                gamma=float(rng.normal(0.0, gamma_sigma_kms)),
                k1=float(k1),
                k2=float(k2),
                eclipsing=eclipsing,
                light_ratio=light_ratio,
                g_mag=g_sys,
                g_rp=g_rp,
                grvs=grvs,
                ecl_lat_deg=ecl_lat,
                n_transits=n_transits,
                ra_deg=float(ra_deg),
                dec_deg=float(dec_deg),
                source="parametric",
                meta={
                    "relation": sequence.relation,
                    "kind": kind,
                    "seed": seed,
                    "ecl_lon": ecl_lon,
                },
            )
        )
    return systems


# ---------------------------------------------------------------------------
# Catalogue adapters
# ---------------------------------------------------------------------------

_DEBCAT_COLUMNS = (
    "system",
    "spt1",
    "spt2",
    "period",
    "vmag",
    "bmv",
    "logm1",
    "logm1e",
    "logm2",
    "logm2e",
    "logr1",
    "logr1e",
    "logr2",
    "logr2e",
    "logg1",
    "logg1e",
    "logg2",
    "logg2e",
    "logt1",
    "logt1e",
    "logt2",
    "logt2e",
    "logl1",
    "logl1e",
    "logl2",
    "logl2e",
    "mh",
    "mhe",
)
_DEBCAT_MISSING = -9.99


def _complete(
    *,
    name: str,
    m1: float,
    m2: float,
    r1: float,
    r2: float,
    teff1: float,
    teff2: float,
    mh: float,
    period: float,
    ecc: float | None,
    omega: float | None,
    incl: float | None,
    gamma: float | None,
    g_mag: float | None,
    g_rp: float | None,
    grvs: float | None,
    ecl_lat: float | None,
    n_transits: int | None,
    eclipsing: bool | None,
    k1: float | None,
    k2: float | None,
    vsini1: float | None,
    vsini2: float | None,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    rng,
    sequence: MainSequence,
    ratio: _ContinuumRatio | None,
    release: str,
    start_bjd: float,
    source: str,
    meta: dict,
) -> BinarySystem:
    """Fill what a catalogue row lacks with the parametric prescriptions."""
    if m2 > m1:  # component 1 is the more massive star
        m1, m2, r1, r2, teff1, teff2 = m2, m1, r2, r1, teff2, teff1
        vsini1, vsini2 = vsini2, vsini1
        if k1 is not None and k2 is not None:
            k1, k2 = k2, k1
        meta = {**meta, "components_swapped": True}
    if ecc is None:
        ecc = _sample_eccentricity(rng, m1, period)
    if omega is None:
        omega = float(rng.uniform(0.0, 2.0 * math.pi))
    a = float(semimajor_axis_rsun(m1 + m2, period))
    p_ecl = float(eclipse_probability(r1, r2, a, ecc, omega))
    if incl is None:
        if eclipsing is True:
            incl = float(math.acos(rng.uniform(0.0, p_ecl)))
        elif eclipsing is False:
            incl = float(math.acos(rng.uniform(p_ecl, 1.0)))
        elif k1 is not None and k2 is not None:
            # The inclination that makes the masses and the catalogue's K agree.
            k1_edge, _ = semi_amplitudes(m1, m2, period, ecc, math.pi / 2.0)
            incl = float(math.asin(min(k1 / float(k1_edge), 1.0)))
        else:
            incl = float(math.acos(rng.uniform(0.0, 1.0)))
    if eclipsing is None:
        eclipsing = bool(math.cos(incl) <= p_ecl)
    if k1 is None or k2 is None:
        k1, k2 = (float(v) for v in semi_amplitudes(m1, m2, period, ecc, incl))
    logg1 = _LOGG_SUN + math.log10(m1) - 2.0 * math.log10(r1)
    logg2 = _LOGG_SUN + math.log10(m2) - 2.0 * math.log10(r2)
    labels1 = {"teff": teff1, "logg": logg1, "mh": mh}
    labels2 = {"teff": teff2, "logg": logg2, "mh": mh}
    if ratio is not None and ratio.inside(labels1) and ratio.inside(labels2):
        light_ratio = float(
            (r2 / r1) ** 2 * math.exp(ratio.log_continuum(labels2) - ratio.log_continuum(labels1))
        )
    else:
        # Outside a library: the Rayleigh-Jeans-like ratio of surface fluxes at 858 nm.
        light_ratio = float((r2 / r1) ** 2 * _planck_ratio(teff2, teff1))
    if vsini1 is None:
        vsini1 = _draw_vsini(rng, r1, period, incl, teff1)
    if vsini2 is None:
        vsini2 = _draw_vsini(rng, r2, period, incl, teff2)
    if g_mag is None:
        g_mag = float(np.clip(rng.normal(9.8, 1.3), 5.0, 12.2))
    if g_rp is None:
        mm1, mm2 = sequence.mass_from_teff(teff1), sequence.mass_from_teff(teff2)
        g_rp = float(0.5 * (sequence.g_minus_rp(mm1) + sequence.g_minus_rp(mm2)))
    if grvs is None or not np.isfinite(grvs):
        grvs = float(predict_grvs(g_mag, g_mag - g_rp))
        if not np.isfinite(grvs):
            grvs = float(g_mag - 0.7)  # a red star outside the colour relation's range
    if ecl_lat is None:
        ecl_lat = float(math.degrees(math.asin(rng.uniform(-1.0, 1.0))))
    if n_transits is None:
        n_transits = int(
            rvs_transit_count(
                1, ecl_lat_deg=ecl_lat, release=release, seed=int(rng.integers(0, 2**31))
            )[0]
        )
    if gamma is None:
        gamma = float(rng.normal(0.0, 30.0))
    return BinarySystem(
        name=name,
        m1=float(m1),
        m2=float(m2),
        r1=float(r1),
        r2=float(r2),
        teff1=float(teff1),
        teff2=float(teff2),
        logg1=float(logg1),
        logg2=float(logg2),
        mh=float(mh),
        vsini1=float(vsini1),
        vsini2=float(vsini2),
        period=float(period),
        ecc=float(ecc),
        omega=float(omega),
        t_peri=float(start_bjd + rng.uniform(0.0, period)),
        incl=float(incl),
        gamma=float(gamma),
        k1=float(k1),
        k2=float(k2),
        eclipsing=bool(eclipsing),
        light_ratio=light_ratio,
        g_mag=float(g_mag),
        g_rp=float(g_rp),
        grvs=float(grvs),
        ecl_lat_deg=float(ecl_lat),
        n_transits=int(n_transits),
        ra_deg=None if ra_deg is None else float(ra_deg),
        dec_deg=None if dec_deg is None else float(dec_deg),
        source=source,
        meta=meta,
    )


def _planck_ratio(t_a: float, t_b: float, wave_angstrom: float = 8580.0) -> float:
    """``B(T_a) / B(T_b)`` at one wavelength: the surface-flux ratio without a library."""
    x = 1.4388e8 / wave_angstrom  # hc/k in K Angstrom, over lambda
    return float(math.expm1(x / t_b) / math.expm1(x / t_a))


def from_debcat(
    path,
    *,
    seed: int = 0,
    library=None,
    relation: MainSequence | str = "mamajek",
    release: str = "dr4",
    start_bjd: float = 2456863.94,
    names: Iterable[str] | None = None,
) -> list[BinarySystem]:
    """Systems from DEBCat's ``debs.dat`` (Southworth 2015), one per catalogue row.

    DEBCat gives the period, masses, radii, temperatures, gravities and metallicity of
    every well-studied detached eclipsing binary, with a V magnitude. It gives no
    eccentricity, no inclination beyond the fact of eclipsing, no G magnitude and no
    rotation, so those are drawn: the eccentricity from the period-conditioned
    prescription, the inclination from the eclipsing range, G from the double-lined
    sample's distribution, and rotation as synchronised or field. A missing metallicity
    (``-9.99``) becomes solar. Systems whose temperatures fall outside the library box
    are kept, with the light ratio from a blackbody ratio instead; the benchmark decides
    what to do with them.

    Parameters
    ----------
    path
        The ``debs.dat`` file (https://www.astro.keele.ac.uk/jkt/debcat/debs.dat).
    seed, library, relation, release, start_bjd
        As for :func:`draw_population`.
    names
        Optional subset of system names to keep.
    """
    rng = np.random.default_rng(seed)
    sequence = relation if isinstance(relation, MainSequence) else MainSequence(relation)
    ratio = _ContinuumRatio(library) if library is not None else None
    keep = None if names is None else {str(n) for n in names}
    systems = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) < len(_DEBCAT_COLUMNS):
                continue
            row = dict(zip(_DEBCAT_COLUMNS, parts[: len(_DEBCAT_COLUMNS)], strict=True))
            name = row["system"]
            if keep is not None and name not in keep:
                continue
            values = {k: float(row[k]) for k in _DEBCAT_COLUMNS[3:]}
            if any(
                values[k] == _DEBCAT_MISSING
                for k in ("period", "logm1", "logm2", "logr1", "logr2", "logt1", "logt2")
            ):
                continue
            mh = 0.0 if values["mh"] == _DEBCAT_MISSING else values["mh"]
            vmag = None if values["vmag"] == _DEBCAT_MISSING else values["vmag"]
            teff1, teff2 = 10 ** values["logt1"], 10 ** values["logt2"]
            g_mag = None
            if vmag is not None:
                # G - V from the sequence at the primary's temperature (the table's M_G - M_V).
                mass = sequence.mass_from_teff(teff1)
                g_minus_v = float(
                    sequence.absolute_g(mass)
                    - np.interp(np.log10(mass), _MAMAJEK_LOGM, _MAMAJEK_BY_MASS[:, 3])
                )
                g_mag = float(vmag + g_minus_v)
            systems.append(
                _complete(
                    name=f"debcat-{name}",
                    m1=10 ** values["logm1"],
                    m2=10 ** values["logm2"],
                    r1=10 ** values["logr1"],
                    r2=10 ** values["logr2"],
                    teff1=teff1,
                    teff2=teff2,
                    mh=mh,
                    period=values["period"],
                    ecc=None,
                    omega=None,
                    incl=None,
                    gamma=None,
                    g_mag=g_mag,
                    g_rp=None,
                    grvs=None,
                    ecl_lat=None,
                    n_transits=None,
                    eclipsing=True,
                    k1=None,
                    k2=None,
                    vsini1=None,
                    vsini2=None,
                    rng=rng,
                    sequence=sequence,
                    ratio=ratio,
                    release=release,
                    start_bjd=start_bjd,
                    source="DEBCat",
                    meta={"spt1": row["spt1"], "spt2": row["spt2"], "vmag": vmag},
                )
            )
    return systems


_GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap/sync"
_GAIA_SB2_ADQL = """
SELECT TOP {limit} nss.source_id, nss.nss_solution_type,
       nss.period, nss.eccentricity, nss.arg_periastron, nss.t_periastron,
       nss.semi_amplitude_primary AS k1, nss.semi_amplitude_secondary AS k2,
       nss.center_of_mass_velocity AS gamma,
       bm.m1, bm.m2,
       s.phot_g_mean_mag, s.phot_rp_mean_mag, s.teff_gspphot, s.ecl_lat,
       s.ra, s.dec,
       nss.rv_n_good_obs_primary
FROM gaiadr3.nss_two_body_orbit AS nss
JOIN gaiadr3.gaia_source AS s ON nss.source_id = s.source_id
LEFT JOIN gaiadr3.binary_masses AS bm ON nss.source_id = bm.source_id
WHERE nss.nss_solution_type IN ('SB2', 'SB2C')
ORDER BY nss.source_id
"""


def query_gaia_sb2(limit: int = 500, *, timeout: float = 120.0) -> list[dict[str, Any]]:
    """The Gaia DR3 double-lined orbits (``SB2`` and ``SB2C``) with their masses and photometry.

    Runs a synchronous, anonymous ADQL query against the Gaia archive (network) and
    returns one dictionary per source with the columns :func:`from_gaia_sb2` reads. The
    eccentricity of a circular solution is stored as NULL by the archive and comes back
    as 0. Gaia publishes no G_RVS, no transit count and no light ratio for its
    double-lined sources (their combined velocities were discarded), so G_RVS is
    predicted from the colour, the transit count is ``rv_n_good_obs_primary``, and the
    light ratio comes from the masses through the main sequence.

    References
    ----------
    Gaia Collaboration, Arenou, F., Babusiaux, C., et al. 2023, A&A, 674, A34
    """
    query = _GAIA_SB2_ADQL.format(limit=int(limit))
    params = urllib.parse.urlencode(
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query}
    )
    request = urllib.request.Request(
        f"{_GAIA_TAP}?{params}", headers={"User-Agent": "albireo (population)"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        text = response.read().decode("utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    header = [h.strip() for h in lines[0].split(",")]
    rows = []
    for line in lines[1:]:
        cells = line.split(",")
        row: dict[str, Any] = {}
        for key, cell in zip(header, cells, strict=False):
            cell = cell.strip().strip('"')
            if cell in ("", "null", "NULL"):
                row[key] = None
            else:
                try:
                    row[key] = float(cell)
                except ValueError:
                    row[key] = cell
        rows.append(row)
    return rows


def from_gaia_sb2(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int = 0,
    library=None,
    relation: MainSequence | str = "mamajek",
    release: str = "dr3",
    start_bjd: float = 2456863.94,
) -> list[BinarySystem]:
    """Systems from Gaia DR3 double-lined orbits (the rows of :func:`query_gaia_sb2`).

    Each row supplies the period, eccentricity, argument of periastron, both
    semi-amplitudes and the systemic velocity, the G magnitude and colour, the ICRS
    position, the ecliptic latitude and the number of good transits. Masses come from
    ``binary_masses`` where
    the archive has them and otherwise from the semi-amplitudes and the main sequence
    (``q = K_1 / K_2``, the primary at the mass whose sequence temperature is the
    GSP-Phot value); radii and temperatures follow from the masses along the sequence,
    and the inclination is the one that reconciles the masses with the catalogue's K.
    Rows without a period or a semi-amplitude are skipped.
    """
    rng = np.random.default_rng(seed)
    sequence = relation if isinstance(relation, MainSequence) else MainSequence(relation)
    ratio = _ContinuumRatio(library) if library is not None else None
    systems = []
    for row in rows:
        period, k1, k2 = row.get("period"), row.get("k1"), row.get("k2")
        if period is None or k1 is None or k2 is None or not (period > 0 and k1 > 0 and k2 > 0):
            continue
        ecc = row.get("eccentricity")
        ecc = 0.0 if ecc is None else float(ecc)
        omega = row.get("arg_periastron")
        omega = None if omega is None else math.radians(float(omega))
        m1, m2 = row.get("m1"), row.get("m2")
        q = float(k1) / float(k2)
        if m1 is None or m2 is None or not (m1 > 0 and m2 > 0):
            teff = row.get("teff_gspphot")
            teff = 6000.0 if teff is None else float(teff)
            lo, hi = sequence.mass_range
            m1 = float(np.clip(sequence.mass_from_teff(teff), lo, hi))
            m2 = float(np.clip(q * m1, lo, hi))
        m1, m2 = float(m1), float(m2)
        lo, hi = sequence.mass_range
        m1c, m2c = float(np.clip(m1, lo, hi)), float(np.clip(m2, lo, hi))
        g_mag = row.get("phot_g_mean_mag")
        rp = row.get("phot_rp_mean_mag")
        g_rp = None if g_mag is None or rp is None else float(g_mag) - float(rp)
        n_good = row.get("rv_n_good_obs_primary")
        source_id = row.get("source_id")
        name = f"gaia-{int(source_id) if source_id is not None else len(systems)}"
        systems.append(
            _complete(
                name=name,
                m1=m1,
                m2=m2,
                r1=float(sequence.radius(m1c)),
                r2=float(sequence.radius(m2c)),
                teff1=float(sequence.teff(m1c)),
                teff2=float(sequence.teff(m2c)),
                mh=0.0,
                period=float(period),
                ecc=ecc,
                omega=omega,
                incl=None,
                gamma=None if row.get("gamma") is None else float(row["gamma"]),
                g_mag=None if g_mag is None else float(g_mag),
                g_rp=g_rp,
                grvs=None,
                ecl_lat=None if row.get("ecl_lat") is None else float(row["ecl_lat"]),
                n_transits=None if n_good is None else int(n_good),
                eclipsing=None,
                k1=float(k1),
                k2=float(k2),
                vsini1=None,
                vsini2=None,
                ra_deg=None if row.get("ra") is None else float(row["ra"]),
                dec_deg=None if row.get("dec") is None else float(row["dec"]),
                rng=rng,
                sequence=sequence,
                ratio=ratio,
                release=release,
                start_bjd=start_bjd,
                source="Gaia DR3 nss_two_body_orbit",
                meta={"solution_type": row.get("nss_solution_type"), "source_id": source_id},
            )
        )
    return systems


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def population_summary(systems: Sequence[BinarySystem]) -> str:
    """A few lines describing a population: counts, and the quartiles of what matters."""
    if not systems:
        return "empty population"

    def q(values):
        v = np.asarray(values, dtype=np.float64)
        return f"{np.percentile(v, 25):.3g} / {np.median(v):.3g} / {np.percentile(v, 75):.3g}"

    n_ecl = sum(s.eclipsing for s in systems)
    lines = [
        f"{len(systems)} systems ({n_ecl} eclipsing), sources "
        + ", ".join(sorted({s.source for s in systems})),
        f"  P [d]        quartiles {q([s.period for s in systems])}",
        f"  e            quartiles {q([s.ecc for s in systems])}",
        f"  K1 + K2 [km/s] quartiles {q([s.k1 + s.k2 for s in systems])}",
        f"  q = M2/M1    quartiles {q([s.q for s in systems])}",
        f"  F2/F1 (RVS)  quartiles {q([s.light_ratio for s in systems])}",
        f"  Teff1 [K]    quartiles {q([s.teff1 for s in systems])}",
        f"  G_RVS        quartiles {q([s.grvs for s in systems])}",
        f"  transits     quartiles {q([s.n_transits for s in systems])}",
    ]
    return "\n".join(lines)

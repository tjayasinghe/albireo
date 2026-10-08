"""Gaia RVS: a simulator of the instrument and of the products it delivers.

**Experimental.** The names and the product presets may change when the DR4 reader is
written against the released data model.

The Radial Velocity Spectrometer records 846-870 nm at a nominal resolving power of 11,500
on a detector that disperses at 0.02453 nm per pixel. The resolution element is three
pixels, and the resolving power measured per CCD in flight ranges from 10,983 to 12,587
(Cropper et al. 2018 Table 6). The archive does not publish the detector pixels. DR3's mean
spectra are resampled onto a 0.01 nm grid (2401 samples in the rest frame of the source, one
mean per source over its epochs and so unusable for disentangling). DR4's epoch spectra are
resampled onto a 0.025 nm grid (961 samples, barycentric frame, one per field-of-view
transit with its three CCDs combined). The resampling correlates the noise, and the
published per-pixel errors do not include the correlation. This module reproduces that
chain so that a synthetic RVS dataset has the same defects as the real product will:

1. the component spectra are combined and observed on the detector grid through the
   operator stack of :func:`albireo.simulate.simulate_dataset`, with the noise added
   there, per detector pixel, as photon counting at the S/N that follows from G_RVS
   (:func:`rvs_snr_per_pixel`);
2. :func:`deliver` then interpolates each epoch onto the delivered grid as the archive
   does, propagates the variance through the interpolation weights, and records the
   lag-one correlation that the diagonal errors cannot express.

The model is that of Rowan's ``SyntheticSB2`` with the ``GAIA_RVS`` preset
(``binaryspectra/notebooks/synthetic_rvs_spectra.ipynb``), with two differences. The
component spectra come from a published grid through :mod:`albireo.library` rather than
from a synthesis code, because albireo does not synthesise spectra. The wavelength scale is
vacuum, as Gaia's is. Every constant of the S/N model is a field of :class:`RVSConstants`
with its source in the docstring.

The transit cadence (:func:`rvs_transit_count`, :func:`rvs_transit_times`) reproduces the
measured structure of the scanning law (the count distribution and its latitude dependence,
the visibility periods and the 106.5-minute field-of-view pairs) but not its phase. Real
transit times for a sky position come from the Gaia Observation Forecast Tool over its IVOA
endpoint (:func:`gost_transits`), which forecasts the nominal law for the whole mission.
:func:`rvs_transit_times_from_gost` restricts them to a release, to the CCD rows the RVS
covers and to the fraction that reaches the ground. Either list is passed as ``bjd``.

References
----------
Cropper, M., Katz, D., Sartoretti, P., et al. 2018, A&A, 616, A5
Katz, D., Sartoretti, P., Guerrier, A., et al. 2023, A&A, 674, A5
Sartoretti, P., Katz, D., Cropper, M., et al. 2018, A&A, 616, A6
Sartoretti, P., Blanco-Cuaresma, S., Katz, D., et al. 2023, A&A, 674, A6
Seabroke, G. M., Fabricius, C., Teyssier, D., et al. 2021, A&A, 653, A160
"""

from __future__ import annotations

import dataclasses
import math
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np

from albireo.data import Dataset, EpochData
from albireo.examples import cache_dir
from albireo.grids import C_KMS, LogGrid, log_doppler_shift
from albireo.simulate import InstrumentSpec, OrbitParams, SimulationTruth, simulate_dataset

__all__ = [
    "GOST_ENDPOINT",
    "GOST_USABLE_FRACTION",
    "RVS_BAND",
    "RVS_CCD_RESOLVING_POWERS",
    "RVS_CCD_ROWS",
    "RVS_CONSTANTS",
    "RVS_DETECTOR_STEP",
    "RVS_DR3_MEAN",
    "RVS_DR4_EPOCH",
    "RVS_RESOLVING_POWER",
    "RVS_SPANS",
    "SIMULATION_VARIANCE_FACTOR",
    "DeliveryRecord",
    "GostTransits",
    "RVSConstants",
    "RVSProduct",
    "RVSTruth",
    "deliver",
    "gost_transits",
    "predict_grvs",
    "quadrature_sigma_kms",
    "random_times",
    "rvs_components",
    "rvs_delivered_sigma_kms",
    "rvs_detector_grid",
    "rvs_lsf_sigma_kms",
    "rvs_model_grid",
    "rvs_snr_per_pixel",
    "rvs_transit_count",
    "rvs_transit_resolving_powers",
    "rvs_transit_times",
    "rvs_transit_times_from_gost",
    "simulate_rvs_dataset",
    "uniform_phase_times",
]

RVS_BAND: tuple[float, float] = (8460.0, 8700.0)
"""The RVS band, 846-870 nm, in vacuum Angstrom."""

RVS_RESOLVING_POWER: float = 11_500.0
"""Nominal resolving power ``lambda / FWHM`` of the RVS: a resolution element of three
pixels (Sartoretti et al. 2018 §5.4), the value the Gaia pipelines convolve their templates
with. The in-flight measurements range from 10,983 to 12,587 per CCD
(:data:`RVS_CCD_RESOLVING_POWERS`)."""

RVS_CCD_RESOLVING_POWERS: dict[tuple[int, int], tuple[float, float, float]] = {
    (1, 4): (12587.0, 12361.0, 12240.0),
    (1, 5): (12159.0, 12430.0, 12085.0),
    (1, 6): (12021.0, 12132.0, 12148.0),
    (1, 7): (12117.0, 11885.0, 11525.0),
    (2, 4): (12065.0, 12106.0, 11954.0),
    (2, 5): (11600.0, 11809.0, 11861.0),
    (2, 6): (11523.0, 11447.0, 11901.0),
    (2, 7): (11078.0, 10983.0, 11377.0),
}
"""Resolving power measured in flight per (telescope, CCD row) for the three RVS strips
15, 16 and 17 (Cropper et al. 2018, Table 6; from the cross-correlation of Fe lines with a
binary mask). A transit crosses one row of one telescope and its epoch spectrum combines
the three strips, so the effective resolving power of a transit is the row's mean, from
11,146 to 12,396. This is a spread of about 7% about the nominal 11,500 that the pipelines
adopt."""

RVS_DETECTOR_STEP: float = 0.245
"""Along-scan dispersion at the detector, 0.02453 nm per pixel, in Angstrom."""

_FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))

_FROM_LIBRARY = "the library's own resolving_power"
"""The default of ``simulate_rvs_dataset(library_resolving_power=...)``: read it from the
library, which must then be given."""


# ---------------------------------------------------------------------------
# Photometry and signal-to-noise
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RVSConstants:
    """The constants of the RVS signal-to-noise model, each with its source.

    The model is ESA's predictive form of ``rv_expected_sig_to_noise`` (the Gaia science
    performance page, section 3, "spectroscopic performance", updated for DR3), which
    substitutes a G_RVS-derived flux for the measured median flux of the catalogue's own
    definition. It reproduces the published column to 3-8% from G_RVS 6 to 14. ``S`` is
    the mean flux of the band per sample, not the continuum, so the S/N it gives is the
    one :class:`albireo.simulate.InstrumentSpec` defines at a reference flux.
    ``rvs_snr_per_pixel`` evaluates it.

    Attributes
    ----------
    zero_point
        G_RVS zero point in the electron-per-second system, 21.317 +- 0.002 mag: a
        G_RVS = 0 source gives ``10**(zero_point / 2.5)`` electrons per second over the
        whole band (Sartoretti et al. 2023, §4.2).
    exposure_s
        Exposure time of one CCD crossing, 4.4167 s (Cropper et al. 2018, Table 5).
    ccds_per_transit
        Number of RVS CCD crossings in one field-of-view transit: three, strips 15-17.
    pixel_nm, band_nm
        Along-scan pixel width (0.0244 nm at 847 nm to 0.0246 at 873, Cropper et al.
        2018 Table 5; 0.02453 is the band mean) and the width of the band, in nm. Their
        ratio is the fraction of the band's electrons collected in one pixel.
    background_e
        Median background per pixel per CCD crossing, in electrons: ESA's mission median,
        after the stray light decreased from December 2015. Individual exposures range
        over a factor of a few with spin phase (Cropper et al. 2018 §9.1).
    ac_pixels
        Across-scan extent of a window, 10 pixels (Sartoretti et al. 2023).
    read_noise_e
        Read-out noise per sample, in electrons. ESA adopts 3.2; the commissioning mean
        over the twelve detectors is 3.1 with a range of 2.9-3.4 (Cropper et al. 2018,
        Table 7). Immaterial below the window-class limit, where the background
        dominates.
    window_class_limit
        Sources brighter than this G_RVS keep two-dimensional windows (window class 0,
        Cropper et al. 2018 §7.1), so every across-scan pixel is read separately and
        contributes its own read noise. Fainter sources are binned across scan on board
        and have one read noise per column.
    """

    zero_point: float = 21.317
    exposure_s: float = 4.4167032
    ccds_per_transit: int = 3
    pixel_nm: float = 0.02453
    band_nm: float = 24.0
    background_e: float = 4.7
    ac_pixels: int = 10
    read_noise_e: float = 3.2
    window_class_limit: float = 7.0


RVS_CONSTANTS = RVSConstants()
"""The default constants; pass another :class:`RVSConstants` to override one."""


def predict_grvs(g_mag, rp_mag):
    """G_RVS from G and G_RP, for sources without a catalogue ``grvs_mag``.

    Two cubic polynomials in ``x = G - G_RP`` (Sartoretti et al. 2023, equations 9 and
    10, with an rms of 0.04 and 0.09 mag), the first for ``x <= 1.2`` and the second
    above it, valid for ``-0.15 <= x <= 1.7``; ``nan`` outside. The catalogue's
    spectroscopic ``grvs_mag`` is to be preferred where it exists.

    Parameters
    ----------
    g_mag, rp_mag
        Gaia G and G_RP magnitudes; scalars or arrays.

    Returns
    -------
    numpy.ndarray
        G_RVS, ``nan`` where the colour is outside the calibrated range.
    """
    g = np.asarray(g_mag, dtype=np.float64)
    rp = np.asarray(rp_mag, dtype=np.float64)
    x = g - rp
    blue = -0.0397 - 0.2852 * x - 0.0330 * x**2 - 0.0867 * x**3
    red = -4.0618 + 10.0187 * x - 9.0532 * x**2 + 2.6089 * x**3
    delta = np.where(x <= 1.2, blue, red)
    delta = np.where((x < -0.15) | (x > 1.7), np.nan, delta)
    return rp + delta


def rvs_snr_per_pixel(grvs, n_transits=1, *, constants: RVSConstants = RVS_CONSTANTS):
    """Signal-to-noise per RVS detector pixel after co-adding ``n_transits`` transits.

    ``S / sqrt(S + B + R)`` with the source electrons ``S``, the background ``B`` and the
    read noise ``R`` accumulated over ``ccds_per_transit * n_transits`` CCD crossings: the
    S/N of the mean flux of the band in one detector pixel, as
    :class:`albireo.simulate.InstrumentSpec` takes ``snr`` under ``shot_noise``.

    Parameters
    ----------
    grvs
        G_RVS magnitude, scalar or array (:func:`predict_grvs` when the catalogue lacks it).
    n_transits
        Transits co-added; 1 for an epoch spectrum. Broadcasts against ``grvs``.
    constants
        The instrument constants.

    Returns
    -------
    numpy.ndarray
        S/N per detector pixel.

    Examples
    --------
    >>> float(np.round(rvs_snr_per_pixel(10.54), 1))
    13.1
    """
    c = constants
    grvs = np.asarray(grvs, dtype=np.float64)
    n = np.asarray(n_transits, dtype=np.float64)
    crossings = c.ccds_per_transit * n
    source = 10.0 ** ((c.zero_point - grvs) / 2.5) * c.exposure_s * crossings
    source = source * (c.pixel_nm / c.band_nm)
    background = c.background_e * crossings * c.ac_pixels
    ac_samples = np.where(grvs <= c.window_class_limit, c.ac_pixels, 1)
    read = c.read_noise_e**2 * crossings * ac_samples
    return source / np.sqrt(source + background + read)


def rvs_lsf_sigma_kms(resolving_power: float = RVS_RESOLVING_POWER) -> float:
    """Gaussian sigma of the RVS line-spread function in km/s, ``c / (R 2 sqrt(2 ln 2))``.

    The DPAC line-spread model is an eight-term basis expansion rather than a Gaussian
    (DR3 documentation §6.3.4), and its departure from a Gaussian is not published. The
    RVS literature convolves with a Gaussian of three pixels FWHM, and its
    wavelength dependence across the band is neglected in the in-flight model (Sartoretti
    et al. 2018 §5.4).
    """
    if not resolving_power > 0.0:
        raise ValueError("resolving_power must be positive")
    return C_KMS / float(resolving_power) * _FWHM_TO_SIGMA


def rvs_transit_resolving_powers(n_transits: int, *, seed: int = 0) -> np.ndarray:
    """Resolving powers for ``n_transits`` transits, one CCD row per transit.

    Each transit crosses one of the eight (telescope, row) combinations of
    :data:`RVS_CCD_RESOLVING_POWERS`, drawn uniformly, and its epoch spectrum combines
    the row's three strips, so the transit takes the row's mean resolving power. The
    result spans 11,146 to 12,396. Passed as ``simulate_rvs_dataset(resolving_power=...)``,
    it sets realistic line-spread widths, which an analysis declares as the nominal 11,500.
    """
    if n_transits < 1:
        raise ValueError("n_transits must be positive")
    rng = np.random.default_rng(seed)
    rows = [float(np.mean(v)) for v in RVS_CCD_RESOLVING_POWERS.values()]
    return np.asarray(rows, dtype=np.float64)[rng.integers(0, len(rows), size=int(n_transits))]


def quadrature_sigma_kms(
    library_resolving_power: float | None, resolving_power: float = RVS_RESOLVING_POWER
) -> float:
    """The Gaussian width that broadens a library from its own resolving power to the RVS's.

    A published grid is already broadened to its resolving power, so the simulator applies
    only the difference in quadrature, ``sqrt(sigma_RVS^2 - sigma_lib^2)``. A fit that
    declares the library's resolving power on its templates makes the same correction
    (:meth:`albireo.todcor.Template.from_library`). ``None`` means an intrinsic-resolution
    grid, and the whole RVS width is returned.

    Raises
    ------
    ValueError
        If the library is coarser than the instrument, which cannot be undone.
    """
    target = rvs_lsf_sigma_kms(resolving_power)
    if library_resolving_power is None:
        return target
    have = rvs_lsf_sigma_kms(library_resolving_power)
    if have >= target:
        raise ValueError(
            f"a library at R = {library_resolving_power:g} is at or below the RVS resolving "
            f"power {resolving_power:g}; it cannot be broadened to it"
        )
    return math.sqrt(target**2 - have**2)


# ---------------------------------------------------------------------------
# Grids and products
# ---------------------------------------------------------------------------


def rvs_detector_grid(step: float = RVS_DETECTOR_STEP, band=RVS_BAND) -> np.ndarray:
    """The detector pixel centres across the band, ``lo + step * k`` while below ``hi``."""
    lo, hi = float(band[0]), float(band[1])
    n = math.ceil((hi - lo) / step - 1e-9)
    return lo + float(step) * np.arange(n)


@dataclasses.dataclass(frozen=True)
class RVSProduct:
    """One delivered RVS product: the grid the archive resamples onto.

    Attributes
    ----------
    name
        A short identifier written into every record.
    wave
        Delivered wavelength grid, vacuum Angstrom, strictly increasing.
    detector_step
        Detector dispersion in Angstrom, the grid the noise is drawn on.
    snr_window
        The window over which the reference flux of the S/N is averaged (a 50 Angstrom
        window centred on the band, as in the reference implementation).
    instrument
        The instrument key of the delivered epochs.
    description
        One line for the reports.
    """

    name: str
    wave: np.ndarray
    detector_step: float = RVS_DETECTOR_STEP
    snr_window: tuple[float, float] = (8555.0, 8605.0)
    instrument: str = "RVS"
    description: str = ""

    def __post_init__(self) -> None:
        wave = np.asarray(self.wave, dtype=np.float64)
        if wave.ndim != 1 or wave.size < 2 or np.any(np.diff(wave) <= 0.0):
            raise ValueError("wave must be 1-D, strictly increasing, with at least 2 samples")
        object.__setattr__(self, "wave", wave)

    @property
    def n_pixels(self) -> int:
        return int(self.wave.size)

    @property
    def step(self) -> float:
        """The delivered sampling in Angstrom (the median step)."""
        return float(np.median(np.diff(self.wave)))


RVS_DR3_MEAN = RVSProduct(
    name="dr3-mean",
    wave=8460.0 + 0.1 * np.arange(2401),
    description="DR3 mean-spectrum sampling: 0.01 nm, 2401 samples from 846.0 to 870.0 nm",
)
"""The DR3 ``rvs_mean_spectrum`` sampling: 846 to 870 nm inclusive in steps of 0.01 nm,
2401 elements (the DR3 data model). The reference implementation's ``GAIA_RVS`` preset
builds it with ``np.arange`` and so stops one sample short, at 869.9 nm. The real DR3
product is in the source rest frame and is not a disentangling dataset; the sampling is
kept for the comparison with the notebook and for the noise-correlation study."""

RVS_DR4_EPOCH = RVSProduct(
    name="dr4-epoch",
    wave=8460.0 + 0.25 * np.arange(961),
    description="DR4 epoch-spectrum sampling: 0.025 nm, 961 samples from 846.0 to 870.0 nm",
)
"""The DR4 ``rvs_epoch_spectrum`` grid from ESA's draft data model of 2026-06-26: 961
samples, 846.0 to 870.0 nm inclusive, step 0.025 nm, barycentric frame and normalised, one
spectrum per field-of-view transit with its three CCDs combined. The ESA DR4 content page
states the last three. The grid is from the draft, which warns that the table is still
under development, so it is to be re-checked against the release."""


def rvs_model_grid(
    v_max_kms: float,
    *,
    dv_kms: float = 2.0,
    vsini_max_kms: float = 0.0,
    lsf_sigma_kms: float | None = None,
    band=RVS_BAND,
    extra_pixels: int = 8,
    length_multiple: int = 1,
) -> LogGrid:
    """A model grid covering the band with the margin the shifted components need.

    The margin is the largest shift any component takes (``v_max_kms``, a semi-amplitude
    plus the systemic velocity plus headroom), the rotational half-width, and four
    line-spread sigmas. A component shifted to the edge of its range then still covers
    the detector grid without zero-fill.

    The number of pixels follows from the margin, so two systems with different velocity
    amplitudes have grids of different lengths, and JAX compiles every array operation
    again for each new length. ``length_multiple`` rounds the number of pixels up to a
    multiple by adding pixels at the red end. The first pixel, the pixel width and every
    pixel the unrounded grid has are unchanged, the added pixels lie beyond the margin, and
    the spectra simulated on the two grids are equal on the detector. A population
    simulated with a multiple of a few hundred pixels has a few grid lengths
    (:func:`albireo.benchmark.simulate_system` uses 512).

    Parameters
    ----------
    v_max_kms
        Largest absolute radial velocity of any component, km/s.
    dv_kms
        Grid pixel in km/s; 2 km/s samples the 11 km/s RVS sigma five times.
    vsini_max_kms
        Largest projected rotational velocity applied to a component.
    lsf_sigma_kms
        Line-spread sigma to reserve; default the RVS's.
    band
        The band to cover, vacuum Angstrom.
    extra_pixels
        Slack on each side.
    length_multiple
        Round the number of pixels up to a multiple of this number, at the red end.
        Default 1: the grid ends where the margin does.
    """
    if v_max_kms < 0.0 or vsini_max_kms < 0.0:
        raise ValueError("v_max_kms and vsini_max_kms must be non-negative")
    if int(length_multiple) != length_multiple or length_multiple < 1:
        raise ValueError(f"length_multiple must be a positive integer; got {length_multiple!r}")
    sigma = rvs_lsf_sigma_kms() if lsf_sigma_kms is None else float(lsf_sigma_kms)
    dx = float(log_doppler_shift(dv_kms))
    margin = (
        abs(float(log_doppler_shift(v_max_kms + vsini_max_kms)))
        + (math.ceil(4.0 * sigma / dv_kms) + int(extra_pixels)) * dx
    )
    lo = math.exp(math.log(float(band[0])) - margin)
    hi = math.exp(math.log(float(band[1])) + margin)
    grid = LogGrid.from_wavelength_range(lo, hi, dv_kms)
    multiple = int(length_multiple)
    if grid.n % multiple:
        grid = dataclasses.replace(grid, n=grid.n + multiple - grid.n % multiple)
    return grid


def rvs_components(
    library,
    labels: Sequence[Mapping[str, float]],
    grid: LogGrid,
    *,
    vsini_kms: Sequence[float] | None = None,
    epsilon: float = 0.6,
) -> list[np.ndarray]:
    """Component deviation spectra on ``grid`` from a library, on the vacuum scale.

    :func:`albireo.simulate.library_component` for each entry of ``labels``, with the
    medium fixed to vacuum because that is the scale Gaia delivers.

    Parameters
    ----------
    library
        A :class:`~albireo.library.SpectralLibrary` covering the band, e.g.
        ``fetch_library("bosz2024-fgk-rvs")``.
    labels
        One mapping per component with every axis the library has (``teff``, ``logg``,
        ``mh`` for BOSZ).
    grid
        The model grid, from :func:`rvs_model_grid`.
    vsini_kms
        Projected rotational velocity per component; default none.
    epsilon
        Linear limb-darkening coefficient of the rotation kernel.
    """
    from albireo.simulate import library_component

    if vsini_kms is None:
        vsini_kms = [0.0] * len(labels)
    if len(vsini_kms) != len(labels):
        raise ValueError("vsini_kms needs one entry per component")
    return [
        library_component(
            library, dict(lab), grid, medium="vacuum", vsini_kms=float(v), epsilon=epsilon
        )
        for lab, v in zip(labels, vsini_kms, strict=True)
    ]


# ---------------------------------------------------------------------------
# Epoch times
# ---------------------------------------------------------------------------


def uniform_phase_times(period: float, n: int, *, start: float = 0.0) -> np.ndarray:
    """``n`` epochs evenly spaced over one period from ``start``, ``start + period`` included.

    The reference implementation's ``init_rv_orbit(uniform_phase=True)``.
    """
    if n < 1:
        raise ValueError("n must be positive")
    return float(start) + np.linspace(0.0, float(period), int(n))


def random_times(n: int, span_days: float, *, start: float = 0.0, seed: int = 0) -> np.ndarray:
    """``n`` epochs drawn without replacement from ``4 n`` evenly spaced slots over the span.

    The reference implementation's ``init_rv_orbit(uniform_phase=False, t_span=...)``.
    """
    if n < 1:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    slots = np.linspace(0.0, float(span_days), 4 * int(n))
    return float(start) + np.sort(rng.choice(slots, size=int(n), replace=False))


_GAIA_SPIN_DAYS = 6.0 / 24.0
_GAIA_FOV_GAP_DAYS = 106.5 / 360.0 * _GAIA_SPIN_DAYS  # 106.5 min: the basic angle
_VISIBILITY_GAP_DAYS = 4.0  # the gap that separates two visibility periods (DR3 data model)
_DR3_START_BJD = 2456863.94  # 2014-07-25, the start of the nominal mission
_DR3_SPAN_DAYS = 1038.0  # to 2017-05-28, the DR3 data span (34 months)
_DR4_SPAN_DAYS = 1998.0  # to 2020-01-20, the DR4 data span (66 months)
_DR4_START_BJD = _DR3_START_BJD

RVS_SPANS: dict[str, tuple[float, float]] = {
    "dr3": (_DR3_START_BJD, _DR3_SPAN_DAYS),
    "dr4": (_DR4_START_BJD, _DR4_SPAN_DAYS),
}
"""``(start BJD, span in days)`` of the data in each release: 34 months for DR3 and 66
for DR4, both from 25 July 2014 (Katz et al. 2023 §3.8; the ESA DR4 content page)."""

_DR3_TRANSITS_MEDIAN = 18.0
_DR3_TRANSITS_P10 = 8.0
_DR3_TRANSITS_P90 = 30.0
_DR3_TRANSITS_MEAN = 18.7
_TRANSITS_PER_VISIBILITY_PERIOD = 1.48
_ECLIPTIC_LATITUDE_DEG = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 90.0)
_ECLIPTIC_LATITUDE_TRANSITS = (12.5, 13.35, 15.3, 17.8, 25.0, 25.2, 19.05, 26.2)


def rvs_transit_count(
    n: int = 1, *, ecl_lat_deg=None, release: str = "dr4", seed: int = 0
) -> np.ndarray:
    """Draw RVS transit counts as the archive records them.

    DR3's ``rv_nb_transits`` over 34 months has a median of 18 and 10th and 90th
    percentiles of 8 and 30 (Katz et al. 2023 §5.2 for the median; the percentiles from a
    uniform subsample of the archive), which a split log-normal reproduces. The count
    scales with the data span, so the DR4 draw is the DR3 one times 66/34. It depends on
    ecliptic latitude through the scanning law: the mean is 12.5 on the ecliptic, about 25
    at 40-50 degrees and 26 at the poles (the archive's own means in 10-degree bins, the
    pole excess being the first month's ecliptic-pole scanning). Given a latitude, the draw
    is scaled by that mean over the sky average of 18.7. Crowding losses in the bulge and
    the disc are not modelled.

    Parameters
    ----------
    n
        How many counts to draw.
    ecl_lat_deg
        Ecliptic latitude in degrees, a scalar or ``(n,)``; ``None`` draws from the
        all-sky distribution.
    release
        ``"dr3"`` or ``"dr4"``, which sets the span the count refers to.
    seed
        Seed.

    Returns
    -------
    numpy.ndarray
        Integer counts, ``(n,)``, at least 2 (the pipeline's minimum).
    """
    if release not in RVS_SPANS:
        raise ValueError(f"release must be one of {sorted(RVS_SPANS)}; got {release!r}")
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, 1.0, size=int(n))
    s_up = math.log(_DR3_TRANSITS_P90 / _DR3_TRANSITS_MEDIAN) / 1.2816
    s_lo = math.log(_DR3_TRANSITS_MEDIAN / _DR3_TRANSITS_P10) / 1.2816
    count = _DR3_TRANSITS_MEDIAN * np.exp(np.where(u > 0.0, s_up, s_lo) * u)
    count = count * RVS_SPANS[release][1] / _DR3_SPAN_DAYS
    if ecl_lat_deg is not None:
        lat = np.abs(np.broadcast_to(np.asarray(ecl_lat_deg, dtype=np.float64), (int(n),)))
        scale = np.interp(lat, _ECLIPTIC_LATITUDE_DEG, _ECLIPTIC_LATITUDE_TRANSITS)
        count = count * scale / _DR3_TRANSITS_MEAN
    return np.maximum(np.rint(count).astype(int), 2)


def rvs_transit_times(
    n_transits: int,
    *,
    start_bjd: float = _DR4_START_BJD,
    span_days: float = _DR4_SPAN_DAYS,
    transits_per_period: float = _TRANSITS_PER_VISIBILITY_PERIOD,
    seed: int = 0,
) -> np.ndarray:
    """Transit times with the structure of the Gaia scanning law, without its phase.

    Gaia spins once in six hours with its two fields of view 106.5 degrees apart, so a
    source crossing the scanned band is observed 106.5 minutes apart by the two fields, then
    4.23 hours later on the next spin. The precession of the spin axis (63 days) moves the
    scanned band away and back, so the transits of one source come in visibility periods
    separated by at least four days. The archive records 1.48 RVS transits per visibility
    period on average (13 periods for a median of 18 transits in DR3), so the number of
    independent orbital phases sampled for a binary scales with the periods, not the
    transits.

    The function draws that structure: ``n_transits / transits_per_period`` visibility
    periods placed uniformly at random over the span and at least four days apart, each
    with one, two or three transits on the 106.5-minute / 4.23-hour ladder, with the
    total adjusted to exactly ``n_transits``. It is an approximation: the phase of the
    scanning law is not reproduced, so the epochs are correct statistically but not
    individually, and the latitude dependence enters only through the count
    (:func:`rvs_transit_count`). Real transit times for a sky position come from the Gaia
    Observation Forecast Tool through :func:`gost_transits` and
    :func:`rvs_transit_times_from_gost`.

    Parameters
    ----------
    n_transits
        Number of transits to place.
    start_bjd, span_days
        The window; the defaults are the DR4 span (:data:`RVS_SPANS`).
    transits_per_period
        Mean number of transits per visibility period.
    seed
        Seed.

    Returns
    -------
    numpy.ndarray
        Sorted BJD of the transits, ``(n_transits,)``.
    """
    n = int(n_transits)
    if n < 1:
        raise ValueError("n_transits must be positive")
    if not transits_per_period >= 1.0:
        raise ValueError("transits_per_period must be at least 1")
    rng = np.random.default_rng(seed)
    n_periods = max(1, round(n / transits_per_period))
    free = float(span_days) - _VISIBILITY_GAP_DAYS * (n_periods - 1)
    if free <= 0.0:
        raise ValueError(
            f"{n_periods} visibility periods do not fit in {span_days:g} days at four days apart"
        )
    # Uniform placement subject to the minimum separation: draw the starts uniformly in
    # the span shortened by the gaps, then shift each one by the gaps before it.
    starts = np.sort(rng.uniform(0.0, free, size=n_periods))
    starts = starts + _VISIBILITY_GAP_DAYS * np.arange(n_periods)
    # One, two or three transits per period, mean transits_per_period; the ladder is the
    # first field of view, the second 106.5 minutes later, the first again one spin later.
    excess = float(transits_per_period) - 1.0
    p_three = max(0.0, min(excess / 2.0, 0.2))
    p_two = max(0.0, min(excess - 2.0 * p_three, 1.0 - p_three))
    ladder = np.array([0.0, _GAIA_FOV_GAP_DAYS, _GAIA_SPIN_DAYS])
    counts = (
        1
        + (rng.uniform(size=n_periods) < p_two + p_three)
        + (rng.uniform(size=n_periods) < p_three / max(p_two + p_three, 1e-12))
        * (rng.uniform(size=n_periods) < p_two + p_three)
    )
    counts = np.minimum(counts.astype(int), 3)
    # Adjust to the exact total: add to or remove from randomly chosen periods.
    while counts.sum() < n:
        k = rng.integers(0, n_periods)
        if counts[k] < 3:
            counts[k] += 1
        elif np.all(counts >= 3):
            # Every period is full: open a new one at a random time in the span.
            starts = np.append(starts, rng.uniform(0.0, float(span_days)))
            counts = np.append(counts, 1)
            n_periods += 1
    while counts.sum() > n:
        k = rng.integers(0, n_periods)
        if counts[k] > 1:
            counts[k] -= 1
        elif np.all(counts <= 1):
            counts[k] = 0
    times = [
        float(start_bjd) + t0 + ladder[j]
        for t0, c in zip(starts, counts, strict=True)
        for j in range(c)
    ]
    return np.sort(np.asarray(times, dtype=np.float64))


# ---------------------------------------------------------------------------
# Real transit times: the Gaia Observation Forecast Tool
# ---------------------------------------------------------------------------

GOST_ENDPOINT = "https://gaia.esac.esa.int/gost/ObjVisSAP/gaiaobjvisap"
"""The anonymous programmatic interface of the Gaia Observation Forecast Tool
(https://gaia.esac.esa.int/gost/): ESA's implementation of the IVOA Object Visibility
Simple Access Protocol, queried as ``?s_ra=<deg>&s_dec=<deg>`` with the optional
``t_min``, ``t_max`` (both MJD), ``MAXREC`` and ``RESPONSEFORMAT``. It returns a
VOTable of one row per field-of-view crossing (GOST user manual GAIA-CU1-UG-ESAC-JFH-005,
issue 08, 2022-08-23)."""

RVS_CCD_ROWS: tuple[int, ...] = (4, 5, 6, 7)
"""The CCD rows of the focal plane the Radial Velocity Spectrometer occupies, of the seven
the astrometric field spans. The RVS is twelve CCDs in three strips over four rows
(Cropper et al. 2018 §3.3.7), so a transit is recorded only when its across-scan position
falls on one of these four. The Gaia DR3 documentation states that the spectroscopic
instrument "is only served by 4 of the 7 Video Processing Units", which reduces the
transit count by a factor 4/7 against the astrometric one."""

GOST_USABLE_FRACTION: float = 0.78
"""Fraction of the RVS transits GOST predicts that yield a usable epoch spectrum.

Three independent numbers are consistent with it. GOST states that "the probability of
the data of the target being received on the ground at the indicated time is about 80%".
Katz et al. (2023) §2 state that dead time and the processing filters reduce the effective
number of transits by about 25%. The published counts, 8 RVS transits per star per year
(Katz et al. 2023 §2) over the 34 months of DR3, predict 22.7 against the published median
of 18, a ratio of 0.79. The fraction applies to a bright, isolated star. For G_RVS fainter
than 14, in fields denser than about 35,000 sources per square degree, or during Galactic
Plane Scanning, the fraction is lower and is set by onboard window allocation rather than
by dead time. None of that is modelled here."""

_MJD_TO_JD = 2400000.5

# The optional columns, lower-cased. ObjVisSAP publishes none of them; the names are the
# ones the GOST web form's CSV export uses, so a table built from such a file is read.
_GOST_ROW_COLUMNS = ("ccd_row", "ccdrow")
_GOST_SCAN_COLUMNS = ("scan_angle_deg", "scan_angle", "scanangle")
_GOST_FOV_NAMES = {"1": 1, "fovp": 1, "preceding": 1, "2": 2, "fovf": 2, "following": 2}


@dataclasses.dataclass(frozen=True)
class GostTransits:
    """The transits GOST forecasts for one sky position.

    Attributes
    ----------
    bjd
        Barycentric Julian date of each transit, float64 and non-decreasing, on the TCB
        scale Gaia uses throughout its archive; see :func:`gost_transits` for the
        conversion from the service's own columns and its precision.
    ra_deg, dec_deg
        The ICRS position the forecast was made for.
    source
        The URL the rows came from.
    fov
        Field of view of each transit, 1 (preceding) or 2 (following), or ``None``.
    ccd_row
        Across-scan CCD row of each transit, 1 to 7, or ``None``. A transit is an RVS
        transit when its row is one of :data:`RVS_CCD_ROWS`.
    scan_angle_deg
        Position angle of the scan at each transit, or ``None``.

    Notes
    -----
    The anonymous ObjVisSAP endpoint publishes none of ``fov``, ``ccd_row`` and
    ``scan_angle_deg``: its six columns are ``t_validity``, ``t_start``,
    ``t_start_gaia_timestamp``, ``t_stop``, ``t_stop_gaia_timestamp`` and ``t_visibility``,
    one row per field-of-view crossing. The three are in the CSV the interactive GOST form
    exports (columns ``Fov[FovP/FovF]``, ``CcdRow[1-7]`` and ``scanAngle[rad]``), so they
    are fields here for a caller who has such a file. :func:`rvs_transit_times_from_gost`
    draws the CCD row when it is absent.
    """

    bjd: np.ndarray
    ra_deg: float
    dec_deg: float
    source: str
    fov: np.ndarray | None = None
    ccd_row: np.ndarray | None = None
    scan_angle_deg: np.ndarray | None = None

    def __post_init__(self) -> None:
        bjd = np.asarray(self.bjd, dtype=np.float64)
        if bjd.ndim != 1:
            raise ValueError("bjd must be one-dimensional")
        if bjd.size and np.any(np.diff(bjd) < 0.0):
            raise ValueError("bjd must be sorted")
        object.__setattr__(self, "bjd", bjd)
        for name, dtype in (("fov", int), ("ccd_row", int), ("scan_angle_deg", np.float64)):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value, dtype=dtype)
            if array.shape != bjd.shape:
                raise ValueError(f"{name} must have one entry per transit, got {array.shape}")
            object.__setattr__(self, name, array)

    def __len__(self) -> int:
        return int(self.bjd.size)

    @property
    def n(self) -> int:
        """Number of transits."""
        return int(self.bjd.size)


def _gost_query(ra_deg, dec_deg, t_min_mjd, t_max_mjd, max_records) -> str:
    params: dict[str, str] = {"s_ra": f"{float(ra_deg):.6f}", "s_dec": f"{float(dec_deg):.6f}"}
    if t_min_mjd is not None:
        params["t_min"] = f"{float(t_min_mjd):.6f}"
    if t_max_mjd is not None:
        params["t_max"] = f"{float(t_max_mjd):.6f}"
    if max_records is not None:
        params["MAXREC"] = str(int(max_records))
    return f"{GOST_ENDPOINT}?{urllib.parse.urlencode(params)}"


def _gost_cache_name(ra_deg, dec_deg, t_min_mjd, t_max_mjd, max_records) -> str:
    """The cache file name of one query, built from every parameter that changes the response."""

    def number(value, sign: str = "") -> str:
        if value is None:
            return "none"
        # The fixed format always writes a point, so the strip does not remove the zeros
        # of the integer part: 60800.0000 becomes 60800, not 608.
        return f"{float(value):{sign}.4f}".rstrip("0").rstrip(".")

    return (
        f"objvissap_ra{number(ra_deg)}_dec{number(dec_deg, '+')}"
        f"_t{number(t_min_mjd)}-{number(t_max_mjd)}"
        f"_n{'none' if max_records is None else int(max_records)}.vot"
    )


def _strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _votable_table(payload: bytes):
    """Field names, field units and rows of the first VOTable table that has fields.

    ``xml.etree`` is used rather than astropy, because the reader must work in an
    installation whose only dependencies are numpy and jax. The service's second RESOURCE
    describes the protocol and has no FIELD, so the first table with fields is the results
    table.
    """
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as error:
        raise ValueError(
            f"the response is not XML ({error}); the service answers some bad queries with "
            "an HTML page"
        ) from error
    for info in root.iter():
        if _strip_namespace(info.tag) == "INFO" and info.get("name") == "QUERY_STATUS":
            status = (info.get("value") or "").strip().upper()
            if status and status != "OK":
                raise RuntimeError(
                    f"the GOST query failed with QUERY_STATUS {status}: {(info.text or '').strip()}"
                )
    for table in root.iter():
        if _strip_namespace(table.tag) != "TABLE":
            continue
        fields = [child for child in table if _strip_namespace(child.tag) == "FIELD"]
        if not fields:
            continue
        names = [(field.get("name") or "").strip().lower() for field in fields]
        units = [(field.get("unit") or "").strip().lower() for field in fields]
        rows = [
            [cell.text for cell in tr if _strip_namespace(cell.tag) == "TD"]
            for tr in table.iter()
            if _strip_namespace(tr.tag) == "TR"
        ]
        return names, units, rows
    raise ValueError("the response holds no VOTable table with fields")


def _votable_column(names, rows, keys) -> np.ndarray | None:
    """The first column named in ``keys``, as floats with ``nan`` for an empty cell."""
    for key in keys:
        if key not in names:
            continue
        index = names.index(key)
        values = []
        for row in rows:
            cell = row[index] if index < len(row) else None
            cell = "" if cell is None else cell.strip()
            values.append(np.nan if not cell else float(cell))
        return np.asarray(values, dtype=np.float64)
    return None


def _votable_fov(names, rows) -> np.ndarray | None:
    """The field of view as 1 (preceding) or 2 (following), numeric or named.

    A table that gives it as 1 and 2 is read as a number; one that gives it as the GOST
    web form does, ``FovP`` and ``FovF``, is read by name.
    """
    if "fov" not in names:
        return None
    index = names.index("fov")
    values = []
    for row in rows:
        cell = row[index] if index < len(row) else None
        cell = "" if cell is None else cell.strip().lower()
        try:
            values.append(round(float(cell)))
        except ValueError:
            if cell not in _GOST_FOV_NAMES:
                raise ValueError(f"unreadable field of view {cell!r}") from None
            values.append(_GOST_FOV_NAMES[cell])
    return np.asarray(values, dtype=int)


def _parse_gost(payload: bytes, *, ra_deg: float, dec_deg: float, source: str) -> GostTransits:
    names, units, rows = _votable_table(payload)
    start = _votable_column(names, rows, ("t_start",))
    if start is None:
        raise ValueError(f"the response has no t_start column; its columns are {', '.join(names)}")
    stop = _votable_column(names, rows, ("t_stop",))
    # The transit time is the middle of the field-of-view crossing, which t_start and
    # t_stop bracket; without t_stop the start is used.
    mjd = start if stop is None else 0.5 * (start + stop)
    bjd = mjd + _MJD_TO_JD
    fov = _votable_fov(names, rows)
    ccd_row = _votable_column(names, rows, _GOST_ROW_COLUMNS)
    scan = _votable_column(names, rows, _GOST_SCAN_COLUMNS)
    if scan is not None:
        for key in _GOST_SCAN_COLUMNS:
            if key in names and units[names.index(key)] == "rad":
                scan = np.degrees(scan)
                break
    order = np.argsort(bjd, kind="stable")
    return GostTransits(
        bjd=bjd[order],
        ra_deg=float(ra_deg),
        dec_deg=float(dec_deg),
        source=source,
        fov=None if fov is None else fov[order],
        ccd_row=None if ccd_row is None else np.rint(ccd_row[order]).astype(int),
        scan_angle_deg=None if scan is None else scan[order],
    )


def gost_transits(
    ra_deg: float,
    dec_deg: float,
    *,
    t_min_mjd: float | None = None,
    t_max_mjd: float | None = None,
    max_records: int | None = 10_000,
    cache: bool = True,
    timeout: float = 60.0,
) -> GostTransits:
    """The transits the Gaia Observation Forecast Tool predicts for a sky position.

    One GET against :data:`GOST_ENDPOINT` with :mod:`urllib`, parsed with
    :mod:`xml.etree`, cached raw under ``cache_dir() / "gost"`` under a name built from
    every parameter that changes the response. The cache is read before the network, so a
    repeated position is fetched once.

    The forecast uses the routine nominal scanning law, not the attitude as flown. The
    manual (§7) states that GOST "will compute the transit crossing the FoVs based on the
    routine nominal scanning law" and that "this is not a full guarantee of getting the
    transits observed". The service returns predictions past the end of science operations
    on 15 January 2025. The forecast does not state whether a transit was received on the
    ground (:data:`GOST_USABLE_FRACTION`), nor which CCD row the source crossed, which
    determines whether the RVS recorded it (:func:`rvs_transit_times_from_gost` applies
    both).

    The service publishes ``t_start`` and ``t_stop``, the bounds of the field-of-view
    crossing, as "Barycentric MJD in TCB". This function returns their mid-point plus
    2400000.5, the barycentric Julian date on the same TCB scale that Gaia's archive uses
    for its epochs. Three effects limit the precision. The MJD to JD offset is a
    definition and exact, but a double near 2.46e6 resolves 4e-5 s. The crossing lasts
    40.5 s, so the mid-point is that far from either bound. TCB runs ahead of TDB by about
    20 s in this era, growing by 0.49 s per year, which is not corrected for. All three are
    far below the accuracy of the nominal law.

    Parameters
    ----------
    ra_deg, dec_deg
        ICRS position in degrees.
    t_min_mjd, t_max_mjd
        Window in MJD; ``None`` leaves the service's own default, which at the time of
        writing runs from 2014-07-25 to 2025-07-01.
    max_records
        ``MAXREC``; ``None`` omits it.
    cache
        Read and write the raw response under ``cache_dir() / "gost"``. ``False`` fetches
        and does not store.
    timeout
        Seconds for the request.

    Returns
    -------
    GostTransits

    Raises
    ------
    RuntimeError
        If the request fails and nothing is cached; the message names the endpoint and the
        cache path the response would have been written to.

    Examples
    --------
    >>> transits = gost_transits(45.0, 20.0)  # doctest: +SKIP
    >>> len(transits)  # doctest: +SKIP
    142
    """
    url = _gost_query(ra_deg, dec_deg, t_min_mjd, t_max_mjd, max_records)
    name = _gost_cache_name(ra_deg, dec_deg, t_min_mjd, t_max_mjd, max_records)
    path = Path(cache_dir()) / "gost" / name
    if cache and path.is_file():
        return _parse_gost(path.read_bytes(), ra_deg=ra_deg, dec_deg=dec_deg, source=url)
    try:
        payload = _fetch_gost(url, timeout=timeout)
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        raise RuntimeError(
            f"could not reach the Gaia Observation Forecast Tool at {GOST_ENDPOINT} ({error}), "
            f"and nothing is cached at {path}. Fetch the position once with a network "
            "connection, or copy a response from another machine to that path."
        ) from error
    # Parsed before it is cached, so that an error page is never written to the cache and
    # read back on every later call.
    transits = _parse_gost(payload, ra_deg=ra_deg, dec_deg=dec_deg, source=url)
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return transits


def _fetch_gost(url: str, *, timeout: float) -> bytes:
    """The one network call, alone so that a test can replace it."""
    request = urllib.request.Request(url, headers={"User-Agent": "albireo (gaia)"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def rvs_transit_times_from_gost(
    transits: GostTransits,
    *,
    release: str = "dr4",
    usable_fraction: float = GOST_USABLE_FRACTION,
    seed: int = 0,
) -> np.ndarray:
    """RVS epoch times from a GOST forecast: the release span, the RVS rows, the losses.

    Three steps, in order:

    1. Keep the transits inside the release's data span (:data:`RVS_SPANS`: 34 months from
       25 July 2014 for DR3, 66 for DR4). GOST forecasts the whole mission, including the
       years after a release's cut-off.
    2. Keep the transits the RVS records. The instrument covers four of the seven CCD rows
       (:data:`RVS_CCD_ROWS`), so 4/7 of the field-of-view transits reach it. When
       ``transits.ccd_row`` is given it is used. The anonymous GOST endpoint does not
       publish it, and then the row is drawn uniformly from 1 to 7 for each transit. The
       draw is an approximation: the across-scan position that sets the row drifts slowly
       with the scanning law and is correlated within a visibility period, so an
       independent draw per transit reproduces the number of RVS epochs but not their
       clumping.
    3. Thin what is left to ``usable_fraction`` (:data:`GOST_USABLE_FRACTION`), the
       fraction of predicted transits that reach the ground and pass the processing
       filters.

    The result is the epoch list of one realisation, not a forecast of which nights Gaia
    observed. Steps 2 and 3 are random, and step 1's input is the nominal law rather than
    the attitude as flown.

    Parameters
    ----------
    transits
        A :class:`GostTransits`, from :func:`gost_transits`.
    release
        ``"dr3"`` or ``"dr4"``.
    usable_fraction
        Fraction kept in step 3; 1 keeps every transit.
    seed
        Seed for steps 2 and 3.

    Returns
    -------
    numpy.ndarray
        Sorted BJD, at most ``len(transits)`` of them and possibly none.
    """
    if release not in RVS_SPANS:
        raise ValueError(f"release must be one of {sorted(RVS_SPANS)}; got {release!r}")
    if not 0.0 < usable_fraction <= 1.0:
        raise ValueError(f"usable_fraction must lie in (0, 1]; got {usable_fraction!r}")
    start, span = RVS_SPANS[release]
    bjd = np.asarray(transits.bjd, dtype=np.float64)
    inside = (bjd >= start) & (bjd <= start + span)
    bjd = bjd[inside]
    rows = None if transits.ccd_row is None else np.asarray(transits.ccd_row)[inside]
    rng = np.random.default_rng(seed)
    if rows is None:
        rows = rng.integers(1, 8, size=bjd.size)
    bjd = bjd[np.isin(rows, np.asarray(RVS_CCD_ROWS))]
    if usable_fraction < 1.0:
        bjd = bjd[rng.uniform(size=bjd.size) < float(usable_fraction)]
    return np.sort(bjd)


# ---------------------------------------------------------------------------
# Delivery: detector grid -> archive grid
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DeliveryRecord:
    """The effect of :func:`deliver` on the noise, for the truth record and the reports.

    Attributes
    ----------
    product
        The product delivered.
    sigma
        Per-epoch delivered noise sigma, propagated through the interpolation weights
        (the ``flux_error`` an archive would publish); ``inf`` at masked pixels.
    lag1
        Per-epoch mean lag-one correlation coefficient of the delivered noise, computed
        exactly from the weights and the detector variances. Zero would mean independent
        pixels; the archive's diagonal errors assume it.
    pixel_ratio
        Delivered pixels per detector pixel over the covered range, the factor by which a
        diagonal likelihood overcounts the samples.
    covered
        Per-epoch mask of delivered pixels that lie inside the detector grid.
    """

    product: RVSProduct
    sigma: tuple[np.ndarray, ...]
    lag1: np.ndarray
    pixel_ratio: float
    covered: tuple[np.ndarray, ...]


def _interpolation_weights(x_out: np.ndarray, x_in: np.ndarray):
    """Left index and right weight of a linear interpolation, and the coverage mask."""
    inside = (x_out >= x_in[0]) & (x_out <= x_in[-1])
    idx = np.searchsorted(x_in, x_out, side="right") - 1
    idx = np.clip(idx, 0, x_in.size - 2)
    span = x_in[idx + 1] - x_in[idx]
    t = np.clip((x_out - x_in[idx]) / span, 0.0, 1.0)
    return idx, t, inside


def deliver(dataset: Dataset, product: RVSProduct, *, lsf_sigma_kms=None):
    """Resample detector-grid epochs onto a product's grid the way the archive does.

    The flux is interpolated linearly (``np.interp``), the variance is propagated through
    the same weights, ``(1 - t)^2 v_i + t^2 v_{i+1}``, and the lag-one correlation is
    recorded. A delivered pixel outside the detector grid is masked (``ivar = 0``, flux 1)
    rather than filled. A detector pixel with ``ivar = 0`` masks every delivered pixel that
    is interpolated from it. The reference implementation assumes linear interpolation; the
    DR3 documentation states only "interpolated to a common wavelength array". The
    interpolation also smooths every line, by a variance of ``Delta_det^2 / 6`` on average,
    and the delivered samples have the detector pixel's width rather than their own. The
    width declared on the delivered epochs includes neither, and
    :func:`rvs_delivered_sigma_kms` gives the width of a delivered line.

    Parameters
    ----------
    dataset
        Epochs on the detector grid, e.g. from :func:`albireo.simulate.simulate_dataset`
        with :func:`rvs_detector_grid` as the instrument grid. The frame is kept.
    product
        The delivered grid.
    lsf_sigma_kms
        The line-spread width to declare on the delivered epochs: one value for all, or
        one per epoch; default the nominal RVS width.

    Returns
    -------
    (Dataset, DeliveryRecord)
        The delivered epochs, on the vacuum scale with ``product.instrument`` as their
        key, and the record.
    """
    if lsf_sigma_kms is None:
        declared = np.full(dataset.n_epochs, rvs_lsf_sigma_kms())
    else:
        declared = np.broadcast_to(
            np.asarray(lsf_sigma_kms, dtype=np.float64), (dataset.n_epochs,)
        ).copy()
    wave_out = product.wave
    epochs, sigmas, lag1s, covered = [], [], [], []
    ratios = []
    for j, epoch in enumerate(dataset):
        wave_in = np.asarray(epoch.wave, dtype=np.float64)
        flux_in = np.array(epoch.flux, dtype=np.float64)
        ivar_in = np.asarray(epoch.effective_ivar, dtype=np.float64)
        good_in = ivar_in > 0.0
        var_in = np.where(good_in, 1.0 / np.where(good_in, ivar_in, 1.0), np.inf)
        flux_in = np.where(good_in, flux_in, np.nan)

        idx, t, inside = _interpolation_weights(wave_out, wave_in)
        w0, w1 = 1.0 - t, t
        flux_out = w0 * flux_in[idx] + w1 * flux_in[idx + 1]
        var_out = w0**2 * var_in[idx] + w1**2 * var_in[idx + 1]
        # A weight of exactly zero on a masked neighbour must not mask the pixel.
        var_out = np.where(w1 == 0.0, var_in[idx], var_out)
        var_out = np.where(w0 == 0.0, var_in[idx + 1], var_out)
        flux_out = np.where(w1 == 0.0, flux_in[idx], flux_out)
        flux_out = np.where(w0 == 0.0, flux_in[idx + 1], flux_out)
        ok = inside & np.isfinite(flux_out) & np.isfinite(var_out) & (var_out > 0.0)
        flux_out = np.where(ok, flux_out, 1.0)
        ivar_out = np.where(ok, 1.0 / np.where(ok, var_out, 1.0), 0.0)
        sigma_out = np.where(ok, np.sqrt(np.where(ok, var_out, 1.0)), np.inf)

        # Lag-one covariance: delivered pixels k and k+1 share detector pixel idx[k] + 1
        # when idx[k+1] == idx[k] + 1, and both of idx[k], idx[k]+1 when idx[k+1] == idx[k].
        cov = np.zeros(wave_out.size - 1)
        same = idx[1:] == idx[:-1]
        step = idx[1:] == idx[:-1] + 1
        pair_ok = ok[:-1] & ok[1:]
        # Masked detector pixels have an infinite variance; their products are discarded
        # through pair_ok, so the arithmetic on them is silenced rather than guarded.
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            cov[same] = (
                w0[:-1] * w0[1:] * var_in[idx[:-1]] + w1[:-1] * w1[1:] * var_in[idx[:-1] + 1]
            )[same]
            cov[step] = (w1[:-1] * w0[1:] * var_in[idx[:-1] + 1])[step]
            rho = cov / np.sqrt(var_out[:-1] * var_out[1:])
        lag1s.append(float(np.mean(rho[pair_ok])) if pair_ok.any() else np.nan)
        n_detector_covered = float(np.sum((wave_in >= wave_out[0]) & (wave_in <= wave_out[-1])))
        ratios.append(float(np.sum(inside)) / max(n_detector_covered, 1.0))

        epochs.append(
            EpochData(
                wave=wave_out,
                flux=flux_out,
                ivar=ivar_out,
                bjd=epoch.bjd,
                v_bary=epoch.v_bary,
                instrument=product.instrument,
                medium="vacuum",
                lsf_sigma_kms=float(declared[j]),
            )
        )
        sigmas.append(sigma_out)
        covered.append(ok)
    record = DeliveryRecord(
        product=product,
        sigma=tuple(sigmas),
        lag1=np.asarray(lag1s, dtype=np.float64),
        pixel_ratio=float(np.mean(ratios)),
        covered=tuple(covered),
    )
    return Dataset(epochs=tuple(epochs), frame=dataset.frame), record


SIMULATION_VARIANCE_FACTOR: float = 5.0 / 12.0
"""The variance :func:`simulate_rvs_dataset` adds to every line, per squared model pixel.

Four discrete steps on the model grid contribute (:func:`rvs_delivered_sigma_kms` derives
each). The epoch comparison of the label fit takes the same steps and one more, its frame
shift, and compensates ``7/12`` (:meth:`albireo.Fit.epoch_statistics`)."""


def rvs_delivered_sigma_kms(
    product: RVSProduct = RVS_DR4_EPOCH,
    resolving_power: float = RVS_RESOLVING_POWER,
    *,
    simulation_dv_kms: float | None = None,
) -> float:
    """The Gaussian width in km/s of a line after :func:`deliver` has resampled it.

    Declare this width, not the nominal one, to an analysis whose operator integrates over
    the delivered pixels. It is the line-spread width with the variances of the delivery,
    and of the simulation when the epochs are simulated, added:

        sigma_eff^2 = sigma_R^2 + Delta_det^2 / 6 + (Delta_det^2 - Delta_prod^2) / 12
                      + (5/12) dv_sim^2,

    with ``sigma_R`` the line-spread width at ``resolving_power``
    (:func:`rvs_lsf_sigma_kms`), ``Delta_det`` the detector step and ``Delta_prod`` the
    delivered step, both in km/s at the centre of the product's band, and ``dv_sim`` the
    model-grid pixel the epochs were simulated on (``simulation_dv_kms``). The last term is
    left out when that argument is ``None``, as it is for archive data.

    **The interpolation.** ``deliver`` reads each delivered sample from the detector grid by
    linear interpolation, which smooths. A sample at a fraction ``t`` of the way from
    detector pixel ``k`` (at ``x_k``) to pixel ``k + 1``, with the detector step ``Delta``,
    is ``(1 - t) f(x_k) + t f(x_k + Delta)``. This is a kernel of two taps, at ``-t Delta``
    and ``(1 - t) Delta`` from the sample, with weights ``1 - t`` and ``t``. Its mean is
    ``(1 - t)(-t Delta) + t (1 - t) Delta = 0`` and its variance is
    ``(1 - t) t^2 Delta^2 + t (1 - t)^2 Delta^2 = t (1 - t) Delta^2``. For a line that is
    smooth on the scale of ``Delta`` a Taylor expansion to second order gives the sample as
    ``f + t (1 - t) Delta^2 f'' / 2``. This is the effect of a convolution with a kernel of
    that variance, so the variances of the line and of the interpolation add. The fraction
    ``t`` runs through every value across the band on the shipped products: the DR4 grid's
    0.25 Å steps against the 0.245 Å detector with a beat period of 49 samples (12.25 Å), and
    the DR3 grid's 0.1 Å steps through ``t = 20 k / 49`` modulo one. The mean of
    ``t (1 - t)`` over ``t`` uniform on ``[0, 1]`` is ``1/2 - 1/3 = 1/6``.

    **The pixel widths.** A delivered sample interpolates detector samples, each of which
    integrated the line over a detector pixel (a box of variance ``Delta_det^2 / 12``),
    while an analysis's operator integrates its model over the delivered pixel
    (``Delta_prod^2 / 12``). The data therefore have a variance
    ``(Delta_det^2 - Delta_prod^2) / 12`` that the operator does not: -0.25 km^2/s^2 on the
    DR4 grid, whose pixel is the wider, and +5.09 km^2/s^2 on the DR3 grid.

    **The simulation.** :func:`simulate_rvs_dataset` renders the epochs on a model grid of
    pixel ``dv_sim`` through four discrete steps, each of which adds a variance that a
    continuous spectrum observed by the RVS would not have. A box average of width ``dv``
    has variance ``dv^2 / 12``, and a linear-interpolation shift by a fraction ``f`` of a
    pixel variance ``f (1 - f) dv^2``, ``dv^2 / 6`` on average. The steps are the library's
    box average onto the model grid (:meth:`albireo.SpectralLibrary.resampled_to`,
    ``dv^2 / 12``), the pixel-integrated rotation kernel (``dv^2 / 12``), the shift of each
    component to its epoch velocity (``dv^2 / 6``), and the model pixel held constant across
    the rebin onto the detector pixels (``dv^2 / 12``). They sum to ``(5/12) dv^2``
    (:data:`SIMULATION_VARIANCE_FACTOR`), 1.67 km^2/s^2 on the 2 km/s grid of
    :func:`rvs_model_grid`. The sum differs from the ``(7/12) dv^2`` of the label fit's epoch
    comparison by the frame shift, which the simulation does not take.

    **Measured.** Narrow Gaussian lines were rendered from a 0.005 Å library at 26 positions
    and 96 epoch velocities through the shipped chain (:func:`rvs_components`,
    :func:`simulate_rvs_dataset`, :func:`deliver`). They were compared with the continuous
    line convolved with the applied width and averaged over the same pixels (WP-AL,
    ``sim_moment.py``). On the 2 km/s grid the detector epochs had an excess of
    1.287 km^2/s^2 without rotation, which is ``(4/12) dv^2`` less the 0.050 by which the
    Gaussian kernel truncated at four sigma is below ``sigma^2``. The excess was 1.52, 1.56
    and 1.58 at ``v sin i`` = 5, 11 and 25 km/s, where the rotation kernel's pixel term
    averages 0.70 to 0.89 of ``dv^2 / 12``. On a 3 km/s grid the values were 2.95 and 3.51.
    The per-epoch excess rose with ``f (1 - f)`` at a slope of 0.998 ``dv^2``. Delivery added
    a further 12.09 +- 0.11 km^2/s^2 on the DR4 grid (predicted 11.96) and 17.314 on the DR3
    grid (predicted 17.303).

    For both shipped products at the nominal 11,500, ``Delta_det = 8.56`` km/s at 8580 Å, and
    ``sigma_eff`` is 11.60 km/s on the DR4 grid and 11.83 km/s on the DR3 grid against the
    nominal 11.07. Simulated on the 2 km/s grid, the values are 11.67 and 11.90 km/s.

    The width is a mean. A line placed where ``t`` is near 0 or 1 is not smoothed by the
    interpolation and one placed where ``t = 1/2`` is smoothed by ``Delta_det^2 / 4``. On
    the DR4 grid the line widths therefore vary with a period of 12.25 Å, which no
    stationary profile represents. The simulation's shift and rotation terms vary in the
    same way with each epoch's phase and each star's rotation. An analysis's own model-grid
    smoothing is part of the model and not of the data, and is not counted here: the label
    fit removes it from its operator (:meth:`albireo.Fit.epoch_statistics`). On the D65
    benchmark products, where the nominal width was declared, the converged label fit had
    absorbed the difference into ``v sin i`` (``docs/math.md`` §9.2a).

    Parameters
    ----------
    product
        The delivered product; its ``detector_step``, delivered step and wavelength range
        set ``Delta_det`` and ``Delta_prod``.
    resolving_power
        The resolving power of the line-spread function before delivery.
    simulation_dv_kms
        The model-grid pixel in km/s on which the epochs were simulated (``grid.dv_kms`` of
        the grid given to :func:`simulate_rvs_dataset`), or ``None`` for archive data.

    Returns
    -------
    float
        ``sigma_eff`` in km/s. The corresponding resolving power is
        ``c / (sigma_eff 2 sqrt(2 ln 2))``.

    Raises
    ------
    ValueError
        If ``simulation_dv_kms`` is given and is not positive.
    """
    sigma_r = rvs_lsf_sigma_kms(resolving_power)
    centre = 0.5 * (float(product.wave[0]) + float(product.wave[-1]))
    det_kms = C_KMS * float(product.detector_step) / centre
    prod_kms = C_KMS * float(product.step) / centre
    variance = sigma_r**2 + det_kms**2 / 6.0 + (det_kms**2 - prod_kms**2) / 12.0
    if simulation_dv_kms is not None:
        if not float(simulation_dv_kms) > 0.0:
            raise ValueError(f"simulation_dv_kms must be positive; got {simulation_dv_kms}")
        variance += SIMULATION_VARIANCE_FACTOR * float(simulation_dv_kms) ** 2
    return math.sqrt(variance)


# ---------------------------------------------------------------------------
# The one-call simulator
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RVSTruth:
    """The quantities :func:`simulate_rvs_dataset` injected and applied.

    Attributes
    ----------
    simulation
        The detector-level :class:`~albireo.simulate.SimulationTruth` (components,
        velocities, light fractions, noiseless detector fluxes, noise sigma).
    detector
        The detector-grid epochs before delivery, with their independent noise.
    delivery
        The :class:`DeliveryRecord`.
    snr
        Per-epoch S/N per detector pixel that was applied.
    grvs
        The G_RVS the S/N was derived from, or ``None`` when ``snr`` was given.
    resolving_power
        Per-epoch resolving power applied, ``(n_epochs,)``.
    lsf_sigma_kms, quadrature_sigma_kms
        Per-epoch line-spread sigma applied (from ``resolving_power``), and the width
        convolved on top of the library's own broadening.
    declared_lsf_sigma_kms
        Per-epoch width declared on the delivered epochs (the nominal one unless
        ``declare_lsf="truth"``), which is the width an analysis reads.
    library_resolving_power
        The library's resolving power the quadrature was computed from.
    """

    simulation: SimulationTruth
    detector: Dataset
    delivery: DeliveryRecord
    snr: np.ndarray
    grvs: float | None
    resolving_power: np.ndarray
    lsf_sigma_kms: np.ndarray
    quadrature_sigma_kms: np.ndarray
    declared_lsf_sigma_kms: np.ndarray
    library_resolving_power: float | None

    @property
    def velocities(self) -> np.ndarray:
        """``(n_comp, n_epochs)`` injected stellar velocities, barycentric."""
        return self.simulation.velocities

    @property
    def components(self) -> tuple[np.ndarray, ...]:
        return self.simulation.components

    @property
    def grid(self) -> LogGrid:
        return self.simulation.grid


def simulate_rvs_dataset(
    components: Sequence[np.ndarray],
    grid: LogGrid,
    *,
    bjd,
    light_fractions,
    orbit: OrbitParams | None = None,
    velocities=None,
    snr=None,
    grvs: float | None = None,
    n_transits_per_epoch=1,
    product: RVSProduct = RVS_DR4_EPOCH,
    library=None,
    library_resolving_power: Any = _FROM_LIBRARY,
    resolving_power=None,
    declare_lsf: str = "nominal",
    shot_noise: bool = True,
    response_order: int = 0,
    response_amplitude: float = 0.0,
    seed: int = 0,
) -> tuple[Dataset, RVSTruth]:
    """Gaia RVS epoch spectra of a binary: the reference implementation's chain in one call.

    The components are shifted to their velocities and summed with the light fractions.
    The sum is broadened from the library's resolving power to the RVS's by the quadrature
    width, then rebinned onto the detector grid. Noise is then added per detector pixel as
    photon counting at the S/N of the band's mean flux, and each epoch is interpolated onto
    the delivered grid with its variance propagated. The result is declared barycentric and
    vacuum, with the RVS width on every epoch.

    Parameters
    ----------
    components
        Deviation spectra on ``grid``, one per star (:func:`rvs_components`).
    grid
        The model grid (:func:`rvs_model_grid`); it must cover the band plus the shifts.
    bjd
        Epoch times, ``(n_ep,)``: :func:`rvs_transit_times`, :func:`uniform_phase_times`,
        or the real transit times of :func:`rvs_transit_times_from_gost`.
    light_fractions
        ``(n_comp,)`` constant or ``(n_comp, n_ep)`` per epoch, summing to one. For a
        flux ratio ``f = F_2 / F_1`` the fractions are ``(1, f) / (1 + f)``.
    orbit, velocities
        Exactly one: an :class:`~albireo.simulate.OrbitParams` or an explicit
        ``(n_comp, n_ep)`` velocity array in km/s.
    snr
        S/N per detector pixel: a scalar for every epoch or ``(n_ep,)``. ``inf`` is not
        accepted; use a large number for a nearly noiseless dataset.
    grvs
        Alternatively, G_RVS, from which the S/N follows through :func:`rvs_snr_per_pixel`
        with ``n_transits_per_epoch`` transits co-added per epoch (1 for DR4 epochs).
    n_transits_per_epoch
        Transits per epoch, a scalar or ``(n_ep,)``.
    product
        The delivered grid; default the DR4 epoch grid.
    library
        The :class:`~albireo.library.SpectralLibrary` the components were rendered from.
        Its :attr:`~albireo.library.SpectralLibrary.resolving_power` is the default of
        ``library_resolving_power``: 20,000 for the BOSZ registry entries, ``None`` (an
        intrinsic grid, which takes the whole RVS width) for a library that declares none.
    library_resolving_power
        The resolving power the components already have, or ``None`` for
        intrinsic-resolution components. Default: the library's own declaration. One of
        ``library`` and ``library_resolving_power`` must be given: assuming a published
        grid would broaden an intrinsic library to 9.06 km/s where the RVS gives 11.07.
    resolving_power
        The resolving power to observe at: a scalar, or one value per epoch such as
        :func:`rvs_transit_resolving_powers` draws from the in-flight measurements.
        Default the nominal :data:`RVS_RESOLVING_POWER` for every epoch.
    declare_lsf
        What the delivered epochs declare as their line-spread width: ``"nominal"``
        (default) puts the nominal width on every epoch, which is the width an analysis
        of the real product has; ``"truth"`` puts each epoch's own width on it.
    shot_noise
        Photon-counting noise (default) or a uniform sigma.
    response_order, response_amplitude
        A multiplicative Chebyshev polynomial of this order on every detector epoch,
        with coefficients drawn from a normal distribution of this standard
        deviation, as :func:`albireo.simulate.simulate_dataset` applies it: the
        residual of a normalisation. The coefficient of order zero is a scale, so an
        order of two gives each transit a scale, a slope and a curvature. The
        default amplitude of zero leaves the normalisation exact.
    seed
        Seed for the noise.

    Returns
    -------
    (Dataset, RVSTruth)
    """
    bjd = np.asarray(bjd, dtype=np.float64)
    n_ep = bjd.size
    if (snr is None) == (grvs is None):
        raise ValueError("give exactly one of snr= and grvs=")
    if grvs is not None:
        snr_ep = np.broadcast_to(
            np.asarray(rvs_snr_per_pixel(grvs, n_transits_per_epoch), dtype=np.float64), (n_ep,)
        ).copy()
    else:
        snr_ep = np.broadcast_to(np.asarray(snr, dtype=np.float64), (n_ep,)).copy()
    if not np.all(np.isfinite(snr_ep) & (snr_ep > 0.0)):
        raise ValueError("the S/N must be finite and positive at every epoch")

    if declare_lsf not in ("nominal", "truth"):
        raise ValueError(f"declare_lsf must be 'nominal' or 'truth'; got {declare_lsf!r}")
    if library_resolving_power is _FROM_LIBRARY:
        if library is None:
            raise ValueError(
                "declare the resolving power the components already carry: pass library= "
                "(the SpectralLibrary they were rendered from, whose resolving_power is read) "
                "or library_resolving_power= (None for an intrinsic grid)"
            )
        library_resolving_power = library.resolving_power
    elif library is not None and library_resolving_power != library.resolving_power:
        raise ValueError(
            f"library_resolving_power = {library_resolving_power} contradicts the library's "
            f"own resolving_power = {library.resolving_power}; pass one of them"
        )
    if resolving_power is None:
        r_ep = np.full(n_ep, RVS_RESOLVING_POWER)
    else:
        r_ep = np.broadcast_to(np.asarray(resolving_power, dtype=np.float64), (n_ep,)).copy()
    if not np.all(np.isfinite(r_ep) & (r_ep > 0.0)):
        raise ValueError("resolving_power must be finite and positive at every epoch")
    sigma_true = np.array([rvs_lsf_sigma_kms(r) for r in r_ep])
    sigma_apply = np.array([quadrature_sigma_kms(library_resolving_power, r) for r in r_ep])
    detector_wave = rvs_detector_grid(product.detector_step)
    if detector_wave[0] < grid.wave[0] or detector_wave[-1] > grid.wave[-1]:
        raise ValueError(
            f"the model grid ({grid.wave[0]:.2f}-{grid.wave[-1]:.2f} A) does not cover the "
            f"detector grid ({detector_wave[0]:.2f}-{detector_wave[-1]:.2f} A); build it "
            "with rvs_model_grid()"
        )
    # One detector instrument per distinct resolving power: the simulator keys the
    # line-spread kernel on the instrument, and every transit shares the detector grid.
    instruments: dict[str, InstrumentSpec] = {}
    epoch_instruments = []
    for j in range(n_ep):
        key = f"{product.instrument}-detector-R{round(float(r_ep[j]))}"
        if key not in instruments:
            instruments[key] = InstrumentSpec(
                wave=detector_wave,
                sigma_v_lsf=float(sigma_apply[j]),
                snr=float(np.median(snr_ep)),
                shot_noise=shot_noise,
                snr_window=product.snr_window,
            )
        epoch_instruments.append(key)
    detector, simulation = simulate_dataset(
        grid,
        components,
        bjd=bjd,
        instruments=instruments,
        light_fractions=light_fractions,
        orbit=orbit,
        velocities=velocities,
        epoch_instruments=epoch_instruments,
        v_bary=np.zeros(n_ep),
        frame="barycentric",
        epoch_snr=snr_ep,
        response_order=response_order,
        response_amplitude=response_amplitude,
        seed=seed,
    )
    declared = sigma_true if declare_lsf == "truth" else np.full(n_ep, rvs_lsf_sigma_kms())
    delivered, record = deliver(detector, product, lsf_sigma_kms=declared)
    truth = RVSTruth(
        simulation=simulation,
        detector=detector,
        delivery=record,
        snr=snr_ep,
        grvs=None if grvs is None else float(grvs),
        resolving_power=r_ep,
        lsf_sigma_kms=sigma_true,
        quadrature_sigma_kms=sigma_apply,
        declared_lsf_sigma_kms=declared,
        library_resolving_power=library_resolving_power,
    )
    return delivered, truth

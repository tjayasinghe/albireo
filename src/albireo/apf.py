"""APF spectra as the California Planet Search pipeline delivers them.

**Experimental.** The names, the defaults and the reader options may change.

The Automated Planet Finder's Levy spectrometer (Lick Observatory) records 79 echelle
orders. The CPS pipeline extracts each exposure into a 79 x 4608 image of counts, the
``r<night>.<number>.fits`` files, and keeps the wavelength solution in a separate file
(``apf_wave_2022.fits``, an image of the same shape in air Angstrom). Row ``r`` is echelle
order ``m = 124 - r``; row 0 is the bluest (3730 A) and row 78 the reddest (10200 A).
:func:`albireo.read_spectrum` recognises these files and calls :func:`extract` with the
options of its ``apf=`` argument. The steps, per exposure:

1. **Wavelengths.** The solution file gives the topocentric air wavelength of every pixel at
   the epoch it was made. The instrument drifts: on exposures from 2023-2026 the 2022 file
   is offset by about 1 km/s and its dispersion within each order differs from these
   exposures' by up to 1.7 km/s near the blue end of the order. :func:`refine_wavelengths`
   fits a smooth correction to a solution from one line-rich exposure: its stellar lines,
   against a synthetic template, and its telluric lines, against the packaged telluric
   line list (:mod:`albireo.telluric`), which fixes the absolute scale.
2. **Zero point.** Every exposure's telluric lines are correlated with the line list
   (:func:`albireo.telluric_shift`) in the O2 gamma and B bands and two H2O regions (rows
   49-61), and the wavelengths are multiplied by ``1 + v / c``. The internal error is
   0.05-0.15 km/s per exposure.
3. **Line-spread function.** The widths of 14 weak, isolated O2 lines, with their own width
   (1.3 km/s FWHM, an assumption) removed in quadrature (:func:`albireo.telluric_lsf`).
   Where fewer than three lines can be fitted the decker's typical value is used
   (:data:`DECKER_LSF_FWHM_KMS`, measured on 2023-2026 exposures: N 3.97, T 5.65,
   W 3.82 km/s FWHM).
4. **Noise.** ``var = (counts + 359.5) / 0.884`` ADU^2 per pixel, fitted to DER_SNR noise
   estimates of exposures of a line-poor early-type star (22,237 chunks of 256 pixels).
5. **Cosmic rays.** Pixels more than 5 sigma above a 7-pixel running median, dilated by one
   pixel, and non-positive counts. The CPS files are single exposures, so cosmic rays are
   not removed upstream.
6. **Blaze.** Each order is divided by its upper envelope (:func:`albireo.fit_continuum`
   over the pixel index), or by a blaze array passed by the caller, such as one made by
   :func:`build_blaze` from exposures of a line-poor star. An envelope follows any
   absorption broader than about 500 pixels (25 A): Balmer-line wings are partly removed.
7. **Merging.** Neighbouring orders overlap below about 7400 A; each overlap is split at
   its midpoint and the pixels are concatenated, the next order scaled to match in the
   overlap. No pixel is resampled.
8. **Sky.** The N decker is 8 arcsec long, so the extracted spectrum contains sky. Night-sky
   emission ([O I] 5577, 6300, 6364 A and Na D) and street-light mercury lines are masked in
   the topocentric frame, +-12 km/s.
9. **Iodine.** Exposures with the iodine cell in the beam (``ICELNAM = 'In'``) carry I2
   absorption over 5000-6200 A. By default they are rejected; ``iodine="mask"`` masks the
   band instead.
10. **Time and velocity.** BJD_TDB at the photon-weighted midpoint (``THEMIDPT``) and the
    barycentric correction, both with astropy for the APF site. The header ``RA``/``DEC`` are
    J2000: on the exposures checked they agree with the catalogue J2000 positions of the
    targets to about an arcsecond, whereas apparent coordinates of the observing dates would
    differ by up to 19 arcmin.

The result is in the topocentric frame, with the barycentric correction recorded for
albireo to apply, and is not continuum-normalised (:func:`albireo.io.to_epoch` fits the
continuum on the merged spectrum).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from albireo.preprocess import fit_continuum
from albireo.telluric import (
    TelluricLines,
    TelluricShift,
    _correlation,
    _parabola_peak,
    _shifted_templates,
    _smooth,
    telluric_lines,
    telluric_lsf,
    telluric_model,
    telluric_shift,
)

__all__ = [
    "APF_SHAPE",
    "DECKER_LSF_FWHM_KMS",
    "SKY_LINES_AIR",
    "ApfExposure",
    "ApfSpectrum",
    "ApfWavelengths",
    "RefinedWavelengths",
    "apf_barycentric",
    "build_blaze",
    "extract",
    "is_apf_exposure",
    "read_apf_exposure",
    "refine_wavelengths",
]

_C_KMS = 299_792.458
_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))

#: Shape of a CPS-extracted APF exposure: (orders, pixels).
APF_SHAPE = (79, 4608)
_N_ROWS, _N_PIX = APF_SHAPE
# Echelle order of row 0; row r is order 124 - r.
_ORDER_OF_ROW_ZERO = 124
_XC = (_N_PIX - 1) / 2.0

# The APF dome on Mt. Hamilton (geodetic).
_APF_LAT_DEG = 37.3425
_APF_LON_DEG = -121.6383
_APF_HEIGHT_M = 1274.0

# Noise model: var = (counts + read_term) / gain, ADU^2 (module docstring, step 4).
_GAIN = 0.884
_READ_TERM_ADU = 359.5

#: Typical LSF FWHM per decker, km/s, the median of the telluric widths of 2023-2026
#: exposures (N 43 exposures, T 49, W 1). Used when an exposure's own lines cannot be fitted.
DECKER_LSF_FWHM_KMS: Mapping[str, float] = {"N": 3.97, "T": 5.65, "W": 3.82}

# Rows used by default: 3846-7470 A, the range the 2022 solution and the noise model were
# checked over. Rows 49-61 hold the telluric bands.
_DEFAULT_ROWS = range(3, 61)
_TELLURIC_ROWS = range(49, 62)

#: Night-sky emission and street-light lines, air Angstrom, masked in the topocentric frame.
SKY_LINES_AIR: tuple[float, ...] = (
    5577.34,
    5889.95,
    5895.92,
    6300.30,
    6363.78,
    4046.56,
    4358.34,
    5460.74,
    5769.60,
    5790.66,
)
_IODINE_BAND = (5000.0, 6200.0)
_BALMER_AIR = (6562.8, 4861.3, 4340.5, 4101.7, 3970.1, 3889.0, 3835.4, 3797.9, 3770.6)


# ---------------------------------------------------------------------------
# Wavelength solutions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApfWavelengths:
    """A wavelength solution: the air wavelength of every pixel of every order.

    Attributes
    ----------
    wave : numpy.ndarray
        Shape ``(79, 4608)``, Angstrom, air, topocentric, strictly increasing along each row.
    source : str
        Where the solution came from.
    """

    wave: np.ndarray
    source: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        wave = np.asarray(self.wave, dtype=np.float64)
        if wave.shape != APF_SHAPE:
            raise ValueError(f"an APF wavelength solution has shape {APF_SHAPE}; got {wave.shape}")
        if not np.all(np.isfinite(wave)) or np.any(np.diff(wave, axis=1) <= 0):
            raise ValueError(
                "an APF wavelength solution must be finite and strictly increasing along each "
                "order (row)"
            )
        if not 3000.0 < float(wave[0, 0]) < float(wave[-1, -1]) < 12000.0:
            raise ValueError(
                f"the solution spans {wave[0, 0]:.1f}-{wave[-1, -1]:.1f}; APF orders cover "
                "about 3730-10200 A. Is it in Angstrom, with row 0 the bluest order?"
            )
        object.__setattr__(self, "wave", wave)

    @classmethod
    def load(cls, path: str | Path) -> ApfWavelengths:
        """Read a solution from a FITS image of shape ``(79, 4608)`` (``apf_wave_2022.fits``)."""
        from astropy.io import fits

        with fits.open(path, memmap=False) as hdulist:
            for hdu in hdulist:
                data = getattr(hdu, "data", None)
                if data is not None and np.shape(data) == APF_SHAPE:
                    source = str(hdu.header.get("ALBSRC", "")).strip() or Path(path).name
                    return cls(np.asarray(data, dtype=np.float64), source=source)
        raise ValueError(f"{path}: no {APF_SHAPE} image in the file")

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Write the solution as a FITS image that :meth:`load` reads back."""
        from astropy.io import fits

        hdu = fits.PrimaryHDU(self.wave)
        # A long origin is written as a CONTINUE long string, so it carries no comment.
        hdu.header["ALBSRC"] = self.source
        hdu.header["MEDIUM"] = ("air", "wavelength medium")
        hdu.header["FRAME"] = ("topocentric", "frame of the wavelengths")
        hdu.header["COMMENT"] = "ALBSRC: origin of this wavelength solution (albireo.apf)"
        hdu.writeto(path, overwrite=overwrite)
        return Path(path)

    def shifted(self, v_kms: float) -> ApfWavelengths:
        """The solution with every wavelength multiplied by ``1 + v_kms / c``."""
        return ApfWavelengths(self.wave * (1.0 + float(v_kms) / _C_KMS), source=self.source)

    @staticmethod
    def order(row: int) -> int:
        """The echelle order of a row."""
        return _ORDER_OF_ROW_ZERO - int(row)


def _as_blaze(value: Any) -> np.ndarray | None:
    """A blaze array from an array, a ``.npy`` file or a FITS image; ``None`` stays ``None``."""
    if value is None:
        return None
    if isinstance(value, str | Path):
        path = Path(value)
        if path.suffix.lower() == ".npy":
            return np.asarray(np.load(path), dtype=float)
        from astropy.io import fits

        return np.asarray(fits.getdata(path), dtype=float)
    return np.asarray(value, dtype=float)


def _as_wavelengths(value: Any) -> ApfWavelengths:
    if isinstance(value, ApfWavelengths):
        return value
    if isinstance(value, str | Path):
        return ApfWavelengths.load(value)
    return ApfWavelengths(np.asarray(value, dtype=np.float64), source="array")


# ---------------------------------------------------------------------------
# Exposures
# ---------------------------------------------------------------------------


def is_apf_exposure(header: Mapping, shape: Sequence[int]) -> bool:
    """Whether a FITS image is a CPS-extracted APF exposure.

    The files have no ``INSTRUME`` keyword. They are recognised by their shape together with
    ``VERSION = 'apf'`` or the APF decker keyword ``DECKRNAM``.
    """
    if tuple(shape) != APF_SHAPE:
        return False
    version = str(header.get("VERSION", "")).strip().lower()
    return version == "apf" or "DECKRNAM" in header


@dataclass(frozen=True)
class ApfExposure:
    """One CPS-extracted exposure: its counts and header.

    Attributes
    ----------
    counts : numpy.ndarray
        Shape ``(79, 4608)``, ADU.
    header : Mapping
        The primary header.
    path : str
        The file it came from.
    """

    counts: np.ndarray
    header: Mapping = field(repr=False)
    path: str = ""

    @property
    def decker(self) -> str:
        """The decker letter (``N``, ``T``, ``W``, ...), from ``DECKRNAM``."""
        return str(self.header.get("DECKRNAM", "")).strip()[:1].upper()

    @property
    def iodine_in(self) -> bool:
        """Whether the iodine cell was in the beam (``ICELNAM = 'In'``)."""
        return str(self.header.get("ICELNAM", "")).strip().lower() == "in"

    @property
    def target(self) -> str:
        """``TOBJECT`` (the telescope's target name), else ``OBJECT``."""
        return str(self.header.get("TOBJECT", self.header.get("OBJECT", ""))).strip()


def read_apf_exposure(path: str | Path) -> ApfExposure:
    """Read a CPS-extracted APF exposure (``r*.fits``)."""
    from astropy.io import fits

    with fits.open(path, memmap=False) as hdulist:
        header = hdulist[0].header.copy()
        data = hdulist[0].data
        if data is None or not is_apf_exposure(header, np.shape(data)):
            raise ValueError(
                f"{path}: not a CPS-extracted APF exposure (a {APF_SHAPE} image with "
                "VERSION = 'apf' or DECKRNAM)"
            )
        counts = np.asarray(data, dtype=np.float64)
    return ApfExposure(counts=counts, header=header, path=str(path))


def _noise_variance(counts: np.ndarray, gain: float, read_term: float) -> np.ndarray:
    from scipy.ndimage import median_filter

    expected = np.maximum(median_filter(counts, size=5, mode="nearest"), 0.0)
    return (expected + read_term) / gain


def _cosmic_rays(counts: np.ndarray, var: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    """``True`` at cosmic-ray hits and unusable pixels of one order."""
    from scipy.ndimage import median_filter

    median = median_filter(counts, size=7, mode="nearest")
    hit = (counts - median) > threshold * np.sqrt(np.maximum(var, 1.0))
    hit = np.convolve(hit.astype(float), np.ones(3), mode="same") > 0
    return hit | ~np.isfinite(counts) | (counts <= 0)


def _envelope(counts: np.ndarray, bad: np.ndarray, smooth_pix: float = 500.0) -> np.ndarray:
    """Upper envelope of one order over the pixel index (the blaze times the continuum)."""
    x = np.arange(counts.size, dtype=float)
    weights = (~bad).astype(float)
    if weights.sum() < 100:
        return np.full(counts.size, np.nan)
    return fit_continuum(
        x, np.where(bad, np.nan, counts), smooth_angstrom=smooth_pix, weights=weights
    )


class _Orders:
    """The noise variance, cosmic-ray mask and envelope of the orders of one exposure.

    Each is computed once per order: the telluric rows 49-61 are normalised for the zero
    point and the line-spread width, and most of them are merged as well.
    """

    def __init__(self, exposure: ApfExposure, gain: float, read_term: float) -> None:
        self._counts = exposure.counts
        self._gain = gain
        self._read_term = read_term
        self._noise: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._envelopes: dict[int, np.ndarray] = {}

    def noise(self, row: int) -> tuple[np.ndarray, np.ndarray]:
        """``(var, bad)`` of one order: the noise variance and the cosmic-ray mask."""
        if row not in self._noise:
            counts = self._counts[row]
            var = _noise_variance(counts, self._gain, self._read_term)
            self._noise[row] = (var, _cosmic_rays(counts, var))
        return self._noise[row]

    def envelope(self, row: int) -> np.ndarray:
        """The upper envelope of one order (:func:`_envelope`)."""
        if row not in self._envelopes:
            self._envelopes[row] = _envelope(self._counts[row], self.noise(row)[1])
        return self._envelopes[row]

    def normalised(self, row: int) -> tuple[np.ndarray, np.ndarray]:
        """``(flux, good)`` of one order divided by its envelope, cosmic rays masked."""
        counts = self._counts[row]
        bad = self.noise(row)[1]
        envelope = self.envelope(row)
        with np.errstate(invalid="ignore", divide="ignore"):
            flux = counts / envelope
        good = ~bad & np.isfinite(flux) & (envelope > 0.03 * np.nanmax(envelope))
        return np.where(good, flux, 1.0), good


def _telluric_segments(
    orders: _Orders, wave: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    segments = []
    for row in _TELLURIC_ROWS:
        flux, good = orders.normalised(row)
        segments.append((wave[row], flux, good))
    return segments


def _line_mask(wave: np.ndarray, centres: Sequence[float], halfwidth_kms: float) -> np.ndarray:
    """``True`` within ``halfwidth_kms`` of any of the ``centres``."""
    mask = np.zeros(wave.shape, dtype=bool)
    for centre in centres:
        mask |= np.abs(wave / centre - 1.0) * _C_KMS < halfwidth_kms
    return mask


def _checked_rows(rows: Iterable[int]) -> list[int]:
    """The rows in increasing order without repeats.

    Raises ``ValueError`` if there are none or one lies outside 0-78.
    """
    checked = sorted({int(row) for row in rows})
    if not checked:
        raise ValueError(f"rows is empty: give at least one row between 0 and {_N_ROWS - 1}")
    if checked[0] < 0 or checked[-1] >= _N_ROWS:
        raise ValueError(
            f"rows must lie between 0 and {_N_ROWS - 1} (row 0 is the bluest order); got "
            f"{checked[0]} to {checked[-1]}"
        )
    return checked


def build_blaze(
    exposures: Sequence[ApfExposure],
    wavelengths,
    *,
    rows: Sequence[int] = _DEFAULT_ROWS,
    gain: float = _GAIN,
    read_term: float = _READ_TERM_ADU,
) -> np.ndarray:
    """A blaze function from exposures of a star with few lines.

    Each order is normalised to its median over pixels 1800-2800, the exposures are
    median-combined with cosmic rays masked, the Balmer lines are masked (+-2200 km/s) and an
    iterative upper-envelope smoothing spline is fitted. An order dominated by a Balmer line
    is interpolated from its neighbours. The blaze of the 2023-2026 validation was made in
    this way from 34 exposures of a line-poor early-type star.

    Parameters
    ----------
    exposures : sequence of ApfExposure
        Exposures of a star with few and weak metal lines (an early-type star).
    wavelengths : ApfWavelengths or array or path
        The wavelength solution, used to find the Balmer lines.
    rows : sequence of int, optional
        The rows to model, between 0 and 78; the others are set to 1.
    gain, read_term : float, optional
        The noise model (module docstring, step 4).

    Returns
    -------
    numpy.ndarray
        Shape ``(79, 4608)``, for the ``blaze`` option of :func:`extract`.

    Raises
    ------
    ValueError
        If ``rows`` is empty or holds a row outside 0-78, or no order could be modelled.
    """
    from scipy.interpolate import UnivariateSpline

    waves = _as_wavelengths(wavelengths).wave
    blaze = np.ones(APF_SHAPE)
    x = np.arange(_N_PIX, dtype=float)
    rows = _checked_rows(rows)
    stack: dict[int, list[np.ndarray]] = {row: [] for row in rows}
    for exposure in exposures:
        for row in rows:
            counts = exposure.counts[row]
            bad = _cosmic_rays(counts, _noise_variance(counts, gain, read_term))
            level = float(np.median(counts[1800:2800]))
            if level > 0:
                stack[row].append(np.where(bad, np.nan, counts / level))
    failed = []
    for row in rows:
        if not stack[row]:
            failed.append(row)
            continue
        median = np.nanmedian(np.array(stack[row]), axis=0)
        free = ~_line_mask(waves[row], _BALMER_AIR, 2200.0)
        ok = np.isfinite(median) & free & (median > 0.02)
        if free.mean() < 0.35 or ok.sum() < 400:
            failed.append(row)
            continue
        weights = ok.astype(float)
        spline = None
        for _ in range(6):
            if (weights > 0).sum() < 200:
                break
            spline = UnivariateSpline(
                x[weights > 0], median[weights > 0], k=3, s=weights.sum() * 2e-5
            )
            model = spline(x)
            weights = (ok & (median - model > -0.015 * np.maximum(model, 0.05))).astype(float)
        if spline is None:
            failed.append(row)
            continue
        blaze[row] = np.maximum(spline(x), 1e-3)
    good_rows = [row for row in rows if row not in failed]
    if not good_rows:
        raise ValueError("no order could be modelled; are these exposures of a star?")
    for row in failed:
        lower = [r for r in good_rows if r < row]
        upper = [r for r in good_rows if r > row]
        if lower and upper:
            lo, hi = lower[-1], upper[0]
            fraction = (row - lo) / (hi - lo)
            blaze[row] = (1.0 - fraction) * blaze[lo] + fraction * blaze[hi]
        else:
            blaze[row] = blaze[lower[-1] if lower else upper[0]]
    return blaze


def _merge_orders(waves, fluxes, ivars, rows):
    """Concatenate orders, splitting each overlap at its midpoint; scale orders to agree.

    Each order is scaled to its predecessor in their overlap, so the joins are continuous.
    The scales compound from order to order: an envelope that is biased by a few per cent at
    the ends of every order makes them drift by a factor of several over 60 orders. The
    merged spectrum is therefore divided by a cubic in log wavelength fitted to the log of
    the scales at the order centres, which keeps the joins and removes the drift.
    """
    keep_w, keep_f, keep_i = [], [], []
    scales, centres = [], []
    scale = 1.0
    previous = None
    for k, row in enumerate(rows):
        w, f, iv = waves[row], fluxes[row], ivars[row]
        lo, hi = -np.inf, np.inf
        if k > 0:
            wp = waves[rows[k - 1]]
            lo = 0.5 * (w[0] + wp[-1]) if wp[-1] > w[0] else w[0]
        if k < len(rows) - 1:
            wn = waves[rows[k + 1]]
            hi = 0.5 * (wn[0] + w[-1]) if w[-1] > wn[0] else w[-1] + 1e-9
        sel = (w >= lo) & (w < hi)
        if previous is not None:
            pw, pf, piv = previous
            overlap = (w > pw[0]) & (w < pw[-1]) & (iv > 0)
            if overlap.sum() > 50 and (piv > 0).sum() > 50:
                on_previous = np.interp(w[overlap], pw[piv > 0], pf[piv > 0])
                with np.errstate(invalid="ignore", divide="ignore"):
                    ratio = np.nanmedian(
                        on_previous / np.where(f[overlap] != 0, f[overlap], np.nan)
                    )
                if np.isfinite(ratio) and ratio > 0:
                    scale = float(ratio)
        f_scaled, iv_scaled = f * scale, iv / scale**2
        keep_w.append(w[sel])
        keep_f.append(f_scaled[sel])
        keep_i.append(iv_scaled[sel])
        previous = (w, f_scaled, iv_scaled)
        scales.append(scale)
        centres.append(float(np.log(w[w.size // 2])))
    wave = np.concatenate(keep_w)
    flux = np.concatenate(keep_f)
    ivar = np.concatenate(keep_i)
    if len(scales) > 1:
        degree = min(3, len(scales) - 1)
        x0 = float(np.mean(centres))
        coef = np.polyfit(np.asarray(centres) - x0, np.log(scales), degree)
        trend = np.exp(np.polyval(coef, np.log(wave) - x0))
        flux, ivar = flux / trend, ivar * trend**2
    return wave, flux, ivar


@dataclass(frozen=True)
class ApfSpectrum:
    """An exposure extracted by :func:`extract`: merged, blaze-removed, not normalised.

    Attributes
    ----------
    wave : numpy.ndarray
        Air wavelengths, topocentric, with the zero-point shift applied, Angstrom.
    flux : numpy.ndarray
        Blaze-removed flux, ``nan`` where ``bad``. With the default envelope it is near 1 at
        the continuum. With a supplied blaze array it keeps the level of the counts (the
        blaze of :func:`build_blaze` is near 1 at the centre of each order) and is not
        continuum-normalised; :func:`albireo.io.to_epoch` normalises it.
    err : numpy.ndarray
        One-sigma uncertainties of ``flux``.
    bad : numpy.ndarray
        ``True`` where a pixel must not be used (cosmic rays, sky lines, iodine band).
    zero_point : TelluricShift or None
        The telluric shift measured and applied, or ``None`` when not measured.
    drift_kms : float
        The shift applied to the solution's wavelengths, km/s.
    lsf_fwhm_kms : float
        The LSF FWHM, km/s.
    lsf_source : str
        How the LSF width was obtained.
    notes : tuple of str
        What was done, step by step.
    """

    wave: np.ndarray
    flux: np.ndarray
    err: np.ndarray
    bad: np.ndarray
    zero_point: TelluricShift | None
    drift_kms: float
    lsf_fwhm_kms: float
    lsf_source: str
    notes: tuple[str, ...] = ()


def extract(
    exposure: ApfExposure,
    wavelengths,
    *,
    zero_point: str | float = "telluric",
    lsf: str | float = "telluric",
    blaze=None,
    rows: Sequence[int] | None = None,
    iodine: str = "error",
    mask_sky: bool = True,
    gain: float = _GAIN,
    read_term: float = _READ_TERM_ADU,
    lines: TelluricLines | None = None,
) -> ApfSpectrum:
    """Calibrate and merge one exposure (module docstring, steps 1-9).

    Parameters
    ----------
    exposure : ApfExposure
        From :func:`read_apf_exposure`.
    wavelengths : ApfWavelengths, array of shape (79, 4608), or path
        The wavelength solution, e.g. ``"apf_wave_2022.fits"`` or the output of
        :func:`refine_wavelengths`.
    zero_point : "telluric", "none" or float, optional
        ``"telluric"`` (default) measures the exposure's shift from its telluric lines;
        ``"none"`` uses the solution as it is; a number is a shift in km/s.
    lsf : "telluric", "decker" or float, optional
        ``"telluric"`` (default) measures the FWHM from the telluric lines and falls back to
        the decker's typical value; ``"decker"`` uses the typical value; a number is a FWHM
        in km/s.
    blaze : array of shape (79, 4608), optional
        A blaze function (:func:`build_blaze`). Default: each order's upper envelope.
    rows : sequence of int, optional
        The orders to merge, rows between 0 and 78 in any order; repeats are ignored.
        Default rows 3-60 (3846-7470 A).
    iodine : {"error", "mask", "ignore"}, optional
        What to do with an exposure taken through the iodine cell. Default ``"error"``.
    mask_sky : bool, optional
        Mask night-sky and street-light emission lines. Default ``True``.
    gain, read_term : float, optional
        The noise model ``var = (counts + read_term) / gain``.
    lines : TelluricLines, optional
        Default: the packaged list.

    Returns
    -------
    ApfSpectrum

    Raises
    ------
    ValueError
        If the exposure was taken through the iodine cell and ``iodine="error"``, the
        telluric zero point cannot be measured, ``rows`` is empty or holds a row outside
        0-78, or a supplied shift or FWHM is not finite (the FWHM also not positive).
    """
    if iodine not in ("error", "mask", "ignore"):
        raise ValueError(f"iodine must be 'error', 'mask' or 'ignore'; got {iodine!r}")
    if exposure.iodine_in and iodine == "error":
        raise ValueError(
            f"{exposure.path}: taken through the iodine cell (ICELNAM = 'In'), whose I2 "
            "absorption covers 5000-6200 A. Pass iodine='mask' to mask that band, or leave the "
            "exposure out."
        )
    solution = _as_wavelengths(wavelengths)
    lines = telluric_lines() if lines is None else lines
    rows = _checked_rows(_DEFAULT_ROWS if rows is None else rows)
    notes: list[str] = [f"wavelengths: {solution.source}"]
    decker_fwhm = DECKER_LSF_FWHM_KMS.get(exposure.decker)
    orders = _Orders(exposure, gain, read_term)

    shift: TelluricShift | None = None
    segments = []
    if zero_point == "telluric" or lsf == "telluric":
        segments = _telluric_segments(orders, solution.wave)
    if isinstance(zero_point, str):
        if zero_point == "telluric":
            sigma_guess = (decker_fwhm or 4.5) / _FWHM_PER_SIGMA
            try:
                shift = telluric_shift(segments, lsf_sigma_kms=sigma_guess, lines=lines)
            except ValueError as exc:
                raise ValueError(
                    f"{exposure.path}: the telluric zero point cannot be measured ({exc}). "
                    "A short or cloudy exposure has too little signal in rows 49-61: leave "
                    "the file out, or pass zero_point=<shift in km/s>."
                ) from exc
            drift = shift.v_kms
            notes.append(
                f"zero point: telluric shift {drift:+.3f} +- {shift.err_kms:.3f} km/s from "
                f"{shift.n_regions} regions"
            )
        elif zero_point == "none":
            drift = 0.0
            notes.append("zero point: the solution as given (no telluric shift)")
        else:
            raise ValueError(
                f"zero_point must be 'telluric', 'none' or a number; got {zero_point!r}"
            )
    else:
        drift = float(zero_point)
        if not math.isfinite(drift):
            raise ValueError(f"zero_point must be a finite shift in km/s; got {zero_point!r}")
        notes.append(f"zero point: shift {drift:+.3f} km/s supplied")
    wave = solution.wave * (1.0 + drift / _C_KMS)

    if isinstance(lsf, str):
        if lsf not in ("telluric", "decker"):
            raise ValueError(f"lsf must be 'telluric', 'decker' or a number; got {lsf!r}")
        measured = None
        if lsf == "telluric":
            measured = telluric_lsf(segments, shift_kms=drift, lines=lines)
        if measured is not None:
            lsf_fwhm, lsf_source = measured[0], f"telluric O2 widths ({measured[1]} lines)"
        elif decker_fwhm is not None:
            lsf_fwhm, lsf_source = decker_fwhm, f"typical for decker {exposure.decker}"
        else:
            raise ValueError(
                f"{exposure.path}: the LSF could not be measured from telluric lines and decker "
                f"{exposure.decker!r} has no typical value; pass lsf=<FWHM in km/s>"
            )
    else:
        lsf_fwhm, lsf_source = float(lsf), "supplied"
        if not (math.isfinite(lsf_fwhm) and lsf_fwhm > 0.0):
            raise ValueError(f"lsf must be a positive FWHM in km/s; got {lsf!r}")
    notes.append(f"LSF FWHM {lsf_fwhm:.2f} km/s ({lsf_source})")

    fluxes: dict[int, np.ndarray] = {}
    ivars: dict[int, np.ndarray] = {}
    n_cosmic = 0
    blaze_array = _as_blaze(blaze)
    if blaze_array is not None and blaze_array.shape != APF_SHAPE:
        raise ValueError(f"blaze must have shape {APF_SHAPE}; got {blaze_array.shape}")
    for row in rows:
        counts = exposure.counts[row]
        var, bad = orders.noise(row)
        n_cosmic += int(bad.sum())
        shape = orders.envelope(row) if blaze_array is None else blaze_array[row]
        usable = ~bad & np.isfinite(shape) & (shape > 0.04 * np.nanmax(shape))
        with np.errstate(invalid="ignore", divide="ignore"):
            fluxes[row] = np.where(usable, counts / shape, 0.0)
            ivars[row] = np.where(usable, shape**2 / var, 0.0)
    notes.append(
        "blaze: each order's upper envelope" if blaze_array is None else "blaze: supplied array"
    )
    w_m, f_m, i_m = _merge_orders(wave, fluxes, ivars, rows)
    bad_m = ~(i_m > 0)
    if mask_sky:
        bad_m |= _line_mask(w_m, SKY_LINES_AIR, 12.0)
    if exposure.iodine_in and iodine == "mask":
        bad_m |= (w_m > _IODINE_BAND[0]) & (w_m < _IODINE_BAND[1])
        notes.append("iodine band 5000-6200 A masked")
    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.where(i_m > 0, 1.0 / np.sqrt(i_m), np.inf)
    notes.append(f"{n_cosmic} pixels masked as cosmic rays or non-positive")
    return ApfSpectrum(
        wave=w_m,
        flux=np.where(bad_m, np.nan, f_m),
        err=err,
        bad=bad_m,
        zero_point=shift,
        drift_kms=float(drift),
        lsf_fwhm_kms=float(lsf_fwhm),
        lsf_source=lsf_source,
        notes=tuple(notes),
    )


def apf_barycentric(header: Mapping, *, path: str = "") -> tuple[float, float, str]:
    """BJD_TDB, the barycentric correction in km/s, and how the time was obtained.

    The time is ``THEMIDPT``, the photon-weighted midpoint from the exposure meter, else
    ``DATE-OBS`` plus half of ``EXPTIME``. The coordinates are the header's J2000 ``RA`` and
    ``DEC`` (module docstring, step 10), the site the APF dome.
    """
    from astropy import units as u
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.time import Time

    location = EarthLocation.from_geodetic(
        lon=_APF_LON_DEG * u.deg, lat=_APF_LAT_DEG * u.deg, height=_APF_HEIGHT_M * u.m
    )
    midpoint = str(header.get("THEMIDPT", "")).strip()
    if midpoint:
        time = Time(midpoint, format="isot", scale="utc", location=location)
        source = "BJD_TDB from THEMIDPT (photon-weighted midpoint)"
    else:
        start = str(header.get("DATE-OBS", "")).strip()
        if not start:
            raise ValueError(f"{path}: neither THEMIDPT nor DATE-OBS is in the header")
        exptime = float(header.get("EXPTIME", 0.0) or 0.0)
        time = Time(start, format="isot", scale="utc", location=location) + 0.5 * exptime * u.s
        source = "BJD_TDB from DATE-OBS + EXPTIME/2"
    ra, dec = str(header.get("RA", "")).strip(), str(header.get("DEC", "")).strip()
    if not ra or not dec:
        raise ValueError(f"{path}: no RA/DEC in the header")
    coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame="icrs")
    bjd = float((time.tdb + time.light_travel_time(coord, kind="barycentric")).jd)
    v_bary = float(
        coord.radial_velocity_correction("barycentric", obstime=time).to(u.km / u.s).value
    )
    return bjd, v_bary, source


# ---------------------------------------------------------------------------
# Refining a solution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RefinedWavelengths:
    """The result of :func:`refine_wavelengths`.

    Attributes
    ----------
    wavelengths : ApfWavelengths
        The refined solution, topocentric at the reference exposure.
    v_star_kms : float
        The reference star's topocentric velocity relative to the template, km/s.
    rms_stellar_kms, rms_telluric_kms : float
        Scatter of the stellar and telluric chunks about the fitted solution, km/s.
    n_stellar, n_telluric : int
        Chunks used.
    correction_range_kms : tuple of float
        Smallest and largest difference from the input solution over the fitted rows, km/s.
    rows : tuple of int
        The rows the solution was fitted over; outside them the input solution is kept,
        scaled by the ratio at the nearest fitted row.
    """

    wavelengths: ApfWavelengths
    v_star_kms: float
    rms_stellar_kms: float
    rms_telluric_kms: float
    n_stellar: int
    n_telluric: int
    correction_range_kms: tuple[float, float]
    rows: tuple[int, ...]

    def summary(self) -> str:
        """A short report of the fit."""
        lo, hi = self.correction_range_kms
        return (
            f"refined APF wavelength solution: {self.n_stellar} stellar chunks "
            f"(rms {self.rms_stellar_kms:.3f} km/s) and {self.n_telluric} telluric chunks "
            f"(rms {self.rms_telluric_kms:.3f} km/s) over rows {self.rows[0]}-{self.rows[-1]}; "
            f"reference star at {self.v_star_kms:+.3f} km/s; differs from the input by "
            f"{lo:+.2f} to {hi:+.2f} km/s"
        )


def _chunk_lag(
    data, ok, lnw, template_lnw, template, lags, *, nuisance=None, min_good=0.8, min_rms=0.004
):
    """Correlation lag (km/s) and peak coefficient of one chunk, ``(nan, 0)`` if unreliable.

    ``nuisance``, a model of a second spectrum in the chunk (the star's lines in a telluric
    chunk), is projected out of the data and of the shifted template before they are
    correlated, so that its lines cannot correlate with the template by chance.
    """
    if ok.mean() < min_good:
        return float("nan"), 0.0
    d = data - data.mean()
    tm = _shifted_templates(lnw, lags, template_lnw, template)
    if nuisance is not None:
        s = nuisance - nuisance.mean()
        ss = float(s @ s)
        if ss > 1e-12:
            d = d - (float(d @ s) / ss) * s
            tm = tm - np.outer(tm @ s / ss, s)
    if np.std(d) < min_rms:
        return float("nan"), 0.0
    r = _correlation(tm, d)
    j = int(np.argmax(r))
    if j == 0 or j == lags.size - 1:
        return float("nan"), float(r[j])
    step = float(lags[1] - lags[0])
    return float(lags[j] + _parabola_peak(r, j) * step), float(r[j])


def _locally_consistent(chunks: np.ndarray, tolerance_kms: float = 2.0, half: int = 3):
    """The chunks whose lag agrees with the median lag of their neighbours in the same order.

    ``chunks`` has rows ``(row, x, lag, r)``. A spurious correlation peak lies km/s away from
    the neighbouring chunks, whereas the true lag changes by less than 1 km/s over the
    ``half`` chunks on either side. Removing such chunks before the fit keeps them from
    pulling the polynomial in a sparsely covered corner of the image, where sigma clipping
    would then reject the good chunks instead of the bad ones.
    """
    keep = np.ones(len(chunks), dtype=bool)
    for row in np.unique(chunks[:, 0]):
        idx = np.where(chunks[:, 0] == row)[0]
        idx = idx[np.argsort(chunks[idx, 1])]
        lags = chunks[idx, 2]
        for k, i in enumerate(idx):
            others = np.concatenate([lags[max(k - half, 0) : k], lags[k + 1 : k + 1 + half]])
            if others.size >= 2 and abs(lags[k] - float(np.median(others))) > tolerance_kms:
                keep[i] = False
    return chunks[keep]


def refine_wavelengths(
    base,
    exposure: ApfExposure,
    *,
    template_wave,
    template_flux,
    template_resolving_power: float | None = 20_000.0,
    lsf_fwhm_kms: float | None = None,
    rows: Sequence[int] | None = None,
    nx: int = 5,
    nm: int = 4,
    search_kms: float = 300.0,
    lines: TelluricLines | None = None,
    gain: float = _GAIN,
    read_term: float = _READ_TERM_ADU,
) -> RefinedWavelengths:
    """Refit a wavelength solution to the stellar and telluric lines of one exposure.

    The input solution serves only to predict where each line falls. The exposure is cut
    into 256-pixel chunks (128-pixel steps). Each chunk is correlated with the stellar
    template near the prediction, after the star's bulk velocity has been found from whole
    orders, and the chunks of rows 49-61 that hold strong telluric lines are also correlated
    with the telluric line list. From a telluric chunk the template spectrum, placed with the
    lags of the order's stellar chunks, is first projected out, so that stellar lines cannot
    correlate with the telluric ones by chance. A chunk whose lag differs by more than
    2 km/s from the median of its neighbours in the order is discarded as a spurious peak.
    The grating equation makes ``m * lambda`` a smooth function of the pixel ``x`` and the
    order ``m``; a two-dimensional Chebyshev series of degree ``nx`` in ``x`` and ``nm`` in
    ``m`` is fitted to both sets jointly by weighted least squares with sigma clipping, with
    the star's velocity as one more linear parameter:

    - a stellar chunk measures the rest wavelength of its centre, which is the topocentric
      one divided by ``1 + v_star / c``;
    - a telluric chunk measures the topocentric wavelength itself, which fixes the absolute
      scale.

    A smooth correction to the input solution cannot take its place: on exposures from
    2023-2026 the 2022 solution's telluric lags run from -3.5 to +6.1 km/s along a single
    order (row 56), a distortion that the series fitted to the lines does not have. Rows
    outside the fitted range keep the input solution, scaled pixel by pixel by the ratio
    found at the nearest fitted row.

    On an exposure of a K-type star from 2023-2026, with a BOSZ template of 4750 K at
    R = 20,000, the fit leaves 0.87 km/s rms per stellar chunk and 0.17 km/s per telluric
    chunk. On ten exposures of the same star not used in the fit, the velocities of
    1024-pixel chunks over rows 8-56 scatter by 0.56 km/s (MAD) about each exposure's
    median, against 0.86 km/s with the 2022 solution, and the blue quarter of the orders
    agrees with the rest to 0.1 km/s, against 1.2 km/s with the 2022 solution.

    Parameters
    ----------
    base : ApfWavelengths, array or path
        The solution to refine, e.g. ``"apf_wave_2022.fits"``.
    exposure : ApfExposure
        A high signal-to-noise exposure of a star with many metal lines (G and K types are
        best), taken without the iodine cell.
    template_wave, template_flux : array_like
        A synthetic spectrum of a star like it, at rest, air wavelengths, continuum 1. It must
        cover 4000-7000 A for the default rows. For example
        ``albireo.Template.from_library(...)`` with a BOSZ grid.
    template_resolving_power : float or None, optional
        The template's resolving power. The data are smoothed to it before the correlation.
        Default 20,000 (the BOSZ grids albireo fetches); ``None`` for an unbroadened
        template, then the data are not smoothed.
    lsf_fwhm_kms : float, optional
        The exposure's LSF FWHM. Default: the decker's typical value.
    rows : sequence of int, optional
        Rows with stellar chunks, between 0 and 78. Default: the rows whose centres lie
        inside the template.
    nx, nm : int, optional
        Degrees of the correction in the pixel and the order. Default 5 and 4.
    search_kms : float, optional
        Half-width of the search for the star's bulk velocity. Default 300.
    lines : TelluricLines, optional
        Default: the packaged list.
    gain, read_term : float, optional
        The noise model.

    Returns
    -------
    RefinedWavelengths

    Raises
    ------
    ValueError
        If too few chunks correlate to fit the correction, or ``rows`` is empty or holds a
        row outside 0-78.
    """
    from numpy.polynomial import chebyshev as cheb

    if exposure.iodine_in:
        raise ValueError(f"{exposure.path}: taken through the iodine cell; use another exposure")
    solution = _as_wavelengths(base)
    lines = telluric_lines() if lines is None else lines
    t_wave = np.asarray(template_wave, dtype=float)
    t_flux = np.asarray(template_flux, dtype=float)
    order = np.argsort(t_wave)
    t_lnw, t_dev = np.log(t_wave[order]), t_flux[order] - 1.0
    data_fwhm = lsf_fwhm_kms or DECKER_LSF_FWHM_KMS.get(exposure.decker, 4.5)
    data_sigma = data_fwhm / _FWHM_PER_SIGMA
    template_sigma = (
        0.0
        if template_resolving_power is None
        else _C_KMS / (float(template_resolving_power) * _FWHM_PER_SIGMA)
    )
    smooth_kms = math.sqrt(max(template_sigma**2 - data_sigma**2, 0.0))
    waves = solution.wave
    if rows is None:
        centres = waves[:, _N_PIX // 2]
        rows = [r for r in range(_N_ROWS) if t_wave.min() + 20 < centres[r] < t_wave.max() - 20]
        if not rows:
            raise ValueError("the template covers none of the orders")
    rows = _checked_rows(rows)

    orders = _Orders(exposure, gain, read_term)
    prepared = {}
    for row in rows:
        flux, good = orders.normalised(row)
        step_kms = float(np.median(np.diff(np.log(waves[row])))) * _C_KMS
        prepared[row] = (*_smooth(flux, good, smooth_kms / step_kms), step_kms)

    # The template shift that best matches the data, from whole-order correlations summed:
    # a lag, about minus the star's velocity relative to the template.
    coarse = np.arange(-search_kms, search_kms + 0.5, 1.0)
    total = np.zeros(coarse.size)
    for row in rows:
        data, ok, _ = prepared[row]
        if ok.mean() < 0.5:
            continue
        d = data[ok] - data[ok].mean()
        tm = _shifted_templates(np.log(waves[row][ok]), coarse, t_lnw, t_dev)
        total += np.clip(_correlation(tm, d), 0, None)
    bulk_lag_kms = float(coarse[int(np.argmax(total))])

    fine = np.arange(-15.0, 15.0 + 0.125, 0.25)
    stellar = []  # (row, x, lag of the chunk, r)
    for row in rows:
        data, ok, _ = prepared[row]
        lnw_all = np.log(waves[row]) + bulk_lag_kms / _C_KMS
        for c0 in range(64, _N_PIX - 64 - 256 + 1, 128):
            sl = slice(c0, c0 + 256)
            lag, r = _chunk_lag(data[sl], ok[sl], lnw_all[sl], t_lnw, t_dev, fine)
            if np.isfinite(lag) and r > 0.3:
                stellar.append((row, c0 + 128.0, lag + bulk_lag_kms, r))
    stellar_arr = _locally_consistent(np.array(stellar, dtype=float).reshape(-1, 4))
    tell_lnw = np.arange(math.log(6200.0), math.log(7500.0), 0.25 / _C_KMS)
    tell_dev = telluric_model(np.exp(tell_lnw), lines)
    telluric = []
    reference_sigma = 7.3 / _FWHM_PER_SIGMA
    # The star's lines in a telluric chunk would correlate with the telluric template by
    # chance. Where the order has stellar chunks, the template spectrum, placed with their
    # lags and brought to the resolution of the smoothed data, is projected out of the chunk
    # before the correlation (_chunk_lag).
    star_smooth_kms = math.sqrt(max(reference_sigma**2 - template_sigma**2, 0.0))
    for row in _TELLURIC_ROWS:
        flux, good = orders.normalised(row)
        step_kms = float(np.median(np.diff(np.log(waves[row])))) * _C_KMS
        smooth = math.sqrt(max(reference_sigma**2 - data_sigma**2, 0.0)) / step_kms
        data, ok = _smooth(flux, good, smooth)
        lnw_all = np.log(waves[row])
        own = stellar_arr[stellar_arr[:, 0] == row]
        star = None
        if len(own) >= 3:
            by_x = np.argsort(own[:, 1])
            shift = np.interp(np.arange(_N_PIX, dtype=float), own[by_x, 1], own[by_x, 2])
            star_dev = np.interp(lnw_all + shift / _C_KMS, t_lnw, t_dev, left=0.0, right=0.0)
            star, _ = _smooth(
                1.0 + star_dev, np.ones(_N_PIX, dtype=bool), star_smooth_kms / step_kms
            )
        for c0 in range(32, _N_PIX - 32 - 256 + 1, 128):
            sl = slice(c0, c0 + 256)
            segment = np.interp(lnw_all[sl], tell_lnw, tell_dev, left=0.0, right=0.0)
            if np.mean(np.abs(segment)) < 0.015 or segment.min() > -0.08:
                continue
            lag, r = _chunk_lag(
                data[sl],
                ok[sl],
                lnw_all[sl],
                tell_lnw,
                tell_dev,
                fine,
                nuisance=None if star is None else star[sl],
            )
            if np.isfinite(lag) and r > 0.5:
                telluric.append((row, c0 + 128.0, lag, r))
    telluric_arr = _locally_consistent(np.array(telluric, dtype=float).reshape(-1, 4))
    n_coef = (nx + 1) * (nm + 1)
    if len(stellar_arr) < 2 * n_coef or len(telluric_arr) < 5:
        raise ValueError(
            f"too few chunks correlate ({len(stellar_arr)} stellar, {len(telluric_arr)} "
            "telluric) to fit the solution: is the exposure line-rich, and does the template "
            "match it?"
        )
    fitted_rows = sorted(set(stellar_arr[:, 0].astype(int)) | set(telluric_arr[:, 0].astype(int)))
    m_lo = float(_ORDER_OF_ROW_ZERO - fitted_rows[-1])
    m_hi = float(_ORDER_OF_ROW_ZERO - fitted_rows[0])

    def design(rows_, xs_):
        m = _ORDER_OF_ROW_ZERO - np.asarray(rows_, float)
        xt = (np.asarray(xs_, float) - _XC) / _XC
        mt = (m - 0.5 * (m_lo + m_hi)) / max(0.5 * (m_hi - m_lo), 1.0)
        return cheb.chebvander2d(xt, mt, [nx, nm]), m

    def chunk_wave(arr):
        """The wavelength each chunk centre measures, from the base prediction and the lag."""
        base_at = np.array(
            [np.interp(x_, np.arange(_N_PIX), waves[int(r_)]) for r_, x_ in arr[:, :2]]
        )
        return base_at * np.exp(arr[:, 2] / _C_KMS)

    # m * lambda is a smooth function of the pixel and the order (the grating equation),
    # modelled as a Chebyshev series P(x, m). A stellar chunk measures the rest wavelength of
    # its centre, lambda_rest, and P = m * lambda_rest * (1 + v_star / c); a telluric chunk
    # measures the topocentric wavelength itself, and P = m * lambda_tell. In velocity units,
    # c * P / (m * lambda) - c equals v_star (stellar) or 0 (telluric), linear in the
    # coefficients and v_star.
    a_s, m_s = design(stellar_arr[:, 0], stellar_arr[:, 1])
    a_t, m_t = design(telluric_arr[:, 0], telluric_arr[:, 1])
    lam_s = chunk_wave(stellar_arr)
    lam_t = chunk_wave(telluric_arr)
    rows_s = _C_KMS * a_s / (m_s * lam_s)[:, None]
    rows_t = _C_KMS * a_t / (m_t * lam_t)[:, None]
    keep_s = np.ones(len(lam_s), bool)
    keep_t = np.ones(len(lam_t), bool)
    sig_s, sig_t = 1.0, 0.3
    coef = np.zeros(n_coef)
    v_star = 0.0
    r_s = np.zeros(len(lam_s))
    r_t = np.zeros(len(lam_t))
    for _ in range(12):
        a = np.vstack(
            [
                np.hstack([rows_s[keep_s], -np.ones((keep_s.sum(), 1))]) / sig_s,
                np.hstack([rows_t[keep_t], np.zeros((keep_t.sum(), 1))]) / sig_t,
            ]
        )
        b = np.concatenate(
            [np.full(keep_s.sum(), _C_KMS) / sig_s, np.full(keep_t.sum(), _C_KMS) / sig_t]
        )
        params, *_ = np.linalg.lstsq(a, b, rcond=None)
        coef, v_star = params[:-1], float(params[-1])
        r_s = rows_s @ coef - _C_KMS - v_star
        r_t = rows_t @ coef - _C_KMS
        sig_s = max(1.4826 * float(np.median(np.abs(r_s[keep_s]))), 0.05)
        sig_t = max(1.4826 * float(np.median(np.abs(r_t[keep_t]))), 0.02)
        new_s = np.abs(r_s) < 3.5 * sig_s
        new_t = np.abs(r_t) < 3.5 * sig_t
        if np.array_equal(new_s, keep_s) and np.array_equal(new_t, keep_t):
            break
        keep_s, keep_t = new_s, new_t
    x = np.arange(_N_PIX, dtype=float)
    lo_row, hi_row = fitted_rows[0], fitted_rows[-1]
    refined_wave = np.array(waves, copy=True)
    for row in range(lo_row, hi_row + 1):
        a_row, m_row = design(np.full(_N_PIX, row), x)
        refined_wave[row] = (a_row @ coef) / m_row
    # Outside the fitted rows the base solution is kept, scaled by the ratio found at the
    # nearest fitted row, pixel by pixel.
    for row in range(_N_ROWS):
        if row < lo_row or row > hi_row:
            edge = lo_row if row < lo_row else hi_row
            refined_wave[row] = waves[row] * (refined_wave[edge] / waves[edge])
    change = (refined_wave[lo_row : hi_row + 1] / waves[lo_row : hi_row + 1] - 1.0) * _C_KMS
    refined = ApfWavelengths(
        refined_wave, source=f"{solution.source} refined on {Path(exposure.path).name}"
    )
    return RefinedWavelengths(
        wavelengths=refined,
        v_star_kms=v_star,
        rms_stellar_kms=float(np.std(r_s[keep_s])),
        rms_telluric_kms=float(np.std(r_t[keep_t])),
        n_stellar=int(keep_s.sum()),
        n_telluric=int(keep_t.sum()),
        correction_range_kms=(float(change.min()), float(change.max())),
        rows=tuple(range(lo_row, hi_row + 1)),
    )

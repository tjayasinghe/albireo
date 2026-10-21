"""Telluric lines as a wavelength and resolution reference for ground-based spectra.

**Experimental.** The names and the packaged line list may change.

Telluric absorption is formed in the Earth's atmosphere, so its lines are at rest in the
topocentric frame whatever the star does. A spectrograph whose wavelength scale has drifted
since it was calibrated shows them displaced, and an instrument with a broader line-spread
function (LSF) shows them broader. Two measurements follow:

- :func:`telluric_shift`, the velocity shift of the wavelength scale that puts a spectrum's
  telluric lines at their rest positions: the corrected wavelengths are ``wave * (1 + v / c)``,
  so lines displaced to the red by ``u`` give ``v = -u``. The correlations of several bands
  are summed before the peak is taken, because the regular line spacing of the O2 B band
  alone has aliases 38 km/s apart.
- :func:`telluric_lsf`, the LSF width from Gaussian fits to weak, isolated O2 lines, with
  the lines' own width (about 1.3 km/s FWHM from pressure and Doppler broadening, an
  assumption) removed in quadrature.

The reference is the packaged line list (:func:`telluric_lines`): 183 lines of O2 and H2O
between 6276 and 7410 A, air wavelengths, measured from the median of 43 PEPSI red-arm
spectra on their ThAr scale (the provenance is in the file's header). A spectrum built from
the list reproduces the zero points of 93 APF exposures measured against the full median
spectrum to 0.031 km/s rms. H2O lines vary in strength with the humidity; their positions do
not, and only the positions are used.

The data must be continuum-normalised and in the topocentric frame. A barycentric-frame
spectrum is put back by dividing its wavelengths by ``1 + v_bary / c`` (albireo's sign
convention for ``v_bary``).
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

__all__ = [
    "INTRINSIC_FWHM_KMS",
    "TELLURIC_REGIONS",
    "TelluricLines",
    "TelluricShift",
    "telluric_lines",
    "telluric_lsf",
    "telluric_model",
    "telluric_shift",
]

_C_KMS = 299_792.458
_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))

#: The bands correlated by :func:`telluric_shift`: name to (lo, hi), air Angstrom,
#: topocentric. The O2 gamma band, the head and the P branch of the O2 B band, and two H2O
#: regions near 7200 A, the five regions the APF zero points were measured in.
TELLURIC_REGIONS: Mapping[str, tuple[float, float]] = {
    "O2_gamma": (6274.0, 6322.0),
    "O2_B_head": (6866.0, 6905.0),
    "O2_B_P": (6900.0, 6950.0),
    "H2O_7200": (7150.0, 7250.0),
    "H2O_7270": (7230.0, 7320.0),
}

#: The assumed width of an O2 line itself (pressure and Doppler broadening), FWHM in km/s,
#: removed in quadrature from the fitted widths by :func:`telluric_lsf`.
INTRINSIC_FWHM_KMS = 1.3

# The observed width of the lines in the median spectrum the list was measured from (the
# PEPSI LSF and the lines' own width). Data narrower than this are smoothed to it before the
# correlation, so that template and data have the same line shapes.
_REFERENCE_SIGMA_KMS = 7.3 / _FWHM_PER_SIGMA
_STEP_KMS = 0.25
_LINE_FILE = Path(__file__).resolve().parent / "data_files" / "telluric_lines_air.csv"


@dataclass(frozen=True, eq=False)
class TelluricLines:
    """A telluric line list.

    Attributes
    ----------
    wave : numpy.ndarray
        Air wavelengths in the topocentric frame, Angstrom.
    depth : numpy.ndarray
        Depth below the continuum in the spectrum the list was measured from.
    fwhm_kms : numpy.ndarray
        Observed FWHM in that spectrum, km/s (its LSF included).
    centre_err_kms : numpy.ndarray
        Formal error of each centre, km/s.
    isolated : numpy.ndarray
        ``True`` where no other line deeper than 0.05 lies within 15 km/s.
    band : numpy.ndarray
        ``"O2"`` or ``"H2O"``.
    source : str
        Where the list came from.
    """

    wave: np.ndarray
    depth: np.ndarray
    fwhm_kms: np.ndarray
    centre_err_kms: np.ndarray
    isolated: np.ndarray
    band: np.ndarray
    source: str = field(default="", compare=False)

    def __len__(self) -> int:
        return int(self.wave.size)

    def select(self, mask: np.ndarray) -> TelluricLines:
        """The lines where ``mask`` is ``True``."""
        mask = np.asarray(mask, dtype=bool)
        return TelluricLines(
            wave=self.wave[mask],
            depth=self.depth[mask],
            fwhm_kms=self.fwhm_kms[mask],
            centre_err_kms=self.centre_err_kms[mask],
            isolated=self.isolated[mask],
            band=self.band[mask],
            source=self.source,
        )

    def lsf_lines(self, depth_range: tuple[float, float] = (0.15, 0.55)) -> TelluricLines:
        """Isolated O2 lines of moderate depth, the ones whose width measures the LSF.

        Deeper lines are saturated and broader than the LSF; shallower ones have widths too
        noisy to use. With the packaged list and the default range this is 14 lines.
        """
        lo, hi = depth_range
        return self.select(
            (self.band == "O2") & self.isolated & (self.depth > lo) & (self.depth < hi)
        )


@lru_cache(maxsize=1)
def _packaged_lines() -> TelluricLines:
    with _LINE_FILE.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(line for line in handle if not line.startswith("#")))
    lines = TelluricLines(
        wave=np.array([float(r["wave_air"]) for r in rows]),
        depth=np.array([float(r["depth"]) for r in rows]),
        fwhm_kms=np.array([float(r["fwhm_kms"]) for r in rows]),
        centre_err_kms=np.array([float(r["centre_err_kms"]) for r in rows]),
        isolated=np.array([r["isolated"].strip() == "1" for r in rows]),
        band=np.array([r["band"].strip() for r in rows]),
        source=f"albireo packaged list ({_LINE_FILE.name})",
    )
    # One object serves every caller in the process, so its arrays are read-only.
    for column in (
        lines.wave,
        lines.depth,
        lines.fwhm_kms,
        lines.centre_err_kms,
        lines.isolated,
        lines.band,
    ):
        column.flags.writeable = False
    return lines


def telluric_lines() -> TelluricLines:
    """The packaged telluric line list (183 O2 and H2O lines, 6276-7410 A, air).

    Every call returns the same object; its arrays are read-only.
    """
    return _packaged_lines()


def telluric_model(
    wave, lines: TelluricLines | None = None, *, shift_kms: float = 0.0
) -> np.ndarray:
    """A telluric spectrum built from a line list, as a deviation from the continuum.

    Each line is a Gaussian of its listed depth and width, so the result has the
    resolution of the spectrum the list was measured from. The sum is limited to -0.98 so
    that blended saturated lines do not go below zero flux.

    Parameters
    ----------
    wave : array_like
        Air wavelengths, Angstrom.
    lines : TelluricLines, optional
        Default: :func:`telluric_lines`.
    shift_kms : float, optional
        Velocity of the lines, km/s (positive to longer wavelengths).

    Returns
    -------
    numpy.ndarray
        ``transmission - 1``, the same shape as ``wave``.
    """
    lines = telluric_lines() if lines is None else lines
    wave = np.asarray(wave, dtype=float)
    # Each line is evaluated only on the samples within 6 sigma of its centre, found by a
    # binary search on the sorted wavelengths with a 1 per cent margin. The samples, the
    # operations and the order in which the lines are added are those of a loop over the
    # whole array, so the result is identical to it.
    flat = wave.ravel()
    order = None
    if not np.all(flat[1:] >= flat[:-1]):
        order = np.argsort(flat, kind="stable")
        flat = flat[order]
    acc = np.zeros_like(flat)
    centres = lines.wave * (1.0 + shift_kms / _C_KMS)
    sigmas = lines.fwhm_kms / _FWHM_PER_SIGMA
    for centre, depth, sigma in zip(centres, lines.depth, sigmas, strict=True):
        half = 1.01 * 6.0 * sigma / _C_KMS
        i0, i1 = np.searchsorted(flat, (centre * (1.0 - half), centre * (1.0 + half)))
        v = (flat[i0:i1] / centre - 1.0) * _C_KMS
        near = np.abs(v) < 6.0 * sigma
        window = acc[i0:i1]
        window[near] -= depth * np.exp(-0.5 * (v[near] / sigma) ** 2)
    if order is None:
        out = acc
    else:
        out = np.empty_like(acc)
        out[order] = acc
    return np.maximum(out.reshape(wave.shape), -0.98)


@dataclass(frozen=True)
class TelluricShift:
    """The result of :func:`telluric_shift`.

    Attributes
    ----------
    v_kms : float
        The shift of the wavelength scale that puts the telluric lines at their rest
        positions, km/s: the corrected wavelengths are ``wave * (1 + v_kms / c)``. Lines
        displaced to the red by ``u`` km/s give ``-u``.
    err_kms : float
        Weighted scatter of the regions about the mean, divided by ``sqrt(n - 1)``; ``nan``
        for a single region.
    regions : Mapping[str, tuple[float, float]]
        Per region, the velocity and the peak correlation coefficient.
    """

    v_kms: float
    err_kms: float
    regions: Mapping[str, tuple[float, float]]

    @property
    def n_regions(self) -> int:
        """Number of regions that gave a correlation peak."""
        return len(self.regions)


def _segments(segments) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out = []
    for segment in segments:
        wave, flux, good = (np.asarray(a) for a in segment)
        out.append((wave.astype(float), flux.astype(float), good.astype(bool)))
    return out


def _covering(segments, lo: float, hi: float):
    """The segment with the most good pixels inside ``[lo, hi]``, or ``None``."""
    best, count = None, 0
    for wave, flux, good in segments:
        inside = good & (wave > lo) & (wave < hi)
        if inside.sum() > count:
            best, count = (wave, flux, good), int(inside.sum())
    return best


def _smooth(flux, good, sigma_pix: float):
    """``flux - 1`` smoothed by a Gaussian of ``sigma_pix``, ignoring bad pixels."""
    deviation = np.where(good, flux - 1.0, 0.0)
    if sigma_pix < 0.3:
        return deviation, good
    half = max(math.ceil(5.0 * sigma_pix), 1)
    x = np.arange(-half, half + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_pix) ** 2)
    kernel /= kernel.sum()
    num = np.convolve(deviation, kernel, mode="same")
    den = np.convolve(good.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0.5, num / den, 0.0)
    return out, den > 0.5


def _parabola_peak(values: np.ndarray, j: int) -> float:
    """Offset of the vertex of the parabola through ``values[j - 1 : j + 2]``, in samples."""
    y0, y1, y2 = values[j - 1], values[j], values[j + 1]
    den = y0 - 2.0 * y1 + y2
    return 0.5 * (y0 - y2) / den if den != 0 else 0.0


def _shifted_templates(
    lnw: np.ndarray, lags: np.ndarray, template_lnw: np.ndarray, template: np.ndarray
) -> np.ndarray:
    """Rows of the template at ``lnw + lag / c``, one per lag (km/s), each less its mean."""
    tm = np.interp(lnw[None, :] + lags[:, None] / _C_KMS, template_lnw, template)
    return tm - tm.mean(axis=1, keepdims=True)


def _correlation(tm: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Correlation coefficient of mean-free ``data`` with each mean-free row of ``tm``."""
    return (tm @ data) / (np.sqrt((tm * tm).sum(axis=1) * (data * data).sum()) + 1e-30)


def telluric_shift(
    segments: Iterable[Sequence[np.ndarray]],
    *,
    lsf_sigma_kms: float,
    lines: TelluricLines | None = None,
    regions: Mapping[str, tuple[float, float]] = TELLURIC_REGIONS,
    v_range: float = 60.0,
    min_pixels: int = 300,
    min_correlation: float = 0.3,
) -> TelluricShift:
    """The velocity shift that puts a spectrum's telluric lines at their rest positions.

    For each region the segment with the most good pixels in it is smoothed to the width of
    the reference lines and correlated with :func:`telluric_model` over
    ``+-v_range`` km/s in steps of 0.25 km/s. The positive parts of the correlations are
    summed over regions and the highest peak is taken. Each region is then refined to its
    own peak within 6 km/s of it, and the regions are averaged with weights ``r**4``, so
    that a region with weak or blended telluric lines contributes little. A region whose
    highest correlation within those 6 km/s lies on the edge of the window is left out,
    because its own peak may lie outside the window.

    Parameters
    ----------
    segments : iterable of (wave, flux, good)
        Continuum-normalised pieces of one spectrum in the topocentric frame (echelle
        orders, or a merged spectrum). ``good`` is ``True`` where a pixel may be used.
    lsf_sigma_kms : float
        The LSF sigma of the data, km/s. Data narrower than the reference lines are smoothed
        to them before the correlation.
    lines : TelluricLines, optional
        Default: :func:`telluric_lines`.
    regions : mapping, optional
        Region name to ``(lo, hi)``, air Angstrom. Default :data:`TELLURIC_REGIONS`.
    v_range : float, optional
        Half-width of the velocity search, km/s. Default 60.
    min_pixels : int, optional
        Good pixels a region needs inside its range. Default 300.
    min_correlation : float, optional
        Peak correlation coefficient a region needs to be used. Default 0.3.

    Returns
    -------
    TelluricShift

    Raises
    ------
    ValueError
        If no region has enough good pixels or a usable correlation peak within 6 km/s of
        the summed correlations' peak.
    """
    lines = telluric_lines() if lines is None else lines
    segments = _segments(segments)
    los = [lo for lo, _ in regions.values()]
    his = [hi for _, hi in regions.values()]
    pad = 1.2 * v_range / _C_KMS
    grid_lnw = np.arange(math.log(min(los)) - pad, math.log(max(his)) + pad, _STEP_KMS / _C_KMS)
    template = telluric_model(np.exp(grid_lnw), lines)
    lags = np.arange(-v_range, v_range + _STEP_KMS / 2.0, _STEP_KMS)
    smooth_kms = math.sqrt(max(_REFERENCE_SIGMA_KMS**2 - float(lsf_sigma_kms) ** 2, 0.0))
    total = np.zeros(lags.size)
    curves: dict[str, np.ndarray] = {}
    for name, (lo, hi) in regions.items():
        found = _covering(segments, lo, hi)
        if found is None:
            continue
        wave, flux, good = found
        step_kms = float(np.median(np.diff(np.log(wave)))) * _C_KMS
        smoothed, ok = _smooth(flux, good, smooth_kms / max(step_kms, 1e-6))
        sel = ok & (wave > lo) & (wave < hi)
        if sel.sum() < min_pixels:
            continue
        tm = _shifted_templates(np.log(wave[sel]), lags, grid_lnw, template)
        curves[name] = _correlation(tm, smoothed[sel] - smoothed[sel].mean())
        total += np.clip(curves[name], 0.0, None)
    if not curves:
        raise ValueError(
            "no telluric region has enough good pixels: the spectrum must cover at least one "
            f"of {dict(regions)} (air Angstrom, topocentric) with {min_pixels} good pixels"
        )
    j0 = int(np.argmax(total))
    idx = np.where(np.abs(lags - lags[j0]) <= 6.0)[0]
    found_regions: dict[str, tuple[float, float]] = {}
    for name, r in curves.items():
        j = int(idx[int(np.argmax(r[idx]))])
        # A maximum on the edge of the window is left out: the region's own peak may lie
        # outside it, and a parabola through a monotonic slope extrapolates without bound.
        # It also drops a peak that lies less than half a lag step inside the edge.
        if idx[0] < j < idx[-1] and r[j] > min_correlation:
            v = float(lags[j] + _parabola_peak(r, j) * _STEP_KMS)
            found_regions[name] = (v, float(r[j]))
    if not found_regions:
        raise ValueError(
            f"no telluric region gave a correlation peak above {min_correlation} within "
            "6 km/s of the peak of the summed correlations: is the spectrum normalised and "
            "in the topocentric frame?"
        )
    values = np.array([v for v, _ in found_regions.values()])
    weights = np.array([r for _, r in found_regions.values()]) ** 4
    mean = float(np.average(values, weights=weights))
    if values.size > 1:
        var = float(np.average((values - mean) ** 2, weights=weights))
        err = math.sqrt(var / (values.size - 1))
    else:
        err = float("nan")
    return TelluricShift(v_kms=mean, err_kms=err, regions=found_regions)


def _gaussian_fwhm(wave, flux, good, centre: float, half_kms: float = 12.0) -> float:
    from scipy.optimize import curve_fit

    sel = good & (np.abs(wave / centre - 1.0) * _C_KMS < half_kms)
    if sel.sum() < 10:
        return float("nan")
    v = (wave[sel] / centre - 1.0) * _C_KMS
    y = flux[sel]

    def profile(vv, a, mu, sig, c0, c1):
        return c0 + c1 * vv - a * np.exp(-0.5 * ((vv - mu) / sig) ** 2)

    try:
        p, _ = curve_fit(
            profile,
            v,
            y,
            p0=[0.3, 0.0, 2.0, 1.0, 0.0],
            bounds=([0.03, -3.0, 0.4, 0.6, -0.05], [1.0, 3.0, 6.0, 1.4, 0.05]),
            maxfev=5000,
        )
    except (RuntimeError, ValueError):
        return float("nan")
    if not 0.45 < p[2] < 5.9:
        return float("nan")
    return float(_FWHM_PER_SIGMA * p[2])


def telluric_lsf(
    segments: Iterable[Sequence[np.ndarray]],
    *,
    shift_kms: float = 0.0,
    lines: TelluricLines | None = None,
    intrinsic_fwhm_kms: float = INTRINSIC_FWHM_KMS,
    min_lines: int = 3,
) -> tuple[float, int] | None:
    """The LSF FWHM from Gaussian fits to weak isolated O2 lines.

    Each line of :meth:`TelluricLines.lsf_lines` that lies inside a segment is fitted with a
    Gaussian on a linear continuum within 12 km/s of its centre. The median fitted FWHM,
    with ``intrinsic_fwhm_kms`` removed in quadrature, is the LSF FWHM.

    Parameters
    ----------
    segments : iterable of (wave, flux, good)
        As for :func:`telluric_shift`.
    shift_kms : float, optional
        The telluric shift of the spectrum (:func:`telluric_shift`), applied to its
        wavelengths before the fits. Default 0.
    lines : TelluricLines, optional
        Default: :func:`telluric_lines`.
    intrinsic_fwhm_kms : float, optional
        The lines' own width, removed in quadrature. Default :data:`INTRINSIC_FWHM_KMS`.
    min_lines : int, optional
        Lines that must give a fit. Default 3.

    Returns
    -------
    (float, int) or None
        The LSF FWHM in km/s and the number of lines used, or ``None`` when fewer than
        ``min_lines`` lines could be fitted.
    """
    lines = (telluric_lines() if lines is None else lines).lsf_lines()
    widths = []
    for wave, flux, good in _segments(segments):
        corrected = wave * (1.0 + shift_kms / _C_KMS)
        for centre in lines.wave:
            if corrected[0] + 0.3 < centre < corrected[-1] - 0.3:
                widths.append(_gaussian_fwhm(corrected, flux, good, float(centre)))
    widths_arr = np.array(widths, dtype=float)
    widths_arr = widths_arr[np.isfinite(widths_arr)]
    if widths_arr.size < min_lines:
        return None
    observed = float(np.median(widths_arr))
    fwhm = math.sqrt(max(observed**2 - intrinsic_fwhm_kms**2, 0.25))
    return fwhm, int(widths_arr.size)

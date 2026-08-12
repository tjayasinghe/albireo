"""Reading reduced 1-D spectra from FITS into an albireo :class:`~albireo.data.Dataset`.

This is the thinnest layer that can honestly get archival data into the model, and its
job is mostly to answer questions the FITS header can answer and the user should not have
to: what frame are these wavelengths in, when exactly was the exposure taken, what
barycentric correction has already been applied, and is the flux normalized.

Two container formats are recognized automatically:

- **Binary-table spectra** — a ``BinTableHDU`` with one row holding ``WAVE``/``FLUX``
  (and usually ``ERR``) array columns. This is the ESO Phase-3 / IVOA ``SPECTRUM v2.0``
  layout used by the ESO Science Archive for FEROS, HARPS, UVES, X-shooter and others.
- **WCS image spectra** — a 1-D image HDU with a linear or log-linear dispersion in
  ``CRVAL1``/``CDELT1``/``CRPIX1``. The classic IRAF-style product.

What the reader will *not* do is guess about the frame or the time. If a header does not
say, you get a warning naming the assumption, not a silent default (:func:`read_spectrum`).

The three header facts that decide whether a velocity comes out right:

**``SPECSYS``** — the frame the wavelengths are in. ``BARYCENT`` means the pipeline has
already applied the correction and albireo must *not* apply it again; it composes the
barycentric motion into the telluric component instead (``docs/math.md`` §1.2). This maps
onto :class:`albireo.data.Dataset`'s ``frame``.

**The applied barycentric velocity** — needed even when the correction has been applied,
because the telluric component is at rest in the topocentric frame and therefore *moves*
in the barycentric one. Read from the pipeline's own keyword where there is one
(``ESO DRS BARYCORR`` for FEROS, ``ESO DRS BERV`` for HARPS, ...), because the pipeline's
value is what actually defines the frame of the delivered wavelengths; recomputed with
astropy only as a fallback. albireo's sign convention is that a barycentric-frame
wavelength is ``exp(xi(v_bary))`` times the topocentric one, which is the same convention
as every ``BERV``-style keyword: the correction you *add* to a measured radial velocity.

**The time** — converted to BJD_TDB at mid-exposure. The barycentric light-travel
correction swings by up to 8.3 minutes over a year, which on a 40-day orbit with
``K = 60 km/s`` is a systematic radial-velocity error of ~0.05 km/s that no amount of data
averages away, because it is a function of the observing date.

Requires ``astropy`` (``pip install "albireo[io]"``); the rest of albireo does not.
"""

from __future__ import annotations

import glob
import math
import os
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from albireo.data import Dataset, EpochData
from albireo.preprocess import (
    estimate_ivar,
    mask_ranges,
    mask_spikes,
    mask_tellurics,
    normalize,
    select_region,
)

__all__ = [
    "RawSpectrum",
    "read_dataset",
    "read_spectrum",
    "to_epoch",
]

_C_KMS = 299_792.458

# Keywords under which a pipeline records the barycentric velocity correction it applied,
# in the order they are trusted. All use the "add this to a measured RV" sign convention.
_BARYCORR_KEYS: tuple[str, ...] = (
    "ESO DRS BARYCORR",  # FEROS (FERN/MIDAS DRS), and the ESO DRS family generally
    "ESO DRS BERV",  # HARPS, HARPS-N
    "ESO QC BERV",  # ESPRESSO
    "HIERARCH ESO DRS BARYCORR",
    "HIERARCH ESO DRS BERV",
    "BERV",
    "BVCORR",
    "BARYCORR",
    "VHELIO",  # heliocentric; ~10 m/s from barycentric, warned about below
    "HELIOCOR",
)

_WAVE_COLUMNS = ("WAVE", "WAVELENGTH", "LAMBDA", "SPECTRAL_AXIS")
_FLUX_COLUMNS = ("FLUX", "FLUX_REDUCED", "SPEC", "INTENSITY")
_ERR_COLUMNS = ("ERR", "ERROR", "SIGMA", "FLUX_ERR", "ERR_REDUCED", "UNCERTAINTY", "NOISE")

# Multiply a wavelength in these units to get Angstrom.
_WAVE_UNITS: dict[str, float] = {
    "angstrom": 1.0,
    "angstroms": 1.0,
    "a": 1.0,
    "aa": 1.0,
    "0.1nm": 1.0,
    "nm": 10.0,
    "nanometer": 10.0,
    "nanometre": 10.0,
    "um": 1e4,
    "micron": 1e4,
    "micrometer": 1e4,
    "m": 1e10,
    "meter": 1e10,
    "metre": 1e10,
}


def _require_astropy():
    """Import astropy, or explain how to get it."""
    try:
        from astropy import units  # noqa: F401
        from astropy.coordinates import EarthLocation, SkyCoord  # noqa: F401
        from astropy.io import fits
        from astropy.time import Time  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without astropy
        raise ImportError(
            "albireo.io needs astropy to read FITS files. Install it with "
            "'pip install \"albireo[io]\"' (or 'pip install astropy'). The rest of "
            "albireo has no astropy dependency — if you already have wavelengths, "
            "fluxes and inverse variances in memory, build EpochData directly."
        ) from exc
    from astropy.io import fits

    return fits


@dataclass(frozen=True)
class RawSpectrum:
    """One spectrum as the file describes it, before any science decisions are made.

    The separation from :class:`~albireo.data.EpochData` is deliberate: this holds what
    the file says, in the file's own units, including the things ``EpochData`` will not
    accept (an unnormalized flux, a missing or all-``NaN`` error array, negative fluxes).
    Turning it into an ``EpochData`` means choosing a continuum, a noise model and a set
    of masks, and those are choices — :func:`to_epoch` makes them explicit.

    Attributes
    ----------
    wave : numpy.ndarray
        Wavelengths in **Angstrom**, converted from whatever unit the file used.
    flux : numpy.ndarray
        Flux as stored. Not normalized unless :attr:`continuum_normalized` is ``True``.
    err : numpy.ndarray or None
        Flux uncertainties in the same units, or ``None`` when the file has no error
        array *or* the one it has is entirely non-finite (which ESO Phase-3 FEROS
        products are — the header even says so: "Error spectrum not available").
    bjd : float
        Mid-exposure time. BJD_TDB when :attr:`time_source` says so.
    v_bary : float
        Barycentric velocity correction in km/s, in albireo's sign convention.
    frame : {"topocentric", "barycentric"}
        The frame :attr:`wave` is in, for :class:`albireo.data.Dataset`.
    instrument : str
        Key into albireo's per-instrument LSF and response tables.
    resolving_power : float or None
        ``R = lambda / dlambda`` from the header, if it gave one. Divide ``c / R`` by
        ``2 sqrt(2 ln 2)`` to get the Gaussian LSF sigma in km/s that
        :func:`albireo.forward.build_problem` wants.
    wave_medium : {"air", "vacuum", "unknown"}
        Whether the wavelengths are air or vacuum. Does not affect disentangling — both
        components share one wavelength solution — but it does affect any comparison with
        line lists or synthetic spectra.
    continuum_normalized : bool
        What the header claims (``CONTNORM``). Not verified against the data.
    time_source : str
        How :attr:`bjd` was obtained, e.g. ``"BJD_TDB from TMID"``. Recorded rather than
        assumed, because a silently uncorrected time is a systematic error that looks
        like orbital scatter.
    path : str
        The file this came from.
    header : Mapping
        The primary FITS header, for anything this class does not model.
    """

    wave: np.ndarray
    flux: np.ndarray
    err: np.ndarray | None
    bjd: float
    v_bary: float
    frame: str
    instrument: str
    resolving_power: float | None = None
    wave_medium: str = "unknown"
    continuum_normalized: bool = False
    time_source: str = "unknown"
    path: str = ""
    header: Mapping = field(default_factory=dict, repr=False)

    @property
    def n_pixels(self) -> int:
        """Number of samples in the spectrum."""
        return int(self.wave.size)

    @property
    def lsf_sigma_kms(self) -> float | None:
        """Gaussian LSF sigma in km/s implied by :attr:`resolving_power`, or ``None``.

        Converts the resolving power to a *sigma* on the assumption that ``R`` quotes a
        FWHM, which is the usual convention: ``sigma = c / (R * 2 sqrt(2 ln 2))``.
        This is a starting value — the true LSF is neither exactly Gaussian nor exactly
        constant with wavelength, which is why albireo can infer the width
        (the ``lsf_sigma`` site of :class:`albireo.inference.MarginalOrbitModel`).
        """
        if self.resolving_power is None or not self.resolving_power > 0:
            return None
        return _C_KMS / (self.resolving_power * 2.0 * math.sqrt(2.0 * math.log(2.0)))

    def summary(self) -> str:
        """One-line human-readable description of this spectrum."""
        err = "no error array" if self.err is None else "with errors"
        res = "" if self.resolving_power is None else f", R={self.resolving_power:.0f}"
        return (
            f"{os.path.basename(self.path) or '<memory>'}: {self.n_pixels} px "
            f"{self.wave[0]:.1f}-{self.wave[-1]:.1f} A ({self.wave_medium}), {err}, "
            f"{self.instrument}{res}, {self.frame}, v_bary={self.v_bary:+.3f} km/s, "
            f"bjd={self.bjd:.5f} ({self.time_source})"
        )


def _header_get(header, keys: Iterable[str], default=None):
    """First present value among ``keys`` (case-insensitive, HIERARCH-tolerant)."""
    for key in keys:
        stripped = key[9:] if key.upper().startswith("HIERARCH ") else key
        if stripped in header:
            value = header[stripped]
            if value is not None and value != "":
                return value
    return default


def _wave_scale(unit: str | None, path: str) -> float:
    """Factor converting ``unit`` to Angstrom; 1.0 with a warning when unrecognized."""
    if not unit:
        return 1.0
    key = str(unit).strip().lower().replace(" ", "")
    if key in _WAVE_UNITS:
        return _WAVE_UNITS[key]
    warnings.warn(
        f"{path}: unrecognized wavelength unit {unit!r}; assuming Angstrom. Pass "
        "wave_scale= to read_spectrum if that is wrong.",
        RuntimeWarning,
        stacklevel=3,
    )
    return 1.0


def _find_column(columns: Sequence[str], candidates: Iterable[str]) -> str | None:
    """First column whose name matches a candidate (case-insensitive)."""
    upper = {name.upper(): name for name in columns}
    for candidate in candidates:
        if candidate in upper:
            return upper[candidate]
    return None


def _read_bintable(hdulist, path: str, wave_scale: float | None):
    """``(wave, flux, err, table_header)`` from the first spectrum-shaped binary table."""
    from astropy.io import fits

    for hdu in hdulist:
        if not isinstance(hdu, fits.BinTableHDU) or hdu.data is None or len(hdu.data) < 1:
            continue
        names = list(hdu.columns.names)
        wave_col = _find_column(names, _WAVE_COLUMNS)
        flux_col = _find_column(names, _FLUX_COLUMNS)
        if wave_col is None or flux_col is None:
            continue
        row = 0
        wave = np.asarray(hdu.data[wave_col][row], dtype=np.float64).ravel()
        flux = np.asarray(hdu.data[flux_col][row], dtype=np.float64).ravel()
        if wave.size != flux.size or wave.size < 2:
            continue
        err_col = _find_column(names, _ERR_COLUMNS)
        err = None
        if err_col is not None:
            candidate = np.asarray(hdu.data[err_col][row], dtype=np.float64).ravel()
            if candidate.size == wave.size and np.any(np.isfinite(candidate) & (candidate > 0)):
                err = candidate
        unit = hdu.columns[wave_col].unit
        scale = wave_scale if wave_scale is not None else _wave_scale(unit, path)
        return wave * scale, flux, err, hdu.header
    return None


def _read_wcs_image(hdulist, path: str, wave_scale: float | None):
    """``(wave, flux, None, header)`` from a 1-D image HDU with a linear/log dispersion."""
    for hdu in hdulist:
        data = getattr(hdu, "data", None)
        if data is None or np.ndim(data) != 1 or np.size(data) < 2:
            continue
        header = hdu.header
        crval = header.get("CRVAL1")
        cdelt = header.get("CDELT1", header.get("CD1_1"))
        if crval is None or cdelt is None:
            continue
        crpix = float(header.get("CRPIX1", 1.0))
        flux = np.asarray(data, dtype=np.float64)
        wave = float(crval) + float(cdelt) * (np.arange(flux.size) + 1.0 - crpix)
        ctype = str(header.get("CTYPE1", "")).upper()
        if "LOG" in ctype or str(header.get("DC-FLAG", "0")) not in ("0", "0.0", "False"):
            wave = np.power(10.0, wave)
        scale = wave_scale if wave_scale is not None else _wave_scale(header.get("CUNIT1"), path)
        return wave * scale, flux, None, header
    return None


def _observatory_location(header, path: str):
    """An astropy ``EarthLocation`` from the header, or ``None`` if it does not say."""
    from astropy import units as u
    from astropy.coordinates import EarthLocation

    lon = _header_get(header, ("ESO TEL GEOLON", "GEOLON", "OBS-LONG", "LONGITUD", "TELLONG"))
    lat = _header_get(header, ("ESO TEL GEOLAT", "GEOLAT", "OBS-LAT", "LATITUDE", "TELLAT"))
    elev = _header_get(
        header, ("ESO TEL GEOELEV", "GEOELEV", "OBS-ELEV", "ALTITUDE", "TELALT"), 0.0
    )
    if lon is not None and lat is not None:
        return EarthLocation.from_geodetic(
            lon=float(lon) * u.deg, lat=float(lat) * u.deg, height=float(elev) * u.m
        )
    x = header.get("OBSGEO-X")
    y = header.get("OBSGEO-Y")
    z = header.get("OBSGEO-Z")
    if x is not None and y is not None and z is not None:
        return EarthLocation.from_geocentric(float(x) * u.m, float(y) * u.m, float(z) * u.m)
    return None


def _sky_coord(header, path: str):
    """An astropy ``SkyCoord`` from the header, or ``None``."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    ra = _header_get(header, ("RA", "CRVAL1_OBJ", "OBJRA", "ESO TEL TARG ALPHA"))
    dec = _header_get(header, ("DEC", "OBJDEC", "ESO TEL TARG DELTA"))
    if ra is None or dec is None:
        return None
    try:
        return SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg, frame="icrs")
    except (TypeError, ValueError):
        return None


def _mid_exposure_mjd_utc(header, table_header) -> tuple[float, str]:
    """``(mjd_utc_at_mid_exposure, provenance)`` from the best keywords available."""
    for hdr in (table_header, header):
        if hdr is not None and "TMID" in hdr:
            return float(hdr["TMID"]), "TMID"
    mjd_obs = _header_get(header, ("MJD-OBS", "MJD_OBS"))
    exptime = _header_get(header, ("TEXPTIME", "EXPTIME", "EXPOSURE"), 0.0)
    if mjd_obs is not None:
        return float(mjd_obs) + 0.5 * float(exptime) / 86400.0, "MJD-OBS + EXPTIME/2"
    jd = _header_get(header, ("JD", "JD-OBS"))
    if jd is not None:
        return float(jd) - 2400000.5 + 0.5 * float(exptime) / 86400.0, "JD + EXPTIME/2"
    date_obs = _header_get(header, ("DATE-OBS",))
    if date_obs is not None:
        from astropy.time import Time

        return (
            float(Time(str(date_obs), format="isot", scale="utc").mjd)
            + 0.5 * float(exptime) / 86400.0,
            "DATE-OBS + EXPTIME/2",
        )
    raise ValueError(
        "cannot determine the observation time: none of TMID, MJD-OBS, JD or DATE-OBS is "
        "present. Pass bjd= to read_spectrum to supply it yourself."
    )


def _barycentric_time(mjd_utc: float, header, path: str) -> tuple[float, str]:
    """``(bjd_tdb, provenance)``; falls back to JD_UTC with a warning if it cannot."""
    from astropy.time import Time

    location = _observatory_location(header, path)
    coord = _sky_coord(header, path)
    if location is None or coord is None:
        missing = "observatory location" if location is None else "target coordinates"
        warnings.warn(
            f"{path}: no {missing} in the header, so the time cannot be put on the "
            "barycentre. Using JD_UTC at mid-exposure, which carries a periodic error of "
            "up to 8.3 minutes — enough to bias a short-period orbit. Pass bjd= "
            "explicitly, or location=/coord= to read_spectrum.",
            RuntimeWarning,
            stacklevel=4,
        )
        return mjd_utc + 2400000.5, "JD_UTC (no barycentric correction)"
    time = Time(mjd_utc, format="mjd", scale="utc", location=location)
    ltt = time.light_travel_time(coord, kind="barycentric")
    return float((time.tdb + ltt).jd), "BJD_TDB"


def _barycentric_velocity(header, mjd_utc: float, path: str) -> tuple[float, str]:
    """``(v_bary_kms, provenance)``: the pipeline's own value, else computed, else 0."""
    for key in _BARYCORR_KEYS:
        stripped = key[9:] if key.upper().startswith("HIERARCH ") else key
        if stripped in header:
            value = header[stripped]
            if value is None or value == "":
                continue
            if "HELIO" in stripped.upper():
                warnings.warn(
                    f"{path}: only a heliocentric velocity ({stripped}) is available; using it "
                    "as the barycentric one. The two differ by up to ~10 m/s, which matters "
                    "only for sub-10 m/s work.",
                    RuntimeWarning,
                    stacklevel=4,
                )
            return float(value), f"header {stripped}"
    location = _observatory_location(header, path)
    coord = _sky_coord(header, path)
    if location is None or coord is None:
        return 0.0, "assumed 0 (no keyword, and too little header information to compute)"
    from astropy.time import Time

    # The location goes on the Time, not on the call: astropy raises if it is given both.
    time = Time(mjd_utc, format="mjd", scale="utc", location=location)
    v = coord.radial_velocity_correction("barycentric", obstime=time)
    return float(v.to("km/s").value), "computed with astropy"


def _frame_from_specsys(header, path: str) -> str:
    """albireo frame name from ``SPECSYS``, warning when the header does not say."""
    specsys = str(_header_get(header, ("SPECSYS", "SPECSYSA"), "") or "").strip().upper()
    if specsys.startswith("BARYCENT") or specsys.startswith("BARY"):
        return "barycentric"
    if specsys.startswith("HELIOCEN") or specsys.startswith("HELIO"):
        warnings.warn(
            f"{path}: SPECSYS={specsys!r}; treating heliocentric as barycentric. The two "
            "frames differ by up to ~10 m/s.",
            RuntimeWarning,
            stacklevel=4,
        )
        return "barycentric"
    if specsys.startswith("TOPOCENT") or specsys.startswith("TOPO"):
        return "topocentric"
    warnings.warn(
        f"{path}: no SPECSYS keyword, so the wavelength frame is undeclared. Assuming "
        "'topocentric' (uncorrected, as observed). If the pipeline already applied the "
        "barycentric correction — most modern echelle pipelines do — pass "
        "frame='barycentric', or every velocity will be offset by the correction.",
        RuntimeWarning,
        stacklevel=4,
    )
    return "topocentric"


def read_spectrum(
    path: str,
    *,
    instrument: str | None = None,
    frame: str | None = None,
    bjd: float | None = None,
    v_bary: float | None = None,
    resolving_power: float | None = None,
    wave_scale: float | None = None,
) -> RawSpectrum:
    """Read one 1-D FITS spectrum into a :class:`RawSpectrum`.

    Detects the container (binary table first, then a WCS image) and reads the frame,
    time and barycentric velocity from the header, warning about anything it had to
    assume. Every derived quantity can be overridden by an argument.

    Parameters
    ----------
    path : str
        Path to the FITS file.
    instrument : str, optional
        Instrument key. Default: the ``INSTRUME`` keyword, else ``"default"``.
        This is what ties an epoch to its LSF width and response, so keep it stable
        across the epochs you want treated as one instrument.
    frame : {"topocentric", "barycentric"}, optional
        Override the frame instead of taking it from ``SPECSYS``.
    bjd : float, optional
        Override the mid-exposure BJD_TDB.
    v_bary : float, optional
        Override the barycentric velocity correction, km/s.
    resolving_power : float, optional
        Override ``R``; otherwise read from ``SPEC_RES``/``SPECRES``/``R``.
    wave_scale : float, optional
        Factor converting the file's wavelengths to Angstrom. Default: from the column
        or ``CUNIT1`` unit string.

    Returns
    -------
    RawSpectrum

    Raises
    ------
    ImportError
        If astropy is not installed.
    ValueError
        If no recognizable spectrum is found in the file, or the wavelengths are not
        strictly increasing after unit conversion.

    Examples
    --------
    >>> raw = read_spectrum("ADP.2016-09-20T12-03-37.453.fits")  # doctest: +SKIP
    >>> print(raw.summary())  # doctest: +SKIP
    ADP...fits: 189628 px 3527.2-9216.1 A (air), no error array, FEROS, R=48000, \
barycentric, v_bary=-21.708 km/s, bjd=2453243.51... (BJD_TDB)
    """
    fits = _require_astropy()
    with fits.open(path, memmap=False) as hdulist:
        parsed = _read_bintable(hdulist, path, wave_scale) or _read_wcs_image(
            hdulist, path, wave_scale
        )
        if parsed is None:
            raise ValueError(
                f"{path}: no 1-D spectrum found. Expected either a binary table with "
                f"{_WAVE_COLUMNS[0]}/{_FLUX_COLUMNS[0]} array columns, or a 1-D image with "
                "CRVAL1/CDELT1. Read the arrays yourself and build an EpochData directly "
                "if the file uses another layout."
            )
        wave, flux, err, table_header = parsed
        header = hdulist[0].header.copy()

    order = np.argsort(wave)
    if not np.array_equal(order, np.arange(wave.size)):
        wave, flux = wave[order], flux[order]
        err = None if err is None else err[order]
    if np.any(np.diff(wave) <= 0):
        raise ValueError(
            f"{path}: wavelengths are not strictly increasing even after sorting "
            "(duplicated samples?). albireo never resamples, so the grid must be clean."
        )

    mjd_utc, time_key = _mid_exposure_mjd_utc(header, table_header)
    if bjd is None:
        bjd_value, time_source = _barycentric_time(mjd_utc, header, path)
        time_source = f"{time_source} from {time_key}"
    else:
        bjd_value, time_source = float(bjd), "supplied by the caller"
    if v_bary is None:
        v_bary_value, _ = _barycentric_velocity(header, mjd_utc, path)
    else:
        v_bary_value = float(v_bary)
    frame_value = frame if frame is not None else _frame_from_specsys(header, path)

    if resolving_power is None:
        res = _header_get(header, ("SPEC_RES", "SPECRES", "RESOLUTI", "R"))
        resolving_power = float(res) if res is not None else None

    medium = "unknown"
    ucd = str(table_header.get("TUCD1", "") if table_header is not None else "").lower()
    ctype = str(header.get("CTYPE1", "")).upper()
    if "obs.atmos" in ucd or ctype.startswith("AWAV"):
        medium = "air"
    elif ucd.startswith("em.wl") or ctype.startswith("WAVE"):
        medium = "vacuum"

    return RawSpectrum(
        wave=wave,
        flux=flux,
        err=err,
        bjd=bjd_value,
        v_bary=v_bary_value,
        frame=frame_value,
        instrument=str(instrument or header.get("INSTRUME", "default")).strip(),
        resolving_power=resolving_power,
        wave_medium=medium,
        continuum_normalized=bool(header.get("CONTNORM", False)),
        time_source=time_source,
        path=str(path),
        header=header,
    )


def to_epoch(
    raw: RawSpectrum,
    *,
    region: Sequence[float] | None = None,
    region_pad_angstrom: float | None = None,
    normalize_continuum: bool | None = None,
    smooth_angstrom: float | None = None,
    ivar_scaling: str = "poisson",
    mask: Iterable[Sequence[float]] | None = None,
    tellurics: bool = False,
    spike_threshold: float | None = 6.0,
    continuum_kwargs: Mapping[str, Any] | None = None,
) -> EpochData:
    """Turn a :class:`RawSpectrum` into a validated :class:`~albireo.data.EpochData`.

    Applies, in this order: region selection (widened by ``region_pad_angstrom`` so the
    continuum fit is not extrapolating at the edges), continuum normalization, inverse
    variances, the final trim to ``region``, then the masks. The order matters — fitting
    a continuum to a region and then trimming its poorly-constrained edges gives a better
    normalization than fitting the trimmed region directly.

    Parameters
    ----------
    raw : RawSpectrum
        From :func:`read_spectrum`.
    region : (float, float), optional
        Wavelength range to keep, Angstrom. Strongly recommended: a full echelle spectrum
        is far more pixels than a disentangling run needs, and the cost of the solve grows
        with every one of them. ``None`` keeps everything.
    region_pad_angstrom : float, optional
        Extra width, on each side, of the range used for the continuum fit before the
        final trim to ``region``. Default: the smoothing scale actually used.
    normalize_continuum : bool, optional
        Whether to fit and divide out a continuum. Default: ``True`` unless the header
        already claimed the spectrum is normalized (``raw.continuum_normalized``).
    smooth_angstrom : float, optional
        Continuum smoothing scale; see :func:`albireo.preprocess.fit_continuum`.
    ivar_scaling : {"poisson", "interpolate", "constant"}, optional
        Noise model used by :func:`albireo.preprocess.estimate_ivar` when the file has no
        error array. Ignored when it does. Default ``"poisson"``.
    mask : iterable of (float, float), optional
        Extra wavelength ranges to zero-weight (interstellar lines, a known bad column,
        a spectral region you distrust).
    tellurics : bool, optional
        Mask the standard telluric bands (:func:`albireo.preprocess.mask_tellurics`).
        Default ``False`` — modelling them with a telluric component usually beats
        throwing them away, and below ~5800 A there is nothing to mask.
    spike_threshold : float or None, optional
        Cosmic-ray rejection threshold in robust sigma, or ``None`` to skip it.
        Default 6.0.
    continuum_kwargs : mapping, optional
        Extra keyword arguments for :func:`albireo.preprocess.fit_continuum`.

    Returns
    -------
    EpochData
        Validated and ready for :class:`albireo.data.Dataset`.

    Raises
    ------
    ValueError
        If the requested region is empty, or masking leaves no usable pixel.
    """
    wave, flux, err = raw.wave, raw.flux, raw.err
    if normalize_continuum is None:
        normalize_continuum = not raw.continuum_normalized

    if region is not None:
        lo, hi = (float(v) for v in region)
        if not hi > lo:
            raise ValueError(f"region must be (wave_min, wave_max) with min < max; got {region!r}")
        pad = region_pad_angstrom
        if pad is None:
            pad = smooth_angstrom if smooth_angstrom is not None else (hi - lo) / 8.0
        sel = (wave >= lo - float(pad)) & (wave <= hi + float(pad))
        if sel.sum() < 5:
            raise ValueError(
                f"{raw.path}: the region [{lo}, {hi}] (padded by {float(pad):g} A) contains "
                f"{int(sel.sum())} pixels; the spectrum spans {wave[0]:.2f} to {wave[-1]:.2f} A"
            )
        wave, flux = wave[sel], flux[sel]
        err = None if err is None else err[sel]

    if normalize_continuum:
        flux_norm, ivar, continuum = normalize(
            wave,
            flux,
            err=err,
            smooth_angstrom=smooth_angstrom,
            **dict(continuum_kwargs or {}),
        )
    else:
        flux_norm, continuum = flux, np.ones_like(flux)
        ivar = None
        if err is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                ivar = np.where(np.isfinite(err) & (err > 0), 1.0 / np.square(err), 0.0)

    if ivar is None:
        ivar = estimate_ivar(
            wave,
            flux_norm,
            continuum=continuum,
            scaling=ivar_scaling,
            mask=~np.isfinite(flux_norm),
        )
    # EpochData tolerates garbage at zero-weight pixels but not the reverse.
    ivar = np.where(np.isfinite(flux_norm), ivar, 0.0)

    epoch = EpochData(
        wave=wave,
        flux=flux_norm,
        ivar=ivar,
        bjd=raw.bjd,
        v_bary=raw.v_bary,
        instrument=raw.instrument,
    )
    if region is not None:
        epoch = select_region(epoch, float(region[0]), float(region[1]))
    if spike_threshold is not None:
        epoch = mask_spikes(epoch, threshold=float(spike_threshold))
    if tellurics:
        epoch = mask_tellurics(epoch)
    if mask is not None:
        epoch = mask_ranges(epoch, mask)
    return epoch


def read_dataset(
    paths: str | Iterable[str],
    *,
    instrument: str | None = None,
    frame: str | None = None,
    sort_by_time: bool = True,
    read_kwargs: Mapping[str, Any] | None = None,
    **epoch_kwargs,
) -> Dataset:
    """Read a set of FITS spectra into one :class:`~albireo.data.Dataset`.

    The one-call path from a directory of archival spectra to something
    :func:`albireo.forward.build_problem` accepts.

    Parameters
    ----------
    paths : str or iterable of str
        A glob pattern (``"data/hr6819/*.fits"``), a directory, or an explicit iterable
        of paths.
    instrument : str, optional
        Instrument key for every epoch; overrides the header. Give the epochs one key if
        they share an LSF — they may still sit on different wavelength grids, which
        albireo handles by splitting them into separate operator groups internally
        (:func:`albireo.forward._epoch_groups`).
    frame : {"topocentric", "barycentric"}, optional
        Override the frame for every epoch. All epochs must agree, since the frame is a
        property of the :class:`~albireo.data.Dataset`.
    sort_by_time : bool, optional
        Order the epochs by ``bjd``. Default ``True``.
    read_kwargs : mapping, optional
        Extra keyword arguments for :func:`read_spectrum`.
    **epoch_kwargs
        Passed to :func:`to_epoch` — ``region``, ``smooth_angstrom``, ``tellurics``, ...

    Returns
    -------
    Dataset

    Raises
    ------
    ValueError
        If no file matches, or the epochs disagree about the frame.

    Examples
    --------
    >>> ds = read_dataset(  # doctest: +SKIP
    ...     "data/hr6819/*.fits",
    ...     instrument="FEROS",
    ...     region=(4000.0, 4600.0),
    ...     smooth_angstrom=150.0,
    ... )
    >>> print(ds.summary())  # doctest: +SKIP
    """
    if isinstance(paths, str):
        expanded = sorted(
            glob.glob(os.path.join(paths, "*.fits") if os.path.isdir(paths) else paths)
        )
    else:
        expanded = [str(p) for p in paths]
    if not expanded:
        raise ValueError(f"no files matched {paths!r}")

    raws = [
        read_spectrum(p, instrument=instrument, frame=frame, **dict(read_kwargs or {}))
        for p in expanded
    ]
    frames = {r.frame for r in raws}
    if len(frames) > 1:
        raise ValueError(
            f"the epochs declare different wavelength frames ({sorted(frames)}); a Dataset "
            "has a single frame. Pass frame= to force one, after checking which is right."
        )
    if sort_by_time:
        raws.sort(key=lambda r: r.bjd)
    epochs = tuple(to_epoch(r, **epoch_kwargs) for r in raws)
    return Dataset(epochs, frame=raws[0].frame)

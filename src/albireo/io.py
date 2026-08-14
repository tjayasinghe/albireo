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
from pathlib import Path
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
    "write_spectra",
]

_C_KMS = 299_792.458

# Keywords under which a pipeline records the barycentric velocity correction it applied,
# in the order they are trusted. All use the "add this to a measured RV" sign convention.
_BARYCORR_KEYS: tuple[str, ...] = (
    "ESO DRS BARYCORR",  # FEROS (FERN/MIDAS DRS), and the ESO DRS family generally
    "ESO DRS BERV",  # HARPS, HARPS-N
    "ESO QC BERV",  # ESPRESSO
    "ESO QC VRAD BARYCOR",  # UVES, X-shooter
    "HIERARCH ESO DRS BARYCORR",
    "HIERARCH ESO DRS BERV",
    "BERV",
    "BVCORR",
    "BARYCORR",  # GIRAFFE, including BLOeM
    "VHELIO",  # heliocentric below here; ~10 m/s from barycentric, warned about
    "ESO QC VRAD HELICOR",  # UVES, X-shooter
    "HELICORR",  # GIRAFFE
    "HELIOCOR",
)

_WAVE_COLUMNS = ("WAVE", "WAVELENGTH", "WAVE_REDUCED", "LAMBDA", "SPECTRAL_AXIS")
_FLUX_COLUMNS = ("FLUX", "FLUX_REDUCED", "SPEC", "INTENSITY")
_ERR_COLUMNS = (
    "ERR",
    "ERROR",
    "SIGMA",
    "FLUX_ERR",
    "ERR_FLUX",
    "ERR_REDUCED",
    "STAT_ERR",
    "UNCERTAINTY",
    "NOISE",
)
# Deliberately not MASK / FLAG / FLAGS. Those names carry no agreed polarity — albireo's own
# EpochData.mask uses True = GOOD, the opposite of the SDP quality convention — and a mask
# read upside down keeps exactly the pixels the file rejected. Losing a mask is recoverable;
# inverting one is not, so a column whose meaning is only guessable from its name is ignored.
_QUALITY_COLUMNS = ("QUAL", "QUAL_REDUCED", "QUALITY")

# The IVOA Spectrum data model records each column's *role* in ``TUTYPn``, and that is the
# only thing in an ESO Phase 3 file that identifies a column unambiguously. Names do not:
# across seven instruments the flux column is variously FLUX, FLUX_REDUCED or both at once.
# Neither do UCDs, and that one is a trap rather than a limitation — a UVES sky-background
# column carries `phot.flux.density;em.wl;stat.uncalib`, byte-identical to the UCD on the
# HARPS *flux* column, so a UCD-keyed reader hands the solver the sky. The utype separates
# them (`BackgroundModel.Value` against `FluxAxis.Value`) and nothing else does.
#
# Three things stop the utype from being a plain string comparison, all seen in real files:
# the namespace prefix is `spec:` in SDP v2, `Spectrum.` in v1 and `eso:` on ESO's own
# reduced columns; ESPRESSO and GIRAFFE misspell `Accuracy` as `Accurancy`; and ESPRESSO
# leaves one utype empty. So match the suffix after `Data.`, case-insensitively, and keep
# the name tables above as a last-resort fallback rather than deleting them.
_ROLE_WAVE = "spectralaxis.value"
_ROLE_FLUX = "fluxaxis.value"
_ROLE_ERR = "fluxaxis.accuracy.staterror"
_ROLE_QUALITY = "fluxaxis.accuracy.qualitystatus"
_ROLE_BACKGROUND = "backgroundmodel.value"

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
            "albireo.io needs astropy to read and write FITS files. Install it with "
            "'pip install \"albireo[io]\"' (or 'pip install astropy'). The rest of "
            "albireo has no astropy dependency — if you already have wavelengths, "
            "fluxes and inverse variances in memory, build EpochData directly, and "
            "albireo.results.write_ascii exports spectra with no extras at all."
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
    quality: np.ndarray | None = field(default=None, repr=False)
    specsys: str = ""
    v_bary_source: str = "unknown"
    err_source: str = "unknown"
    columns: Mapping[str, str] = field(default_factory=dict, repr=False)

    @property
    def n_pixels(self) -> int:
        """Number of samples in the spectrum."""
        return int(self.wave.size)

    @property
    def bad_pixels(self) -> np.ndarray:
        """Boolean mask of pixels that must not be fitted, ``True`` where bad.

        The union of every way a Phase 3 product marks a pixel as carrying no measurement:
        a nonzero quality flag, a non-finite flux, and — where the file has a real error
        array — a non-finite, zero or negative uncertainty. A zero error is not a
        measurement of infinite precision; it is how these pipelines write "nothing here".

        One case is deliberately *not* covered, because no generic rule can be: UVES pads
        the ends of its merged spectra with ``flux = 0.0, err = 1.0`` exactly, which no
        column distinguishes from a genuine measurement of zero flux at unit uncertainty.
        Those files carry no quality column either. Trim the ends, or mask them by hand.
        """
        bad = ~np.isfinite(self.flux)
        if self.quality is not None:
            bad |= np.nan_to_num(self.quality, nan=1.0) != 0.0
        if self.err is not None:
            bad |= ~np.isfinite(self.err) | (self.err <= 0.0)
        return bad

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


@dataclass(frozen=True)
class _Column:
    """One table column, described as the IVOA Spectrum data model describes it."""

    index: int  # 1-based, i.e. the n in TTYPEn
    name: str
    namespace: str
    role: str  # lowercased utype suffix after "Data.", or "" when the file omits it
    ucd: str
    unit: str | None


@dataclass(frozen=True)
class _TableRead:
    """What a container yielded, before any science decision is made about it."""

    wave: np.ndarray
    flux: np.ndarray
    err: np.ndarray | None
    quality: np.ndarray | None
    header: Mapping
    medium: str
    columns: dict[str, str]
    err_note: str
    err_rejected: bool = False


def _utype_parts(utype: object) -> tuple[str, str]:
    """``(namespace, role)`` from a ``TUTYPn`` value; ``("", "")`` when it says nothing.

    The role is everything after the first ``Data.``, lowercased, with ESO's ``Accurancy``
    misspelling folded onto ``Accuracy`` so that ESPRESSO and GIRAFFE agree with X-shooter.
    """
    text = str(utype or "").strip()
    if not text:
        return "", ""
    lowered = text.lower()
    marker = lowered.find("data.")
    if marker == -1:
        return text, ""
    role = lowered[marker + len("data.") :].replace("accurancy", "accuracy")
    return text[:marker], role


def _table_columns(hdu) -> list[_Column]:
    """Describe every column of a binary table by index, utype, UCD and unit."""
    columns = []
    for position, name in enumerate(hdu.columns.names, start=1):
        namespace, role = _utype_parts(hdu.header.get(f"TUTYP{position}"))
        columns.append(
            _Column(
                index=position,
                name=name,
                namespace=namespace,
                role=role,
                ucd=str(hdu.header.get(f"TUCD{position}", "") or "").strip().lower(),
                unit=hdu.columns[name].unit,
            )
        )
    return columns


def _pick_wave(columns: Sequence[_Column]) -> _Column | None:
    """The spectral axis: by utype, else by UCD plus a known name, else by name alone.

    ``SpectralAxis`` does not mean *wavelength* — the same utype carries a frequency or an
    energy axis, distinguished only by the UCD (``em.freq``, ``em.energy``). Reading one of
    those as Angstrom would produce a strictly increasing, entirely wrong grid, so an axis
    that declares itself as something other than ``em.wl`` is refused rather than converted.
    """
    for column in columns:
        if column.role == _ROLE_WAVE:
            if column.ucd and not column.ucd.startswith("em.wl"):
                raise ValueError(
                    f"the spectral axis column {column.name!r} is not a wavelength: its UCD "
                    f"is {column.ucd!r}. albireo works in wavelength; convert the axis and "
                    "build an EpochData directly."
                )
            return column
    for column in columns:
        if column.ucd.startswith("em.wl") and column.name.upper() in _WAVE_COLUMNS:
            return column
    name = _find_column([c.name for c in columns], _WAVE_COLUMNS)
    return next((c for c in columns if c.name == name), None)


def _pick_flux(columns: Sequence[_Column]) -> _Column | None:
    """The flux axis, preferring the calibrated column when a file carries two.

    Candidates are the columns whose utype role is ``FluxAxis.Value``, which is what keeps
    a column typed ``BackgroundModel.Value`` out; the UCD then breaks ties among real flux
    columns. Files carrying both a calibrated and a raw flux (X-shooter, some UVES) label
    the calibrated one ``meta.main`` and the raw one ``stat.uncalib``, and that pair of keys
    is what decides between them.

    The namespace is deliberately *not* a key. It looks like one — ESO's raw columns often
    sit in ``eso:`` beside a ``spec:`` calibrated column — but the association does not hold:
    XShootU products put the science flux in ``eso:Data.FluxAxis.Value`` and a *derived*
    telluric-corrected column in ``spec:``, so ranking by namespace would prefer the derived
    one. The two UCD keys already separate every real case.
    """
    candidates = [c for c in columns if c.role == _ROLE_FLUX]
    if not candidates:
        # No utype at all — a non-ESO file. Fall back to names, but only for columns the
        # file did not label as something else: a declared background stays excluded.
        candidates = [c for c in columns if c.role == "" and c.name.upper() in _FLUX_COLUMNS]
    if not candidates:
        return None

    def rank(column: _Column) -> tuple:
        try:
            by_name = _FLUX_COLUMNS.index(column.name.upper())
        except ValueError:
            by_name = len(_FLUX_COLUMNS)
        return (
            0 if "meta.main" in column.ucd else 1,
            1 if "stat.uncalib" in column.ucd else 0,
            by_name,
            column.index,
        )

    return min(candidates, key=rank)


def _pick_err(columns: Sequence[_Column], flux: _Column | None) -> _Column | None:
    """The statistical error on the chosen flux column.

    Matched to the flux by namespace, then by unit. Pairing X-shooter's calibrated ``FLUX``
    (erg/cm2/s/A) with its ``ERR_REDUCED`` (adu) would be a silent unit catastrophe: the
    numbers are both finite and positive, and the resulting weights would be wrong by the
    flux calibration. This ranks; the caller rejects a candidate that matches on neither
    key, because that one is the error on the file's *other* flux column.
    """
    candidates = [c for c in columns if c.role == _ROLE_ERR]
    if not candidates:
        name = _find_column([c.name for c in columns], _ERR_COLUMNS)
        candidates = [c for c in columns if c.name == name and c.role == ""]
    if not candidates or flux is None:
        return candidates[0] if candidates else None

    def rank(column: _Column) -> tuple:
        return (
            0 if column.namespace.lower() == flux.namespace.lower() else 1,
            0 if (column.unit or "") == (flux.unit or "") else 1,
            column.index,
        )

    return min(candidates, key=rank)


def _pick_quality(columns: Sequence[_Column]) -> _Column | None:
    """The per-pixel quality flag, if the product carries one. Nonzero means bad."""
    for column in columns:
        if column.role == _ROLE_QUALITY:
            return column
    for column in columns:
        if "meta.code.qual" in column.ucd:
            return column
    for column in columns:
        if column.role == "" and column.name.upper() in _QUALITY_COLUMNS:
            return column
    return None


def _medium_from_ucd(ucd: str) -> str:
    """``"air"``, ``"vacuum"`` or ``"unknown"`` from a spectral-axis UCD.

    The ESO Science Data Product standard puts this and only this in the UCD: an
    ``obs.atmos`` qualifier on ``em.wl`` means the wavelengths are air, its absence means
    vacuum. No file carries an ``AIR``/``VACUUM`` keyword, and the human-readable column
    comments contradict each other across collections, so the UCD is the whole of it.
    """
    if "obs.atmos" in ucd:
        return "air"
    if ucd.startswith("em.wl"):
        return "vacuum"
    return "unknown"


def _spectrum_hdu(hdulist, path: str):
    """The HDU holding the spectrum, chosen by utype, then EXTNAME, then column names.

    ESO's own products disagree about where to put it: most write ``EXTNAME='SPECTRUM'``,
    while the Gaia-ESO community release writes ``phase3spectrum``. Choosing the first
    table that merely *has* columns called WAVE and FLUX is what lets a small calibration
    or response table earlier in the file win over the real spectrum.
    """
    from astropy.io import fits

    tiers: dict[int, list] = {0: [], 1: [], 2: []}
    for hdu in hdulist:
        if not isinstance(hdu, fits.BinTableHDU) or hdu.data is None or len(hdu.data) < 1:
            continue
        columns = _table_columns(hdu)
        if any(c.role == _ROLE_WAVE for c in columns):
            tiers[0].append(hdu)
        elif str(hdu.header.get("EXTNAME", "")).strip().lower() in ("spectrum", "phase3spectrum"):
            tiers[1].append(hdu)
        elif _find_column([c.name for c in columns], _WAVE_COLUMNS) and _find_column(
            [c.name for c in columns], _FLUX_COLUMNS
        ):
            tiers[2].append(hdu)
    for tier in (0, 1, 2):
        found = tiers[tier]
        if not found:
            continue
        if len(found) > 1:
            warnings.warn(
                f"{path}: {len(found)} binary tables look like the spectrum "
                f"({', '.join(str(h.header.get('EXTNAME', '?')) for h in found)}); reading "
                "the first. If that is the wrong one, read the arrays yourself and build an "
                "EpochData directly.",
                RuntimeWarning,
                stacklevel=4,
            )
        return found[0]
    return None


def _read_bintable(hdulist, path: str, wave_scale: float | None) -> _TableRead | None:
    """Read the spectrum table, dispatching on the IVOA utypes rather than column names."""
    hdu = _spectrum_hdu(hdulist, path)
    if hdu is None:
        return None
    columns = _table_columns(hdu)
    wave_col = _pick_wave(columns)
    flux_col = _pick_flux(columns)
    if wave_col is None or flux_col is None:
        # This HDU declared itself the spectrum, so falling back to the image reader would
        # mean silently returning something else in the same file — a variance plane, a
        # thumbnail — as the science spectrum. Say what is missing instead.
        missing = "wavelength" if wave_col is None else "flux"
        raise ValueError(
            f"{path}: the spectrum table {str(hdu.header.get('EXTNAME', '?'))!r} has no "
            f"{missing} column that albireo can identify. Its columns are "
            f"{[c.name for c in columns]} with utypes {[c.role or '-' for c in columns]}. "
            "Read the arrays yourself and build an EpochData directly."
        )

    # Two layouts in the wild: the IVOA one, a single row whose cells are whole arrays, and
    # the ordinary one, N rows of scalars (which is what albireo's own write_spectra emits).
    vector = np.asarray(hdu.data[wave_col.name]).ndim > 1
    n_rows = len(hdu.data)
    if vector and n_rows > 1:
        warnings.warn(
            f"{path}: the spectrum table has {n_rows} rows of array cells; reading the "
            "first. Multi-row products are usually per-order or per-fibre extractions, and "
            "which one you want is not something the file says.",
            RuntimeWarning,
            stacklevel=4,
        )

    def read(column: _Column | None) -> np.ndarray | None:
        if column is None:
            return None
        values = np.asarray(hdu.data[column.name])
        cell = values[0] if vector else values
        return np.asarray(cell, dtype=np.float64).ravel()

    wave = read(wave_col)
    flux = read(flux_col)
    if wave is None or flux is None or wave.size != flux.size or wave.size < 2:
        return None

    err_col = _pick_err(columns, flux_col)
    err_note, err_rejected = "no error column in the file", False
    # An error column that shares neither the namespace nor the unit of the chosen flux is
    # the error on the file's *other* flux column. Its values are finite and positive, so
    # nothing downstream would object to weights wrong by the flux calibration.
    if (
        err_col is not None
        and err_col.namespace.lower() != flux_col.namespace.lower()
        and (err_col.unit or "").strip().lower() != (flux_col.unit or "").strip().lower()
        and (err_col.unit or "")
        and (flux_col.unit or "")
    ):
        err_note = (
            f"the only error column, {err_col.name} ({err_col.unit}), belongs to a different "
            f"flux column than {flux_col.name} ({flux_col.unit})"
        )
        err_col, err_rejected = None, True
    err = read(err_col)
    if err is not None and err_col is not None:
        if err.size != wave.size:
            err_note = f"the {err_col.name} column has {err.size} values for {wave.size} pixels"
            err, err_rejected = None, True
        elif not np.any(np.isfinite(err) & (err > 0)):
            # FEROS and HARPS ship an all-NaN ERR: those pipelines produce no error
            # spectrum, and the header says so only in a comment nobody reads.
            err_note = f"the {err_col.name} column holds no finite positive value"
            err, err_rejected = None, True
        else:
            err_note = f"the {err_col.name} column"

    quality_col = _pick_quality(columns)
    quality = read(quality_col)
    if quality is not None and quality.size != wave.size:
        quality = None
    elif quality is not None and not np.any(quality == 0.0):
        # "0 = good" is the SDP convention, so a flag column in which zero never occurs is
        # not using it — UVES_SQUAD's STATUS runs {-5, 1}, and taken at face value it marks
        # every pixel of all 467 products bad. A column whose polarity cannot be read is
        # dropped rather than inverted; the -5 pixels there also carry err < 0 and are
        # caught by that rule anyway.
        warnings.warn(
            f"{path}: the quality column {quality_col.name!r} never takes the value 0, so it "  # type: ignore[union-attr]
            f"is not using the standard 'zero means good' convention (it holds "
            f"{np.unique(quality)[:6]}). Ignoring it — inverting a mask would keep exactly "
            "the pixels the file rejected. Mask them yourself with albireo.mask_ranges if "
            "they matter.",
            RuntimeWarning,
            stacklevel=4,
        )
        quality, quality_col = None, None

    scale = wave_scale if wave_scale is not None else _wave_scale(wave_col.unit, path)
    chosen = {"wave": wave_col.name, "flux": flux_col.name}
    if err is not None and err_col is not None:
        chosen["err"] = err_col.name
    if quality is not None and quality_col is not None:
        chosen["quality"] = quality_col.name
    return _TableRead(
        wave=wave * scale,
        flux=flux,
        err=err,
        quality=quality,
        header=hdu.header,
        medium=_medium_from_ucd(wave_col.ucd),
        columns=chosen,
        err_note=err_note,
        err_rejected=err_rejected,
    )


def _read_wcs_image(hdulist, path: str, wave_scale: float | None) -> _TableRead | None:
    """A 1-D image HDU with a linear or log dispersion in ``CRVAL1``/``CDELT1``."""
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
        # The image convention for the same fact the SDP tables put in a UCD.
        medium = "air" if ctype.startswith("AWAV") else "vacuum" if ctype.startswith("WAVE") else ""
        return _TableRead(
            wave=wave * scale,
            flux=flux,
            err=None,
            quality=None,
            header=header,
            medium=medium or "unknown",
            columns={},
            err_note="a WCS image spectrum carries no error array",
        )
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


def _sexagesimal(value: object) -> float | None:
    """Decode ESO's packed ``[+-]DDMMSS.sss`` telescope coordinates to degrees.

    ``ESO TEL TARG ALPHA = 182703.542`` is 18h 27m 03.542s, not 182703 degrees. Read as
    degrees it wraps modulo 360 to a plausible-looking wrong position, which moves the
    barycentric light-travel correction by minutes and the barycentric velocity by km/s —
    both silently, since ``SkyCoord`` accepts any real number as a right ascension.
    """
    try:
        packed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    sign = -1.0 if packed < 0 else 1.0
    packed = abs(packed)
    degrees, rest = divmod(packed, 10000.0)
    minutes, seconds = divmod(rest, 100.0)
    return sign * (degrees + minutes / 60.0 + seconds / 3600.0)


def _sky_coord(header, path: str):
    """An astropy ``SkyCoord`` from the header, or ``None``."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    ra = _header_get(header, ("RA", "CRVAL1_OBJ", "OBJRA"))
    dec = _header_get(header, ("DEC", "OBJDEC"))
    if ra is None or dec is None:
        # The telescope keywords are the fallback, and they are in a different unit: RA is
        # packed sexagesimal *hours*, declination packed sexagesimal degrees.
        packed_ra = _sexagesimal(_header_get(header, ("ESO TEL TARG ALPHA",)))
        packed_dec = _sexagesimal(_header_get(header, ("ESO TEL TARG DELTA",)))
        if packed_ra is None or packed_dec is None:
            return None
        ra, dec = packed_ra * 15.0, packed_dec
    try:
        return SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg, frame="icrs")
    except (TypeError, ValueError):
        return None


def _mid_exposure_mjd_utc(header, table_header, path: str) -> tuple[float, str]:
    """``(mjd_utc_at_mid_exposure, provenance)`` from the best keywords available.

    ``TMID`` is authoritative and lives in the *extension* header of every ESO Phase 3
    product. The fallbacks matter because the obvious one is wrong for coadded products:
    ``MJD-OBS + EXPTIME/2`` is the middle of the exposure only when the product is a single
    exposure. Where ``TELAPSE`` says the product spans much longer than ``EXPTIME`` — a
    stack of nine exposures over thirty days, in one real Gaia-ESO file — that formula puts
    the epoch a week early. ``MJD-END`` is used first where it exists, and the ``EXPTIME``
    fallback is refused rather than guessed when ``TELAPSE`` disagrees with it.
    """
    if _header_get(header, ("M_EPOCH",)) in (True, "T", "True"):
        warnings.warn(
            f"{path}: M_EPOCH is true, so this product coadds exposures taken at different "
            "times. It has one nominal timestamp but no single epoch, and a radial velocity "
            "measured from it is an average over the span. Use the per-exposure products.",
            RuntimeWarning,
            stacklevel=4,
        )
    for hdr in (table_header, header):
        if hdr is not None and "TMID" in hdr:
            return float(hdr["TMID"]), "TMID"
    mjd_obs = _header_get(header, ("MJD-OBS", "MJD_OBS"))
    mjd_end = _header_get(header, ("MJD-END",))
    exptime = _header_get(header, ("TEXPTIME", "EXPTIME", "EXPOSURE"), 0.0)
    if mjd_obs is not None and mjd_end is not None:
        return 0.5 * (float(mjd_obs) + float(mjd_end)), "(MJD-OBS + MJD-END)/2"
    if mjd_obs is not None:
        elapsed = _header_get(header, ("TELAPSE",))
        if elapsed is not None and float(exptime) > 0 and float(elapsed) > 1.5 * float(exptime):
            raise ValueError(
                f"{path}: the only usable time keyword is MJD-OBS, but TELAPSE "
                f"({float(elapsed):.0f} s) is much longer than EXPTIME ({float(exptime):.0f} s), "
                "so this product spans more than one exposure and 'MJD-OBS + EXPTIME/2' is not "
                "its mid-point. Pass bjd= to read_spectrum with the epoch you mean."
            )
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
            # HELICORR and 'ESO QC VRAD HELICOR' are heliocentric too, and neither spells
            # out HELIO — a substring test on that alone silently trusts them as barycentric.
            if any(mark in stripped.upper() for mark in ("HELIO", "HELICOR", "VHELIO")):
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
        warnings.warn(
            f"{path}: no barycentric-velocity keyword, and too little header information to "
            "compute one (missing the observatory location or the target coordinates), so "
            "v_bary is 0. That is a fabricated value, not a measured one: it is right only "
            "if the pipeline applied no correction. Pass v_bary= to read_spectrum. The "
            "telluric component is placed by this number, and up to 30 km/s of it.",
            RuntimeWarning,
            stacklevel=4,
        )
        return 0.0, "assumed 0 (no keyword, and too little header information to compute)"
    from astropy.time import Time

    # The location goes on the Time, not on the call: astropy raises if it is given both.
    time = Time(mjd_utc, format="mjd", scale="utc", location=location)
    v = coord.radial_velocity_correction("barycentric", obstime=time)
    return float(v.to("km/s").value), "computed with astropy"


def _frame_from_specsys(header, table_header, path: str, declared: bool = False) -> tuple[str, str]:
    """``(albireo_frame, raw_SPECSYS)``, warning when the header does not say.

    albireo models two frames, and a heliocentric spectrum is reported as barycentric
    because the difference is a few m/s — a hundredth of a pixel at any resolving power
    this package is used at. The raw value is returned rather than discarded so that
    :attr:`RawSpectrum.specsys` still says what the file actually claimed.

    ``declared`` says the caller passed ``frame=``. Every warning here ends by advising
    exactly that, so with it already supplied they are noise on a question already answered.
    """
    for source in (table_header, header):
        if source is None:
            continue
        specsys = str(_header_get(source, ("SPECSYS", "SPECSYSA"), "") or "").strip().upper()
        if not specsys:
            continue
        if specsys.startswith(("BARYCENT", "BARY")):
            return "barycentric", specsys
        if specsys.startswith(("HELIOCEN", "HELIO")):
            if not declared:
                warnings.warn(
                    f"{path}: SPECSYS={specsys!r}; treating heliocentric as barycentric. The "
                    "two frames differ by up to ~10 m/s.",
                    RuntimeWarning,
                    stacklevel=4,
                )
            return "barycentric", specsys
        if specsys.startswith(("TOPOCENT", "TOPO")):
            return "topocentric", specsys
        # A frame albireo does not model — LSRK, GEOCENT, CMBDIPOL. Assuming topocentric is
        # the least-wrong default, but the value must not be thrown away or reported as a
        # missing keyword: the file was perfectly clear, it just said something else.
        if not declared:
            warnings.warn(
                f"{path}: SPECSYS={specsys!r} is a frame albireo does not model (it knows "
                "barycentric and topocentric). Assuming 'topocentric'; pass frame= if that "
                "is wrong. The declared value is kept on RawSpectrum.specsys.",
                RuntimeWarning,
                stacklevel=4,
            )
        return "topocentric", specsys
    if not declared:
        warnings.warn(
            f"{path}: no SPECSYS keyword, so the wavelength frame is undeclared. Assuming "
            "'topocentric' (uncorrected, as observed). If the pipeline already applied the "
            "barycentric correction — most modern echelle pipelines do — pass "
            "frame='barycentric', or every velocity will be offset by the correction.",
            RuntimeWarning,
            stacklevel=4,
        )
    return "topocentric", ""


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
        Override ``R``; otherwise read from ``SPEC_RES``/``SPECRES``/``RESOLUTI``. A bare
        ``R`` keyword is deliberately not consulted: a one-character card collides with too
        much, and ``R = 3.7`` would imply a 34,000 km/s line-spread function in silence.
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
                f"{_WAVE_COLUMNS[0]}/{_FLUX_COLUMNS[0]} columns (or the IVOA utypes that "
                "name them), or a 1-D image with CRVAL1/CDELT1. Read the arrays yourself "
                "and build an EpochData directly if the file uses another layout."
            )
        header = hdulist[0].header.copy()

    wave, flux, err, quality = parsed.wave, parsed.flux, parsed.err, parsed.quality
    table_header = parsed.header
    order = np.argsort(wave)
    if not np.array_equal(order, np.arange(wave.size)):
        wave, flux = wave[order], flux[order]
        err = None if err is None else err[order]
        quality = None if quality is None else quality[order]
    if np.any(np.diff(wave) <= 0):
        raise ValueError(
            f"{path}: wavelengths are not strictly increasing even after sorting "
            "(duplicated samples?). albireo never resamples, so the grid must be clean."
        )

    # Only look for a time when one is actually needed. The error raised for a file with no
    # time keyword tells the caller to pass bjd=, and that advice has to work — on a file
    # albireo wrote itself, among others.
    mjd_utc, time_key = 0.0, ""
    if bjd is None:
        mjd_utc, time_key = _mid_exposure_mjd_utc(header, table_header, path)
    elif v_bary is None:
        # A time is still wanted, but only as the argument to an astropy-computed v_bary,
        # itself the fallback for a file with no barycentric keyword. Never let that block a
        # caller who supplied the epoch: fall back to the given BJD, which differs from
        # MJD_UTC by at most the 8.3-minute light-travel term and so moves a barycentric
        # velocity by under 0.05 km/s.
        try:
            mjd_utc, time_key = _mid_exposure_mjd_utc(header, table_header, path)
        except ValueError:
            mjd_utc, time_key = float(bjd) - 2400000.5, "approximated from the supplied bjd"
    if bjd is None:
        bjd_value, time_source = _barycentric_time(mjd_utc, header, path)
        time_source = f"{time_source} from {time_key}"
    else:
        bjd_value, time_source = float(bjd), "supplied by the caller"
    if v_bary is None:
        v_bary_value, v_bary_source = _barycentric_velocity(header, mjd_utc, path)
    else:
        v_bary_value, v_bary_source = float(v_bary), "supplied by the caller"
    frame_value, specsys = _frame_from_specsys(
        header, table_header, path, declared=frame is not None
    )
    if frame is not None:
        frame_value = frame

    if resolving_power is None:
        # Deliberately not "R": a one-character keyword collides with anything, and a
        # stray R = 3.7 would imply a 34,000 km/s line-spread function without complaint.
        res = _header_get(header, ("SPEC_RES", "SPECRES", "RESOLUTI"))
        resolving_power = float(res) if res is not None else None

    if parsed.err_rejected:
        warnings.warn(
            f"{path}: {parsed.err_note}, so the file carries no usable uncertainties and "
            "albireo will estimate them from the scatter instead (to_epoch's ivar_scaling). "
            "This is normal for FEROS and HARPS, whose pipelines produce no error spectrum, "
            "but it means the weights are albireo's assumption rather than the archive's.",
            RuntimeWarning,
            stacklevel=3,
        )

    return RawSpectrum(
        wave=wave,
        flux=flux,
        err=err,
        bjd=bjd_value,
        v_bary=v_bary_value,
        frame=frame_value,
        instrument=str(instrument or header.get("INSTRUME", "default")).strip(),
        resolving_power=resolving_power,
        wave_medium=parsed.medium,
        continuum_normalized=bool(header.get("CONTNORM", False)),
        time_source=time_source,
        path=str(path),
        header=header,
        quality=quality,
        specsys=specsys,
        v_bary_source=v_bary_source,
        err_source=parsed.err_note,
        columns=dict(parsed.columns),
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
    bad = raw.bad_pixels
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
        wave, flux, bad = wave[sel], flux[sel], bad[sel]
        err = None if err is None else err[sel]
        # The guard below asks about the pixels that survive the *final* trim, not about the
        # padded slice the continuum is fitted on. A region that is entirely dead beside a
        # live pad would otherwise pass, and produce a zero-weight epoch that the solver
        # accepts in silence: the fit reports N epochs and is informed by N-1.
        core = (wave >= lo) & (wave <= hi)
    else:
        core = np.ones(bad.shape, dtype=bool)

    if bad[core].all():
        raise ValueError(
            f"{raw.path}: every pixel in the selected range is flagged bad by the file "
            "(quality flag, non-finite flux, or non-positive uncertainty). There is nothing "
            "to fit here — check the region, or the product."
        )

    continuum_options = dict(continuum_kwargs or {})
    if normalize_continuum:
        # Bad pixels are kept in place (albireo never resamples or drops samples) but given
        # no say in the continuum: a flagged cosmic pulls the upper envelope up, and a dead
        # column pulls it down, and either one propagates into every line depth.
        continuum_options.setdefault("weights", (~bad).astype(np.float64))
        flux_norm, ivar, continuum = normalize(
            wave,
            flux,
            err=err,
            smooth_angstrom=smooth_angstrom,
            **continuum_options,
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
            mask=bad | ~np.isfinite(flux_norm),
        )
    # EpochData tolerates garbage at zero-weight pixels but not the reverse.
    ivar = np.where(np.isfinite(flux_norm) & ~bad, ivar, 0.0)

    epoch = EpochData(
        wave=wave,
        flux=flux_norm,
        ivar=ivar,
        bjd=raw.bjd,
        v_bary=raw.v_bary,
        instrument=raw.instrument,
        medium=None if raw.wave_medium == "unknown" else raw.wave_medium,
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


def _read_many(paths: Sequence[str], *, instrument, frame, options) -> list[RawSpectrum]:
    """Read every file, collapsing warnings that repeat across files into one each.

    Every warning in this module names its file, which is right for one spectrum and wrong
    for fifty: reading the 51 FEROS epochs of HR 6819 emits the same "no usable error
    array" warning 51 times, differing only in the path. A wall of identical text is how
    users learn to ignore warnings, so the batch entry point reports each *distinct* one
    once, naming the first file it happened to and how many others followed.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        raws = [read_spectrum(p, instrument=instrument, frame=frame, **options) for p in paths]

    collapsed: dict[tuple, list] = {}
    for entry in caught:
        text = str(entry.message)
        # The messages are formatted "<path>: <complaint>"; the complaint is the identity.
        _, _, complaint = text.partition(": ")
        key = (entry.category, complaint or text)
        collapsed.setdefault(key, [0, text])[0] += 1
    for (category, _), (count, first) in collapsed.items():
        message = first
        if count > 1:
            message = f"{first}\n(the same for {count - 1} more of the {len(paths)} files.)"
        warnings.warn(message, category, stacklevel=3)
    return raws


def read_dataset(
    paths: str | Iterable[str],
    *,
    instrument: str | None = None,
    frame: str | None = None,
    medium: str | None = None,
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
    medium : {"air", "vacuum"}, optional
        Declare the wavelength scale for every epoch, overriding what the files say. Use
        this only after converting them onto a common scale — it changes the label, not
        the wavelengths. Without it, files that disagree are refused.
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
        if os.path.isdir(paths):
            # Deduplicated by normalized case because Windows matches *.fits and *.FITS to
            # the same file, which would otherwise read every epoch twice.
            seen = {
                os.path.normcase(path): path
                for pattern in ("*.fits", "*.fit", "*.fits.gz", "*.FITS")
                for path in glob.glob(os.path.join(paths, pattern))
            }
            expanded = sorted(seen.values())
        else:
            expanded = sorted(glob.glob(paths))
    else:
        expanded = [str(p) for p in paths]
    if not expanded:
        raise ValueError(f"no files matched {paths!r}")

    options = dict(read_kwargs or {})
    options.pop("instrument", None)
    options.pop("frame", None)
    raws = _read_many(expanded, instrument=instrument, frame=frame, options=options)
    frames = {r.frame for r in raws}
    if len(frames) > 1:
        raise ValueError(
            f"the epochs declare different wavelength frames ({sorted(frames)}); a Dataset "
            "has a single frame. Pass frame= to force one, after checking which is right."
        )
    media = {r.wave_medium for r in raws}
    if medium is None and len(media) > 1:
        raise ValueError(
            f"the epochs are on different wavelength scales ({sorted(media)}); combining them "
            "unconverted would place the same line at two different model pixels, an offset "
            "of about 83 km/s. Convert with albireo.air_to_vacuum or albireo.vacuum_to_air "
            "and pass medium= to declare the result, or read the two sets separately."
        )
    if sort_by_time:
        raws.sort(key=lambda r: r.bjd)
    epochs = tuple(to_epoch(r, **epoch_kwargs) for r in raws)
    if medium is not None:
        from albireo.preprocess import _replace

        epochs = tuple(_replace(epoch, medium=medium) for epoch in epochs)
    return Dataset(epochs, frame=raws[0].frame)


def write_spectra(
    path,
    grid,
    d_hat,
    std=None,
    *,
    format: str = "fits",
    light_fractions=None,
    prior=None,
    meta: Mapping[str, object] | None = None,
    overwrite: bool = True,
):
    """Write disentangled component spectra to FITS or ECSV.

    The disentangled spectrum and its uncertainty band are what the rest of a project
    consumes — an atmosphere code, a line-profile fit, a co-author. This is how they leave.

    Parameters
    ----------
    path
        Output path. The suffix is set from ``format`` if it is missing.
    grid
        The :class:`~albireo.grids.LogGrid` the spectra were solved on.
    d_hat
        Deviation spectra, shape ``(n_comp, n_pix)`` or ``(n_pix,)``. Written as flux
        ``1 + d``, i.e. the normalized component spectrum.
    std
        Pointwise standard deviations with the same shape, e.g. from
        :func:`albireo.likelihood.spectra_std`. Written as the ``ERR`` column.
    format
        ``"fits"`` — one ``BinTableHDU`` per component, each with ``WAVE`` / ``FLUX`` and,
        if ``std`` was given, ``ERR``. ``"ecsv"`` — an Astropy ECSV table, which carries
        column metadata as readable YAML and is the better choice for handing spectra to
        another Python tool.
    light_fractions
        Recorded in the header as ``LIGHTFR*``. Worth passing: the recovered quantity is
        the light-weighted contribution, so the line depths are only interpretable
        alongside the light fractions assumed or inferred for the fit.
    prior
        A :class:`~albireo.priors.SmoothnessPrior`, recorded as ``TAU*`` / ``ETA*``.
    meta
        Extra header cards (FITS) or table metadata (ECSV). Keys longer than eight
        characters are written as FITS ``HIERARCH`` cards.
    overwrite
        Overwrite an existing file.

    Returns
    -------
    pathlib.Path
        The path written.

    Notes
    -----
    The written flux is the *component* spectrum ``1 + d``, not its contribution to the
    composite, which is ``l_i * d_i``. Where the smoothness prior dominates — between
    lines, and wherever the epochs give little leverage — the values are prior-set rather
    than data-set; the ``ERR`` column is what says so, and it is the reason to write it.
    See ``docs/math.md`` §5.1.
    """
    fmt = format.lower()
    if fmt not in {"fits", "ecsv"}:
        raise ValueError(f"format must be 'fits' or 'ecsv', got {format!r}")

    _require_astropy()

    d_hat = np.atleast_2d(np.asarray(d_hat))
    std_arr = None if std is None else np.atleast_2d(np.asarray(std))
    wave = np.asarray(grid.wave)
    if d_hat.shape[-1] != wave.size:
        raise ValueError(
            f"d_hat has {d_hat.shape[-1]} pixels but the grid has {wave.size}; "
            "they must come from the same fit."
        )
    if std_arr is not None and std_arr.shape != d_hat.shape:
        raise ValueError(f"std has shape {std_arr.shape}, expected {d_hat.shape}")

    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".fits" if fmt == "fits" else ".ecsv")
    path.parent.mkdir(parents=True, exist_ok=True)

    provenance = _spectra_provenance(d_hat.shape[0], light_fractions, prior, meta)
    if fmt == "fits":
        _write_spectra_fits(path, wave, d_hat, std_arr, provenance, overwrite)
    else:
        _write_spectra_ecsv(path, wave, d_hat, std_arr, provenance, overwrite)
    return path


def _spectra_provenance(n_comp, light_fractions, prior, meta) -> dict[str, object]:
    from albireo import __version__

    out: dict[str, object] = {"ALBIREO": __version__, "NCOMP": int(n_comp)}
    if light_fractions is not None:
        for i, value in enumerate(np.atleast_1d(np.asarray(light_fractions, dtype=float)).ravel()):
            out[f"LIGHTFR{i + 1}"] = float(value)
    if prior is not None:
        for name in ("tau", "eta"):
            values = getattr(prior, name, None)
            if values is None:
                continue
            for i, value in enumerate(np.atleast_1d(np.asarray(values, dtype=float)).ravel()):
                out[f"{name.upper()}{i + 1}"] = float(value)
    if meta:
        out.update(meta)
    return out


def _write_spectra_fits(path, wave, d_hat, std_arr, provenance, overwrite) -> None:
    fits = _require_astropy()

    primary = fits.PrimaryHDU()
    for key, value in provenance.items():
        primary.header[key] = value
    primary.header["COMMENT"] = "Disentangled component spectra from albireo."
    primary.header["COMMENT"] = "FLUX is the normalized component spectrum 1 + d."
    primary.header["COMMENT"] = "The observable is l_i * d_i; depths depend on the light"
    primary.header["COMMENT"] = "fractions recorded above. See docs/math.md section 5.2."
    if std_arr is not None:
        primary.header["COMMENT"] = "ERR is the marginal posterior standard deviation;"
        primary.header["COMMENT"] = "it is large where the prior, not the data, sets the"
        primary.header["COMMENT"] = "spectrum. See docs/math.md section 5.1."

    hdus = [primary]
    for i in range(d_hat.shape[0]):
        columns = [
            fits.Column(name="WAVE", format="D", unit="Angstrom", array=wave),
            fits.Column(name="FLUX", format="D", array=1.0 + d_hat[i]),
        ]
        if std_arr is not None:
            columns.append(fits.Column(name="ERR", format="D", array=std_arr[i]))
        hdu = fits.BinTableHDU.from_columns(columns, name=f"COMP{i + 1}")
        hdu.header["COMPONEN"] = (i + 1, "component index, 1-based")
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(path, overwrite=overwrite)


def _write_spectra_ecsv(path, wave, d_hat, std_arr, provenance, overwrite) -> None:
    from astropy.table import Table

    table = Table()
    table["wave"] = wave
    table["wave"].unit = "Angstrom"
    for i in range(d_hat.shape[0]):
        table[f"flux_{i + 1}"] = 1.0 + d_hat[i]
        if std_arr is not None:
            table[f"err_{i + 1}"] = std_arr[i]
    table.meta.update(provenance)
    table.meta["comment"] = (
        "Disentangled component spectra from albireo. flux_i is the normalized "
        "component spectrum 1 + d for component i; the observable is l_i * d_i, so "
        "depths depend on the recorded light fractions (docs/math.md section 5.2)."
    )
    table.write(path, format="ascii.ecsv", overwrite=overwrite)

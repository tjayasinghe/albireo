"""PEPSI spectra as the Spectroscopic Data Systems pipeline delivers them.

**Experimental.** The names and the default resolving power may change.

PEPSI, the Potsdam Echelle Polarimetric and Spectroscopic Instrument on the Large Binocular
Telescope, records one cross-disperser per arm and exposure: CD1-CD3 in the blue arm and
CD4-CD6 in the red. The pipeline's products (``pepsib.<date>.<number>.<side>.<kind>`` for the
blue arm, ``pepsir.`` for the red, ``.bwl``, ``.avr`` and ``.all`` among the kinds) are FITS
binary tables with the columns ``Arg`` (wavelength, Angstrom), ``Fun`` (flux), ``Var``
(variance of the flux) and ``Mask`` (``True`` where a pixel is flagged; no pixel of the
2023-2026 products checked here is). :func:`albireo.read_spectrum` recognises them by
``INSTRUME = 'PEPSI'`` and reads them with :func:`read_pepsi`. The quantities albireo needs
come from the primary header:

- ``JD-TDB``, the barycentric Julian date in TDB at mid-exposure. It agrees with astropy's
  BJD_TDB at the header's mid-exposure time (``JD-OBS``) to 0.01 s.
- ``SSBVEL``, the velocity of the observer relative to the solar-system barycentre, m/s.
  The pipeline has removed it from the wavelengths, which are therefore barycentric. In
  albireo's sign convention the barycentric correction is ``-SSBVEL / 1000`` km/s, which
  agrees with astropy's to 3 m/s. albireo needs it even for barycentric wavelengths, to place
  the telluric lines.
- ``FIBER``, the core diameter of the science fibre in micrometres, which sets the resolving
  power. For the 300 um fibre the default is R = 44,300 (LSF FWHM 6.77 km/s), measured from
  the widths of 14 weak, isolated telluric O2 lines in red-arm spectra of 2023-2026 with
  their own width (1.3 km/s FWHM, an assumption) removed in quadrature. Other fibres have no
  default, and their resolving power must be declared.
- ``ARM`` and ``CROSDIS`` (for example ``'3: 4800 - 5441'``): the arm and the cross-disperser.

The wavelengths are in air: velocities measured against air synthetic templates are
consistent with the catalogue velocities in the headers (``RADVEL``), whereas a vacuum
scale would offset them by 83 km/s. The flux is continuum-normalised by the pipeline.
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["PEPSI_RESOLVING_POWER", "PepsiSpectrum", "is_pepsi", "read_pepsi"]

_C_KMS = 299_792.458

#: Default resolving power per science fibre (the ``FIBER`` keyword, micrometres).
PEPSI_RESOLVING_POWER: Mapping[str, float] = {"300": 44_300.0}


def is_pepsi(header: Mapping) -> bool:
    """Whether a primary header belongs to a PEPSI product."""
    return str(header.get("INSTRUME", "")).strip().upper() == "PEPSI"


@dataclass(frozen=True)
class PepsiSpectrum:
    """One PEPSI product as read from its file.

    Attributes
    ----------
    wave : numpy.ndarray
        Air wavelengths, barycentric, Angstrom, strictly increasing over the finite values.
        A pixel without a finite wavelength is bad.
    flux : numpy.ndarray
        Continuum-normalised flux.
    err : numpy.ndarray
        One-sigma uncertainties, the square root of ``Var``.
    bad : numpy.ndarray
        ``True`` where the pixel is flagged or has no valid flux or variance.
    bjd : float
        ``JD-TDB``, BJD_TDB at mid-exposure.
    v_bary : float
        Barycentric correction, km/s (``-SSBVEL / 1000``).
    arm : str
        ``"blue"`` or ``"red"`` (``"both"`` for a product covering both arms).
    cross_disperser : int or None
        Cross-disperser number from ``CROSDIS``.
    fiber : str
        ``FIBER``, micrometres.
    resolving_power : float or None
        The default for the fibre, or ``None`` when there is none.
    kind : str
        The product kind, the file name's last suffix (``bwl``, ``avr``, ``all``).
    target : str
        ``OBJECT``.
    path : str
        The file read.
    header : Mapping
        The primary header.
    """

    wave: np.ndarray
    flux: np.ndarray
    err: np.ndarray
    bad: np.ndarray
    bjd: float
    v_bary: float
    arm: str
    cross_disperser: int | None
    fiber: str
    resolving_power: float | None
    kind: str
    target: str
    path: str = ""
    header: Mapping = field(default_factory=dict, repr=False)

    @property
    def lsf_sigma_kms(self) -> float | None:
        """Gaussian LSF sigma in km/s implied by :attr:`resolving_power`, or ``None``."""
        if self.resolving_power is None:
            return None
        return _C_KMS / (self.resolving_power * 2.0 * math.sqrt(2.0 * math.log(2.0)))


def _column(table, name: str):
    for candidate in table.columns.names:
        if candidate.strip().lower() == name.lower():
            return np.asarray(table[candidate])
    return None


def read_pepsi(path: str | Path, *, resolving_power: float | None = None) -> PepsiSpectrum:
    """Read one PEPSI product.

    Parameters
    ----------
    path : str or Path
        A ``pepsib.*`` or ``pepsir.*`` file.
    resolving_power : float, optional
        Override the fibre's default resolving power.

    Returns
    -------
    PepsiSpectrum

    Raises
    ------
    ValueError
        If the file is not a PEPSI product, has no ``Arg``/``Fun``/``Var`` table, lacks
        ``JD-TDB`` or ``SSBVEL``, or repeats a wavelength.
    """
    from astropy.io import fits

    path = str(path)
    with fits.open(path, memmap=False) as hdulist:
        header = hdulist[0].header.copy()
        if not is_pepsi(header):
            raise ValueError(f"{path}: not a PEPSI product (INSTRUME = 'PEPSI' is missing)")
        table = None
        for hdu in hdulist[1:]:
            data = getattr(hdu, "data", None)
            if data is not None and hasattr(data, "columns"):
                names = {n.strip().lower() for n in data.columns.names}
                if {"arg", "fun", "var"} <= names:
                    table = data
                    break
        if table is None:
            raise ValueError(f"{path}: no binary table with the columns Arg, Fun and Var")
        wave = np.asarray(_column(table, "Arg"), dtype=np.float64)
        flux = np.asarray(_column(table, "Fun"), dtype=np.float64)
        var = np.asarray(_column(table, "Var"), dtype=np.float64)
        mask = _column(table, "Mask")
    for key in ("JD-TDB", "SSBVEL"):
        if key not in header:
            raise ValueError(f"{path}: {key} is missing from the header")
    order = np.argsort(wave)
    wave, flux, var = wave[order], flux[order], var[order]
    finite = np.isfinite(wave)
    if np.any(np.diff(wave[finite]) <= 0):
        raise ValueError(
            f"{path}: wavelengths are not strictly increasing even after sorting "
            "(duplicated samples?). albireo never resamples, so the grid must be clean."
        )
    bad = ~(np.isfinite(wave) & np.isfinite(flux) & np.isfinite(var) & (var > 0))
    if mask is not None:
        flagged = np.asarray(mask, dtype=bool)[order]
        if flagged.mean() > 0.5:
            # True is read as "flagged" because no pixel of the products checked is flagged
            # while all are valid. A Mask that flags most of a spectrum would mean the opposite
            # polarity, so it is not applied.
            warnings.warn(
                f"{path}: the Mask column flags {flagged.mean():.0%} of the pixels; it is "
                "ignored, since flagging most of a spectrum suggests the opposite polarity",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            bad |= flagged
    with np.errstate(invalid="ignore"):
        err = np.where(bad, np.inf, np.sqrt(np.where(var > 0, var, np.nan)))
    fiber = str(header.get("FIBER", "")).strip()
    if resolving_power is None:
        resolving_power = PEPSI_RESOLVING_POWER.get(fiber)
        if resolving_power is None:
            warnings.warn(
                f"{path}: no default resolving power for the {fiber or 'unknown'} um fibre; "
                "declare the LSF for instrument 'PEPSI' (a resolving power or a sigma in km/s)",
                RuntimeWarning,
                stacklevel=2,
            )
    crosdis = str(header.get("CROSDIS", "")).strip()
    match = re.match(r"\s*(\d+)", crosdis)
    cross_disperser = int(match.group(1)) if match else None
    name = Path(path).name.lower()
    kind = name.rsplit(".", 1)[-1] if "." in name else ""
    arm_header = str(header.get("ARM", "")).strip().lower()
    arm = (
        "blue" if name.startswith("pepsib") else "red" if name.startswith("pepsir") else arm_header
    )
    if kind == "all" and finite.any() and wave[finite][0] < 6000.0 < wave[finite][-1]:
        arm = "both"
    return PepsiSpectrum(
        wave=wave,
        flux=np.where(bad, np.nan, flux),
        err=err,
        bad=bad,
        bjd=float(header["JD-TDB"]),
        v_bary=-float(header["SSBVEL"]) / 1000.0,
        arm=arm,
        cross_disperser=cross_disperser,
        fiber=fiber,
        resolving_power=None if resolving_power is None else float(resolving_power),
        kind=kind,
        target=str(header.get("OBJECT", "")).strip(),
        path=path,
        header=header,
    )

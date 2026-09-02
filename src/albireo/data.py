"""Observed spectra: the :class:`EpochData` and :class:`Dataset` containers and their validation.

This module is pure NumPy. It is the boundary at which user data enter albireo, so it is
importable and inspectable without JAX; everything downstream consumes the arrays validated
here (``internal/design.md`` §3). The conventions it enforces are the following.

Masking is ``ivar == 0``. Chip gaps, cosmic rays, interstellar lines, saturated pixels and
deep tellurics are all zero-weight pixels, so no operator or solve downstream needs a special
case. A masked pixel carries no weight and its ``flux`` value is never read, so non-finite
values are permitted there (a cosmic-ray spike, or a NaN from a reduction pipeline). ``flux``
must be finite wherever ``ivar > 0``.

``mask`` is optional, and ``True`` means good. It allows a caller to keep a boolean quality
flag alongside the inverse variances instead of zeroing them destructively. It is folded into
the weights in one place, :attr:`EpochData.effective_ivar`, which is what downstream code
consumes; nothing else in albireo reads ``mask``. The finite-``flux`` requirement is keyed on
``ivar > 0`` alone, so a pixel that is to hold non-finite values requires ``ivar = 0``, not
merely ``mask = False``.

Data are never resampled (``internal/design.md`` D4, ``docs/math.md`` §1.1). Interpolating
observations onto a common grid correlates the noise and invalidates the diagonal ``ivar``
model. Each epoch keeps its own native, strictly increasing ``wave`` array, and the model is
projected onto it by a static rebin operator, so mixed instruments, resolutions and samplings
are supported directly.

A :class:`Dataset` declares the frame its wavelengths are in: ``"topocentric"`` (as observed,
the default) or ``"barycentric"`` (already corrected). The declaration is not a
transformation, since the barycentric correction ``v_bary`` is composed inside the forward
model instead (``docs/math.md`` §1.2). In the topocentric frame a stellar component is shifted
by ``xi(v_ij) - xi(v_bary_j)`` and the tellurics are static; in the barycentric frame the star
is shifted by ``xi(v_ij)`` and the tellurics carry ``+xi(v_bary_j)``. Both frames are exact,
since log-shifts compose by addition. Declaring the wrong frame offsets every velocity without
raising an error.

Wavelengths are conventionally in Angstrom, on an air or vacuum scale that is declared rather
than inferred, ``v_bary`` is in km/s, and ``bjd`` is BJD_TDB at mid-exposure.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterator

import numpy as np

__all__ = ["Dataset", "EpochData"]

_FRAMES: tuple[str, ...] = ("topocentric", "barycentric")
_MEDIA: tuple[str, ...] = ("air", "vacuum")


def _as_float64_1d(value: object, name: str) -> np.ndarray:
    """Coerce ``value`` to a 1-D float64 array, with an informative error on failure."""
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be convertible to a float64 array ({exc})") from exc
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    return arr


def _as_finite_float(value: object, name: str) -> float:
    """Coerce ``value`` to a finite Python float, with an informative error on failure."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float, got {value!r} ({exc})") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {out!r}")
    return out


@dataclasses.dataclass(frozen=True)
class EpochData:
    """One observed spectrum, on its own native wavelength grid.

    Inputs are coerced (``wave``/``flux``/``ivar`` to 1-D float64 arrays, ``mask`` to bool)
    and validated on construction, so any :class:`EpochData` that exists is usable by every
    operator downstream without re-checking.

    Parameters
    ----------
    wave : array_like
        Native wavelength grid, shape ``(n,)``, conventionally Angstrom. Must be strictly
        increasing, strictly positive and finite. Never resampled by albireo.
    flux : array_like
        Continuum-normalized flux, shape ``(n,)``. Must be finite wherever ``ivar > 0``;
        at zero-weight pixels any value (including ``nan``/``inf``) is accepted and ignored.
    ivar : array_like
        Inverse variance of ``flux``, shape ``(n,)``. Must be finite and non-negative;
        ``0`` marks a masked pixel.
    bjd : float
        Mid-exposure time as BJD_TDB.
    v_bary : float, optional
        Barycentric correction velocity in km/s, defined so that
        ``v_bary_frame = v_topo_frame (+) v_bary``. Default ``0.0``.
    instrument : str, optional
        Key into the per-instrument LSF and response tables. Default ``"default"``.
    mask : array_like of bool or None, optional
        Optional quality flag, shape ``(n,)``, where ``True`` means GOOD. Folded into the
        weights by :attr:`effective_ivar` and nowhere else. Default ``None``.
    medium : {"air", "vacuum"} or None, optional
        Which wavelength scale ``wave`` is on. Default ``None``, meaning undeclared, which
        is accepted but is not equivalent to ``"air"``: nothing may assume a value for it.

        Air and vacuum wavelengths differ by a nearly constant 83 km/s across the optical
        (0.87 Angstrom at 3000 A, 2.74 A at 10000 A), the same order as the semi-amplitudes
        albireo measures, and archives mix the two. ESO Phase 3 spectra declare the scale
        per file in ``TUCD1`` (``em.wl;obs.atmos`` is air, ``em.wl`` is vacuum): FEROS,
        HARPS, UVES and GIRAFFE deliver air, while ESPRESSO and XQ-100 deliver vacuum. A
        :class:`Dataset` combining them without conversion would put the same physical line
        at two different model pixels.

        :class:`Dataset` therefore requires every epoch to agree and raises on a mixture
        (:func:`albireo.air_to_vacuum` and :func:`albireo.vacuum_to_air` convert).
        Undeclared epochs may be combined only with other undeclared epochs, since
        "unknown" cannot be checked against "air".

    Raises
    ------
    ValueError
        If any of the above conditions is violated; the message names the offending field
        and, where meaningful, the first offending pixel index.

    See Also
    --------
    Dataset : A validated collection of epochs plus the frame declaration.

    Examples
    --------
    >>> import numpy as np
    >>> ep = EpochData(
    ...     wave=[4000.0, 4000.1, 4000.2],
    ...     flux=[1.0, 0.7, np.nan],
    ...     ivar=[100.0, 100.0, 0.0],  # last pixel masked, so its nan is fine
    ...     bjd=2459000.5,
    ... )
    >>> ep.n_pixels
    3
    >>> ep.good
    array([ True,  True, False])
    """

    wave: np.ndarray
    flux: np.ndarray
    ivar: np.ndarray
    bjd: float
    v_bary: float = 0.0
    instrument: str = "default"
    mask: np.ndarray | None = None
    medium: str | None = None

    def __post_init__(self) -> None:
        wave = _as_float64_1d(self.wave, "wave")
        flux = _as_float64_1d(self.flux, "flux")
        ivar = _as_float64_1d(self.ivar, "ivar")

        if not (wave.size == flux.size == ivar.size):
            raise ValueError(
                "wave, flux and ivar must have the same length; got "
                f"{wave.size}, {flux.size}, {ivar.size}"
            )
        if wave.size < 2:
            raise ValueError(f"an epoch needs at least 2 pixels, got {wave.size}")

        finite_wave = np.isfinite(wave)
        if not finite_wave.all():
            bad = int(np.argmax(~finite_wave))
            raise ValueError(f"wave must be finite; wave[{bad}] = {wave[bad]}")
        if not (wave > 0.0).all():
            bad = int(np.argmax(wave <= 0.0))
            raise ValueError(f"wave must be strictly positive; wave[{bad}] = {wave[bad]}")
        steps = np.diff(wave)
        if not (steps > 0.0).all():
            bad = int(np.argmax(steps <= 0.0))
            raise ValueError(
                "wave must be strictly increasing (albireo never resamples or reorders data); "
                f"wave[{bad}] = {wave[bad]} is not less than wave[{bad + 1}] = {wave[bad + 1]}"
            )

        finite_ivar = np.isfinite(ivar)
        if not finite_ivar.all():
            bad = int(np.argmax(~finite_ivar))
            raise ValueError(
                f"ivar must be finite (use ivar = 0 to mask a pixel); ivar[{bad}] = {ivar[bad]}"
            )
        if not (ivar >= 0.0).all():
            bad = int(np.argmax(ivar < 0.0))
            raise ValueError(f"ivar must be non-negative; ivar[{bad}] = {ivar[bad]}")

        bad_flux = ~np.isfinite(flux) & (ivar > 0.0)
        if bad_flux.any():
            n_bad = int(bad_flux.sum())
            bad = int(np.argmax(bad_flux))
            raise ValueError(
                f"flux must be finite wherever ivar > 0; {n_bad} offending pixel(s), the first "
                f"at index {bad} with flux = {flux[bad]}, ivar = {ivar[bad]}. Set ivar = 0 at "
                "pixels that should be ignored (cosmics, chip gaps, deep tellurics)."
            )

        bjd = _as_finite_float(self.bjd, "bjd")
        v_bary = _as_finite_float(self.v_bary, "v_bary")

        if self.medium is not None and self.medium not in _MEDIA:
            raise ValueError(
                f"medium must be one of {_MEDIA} or None (undeclared), got {self.medium!r}"
            )

        mask: np.ndarray | None = None
        if self.mask is not None:
            try:
                mask = np.asarray(self.mask, dtype=bool)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"mask must be convertible to a bool array ({exc})") from exc
            if mask.ndim != 1:
                raise ValueError(f"mask must be 1-D, got shape {mask.shape}")
            if mask.size != wave.size:
                raise ValueError(
                    f"mask must have the same length as wave ({wave.size}), got {mask.size}"
                )

        object.__setattr__(self, "wave", wave)
        object.__setattr__(self, "flux", flux)
        object.__setattr__(self, "ivar", ivar)
        object.__setattr__(self, "bjd", bjd)
        object.__setattr__(self, "v_bary", v_bary)
        object.__setattr__(self, "instrument", str(self.instrument))
        object.__setattr__(self, "mask", mask)

    @property
    def n_pixels(self) -> int:
        """Number of pixels in this epoch.

        Returns
        -------
        int
            The common length of ``wave``, ``flux`` and ``ivar``.
        """
        return int(self.wave.size)

    @property
    def good(self) -> np.ndarray:
        """Boolean array of usable pixels: positive weight, and not flagged out by ``mask``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_pixels,)``, dtype ``bool``. ``ivar > 0`` when ``mask`` is ``None``,
            otherwise ``(ivar > 0) & mask``. A fresh array is returned on every access.
        """
        ok = self.ivar > 0.0
        if self.mask is not None:
            ok &= self.mask
        return ok

    @property
    def effective_ivar(self) -> np.ndarray:
        """Inverse variances with the mask folded in: the weights downstream code consumes.

        This is the single place where ``mask`` affects anything. The result is ``ivar`` with
        zeros wherever :attr:`good` is ``False``, restoring the convention that a masked pixel
        is a zero-weight pixel.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_pixels,)``, dtype ``float64``, non-negative. A fresh array is returned
            on every access.
        """
        return np.where(self.good, self.ivar, 0.0)


@dataclasses.dataclass(frozen=True)
class Dataset:
    """A validated collection of epochs plus the frame their wavelengths are in.

    Parameters
    ----------
    epochs : sequence of EpochData
        One or more epochs, in any order. A list is coerced to a tuple; epochs are not
        sorted, so the caller's ordering is preserved everywhere (including in
        :attr:`bjd` and :attr:`v_bary`).
    frame : {"topocentric", "barycentric"}, optional
        The frame the ``wave`` arrays are in. Declaring the frame does not transform
        anything: the barycentric correction is applied inside the forward model
        (``docs/math.md`` §1.2). Default ``"topocentric"``.

    Raises
    ------
    ValueError
        If ``epochs`` is empty, contains a non-:class:`EpochData` element, or ``frame`` is
        not one of the two recognized values.

    Examples
    --------
    >>> ep = EpochData(wave=[4000.0, 4001.0], flux=[1.0, 1.0], ivar=[9.0, 9.0], bjd=2459000.5)
    >>> ds = Dataset([ep, ep], frame="barycentric")
    >>> len(ds), ds.instruments
    (2, ('default',))
    """

    epochs: tuple[EpochData, ...]
    frame: str = "topocentric"

    def __post_init__(self) -> None:
        if isinstance(self.epochs, EpochData):
            raise ValueError(
                "epochs must be a sequence of EpochData, not a single EpochData (wrap it in a list)"
            )
        try:
            epochs = tuple(self.epochs)
        except TypeError as exc:
            raise ValueError(
                f"epochs must be a sequence of EpochData, got {type(self.epochs).__name__}"
            ) from exc

        if not epochs:
            raise ValueError("a Dataset needs at least one epoch, got an empty sequence")
        for k, epoch in enumerate(epochs):
            if not isinstance(epoch, EpochData):
                raise ValueError(f"epochs[{k}] must be an EpochData, got {type(epoch).__name__}")

        if self.frame not in _FRAMES:
            raise ValueError(f"frame must be one of {_FRAMES}, got {self.frame!r}")

        # Every epoch must be on the same wavelength scale. Air and vacuum differ by a
        # nearly constant 83 km/s across the optical, the same order as the orbits albireo
        # measures, so a mixture puts one physical line at two model pixels and biases every
        # velocity that depends on the offending epochs. Archives mix them (ESO delivers air
        # from FEROS/HARPS/UVES/GIRAFFE and vacuum from ESPRESSO), so the agreement is
        # checked rather than assumed.
        media = {epoch.medium for epoch in epochs}
        if len(media) > 1:
            where: dict[str | None, list[object]] = {}
            for k, epoch in enumerate(epochs):
                where.setdefault(epoch.medium, []).append(k)
            summary = "; ".join(
                f"{name if name is not None else 'undeclared'}: epochs "
                f"{indices if len(indices) <= 4 else [*indices[:4], '...']}"
                for name, indices in sorted(where.items(), key=lambda kv: str(kv[0]))
            )
            raise ValueError(
                f"the epochs disagree about their wavelength scale ({summary}). Air and "
                "vacuum wavelengths differ by a nearly constant 83 km/s across the "
                "optical, so combining them without conversion would put the same line "
                "at different model pixels. Convert with albireo.air_to_vacuum or "
                "albireo.vacuum_to_air and set medium= on every epoch. An undeclared "
                "epoch cannot be combined with a declared one: 'unknown' is not a value "
                "this can be checked against."
            )

        object.__setattr__(self, "epochs", epochs)

    def __len__(self) -> int:
        """Number of epochs."""
        return len(self.epochs)

    def __iter__(self) -> Iterator[EpochData]:
        """Iterate over the epochs, in the order they were supplied."""
        return iter(self.epochs)

    def __getitem__(self, index: int) -> EpochData:
        """Return the epoch at position ``index``."""
        return self.epochs[index]

    @property
    def n_epochs(self) -> int:
        """Number of epochs.

        Returns
        -------
        int
        """
        return len(self.epochs)

    @property
    def bjd(self) -> np.ndarray:
        """Mid-exposure times of all epochs.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_epochs,)``, dtype ``float64``, in supplied order (not sorted).
        """
        return np.array([epoch.bjd for epoch in self.epochs], dtype=np.float64)

    @property
    def v_bary(self) -> np.ndarray:
        """Barycentric correction velocities of all epochs.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_epochs,)``, dtype ``float64``, in km/s and in supplied order.
        """
        return np.array([epoch.v_bary for epoch in self.epochs], dtype=np.float64)

    @property
    def instruments(self) -> tuple[str, ...]:
        """The distinct instrument keys present, sorted alphabetically.

        Returns
        -------
        tuple of str
            Each key appears once; these are the keys the LSF and response tables must cover.
        """
        return tuple(sorted({epoch.instrument for epoch in self.epochs}))

    def summary(self) -> str:
        """Human-readable multi-line summary of the dataset.

        Reports the epoch count and frame, the time span, a per-instrument breakdown
        (epoch count, wavelength coverage, pixel count) and the overall good-pixel fraction.
        Intended for logging and notebooks; the format is not a stable API.

        Returns
        -------
        str
            The summary, without a trailing newline.
        """
        bjd = self.bjd
        n_pixels = sum(epoch.n_pixels for epoch in self.epochs)
        n_good = sum(int(epoch.good.sum()) for epoch in self.epochs)
        instruments = self.instruments
        width = max(len(name) for name in instruments)

        lines = [
            f"Dataset: {_plural(self.n_epochs, 'epoch')}, frame={self.frame!r}",
            f"  BJD_TDB {bjd.min():.5f} to {bjd.max():.5f} "
            f"(span {float(bjd.max() - bjd.min()):.5f} d)",
            f"  instruments ({len(instruments)}):",
        ]
        for name in instruments:
            subset = [epoch for epoch in self.epochs if epoch.instrument == name]
            wave_min = min(epoch.wave[0] for epoch in subset)
            wave_max = max(epoch.wave[-1] for epoch in subset)
            n_sub = sum(epoch.n_pixels for epoch in subset)
            lines.append(
                f"    {name:<{width}}  {_plural(len(subset), 'epoch')}, "
                f"{wave_min:.2f}-{wave_max:.2f} A, {n_sub} px"
            )
        lines.append(f"  good pixels: {n_good} / {n_pixels} ({100.0 * n_good / n_pixels:.1f}%)")
        return "\n".join(lines)


def _plural(count: int, noun: str) -> str:
    """Format ``count`` with ``noun``, adding a naive plural ``s`` when needed."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"

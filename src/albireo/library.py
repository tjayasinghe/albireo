"""Synthetic spectral libraries: standardized containers and differentiable interpolation.

A library is a published grid of synthetic spectra (BOSZ, Bohlin et al. 2017 and Mészáros
et al. 2024; POLLUX, Palacios et al. 2010; PHOENIX, Husser et al. 2013) reduced to the four
quantities a label fit needs: the node labels, the normalized flux at each node, the
continuum at each node, and the wavelength scale those are tabulated on. The medium of that
wavelength scale is a required field of :class:`SpectralLibrary`. Everything else the
upstream distributions carry (SEDs, stratifications, ionizing fluxes) is dropped at ingest.

The module does not synthesize spectra: there is no line list, no model atmosphere and no
radiative transfer. albireo reads grids computed elsewhere and cites them; bespoke synthesis
is reached through :mod:`albireo.handoff` and GSSP, iSpec, Korg.jl or PySME. Nor does the
module guess the wavelength medium. Air and vacuum differ by ~83 km/s across the optical,
the same order as the orbital semi-amplitudes albireo measures, and the distributions are
not a reliable source: BOSZ 2017 was vacuum throughout while BOSZ 2024 is air above 200 nm,
under the same name. :func:`line_core_medium` measures the convention from the spectra, and
the ingest paths use it to verify a declaration rather than to supply one.

The module also does not interpolate model atmospheres. Interpolation is performed in flux,
which is more accurate: on the 250 K / 0.5 dex spacing BOSZ uses, Mészáros & Allende Prieto
(2013) measured 0.19% scatter interpolating atmospheres against 0.051% interpolating fluxes
linearly and 0.031% with a cubic. :func:`library_interpolator` therefore defaults to a cubic
in flux space, and an emulator is not an evident improvement on it.

Continua are stored and interpolated in the log. They are positive and span decades across
the Teff range, so the log makes the interpolation accurate and the exponential guarantees
the positivity the light-fraction simplex requires.

References
----------
Bohlin, R. C., Mészáros, Sz., Fleming, S. W., et al. 2017, AJ, 153, 234
Husser, T.-O., Wende-von Berg, S., Dreizler, S., et al. 2013, A&A, 553, A6
Mészáros, Sz. & Allende Prieto, C. 2013, MNRAS, 430, 3285
Mészáros, Sz., Bohlin, R., Allende Prieto, C., et al. 2024, A&A, 688, A171
Palacios, A., Gebran, M., Josselin, E., et al. 2010, A&A, 516, A13
"""

from __future__ import annotations

import datetime as _dt
import gzip
import hashlib
import json
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from albireo.examples import _download_with_retries, _sha256, cache_dir
from albireo.grids import LogGrid, air_to_vacuum, vacuum_to_air
from albireo.operators import rebin_operator

__all__ = [
    "SUPPORTED_MEDIA",
    "BoxInterpolator",
    "SimplexInterpolator",
    "SpectralLibrary",
    "clear_library_cache",
    "crossval_library",
    "fetch_library",
    "ingest_bosz",
    "ingest_pollux",
    "library_info",
    "library_interpolator",
    "library_names",
    "line_core_medium",
    "load_library",
    "save_library",
]

SUPPORTED_MEDIA = ("air", "vacuum")
"""The two wavelength scales a library may declare. There is no third option and no default."""


# ---------------------------------------------------------------------------
# The container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpectralLibrary:
    """A grid of synthetic spectra, standardized for label fitting.

    Build-time container: plain NumPy, no tracing. The traced object is the interpolator
    that :func:`library_interpolator` builds from it.

    Attributes
    ----------
    label_names
        Names of the label axes, in the column order of ``nodes``, for example
        ``("teff", "logg", "mh")``. A grid computed at one fixed metallicity omits that
        axis; the fit then requires ``mh`` to be ``Fixed``, and reports it.
    nodes
        Label values at each node, shape ``(n_nodes, n_labels)``, in physical units
        (K, dex, dex).
    normalized
        Continuum-normalized flux, shape ``(n_nodes, n_pix)``. Order matches ``nodes``.
    log_continuum
        Natural log of the continuum flux, same shape. The unit is that of the upstream
        grid; only ratios between components enter the model, so the unit cancels, but two
        libraries mixed in one fit must share it, which ``meta["continuum_unit"]`` records
        and the fit checks.
    wave
        Wavelengths in Angstrom, shape ``(n_pix,)``, strictly increasing.
    medium
        ``"air"`` or ``"vacuum"``. Required.
    meta
        Provenance: grid name, upstream version, retrieval date, checksum, microturbulence,
        continuum unit, licence, citation. Carried into every template file written from a
        fit, so that the template remains reproducible.
    """

    label_names: tuple[str, ...]
    nodes: np.ndarray
    normalized: np.ndarray
    log_continuum: np.ndarray
    wave: np.ndarray
    medium: str
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        nodes = np.asarray(self.nodes, dtype=np.float64)
        normalized = np.asarray(self.normalized, dtype=np.float64)
        log_continuum = np.asarray(self.log_continuum, dtype=np.float64)
        wave = np.asarray(self.wave, dtype=np.float64)
        object.__setattr__(self, "label_names", tuple(self.label_names))
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "normalized", normalized)
        object.__setattr__(self, "log_continuum", log_continuum)
        object.__setattr__(self, "wave", wave)

        if self.medium not in SUPPORTED_MEDIA:
            raise ValueError(
                f"medium must be one of {SUPPORTED_MEDIA}, got {self.medium!r}. This is "
                "required and has no default: air and vacuum differ by ~83 km/s, and the "
                "upstream documentation is not reliable (BOSZ changed convention between "
                "2017 and 2024 under one name). Use line_core_medium() to measure it."
            )
        if nodes.ndim != 2 or nodes.shape[1] != len(self.label_names):
            raise ValueError(
                f"nodes must be (n_nodes, {len(self.label_names)}) to match label_names "
                f"{self.label_names}; got {nodes.shape}"
            )
        if normalized.shape != log_continuum.shape:
            raise ValueError(
                f"normalized {normalized.shape} and log_continuum {log_continuum.shape} "
                "must have the same shape"
            )
        if normalized.shape != (nodes.shape[0], wave.size):
            raise ValueError(
                f"normalized must be (n_nodes, n_pix) = ({nodes.shape[0]}, {wave.size}); "
                f"got {normalized.shape}"
            )
        if wave.ndim != 1 or wave.size < 2 or np.any(np.diff(wave) <= 0):
            raise ValueError("wave must be 1-D, strictly increasing, with at least 2 points")
        if not np.all(np.isfinite(normalized)) or not np.all(np.isfinite(log_continuum)):
            raise ValueError("normalized and log_continuum must be finite everywhere")
        if not np.all(np.isfinite(nodes)):
            raise ValueError("nodes must be finite")
        if nodes.shape[0] != len({tuple(row) for row in nodes}):
            raise ValueError("nodes contains duplicate label vectors")

    # -- geometry ----------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        """Number of grid nodes."""
        return int(self.nodes.shape[0])

    @property
    def n_pix(self) -> int:
        """Number of wavelength pixels."""
        return int(self.wave.size)

    @property
    def bounds(self) -> dict[str, tuple[float, float]]:
        """Per-label ``(min, max)`` over the nodes: the box a fit must stay inside."""
        return {
            name: (float(self.nodes[:, i].min()), float(self.nodes[:, i].max()))
            for i, name in enumerate(self.label_names)
        }

    def axes(self) -> list[np.ndarray] | None:
        """Sorted unique values per label axis if the nodes form a complete box, else None.

        A complete box means the node set is exactly the Cartesian product of its axes. A
        BOSZ subset is such a set; a grid whose corners are cut away by physics, such as
        POLLUX's OB models, is not. :func:`library_interpolator` dispatches on this.
        """
        axes = [np.unique(self.nodes[:, i]) for i in range(self.nodes.shape[1])]
        if int(np.prod([a.size for a in axes])) != self.n_nodes:
            return None
        return axes

    # -- transforms --------------------------------------------------------

    def in_medium(self, medium: str) -> SpectralLibrary:
        """The same library with wavelengths on the requested scale.

        Converts the wavelength axis only: the fluxes are unchanged, since they are
        tabulated per pixel rather than per unit wavelength. Returns ``self`` when the
        medium already matches, so it may be called unconditionally.
        """
        if medium not in SUPPORTED_MEDIA:
            raise ValueError(f"medium must be one of {SUPPORTED_MEDIA}, got {medium!r}")
        if medium == self.medium:
            return self
        convert = air_to_vacuum if medium == "vacuum" else vacuum_to_air
        return self.replace(wave=np.asarray(convert(self.wave), dtype=np.float64), medium=medium)

    def sliced(self, wave_min: float, wave_max: float) -> SpectralLibrary:
        """Restrict to a wavelength window, keeping one pixel of margin on each side."""
        if not wave_max > wave_min:
            raise ValueError("need wave_min < wave_max")
        lo = int(np.searchsorted(self.wave, wave_min, side="right") - 1)
        hi = int(np.searchsorted(self.wave, wave_max, side="left") + 1)
        lo, hi = max(lo, 0), min(hi + 1, self.n_pix)
        if hi - lo < 2:
            raise ValueError(
                f"window [{wave_min}, {wave_max}] Angstrom does not overlap the library, "
                f"which spans [{self.wave[0]:.1f}, {self.wave[-1]:.1f}]"
            )
        return self.replace(
            normalized=self.normalized[:, lo:hi],
            log_continuum=self.log_continuum[:, lo:hi],
            wave=self.wave[lo:hi],
        )

    def resampled_to(self, grid: LogGrid, *, medium: str) -> SpectralLibrary:
        """Project onto a model grid, converting the wavelength scale on the way.

        Flux-conserving pixel integration (:func:`albireo.operators.rebin_operator`) is
        used going from a high-resolution library down to a model grid: a point sample would
        alias the unresolved lines instead of averaging them. The box average adds a
        broadening of ``dv/sqrt(12)``, about 1 km/s at ``dv = 3.5`` km/s, negligible in
        quadrature against any real LSF and identical on both sides of the comparison, since
        the data are convolved by the same instrument profile.

        This moves the model onto the data's grid. The data are never resampled
        never resampled.
        """
        library = self.in_medium(medium)
        target = np.asarray(grid.wave, dtype=np.float64)
        if library.wave[0] > target[0] or library.wave[-1] < target[-1]:
            raise ValueError(
                f"library spans [{library.wave[0]:.2f}, {library.wave[-1]:.2f}] Angstrom in "
                f"{medium} and cannot cover the model grid "
                f"[{target[0]:.2f}, {target[-1]:.2f}]. Fetch a wider band, or narrow the "
                "analysis window."
            )
        operator = rebin_operator(library.wave, target)
        apply = jax.jit(jax.vmap(operator))
        return SpectralLibrary(
            label_names=library.label_names,
            nodes=library.nodes,
            normalized=np.asarray(apply(jnp.asarray(library.normalized))),
            log_continuum=np.asarray(apply(jnp.asarray(library.log_continuum))),
            wave=target,
            medium=medium,
            meta={**library.meta, "resampled_to_grid": True},
        )

    def replace(self, **changes) -> SpectralLibrary:
        """A copy with fields replaced (``dataclasses.replace``, re-validated)."""
        fields = {
            "label_names": self.label_names,
            "nodes": self.nodes,
            "normalized": self.normalized,
            "log_continuum": self.log_continuum,
            "wave": self.wave,
            "medium": self.medium,
            "meta": dict(self.meta),
        }
        fields.update(changes)
        return SpectralLibrary(**fields)

    def summary(self) -> str:
        """A short human-readable description, including the library's provenance."""
        lines = [
            f"SpectralLibrary: {self.n_nodes} nodes x {self.n_pix} pixels",
            f"  wavelengths  {self.wave[0]:.2f} - {self.wave[-1]:.2f} Angstrom ({self.medium})",
        ]
        for name, (lo, hi) in self.bounds.items():
            n_unique = np.unique(self.nodes[:, self.label_names.index(name)]).size
            lines.append(f"  {name:<12} {lo:g} to {hi:g}  ({n_unique} values)")
        lines.append(f"  geometry     {'complete box' if self.axes() else 'irregular coverage'}")
        for key in ("grid", "version", "retrieved", "vmicro", "citation"):
            if key in self.meta:
                lines.append(f"  {key:<12} {self.meta[key]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interpolators
# ---------------------------------------------------------------------------


def _axis_weights(axis: jax.Array, value, cubic: bool):
    """Stencil indices and weights for one label axis.

    Returns ``(idx, w)`` with ``idx`` of length 2 (linear) or 4 (Catmull-Rom). Both
    reproduce a node exactly: at a node the fractional coordinate is exactly 0 and the
    weight vector is exactly ``(1, 0, ...)``, so the gather returns the stored spectrum
    bit for bit. The tests assert that with ``==``.
    """
    n = axis.shape[0]
    if n == 1:
        # A degenerate axis, for example a library sliced to one metallicity. The
        # interpolant is constant along it, which is the only defensible reading: there is
        # no second node. Without this branch the clip below produces an empty cell,
        # lo == hi, and a 0/0 that propagates as a NaN through every pixel.
        return jnp.zeros(1, dtype=int), jnp.ones(1, dtype=jnp.float64)
    i = jnp.clip(jnp.searchsorted(axis, value, side="right") - 1, 0, n - 2)
    lo, hi = axis[i], axis[i + 1]
    t = (value - lo) / (hi - lo)
    if not cubic or n < 3:
        return jnp.stack([i, i + 1]), jnp.stack([1.0 - t, t])
    # Catmull-Rom in its uniform-parameter form: exact at t = 0 and t = 1.
    t2, t3 = t * t, t * t * t
    w_m1 = -0.5 * t3 + t2 - 0.5 * t
    w_0 = 1.5 * t3 - 2.5 * t2 + 1.0
    w_p1 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
    w_p2 = 0.5 * t3 - 0.5 * t2

    # Edge cells need a phantom node. Clamping it to the end node destroys linear
    # reproduction there, and on a grid with only a handful of values per axis the cubic
    # then loses to plain multilinear over about a third of the range. Extrapolating the
    # phantom linearly, f(-1) := 2 f(0) - f(1), keeps the interpolant exact on linear data
    # everywhere.
    at_low = i == 0
    at_high = i == n - 2
    w_0, w_p1 = (
        jnp.where(at_low, w_0 + 2.0 * w_m1, w_0),
        jnp.where(at_low, w_p1 - w_m1, w_p1),
    )
    w_m1 = jnp.where(at_low, 0.0, w_m1)
    w_p1, w_0 = (
        jnp.where(at_high, w_p1 + 2.0 * w_p2, w_p1),
        jnp.where(at_high, w_0 - w_p2, w_0),
    )
    w_p2 = jnp.where(at_high, 0.0, w_p2)

    idx = jnp.stack([jnp.maximum(i - 1, 0), i, i + 1, jnp.minimum(i + 2, n - 1)])
    return idx, jnp.stack([w_m1, w_0, w_p1, w_p2])


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BoxInterpolator:
    """Separable interpolation on a complete axis-product grid.

    Multilinear or Catmull-Rom cubic (the default), applied to the flux itself. The cubic
    costs 4^k taps rather than 2^k and returns three properties: it is C^1 in the labels,
    which both L-BFGS and NUTS require; it has local support, so it cannot ring across a
    Balmer jump; and on BOSZ's spacing it halves the interpolation error (Mészáros & Allende
    Prieto 2013).

    Call with a label vector; returns ``(normalized, log_continuum)``, each ``(n_pix,)``.

    References
    ----------
    Mészáros, Sz. & Allende Prieto, C. 2013, MNRAS, 430, 3285
    """

    axes: tuple[jax.Array, ...]
    normalized: jax.Array
    log_continuum: jax.Array
    cubic: bool = True

    def __call__(self, labels):
        labels = jnp.atleast_1d(jnp.asarray(labels, dtype=jnp.float64))
        stencils = [_axis_weights(a, labels[i], self.cubic) for i, a in enumerate(self.axes)]
        out = []
        for values in (self.normalized, self.log_continuum):
            acc = values
            # Contract one axis at a time: gather the stencil, then weight it away.
            for idx, w in stencils:
                acc = jnp.tensordot(w, acc[idx], axes=(0, 0))
            out.append(acc)
        return out[0], out[1]

    def hull_margin(self, labels):
        """Signed distance into the box, in units of each axis span (>= 0 inside)."""
        labels = jnp.atleast_1d(jnp.asarray(labels, dtype=jnp.float64))
        margins = [
            jnp.minimum(labels[i] - a[0], a[-1] - labels[i]) / jnp.maximum(a[-1] - a[0], 1e-30)
            for i, a in enumerate(self.axes)
        ]
        return jnp.min(jnp.stack(margins))

    def tree_flatten(self):
        return (self.axes, self.normalized, self.log_continuum), self.cubic

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(axes=children[0], normalized=children[1], log_continuum=children[2], cubic=aux)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SimplexInterpolator:
    """Barycentric interpolation over a Delaunay triangulation of scattered nodes.

    For grids whose coverage is bounded by physics rather than by a box: POLLUX's OB models
    have no cool, low-gravity corner, because no such star exists, so the axis product is not
    the node set and :class:`BoxInterpolator` does not apply.

    The triangulation is built once in NumPy. Under trace, barycentric coordinates are
    evaluated against every simplex by one batched affine map, and the containing simplex is
    the one whose minimum coordinate is largest. That is a few thousand floating-point
    operations for a realistic grid, and it keeps the object jit- and vmap-safe with no
    callbacks. Outside the hull the weights are clipped and renormalized, which extrapolates
    flat rather than diverging; :meth:`hull_margin` is negative there, so the fit can be
    told.

    References
    ----------
    Palacios, A., Gebran, M., Josselin, E., et al. 2010, A&A, 516, A13
    """

    transform: jax.Array
    simplices: jax.Array
    scale: jax.Array
    origin: jax.Array
    normalized: jax.Array
    log_continuum: jax.Array

    def _barycentric(self, labels):
        x = (jnp.atleast_1d(jnp.asarray(labels, dtype=jnp.float64)) - self.origin) / self.scale
        k = x.shape[0]
        offset = x[None, :] - self.transform[:, k, :]
        first = jnp.einsum("sij,sj->si", self.transform[:, :k, :], offset)
        return jnp.concatenate([first, 1.0 - jnp.sum(first, axis=1, keepdims=True)], axis=1)

    def __call__(self, labels):
        bary = self._barycentric(labels)
        best = jnp.argmax(jnp.min(bary, axis=1))
        weights = jnp.clip(bary[best], 0.0, None)
        weights = weights / jnp.sum(weights)
        idx = self.simplices[best]
        return (
            jnp.tensordot(weights, self.normalized[idx], axes=(0, 0)),
            jnp.tensordot(weights, self.log_continuum[idx], axes=(0, 0)),
        )

    def hull_margin(self, labels):
        """Largest minimum barycentric coordinate: ``>= 0`` inside the convex hull."""
        return jnp.max(jnp.min(self._barycentric(labels), axis=1))

    def tree_flatten(self):
        children = (
            self.transform,
            self.simplices,
            self.scale,
            self.origin,
            self.normalized,
            self.log_continuum,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


def library_interpolator(
    library: SpectralLibrary, *, method: str = "auto"
) -> BoxInterpolator | SimplexInterpolator:
    """Build a differentiable interpolator over a library's nodes.

    Parameters
    ----------
    library
        The grid to interpolate. Already resampled onto the model grid, normally.
    method
        ``"auto"`` (default) selects a Catmull-Rom cubic when the nodes form a complete
        box and barycentric interpolation when they do not. ``"linear"`` forces multilinear
        on a box, ``"cubic"`` forces the cubic, and ``"simplex"`` forces the scattered path
        even for a box, which measures what the box structure contributes.

    Returns
    -------
    BoxInterpolator or SimplexInterpolator
        A callable pytree, safe to pass through ``jit`` as a traced model argument.
    """
    if method not in ("auto", "linear", "cubic", "simplex"):
        raise ValueError(f"method must be auto, linear, cubic or simplex; got {method!r}")
    axes = None if method == "simplex" else library.axes()

    if axes is not None:
        order = np.lexsort(tuple(library.nodes[:, i] for i in reversed(range(len(axes)))))
        shape = tuple(a.size for a in axes)
        return BoxInterpolator(
            axes=tuple(jnp.asarray(a) for a in axes),
            normalized=jnp.asarray(library.normalized[order].reshape(*shape, library.n_pix)),
            log_continuum=jnp.asarray(library.log_continuum[order].reshape(*shape, library.n_pix)),
            cubic=method != "linear",
        )

    if method in ("linear", "cubic"):
        raise ValueError(
            f"method={method!r} needs a complete axis-product grid, but this library's "
            f"{library.n_nodes} nodes do not form one (its coverage is irregular). Use "
            "method='auto' or 'simplex'."
        )
    from scipy.spatial import Delaunay  # scipy ships with jax; imported here to keep it local

    origin = library.nodes.min(axis=0)
    span = np.maximum(library.nodes.max(axis=0) - origin, 1e-30)
    if library.nodes.shape[1] < 2:
        raise ValueError(
            "a one-label library is always a complete box; irregular coverage needs at "
            "least two label axes"
        )
    tri = Delaunay((library.nodes - origin) / span)
    # Published grids are lattices with pieces removed, and a lattice is the cospherical
    # configuration Qhull cannot triangulate uniquely: it emits zero-volume simplices whose
    # barycentric transform is NaN. Those cover no volume, so dropping them loses no domain,
    # while keeping them would give `argmax` a NaN spectrum to return. Joggling the input
    # would fix the degeneracy at the cost of moving the nodes, which would break exact node
    # reproduction.
    usable = ~np.isnan(tri.transform).any(axis=(1, 2))
    if not usable.any():
        raise ValueError(
            f"the Delaunay triangulation of these {library.n_nodes} nodes is entirely "
            "degenerate; the labels are probably collinear or duplicated"
        )
    return SimplexInterpolator(
        transform=jnp.asarray(tri.transform[usable]),
        simplices=jnp.asarray(tri.simplices[usable], dtype=jnp.int32),
        scale=jnp.asarray(span),
        origin=jnp.asarray(origin),
        normalized=jnp.asarray(library.normalized),
        log_continuum=jnp.asarray(library.log_continuum),
    )


# ---------------------------------------------------------------------------
# Measuring what the interpolation costs
# ---------------------------------------------------------------------------


def crossval_library(library: SpectralLibrary, *, method: str = "auto", seed: int = 0) -> dict:
    """Measure interpolation error by holding nodes out and predicting them.

    The result decides whether a library needs a learned emulator. Published context for
    the same quantity: on a 250 K / 0.5 dex FGK grid, Mészáros & Allende Prieto (2013)
    report 0.051% for linear and 0.031% for cubic flux interpolation, against roughly 0.1%
    for a Payne-style network. A library near those numbers does not require an emulator; a
    coarse, strongly non-linear grid may.

    For a complete box the held-out set is every other node along each axis, so the
    surviving grid has twice the spacing. That is a pessimistic proxy, since the real fit
    interpolates on the full grid. For irregular coverage a random fifth of the nodes is held
    out and the triangulation rebuilt without them.

    Returns
    -------
    dict
        ``n_tested``, ``rms``, ``median``, ``p95``, ``max``, the fractional flux errors on
        the normalized spectra, plus ``spacing`` (``"doubled"`` or ``"scattered"``) and
        ``method``.

    References
    ----------
    Mészáros, Sz. & Allende Prieto, C. 2013, MNRAS, 430, 3285
    """
    axes = library.axes()
    if axes is not None and method != "simplex":
        keep_values = [a[::2] if a.size >= 5 else a for a in axes]
        keep = np.ones(library.n_nodes, dtype=bool)
        for i, values in enumerate(keep_values):
            keep &= np.isin(library.nodes[:, i], values)
        spacing = "doubled"
    else:
        rng = np.random.default_rng(seed)
        keep = np.ones(library.n_nodes, dtype=bool)
        keep[rng.choice(library.n_nodes, size=max(1, library.n_nodes // 5), replace=False)] = False
        spacing = "scattered"

    if keep.sum() < 4 or (~keep).sum() == 0:
        raise ValueError(
            f"library has too few nodes ({library.n_nodes}) to hold any out for cross-validation"
        )
    reduced = library.replace(
        nodes=library.nodes[keep],
        normalized=library.normalized[keep],
        log_continuum=library.log_continuum[keep],
    )
    interpolator = library_interpolator(reduced, method=method)
    predict = jax.jit(jax.vmap(interpolator))

    test = np.flatnonzero(~keep)
    inside = np.asarray(jax.jit(jax.vmap(interpolator.hull_margin))(library.nodes[test])) >= 0.0
    test = test[inside]
    if test.size == 0:
        raise ValueError("every held-out node fell outside the reduced grid; nothing to measure")
    predicted = np.asarray(predict(library.nodes[test])[0])
    error = predicted - library.normalized[test]
    return {
        "n_tested": int(test.size),
        "rms": float(np.sqrt(np.mean(error**2))),
        "median": float(np.median(np.abs(error))),
        "p95": float(np.percentile(np.abs(error), 95)),
        "max": float(np.max(np.abs(error))),
        "spacing": spacing,
        "method": method,
    }


# ---------------------------------------------------------------------------
# Measuring the wavelength medium
# ---------------------------------------------------------------------------

_MEDIUM_LINES: tuple[tuple[str, float], ...] = (
    # Strong, isolated, and present in essentially every optical stellar spectrum.
    # Vacuum wavelengths in Angstrom; the air counterparts are derived with albireo's own
    # converter, so a library that agrees with one disagrees with the other by ~1.5 A.
    ("H-delta", 4102.8991),
    ("H-gamma", 4341.6837),
    ("H-beta", 4862.6830),
    ("Mg b1", 5185.0479),
    ("Na D2", 5891.5833),
    ("H-alpha", 6564.6127),
)


def line_core_medium(
    wave, flux, *, window_angstrom: float = 3.0, decisive: float = 4.0
) -> dict[str, Any]:
    """Decide whether a spectrum is on the air or the vacuum scale, by measuring it.

    Locates the core of each strong line in range, refines it by fitting a parabola to the
    three samples around the minimum, and compares the result against both conventions. The
    two differ by ~1.5 Angstrom in the optical while a correctly identified core lands within
    a few hundredths, so the verdict is not marginal. BOSZ is the case that requires the
    measurement: the 2017 release was vacuum throughout and the 2024 release is air above
    200 nm, under one name (Bohlin et al. 2017; Mészáros et al. 2024).

    Parameters
    ----------
    wave, flux
        A spectrum covering at least two of the reference lines. Normalized or not.
    window_angstrom
        Half-width of the search window around each reference position.
    decisive
        Required ratio between the losing and winning mean residuals. Below it the verdict
        is refused rather than guessed.

    Returns
    -------
    dict
        ``medium`` (``"air"`` or ``"vacuum"``), ``ratio``, ``n_lines``, and per-line
        ``residuals`` in Angstrom for both conventions.

    Raises
    ------
    ValueError
        If fewer than two reference lines are covered, or the verdict is not decisive. Both
        mean the medium must be established another way rather than guessed.

    References
    ----------
    Bohlin, R. C., Mészáros, Sz., Fleming, S. W., et al. 2017, AJ, 153, 234
    Mészáros, Sz., Bohlin, R., Allende Prieto, C., et al. 2024, A&A, 688, A171
    """
    wave = np.asarray(wave, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    if wave.shape != flux.shape or wave.ndim != 1:
        raise ValueError("wave and flux must be 1-D arrays of the same length")

    residuals: dict[str, dict[str, float]] = {}
    for name, vac in _MEDIUM_LINES:
        air = float(vacuum_to_air(vac))
        # Search around the midpoint so neither convention is favoured by the window.
        center = 0.5 * (vac + air)
        lo = int(np.searchsorted(wave, center - window_angstrom))
        hi = int(np.searchsorted(wave, center + window_angstrom))
        if hi - lo < 5 or lo == 0 or hi >= wave.size:
            continue
        k = lo + int(np.argmin(flux[lo:hi]))
        if k <= 0 or k >= wave.size - 1:
            continue
        y0, y1, y2 = flux[k - 1], flux[k], flux[k + 1]
        denominator = y0 - 2.0 * y1 + y2
        shift = 0.5 * (y0 - y2) / denominator if denominator > 0 else 0.0
        measured = wave[k] + shift * (wave[k + 1] - wave[k - 1]) * 0.5
        residuals[name] = {"air": abs(measured - air), "vacuum": abs(measured - vac)}

    if len(residuals) < 2:
        raise ValueError(
            "need at least two reference lines in range to measure the wavelength medium; "
            f"the spectrum spans {wave[0]:.1f}-{wave[-1]:.1f} Angstrom and covered "
            f"{len(residuals)}. Declare the medium explicitly instead."
        )
    means = {m: float(np.mean([r[m] for r in residuals.values()])) for m in SUPPORTED_MEDIA}
    winner = min(means, key=means.__getitem__)
    loser = next(m for m in SUPPORTED_MEDIA if m != winner)
    ratio = means[loser] / max(means[winner], 1e-12)
    if ratio < decisive:
        raise ValueError(
            f"the wavelength medium is not decisive: mean line-core residual "
            f"{means['air']:.4f} A against air and {means['vacuum']:.4f} A against vacuum "
            f"(ratio {ratio:.1f} < {decisive}). The lines may be too broad, too blended, or "
            "too coarsely sampled to locate. Declare the medium explicitly."
        )
    return {
        "medium": winner,
        "ratio": float(ratio),
        "n_lines": len(residuals),
        "residuals": residuals,
    }


# ---------------------------------------------------------------------------
# Named libraries: registry, download, cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Library:
    """One named library: its coverage, its source, and its citation.

    ``version`` is a cache-busting token rather than the upstream's version. It is bumped
    whenever the build changes (a different band, a different node box, a fixed axis moved),
    because the cached ``.npz`` is named after it and a stale cache would otherwise be
    indistinguishable from a fresh one.
    """

    name: str
    description: str
    source: str
    version: str
    wave_range: tuple[float, float]
    medium: str
    label_names: tuple[str, ...]
    axes: dict[str, Any]
    fixed: dict[str, Any]
    licence: str
    citation: str
    doi: str | None = None
    upstream_note: str = ""
    caveats: tuple[str, ...] = ()
    known_gaps: tuple[str, ...] = ()
    download_mb: float | None = None
    cache_mb: float | None = None


# The FGK box is the one Mészáros & Allende Prieto (2013) benchmarked interpolation on,
# 250 K in Teff and 0.5 dex in log g, so their measured 0.051% linear and 0.031% cubic apply
# to this grid directly rather than by analogy. Verified against the archive listing on
# 2026-08-27: Teff runs 4000..7000 in exact 250 K steps there.
_BOSZ_FGK_AXES: dict[str, Any] = {
    "teff": [float(t) for t in range(4000, 7001, 250)],
    "logg": [3.0, 3.5, 4.0, 4.5, 5.0],
    "mh": [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5],
}
_BOSZ_FIXED: dict[str, Any] = {"alpha": 0.0, "carbon": 0.0, "vmicro": 2, "resolution": 20000}

_BOSZ_CAVEATS = (
    "MARCS switches geometry inside the log g axis: spherical below log g 3.5, "
    "plane-parallel at and above it. That is the upstream's own arrangement, confirmed "
    "against the archive listing, but it means interpolating across log g 3.5 crosses a "
    "change of model geometry as well as of gravity.",
    "Microturbulence is pinned at 2 km/s. BOSZ offers 0, 1, 2 and 4, and fixing it silently "
    "is a documented way to bias [M/H]; the fit reports it as an assumption rather than "
    "hiding it.",
)

# Confirmed against the archive listing on 2026-08-27: this one model is absent while every
# carbon-varied version of it is present, so it is a gap in the published calculation rather
# than a naming error here. The grid is therefore not a complete box, which is why the
# shipped FGK library interpolates barycentrically rather than with the cubic.
_BOSZ_GAPS = ("Teff 5750 K, log g 3.0, [M/H] -0.75 is not published (a+0.00, c+0.00, v2).",)

_BOSZ_RECOMPUTE_NOTE = (
    "BOSZ 2024 was recomputed on 2025-09-25 to correct the hydrogen lines and the OH+ band "
    "strength. Anything cached before that date is the earlier calculation."
)

_LIBRARIES: dict[str, _Library] = {
    "bosz2024-fgk-r20000": _Library(
        name="bosz2024-fgk-r20000",
        description="BOSZ 2024 (MARCS) FGK optical, R = 20,000, 4000-7000 Angstrom",
        source="bosz2024",
        version="1",
        wave_range=(4000.0, 7000.0),
        medium="air",
        label_names=("teff", "logg", "mh"),
        axes=_BOSZ_FGK_AXES,
        fixed=_BOSZ_FIXED,
        licence="CC BY 4.0",
        citation="Meszaros et al. 2024, A&A 688, A171 (arXiv:2407.10872)",
        doi="10.17909/T95G68",
        upstream_note=_BOSZ_RECOMPUTE_NOTE,
        caveats=_BOSZ_CAVEATS,
        known_gaps=_BOSZ_GAPS,
        download_mb=645.0,
        cache_mb=95.0,
    ),
    "bosz2024-fgk-rvs": _Library(
        name="bosz2024-fgk-rvs",
        description="BOSZ 2024 (MARCS) FGK in the Gaia RVS band, R = 20,000, 8350-8850 Angstrom",
        source="bosz2024",
        version="1",
        wave_range=(8350.0, 8850.0),
        medium="air",
        label_names=("teff", "logg", "mh"),
        axes=_BOSZ_FGK_AXES,
        fixed=_BOSZ_FIXED,
        licence="CC BY 4.0",
        citation="Meszaros et al. 2024, A&A 688, A171 (arXiv:2407.10872)",
        doi="10.17909/T95G68",
        upstream_note=_BOSZ_RECOMPUTE_NOTE,
        caveats=(
            *_BOSZ_CAVEATS,
            "Gaia publishes RVS spectra on the vacuum scale and this library is air. "
            "Convert with SpectralLibrary.in_medium('vacuum') before comparing the two.",
        ),
        known_gaps=_BOSZ_GAPS,
        download_mb=645.0,
        cache_mb=16.0,
    ),
    "pollux-ob-smc24": _Library(
        name="pollux-ob-smc24",
        description="POLLUX OB-SMC-24 (CMFGEN) SMC OB stars, 3850-4650 Angstrom",
        source="pollux",
        version="1",
        wave_range=(3850.0, 4650.0),
        medium="unverified",
        label_names=("teff", "logg"),
        axes={"teff": None, "logg": None},
        fixed={"mh": -0.73},
        licence="CC BY 4.0",
        citation="Palacios et al. 2010, A&A 516, A13 (POLLUX database)",
        doi=None,
        upstream_note=(
            "POLLUX serves its collections through a form that posts to /download/, so there "
            "is no stable URL to fetch and no automatic path. Download the 629 MB OB-SMC-24 "
            "archive by hand from https://pollux.oreme.org/ and pass it to ingest_pollux()."
        ),
        caveats=(
            "[M/H] is fixed at -0.73 (0.19 Z_sun, the value BLOeM adopts for the SMC). It is "
            "not a fitted axis in this regime, and the fit requires mh to be Fixed.",
            "Coverage starts at 23,000 K, so early-B stars below that are absent. The PoWR "
            "SMC-OB grids span 15-50 kK and are the intended complement.",
            "The wavelength medium is undocumented upstream, so ingest_pollux() measures it "
            "with line_core_medium() instead of assuming one.",
        ),
        download_mb=629.0,
        cache_mb=120.0,
    ),
}

_BOSZ_ROOT = "https://archive.stsci.edu/hlsps/bosz/bosz2024"


def library_names() -> list[str]:
    """Names accepted by :func:`fetch_library`."""
    return sorted(_LIBRARIES)


def library_info(name: str) -> dict[str, Any]:
    """Everything the registry knows about one library, without downloading it.

    Carries the licence, the citation, the node box, the pinned axes, the download and
    cache sizes, and the caveats that belong beside any number the library produces.
    """
    lib = _lookup_library(name)
    return {
        "name": lib.name,
        "description": lib.description,
        "source": lib.source,
        "version": lib.version,
        "wave_range": lib.wave_range,
        "medium": lib.medium,
        "label_names": lib.label_names,
        "axes": {k: (list(v) if v is not None else None) for k, v in lib.axes.items()},
        "fixed": dict(lib.fixed),
        "n_nodes": _n_nodes(lib),
        "licence": lib.licence,
        "citation": lib.citation,
        "doi": lib.doi,
        "upstream_note": lib.upstream_note,
        "caveats": list(lib.caveats),
        "known_gaps": list(lib.known_gaps),
        "download_mb": lib.download_mb,
        "cache_mb": lib.cache_mb,
        "cached": _library_cache_path(lib).is_file(),
        "cache_path": str(_library_cache_path(lib)),
    }


def _lookup_library(name: str) -> _Library:
    try:
        return _LIBRARIES[name]
    except KeyError:
        raise KeyError(f"unknown library {name!r}; available: {library_names()}") from None


def _n_nodes(lib: _Library) -> int | None:
    if any(values is None for values in lib.axes.values()):
        return None
    total = 1
    for values in lib.axes.values():
        total *= len(values)
    return total


def _library_cache_path(lib: _Library) -> Path:
    # The version token is part of the filename: a re-pinned registry then cannot read a
    # cache built under the old definition.
    return cache_dir() / "libraries" / f"{lib.name}-v{lib.version}.npz"


def clear_library_cache(name: str | None = None) -> list[Path]:
    """Delete cached libraries; returns the paths removed.

    Raw upstream shards under ``libraries/_raw`` are left in place, because re-slicing
    another band out of them requires no download. Pass ``name="_raw"`` to clear those as
    well.
    """
    root = cache_dir() / "libraries"
    if name == "_raw":
        removed = []
        raw = root / "_raw"
        if raw.is_dir():
            for path in sorted(raw.rglob("*")):
                if path.is_file():
                    path.unlink()
                    removed.append(path)
        return removed
    targets = [_lookup_library(name)] if name else list(_LIBRARIES.values())
    removed = []
    for lib in targets:
        path = _library_cache_path(lib)
        if path.is_file():
            path.unlink()
            removed.append(path)
    return removed


# ---------------------------------------------------------------------------
# On-disk format
# ---------------------------------------------------------------------------


def _content_digest(library: SpectralLibrary) -> str:
    """A hash of the library's content, not of the file containing it.

    Taken over the arrays in their stored precision, so it survives a save/load round trip
    and two machines that built the same library agree on it whatever their npz compression
    produced. Hashing the file is not possible here: the digest is recorded inside the file,
    so it would have to describe bytes it is part of.
    """
    digest = hashlib.sha256()
    digest.update("|".join(library.label_names).encode())
    digest.update(library.medium.encode())
    for array, dtype in (
        (library.nodes, np.float64),
        (library.wave, np.float64),
        (library.normalized, np.float32),
        (library.log_continuum, np.float32),
    ):
        contiguous = np.ascontiguousarray(array, dtype=dtype)
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def save_library(library: SpectralLibrary, path) -> Path:
    """Write a library to a compressed ``.npz``.

    Fluxes are stored as float32 and restored to float64 on load. Interpolation error is
    ~1e-4 in normalized flux while float32 resolves ~1e-7, so the stored precision does not
    limit the result, and the file is substantially smaller.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    np.savez_compressed(
        tmp,
        label_names=np.asarray(library.label_names, dtype=object),
        nodes=library.nodes,
        normalized=library.normalized.astype(np.float32),
        log_continuum=library.log_continuum.astype(np.float32),
        wave=library.wave,
        medium=np.asarray(library.medium),
        meta=np.asarray(json.dumps(library.meta, default=str)),
    )
    # numpy appends .npz to a name that lacks it; locate the file it wrote.
    written = tmp if tmp.is_file() else tmp.with_name(tmp.name + ".npz")
    written.replace(path)
    return path


def load_library(path) -> SpectralLibrary:
    """Read a library written by :func:`save_library`."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as handle:
        library = SpectralLibrary(
            label_names=tuple(str(name) for name in handle["label_names"]),
            nodes=handle["nodes"].astype(np.float64),
            normalized=handle["normalized"].astype(np.float64),
            log_continuum=handle["log_continuum"].astype(np.float64),
            wave=handle["wave"].astype(np.float64),
            medium=str(handle["medium"]),
            meta=json.loads(str(handle["meta"])),
        )
    return library


# ---------------------------------------------------------------------------
# BOSZ 2024: deterministic URLs, so no search API and no scraping
# ---------------------------------------------------------------------------
#
# Every fact encoded below was checked against the live archive on 2026-08-27 rather than
# read from a paper, because two of them are not what the documentation would predict:
#
#   * Teff is NOT zero-padded. The token is "t6000" and "t10000", not "t06000".
#   * The atmosphere code varies across the grid: "ms" (MARCS spherical) below log g 3.5,
#     "mp" (MARCS plane-parallel) at and above it, "ap" (ATLAS9) above 8000 K, with both
#     families published in the 7500-8000 K overlap.
#
# A resampled file is two whitespace columns, flux and continuum, with no header and no
# wavelength; the wavelengths are in one shared file per resolution, in Angstrom.


def _bosz_atmosphere(teff: float, logg: float) -> str:
    """Which model family BOSZ published at this node.

    MARCS below 8000 K and ATLAS9 above it, with MARCS split by geometry at log g 3.5.
    Both exist inside the 7500-8000 K overlap; MARCS is chosen so that a library staying
    under 8000 K is one family throughout and never interpolates across a change of code.
    """
    if teff <= 8000.0:
        return "ms" if logg < 3.5 else "mp"
    return "ap"


def _bosz_filename(teff, logg, mh, *, alpha, carbon, vmicro, resolution) -> str:
    return (
        f"bosz2024_{_bosz_atmosphere(teff, logg)}_t{int(teff)}_g{logg:+.1f}_"
        f"m{mh:+.2f}_a{alpha:+.2f}_c{carbon:+.2f}_v{int(vmicro)}_"
        f"r{int(resolution)}_resam.txt.gz"
    )


def _bosz_url(teff, logg, mh, *, alpha, carbon, vmicro, resolution) -> str:
    name = _bosz_filename(
        teff, logg, mh, alpha=alpha, carbon=carbon, vmicro=vmicro, resolution=resolution
    )
    return f"{_BOSZ_ROOT}/r{int(resolution)}/m{mh:+.2f}/{name}"


def _bosz_wave_url(resolution: int) -> str:
    return f"{_BOSZ_ROOT}/wavelength_grids/bosz2024_wave_r{int(resolution)}.txt"


def _raw_dir() -> Path:
    return cache_dir() / "libraries" / "_raw" / "bosz2024"


def _fetch_shard(url: str, destination: Path) -> Path:
    """Download one file unless it is already present, atomically."""
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".part")
    _download_with_retries(url, tmp)
    tmp.replace(destination)
    return destination


def _read_bosz_shard(path: Path, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (normalized, log_continuum) over the kept pixels of one BOSZ file."""
    with gzip.open(path, "rt") as handle:
        table = np.loadtxt(handle, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] != 2:
        raise ValueError(
            f"{path.name}: expected two columns (flux, continuum), got shape {table.shape}. "
            "The upstream format may have changed; re-check before trusting the ingest."
        )
    flux, continuum = table[keep, 0], table[keep, 1]
    if not np.all(continuum > 0.0):
        raise ValueError(
            f"{path.name}: the continuum column is not everywhere positive over the "
            "requested band, so it cannot be divided out or logged."
        )
    return flux / continuum, np.log(continuum)


def ingest_bosz(
    name: str = "bosz2024-fgk-r20000",
    *,
    jobs: int = 4,
    progress: bool = True,
    keep_raw: bool = True,
) -> SpectralLibrary:
    """Build a BOSZ library from the archive, sliced to the registry's band.

    Downloads one file per node from MAST, keeps the raw shards so that a different band can
    be cut later without re-downloading, and writes the standardized ``.npz`` that
    :func:`fetch_library` reads. The URLs are deterministic, so the build is reproducible.

    The medium is measured from the assembled spectra with :func:`line_core_medium` and
    checked against the registry's declaration. A disagreement raises rather than being
    reconciled, because it means the upstream convention has changed.

    References
    ----------
    Mészáros, Sz., Bohlin, R., Allende Prieto, C., et al. 2024, A&A, 688, A171
    """
    lib = _lookup_library(name)
    if lib.source != "bosz2024":
        raise ValueError(f"{name!r} is not a BOSZ library; its source is {lib.source!r}")

    fixed = lib.fixed
    resolution = int(fixed["resolution"])
    nodes = [
        (teff, logg, mh)
        for teff in lib.axes["teff"]
        for logg in lib.axes["logg"]
        for mh in lib.axes["mh"]
    ]

    raw = _raw_dir()
    if progress:
        print(
            f"albireo: building {name!r} from {len(nodes)} BOSZ nodes.\n"
            f"  up to ~{lib.download_mb:.0f} MB of downloads into {raw}\n"
            f"  (already-downloaded shards are reused; clear_library_cache('_raw') frees them)"
        )

    wave_path = raw / f"bosz2024_wave_r{resolution}.txt"
    _fetch_shard(_bosz_wave_url(resolution), wave_path)
    wave_full = np.loadtxt(wave_path, dtype=np.float64)

    lo, hi = lib.wave_range
    keep = np.flatnonzero((wave_full >= lo) & (wave_full <= hi))
    if keep.size < 16:
        raise ValueError(
            f"the shared wavelength grid has only {keep.size} samples between {lo} and "
            f"{hi} Angstrom; the band or the resolution is wrong."
        )
    wave = wave_full[keep]

    targets = []
    for teff, logg, mh in nodes:
        url = _bosz_url(
            teff,
            logg,
            mh,
            alpha=fixed["alpha"],
            carbon=fixed["carbon"],
            vmicro=fixed["vmicro"],
            resolution=resolution,
        )
        targets.append((url, raw / Path(url).name))

    # A published grid is not always the box its axes imply: BOSZ is missing exactly one
    # model in this box, (5750 K, log g 3.0, [M/H] -0.75), while every carbon-varied version
    # of it is present, so it is a gap in the calculation rather than a naming error. The
    # node is dropped, named, and recorded in the metadata rather than raising or being
    # substituted by a neighbour.
    missing: list[int] = []
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, int(jobs))) as pool:
        futures = {pool.submit(_fetch_shard, url, path): i for i, (url, path) in enumerate(targets)}
        for future in as_completed(futures):
            index = futures[future]
            try:
                future.result()
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    raise
                missing.append(index)
            done += 1
            if progress and (done % 25 == 0 or done == len(targets)):
                print(f"  {done}/{len(targets)} shards")

    if missing:
        absent = sorted(missing)
        kept = [i for i in range(len(nodes)) if i not in set(absent)]
        if progress:
            print(f"  {len(absent)} node(s) are not published and were dropped:")
            for i in absent[:5]:
                teff, logg, mh = nodes[i]
                print(f"    Teff {teff:.0f}, log g {logg:.1f}, [M/H] {mh:+.2f}")
        nodes = [nodes[i] for i in kept]
        targets = [targets[i] for i in kept]

    normalized = np.empty((len(nodes), wave.size), dtype=np.float64)
    log_continuum = np.empty_like(normalized)
    for i, (_, path) in enumerate(targets):
        normalized[i], log_continuum[i] = _read_bosz_shard(path, keep)

    meta = {
        "grid": name,
        "source": "BOSZ 2024 (MAST HLSP)",
        "root": _BOSZ_ROOT,
        "resolution": resolution,
        "version": lib.version,
        "retrieved": _today(),
        "upstream_note": lib.upstream_note,
        "licence": lib.licence,
        "citation": lib.citation,
        "doi": lib.doi,
        "continuum_unit": "upstream BOSZ surface flux",
        "vmicro": fixed["vmicro"],
        "alpha": fixed["alpha"],
        "carbon": fixed["carbon"],
        "caveats": list(lib.caveats),
        "n_requested": len(lib.axes["teff"]) * len(lib.axes["logg"]) * len(lib.axes["mh"]),
        "n_missing": len(missing),
        "shard_sha256": {Path(p).name: _sha256(p) for _, p in targets[:1]},
    }
    if missing:
        meta["missing_note"] = (
            "One or more nodes the axes imply are not published upstream and were dropped, "
            "so the grid is not a complete box and interpolation falls back from the cubic "
            "to barycentric over a triangulation. See library_info()['known_gaps']."
        )

    library = SpectralLibrary(
        label_names=lib.label_names,
        nodes=np.asarray(nodes, dtype=np.float64),
        normalized=normalized,
        log_continuum=log_continuum,
        wave=wave,
        medium=lib.medium,
        meta=meta,
    )
    _verify_declared_medium(library, lib)

    if not keep_raw:
        for _, path in targets:
            path.unlink(missing_ok=True)
    return library


def _verify_declared_medium(library: SpectralLibrary, lib: _Library) -> None:
    """Check the registry's medium against the spectra, where the band allows it.

    A band too narrow to hold two reference lines, such as the RVS window, is not an error.
    It means only that this check cannot run, and the declaration rests on the measurement
    made in a band that could.
    """
    middle = library.normalized[library.normalized.shape[0] // 2]
    try:
        verdict = line_core_medium(library.wave, middle)
    except ValueError:
        return
    if verdict["medium"] != lib.medium:
        raise ValueError(
            f"{lib.name}: the registry declares {lib.medium!r} but the spectra measure as "
            f"{verdict['medium']!r} (ratio {verdict['ratio']:.1f}). The upstream convention "
            "has moved -- BOSZ has done this once already, vacuum in 2017 and air in 2024 "
            "under one name. Do not override this; re-check the release notes."
        )


def ingest_pollux(archive_path, name: str = "pollux-ob-smc24") -> SpectralLibrary:
    """Build the POLLUX OB library from a hand-downloaded archive.

    Not implemented. POLLUX has no stable download URL, since its collections are served
    through a form that posts to ``/download/``, so the archive cannot be fetched here, and a
    parser written against an unseen file format would be a guess. The registry entry, the
    citation and the caveats are in place; the reader follows once the archive is available.

    References
    ----------
    Palacios, A., Gebran, M., Josselin, E., et al. 2010, A&A, 516, A13
    """
    lib = _lookup_library(name)
    raise NotImplementedError(
        f"{name!r} cannot be built automatically. {lib.upstream_note}\n"
        f"Once the archive is in hand, build the library directly with SpectralLibrary("
        f"label_names={lib.label_names}, nodes=..., normalized=..., log_continuum=..., "
        f"wave=..., medium=line_core_medium(...)['medium']) and save_library() it to "
        f"{_library_cache_path(lib)}."
    )


# ---------------------------------------------------------------------------
# The public entry point
# ---------------------------------------------------------------------------


def fetch_library(
    name: str,
    *,
    wave_range: tuple[float, float] | None = None,
    progress: bool = True,
    jobs: int = 4,
) -> SpectralLibrary:
    """Load a named library, downloading and building it on first use.

    The build is cached under :func:`albireo.examples.cache_dir` and reused thereafter, so
    the cost is paid once per machine. ``$ALBIREO_DATA_DIR`` redirects the cache, which is
    also how a shared or pre-populated directory is selected on a cluster.

    Parameters
    ----------
    name
        One of :func:`library_names`.
    wave_range
        Optional sub-slice, in Angstrom, within the library's own band. Slicing narrower
        happens after loading. Widening is refused: the band is what was downloaded, and
        returning less than was asked for without saying so would be worse than an error.
    progress
        Print download and build progress. Downloads run to hundreds of megabytes, so this
        defaults on.
    jobs
        Parallel downloads during a build.

    Notes
    -----
    The cached build is checksummed on every load against the digest recorded when it was
    written, so later corruption is caught. The upstream shards are verified structurally,
    and the assembled library has its wavelength medium measured rather than trusted. A
    registry-level pin published under a DOI is not yet in place; until then two machines can
    compare ``library.meta["content_sha256"]``, a hash of the arrays themselves and therefore
    reproducible across machines, to confirm they built the same library.
    """
    lib = _lookup_library(name)
    path = _library_cache_path(lib)

    if path.is_file():
        library = load_library(path)
        recorded = library.meta.get("content_sha256")
        if recorded:
            digest = _content_digest(library)
            if digest != recorded:
                raise RuntimeError(
                    f"cached library {name!r} at {path} hashes to {digest}, but it records "
                    f"{recorded}. The file was corrupted or edited after it was written; "
                    f"clear_library_cache({name!r}) and rebuild."
                )
        return _subset(library, wave_range, lib)

    if lib.source == "bosz2024":
        library = ingest_bosz(name, jobs=jobs, progress=progress)
    elif lib.source == "pollux":
        ingest_pollux(None, name)  # raises with the manual instructions
        raise AssertionError("unreachable")  # pragma: no cover
    else:  # pragma: no cover - registry bug, not a user error
        raise RuntimeError(f"library {name!r} has an unknown source {lib.source!r}")

    library.meta["content_sha256"] = _content_digest(library)
    save_library(library, path)
    if progress:
        print(f"albireo: cached {name!r} ({path.stat().st_size / 1e6:.1f} MB) at {path}")
    # Read back what was just written rather than returning the in-memory build. Fluxes are
    # stored as float32, so the two differ in the last few digits, and a function whose
    # precision depends on whether the cache was warm is a defect.
    return _subset(load_library(path), wave_range, lib)


def _subset(
    library: SpectralLibrary, wave_range: tuple[float, float] | None, lib: _Library
) -> SpectralLibrary:
    if wave_range is None:
        return library
    lo, hi = float(wave_range[0]), float(wave_range[1])
    band_lo, band_hi = lib.wave_range
    if lo < band_lo - 1e-6 or hi > band_hi + 1e-6:
        raise ValueError(
            f"requested {lo}-{hi} Angstrom but {lib.name!r} was built over "
            f"{band_lo}-{band_hi}. Sub-slicing within the band is free; widening it means "
            "rebuilding, which is a registry change so that the band stays pinnable."
        )
    keep = np.flatnonzero((library.wave >= lo) & (library.wave <= hi))
    if keep.size < 16:
        raise ValueError(f"{lo}-{hi} Angstrom keeps only {keep.size} samples")
    meta = dict(library.meta)
    meta["wave_range"] = [lo, hi]
    return SpectralLibrary(
        label_names=library.label_names,
        nodes=library.nodes,
        normalized=library.normalized[:, keep],
        log_continuum=library.log_continuum[:, keep],
        wave=library.wave[keep],
        medium=library.medium,
        meta=meta,
    )


def _today() -> str:
    return _dt.date.today().isoformat()

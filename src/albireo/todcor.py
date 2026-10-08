"""Epoch radial velocities of every component by N-dimensional correlation (TODCOR).

**Experimental.** The names and return shapes defined here may change; the joint
:class:`~albireo.inference.MarginalOrbitModel` path infers velocities without them.

The estimator is the weighted least-squares fit of N shifted, LSF-convolved, rebinned
templates to each epoch's pixels, with the amplitudes held at declared light fractions or
profiled, and an additive low-order nuisance polynomial (``docs/math.md`` §10.1). The
epoch model is that of ``docs/math.md`` §1.4 with the component spectra given rather than
marginalized:

    y_j = 1 + sum_i a_ij R_j B_j T(delta_ij) t_i + (nuisance) + noise,   Var = 1 / w

with ``t_i`` the templates normalized to their own continua, ``T`` the shift, ``B_j`` the
instrument's line-spread function, ``R_j`` the projection onto the epoch's native pixels
(data are never resampled), and ``w`` the inverse variances with the masks applied. For
every set of shifts the amplitudes are either held or solved in closed form. The
chi-square is minimized over the integer shifts of the template grid and then refined
below a pixel. Where the lines of two components overlap the surface has a second minimum,
at the pair exchanged about its light-weighted mean velocity, and the search is made from
both and returns the lower (``docs/math.md`` §10.3). An epoch at which another minimum fits
as well and gives a different pair of velocities is flagged ``blended``.

On a uniform grid with uniform weights and free amplitudes the chi-square surface is
identical to the two-dimensional correlation of Zucker & Mazeh (1994),
``R^2 = 1 - chi^2 / |z|^2`` with ``z = y - 1``, and with a fixed light ratio to their
``R(s_1, s_2; alpha)``. The test suite checks both identities to 1e-10 (``docs/math.md``
§10.2). The three- and four-component extensions (Zucker, Torres & Mazeh 1995; Torres,
Latham & Stefanik 2007) are the same block solve with more templates. The uncertainties
are the maximum-likelihood curvature errors of Zucker (2003), rescaled by the reduced
chi-square (§10.4). The least-squares form handles masks, chip gaps, cosmic rays,
per-pixel weights, mixed instruments and mixed samplings without change to the formulae.
It applies each instrument's LSF to intrinsic templates in quadrature above their own
resolution and evaluates the chi-square exactly at fractional shifts (§10.3). The residual
pixel-locking error of the linear shift operator is of order ``0.1 / sigma_px^2`` pixels.
It is below 0.01 px when the template grid samples the narrowest LSF with three or more
pixels per sigma.

Templates come from :meth:`albireo.Fit.templates` (disentangled components),
:meth:`Template.from_library` (a synthetic grid rendered at given labels) or
:meth:`Template.from_labels` (the model spectrum of a label match).
:meth:`albireo.Fit.measure_velocities` measures a fit's epochs against its own components,
and :func:`todcor_batch` measures many stars in one call. Velocities are reported
barycentric (§10.5). A disentangled component has an unidentified zero point
(``docs/math.md`` §5.3, §7.6), so velocities measured against it are differential, with one
arbitrary constant per component, unless a label match (:mod:`albireo.match`) has
determined the offset. ``VelocityTable.absolute`` records the status of each component. A
per-epoch table discards the phase coherence by which disentangling separates components
whose lines are never resolved, and its accuracy is bounded by the agreement between
templates and stars. Where the components are unknown, disentangling comes first.

References
----------
Zucker, S. & Mazeh, T. 1994, ApJ, 420, 806
Zucker, S., Torres, G. & Mazeh, T. 1995, ApJ, 452, 863
Torres, G., Latham, D. W. & Stefanik, R. P. 2007, ApJ, 662, 602
Zucker, S. 2003, MNRAS, 342, 1291
"""

from __future__ import annotations

import itertools
import math
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from albireo.data import Dataset, EpochData
from albireo.forward import _is_per_epoch
from albireo.grids import C_KMS, LogGrid, _in_epoch_blocks, log_doppler_shift
from albireo.operators import (
    _gaussian_kernel_numpy,
    _rotational_kernel_numpy,
    convolve_spectrum,
    convolve_varying,
    gaussian_kernel,
    gaussian_lsf_profiles,
    rebin_operator,
)

__all__ = [
    "Template",
    "TodcorBatch",
    "TodcorSurface",
    "VelocityTable",
    "todcor",
    "todcor_batch",
    "todcor_surface",
]

_ROW_BUCKET = 1024
_SHIFT_CHUNK = 32
_EPS = 1e-12
# The largest number of local minima refined per epoch: of the coarse surface at full
# resolution, and of each fine window below a pixel (see `_candidate_minima`). A fine
# window is moved at most `_WINDOW_MOVES` times while its minimum is on its edge.
_COARSE_STARTS = 6
_FINE_STARTS = 4
_WINDOW_MOVES = 6
# The blend flag (`VelocityTable.blended`) is raised by another minimum of the surface within
# `_BLEND_CHI2` of the one returned, in units of the reduced chi-square under
# ``errors="profiled"``, whose velocities differ from those returned by more than
# `_BLEND_SIGMAS` quoted errors in every order of the components (`_margin`).
_BLEND_CHI2 = 9.0
_BLEND_SIGMAS = 3.0


class _LightNotMeasured(ValueError):
    """The free pass of ``light="global"`` gave no epoch with positive amplitudes."""


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Template:
    """One component's spectrum, in the form the epochs are correlated against.

    Parameters
    ----------
    name
        The component's name; it labels every column of the velocity table.
    grid
        The :class:`~albireo.grids.LogGrid` the template is sampled on. Every template
        passed to one :func:`todcor` call must share it, and it must extend beyond the data
        by the velocity range searched (:meth:`LogGrid.covering` builds such a grid).
    deviation
        ``flux - 1`` on ``grid``, normalized to the template's own continuum, so that the
        amplitude assigned by the fit is a light fraction.
    sigma_kms
        Gaussian broadening already present in the template, as a sigma in km/s: zero for
        an intrinsic (deconvolved, or synthetic at infinite resolution) spectrum, the
        library's own resolution for a synthetic grid. The instrument LSF is applied in
        quadrature above it, so a template rendered at R = 20,000 is not broadened twice.
    v_zero_kms
        Velocity of the template's rest frame relative to the star's true rest frame, when
        known, so that velocities measured against it can be reported as absolute. A
        synthetic spectrum is at zero. A disentangled component is at an unknown offset,
        because its zero point is unidentified (``docs/math.md`` §5.3). ``None`` declares
        the offset unknown, and every table built from the template records that.
    meta
        Free-form provenance (library name, labels, the fit it came from).
    """

    name: str
    grid: LogGrid
    deviation: np.ndarray
    sigma_kms: float = 0.0
    v_zero_kms: float | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        deviation = np.asarray(self.deviation, dtype=np.float64)
        if deviation.shape != (self.grid.n,):
            raise ValueError(
                f"template {self.name!r}: deviation has shape {deviation.shape} but the grid "
                f"has {self.grid.n} pixels"
            )
        if not np.all(np.isfinite(deviation)):
            raise ValueError(f"template {self.name!r}: deviation must be finite everywhere")
        if not float(self.sigma_kms) >= 0.0:
            raise ValueError(f"template {self.name!r}: sigma_kms must be non-negative")
        if self.v_zero_kms is not None and not math.isfinite(float(self.v_zero_kms)):
            raise ValueError(f"template {self.name!r}: v_zero_kms must be finite or None")
        object.__setattr__(self, "deviation", deviation)
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "sigma_kms", float(self.sigma_kms))
        object.__setattr__(
            self, "v_zero_kms", None if self.v_zero_kms is None else float(self.v_zero_kms)
        )

    @property
    def flux(self) -> np.ndarray:
        """The normalized spectrum ``1 + deviation``."""
        return 1.0 + self.deviation

    @property
    def absolute(self) -> bool:
        """Whether velocities measured against this template have an absolute zero point."""
        return self.v_zero_kms is not None

    @classmethod
    def from_flux(cls, name: str, grid: LogGrid, flux, **kwargs) -> Template:
        """Build from a normalized flux array on ``grid`` (``deviation = flux - 1``)."""
        return cls(
            name=name, grid=grid, deviation=np.asarray(flux, dtype=np.float64) - 1.0, **kwargs
        )

    @classmethod
    def from_library(
        cls,
        name: str,
        library,
        labels: Mapping[str, float],
        *,
        grid: LogGrid,
        medium: str,
        vsini_kms: float = 0.0,
        macro_kms: float = 0.0,
        epsilon: float = 0.6,
        resolving_power: float | None = None,
        method: str = "auto",
        v_zero_kms: float | None = 0.0,
    ) -> Template:
        """Render a synthetic template from a :class:`~albireo.library.SpectralLibrary`.

        The grid is interpolated at ``labels`` (``docs/math.md`` §9.3), the deviation is
        rotationally broadened by ``vsini_kms`` with the pixel-integrated Gray kernel, and
        any fixed macroturbulence is applied as a Gaussian. The result is placed at rest
        (``v_zero_kms = 0``), i.e. it is an absolute template.

        Parameters
        ----------
        name
            Component name.
        library
            The grid, e.g. from :func:`albireo.fetch_library`. It is resampled onto
            ``grid`` in the data's ``medium`` first, so that the wavelength scale is
            handled explicitly (air and vacuum wavelengths differ by about 83 km/s).
        labels
            ``{"teff": ..., "logg": ..., "mh": ...}``: whichever axes the library has, in
            any order. The fitted values in :attr:`albireo.LabelMatch.labels` can be
            passed directly.
        grid, medium
            The template grid and the wavelength scale of the data.
        vsini_kms, macro_kms, epsilon
            Rotational broadening [km/s], Gaussian macroturbulence [km/s] and the
            limb-darkening coefficient.
        resolving_power
            The library's own resolving power (``R = lambda / FWHM``), recorded as
            :attr:`sigma_kms` so that the instrument LSF is applied in quadrature above
            it. ``None`` declares an intrinsic-resolution grid.
        method
            Interpolation method, as :func:`albireo.library_interpolator`.
        v_zero_kms
            Rest-frame velocity of the rendered template. The default is zero because a
            synthetic spectrum is at rest. ``None`` declares it unknown.
        """
        from albireo.library import library_interpolator

        resampled = library.resampled_to(grid, medium=medium)
        interpolator = library_interpolator(resampled, method=method)
        missing = [axis for axis in resampled.label_names if axis not in labels]
        if missing:
            raise ValueError(
                f"template {name!r}: labels are missing the library axes {missing} "
                f"(the library has {list(resampled.label_names)})"
            )
        point = jnp.asarray([float(labels[axis]) for axis in resampled.label_names])
        normalized, _ = interpolator(point)
        deviation = np.asarray(normalized, dtype=np.float64) - 1.0
        if vsini_kms < 0.0 or macro_kms < 0.0:
            raise ValueError(f"template {name!r}: vsini_kms and macro_kms must be non-negative")
        if vsini_kms > 0.0:
            kernel = _rotational_kernel_numpy(vsini_kms / grid.dv_kms, epsilon=epsilon)
            deviation = np.convolve(deviation, kernel, mode="same")
        if macro_kms > 0.0:
            kernel = _gaussian_kernel_numpy(macro_kms / grid.dv_kms)
            deviation = np.convolve(deviation, kernel, mode="same")
        sigma = 0.0
        if resolving_power is not None:
            if not resolving_power > 0.0:
                raise ValueError(f"template {name!r}: resolving_power must be positive")
            sigma = C_KMS / (float(resolving_power) * 2.0 * math.sqrt(2.0 * math.log(2.0)))
        meta = {
            "source": "library",
            "library": dict(getattr(library, "meta", {})).get("name", "unnamed"),
            "labels": {axis: float(labels[axis]) for axis in resampled.label_names},
            "vsini_kms": float(vsini_kms),
            "macro_kms": float(macro_kms),
            "medium": medium,
        }
        return cls(
            name=name,
            grid=grid,
            deviation=deviation,
            sigma_kms=sigma,
            v_zero_kms=v_zero_kms,
            meta=meta,
        )

    @classmethod
    def from_labels(cls, match, name: str) -> Template:
        """The MAP model spectrum of one component of a :class:`~albireo.LabelMatch`.

        The spectrum is rendered as the label fit rendered it: interpolated, rotationally
        broadened, and shifted by the fitted ``v_kms`` so that it is in the disentangled
        component's frame, with that shift recorded as :attr:`v_zero_kms`. Velocities
        measured against it are therefore absolute, since the label fit measures the zero
        point of the disentangled frame (``docs/math.md`` §9).

        The Gaussian width the spectrum already has is recorded in :attr:`sigma_kms`, so
        that :func:`todcor` applies only the rest of the instrument profile, as for
        :meth:`from_library`. A library that declares its resolving power
        (:attr:`albireo.SpectralLibrary.resolving_power`) has
        ``sigma_lib = c / (R 2 sqrt(2 ln 2))``, the width a ``"native"`` or ``"epochs"``
        match renders. A ``"matched"`` match convolves the template further with
        ``sqrt(sigma_inst^2 - sigma_lib^2)`` (the whole declared width for an intrinsic
        library), so its template has the declared instrument width, and that is recorded.
        """
        from albireo.match import _quadrature_width

        if name not in match.names:
            raise ValueError(f"unknown component {name!r}; the match has {list(match.names)}")
        index = list(match.names).index(name)
        labels = match.labels[name]
        grid = LogGrid(
            x0=float(np.log(match.wave[0])),
            dx=float(match.problem.dx),
            n=int(match.wave.size),
            relativistic=bool(match.problem.relativistic),
        )
        deviation = np.asarray(match.template(name), dtype=np.float64) - 1.0
        resolving_power = getattr(match.libraries[index], "resolving_power", None)
        library_sigma = (
            0.0
            if resolving_power is None
            else C_KMS / (float(resolving_power) * 2.0 * math.sqrt(2.0 * math.log(2.0)))
        )
        sigma = library_sigma
        if bool(match.problem.matched):
            declared = float(match.assumptions["lsf_sigma_kms"])
            applied = float(_quadrature_width(declared, resolving_power, f"star {name!r}")[0])
            sigma = math.sqrt(library_sigma**2 + applied**2)
        return cls(
            name=name,
            grid=grid,
            deviation=deviation,
            sigma_kms=sigma,
            v_zero_kms=float(labels["v_kms"]),
            meta={
                "source": "label match",
                "labels": dict(labels),
                "compare": str(match.assumptions.get("compare", "native")),
                "library_resolving_power": resolving_power,
            },
        )


def _templates_share_grid(templates: Sequence[Template]) -> LogGrid:
    if len(templates) == 0:
        raise ValueError("todcor needs at least one template")
    grid = templates[0].grid
    for t in templates[1:]:
        same = (
            math.isclose(t.grid.x0, grid.x0, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(t.grid.dx, grid.dx, rel_tol=1e-12, abs_tol=0.0)
            and t.grid.n == grid.n
            and t.grid.relativistic == grid.relativistic
        )
        if not same:
            raise ValueError(
                f"templates {templates[0].name!r} and {t.name!r} are on different grids; "
                "resample them onto one LogGrid first"
            )
    names = [t.name for t in templates]
    if len(set(names)) != len(names):
        raise ValueError(f"template names must be distinct; got {names}")
    return grid


# ---------------------------------------------------------------------------
# The per-epoch terms: every inner product the chi-square surface needs
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("chunk",))
def _epoch_terms(t_stack, rows, cols, vals, z, w, deltas, basis, chunk: int):
    """All inner products between the data and the shifted, projected templates.

    ``t_stack`` holds the templates (already convolved with this instrument's LSF) on the
    model grid, ``(rows, cols, vals)`` is the epoch's rebin operator, ``deltas`` is an
    ``(n_tmpl, n_shift)`` table of integer shifts (one row per template), and ``basis``
    the additive nuisance basis on the native pixels. Returns the projections ``b``, the
    full Gram tensor ``G`` (including each template against itself at every pair of
    shifts, which makes the chi-square exact at fractional shifts), the
    template-nuisance cross terms, and the data scalars (``docs/math.md`` §10.1).

    A shifted template projected onto the native grid is one gather of the template at
    ``cols - delta`` weighted by the rebin values, so all shifts are built by one
    segment-sum per chunk of shifts. The chunking bounds the ``nnz x n_shift`` temporary.
    """
    n_out = z.shape[0]
    n_grid = t_stack.shape[1]

    def columns(t, d_chunk):
        idx = cols[:, None] - d_chunk[None, :]
        ok = (idx >= 0) & (idx < n_grid)
        contrib = vals[:, None] * jnp.where(ok, t[jnp.clip(idx, 0, n_grid - 1)], 0.0)
        return jax.ops.segment_sum(contrib, rows, num_segments=n_out)

    def all_columns(t, d):
        pieces = jax.lax.map(lambda dc: columns(t, dc), d.reshape(-1, chunk))
        return jnp.transpose(pieces, (1, 0, 2)).reshape(n_out, -1)

    a = jax.vmap(all_columns)(t_stack, deltas)  # (n_tmpl, n_out, n_shift)
    wz = w * z
    b = jnp.einsum("ins,n->is", a, wz)
    wa = a * w[None, :, None]
    gram = jnp.einsum("ins,knt->ikst", a, wa)
    pwa = jnp.einsum("nm,ins->ims", basis, wa)
    pwp = basis.T @ (w[:, None] * basis)
    pwz = basis.T @ wz
    zwz = wz @ z
    return b, gram, pwa, pwp, pwz, zwz


def _bcast(x, axis: int, ndim: int):
    shape = [1] * ndim
    shape[axis] = -1
    return jnp.reshape(x, shape)


def _bcast_pair(g, i: int, k: int, ndim: int):
    shape = [1] * ndim
    shape[i] = g.shape[0]
    shape[k] = g.shape[1]
    return jnp.reshape(g, shape)


@partial(jax.jit, static_argnames=("free_scale",))
def _chi2_grid_fixed(b, gram, pwa, pwp, pwz, zwz, amps, free_scale: bool = False):
    """Chi-square over the full integer-shift grid with the amplitudes held at ``amps``.

    With ``free_scale`` the amplitudes are ``a * amps`` with one overall ``a`` solved per
    point: the original TODCOR with a known light ratio, whose correlation is invariant to
    the composite's scale (``docs/math.md`` §10.2).
    """
    n_tmpl = b.shape[0]
    m = pwp.shape[0]
    # B = sum_i l_i b_i, Q = l.G.l, and the nuisance cross term PB = sum_i l_i PWA_i.
    lin = 0.0
    quad = 0.0
    for i in range(n_tmpl):
        lin = lin + amps[i] * _bcast(b[i], i, n_tmpl)
        quad = quad + amps[i] ** 2 * _bcast(jnp.diagonal(gram[i, i]), i, n_tmpl)
        for k in range(i + 1, n_tmpl):
            quad = quad + 2.0 * amps[i] * amps[k] * _bcast_pair(gram[i, k], i, k, n_tmpl)
    if m > 0:
        inv = jnp.linalg.inv(pwp)
        pb = 0.0
        for i in range(n_tmpl):
            pb = pb + amps[i] * jnp.reshape(pwa[i], (m, *_bcast(b[i], i, n_tmpl).shape))
        pwz_b = jnp.reshape(pwz, (m,) + (1,) * n_tmpl)
        if free_scale:
            # Schur complement: profile the nuisance, then the scale.
            null = zwz - pwz @ inv @ pwz
            lin_eff = lin - jnp.einsum("m...,mn,n->...", pb, inv, pwz)
            quad_eff = quad - jnp.einsum("m...,mn,n...->...", pb, inv, pb)
            return null - lin_eff**2 / jnp.maximum(quad_eff, _EPS)
        resid = pwz_b - pb
        chi2 = zwz - 2.0 * lin + quad
        return chi2 - jnp.einsum("m...,mn,n...->...", resid, inv, resid)
    if free_scale:
        return zwz - lin**2 / jnp.maximum(quad, _EPS)
    return zwz - 2.0 * lin + quad


@jax.jit
def _chi2_grid_free(b, gram, pwa, pwp, pwz, zwz):
    """Chi-square over the full integer-shift grid with the amplitudes solved per point.

    Mapped over the first template's shifts so the batched ``(n_tmpl + m)`` linear solves
    never materialize more than one slab of the grid at a time.
    """
    n_tmpl, n_shift = b.shape
    m = pwp.shape[0]
    size = n_tmpl + m
    inner = jnp.meshgrid(*[jnp.arange(n_shift)] * (n_tmpl - 1), indexing="ij")
    inner = [x.reshape(-1) for x in inner]
    n_points = inner[0].shape[0] if inner else 1

    def slab(s0):
        idx = [jnp.full((n_points,), s0, dtype=jnp.int32)] + [x.astype(jnp.int32) for x in inner]
        mat = jnp.zeros((n_points, size, size))
        rhs = jnp.zeros((n_points, size))
        for i in range(n_tmpl):
            rhs = rhs.at[:, i].set(b[i][idx[i]])
            for k in range(n_tmpl):
                mat = mat.at[:, i, k].set(gram[i, k][idx[i], idx[k]])
            for j in range(m):
                mat = mat.at[:, i, n_tmpl + j].set(pwa[i, j][idx[i]])
                mat = mat.at[:, n_tmpl + j, i].set(pwa[i, j][idx[i]])
        for j in range(m):
            rhs = rhs.at[:, n_tmpl + j].set(pwz[j])
            for jj in range(m):
                mat = mat.at[:, n_tmpl + j, n_tmpl + jj].set(pwp[j, jj])
        sol = jnp.linalg.solve(mat, rhs[..., None])[..., 0]
        return zwz - jnp.sum(rhs * sol, axis=-1)

    out = jax.lax.map(slab, jnp.arange(n_shift))
    return out.reshape((n_shift,) * n_tmpl)


# ---------------------------------------------------------------------------
# Exact evaluation at fractional shifts, and the refinement
# ---------------------------------------------------------------------------


@dataclass
class _Terms:
    """NumPy copies of the fine-window terms, indexed by window position."""

    b: np.ndarray  # (n_tmpl, S)
    gram: np.ndarray  # (n_tmpl, n_tmpl, S, S)
    pwa: np.ndarray  # (n_tmpl, m, S)
    pwp: np.ndarray  # (m, m)
    pwz: np.ndarray  # (m,)
    zwz: float

    @property
    def n_tmpl(self) -> int:
        return int(self.b.shape[0])

    @property
    def n_shift(self) -> int:
        return int(self.b.shape[1])

    @property
    def m(self) -> int:
        return int(self.pwp.shape[0])

    def at(self, pos: np.ndarray):
        """The terms at fractional window positions, by the linear-interpolation identity.

        A template shifted by ``n + f`` is ``(1 - f)`` times the template shifted by ``n``
        plus ``f`` times the template shifted by ``n + 1``, because the shift operator is
        linear in the template. Every inner product at a fractional shift is then a bilinear
        combination of the integer-shift ones. The chi-square is therefore exact at any
        fractional position inside the window, for the same shift operator the forward
        model uses (``docs/math.md`` §10.3).
        """
        pos = np.asarray(pos, dtype=np.float64)
        n = np.clip(np.floor(pos).astype(int), 0, self.n_shift - 2)
        f = pos - n
        u = np.stack([1.0 - f, f], axis=1)  # (n_tmpl, 2)
        n_tmpl = self.n_tmpl
        b = np.array([u[i] @ self.b[i, n[i] : n[i] + 2] for i in range(n_tmpl)])
        gram = np.empty((n_tmpl, n_tmpl))
        for i in range(n_tmpl):
            for k in range(n_tmpl):
                block = self.gram[i, k, n[i] : n[i] + 2, n[k] : n[k] + 2]
                gram[i, k] = u[i] @ block @ u[k]
        pwa = np.array([self.pwa[i][:, n[i] : n[i] + 2] @ u[i] for i in range(n_tmpl)])
        return b, gram, pwa

    def chi2(self, pos, amps=None, free_scale=False):
        """Chi-square at fractional positions; amplitudes fixed, scaled, or solved."""
        b, gram, pwa = self.at(pos)
        return _chi2_from_terms(b, gram, pwa, self.pwp, self.pwz, self.zwz, amps, free_scale)

    def null_chi2(self) -> float:
        """Chi-square with no template, i.e. with the nuisance alone."""
        if self.m == 0:
            return float(self.zwz)
        return float(self.zwz - self.pwz @ np.linalg.solve(self.pwp, self.pwz))


def _chi2_from_terms(b, gram, pwa, pwp, pwz, zwz, amps=None, free_scale=False):
    """Chi-square from the terms at one point. Returns ``(chi2, amplitudes, nuisance)``.

    ``amps=None`` solves every amplitude, and ``free_scale`` solves one overall scale on the
    given ``amps``. Otherwise the amplitudes are held at ``amps``.
    """
    n_tmpl = b.shape[0]
    m = pwp.shape[0]
    if free_scale:
        amps = np.asarray(amps, dtype=np.float64)
        col_b = float(amps @ b)
        col_q = float(amps @ gram @ amps)
        if m:
            inv = np.linalg.inv(pwp)
            pb = pwa.T @ amps
            null = zwz - pwz @ inv @ pwz
            lin = col_b - pb @ inv @ pwz
            quad = col_q - pb @ inv @ pb
            scale = lin / max(quad, _EPS)
            nuisance = inv @ (pwz - scale * pb)
            return float(null - lin * scale), scale * amps, nuisance
        scale = col_b / max(col_q, _EPS)
        return float(zwz - col_b * scale), scale * amps, np.zeros(0)
    if amps is None:
        size = n_tmpl + m
        mat = np.zeros((size, size))
        rhs = np.zeros(size)
        mat[:n_tmpl, :n_tmpl] = gram
        rhs[:n_tmpl] = b
        if m:
            mat[:n_tmpl, n_tmpl:] = pwa
            mat[n_tmpl:, :n_tmpl] = pwa.T
            mat[n_tmpl:, n_tmpl:] = pwp
            rhs[n_tmpl:] = pwz
        try:
            sol = np.linalg.solve(mat, rhs)
        except np.linalg.LinAlgError:
            # Two templates that coincide at this shift (the same spectrum for both
            # components, as in Gaia's own screening, or twins) make the Gram matrix
            # singular. The chi-square of the best fit in the column span is still
            # defined. The minimum-norm amplitudes are one of the equivalent solutions.
            sol = np.linalg.lstsq(mat, rhs, rcond=None)[0]
        return float(zwz - rhs @ sol), sol[:n_tmpl], sol[n_tmpl:]
    amps = np.asarray(amps, dtype=np.float64)
    chi2 = zwz - 2.0 * amps @ b + amps @ gram @ amps
    nuisance = np.zeros(m)
    if m:
        resid = pwz - pwa.T @ amps
        nuisance = np.linalg.solve(pwp, resid)
        chi2 = chi2 - resid @ nuisance
    return float(chi2), amps, nuisance


def _quadratic_fit(points: np.ndarray, values: np.ndarray):
    """Fit ``c + g.x + x.H.x / 2`` to ``values`` at ``points``; returns ``(c, g, H)``."""
    n_dim = points.shape[1]
    columns = [np.ones(points.shape[0])]
    columns += [points[:, i] for i in range(n_dim)]
    pairs = [(i, k) for i in range(n_dim) for k in range(i, n_dim)]
    columns += [points[:, i] * points[:, k] for i, k in pairs]
    design = np.stack(columns, axis=1)
    coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    c = coef[0]
    g = coef[1 : 1 + n_dim]
    hess = np.zeros((n_dim, n_dim))
    for (i, k), value in zip(pairs, coef[1 + n_dim :], strict=True):
        if i == k:
            hess[i, i] = 2.0 * value
        else:
            hess[i, k] = hess[k, i] = value
    return c, g, hess


def _stencil(center: np.ndarray, lower: np.ndarray, upper: np.ndarray, h: float) -> np.ndarray:
    """A ``3^N`` stencil of half-width ``h`` around ``center``, kept inside ``[lower, upper]``."""
    axes = []
    for c, lo, hi in zip(center, lower, upper, strict=True):
        offsets = np.array([-h, 0.0, h])
        if c - h < lo:
            offsets = np.array([0.0, h, 2.0 * h]) + (lo - c)
        elif c + h > hi:
            offsets = np.array([-2.0 * h, -h, 0.0]) + (hi - c)
        axes.append(c + offsets)
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([x.reshape(-1) for x in mesh], axis=1)


def _minimize_box_quadratic(g: np.ndarray, hess: np.ndarray) -> tuple[np.ndarray, float]:
    """Minimize the convex ``g.x + x.H.x/2`` over the unit box exactly; ``(x, value)``.

    The stationary point is the minimum where it lies inside the box. Otherwise the
    minimum is on the boundary, and it is the lowest of the minima over the ``2 N``
    facets, each a problem of the same form in one dimension fewer. A singular ``H`` has
    a direction along which the quadratic is linear or flat, so its minimum is also on the
    boundary. A coordinate descent is not used. It converges at the rate of the squared
    correlation of ``H`` per sweep, which is above 0.99 at a blended epoch.
    """
    n = g.size
    if n == 1:
        g0, h0 = float(g[0]), float(hess[0, 0])
        x = min(max(-g0 / h0, 0.0), 1.0) if h0 > _EPS else (0.0 if g0 > 0.0 else 1.0)
        return np.array([x]), g0 * x + 0.5 * h0 * x * x
    scale = float(np.prod(np.clip(np.diag(hess), 0.0, None)))
    if scale > 0.0 and np.linalg.det(hess) > 1e-12 * scale:
        x = np.linalg.solve(hess, -g)
        if np.all((x >= 0.0) & (x <= 1.0)):
            return x, float(g @ x + 0.5 * x @ hess @ x)
    best_x, best_value = None, np.inf
    keep = np.arange(n)
    for axis in range(n):
        rest = keep[keep != axis]
        for bound in (0.0, 1.0):
            # On the facet x[axis] = bound the gradient gains H[rest, axis] * bound.
            sub_x, sub_value = _minimize_box_quadratic(
                g[rest] + hess[rest, axis] * bound, hess[np.ix_(rest, rest)]
            )
            value = sub_value + g[axis] * bound + 0.5 * hess[axis, axis] * bound * bound
            if value < best_value:
                best_x = np.empty(n)
                best_x[rest] = sub_x
                best_x[axis] = bound
                best_value = float(value)
    return best_x, best_value


def _cell_quadratic(terms: _Terms, corner: np.ndarray, amps: np.ndarray):
    """The chi-square inside one unit cell of the fine window as ``c + g.f + f.H.f / 2``.

    With the amplitudes held, every inner product is bilinear in the fractional shifts
    ``f`` of the cell whose lowest corner is ``corner`` (:meth:`_Terms.at`), so the
    chi-square with the nuisance profiled is exactly a quadratic in them
    (``docs/math.md`` §10.3). Its coefficients are differences of the integer-shift terms
    at the corners of the cell. ``H`` is positive semi-definite, since the chi-square is
    the squared norm of a residual that is affine in ``f``.
    """
    n_tmpl = terms.n_tmpl
    lo = np.asarray(corner, dtype=int)
    rows = np.arange(n_tmpl)
    b_lo = terms.b[rows, lo]
    b_step = terms.b[rows, lo + 1] - b_lo
    gram_lo = np.empty((n_tmpl, n_tmpl))
    gram_step = np.empty((n_tmpl, n_tmpl))  # template i moved one pixel, k at the corner
    gram_both = np.empty((n_tmpl, n_tmpl))
    for i in range(n_tmpl):
        for k in range(n_tmpl):
            block = terms.gram[i, k, lo[i] : lo[i] + 2, lo[k] : lo[k] + 2]
            gram_lo[i, k] = block[0, 0]
            gram_step[i, k] = block[1, 0] - block[0, 0]
            gram_both[i, k] = block[1, 1] - block[1, 0] - block[0, 1] + block[0, 0]
    const = terms.zwz - 2.0 * amps @ b_lo + amps @ gram_lo @ amps
    grad = 2.0 * amps * (gram_step @ amps - b_step)
    hess = 2.0 * np.outer(amps, amps) * gram_both
    if terms.m:
        pwa_lo = terms.pwa[rows, :, lo]  # (n_tmpl, m)
        pwa_step = terms.pwa[rows, :, lo + 1] - pwa_lo
        resid = terms.pwz - amps @ pwa_lo
        solved = np.linalg.solve(terms.pwp, np.column_stack([resid, pwa_step.T]))
        const = const - resid @ solved[:, 0]
        grad = grad + 2.0 * amps * (pwa_step @ solved[:, 0])
        hess = hess - 2.0 * np.outer(amps, amps) * (pwa_step @ solved[:, 1:])
    return float(const), grad, 0.5 * (hess + hess.T)


def _refine_cell(terms: _Terms, corner: np.ndarray, amps):
    """Exact minimum of the chi-square over one unit cell of the fine window.

    The chi-square with the amplitudes held is a quadratic in the fractional shifts inside
    the cell (:func:`_cell_quadratic`), and its minimum over the cell is found in closed
    form (:func:`_minimize_box_quadratic`). Returns ``(chi2, position)``.
    """
    const, grad, hess = _cell_quadratic(terms, corner, np.asarray(amps, dtype=np.float64))
    x, value = _minimize_box_quadratic(grad, hess)
    return const + value, corner + x


def _profiled(terms: _Terms, pos, amps, mode: str):
    """The (profiled) chi-square and the amplitudes it used, in the given ``mode``."""
    if mode == "free":
        return terms.chi2(pos, None)
    if mode == "scale":
        return terms.chi2(pos, amps, free_scale=True)
    return terms.chi2(pos, amps)


def _refine(terms: _Terms, start: np.ndarray, amps, mode: str):
    """Sub-pixel minimum reached from the integer point ``start`` of the fine window.

    Every unit cell touching the position is minimized exactly with the amplitudes held,
    and the position moves to the lowest of them: all ``2^N`` cells at an integer point,
    the two on either side of a cell face, and the one cell that contains an interior
    point. The step is repeated until no cell touching the position is lower, so that the
    descent crosses cell faces and follows the surface for more than a pixel. At a blended
    epoch the surface is a valley along which the light-weighted mean velocity is constant,
    and its lowest integer sample is the one nearest its floor, which can be several pixels
    from its minimum. When the amplitudes are profiled (``mode`` ``"free"`` or
    ``"scale"``) the amplitude solve alternates with the step until the position
    converges.

    The descent stops at the boundary of the window, where the caller moves the window
    (:func:`_refine_at`). Falls back to ``start``, flagged as unrefined, if ``start`` is
    itself on that boundary.
    """
    hi = terms.n_shift - 1
    pos = start.astype(np.float64)
    if np.any(start <= 0) or np.any(start >= hi):
        chi2, a, _ = _profiled(terms, pos, amps, mode)
        return chi2, pos, a, False
    value, current, _ = _profiled(terms, pos, amps, mode)
    cells: dict[tuple[int, ...], tuple[float, np.ndarray]] = {}
    for _ in range(max(12, 4 * terms.n_shift)):
        base = np.floor(pos).astype(int)
        on_face = pos == base
        axes = [
            sorted({int(np.clip(c - 1, 0, hi - 1)), int(np.clip(c, 0, hi - 1))}) if face else [c]
            for c, face in zip(base.tolist(), on_face.tolist(), strict=True)
        ]
        best = None
        for corner in itertools.product(*axes):
            if corner not in cells:
                cells[corner] = _refine_cell(terms, np.array(corner), current)
            if best is None or cells[corner][0] < best[0]:
                best = cells[corner]
        if mode == "fixed":
            if not best[0] < value or np.array_equal(best[1], pos):
                break  # no cell touching the position is lower: a minimum of the surface
            value, pos = best
            continue
        _, new_amps, _ = _profiled(terms, best[1], amps, mode)
        settled = np.max(np.abs(best[1] - pos)) < 1e-6
        pos, current = best[1], new_amps
        cells.clear()  # the cell quadratics hold the amplitudes
        if settled:
            break
    chi2, current, _ = _profiled(terms, pos, amps, mode)
    return chi2, pos, current, True


def _candidate_minima(
    surface: np.ndarray, n_max: int, *, interior: bool = False, slack: float = 0.0
):
    """The local minima of a sampled chi-square surface that can contain its lowest minimum.

    The surface of two components has two minima of comparable depth wherever their lines
    overlap: the solution, and the one with the components exchanged about their
    light-weighted mean velocity, which reproduces the first two moments of the blended
    profile. A sampled surface does not show which is the lower. A basin sampled at a
    distance ``d`` from its minimum is ``d.H.d / 2`` above it, with ``H`` the curvature,
    and at a S/N of 100 that exceeds the difference in depth of the two basins on the
    integer shifts as well as on the coarse stride. Refining the lowest sample alone
    therefore returns either minimum.

    Every local minimum that can be the lowest is returned, the lowest sample first. A
    point is a local minimum where it is finite and no neighbour within one step along
    every axis is lower. Its basin can be lower than its sample by at most the largest
    value of ``d.H.d / 2`` over half a step along every axis, which for a quadratic basin is
    bounded by a quarter of the sum over the axes of the second differences
    ``f(+1) + f(-1) - 2 f(0)``. A minimum is kept where its sample minus that bound does
    not exceed the lowest sample, and at most ``n_max`` are kept. A neighbour outside the
    surface is replaced by the one opposite. With ``interior`` a minimum on the boundary
    of the surface is left out unless it is the lowest sample.

    With ``slack`` the local minima whose sample minus the bound exceeds the lowest
    sample by at most ``slack`` are appended, up to ``n_max`` more. They cannot contain the
    lowest minimum. They are the ones that can lie within ``slack`` of it, which the blend
    flag needs (:func:`_margin`). The first ``n_max`` entries do not depend on ``slack``.

    Returns the indices and, for each, its floor: the sample minus the bound, below which
    the minimum of a quadratic basin cannot lie. A caller that has already refined a
    minimum below the floor of a candidate need not refine that candidate.

    On 360 simulated Gaia RVS epochs of a pair with light fractions 0.625 and 0.375 and
    separations of 0 to 60 km/s, refining the lowest coarse sample alone, within the cells
    touching the lowest integer shift, returned a minimum above the lowest at 43, 70, 73
    and 90 epochs for a S/N of 15, 40, 100 and 300, by up to 8.5, 34, 166 and 1421 in
    chi-square (``scripts/todcor_blend_bench.py``, ``docs/benchmarks.md``).
    """
    clean = np.where(np.isfinite(surface), surface, np.inf)
    shape, n_dim = clean.shape, clean.ndim
    padded = np.pad(clean, 1, constant_values=np.inf)

    def neighbour(offset):
        return padded[tuple(slice(1 + o, 1 + o + n) for o, n in zip(offset, shape, strict=True))]

    is_minimum = np.isfinite(clean)
    for offset in np.ndindex(*(3,) * n_dim):
        offset = tuple(o - 1 for o in offset)
        if any(offset):
            is_minimum &= clean <= neighbour(offset)
    if not is_minimum.any():  # no finite sample: the caller's handling of the surface applies
        return [np.array(np.unravel_index(int(np.nanargmin(surface)), shape))], [-np.inf]
    bound = np.zeros(shape)
    with np.errstate(invalid="ignore"):
        for axis in range(n_dim):
            above = neighbour(tuple(int(a == axis) for a in range(n_dim)))
            below = neighbour(tuple(-int(a == axis) for a in range(n_dim)))
            mirrored_above = np.where(np.isfinite(above), above, below)
            mirrored_below = np.where(np.isfinite(below), below, above)
            bound += 0.25 * (mirrored_above + mirrored_below - 2.0 * clean)
        lowest = float(clean[is_minimum].min())
        floor = clean - bound
        can_be_lowest = is_minimum & ~(floor > lowest)
        near = is_minimum & ~can_be_lowest & ~(floor > lowest + slack)
    last = np.array(shape) - 1

    def ordered(keep, *, with_lowest):
        index = np.argwhere(keep)
        starts = [index[i] for i in np.argsort(clean[keep], kind="stable")]
        if interior:
            starts = [
                start
                for k, start in enumerate(starts)
                if (with_lowest and k == 0) or (np.all(start > 0) and np.all(start < last))
            ]
        return starts[: max(int(n_max), 1)]

    starts = ordered(can_be_lowest, with_lowest=True)
    if slack > 0.0:
        starts += ordered(near, with_lowest=False)
    return starts, [float(floor[tuple(start)]) for start in starts]


def _hessian(terms: _Terms, pos: np.ndarray, amps, mode: str, h: float = 0.2):
    """Curvature of the (profiled) chi-square at ``pos``, from a stencil inside its cell."""
    hi = terms.n_shift - 1
    cell_lo = np.clip(np.floor(pos), 0, hi - 1)
    cell_hi = cell_lo + 1.0
    pts = _stencil(pos, cell_lo, cell_hi, h)
    values = np.array([_profiled(terms, p, amps, mode)[0] for p in pts])
    _, _, hess = _quadratic_fit(pts - pos, values)
    return 0.5 * (hess + hess.T)


@jax.jit
def _epoch_columns(t_stack, rows, cols, vals, w, deltas):
    """The projected templates at a few integer shifts, ``(n_tmpl, n_out, n_shift)``.

    This is the gather from which :func:`_epoch_terms` builds its inner products, returned
    as pixel vectors. The Jacobian of the model at the solution is assembled from them.
    """
    n_out = w.shape[0]
    n_grid = t_stack.shape[1]

    def columns(t, d):
        idx = cols[:, None] - d[None, :]
        ok = (idx >= 0) & (idx < n_grid)
        contrib = vals[:, None] * jnp.where(ok, t[jnp.clip(idx, 0, n_grid - 1)], 0.0)
        return jax.ops.segment_sum(contrib, rows, num_segments=n_out)

    return jax.vmap(columns)(t_stack, deltas)


def _ar1_apply(x: np.ndarray, phi: float) -> np.ndarray:
    """``R x`` for the AR(1) correlation ``R_pq = phi^|p - q|`` over the pixel index.

    Two first-order recursions are used, forward and backward, each a filter with one
    pole. Their sum counts the diagonal twice. Rows with no weight contribute nothing to
    ``x`` and are still traversed, which is the correlation of a subset of a Markov chain.
    """
    from scipy.signal import lfilter

    forward = lfilter([1.0], [1.0, -phi], x, axis=0)
    backward = lfilter([1.0], [1.0, -phi], x[::-1], axis=0)[::-1]
    return forward + backward - x


def _correlated_covariance(stack, work: _EpochWork, fine, pos, amps, amp_mode: str, phi: float):
    """The shift covariance in pixels squared under AR(1) noise, by the sandwich.

    The model at the solution is linear in the amplitudes, in the nuisance, and in the
    shifts within their cells (a template at a fractional shift is the linear interpolation
    of its two integer neighbours, the identity used by :meth:`_Terms.at`). Its Jacobian
    ``J`` is therefore assembled from the projected templates at the two integer shifts
    bracketing each position. With ``W = diag(w)`` and ``R`` the AR(1) correlation of the
    standardized noise, ``Cov = (J^T W J)^-1 J^T W^1/2 R W^1/2 J (J^T W J)^-1``. At
    ``phi = 0`` this is the white-noise curvature error. The shift block is returned.
    """
    n_tmpl = fine.shape[0]
    hi = fine.shape[1] - 1
    cell = np.clip(np.floor(np.asarray(pos, dtype=np.float64)).astype(int), 0, hi - 1)
    frac = np.asarray(pos, dtype=np.float64) - cell
    deltas = np.stack([fine[i, cell[i] : cell[i] + 2] for i in range(n_tmpl)]).astype(np.int32)
    columns = np.asarray(
        _epoch_columns(stack, work.rows, work.cols, work.vals, work.w, jnp.asarray(deltas))
    )
    model = (1.0 - frac)[:, None] * columns[:, :, 0] + frac[:, None] * columns[:, :, 1]
    slope = columns[:, :, 1] - columns[:, :, 0]
    amps = np.asarray(amps, dtype=np.float64)
    jacobian = [amps[i] * slope[i] for i in range(n_tmpl)]
    if amp_mode == "free":
        jacobian += [model[i] for i in range(n_tmpl)]
    elif amp_mode == "scale":
        jacobian.append(amps @ model)
    basis = np.asarray(work.basis)
    jacobian += [basis[:, k] for k in range(basis.shape[1])]
    whitened = np.sqrt(np.asarray(work.w))[:, None] * np.stack(jacobian, axis=1)
    curvature = whitened.T @ whitened
    try:
        cov = np.linalg.inv(curvature)
    except np.linalg.LinAlgError:
        return np.full((n_tmpl, n_tmpl), np.nan)
    middle = whitened.T @ _ar1_apply(whitened, phi)
    sandwich = cov @ middle @ cov
    return sandwich[:n_tmpl, :n_tmpl]


# ---------------------------------------------------------------------------
# Instrument preparation
# ---------------------------------------------------------------------------


def _effective_sigma(instrument: str, sigma_inst, template: Template) -> np.ndarray:
    sig = np.atleast_1d(np.asarray(sigma_inst, dtype=np.float64))
    if np.any(sig <= 0.0):
        raise ValueError(f"instrument {instrument!r}: LSF widths must be positive")
    excess = sig**2 - template.sigma_kms**2
    if np.any(excess < -1e-9):
        warnings.warn(
            f"template {template.name!r} is broader ({template.sigma_kms:.2f} km/s) than "
            f"instrument {instrument!r}'s LSF ({float(np.min(sig)):.2f} km/s); it is used "
            "without further broadening, and the correlation peak will be wider than the "
            "data's lines",
            stacklevel=3,
        )
    return np.sqrt(np.clip(excess, 0.0, None))


def _lsf_key(epoch, lsf_sigma_v):
    """The key under which an epoch's convolved templates are cached.

    The key is the instrument, with the epoch's own width added for an instrument declared
    :data:`albireo.forward.PER_EPOCH`, so that epochs at different resolving powers are
    correlated against templates at theirs.
    """
    if lsf_sigma_v is not None and _is_per_epoch(lsf_sigma_v.get(epoch.instrument)):
        if epoch.lsf_sigma_kms is None:
            raise ValueError(
                f"instrument {epoch.instrument!r} is declared PER_EPOCH but an epoch at "
                f"BJD {epoch.bjd:.5f} declares no LSF width (EpochData.lsf_sigma_kms)"
            )
        return (epoch.instrument, float(epoch.lsf_sigma_kms))
    return (epoch.instrument, None)


def _convolved_templates(
    templates: Sequence[Template],
    grid: LogGrid,
    instrument: str,
    lsf_sigma_v,
    lsf_anchors_angstrom,
    epoch_sigma_kms: float | None = None,
) -> tuple[np.ndarray, float]:
    """Templates convolved with one instrument's LSF, and the narrowest sigma in pixels.

    ``epoch_sigma_kms`` supplies the width for an instrument declared
    :data:`albireo.forward.PER_EPOCH`; it is ignored otherwise.
    """
    if lsf_sigma_v is None:
        stack = np.stack([t.deviation for t in templates])
        return stack, 0.0
    if instrument not in lsf_sigma_v:
        raise ValueError(
            f"no LSF width for instrument {instrument!r}; lsf_sigma_v covers "
            f"{sorted(lsf_sigma_v)}. Pass lsf_sigma_v=None only if the templates are "
            "already at the instruments' resolution."
        )
    sigma_inst = lsf_sigma_v[instrument]
    if _is_per_epoch(sigma_inst):
        if epoch_sigma_kms is None:
            raise ValueError(
                f"instrument {instrument!r} is declared PER_EPOCH: the epoch's own width "
                "is needed to convolve the templates"
            )
        sigma_inst = float(epoch_sigma_kms)
    anchors = None if lsf_anchors_angstrom is None else lsf_anchors_angstrom.get(instrument)
    rows = []
    narrowest = np.inf
    for t in templates:
        sigma = _effective_sigma(instrument, sigma_inst, t)
        if anchors is None:
            if sigma.size != 1:
                raise ValueError(
                    f"instrument {instrument!r}: {sigma.size} LSF widths but no anchors; "
                    "per-anchor widths need lsf_anchors_angstrom"
                )
            s = float(sigma[0])
            narrowest = min(narrowest, s)
            if s / grid.dv_kms < 1e-3:
                rows.append(t.deviation)
            else:
                kernel = gaussian_kernel(s / grid.dv_kms)
                rows.append(np.asarray(convolve_spectrum(jnp.asarray(t.deviation), kernel)))
        else:
            anchor_wave = tuple(float(x) for x in anchors)
            if sigma.size == 1:
                sigma = np.full(len(anchor_wave), sigma[0])
            if sigma.size != len(anchor_wave):
                raise ValueError(
                    f"instrument {instrument!r}: {sigma.size} LSF widths for "
                    f"{len(anchor_wave)} anchors"
                )
            narrowest = min(narrowest, float(np.min(sigma)))
            sigma_px = np.maximum(sigma / grid.dv_kms, 1e-3)
            profiles = gaussian_lsf_profiles(sigma_px, anchor_wave, grid.wave)
            rows.append(np.asarray(convolve_varying(jnp.asarray(t.deviation), profiles)))
    return np.stack(rows), float(narrowest / grid.dv_kms if np.isfinite(narrowest) else 0.0)


def _chebyshev_basis(n: int, order: int | None) -> np.ndarray:
    if order is None:
        return np.zeros((n, 0))
    x = np.linspace(-1.0, 1.0, n)
    return np.polynomial.chebyshev.chebvander(x, order)


def _round_up(n: int, multiple: int) -> int:
    return int(math.ceil(n / multiple) * multiple)


@dataclass
class _EpochWork:
    """One epoch's static arrays, padded to bucketed shapes for the jitted terms."""

    index: int
    instrument: str
    bary_pix: float
    z: jax.Array
    w: jax.Array
    rows: jax.Array
    cols: jax.Array
    vals: jax.Array
    basis: jax.Array
    n_good: int


def _prepare_epoch(index: int, epoch: EpochData, grid: LogGrid, nuisance_order) -> _EpochWork:
    rebin = rebin_operator(x_in=grid.wave, x_out=epoch.wave)
    coverage = np.asarray(rebin.coverage)
    w = epoch.effective_ivar * (coverage >= 1.0 - 1e-10)
    z = np.where(w > 0.0, epoch.flux - 1.0, 0.0)
    n_native = epoch.n_pixels
    n_pad = _round_up(n_native, _ROW_BUCKET)
    rows = np.asarray(rebin.rows)
    cols = np.asarray(rebin.cols)
    vals = np.asarray(rebin.vals)
    nnz_pad = _round_up(rows.size, _ROW_BUCKET)
    rows_p = np.full(nnz_pad, n_pad - 1, dtype=np.int32)
    cols_p = np.zeros(nnz_pad, dtype=np.int32)
    vals_p = np.zeros(nnz_pad)
    rows_p[: rows.size] = rows
    cols_p[: cols.size] = cols
    vals_p[: vals.size] = vals
    basis = np.zeros((n_pad, _chebyshev_basis(2, nuisance_order).shape[1]))
    basis[:n_native] = _chebyshev_basis(n_native, nuisance_order)
    return _EpochWork(
        index=index,
        instrument=epoch.instrument,
        bary_pix=float(np.asarray(grid.velocity_to_pixels(epoch.v_bary))),
        z=jnp.asarray(np.pad(z, (0, n_pad - n_native))),
        w=jnp.asarray(np.pad(w, (0, n_pad - n_native))),
        rows=jnp.asarray(rows_p),
        cols=jnp.asarray(cols_p),
        vals=jnp.asarray(vals_p),
        basis=jnp.asarray(basis),
        n_good=int(np.sum(w > 0.0)),
    )


def _terms_numpy(out, n_shift: int) -> _Terms:
    b, gram, pwa, pwp, pwz, zwz = out
    return _Terms(
        b=np.asarray(b)[:, :n_shift],
        gram=np.asarray(gram)[:, :, :n_shift, :n_shift],
        pwa=np.asarray(pwa)[:, :, :n_shift],
        pwp=np.asarray(pwp),
        pwz=np.asarray(pwz),
        zwz=float(zwz),
    )


def _velocity_from_shift(grid: LogGrid, shift, frame: str, bary_pix: float):
    """Barycentric velocity of a component shifted by ``shift`` pixels in the data frame."""
    total = np.asarray(shift, dtype=np.float64) + (bary_pix if frame == "topocentric" else 0.0)
    return np.asarray(grid.pixels_to_velocity(total), dtype=np.float64), total


def _dv_dpix(grid: LogGrid, total_pix) -> np.ndarray:
    """Exact Jacobian ``dv/d(pixel)`` at the given total shift (D2)."""
    xi = np.asarray(total_pix, dtype=np.float64) * grid.dx
    if grid.relativistic:
        return C_KMS * grid.dx / np.cosh(xi) ** 2
    return C_KMS * grid.dx * np.exp(xi)


def _compose(v_kms, v_zero_kms: float | None, relativistic: bool):
    """Compose a velocity with a rest-frame offset by relativistic velocity addition.

    The composition law of shifts in log-wavelength (``docs/math.md`` §7.6, §10.5).
    """
    if v_zero_kms is None or v_zero_kms == 0.0:
        return v_kms
    xi = np.asarray(log_doppler_shift(v_kms, relativistic=relativistic)) + float(
        np.asarray(log_doppler_shift(v_zero_kms, relativistic=relativistic))
    )
    return C_KMS * (np.tanh(xi) if relativistic else np.expm1(xi))


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------


def _finite_reduce(func, values) -> float:
    """``func`` over the finite entries of ``values``; ``nan`` when there are none.

    An epoch at the search edge has no velocity and no uncertainty, so a column of a
    small table can be entirely NaN. The NumPy ``nan*`` reductions warn and return NaN
    there, and the summary line should not emit a warning.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    return float(func(finite)) if finite.size else float("nan")


@dataclass(frozen=True)
class VelocityTable:
    """Per-epoch velocities of every component, with the diagnostics that qualify them.

    Rows are epochs in the dataset's order; component axes follow ``names``. Every array
    is NumPy. Velocities are barycentric whatever frame the data were declared in. They are
    absolute for a component only where ``absolute`` is true. A template whose rest frame
    is unknown (a disentangled component) yields velocities that include that component's
    own unidentified zero point, as ``docs/math.md`` §7.6 describes for the free-velocity
    table (§10.5).

    Attributes
    ----------
    names
        Component names, from the templates.
    bjd, instrument
        Per epoch.
    velocity
        ``(n_comp, n_epochs)`` km/s, ``nan`` where nothing was measured: an epoch with too
        few weighted pixels, and any component flagged ``at_edge``, whose chi-square was
        still decreasing at the end of the search interval.
    sigma
        ``(n_comp, n_epochs)`` km/s, the quoted uncertainty: the curvature of the
        chi-square surface at its minimum, rescaled by the reduced chi-square so that the
        noise level is estimated from the residuals rather than taken from ``ivar``. This
        is the estimator of Zucker (2003) (``docs/math.md`` §10.4). ``sigma_ivar`` is the
        same curvature without the rescaling. Both are ``nan`` wherever ``velocity`` is.
    covariance
        ``(n_epochs, n_comp, n_comp)`` in km/s², on the ``sigma`` scale. Its off-diagonal
        is the blending diagnostic: velocities that are highly correlated were measured
        along a ridge rather than at a peak.
    light
        ``(n_comp, n_epochs)`` amplitudes assigned to the templates: the light fractions
        as declared (``light_mode == "fixed"``), or as measured per epoch or globally.
    chi2, chi2_null, n_pixels
        The minimum chi-square, the chi-square with no template (the nuisance alone), and
        the number of pixels with non-zero weight.
    r_squared
        ``1 - chi2 / chi2_null``, the correlation ``R^2`` of Zucker & Mazeh (1994) at the
        maximum.
    delta_chi2
        ``(n_comp, n_epochs)``: the rise in chi-square when that component is removed and
        the rest refitted. This is the per-epoch detection statistic. It is small for a
        companion not detected at that epoch.
    blended
        Per epoch: the epoch does not determine the velocities. The flag is raised where
        they lie on a ridge (a covariance correlation above 0.9, or a curvature that is not
        positive definite), and where another minimum of the surface fits as well and gives
        different velocities: ``margin`` below 9 times the reduced chi-square, or below 9
        under ``errors="ivar"`` (:func:`todcor`, Notes).
    margin
        Per epoch: the rise in chi-square from the minimum returned to the lowest other
        minimum the search refined at which the velocities differ. Two minima give the same
        velocities where, in some order of the components, every velocity of one is within
        three quoted errors of the other's. The same pair of velocities assigned to the
        stars in the other order is therefore not counted. One epoch does not determine
        that assignment for two alike stars, and an orbit does
        (:func:`albireo.rvorbit.assign_components`). ``inf`` where the search found no such
        minimum, and in a table built without one. ``nan`` where nothing was measured or
        the curvature gave no error.
    alternative, alternative_covariance
        ``(n_comp, n_epochs)`` km/s and ``(n_epochs, n_comp, n_comp)`` km/s²: the
        velocities at the minimum ``margin`` refers to and their covariance, where that
        minimum flags the epoch (``second_minimum``), and ``nan`` elsewhere. The epoch
        alone does not decide between the two pairs, and an orbit can
        (:func:`albireo.rvorbit.assign_components`). The covariance is on the scale of
        ``covariance``, with the reduced chi-square of that minimum, and
        ``alternative_sigma`` is the square root of its diagonal. ``light`` and
        ``delta_chi2`` are those of the minimum returned.
    at_edge
        ``(n_comp, n_epochs)``: the chi-square of that component was still decreasing at
        the edge of the range searched, either on the coarse grid over ``v_range`` or on
        the refinement window that moves within it. The minimum is then not bracketed and
        nothing is measured. ``velocity`` and ``sigma`` are ``nan`` for such a component,
        every diagnostic below is that of the last point evaluated, and ``v_range`` should
        be widened (or, when it is already wide, the templates are the wrong ones).
    refined
        Per epoch: the sub-pixel refinement succeeded (otherwise the integer-grid minimum
        is reported, with its curvature).
    absolute
        Per component: whether the velocities have an absolute zero point.
    """

    names: tuple[str, ...]
    bjd: np.ndarray
    instrument: tuple[str, ...]
    velocity: np.ndarray
    sigma: np.ndarray
    sigma_ivar: np.ndarray
    covariance: np.ndarray
    light: np.ndarray
    light_mode: str
    chi2: np.ndarray
    chi2_null: np.ndarray
    n_pixels: np.ndarray
    delta_chi2: np.ndarray
    blended: np.ndarray
    at_edge: np.ndarray
    refined: np.ndarray
    absolute: tuple[bool, ...]
    frame: str
    margin: np.ndarray | None = None
    alternative: np.ndarray | None = None
    alternative_covariance: np.ndarray | None = None
    settings: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.margin is None:
            object.__setattr__(self, "margin", np.full(np.shape(self.bjd), np.inf))
        if self.alternative is None:
            object.__setattr__(self, "alternative", np.full(np.shape(self.velocity), np.nan))
        if self.alternative_covariance is None:
            empty = np.full(np.shape(self.covariance), np.nan)
            object.__setattr__(self, "alternative_covariance", empty)

    @property
    def n_epochs(self) -> int:
        return int(self.bjd.size)

    @property
    def n_components(self) -> int:
        return len(self.names)

    @property
    def r_squared(self) -> np.ndarray:
        """``1 - chi2 / chi2_null`` per epoch.

        This is the correlation ``R^2`` of Zucker & Mazeh (1994) evaluated at the maximum.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.chi2_null > 0, 1.0 - self.chi2 / self.chi2_null, np.nan)

    @property
    def reduced_chi2(self) -> np.ndarray:
        """``chi2 / (n_pixels - n_parameters)`` per epoch.

        ``sigma`` is ``sigma_ivar`` multiplied by the square root of this value.
        """
        dof = np.maximum(self.n_pixels - self.settings.get("n_parameters", 0), 1)
        return self.chi2 / dof

    @property
    def alternative_sigma(self) -> np.ndarray:
        """``(n_comp, n_epochs)`` km/s: the quoted errors of ``alternative``."""
        variance = np.diagonal(self.alternative_covariance, axis1=1, axis2=2).T
        with np.errstate(invalid="ignore"):
            return np.sqrt(np.where(variance >= 0.0, variance, np.nan))

    @property
    def second_minimum(self) -> np.ndarray:
        """Per epoch: ``margin`` is below the threshold that raises ``blended``.

        The threshold is 9 times the reduced chi-square, or 9 under ``errors="ivar"``.
        These epochs are blended whatever the curvature at the minimum returned.
        """
        scale = 1.0 if self.settings.get("errors") == "ivar" else self.reduced_chi2
        with np.errstate(invalid="ignore"):
            return np.asarray(self.margin < _BLEND_CHI2 * scale, dtype=bool)

    @property
    def good(self) -> np.ndarray:
        """Epochs with finite velocities, off the search edge and not blended."""
        finite = np.all(np.isfinite(self.velocity), axis=0)
        return finite & ~self.blended & ~np.any(self.at_edge, axis=0)

    def component(self, name: str) -> dict[str, np.ndarray]:
        """The per-epoch columns of one named component."""
        if name not in self.names:
            raise KeyError(f"no component {name!r}; this table has {list(self.names)}")
        i = self.names.index(name)
        return {
            "bjd": self.bjd,
            "velocity": self.velocity[i],
            "sigma": self.sigma[i],
            "light": self.light[i],
            "delta_chi2": self.delta_chi2[i],
            "at_edge": self.at_edge[i],
        }

    def wilson(self) -> tuple[float, float] | None:
        """Slope and intercept of component 2 against component 1 over the good epochs.

        The slope is ``-K_2 / K_1``, the inverse mass ratio, and is unaffected by either
        zero point. Returns ``None`` for fewer than two components or fewer than three
        usable epochs.
        """
        if self.n_components < 2:
            return None
        ok = self.good  # excludes the epochs left NaN at the search edge
        if int(ok.sum()) < 3:
            return None
        slope, intercept = np.polyfit(self.velocity[0, ok], self.velocity[1, ok], 1)
        return float(slope), float(intercept)

    def to_dict(self) -> dict[str, np.ndarray]:
        """Flat columns keyed like the written table.

        ``pandas.DataFrame(table.to_dict())`` builds a data frame from them.
        """
        out: dict[str, np.ndarray] = {"bjd": self.bjd, "instrument": np.asarray(self.instrument)}
        for i, name in enumerate(self.names):
            out[f"v_{name}"] = self.velocity[i]
            out[f"sigma_{name}"] = self.sigma[i]
        for i, name in enumerate(self.names):
            out[f"light_{name}"] = self.light[i]
            out[f"dchi2_{name}"] = self.delta_chi2[i]
        out["chi2_red"] = self.reduced_chi2
        out["r2"] = self.r_squared
        out["n_pix"] = self.n_pixels
        out["margin"] = self.margin
        other_sigma = self.alternative_sigma
        for i, name in enumerate(self.names):
            out[f"alt_{name}"] = self.alternative[i]
            out[f"alt_sigma_{name}"] = other_sigma[i]
        out["blended"] = self.blended
        out["at_edge"] = np.any(self.at_edge, axis=0)
        out["refined"] = self.refined
        return out

    def write(self, path, *, header: str = "") -> Path:
        """Write the table as whitespace-separated ASCII with a commented header.

        The epoch times are written with enough digits to recover the same float64, and
        every other number with six decimals. Six decimals of a BJD is an error of up to
        5e-7 d, and shifts of that size reorder the near-degenerate short-period peaks of a
        period search (:func:`albireo.rvorbit.find_period`), which would then not be
        reproduced from the written table.
        """
        path = Path(path)
        columns = self.to_dict()
        lines = ["# albireo.todcor velocity table"]
        if header:
            lines += [f"# {line}" for line in header.splitlines()]
        lines.append(f"# frame of the velocities: barycentric (data declared {self.frame})")
        for name, absolute in zip(self.names, self.absolute, strict=True):
            zero = "absolute" if absolute else "own unidentified zero point (differential)"
            lines.append(f"# component {name}: {zero}")
        lines.append(f"# light fractions: {self.light_mode}")
        lines.append("# " + " ".join(columns))
        for j in range(self.n_epochs):
            fields = []
            for key, col in columns.items():
                value = col[j]
                if key == "instrument":
                    fields.append(str(value))
                elif key == "bjd":
                    fields.append(repr(float(value)))
                elif key in ("n_pix",):
                    fields.append(f"{int(value):d}")
                elif isinstance(value, (bool, np.bool_)):
                    fields.append("1" if value else "0")
                else:
                    fields.append(f"{float(value):.6f}")
            lines.append(" ".join(fields))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def summary(self) -> str:
        """A text report: frame, zero points per component, flags and the Wilson slope."""
        lines = [
            f"TODCOR velocities: {self.n_components} components x {self.n_epochs} epochs, "
            f"{int(self.good.sum())} usable "
            f"(data {self.frame}; velocities barycentric)"
        ]
        for i, name in enumerate(self.names):
            zero = "absolute" if self.absolute[i] else "differential (template zero point unknown)"
            ok = np.isfinite(self.velocity[i])
            span = (
                f"{_finite_reduce(np.min, self.velocity[i]):+.3f} to "
                f"{_finite_reduce(np.max, self.velocity[i]):+.3f} km/s"
                if ok.any()
                else "no finite velocities"
            )
            lines.append(
                f"  {name}: {span}, median sigma "
                f"{_finite_reduce(np.median, self.sigma[i]):.4f} km/s, light "
                f"{_finite_reduce(np.median, self.light[i]):.3f} ({self.light_mode}); {zero}"
            )
        lines.append(
            f"  reduced chi-square: median {_finite_reduce(np.median, self.reduced_chi2):.3f} "
            f"(range {_finite_reduce(np.min, self.reduced_chi2):.3f}-"
            f"{_finite_reduce(np.max, self.reduced_chi2):.3f}); "
            f"R^2 median {_finite_reduce(np.median, self.r_squared):.3f}"
        )
        n_blend = int(self.blended.sum())
        n_edge = int(np.any(self.at_edge, axis=0).sum())
        n_unrefined = int((~self.refined).sum())
        if n_blend or n_edge or n_unrefined:
            n_second = int((self.blended & self.second_minimum).sum())
            second = f" ({n_second} by a second minimum)" if n_second else ""
            lines.append(
                f"  flags: {n_blend} blended{second}, {n_edge} at the search edge, not "
                f"measured, {n_unrefined} not refined below a pixel"
            )
        weak = [
            f"{name} in {int((self.delta_chi2[i] < 25.0).sum())} epoch(s)"
            for i, name in enumerate(self.names)
            if np.any(self.delta_chi2[i] < 25.0)
        ]
        if weak:
            lines.append("  weakly detected (delta chi2 < 25): " + ", ".join(weak))
        wilson = self.wilson()
        if wilson is not None:
            lines.append(
                f"  Wilson slope {self.names[1]} vs {self.names[0]}: {wilson[0]:.4f} "
                f"(= -K_{self.names[1]}/K_{self.names[0]})"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------


def _resolve_ranges(v_range, n_tmpl: int) -> np.ndarray:
    arr = np.asarray(v_range, dtype=np.float64)
    if arr.shape == (2,):
        arr = np.repeat(arr[None, :], n_tmpl, axis=0)
    if arr.shape != (n_tmpl, 2) or np.any(arr[:, 1] <= arr[:, 0]):
        raise ValueError(
            "v_range must be (lo, hi) or one (lo, hi) pair per template with lo < hi; "
            f"got {np.asarray(v_range).tolist()} for {n_tmpl} templates"
        )
    return arr


def _resolve_light(light, names: Sequence[str]):
    """Parse the ``light`` argument into ``(mode, fixed amplitudes or None)``."""
    if isinstance(light, str):
        if light not in ("free", "global"):
            raise ValueError(f"light must be 'free', 'global' or a sequence; got {light!r}")
        return light, None
    if isinstance(light, Mapping):
        missing = [n for n in names if n not in light]
        if missing:
            raise ValueError(f"light is missing entries for {missing}")
        arr = np.array([float(light[n]) for n in names])
    else:
        arr = np.asarray(light, dtype=np.float64).reshape(-1)
        if arr.size != len(names):
            raise ValueError(f"light has {arr.size} entries for {len(names)} templates")
    if np.any(arr < 0.0):
        raise ValueError("light fractions must be non-negative")
    if not math.isclose(float(arr.sum()), 1.0, abs_tol=1e-6):
        raise ValueError(
            f"light fractions must sum to 1 (got {float(arr.sum()):.6f}); the templates are "
            "normalized to their own continua, so the amplitudes are fractions of the light"
        )
    return "fixed", arr


def _check_margins(grid: LogGrid, dataset: Dataset, shift_lo: float, shift_hi: float) -> None:
    x_lo = min(math.log(float(e.wave[0])) for e in dataset)
    x_hi = max(math.log(float(e.wave[-1])) for e in dataset)
    need_lo = shift_hi * grid.dx  # a redshifted template must still cover the blue end
    need_hi = -shift_lo * grid.dx
    have_lo = x_lo - grid.x0
    have_hi = (grid.x0 + grid.dx * (grid.n - 1)) - x_hi
    short = max(need_lo - have_lo, need_hi - have_hi) / grid.dx
    if short > 2.0:
        warnings.warn(
            f"the template grid is {short:.0f} pixels too narrow for the velocity range "
            "searched: a template shifted to the edge of the range runs off the grid and "
            "zero-fills part of the data. Build the grid with LogGrid.covering(dataset, ..., "
            "v_margin_kms=<the largest |velocity| searched>).",
            stacklevel=3,
        )


def _global_light(first: VelocityTable) -> dict[str, np.ndarray]:
    """Per-instrument light fractions from a free-amplitude pass: a median over epochs.

    The epochs are those of the instrument with every amplitude positive, usable, and
    every component detected at a rise above 9. Where none qualifies, every epoch with
    positive amplitudes is taken. Where no epoch has positive amplitudes the pass has not
    measured the light, and an error is raised. Two templates cooler than both stars fit a
    blend best as a difference, with one amplitude negative, at every epoch of a pair that
    is never resolved. Before D68 the median was then taken over those epochs, and the
    held fractions, one of them negative, left every epoch of the table on a ridge.
    """
    out = {}
    instruments = sorted(set(first.instrument))
    for inst in instruments:
        here = np.array([i == inst for i in first.instrument])
        with np.errstate(invalid="ignore"):
            positive = here & np.all(np.isfinite(first.light) & (first.light > 0.0), axis=0)
            sel = positive & first.good & np.all(first.delta_chi2 > 9.0, axis=0)
        if not sel.any():
            sel = positive
        if not sel.any():
            raise _LightNotMeasured(
                f"instrument {inst!r}: no epoch gives a positive amplitude to every template, "
                "so the free pass does not measure the light fractions (light='free' shows "
                "the amplitudes). The templates do not describe every component at any "
                "epoch: pass light=<fractions>, or templates closer to the components"
            )
        fractions = first.light[:, sel] / first.light[:, sel].sum(axis=0, keepdims=True)
        med = np.median(fractions, axis=1)
        out[inst] = med / med.sum()
    return out


def todcor(
    dataset: Dataset,
    templates: Sequence[Template],
    *,
    v_range=(-300.0, 300.0),
    light="global",
    lsf_sigma_v: Mapping[str, Any] | None = None,
    lsf_anchors_angstrom: Mapping[str, Sequence[float]] | None = None,
    nuisance_order: int | None = 0,
    coarse_step: int | None = None,
    errors: str = "profiled",
    scale: str = "fixed",
    noise_correlation: float | Mapping[str, float] | None = None,
    progress: bool = False,
) -> VelocityTable:
    """Measure every component's velocity in every epoch by N-dimensional correlation.

    The estimator is the weighted least-squares fit of ``docs/math.md`` §10.1. The
    chi-square of the shifted, LSF-convolved and rebinned templates against each epoch's
    pixels is minimized over the shifts, on the integer grid first and then exactly below
    a pixel (§10.3). On a uniform grid with uniform weights and free amplitudes the surface
    is the two-dimensional correlation of Zucker & Mazeh (1994) (§10.2).

    Parameters
    ----------
    dataset
        The epochs, continuum-normalized, with their masks in ``ivar`` (``0`` = ignored).
        The data are never resampled: the shifted templates are projected onto each
        epoch's own pixels.
    templates
        One :class:`Template` per component, all on one grid. Two give TODCOR, three and
        four the extensions of Zucker, Torres & Mazeh (1995) and Torres, Latham &
        Stefanik (2007), and so on. The search grid grows as a power of the number of
        templates, so beyond three components ``v_range`` should be narrowed and
        ``coarse_step`` increased.
    v_range
        Barycentric velocity range to search, km/s: one ``(lo, hi)`` for all components or
        one per template. The range is in each template's own frame, before that
        template's ``v_zero_kms`` is composed into the reported velocity. Templates whose
        zero points differ therefore need one pair each to search the same interval of
        reported velocity (:meth:`albireo.Fit.measure_velocities` builds them that way).
        The search is confined to the range: a component whose chi-square is still
        decreasing at its end is flagged ``at_edge`` and its velocity is ``nan``, since its
        minimum was not bracketed. The template grid must extend beyond the data by this
        much (:meth:`LogGrid.covering`); otherwise a warning reports the shortfall.
    light
        Treatment of the templates' amplitudes, i.e. their light fractions.
        ``"global"`` (default) fits them freely in every epoch, takes the weighted median
        over the well-detected, unblended epochs of each instrument, and re-measures with
        them held fixed. A per-epoch light ratio is noisy, and a ratio fitted at a blended
        phase is not a measurement. ``"free"`` reports the per-epoch fit itself. An
        amplitude of that fit can be negative: templates that do not describe the
        components fit a blend as a difference. ``"global"`` takes the median over the
        epochs at which every amplitude is positive and raises an error where there is
        none. A
        sequence or a ``{name: fraction}`` mapping, summing to one, holds them fixed. When
        the templates are the components of a disentangling that assumed fractions, those
        should be held fixed, because no other choice is consistent with the definition of
        the components (``docs/math.md`` §9.1).
    lsf_sigma_v
        Per-instrument Gaussian LSF sigma in km/s, as :func:`albireo.build_problem` takes
        it (a scalar, or one width per anchor with ``lsf_anchors_angstrom``). Applied to
        each template in quadrature above the template's own ``sigma_kms``. ``None``
        means the templates are already at the instruments' resolution, as when the
        template is an observed single-star spectrum from the same spectrograph.
    lsf_anchors_angstrom
        Optional per-instrument anchors for a wavelength-dependent LSF.
    nuisance_order
        Order of an additive Chebyshev polynomial fitted alongside the templates in every
        epoch: ``0`` (default) a constant, which absorbs the residual of the continuum
        normalization; ``None`` for none. The term is additive rather than multiplicative
        for the reason given in :mod:`albireo.match`: what it absorbs lies in the
        continuum, where a multiplicative term is identically zero.
    coarse_step
        Stride of the global search, in template pixels. Default: the narrowest effective
        LSF sigma in pixels (at least one), which cannot step over a correlation peak.
        Every minimum of the coarse surface that can contain the lowest one, or lie
        within the blend threshold of it, is then searched at full resolution and refined
        below a pixel, and where the lines of two components overlap the search is repeated
        from the exchanged pair (Notes). The result does not depend on the stride.
    errors
        ``"profiled"`` (default) rescales the curvature error by the reduced chi-square,
        so that the noise level is estimated from the residuals: the maximum-likelihood
        estimator of Zucker (2003), appropriate when ``ivar`` is known only to a scale.
        ``"ivar"`` omits the rescaling.
    scale
        With fixed or global light fractions, ``"fixed"`` (default) holds the composite at
        the fractions exactly, since continuum-normalized data fix its scale. ``"free"``
        solves one overall scale per epoch on top of the fixed ratios, the original form
        of TODCOR with a known light ratio (its correlation is scale-invariant). ``"free"``
        is appropriate when the normalization is uncertain. The fitted scale is then the
        sum of the reported ``light`` row, and its departure from one is a normalization
        diagnostic. Ignored when ``light="free"``.
    noise_correlation
        Lag-one correlation of each epoch's noise along its pixel index, one value or one
        per instrument, as produced by a pipeline that resampled the spectra onto a common
        step (Gaia's RVS grids have 0.27 and 0.81; :mod:`albireo.gaia` measures it). The
        weighted least-squares estimator is unchanged, but its error is not. With AR(1)
        noise along the pixel index the covariance is the sandwich of ``docs/math.md``
        §10.4, which grows with the correlation, and the diagonal curvature error is too
        small by that factor. ``None`` (default) takes the noise as white.
    progress
        Print one line per epoch.

    Returns
    -------
    VelocityTable

    Notes
    -----
    The estimator is the weighted least-squares fit, which Zucker (2003) showed to be the
    maximum-likelihood estimator. Its per-epoch error is the curvature of the chi-square
    surface (``docs/math.md`` §10.4), or the sandwich through that curvature when a noise
    correlation is declared. Two systematics are not included in that error. Template
    mismatch mostly moves each component by a constant (the zero point). The pixel-locking
    ripple of the linear shift operator is of order ``0.1 / sigma_px^2`` pixels (measured:
    0.006 px at five pixels per LSF sigma, 0.03 px at one). It is negligible when the
    template grid samples the narrowest LSF with three or more pixels per sigma (§10.3).
    :meth:`albireo.Fit.templates` upsamples to that. A library template's grid should be
    built the same way.

    Where the lines of two components overlap, the surface is a valley along which their
    light-weighted mean velocity is constant, with two minima: the solution, and the pair
    with the same mean and the velocity difference of the opposite sign, which reproduces
    the first two moments of the blended profile (``docs/math.md`` §10.3). The search is
    made from both and the lower is returned, to the minimum of a lattice of 0.125 pixel
    on all 1440 simulated Gaia RVS epochs it was compared on (``docs/benchmarks.md``). The
    two minima differ in depth only through the asymmetry of the blended profile, so noise
    can make the exchanged pair the lower. Both velocities are then wrong by about their
    separation, and the curvature at that minimum is regular, so the quoted errors do not
    show it.

    The search keeps every minimum it refines, and ``VelocityTable.margin`` is the rise in
    chi-square from the one returned to the lowest other one at which the velocities
    differ. An epoch is flagged ``blended`` where the margin is below 9 times the reduced
    chi-square (9 under ``errors="ivar"``). The velocities of two minima differ where, in
    every order of the components, some velocity differs by more than three quoted
    errors. The same pair in the other order is not counted: the surface of two alike
    stars has that minimum at every separation, the pair is measured, and an orbit assigns
    it to the stars (:func:`albireo.rvorbit.assign_components`).

    On those epochs (R = 11,500, light fractions 0.625 and 0.375, 480 at each S/N) either
    velocity was more than five quoted errors and 3 km/s from the injected one at 46, 17,
    1 and 0 epochs for a S/N of 15, 40, 100 and 300 per pixel, all with the lines 10 to
    40 km/s apart, up to 1.5 times the FWHM of the line-spread function. At 63 of these 64
    the minimum returned was the exchanged pair, and at all 64 the injected pair was
    another minimum the search refined, at a rise of at most 8.4. The margin flags 33 of
    the 64 and 96 epochs that were measured correctly. At the other 31 the two minima are
    the same pair of velocities in the two orders to within three quoted errors, the case
    the margin does not count. With the two velocities interchanged, 30 of them are within
    five quoted errors or 3 km/s of the injected ones (``docs/benchmarks.md``).

    Where the margin raises the flag, the table records the velocities of that minimum
    and their covariance (``VelocityTable.alternative``, ``alternative_covariance``). At
    each of the 33 flagged wrong epochs they are the injected pair. One epoch does not
    decide between the two minima, and an orbit does
    (:func:`albireo.rvorbit.assign_by_orbit`). Without one a flagged epoch is left out.

    One limit applies. The margin counts the minima the search refined. Where the lines
    are closer than their width the two minima merge into one elongated minimum, which
    is flagged only where the correlation of the two velocities exceeds 0.9 and has no
    other minimum to record.

    Velocities are barycentric. For topocentric data the shift searched is
    ``xi(v) - xi(v_bary)`` in log-wavelength (``docs/math.md`` §1.2). The composition is
    exact because log-shifts add (§10.5).

    References
    ----------
    Zucker, S. & Mazeh, T. 1994, ApJ, 420, 806
    Zucker, S., Torres, G. & Mazeh, T. 1995, ApJ, 452, 863
    Torres, G., Latham, D. W. & Stefanik, R. P. 2007, ApJ, 662, 602
    Zucker, S. 2003, MNRAS, 342, 1291
    """
    if not isinstance(dataset, Dataset):
        raise TypeError("dataset must be an albireo Dataset")
    templates = list(templates)
    grid = _templates_share_grid(templates)
    names = tuple(t.name for t in templates)
    n_tmpl = len(templates)
    ranges = _resolve_ranges(v_range, n_tmpl)
    mode, fixed = _resolve_light(light, names)
    if errors not in ("profiled", "ivar"):
        raise ValueError(f"errors must be 'profiled' or 'ivar'; got {errors!r}")
    if scale not in ("fixed", "free"):
        raise ValueError(f"scale must be 'fixed' or 'free'; got {scale!r}")
    if nuisance_order is not None and nuisance_order < 0:
        raise ValueError("nuisance_order must be None or >= 0")
    phi_of = _resolve_correlation(noise_correlation, dataset)

    if mode == "global":
        # The first pass supplies the light fractions only, so it does not search for the
        # second minima that the blend flag of the table needs.
        first = _run(
            dataset,
            templates,
            grid,
            ranges,
            "free",
            None,
            lsf_sigma_v,
            lsf_anchors_angstrom,
            nuisance_order,
            coarse_step,
            errors,
            scale,
            phi_of,
            progress,
            blend=False,
        )
        per_instrument = _global_light(first)
        table = _run(
            dataset,
            templates,
            grid,
            ranges,
            "global",
            per_instrument,
            lsf_sigma_v,
            lsf_anchors_angstrom,
            nuisance_order,
            coarse_step,
            errors,
            scale,
            phi_of,
            progress,
        )
        settings = dict(table.settings)
        settings["first_pass_light"] = first.light
        settings["global_light"] = {k: v.tolist() for k, v in per_instrument.items()}
        return replace(table, settings=settings)
    return _run(
        dataset,
        templates,
        grid,
        ranges,
        mode,
        fixed,
        lsf_sigma_v,
        lsf_anchors_angstrom,
        nuisance_order,
        coarse_step,
        errors,
        scale,
        phi_of,
        progress,
    )


def _resolve_correlation(noise_correlation, dataset: Dataset) -> dict[str, float]:
    """One lag-one correlation per instrument of the dataset, zero where none is declared."""
    instruments = list(dict.fromkeys(epoch.instrument for epoch in dataset))
    if noise_correlation is None:
        return dict.fromkeys(instruments, 0.0)
    if isinstance(noise_correlation, Mapping):
        unknown = sorted(set(noise_correlation) - set(instruments))
        if unknown:
            raise ValueError(
                f"noise_correlation names instrument(s) {unknown} that the dataset does not "
                f"have; it has {instruments}"
            )
        out = {name: float(noise_correlation.get(name, 0.0)) for name in instruments}
    else:
        out = dict.fromkeys(instruments, float(noise_correlation))
    for name, phi in out.items():
        if not (np.isfinite(phi) and -1.0 < phi < 1.0):
            raise ValueError(
                f"noise_correlation for {name!r} must lie in (-1, 1); got {phi}. It is the "
                "lag-one correlation of the pixel noise, not a variance."
            )
    return out


def _lower_of(best, candidate):
    """The candidate where its chi-square is finite and below that of ``best``, else ``best``."""
    if best is None or (np.isfinite(candidate[0]) and not candidate[0] >= best[0]):
        return candidate
    return best


def _refine_at(
    centre: np.ndarray,
    edge: np.ndarray,
    stack,
    work: _EpochWork,
    amps,
    amp_mode: str,
    window: tuple[np.ndarray, np.ndarray],
    radius: int,
    n_fine: int,
    n_fine_raw: int,
    blend: tuple[float, float],
):
    """The refined minimum of one epoch's chi-square reached from the integer shifts ``centre``.

    The fine pass evaluates every integer shift in a window of ``n_fine_raw`` shifts per
    template around ``centre``. The refinement below a pixel (:func:`_refine`) is started
    from ``centre`` and from each minimum of the window that can be its lowest
    (:func:`_candidate_minima`). A refined position inside the window is a minimum of the
    surface, and the lowest of these is kept. While the lowest point of the window is on
    its edge, an integer shift or the position a refinement reached, the minimum there is
    not bracketed, and the window is centred on that point and evaluated again, up to
    ``_WINDOW_MOVES`` times. The window is kept inside ``window``, the shift range the
    coarse pass searched, so that no reported velocity lies outside the requested
    ``v_range``.

    Returns the solution and the minima found. The solution is ``(chi2, position,
    amplitudes, refined, evaluated_start, fine, terms, edge)``. ``position`` is in pixels
    from ``evaluated_start``, the start of the window in which the result was evaluated,
    and ``fine`` and ``terms`` are that window's. The result is the lowest minimum inside a
    window, with ``edge`` as given (per template, a coarse point on the end of its range).
    Where the point on the edge of the last window is lower still, the surface is
    decreasing where the search stops. That point is then returned, with the templates on
    the edge added to ``edge``, and nothing is measured for them.

    The minima found are every refined minimum inside a window, each in the form of the
    solution and the solution among them. Each window is also refined from its local minima
    that can lie within the blend threshold of its lowest (``blend`` sets that threshold,
    :func:`_blend_slack`), so that a second minimum inside the window of the first is
    recorded.
    """
    n_tmpl = centre.shape[0]
    window_lo, window_hi = window
    last = n_fine_raw - 1
    inside = None  # the lowest refined minimum inside a window, with that window's terms
    found = []  # every refined minimum inside a window, in the form of the solution
    fine_start = np.clip(centre - radius, window_lo, window_hi)
    for attempt in range(_WINDOW_MOVES):
        evaluated_start = fine_start
        fine = np.stack([evaluated_start[i] + np.arange(n_fine) for i in range(n_tmpl)]).astype(
            np.int32
        )
        out = _epoch_terms(
            stack,
            work.rows,
            work.cols,
            work.vals,
            work.z,
            work.w,
            jnp.asarray(fine),
            work.basis,
            chunk=_SHIFT_CHUNK,
        )
        terms = _terms_numpy(out, n_fine_raw)
        if amps is None:
            fine_surface = np.asarray(_chi2_grid_free(*out))
        else:
            fine_surface = np.asarray(
                _chi2_grid_fixed(*out, jnp.asarray(amps), free_scale=amp_mode == "scale")
            )
        fine_surface = fine_surface[(slice(0, n_fine_raw),) * n_tmpl]
        # The integer shifts do not resolve which of two basins inside the window is the
        # lower, and a basin need not contain a local minimum of the integer shifts, so
        # the refinement starts from each candidate and, in the first window, from the
        # centre asked for. The first start is the lowest integer shift.
        finite = fine_surface[np.isfinite(fine_surface)]
        slack = _blend_slack(float(finite.min()), blend) if finite.size else 0.0
        starts, _ = _candidate_minima(fine_surface, _FINE_STARTS, interior=True, slack=slack)
        asked = centre - evaluated_start
        if (
            attempt == 0
            and np.all((asked >= 0) & (asked <= last))
            and not any(np.array_equal(asked, start) for start in starts)
        ):
            starts.append(asked)
        lowest = None
        for start in starts:
            result = _refine(terms, start, amps, amp_mode)
            lowest = _lower_of(lowest, result)
            if result[3] and np.all((result[1] > 0.0) & (result[1] < last)):
                bracketed = (*result, evaluated_start, fine, terms, edge)
                inside = _lower_of(inside, bracketed)
                found.append(bracketed)
        on_edge = (lowest[1] <= 0.0) | (lowest[1] >= last)
        if not on_edge.any():
            break
        fine_start = np.clip(
            evaluated_start + np.rint(lowest[1]).astype(int) - radius, window_lo, window_hi
        )
        if np.array_equal(fine_start, evaluated_start):
            break  # the window is already at the end of the requested range
    unbracketed = (*lowest, evaluated_start, fine, terms, edge | on_edge)
    solution = _lower_of(inside, unbracketed) if on_edge.any() or inside is None else inside
    return solution, found


def _blend_threshold(dof: int, errors: str) -> tuple[float, float]:
    """``(a, b)`` such that the blend threshold at a minimum of chi-square ``c`` is ``a c + b``.

    ``_BLEND_CHI2`` times the reduced chi-square under ``errors="profiled"``, which
    estimates the noise level from the residuals as the quoted errors do, and
    ``_BLEND_CHI2`` otherwise.
    """
    return (_BLEND_CHI2 / dof, 0.0) if errors == "profiled" else (0.0, _BLEND_CHI2)


def _blend_slack(chi2: float, blend: tuple[float, float]) -> float:
    """The rise in chi-square within which another minimum raises the blend flag.

    ``blend`` is the pair of :func:`_blend_threshold`, or ``(0, 0)`` for a search that
    keeps no second minimum.
    """
    return blend[0] * max(chi2, 0.0) + blend[1]


def _reported_velocities(grid: LogGrid, templates, shift, frame: str, bary_pix: float):
    """The velocities reported for templates at ``shift`` pixels, zero points composed."""
    v, _ = _velocity_from_shift(grid, shift, frame, bary_pix)
    return np.array(
        [float(_compose(v[i], t.v_zero_kms, grid.relativistic)) for i, t in enumerate(templates)]
    )


def _margin(chi2_min: float, velocity, sigma, alternatives) -> tuple[float, int | None]:
    """The rise in chi-square to the lowest other minimum that gives different velocities.

    ``alternatives`` holds the chi-square and the velocities of every minimum the search
    refined, the one returned among them. An alternative gives the same velocities where,
    in some order of its components, every velocity is within ``_BLEND_SIGMAS`` quoted
    errors of the one returned. Returns the rise and the index of that minimum in
    ``alternatives``: infinity and ``None`` where no alternative gives different ones.

    The order is free because the surface of two alike stars has a second minimum with the
    two velocities interchanged at every separation, as deep as the first for equal light
    fractions. That minimum gives the same pair of velocities and a different assignment
    to the stars, which one epoch does not determine and an orbit does
    (:func:`albireo.rvorbit.assign_components`). A flag that counted it marked every
    epoch of three benchmark systems of 10 to 15 epochs and 272 of 1037 usable epochs of
    the 33 (``docs/benchmarks.md``).
    """
    velocity = np.asarray(velocity, dtype=np.float64)
    tolerance = _BLEND_SIGMAS * np.asarray(sigma, dtype=np.float64)
    margin, index = np.inf, None
    for k, (chi2, other) in enumerate(alternatives):
        if not chi2 - chi2_min < margin:
            continue
        other = np.asarray(other, dtype=np.float64)
        same = any(
            np.all(np.abs(other[list(order)] - velocity) <= tolerance)
            for order in itertools.permutations(range(velocity.size))
        )
        if not same:
            margin, index = float(chi2 - chi2_min), k
    return margin, index


def _velocity_covariance(
    grid: LogGrid, stack, work, minimum, amps, amp_mode: str, phi: float, frame: str
):
    """The covariance of the velocities at a refined minimum, and whether it is defined.

    ``minimum`` has the form of the solution of :func:`_refine_at`. The covariance is in
    (km/s)^2 on the scale of the weights as given: twice the inverse curvature of the
    chi-square, or, with the noise correlated along the pixel index (``phi``), the sandwich
    through it evaluated from the model's Jacobian at the minimum (``docs/math.md`` §10.4).
    It is undefined where the curvature is not positive definite.
    """
    _, pos, fitted_amps, _, evaluated_start, fine, terms, _ = minimum
    n_tmpl = terms.n_tmpl
    hess = _hessian(terms, pos, amps, amp_mode)
    try:
        cov_pix = 2.0 * np.linalg.inv(hess)
        defined = bool(np.all(np.linalg.eigvalsh(hess) > 0.0))
    except np.linalg.LinAlgError:
        cov_pix = np.full((n_tmpl, n_tmpl), np.nan)
        defined = False
    if defined and phi != 0.0:
        cov_pix = _correlated_covariance(stack, work, fine, pos, fitted_amps, amp_mode, phi)
        defined = bool(np.all(np.isfinite(cov_pix)))
    total = evaluated_start + pos + (work.bary_pix if frame == "topocentric" else 0.0)
    jac = _dv_dpix(grid, total)
    return jac[:, None] * cov_pix * jac[None, :], defined


def _exchanged_shifts(shift: np.ndarray, weights: np.ndarray, i: int, k: int) -> np.ndarray:
    """The shifts with components ``i`` and ``k`` exchanged about their weighted mean.

    Two overlapping line systems of weights ``w_i`` and ``w_k`` (light times line strength)
    at shifts ``s_i`` and ``s_k`` give a blended profile whose first two moments are set by
    the mean ``c = (w_i s_i + w_k s_k) / (w_i + w_k)`` and by the square of the difference
    ``d = s_i - s_k``. The pair with the same mean and the difference ``-d`` has the same
    two moments, and the chi-square has a second minimum near it. For equal weights it is
    the two shifts interchanged.
    """
    out = np.array(shift, dtype=np.float64)
    w_i, w_k = float(weights[i]), float(weights[k])
    total = w_i + w_k
    out[i] = ((w_i - w_k) * shift[i] + 2.0 * w_k * shift[k]) / total
    out[k] = (2.0 * w_i * shift[i] + (w_k - w_i) * shift[k]) / total
    return out


def _run(
    dataset,
    templates,
    grid,
    ranges,
    mode,
    amplitudes,
    lsf_sigma_v,
    lsf_anchors_angstrom,
    nuisance_order,
    coarse_step,
    errors,
    scale,
    phi_of,
    progress,
    blend: bool = True,
) -> VelocityTable:
    n_tmpl = len(templates)
    names = tuple(t.name for t in templates)
    frame = dataset.frame
    relativistic = grid.relativistic
    # How the amplitudes enter the chi-square: held, scaled together, or all solved.
    amp_mode = "free" if mode == "free" else ("scale" if scale == "free" else "fixed")

    # The templates are convolved once per instrument (per declared width, for a PER_EPOCH
    # instrument). The narrowest sigma sets the coarse step.
    convolved: dict[tuple, np.ndarray] = {}
    narrowest_px = np.inf
    epoch_keys = [_lsf_key(epoch, lsf_sigma_v) for epoch in dataset]
    for key in dict.fromkeys(epoch_keys):
        stack, sigma_px = _convolved_templates(
            templates, grid, key[0], lsf_sigma_v, lsf_anchors_angstrom, epoch_sigma_kms=key[1]
        )
        convolved[key] = stack
        narrowest_px = min(narrowest_px, sigma_px)
    if not np.isfinite(narrowest_px):
        narrowest_px = 0.0
    if coarse_step is None:
        coarse_step = max(1, math.floor(narrowest_px))
    if coarse_step < 1:
        raise ValueError("coarse_step must be at least 1")
    if 0.0 < narrowest_px < 2.0:
        warnings.warn(
            f"the narrowest instrument LSF is {narrowest_px:.2f} template pixels wide; the "
            "shift interpolation's pixel-locking ripple is of order 0.1/sigma_px^2 pixels, so "
            "build the template grid at least three pixels per LSF sigma for sub-pixel accuracy",
            stacklevel=3,
        )
    stacks = {key: jnp.asarray(stack) for key, stack in convolved.items()}

    # Shift ranges in log-wavelength pixels (barycentric); composed per epoch below.
    xi_lo = np.asarray(log_doppler_shift(ranges[:, 0], relativistic=relativistic)) / grid.dx
    xi_hi = np.asarray(log_doppler_shift(ranges[:, 1], relativistic=relativistic)) / grid.dx
    # In blocks of epochs: outside jit the conversion is compiled once per array length.
    bary_all = _in_epoch_blocks(grid.velocity_to_pixels, dataset.v_bary)
    bary_span = (bary_all.max() - bary_all.min()) if frame == "topocentric" else 0.0
    _check_margins(grid, dataset, float(xi_lo.min() - bary_span), float(xi_hi.max() + bary_span))
    n_coarse = math.ceil((xi_hi - xi_lo).max() / coarse_step) + 1
    n_coarse = _round_up(n_coarse, _SHIFT_CHUNK)
    radius = coarse_step + 2
    n_fine_raw = 2 * radius + 1
    n_fine = _round_up(n_fine_raw, _SHIFT_CHUNK)
    m = 0 if nuisance_order is None else nuisance_order + 1
    n_par = n_tmpl + m + {"free": n_tmpl, "scale": 1, "fixed": 0}[amp_mode]

    n_ep = dataset.n_epochs
    velocity = np.full((n_tmpl, n_ep), np.nan)
    sigma = np.full((n_tmpl, n_ep), np.nan)
    sigma_ivar = np.full((n_tmpl, n_ep), np.nan)
    covariance = np.full((n_ep, n_tmpl, n_tmpl), np.nan)
    light = np.full((n_tmpl, n_ep), np.nan)
    chi2 = np.full(n_ep, np.nan)
    chi2_null = np.full(n_ep, np.nan)
    n_pixels = np.zeros(n_ep, dtype=int)
    delta_chi2 = np.full((n_tmpl, n_ep), np.nan)
    blended = np.zeros(n_ep, dtype=bool)
    margin = np.full(n_ep, np.nan)
    alternative = np.full((n_tmpl, n_ep), np.nan)
    alternative_covariance = np.full((n_ep, n_tmpl, n_tmpl), np.nan)
    at_edge = np.zeros((n_tmpl, n_ep), dtype=bool)
    refined = np.zeros(n_ep, dtype=bool)
    instruments = []

    for j, epoch in enumerate(dataset):
        t0 = time.perf_counter()
        instruments.append(epoch.instrument)
        work = _prepare_epoch(j, epoch, grid, nuisance_order)
        n_pixels[j] = work.n_good
        if work.n_good < max(8, n_par + 2):
            warnings.warn(f"epoch {j}: only {work.n_good} weighted pixels; skipped", stacklevel=3)
            continue
        dof = max(work.n_good - n_par, 1)
        threshold = _blend_threshold(dof, errors) if blend else (0.0, 0.0)
        bary = work.bary_pix if frame == "topocentric" else 0.0
        amps = None
        if mode == "fixed":
            amps = np.asarray(amplitudes, dtype=np.float64)
        elif mode == "global":
            amps = np.asarray(amplitudes[epoch.instrument], dtype=np.float64)

        # Coarse pass: integer shifts at the coarse stride, per template.
        starts = np.ceil(xi_lo - bary).astype(int)
        ends = np.floor(xi_hi - bary).astype(int)
        deltas = np.stack(
            [starts[i] + coarse_step * np.arange(n_coarse) for i in range(n_tmpl)]
        ).astype(np.int32)
        out = _epoch_terms(
            stacks[epoch_keys[j]],
            work.rows,
            work.cols,
            work.vals,
            work.z,
            work.w,
            jnp.asarray(deltas),
            work.basis,
            chunk=_SHIFT_CHUNK,
        )
        valid_count = np.array(
            [min(n_coarse, (ends[i] - starts[i]) // coarse_step + 1) for i in range(n_tmpl)]
        )
        if amps is None:
            surface = np.array(_chi2_grid_free(*out))
        else:
            surface = np.array(
                _chi2_grid_fixed(*out, jnp.asarray(amps), free_scale=amp_mode == "scale")
            )
        # Shifts past each template's own upper end are outside the requested range.
        for i in range(n_tmpl):
            index = [slice(None)] * n_tmpl
            index[i] = slice(int(valid_count[i]), None)
            surface[tuple(index)] = np.inf
        # The coarse stride does not resolve which of two basins of similar depth is the
        # lower, so every coarse minimum that can be the lowest (`_candidate_minima`) is
        # refined and the lowest refined chi-square is the solution. The first candidate
        # is the lowest coarse point, which is kept on a tie. The minima that can lie within
        # the blend threshold of the lowest are refined as well, and every refined minimum
        # is kept, for the margin of the solution (`_margin`).
        search = (
            stacks[epoch_keys[j]],
            work,
            amps,
            amp_mode,
            (starts, np.maximum(starts, ends - (n_fine_raw - 1))),
            radius,
            n_fine,
            n_fine_raw,
            threshold,
        )
        best = None
        minima = []  # every refined minimum of the epoch: its chi-square and its shifts
        finite = surface[np.isfinite(surface)]
        slack = _blend_slack(float(finite.min()), threshold) if finite.size else 0.0
        coarse_starts, floors = _candidate_minima(surface, _COARSE_STARTS, slack=slack)
        for index, floor in zip(coarse_starts, floors, strict=True):
            if best is not None and floor > best[0] + _blend_slack(best[0], threshold):
                continue  # this basin cannot be lower than the minimum refined, or near it
            edge = np.array(
                [index[i] == 0 or index[i] >= valid_count[i] - 1 for i in range(n_tmpl)]
            )
            solution, found = _refine_at(deltas[np.arange(n_tmpl), index], edge, *search)
            best = _lower_of(best, solution)
            minima += found
        # Two components whose lines overlap have a second minimum with the pair exchanged
        # about their weighted mean (`_exchanged_shifts`). It lies on the same narrow valley
        # of the surface as the solution, where the coarse samples do not separate the
        # two, so the search is also started from it for every pair of components. The
        # start is left out where the coarse samples around the exchanged pair, less twice
        # the sampling bound of the lowest coarse minimum, are above the minimum found by
        # more than the blend threshold: the lines do not overlap there, and the exchanged
        # pair is a basin of its own that the coarse minima cover.
        strength = np.abs(convolved[epoch_keys[j]]).sum(axis=1)
        allowance = 2.0 * (float(surface[tuple(coarse_starts[0])]) - floors[0])
        for i, k in itertools.combinations(range(n_tmpl), 2):
            weights = np.asarray(best[2], dtype=np.float64) * strength
            if not (np.all(np.isfinite(weights[[i, k]])) and np.all(weights[[i, k]] > 0.0)):
                continue
            exchanged = _exchanged_shifts(best[4] + best[1], weights, i, k)
            centre = np.rint(exchanged).astype(int)
            node = np.clip(np.rint((centre - starts) / coarse_step).astype(int), 0, valid_count - 1)
            around = surface[tuple(slice(max(int(n) - 1, 0), int(n) + 2) for n in node)]
            around = around[np.isfinite(around)]
            near = best[0] + _blend_slack(best[0], threshold)
            if around.size == 0 or float(around.min()) - allowance > near:
                continue
            solution, found = _refine_at(centre, np.zeros(n_tmpl, dtype=bool), *search)
            best = _lower_of(best, solution)
            minima += found
        chi2_min, pos, fitted_amps, ok, evaluated_start, _, terms, edge = best
        at_edge[:, j] = edge
        refined[j] = bool(ok)
        shift = evaluated_start + pos
        v_bary_frame, _ = _velocity_from_shift(grid, shift, frame, work.bary_pix)

        # Curvature, covariance, and the scale. The curvature is the white-noise covariance.
        # With the noise correlated along the pixel index the estimator's covariance is the
        # sandwich through it, evaluated from the model's Jacobian at the solution
        # (math.md 10.4).
        phi = phi_of.get(epoch.instrument, 0.0)
        curvature = (grid, stacks[epoch_keys[j]], work)
        cov_v, pd = _velocity_covariance(*curvature, best, amps, amp_mode, phi, frame)
        # Two templates that are the same spectrum, fitted with free amplitudes at nearly
        # equal shifts, give normal equations that are singular to rounding. The minimum
        # chi-square can then be negative and a variance of the sandwich not positive.
        # Such an epoch has no error and is flagged, as one with an indefinite curvature.
        pd = pd and chi2_min > 0.0 and bool(np.all(np.diag(cov_v) > 0.0))
        rescale = chi2_min / dof if errors == "profiled" else 1.0
        if pd:
            diag = np.diag(cov_v)
            sigma_ivar[:, j] = np.sqrt(np.clip(diag, 0.0, None))
            sigma[:, j] = sigma_ivar[:, j] * math.sqrt(rescale)
            covariance[j] = cov_v * rescale
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = cov_v / np.sqrt(np.outer(diag, diag))
            off = corr[~np.eye(n_tmpl, dtype=bool)]
            blended[j] = bool(off.size and np.any(np.abs(off) > 0.9))
        else:
            blended[j] = True
        if pd and not edge.any():
            # Another minimum that fits as well and gives different velocities: the epoch
            # does not determine them, whatever the curvature at the minimum returned.
            found = [
                (m[0], _reported_velocities(grid, templates, m[4] + m[1], frame, work.bary_pix))
                for m in minima
            ]
            returned = _reported_velocities(grid, templates, shift, frame, work.bary_pix)
            margin[j], other = _margin(chi2_min, returned, sigma[:, j], found)
            if margin[j] < _blend_slack(chi2_min, threshold):
                blended[j] = True
                # The velocities of that minimum and their covariance, on the scale of its
                # own chi-square, so that an orbit can choose between the two.
                cov_other, defined = _velocity_covariance(
                    *curvature, minima[other], amps, amp_mode, phi, frame
                )
                if defined:
                    scale_other = found[other][0] / dof if errors == "profiled" else 1.0
                    alternative[:, j] = found[other][1]
                    alternative_covariance[j] = cov_other * scale_other
        # Nothing was measured for a component whose minimum was on an edge, so its
        # velocity and uncertainty stay NaN. The diagnostics of the point evaluated are
        # kept, since they show that the epoch is at the edge rather than at a peak.
        unmeasured = at_edge[:, j]
        sigma[unmeasured, j] = np.nan
        sigma_ivar[unmeasured, j] = np.nan

        # Detection statistics with the amplitudes free, at the solution.
        b_at, gram_at, pwa_at = terms.at(pos)
        chi2_all, _, _ = _chi2_from_terms(b_at, gram_at, pwa_at, terms.pwp, terms.pwz, terms.zwz)
        for i in range(n_tmpl):
            keep = [k for k in range(n_tmpl) if k != i]
            if keep:
                chi2_without, _, _ = _chi2_from_terms(
                    b_at[keep],
                    gram_at[np.ix_(keep, keep)],
                    pwa_at[keep],
                    terms.pwp,
                    terms.pwz,
                    terms.zwz,
                )
            else:
                chi2_without = terms.null_chi2()
            delta_chi2[i, j] = max(chi2_without - chi2_all, 0.0)

        for i, t in enumerate(templates):
            if not unmeasured[i]:
                velocity[i, j] = _compose(v_bary_frame[i], t.v_zero_kms, relativistic)
        light[:, j] = fitted_amps
        chi2[j] = chi2_min
        chi2_null[j] = terms.null_chi2()
        if progress:
            print(
                f"  epoch {j:3d} ({epoch.instrument}): "
                + " ".join(
                    f"{n}={v:+.3f}+-{s:.3f}"
                    for n, v, s in zip(names, velocity[:, j], sigma[:, j], strict=True)
                )
                + f"  chi2/dof {chi2_min / dof:.3f}  [{time.perf_counter() - t0:.2f} s]"
            )

    return VelocityTable(
        names=names,
        bjd=dataset.bjd,
        instrument=tuple(instruments),
        velocity=velocity,
        sigma=sigma,
        sigma_ivar=sigma_ivar,
        covariance=covariance,
        light=light,
        light_mode={"fixed": "fixed", "free": "free per epoch", "global": "global median"}[mode],
        chi2=chi2,
        chi2_null=chi2_null,
        n_pixels=n_pixels,
        delta_chi2=delta_chi2,
        blended=blended,
        at_edge=at_edge,
        refined=refined,
        absolute=tuple(t.absolute for t in templates),
        frame=frame,
        margin=margin,
        alternative=alternative,
        alternative_covariance=alternative_covariance,
        settings={
            "v_range": ranges.tolist(),
            "coarse_step": int(coarse_step),
            "nuisance_order": nuisance_order,
            "errors": errors,
            "scale": scale,
            "noise_correlation": dict(phi_of),
            "n_parameters": int(n_par),
            "lsf_sigma_v": None if lsf_sigma_v is None else {k: v for k, v in lsf_sigma_v.items()},
            "template_sigma_kms": [t.sigma_kms for t in templates],
            "grid": {"x0": grid.x0, "dx": grid.dx, "n": grid.n},
        },
    )


# ---------------------------------------------------------------------------
# The two-dimensional surface, for plotting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TodcorSurface:
    """The two-dimensional chi-square / correlation surface of one epoch.

    Attributes
    ----------
    names
        The two component names, in axis order.
    v1, v2
        Barycentric velocity axes (km/s).
    chi2
        ``(len(v1), len(v2))`` chi-square at every pair of integer shifts.
    r_squared
        ``1 - chi2 / chi2_null`` on the same grid, the TODCOR correlation ``R^2``.
    """

    names: tuple[str, str]
    v1: np.ndarray
    v2: np.ndarray
    chi2: np.ndarray
    r_squared: np.ndarray
    chi2_null: float

    @property
    def peak(self) -> tuple[float, float]:
        """The velocities at the maximum of ``r_squared``, at integer-shift resolution."""
        i, k = np.unravel_index(
            int(np.argmin(np.where(np.isfinite(self.chi2), self.chi2, np.inf))), self.chi2.shape
        )
        return float(self.v1[i]), float(self.v2[k])


def todcor_surface(
    dataset: Dataset,
    epoch_index: int,
    templates: Sequence[Template],
    *,
    v_range=(-300.0, 300.0),
    light="free",
    lsf_sigma_v=None,
    lsf_anchors_angstrom=None,
    nuisance_order: int | None = 0,
    scale: str = "fixed",
    step: int = 1,
) -> TodcorSurface:
    """The full chi-square surface of one epoch over two templates, for plotting.

    Follows the conventions of :func:`todcor`, except that ``light`` is either ``"free"``
    or a fixed pair (there is no global pass here), and ``step`` is the integer-shift stride.
    """
    if scale not in ("fixed", "free"):
        raise ValueError(f"scale must be 'fixed' or 'free'; got {scale!r}")
    templates = list(templates)
    if len(templates) != 2:
        raise ValueError("todcor_surface draws a two-dimensional surface: pass two templates")
    grid = _templates_share_grid(templates)
    names = (templates[0].name, templates[1].name)
    ranges = _resolve_ranges(v_range, 2)
    mode, fixed = _resolve_light(light, names)
    if mode == "global":
        raise ValueError("todcor_surface takes light='free' or fixed fractions")
    epoch = dataset[epoch_index]
    stack, _ = _convolved_templates(
        templates,
        grid,
        epoch.instrument,
        lsf_sigma_v,
        lsf_anchors_angstrom,
        epoch_sigma_kms=_lsf_key(epoch, lsf_sigma_v)[1],
    )
    work = _prepare_epoch(epoch_index, epoch, grid, nuisance_order)
    bary = work.bary_pix if dataset.frame == "topocentric" else 0.0
    xi_lo = np.asarray(log_doppler_shift(ranges[:, 0], relativistic=grid.relativistic)) / grid.dx
    xi_hi = np.asarray(log_doppler_shift(ranges[:, 1], relativistic=grid.relativistic)) / grid.dx
    starts = np.ceil(xi_lo - bary).astype(int)
    ends = np.floor(xi_hi - bary).astype(int)
    n_shift = int(((ends - starts) // step).max()) + 1
    n_pad = _round_up(n_shift, _SHIFT_CHUNK)
    deltas = np.stack([starts[i] + step * np.arange(n_pad) for i in range(2)]).astype(np.int32)
    out = _epoch_terms(
        jnp.asarray(stack),
        work.rows,
        work.cols,
        work.vals,
        work.z,
        work.w,
        jnp.asarray(deltas),
        work.basis,
        chunk=_SHIFT_CHUNK,
    )
    if fixed is None:
        surface = np.asarray(_chi2_grid_free(*out))[:n_shift, :n_shift]
    else:
        surface = np.asarray(
            _chi2_grid_fixed(*out, jnp.asarray(fixed), free_scale=scale == "free")
        )[:n_shift, :n_shift]
    terms = _terms_numpy(out, n_shift)
    null = terms.null_chi2()
    axes = []
    for i, t in enumerate(templates):
        v, _ = _velocity_from_shift(grid, deltas[i, :n_shift], dataset.frame, work.bary_pix)
        axes.append(_compose(v, t.v_zero_kms, grid.relativistic))
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - surface / null
    return TodcorSurface(
        names=names, v1=axes[0], v2=axes[1], chi2=surface, r_squared=r2, chi2_null=float(null)
    )


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TodcorBatch:
    """Velocity tables for many stars, and the failures recorded per star."""

    tables: dict[str, VelocityTable]
    failures: dict[str, str]
    seconds: dict[str, float] = field(default_factory=dict)

    def write(self, directory, *, suffix: str = ".rv") -> list[Path]:
        """One table per star, ``<directory>/<star><suffix>``, plus ``failures.txt``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        written = []
        for star, table in self.tables.items():
            written.append(table.write(directory / f"{star}{suffix}", header=f"star: {star}"))
        if self.failures:
            (directory / "failures.txt").write_text(
                "\n".join(f"{star}: {why}" for star, why in self.failures.items()) + "\n",
                encoding="utf-8",
            )
        return written

    def summary(self) -> str:
        """A text report: one line per star with its usable epochs and median uncertainty."""
        lines = [f"todcor batch: {len(self.tables)} stars measured, {len(self.failures)} failed"]
        for star, table in self.tables.items():
            good = int(table.good.sum())
            meds = ", ".join(
                f"{n} {_finite_reduce(np.median, table.sigma[i]):.3f}"
                for i, n in enumerate(table.names)
            )
            lines.append(
                f"  {star}: {good}/{table.n_epochs} usable epochs, median sigma [km/s] {meds}"
                + (f", {self.seconds[star]:.1f} s" if star in self.seconds else "")
            )
        for star, why in self.failures.items():
            lines.append(f"  {star}: FAILED - {why}")
        return "\n".join(lines)


def todcor_batch(
    datasets: Mapping[str, Dataset],
    templates,
    *,
    on_error: str = "record",
    progress: bool = True,
    **kwargs,
) -> TodcorBatch:
    """Run :func:`todcor` over many stars.

    Parameters
    ----------
    datasets
        ``{star: Dataset}``.
    templates
        Either one sequence of :class:`Template` used for every star, or
        ``{star: sequence}``. Shared templates suit a survey of similar objects measured
        against a synthetic grid. The disentangling route produces per-star templates.
    on_error
        ``"record"`` (default) catches an exception in one star, records its message in
        :attr:`TodcorBatch.failures`, and continues; ``"raise"`` stops at the first.
    progress
        Print one line per star.
    **kwargs
        Passed to :func:`todcor` (``v_range``, ``light``, ``lsf_sigma_v``, and so on).
    """
    if on_error not in ("record", "raise"):
        raise ValueError("on_error must be 'record' or 'raise'")
    tables: dict[str, VelocityTable] = {}
    failures: dict[str, str] = {}
    seconds: dict[str, float] = {}
    for star, dataset in datasets.items():
        per_star = templates[star] if isinstance(templates, Mapping) else templates
        t0 = time.perf_counter()
        try:
            tables[star] = todcor(dataset, per_star, **kwargs)
        except Exception as exc:  # record the failure and continue with the next star
            if on_error == "raise":
                raise
            failures[star] = f"{type(exc).__name__}: {exc}"
            if progress:
                print(f"{star}: FAILED ({type(exc).__name__}: {exc})")
            continue
        seconds[star] = time.perf_counter() - t0
        if progress:
            table = tables[star]
            meds = " ".join(
                f"{n} {_finite_reduce(np.median, table.sigma[i]):.3f}"
                for i, n in enumerate(table.names)
            )
            print(
                f"{star}: {int(table.good.sum())}/{table.n_epochs} usable epochs, "
                f"median sigma [km/s] {meds}, {seconds[star]:.1f} s"
            )
    return TodcorBatch(tables=tables, failures=failures, seconds=seconds)

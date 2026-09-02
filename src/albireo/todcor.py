"""Epoch radial velocities of every component by N-dimensional correlation (TODCOR).

The estimator is the weighted least-squares fit of N shifted, LSF-convolved, rebinned
templates to each epoch's pixels, with the amplitudes held at declared light fractions or
profiled, and an additive low-order nuisance polynomial (``docs/math.md`` §10.1). The
epoch model is that of ``docs/math.md`` §1.4 with the component spectra given rather than
marginalized:

    y_j = 1 + sum_i a_ij R_j B_j T(delta_ij) t_i + (nuisance) + noise,   Var = 1 / w

with ``t_i`` the templates normalized to their own continua, ``T`` the shift, ``B_j`` the
instrument's line-spread function, ``R_j`` the projection onto the epoch's native pixels
(data are never resampled), and ``w`` the inverse variances with
masks folded in. For every set of shifts the amplitudes are either held or solved in
closed form; the chi-square is minimized over the integer shifts of the template grid and
then refined below a pixel.

On a uniform grid with uniform weights and free amplitudes the chi-square surface is
identical to the two-dimensional correlation of Zucker & Mazeh (1994),
``R^2 = 1 - chi^2 / |z|^2`` with ``z = y - 1``, and with a fixed light ratio to their
``R(s_1, s_2; alpha)``; the test suite pins both identities to 1e-10 (``docs/math.md``
§10.2). The three- and four-component extensions (Zucker, Torres & Mazeh 1995; Torres,
Latham & Stefanik 2007) are the same block solve with more templates. The uncertainties
are the maximum-likelihood curvature errors of Zucker (2003), rescaled by the reduced
chi-square (§10.4). The least-squares form admits masks, chip gaps, cosmic rays, per-pixel
weights, mixed instruments and mixed samplings without change to the formulae, applies
each instrument's LSF to intrinsic templates in quadrature above their own resolution, and
evaluates the chi-square exactly at fractional shifts (§10.3). The residual pixel-locking
error of the linear shift operator is of order ``0.1 / sigma_px^2`` pixels; it is below
0.01 px when the template grid samples the narrowest LSF with three or more pixels per
sigma.

Templates come from :meth:`albireo.Fit.templates` (disentangled components),
:meth:`Template.from_library` (a synthetic grid rendered at given labels) or
:meth:`Template.from_labels` (the model spectrum of a label match);
:meth:`albireo.Fit.measure_velocities` runs a fit's components back through its epochs, and
:func:`todcor_batch` measures many stars in one call. Velocities are reported barycentric
(§10.5). A disentangled component has an unidentified zero point (``docs/math.md`` §5.3,
§7.6), so velocities measured against it are differential, with one arbitrary constant per
component, unless a label match (:mod:`albireo.match`) has determined the offset;
``VelocityTable.absolute`` records the status of each component. A per-epoch table
discards the phase coherence that lets disentangling separate components whose lines never
resolve, and its accuracy is bounded by the agreement between templates and stars; where
the components are unknown, disentangling comes first.

References
----------
Zucker, S. & Mazeh, T. 1994, ApJ, 420, 806
Zucker, S., Torres, G. & Mazeh, T. 1995, ApJ, 452, 863
Torres, G., Latham, D. W. & Stefanik, R. P. 2007, ApJ, 662, 602
Zucker, S. 2003, MNRAS, 342, 1291
"""

from __future__ import annotations

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
from albireo.grids import C_KMS, LogGrid, log_doppler_shift
from albireo.operators import (
    convolve_spectrum,
    convolve_varying,
    gaussian_kernel,
    gaussian_lsf_profiles,
    rebin_operator,
    rotational_kernel,
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
        amplitude assigned by the fit is a light fraction rather than an arbitrary scale.
    sigma_kms
        Gaussian broadening already present in the template, as a sigma in km/s: zero for
        an intrinsic (deconvolved, or synthetic at infinite resolution) spectrum, the
        library's own resolution for a synthetic grid. The instrument LSF is applied in
        quadrature above it, so a template rendered at R = 20,000 is not broadened twice.
    v_zero_kms
        Velocity of the template's rest frame relative to the star's true rest frame, when
        known, so that velocities measured against it can be reported as absolute. A
        synthetic spectrum is at zero. A disentangled component is at an unknown offset,
        because its zero point is unidentified (``docs/math.md`` §5.3); ``None`` declares
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
        rotationally broadened by ``vsini_kms`` with the pixel-integrated Gray kernel, any
        fixed macroturbulence is applied as a Gaussian, and the result is placed at rest
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
            Rest-frame velocity of the rendered template. The default of zero is the
            meaning of a synthetic spectrum; ``None`` declares it unknown.
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
            kernel = np.asarray(rotational_kernel(vsini_kms / grid.dv_kms, epsilon=epsilon))
            deviation = np.convolve(deviation, kernel, mode="same")
        if macro_kms > 0.0:
            kernel = np.asarray(gaussian_kernel(macro_kms / grid.dv_kms))
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
        broadened, and shifted by the fitted ``v_kms`` so that it sits in the disentangled
        component's frame, with that shift recorded as :attr:`v_zero_kms`. Velocities
        measured against it are therefore absolute, since the label fit measures the zero
        point of the disentangled frame (``docs/math.md`` §9).

        If the match was run with ``compare="matched"`` the model already carries the
        instrument LSF, which is recorded in :attr:`sigma_kms` so that it is not applied
        twice.
        """
        if name not in match.names:
            raise ValueError(f"unknown component {name!r}; the match has {list(match.names)}")
        labels = match.labels[name]
        grid = LogGrid(
            x0=float(np.log(match.wave[0])),
            dx=float(match.problem.dx),
            n=int(match.wave.size),
            relativistic=bool(match.problem.relativistic),
        )
        deviation = np.asarray(match.template(name), dtype=np.float64) - 1.0
        sigma = 0.0
        if bool(match.problem.matched):
            kernel = np.asarray(match.problem.lsf_kernel)
            sigma = float(match.config.get("lsf_sigma_kms", 0.0)) if kernel.size > 1 else 0.0
        return cls(
            name=name,
            grid=grid,
            deviation=deviation,
            sigma_kms=sigma,
            v_zero_kms=float(labels["v_kms"]),
            meta={"source": "label match", "labels": dict(labels)},
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
    shifts, which is what makes the chi-square exact at fractional shifts), the
    template-nuisance cross terms, and the data scalars (``docs/math.md`` §10.1).

    A shifted template projected onto the native grid is one gather of the template at
    ``cols - delta`` weighted by the rebin values, so all shifts are built by one
    segment-sum per chunk of shifts; the chunking bounds the ``nnz x n_shift`` temporary.
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
        linear in the template, so every inner product at a fractional shift is a bilinear
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

    ``amps=None`` solves every amplitude; ``free_scale`` solves one overall scale on the
    given ``amps``; otherwise the amplitudes are held exactly at ``amps``.
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
        sol = np.linalg.solve(mat, rhs)
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


def _minimize_box_quadratic(g: np.ndarray, hess: np.ndarray, x0: np.ndarray, lower, upper):
    """Minimize ``g.x + x.H.x/2`` over a box by exact coordinate descent from ``x0``."""
    x = np.clip(np.asarray(x0, dtype=np.float64), lower, upper)
    n_dim = x.size
    for _ in range(60):
        moved = 0.0
        for i in range(n_dim):
            slope = g[i] + hess[i] @ x
            curv = hess[i, i]
            if curv > _EPS:
                new = x[i] - slope / curv
            else:  # flat or concave along this axis: walk downhill to the edge
                new = lower[i] if slope > 0 else upper[i]
            new = float(np.clip(new, lower[i], upper[i]))
            moved = max(moved, abs(new - x[i]))
            x[i] = new
        if moved < 1e-10:
            break
    return x


def _refine_cell(terms: _Terms, corner: np.ndarray, amps):
    """Exact minimum of the chi-square over one unit cell of the fine window.

    With the amplitudes fixed the chi-square is exactly a quadratic in the fractional
    shifts inside a cell (every term is bilinear in them), so it is reconstructed from
    ``3^N`` exact evaluations and minimized in closed form, with the box constraint
    handled by coordinate descent (``docs/math.md`` §10.3). Returns ``(chi2, position)``.
    """
    n_dim = terms.n_tmpl
    grid = np.meshgrid(*[np.array([0.0, 0.5, 1.0])] * n_dim, indexing="ij")
    frac = np.stack([x.reshape(-1) for x in grid], axis=1)
    values = np.array([terms.chi2(corner + f, amps)[0] for f in frac])
    _, g, hess = _quadratic_fit(frac, values)
    best = frac[int(np.argmin(values))]
    x = _minimize_box_quadratic(g, hess, best, np.zeros(n_dim), np.ones(n_dim))
    pos = corner + x
    return terms.chi2(pos, amps)[0], pos


def _profiled(terms: _Terms, pos, amps, mode: str):
    """The (profiled) chi-square and the amplitudes it used, in the given ``mode``."""
    if mode == "free":
        return terms.chi2(pos, None)
    if mode == "scale":
        return terms.chi2(pos, amps, free_scale=True)
    return terms.chi2(pos, amps)


def _refine(terms: _Terms, start: np.ndarray, amps, mode: str):
    """Sub-pixel minimum from the integer minimum ``start`` of the fine window.

    Every unit cell touching ``start`` is minimized exactly with the amplitudes held; when
    they are profiled (``mode`` ``"free"`` or ``"scale"``) the amplitude solve and the cell
    minimization alternate until the position settles. Falls back to ``start``, flagged as
    unrefined, if the minimum is not interior to the window.
    """
    n_dim = terms.n_tmpl
    hi = terms.n_shift - 1
    pos = start.astype(np.float64)
    if np.any(start <= 0) or np.any(start >= hi):
        chi2, a, _ = _profiled(terms, pos, amps, mode)
        return chi2, pos, a, False
    _, current, _ = _profiled(terms, pos, amps, mode)
    for _ in range(12 if mode != "fixed" else 1):
        candidates = []
        for signs in np.ndindex(*(2,) * n_dim):
            corner = np.floor(pos).astype(int) - np.array(signs)
            corner = np.clip(corner, 0, hi - 1)
            candidates.append(_refine_cell(terms, corner, current))
        chi2, new_pos = min(candidates, key=lambda c: c[0])
        if mode == "fixed":
            return chi2, new_pos, current, True
        _, new_amps, _ = _profiled(terms, new_pos, amps, mode)
        settled = np.max(np.abs(new_pos - pos)) < 1e-6
        pos, current = new_pos, new_amps
        if settled:
            break
    chi2, current, _ = _profiled(terms, pos, amps, mode)
    return chi2, pos, current, True


def _hessian(terms: _Terms, pos: np.ndarray, amps, mode: str, h: float = 0.2):
    """Curvature of the (profiled) chi-square at ``pos``, from a stencil inside its cell."""
    hi = terms.n_shift - 1
    cell_lo = np.clip(np.floor(pos), 0, hi - 1)
    cell_hi = cell_lo + 1.0
    pts = _stencil(pos, cell_lo, cell_hi, h)
    values = np.array([_profiled(terms, p, amps, mode)[0] for p in pts])
    _, _, hess = _quadratic_fit(pts - pos, values)
    return 0.5 * (hess + hess.T)


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


def _convolved_templates(
    templates: Sequence[Template],
    grid: LogGrid,
    instrument: str,
    lsf_sigma_v,
    lsf_anchors_angstrom,
) -> tuple[np.ndarray, float]:
    """Templates convolved with one instrument's LSF, and the narrowest sigma in pixels."""
    if lsf_sigma_v is None:
        stack = np.stack([t.deviation for t in templates])
        return stack, 0.0
    if instrument not in lsf_sigma_v:
        raise ValueError(
            f"no LSF width for instrument {instrument!r}; lsf_sigma_v covers "
            f"{sorted(lsf_sigma_v)}. Pass lsf_sigma_v=None only if the templates are "
            "already at the instruments' resolution."
        )
    anchors = None if lsf_anchors_angstrom is None else lsf_anchors_angstrom.get(instrument)
    rows = []
    narrowest = np.inf
    for t in templates:
        sigma = _effective_sigma(instrument, lsf_sigma_v[instrument], t)
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


@dataclass(frozen=True)
class VelocityTable:
    """Per-epoch velocities of every component, with the diagnostics that qualify them.

    Rows are epochs in the dataset's order; component axes follow ``names``. Every array
    is NumPy. Velocities are barycentric whatever frame the data were declared in, and
    absolute for a component only where ``absolute`` says so: a template whose rest frame
    is unknown (a disentangled component) yields velocities that carry that component's
    own unidentified zero point, as ``docs/math.md`` §7.6 describes for the free-velocity
    table (§10.5).

    Attributes
    ----------
    names
        Component names, from the templates.
    bjd, instrument
        Per epoch.
    velocity
        ``(n_comp, n_epochs)`` km/s.
    sigma
        ``(n_comp, n_epochs)`` km/s, the quoted uncertainty: the curvature of the
        chi-square surface at its minimum, rescaled by the reduced chi-square so that the
        noise level is estimated from the residuals rather than taken from ``ivar``; this
        is the estimator of Zucker (2003) (``docs/math.md`` §10.4). ``sigma_ivar`` is the
        same curvature with the declared weights trusted.
    covariance
        ``(n_epochs, n_comp, n_comp)`` in km/s², on the ``sigma`` scale. Its off-diagonal
        is the blending diagnostic: velocities that are highly correlated were measured
        along a ridge rather than at a peak.
    light
        ``(n_comp, n_epochs)`` amplitudes assigned to the templates: the light fractions
        as declared (``light_mode == "fixed"``), or as measured per epoch or globally.
    chi2, chi2_null, n_pixels
        The minimum chi-square, the chi-square with no template (the nuisance alone), and
        the number of pixels that carried weight.
    r_squared
        ``1 - chi2 / chi2_null``, the correlation ``R^2`` of Zucker & Mazeh (1994) at the
        maximum.
    delta_chi2
        ``(n_comp, n_epochs)``: the rise in chi-square when that component is removed and
        the rest refitted. This is the per-epoch detection statistic; it is small for a
        companion the epoch does not detect.
    blended
        Per epoch: the velocities lie on a ridge (a covariance correlation above 0.9, or a
        curvature that is not positive definite).
    at_edge
        ``(n_comp, n_epochs)``: the minimum lay at the edge of the search range, and
        ``v_range`` should be widened.
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
    settings: dict = field(default_factory=dict, repr=False)

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
        ok = self.good
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
        out["blended"] = self.blended
        out["at_edge"] = np.any(self.at_edge, axis=0)
        out["refined"] = self.refined
        return out

    def write(self, path, *, header: str = "") -> Path:
        """Write the table as whitespace-separated ASCII with a commented header."""
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
                f"{np.nanmin(self.velocity[i]):+.3f} to {np.nanmax(self.velocity[i]):+.3f} km/s"
                if ok.any()
                else "no finite velocities"
            )
            med = float(np.nanmedian(self.sigma[i])) if ok.any() else float("nan")
            lines.append(
                f"  {name}: {span}, median sigma {med:.4f} km/s, light "
                f"{np.nanmedian(self.light[i]):.3f} ({self.light_mode}); {zero}"
            )
        lines.append(
            f"  reduced chi-square: median {np.nanmedian(self.reduced_chi2):.3f} "
            f"(range {np.nanmin(self.reduced_chi2):.3f}-{np.nanmax(self.reduced_chi2):.3f}); "
            f"R^2 median {np.nanmedian(self.r_squared):.3f}"
        )
        n_blend = int(self.blended.sum())
        n_edge = int(np.any(self.at_edge, axis=0).sum())
        n_unrefined = int((~self.refined).sum())
        if n_blend or n_edge or n_unrefined:
            lines.append(
                f"  flags: {n_blend} blended, {n_edge} at the search edge, "
                f"{n_unrefined} not refined below a pixel"
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
    """Per-instrument light fractions from a free-amplitude pass: a weighted median."""
    out = {}
    instruments = sorted(set(first.instrument))
    for inst in instruments:
        sel = np.array([i == inst for i in first.instrument]) & first.good
        sel &= np.all(first.light > 0.0, axis=0) & np.all(np.isfinite(first.light), axis=0)
        sel &= np.all(first.delta_chi2 > 9.0, axis=0)
        if not sel.any():
            sel = np.array([i == inst for i in first.instrument]) & np.all(
                np.isfinite(first.light), axis=0
            )
        if not sel.any():
            raise ValueError(
                f"instrument {inst!r}: no epoch yielded usable free light fractions; pass "
                "light=<fractions> explicitly"
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
    progress: bool = False,
) -> VelocityTable:
    """Measure every component's velocity in every epoch by N-dimensional correlation.

    The estimator is the weighted least-squares fit of ``docs/math.md`` §10.1: the
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
        one per template. The template grid must extend beyond the data by this much
        (:meth:`LogGrid.covering`); otherwise a warning reports the shortfall.
    light
        Treatment of the templates' amplitudes, i.e. their light fractions.
        ``"global"`` (default) fits them freely in every epoch, takes the weighted median
        over the well-detected, unblended epochs of each instrument, and re-measures with
        them held fixed; a per-epoch light ratio is noisy, and a ratio fitted at a blended
        phase is not a measurement. ``"free"`` reports the per-epoch fit itself. A
        sequence or a ``{name: fraction}`` mapping, summing to one, holds them fixed. Fixed
        fractions are the appropriate choice when they were assumed by a disentangling
        whose components are the templates, since that is the only choice consistent with
        the definition of those components (``docs/math.md`` §9.1).
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
        The minimum found is then refined at full resolution and below a pixel.
    errors
        ``"profiled"`` (default) rescales the curvature error by the reduced chi-square,
        so that the noise level is estimated from the residuals; this is the
        maximum-likelihood estimator of Zucker (2003) and the appropriate choice when
        ``ivar`` is known only to a scale. ``"ivar"`` trusts the declared weights.
    scale
        With fixed or global light fractions, ``"fixed"`` (default) holds the composite at
        the fractions exactly, since continuum-normalized data pin its scale, while
        ``"free"`` solves one overall scale per epoch on top of the fixed ratios, which is
        the original form of TODCOR with a known light ratio (its correlation is
        scale-invariant). ``"free"`` is appropriate when the normalization is uncertain;
        the fitted scale is then the sum of the reported ``light`` row, and its departure
        from one is a normalization diagnostic. Ignored when ``light="free"``.
    progress
        Print one line per epoch.

    Returns
    -------
    VelocityTable

    Notes
    -----
    The estimator is the weighted least-squares fit, which Zucker (2003) showed to be the
    maximum-likelihood estimator, and its per-epoch error is the curvature of the
    chi-square surface (``docs/math.md`` §10.4). Two systematics lie outside that error:
    template mismatch, which mostly moves each component by a constant (the zero point),
    and the pixel-locking ripple of the linear shift operator, of order
    ``0.1 / sigma_px^2`` pixels (measured: 0.006 px at five pixels per LSF sigma, 0.03 px
    at one), which is negligible when the template grid samples the narrowest LSF with
    three or more pixels per sigma (§10.3). :meth:`albireo.Fit.templates` upsamples to
    that; a library template's grid should be built the same way.

    Velocities are barycentric. For topocentric data the shift searched is
    ``xi(v) - xi(v_bary)`` in log-wavelength (``docs/math.md`` §1.2); the composition is
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

    if mode == "global":
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
            progress,
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
        progress,
    )


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
    progress,
) -> VelocityTable:
    n_tmpl = len(templates)
    names = tuple(t.name for t in templates)
    frame = dataset.frame
    relativistic = grid.relativistic
    # How the amplitudes enter the chi-square: held, scaled together, or all solved.
    amp_mode = "free" if mode == "free" else ("scale" if scale == "free" else "fixed")

    # Templates convolved once per instrument; the narrowest sigma sets the coarse step.
    convolved: dict[str, np.ndarray] = {}
    narrowest_px = np.inf
    for inst in dataset.instruments:
        stack, sigma_px = _convolved_templates(
            templates, grid, inst, lsf_sigma_v, lsf_anchors_angstrom
        )
        convolved[inst] = stack
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
    stacks = {inst: jnp.asarray(stack) for inst, stack in convolved.items()}

    # Shift ranges in log-wavelength pixels (barycentric); composed per epoch below.
    xi_lo = np.asarray(log_doppler_shift(ranges[:, 0], relativistic=relativistic)) / grid.dx
    xi_hi = np.asarray(log_doppler_shift(ranges[:, 1], relativistic=relativistic)) / grid.dx
    bary_all = np.asarray(grid.velocity_to_pixels(dataset.v_bary), dtype=np.float64)
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
            stacks[epoch.instrument],
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
        flat = int(np.nanargmin(surface))
        coarse_idx = np.array(np.unravel_index(flat, surface.shape))
        for i in range(n_tmpl):
            at_edge[i, j] = coarse_idx[i] == 0 or coarse_idx[i] >= valid_count[i] - 1
        centre = deltas[np.arange(n_tmpl), coarse_idx]

        # Fine pass: full resolution around the coarse minimum, moved if the minimum is on its edge.
        fine_start = centre - radius
        for _attempt in range(4):
            fine = np.stack([fine_start[i] + np.arange(n_fine) for i in range(n_tmpl)]).astype(
                np.int32
            )
            out = _epoch_terms(
                stacks[epoch.instrument],
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
            fine_idx = np.array(np.unravel_index(int(np.argmin(fine_surface)), fine_surface.shape))
            interior = np.all(fine_idx > 0) & np.all(fine_idx < n_fine_raw - 1)
            if interior:
                break
            fine_start = fine_start + (fine_idx - radius)
        chi2_min, pos, fitted_amps, ok = _refine(terms, fine_idx, amps, amp_mode)
        refined[j] = bool(ok)
        shift = fine_start + pos
        v_bary_frame, total = _velocity_from_shift(grid, shift, frame, work.bary_pix)

        # Curvature, covariance, and the scale.
        hess = _hessian(terms, pos, amps, amp_mode)
        jac = _dv_dpix(grid, total)
        try:
            cov_pix = 2.0 * np.linalg.inv(hess)
            eig = np.linalg.eigvalsh(hess)
            pd = bool(np.all(eig > 0.0))
        except np.linalg.LinAlgError:
            cov_pix = np.full((n_tmpl, n_tmpl), np.nan)
            pd = False
        cov_v = jac[:, None] * cov_pix * jac[None, :]
        dof = max(work.n_good - n_par, 1)
        scale = chi2_min / dof if errors == "profiled" else 1.0
        if pd:
            diag = np.diag(cov_v)
            sigma_ivar[:, j] = np.sqrt(np.clip(diag, 0.0, None))
            sigma[:, j] = sigma_ivar[:, j] * math.sqrt(scale)
            covariance[j] = cov_v * scale
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = cov_v / np.sqrt(np.outer(diag, diag))
            off = corr[~np.eye(n_tmpl, dtype=bool)]
            blended[j] = bool(off.size and np.any(np.abs(off) > 0.9))
        else:
            blended[j] = True

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
        settings={
            "v_range": ranges.tolist(),
            "coarse_step": int(coarse_step),
            "nuisance_order": nuisance_order,
            "errors": errors,
            "scale": scale,
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
        i, k = np.unravel_index(int(np.argmin(self.chi2)), self.chi2.shape)
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

    Same conventions as :func:`todcor`, except that ``light`` is either ``"free"`` or a
    fixed pair (there is no global pass here), and ``step`` strides the integer shifts.
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
        templates, grid, epoch.instrument, lsf_sigma_v, lsf_anchors_angstrom
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
                f"{n} {np.nanmedian(table.sigma[i]):.3f}" for i, n in enumerate(table.names)
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
        against a synthetic grid; per-star templates are what the disentangling route
        produces.
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
                f"{n} {np.nanmedian(table.sigma[i]):.3f}" for i, n in enumerate(table.names)
            )
            print(
                f"{star}: {int(table.good.sum())}/{table.n_epochs} usable epochs, "
                f"median sigma [km/s] {meds}, {seconds[star]:.1f} s"
            )
    return TodcorBatch(tables=tables, failures=failures, seconds=seconds)

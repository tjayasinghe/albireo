"""Sparse linear operators: Doppler shift, interpolation, and flux-conserving rebinning.

Every operator here is *linear* in the flux vector and ships with an exact adjoint
(verified in the tests against ``jax.linear_transpose``). Conventions (``docs/math.md``
§1.1):

- Operators act on **deviation** spectra ``d = s - 1`` and zero-fill outside their input
  domain, which is exact for signals that vanish in the continuum.
- The shift operator acts on a *uniform* log-wavelength grid, where a Doppler shift is a
  translation by ``delta`` pixels; it is differentiable with respect to ``delta``.
- Grid-to-grid operators (:class:`InterpOperator`, :class:`RebinOperator`) are static:
  their sparsity pattern and weights are precomputed once in NumPy and applied as
  gathers / segment-sums in JAX. Data are never resampled — these operators project the
  *model* onto each epoch's native grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "InterpOperator",
    "RebinOperator",
    "bin_edges_from_centers",
    "convolve_spectrum",
    "convolve_varying",
    "convolve_varying_adjoint",
    "gauss_hermite_kernel_traced",
    "gaussian_kernel",
    "gaussian_kernel_traced",
    "gaussian_lsf_profiles",
    "interp_operator",
    "lsf_anchor_tables",
    "rebin_link_pair_tables",
    "rebin_operator",
    "rebin_pair_tables",
    "shift_spectrum",
    "shift_spectrum_adjoint",
]


# ---------------------------------------------------------------------------
# Doppler shift on a uniform grid
# ---------------------------------------------------------------------------


def shift_spectrum(flux, shift_pix):
    """Apply the Doppler-shift operator ``T(delta)`` on a uniform grid.

    Samples the zero-padded input at fractional positions ``p - shift_pix`` by linear
    interpolation:

    ``out[p] = (1 - f) * flux[i] + f * flux[i + 1]``,  ``i = floor(p - shift_pix)``,

    with ``flux`` treated as zero outside ``[0, n-1]``. Positive ``shift_pix`` moves
    features redward (toward higher pixel index), matching ``xi(v) > 0`` for a receding
    source.

    Parameters
    ----------
    flux
        Input spectrum on the uniform grid, shape ``(n,)``. Should be a *deviation*
        spectrum (zero in the continuum) so that zero padding is exact.
    shift_pix
        Shift in pixels (scalar). Differentiable; the derivative of the output with
        respect to ``shift_pix`` is piecewise constant (kinks at integer shifts).

    Returns
    -------
    jax.Array
        Shifted spectrum, shape ``(n,)``.
    """
    flux = jnp.asarray(flux)
    n = flux.shape[-1]
    pos = jnp.arange(n) - shift_pix
    lo = jnp.floor(pos)
    frac = pos - lo
    i0 = lo.astype(jnp.int32)
    i1 = i0 + 1
    in0 = (i0 >= 0) & (i0 < n)
    in1 = (i1 >= 0) & (i1 < n)
    g0 = jnp.where(in0, flux[jnp.clip(i0, 0, n - 1)], 0.0)
    g1 = jnp.where(in1, flux[jnp.clip(i1, 0, n - 1)], 0.0)
    return (1.0 - frac) * g0 + frac * g1


def shift_spectrum_adjoint(flux, shift_pix):
    """Apply the exact adjoint ``T(delta)^T`` (a scatter-add).

    Satisfies ``<T u, w> == <u, T^T w>`` for all ``u, w``; verified in the tests against
    ``jax.linear_transpose`` of :func:`shift_spectrum`.
    """
    flux = jnp.asarray(flux)
    n = flux.shape[-1]
    pos = jnp.arange(n) - shift_pix
    lo = jnp.floor(pos)
    frac = pos - lo
    i0 = lo.astype(jnp.int32)
    i1 = i0 + 1
    in0 = (i0 >= 0) & (i0 < n)
    in1 = (i1 >= 0) & (i1 < n)
    out = jnp.zeros_like(flux)
    out = out.at[jnp.clip(i0, 0, n - 1)].add(jnp.where(in0, (1.0 - frac) * flux, 0.0))
    out = out.at[jnp.clip(i1, 0, n - 1)].add(jnp.where(in1, frac * flux, 0.0))
    return out


# ---------------------------------------------------------------------------
# Stationary (LSF) convolution on a uniform grid
# ---------------------------------------------------------------------------


def gaussian_kernel(sigma_px, *, truncate: float = 4.0):
    """Normalized Gaussian kernel at integer pixel offsets, truncated at ±truncate·sigma.

    On the uniform log-wavelength grid a constant-resolving-power Gaussian LSF of
    velocity width ``sigma_v`` has ``sigma_px = sigma_v / grid.dv_kms``. The kernel has
    odd length ``2*radius + 1`` and sums to exactly 1.
    """
    sigma = float(sigma_px)
    if sigma <= 0:
        raise ValueError("sigma_px must be positive")
    radius = max(1, int(np.ceil(truncate * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    return jnp.asarray(kernel / kernel.sum())


def gaussian_kernel_traced(sigma_px, radius: int):
    """Normalized Gaussian kernel with a *static* radius and a traced ``sigma_px``.

    The jit-safe counterpart of :func:`gaussian_kernel` for inference over LSF
    widths: the kernel length ``2*radius + 1`` is fixed at trace time while the
    values are differentiable in ``sigma_px``. The caller must ensure
    ``radius >= truncate * sigma_px`` (build the radius from an upper bound on
    sigma) — a radius too small for the realized sigma truncates the Gaussian and
    degrades accuracy, which is why the inference model guards the bound.
    ``sigma_px`` must be positive (enforce via the prior's support).
    """
    offsets = jnp.arange(-radius, radius + 1, dtype=jnp.float64)
    kernel = jnp.exp(-0.5 * (offsets / sigma_px) ** 2)
    return kernel / jnp.sum(kernel)


def gauss_hermite_kernel_traced(sigma_px, h3, radius: int):
    """Normalized Gauss-Hermite kernel: a Gaussian with traced skewness ``h3`` (D38).

    The van der Marel & Franx (1993) series truncated at the first asymmetric term,

    ``psi(u) = exp(-u^2/2) * (1 + h3 * H3(u))``,
    ``H3(u) = (2*sqrt(2)*u^3 - 3*sqrt(2)*u) / sqrt(6)``,   ``u = offset / sigma_px``,

    normalized to unit sum. ``h3 = 0`` reproduces :func:`gaussian_kernel_traced`
    exactly. Positive ``h3`` skews the profile redward: the kernel's first moment
    (the centroid shift an unmodeled asymmetry imprints on every line) is
    ``~ sqrt(3) * h3 * sigma`` for small ``h3``. Keep ``|h3|`` modest (the inference
    model clips at 0.2): the series goes slightly negative in the far tail beyond
    that, and real instrument profiles sit well below it. Radius contract as in
    :func:`gaussian_kernel_traced` (fixed by the build-time width bound; ``h3``
    does not change the support).
    """
    offsets = jnp.arange(-radius, radius + 1, dtype=jnp.float64)
    u = offsets / sigma_px
    h3_poly = (2.0 * jnp.sqrt(2.0) * u**3 - 3.0 * jnp.sqrt(2.0) * u) / jnp.sqrt(6.0)
    kernel = jnp.exp(-0.5 * u**2) * (1.0 + h3 * h3_poly)
    return kernel / jnp.sum(kernel)


def convolve_spectrum(flux, kernel):
    """Zero-padded 'same' convolution (stationary LSF on the uniform grid).

    Linear in ``flux``; for a symmetric kernel the operator is self-adjoint, and in
    general the adjoint is convolution with the reversed kernel (verified in tests).
    Zero padding is exact for deviation spectra that vanish near the grid edges.
    """
    return jnp.convolve(jnp.asarray(flux), jnp.asarray(kernel), mode="same")


def convolve_varying(flux, profiles):
    """Row-varying 'same' convolution (wavelength-dependent LSF, D37).

    Each output pixel applies its own kernel row, in the convention of
    :func:`convolve_spectrum`:

    ``out[m] = sum_d profiles[m, d + r] * flux[m - d]``,   ``r = (w - 1) // 2``,

    so a ``profiles`` whose rows are all equal to ``kernel`` reproduces
    ``convolve_spectrum(flux, kernel)`` exactly. The matrix realized is banded,
    ``K[m, c] = profiles[m, m - c + r]`` for ``|m - c| <= r`` — the "banded matrix"
    slot design.md D8 reserved for a tabulated LSF. Zero-padded at the grid edges
    (exact for deviation spectra); linear in ``flux`` *and* in ``profiles``, so a
    traced profile bank (LSF inference) differentiates through it.
    """
    flux = jnp.asarray(flux)
    profiles = jnp.asarray(profiles)
    n = flux.shape[0]
    w = profiles.shape[1]
    r = (w - 1) // 2
    fp = jnp.pad(flux, (r, r))
    out = jnp.zeros(n, dtype=jnp.result_type(flux, profiles))
    for j in range(w):
        out = out + profiles[:, j] * jax.lax.slice(fp, (2 * r - j,), (2 * r - j + n,))
    return out


def convolve_varying_adjoint(flux, profiles):
    """Exact adjoint of :func:`convolve_varying` (same ``profiles`` argument).

    ``out[c] = sum_d profiles[c + d, d + r] * flux[c + d]`` — each *input* pixel
    scatters back through the rows that read it. Verified in the tests against
    ``jax.linear_transpose``.
    """
    flux = jnp.asarray(flux)
    profiles = jnp.asarray(profiles)
    n = flux.shape[0]
    w = profiles.shape[1]
    r = (w - 1) // 2
    fp = jnp.pad(flux, (r, r))
    pp = jnp.pad(profiles, ((r, r), (0, 0)))
    out = jnp.zeros(n, dtype=jnp.result_type(flux, profiles))
    for j in range(w):
        out = out + jax.lax.slice(pp, (j, j), (j + n, j + 1))[:, 0] * jax.lax.slice(
            fp, (j,), (j + n,)
        )
    return out


def gaussian_lsf_profiles(sigma_px, anchor_wave, grid_wave, h3=None):
    """Per-pixel LSF profiles from per-anchor Gaussian widths (build time, NumPy).

    Builds one normalized Gaussian kernel per anchor — the radius follows the
    *largest* width (truncate at 4 sigma, as :func:`gaussian_kernel`) — and linearly
    interpolates them onto the grid through :func:`lsf_anchor_tables`. Rows are convex
    combinations of unit-sum kernels, so every row sums to exactly 1. Shape
    ``(len(grid_wave), 2 * radius + 1)``, ready for :func:`convolve_varying`.
    ``sigma_px`` must supply one positive width per anchor. Optional ``h3`` (one
    skewness per anchor) makes each anchor kernel the Gauss-Hermite profile of
    :func:`gauss_hermite_kernel_traced` (D38); None or all-zero is exactly Gaussian.
    """
    sigma_px = np.atleast_1d(np.asarray(sigma_px, dtype=np.float64))
    idx, t = lsf_anchor_tables(anchor_wave, grid_wave)
    if sigma_px.shape != (len(anchor_wave),):
        raise ValueError(f"need one width per anchor; got {sigma_px.shape} for {len(anchor_wave)}")
    if np.any(sigma_px <= 0):
        raise ValueError("sigma_px must be positive")
    radius = max(1, int(np.ceil(4.0 * float(np.max(sigma_px)))))
    off = np.arange(-radius, radius + 1, dtype=np.float64)
    u = off[None, :] / sigma_px[:, None]
    bank = np.exp(-0.5 * u**2)
    if h3 is not None:
        h3 = np.atleast_1d(np.asarray(h3, dtype=np.float64))
        if h3.shape != (len(anchor_wave),):
            raise ValueError(f"need one h3 per anchor; got {h3.shape} for {len(anchor_wave)}")
        h3_poly = (2.0 * np.sqrt(2.0) * u**3 - 3.0 * np.sqrt(2.0) * u) / np.sqrt(6.0)
        bank = bank * (1.0 + h3[:, None] * h3_poly)
    bank /= bank.sum(axis=1, keepdims=True)
    return (1.0 - t)[:, None] * bank[idx] + t[:, None] * bank[idx + 1]


def lsf_anchor_tables(anchor_wave, grid_wave):
    """Per-pixel linear-interpolation tables onto LSF anchor wavelengths (build time).

    Maps each model-grid pixel to its bracketing anchor pair: the realized kernel row
    at pixel ``m`` is ``(1 - t[m]) * bank[idx[m]] + t[m] * bank[idx[m] + 1]``.
    Piecewise linear in log-wavelength (the grid's uniform coordinate) and clamped
    beyond the end anchors, so anchors need not span the grid. NumPy-only: the tables
    depend on the (static) grid and anchor positions, never on kernel values, so the
    θ-path (:func:`albireo.forward.with_lsf`) reuses them with traced banks.
    """
    a = np.asarray(anchor_wave, dtype=np.float64)
    g = np.asarray(grid_wave, dtype=np.float64)
    if a.ndim != 1 or a.size < 2:
        raise ValueError("anchor_wave must be a 1-D array of at least 2 wavelengths")
    if np.any(np.diff(a) <= 0):
        raise ValueError("anchor_wave must be strictly increasing")
    la, lg = np.log(a), np.log(g)
    idx = np.clip(np.searchsorted(la, lg, side="right") - 1, 0, a.size - 2)
    t = np.clip((lg - la[idx]) / (la[idx + 1] - la[idx]), 0.0, 1.0)
    return idx.astype(np.int32), t


# ---------------------------------------------------------------------------
# Point interpolation between arbitrary grids
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class InterpOperator:
    """Static sparse linear-interpolation operator between two fixed grids.

    Two nonzeros per output row. Output positions outside the input span get zero
    (zero-fill convention, consistent with deviation spectra). Build with
    :func:`interp_operator`.
    """

    idx0: jax.Array
    idx1: jax.Array
    w0: jax.Array
    w1: jax.Array
    n_in: int

    def __call__(self, flux):
        """Apply the operator: ``out = w0 * flux[idx0] + w1 * flux[idx1]``."""
        flux = jnp.asarray(flux)
        return self.w0 * flux[self.idx0] + self.w1 * flux[self.idx1]

    def adjoint(self, cotangent):
        """Apply the exact adjoint (scatter-add into the input grid)."""
        cotangent = jnp.asarray(cotangent)
        out = jnp.zeros(self.n_in, dtype=cotangent.dtype)
        out = out.at[self.idx0].add(self.w0 * cotangent)
        out = out.at[self.idx1].add(self.w1 * cotangent)
        return out

    # pytree protocol: index/weight arrays are children, sizes are static
    def tree_flatten(self):
        return (self.idx0, self.idx1, self.w0, self.w1), self.n_in

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, n_in=aux)


def interp_operator(x_in, x_out) -> InterpOperator:
    """Build an :class:`InterpOperator` sampling ``x_in``-gridded flux at ``x_out``.

    Parameters
    ----------
    x_in
        Strictly increasing input coordinates, shape ``(n_in,)`` (need not be uniform).
    x_out
        Output coordinates, shape ``(n_out,)``. Points outside ``[x_in[0], x_in[-1]]``
        get identically zero rows.
    """
    x_in = np.asarray(x_in, dtype=np.float64)
    x_out = np.asarray(x_out, dtype=np.float64)
    if x_in.ndim != 1 or x_in.size < 2 or np.any(np.diff(x_in) <= 0):
        raise ValueError("x_in must be 1-D, strictly increasing, with at least 2 points")
    j = np.searchsorted(x_in, x_out, side="right") - 1
    j = np.clip(j, 0, x_in.size - 2)
    t = (x_out - x_in[j]) / (x_in[j + 1] - x_in[j])
    inside = (x_out >= x_in[0]) & (x_out <= x_in[-1])
    w0 = np.where(inside, 1.0 - t, 0.0)
    w1 = np.where(inside, t, 0.0)
    return InterpOperator(
        idx0=jnp.asarray(j, dtype=jnp.int32),
        idx1=jnp.asarray(j + 1, dtype=jnp.int32),
        w0=jnp.asarray(w0),
        w1=jnp.asarray(w1),
        n_in=int(x_in.size),
    )


# ---------------------------------------------------------------------------
# Flux-conserving (pixel-integral) rebinning
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RebinOperator:
    """Static sparse pixel-integral rebinning operator (bin-average convention).

    Output pixel ``k`` with edges ``[a_k, b_k]`` receives
    ``sum_l |[a_k, b_k] ∩ [e_l, e_{l+1}]| * flux[l] / (b_k - a_k)``,
    i.e. the average of the (piecewise-constant) input flux density over the output bin.
    Conserves integrated flux exactly over fully covered ranges. Build with
    :func:`rebin_operator`.

    ``coverage[k]`` is the fraction of output bin ``k`` overlapped by the input grid
    (1 where fully covered); callers should mask pixels with ``coverage < 1``.
    """

    rows: jax.Array
    cols: jax.Array
    vals: jax.Array
    coverage: jax.Array
    n_out: int
    n_in: int

    def __call__(self, flux):
        """Apply the operator via segment-sum over the sparse entries."""
        flux = jnp.asarray(flux)
        return jax.ops.segment_sum(self.vals * flux[self.cols], self.rows, num_segments=self.n_out)

    def adjoint(self, cotangent):
        """Apply the exact adjoint (segment-sum onto the input grid)."""
        cotangent = jnp.asarray(cotangent)
        return jax.ops.segment_sum(
            self.vals * cotangent[self.rows], self.cols, num_segments=self.n_in
        )

    def tree_flatten(self):
        return (self.rows, self.cols, self.vals, self.coverage), (self.n_out, self.n_in)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, n_out=aux[0], n_in=aux[1])


def bin_edges_from_centers(centers) -> np.ndarray:
    """Bin edges from pixel centers: interior midpoints, end bins extrapolated."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 1 or centers.size < 2 or np.any(np.diff(centers) <= 0):
        raise ValueError("centers must be 1-D, strictly increasing, with at least 2 points")
    mid = 0.5 * (centers[1:] + centers[:-1])
    first = centers[0] - (mid[0] - centers[0])
    last = centers[-1] + (centers[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def rebin_operator(
    x_in=None,
    x_out=None,
    *,
    edges_in=None,
    edges_out=None,
) -> RebinOperator:
    """Build a :class:`RebinOperator` from input to output bins.

    Provide either pixel centers (``x_in``/``x_out`` — edges are taken at midpoints via
    :func:`bin_edges_from_centers`) or explicit ``edges_in``/``edges_out``.
    """
    if edges_in is None:
        if x_in is None:
            raise ValueError("provide x_in or edges_in")
        edges_in = bin_edges_from_centers(x_in)
    if edges_out is None:
        if x_out is None:
            raise ValueError("provide x_out or edges_out")
        edges_out = bin_edges_from_centers(x_out)
    edges_in = np.asarray(edges_in, dtype=np.float64)
    edges_out = np.asarray(edges_out, dtype=np.float64)
    for name, e in (("edges_in", edges_in), ("edges_out", edges_out)):
        if e.ndim != 1 or e.size < 2 or np.any(np.diff(e) <= 0):
            raise ValueError(f"{name} must be 1-D, strictly increasing, with at least 2 points")
    n_in = edges_in.size - 1
    n_out = edges_out.size - 1

    # For each output bin, the (inclusive) range of input bins it can overlap.
    lo = np.searchsorted(edges_in, edges_out[:-1], side="right") - 1
    hi = np.searchsorted(edges_in, edges_out[1:], side="left") - 1
    lo = np.clip(lo, 0, n_in - 1)
    hi = np.clip(hi, 0, n_in - 1)
    counts = hi - lo + 1

    rows = np.repeat(np.arange(n_out), counts)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    cols = lo[rows] + (np.arange(counts.sum()) - offsets[rows])

    overlap = np.minimum(edges_out[rows + 1], edges_in[cols + 1]) - np.maximum(
        edges_out[rows], edges_in[cols]
    )
    keep = overlap > 0
    rows, cols, overlap = rows[keep], cols[keep], overlap[keep]
    width_out = np.diff(edges_out)
    vals = overlap / width_out[rows]
    coverage = np.bincount(rows, weights=vals, minlength=n_out)

    return RebinOperator(
        rows=jnp.asarray(rows, dtype=jnp.int32),
        cols=jnp.asarray(cols, dtype=jnp.int32),
        vals=jnp.asarray(vals),
        coverage=jnp.asarray(coverage),
        n_out=n_out,
        n_in=n_in,
    )


def rebin_pair_tables(rebin: RebinOperator):
    """Static pair tables for the banded normal matrix ``R^T diag(w') R``.

    For every native row kappa and every ordered column pair ``(c, c + o)`` with
    ``o >= 0`` inside that row, ``H[c, c + o] += v_1 v_2 w'_kappa``. Columns within a
    rebin row are consecutive integers (guaranteed by :func:`rebin_operator`; asserted
    here), so the pairs at offset ``o`` are exactly the entry pairs ``(t, t + o)``
    with equal rows. Returns ``(pair_val, pair_sid, pair_row, h)``: static value
    products, flattened band indices into ``(n_in, h)`` (``sid = c * h + o``), the
    native row of each pair, and the row support ``h``. Per epoch, the upper band of
    ``H`` is then one ``segment_sum(pair_val * w'[pair_row], pair_sid)``.

    NumPy-only (concrete arrays): call at build time, never under trace.
    """
    rows = np.asarray(rebin.rows)
    cols = np.asarray(rebin.cols)
    vals = np.asarray(rebin.vals)
    if rows.size == 0:
        raise ValueError("empty rebin operator")
    same = rows[:-1] == rows[1:]
    if not np.all(np.diff(cols)[same] == 1):
        raise AssertionError("rebin columns are not consecutive within a row")
    counts = np.bincount(rows.astype(np.int64), minlength=int(rows.max()) + 1)
    h = int(counts.max())
    idx = np.arange(rows.size)
    pv, ps, pr = [], [], []
    for o in range(h):
        i = idx if o == 0 else idx[:-o][rows[:-o] == rows[o:]]
        pv.append(vals[i] * vals[i + o])
        ps.append(cols[i].astype(np.int64) * h + o)
        pr.append(rows[i])
    return (
        jnp.asarray(np.concatenate(pv)),
        jnp.asarray(np.concatenate(ps), dtype=jnp.int32),
        jnp.asarray(np.concatenate(pr), dtype=jnp.int32),
        h,
    )


def rebin_link_pair_tables(rebin: RebinOperator, link_row, link_gap, width: int):
    """Static pair tables for the AR(1) cross-row band of ``R^T W R``.

    An AR(1) link between native rows ``n`` and ``p = n - g`` contributes
    ``w_link (R[n]^T R[p] + R[p]^T R[n])`` to ``H`` — the symmetrized outer product of
    two *different* rebin rows, which :func:`rebin_pair_tables` (equal rows only)
    cannot express. ``(link_row, link_gap)`` lists the realized links — the union over
    epochs of ``ar_gap[e, n] == g`` — because only realized links are covered by the
    build-time ``ar_step`` bound that sizes ``width``. For every listed link and every
    ordered entry pair ``(t1 in row n, t2 in row p)``, the product ``v1 v2`` lands on
    the upper band entry ``(min(c1, c2), |c1 - c2|)``: the two orderings supply the
    two transposes of each off-diagonal entry, and coincide on the diagonal, where the
    value is doubled instead. Returns ``(link_val, link_sid, link_row, link_gap)``
    with ``sid = cmin * width + o``; per epoch the band increment is one
    ``segment_sum(link_val * wl[link_row] * (gap_row[link_row] == link_gap), link_sid)``
    — the gap test keeps each epoch's own realized links, since masks differ by epoch.

    NumPy-only (concrete arrays): call at build time, never under trace.
    """
    rows = np.asarray(rebin.rows).astype(np.int64)
    cols = np.asarray(rebin.cols).astype(np.int64)
    vals = np.asarray(rebin.vals)
    link_row = np.asarray(link_row, dtype=np.int64)
    link_gap = np.asarray(link_gap, dtype=np.int64)
    if link_row.size and (np.any(link_gap < 1) or np.any(link_row - link_gap < 0)):
        raise ValueError("links must have gap >= 1 and a non-negative partner row")

    empty = (
        jnp.zeros(0),
        jnp.zeros(0, dtype=jnp.int32),
        jnp.zeros(0, dtype=jnp.int32),
        jnp.zeros(0, dtype=jnp.int32),
    )
    if link_row.size == 0 or rows.size == 0:
        return empty
    counts = np.bincount(rows, minlength=int(rebin.n_out))
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    prev_row = link_row - link_gap
    c1 = counts[link_row]
    c2 = counts[prev_row]
    pairs = c1 * c2  # rows without rebin entries (never good, never linked) drop out
    total = int(pairs.sum())
    if total == 0:
        return empty
    k = np.repeat(np.arange(link_row.size), pairs)
    off = np.arange(total) - np.repeat(np.concatenate(([0], np.cumsum(pairs)[:-1])), pairs)
    t1 = starts[link_row[k]] + off // c2[k]
    t2 = starts[prev_row[k]] + off % c2[k]
    ca, cb = cols[t1], cols[t2]
    o = np.abs(ca - cb)
    if int(o.max()) >= width:
        raise AssertionError(
            f"link pair offset {int(o.max())} exceeds the declared band width {width} "
            "(ar_step must bound the model-pixel offset of every realized link)"
        )
    return (
        jnp.asarray(vals[t1] * vals[t2] * np.where(ca == cb, 2.0, 1.0)),
        jnp.asarray(np.minimum(ca, cb) * width + o, dtype=jnp.int32),
        jnp.asarray(link_row[k], dtype=jnp.int32),
        jnp.asarray(link_gap[k], dtype=jnp.int32),
    )

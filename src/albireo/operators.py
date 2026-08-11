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
    "interp_operator",
    "rebin_operator",
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

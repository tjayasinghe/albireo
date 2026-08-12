"""Direct band assembly of the posterior precision (no global probing).

The data term of the posterior precision is a sum of per-epoch normal matrices,

    A^T W A = sum_e S_e^T W'_e S_e,   S_e = R K sum_i l_ie T(delta_ie),

and the (i, j) component block of one epoch's term is

    l_ie l_je T(delta_ie)^T G_e T(delta_je),   G_e = K^T (R^T W'_e R) K,

a *narrow* band: G_e has half-bandwidth (s - 1) + 2r (rebin row support s, kernel
radius r) regardless of the velocities, and the T-sandwich only translates it to the
epoch's relative-shift offset and mixes 2x2 neighboring entries (linear-interpolation
tent weights). Global comb probing (``docs/math.md`` §4.2, ``solver.py``) instead pays
for the *union* of those offsets over all epochs: 2p + 1 full matvecs with
p ~ max relative shift. Assembling per epoch replaces O(p) operator applications with
O(w) banded work per epoch (w = band width ~ 2s + 4r + 3), a >10x flop reduction at
survey bandwidths — with the identical exact result (same matrix, different summation
order; agreement is verified against probing and dense construction in the tests).

Stages per epoch (all exact, all differentiable in shifts / lights / kernel / weights):

1. ``H = R^T diag(w r^2) R`` via static *pair tables* precomputed from the rebin
   sparsity at build time (``forward.build_problem``): one ``segment_sum`` per epoch.
2. ``G = K^T H K`` as two unrolled diagonal-shifted accumulations (the band-image
   form of the two convolutions; static slices only).
3. The T-sandwich: column q of ``T(delta)`` has entries at rows ``floor(q + delta)``
   (weight ``1 - frac(delta)``) and ``floor(q + delta) + 1`` (weight ``frac(delta)``),
   so each block band is a 4-term tent-weighted combination of row-translated copies
   of ``G`` — one dynamic row-gather per (component, 0/1), static column slices.
4. Accumulation into a global band tensor ``BAND[q, i, k, d]`` holding the interleaved
   band entry at row ``q * nc + i``, column offset ``k * nc + d`` — the per-epoch
   integer offset enters as a traced ``dynamic_update_slice`` start, so no scatter is
   needed. The epoch loop is a ``lax.scan`` with the band tensor as carry (buffer
   reuse; the body is rematerialized in reverse mode — recomputing one epoch's band
   is far cheaper than storing 50 of them).

The bandwidth contract is inherited from probing: entries beyond the static
half-bandwidth ``p`` are dropped (out of contract; the inference model guards the
velocity bound), and ``marginal_loglikelihood(validate=True)`` checks the assembled
matrix against the matrix-free operator.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from albireo.forward import EpochGroup, Problem
from albireo.priors import SmoothnessPrior
from albireo.solver import BlockTridiagonal

__all__ = [
    "band_block_tridiagonal",
    "prior_block_tridiagonal",
]


def _band_offsets(p: int, nc: int):
    """Static layout of the band tensor: column offset ``M = k_slot * nc + d``."""
    off0 = (p + nc - 1) // nc
    n_k = off0 + p // nc + 1
    return off0, n_k


def _epoch_band_scan(group: EpochGroup, band, n_pix: int, p: int, nc: int, remat: bool):
    """Accumulate every epoch of one group into the band tensor via ``lax.scan``."""
    off0, n_k = _band_offsets(p, nc)
    h = group.row_support
    kernel = group.kernel
    r = (kernel.shape[0] - 1) // 2
    w_h = 2 * h - 1  # full H band width
    w_u = w_h + 2 * r
    w_g = w_u + 2 * r
    h_g = h - 1 + 2 * r
    h_f = h_g + 1
    w_f = 2 * h_f + 1

    pair_val, pair_sid, pair_row = group.pair_val, group.pair_sid, group.pair_row
    wprime = group.w * group.r**2  # (n_ep, n_native)
    floors = jnp.floor(group.shifts).astype(jnp.int32)  # (n_ep, nc)
    fracs = group.shifts - floors
    q_pix = jnp.arange(n_pix)

    def epoch_g(wp):
        """G = K^T (R^T diag(w') R) K for one epoch, as a band image (n_pix, w_g + 4).

        Depends only on the (static) weights and the LSF kernel — not on the
        velocities — so it runs vmapped over epochs *outside* the accumulation scan
        and outside its rematerialized backward.
        """
        # 1. H = R^T diag(w') R, upper diagonals (n_pix, h).
        h_up = jax.ops.segment_sum(
            pair_val * wp[pair_row], pair_sid, num_segments=n_pix * h
        ).reshape(n_pix, h)

        # 2. Mirror to the full band image B_H[c, (h-1) + o] = H[c, c + o].
        b_h = jnp.zeros((n_pix, w_h))
        b_h = b_h.at[:, h - 1 :].set(h_up)
        for o in range(1, h):
            b_h = b_h.at[o:, h - 1 - o].set(h_up[: n_pix - o, o])

        # 3. G = K^T H K as two unrolled shifted accumulations.
        bhp = jnp.pad(b_h, ((r, r), (0, 0)))
        u = jnp.zeros((n_pix, w_u))
        for d1 in range(-r, r + 1):
            u = u.at[:, r + d1 : r + d1 + w_h].add(
                kernel[d1 + r] * jax.lax.slice(bhp, (r + d1, 0), (r + d1 + n_pix, w_h))
            )
        g = jnp.zeros((n_pix, w_g))
        for d2 in range(-r, r + 1):
            g = g.at[:, r - d2 : r - d2 + w_u].add(kernel[d2 + r] * u)
        return jnp.pad(g, ((0, 0), (2, 2)))

    gp_all = jax.vmap(epoch_g)(wprime)

    def epoch_body(band, xs):
        gp, light, floor_e, frac_e = xs

        # 4. Row-translated copies for the T-sandwich, zero-filled outside the grid.
        # A translation's adjoint is the opposite translation, so implement it as a
        # clip-safe dynamic_slice of a zero-padded copy (pads of n_pix rows make any
        # clamped start land the window in an all-zero region, reproducing zero-fill
        # exactly for arbitrary shifts) — reverse mode then costs a contiguous copy
        # instead of the general scatter that a gather-based translation would pay.
        gpp = jnp.pad(gp, ((n_pix, n_pix), (0, 0)))
        w_gp = w_g + 4

        def shift_rows(start):
            s = jnp.clip(start + n_pix, 0, gpp.shape[0] - n_pix).astype(jnp.int32)
            return jax.lax.dynamic_slice(gpp, (s, jnp.int32(0)), (n_pix, w_gp))

        g_rows = [[shift_rows(floor_e[i] + a) for a in (0, 1)] for i in range(nc)]
        t_idx = jnp.arange(w_f) - h_f

        for i in range(nc):
            w_i = (1.0 - frac_e[i], frac_e[i])
            for j in range(nc):
                w_j = (1.0 - frac_e[j], frac_e[j])
                delta = floor_e[i] - floor_e[j]
                f = jnp.zeros((n_pix, w_f))
                for a in (0, 1):
                    for b in (0, 1):
                        f = f + (w_i[a] * w_j[b]) * jax.lax.slice(
                            g_rows[i][a], (0, 1 + b - a), (n_pix, 1 + b - a + w_f)
                        )
                f = f * (light[i] * light[j])
                # Column validity (T columns exist only on the grid) and band contract.
                col = q_pix[:, None] + delta + t_idx[None, :]
                m_off = (delta + t_idx) * nc + (j - i)
                f = jnp.where((col >= 0) & (col < n_pix) & (jnp.abs(m_off) <= p)[None, :], f, 0.0)
                d = (j - i) % nc
                s0 = off0 + delta - h_f + ((j - i) - d) // nc
                start = jnp.clip(s0, 0, n_k - w_f).astype(jnp.int32)
                idx = (jnp.int32(0), jnp.int32(i), start, jnp.int32(d))
                band = jax.lax.dynamic_update_slice(
                    band,
                    jax.lax.dynamic_slice(band, idx, (n_pix, 1, w_f, 1)) + f[:, None, :, None],
                    idx,
                )
        return band, None

    if remat:
        epoch_body = jax.checkpoint(epoch_body)
    band, _ = jax.lax.scan(epoch_body, band, (gp_all, group.light, floors, fracs))
    return band


def _prior_diagonals(prior: SmoothnessPrior, n_pix: int):
    """Analytic diagonals of ``tau * D2^T D2 + eta * I`` per component: (d0, d1, d2).

    ``D2^T D2`` is pentadiagonal Toeplitz with additive boundary corrections
    (main diagonal 6 with ends 1, 5; first diagonal -4 with ends -2; second
    diagonal 1). Verified against :meth:`SmoothnessPrior.dense` in the tests.
    """
    d0 = jnp.full(n_pix, 6.0).at[0].add(-5.0).at[1].add(-1.0).at[-1].add(-5.0).at[-2].add(-1.0)
    d1 = jnp.full(n_pix - 1, -4.0).at[0].add(2.0).at[-1].add(2.0)
    d2 = jnp.ones(n_pix - 2)
    tau = prior.tau[:, None]
    eta = prior.eta[:, None]
    return tau * d0[None, :] + eta, tau * d1[None, :], tau * d2[None, :]


def _add_prior_band(band, prior: SmoothnessPrior, n_pix: int, p: int, nc: int):
    """Add the (component-diagonal, pentadiagonal) prior band to the band tensor."""
    off0, _ = _band_offsets(p, nc)
    d0, d1, d2 = _prior_diagonals(prior, n_pix)
    band = band.at[:, :, off0, 0].add(d0.T)
    band = band.at[: n_pix - 1, :, off0 + 1, 0].add(d1.T)
    band = band.at[1:, :, off0 - 1, 0].add(d1.T)
    band = band.at[: n_pix - 2, :, off0 + 2, 0].add(d2.T)
    band = band.at[2:, :, off0 - 2, 0].add(d2.T)
    return band


def _pack_band(band, n_pix: int, nc: int, p: int, block_size: int) -> BlockTridiagonal:
    """Gather the band tensor into :class:`BlockTridiagonal` (identity pad block)."""
    off0, n_k = _band_offsets(p, nc)
    n = nc * n_pix
    bs = block_size
    k_blocks = max(1, -(-n // bs))
    row = jnp.arange(bs)[:, None]
    col = jnp.arange(bs)[None, :]

    def value_at(rr, cc):
        m = cc - rr
        q = rr // nc
        i = rr % nc
        d = m % nc
        slot = off0 + (m - d) // nc
        valid = (jnp.abs(m) <= p) & (rr < n) & (cc < n) & (slot >= 0) & (slot < n_k)
        v = band[
            jnp.clip(q, 0, n_pix - 1),
            i,
            jnp.clip(slot, 0, n_k - 1),
            d,
        ]
        pad_eye = ((rr == cc) & (rr >= n)).astype(band.dtype)
        return jnp.where(valid, v, 0.0) + pad_eye

    def blocks_at(kk):
        diag_k = value_at(kk * bs + row, kk * bs + col)
        lower_k = value_at((kk + 1) * bs + row, kk * bs + col)
        return diag_k, lower_k

    diag, lower_all = jax.lax.map(blocks_at, jnp.arange(k_blocks))
    lower = lower_all[:-1] if k_blocks > 1 else lower_all[:0]
    return BlockTridiagonal(diag=diag, lower=lower, n=n)


def band_block_tridiagonal(
    problem: Problem,
    prior: SmoothnessPrior,
    half_bandwidth: int,
    block_size: int | None = None,
    *,
    remat: bool = True,
) -> BlockTridiagonal:
    """Assemble the posterior precision ``Lambda_p + A^T W A`` by direct band assembly.

    Drop-in replacement for probing the full operator: returns the same
    :class:`BlockTridiagonal` (to floating-point reordering) at a fraction of the
    cost. ``half_bandwidth`` is the *per-component* bound ``b_nat`` (as in
    :func:`albireo.likelihood.marginal_loglikelihood`); the stacked bandwidth is
    ``p = nc * b_nat + nc - 1``.

    Parameters
    ----------
    problem
        Output of :func:`albireo.forward.build_problem` (pair tables included).
    prior
        Spectral prior (one component per problem component).
    half_bandwidth
        Static per-component half-bandwidth bound ``b_nat``.
    block_size
        Solver block size ``B >= p`` (default ``p``).
    remat
        Rematerialize the per-epoch band in reverse mode (default True: one epoch's
        band is cheap to recompute and expensive to store 50 times).
    """
    nc, n_pix = problem.n_components, problem.grid.n
    p = nc * int(half_bandwidth) + nc - 1
    bs = int(block_size) if block_size is not None else max(p, 8)
    if bs < p:
        raise ValueError(f"block_size ({bs}) must be >= half_bandwidth ({p})")
    _, n_k = _band_offsets(p, nc)
    band = jnp.zeros((n_pix, nc, n_k, nc))
    for g in problem.groups:
        band = _epoch_band_scan(g, band, n_pix, p, nc, remat)
    band = _add_prior_band(band, prior, n_pix, p, nc)
    return _pack_band(band, n_pix, nc, p, bs)


def prior_block_tridiagonal(
    prior: SmoothnessPrior, n_pix: int, nc: int, block_size: int
) -> BlockTridiagonal:
    """The prior precision alone as :class:`BlockTridiagonal` (analytic diagonals)."""
    p = 2 * nc
    _off0, n_k = _band_offsets(p, nc)
    band = jnp.zeros((n_pix, nc, n_k, nc))
    band = _add_prior_band(band, prior, n_pix, p, nc)
    return _pack_band(band, n_pix, nc, p, block_size)

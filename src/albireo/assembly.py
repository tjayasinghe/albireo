"""Direct band assembly of the posterior precision (no global probing).

The data term of the posterior precision is a sum of per-epoch normal matrices,

    A^T W A = sum_e S_e^T W'_e S_e,   S_e = R K sum_i l_ie T(delta_ie),

and the (i, j) component block of one epoch's term is

    l_ie l_je T(delta_ie)^T G_e T(delta_je),   G_e = K^T (R^T W'_e R) K,

a narrow band: G_e has half-bandwidth (s - 1) + 2r (rebin row support s, kernel radius
r) regardless of the velocities, and the T-sandwich only translates it to the epoch's
relative-shift offset and mixes 2x2 neighboring entries (linear-interpolation tent
weights). Global comb probing (``docs/math.md`` §4.2, ``solver.py``) instead pays
for the union of those offsets over all epochs: 2p + 1 full matvecs with p ~ max
relative shift. Assembling per epoch replaces O(p) operator applications with O(w)
banded work per epoch (w = band width ~ 2s + 4r + 3), a >10x flop reduction at survey
bandwidths, for the identical exact result (same matrix, different summation order;
agreement is verified against probing and dense construction in the tests).

Stages per epoch (all exact, all differentiable in shifts, lights, kernel and weights):

1. ``H = R^T W' R`` via static pair tables precomputed from the rebin sparsity at
   build time (``forward.build_problem``): one ``segment_sum`` per epoch for the
   diagonal part ``diag(w r^2)``, plus, when the problem carries AR(1) correlated
   noise (``forward.with_ar1``), a second ``segment_sum`` over cross-row link tables
   for the tridiagonal chain terms, which widen ``H`` by the group's static
   ``ar_step`` (``operators.rebin_link_pair_tables``; the traced link weights carry
   phi, so the whole band stays differentiable in it).
2. ``G = K^T H K`` as the band-image form of the two convolutions. The first
   application translates rows, so it stays an unrolled accumulation over the 2r + 1
   taps (static slices only); the second translates columns alone, which makes it a
   contraction of the band image against a static ``(w_u, w_g)`` banded matrix, one
   GEMM rather than 2r + 1 further read-modify-write passes over the widest image in
   the assembly. As elsewhere in this module the equality holds up to summation
   order (measured 0.5 ulp; XLA does not promise a GEMM's accumulation order, although
   it matched the loop's exactly on the benchmark configurations). A
   wavelength-dependent LSF (``forward.build_problem(lsf_anchors_angstrom=...)``)
   keeps the identical structure with the scalar kernel taps replaced by row-shifted
   profile columns; there both applications translate rows, so both stay loops, and
   the second runs against the band-transpose of the first, since only left
   applications broadcast on a row-major band image and ``G`` is symmetric,
   ``G = K^T (K^T H)^T``. ``G`` is a matrix on the model grid, so its off-grid columns,
   which the kernel populates by smearing in-grid mass outward, are masked; the
   sandwich would otherwise read them whenever a shift places a component's support
   against a grid edge. Being velocity-independent, this whole stage is a pre-pass
   over epochs, batched by ``epoch_chunk`` (memory only; see
   :func:`_epoch_chunk_default`).
3. The T-sandwich: column q of ``T(delta)`` has entries at rows ``floor(q + delta)``
   (weight ``1 - frac(delta)``) and ``floor(q + delta) + 1`` (weight ``frac(delta)``),
   so each block band is a 4-term tent-weighted combination of row-translated copies
   of ``G``: one dynamic row-gather per (component, 0/1), with static column slices.
4. Accumulation into a global band tensor ``BAND[q, i, k, d]`` holding the interleaved
   band entry at row ``q * nc + i``, column offset ``k * nc + d``. The per-epoch
   integer offset enters as a traced ``dynamic_update_slice`` start, so no scatter is
   needed. The update is ``band + place(f)``, the identity in ``band``, but reverse
   mode reassembles that identity out of three whole-tensor passes unless told
   otherwise, so it goes through the closed-form :func:`_band_accumulate` (3.5 s
   of a 5.9 s backward at the benchmark ladder's first row). The epoch loop is a
   ``lax.scan`` with the band tensor as carry (buffer reuse; the body is
   rematerialized in reverse mode, since recomputing one epoch's band is much cheaper
   than storing 50 of them).

The bandwidth contract is inherited from probing: entries beyond the static
half-bandwidth ``p`` are dropped (out of contract; the inference model guards the
velocity bound), and ``marginal_loglikelihood(validate=True)`` checks the assembled
matrix against the matrix-free operator.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from albireo.forward import EpochGroup, Problem, ar1_band_weights
from albireo.priors import SmoothnessPrior
from albireo.solver import BlockTridiagonal

__all__ = [
    "band_block_tridiagonal",
    "prior_block_tridiagonal",
    "prior_logdet",
]

# Keep every epoch's G live while the pre-pass is under this; batch it beyond.
_GP_HOIST_BYTES = 1024**3
# Target live bytes for one batch of G once batching kicks in.
_GP_CHUNK_BYTES = 512 * 1024**2


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _band_accumulate(band, f, start, comp: int, d: int):
    """``band`` with ``f`` added at ``[:, comp, start : start + f.shape[1], d]``.

    Mathematically ``band + place(f)``: linear in both arguments and the identity in
    ``band``. The forward is one in-place slice update either way; the ``custom_vjp``
    is there for the reverse pass. Written as nested dynamic slices, reverse mode
    transposes the two primitives separately and reassembles the identity as

        ``band_bar = dus(out_bar, 0, idx) + dus(zeros_like(band), ds(out_bar, idx), idx)``

    which is three passes over the whole band tensor to reproduce its own input, once
    per (i, j) block per epoch. At the benchmark ladder's first row that is 4 x 50 x 3
    passes over 522 MB = 313 GB of traffic, measured at 3.5 s of a 5.9 s backward, and
    it grows with the band tensor rather than with the slice touched (D49). The closed
    form here is exact: the operand cotangent is the output cotangent, and ``f``'s is
    the corresponding slice of it. Values and gradients are bit-identical to the
    nested-slice route (``test_assembly.py``).

    ``comp`` and ``d`` are Python ints (the component and interleave-offset loop
    counters), hence static; only ``start`` is traced, and being an integer it
    takes a ``float0`` cotangent.
    """
    idx = (jnp.int32(0), jnp.int32(comp), start, jnp.int32(d))
    return jax.lax.dynamic_update_slice(
        band,
        jax.lax.dynamic_slice(band, idx, (band.shape[0], 1, f.shape[1], 1)) + f[:, None, :, None],
        idx,
    )


def _band_accumulate_fwd(band, f, start, comp: int, d: int):
    # Recompute inline rather than calling _band_accumulate: the forward trace must
    # hold only plain operations, so that a second reverse differentiation
    # (jacrev-of-jacrev, as in laplace_inverse_mass) walks an ordinary graph instead
    # of re-entering the custom boundary. Calling the custom function here is
    # first-order exact but returns the transpose of the true Hessian (measured by
    # test_second_order_reverse_matches_plain_autodiff): the same re-entry defect D28
    # found in _solve_stage_fwd.
    idx = (jnp.int32(0), jnp.int32(comp), start, jnp.int32(d))
    out = jax.lax.dynamic_update_slice(
        band,
        jax.lax.dynamic_slice(band, idx, (band.shape[0], 1, f.shape[1], 1)) + f[:, None, :, None],
        idx,
    )
    return out, (band.shape[0], f.shape[1], start)


def _band_accumulate_bwd(comp: int, d: int, res, g):
    n_pix, w_f, start = res
    idx = (jnp.int32(0), jnp.int32(comp), start, jnp.int32(d))
    f_bar = jax.lax.dynamic_slice(g, idx, (n_pix, 1, w_f, 1))[:, 0, :, 0]
    return g, f_bar, np.zeros((), dtype=jax.dtypes.float0)


_band_accumulate.defvjp(_band_accumulate_fwd, _band_accumulate_bwd)


def _band_offsets(p: int, nc: int):
    """Static layout of the band tensor: column offset ``M = k_slot * nc + d``."""
    off0 = (p + nc - 1) // nc
    n_k = off0 + p // nc + 1
    return off0, n_k


def _epoch_chunk_default(n_ep: int, n_pix: int, w_gp: int, n_groups: int = 1) -> int:
    """Epochs per G batch: hoist the whole pre-pass while it is cheap, else batch it.

    The velocity-independent stage ``G_e`` is computed once per epoch either way, so any
    chunking below ``n_ep`` costs exactly one extra G pass in the backward (the chunk
    body is rematerialized); the chunk size only trades live bytes against ``vmap``
    width. Hence the two-regime policy: keep every epoch's G live while that is under
    ``_GP_HOIST_BYTES``, and otherwise batch to about ``_GP_CHUNK_BYTES``, which at the
    design target reduces 4.5 GB of ``gp_all`` to under 0.5 GB and is the difference
    between a gradient that fits in 32 GB and one that does not.

    The budgets are shared out over ``n_groups``, because the group loop is unrolled into
    a single jit graph: every group's pre-pass is live in the same buffer-assignment plan,
    so a per-group budget would be multiplied by the group count. That count is 1 for
    simulations and for any pipeline delivering one wavelength solution, but real archival
    data routinely gives one group per exposure (:func:`albireo.forward._epoch_groups`),
    where the un-divided budget was measured at 40 GB against the 11 GB a shared grid
    needs.
    """
    per_epoch = n_pix * w_gp * 8
    groups = max(1, int(n_groups))
    if n_ep * per_epoch * groups <= _GP_HOIST_BYTES:
        return n_ep
    return max(1, min(n_ep, _GP_CHUNK_BYTES // groups // max(per_epoch, 1)))


def _epoch_band_scan(
    group: EpochGroup,
    band,
    n_pix: int,
    p: int,
    nc: int,
    remat: bool,
    chunk: int | None,
    correlated: bool = False,
    n_groups: int = 1,
):
    """Accumulate every epoch of one group into the band tensor via ``lax.scan``."""
    off0, n_k = _band_offsets(p, nc)
    h = group.row_support
    kernel = group.kernel
    r = (kernel.shape[-1] - 1) // 2
    # Static structural branch (D37): one bank row = stationary kernel, scalar taps;
    # a full per-pixel bank = wavelength-dependent LSF, row-indexed taps.
    varying = kernel.shape[0] > 1
    # The AR(1) chain couples rebin rows across links, so H widens by the group's
    # static ar_step (Problem.natural_half_bandwidth reserves the same amount).
    h_eff = h + group.ar_step if correlated else h
    w_h = 2 * h_eff - 1  # full H band width
    w_u = w_h + 2 * r
    w_g = w_u + 2 * r
    h_g = h_eff - 1 + 2 * r
    h_f = h_g + 1
    w_f = 2 * h_f + 1
    if n_k < w_f:
        raise ValueError(
            f"half_bandwidth too small for instrument {group.instrument!r}: the band has "
            f"{n_k} slots but one epoch's block spans {w_f}. The per-component bound must be "
            f"at least {h_eff + 2 * r} (rebin row support {h}"
            + (f" + AR(1) step {group.ar_step}" if correlated else "")
            + f", kernel radius {r}) even at zero "
            "relative shift, and in general that plus the maximum relative shift in pixels "
            "; use Problem.half_bandwidth_bound."
        )

    pair_val, pair_sid, pair_row = group.pair_val, group.pair_sid, group.pair_row
    weights: tuple[jax.Array, ...]
    if correlated:
        wp_all, wl_all = ar1_band_weights(group)  # jitter folded into both
        weights = (wp_all, wl_all, group.ar_gap)
        link_val, link_sid = group.link_val, group.link_sid
        link_row, link_gap = group.link_row, group.link_gap
    else:
        weights = (group.effective_w * group.r**2,)  # (n_ep, n_native); jitter folded in
    floors = jnp.floor(group.shifts).astype(jnp.int32)  # (n_ep, nc)
    fracs = group.shifts - floors
    q_pix = jnp.arange(n_pix, dtype=jnp.int32)

    # G lives on the model grid, so entry (x, y) exists only for y in [0, n_pix). The
    # band image is built by convolving H along its columns, which writes entries at
    # |y| beyond the grid: H itself is clean there (no rebin pairs), but the kernel
    # smears in-grid mass outward. Those phantom columns are read by the T-sandwich
    # whenever an epoch's shift places a component's support against a grid edge, so
    # mask them here, once, for every epoch. Rows are already zero-filled by the
    # translation in `shift_rows`; this is the same guard on the other index.
    # (Fixtures whose native grid stops short of the model grid never see it: the
    # weights vanish there, so the phantom entries are multiplied by zero.)
    gp_offsets = jnp.arange(w_g + 4, dtype=jnp.int32) - jnp.int32(h_g + 2)
    y_of = q_pix[:, None] + gp_offsets[None, :]
    gp_valid = (y_of >= 0) & (y_of < n_pix)

    def epoch_g(ws):
        """G = K^T (R^T W' R) K for one epoch, as a band image (n_pix, w_g + 4).

        Depends only on the weights and the LSF kernel, not on the velocities, so it
        is computed in a ``vmap``ped pre-pass rather than inside the accumulation
        body. How many epochs' worth are kept live at once is the ``chunk`` policy of
        :func:`_epoch_chunk_default`. The weights are velocity-independent but still
        traced: jitter, response and phi all flow through them.
        """
        # 1. H = R^T W' R, upper diagonals (n_pix, h_eff): the diagonal part through
        # the equal-row pair tables, plus, on a correlated problem, one symmetrized
        # cross-row term per AR(1) link through the link tables. The gap test keeps
        # each epoch's own realized links: the tables hold the union over epochs,
        # because masks differ by epoch.
        h_up = jax.ops.segment_sum(
            pair_val * ws[0][pair_row], pair_sid, num_segments=n_pix * h
        ).reshape(n_pix, h)
        if correlated:
            wl, gap_e = ws[1], ws[2]
            lw = wl[link_row] * (gap_e[link_row] == link_gap)
            h_up = (
                jax.ops.segment_sum(link_val * lw, link_sid, num_segments=n_pix * h_eff)
                .reshape(n_pix, h_eff)
                .at[:, :h]
                .add(h_up)
            )

        # 2. Mirror to the full band image B_H[c, (h_eff-1) + o] = H[c, c + o].
        b_h = jnp.zeros((n_pix, w_h))
        b_h = b_h.at[:, h_eff - 1 :].set(h_up)
        for o in range(1, h_eff):
            b_h = b_h.at[o:, h_eff - 1 - o].set(h_up[: n_pix - o, o])

        # 3. G = K^T H K. Stationary: two unrolled shifted accumulations with scalar
        # kernel taps (the v1 path, unchanged). Wavelength-dependent (D37): the left
        # application K^T M generalizes tap by tap, the scalar becoming a row-shifted
        # profile column, K[c + d, c] = P[c + d, d + r], but a right application's
        # taps would vary along the band image's columns, which the row-major layout
        # cannot broadcast. H (hence G) is symmetric, so compute G = K^T (K^T H)^T
        # instead: band-transpose U = K^T H (a static column-slice shuffle) and run
        # the same left application once more.
        if not varying:
            bhp = jnp.pad(b_h, ((r, r), (0, 0)))
            u = jnp.zeros((n_pix, w_u))
            for d1 in range(-r, r + 1):
                u = u.at[:, r + d1 : r + d1 + w_h].add(
                    kernel[0, d1 + r] * jax.lax.slice(bhp, (r + d1, 0), (r + d1 + n_pix, w_h))
                )
            # The second application translates columns only (unlike the first, it has
            # no row shift), so it is a contraction of the band image against a static
            # (w_u, w_g) banded matrix instead of 2r + 1 accumulate passes over the
            # whole image. The loop form re-read and rewrote the (n_pix, w_g) output
            # once per tap, the largest block of memory traffic in the assembly at
            # survey scale (measured 0.55 s of a 0.88 s G pre-pass at the benchmark
            # ladder's first row, against 0.03 s here; D49). Column k of the output
            # collects tap s = k' + 2r - k. Equality is to summation order, as
            # everywhere else in this module: increasing k' is increasing s, so the
            # ideal orders agree, but XLA may block a GEMM's accumulation as it likes
            # (measured 0.5 ulp against the loop on a random kernel, and bit-identical
            # log-likelihoods on the benchmark configurations).
            tap = jnp.arange(w_u)[:, None] + 2 * r - jnp.arange(w_g)[None, :]
            g = u @ jnp.where((tap >= 0) & (tap <= 2 * r), kernel[0, jnp.clip(tap, 0, 2 * r)], 0.0)
        else:
            kp = jnp.pad(kernel, ((r, r), (0, 0)))

            def kt_left(img, w_in):
                """(K^T M) band image from M's, widening w_in by 2r (zero-fill rows)."""
                imgp = jnp.pad(img, ((r, r), (0, 0)))
                out = jnp.zeros((n_pix, w_in + 2 * r))
                for d in range(-r, r + 1):
                    tap = jax.lax.slice(kp, (r + d, d + r), (r + d + n_pix, d + r + 1))
                    out = out.at[:, r + d : r + d + w_in].add(
                        tap * jax.lax.slice(imgp, (r + d, 0), (r + d + n_pix, w_in))
                    )
                return out

            u = kt_left(b_h, w_h)
            hw_u = (w_u - 1) // 2
            up = jnp.pad(u, ((hw_u, hw_u), (0, 0)))
            ut = jnp.stack(
                [
                    jax.lax.slice(up, (j, w_u - 1 - j), (j + n_pix, w_u - j))[:, 0]
                    for j in range(w_u)
                ],
                axis=1,
            )
            g = kt_left(ut, w_u)
        return jnp.where(gp_valid, jnp.pad(g, ((0, 0), (2, 2))), 0.0)

    def epoch_body(band, xs):
        gp, light, floor_e, frac_e = xs

        # 4. Row-translated copies for the T-sandwich, zero-filled outside the grid.
        # A translation's adjoint is the opposite translation, so implement it as a
        # clip-safe dynamic_slice of a zero-padded copy (pads of n_pix rows make any
        # clamped start land the window in an all-zero region, reproducing zero-fill
        # exactly for arbitrary shifts). Reverse mode then costs a contiguous copy
        # instead of the general scatter that a gather-based translation would pay.
        gpp = jnp.pad(gp, ((n_pix, n_pix), (0, 0)))
        w_gp = w_g + 4

        def shift_rows(start):
            s = jnp.clip(start + n_pix, 0, gpp.shape[0] - n_pix).astype(jnp.int32)
            return jax.lax.dynamic_slice(gpp, (s, jnp.int32(0)), (n_pix, w_gp))

        g_rows = [[shift_rows(floor_e[i] + a) for a in (0, 1)] for i in range(nc)]
        # int32 throughout: under x64 the default integer is int64, and the column
        # index below is an (n_pix, w_f) array whose only use is a comparison.
        t_idx = jnp.arange(w_f, dtype=jnp.int32) - jnp.int32(h_f)

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
                band = _band_accumulate(band, f, start, i, d)
        return band, None

    n_ep = group.shifts.shape[0]
    if chunk is None:
        chunk = _epoch_chunk_default(n_ep, n_pix, w_g + 4, n_groups)
    chunk = max(1, min(int(chunk), n_ep))

    if chunk >= n_ep:
        # Hoisted: every epoch's G stays live, so the rematerialized backward never
        # recomputes it. Cheapest in time, most expensive in memory.
        gp_all = jax.vmap(epoch_g)(weights)
        body = jax.checkpoint(epoch_body) if remat else epoch_body
        band, _ = jax.lax.scan(body, band, (gp_all, group.light, floors, fracs))
        return band

    # Batched: only one chunk's G is live. The batch (not the epoch) is the unit of
    # rematerialization, so the backward recomputes each epoch's G exactly once.
    # Epochs padding the last batch carry zero weight, hence G = 0 and no contribution.
    n_chunks = -(-n_ep // chunk)
    pad = n_chunks * chunk - n_ep
    zpad = ((0, pad), (0, 0))
    weights_c = tuple(jnp.pad(a, zpad) for a in weights)
    light_c = jnp.pad(group.light, zpad)
    floor_c = jnp.pad(floors, zpad)
    frac_c = jnp.pad(fracs, zpad)

    def chunk_body(band, xs):
        ws_b, light_b, floor_b, frac_b = xs
        band, _ = jax.lax.scan(
            epoch_body, band, (jax.vmap(epoch_g)(ws_b), light_b, floor_b, frac_b)
        )
        return band, None

    if remat:
        chunk_body = jax.checkpoint(chunk_body)

    def batched(a):
        return a.reshape(n_chunks, chunk, *a.shape[1:])

    band, _ = jax.lax.scan(
        chunk_body,
        band,
        (jax.tree.map(batched, weights_c), batched(light_c), batched(floor_c), batched(frac_c)),
    )
    return band


def _prior_diagonals(prior: SmoothnessPrior, n_pix: int):
    """Analytic diagonals of ``D2^T diag(t) D2 + diag(e)`` per component: (d0, d1, d2).

    With row weights ``t_k`` (row ``k`` of ``D2`` spans pixels ``k, k+1, k+2``) the
    pentadiagonal entries are

        ``d0_a = t_a + 4 t_{a-1} + t_{a-2}``,
        ``d1_a = -2 (t_a + t_{a-1})``,
        ``d2_a = t_a``,

    with ``t`` read as zero outside ``[0, n_pix - 3]``, which is where the boundary
    corrections come from. Uniform weights recover the Toeplitz form (main diagonal 6
    with ends 1, 5; first diagonal -4 with ends -2; second diagonal 1). The per-pixel
    profiles of D40 enter entirely through ``t`` and ``e``
    (:meth:`SmoothnessPrior.curvature_weights` / :meth:`~SmoothnessPrior.ridge_weights`),
    so nothing downstream changes. Verified against :meth:`SmoothnessPrior.dense` in
    the tests.
    """
    t = prior.curvature_weights(n_pix)  # (nc, n_pix - 2)
    # Pad by 2 on each side so the three shifted reads below are plain slices: with
    # p = pad(t, 2), p[2 + a] is t_a and p[2 + a - s] is t_{a-s}, zero out of range.
    p = jnp.pad(t, ((0, 0), (2, 2)))
    d0 = p[:, 2 : 2 + n_pix] + 4.0 * p[:, 1 : 1 + n_pix] + p[:, 0:n_pix]
    d1 = -2.0 * (p[:, 2 : 1 + n_pix] + p[:, 1:n_pix])
    return d0 + prior.ridge_weights(n_pix), d1, t


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

    # The column offset m = cc - rr, and therefore the band's (slot, d) coordinates and
    # the bandwidth mask, depend only on the within-block (row, col), not on which block
    # is being read. Hoisting them out of the loop body keeps the per-iteration
    # residuals at O(B) instead of O(B^2): reverse mode stacks a scanned body's live
    # intermediates over all K iterations, so leaving the (B, B) index and mask arrays
    # inside costs K * B^2 of stacked int32/bool residuals (several GB at the design
    # target) to re-derive numbers that never varied.
    def band_coords(m):
        d = m % nc
        slot = off0 + (m - d) // nc
        ok = (jnp.abs(m) <= p) & (slot >= 0) & (slot < n_k)
        return jnp.clip(slot, 0, n_k - 1), d, ok

    slot_d, dd_d, ok_d = band_coords(col - row)  # diagonal block: rr = cc = kk*bs + .
    slot_l, dd_l, ok_l = band_coords(col - row - bs)  # lower block: rr is bs rows later
    eye_mask = row == col

    def gather(rr, cc, slot, dd, ok):
        """``band`` values for one block, given its (bs,) row/column indices."""
        v = band[jnp.clip(rr // nc, 0, n_pix - 1)[:, None], (rr % nc)[:, None], slot, dd]
        return jnp.where(ok & (rr < n)[:, None] & (cc < n)[None, :], v, 0.0)

    # Rematerialized: what is left inside the body after the hoist is index arithmetic
    # and one gather, but reverse mode would still stack the (B, B) validity mask over
    # all K iterations. Recomputing it is free next to the block Cholesky it feeds.
    @jax.checkpoint
    def blocks_at(kk):
        base = kk * bs
        rr = base + jnp.arange(bs)
        cc = base + jnp.arange(bs)
        # Pad rows of the identity block: only the diagonal can hit them, and only
        # within a diagonal block (a lower block's rows sit bs beyond its columns).
        pad_eye = (eye_mask & (rr >= n)[:, None]).astype(band.dtype)
        diag_k = gather(rr, cc, slot_d, dd_d, ok_d) + pad_eye
        lower_k = gather(rr + bs, cc, slot_l, dd_l, ok_l)
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
    epoch_chunk: int | None = None,
) -> BlockTridiagonal:
    """Assemble the posterior precision ``Lambda_p + A^T W A`` by direct band assembly.

    Drop-in replacement for probing the full operator: returns the same
    :class:`BlockTridiagonal` (to floating-point reordering) at a fraction of the
    cost. ``half_bandwidth`` is the per-component bound ``b_nat`` (as in
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
    epoch_chunk
        Epochs per batch of the velocity-independent ``G`` pre-pass. ``None``
        (default) applies the size-adaptive policy of :func:`_epoch_chunk_default`:
        hoist the whole pre-pass while it is under 1 GB, otherwise batch to ~0.5 GB.
        Pass ``n_epochs`` to force the fully hoisted (fastest, most memory-hungry)
        path, or a small integer to cap live memory further. Raise it on a GPU with
        spare memory; lower it if the gradient does not fit.
    """
    nc, n_pix = problem.n_components, problem.grid.n
    p = nc * int(half_bandwidth) + nc - 1
    bs = int(block_size) if block_size is not None else max(p, 8)
    if bs < p:
        raise ValueError(f"block_size ({bs}) must be >= half_bandwidth ({p})")
    _, n_k = _band_offsets(p, nc)
    band = jnp.zeros((n_pix, nc, n_k, nc))
    for g in problem.groups:
        band = _epoch_band_scan(
            g, band, n_pix, p, nc, remat, epoch_chunk, problem.correlated, len(problem.groups)
        )
    band = _add_prior_band(band, prior, n_pix, p, nc)
    return _pack_band(band, n_pix, nc, p, bs)


def prior_logdet(prior: SmoothnessPrior, n_pix: int):
    """``log det(Lambda_p)`` by scalar banded Cholesky, without block factorization.

    ``Lambda_p`` is block diagonal over components and pentadiagonal within each
    (half-bandwidth 2), so its determinant needs only the three Cholesky diagonals

        ``a_i = L[i, i-2]``, ``b_i = L[i, i-1]``, ``c_i = L[i, i]``

    obtained from ``alpha_i = Lambda[i, i-2]``, ``beta_i = Lambda[i, i-1]``,
    ``gamma_i = Lambda[i, i]`` by

        ``a_i = alpha_i / c_{i-2}``,
        ``b_i = (beta_i - a_i b_{i-1}) / c_{i-1}``,
        ``c_i = sqrt(gamma_i - a_i^2 - b_i^2)``,

    with ``log det = 2 sum_i log c_i`` accumulated in the carry. Routing this through
    :func:`prior_block_tridiagonal` and :func:`albireo.solver.block_cholesky` instead
    pads the bandwidth-2 matrix out to dense blocks of size 64 and factorizes
    ``n_comp * n_pix / 64`` of them, 0.78 GB of live blocks at the design target, for a
    quantity that is exactly this ``O(n_pix)`` recursion. Components are carried as a
    leading ``vmap``-free axis, so the scan is one pass regardless of ``n_comp``.
    """
    d0, d1, d2 = _prior_diagonals(prior, n_pix)  # (nc, n), (nc, n-1), (nc, n-2)
    nc = d0.shape[0]
    gamma = d0.T
    beta = jnp.pad(d1.T, ((1, 0), (0, 0)))
    alpha = jnp.pad(d2.T, ((2, 0), (0, 0)))

    def step(carry, x):
        c1, c2, b1, acc = carry
        al, be, ga = x
        a = al / c2
        b = (be - a * b1) / c1
        # The pivot is positive for any positive (tau, eta), but it is a difference of
        # like-sized quantities: below eta/tau ~ 1e-13 it rounds to <= 0 and the sqrt
        # returns nan, taking the whole likelihood with it. ML-II reaches that region on
        # its own, since nothing bounds log_eta from below and a component with little
        # signal in the data has no reason to stay away. Flooring keeps the recursion
        # finite and the value monotone, so the optimizer is pushed back rather than
        # derailed.
        c = jnp.sqrt(jnp.maximum(ga - a * a - b * b, jnp.finfo(ga.dtype).tiny))
        return (c, c1, b, acc + jnp.log(c)), None

    init = (jnp.ones(nc), jnp.ones(nc), jnp.zeros(nc), jnp.zeros(nc))
    (_, _, _, acc), _ = jax.lax.scan(step, init, (alpha, beta, gamma))
    return 2.0 * jnp.sum(acc)


def prior_block_tridiagonal(
    prior: SmoothnessPrior, n_pix: int, nc: int, block_size: int
) -> BlockTridiagonal:
    """The prior precision alone as :class:`BlockTridiagonal` (analytic diagonals).

    Retained as the reference construction and test oracle; the likelihood takes its
    determinant from :func:`prior_logdet` instead.
    """
    p = 2 * nc
    _off0, n_k = _band_offsets(p, nc)
    band = jnp.zeros((n_pix, nc, n_k, nc))
    band = _add_prior_band(band, prior, n_pix, p, nc)
    return _pack_band(band, n_pix, nc, p, block_size)

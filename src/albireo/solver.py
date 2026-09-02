"""Block-tridiagonal linear algebra: comb-probe assembly, Cholesky, solves, sampling and
selected inversion.

Implements Strategy A of ``docs/math.md`` §4.2. A banded symmetric positive definite
operator of half-bandwidth ``p`` is partitioned into ``K`` dense blocks of size
``B >= p``, which makes it exactly block-tridiagonal. Every stage (factorization,
triangular solves, sampling, selected inversion) is a ``lax.scan`` over the blocks and is
differentiable.

Assembly by comb probing applies the matrix-free operator to unit combs of stride
``2p + 1``. Columns of one comb are separated by more than the bandwidth, so they do not
alias within the band, and ``2p + 1`` operator applications recover every band entry
exactly from the tested forward and adjoint operators. Direct band assembly
(``docs/math.md`` §4.5) has replaced probing on the likelihood path; probing remains
the reference construction and the validation oracle.

The logical dimension ``n`` is padded to ``K * B``. The pad block is the identity (probing
passes the pad coordinates through), so solves, log-determinants and samples restricted to
the first ``n`` coordinates are unaffected.

The selected inverse (the block diagonal and first block subdiagonal of the inverse) is
computed by the Takahashi recursion on the block Cholesky factor (Takahashi et al. 1973).

References
----------
Takahashi, K., Fagan, J. & Chin, M.-S. 1973, in Proc. 8th PICA Conference, 63
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

__all__ = [
    "BlockCholesky",
    "BlockTridiagonal",
    "block_cholesky",
    "dense_from_block_tridiagonal",
    "logdet",
    "probe_block_tridiagonal",
    "sample_standard",
    "selected_inverse_blocks",
    "selected_inverse_cotangent",
    "selected_inverse_diag",
    "solve",
    "solve_lower",
    "solve_upper",
]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BlockTridiagonal:
    """Symmetric block-tridiagonal matrix in block storage.

    ``diag[k]`` is the ``k``-th diagonal block and ``lower[k] = M[block k+1, block k]``
    the block below it; the blocks above the diagonal are the transposes and are not
    stored. ``n`` is the logical (unpadded) dimension; the stored dimension is ``K * B``.
    """

    diag: jax.Array  # (K, B, B)
    lower: jax.Array  # (K-1, B, B)
    n: int

    @property
    def num_blocks(self) -> int:
        return self.diag.shape[0]

    @property
    def block_size(self) -> int:
        return self.diag.shape[1]

    def matvec(self, x):
        """Apply the matrix to a padded vector of length ``K * B``."""
        k, b = self.num_blocks, self.block_size
        xb = x.reshape(k, b)
        out = jnp.einsum("kij,kj->ki", self.diag, xb)
        if k > 1:
            out = out.at[1:].add(jnp.einsum("kij,kj->ki", self.lower, xb[:-1]))
            out = out.at[:-1].add(jnp.einsum("kji,kj->ki", self.lower, xb[1:]))
        return out.reshape(-1)

    def tree_flatten(self):
        return (self.diag, self.lower), self.n

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, n=aux)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BlockCholesky:
    """Lower block-bidiagonal Cholesky factor ``L`` with ``M = L L^T``."""

    diag: jax.Array  # (K, B, B) lower-triangular
    lower: jax.Array  # (K-1, B, B)
    n: int

    @property
    def num_blocks(self) -> int:
        return self.diag.shape[0]

    @property
    def block_size(self) -> int:
        return self.diag.shape[1]

    def tree_flatten(self):
        return (self.diag, self.lower), self.n

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, n=aux)


_SMALL_PROBE_BYTES = 64 * 1024 * 1024  # outputs-array size below which memory is unconstrained


def probe_block_tridiagonal(
    matvec,
    n: int,
    half_bandwidth: int,
    block_size: int | None = None,
    *,
    probe_chunk: int | None = None,
    remat: bool | None = None,
):
    """Assemble a banded SPD operator into a :class:`BlockTridiagonal` by comb probing.

    Probes are applied in sequential batches of ``probe_chunk`` under ``lax.map``, with
    the combs generated from their offsets inside the loop body, and the block storage is
    read out of the probe outputs with per-block gathers, also under ``lax.map``.
    Sequential batching bounds the memory: unrolled probe batches have no data
    dependence, so XLA schedules them with overlapping live ranges and the
    temporary-buffer plan grows with the total probe count (about 80 GB at survey scale,
    measured). A scan reuses one batch's buffers for the next, so peak memory is the
    ``(2p + 1, n_pad)`` output array plus one batch's intermediates.

    Parameters
    ----------
    matvec
        Linear map on vectors of length ``n``. Its half-bandwidth must not exceed
        ``half_bandwidth``; otherwise probing aliases band entries and the assembled
        matrix is wrong without any error being raised. Validate by comparing
        :meth:`BlockTridiagonal.matvec` against ``matvec`` on a random vector.
    n
        Logical dimension.
    half_bandwidth
        Bound ``p`` on ``|q - c|`` for nonzero entries ``M[q, c]``.
    block_size
        Block size ``B >= p`` (default ``max(p, 8)``). The dimension is padded to a
        multiple of ``B``; the pad block is the identity.
    probe_chunk
        Probes per sequential batch. Peak probing memory scales linearly with it, and
        larger values amortize the per-step overhead (on a GPU, where memory bandwidth
        favours larger batches, it should be raised). Default: size-adaptive, all
        ``2p + 1`` probes in one batch for small problems and 8 once the outputs array
        exceeds about 64 MB.
    remat
        Rematerialize probe batches in the backward pass. Saves gradient memory
        proportional to the probe count at about 1.5-2x the backward probing cost.
        Default: size-adaptive, off for small problems (a measured 2x NUTS slowdown at
        the gate scale with no memory benefit) and on at scale, where storing every
        batch would need hundreds of GB.

    Returns
    -------
    BlockTridiagonal
        The assembled operator in block storage.

    Raises
    ------
    ValueError
        If ``block_size`` is smaller than ``half_bandwidth``.
    """
    p = int(half_bandwidth)
    b = int(block_size) if block_size is not None else max(p, 8)
    if b < p:
        raise ValueError(f"block_size ({b}) must be >= half_bandwidth ({p})")
    k = max(1, -(-n // b))
    n_pad = k * b
    stride = 2 * p + 1
    small = stride * n_pad * 8 <= _SMALL_PROBE_BYTES
    if probe_chunk is None:
        # Small problems: one batch holds all probes and runs in parallel across cores;
        # the serialization that bounds memory at scale would cost wall time here.
        probe_chunk = stride if small else 8
    if remat is None:
        remat = not small

    def matvec_padded(v):
        return jnp.concatenate([matvec(v[:n]), v[n:]])

    pos = jnp.arange(n_pad)

    # Sequential scan over probe batches; see the probe_chunk/remat notes above.
    def apply_batch(offsets):
        def apply_probe(offset):
            comb = (pos % stride == offset).astype(jnp.float64)
            return matvec_padded(comb)

        return jax.vmap(apply_probe)(offsets)

    if remat:
        apply_batch = jax.checkpoint(apply_batch)

    n_batches = -(-stride // probe_chunk)
    all_offsets = jnp.arange(n_batches * probe_chunk).reshape(n_batches, probe_chunk)

    def scan_body(carry, offsets):
        return carry, apply_batch(offsets)

    # (stride, n_pad); outputs[c % stride, q] = A[q, c]. Padded offsets >= stride
    # produce all-zero combs whose rows are sliced away.
    _, batched = jax.lax.scan(scan_body, None, all_offsets)
    outputs = batched.reshape(n_batches * probe_chunk, n_pad)[:stride]

    # Per-block readout: within the band, probe c % stride carries column c exactly
    # (the nearest comb alias c +- (2p + 1) is outside the band); entries beyond the
    # band are masked, which also reproduces the identity pad block.
    row = jnp.arange(b)[:, None]
    col = jnp.arange(b)[None, :]
    in_band_diag = jnp.abs(row - col) <= p
    in_band_lower = (row - col + b) <= p

    def blocks_at(kk):
        probe_of_col = (kk * b + col) % stride
        diag_k = jnp.where(in_band_diag, outputs[probe_of_col, kk * b + row], 0.0)
        q_low = (kk + 1) * b + row
        valid = in_band_lower & (q_low < n_pad)
        lower_k = jnp.where(valid, outputs[probe_of_col, jnp.minimum(q_low, n_pad - 1)], 0.0)
        return diag_k, lower_k

    diag, lower_all = jax.lax.map(blocks_at, jnp.arange(k))
    lower = lower_all[:-1] if k > 1 else lower_all[:0]
    return BlockTridiagonal(diag=diag, lower=lower, n=n)


def block_cholesky(bt: BlockTridiagonal) -> BlockCholesky:
    """Block Cholesky factorization by a ``lax.scan`` over blocks (``docs/math.md`` §4.2).

    The recursion is ``L_kk L_kk^T = D_k - L_{k,k-1} L_{k,k-1}^T`` and
    ``L_{k+1,k} = E_k L_kk^{-T}``, with ``D_k`` the diagonal and ``E_k`` the subdiagonal
    blocks of the input.
    """
    d, e = bt.diag, bt.lower
    l0 = jnp.linalg.cholesky(d[0])
    if bt.num_blocks == 1:
        return BlockCholesky(diag=l0[None], lower=e, n=bt.n)

    def step(prev_l, inputs):
        dk, ek = inputs
        # L_{k,k-1} L_{k-1,k-1}^T = E_k
        ll = solve_triangular(prev_l, ek.T, lower=True).T
        lk = jnp.linalg.cholesky(dk - ll @ ll.T)
        return lk, (lk, ll)

    _, (ls, lls) = jax.lax.scan(step, l0, (d[1:], e))
    return BlockCholesky(diag=jnp.concatenate([l0[None], ls]), lower=lls, n=bt.n)


def solve_lower(chol: BlockCholesky, rhs):
    """Solve ``L y = rhs`` (padded length ``K * B``)."""
    k, b = chol.num_blocks, chol.block_size
    rb = rhs.reshape(k, b)
    y0 = solve_triangular(chol.diag[0], rb[0], lower=True)
    if k == 1:
        return y0

    def step(prev_y, inputs):
        lk, llk, bk = inputs
        y = solve_triangular(lk, bk - llk @ prev_y, lower=True)
        return y, y

    _, ys = jax.lax.scan(step, y0, (chol.diag[1:], chol.lower, rb[1:]))
    return jnp.concatenate([y0[None], ys]).reshape(-1)


def solve_upper(chol: BlockCholesky, rhs):
    """Solve ``L^T x = rhs`` (padded length ``K * B``)."""
    k, b = chol.num_blocks, chol.block_size
    rb = rhs.reshape(k, b)
    x_last = solve_triangular(chol.diag[-1].T, rb[-1], lower=False)
    if k == 1:
        return x_last

    def step(next_x, inputs):
        lk, llk, bk = inputs
        x = solve_triangular(lk.T, bk - llk.T @ next_x, lower=False)
        return x, x

    _, xs = jax.lax.scan(step, x_last, (chol.diag[:-1], chol.lower, rb[:-1]), reverse=True)
    return jnp.concatenate([xs, x_last[None]]).reshape(-1)


def solve(chol: BlockCholesky, rhs):
    """Solve ``L L^T x = rhs``."""
    return solve_upper(chol, solve_lower(chol, rhs))


def logdet(chol: BlockCholesky):
    """``log det(L L^T) = 2 sum log diag(L)`` (pad block contributes exactly 0)."""
    return 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol.diag, axis1=-2, axis2=-1)))


def sample_standard(chol: BlockCholesky, z):
    """Map standard normals ``z`` to samples with covariance ``(L L^T)^{-1}``."""
    return solve_upper(chol, z)


def selected_inverse_diag(chol: BlockCholesky):
    """Diagonal of ``(L L^T)^{-1}`` by the block Takahashi recursion (backward scan).

    The recursion is the one written out in :func:`selected_inverse_blocks`; only the
    diagonal of each block is emitted.

    References
    ----------
    Takahashi, K., Fagan, J. & Chin, M.-S. 1973, in Proc. 8th PICA Conference, 63
    """
    k, b = chol.num_blocks, chol.block_size
    eye = jnp.eye(b)
    inv_last = solve_triangular(chol.diag[-1], eye, lower=True)
    s_last = inv_last.T @ inv_last
    if k == 1:
        return jnp.diagonal(s_last)[: chol.n]

    def step(s_next, inputs):
        lk, llk = inputs
        inv_lk = solve_triangular(lk, eye, lower=True)
        w = llk @ inv_lk
        s_k = inv_lk.T @ inv_lk + w.T @ s_next @ w
        return s_k, jnp.diagonal(s_k)

    _, diags = jax.lax.scan(step, s_last, (chol.diag[:-1], chol.lower), reverse=True)
    return jnp.concatenate([diags.reshape(-1), jnp.diagonal(s_last)])[: chol.n]


def selected_inverse_blocks(chol: BlockCholesky):
    """Block diagonal and first block subdiagonal of ``(L L^T)^{-1}`` (Takahashi recursion).

    Returns ``(s_diag, s_sub)`` of shapes ``(K, B, B)`` and ``(K-1, B, B)`` with
    ``s_diag[k] = Sigma[block k, block k]`` and ``s_sub[k] = Sigma[block k+1, block k]``:
    the banded part of the inverse that a block-tridiagonal perturbation contracts
    against, as the closed-form marginal-likelihood gradient requires (``docs/math.md``
    §4.5). The recursion runs backward from the last block,
    ``Sigma_{k+1,k} = -Sigma_{k+1,k+1} W_k`` and
    ``Sigma_{kk} = L_kk^{-T} L_kk^{-1} + W_k^T Sigma_{k+1,k+1} W_k`` with
    ``W_k = L_{k+1,k} L_kk^{-1}``.

    This is the reference implementation and the test oracle. The likelihood gradient
    uses the fused :func:`selected_inverse_cotangent`, which does not materialize these
    ``2K - 1`` blocks (3.1 GB at the design target).

    References
    ----------
    Takahashi, K., Fagan, J. & Chin, M.-S. 1973, in Proc. 8th PICA Conference, 63
    """
    k, b = chol.num_blocks, chol.block_size
    eye = jnp.eye(b)
    inv_last = solve_triangular(chol.diag[-1], eye, lower=True)
    s_last = inv_last.T @ inv_last
    if k == 1:
        return s_last[None], chol.lower[:0]

    def step(s_next, inputs):
        lk, llk = inputs
        inv_lk = solve_triangular(lk, eye, lower=True)
        w = llk @ inv_lk
        t = s_next @ w
        s_k = inv_lk.T @ inv_lk + w.T @ t
        return s_k, (s_k, -t)

    _, (s_all, subs) = jax.lax.scan(step, s_last, (chol.diag[:-1], chol.lower), reverse=True)
    return jnp.concatenate([s_all, s_last[None]]), subs


def selected_inverse_cotangent(chol: BlockCholesky, d_blocks, u_blocks, g_logdet, g_quad):
    """Marginal-likelihood cotangent of ``Lambda``, fused into the Takahashi sweep.

    Returns the block-tridiagonal storage of

        ``g_logdet * Sigma  -  g_quad * d d^T  -  sym(u d^T)``

    restricted to the block-tridiagonal pattern, where ``Sigma = (L L^T)^{-1}``. This is
    the quantity the reverse rule of :func:`albireo.likelihood._solve_stage` requires
    (``docs/math.md`` §4.5). Forming it inside the recursion consumes each ``Sigma`` block
    at the step that produces it, so the selected inverse is never stored. Compared with
    the unfused route (:func:`selected_inverse_blocks` followed by the outer products)
    this removes ``2K - 1`` blocks of live storage and the outer-product temporaries,
    3.1 GB at the design target, with identical arithmetic.

    The subdiagonal blocks carry a factor 2 because ``BlockTridiagonal.lower[k]`` is the
    sole storage for both ``Lambda[k+1, k]`` and ``Lambda[k, k+1]`` (see
    :meth:`BlockTridiagonal.matvec`, which applies it and its transpose).

    Parameters
    ----------
    chol
        Factor of ``Lambda``.
    d_blocks, u_blocks
        ``(K, B)`` block views of ``d = Lambda^{-1} b`` and ``u = Lambda^{-1} g_d``.
    g_logdet, g_quad
        Scalar cotangents of ``log det Lambda`` and of ``b^T Lambda^{-1} b``.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        ``(diag_bar, lower_bar)`` of shapes ``(K, B, B)`` and ``(K-1, B, B)``.

    References
    ----------
    Takahashi, K., Fagan, J. & Chin, M.-S. 1973, in Proc. 8th PICA Conference, 63
    """
    k, b = chol.num_blocks, chol.block_size
    eye = jnp.eye(b)
    d, u = d_blocks, u_blocks

    def outer(x, y):
        return x[:, None] * y[None, :]

    inv_last = solve_triangular(chol.diag[-1], eye, lower=True)
    s_last = inv_last.T @ inv_last
    diag_last = g_logdet * s_last - g_quad * outer(d[-1], d[-1]) - outer(u[-1], d[-1])
    if k == 1:
        return diag_last[None], chol.lower[:0]

    def step(s_next, inputs):
        lk, llk, dk, uk, dn, un = inputs
        inv_lk = solve_triangular(lk, eye, lower=True)
        w = llk @ inv_lk
        t = s_next @ w  # = -Sigma[k+1, k]
        s_k = inv_lk.T @ inv_lk + w.T @ t
        diag_k = g_logdet * s_k - g_quad * outer(dk, dk) - outer(uk, dk)
        sub_k = -2.0 * g_logdet * t - 2.0 * g_quad * outer(dn, dk) - (outer(un, dk) + outer(dn, uk))
        return s_k, (diag_k, sub_k)

    _, (diags, subs) = jax.lax.scan(
        step,
        s_last,
        (chol.diag[:-1], chol.lower, d[:-1], u[:-1], d[1:], u[1:]),
        reverse=True,
    )
    return jnp.concatenate([diags, diag_last[None]]), subs


def dense_from_block_tridiagonal(bt: BlockTridiagonal) -> np.ndarray:
    """Dense (unpadded) matrix, for small-problem tests only."""
    k, b = bt.num_blocks, bt.block_size
    out = np.zeros((k * b, k * b))
    d = np.asarray(bt.diag)
    e = np.asarray(bt.lower)
    for i in range(k):
        out[i * b : (i + 1) * b, i * b : (i + 1) * b] = d[i]
    for i in range(k - 1):
        out[(i + 1) * b : (i + 2) * b, i * b : (i + 1) * b] = e[i]
        out[i * b : (i + 1) * b, (i + 1) * b : (i + 2) * b] = e[i].T
    return out[: bt.n, : bt.n]

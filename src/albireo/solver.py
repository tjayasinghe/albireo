"""Block-tridiagonal solver: exact probe assembly, Cholesky, solves, sampling, Takahashi.

Strategy A of ``docs/math.md`` §4.2. A banded SPD operator of half-bandwidth ``p`` is
partitioned into ``K`` dense blocks of size ``B >= p``, making it block-tridiagonal
exactly. Assembly uses *comb probing*: probing a banded matrix with unit combs of
stride ``2p + 1`` recovers every entry exactly (columns of the same comb are too far
apart to alias within the band), so the matrix is built from ``2p + 1`` matrix-free
operator applications — reusing the tested forward/adjoint operators and nothing else.
All stages are ``lax.scan``-based and differentiable.

The logical dimension ``n`` is padded to ``K * B``; the pad block is the identity
(probe pass-through), which leaves solves, log-determinants, and sampling of the
first ``n`` coordinates untouched.
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
    "selected_inverse_diag",
    "solve",
    "solve_lower",
    "solve_upper",
]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BlockTridiagonal:
    """Symmetric block-tridiagonal matrix: ``diag[k]`` on the diagonal and
    ``lower[k] = M[block k+1, block k]`` below it. ``n`` is the logical (unpadded)
    dimension; the stored dimension is ``K * B``."""

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


def _probe_scatter_indices(n_pad: int, p: int, block_size: int, num_blocks: int):
    """Static index maps turning comb-probe outputs into block-tridiagonal storage.

    For every matrix entry ``(q, c)`` with ``|q - c| <= p``, the probe containing
    column ``c`` is ``c % (2p + 1)`` and the entry value sits at row ``q`` of that
    probe's output. Entries land either in the diagonal block of ``c`` or (for
    ``q`` one block below) in a lower block; upper entries follow by symmetry.
    """
    b = block_size
    stride = 2 * p + 1
    c = np.arange(n_pad)[:, None]
    q = c + np.arange(-p, p + 1)[None, :]
    valid = (q >= 0) & (q < n_pad)
    kc, kq = c // b, q // b
    probe = np.broadcast_to(c % stride, q.shape)

    same = valid & (kq == kc)
    down = valid & (kq == kc + 1)

    def gather(mask):
        qq, cc = q[mask], np.broadcast_to(c, q.shape)[mask]
        return (
            (cc // b).astype(np.int32),  # block index (column block)
            (qq % b).astype(np.int32),  # row within block
            (cc % b).astype(np.int32),  # col within block
            probe[mask].astype(np.int32),
            qq.astype(np.int32),
        )

    d_idx = gather(same)
    e_idx = gather(down)
    if num_blocks == 1 and e_idx[0].size:  # no lower blocks exist
        e_idx = tuple(a[:0] for a in e_idx)
    return stride, d_idx, e_idx


def probe_block_tridiagonal(
    matvec, n: int, half_bandwidth: int, block_size: int | None = None, *, probe_chunk: int = 64
):
    """Assemble a banded SPD operator into :class:`BlockTridiagonal` by comb probing.

    Parameters
    ----------
    matvec
        Linear map on vectors of length ``n``. Must have half-bandwidth at most
        ``half_bandwidth`` — probing silently corrupts entries otherwise (validate with
        :meth:`BlockTridiagonal.matvec` against ``matvec`` on a random vector).
    n
        Logical dimension.
    half_bandwidth
        Bound ``p`` on ``|q - c|`` for nonzero entries.
    block_size
        Block size ``B >= p`` (default ``max(p, 8)``). Dimension is padded to a
        multiple of ``B``; the pad block is the identity.
    probe_chunk
        Probes are applied in ``vmap`` batches of this size to bound peak memory.
    """
    p = int(half_bandwidth)
    b = int(block_size) if block_size is not None else max(p, 8)
    if b < p:
        raise ValueError(f"block_size ({b}) must be >= half_bandwidth ({p})")
    k = max(1, -(-n // b))
    n_pad = k * b

    def matvec_padded(v):
        return jnp.concatenate([matvec(v[:n]), v[n:]])

    stride, d_idx, e_idx = _probe_scatter_indices(n_pad, p, b, k)
    probes = (jnp.arange(n_pad)[None, :] % stride == jnp.arange(stride)[:, None]).astype(
        jnp.float64
    )
    chunks = [
        jax.vmap(matvec_padded)(probes[i : i + probe_chunk]) for i in range(0, stride, probe_chunk)
    ]
    outputs = jnp.concatenate(chunks)  # (stride, n_pad)

    kd, qi, ci, pr, qg = d_idx
    diag = jnp.zeros((k, b, b)).at[kd, qi, ci].set(outputs[pr, qg])
    ke, qie, cie, pre, qge = e_idx
    lower = jnp.zeros((max(k - 1, 0), b, b))
    if k > 1:
        lower = lower.at[ke, qie, cie].set(outputs[pre, qge])
    return BlockTridiagonal(diag=diag, lower=lower, n=n)


def block_cholesky(bt: BlockTridiagonal) -> BlockCholesky:
    """Cholesky factorization via a ``lax.scan`` over blocks (``docs/math.md`` §4.2)."""
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
    """Diagonal of ``(L L^T)^{-1}`` via block Takahashi recursions (backward scan)."""
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

"""The analytically-marginalized likelihood (Strategy A) and conditional spectra.

Implements ``docs/math.md`` §3: with the component spectra marginalized in closed form,

    log p(y | theta) = -1/2 [ z^T W z - b^T Lt^{-1} b ] - 1/2 log det(Lt)
                       + 1/2 log det(Lp) + 1/2 sum log(w/2pi)

where ``Lt = Lp + A^T W A`` is the posterior precision of the stacked deviation
spectra, ``b = A^T W z``, and both determinants come from block-tridiagonal Cholesky
factorizations (``docs/math.md`` §4.2) assembled by exact comb probing of the
matrix-free operators.

Component interleaving: the stacked vector uses index ``q * n_comp + i`` (pixel-major),
which keeps the posterior precision banded with half-bandwidth
``n_comp * b_natural + n_comp - 1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from albireo.assembly import band_block_tridiagonal, prior_block_tridiagonal
from albireo.forward import Problem, normal_matvec, rhs, weighted_data_terms
from albireo.priors import SmoothnessPrior
from albireo.solver import (
    BlockCholesky,
    BlockTridiagonal,
    block_cholesky,
    logdet,
    probe_block_tridiagonal,
    sample_standard,
    selected_inverse_blocks,
    selected_inverse_diag,
    solve,
    solve_lower,
    solve_upper,
)

__all__ = [
    "MarginalResult",
    "draw_spectra",
    "marginal_loglikelihood",
    "spectra_std",
]


def _pack(d_stack, n_comp: int, n_pix: int):
    """(n_comp, n_pix) -> interleaved vector with index q * n_comp + i."""
    return jnp.asarray(d_stack).T.reshape(n_comp * n_pix)


def _unpack(v, n_comp: int, n_pix: int):
    return jnp.asarray(v).reshape(n_pix, n_comp).T


@jax.custom_vjp
def _solve_stage(bt: BlockTridiagonal, b_pad):
    """Cholesky + solves with a closed-form reverse pass (``docs/math.md`` §3).

    Returns ``(logdet, quad, d_pad, chol)`` for the posterior precision ``bt`` and
    right-hand side ``b_pad``. The custom VJP replaces reverse-mode through the
    Cholesky/solve scans with the analytic identities

        d(logdet)/d(Lambda) = Sigma,      d(quad)/d(Lambda) = -d d^T,
        cotangent(d_pad) = g  ->  b_bar += Sigma g,  Lambda_bar += -(u d^T)|_band,

    where only the *banded* part of ``Sigma`` (block Takahashi,
    :func:`albireo.solver.selected_inverse_blocks`) is ever formed — the
    perturbation is block-tridiagonal, so nothing else contributes. This removes
    the stored backward pass of both scans (memory ~ the factor itself) and costs
    about one extra Cholesky-equivalent instead of two to three.

    Gradient contract: gradients flow through ``logdet``, ``quad``, and ``d_pad``.
    They do **not** flow through the returned ``chol`` factor (its cotangent is
    dropped; the factor is a numerical artifact for sampling/variance diagnostics).
    """
    chol = block_cholesky(bt)
    y = solve_lower(chol, b_pad)
    quad = jnp.sum(y * y)
    d_pad = solve_upper(chol, y)
    return logdet(chol), quad, d_pad, chol


def _solve_stage_fwd(bt, b_pad):
    # Recompute inline rather than calling _solve_stage: the fwd trace must contain
    # only plain operations, so that a second reverse differentiation (Hessians via
    # jacrev-of-jacrev, as in laplace_inverse_mass) walks ordinary graphs instead of
    # re-entering the custom boundary — where the chol cotangent is dropped by
    # contract and second derivatives would silently lose the chol-mediated terms
    # (measured 8e-3 relative before this change; equal to plain autodiff after).
    chol = block_cholesky(bt)
    y = solve_lower(chol, b_pad)
    quad = jnp.sum(y * y)
    d_pad = solve_upper(chol, y)
    return (logdet(chol), quad, d_pad, chol), (chol, d_pad, bt.n)


def _solve_stage_bwd(res, cot):
    chol, d_pad, n = res
    g_ld, g_quad, g_d, _g_chol = cot  # chol cotangent dropped by contract
    k, b = chol.num_blocks, chol.block_size
    u = solve(chol, g_d)
    s_diag, s_sub = selected_inverse_blocks(chol)
    db = d_pad.reshape(k, b)
    ub = u.reshape(k, b)

    def outer(x, y):
        return x[:, :, None] * y[:, None, :]

    diag_bar = g_ld * s_diag - g_quad * outer(db, db) - outer(ub, db)
    lower_bar = (
        2.0 * g_ld * s_sub
        - 2.0 * g_quad * outer(db[1:], db[:-1])
        - (outer(ub[1:], db[:-1]) + outer(db[1:], ub[:-1]))
    )
    b_bar = 2.0 * g_quad * d_pad + u
    return BlockTridiagonal(diag=diag_bar, lower=lower_bar, n=n), b_bar


_solve_stage.defvjp(_solve_stage_fwd, _solve_stage_bwd)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MarginalResult:
    """Marginal log-likelihood plus everything needed to recover the spectra."""

    log_likelihood: jax.Array
    d_hat: jax.Array  # (n_comp, n_pix) posterior-mean deviation spectra
    chol: BlockCholesky  # factor of the (interleaved, padded) posterior precision
    n_components: int
    n_pixels: int

    @property
    def n_padded(self) -> int:
        return self.chol.num_blocks * self.chol.block_size

    def tree_flatten(self):
        return (self.log_likelihood, self.d_hat, self.chol), (self.n_components, self.n_pixels)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, n_components=aux[0], n_pixels=aux[1])


def marginal_loglikelihood(
    problem: Problem,
    prior: SmoothnessPrior,
    *,
    block_size: int | None = None,
    half_bandwidth: int | None = None,
    validate: bool = False,
    assembly: str = "band",
) -> MarginalResult:
    """Evaluate the marginal log-likelihood for a fixed-parameter :class:`Problem`.

    Parameters
    ----------
    problem
        Output of :func:`albireo.forward.build_problem`.
    prior
        Spectral prior with one (tau, eta) pair per component (including telluric).
    block_size
        Solver block size; default = the required half-bandwidth.
    half_bandwidth
        Static override for the per-component half-bandwidth ``b_natural``. Required
        under ``jax.jit`` with traced shifts (where :attr:`Problem.natural_half_bandwidth`
        cannot be computed); use :meth:`Problem.half_bandwidth_bound`. Probing with any
        value >= the true bandwidth is exact, so an overestimate costs time, not
        accuracy; an *underestimate* silently corrupts the result — hence ``validate``.
    validate
        If True, verify the assembled matrix reproduces the matrix-free operator on a
        random vector (guards against a bandwidth underestimate and any assembly
        defect) — raises AssertionError on mismatch. Cheap relative to assembly;
        enabled in tests. Not jit-compatible.
    assembly
        ``"band"`` (default): direct per-epoch band assembly
        (:func:`albireo.assembly.band_block_tridiagonal`) — O(band width) work per
        epoch instead of O(bandwidth) operator applications; >10x faster at survey
        bandwidths, identical result up to floating-point summation order.
        ``"probe"``: the original global comb probing (reference implementation).
    """
    n_comp, n_pix = problem.n_components, problem.grid.n
    if prior.n_components != n_comp:
        raise ValueError(
            f"prior has {prior.n_components} components, problem has {n_comp} "
            "(remember the telluric component if enabled)"
        )
    n = n_comp * n_pix
    b_nat = int(half_bandwidth) if half_bandwidth is not None else problem.natural_half_bandwidth
    b_nat = max(b_nat, prior.half_bandwidth)
    p = n_comp * b_nat + n_comp - 1

    def full_matvec(v):
        d = _unpack(v, n_comp, n_pix)
        return _pack(normal_matvec(problem, d) + prior.apply(d), n_comp, n_pix)

    def prior_matvec(v):
        return _pack(prior.apply(_unpack(v, n_comp, n_pix)), n_comp, n_pix)

    # The prior's bandwidth is tiny (2 per component), so its factor gets its own
    # small block size: factorizing it at the posterior's block size would double
    # the Cholesky cost at scale for a determinant that is nearly free.
    prior_block = max(2 * n_comp, 64)
    if assembly == "band":
        bt = band_block_tridiagonal(problem, prior, b_nat, block_size)
        bt_prior = prior_block_tridiagonal(prior, n_pix, n_comp, prior_block)
    elif assembly == "probe":
        bt = probe_block_tridiagonal(full_matvec, n, p, block_size)
        bt_prior = probe_block_tridiagonal(prior_matvec, n, 2 * n_comp, prior_block)
    else:
        raise ValueError(f"assembly must be 'band' or 'probe'; got {assembly!r}")
    if validate:
        key = jax.random.PRNGKey(0)
        v = jax.random.normal(key, (n,))
        v_pad = jnp.pad(v, (0, bt.num_blocks * bt.block_size - n))
        got = bt.matvec(v_pad)[:n]
        want = full_matvec(v)
        err = jnp.max(jnp.abs(got - want)) / (1.0 + jnp.max(jnp.abs(want)))
        assert float(err) < 1e-10, (
            f"probed matrix disagrees with operator (rel err {float(err):.2e}); "
            "half-bandwidth underestimated?"
        )

    chol_prior = block_cholesky(bt_prior)

    b_vec = _pack(rhs(problem), n_comp, n_pix)
    b_pad = jnp.pad(b_vec, (0, bt.num_blocks * bt.block_size - n))
    ld, quad, d_pad, chol = _solve_stage(bt, b_pad)
    d_hat = _unpack(d_pad[:n], n_comp, n_pix)

    zwz, logw, n_good = weighted_data_terms(problem)
    logp = (
        -0.5 * (zwz - quad)
        - 0.5 * ld
        + 0.5 * logdet(chol_prior)
        + 0.5 * logw
        - 0.5 * n_good * jnp.log(2.0 * jnp.pi)
    )
    return MarginalResult(
        log_likelihood=logp, d_hat=d_hat, chol=chol, n_components=n_comp, n_pixels=n_pix
    )


def draw_spectra(result: MarginalResult, key, num_draws: int):
    """Draw conditional posterior spectra, shape ``(num_draws, n_comp, n_pix)``.

    Draws are ``d_hat + L^{-T} z`` with standard-normal ``z`` (``docs/math.md`` §3.3);
    pad coordinates are independent standard normals and are sliced away.
    """
    n = result.n_components * result.n_pixels
    z = jax.random.normal(key, (num_draws, result.n_padded))
    x = jax.vmap(lambda zz: sample_standard(result.chol, zz))(z)[:, :n]
    return jax.vmap(lambda v: _unpack(v, result.n_components, result.n_pixels))(x) + result.d_hat


def spectra_std(result: MarginalResult):
    """Pointwise posterior standard deviation, shape ``(n_comp, n_pix)`` (Takahashi)."""
    var = selected_inverse_diag(result.chol)
    return jnp.sqrt(_unpack(var, result.n_components, result.n_pixels))

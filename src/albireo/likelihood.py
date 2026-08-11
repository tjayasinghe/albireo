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

from albireo.forward import Problem, normal_matvec, rhs, weighted_data_terms
from albireo.priors import SmoothnessPrior
from albireo.solver import (
    BlockCholesky,
    block_cholesky,
    logdet,
    probe_block_tridiagonal,
    sample_standard,
    selected_inverse_diag,
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


def marginal_loglikelihood(
    problem: Problem,
    prior: SmoothnessPrior,
    *,
    block_size: int | None = None,
    validate: bool = False,
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
    validate
        If True, verify the probed matrix reproduces the matrix-free operator on a
        random vector (guards against a bandwidth underestimate) — raises AssertionError
        on mismatch. Cheap relative to assembly; enabled in tests.
    """
    n_comp, n_pix = problem.n_components, problem.grid.n
    if prior.n_components != n_comp:
        raise ValueError(
            f"prior has {prior.n_components} components, problem has {n_comp} "
            "(remember the telluric component if enabled)"
        )
    n = n_comp * n_pix
    b_nat = max(problem.natural_half_bandwidth, prior.half_bandwidth)
    p = n_comp * b_nat + n_comp - 1

    def full_matvec(v):
        d = _unpack(v, n_comp, n_pix)
        return _pack(normal_matvec(problem, d) + prior.apply(d), n_comp, n_pix)

    def prior_matvec(v):
        return _pack(prior.apply(_unpack(v, n_comp, n_pix)), n_comp, n_pix)

    bt = probe_block_tridiagonal(full_matvec, n, p, block_size)
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

    bt_prior = probe_block_tridiagonal(prior_matvec, n, 2 * n_comp, bt.block_size)
    chol = block_cholesky(bt)
    chol_prior = block_cholesky(bt_prior)

    b_vec = _pack(rhs(problem), n_comp, n_pix)
    b_pad = jnp.pad(b_vec, (0, bt.num_blocks * bt.block_size - n))
    y = solve_lower(chol, b_pad)
    quad = jnp.sum(y * y)  # b^T Lt^{-1} b
    d_hat = _unpack(solve_upper(chol, y)[:n], n_comp, n_pix)

    zwz, logw, n_good = weighted_data_terms(problem)
    logp = (
        -0.5 * (zwz - quad)
        - 0.5 * logdet(chol)
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

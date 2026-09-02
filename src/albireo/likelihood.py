"""The analytically marginalized likelihood (Strategy A) and the conditional spectra.

Implements ``docs/math.md`` §3. Conditional on the nonlinear parameters the model is
linear-Gaussian in the stacked deviation spectra, so the spectra integrate out in closed
form (Rasmussen & Williams 2006):

    log p(y | theta) = -1/2 [ z^T W z - b^T Lt^{-1} b ] - 1/2 log det(Lt)
                       + 1/2 log det(Lp) + 1/2 sum log(w/2pi)

where ``Lt = Lp + A^T W A`` is the posterior precision of the stacked deviation spectra,
``Lp`` the prior precision, ``z`` the data with the offset term removed
(``docs/math.md`` §3.1), and ``b = A^T W z``. Both determinants come from
block-tridiagonal Cholesky factorizations (``docs/math.md`` §4.2). The band of ``Lt`` is
assembled per epoch from its analytic structure by default (``docs/math.md`` §4.5, D28);
exact comb probing of the matrix-free operators is retained as the reference path. The
pointwise posterior variance of the spectra, and the selected inverse required by the
closed-form gradient, come from the Takahashi recursion on the block factor (Takahashi
et al. 1973).

Component interleaving: the stacked vector uses index ``q * n_comp + i`` (pixel-major),
which keeps the posterior precision banded with half-bandwidth
``n_comp * b_natural + n_comp - 1``.

References
----------
Rasmussen, C. E. & Williams, C. K. I. 2006, Gaussian Processes for Machine Learning
    (MIT Press)
Takahashi, K., Fagan, J. & Chin, M.-S. 1973, in Proc. 8th PICA Conference, 63
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from albireo.assembly import band_block_tridiagonal, prior_logdet
from albireo.forward import Problem, normal_matvec, rhs, weighted_data_terms
from albireo.priors import SmoothnessPrior
from albireo.solver import (
    BlockCholesky,
    BlockTridiagonal,
    block_cholesky,
    logdet,
    probe_block_tridiagonal,
    sample_standard,
    selected_inverse_cotangent,
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
    """Cholesky and solves with a closed-form reverse pass (``docs/math.md`` §3, §4.5).

    Returns ``(logdet, quad, d_pad)`` for the posterior precision ``bt`` and right-hand
    side ``b_pad``. The custom VJP replaces reverse mode through the Cholesky and solve
    scans with the analytic identities

        d(logdet)/d(Lambda) = Sigma,      d(quad)/d(Lambda) = -d d^T,
        cotangent(d_pad) = g  ->  b_bar += Sigma g,  Lambda_bar += -(u d^T)|_band,

    where only the banded part of ``Sigma`` is formed (the perturbation is
    block-tridiagonal, so no other entries contribute) and is contracted into the
    cotangent inside the Takahashi sweep that produces it
    (:func:`albireo.solver.selected_inverse_cotangent`). This removes the stored backward
    pass of both scans, whose memory is of the order of the factor itself, and costs
    about one extra Cholesky-equivalent instead of two to three.

    The Cholesky factor is not an output. A cotangent on it could not be honoured by this
    rule (propagating it is the reverse-mode pass through the factorization that the rule
    avoids), and returning it would make gradients of any factor-derived quantity zero
    without warning. :attr:`MarginalResult.chol` therefore rebuilds the factor outside
    this boundary, where plain autodiff applies.

    ``diag_bar`` is left unsymmetrized: ``BlockTridiagonal.diag[k]`` stores both
    triangles of a symmetric block, each read once by the band packing, so mirror entries
    receive the two halves of ``u d^T + d u^T`` separately and the assembled parameter
    gradient is the same. The off-diagonal blocks are stored once for two triangles,
    hence their factor of 2.
    """
    chol = block_cholesky(bt)
    y = solve_lower(chol, b_pad)
    quad = jnp.sum(y * y)
    d_pad = solve_upper(chol, y)
    return logdet(chol), quad, d_pad


def _solve_stage_fwd(bt, b_pad):
    # Recompute inline rather than calling _solve_stage: the forward trace must contain
    # only plain operations, so that a second reverse differentiation (Hessians by
    # jacrev-of-jacrev, as in laplace_inverse_mass) walks ordinary graphs instead of
    # re-entering the custom boundary, where the chol cotangent is dropped by contract
    # and second derivatives would lose the chol-mediated terms without warning
    # (measured 8e-3 relative before this change; equal to plain autodiff after).
    chol = block_cholesky(bt)
    y = solve_lower(chol, b_pad)
    quad = jnp.sum(y * y)
    d_pad = solve_upper(chol, y)
    return (logdet(chol), quad, d_pad), (chol, d_pad, bt.n)


def _solve_stage_bwd(res, cot):
    chol, d_pad, n = res
    g_ld, g_quad, g_d = cot
    k, b = chol.num_blocks, chol.block_size
    u = solve(chol, g_d)
    # Fused: the Takahashi sweep emits the cotangent blocks directly, so the selected
    # inverse and the outer-product temporaries are never materialized (2K - 1 blocks,
    # 3.1 GB at the design target). selected_inverse_blocks remains the test oracle.
    diag_bar, lower_bar = selected_inverse_cotangent(
        chol, d_pad.reshape(k, b), u.reshape(k, b), g_ld, g_quad
    )
    b_bar = 2.0 * g_quad * d_pad + u
    return BlockTridiagonal(diag=diag_bar, lower=lower_bar, n=n), b_bar


_solve_stage.defvjp(_solve_stage_fwd, _solve_stage_bwd)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MarginalResult:
    """Marginal log-likelihood and the quantities needed to recover the spectra.

    Attributes
    ----------
    log_likelihood
        The marginal log-likelihood ``log p(y | theta)``.
    d_hat
        ``(n_comp, n_pix)`` posterior-mean deviation spectra.
    precision
        The interleaved, padded posterior precision ``Lt`` in block storage.
    n_components, n_pixels
        Dimensions of ``d_hat``.
    """

    log_likelihood: jax.Array
    d_hat: jax.Array  # (n_comp, n_pix) posterior-mean deviation spectra
    precision: BlockTridiagonal  # the (interleaved, padded) posterior precision
    n_components: int
    n_pixels: int

    @property
    def chol(self) -> BlockCholesky:
        """Cholesky factor of :attr:`precision`, built on demand.

        The likelihood's own factorization is computed inside a ``custom_vjp`` whose
        reverse rule cannot carry a cotangent on the factor, so returning that factor
        would make gradients of anything derived from it (spectral uncertainties, draws)
        zero without warning. Refactorizing here keeps those paths on plain autodiff, at
        the cost of one extra block Cholesky, paid only by callers that use the factor.
        The sampling path reads :attr:`log_likelihood` and :attr:`d_hat` and never
        triggers it.

        Raises
        ------
        ValueError
            If the result carries no precision, which happens only for a result read
            back by :func:`albireo.results.load_fit` from a file saved without
            ``precision=True``.
        """
        if self.precision is None:
            raise ValueError(
                "this MarginalResult carries no posterior precision, so it cannot be "
                "factorized. It was read back by albireo.results.load_fit, which stores "
                "the precision only when saved with precision=True (the blocks are large). "
                "The posterior mean is in .d_hat and, if it was saved, the pointwise "
                "standard deviation is in .d_std; re-run the fit to draw new spectra."
            )
        return block_cholesky(self.precision)

    @property
    def n_padded(self) -> int:
        return self.precision.num_blocks * self.precision.block_size

    def tree_flatten(self):
        return (self.log_likelihood, self.d_hat, self.precision), (
            self.n_components,
            self.n_pixels,
        )

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
    assembly: str | None = None,
) -> MarginalResult:
    """Evaluate the marginal log-likelihood of a fixed-parameter :class:`Problem`.

    Implements the marginal log-likelihood of ``docs/math.md`` §3.1 and returns with it
    the posterior-mean deviation spectra and the posterior precision from which the
    conditional spectra are recovered (§3.3).

    Parameters
    ----------
    problem
        Output of :func:`albireo.forward.build_problem`.
    prior
        Spectral prior with one ``(tau, eta)`` pair per component, including the
        telluric and nebular components when they are enabled, in that trailing order.
        Per-pixel profiles, if present, must match the model grid.
    block_size
        Solver block size; the default is the required half-bandwidth.
    half_bandwidth
        Static override for the per-component half-bandwidth ``b_natural``. Required
        under ``jax.jit`` with traced shifts, where :attr:`Problem.natural_half_bandwidth`
        cannot be computed; use :meth:`Problem.half_bandwidth_bound`. Assembly with any
        value at or above the true bandwidth is exact, so an overestimate costs time but
        not accuracy. An underestimate corrupts the result without raising an error; see
        ``validate``.
    validate
        If True, verify that the assembled matrix reproduces the matrix-free operator on
        a random vector, which guards against a bandwidth underestimate and any assembly
        defect; raises ``AssertionError`` on mismatch. Cheap relative to assembly and
        enabled in the tests. Not jit-compatible.
    assembly
        ``None`` (default) selects ``"band"``: direct per-epoch band assembly
        (:func:`albireo.assembly.band_block_tridiagonal`, ``docs/math.md`` §4.5, D28),
        O(band width) work per epoch instead of O(bandwidth) operator applications, more
        than 10x faster at survey bandwidths, with an identical result up to
        floating-point summation order. Correlated AR(1) noise runs on the same path
        (D35): the chain's cross-row terms enter through static link pair tables
        (:func:`albireo.operators.rebin_link_pair_tables`). ``"probe"`` selects global
        comb probing, retained as the reference implementation and as the independent
        construction behind the ``validate`` oracle.

    Returns
    -------
    MarginalResult
        Log-likelihood, posterior-mean spectra and posterior precision.

    Raises
    ------
    ValueError
        If the prior's component count or profile length does not match the problem,
        or ``assembly`` is neither ``"band"`` nor ``"probe"``.
    AssertionError
        If ``validate`` is True and the assembled matrix disagrees with the operator.
    """
    n_comp, n_pix = problem.n_components, problem.grid.n
    if prior.n_components != n_comp:
        extra = [
            n for n, on in (("telluric", problem.telluric), ("nebular", problem.nebular)) if on
        ]
        raise ValueError(
            f"prior has {prior.n_components} components, problem has {n_comp}"
            + (f" (including the {' and '.join(extra)} component(s))" if extra else "")
            + ": one (tau, eta) pair per model component, in the order stellar,"
            " telluric, nebular"
        )
    if prior.n_pixels is not None and prior.n_pixels != n_pix:
        raise ValueError(
            f"prior profiles cover {prior.n_pixels} pixels, the model grid has {n_pix}. "
            "Per-pixel profiles are tied to the grid they were built on: rebuild with "
            "albireo.priors.window_profile(grid.wave, ...)."
        )
    if assembly is None:
        assembly = "band"
    n = n_comp * n_pix
    b_nat = int(half_bandwidth) if half_bandwidth is not None else problem.natural_half_bandwidth
    b_nat = max(b_nat, prior.half_bandwidth)
    p = n_comp * b_nat + n_comp - 1

    def full_matvec(v):
        d = _unpack(v, n_comp, n_pix)
        return _pack(normal_matvec(problem, d) + prior.apply(d), n_comp, n_pix)

    def prior_matvec(v):
        return _pack(prior.apply(_unpack(v, n_comp, n_pix)), n_comp, n_pix)

    # The prior's bandwidth is 2 per component and it is block diagonal over components,
    # so its determinant comes from a scalar banded recursion rather than a block
    # factorization (assembly.prior_logdet). The probe path keeps the blocked route as
    # its reference.
    prior_block = max(2 * n_comp, 64)
    if assembly == "band":
        bt = band_block_tridiagonal(problem, prior, b_nat, block_size)
        ld_prior = prior_logdet(prior, n_pix)
    elif assembly == "probe":
        bt = probe_block_tridiagonal(full_matvec, n, p, block_size)
        bt_prior = probe_block_tridiagonal(prior_matvec, n, 2 * n_comp, prior_block)
        ld_prior = logdet(block_cholesky(bt_prior))
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
            f"assembled matrix disagrees with the matrix-free operator (rel err "
            f"{float(err):.2e}) using assembly={assembly!r}. Most likely an "
            "underestimated half_bandwidth (use Problem.half_bandwidth_bound); "
            "otherwise an assembly defect: note that the operator is symmetric, so "
            "checking the assembled matrix against its own transpose separates the two."
        )

    b_vec = _pack(rhs(problem), n_comp, n_pix)
    b_pad = jnp.pad(b_vec, (0, bt.num_blocks * bt.block_size - n))
    ld, quad, d_pad = _solve_stage(bt, b_pad)
    d_hat = _unpack(d_pad[:n], n_comp, n_pix)

    zwz, logw, n_good = weighted_data_terms(problem)
    logp = (
        -0.5 * (zwz - quad)
        - 0.5 * ld
        + 0.5 * ld_prior
        + 0.5 * logw
        - 0.5 * n_good * jnp.log(2.0 * jnp.pi)
    )
    return MarginalResult(
        log_likelihood=logp, d_hat=d_hat, precision=bt, n_components=n_comp, n_pixels=n_pix
    )


def draw_spectra(result: MarginalResult, key, num_draws: int):
    """Draw conditional posterior spectra, shape ``(num_draws, n_comp, n_pix)``.

    Draws are ``d_hat + L^{-T} z`` with standard-normal ``z`` (``docs/math.md`` §3.3),
    where ``L`` is the block Cholesky factor of the posterior precision. Pad coordinates
    are independent standard normals and are sliced away. The draws are conditional on
    the parameters of ``result``; :func:`albireo.inference.posterior_spectra` mixes them
    over the parameter posterior.

    Parameters
    ----------
    result
        A :class:`MarginalResult` carrying its posterior precision.
    key
        JAX PRNG key.
    num_draws
        Number of draws.

    Returns
    -------
    jax.Array
        ``(num_draws, n_comp, n_pix)`` deviation spectra.
    """
    n = result.n_components * result.n_pixels
    z = jax.random.normal(key, (num_draws, result.n_padded))
    chol = result.chol  # factorize once, not once per draw
    x = jax.vmap(lambda zz: sample_standard(chol, zz))(z)[:, :n]
    return jax.vmap(lambda v: _unpack(v, result.n_components, result.n_pixels))(x) + result.d_hat


def spectra_std(result: MarginalResult):
    """Pointwise posterior standard deviation of the spectra, shape ``(n_comp, n_pix)``.

    The variances are the diagonal of the posterior covariance ``Lt^{-1}``, computed by
    the Takahashi selected-inverse recursion on the block Cholesky factor
    (:func:`albireo.solver.selected_inverse_diag`; ``docs/math.md`` §3.3) without a
    dense inversion. This is the uncertainty conditional on the parameters of
    ``result``.

    References
    ----------
    Takahashi, K., Fagan, J. & Chin, M.-S. 1973, in Proc. 8th PICA Conference, 63
    """
    var = selected_inverse_diag(result.chol)
    return jnp.sqrt(_unpack(var, result.n_components, result.n_pixels))

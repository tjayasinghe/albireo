"""Tests for the block-tridiagonal solver against dense linear algebra."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from albireo.solver import (
    block_cholesky,
    dense_from_block_tridiagonal,
    logdet,
    probe_block_tridiagonal,
    sample_standard,
    selected_inverse_diag,
    solve,
    solve_lower,
    solve_upper,
)

RNG = np.random.default_rng(11)
N, P = 37, 5


def random_banded_spd(n, p, rng):
    """Symmetric banded matrix made SPD by diagonal dominance."""
    m = np.zeros((n, n))
    for o in range(1, p + 1):
        vals = rng.standard_normal(n - o)
        m[np.arange(n - o), np.arange(o, n)] = vals
        m[np.arange(o, n), np.arange(n - o)] = vals
    m[np.diag_indices(n)] = np.abs(m).sum(axis=1) + 1.0 + rng.uniform(0, 1, n)
    return m


M_DENSE = random_banded_spd(N, P, RNG)


def matvec(v):
    return jnp.asarray(M_DENSE) @ v


@pytest.mark.parametrize("block_size", [P, P + 3, 2 * P + 1, N])
def test_probe_assembly_is_exact(block_size):
    bt = probe_block_tridiagonal(matvec, N, P, block_size)
    np.testing.assert_allclose(dense_from_block_tridiagonal(bt), M_DENSE, rtol=0, atol=1e-13)
    # padded matvec agrees with the operator
    n_pad = bt.num_blocks * bt.block_size
    v = RNG.standard_normal(n_pad)
    v[N:] = 0.0
    np.testing.assert_allclose(
        np.asarray(bt.matvec(jnp.asarray(v)))[:N], M_DENSE @ v[:N], rtol=1e-12
    )


def test_block_size_smaller_than_bandwidth_raises():
    with pytest.raises(ValueError):
        probe_block_tridiagonal(matvec, N, P, P - 1)


@pytest.mark.parametrize("block_size", [P, P + 3, N])
def test_cholesky_logdet_solve(block_size):
    bt = probe_block_tridiagonal(matvec, N, P, block_size)
    chol = block_cholesky(bt)

    sign, ld = np.linalg.slogdet(M_DENSE)
    assert sign > 0
    np.testing.assert_allclose(float(logdet(chol)), ld, rtol=1e-12)

    n_pad = bt.num_blocks * bt.block_size
    b = np.zeros(n_pad)
    b[:N] = RNG.standard_normal(N)
    x = np.asarray(solve(chol, jnp.asarray(b)))
    np.testing.assert_allclose(x[:N], np.linalg.solve(M_DENSE, b[:N]), rtol=1e-10)
    # pad coordinates are identity-decoupled
    np.testing.assert_allclose(x[N:], 0.0, atol=1e-14)

    # forward then backward substitution equals the full solve
    y = solve_lower(chol, jnp.asarray(b))
    np.testing.assert_allclose(np.asarray(solve_upper(chol, y)), x, rtol=1e-12)
    # quadratic form b^T M^-1 b = ||L^-1 b||^2
    np.testing.assert_allclose(
        float(jnp.sum(y * y)), b[:N] @ np.linalg.solve(M_DENSE, b[:N]), rtol=1e-10
    )


@pytest.mark.parametrize("block_size", [P + 3, N])
def test_selected_inverse_diag(block_size):
    bt = probe_block_tridiagonal(matvec, N, P, block_size)
    chol = block_cholesky(bt)
    np.testing.assert_allclose(
        np.asarray(selected_inverse_diag(chol)), np.diag(np.linalg.inv(M_DENSE)), rtol=1e-10
    )


def test_sampling_covariance_is_inverse():
    # deterministic check: S = L^{-T} applied to the identity satisfies S S^T = M^{-1}
    bt = probe_block_tridiagonal(matvec, N, P, P + 3)
    chol = block_cholesky(bt)
    n_pad = bt.num_blocks * bt.block_size
    s = np.asarray(jax.vmap(lambda z: sample_standard(chol, z))(jnp.eye(n_pad))).T
    cov = (s @ s.T)[:N, :N]
    np.testing.assert_allclose(cov, np.linalg.inv(M_DENSE), rtol=1e-9, atol=1e-12)

"""Tests for the banded spectral priors."""

import jax.numpy as jnp
import numpy as np
import pytest

from albireo.priors import SmoothnessPrior, second_difference, second_difference_adjoint

RNG = np.random.default_rng(5)


def test_second_difference_adjoint():
    u = jnp.asarray(RNG.standard_normal(50))
    v = jnp.asarray(RNG.standard_normal(48))
    lhs = jnp.vdot(second_difference(u), v)
    rhs = jnp.vdot(u, second_difference_adjoint(v))
    np.testing.assert_allclose(float(lhs), float(rhs), rtol=1e-13)


def test_apply_matches_dense():
    n = 30
    prior = SmoothnessPrior(tau=[3.0, 0.5], eta=[1e-2, 1e-3])
    d = RNG.standard_normal((2, n))
    dense = prior.dense(n)
    expected = (dense @ d.reshape(-1)).reshape(2, n)
    np.testing.assert_allclose(np.asarray(prior.apply(jnp.asarray(d))), expected, rtol=1e-12)


def test_dense_is_spd():
    prior = SmoothnessPrior(tau=[1.0], eta=[1e-4])
    dense = prior.dense(25)
    np.testing.assert_allclose(dense, dense.T, rtol=1e-14)
    assert np.linalg.eigvalsh(dense).min() > 0


def test_validation():
    with pytest.raises(ValueError):
        SmoothnessPrior(tau=[1.0, 2.0], eta=[1.0])
    prior = SmoothnessPrior(tau=[1.0], eta=[1e-4])
    with pytest.raises(ValueError):
        prior.apply(jnp.zeros((2, 10)))

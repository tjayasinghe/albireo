"""Gaussian priors on the component deviation spectra (banded precision).

The default prior (``docs/math.md`` §2) on each deviation spectrum is

    Lambda_i = tau_i * D2^T D2 + eta_i * I

where ``D2`` is the second-difference operator: a curvature (smoothness) penalty whose
affine nullspace — exactly the low-frequency separation degeneracy — is made proper by
the weak ridge ``eta_i`` anchoring the spectrum to the continuum. Precisions are always
banded (half-bandwidth 2); dense kernels are deliberately avoided.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["SmoothnessPrior", "second_difference", "second_difference_adjoint"]


def second_difference(d):
    """Apply ``D2``: ``(D2 d)_k = d_k - 2 d_{k+1} + d_{k+2}``, shape ``(n,) -> (n-2,)``."""
    d = jnp.asarray(d)
    return d[..., :-2] - 2.0 * d[..., 1:-1] + d[..., 2:]


def second_difference_adjoint(v):
    """Apply ``D2^T``, shape ``(n-2,) -> (n,)`` (exact adjoint of :func:`second_difference`)."""
    v = jnp.asarray(v)
    n = v.shape[-1] + 2
    out = jnp.zeros((*v.shape[:-1], n), dtype=v.dtype)
    out = out.at[..., :-2].add(v)
    out = out.at[..., 1:-1].add(-2.0 * v)
    out = out.at[..., 2:].add(v)
    return out


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SmoothnessPrior:
    """Independent smoothness + continuum-anchor prior per component.

    Attributes
    ----------
    tau
        Curvature penalty weights, shape ``(n_components,)``. Larger = smoother.
    eta
        Ridge weights, shape ``(n_components,)``. Anchors the affine nullspace of the
        curvature penalty to the continuum (d = 0) with variance ~ 1/eta per pixel —
        an explicit, documented scale for the unavoidable low-frequency uncertainty.
    """

    tau: jax.Array
    eta: jax.Array

    def __post_init__(self):
        tau = jnp.atleast_1d(jnp.asarray(self.tau, dtype=jnp.float64))
        eta = jnp.atleast_1d(jnp.asarray(self.eta, dtype=jnp.float64))
        if tau.shape != eta.shape:
            raise ValueError("tau and eta must have the same shape")
        object.__setattr__(self, "tau", tau)
        object.__setattr__(self, "eta", eta)

    @property
    def n_components(self) -> int:
        return self.tau.shape[0]

    half_bandwidth: int = 2  # of each per-component precision block

    def apply(self, d_stack):
        """Apply the block-diagonal precision to stacked spectra, shape ``(n_comp, n)``."""
        d_stack = jnp.asarray(d_stack)
        if d_stack.ndim != 2 or d_stack.shape[0] != self.n_components:
            raise ValueError(
                f"expected d_stack of shape ({self.n_components}, n); got {d_stack.shape}"
            )
        curv = second_difference_adjoint(second_difference(d_stack))
        return self.tau[:, None] * curv + self.eta[:, None] * d_stack

    def dense(self, n: int) -> np.ndarray:
        """Dense ``(n_comp * n, n_comp * n)`` precision, for small-problem tests only."""
        d2 = np.zeros((n - 2, n))
        idx = np.arange(n - 2)
        d2[idx, idx] = 1.0
        d2[idx, idx + 1] = -2.0
        d2[idx, idx + 2] = 1.0
        blocks = [
            float(t) * (d2.T @ d2) + float(e) * np.eye(n)
            for t, e in zip(np.asarray(self.tau), np.asarray(self.eta), strict=True)
        ]
        out = np.zeros((self.n_components * n, self.n_components * n))
        for i, b in enumerate(blocks):
            out[i * n : (i + 1) * n, i * n : (i + 1) * n] = b
        return out

    def tree_flatten(self):
        return (self.tau, self.eta), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

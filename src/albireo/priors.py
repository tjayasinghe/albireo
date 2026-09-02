"""Gaussian priors on the component deviation spectra, with banded precision.

The default prior (``docs/math.md`` §2) on each deviation spectrum is

    Lambda_i = tau_i * D2^T D2 + eta_i * I

where ``D2`` is the second-difference operator. The curvature penalty ``tau_i`` is a
smoothness prior whose affine nullspace (constant plus slope per component) coincides
with the low-frequency separation degeneracy of ``docs/math.md`` §5.1; the weak ridge
``eta_i`` makes those directions proper by anchoring the spectrum to the continuum.
Precisions are banded (half-bandwidth 2); dense covariance kernels are not used.

Either strength may carry a static per-pixel profile (D40),

    Lambda_i = D2^T diag(tau_i * p^tau_i) D2 + diag(eta_i * p^eta_i),

with the scalars ``tau_i, eta_i`` remaining the inferred (ML-II) hyperparameters. A
profile sets where a component may deviate from the continuum without adding a sampled
parameter. The main use is confining a component to line windows
(:func:`window_profile`): a nebular emission component has structure only at the Balmer,
He I and forbidden lines, and a large ridge elsewhere encodes that. Curvature rows are
weighted by their center pixel, so a profile is indexed like the spectrum it regularizes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "NEBULAR_LINES",
    "SmoothnessPrior",
    "nebular_windows",
    "second_difference",
    "second_difference_adjoint",
    "window_profile",
]


# Optical emission lines of an H II region, in air angstrom. Wavelengths are taken from
# the standard nebular references (the tables of Osterbrock, D. E. & Ferland, G. J. 2006,
# Astrophysics of Gaseous Nebulae and Active Galactic Nuclei, 2nd ed. (University Science
# Books); NIST and the Atomic Line List for the recombination lines), rounded to 0.01 A,
# far below the width of any window built around them. The list is not exhaustive: it
# holds the features strong enough to matter in a normalized stellar spectrum. A window
# relaxes the prior locally without changing the precision bandwidth or the pixel count.
# A different line set can be passed to :func:`nebular_windows`.
NEBULAR_LINES: Mapping[str, float] = {
    "[O II] 3726": 3726.03,
    "[O II] 3729": 3728.82,
    "H8": 3889.05,  # blended with He I 3888.65 in practice
    "H-epsilon": 3970.07,
    "H-delta": 4101.73,
    "H-gamma": 4340.47,
    "[O III] 4363": 4363.21,
    "He I 4471": 4471.48,
    "He II 4686": 4685.68,
    "H-beta": 4861.33,
    "[O III] 4959": 4958.91,
    "[O III] 5007": 5006.84,
    "[N II] 5755": 5754.60,
    "He I 5876": 5875.62,
    "[O I] 6300": 6300.30,
    "[S III] 6312": 6312.06,
    "[O I] 6364": 6363.78,
    "[N II] 6548": 6548.05,
    "H-alpha": 6562.80,
    "[N II] 6583": 6583.45,
    "He I 6678": 6678.15,
    "[S II] 6716": 6716.44,
    "[S II] 6731": 6730.82,
    "He I 7065": 7065.19,
}
"""Optical H II region emission lines (air angstrom) used by :func:`nebular_windows`.

Wavelengths follow Osterbrock & Ferland (2006), with NIST and the Atomic Line List for
the recombination lines, rounded to 0.01 A.

References
----------
Osterbrock, D. E. & Ferland, G. J. 2006, Astrophysics of Gaseous Nebulae and Active
    Galactic Nuclei, 2nd ed. (University Science Books)
"""


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


def nebular_windows(
    *,
    lines: Mapping[str, float] | Sequence[float] | None = None,
    halfwidth_kms: float = 300.0,
    v_kms: float = 0.0,
    wave_range: tuple[float, float] | None = None,
) -> tuple[tuple[float, float], ...]:
    """Wavelength windows around the nebular lines, merged and sorted.

    The windows are where a nebular component is allowed to have structure
    (:func:`window_profile`), so they should be generous. A window that is too narrow
    clips real emission and pushes the residual into the stellar components, which is
    the failure the component exists to prevent; a window that is too wide only returns
    some of the freedom the profile removes. The default half-width of 300 km/s covers
    the nebular line, the velocity spread of an H II region, and a margin.

    Parameters
    ----------
    lines
        Rest wavelengths in air angstrom: a mapping (its values are used) or a sequence.
        Default :data:`NEBULAR_LINES`.
    halfwidth_kms
        Half-width of each window in velocity [km/s], converted to wavelength at the
        line.
    v_kms
        Velocity of the nebula in the frame of the model grid [km/s]. Windows are built
        at ``lambda * (1 + v_kms / c)``, so this value must equal the ``nebular_v_kms``
        passed to :func:`albireo.forward.build_problem`: that shift decides where the
        component's lines fall on the model grid, and the profile must agree with it.
        Both default to 0, which places the component at the observed barycentric
        wavelengths, the convention the stellar components follow (their systemic
        velocity is absorbed into their spectra, D14).
    wave_range
        Optional ``(min, max)`` in angstrom; windows disjoint from it are dropped.
        ``(grid.wave[0], grid.wave[-1])`` keeps only the lines a given model grid
        covers.

    Returns
    -------
    tuple[tuple[float, float], ...]
        ``(lambda_min, lambda_max)`` pairs, sorted, with overlaps merged (adjacent
        doublets such as [O II] 3726/3729 or [S II] 6716/6731 return as one window).

    Raises
    ------
    ValueError
        If ``halfwidth_kms`` is not positive.
    """
    from albireo.grids import C_KMS

    if lines is None:
        lines = NEBULAR_LINES
    values = list(lines.values()) if isinstance(lines, Mapping) else list(lines)
    if halfwidth_kms <= 0:
        raise ValueError("halfwidth_kms must be positive")
    scale = 1.0 + float(v_kms) / C_KMS
    raw = []
    for lam in values:
        lam0 = float(lam) * scale
        half = lam0 * float(halfwidth_kms) / C_KMS
        raw.append((lam0 - half, lam0 + half))
    if wave_range is not None:
        lo, hi = float(wave_range[0]), float(wave_range[1])
        raw = [(a, b) for a, b in raw if b >= lo and a <= hi]
    merged: list[list[float]] = []
    for a, b in sorted(raw):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return tuple((a, b) for a, b in merged)


def window_profile(
    wave,
    windows: Sequence[tuple[float, float]],
    *,
    inside: float = 1.0,
    outside: float = 1.0e6,
) -> np.ndarray:
    """Per-pixel prior multiplier: ``inside`` within any window, ``outside`` elsewhere.

    The result is passed to :class:`SmoothnessPrior` as an ``eta_profile`` row. A ridge
    scaled by ``outside`` pins the component to the continuum away from the windows with
    prior standard deviation ``1/sqrt(eta * outside)`` per pixel. The default ``1e6`` is
    a factor of 1000 in amplitude, negligible against any line, and leaves the precision
    well conditioned. The confinement is soft: a hard zero would be a constraint, would
    require different linear algebra, and would remove the model's ability to report a
    disagreement with the windows (``docs/math.md`` §2).

    The function is not specific to nebular emission. Interstellar bands, diffuse
    interstellar bands, or any component known a priori to be line-poor take the same
    treatment.

    Parameters
    ----------
    wave
        Model grid wavelengths, ``(n_pix,)`` angstrom (:attr:`albireo.grids.LogGrid.wave`).
    windows
        ``(lambda_min, lambda_max)`` pairs in angstrom; overlaps are permitted. An empty
        sequence would give a uniform ``outside`` profile (no pixel allowed to deviate)
        and raises instead.
    inside, outside
        Multipliers inside and outside the windows. Both must be positive so that the
        prior stays proper.

    Returns
    -------
    numpy.ndarray
        ``(n_pix,)`` float multiplier.

    Raises
    ------
    ValueError
        If ``wave`` is not one-dimensional, a multiplier is not positive, ``windows`` is
        empty, a window is empty or reversed, or no window overlaps the grid.
    """
    wave = np.asarray(wave, dtype=np.float64)
    if wave.ndim != 1:
        raise ValueError(f"wave must be 1-D; got shape {wave.shape}")
    if not (inside > 0 and outside > 0):
        raise ValueError("inside and outside must be positive (the prior must stay proper)")
    windows = list(windows)
    if not windows:
        raise ValueError(
            "no windows: the profile would pin the component to the continuum everywhere. "
            "If that is what you want, raise eta instead; if the line list simply missed "
            "this grid, check nebular_windows(wave_range=...)."
        )
    inside_mask = np.zeros(wave.size, dtype=bool)
    for lo, hi in windows:
        if not hi > lo:
            raise ValueError(f"window ({lo}, {hi}) is empty or reversed")
        inside_mask |= (wave >= lo) & (wave <= hi)
    if not inside_mask.any():
        raise ValueError(
            f"none of the {len(windows)} windows overlap the grid "
            f"({wave[0]:.2f}-{wave[-1]:.2f} A), so the profile would pin the component to "
            "the continuum everywhere. Pass wave_range= to nebular_windows to see which "
            "lines this grid can reach."
        )
    return np.where(inside_mask, float(inside), float(outside))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SmoothnessPrior:
    """Independent smoothness-plus-continuum-anchor prior per component.

    Implements the banded precision of ``docs/math.md`` §2, with the optional per-pixel
    profiles of D40.

    Attributes
    ----------
    tau
        Curvature penalty weights, shape ``(n_components,)``. Larger values give smoother
        spectra.
    eta
        Ridge weights, shape ``(n_components,)``. Anchors the affine nullspace of the
        curvature penalty to the continuum (``d = 0``) with variance about ``1/eta`` per
        pixel, which sets the scale of the low-frequency uncertainty (``docs/math.md``
        §5.1).
    tau_profile, eta_profile
        Optional static ``(n_components, n_pixels)`` per-pixel multipliers on the two
        strengths (D40): the effective weights are ``tau[i] * tau_profile[i]`` and
        ``eta[i] * eta_profile[i]``. ``None`` (default) is a uniform profile and
        reproduces the v1 prior exactly. The scalars stay separate from the profiles, so
        the ML-II hyperparameter fit is unchanged: a profile sets where the freedom is,
        the scalar sets how much. :func:`window_profile` builds one. Curvature rows take
        the weight of their center pixel, so ``tau_profile`` is indexed like the
        spectrum.

    Raises
    ------
    ValueError
        If ``tau`` and ``eta`` differ in shape, or a profile has the wrong shape or the
        two profiles disagree on the pixel count.
    """

    tau: jax.Array
    eta: jax.Array
    tau_profile: jax.Array | None = None
    eta_profile: jax.Array | None = None

    def __post_init__(self):
        tau = jnp.atleast_1d(jnp.asarray(self.tau, dtype=jnp.float64))
        eta = jnp.atleast_1d(jnp.asarray(self.eta, dtype=jnp.float64))
        if tau.shape != eta.shape:
            raise ValueError("tau and eta must have the same shape")
        object.__setattr__(self, "tau", tau)
        object.__setattr__(self, "eta", eta)
        n_pix = None
        for name in ("tau_profile", "eta_profile"):
            p = getattr(self, name)
            if p is None:
                continue
            p = jnp.asarray(p, dtype=jnp.float64)
            if p.ndim == 1:  # one row broadcast to every component
                p = jnp.broadcast_to(p, (tau.shape[0], p.shape[0]))
            if p.ndim != 2 or p.shape[0] != tau.shape[0]:
                raise ValueError(
                    f"{name} must have shape ({tau.shape[0]}, n_pixels) or (n_pixels,); "
                    f"got {tuple(p.shape)}"
                )
            if n_pix is not None and p.shape[1] != n_pix:
                raise ValueError("tau_profile and eta_profile must have the same pixel count")
            n_pix = p.shape[1]
            object.__setattr__(self, name, p)

    @property
    def n_components(self) -> int:
        return self.tau.shape[0]

    @property
    def n_pixels(self) -> int | None:
        """Pixel count the profiles were built for, or ``None`` when there are none."""
        for p in (self.tau_profile, self.eta_profile):
            if p is not None:
                return int(p.shape[1])
        return None

    half_bandwidth: int = 2  # of each per-component precision block

    def curvature_weights(self, n_pixels: int):
        """Per-row curvature weights, shape ``(n_comp, n_pixels - 2)``.

        Row ``k`` of ``D2`` spans pixels ``k, k+1, k+2`` and takes the profile value of
        its center pixel ``k+1``, so a window's edge rows are weighted by whether the
        center of the stencil is inside.
        """
        self._check_pixels(n_pixels)
        if self.tau_profile is None:
            return jnp.broadcast_to(self.tau[:, None], (self.n_components, n_pixels - 2))
        return self.tau[:, None] * self.tau_profile[:, 1:-1]

    def ridge_weights(self, n_pixels: int):
        """Per-pixel ridge weights, shape ``(n_comp, n_pixels)``."""
        self._check_pixels(n_pixels)
        if self.eta_profile is None:
            return jnp.broadcast_to(self.eta[:, None], (self.n_components, n_pixels))
        return self.eta[:, None] * self.eta_profile

    def _check_pixels(self, n_pixels: int) -> None:
        own = self.n_pixels
        if own is not None and own != n_pixels:
            raise ValueError(
                f"prior profiles were built for {own} pixels but the model grid has "
                f"{n_pixels}. A profile is tied to the grid it was built on: rebuild it "
                "with window_profile(grid.wave, ...)."
            )

    def apply(self, d_stack):
        """Apply the block-diagonal precision to stacked spectra, shape ``(n_comp, n)``."""
        d_stack = jnp.asarray(d_stack)
        if d_stack.ndim != 2 or d_stack.shape[0] != self.n_components:
            raise ValueError(
                f"expected d_stack of shape ({self.n_components}, n); got {d_stack.shape}"
            )
        n_pix = d_stack.shape[1]
        curv = second_difference_adjoint(self.curvature_weights(n_pix) * second_difference(d_stack))
        return curv + self.ridge_weights(n_pix) * d_stack

    def dense(self, n: int) -> np.ndarray:
        """Dense ``(n_comp * n, n_comp * n)`` precision, for small-problem tests only."""
        d2 = np.zeros((n - 2, n))
        idx = np.arange(n - 2)
        d2[idx, idx] = 1.0
        d2[idx, idx + 1] = -2.0
        d2[idx, idx + 2] = 1.0
        t = np.asarray(self.curvature_weights(n))
        e = np.asarray(self.ridge_weights(n))
        out = np.zeros((self.n_components * n, self.n_components * n))
        for i in range(self.n_components):
            block = d2.T @ (t[i][:, None] * d2) + np.diag(e[i])
            out[i * n : (i + 1) * n, i * n : (i + 1) * n] = block
        return out

    def tree_flatten(self):
        return (self.tau, self.eta, self.tau_profile, self.eta_profile), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

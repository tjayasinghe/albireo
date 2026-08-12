"""Fixed-parameter forward model: grouped epoch operators, matvecs, and adjoints.

Implements the affine model of ``docs/math.md`` §1.4 conditional on the nonlinear
parameters (velocities, light fractions, LSF, response):

    y_j = r_j ⊙ [ R_j (1 + B_j sum_i l_ij T(delta_ij) d_i) ] + n_j

Epochs are grouped by instrument so that all epochs in a group share the (static) rebin
operator and LSF kernel and can be batched with ``vmap``. The response enters only
through effective weights and targets: with ``C_j = R_j B_j sum_i l_ij T_ij`` the normal
equations use ``C^T diag(r^2 w) C`` and ``C^T (r w z)`` where ``z = y - r ⊙ (R 1)``.

Everything here is linear in the stacked deviation spectra and ships with an exact
adjoint; masked pixels (ivar = 0, incomplete rebin coverage) carry zero weight
everywhere.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from albireo.data import Dataset
from albireo.grids import LogGrid
from albireo.operators import (
    RebinOperator,
    gaussian_kernel,
    gaussian_kernel_traced,
    rebin_operator,
    rebin_pair_tables,
    shift_spectrum,
)
from albireo.operators import shift_spectrum_adjoint as shift_adjoint
from albireo.simulate import chebyshev_response

__all__ = [
    "EpochGroup",
    "Problem",
    "apply_model",
    "apply_model_adjoint",
    "build_problem",
    "data_residual_zscores",
    "normal_matvec",
    "rhs",
    "weighted_data_terms",
    "with_light_fractions",
    "with_lsf",
    "with_velocities",
]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EpochGroup:
    """All epochs sharing one instrument (one rebin operator, one LSF kernel).

    Registered as a pytree so a whole :class:`Problem` can be passed as a ``jax.jit``
    *argument* — its arrays then enter the graph as runtime parameters rather than
    embedded constants, which at scale would trigger multi-GB XLA constant folding.
    """

    instrument: str
    epoch_indices: tuple[int, ...]
    rebin: RebinOperator
    kernel: jax.Array
    kernel_rev: jax.Array
    shifts: jax.Array  # (n_epochs, n_comp) shift in model pixels
    light: jax.Array  # (n_epochs, n_comp)
    z: jax.Array  # (n_epochs, n_native) y - r * (R 1)
    w: jax.Array  # (n_epochs, n_native) effective ivar (masks + coverage folded in)
    r: jax.Array  # (n_epochs, n_native) response
    row_support: int  # max model-pixel span of a rebin row (bandwidth bookkeeping)
    bary_pix: jax.Array  # (n_epochs,) barycentric shift in model pixels (static)
    pair_val: jax.Array  # static rebin pair tables for direct band assembly
    pair_sid: jax.Array  # (albireo.operators.rebin_pair_tables; sid = c * row_support + o)
    pair_row: jax.Array

    @property
    def n_epochs(self) -> int:
        return self.shifts.shape[0]

    def tree_flatten(self):
        children = (
            self.rebin,
            self.kernel,
            self.kernel_rev,
            self.shifts,
            self.light,
            self.z,
            self.w,
            self.r,
            self.bary_pix,
            self.pair_val,
            self.pair_sid,
            self.pair_row,
        )
        return children, (self.instrument, self.epoch_indices, self.row_support)

    @classmethod
    def tree_unflatten(cls, aux, children):
        rebin, kernel, kernel_rev, shifts, light, z, w, r, bary_pix, pv, ps, pr = children
        instrument, epoch_indices, row_support = aux
        return cls(
            instrument=instrument,
            epoch_indices=epoch_indices,
            rebin=rebin,
            kernel=kernel,
            kernel_rev=kernel_rev,
            shifts=shifts,
            light=light,
            z=z,
            w=w,
            r=r,
            row_support=row_support,
            bary_pix=bary_pix,
            pair_val=pv,
            pair_sid=ps,
            pair_row=pr,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Problem:
    """A fixed-parameter disentangling problem: data + operators, ready to solve.

    A registered pytree (see :class:`EpochGroup`); ``grid`` is hashable static
    metadata and lives in the aux data.
    """

    grid: LogGrid
    n_components: int
    groups: tuple[EpochGroup, ...]
    frame: str = "barycentric"
    telluric: bool = False

    def tree_flatten(self):
        return (self.groups,), (self.grid, self.n_components, self.frame, self.telluric)

    @classmethod
    def tree_unflatten(cls, aux, children):
        grid, n_components, frame, telluric = aux
        return cls(
            grid=grid,
            n_components=n_components,
            groups=children[0],
            frame=frame,
            telluric=telluric,
        )

    @property
    def n_linear(self) -> int:
        """Dimension of the stacked linear system (n_components * grid pixels)."""
        return self.n_components * self.grid.n

    @property
    def n_stellar(self) -> int:
        return self.n_components - (1 if self.telluric else 0)

    @property
    def n_epochs(self) -> int:
        return sum(len(g.epoch_indices) for g in self.groups)

    @property
    def kernel_radius(self) -> int:
        return max((g.kernel.shape[0] - 1) // 2 for g in self.groups)

    @property
    def max_relative_shift(self) -> float:
        """Max over epochs and component pairs of |delta_i - delta_i'| in pixels."""
        out = 0.0
        for g in self.groups:
            s = np.asarray(g.shifts)
            out = max(out, float(np.max(np.abs(s[:, :, None] - s[:, None, :]))))
        return out

    @property
    def natural_half_bandwidth(self) -> int:
        """Half-bandwidth of A^T W A between any two components, in model pixels."""
        support = max(g.row_support for g in self.groups)
        return int(np.ceil(self.max_relative_shift)) + 1 + 2 * self.kernel_radius + support

    def half_bandwidth_bound(self, v_rel_max_kms: float) -> int:
        """Static upper bound on :attr:`natural_half_bandwidth` given a velocity bound.

        ``v_rel_max_kms`` must bound the largest *relative* radial velocity between any
        two model components at any epoch — for an SB2, ``(K_1 + K_2)(1 + e)`` plus, if
        a telluric component is present, the stellar velocity relative to the telluric
        frame (which includes the barycentric motion, up to ~30 km/s). Because the
        bound is static (independent of the velocity values), it can be passed to
        :func:`albireo.likelihood.marginal_loglikelihood` as ``half_bandwidth`` inside
        ``jax.jit`` with traced velocities. The ``+ 1`` pixel of slack in the bandwidth
        formula also absorbs the (< 1e-5 relative at 1000 km/s) curvature of the
        relativistic velocity-to-shift mapping.
        """
        shift = abs(float(self.grid.velocity_to_pixels(float(v_rel_max_kms))))
        support = max(g.row_support for g in self.groups)
        return int(np.ceil(shift)) + 1 + 2 * self.kernel_radius + support


def build_problem(
    grid: LogGrid,
    dataset: Dataset,
    *,
    velocities,
    light_fractions,
    lsf_sigma_v: Mapping[str, float],
    response_coeffs: Sequence[np.ndarray] | None = None,
    telluric: bool = False,
) -> Problem:
    """Assemble a :class:`Problem` from a dataset and fixed nonlinear parameters.

    Parameters
    ----------
    grid
        Common model grid.
    dataset
        Observed epochs (never resampled; the model is projected onto each native grid).
    velocities
        Stellar radial velocities, shape ``(n_stellar, n_epochs)``, in the barycentric
        frame (km/s). Frame composition with each epoch's ``v_bary`` follows
        ``docs/math.md`` §1.2 using ``dataset.frame``.
    light_fractions
        ``(n_stellar,)`` or ``(n_stellar, n_epochs)``; must sum to 1 per epoch.
    lsf_sigma_v
        Per-instrument Gaussian LSF width (km/s).
    response_coeffs
        Optional per-epoch Chebyshev response coefficients (empty/None = unit response).
    telluric
        If True, append a telluric component (light fraction 1) whose velocity law is
        the topocentric one — static for topocentric-frame data, ``+v_bary`` for
        barycentric-frame data.
    """
    vel = np.atleast_2d(np.asarray(velocities, dtype=np.float64))
    n_stellar, n_ep = vel.shape
    if n_ep != dataset.n_epochs:
        raise ValueError(f"velocities has {n_ep} epochs, dataset has {dataset.n_epochs}")

    ell = np.asarray(light_fractions, dtype=np.float64)
    if ell.ndim == 1:
        ell = np.repeat(ell[:, None], n_ep, axis=1)
    if ell.shape != (n_stellar, n_ep):
        raise ValueError(f"light_fractions must be ({n_stellar},) or ({n_stellar}, {n_ep})")
    if not np.allclose(ell.sum(axis=0), 1.0, atol=1e-10):
        raise ValueError("light fractions must sum to 1 at every epoch")

    # Frame composition (log-shifts are exactly additive; docs/math.md §1.2).
    v_bary = dataset.v_bary
    bary_pix = np.asarray(grid.velocity_to_pixels(v_bary))
    star_pix = np.asarray(grid.velocity_to_pixels(vel))
    if dataset.frame == "topocentric":
        star_pix = star_pix - bary_pix[None, :]
        tell_pix = np.zeros(n_ep)
    else:
        tell_pix = bary_pix

    shifts = np.vstack([star_pix, tell_pix[None, :]]) if telluric else star_pix
    light = np.vstack([ell, np.ones((1, n_ep))]) if telluric else ell
    n_comp = shifts.shape[0]

    if response_coeffs is None:
        response_coeffs = [np.zeros(0)] * n_ep
    if len(response_coeffs) != n_ep:
        raise ValueError("response_coeffs must have one entry per epoch")

    # Group epochs by instrument; all epochs in a group share static operators.
    by_instrument: dict[str, list[int]] = {}
    for j, ep in enumerate(dataset):
        by_instrument.setdefault(ep.instrument, []).append(j)

    groups = []
    for instrument, idx in by_instrument.items():
        if instrument not in lsf_sigma_v:
            raise ValueError(f"no LSF width supplied for instrument {instrument!r}")
        wave_native = dataset[idx[0]].wave
        for j in idx[1:]:
            if not np.array_equal(dataset[j].wave, wave_native):
                raise ValueError(
                    f"epochs of instrument {instrument!r} have differing wavelength grids; "
                    "give them distinct instrument labels"
                )
        rebin = rebin_operator(x_in=grid.wave, x_out=wave_native)
        coverage = np.asarray(rebin.coverage)
        covered = coverage > 1.0 - 1e-10
        rows = np.asarray(rebin.rows)
        cols = np.asarray(rebin.cols)
        span = np.zeros(wave_native.size, dtype=np.int64)
        np.maximum.at(span, rows, cols)
        lo = np.full(wave_native.size, np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(lo, rows, cols)
        row_support = int(np.max(span - lo) + 1)

        sigma_px = float(lsf_sigma_v[instrument]) / grid.dv_kms
        kernel = gaussian_kernel(sigma_px)
        base = np.asarray(rebin(jnp.ones(grid.n)))  # R 1 (= coverage)

        z_rows, w_rows, r_rows = [], [], []
        for j in idx:
            ep = dataset[j]
            r = chebyshev_response(ep.wave, response_coeffs[j])
            z_rows.append(ep.flux - r * base)
            w_rows.append(np.where(covered, ep.effective_ivar, 0.0))
            r_rows.append(r)

        pair_val, pair_sid, pair_row, pair_h = rebin_pair_tables(rebin)
        if pair_h != row_support:
            raise AssertionError(f"pair-table support {pair_h} != row support {row_support}")

        groups.append(
            EpochGroup(
                instrument=instrument,
                epoch_indices=tuple(idx),
                rebin=rebin,
                kernel=kernel,
                kernel_rev=kernel[::-1],
                shifts=jnp.asarray(shifts[:, idx].T),
                light=jnp.asarray(light[:, idx].T),
                z=jnp.asarray(np.stack(z_rows)),
                w=jnp.asarray(np.stack(w_rows)),
                r=jnp.asarray(np.stack(r_rows)),
                row_support=row_support,
                bary_pix=jnp.asarray(bary_pix[list(idx)]),
                pair_val=pair_val,
                pair_sid=pair_sid,
                pair_row=pair_row,
            )
        )

    return Problem(
        grid=grid,
        n_components=n_comp,
        groups=tuple(groups),
        frame=dataset.frame,
        telluric=telluric,
    )


def with_velocities(problem: Problem, velocities) -> Problem:
    """Return ``problem`` with the stellar velocities replaced (differentiable in them).

    This is the θ-dependent path for joint inference: only the per-epoch shift columns
    are recomputed — with the same frame composition as :func:`build_problem` — while
    every static piece (rebin operators, kernels, weights, targets, response) is reused
    unchanged. Safe to call inside ``jax.jit`` with traced ``velocities``; combine with
    :meth:`Problem.half_bandwidth_bound` for a static solver bandwidth.

    Parameters
    ----------
    problem
        Output of :func:`build_problem` (any velocities).
    velocities
        Stellar radial velocities in the barycentric frame, shape
        ``(n_stellar, n_epochs)`` (km/s). The telluric column, if present, is
        reconstructed from the stored barycentric shifts.
    """
    vel = jnp.atleast_2d(jnp.asarray(velocities))
    if vel.shape != (problem.n_stellar, problem.n_epochs):
        raise ValueError(
            f"velocities must have shape ({problem.n_stellar}, {problem.n_epochs}); got {vel.shape}"
        )
    star_pix = problem.grid.velocity_to_pixels(vel)
    groups = []
    for g in problem.groups:
        idx = list(g.epoch_indices)
        sp = star_pix[:, idx].T  # (n_epochs_group, n_stellar)
        if problem.frame == "topocentric":
            sp = sp - g.bary_pix[:, None]
            tell_col = jnp.zeros((len(idx), 1))
        else:
            tell_col = g.bary_pix[:, None]
        if problem.telluric:
            sp = jnp.concatenate([sp, tell_col], axis=1)
        groups.append(replace(g, shifts=sp))
    return replace(problem, groups=tuple(groups))


def with_light_fractions(problem: Problem, light_fractions) -> Problem:
    """Return ``problem`` with the stellar light fractions replaced (differentiable).

    The θ-dependent path for light-fraction inference: only the per-epoch light
    columns are swapped; the telluric column (if present) keeps light fraction 1.
    Safe inside ``jax.jit`` with traced values. The simplex constraint (non-negative,
    sum to 1 per epoch) cannot be checked on traced input — it is the caller's
    responsibility (in the numpyro model it is guaranteed by a Dirichlet prior).

    Parameters
    ----------
    problem
        Output of :func:`build_problem`.
    light_fractions
        ``(n_stellar,)`` constant or ``(n_stellar, n_epochs)`` per-epoch.
    """
    ell = jnp.asarray(light_fractions)
    if ell.ndim == 1:
        ell = jnp.broadcast_to(ell[:, None], (ell.shape[0], problem.n_epochs))
    if ell.shape != (problem.n_stellar, problem.n_epochs):
        raise ValueError(
            f"light_fractions must have shape ({problem.n_stellar},) or "
            f"({problem.n_stellar}, {problem.n_epochs}); got {ell.shape}"
        )
    groups = []
    for g in problem.groups:
        idx = list(g.epoch_indices)
        le = ell[:, idx].T  # (n_epochs_group, n_stellar)
        if problem.telluric:
            le = jnp.concatenate([le, jnp.ones((len(idx), 1))], axis=1)
        groups.append(replace(g, light=le))
    return replace(problem, groups=tuple(groups))


def with_lsf(problem: Problem, lsf_sigma_v: Mapping) -> Problem:
    """Return ``problem`` with the Gaussian LSF widths replaced (differentiable).

    The θ-dependent path for LSF inference: each group's kernel *values* are
    recomputed from the traced width while the kernel *radius* stays the one fixed
    at :func:`build_problem` time — so the ``lsf_sigma_v`` passed at build time must
    be an upper bound on any width used here (a larger width would be truncated by
    the fixed radius; the inference model rejects that region). Safe inside
    ``jax.jit``; widths must be positive (enforce via the prior's support).

    Parameters
    ----------
    problem
        Output of :func:`build_problem`.
    lsf_sigma_v
        Per-instrument Gaussian LSF width in km/s (traced scalars allowed); must
        cover every instrument in the problem.
    """
    groups = []
    for g in problem.groups:
        if g.instrument not in lsf_sigma_v:
            raise ValueError(f"no LSF width supplied for instrument {g.instrument!r}")
        sigma_px = jnp.asarray(lsf_sigma_v[g.instrument]) / problem.grid.dv_kms
        radius = (g.kernel.shape[0] - 1) // 2
        kernel = gaussian_kernel_traced(sigma_px, radius)
        groups.append(replace(g, kernel=kernel, kernel_rev=kernel[::-1]))
    return replace(problem, groups=tuple(groups))


# ---------------------------------------------------------------------------
# Linear maps (all exact adjoint pairs)
# ---------------------------------------------------------------------------


def _epoch_model(group: EpochGroup, d_stack):
    """Per-epoch model deviation on the native grid: ``R B sum_i l_i T_i d_i``."""

    def one_epoch(shifts_e, light_e):
        acc = jnp.zeros(d_stack.shape[1])
        for i in range(d_stack.shape[0]):
            acc = acc + light_e[i] * shift_spectrum(d_stack[i], shifts_e[i])
        conv = jnp.convolve(acc, group.kernel, mode="same")
        return group.rebin(conv)

    return jax.vmap(one_epoch)(group.shifts, group.light)


def _epoch_model_adjoint(group: EpochGroup, v):
    """Adjoint of :func:`_epoch_model`: native-space ``(n_epochs, n_native)`` -> stack."""
    n_comp = group.shifts.shape[1]

    def one_epoch(v_e, shifts_e, light_e):
        t = group.rebin.adjoint(v_e)
        t = jnp.convolve(t, group.kernel_rev, mode="same")
        return jnp.stack([light_e[i] * shift_adjoint(t, shifts_e[i]) for i in range(n_comp)])

    return jnp.sum(jax.vmap(one_epoch)(v, group.shifts, group.light), axis=0)


def apply_model(problem: Problem, d_stack):
    """Model deviations per group: list of ``(n_epochs, n_native)`` arrays (no response)."""
    return [_epoch_model(g, jnp.asarray(d_stack)) for g in problem.groups]


def apply_model_adjoint(problem: Problem, per_group):
    """Adjoint of :func:`apply_model`."""
    out = jnp.zeros((problem.n_components, problem.grid.n))
    for g, v in zip(problem.groups, per_group, strict=True):
        out = out + _epoch_model_adjoint(g, jnp.asarray(v))
    return out


def normal_matvec(problem: Problem, d_stack):
    """Apply ``A^T W A`` (data term of the posterior precision), matrix-free."""
    per_group = []
    for g in problem.groups:
        m = _epoch_model(g, jnp.asarray(d_stack))
        per_group.append(g.w * g.r**2 * m)
    return apply_model_adjoint(problem, per_group)


def rhs(problem: Problem):
    """Right-hand side ``b = A^T W z``, shape ``(n_comp, n)``."""
    return apply_model_adjoint(problem, [g.w * g.r * g.z for g in problem.groups])


def weighted_data_terms(problem: Problem):
    """Data-only scalars of the marginal likelihood: ``(z^T W z, sum log w_good, n_good)``."""
    zwz, logw, n_good = 0.0, 0.0, 0
    for g in problem.groups:
        good = g.w > 0
        zwz = zwz + jnp.sum(g.w * g.z**2)
        logw = logw + jnp.sum(jnp.where(good, jnp.log(jnp.where(good, g.w, 1.0)), 0.0))
        n_good = n_good + jnp.sum(good)
    return zwz, logw, n_good


def data_residual_zscores(problem: Problem, d_stack) -> np.ndarray:
    """Whitened data residuals ``(z - r * model) sqrt(w)`` over unmasked pixels."""
    out = []
    for g, m in zip(problem.groups, apply_model(problem, d_stack), strict=True):
        resid = np.asarray((g.z - g.r * m) * jnp.sqrt(g.w))
        out.append(resid[np.asarray(g.w) > 0])
    return np.concatenate(out)

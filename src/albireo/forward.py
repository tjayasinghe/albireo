"""Fixed-parameter forward model: grouped epoch operators, matvecs, and adjoints.

Implements the affine model of ``docs/math.md`` §1.4 conditional on the nonlinear
parameters (velocities, light fractions, LSF, response):

    y_j = r_j ⊙ [ R_j (1 + B_j sum_i l_ij T(delta_ij) d_i) ] + n_j

Epochs are grouped by instrument *and* native wavelength grid so that all epochs in a
group share the (static) rebin operator and LSF kernel and can be batched with ``vmap``
(see :func:`_epoch_groups`). The response enters only
through effective weights and targets: with ``C_j = R_j B_j sum_i l_ij T_ij`` the normal
equations use ``C^T diag(r^2 w) C`` and ``C^T (r w z)`` where ``z = y - r ⊙ (R 1)``.

Everything here is linear in the stacked deviation spectra and ships with an exact
adjoint; masked pixels (ivar = 0, incomplete rebin coverage) carry zero weight
everywhere.
"""

from __future__ import annotations

import hashlib
import warnings
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
    "with_jitter",
    "with_light_fractions",
    "with_lsf",
    "with_velocities",
]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EpochGroup:
    """All epochs sharing one instrument *and* one native grid (one rebin operator, one LSF).

    One instrument can own several groups when its exposures do not share a wavelength
    array (:func:`_epoch_groups`); ``instrument`` is then the same string in each, since
    it is the key into the LSF and response tables.

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
    jitter: jax.Array  # (n_epochs,) noise-inflation factor alpha_j; see effective_w
    row_support: int  # max model-pixel span of a rebin row (bandwidth bookkeeping)
    bary_pix: jax.Array  # (n_epochs,) barycentric shift in model pixels (static)
    pair_val: jax.Array  # static rebin pair tables for direct band assembly
    pair_sid: jax.Array  # (albireo.operators.rebin_pair_tables; sid = c * row_support + o)
    pair_row: jax.Array

    @property
    def n_epochs(self) -> int:
        return self.shifts.shape[0]

    @property
    def effective_w(self):
        """The weights the likelihood actually uses: ``w / alpha_j^2`` (docs/math.md §1.4).

        Every consumer of the weights reads *this*, never :attr:`w`, so a jitter factor
        enters the normal equations, the right-hand side, and the ``sum log w`` term of
        the marginal likelihood consistently — which is what makes it identifiable
        (inflating the noise buys misfit at the cost of that determinant term).
        :attr:`w` stays the measurement's own inverse variance, so applying a jitter is
        idempotent rather than cumulative.
        """
        return self.w / self.jitter[:, None] ** 2

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
            self.jitter,
            self.bary_pix,
            self.pair_val,
            self.pair_sid,
            self.pair_row,
        )
        return children, (self.instrument, self.epoch_indices, self.row_support)

    @classmethod
    def tree_unflatten(cls, aux, children):
        rebin, kernel, kernel_rev, shifts, light, z, w, r, jit, bary_pix, pv, ps, pr = children
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
            jitter=jit,
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


def _warn_if_row_support_is_an_artifact(instrument, wave_native, per_row, row_support):
    """Flag a rebin row support set by a few anomalously wide native pixels.

    ``row_support`` is a *max* over native pixels and it propagates straight into the
    solver half-bandwidth, whose cost is quadratic in the block size. A handful of wide
    rows therefore taxes the whole run. In real spectra they are usually not wide pixels
    at all but *deleted* samples — telluric windows, cosmic-ray hits, order or chip gaps
    — because :func:`albireo.operators.bin_edges_from_centers` places edges at
    midpoints, so dropping samples makes the two bracketing pixels absorb half the gap
    each. Masking (``ivar = 0``) keeps the sampling regular; deleting does not.
    """
    # Only rows the operator actually touches: a native pixel lying entirely outside the
    # model grid has no rebin entries at all, and counting it as support 0 would drag the
    # median down (it used to do worse — the empty rows carried a sentinel, so a
    # region-selected epoch could produce a warning quoting int64 minimum as the median).
    covered = per_row > 0
    if not covered.any():
        return
    med = float(np.median(per_row[covered]))
    if row_support <= max(4.0 * med, med + 4.0):
        return
    wide = np.flatnonzero(per_row > max(4.0 * med, med + 4.0))
    k = int(wide[np.argmax(per_row[wide])])
    warnings.warn(
        f"instrument {instrument!r}: {wide.size} of {per_row.size} native pixels span far "
        f"more model pixels than typical (worst is pixel {k} at {wave_native[k]:.4f} with "
        f"{row_support}, median {med:.0f}). The solver bandwidth follows this maximum, so "
        f"the run costs roughly ({row_support / max(med, 1.0):.0f}x)^2 more than it need. "
        "This pattern almost always means samples were *deleted* (telluric window, bad "
        "pixels, order or chip gap) rather than genuinely wide pixels: keep the pixels and "
        "set their ivar to 0 instead, or give the disjoint segments distinct instrument "
        "labels.",
        RuntimeWarning,
        stacklevel=3,
    )


def _warn_if_data_extends_past_grid(instrument, dataset, idx, covered) -> None:
    """Flag weighted native pixels that the model grid does not fully cover.

    Such pixels are silently zero-weighted (``w = 0`` where ``coverage < 1``), so the fit
    quietly discards data the user believes it is using. It is also the visible end of a
    more damaging condition: near a grid edge the shift and LSF operators zero-fill, so
    the *covered* pixels within a shift-plus-kernel-radius of the boundary are modelled
    with missing flux while still carrying full weight. :meth:`albireo.grids.LogGrid.covering`
    sizes the margin correctly; this warning catches the case where it was not used.
    """
    weighted_outside = 0
    for j in idx:
        weighted_outside += int(np.count_nonzero((dataset[j].effective_ivar > 0) & ~covered))
    if not weighted_outside:
        return
    warnings.warn(
        f"instrument {instrument!r}: {weighted_outside} pixel(s) with positive weight lie "
        "outside the model grid and have been zero-weighted. Widen the grid — and by more "
        "than the bare data range: it must also clear the largest component shift plus the "
        "LSF kernel radius, or the pixels just *inside* the edge are modelled with "
        "zero-filled flux at full weight. LogGrid.covering(dataset, dv_kms, "
        "v_margin_kms=..., lsf_sigma_kms=...) computes the margin.",
        RuntimeWarning,
        stacklevel=3,
    )


def _epoch_groups(dataset: Dataset) -> list[tuple[str, list[int]]]:
    """Partition epoch indices into ``(instrument, indices)`` sharing one native grid.

    A group is the unit that shares static operators, so it must be an instrument *and*
    a single wavelength array — one rebin operator serves the whole group. Instruments
    whose epochs sit on a common grid (simulations, and any pipeline that resamples every
    exposure onto one wavelength solution) therefore give exactly one group each, batched
    by ``vmap`` as before.

    Pipelines that apply the barycentric correction by shifting *before* rebinning do not:
    ESO Phase-3 FEROS spectra, for instance, carry a per-exposure grid whose start moves
    with the correction, so 51 epochs can have 51 grids differing in length and in
    sub-pixel phase. Splitting them here is exact — each epoch keeps its own grid and its
    own operator, per ``docs/design.md`` D4 — and costs one operator per distinct grid,
    which is small next to the solve. The alternative available to callers, relabelling
    the epochs as distinct *instruments*, would also fork the LSF width and response
    tables, so widths that are physically one number would have to be inferred (or
    supplied) once per exposure. The instrument key is what identifies the LSF, so it
    stays shared across the subgroups here.

    Grids are matched by content hash, so the partition costs one pass over the data
    rather than a comparison against every group seen so far.
    """
    order: list[tuple[str, list[int]]] = []
    seen: dict[tuple[str, bytes], int] = {}
    for j, epoch in enumerate(dataset):
        key = (epoch.instrument, hashlib.blake2b(epoch.wave.tobytes(), digest_size=16).digest())
        slot = seen.get(key)
        if slot is None or not np.array_equal(dataset[order[slot][1][0]].wave, epoch.wave):
            seen[key] = len(order)  # a hash collision (never observed) just makes a group
            order.append((epoch.instrument, [j]))
        else:
            order[slot][1].append(j)
    return order


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

    groups = []
    for instrument, idx in _epoch_groups(dataset):
        if instrument not in lsf_sigma_v:
            raise ValueError(f"no LSF width supplied for instrument {instrument!r}")
        wave_native = dataset[idx[0]].wave
        rebin = rebin_operator(x_in=grid.wave, x_out=wave_native)
        if np.asarray(rebin.rows).size == 0:
            raise ValueError(
                f"instrument {instrument!r}: the model grid ({grid.wave[0]:.3f}-"
                f"{grid.wave[-1]:.3f}) does not overlap this epoch's wavelengths "
                f"({wave_native[0]:.3f}-{wave_native[-1]:.3f}), so there is nothing to fit. "
                "Build the grid from the data — LogGrid.covering(dataset, dv_kms, ...)."
            )
        coverage = np.asarray(rebin.coverage)
        covered = coverage > 1.0 - 1e-10
        rows = np.asarray(rebin.rows)
        cols = np.asarray(rebin.cols)
        touched = np.bincount(rows, minlength=wave_native.size) > 0
        span = np.zeros(wave_native.size, dtype=np.int64)
        np.maximum.at(span, rows, cols)
        lo = np.full(wave_native.size, np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(lo, rows, cols)
        per_row = np.where(touched, span - np.where(touched, lo, 0) + 1, 0)
        row_support = int(np.max(per_row))
        _warn_if_row_support_is_an_artifact(instrument, wave_native, per_row, row_support)
        _warn_if_data_extends_past_grid(instrument, dataset, idx, covered)

        sigma_px = float(lsf_sigma_v[instrument]) / grid.dv_kms
        kernel = gaussian_kernel(sigma_px)
        base = np.asarray(rebin(jnp.ones(grid.n)))  # R 1 (= coverage)

        z_rows, w_rows, r_rows = [], [], []
        for j in idx:
            ep = dataset[j]
            r = chebyshev_response(ep.wave, response_coeffs[j])
            w = np.where(covered, ep.effective_ivar, 0.0)
            # Zero-weight pixels are allowed to hold anything, including nan and inf
            # (albireo.data: a masked pixel's flux "is never read"). Every consumer of z
            # multiplies it by w, so zeroing it here changes nothing — except that
            # `0 * nan` is `nan`, and one such pixel takes the whole marginal likelihood
            # to nan. Real data reaches this path routinely: albireo.preprocess.normalize
            # marks pixels where the fitted continuum collapses by writing nan.
            flux = np.where(np.isfinite(ep.flux), ep.flux, 0.0)
            z_rows.append(np.where(w > 0.0, flux - r * base, 0.0))
            w_rows.append(w)
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
                jitter=jnp.ones(len(idx)),
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


def with_jitter(problem: Problem, jitter) -> Problem:
    """Return ``problem`` with per-epoch noise-inflation factors ``alpha_j`` (differentiable).

    ``docs/math.md`` §1.4 / ``docs/design.md`` D15: the weights become
    ``w_j -> w_j / alpha_j^2``, so ``alpha = 1`` is exactly the unmodified problem and
    ``alpha > 1`` says this epoch's quoted inverse variances are optimistic by that
    factor. The marginal likelihood keeps its ``+1/2 sum log w`` term, which is what
    makes ``alpha`` identifiable rather than a free knob, and it supplies the right
    denominator for free. In the data-dominated limit ``-1/2 log det(Lambda + A^T W A)``
    contributes ``+p_eff log alpha`` against that term's ``-N log alpha``, so profiling
    gives ``alpha^2 = chi2 / (N - p_eff)`` with ``p_eff = tr[(Lambda + A^T W A)^-1 A^T W A]``
    the effective number of parameters the marginalized spectra consume. Whitening the
    residuals by hand and reading off their standard deviation instead gives ``chi2 / N``,
    low by ``sqrt(1 - p_eff/N)``. How much that matters is a property of the run, and
    ``p_eff`` is the *effective* count, not ``n_comp * n_pix``: an oversampled model grid
    with a fitted smoothness prior can put it an order of magnitude below the pixel count
    (measured on HR 6819: ~2900 against 19,876 pixels, so the correction was 0.4% — while
    the weak-prior fixture in ``tests/test_jitter.py`` sees 4.6%). Being joint with the
    orbit, the widened uncertainties then propagate.

    Two things this is and is not. It is the right handle for archival spectra whose
    inverse variances were *estimated* rather than measured
    (:func:`albireo.preprocess.estimate_ivar`), where a scale error is expected and
    unknowable a priori. It is **not** a repair for unmodelled structure: a jitter fitted
    against systematics (imperfect continuum, LSF mismatch, line-profile variability)
    reports a wider — but still wrong — orbit, because inflating a diagonal noise model
    cannot represent a residual that is correlated across pixels. Check
    :func:`data_residual_zscores` for structure before trusting the widening; on real
    data the honest error bar is usually still the scatter between independent
    wavelength windows.

    Parameters
    ----------
    problem
        Output of :func:`build_problem`.
    jitter
        Scalar (one factor shared by every epoch) or ``(n_epochs,)``. Must be positive;
        supply it as ``exp(log_jitter)`` from an unconstrained parameter. Traced values
        are fine, so this is safe inside ``jax.jit``.
    """
    alpha = jnp.asarray(jitter)
    if alpha.ndim == 0:
        alpha = jnp.broadcast_to(alpha, (problem.n_epochs,))
    if alpha.shape != (problem.n_epochs,):
        raise ValueError(
            f"jitter must be a scalar or have shape ({problem.n_epochs},); got {alpha.shape}"
        )
    groups = [
        replace(g, jitter=alpha[np.asarray(g.epoch_indices, dtype=np.int64)])
        for g in problem.groups
    ]
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
        per_group.append(g.effective_w * g.r**2 * m)
    return apply_model_adjoint(problem, per_group)


def rhs(problem: Problem):
    """Right-hand side ``b = A^T W z``, shape ``(n_comp, n)``."""
    return apply_model_adjoint(problem, [g.effective_w * g.r * g.z for g in problem.groups])


def weighted_data_terms(problem: Problem):
    """Data-only scalars of the marginal likelihood: ``(z^T W z, sum log w_good, n_good)``."""
    zwz = jnp.asarray(0.0)
    logw = jnp.asarray(0.0)
    n_good = jnp.asarray(0)
    for g in problem.groups:
        good = g.w > 0
        zwz = zwz + jnp.sum(g.effective_w * g.z**2)
        # sum log(w / alpha^2) = sum log w - 2 n_j log alpha_j, split so that the jitter
        # gradient runs through one scalar per epoch instead of an (n_ep, n_native) log.
        logw = logw + jnp.sum(jnp.where(good, jnp.log(jnp.where(good, g.w, 1.0)), 0.0))
        logw = logw - 2.0 * jnp.sum(jnp.sum(good, axis=1) * jnp.log(g.jitter))
        n_good = n_good + jnp.sum(good)
    return zwz, logw, n_good


def data_residual_zscores(problem: Problem, d_stack) -> np.ndarray:
    """Whitened data residuals ``(z - r * model) sqrt(w)`` over unmasked pixels.

    Whitened by the weights the model *assumes*, jitter included, so a standard
    deviation of 1 always means "the noise model matches the residuals". With no
    jitter that is a statement about the supplied inverse variances; with a fitted
    jitter it is close to 1 by construction, and the diagnostic that still bites is the
    *shape* of the distribution (correlated structure, outlying epochs), which no
    diagonal inflation can fix.
    """
    out = []
    for g, m in zip(problem.groups, apply_model(problem, d_stack), strict=True):
        resid = np.asarray((g.z - g.r * m) * jnp.sqrt(g.effective_w))
        out.append(resid[np.asarray(g.w) > 0])
    return np.concatenate(out)

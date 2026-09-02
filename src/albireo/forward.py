"""Fixed-parameter forward model: grouped epoch operators, matvecs, and adjoints.

Implements the affine model of ``docs/math.md`` §1.4 conditional on the nonlinear
parameters (velocities, light fractions, LSF, response):

    y_j = r_j ⊙ [ R_j (1 + B_j sum_i l_ij T(delta_ij) d_i) ] + n_j

Epochs are grouped by instrument and native wavelength grid, so that all epochs in a
group share the (static) rebin operator and LSF kernel and can be batched with ``vmap``
(see :func:`_epoch_groups`). The response enters only through effective weights and
targets: with ``C_j = R_j B_j sum_i l_ij T_ij`` the normal equations use
``C^T diag(r^2 w) C`` and ``C^T (r w z)`` where ``z = y - r ⊙ (R 1)``.

The noise model is diagonal by default and AR(1) over each epoch's chain of unmasked
pixels once :func:`with_ar1` is applied (``docs/math.md`` §1.4a).

Everything here is linear in the stacked deviation spectra and has an exact adjoint;
masked pixels (ivar = 0, incomplete rebin coverage) carry zero weight everywhere.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from albireo.data import Dataset
from albireo.grids import LogGrid
from albireo.operators import (
    RebinOperator,
    convolve_varying,
    convolve_varying_adjoint,
    gauss_hermite_kernel_traced,
    gaussian_kernel,
    gaussian_kernel_traced,
    gaussian_lsf_profiles,
    lsf_anchor_tables,
    rebin_link_pair_tables,
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
    "apply_noise_precision",
    "ar1_band_weights",
    "build_problem",
    "data_residual_zscores",
    "normal_matvec",
    "rhs",
    "weighted_data_terms",
    "with_ar1",
    "with_jitter",
    "with_light_fractions",
    "with_lsf",
    "with_nebular_amplitudes",
    "with_response",
    "with_velocities",
]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EpochGroup:
    """All epochs sharing one instrument and one native grid (one rebin operator, one LSF).

    One instrument can own several groups when its exposures do not share a wavelength
    array (:func:`_epoch_groups`); ``instrument`` is then the same string in each, since
    it is the key into the LSF and response tables.

    Registered as a pytree so a whole :class:`Problem` can be passed as a ``jax.jit``
    argument: its arrays then enter the graph as runtime parameters rather than embedded
    constants, which at scale would trigger multi-GB XLA constant folding.
    """

    instrument: str
    epoch_indices: tuple[int, ...]
    rebin: RebinOperator
    kernel: jax.Array  # (n_rows, 2r+1) LSF profile bank: one row = stationary kernel
    #   (applied with jnp.convolve, exactly the v1 operator); n_grid rows = per-model-pixel
    #   profiles in the same convention (operators.convolve_varying), realized from
    #   per-anchor kernels via the static lsf_anchor_wave interpolation tables (D37)
    shifts: jax.Array  # (n_epochs, n_comp) shift in model pixels
    light: jax.Array  # (n_epochs, n_comp)
    z: jax.Array  # (n_epochs, n_native) y - r * (R 1)
    w: jax.Array  # (n_epochs, n_native) effective ivar (masks + coverage folded in)
    r: jax.Array  # (n_epochs, n_native) response
    jitter: jax.Array  # (n_epochs,) noise-inflation factor alpha_j; see effective_w
    row_support: int  # max model-pixel span of a rebin row (bandwidth bookkeeping)
    bary_pix: jax.Array  # (n_epochs,) barycentric shift in model pixels (static)
    base: jax.Array  # (n_native,) R 1, the response-independent rebinned unit continuum
    cheb_x: jax.Array  # (n_native,) response Chebyshev abscissa 2(l - l_0)/(l_N - l_0) - 1
    ar_phi: jax.Array  # (n_epochs,) AR(1) correlation of the standardized noise (0 = diagonal)
    ar_gap: jax.Array  # (n_epochs, n_native) int link table: distance to the previous good
    #   pixel (0 = no link; capped at build_problem's ar1_max_gap); see with_ar1
    ar_step: int  # static max model-pixel offset between a stored link's rebin supports
    pair_val: jax.Array  # static rebin pair tables for direct band assembly
    pair_sid: jax.Array  # (albireo.operators.rebin_pair_tables; sid = c * row_support + o)
    pair_row: jax.Array
    link_val: jax.Array  # static cross-row pair tables for the AR(1) band assembly
    link_sid: jax.Array  # (albireo.operators.rebin_link_pair_tables;
    link_row: jax.Array  #  sid = cmin * (row_support + ar_step) + o); link_row is the
    link_gap: jax.Array  #  link's later endpoint, link_gap its stored native-pixel gap
    # Static LSF anchor wavelengths [A]; () = stationary. Aux (not traced): with_lsf
    # rebuilds the interpolation tables from these and the (static) grid, so a traced
    # per-anchor width vector can re-realize the profile bank without new structure.
    lsf_anchor_wave: tuple[float, ...] = ()

    @property
    def n_epochs(self) -> int:
        return self.shifts.shape[0]

    @property
    def effective_w(self):
        """The weights the likelihood uses: ``w / alpha_j^2`` (``docs/math.md`` §1.4).

        Every consumer of the weights reads this property, never :attr:`w`, so a jitter
        factor enters the normal equations, the right-hand side, and the ``sum log w``
        term of the marginal likelihood consistently. That consistency is what makes the
        jitter identifiable: inflating the noise reduces the misfit term at the cost of
        the determinant term. :attr:`w` stays the measurement's own inverse variance, so
        applying a jitter is idempotent rather than cumulative.
        """
        return self.w / self.jitter[:, None] ** 2

    def tree_flatten(self):
        children = (
            self.rebin,
            self.kernel,
            self.shifts,
            self.light,
            self.z,
            self.w,
            self.r,
            self.jitter,
            self.bary_pix,
            self.base,
            self.cheb_x,
            self.ar_phi,
            self.ar_gap,
            self.pair_val,
            self.pair_sid,
            self.pair_row,
            self.link_val,
            self.link_sid,
            self.link_row,
            self.link_gap,
        )
        return children, (
            self.instrument,
            self.epoch_indices,
            self.row_support,
            self.ar_step,
            self.lsf_anchor_wave,
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            rebin,
            kernel,
            shifts,
            light,
            z,
            w,
            r,
            jit,
            bary,
            base,
            cx,
            phi,
            gap,
            pv,
            ps,
            pr,
            lv,
            ls,
            lr,
            lg,
        ) = children
        instrument, epoch_indices, row_support, ar_step, lsf_anchor_wave = aux
        return cls(
            instrument=instrument,
            epoch_indices=epoch_indices,
            rebin=rebin,
            kernel=kernel,
            shifts=shifts,
            light=light,
            z=z,
            w=w,
            r=r,
            jitter=jit,
            row_support=row_support,
            bary_pix=bary,
            base=base,
            cheb_x=cx,
            ar_phi=phi,
            ar_gap=gap,
            ar_step=ar_step,
            pair_val=pv,
            pair_sid=ps,
            pair_row=pr,
            link_val=lv,
            link_sid=ls,
            link_row=lr,
            link_gap=lg,
            lsf_anchor_wave=lsf_anchor_wave,
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
    # Static: True once with_ar1 has been applied (even with numerically zero phi).
    # The assembly reads it to include the chain's link terms and widen the
    # bandwidth. This is a structural decision, so it cannot depend on traced values.
    correlated: bool = False
    nebular: bool = False

    def tree_flatten(self):
        return (self.groups,), (
            self.grid,
            self.n_components,
            self.frame,
            self.telluric,
            self.correlated,
            self.nebular,
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        grid, n_components, frame, telluric, correlated, nebular = aux
        return cls(
            grid=grid,
            n_components=n_components,
            groups=children[0],
            frame=frame,
            telluric=telluric,
            correlated=correlated,
            nebular=nebular,
        )

    @property
    def n_linear(self) -> int:
        """Dimension of the stacked linear system (n_components * grid pixels)."""
        return self.n_components * self.grid.n

    @property
    def n_stellar(self) -> int:
        """Stellar components, the ones an orbit moves (component order is stellar,
        telluric, nebular, so the non-stellar columns are always the trailing ones)."""
        return self.n_components - (1 if self.telluric else 0) - (1 if self.nebular else 0)

    @property
    def n_epochs(self) -> int:
        return sum(len(g.epoch_indices) for g in self.groups)

    @property
    def kernel_radius(self) -> int:
        return max((g.kernel.shape[-1] - 1) // 2 for g in self.groups)

    @property
    def max_relative_shift(self) -> float:
        """Max over epochs and component pairs of |delta_i - delta_i'| in pixels."""
        out = 0.0
        for g in self.groups:
            s = np.asarray(g.shifts)
            out = max(out, float(np.max(np.abs(s[:, :, None] - s[:, None, :]))))
        return out

    @property
    def ar_bandwidth_extra(self) -> int:
        """Extra half-bandwidth consumed by the AR(1) noise coupling (static).

        A noise precision that is tridiagonal over the observed chain couples rebin
        rows up to ``ar1_max_gap`` native pixels apart, so ``A^T W A`` widens by the
        largest model-pixel offset between any stored link's row supports. That offset
        is :attr:`EpochGroup.ar_step`, computed exactly at build time, and is zero when
        no links were stored. Read it even for a diagonal problem when a later
        :func:`with_ar1` swap is planned: probing with the widened bandwidth is exact
        either way (D21: an overestimate costs time, an underestimate silently
        corrupts).
        """
        return max(g.ar_step for g in self.groups)

    @property
    def natural_half_bandwidth(self) -> int:
        """Half-bandwidth of A^T W A between any two components, in model pixels."""
        support = max(g.row_support for g in self.groups)
        base = int(np.ceil(self.max_relative_shift)) + 1 + 2 * self.kernel_radius + support
        return base + (self.ar_bandwidth_extra if self.correlated else 0)

    def half_bandwidth_bound(self, v_rel_max_kms: float) -> int:
        """Static upper bound on :attr:`natural_half_bandwidth` given a velocity bound.

        ``v_rel_max_kms`` must bound the largest relative radial velocity between any
        two model components at any epoch: for an SB2, ``(K_1 + K_2)(1 + e)`` plus, if
        a telluric component is present, the stellar velocity relative to the telluric
        frame (which includes the barycentric motion, up to ~30 km/s). A nebular
        component (D40) adds ``|nebular_v_kms| + max K`` against the stars and, since
        the two sit in opposite frames, the barycentric motion again against the
        telluric column.

        Because the bound is static (independent of the velocity values), it can be
        passed to :func:`albireo.likelihood.marginal_loglikelihood` as
        ``half_bandwidth`` inside ``jax.jit`` with traced velocities. The ``+ 1`` pixel
        of slack in the bandwidth formula also absorbs the (< 1e-5 relative at
        1000 km/s) curvature of the relativistic velocity-to-shift mapping.
        """
        shift = abs(float(self.grid.velocity_to_pixels(float(v_rel_max_kms))))
        support = max(g.row_support for g in self.groups)
        return int(np.ceil(shift)) + 1 + 2 * self.kernel_radius + support


def _warn_if_row_support_is_an_artifact(instrument, wave_native, per_row, row_support):
    """Flag a rebin row support set by a few anomalously wide native pixels.

    ``row_support`` is a maximum over native pixels and it propagates into the solver
    half-bandwidth, whose cost is quadratic in the block size, so a handful of wide rows
    taxes the whole run. In real spectra such rows are usually not wide pixels but
    deleted samples (telluric windows, cosmic-ray hits, order or chip gaps):
    :func:`albireo.operators.bin_edges_from_centers` places edges at midpoints, so
    dropping samples makes the two bracketing pixels absorb half the gap each. Masking
    (``ivar = 0``) keeps the sampling regular; deleting does not.
    """
    # Only rows the operator touches: a native pixel lying entirely outside the model
    # grid has no rebin entries at all, and counting it as support 0 would drag the
    # median down.
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
        "pixels, order or chip gap) rather than physically wide pixels: keep the pixels and "
        "set their ivar to 0 instead, or give the disjoint segments distinct instrument "
        "labels.",
        RuntimeWarning,
        stacklevel=3,
    )


def _warn_if_data_extends_past_grid(instrument, dataset, idx, covered) -> None:
    """Flag weighted native pixels that the model grid does not fully cover.

    Such pixels are silently zero-weighted (``w = 0`` where ``coverage < 1``), so the fit
    discards data the caller believes it is using. It is also the visible end of a more
    damaging condition: near a grid edge the shift and LSF operators zero-fill, so the
    covered pixels within a shift-plus-kernel-radius of the boundary are modelled with
    missing flux while still carrying full weight.
    :meth:`albireo.grids.LogGrid.covering` sizes the margin correctly; this warning
    catches the case where it was not used.
    """
    weighted_outside = 0
    for j in idx:
        weighted_outside += int(np.count_nonzero((dataset[j].effective_ivar > 0) & ~covered))
    if not weighted_outside:
        return
    warnings.warn(
        f"instrument {instrument!r}: {weighted_outside} pixel(s) with positive weight lie "
        "outside the model grid and have been zero-weighted. Widen the grid: and by more "
        "than the bare data range: it must also clear the largest component shift plus the "
        "LSF kernel radius, or the pixels just *inside* the edge are modelled with "
        "zero-filled flux at full weight. LogGrid.covering(dataset, dv_kms, "
        "v_margin_kms=..., lsf_sigma_kms=...) computes the margin.",
        RuntimeWarning,
        stacklevel=3,
    )


def _epoch_groups(dataset: Dataset) -> list[tuple[str, list[int]]]:
    """Partition epoch indices into ``(instrument, indices)`` sharing one native grid.

    A group is the unit that shares static operators, so it must be one instrument and
    one wavelength array: a single rebin operator serves the whole group. Instruments
    whose epochs sit on a common grid (simulations, and any pipeline that resamples every
    exposure onto one wavelength solution) therefore give exactly one group each, batched
    by ``vmap``.

    Pipelines that apply the barycentric correction by shifting before rebinning do not.
    ESO Phase-3 FEROS spectra, for instance, carry a per-exposure grid whose start moves
    with the correction, so 51 epochs can have 51 grids differing in length and in
    sub-pixel phase. Splitting them here is exact, each epoch keeping its own grid and
    operator (``internal/design.md`` D4), and costs one operator per distinct grid, which is
    small next to the solve. Relabelling the epochs as distinct instruments would instead
    fork the LSF width and response tables, so widths that are physically one number
    would have to be inferred or supplied once per exposure. The instrument key
    identifies the LSF, so it stays shared across the subgroups here.

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


def _lsf_bank(instrument, lsf_sigma_v, lsf_anchors_angstrom, lsf_h3, grid):
    """Realized LSF profile bank for one instrument (build time, NumPy).

    Returns ``(bank, anchor_wave)``: a ``(1, 2r+1)`` stationary kernel and ``()``
    without anchors, else the ``(grid.n, 2r+1)`` per-model-pixel profiles from
    per-anchor Gauss-Hermite kernels (pure Gaussians when ``lsf_h3`` carries nothing
    for this instrument) through the static interpolation tables. The radius follows
    the largest width (truncated at 4 sigma, as in
    :func:`albireo.operators.gaussian_kernel`), so every later :func:`with_lsf` swap
    bounded by the build widths stays untruncated; ``h3`` does not change the support.
    """
    sig = np.atleast_1d(np.asarray(lsf_sigma_v[instrument], dtype=np.float64))
    if sig.ndim != 1 or np.any(sig <= 0):
        raise ValueError(f"instrument {instrument!r}: LSF widths must be positive scalars")
    anchors = None if lsf_anchors_angstrom is None else lsf_anchors_angstrom.get(instrument)
    h3 = None if lsf_h3 is None else lsf_h3.get(instrument)
    if anchors is None:
        if sig.size != 1:
            raise ValueError(
                f"instrument {instrument!r}: {sig.size} LSF widths supplied but no "
                "anchors: per-anchor widths need lsf_anchors_angstrom for this instrument."
            )
        if h3 is not None:
            raise ValueError(
                f"instrument {instrument!r}: lsf_h3 needs lsf_anchors_angstrom: a "
                "stationary asymmetric LSF is absorbed by the free spectra "
                "(docs/math.md §1.3); only its wavelength variation is identified."
            )
        return jnp.asarray(gaussian_kernel(float(sig[0]) / grid.dv_kms))[None, :], ()
    anchor_wave = tuple(float(x) for x in anchors)
    if sig.size == 1:
        sig = np.full(len(anchor_wave), sig[0])
    if sig.size != len(anchor_wave):
        raise ValueError(
            f"instrument {instrument!r}: {sig.size} LSF widths for "
            f"{len(anchor_wave)} anchors: supply one per anchor (or one scalar)."
        )
    if h3 is not None:
        h3 = np.atleast_1d(np.asarray(h3, dtype=np.float64))
        if h3.size == 1:
            h3 = np.full(len(anchor_wave), h3[0])
        if h3.size != len(anchor_wave):
            raise ValueError(
                f"instrument {instrument!r}: {h3.size} h3 values for "
                f"{len(anchor_wave)} anchors: supply one per anchor (or one scalar)."
            )
    profiles = gaussian_lsf_profiles(sig / grid.dv_kms, anchor_wave, grid.wave, h3=h3)
    return jnp.asarray(profiles), anchor_wave


def build_problem(
    grid: LogGrid,
    dataset: Dataset,
    *,
    velocities,
    light_fractions,
    lsf_sigma_v: Mapping[str, float | Sequence[float]],
    lsf_anchors_angstrom: Mapping[str, Sequence[float]] | None = None,
    lsf_h3: Mapping[str, float | Sequence[float]] | None = None,
    response_coeffs: Sequence[np.ndarray] | None = None,
    telluric: bool = False,
    nebular: bool = False,
    nebular_v_kms: float = 0.0,
    nebular_amplitudes=None,
    ar1_max_gap: int = 4,
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
        Per-instrument Gaussian LSF width (km/s): a scalar for a stationary LSF, or,
        with matching ``lsf_anchors_angstrom``, one width per anchor for a
        wavelength-dependent LSF (a scalar then broadcasts to every anchor). The
        largest width fixes the kernel radius, so build-time widths are the upper
        bounds a later :func:`with_lsf` swap must respect.
    lsf_anchors_angstrom
        Optional per-instrument anchor wavelengths (strictly increasing, >= 2). When
        given, the instrument's LSF varies across the grid: per-anchor Gaussian
        kernels are linearly interpolated (in log-wavelength, clamped beyond the end
        anchors) into a per-model-pixel profile bank applied by
        :func:`albireo.operators.convolve_varying`, the tabulated-LSF form of
        design.md D8. Instruments absent from the mapping stay stationary.
    lsf_h3
        Optional per-instrument Gauss-Hermite skewness (D38): a scalar or one value
        per anchor, requiring ``lsf_anchors_angstrom`` for that instrument. A
        stationary asymmetric LSF is absorbed by the free spectra, so only the
        wavelength variation of the asymmetry is identified (``docs/math.md`` §1.3).
        Absent instruments (or None) keep pure Gaussian anchors.
    response_coeffs
        Optional per-epoch Chebyshev response coefficients (empty/None = unit response).
    telluric
        If True, append a telluric component (light fraction 1) whose velocity law is
        the topocentric one: static for topocentric-frame data, ``+v_bary`` for
        barycentric-frame data.
    nebular
        If True, append a nebular component (D40), static in the barycentric frame (the
        opposite convention from the telluric one) and carrying a free per-epoch
        amplitude rather than a light fraction. Component order is stellar, telluric,
        nebular, so enabling it adds one trailing column and one more entry to the
        spectral prior.

        Nebular emission is added on top of the total stellar continuum and takes no
        light from the stars, so its amplitude lies outside the simplex the light
        fractions live on, and its night-to-night variation (seeing, slit losses, sky
        subtraction) is a scale on one fixed shape. Left in the data, a static emission
        feature sitting on a moving absorption line is absorbed by the stellar
        components as a spurious core-fill, which narrows the disentangled profile and
        biases every temperature and gravity derived from it.
    nebular_v_kms
        Velocity of the nebula, in the frame the stellar velocities are measured in
        (km/s). It is not identified by the data: the shift is the same at every epoch
        (barycentric-frame data) or differs only by ``v_bary`` (topocentric), and a
        constant shift of a free spectrum is a reparameterization ``d -> T(delta) d``,
        exactly as the systemic velocity is for the stellar components (D14). What it
        decides is where the component's lines land on the model grid, which matters as
        soon as the prior confines it to windows
        (:func:`albireo.priors.window_profile`), since the windows and the shift must
        agree. Pass the same value to :func:`albireo.priors.nebular_windows`.
    nebular_amplitudes
        Per-epoch amplitudes ``(n_epochs,)`` for the nebular component (default: all
        ones). Only the product ``amplitude * spectrum`` is observable, so the overall
        scale is degenerate with the component spectrum and is fixed by convention:
        :func:`albireo.inference.nebular_amplitudes` normalizes the geometric mean to
        1. Must be positive here; the traced swap (:func:`with_nebular_amplitudes`)
        cannot check.
    ar1_max_gap
        Longest masked gap (in native pixels) an AR(1) noise link may span
        (:func:`with_ar1`, ``docs/math.md`` §1.4a). Links between good pixels ``gap``
        apart carry correlation ``phi**gap``; beyond the cap the chain restarts, which
        matches the short-range character of resampling correlation and bounds the
        coupling's bandwidth cost (:attr:`Problem.ar_bandwidth_extra`). The tables are
        static and always built; they enter the model only when :func:`with_ar1` is
        applied.
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

    shift_rows = [star_pix]
    light_rows = [ell]
    if telluric:
        shift_rows.append(tell_pix[None, :])
        light_rows.append(np.ones((1, n_ep)))
    if nebular:
        # Static in the barycentric frame: the mirror image of the telluric law, so it
        # is the *topocentric*-frame data that carry the per-epoch -v_bary term.
        neb_pix = float(np.asarray(grid.velocity_to_pixels(float(nebular_v_kms))))
        neb_row = np.full(n_ep, neb_pix)
        if dataset.frame == "topocentric":
            neb_row = neb_row - bary_pix
        if nebular_amplitudes is None:
            amp = np.ones(n_ep)
        else:
            amp = np.atleast_1d(np.asarray(nebular_amplitudes, dtype=np.float64))
        if amp.shape != (n_ep,):
            raise ValueError(f"nebular_amplitudes must have shape ({n_ep},); got {amp.shape}")
        if np.any(amp <= 0) or not np.all(np.isfinite(amp)):
            raise ValueError("nebular_amplitudes must be positive and finite")
        shift_rows.append(neb_row[None, :])
        light_rows.append(amp[None, :])
    shifts = np.vstack(shift_rows)
    light = np.vstack(light_rows)
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
                "Build the grid from the data: LogGrid.covering(dataset, dv_kms, ...)."
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

        kernel, anchor_wave = _lsf_bank(instrument, lsf_sigma_v, lsf_anchors_angstrom, lsf_h3, grid)
        base = np.asarray(rebin(jnp.ones(grid.n)))  # R 1 (= coverage)

        z_rows, w_rows, r_rows, gap_rows = [], [], [], []
        ar_step = 0
        for j in idx:
            ep = dataset[j]
            r = chebyshev_response(ep.wave, response_coeffs[j])
            w = np.where(covered, ep.effective_ivar, 0.0)
            # Zero-weight pixels are allowed to hold anything, including nan and inf
            # (albireo.data: a masked pixel's flux is never read). Every consumer of z
            # multiplies it by w, so zeroing it here changes no value, except that
            # `0 * nan` is `nan` and one such pixel takes the whole marginal likelihood
            # to nan. Real data reaches this path routinely: albireo.preprocess.normalize
            # marks pixels where the fitted continuum collapses by writing nan.
            flux = np.where(np.isfinite(ep.flux), ep.flux, 0.0)
            z_rows.append(np.where(w > 0.0, flux - r * base, 0.0))
            w_rows.append(w)
            r_rows.append(r)
            # AR(1) link table (with_ar1): each good pixel links to the previous good
            # pixel when the index gap is within ar1_max_gap; the link carries phi**gap.
            good_px = w > 0.0
            pix = np.arange(w.size)
            prev = np.concatenate(([-1], np.maximum.accumulate(np.where(good_px, pix, -1))[:-1]))
            gap = np.where(good_px & (prev >= 0), pix - prev, 0)
            gap = np.where(gap <= ar1_max_gap, gap, 0)
            link = gap > 0
            if np.any(link):
                dlo = lo[pix[link]] - lo[pix[link] - gap[link]]
                ar_step = max(ar_step, int(np.max(np.maximum(dlo, 0))))
            gap_rows.append(gap)

        pair_val, pair_sid, pair_row, pair_h = rebin_pair_tables(rebin)
        if pair_h != row_support:
            raise AssertionError(f"pair-table support {pair_h} != row support {row_support}")

        # Cross-row pair tables for the AR(1) band assembly, over the union of links
        # realized in any epoch (masks differ per epoch; the per-epoch gap test in the
        # assembly selects each epoch's own). Realized links are exactly the ones whose
        # model-pixel offset entered ar_step, so the table width is bounded.
        gap_stack = np.stack(gap_rows)
        ep_i, px_i = np.nonzero(gap_stack > 0)
        link_key = np.unique(px_i.astype(np.int64) * (ar1_max_gap + 1) + gap_stack[ep_i, px_i])
        link_val, link_sid, link_row, link_gap = rebin_link_pair_tables(
            rebin,
            link_key // (ar1_max_gap + 1),
            link_key % (ar1_max_gap + 1),
            row_support + ar_step,
        )

        groups.append(
            EpochGroup(
                instrument=instrument,
                epoch_indices=tuple(idx),
                rebin=rebin,
                kernel=kernel,
                shifts=jnp.asarray(shifts[:, idx].T),
                light=jnp.asarray(light[:, idx].T),
                z=jnp.asarray(np.stack(z_rows)),
                w=jnp.asarray(np.stack(w_rows)),
                r=jnp.asarray(np.stack(r_rows)),
                jitter=jnp.ones(len(idx)),
                row_support=row_support,
                bary_pix=jnp.asarray(bary_pix[list(idx)]),
                base=jnp.asarray(base),
                cheb_x=jnp.asarray(
                    2.0 * (wave_native - wave_native[0]) / (wave_native[-1] - wave_native[0]) - 1.0
                ),
                ar_phi=jnp.zeros(len(idx)),
                ar_gap=jnp.asarray(np.stack(gap_rows)),
                ar_step=ar_step,
                pair_val=pair_val,
                pair_sid=pair_sid,
                pair_row=pair_row,
                link_val=link_val,
                link_sid=link_sid,
                link_row=link_row,
                link_gap=link_gap,
                lsf_anchor_wave=anchor_wave,
            )
        )

    return Problem(
        grid=grid,
        n_components=n_comp,
        groups=tuple(groups),
        frame=dataset.frame,
        telluric=telluric,
        nebular=nebular,
    )


def with_data(problem: Problem, z_per_group) -> Problem:
    """Return ``problem`` with the data term ``z`` replaced, everything else untouched.

    ``z = y - r (R 1)`` is the only place the observed fluxes enter the problem, so
    swapping it re-points the whole operator stack at a different realization of the
    same experiment. This is what makes a parametric bootstrap cheap: the rebin
    operators, pair tables, LSF bank, weights, masks and response are built once and
    reused, and each trial costs one forward apply instead of a fresh
    :func:`build_problem`. :func:`albireo.simulate.resimulate` draws the replacement,
    and :mod:`albireo.calibrate` runs the loop.

    Nothing the structure was built from may change: the native wavelength grids, the
    masks (the ``w == 0`` pattern), and the AR(1) link tables derived from them are all
    static here, so this is a swap of numbers into a fixed graph. Data with a different
    mask silently reuses the old one.

    Parameters
    ----------
    problem
        Output of :func:`build_problem`.
    z_per_group
        One ``(n_epochs, n_native)`` array per group, in ``problem.groups`` order: the
        layout :func:`apply_model` returns and :attr:`EpochGroup.z` stores.

    Returns
    -------
    Problem
    """
    groups, new = [], list(z_per_group)
    if len(new) != len(problem.groups):
        raise ValueError(f"expected {len(problem.groups)} z arrays, got {len(new)}")
    for g, z in zip(problem.groups, new, strict=True):
        z = jnp.asarray(z)
        if z.shape != g.z.shape:
            raise ValueError(
                f"group {g.instrument!r}: z must have shape {g.z.shape}; got {z.shape}"
            )
        # Masked pixels are allowed to hold anything upstream (albireo.data), and every
        # consumer multiplies z by w, but 0 * nan is nan, so zero them here as
        # build_problem does rather than trust the caller.
        groups.append(replace(g, z=jnp.where(g.w > 0.0, z, 0.0)))
    return replace(problem, groups=tuple(groups))


def with_velocities(problem: Problem, velocities) -> Problem:
    """Return ``problem`` with the stellar velocities replaced (differentiable in them).

    This is the θ-dependent path for joint inference: only the per-epoch shift columns
    are recomputed, with the same frame composition as :func:`build_problem`, while
    every static piece (rebin operators, kernels, weights, targets, response) is reused
    unchanged. Safe to call inside ``jax.jit`` with traced ``velocities``; combine with
    :meth:`Problem.half_bandwidth_bound` for a static solver bandwidth.

    Parameters
    ----------
    problem
        Output of :func:`build_problem` (any velocities).
    velocities
        Stellar radial velocities in the barycentric frame, shape
        ``(n_stellar, n_epochs)`` (km/s). The telluric and nebular columns, if
        present, are carried over unchanged: their velocity laws depend only on the
        frame and each epoch's ``v_bary``, both fixed at build time, so there is
        nothing in them for a stellar velocity to change.
    """
    vel = jnp.atleast_2d(jnp.asarray(velocities))
    if vel.shape != (problem.n_stellar, problem.n_epochs):
        raise ValueError(
            f"velocities must have shape ({problem.n_stellar}, {problem.n_epochs}); got {vel.shape}"
        )
    return with_shifts(problem, problem.grid.velocity_to_pixels(vel))


def with_shifts(problem: Problem, star_pix) -> Problem:
    """Return ``problem`` with the stellar shifts replaced, in *model pixels*.

    The pixel-space core of :func:`with_velocities`, which is a one-line wrapper over it.
    Two uses call for this layer rather than the velocity one.

    First, shift composition is exact in pixels: with the relativistic mapping
    ``xi = artanh(v/c)`` the log-wavelength shift turns relativistic velocity addition
    into ordinary addition, so adding a constant here is exactly a translation, while
    adding a constant to a velocity is not. Anything that adds, subtracts or centers
    shifts, above all the free-velocity table's zero point
    (:func:`albireo.inference.relative_velocities`), must do it here to be exact rather
    than first-order.

    Second, it is the entry point for a shift the caller computed some other way: a
    template cross-correlation lag, or a per-epoch offset read off a line centroid.

    Parameters
    ----------
    problem
        Output of :func:`build_problem` (any velocities).
    star_pix
        Stellar shifts in model pixels, shape ``(n_stellar, n_epochs)``, in the
        barycentric frame. The frame composition (the per-epoch ``-v_bary`` term on
        topocentric data) is applied here, exactly as :func:`build_problem` applies it.
        Telluric and nebular columns are carried over unchanged.
    """
    star_pix = jnp.atleast_2d(jnp.asarray(star_pix))
    if star_pix.shape != (problem.n_stellar, problem.n_epochs):
        raise ValueError(
            f"star_pix must have shape ({problem.n_stellar}, {problem.n_epochs}); "
            f"got {star_pix.shape}"
        )
    groups = []
    for g in problem.groups:
        idx = list(g.epoch_indices)
        sp = star_pix[:, idx].T  # (n_epochs_group, n_stellar)
        if problem.frame == "topocentric":
            sp = sp - g.bary_pix[:, None]
        sp = jnp.concatenate([sp, g.shifts[:, problem.n_stellar :]], axis=1)
        groups.append(replace(g, shifts=sp))
    return replace(problem, groups=tuple(groups))


def with_light_fractions(problem: Problem, light_fractions) -> Problem:
    """Return ``problem`` with the stellar light fractions replaced (differentiable).

    The θ-dependent path for light-fraction inference: only the stellar light columns
    are swapped. The telluric column (if present) keeps light fraction 1 and the nebular
    column keeps whatever amplitudes it carries, since the nebular amplitude is outside
    the simplex by construction and has its own swap
    (:func:`with_nebular_amplitudes`). Safe inside ``jax.jit`` with traced values. The
    simplex constraint (non-negative, sum to 1 per epoch) cannot be checked on traced
    input and is the caller's responsibility; in the numpyro model it is guaranteed by
    a Dirichlet prior.

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
        le = jnp.concatenate([le, g.light[:, problem.n_stellar :]], axis=1)
        groups.append(replace(g, light=le))
    return replace(problem, groups=tuple(groups))


def with_nebular_amplitudes(problem: Problem, amplitudes) -> Problem:
    """Return ``problem`` with the nebular component's per-epoch amplitudes replaced.

    The θ-dependent path for the D40 nebular component: only its light column moves,
    and the stellar simplex and the telluric column are untouched. Differentiable and
    safe inside ``jax.jit`` with traced values.

    The overall scale is a convention, not a measurement. The model sees only the
    products ``a_j * d_neb``, so ``(c a_j, d_neb / c)`` is the same fit for any
    ``c > 0``: the amplitudes carry the epoch-to-epoch variation and the spectrum
    carries the level. The prior on ``d_neb`` breaks the tie weakly rather than not at
    all, which samples worse than either extreme, so the scale is pinned explicitly.
    :func:`albireo.inference.nebular_amplitudes` normalizes the geometric mean to 1,
    and that is what the ``log_nebular_amp`` site feeds through here.

    Positivity is likewise not enforced (a traced value cannot be checked, and the sign
    is degenerate with the spectrum's); sample ``exp`` of an unconstrained parameter.

    Parameters
    ----------
    problem
        Output of :func:`build_problem` with ``nebular=True``.
    amplitudes
        Scalar (one amplitude shared by every epoch, i.e. a static nebular
        contribution) or ``(n_epochs,)``.
    """
    if not problem.nebular:
        raise ValueError(
            "this problem has no nebular component: rebuild it with "
            "build_problem(..., nebular=True). Enabling it adds a component, so it "
            "changes the linear system's size and the spectral prior's length; it "
            "cannot be a traced swap."
        )
    amp = jnp.asarray(amplitudes)
    if amp.ndim == 0:
        amp = jnp.broadcast_to(amp, (problem.n_epochs,))
    if amp.shape != (problem.n_epochs,):
        raise ValueError(
            f"amplitudes must be a scalar or have shape ({problem.n_epochs},); got {amp.shape}"
        )
    groups = []
    for g in problem.groups:
        col = amp[np.asarray(g.epoch_indices, dtype=np.int64)]
        groups.append(replace(g, light=g.light.at[:, -1].set(col)))
    return replace(problem, groups=tuple(groups))


def with_jitter(problem: Problem, jitter) -> Problem:
    """Return ``problem`` with per-epoch noise-inflation factors ``alpha_j`` (differentiable).

    ``docs/math.md`` §1.4 and ``internal/design.md`` D15: the weights become
    ``w_j -> w_j / alpha_j^2``, so ``alpha = 1`` is exactly the unmodified problem and
    ``alpha > 1`` says this epoch's quoted inverse variances are optimistic by that
    factor. The marginal likelihood keeps its ``+1/2 sum log w`` term, which is what
    makes ``alpha`` identifiable rather than a free knob, and it supplies the correct
    denominator. In the data-dominated limit ``-1/2 log det(Lambda + A^T W A)``
    contributes ``+p_eff log alpha`` against that term's ``-N log alpha``, so profiling
    gives ``alpha^2 = chi2 / (N - p_eff)`` with
    ``p_eff = tr[(Lambda + A^T W A)^-1 A^T W A]`` the effective number of parameters the
    marginalized spectra consume. Whitening the residuals by hand and reading off their
    standard deviation instead gives ``chi2 / N``, low by ``sqrt(1 - p_eff/N)``. The size
    of that difference is a property of the run, since ``p_eff`` is an effective count
    rather than ``n_comp * n_pix``: an oversampled model grid with a fitted smoothness
    prior can put it an order of magnitude below the pixel count (measured on HR 6819,
    ~2900 against 19,876 pixels, for a 0.4% correction, while the weak-prior fixture in
    ``tests/test_jitter.py`` sees 4.6%). Being joint with the orbit, the widened
    uncertainties propagate.

    The jitter is the appropriate handle for archival spectra whose inverse variances
    were estimated rather than measured (:func:`albireo.preprocess.estimate_ivar`),
    where a scale error is expected and unknowable a priori. It is not a repair for
    unmodelled structure: a jitter fitted against systematics (imperfect continuum, LSF
    mismatch, line-profile variability) reports a wider but still biased orbit, because
    inflating a diagonal noise model cannot represent a residual correlated across
    pixels. Check :func:`data_residual_zscores` for structure before trusting the
    widening; on real data the reliable error bar is usually still the scatter between
    independent wavelength windows.

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


def _ar1_links(g: EpochGroup):
    """Per-link AR coefficients ``(rho, a, c, prev)`` for the masked chain (traced).

    Link ``i`` connects pixel ``i`` to ``prev_i = i - gap_i`` with correlation
    ``rho_i = phi**gap_i`` (0 where there is no link): a subset of an AR(1) chain is
    still Markov, with the correlation across a masked gap equal to the process
    correlation at that index distance, so masking is exact rather than approximate
    (``docs/math.md`` §1.4a). ``a = rho^2/(1-rho^2)`` and ``c = rho/(1-rho^2)`` are
    the precision increments: on top of the identity, each link adds ``a`` to both
    endpoints' diagonal and ``-c`` to their off-diagonal pair. The gap-1 branch uses
    ``phi`` directly so the gradient at ``phi = 0`` is exact (``jnp.power`` has a nan
    gradient at a zero base); multi-gap links go through a zero-guarded base, whose
    true gradient at 0 is 0 and stays 0.
    """
    gap = g.ar_gap
    phi = g.ar_phi[:, None]
    phi_nz = jnp.where(jnp.abs(phi) > 1e-150, phi, 1e-150)
    rho = jnp.where(gap == 1, phi, jnp.where(gap >= 2, phi_nz ** gap.astype(jnp.float64), 0.0))
    one_m = 1.0 - rho * rho
    a = rho * rho / one_m
    c = rho / one_m
    prev = jnp.arange(gap.shape[1])[None, :] - gap
    return rho, a, c, prev


def ar1_band_weights(g: EpochGroup):
    """Diagonal and link weights of ``diag(r) W diag(r)`` for direct band assembly.

    The chain precision (``docs/math.md`` §1.4a) splits into a diagonal part, the
    pixel's own weight scaled by the chain diagonal ``1 + sum of the a's of the links
    touching it``, and one symmetric off-diagonal term per link,
    ``-c sqrt(w_n w_p) r_n r_p / alpha^2``. :func:`apply_noise_precision` is the
    matrix-free form of the same split. Returns ``(wp, wl)``, each
    ``(n_epochs, n_native)``: ``wp`` feeds the diagonal pair tables exactly where
    ``effective_w * r**2`` does on the diagonal path, and ``wl[.., n]`` weights the link
    whose later endpoint is ``n`` (zero where the epoch has none), consumed against the
    static cross-row tables (:func:`albireo.operators.rebin_link_pair_tables`).
    """
    _, a, c, prev = _ar1_links(g)

    def one(a_e, c_e, prev_e, w_e, r_e):
        dchain = 1.0 + a_e + jnp.zeros_like(a_e).at[prev_e].add(a_e)
        s = jnp.sqrt(w_e)
        return w_e * r_e**2 * dchain, -c_e * s * s[prev_e] * r_e * r_e[prev_e]

    wp, wl = jax.vmap(one)(a, c, prev, g.w, g.r)
    alpha2 = g.jitter[:, None] ** 2
    return wp / alpha2, wl / alpha2


def apply_noise_precision(g: EpochGroup, x):
    """``W x`` for one group's noise model, ``W = D^{1/2} R^{-1} D^{1/2} / alpha^2``.

    ``D = diag(w)``, ``R`` the AR(1) correlation of the standardized residuals over the
    masked chain (:func:`_ar1_links`, ``docs/math.md`` §1.4a), ``alpha`` the jitter.
    Rows and columns at masked pixels are exactly zero (``sqrt(w) = 0`` on both sides).
    With ``rho = 0`` this is ``effective_w * x`` up to floating-point ordering; callers
    on the diagonal path keep the direct product, gated by
    :attr:`Problem.correlated`.
    """
    s = jnp.sqrt(g.w)
    u = s * x
    _, a, c, prev = _ar1_links(g)

    def one(u_e, a_e, c_e, prev_e):
        diag = 1.0 + a_e + jnp.zeros_like(a_e).at[prev_e].add(a_e)
        return diag * u_e - c_e * u_e[prev_e] - jnp.zeros_like(u_e).at[prev_e].add(c_e * u_e)

    return s * jax.vmap(one)(u, a, c, prev) / g.jitter[:, None] ** 2


def _noise_quad_logdet(g: EpochGroup, x):
    """``(x^T W x, log det W|_good)`` for one group (docs/math.md §1.4a).

    ``log det W`` over the good-pixel subspace is closed-form:
    ``sum_good log w  -  2 n_good log alpha  -  sum_links log(1 - rho^2)``.
    """
    s = jnp.sqrt(g.w)
    u = s * x
    rho, a, c, prev = _ar1_links(g)

    def one(u_e, a_e, c_e, prev_e):
        diag = 1.0 + a_e + jnp.zeros_like(a_e).at[prev_e].add(a_e)
        return jnp.sum(diag * u_e * u_e) - 2.0 * jnp.sum(c_e * u_e * u_e[prev_e])

    quad = jnp.sum(jax.vmap(one)(u, a, c, prev) / g.jitter**2)
    good = g.w > 0
    logdet = (
        jnp.sum(jnp.where(good, jnp.log(jnp.where(good, g.w, 1.0)), 0.0))
        - 2.0 * jnp.sum(jnp.sum(good, axis=1) * jnp.log(g.jitter))
        - jnp.sum(jnp.log1p(-rho * rho))
    )
    return quad, logdet


def _whiten_residuals(g: EpochGroup, resid):
    """Exact chain whitener of the group noise model applied to native residuals.

    ``eps = sqrt(w) resid / alpha`` standardizes; the Markov factorization
    ``(eps_i - rho_i eps_prev) / sqrt(1 - rho_i^2)`` then decorrelates the links, so
    the output is iid N(0, 1) exactly when the noise model holds. With ``rho = 0``
    it reduces to the diagonal whitening.
    """
    eps = jnp.sqrt(g.w) * resid / g.jitter[:, None]
    rho, _, _, prev = _ar1_links(g)

    def one(e_e, rho_e, prev_e):
        return (e_e - rho_e * e_e[prev_e]) / jnp.sqrt(1.0 - rho_e * rho_e)

    return jax.vmap(one)(eps, rho, prev)


def with_ar1(problem: Problem, phi) -> Problem:
    """Return ``problem`` with AR(1)-correlated noise (differentiable in ``phi``).

    The noise model D15 could not express and D31 measured the need for. Per epoch the
    noise covariance is ``C = alpha^2 D^{-1/2} R_phi D^{-1/2}`` with ``D = diag(w)``
    and ``R_phi`` the AR(1) correlation in native-pixel index: adjacent pixels of the
    standardized residual share correlation ``phi``, the expected shape when a pipeline
    resamples spectra onto a common step (each output pixel mixes the same input pixels
    as its neighbours). ``phi = 0`` is exactly the D31 model. The jitter ``alpha``
    scales and ``phi`` correlates, and the two compose: ``with_jitter`` and this swap
    are independent, and each replaces its own parameter.

    Everything stays closed-form (``docs/math.md`` §1.4a). The precision
    ``W = D^{1/2} R^{-1} D^{1/2} / alpha^2`` is tridiagonal over the observed chain
    with per-link entries. Masked pixels are handled exactly, because a subset of a
    Markov chain is Markov and a link across a gap of ``d`` pixels carries ``phi**d``,
    up to ``build_problem``'s ``ar1_max_gap``, beyond which the chain restarts
    (short-range noise does not span a chip gap, and the cap bounds the solver-bandwidth
    cost). ``log det W`` needs one extra term, ``-sum_links log(1 - rho^2)``, which is
    what makes ``phi`` identifiable in the marginal rather than a free knob: the same
    logdet discipline as the jitter.

    Two structural consequences, both static:

    * the D28 band assembly carries the chain's off-diagonal terms through static
      cross-row link pair tables built at :func:`build_problem` time (D35,
      :func:`albireo.operators.rebin_link_pair_tables`), so the correlated marginal
      stays on the fast path; global comb probing remains the reference
      implementation (``assembly="probe"``) and the ``validate`` oracle;
    * ``A^T W A`` widens by :attr:`Problem.ar_bandwidth_extra` model pixels.
      :attr:`Problem.natural_half_bandwidth` includes it once this swap is applied,
      and a static ``half_bandwidth`` chosen before the swap must add it
      (:class:`albireo.inference.MarginalOrbitModel` does, behind its ``ar1`` flag).

    Applying this swap marks the problem correlated even at ``phi = 0``, since a traced
    value cannot make structural decisions; the result then equals the diagonal model to
    floating-point ordering, with the widened bandwidth.

    Parameters
    ----------
    problem
        Output of :func:`build_problem`.
    phi
        Scalar (shared) or ``(n_epochs,)`` AR(1) correlation, ``|phi| < 1``; clipped
        to ``+-0.999`` so the likelihood stays finite (and rejectable) under
        optimizer excursions. Traced values are fine.
    """
    phi = jnp.asarray(phi)
    if phi.ndim == 0:
        phi = jnp.broadcast_to(phi, (problem.n_epochs,))
    if phi.shape != (problem.n_epochs,):
        raise ValueError(
            f"phi must be a scalar or have shape ({problem.n_epochs},); got {phi.shape}"
        )
    phi = jnp.clip(phi, -0.999, 0.999)
    groups = [
        replace(g, ar_phi=phi[np.asarray(g.epoch_indices, dtype=np.int64)]) for g in problem.groups
    ]
    return replace(problem, groups=tuple(groups), correlated=True)


def _chebval_traced(x, c):
    """Clenshaw evaluation of ``sum_m c_m T_m(x)`` with traced coefficients.

    Written with numpy's ``polynomial.chebyshev.chebval`` operation order so that the
    θ-path (:func:`with_response`) reproduces a fresh :func:`build_problem` to float
    precision, not merely to an interpolation tolerance. ``c`` must have static length
    >= 1 (the coefficient count is graph structure, like a kernel radius).
    """
    n = c.shape[0]
    if n == 1:
        c0, c1 = c[0], jnp.asarray(0.0)
    elif n == 2:
        c0, c1 = c[0], c[1]
    else:
        x2 = 2.0 * x
        c0, c1 = c[n - 2], c[n - 1]
        for i in range(3, n + 1):
            c0, c1 = c[n - i] - c1, c0 + c1 * x2
    return c0 + c1 * x


def with_response(problem: Problem, response_coeffs) -> Problem:
    """Return ``problem`` with the multiplicative per-epoch response replaced (differentiable).

    The θ-dependent path for response and continuum inference, the swap D7 deferred:
    the response enters the targets ``z_j = y_j - r_j (R 1)`` and the normal-matrix
    weights ``w r^2``, not only the forward operator. The stored ``base = R 1`` is
    response-independent, which gives the target update in place,
    ``z_new = z_old + (r_old - r_new) * base``, exactly and without carrying the raw
    fluxes. Masked pixels stay exactly zero (every ``z`` entry at ``w = 0`` was zeroed
    at build time and the update is re-masked), so the D30 ``0 * nan`` failure cannot
    resurface here. This replaces rather than compounds, as :func:`with_jitter` does;
    the ``sum log w`` term is untouched because the noise lives on the data, not on the
    response-divided data.

    The convention matches :func:`albireo.simulate.chebyshev_response` and
    :func:`build_problem`: ``r = 1 + sum_m c_m T_m(x)`` with ``x`` the epoch's native
    wavelength grid scaled to [-1, 1] (per group, so mixed instruments each use their
    own abscissa); an all-zero coefficient vector is exactly the unit response.

    Identifiability follows the ``internal/design.md`` §5 response row: a low-order response
    trades against the components' broad spectral features, so the epoch-shared part of
    a free response is only weakly identified, while the epoch-to-epoch differences,
    which are what a per-epoch continuum treatment is for, are well constrained. Keep
    the order low and the priors tight and zero-centered; the anchor policy of D13 and
    D25 applies.

    Parameters
    ----------
    problem
        Output of :func:`build_problem`.
    response_coeffs
        ``(n_coef,)`` shared across epochs or ``(n_epochs, n_coef)`` per-epoch
        Chebyshev coefficients with ``n_coef >= 1`` static. Traced values are fine,
        so this is safe inside ``jax.jit``.
    """
    c = jnp.asarray(response_coeffs)
    if c.ndim == 1:
        c = jnp.broadcast_to(c[None, :], (problem.n_epochs, c.shape[0]))
    if c.ndim != 2 or c.shape[0] != problem.n_epochs:
        raise ValueError(
            f"response_coeffs must have shape (n_coef,) or ({problem.n_epochs}, n_coef); "
            f"got {jnp.asarray(response_coeffs).shape}"
        )
    if c.shape[1] < 1:
        raise ValueError("response_coeffs needs at least one coefficient per epoch")
    groups = []
    for g in problem.groups:
        ce = c[np.asarray(g.epoch_indices, dtype=np.int64)]
        r_new = 1.0 + jax.vmap(partial(_chebval_traced, g.cheb_x))(ce)
        z_new = jnp.where(g.w > 0.0, g.z + (g.r - r_new) * g.base[None, :], 0.0)
        groups.append(replace(g, r=r_new, z=z_new))
    return replace(problem, groups=tuple(groups))


def with_lsf(problem: Problem, lsf_sigma_v: Mapping, lsf_h3: Mapping | None = None) -> Problem:
    """Return ``problem`` with the Gaussian LSF widths replaced (differentiable).

    The θ-dependent path for LSF inference: each group's kernel values are recomputed
    from the traced width while the kernel radius stays the one fixed at
    :func:`build_problem` time. The ``lsf_sigma_v`` passed at build time must therefore
    be an upper bound on any width used here (a larger width would be truncated by the
    fixed radius; the inference model rejects that region). Safe inside ``jax.jit``;
    widths must be positive (enforce via the prior's support).

    Parameters
    ----------
    problem
        Output of :func:`build_problem`.
    lsf_sigma_v
        Per-instrument Gaussian LSF width in km/s (traced values allowed); must
        cover every instrument in the problem. For a group built with LSF anchors
        (``lsf_anchors_angstrom``), one width per anchor, in which case the traced bank
        is re-interpolated into the per-pixel profiles through the same static tables
        the build used, or a scalar, which broadcasts to every anchor. A group built
        without anchors takes a scalar only.
    lsf_h3
        Optional per-instrument Gauss-Hermite skewness (traced values allowed; D38):
        one value per anchor or a scalar, for anchored instruments only. Instruments
        absent from the mapping keep pure Gaussian anchors. Keep ``|h3|`` within the
        inference bound (0.2); the kernel radius is unchanged.
    """
    lsf_h3 = lsf_h3 or {}
    groups = []
    for g in problem.groups:
        if g.instrument not in lsf_sigma_v:
            raise ValueError(f"no LSF width supplied for instrument {g.instrument!r}")
        sigma_px = jnp.atleast_1d(jnp.asarray(lsf_sigma_v[g.instrument])) / problem.grid.dv_kms
        radius = (g.kernel.shape[-1] - 1) // 2
        if not g.lsf_anchor_wave:
            if sigma_px.shape[0] != 1:
                raise ValueError(
                    f"instrument {g.instrument!r} was built without LSF anchors: "
                    "with_lsf takes a scalar width for it."
                )
            if g.instrument in lsf_h3:
                raise ValueError(
                    f"instrument {g.instrument!r}: lsf_h3 needs LSF anchors: a "
                    "stationary asymmetric LSF is absorbed by the free spectra "
                    "(docs/math.md §1.3)."
                )
            kernel = gaussian_kernel_traced(sigma_px[0], radius)
            groups.append(replace(g, kernel=kernel[None, :]))
            continue
        n_anchor = len(g.lsf_anchor_wave)
        if sigma_px.shape[0] == 1:
            sigma_px = jnp.broadcast_to(sigma_px, (n_anchor,))
        if sigma_px.shape[0] != n_anchor:
            raise ValueError(
                f"instrument {g.instrument!r}: {sigma_px.shape[0]} LSF widths for "
                f"{n_anchor} anchors: supply one per anchor (or one scalar)."
            )
        if g.instrument in lsf_h3:
            h3 = jnp.atleast_1d(jnp.asarray(lsf_h3[g.instrument]))
            if h3.shape[0] == 1:
                h3 = jnp.broadcast_to(h3, (n_anchor,))
            if h3.shape[0] != n_anchor:
                raise ValueError(
                    f"instrument {g.instrument!r}: {h3.shape[0]} h3 values for "
                    f"{n_anchor} anchors: supply one per anchor (or one scalar)."
                )
            bank = jax.vmap(partial(gauss_hermite_kernel_traced, radius=radius))(sigma_px, h3)
        else:
            bank = jax.vmap(partial(gaussian_kernel_traced, radius=radius))(sigma_px)
        idx, t = lsf_anchor_tables(g.lsf_anchor_wave, problem.grid.wave)
        profiles = (1.0 - t)[:, None] * bank[idx] + t[:, None] * bank[idx + 1]
        groups.append(replace(g, kernel=profiles))
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
        # Static branch on the bank shape: one row is the stationary kernel (v1 path,
        # bit-identical), a full bank is the wavelength-dependent LSF (D37).
        if group.kernel.shape[0] == 1:
            conv = jnp.convolve(acc, group.kernel[0], mode="same")
        else:
            conv = convolve_varying(acc, group.kernel)
        return group.rebin(conv)

    return jax.vmap(one_epoch)(group.shifts, group.light)


def _epoch_model_adjoint(group: EpochGroup, v):
    """Adjoint of :func:`_epoch_model`: native-space ``(n_epochs, n_native)`` -> stack."""
    n_comp = group.shifts.shape[1]

    def one_epoch(v_e, shifts_e, light_e):
        t = group.rebin.adjoint(v_e)
        if group.kernel.shape[0] == 1:
            t = jnp.convolve(t, group.kernel[0, ::-1], mode="same")
        else:
            t = convolve_varying_adjoint(t, group.kernel)
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
        if problem.correlated:
            per_group.append(g.r * apply_noise_precision(g, g.r * m))
        else:
            per_group.append(g.effective_w * g.r**2 * m)
    return apply_model_adjoint(problem, per_group)


def rhs(problem: Problem):
    """Right-hand side ``b = A^T W z``, shape ``(n_comp, n)``."""
    if problem.correlated:
        per_group = [g.r * apply_noise_precision(g, g.z) for g in problem.groups]
    else:
        per_group = [g.effective_w * g.r * g.z for g in problem.groups]
    return apply_model_adjoint(problem, per_group)


def weighted_data_terms(problem: Problem):
    """Data-only scalars of the marginal likelihood: ``(z^T W z, log det W_good, n_good)``.

    For a diagonal noise model the determinant term is ``sum log(w/alpha^2)`` over
    good pixels; the correlated model adds its closed-form link correction
    (:func:`_noise_quad_logdet`).
    """
    zwz = jnp.asarray(0.0)
    logw = jnp.asarray(0.0)
    n_good = jnp.asarray(0)
    for g in problem.groups:
        good = g.w > 0
        if problem.correlated:
            quad, ld = _noise_quad_logdet(g, g.z)
            zwz = zwz + quad
            logw = logw + ld
        else:
            zwz = zwz + jnp.sum(g.effective_w * g.z**2)
            # sum log(w / alpha^2) = sum log w - 2 n_j log alpha_j, split so the jitter
            # gradient runs through one scalar per epoch, not an (n_ep, n_native) log.
            logw = logw + jnp.sum(jnp.where(good, jnp.log(jnp.where(good, g.w, 1.0)), 0.0))
            logw = logw - 2.0 * jnp.sum(jnp.sum(good, axis=1) * jnp.log(g.jitter))
        n_good = n_good + jnp.sum(good)
    return zwz, logw, n_good


def data_residual_zscores(problem: Problem, d_stack, *, per_epoch: bool = False):
    """Whitened data residuals over unmasked pixels, under the assumed noise model.

    The whitening uses the noise model as assumed, jitter and AR(1) correlation
    included, so a standard deviation of 1 means the noise model matches the residuals.
    With no fitted noise parameters that is a statement about the supplied inverse
    variances; with fitted ones it is close to 1 by construction, and the informative
    diagnostic is then the shape of the distribution. For the correlated model the
    whitener is the exact Markov factorization (:func:`_whiten_residuals`,
    ``docs/math.md`` §1.4a), so surviving structure means the correlation is not
    AR(1)-shaped (longer-range, line-locked, or epoch-outlier structure).

    Parameters
    ----------
    problem
        The :class:`Problem` the residuals are taken against.
    d_stack
        Component deviation spectra, shape ``(n_comp, n_pix)``.
    per_epoch
        If True, return a list of 1-D arrays, one per epoch and ordered as in the
        :class:`~albireo.data.Dataset`, instead of one concatenated array. Per-epoch
        structure is what makes an outlying exposure or a drifting night visible, and it
        is what the lag-1 autocorrelation test needs: the lag-1 statistic is meaningful
        only within a single spectrum, since consecutive pixels of different epochs are
        unrelated.

    Returns
    -------
    numpy.ndarray or list[numpy.ndarray]
    """
    per_index: dict[int, np.ndarray] = {}
    out = []
    for g, m in zip(problem.groups, apply_model(problem, d_stack), strict=True):
        resid = g.z - g.r * m
        if problem.correlated:
            rows = np.asarray(_whiten_residuals(g, resid))
        else:
            rows = np.asarray(resid * jnp.sqrt(g.effective_w))
        good = np.asarray(g.w) > 0
        if per_epoch:
            for local, index in enumerate(g.epoch_indices):
                per_index[index] = rows[local][good[local]]
        else:
            out.append(rows[good])
    if per_epoch:
        return [per_index[i] for i in sorted(per_index)]
    return np.concatenate(out)

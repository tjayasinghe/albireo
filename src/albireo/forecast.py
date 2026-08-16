"""Observing-strategy forecasts: what a set of epochs buys, before they are taken.

"Will twelve more nights at these phases separate the two stars?" is a question about
the posterior covariance of the component spectra, and that covariance

    Sigma = (Lambda_p + A^T W A)^{-1}

**contains no fluxes**. Only the epoch times (through the velocities, hence the shifts),
the per-pixel weights, the masks, the line-spread functions, the light fractions, the
response and the prior enter it. Every one of those is knowable for an observation that
has not happened yet, so the whole thing can be computed in advance. That is not a
convenience — it is the one question about a disentangling run that has an exact answer
before the data exist, and no other disentangling code answers it at all.

The claim is structural rather than asserted: this module assembles the precision with
:func:`albireo.assembly.band_block_tridiagonal`, which reads weights, response, shifts,
light fractions and kernels and never reads :attr:`albireo.forward.EpochGroup.z`. The
right-hand side ``b = A^T W z`` — the only place fluxes enter the marginal likelihood —
is never formed. ``tests/test_forecast.py`` pins that by overwriting every flux with
garbage and checking the forecast comes back bit-identical.

**What it forecasts, and what it cannot.** The covariance of the *spectra* is
design-only; the covariance of the *orbit* is not. The Fisher information for a velocity
runs through ``d(model)/dv ~ ell_i d_i'``, the derivative of the component spectrum — so
forecasting an error bar on ``K_2`` requires the line depths, which is exactly what has
not been measured yet. albireo therefore forecasts the spectra and declines to forecast
the orbit, rather than forecasting the orbit against an assumed template and presenting
the assumption as a result.

**Everything it does report is conditional on the same assumptions the analysis makes**:
the orbit that sets the shifts, the light fractions (D13 — the observable is
``ell_i d_i``), the LSF widths, the response, the noise model, and the prior
hyperparameters ``(tau, eta)``. For a star already observed those come from a fit and
the forecast is nearly assumption-free; for a proposal on a star with no spectra they
are guesses, and the honest use is a *comparison* between candidate designs under one
set of guesses rather than an absolute number. The weights deserve the sharpest warning
of the set: an exposure-time calculator's inverse variances are the most optimistic
number in a proposal, and the forecast scales with them.

**The prior is quoted alongside every answer, for the D42 reason.** A posterior standard
deviation that has collapsed back onto the prior looks exactly as convincing as one the
data earned. So :class:`SensitivityForecast` carries the prior-only standard deviation
and the prior-only worst mode next to the posterior ones: if the ratio is 1, the design
is learning nothing there, and no number of extra epochs at those phases will change it.

Three quantities do the work.

*Pointwise band* (:attr:`SensitivityForecast.component_std`) — the thing a user plots,
one standard deviation per component per model pixel.

*The worst-determined modes* — the top eigenpairs of ``Sigma``, by subspace iteration on
the banded factor. This is the exact form of the degeneracy that ``docs/math.md`` §5.1
derives asymptotically: with poor phase coverage the leading eigenvector is a
low-frequency see-saw that adds flux to one component and removes it from the other, and
its standard deviation is what "breaking the degeneracy" means quantitatively. The
eigenvector is worth looking at, not only its eigenvalue — it says *which* spectral
pattern the design cannot pin down.

*Constrained degrees of freedom* ``p_eff = tr[(Lambda_p + A^T W A)^{-1} A^T W A]`` — how
many of the ``n_comp * n_pix`` spectral parameters the data actually determine, as
opposed to the prior. It is the same ``p_eff`` that
:func:`albireo.forward.with_jitter` profiles against, and it is obtained here exactly,
as a directional derivative of ``log det`` in the noise scale, rather than by a
stochastic trace estimator.

Alongside them sits the closed-form diagnostic of ``docs/math.md`` §5.1 for the
two-component case: the spread of the *differential* shift over epochs,
``Var_j(Delta_j)``, and the resulting per-scale noise amplification of the difference
mode. It costs no linear algebra at all, which is what makes screening a hundred
candidate phase sets cheap before confirming the shortlist exactly. Being the idealized
model (two components, unit weights, one grid, no LSF), it is labelled as such and
reported next to the exact numbers rather than instead of them — and there is a specific
reason to distrust it on its own. ``Var_j(Delta_j)`` is *maximized* by observing only at
the two quadratures, and that design is a bad one: two values of ``Delta`` leave the
difference mode degenerate at a whole comb of feature scales. The variance is the
small-``k`` expansion of the thing that matters, not the thing itself. See
:func:`_separation_diagnostics` for the measurement, and prefer ``blind_fraction`` and
the exact numbers when ranking designs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from albireo.assembly import (
    _prior_diagonals,
    band_block_tridiagonal,
    prior_block_tridiagonal,
    prior_logdet,
)
from albireo.data import Dataset, EpochData
from albireo.forward import build_problem, with_ar1, with_jitter
from albireo.grids import LogGrid
from albireo.priors import SmoothnessPrior
from albireo.simulate import InstrumentSpec, OrbitParams
from albireo.solver import BlockCholesky, block_cholesky, logdet, selected_inverse_diag, solve

__all__ = ["SensitivityForecast", "plan_epochs", "sensitivity_forecast"]


# ---------------------------------------------------------------------------
# Planned epochs
# ---------------------------------------------------------------------------


def plan_epochs(
    template: EpochData | InstrumentSpec,
    bjd,
    *,
    snr: float | None = None,
    v_bary=0.0,
    instrument: str | None = None,
    medium: str | None = None,
) -> tuple[EpochData, ...]:
    """Epochs for observations that have not been taken, for :func:`sensitivity_forecast`.

    A planned epoch is a real :class:`~albireo.data.EpochData` in every respect the
    forecast reads — native wavelength grid, inverse variances, time, barycentric
    velocity, instrument — and carries a **placeholder flux of exactly 1.0**, the
    continuum. The forecast never reads it (see the module docstring), and the choice of
    value is deliberate: a planned dataset fed to a *fit* by mistake returns featureless
    spectra, which is visibly wrong, rather than something that looks like a result.

    Parameters
    ----------
    template
        Either an existing :class:`~albireo.data.EpochData` — "more nights of what I
        already have", which reuses its wavelength grid, inverse variances, mask,
        instrument key and wavelength medium — or an
        :class:`albireo.simulate.InstrumentSpec`, for an instrument with no data yet
        (its ``wave`` and ``snr`` become the grid and a uniform ``ivar = snr**2``).
    bjd
        Planned mid-exposure times, ``(n_new,)`` BJD_TDB. These are what the forecast is
        *about*: they set the orbital phases, hence the shifts, hence the whole
        covariance.
    snr
        Optional per-pixel continuum signal-to-noise for the planned epochs, replacing
        the template's inverse variances with a uniform ``snr**2`` (masked pixels of an
        :class:`~albireo.data.EpochData` template stay masked). Leave it out to plan
        epochs exactly as good as the template one.
    v_bary
        Barycentric correction, a scalar or ``(n_new,)`` km/s. It matters whenever a
        telluric or nebular component is enabled — those sit in the opposite frame from
        the stars, so the barycentric motion is part of the separation the design buys.
        Default 0, which is a *choice*, not a measurement: real epochs months apart
        differ by up to ~60 km/s, and pretending otherwise understates a telluric
        model's separation. Compute the real values (astropy's ``radial_velocity_correction``)
        when the target and dates are known.
    instrument
        Instrument key; default is the template's (``"default"`` for an
        :class:`~albireo.simulate.InstrumentSpec`, whose specs are not named).
    medium
        Wavelength scale, as in :class:`~albireo.data.EpochData`. Default is the
        template's, which is what keeps a planned epoch combinable with the real ones
        in one :class:`~albireo.data.Dataset` (D43 refuses a mixture).

    Returns
    -------
    tuple of EpochData
        One per entry of ``bjd``, in order.

    Examples
    --------
    >>> plan = plan_epochs(dataset[0], bjd=t0 + np.arange(12) * 3.1)  # doctest: +SKIP
    >>> design = Dataset([*dataset, *plan], frame=dataset.frame)  # doctest: +SKIP
    """
    if isinstance(template, InstrumentSpec):
        wave = np.asarray(template.wave, dtype=np.float64)
        if snr is None:
            snr = float(template.snr)
        ivar = np.full(wave.size, float(snr) ** 2)
        mask = None
        base_instrument, base_medium = "default", None
    elif isinstance(template, EpochData):
        wave = template.wave
        ivar = template.ivar if snr is None else np.where(template.ivar > 0.0, float(snr) ** 2, 0.0)
        mask = template.mask
        base_instrument, base_medium = template.instrument, template.medium
    else:
        raise ValueError(
            "template must be an EpochData (plan more of what you already have) or an "
            f"InstrumentSpec (plan for an instrument with no data yet); got "
            f"{type(template).__name__}"
        )
    if snr is not None and float(snr) <= 0.0:
        raise ValueError(f"snr must be positive; got {snr}")

    times = np.atleast_1d(np.asarray(bjd, dtype=np.float64))
    if times.ndim != 1 or times.size == 0:
        raise ValueError(f"bjd must be a non-empty 1-D array of times; got shape {times.shape}")
    vb = np.broadcast_to(np.asarray(v_bary, dtype=np.float64), times.shape)

    flux = np.ones(wave.size)  # the continuum: never read, and harmless if it ever is
    return tuple(
        EpochData(
            wave=wave,
            flux=flux,
            ivar=ivar,
            bjd=float(t),
            v_bary=float(v),
            instrument=base_instrument if instrument is None else str(instrument),
            mask=mask,
            medium=base_medium if medium is None else medium,
        )
        for t, v in zip(times, vb, strict=True)
    )


# ---------------------------------------------------------------------------
# Linear-algebra pieces
# ---------------------------------------------------------------------------


def _top_eigenpairs(
    chol: BlockCholesky, n_modes: int, keep, *, iterations: int, seed: int, oversample: int = 4
):
    """Largest eigenpairs of ``Sigma = (L L^T)^{-1}`` restricted to ``keep``.

    The worst-determined directions of the fit are the *largest* eigenvalues of the
    covariance, i.e. the smallest of the precision — the end of the spectrum an
    ordinary factorization does not hand you. Subspace iteration on ``Sigma`` reaches
    them with one banded solve per vector per step and nothing dense ever formed.

    ``keep`` is a boolean mask over the padded coordinates; iterating on ``P Sigma P``
    with ``P = diag(keep)`` converges to the eigenpairs of the *submatrix*
    ``Sigma[keep, keep]`` — the covariance of the retained coordinates with the rest
    marginalized out, which is the right object, not a conditional. Two things are
    excluded through it.

    The **pad** coordinates: the pad block of an assembled
    :class:`~albireo.solver.BlockTridiagonal` is the identity and is decoupled from the
    real block, so they are eigenvectors of ``Sigma`` with eigenvalue 1 — a value that
    would sit in the middle of a real spectrum and could outrank genuine modes.

    The **grid margin**: a model grid is always built wider than the data (shifts plus
    kernel radius, :meth:`albireo.grids.LogGrid.covering`), and the pixels in that
    margin are constrained by the prior alone. They are therefore the largest
    eigenvalue of ``Sigma`` on essentially every real problem, and reporting them as
    "the worst-determined mode" would say nothing about the observing design — it would
    report how much margin the grid was given. Restricting to the pixels some epoch
    actually weights is what makes the answer a property of the epochs.

    Returns ``(values, vectors, residual)`` with ``vectors`` of shape ``(n_modes, n)``
    (rows, unpadded) and ``residual`` the largest relative Rayleigh residual
    ``||Sigma y - theta y|| / theta`` over the returned modes. Report it: eigenvalues
    converge much faster than eigenvectors, and a cluster of near-degenerate
    low-frequency modes — the normal situation here — leaves the individual vectors far
    less determined than the subspace they span.
    """
    n = chol.n
    n_pad = chol.num_blocks * chol.block_size
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != (n,):
        raise ValueError(f"keep must have shape ({n},); got {keep.shape}")
    keep = jnp.asarray(np.pad(keep, (0, n_pad - n)))  # the pad block is never retained
    n_keep = int(np.count_nonzero(keep))
    n_modes = int(min(n_modes, n_keep))
    m = int(min(n_keep, n_modes + oversample))

    def sigma(v):  # (m, n_pad) rows -> P Sigma P applied to each row
        return jnp.where(keep, jax.vmap(lambda x: solve(chol, x))(v), 0.0)

    def orth(v):
        q, _ = jnp.linalg.qr(v.T)
        return q.T

    v0 = orth(jnp.where(keep, jax.random.normal(jax.random.PRNGKey(int(seed)), (m, n_pad)), 0.0))
    v = jax.lax.fori_loop(0, int(iterations), lambda _, vv: orth(sigma(vv)), v0)

    # Rayleigh-Ritz on the converged subspace: T = V Sigma V^T with V's rows orthonormal.
    sv = sigma(v)
    t = v @ sv.T
    vals, u = jnp.linalg.eigh(0.5 * (t + t.T))
    order = jnp.argsort(vals)[::-1][:n_modes]
    vals = vals[order]
    vecs = u[:, order].T @ v
    resid = sigma(vecs) - vals[:, None] * vecs
    rel = jnp.max(jnp.linalg.norm(resid, axis=1) / jnp.maximum(jnp.abs(vals), 1e-300))
    # The sign of an eigenvector is arbitrary; pin it so repeated runs and plots agree.
    lead = jnp.take_along_axis(vecs, jnp.argmax(jnp.abs(vecs), axis=1)[:, None], axis=1)
    vecs = vecs * jnp.sign(jnp.where(lead == 0.0, 1.0, lead))
    # Re-mask: Householder QR leaves ~1e-33 of dust on the excluded rows rather than the
    # exact zeros the inputs carried. Nothing numerical turns on it, but "the modes live
    # in the region" should be a fact about the output, not a statement about its
    # magnitude — plots and participation ratios read these directly.
    return np.asarray(vals), np.asarray(jnp.where(keep, vecs, 0.0)[:, :n]), float(rel)


def _effective_parameters(problem, prior, alpha, b_nat, block_size) -> float:
    """``p_eff = tr[(Lambda_p + A^T W A)^{-1} A^T W A]``, exactly, from one derivative.

    Scaling every epoch's noise by ``alpha -> alpha e^{t}`` sends ``A^T W A -> e^{-2t}
    A^T W A`` and leaves ``Lambda_p`` alone, so

        ``d/dt log det(Lambda_p + e^{-2t} A^T W A)|_{t=0} = -2 tr[Sigma A^T W A]``.

    One derivative of the log-determinant therefore gives the effective parameter count
    with no stochastic trace estimator and no selected inverse — the jitter swap
    (:func:`albireo.forward.with_jitter`) is already exactly this one-parameter family,
    and the assembly is already differentiable through it.

    ``t`` is a scalar and so is the log-determinant, so forward and reverse mode return
    the same single number; this takes it in **reverse**. The band assembly carries a
    ``custom_vjp`` for its accumulate (D49 — reverse mode otherwise rebuilds the whole
    band tensor to reproduce its own input), and ``custom_vjp`` rejects ``jax.jvp``
    outright. That is the same trade D28 made one stage later at ``_solve_stage``, and
    it is paid in the right place: this runs once per forecast, while the rule it buys
    runs once per leapfrog step.
    """

    def logdet_at(t):
        scaled = with_jitter(problem, alpha * jnp.exp(t))
        return logdet(block_cholesky(band_block_tridiagonal(scaled, prior, b_nat, block_size)))

    return -0.5 * float(jax.grad(logdet_at)(jnp.asarray(0.0)))


def _prior_std(prior: SmoothnessPrior, n_pix: int, n_comp: int) -> np.ndarray:
    """Pointwise sigma under the prior alone — the "learned nothing" line.

    ``Lambda_p`` is block diagonal over components and pentadiagonal within each, so this
    is cheap; what it buys is the reference every posterior number is quoted against.
    A band that sits at the prior's is the D42 failure mode — a number that looks
    equally convincing on a good design and a useless one.
    """
    chol = block_cholesky(prior_block_tridiagonal(prior, n_pix, n_comp, max(2 * n_comp, 64)))
    return np.asarray(jnp.sqrt(selected_inverse_diag(chol))).reshape(n_pix, n_comp).T


def _prior_mode_std(
    prior: SmoothnessPrior, n_pix: int, n_comp: int, n_modes: int, keep, **modes_kw
) -> np.ndarray:
    """Worst-determined modes under the prior alone, on the *same* subspace.

    Quoted next to the posterior modes, so it has to be the same eigenproblem with the
    data term removed — same coordinates, same restriction. Comparing a posterior mode
    over the observed pixels against a prior mode over the whole grid would make every
    design look like it had learned something.
    """
    if n_modes <= 0:
        return np.zeros(0)
    block = max(2 * n_comp, 64)
    chol = block_cholesky(prior_block_tridiagonal(prior, n_pix, n_comp, block))
    vals, _, _ = _top_eigenpairs(chol, n_modes, keep, **modes_kw)
    return np.sqrt(np.maximum(vals, 0.0))


def _data_diagonal(precision, prior: SmoothnessPrior, n_pix: int, n_comp: int) -> np.ndarray:
    """``diag(A^T W A)`` as ``(n_comp, n_pix)`` — how much weight the design puts where.

    Exactly ``diag(precision) - diag(Lambda_p)``, and the subtraction is exact where it
    matters: a coordinate no epoch reaches receives a sum of products of zeros in the
    band assembly, so its precision entry *is* the prior's own float and the difference
    comes back as exactly ``0.0``.

    Per *component*, not per pixel, which matters twice over — a telluric or nebular
    component sits in a different frame from the stars and is therefore observable at a
    different set of pixels, and a faint companion's whole row is scaled by
    ``ell_2**2``, four orders of magnitude below the primary's at a 1% light fraction.
    """
    diag = np.asarray(jnp.diagonal(precision.diag, axis1=-2, axis2=-1)).reshape(-1)[
        : n_comp * n_pix
    ]
    prior_diag = np.asarray(_prior_diagonals(prior, n_pix)[0])  # (n_comp, n_pix), ridge included
    return diag.reshape(n_pix, n_comp).T - prior_diag


def _auto_region(data_diag: np.ndarray, floor: float) -> np.ndarray:
    """The pixels the design covers properly, as opposed to grazes.

    A model grid is always built wider than the data — by the largest component shift
    plus the LSF kernel radius (:meth:`albireo.grids.LogGrid.covering`) — so its margin
    pixels are reachable *only* at the most extreme shifts, and then only through the
    tail of the kernel. Their weight is therefore not zero but a ramp down to zero, and
    they are the worst-determined directions of the covariance on essentially every real
    problem. Reporting one of them as "the worst-determined mode" would answer a question
    about how much margin the grid was given, not about the observing design.

    So the cut is placed in that ramp: keep coordinates carrying at least ``floor``
    times the *median* weight of the coordinates this component is weighted at. The
    interior of a design is flat in this quantity and the margin ramps through it, so
    any value well inside ``(0, 1)`` separates them and the exact choice does not matter
    — which is the argument for having a default at all. The median is taken per
    component for the ``ell_2**2`` reason above; a global one would delete a faint
    companion entirely.

    It is a convention, not a measurement, so it is a named argument and it is reported.
    """
    out = np.zeros(data_diag.shape, dtype=bool)
    for i, row in enumerate(data_diag):
        touched = row > 0.0
        if touched.any():
            out[i] = row >= float(floor) * np.median(row[touched])
    return out


def _resolve_region(region, observed: np.ndarray, grid: LogGrid) -> np.ndarray:
    """Intersect a user's region of interest with the pixels the design actually covers.

    Always an intersection: a window the data do not cover cannot be forecast, and
    quietly including it would put the grid margin back into the answer through the
    front door.
    """
    if region is None:
        return observed
    arr = np.asarray(region)
    if arr.dtype != bool:
        if arr.shape != (2,):
            raise ValueError(
                "region must be None, a (wave_min, wave_max) pair, or a boolean mask of "
                f"shape ({grid.n},) or {observed.shape}; got array of shape {arr.shape} "
                f"and dtype {arr.dtype}"
            )
        lo, hi = float(arr[0]), float(arr[1])
        if not hi > lo:
            raise ValueError(f"region ({lo}, {hi}) is empty or reversed")
        arr = ((grid.wave >= lo) & (grid.wave <= hi))[None, :]
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[-1] != grid.n or arr.ndim != 2 or arr.shape[0] not in (1, observed.shape[0]):
        raise ValueError(
            f"a boolean region must have shape ({grid.n},) or {observed.shape}; got {arr.shape}"
        )
    return observed & np.broadcast_to(arr, observed.shape)


def _separation_diagnostics(delta_pix: np.ndarray, grid: LogGrid, scales_kms, threshold: float):
    """The closed-form two-component diagnostic of ``docs/math.md`` §5.1.

    With the light fractions absorbed into ``u_i = ell_i d_i``, the per-mode information
    matrix over the two components has eigenvalues ``J +- |g(k)|`` for the sum and
    difference modes, with ``g(k) = sum_j exp(i k Delta_j)`` and ``Delta_j`` the
    differential log-shift at epoch ``j``. The difference mode — telling the two stars
    apart — is therefore noisier than the sum by

        ``sqrt[(J + |g(k)|) / (J - |g(k)|)]``,

    exactly (no small-``k`` expansion), diverging at ``k = 0`` where the split is
    singular for any design. Returned against a *feature scale*: one full period of the
    mode, expressed as a velocity, which is the axis a spectroscopist reads — a scale of
    30 km/s is a line, 1000 km/s is a continuum undulation.

    Returns ``(k, penalty, blind_fraction)``, the last being the fraction of the sampled
    scales at which the penalty exceeds ``threshold``.

    This is the idealized problem (two components, unit weights, one grid, no LSF, no
    prior), so it is a screening tool and an explanation, not the answer; the exact
    numbers come from the assembled covariance. It is nonetheless the part an observer
    can act on, since the distribution of ``Delta_j`` is the only thing on this page a
    proposal gets to choose.

    **``Var_j(Delta_j)`` alone is the wrong thing to maximize, and this is where that
    shows.** §5.1's expansion keeps only the second moment, but ``|g(k)|`` depends on
    the whole distribution: a cadence aliased to the orbital period visits the *two*
    extreme values of ``Delta`` over and over, which maximizes the variance and is a
    poor design, because two values leave ``|g|`` recurring to ``J`` at a comb of
    scales. Measured in ``examples/08_forecast.py`` — a 13.7 d circular SB2, eight
    epochs in hand and twelve to plan — twelve more at ``P/2`` keep the RMS differential
    velocity at 117.8 km/s and stay blind over 58% of the sampled scale range, while the
    same twelve spread over phase *lower* the RMS to 99.3 km/s and the blind fraction to
    33%. The exact calculation prefers the second, by **375 nats against 243**.
    """
    j = delta_pix.size
    scales = np.asarray(scales_kms, dtype=np.float64)
    # k is in radians per model pixel; one period of the mode spans scale/dv pixels.
    k = 2.0 * np.pi / (scales / grid.dv_kms)
    g = np.abs(np.exp(1j * k[:, None] * delta_pix[None, :]).sum(axis=1))
    # |g| <= J with equality at k = 0 (and, for an aliased design, at a comb of other k),
    # so the denominator is floored at the point where the difference mode carries a
    # rounding error's worth of the sum mode's information. That is a penalty of ~7e7 —
    # blind by any reading, and finite, which an unfloored division is not.
    penalty = np.sqrt((j + g) / np.maximum(j - g, j * np.finfo(np.float64).eps))

    # The summary statistic is the *fraction* of the sampled range the design is blind
    # over, and not a "clean down to X km/s" crossing, because the penalty is not
    # monotone and a crossing is therefore not a meaningful number. |g(k)| decays with k
    # only when the Delta values are well spread; a cadence aliased to the period gives
    # two values of Delta and |g(k)| = J |cos(k Delta_sep / 2)|, which returns to J
    # periodically — a comb of blind spots at narrow scales. Measured on an 8-epoch
    # aliased design, one and the same curve puts its first upcrossing at 8.8 km/s and
    # its last at 802, which is how little a crossing means here; the blind fraction is
    # the statistic that both survives the comb and orders the designs the way the exact
    # calculation does (58% aliased against 33% spread, in examples/08_forecast.py).
    return k, penalty, float(np.mean(penalty > threshold))


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensitivityForecast:
    """What one observing design determines about the component spectra.

    Every field is a property of the design — epochs, phases, weights, masks, LSF,
    light fractions, response and prior — and none of it depends on any flux. See the
    module docstring for what that does and does not license.

    ``baseline`` is the forecast for the subset of epochs already in hand, when
    :func:`sensitivity_forecast` was given one; it is the same class with its own
    ``baseline`` of ``None``, computed through the identical code path on a subset of
    the identical inputs. That sameness is the point: a comparison between two designs
    assembled by two different routes is exactly where a wrong answer hides, so there is
    only one route.
    """

    grid: LogGrid
    component_labels: tuple[str, ...]
    epoch_indices: tuple[int, ...]
    component_std: np.ndarray  # (n_comp, n_pix) posterior sigma per component per pixel
    prior_std: np.ndarray  # (n_comp, n_pix) the same under the prior alone
    region: np.ndarray  # (n_comp, n_pix) bool: the coordinates summarized below
    logdet_posterior: float  # log det(Lambda_p + A^T W A)
    logdet_prior: float  # log det(Lambda_p)
    p_eff: float  # constrained spectral degrees of freedom
    mode_std: np.ndarray  # (n_modes,) sigma of the worst-determined modes
    mode_vectors: np.ndarray  # (n_modes, n_comp, n_pix) the modes themselves
    prior_mode_std: np.ndarray  # (n_modes,) the same under the prior alone
    mode_residual: float  # largest relative Rayleigh residual over the modes
    rms_delta_kms: float  # RMS differential velocity over epochs (nan unless 2 stellar)
    feature_scale_kms: np.ndarray  # scale axis of the idealized diagnostic
    separation_penalty: np.ndarray  # sqrt[(J + |g|) / (J - |g|)] on that axis
    blind_fraction: float  # fraction of that axis above `penalty_threshold`
    penalty_threshold: float
    baseline: SensitivityForecast | None = None

    @property
    def n_epochs(self) -> int:
        return len(self.epoch_indices)

    @property
    def n_planned(self) -> int:
        """Epochs this design adds to :attr:`baseline` (0 without one)."""
        return 0 if self.baseline is None else self.n_epochs - self.baseline.n_epochs

    @property
    def n_linear(self) -> int:
        """Spectral parameters in the linear system, ``n_components * n_pixels``."""
        return int(self.component_std.size)

    @property
    def information_nats(self) -> float:
        """Expected information gain over the prior, in nats.

        For a linear-Gaussian model the expected Kullback-Leibler divergence from prior
        to posterior, averaged over the prior predictive, is exactly
        ``0.5 (log det Lambda_post - log det Lambda_prior)`` — the data-free half of the
        marginal likelihood, and the Bayesian D-optimality criterion. It is the single
        scalar that ranks designs when there is no one component or wavelength region
        the run is about; when there is, read :attr:`mode_std` and
        :attr:`component_std` instead, because a determinant can be raised by
        constraining directions nobody cares about.
        """
        return 0.5 * (self.logdet_posterior - self.logdet_prior)

    @property
    def gain_nats(self) -> float:
        """Information the planned epochs add to :attr:`baseline`, in nats (nan without one)."""
        if self.baseline is None:
            return float("nan")
        return self.information_nats - self.baseline.information_nats

    @property
    def median_std(self) -> np.ndarray:
        """Median posterior sigma per component over :attr:`region`, ``(n_comp,)``.

        Over the region, not the whole grid: a model grid carries a margin of pixels no
        epoch reaches (:meth:`albireo.grids.LogGrid.covering`), and those sit at the
        prior's width, so a median over everything reports how generous the margin was.
        :attr:`component_std` is the full grid, for plotting — where seeing the band
        flare outside the data is informative rather than misleading.
        """
        return np.array(
            [
                np.median(row[mask]) if mask.any() else np.nan
                for row, mask in zip(self.component_std, self.region, strict=True)
            ]
        )

    @property
    def prior_median_std(self) -> np.ndarray:
        """The same under the prior alone, ``(n_comp,)`` — the width of knowing nothing."""
        return np.array(
            [
                np.median(row[mask]) if mask.any() else np.nan
                for row, mask in zip(self.prior_std, self.region, strict=True)
            ]
        )

    @property
    def worst_mode_std(self) -> float:
        """Standard deviation of the worst-determined mode (nan if none were computed)."""
        return float(self.mode_std[0]) if self.mode_std.size else float("nan")

    @property
    def prior_worst_mode_std(self) -> float:
        """The same under the prior alone — the ceiling the data have to beat."""
        return float(self.prior_mode_std[0]) if self.prior_mode_std.size else float("nan")

    @property
    def worst_mode_gain(self) -> float:
        """How much better than the prior the worst mode is determined.

        **Expect roughly 1, and do not read it as a failure.** For a two-component fit
        the leading mode is the ``k = 0`` exchange — add a constant to one star's
        spectrum, subtract it from the other's — which ``docs/math.md`` §5.1 shows is
        *exactly* singular for every design, since the difference-mode information
        ``lambda_-(k) ~ J k^2 Var_j(Delta)/2`` vanishes at ``k = 0``. No phase sampling
        removes it; only the ridge ``eta`` makes it proper, and only an external
        constraint (a light ratio, a template) measures it.

        So this number is a check that the leading mode is the one theory predicts, and
        :attr:`mode_std` as a whole — how fast the ladder falls away from it — is what
        distinguishes designs. :attr:`p_eff` is the scalar that ranks them.
        """
        if not self.mode_std.size or self.worst_mode_std <= 0.0:
            return float("nan")
        return self.prior_worst_mode_std / self.worst_mode_std

    @property
    def n_covered(self) -> int:
        """Component-pixels inside :attr:`region` — the denominator :attr:`p_eff` is out of."""
        return int(self.region.sum())

    def summary(self) -> str:
        """The paragraph a proposal or an observing-run decision quotes."""
        base = self.baseline
        lines = []
        have = (
            f"{self.n_epochs} epochs"
            if base is None
            else f"{base.n_epochs} epochs in hand + {self.n_planned} planned"
        )
        lines.append(
            f"Sensitivity forecast: {have}, {len(self.component_labels)} components "
            f"({', '.join(self.component_labels)}), "
            f"{int(self.region.sum())} of {self.region.size} component-pixels weighted."
        )

        def pair(before: float, after: float, fmt: str = ".4g") -> str:
            return f"{after:{fmt}}" if base is None else f"{before:{fmt}} -> {after:{fmt}}"

        if np.isfinite(self.rms_delta_kms):
            lines.append(
                "  differential velocity spread (RMS over epochs): "
                + pair(getattr(base, "rms_delta_kms", np.nan), self.rms_delta_kms, ".1f")
                + " km/s"
            )
            blind = 100.0 * self.blind_fraction
            base_blind = 100.0 * getattr(base, "blind_fraction", np.nan)
            lines.append(
                f"  feature scales where separating the pair costs more than "
                f"{self.penalty_threshold:g}x: "
                + pair(base_blind, blind, ".0f")
                + "% of the sampled range (idealized)"
            )
        if self.mode_std.size:

            def ladder(values, label):
                return "    " + " ".join(f"{v:9.4g}" for v in values) + f"   [{label}]"

            lines.append("  worst-determined modes, sigma (worst first):")
            if base is not None:
                lines.append(ladder(base.mode_std, "in hand"))
                lines.append(ladder(self.mode_std, f"with the {self.n_planned} planned"))
            else:
                lines.append(ladder(self.mode_std, "this design"))
            lines.append(ladder(self.prior_mode_std, "prior alone"))
            lines.append(
                f"    the leading mode sits at {self.worst_mode_gain:.2f}x the prior: for two "
                "components that is the k=0 exchange, which is degenerate for every design "
                "(math.md 5.1). What a design buys is how fast the rest of the ladder falls."
            )
        med, base_med = self.median_std, None if base is None else base.median_std
        prior_med = self.prior_median_std
        for i, name in enumerate(self.component_labels):
            lines.append(
                f"  median band, {name}: "
                + pair(np.nan if base_med is None else base_med[i], med[i])
                + f" (prior alone {prior_med[i]:.4g})"
            )
        lines.append(
            "  constrained spectral degrees of freedom: "
            + pair(getattr(base, "p_eff", np.nan), self.p_eff, ".0f")
            + f" of {self.n_covered} covered ({self.n_linear} on the full grid)"
        )
        if base is not None:
            lines.append(
                f"  expected information gain from the {self.n_planned} planned epochs: "
                f"{self.gain_nats:.4g} nats"
            )
        lines.append(
            "  conditional on the assumed orbit, light fractions, LSF, response, weights "
            "and prior; no fluxes were used."
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The forecast
# ---------------------------------------------------------------------------


def _velocity_rows(orbit, velocities, bjd, n_epochs: int):
    """Stellar velocities ``(n_stellar, n_epochs)`` from an orbit or an explicit table."""
    if (orbit is None) == (velocities is None):
        raise ValueError(
            "provide exactly one of orbit= (a theta mapping with period/t_conj/secosw/"
            "sesinw/k, or an albireo.simulate.OrbitParams) or velocities= (an explicit "
            "(n_stellar, n_epochs) table)"
        )
    if velocities is not None:
        vel = np.atleast_2d(np.asarray(velocities, dtype=np.float64))
    elif isinstance(orbit, OrbitParams):
        vel = np.asarray(orbit.component_velocities(np.asarray(bjd, dtype=np.float64)))
    elif isinstance(orbit, Mapping):
        from albireo.inference import orbit_velocities

        vel = np.asarray(orbit_velocities(dict(orbit), jnp.asarray(bjd)))
    else:
        raise ValueError(
            "orbit must be a mapping of theta sites (period, t_conj, secosw, sesinw, k) "
            f"or an albireo.simulate.OrbitParams; got {type(orbit).__name__}"
        )
    if vel.ndim != 2 or vel.shape[1] != n_epochs:
        raise ValueError(
            f"the orbit gives velocities of shape {vel.shape}; expected (n_stellar, {n_epochs})"
        )
    return vel


def _default_scales(grid: LogGrid, n_scales: int) -> np.ndarray:
    """Log-spaced feature scales from two pixels to the whole grid, as velocities."""
    return np.geomspace(2.0 * grid.dv_kms, grid.n * grid.dv_kms, int(n_scales))


def sensitivity_forecast(
    grid: LogGrid,
    dataset: Dataset,
    *,
    light_fractions,
    lsf_sigma_v: Mapping[str, float | Sequence[float]],
    prior: SmoothnessPrior,
    orbit: Mapping | OrbitParams | None = None,
    velocities=None,
    baseline=None,
    lsf_anchors_angstrom: Mapping[str, Sequence[float]] | None = None,
    lsf_h3: Mapping[str, float | Sequence[float]] | None = None,
    response_coeffs=None,
    telluric: bool = False,
    nebular: bool = False,
    nebular_v_kms: float = 0.0,
    nebular_amplitudes=None,
    jitter=None,
    ar1_phi=None,
    region=None,
    region_floor: float = 0.1,
    n_modes: int = 4,
    mode_iterations: int = 120,
    mode_seed: int = 0,
    feature_scales_kms=None,
    n_scales: int = 48,
    penalty_threshold: float = 2.0,
    block_size: int | None = None,
    half_bandwidth: int | None = None,
) -> SensitivityForecast:
    """Forecast what a set of epochs determines about the component spectra.

    Computes the posterior covariance ``(Lambda_p + A^T W A)^{-1}`` of the stacked
    deviation spectra for ``dataset`` — which needs no fluxes, so the epochs may be
    planned rather than observed (:func:`plan_epochs`) — and summarizes it as the
    pointwise band, the worst-determined modes, and the constrained degree-of-freedom
    count, each quoted against the prior-only value.

    Parameters
    ----------
    grid, dataset, light_fractions, lsf_sigma_v, lsf_anchors_angstrom, lsf_h3,
    response_coeffs, telluric, nebular, nebular_v_kms, nebular_amplitudes
        As in :func:`albireo.forward.build_problem`. ``dataset`` is the *design*: the
        epochs already in hand, the ones being planned, or both. Per-epoch quantities
        (``light_fractions``, ``response_coeffs``, ``nebular_amplitudes``) must cover
        every epoch of the design.
    prior
        The :class:`~albireo.priors.SmoothnessPrior` the fit will use, with one
        ``(tau, eta)`` pair per component in the order stellar, telluric, nebular.
        **It is part of the forecast, not a detail of it**: ``eta`` sets the scale of
        the low-frequency directions the design is being asked to constrain, so a
        forecast against a prior looser than the one you will fit with reports a
        degeneracy you will not have, and one tighter reports a design that works
        because the prior is doing the work. Take ``(tau, eta)`` from an ML-II fit to
        comparable data where there is one.
    orbit, velocities
        Exactly one. ``orbit`` is either a ``theta`` mapping (``period``, ``t_conj``,
        ``secosw``, ``sesinw``, ``k`` — what a fit returns, via
        :func:`albireo.inference.orbit_parameters`) or an
        :class:`albireo.simulate.OrbitParams` (what a paper's table gives). Either is
        evaluated at every epoch's ``bjd``. ``velocities`` is an explicit
        ``(n_stellar, n_epochs)`` table for designs that are not Keplerian — a
        wide pair, a moving group, a fixed set of assumed offsets.
    baseline
        Optional epoch indices into ``dataset`` naming the observations already in
        hand; the rest of the design is what is being planned. The result then carries
        a full :attr:`~SensitivityForecast.baseline` forecast for that subset and
        reports the difference, and the two are computed by the same code on the same
        inputs, so the comparison cannot drift. ``None`` (default) forecasts the design
        on its own.
    jitter
        Optional noise-inflation factor ``alpha`` (scalar or per-epoch), as in
        :func:`albireo.forward.with_jitter`. This is where to put the suspicion that
        the promised inverse variances are optimistic — which, for a design built from
        an exposure-time calculator, they usually are. The whole forecast scales with
        the weights.
    ar1_phi
        Optional AR(1) correlation of the pixel noise (:func:`albireo.forward.with_ar1`).
        Correlated noise carries materially less information per pixel than the same
        variance would if it were white, so a forecast that assumes white noise for a
        pipeline that resamples is optimistic in a direction no amount of exposure time
        fixes.
    region
        Which model pixels the mode and median summaries are taken over: ``None``
        (default) for every pixel the design puts weight on, a ``(wave_min, wave_max)``
        pair, or a boolean mask of shape ``(n_pixels,)`` or ``(n_components, n_pixels)``.
        Whatever is given is **intersected** with the covered set, because the model
        grid is always wider than the data (shift plus kernel margin,
        :meth:`albireo.grids.LogGrid.covering`) and those margin pixels sit at the prior
        by construction — left in, they *are* the worst-determined mode of essentially
        every real problem, and the forecast would report the grid's margin instead of
        the observing design. Narrow it to a line window to ask what a design does for
        the feature the science depends on, which is usually a sharper question than
        what it does on average.
    region_floor
        Where the automatic region cuts through the grid margin: a coordinate is kept
        when the design weights it at least this fraction of the median weight over
        that component's weighted coordinates (:func:`_auto_region`). The interior is
        flat in that quantity and the margin ramps through it, so the default 0.1 and
        anything else well inside ``(0, 1)`` give the same region; raise it to trim
        partially-covered wings, set it to 0 to keep every pixel any epoch touches.
    n_modes
        Worst-determined modes to extract (0 to skip the eigenproblem). The default 4
        is enough to see whether the leading mode is isolated or one of a cluster.
    mode_iterations, mode_seed
        Subspace-iteration steps and the seed for its starting block. The default is
        generous because the steps are nearly free — each is a handful of banded solves
        against an assembly and a Cholesky that dominate the call, and a run measured at
        2.2 s took the same 2.2 s at 40 steps and at 320. Eigen*values* converge long
        before eigen*vectors* (seven significant figures by step 20 in that run, against
        a vector residual still at 2e-3), so raise this only if
        :attr:`~SensitivityForecast.mode_residual` matters to what you are doing —
        plotting the mode, rather than reading its width.
    feature_scales_kms, n_scales
        Scale axis for the idealized §5.1 diagnostic: an explicit array of feature
        scales (as velocities), or ``n_scales`` points log-spaced from two model pixels
        to the whole grid. :attr:`~SensitivityForecast.blind_fraction` is a fraction
        *of this axis*, so its absolute value carries the axis's arbitrariness — the
        broad end is blind for every design, since that is what §5.1's ``k -> 0``
        singularity says. Compare it between candidate designs on one axis, which is
        what it is for; narrow the axis to the scales the science needs if the absolute
        number is going to be quoted.
    penalty_threshold
        Noise penalty counted as blind by :attr:`~SensitivityForecast.blind_fraction`
        (default 2, i.e. "separating the pair costs a factor of two"). A convention,
        stated so the number can be read.
    block_size, half_bandwidth
        Solver block size, and a static per-component half-bandwidth override. The
        default is the exact bandwidth the design's own shifts need
        (:attr:`albireo.forward.Problem.natural_half_bandwidth`), which is what a
        non-traced computation should use.

    Returns
    -------
    SensitivityForecast

    Examples
    --------
    >>> design = Dataset([*observed, *plan_epochs(observed[0], t)], frame="barycentric")
    ... # doctest: +SKIP
    >>> fc = sensitivity_forecast(grid, design, orbit=fit_orbit, light_fractions=(0.6, 0.4),
    ...                           lsf_sigma_v={"FEROS": 4.1}, prior=prior,
    ...                           baseline=range(len(observed)))  # doctest: +SKIP
    >>> print(fc.summary())  # doctest: +SKIP

    See Also
    --------
    plan_epochs : build the epochs of an observation that has not happened.
    albireo.calibrate.detection_limit : the other planning question — *how faint a
        companion would this data have found* — which does need fluxes, because it is
        about a realized dataset rather than a design.
    """
    n_ep = dataset.n_epochs
    vel = _velocity_rows(orbit, velocities, dataset.bjd, n_ep)
    n_stellar = vel.shape[0]

    ell = np.asarray(light_fractions, dtype=np.float64)
    if ell.ndim == 1:
        ell = np.repeat(ell[:, None], n_ep, axis=1)
    if ell.shape != (n_stellar, n_ep):
        raise ValueError(
            f"light_fractions must be ({n_stellar},) or ({n_stellar}, {n_ep}) to match the "
            f"{n_stellar} stellar components the orbit implies; got {ell.shape}"
        )

    extra = [name for name, on in (("telluric", telluric), ("nebular", nebular)) if on]
    n_comp = n_stellar + len(extra)
    if prior.n_components != n_comp:
        raise ValueError(
            f"prior has {prior.n_components} components, the design has {n_comp} "
            f"({n_stellar} stellar"
            + "".join(f" + {name}" for name in extra)
            + ") — one (tau, eta) pair per component, in the order stellar, telluric, nebular"
        )
    if prior.n_pixels is not None and prior.n_pixels != grid.n:
        raise ValueError(
            f"prior profiles cover {prior.n_pixels} pixels, the model grid has {grid.n} — "
            "rebuild them with albireo.priors.window_profile(grid.wave, ...)."
        )
    # Every check in this function runs before any linear algebra does: a forecast of a
    # survey-scale design is minutes of work, and finding out then that an epoch index was
    # out of range is not a diagnostic, it is a wasted afternoon.
    have = None
    if baseline is not None:
        have = tuple(int(i) for i in baseline)
        if not have:
            raise ValueError("baseline is empty — pass baseline=None to forecast the design alone")
        if len(set(have)) != len(have):
            raise ValueError("baseline contains repeated epoch indices")
        if any(i < 0 or i >= n_ep for i in have):
            raise ValueError(f"baseline indices must lie in [0, {n_ep})")
        if len(have) == n_ep:
            raise ValueError(
                "baseline names every epoch of the design, so there is nothing planned to "
                "forecast. Pass the design as observed-plus-planned and baseline as the "
                "indices of the observed epochs only."
            )
    if not 0.0 <= region_floor < 1.0:
        raise ValueError(f"region_floor must lie in [0, 1); got {region_floor}")
    if penalty_threshold <= 1.0:
        raise ValueError(
            f"penalty_threshold must exceed 1 (the difference mode is never *better* "
            f"determined than the sum mode); got {penalty_threshold}"
        )

    # Per-epoch nuisances, normalized to full-design arrays once so that the design and
    # the baseline can be sliced from the same objects rather than re-derived.
    coeffs = [np.zeros(0)] * n_ep if response_coeffs is None else list(response_coeffs)
    if len(coeffs) != n_ep:
        raise ValueError(f"response_coeffs must have one entry per epoch ({n_ep})")
    amps = None
    if nebular_amplitudes is not None:
        amps = np.broadcast_to(
            np.atleast_1d(np.asarray(nebular_amplitudes, dtype=np.float64)), (n_ep,)
        )
    alpha = np.broadcast_to(
        np.atleast_1d(np.asarray(1.0 if jitter is None else jitter, dtype=np.float64)), (n_ep,)
    )
    if np.any(alpha <= 0.0):
        raise ValueError("jitter must be positive")
    phi = None
    if ar1_phi is not None:
        phi = np.broadcast_to(np.atleast_1d(np.asarray(ar1_phi, dtype=np.float64)), (n_ep,))

    scales = (
        _default_scales(grid, n_scales)
        if feature_scales_kms is None
        else np.atleast_1d(np.asarray(feature_scales_kms, dtype=np.float64))
    )
    if scales.ndim != 1 or scales.size < 2 or np.any(scales <= 0) or np.any(np.diff(scales) <= 0):
        raise ValueError("feature_scales_kms must be a strictly ascending, positive 1-D array")

    labels = tuple([f"star {i + 1}" for i in range(n_stellar)] + list(extra))
    modes_kw = {"iterations": mode_iterations, "seed": mode_seed}
    prior_std = _prior_std(prior, grid.n, n_comp)

    def forecast(indices: tuple[int, ...], mask=None) -> SensitivityForecast:
        """One design, from the shared inputs restricted to ``indices``.

        The design and its baseline both come through here — same construction, same
        assembly, same summaries — because the class of defect a comparison invites is
        two subtly different routes to two numbers that are then subtracted (D46).

        ``mask`` is the coordinate subspace the mode and median summaries are taken
        over. It is derived from the *full design* and then handed to the baseline, so
        the two are compared on the same coordinates. Letting each derive its own would
        be the same defect wearing a different hat: the baseline's own region is
        strictly smaller (fewer epochs reach fewer pixels), so a "before and after"
        would be quietly measuring two different questions.
        """
        idx = list(indices)
        subset = Dataset(tuple(dataset[i] for i in idx), frame=dataset.frame)
        problem = build_problem(
            grid,
            subset,
            velocities=vel[:, idx],
            light_fractions=ell[:, idx],
            lsf_sigma_v=lsf_sigma_v,
            lsf_anchors_angstrom=lsf_anchors_angstrom,
            lsf_h3=lsf_h3,
            response_coeffs=[coeffs[i] for i in idx],
            telluric=telluric,
            nebular=nebular,
            nebular_v_kms=nebular_v_kms,
            nebular_amplitudes=None if amps is None else amps[idx],
        )
        problem = with_jitter(problem, jnp.asarray(alpha[idx]))
        if phi is not None:
            problem = with_ar1(problem, jnp.asarray(phi[idx]))

        b_nat = (
            int(half_bandwidth)
            if half_bandwidth is not None
            else max(problem.natural_half_bandwidth, prior.half_bandwidth)
        )
        precision = band_block_tridiagonal(problem, prior, b_nat, block_size)
        chol = block_cholesky(precision)
        std = np.asarray(jnp.sqrt(selected_inverse_diag(chol))).reshape(grid.n, n_comp).T

        if mask is None:
            covered = _auto_region(_data_diagonal(precision, prior, grid.n, n_comp), region_floor)
            mask = _resolve_region(region, covered, grid)
            if not mask.any():
                raise ValueError(
                    "the region is empty: no model pixel carries weight from any epoch "
                    "(after intersecting with region=, if given). Check that the model "
                    "grid overlaps the data and that region= names wavelengths the "
                    "epochs cover."
                )
        keep = mask.T.reshape(-1)  # interleaved, index q * n_comp + i

        if n_modes > 0:
            vals, vecs, resid = _top_eigenpairs(chol, n_modes, keep, **modes_kw)
            mode_std = np.sqrt(np.maximum(vals, 0.0))
            mode_vectors = vecs.reshape(vals.size, grid.n, n_comp).transpose(0, 2, 1)
            prior_mode_std = _prior_mode_std(prior, grid.n, n_comp, n_modes, keep, **modes_kw)
        else:
            mode_std = np.zeros(0)
            mode_vectors = np.zeros((0, n_comp, grid.n))
            prior_mode_std = np.zeros(0)
            resid = 0.0

        if n_stellar == 2:
            # Delta is a difference of log-shifts, so it is the relative velocity's own
            # shift exactly (they compose by addition, D2) and the spread maps back
            # through the same relation rather than approximately.
            delta = np.asarray(
                grid.velocity_to_pixels(vel[0, idx]) - grid.velocity_to_pixels(vel[1, idx])
            )
            rms = float(np.asarray(grid.pixels_to_velocity(float(np.std(delta)))))
            _, penalty, blind = _separation_diagnostics(
                delta, grid, scales, float(penalty_threshold)
            )
        else:
            rms, blind = float("nan"), float("nan")
            penalty = np.full(scales.size, np.nan)

        return SensitivityForecast(
            grid=grid,
            component_labels=labels,
            epoch_indices=tuple(idx),
            component_std=std,
            prior_std=prior_std,
            region=mask,
            logdet_posterior=float(logdet(chol)),
            logdet_prior=float(prior_logdet(prior, grid.n)),
            p_eff=_effective_parameters(problem, prior, jnp.asarray(alpha[idx]), b_nat, block_size),
            mode_std=mode_std,
            mode_vectors=mode_vectors,
            prior_mode_std=prior_mode_std,
            mode_residual=float(resid),
            rms_delta_kms=rms,
            feature_scale_kms=scales,
            separation_penalty=penalty,
            blind_fraction=blind,
            penalty_threshold=float(penalty_threshold),
        )

    design = forecast(tuple(range(n_ep)))
    if have is None:
        return design
    return replace(design, baseline=forecast(have, mask=design.region))

"""Joint Bayesian inference of the orbit with the spectra marginalized (M3).

The nonlinear parameter vector ``theta`` is a dict of JAX arrays with sites

- ``period``      orbital period [d]
- ``t_conj``      time of conjunction of component 1 (``nu + omega = pi/2``) [d]
- ``secosw``      ``sqrt(e) cos(omega)``
- ``sesinw``      ``sqrt(e) sin(omega)``
- ``k``           RV semi-amplitudes ``(K_1, K_2, ...)`` [km/s], one per inner
  stellar component; even components use ``omega``, odd use ``omega + pi``
- ``log_tau``, ``log_eta`` (optional) — log spectral-prior hyperparameters, one per
  model component (including the telluric component when enabled)
- ``period_out``, ``t_conj_out``, ``secosw_out``, ``sesinw_out``, ``k_out``
  (optional, all five together) — a hierarchical outer orbit (SB3): the inner
  components' center of mass moves with semi-amplitude ``k_out[0]`` and argument
  ``omega_out``, and one additional tertiary component is appended moving with
  ``k_out[1]`` and ``omega_out + pi``
- ``velocity`` (optional) — free per-epoch radial velocities [km/s],
  ``(n_stellar, n_epochs)``, *replacing* the Keplerian entirely (D42): with this site
  present the orbital sites must be absent, and the model is the diagnostic mode in
  which each epoch's velocity is its own parameter. The per-component zero points are
  removed before use, in pixel space where the removal is exact, because a table left
  uncentered would have its absolute level pinned by shift-interpolation error rather
  than by data. Read the ``velocity_rel`` deterministic
  (:func:`relative_velocities`) rather than the raw site, and see
  :func:`keplerian_residuals` for the model check this mode exists to support.
- ``light`` (optional) — stellar light fractions, ``(n_stellar,)`` constant or
  ``(n_epochs, n_stellar)`` per-epoch, rows on the simplex (use Dirichlet priors)
- ``lsf_sigma`` (optional) — Gaussian LSF widths [km/s]: one entry per LSF anchor
  for instruments built with ``lsf_anchors_angstrom`` (wavelength-dependent LSF,
  D37), one entry per un-anchored instrument, concatenated in instrument order; the
  construction-time ``lsf_sigma_v`` values act as per-entry upper bounds (they fix
  the kernel radii)
- ``lsf_h3`` (optional) — Gauss-Hermite LSF skewness (D38): one entry per LSF
  anchor of each *anchored* instrument (un-anchored instruments have no slot — a
  stationary asymmetric LSF is absorbed by the free spectra, math.md §1.3),
  concatenated in instrument order; ``|h3| <= 0.2``. May appear with or without
  ``lsf_sigma`` (without, the widths stay at the build values)
- ``response`` (optional) — multiplicative per-epoch Chebyshev response coefficients,
  ``(n_coef,)`` shared or ``(n_epochs, n_coef)`` per-epoch
  (:func:`albireo.forward.with_response`, D7/D33; ``r = 1 + sum_m c_m T_m``, so
  all-zero coefficients are the unit response). This is the per-epoch continuum
  treatment: keep the order low and the priors tight and zero-centered — a low-order
  response trades against the components' broad features (design.md §5), and when a
  ``response`` site is present the construction-time ``response_coeffs`` are replaced
  outright
- ``log_jitter`` (optional) — log noise-inflation factor, scalar (shared) or one per
  epoch; the weights become ``w_j / exp(2 log_jitter_j)``
  (:func:`albireo.forward.with_jitter`, D15). Read its caveats before using it: a
  jitter fitted against systematics widens the error bars around a still-biased point.
- ``ar1_phi`` (optional) — AR(1) correlation of the standardized noise, scalar
  (shared) or one per epoch, ``|phi| < 1`` (:func:`albireo.forward.with_ar1`, D34).
  Requires the model to be built with ``ar1=True``, because the correlated coupling
  widens the static solver bandwidth; the marginal stays on the band assembly path
  (D35). Composes with ``log_jitter``: alpha scales, phi correlates.
- ``log_nebular_amp`` (optional) — log per-epoch amplitude of the nebular component,
  ``(n_epochs,)`` (D40; requires ``nebular=True`` at construction). The site is
  *centered* before use — ``a_j = exp(u_j - mean(u))`` — because only ``a_j d_neb``
  is observable, so the overall scale is degenerate with the component spectrum and
  the prior would otherwise have to break it. The consequence is that the prior on
  this site is a prior on the epoch-to-epoch *variation* (a zero-mean Normal with
  sigma 0.2 says "roughly ±20% night to night"), and that shifting every entry by a
  constant does nothing at all. The common mode is therefore an exactly flat direction
  of the *likelihood*, held only by the site's own prior — the posterior stays proper
  and well conditioned (curvature 1/sigma^2 along it), but ``mean(log_nebular_amp)``
  in the samples is the prior and nothing else. Read the ``nebular_amp`` deterministic
  (:func:`nebular_amplitudes`) rather than the raw site.

``gamma`` is identically zero (design decision D14: a systemic velocity is exactly
degenerate with a common shift of the component spectra). The ``(secosw, sesinw)``
parameterization is smooth through ``e = 0`` — where ``omega`` and a time of
periastron are undefined — and carries a uniform-on-the-unit-disk prior to a uniform
prior on ``e``; the disk constraint ``e < 1`` enters the model as a ``-inf`` factor,
with ``e`` clipped to ``ecc_max`` before the Kepler solve so the likelihood stays
finite (and rejectable) outside it. The map is non-differentiable only at the exact
point ``secosw = sesinw = 0`` — initialize circular orbits slightly off the origin.

Hyperparameters follow empirical Bayes by default (``docs/math.md`` §5.1: the prior
curvature scale *is* information the data cannot supply below the LSF scale, so it
must be estimated deliberately): :func:`run_map` maximizes the joint over
``(theta, log_tau, log_eta)`` — the marginal likelihood is already integrated over
the spectra, so this is ML-II up to the weak hyperpriors — and NUTS then runs with
the hyperparameters held at those values (pass them via ``fixed=``), or sampled, at
the user's choice.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
from jax.flatten_util import ravel_pytree
from numpyro.infer import MCMC, NUTS, init_to_value
from numpyro.infer.util import initialize_model

from albireo.data import Dataset
from albireo.forward import (
    build_problem,
    with_ar1,
    with_jitter,
    with_light_fractions,
    with_lsf,
    with_nebular_amplitudes,
    with_response,
    with_shifts,
    with_velocities,
)
from albireo.grids import LogGrid
from albireo.kepler import radial_velocity, t_peri_from_t_conj
from albireo.likelihood import MarginalResult, draw_spectra, marginal_loglikelihood
from albireo.priors import SmoothnessPrior

__all__ = [
    "MAPResult",
    "MarginalOrbitModel",
    "keplerian_residuals",
    "laplace_inverse_mass",
    "nebular_amplitudes",
    "orbit_parameters",
    "orbit_velocities",
    "posterior_spectra",
    "relative_velocities",
    "relative_velocity_errors",
    "run_map",
    "run_nuts",
]

_ECC_MAX_DEFAULT = 0.95  # the Kepler solver is verified up to e = 0.95
# Live-bytes budget for one vmapped batch of a log_likelihood_sweep. The sweep is
# forward-only (no tape), so the constraint is the batch's own working set rather than
# assembly.py's gradient budget, and it can be a good deal more generous.
_SWEEP_BATCH_BYTES = 1 << 30
# Gauss-Hermite skewness bound: beyond ~0.2 the truncated series dips measurably
# negative in the tail, and real instrument profiles sit well below it (D38).
_H3_MAX = 0.2
_OUTER_SITES = ("period_out", "t_conj_out", "secosw_out", "sesinw_out", "k_out")
_THETA_SITES = (
    "period",
    "t_conj",
    "secosw",
    "sesinw",
    "k",
    *_OUTER_SITES,
    "velocity",
    "light",
    "lsf_sigma",
    "lsf_h3",
    "response",
    "log_jitter",
    "ar1_phi",
    "log_nebular_amp",
    "log_tau",
    "log_eta",
)


def _sweep_batch_default(n_trials: int, n: int, bandwidth: int) -> int:
    """Trials per vmapped batch of :meth:`MarginalOrbitModel.log_likelihood_sweep`.

    Same two-regime shape as :func:`albireo.assembly._epoch_chunk_default`: run the whole
    sweep as one batch while that stays under ``_SWEEP_BATCH_BYTES``, otherwise cut it to
    fit. The estimate is the handful of ``(n, bandwidth)`` float64 arrays a marginal
    solve keeps live — the band tensor and the block-tridiagonal factor dominate, and
    everything smaller rides along inside the constant.
    """
    per_trial = 48 * max(n, 1) * max(bandwidth, 1)
    if n_trials * per_trial <= _SWEEP_BATCH_BYTES:
        return max(1, n_trials)
    return max(1, min(n_trials, _SWEEP_BATCH_BYTES // per_trial))


def _has_outer_orbit(theta: Mapping) -> bool:
    present = [s for s in _OUTER_SITES if s in theta]
    if present and len(present) != len(_OUTER_SITES):
        missing = [s for s in _OUTER_SITES if s not in theta]
        raise ValueError(f"outer orbit needs all of {_OUTER_SITES}; missing {missing}")
    return bool(present)


def _max_relative_shift(problem) -> jax.Array:
    """Max over epochs/component pairs of |delta_i - delta_i'| in pixels (traced)."""
    rel = jnp.asarray(0.0)
    for g in problem.groups:
        s = g.shifts
        rel = jnp.maximum(rel, jnp.max(jnp.abs(s[:, :, None] - s[:, None, :])))
    return rel


def _kepler_block(theta: Mapping, suffix: str, ecc_max: float) -> dict:
    """One Keplerian's physical parameters from suffixed theta sites (differentiable)."""
    h = jnp.asarray(theta["secosw" + suffix])
    s = jnp.asarray(theta["sesinw" + suffix])
    return {
        "period": jnp.asarray(theta["period" + suffix]),
        "t_conj": jnp.asarray(theta["t_conj" + suffix]),
        "ecc": jnp.minimum(h * h + s * s, ecc_max),
        "omega": jnp.arctan2(s, h),
        "k": jnp.atleast_1d(jnp.asarray(theta["k" + suffix])),
    }


def _block_velocity(par: Mapping, bjd, *, component: int):
    """RV of one component of a Keplerian block (odd components use ``omega + pi``)."""
    t_peri = t_peri_from_t_conj(
        par["t_conj"], period=par["period"], ecc=par["ecc"], omega=par["omega"]
    )
    return radial_velocity(
        bjd,
        period=par["period"],
        t_peri=t_peri,
        ecc=par["ecc"],
        omega=par["omega"] + (component % 2) * jnp.pi,
        k=par["k"][component],
    )


def orbit_parameters(theta: Mapping, *, ecc_max: float = _ECC_MAX_DEFAULT) -> dict:
    """Physical orbit parameters from a ``theta`` dict (differentiable).

    Returns ``{"period", "t_conj", "ecc", "omega", "k"}`` with ``ecc`` clipped to
    ``ecc_max`` (the unclipped value is enforced separately by the model's disk
    constraint). If the hierarchical outer-orbit sites are present, an ``"outer"``
    key holds the same structure for the outer Keplerian
    (``k = (K_inner_com, K_tertiary)``).
    """
    par = _kepler_block(theta, "", ecc_max)
    if _has_outer_orbit(theta):
        par["outer"] = _kepler_block(theta, "_out", ecc_max)
    return par


def orbit_velocities(theta: Mapping, bjd, *, ecc_max: float = _ECC_MAX_DEFAULT):
    """Stellar radial velocities, shape ``(n_stellar, n_epochs)``, barycentric frame.

    Inner components follow :class:`albireo.simulate.OrbitParams` conventions:
    component ``i`` uses ``omega + (i % 2) * pi`` and semi-amplitude ``k[i]``;
    ``gamma = 0`` (D14). With the outer-orbit sites present (SB3), the outer
    center-of-mass velocity (semi-amplitude ``k_out[0]``, argument ``omega_out``,
    conjunction convention on the inner pair's center of mass) is added to every
    inner component, and one tertiary row is appended with semi-amplitude
    ``k_out[1]`` and argument ``omega_out + pi``.
    """
    par = orbit_parameters(theta, ecc_max=ecc_max)
    bjd = jnp.asarray(bjd)
    rows = [_block_velocity(par, bjd, component=i) for i in range(par["k"].shape[0])]
    if "outer" in par:
        out = par["outer"]
        if out["k"].shape[0] != 2:
            raise ValueError("k_out must have exactly two entries (K_inner_com, K_tertiary)")
        v_com = _block_velocity(out, bjd, component=0)
        rows = [v + v_com for v in rows]
        rows.append(_block_velocity(out, bjd, component=1))
    return jnp.stack(rows)


def _centered_shifts(velocity, grid):
    """Per-epoch pixel shifts with each component's zero point removed.

    The centering is the whole design decision behind the free-velocity table (D42), and
    it belongs in *pixel* space rather than velocity space for a reason that is exact
    rather than stylistic: ``xi = artanh(v/c)`` turns relativistic velocity addition into
    ordinary addition, so subtracting a constant here is exactly a translation of the
    component's spectrum, while subtracting a constant velocity is only a first-order
    approximation of one.
    """
    pix = grid.velocity_to_pixels(jnp.atleast_2d(jnp.asarray(velocity)))
    return pix - jnp.mean(pix, axis=1, keepdims=True)


def relative_velocities(velocity, grid):
    """The identified part of a free-velocity table: velocities per component, zero-pointed.

    A table of free per-epoch velocities has **one arbitrary zero point per stellar
    component**, not one in total. Each component's deviation spectrum is a free vector,
    so translating it absorbs a constant added to that component's shifts, and the
    likelihood cannot tell the difference — the generalization of ``gamma = 0`` (D14),
    except that with no Keplerian tying the components together each of them gets its own.

    That degeneracy is exact only for whole-pixel translations, because the model shifts
    spectra by linear interpolation and a fractional shift blurs slightly as well as
    moves. Measured on a 10-epoch SB2 at SNR 200: a one-pixel common shift of one
    component costs **4e-9 of the log-likelihood in relative terms** (boundary effects
    only), while a 0.1-pixel one costs 7.3 nats. So a table left uncentered would have its
    absolute zero point pinned not by the data but by *interpolation error* — a number
    that looks like a systemic velocity, moves when the model grid is resampled, and means
    nothing. albireo removes it instead, and reports what remains.

    What remains is fully identified and is what the science needs: each component's
    velocity *variation* (hence its semi-amplitude), the epoch-to-epoch differences, and
    the slope of component 1 against component 2 — the Wilson mass ratio, which is a slope
    and therefore untouched by either zero point. What is *not* recoverable from this
    table is the systemic velocity, or the absolute velocity of either star. Measure those
    from the disentangled spectra afterwards, exactly as D14 prescribes for gamma.

    Parameters
    ----------
    velocity
        Free per-epoch velocities [km/s], ``(n_stellar, n_epochs)`` — the raw
        ``velocity`` theta site, or a posterior sample of it.
    grid
        The model :class:`~albireo.grids.LogGrid` the shifts are taken on. The centering
        is grid-dependent by construction, since it is done in pixel space.

    Returns
    -------
    jax.Array
        ``(n_stellar, n_epochs)`` velocities [km/s], each row relativistically
        zero-pointed to that component's mean epoch. Rows sum to zero *in pixel space*,
        so they will not sum to exactly zero in km/s — that is the nonlinearity being
        handled correctly rather than approximated away.
    """
    return grid.pixels_to_velocity(_centered_shifts(velocity, grid))


def relative_velocity_errors(covariance, unconstrained: Mapping, *, site: str = "velocity"):
    """Per-epoch standard errors of the free-velocity table, zero points projected out.

    **Reading the Laplace covariance's diagonal directly gives the prior back, not an
    error bar.** Each component's zero point is an exactly flat direction of the
    likelihood (:func:`relative_velocities`), so its posterior width is whatever the
    prior said, and every epoch's marginal variance inherits it. Measured on the D42
    fixture with a ``Normal(0, 120)`` prior over 10 epochs: every raw marginal sigma came
    out at **37.95 km/s = 120/sqrt(10)**, identical to four digits across both components
    and all epochs, while the honest per-epoch error is **0.059 km/s**. A factor of 640,
    and — worse than merely wrong — completely insensitive to the data, so it would look
    equally plausible on a good dataset and a useless one.

    Projecting each component's mean out of the covariance removes exactly those
    directions and leaves the identified ones. As a check that the count is right, the
    projected block has exactly ``n_stellar`` zero eigenvalues.

    The projection is applied in velocity units, where it is the identity to
    ``O(v^2/c^2)`` — the Jacobian of the pixel-space centering differs from unity by
    ``~1e-8`` at stellar velocities, far below the approximation already made by using a
    Laplace covariance at all.

    Prefer posterior samples when you have them: the ``velocity_rel`` deterministic that
    :meth:`MarginalOrbitModel.model` records is already the identified table, so
    ``samples["velocity_rel"].std(axis=0)`` needs no projection and no Gaussian
    assumption. This function is for the MAP + Laplace route.

    Parameters
    ----------
    covariance
        Full unconstrained-space covariance from :func:`laplace_inverse_mass`.
    unconstrained
        :attr:`MAPResult.unconstrained` from the *same* fit — it supplies both the site
        ordering within the flattened vector and the table's shape.
    site
        Name of the free-velocity site.

    Returns
    -------
    numpy.ndarray
        ``(n_stellar, n_epochs)`` standard errors [km/s].

    Notes
    -----
    A Laplace covariance is a local Gaussian approximation with the hyperparameters held
    at their MAP values, so it does not carry the widening that marginalizing over
    ``log_tau``/``log_eta`` would add. On the D42 fixture the resulting bars run about
    1.4x optimistic against the realized errors. Treat them as a fast estimate and NUTS
    as the answer.
    """
    unconstrained = dict(unconstrained)
    if site not in unconstrained:
        raise ValueError(f"no {site!r} site in the fit (sites: {sorted(unconstrained)})")
    shape = jnp.shape(jnp.asarray(unconstrained[site]))
    if len(shape) != 2:
        raise ValueError(f"the {site!r} site must be (n_stellar, n_epochs); got shape {shape}")
    # Locate the site's slots without assuming how ravel_pytree orders the dict: flatten
    # a matching pytree of markers and read off where they land.
    marks, _ = ravel_pytree(
        {
            name: jnp.full(jnp.shape(jnp.asarray(value)), 1.0 if name == site else 0.0)
            for name, value in unconstrained.items()
        }
    )
    sel = np.asarray(marks) > 0.5
    cov = np.asarray(covariance)
    if cov.shape != (sel.size, sel.size):
        raise ValueError(
            f"covariance is {cov.shape} but the fit has {sel.size} unconstrained "
            "parameters — they must come from the same model"
        )
    block = cov[np.ix_(sel, sel)]
    n_stellar, n_epochs = shape
    centre = np.eye(n_epochs) - np.ones((n_epochs, n_epochs)) / n_epochs
    projector = np.zeros((n_stellar * n_epochs, n_stellar * n_epochs))
    for i in range(n_stellar):
        s = slice(i * n_epochs, (i + 1) * n_epochs)
        projector[s, s] = centre
    projected = projector @ block @ projector.T
    return np.sqrt(np.clip(np.diag(projected), 0.0, None)).reshape(n_stellar, n_epochs)


def keplerian_residuals(velocity, theta: Mapping, bjd, grid, *, ecc_max: float = _ECC_MAX_DEFAULT):
    """Per-epoch velocity residuals of a free table against a Keplerian orbit.

    The model check the free-velocity mode exists for: fit per-epoch velocities with no
    orbit imposed, then ask whether a Keplerian threads them. A period that is slightly
    wrong, an unmodelled third body, or line-profile variability that the Keplerian
    absorbs into ``e`` all show up here as *structured* residuals — phase-correlated,
    or one epoch far out — where noise alone would not.

    Both tables are zero-pointed the same way before subtracting, and the subtraction is
    done in pixel space so the result is an exact relativistic velocity difference rather
    than a first-order one. That matters because the two tables' arbitrary zero points
    (:func:`relative_velocities`) must cancel *exactly*, or the residual would carry a
    constant offset that means nothing.

    Parameters
    ----------
    velocity
        Free per-epoch velocities [km/s], ``(n_stellar, n_epochs)`` — the ``velocity``
        site from a fit, or one posterior sample of it.
    theta
        Keplerian parameters (``period``, ``t_conj``, ``secosw``, ``sesinw``, ``k``, and
        the outer-orbit sites if any), as :func:`orbit_velocities` takes them.
    bjd
        Epoch times, matching the columns of ``velocity``.
    grid
        The model :class:`~albireo.grids.LogGrid`.
    ecc_max
        Eccentricity clip, matching the model the Keplerian came from.

    Returns
    -------
    jax.Array
        ``(n_stellar, n_epochs)`` residuals [km/s]. Compare them against the per-epoch
        uncertainties of the free table, not against zero.
    """
    free = _centered_shifts(velocity, grid)
    kep = _centered_shifts(orbit_velocities(theta, bjd, ecc_max=ecc_max), grid)
    if free.shape != kep.shape:
        raise ValueError(
            f"the free table is {free.shape} but the Keplerian implies {kep.shape} — "
            "they must agree on both the component count and the epoch count"
        )
    return grid.pixels_to_velocity(free - kep)


def nebular_amplitudes(theta: Mapping):
    """Per-epoch nebular amplitudes from ``theta['log_nebular_amp']`` (differentiable).

    ``a_j = exp(u_j - mean(u))``: the geometric mean is pinned to 1, which is a
    *convention*, not an inference. The model sees only the products ``a_j d_neb``, so
    without a pinned scale the pair ``(c a_j, d_neb / c)`` is the same fit for every
    ``c > 0`` and the only thing separating them is the spectral prior — a direction
    that is nearly flat, unbounded in one coordinate, and would dominate the sampler's
    step size for no return. Centering removes it exactly and costs one degree of
    freedom, the one that was never identified.

    Two consequences to keep in mind when reading a fit. The recovered ``d_neb`` is on
    the scale of a *typical* epoch, so its line strengths are comparable to injected or
    published ones only up to that convention; and the posterior for ``a`` describes
    relative variation, so "epoch 7 is 1.4x" is a statement the data support while
    "the nebular emission was 1.4 units strong" is not.

    Returns
    -------
    jax.Array
        ``(n_epochs,)`` positive amplitudes with geometric mean 1.
    """
    u = jnp.atleast_1d(jnp.asarray(theta["log_nebular_amp"]))
    return jnp.exp(u - jnp.mean(u))


class MarginalOrbitModel:
    """The marginal posterior over orbital parameters for one dataset.

    Bundles the static problem structure (rebin operators, kernels, weights — built
    once) with the θ-dependent path (Kepler velocities → shifts → marginal likelihood)
    so that :meth:`log_likelihood` is a single jit-compiled, differentiable function
    of ``theta``. The solver bandwidth is fixed by ``v_rel_max_kms`` (see
    :meth:`albireo.forward.Problem.half_bandwidth_bound`) so the computation graph is
    static; the numpyro model rejects (``-inf``) any configuration whose actual
    relative shifts exceed that budget, so a prior wider than ``v_rel_max_kms`` slows
    mixing near the bound but can never corrupt the result. The direct
    :meth:`log_likelihood` entry point carries no such guard — keep explicit calls
    within the bound.

    Parameters
    ----------
    grid, dataset, light_fractions, lsf_sigma_v, lsf_anchors_angstrom, response_coeffs
        As in :func:`albireo.forward.build_problem`. ``light_fractions`` and
        ``lsf_sigma_v`` are the build-time values, used whenever θ carries no
        ``light`` / ``lsf_sigma`` site; when those sites *are* inferred, the
        build-time light fractions only set ``n_stellar``, and the build-time LSF
        widths become strict upper bounds (they fix the kernel radii — the model
        rejects wider widths, which the fixed radii would silently truncate).
        ``lsf_anchors_angstrom`` makes an instrument's LSF wavelength-dependent
        (D37) and gives it one ``lsf_sigma`` site entry per anchor rather than one
        in total. ``response_coeffs`` likewise is the fixed response used whenever
        θ carries no ``response`` site, and is replaced outright when it does
        (:func:`albireo.forward.with_response`).
    telluric, nebular, nebular_v_kms
        Extra non-stellar components, as in :func:`albireo.forward.build_problem`.
        Each one enabled adds a trailing row to the recovered spectra and a trailing
        entry to ``prior`` (order: stellar, telluric, nebular), and must be paid for
        in ``v_rel_max_kms``. ``nebular=True`` also enables the ``log_nebular_amp``
        θ site; without the site the amplitudes stay at 1 and the component is static.
    v_rel_max_kms
        Bound on the largest relative velocity between any two model components at
        any epoch (for an SB2: ``(K_1 + K_2)(1 + e)``; for an SB3 add the outer
        orbit's ``(K_AB + K_C)(1 + e_out)``; plus barycentric motion if a telluric
        component is enabled). Give it headroom — the priors must not allow
        configurations that exceed it, or mixing will stall at the guard.
    prior
        Fixed :class:`SmoothnessPrior`, used whenever ``theta`` carries no
        ``log_tau``/``log_eta`` sites. Optional if the hyperparameters are always in
        ``theta`` — with one exception: its **per-pixel profiles are kept even when
        the scalars are inferred** (D40), because a profile is structure rather than a
        hyperparameter. A windowed component therefore needs its prior passed here
        even in a pure ML-II run.
    ecc_max
        Eccentricity clip/constraint (default 0.95, the solver's verified range).
    block_size
        Solver block size passed through to the marginal likelihood.
    ar1
        Allow an ``ar1_phi`` site (correlated noise, D34). The AR coupling widens
        ``A^T W A`` by a static amount (:attr:`albireo.forward.Problem.ar_bandwidth_extra`),
        which must be reserved in the solver bandwidth *up front* — so the site is an
        explicit construction-time choice, like the bandwidth itself (D21). Costs a
        few extra pixels of bandwidth whether or not θ ends up carrying the site.
    """

    def __init__(
        self,
        grid: LogGrid,
        dataset: Dataset,
        *,
        light_fractions,
        lsf_sigma_v: Mapping[str, float | Sequence[float]],
        v_rel_max_kms: float,
        lsf_anchors_angstrom: Mapping[str, Sequence[float]] | None = None,
        response_coeffs=None,
        telluric: bool = False,
        nebular: bool = False,
        nebular_v_kms: float = 0.0,
        prior: SmoothnessPrior | None = None,
        ecc_max: float = _ECC_MAX_DEFAULT,
        block_size: int | None = None,
        ar1: bool = False,
    ):
        ell = np.asarray(light_fractions, dtype=np.float64)
        n_stellar = ell.shape[0]
        self.problem = build_problem(
            grid,
            dataset,
            velocities=np.zeros((n_stellar, dataset.n_epochs)),
            light_fractions=ell,
            lsf_sigma_v=lsf_sigma_v,
            lsf_anchors_angstrom=lsf_anchors_angstrom,
            response_coeffs=response_coeffs,
            telluric=telluric,
            nebular=nebular,
            nebular_v_kms=nebular_v_kms,
        )
        self.bjd = jnp.asarray(dataset.bjd)
        hb = self.problem.half_bandwidth_bound(v_rel_max_kms)
        # The shift budget inside half_bandwidth (inverse of half_bandwidth_bound);
        # the numpyro model rejects any configuration whose actual relative shifts
        # exceed it, so a prior wider than v_rel_max cannot silently corrupt probing.
        # Computed from the *base* bandwidth: the AR extra below is reserved for the
        # noise coupling's reach and must not be spent on shifts.
        support = max(g.row_support for g in self.problem.groups)
        self._shift_bound = hb - 1 - 2 * self.problem.kernel_radius - support
        self.ar1 = bool(ar1)
        self.half_bandwidth = hb + (self.problem.ar_bandwidth_extra if self.ar1 else 0)
        self.block_size = block_size
        self.ecc_max = float(ecc_max)
        self.fixed_prior = prior
        # Instrument order for the (optional) "lsf_sigma" theta site; the widths the
        # kernels were built with are the upper bounds (they fix the kernel radii).
        # Deduplicated, first-seen order: one instrument owns several groups whenever its
        # epochs sit on different native grids (albireo.forward._epoch_groups), and the
        # LSF width is a property of the instrument, not of the grid — one sampled width
        # per group would both mis-shape the site and (via dict(zip(...))) silently drop
        # all but the last group's value.
        self.instruments = tuple(dict.fromkeys(g.instrument for g in self.problem.groups))
        # Site layout: one width per LSF anchor for anchored instruments (D37), one per
        # un-anchored instrument, concatenated in instrument order. build_problem has
        # already validated the counts.
        anchors = lsf_anchors_angstrom or {}
        sizes, maxima = [], []
        for name in self.instruments:
            n_a = max(len(anchors.get(name, ())), 1)
            sig = np.atleast_1d(np.asarray(lsf_sigma_v[name], dtype=np.float64))
            maxima.append(np.full(n_a, sig[0]) if sig.size == 1 else sig)
            sizes.append(n_a)
        self._lsf_sizes = tuple(sizes)
        self._lsf_anchored = tuple(len(anchors.get(name, ())) > 0 for name in self.instruments)
        self._lsf_sigma_max = jnp.asarray(np.concatenate(maxima))
        # The problem is passed as a jit *argument* (Problem is a registered
        # pytree): its arrays enter the graph as runtime parameters. Capturing
        # them as closure constants instead triggers XLA constant folding that
        # allocates tens of GB at survey scale.
        self._marginal_jit = jax.jit(self._marginal_at)
        self._sweep_jit = jax.jit(self._sweep_at, static_argnums=3)

    @property
    def n_stellar(self) -> int:
        return self.problem.n_stellar

    def _prior(self, theta: Mapping) -> SmoothnessPrior:
        if "log_tau" in theta or "log_eta" in theta:
            if "log_tau" not in theta or "log_eta" not in theta:
                raise ValueError("theta must carry both log_tau and log_eta, or neither")
            # The per-pixel profiles are *structure*, not hyperparameters (D40): they say
            # where a component may deviate from the continuum, the sampled scalars say
            # how much. So an inferred (tau, eta) replaces the scalars and keeps the
            # construction-time profiles — dropping them here would silently un-confine a
            # windowed component the moment ML-II was switched on, which is exactly the
            # kind of quiet wrongness the guards elsewhere exist to prevent.
            base = self.fixed_prior
            return SmoothnessPrior(
                jnp.exp(theta["log_tau"]),
                jnp.exp(theta["log_eta"]),
                None if base is None else base.tau_profile,
                None if base is None else base.eta_profile,
            )
        if self.fixed_prior is None:
            raise ValueError(
                "no spectral prior: pass prior= at construction or include log_tau/log_eta in theta"
            )
        return self.fixed_prior

    def _theta_problem(self, theta: Mapping, base=None):
        """Problem at θ: velocities always; light fractions / LSF widths if present."""
        base = self.problem if base is None else base
        if "velocity" in theta:
            # Free per-epoch velocities: no Keplerian at all (D42). The zero point is
            # removed here, in pixel space, where the removal is exact.
            clash = [s for s in ("period", "t_conj", "secosw", "sesinw", "k", *_OUTER_SITES)]
            present = [s for s in clash if s in theta]
            if present:
                raise ValueError(
                    f"theta carries both a free-velocity site and Keplerian sites {present}. "
                    "They are alternatives: 'velocity' replaces the orbit entirely, so a "
                    "Keplerian site alongside it would be silently ignored. Drop one."
                )
            pix = _centered_shifts(jnp.asarray(theta["velocity"]), base.grid)
            if pix.shape != (self.n_stellar, base.n_epochs):
                raise ValueError(
                    f"velocity must have shape ({self.n_stellar}, {base.n_epochs}) — one "
                    f"row per stellar component, one column per epoch; got {pix.shape}"
                )
            problem = with_shifts(base, pix)
        else:
            vel = orbit_velocities(theta, self.bjd, ecc_max=self.ecc_max)
            if vel.shape[0] != self.n_stellar:
                raise ValueError(
                    f"theta implies {vel.shape[0]} stellar components "
                    f"(len(k){' + tertiary' if _has_outer_orbit(theta) else ''}) but the model "
                    f"was built with {self.n_stellar} light fractions"
                )
            problem = with_velocities(base, vel)
        if "light" in theta:
            ell = jnp.asarray(theta["light"])
            if ell.ndim == 2:  # per-epoch, Dirichlet layout (n_epochs, n_stellar)
                ell = ell.T
            problem = with_light_fractions(problem, ell)
        if "lsf_sigma" in theta or "lsf_h3" in theta:
            if "lsf_sigma" in theta:
                sig = jnp.atleast_1d(jnp.asarray(theta["lsf_sigma"]))
                if sig.shape != self._lsf_sigma_max.shape:
                    raise ValueError(
                        f"lsf_sigma must have {self._lsf_sigma_max.shape[0]} entries — one per "
                        f"LSF anchor of an anchored instrument, one per un-anchored instrument "
                        f"(instruments {self.instruments}, sizes {self._lsf_sizes}); "
                        f"got shape {sig.shape}"
                    )
                # Clip into the kernel-radius-valid range so the likelihood stays finite
                # (and rejectable, via the model's lsf_bound guard) outside it.
                sig = jnp.clip(sig, 1e-3 * self._lsf_sigma_max, self._lsf_sigma_max)
            else:
                sig = self._lsf_sigma_max  # h3 alone: widths stay the build values
            parts, start = {}, 0
            for name, n_a in zip(self.instruments, self._lsf_sizes, strict=True):
                parts[name] = sig[start] if n_a == 1 else sig[start : start + n_a]
                start += n_a
            h3_parts = None
            if "lsf_h3" in theta:
                h3 = jnp.atleast_1d(jnp.asarray(theta["lsf_h3"]))
                n_h3 = sum(n for n, a in zip(self._lsf_sizes, self._lsf_anchored, strict=True) if a)
                if h3.shape != (n_h3,):
                    raise ValueError(
                        f"lsf_h3 must have {n_h3} entries — one per LSF anchor of an "
                        f"anchored instrument, skipping un-anchored instruments "
                        f"(instruments {self.instruments}, sizes {self._lsf_sizes}, "
                        f"anchored {self._lsf_anchored}); got shape {h3.shape}"
                    )
                # Clip into the truncated-series-valid range; the model's
                # lsf_h3_bound guard rejects outside it.
                h3 = jnp.clip(h3, -_H3_MAX, _H3_MAX)
                h3_parts, start = {}, 0
                for name, n_a, anchored in zip(
                    self.instruments, self._lsf_sizes, self._lsf_anchored, strict=True
                ):
                    if anchored:
                        h3_parts[name] = h3[start : start + n_a]
                        start += n_a
            problem = with_lsf(problem, parts, h3_parts)
        if "response" in theta:
            problem = with_response(problem, jnp.asarray(theta["response"]))
        if "log_nebular_amp" in theta:
            if not problem.nebular:
                raise ValueError(
                    "theta carries a log_nebular_amp site but the model was built "
                    "without nebular=True. The nebular component changes the size of "
                    "the linear system and the length of the spectral prior, so it "
                    "cannot be switched on by a θ site — rebuild the "
                    "MarginalOrbitModel with nebular=True (and one more (tau, eta))."
                )
            problem = with_nebular_amplitudes(problem, nebular_amplitudes(theta))
        if "log_jitter" in theta:
            problem = with_jitter(problem, jnp.exp(jnp.asarray(theta["log_jitter"])))
        if "ar1_phi" in theta:
            if not self.ar1:
                raise ValueError(
                    "theta carries an ar1_phi site but the model was built without "
                    "ar1=True — the AR coupling's bandwidth was not reserved, so the "
                    "probed marginal would be silently wrong. Rebuild the "
                    "MarginalOrbitModel with ar1=True."
                )
            problem = with_ar1(problem, jnp.asarray(theta["ar1_phi"]))
        return problem

    def problem_at(self, theta: Mapping):
        """The :class:`albireo.forward.Problem` at ``theta`` — data, weights and operators.

        What you need to look at residuals, which on real data is not optional: the
        inverse variances of an archival spectrum are usually estimated rather than
        measured (:func:`albireo.preprocess.estimate_ivar`), and a factor-of-two error
        there rescales every uncertainty the run reports. Run this first, without a
        ``log_jitter`` site, to *measure* the discrepancy; only then decide whether to
        let a jitter absorb it (:func:`albireo.forward.with_jitter` explains when that is
        legitimate and when it merely widens a biased answer).

        Parameters
        ----------
        theta
            Parameter dict, as passed to :meth:`log_likelihood`.

        Returns
        -------
        Problem
            The problem with ``theta``'s velocities (and light fractions, LSF widths or
            jitter, if those sites are present) substituted in.

        Examples
        --------
        >>> from albireo.forward import data_residual_zscores  # doctest: +SKIP
        >>> z = data_residual_zscores(model.problem_at(theta), model.marginal(theta).d_hat)
        >>> z.std()  # 1.0 if the inverse variances are calibrated  # doctest: +SKIP
        """
        return self._theta_problem(theta)

    def _marginal_from_problem(self, problem, theta: Mapping) -> MarginalResult:
        return marginal_loglikelihood(
            problem,
            self._prior(theta),
            block_size=self.block_size,
            half_bandwidth=self.half_bandwidth,
        )

    def _marginal_at(self, base, theta: Mapping) -> MarginalResult:
        return self._marginal_from_problem(self._theta_problem(theta, base=base), theta)

    def _marginal(self, theta: Mapping) -> MarginalResult:
        """Un-jitted marginal at θ (for gradient composition and tests)."""
        return self._marginal_at(self.problem, theta)

    def marginal(self, theta: Mapping) -> MarginalResult:
        """Jit-compiled marginal result (log-likelihood + conditional spectra) at θ."""
        return self._marginal_jit(self.problem, dict(theta))

    def log_likelihood(self, theta: Mapping):
        """Jit-compiled marginal log-likelihood at θ (differentiable)."""
        return self.marginal(theta).log_likelihood

    def _sweep_at(self, base, theta: Mapping, sweep: Mapping, batch_size: int):
        def one(sw):
            return self._marginal_at(base, {**theta, **sw}).log_likelihood

        return jax.lax.map(one, sweep, batch_size=batch_size)

    def log_likelihood_sweep(
        self,
        theta: Mapping,
        sweep: Mapping,
        *,
        batch_size: int | None = None,
        problem=None,
    ):
        """Marginal log-likelihood over a grid of θ values — one graph, no host round-trip.

        A scan over trial parameters is not inference: nothing is being sampled, the
        points are independent, and the answer is one number each. Done as a Python loop
        over :meth:`log_likelihood` it still pays a device synchronization per point,
        and every point's linear algebra is dispatched alone; done here it is a single
        ``lax.map``, so the trials share one compiled graph and batch into the same
        kernels. That is what makes a 2-D ``(K_1, K_2)`` grid — and the thousands of
        scans an injection-recovery calibration runs on top of it
        (:mod:`albireo.calibrate`) — affordable rather than merely possible.

        Parameters
        ----------
        theta
            The fixed part of the parameter dict, exactly as :meth:`log_likelihood`
            takes it.
        sweep
            The varying part: a mapping from θ site name to an array whose *leading*
            axis is the trial axis. Every entry must share that leading length, and each
            trailing shape must be what the site takes at a single θ (so sweeping ``k``
            on a two-component model wants ``(n_trials, 2)``). Entries override ``theta``.
        batch_size
            Trials per vmapped batch. ``None`` (default) applies the size-adaptive
            policy of :func:`_sweep_batch_default`; 1 is a pure sequential scan (least
            memory); ``n_trials`` forces one wide batch (fastest, most memory).
        problem
            Alternative base :class:`~albireo.forward.Problem` — same structure, other
            numbers. The one use is a resimulated dataset
            (:func:`albireo.simulate.resimulate`), which is why it exists: it lets a
            bootstrap reuse this model's operators instead of rebuilding them per trial.

        Returns
        -------
        jax.Array
            ``(n_trials,)`` marginal log-likelihoods, in ``sweep`` order.

        Examples
        --------
        >>> k = jnp.stack([jnp.full_like(k2s, 60.0), k2s], axis=1)  # doctest: +SKIP
        >>> ll = model.log_likelihood_sweep(orbit, {"k": k})  # doctest: +SKIP
        """
        sweep = dict(sweep)
        if not sweep:
            raise ValueError("sweep is empty — pass at least one site to vary")
        unknown = [s for s in sweep if s not in _THETA_SITES]
        if unknown:
            raise ValueError(f"unknown sites in sweep: {unknown} (expected {_THETA_SITES})")
        sweep = {name: jnp.asarray(v) for name, v in sweep.items()}
        lengths = {name: v.shape[0] for name, v in sweep.items() if v.ndim > 0}
        if len(lengths) != len(sweep):
            flat = [n for n, v in sweep.items() if v.ndim == 0]
            raise ValueError(f"sweep entries need a leading trial axis; {flat} are scalars")
        if len(set(lengths.values())) != 1:
            raise ValueError(f"sweep entries disagree on the trial-axis length: {lengths}")
        n_trials = next(iter(lengths.values()))
        if n_trials == 0:
            raise ValueError("sweep has no trials")
        if batch_size is None:
            batch_size = _sweep_batch_default(
                n_trials, self.problem.n_components * self.problem.grid.n, self.half_bandwidth
            )
        batch_size = max(1, min(int(batch_size), n_trials))
        base = self.problem if problem is None else problem
        return self._sweep_jit(base, dict(theta), sweep, batch_size)

    def model(self, priors: Mapping[str, dist.Distribution], *, fixed: Mapping | None = None):
        """Build a numpyro model: sample ``priors``, add the marginal likelihood.

        Parameters
        ----------
        priors
            Distribution per sampled site (``period``, ``t_conj``, ``secosw``,
            ``sesinw``, ``k``; optionally ``log_tau``/``log_eta``). The ``k`` site is
            vector-valued: use a distribution with batch shape ``(n_stellar,)``.
        fixed
            Values injected as constants instead of sampled — the empirical-Bayes
            route: fix ``log_tau``/``log_eta`` at their :func:`run_map` values for the
            NUTS run. Keys must not also appear in ``priors``.

        Returns
        -------
        callable
            A numpyro model for :func:`run_map` / :func:`run_nuts`. Records ``ecc``
            and ``omega`` as deterministic sites. The model takes the base
            :class:`~albireo.forward.Problem` as an optional argument and advertises
            it via a ``model_args`` attribute, and the runners pass it through
            numpyro as a *traced* jit argument — the same contract as
            :meth:`marginal` (D27): captured as a closure constant instead, the
            problem's arrays are baked into the jitted potential as XLA constants,
            whose compile-time folding allocates multi-GB temporaries at survey
            scale. Calling the model with no argument (any plain numpyro utility,
            e.g. ``log_density``) falls back to the closure — correct, just not
            compile-safe at scale.
        """
        unknown = [s for s in priors if s not in _THETA_SITES]
        if unknown:
            raise ValueError(f"unknown sites in priors: {unknown} (expected {_THETA_SITES})")
        fixed = dict(fixed or {})
        overlap = set(fixed) & set(priors)
        if overlap:
            raise ValueError(f"sites both fixed and sampled: {sorted(overlap)}")

        free_velocity = "velocity" in priors or "velocity" in fixed
        if free_velocity:
            keplerian = [s for s in ("period", "t_conj", "secosw", "sesinw", "k", *_OUTER_SITES)]
            clash = [s for s in keplerian if s in priors or s in fixed]
            if clash:
                raise ValueError(
                    f"a free-velocity model cannot also carry Keplerian sites {clash} — "
                    "'velocity' replaces the orbit rather than supplementing it"
                )
        elif not all(s in priors or s in fixed for s in ("period", "t_conj", "secosw", "sesinw")):
            missing = [
                s
                for s in ("period", "t_conj", "secosw", "sesinw")
                if s not in priors and s not in fixed
            ]
            raise ValueError(
                f"missing orbital sites {missing}. Sample or fix them, or build a "
                "free-velocity model instead by giving a 'velocity' site (D42)."
            )

        def _model(base=None):
            theta = {name: numpyro.sample(name, d) for name, d in priors.items()}
            theta.update({name: jnp.asarray(v) for name, v in fixed.items()})
            if free_velocity:
                # No Keplerian: record the identified table rather than the raw site,
                # whose per-component zero points the likelihood cannot see (D42).
                numpyro.deterministic(
                    "velocity_rel", relative_velocities(theta["velocity"], self.problem.grid)
                )
            else:
                ecc_raw = theta["secosw"] ** 2 + theta["sesinw"] ** 2
                numpyro.deterministic("ecc", jnp.minimum(ecc_raw, self.ecc_max))
                numpyro.deterministic("omega", jnp.arctan2(theta["sesinw"], theta["secosw"]))
                numpyro.factor("ecc_disk", jnp.where(ecc_raw <= self.ecc_max, 0.0, -jnp.inf))
            if _has_outer_orbit(theta):
                ecc_out_raw = theta["secosw_out"] ** 2 + theta["sesinw_out"] ** 2
                numpyro.deterministic("ecc_out", jnp.minimum(ecc_out_raw, self.ecc_max))
                numpyro.deterministic(
                    "omega_out", jnp.arctan2(theta["sesinw_out"], theta["secosw_out"])
                )
                numpyro.factor(
                    "ecc_disk_out", jnp.where(ecc_out_raw <= self.ecc_max, 0.0, -jnp.inf)
                )
            if "lsf_sigma" in theta:
                sig = jnp.atleast_1d(theta["lsf_sigma"])
                # The construction-time widths fix the kernel radii: wider LSFs would
                # be silently truncated, so they are rejected (see with_lsf).
                numpyro.factor(
                    "lsf_bound",
                    jnp.where(jnp.all(sig <= self._lsf_sigma_max), 0.0, -jnp.inf),
                )
            if "lsf_h3" in theta:
                h3 = jnp.atleast_1d(theta["lsf_h3"])
                # Beyond the bound the truncated Gauss-Hermite series is no longer a
                # credible line-spread profile (see _H3_MAX); clipped + rejected.
                numpyro.factor(
                    "lsf_h3_bound",
                    jnp.where(jnp.all(jnp.abs(h3) <= _H3_MAX), 0.0, -jnp.inf),
                )
            if "ar1_phi" in theta:
                phi = jnp.atleast_1d(theta["ar1_phi"])
                # with_ar1 clips at +-0.999 so the likelihood stays finite (and
                # rejectable, via this factor) outside the stationary region.
                numpyro.factor("ar1_bound", jnp.where(jnp.all(jnp.abs(phi) < 1.0), 0.0, -jnp.inf))
            if "log_nebular_amp" in theta:
                # Record the amplitudes the model actually applied: the site itself is
                # only identified up to an additive constant (nebular_amplitudes
                # centers it), so reading the raw samples would mislead.
                numpyro.deterministic("nebular_amp", nebular_amplitudes(theta))
            problem = self._theta_problem(theta, base=base)
            # Reject configurations whose relative shifts exceed the static bandwidth
            # (the probed marginal likelihood would be silently wrong out there).
            rel = _max_relative_shift(problem)
            numpyro.factor("bandwidth_guard", jnp.where(rel <= self._shift_bound, 0.0, -jnp.inf))
            numpyro.factor(
                "marginal_loglike", self._marginal_from_problem(problem, theta).log_likelihood
            )

        # Advertise the problem as a model argument (picked up by run_map /
        # laplace_inverse_mass / run_nuts) so numpyro traces it instead of folding it.
        _model.model_args = (self.problem,)  # type: ignore[attr-defined]
        return _model


def _resolve_model_args(model, model_args) -> tuple:
    """The model's traced arguments: explicit ``model_args`` wins, else the model's
    own ``model_args`` attribute (:meth:`MarginalOrbitModel.model`), else none."""
    if model_args is None:
        model_args = getattr(model, "model_args", ())
    return tuple(model_args)


@dataclass(frozen=True)
class MAPResult:
    """Result of :func:`run_map`.

    ``potential`` and ``grad_norm`` are evaluated at the convergence-check point,
    which (by the check-then-step loop) is one accepted L-BFGS step *behind*
    ``params`` — irrelevant at convergence, stated for exactness.
    """

    params: dict  # constrained values of all sites, including deterministics
    unconstrained: dict
    potential: float  # potential energy (-log joint, up to constants)
    grad_norm: float  # unconstrained-space gradient norm
    converged: bool
    num_steps: int


def run_map(
    model,
    *,
    init: Mapping,
    rng_key=None,
    max_steps: int = 200,
    tol: float = 1e-2,
    callback=None,
    model_args: tuple | None = None,
) -> MAPResult:
    """MAP over all sampled sites of ``model`` via L-BFGS on numpyro's potential.

    Runs in numpyro's unconstrained space (so constrained priors are handled by the
    standard transforms) with a zoom linesearch; with ``log_tau``/``log_eta`` among
    the sampled sites this is the ML-II / empirical-Bayes hyperparameter fit, since
    the spectra are already marginalized out of the likelihood.

    Parameters
    ----------
    model
        A numpyro model (from :meth:`MarginalOrbitModel.model`).
    init
        Constrained initial values for every sampled site. Start circular orbits at
        small nonzero ``(secosw, sesinw)`` — the origin is the one non-smooth point.
    tol
        Convergence threshold on the unconstrained-space gradient norm. Note that this is
        an *absolute* threshold on a potential whose scale grows with the number of good
        pixels: on a survey-sized dataset the gradient norm at the true parameters is
        already in the hundreds, so the default is unreachable and ``converged`` will be
        ``False`` however good the fit is. Watch the parameters, via ``callback``, rather
        than trusting the flag.
    callback
        Called after every accepted step as ``callback(step, potential, grad_norm, params)``
        with ``params`` the *constrained* site values. Without it this function is silent
        for however long it runs, and a real-data fit can run for hours — a first MAP on a
        new dataset should always pass one. Return ``True`` to stop early.
    model_args
        Positional arguments for ``model``, passed through numpyro as *traced* jit
        arguments rather than closure constants (which XLA constant-folds — the
        multi-GB-at-scale trap D27 documents). Default: the model's own
        ``model_args`` attribute when it has one (:meth:`MarginalOrbitModel.model`
        advertises its base problem there), else ``()``. Pass ``()`` explicitly to
        force the closure path.

    Examples
    --------
    >>> def log(step, potential, grad_norm, params):  # doctest: +SKIP
    ...     print(f"{step:4d}  {potential:.3f}  |g|={grad_norm:.3g}  K={params['k']}")
    """
    rng_key = jax.random.PRNGKey(0) if rng_key is None else rng_key
    model_args = _resolve_model_args(model, model_args)
    model_info = initialize_model(
        rng_key,
        model,
        init_strategy=init_to_value(values=dict(init)),
        dynamic_args=True,
        model_args=model_args,
    )
    potential_gen = model_info.potential_fn  # potential_gen(*model_args) -> z -> potential
    postprocess = model_info.postprocess_fn(*model_args)
    opt = optax.lbfgs()

    @jax.jit
    def step(params, state, args):
        potential = potential_gen(*args)
        value, grad = optax.value_and_grad_from_state(potential)(params, state=state)
        updates, state = opt.update(grad, state, params, value=value, grad=grad, value_fn=potential)
        params = optax.apply_updates(params, updates)
        return params, state, value, grad

    # numpyro may return python-float leaves for scalar sites; optax needs arrays
    params = jax.tree.map(jnp.asarray, model_info.param_info.z)
    state = opt.init(params)
    value, grad_norm, steps_taken = np.inf, np.inf, 0
    for steps_taken in range(1, max_steps + 1):
        params, state, value, grad = step(params, state, model_args)
        grad_norm = float(optax.tree.norm(grad))
        if not np.isfinite(grad_norm):
            raise FloatingPointError(
                f"non-finite gradient at L-BFGS step {steps_taken} (potential {float(value)})"
            )
        if callback is not None and callback(
            steps_taken, float(value), grad_norm, postprocess(params)
        ):
            break
        if grad_norm < tol:
            break
    constrained = postprocess(params)
    return MAPResult(
        params={k: v for k, v in constrained.items()},
        unconstrained=dict(params),
        potential=float(value),
        grad_norm=grad_norm,
        converged=grad_norm < tol,
        num_steps=steps_taken,
    )


def laplace_inverse_mass(
    model, params: Mapping, *, rng_key=None, floor: float = 1e-10, model_args: tuple | None = None
):
    """Unconstrained-space Laplace covariance at ``params`` — a NUTS starting mass matrix.

    Evaluates the Hessian of the model potential at the given constrained site values
    (typically :attr:`MAPResult.params`; extra keys are ignored), symmetrizes, floors
    the eigenvalues at ``floor * max_eig``, and returns the inverse as a dense array.
    Pass it as ``inverse_mass_matrix`` to :func:`run_nuts` built from the *same*
    model: with the mass matrix pre-set to (approximately) the posterior covariance,
    warmup only tunes the step size — without it, parameter scales spanning many
    orders of magnitude drive early trajectories to the tree-depth cap and warmup
    costs more than sampling. ``model_args`` follows the :func:`run_map` contract
    (default: the model's own ``model_args`` attribute).
    """
    rng_key = jax.random.PRNGKey(0) if rng_key is None else rng_key
    model_args = _resolve_model_args(model, model_args)
    model_info = initialize_model(
        rng_key,
        model,
        init_strategy=init_to_value(values=dict(params)),
        dynamic_args=True,
        model_args=model_args,
    )
    potential = model_info.potential_fn(*model_args)
    z = jax.tree.map(jnp.asarray, model_info.param_info.z)
    flat, unravel = ravel_pytree(z)
    # Reverse-over-reverse, NOT jax.hessian (= forward-over-reverse). Forward-over-
    # reverse runs here — it differentiates the custom rule's *backward* pass, which is
    # plain operations — but it was measured to return an appreciably *asymmetric*
    # Hessian on this stack, and that on the plain-autodiff path too, so the cause is
    # the solver scans rather than the custom VJP. jacrev(jacrev(...)) matches central
    # finite differences of the gradient to 8 digits where forward-over-reverse does
    # not (D28). Applying forward mode *directly* to the marginal is separately
    # impossible: JAX rejects jvp of a custom_vjp function.
    hess = jax.jacrev(jax.jacrev(lambda zf: potential(unravel(zf))))(flat)
    hess = 0.5 * (hess + hess.T)
    eigval, eigvec = jnp.linalg.eigh(hess)
    eigval = jnp.maximum(eigval, floor * jnp.max(eigval))
    return np.asarray(eigvec @ jnp.diag(1.0 / eigval) @ eigvec.T)


def run_nuts(
    model,
    *,
    rng_key,
    init: Mapping | None = None,
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 2,
    target_accept: float = 0.9,
    dense_mass: bool = True,
    inverse_mass_matrix=None,
    adapt_mass_matrix: bool | None = None,
    max_tree_depth: int = 8,
    progress_bar: bool = False,
    model_args: tuple | None = None,
) -> MCMC:
    """NUTS over the sampled sites of ``model`` (spectra stay marginalized).

    ``init`` should be the :attr:`MAPResult.params` dict (extra keys are ignored);
    ``dense_mass=True`` is the right default for the low-dimensional, correlated
    orbital posterior, and ``inverse_mass_matrix`` from :func:`laplace_inverse_mass`
    makes warmup cheap. When an explicit mass matrix is supplied, mass adaptation
    defaults to *off* — warmup's early adaptation windows would overwrite the Laplace
    matrix with a poor few-sample estimate and give back the slow, deep-tree warmup
    the matrix was meant to avoid (override via ``adapt_mass_matrix=True``). Returns
    the numpyro ``MCMC`` object (``.get_samples()``, ``.print_summary()``);
    divergences and tree depths are collected as extra fields.

    ``model_args`` follows the :func:`run_map` contract (default: the model's own
    ``model_args`` attribute); with arguments present the MCMC runs with
    ``jit_model_args=True``, so they are traced through the jitted sample loop
    rather than baked into it as XLA constants.
    """
    if adapt_mass_matrix is None:
        adapt_mass_matrix = inverse_mass_matrix is None
    model_args = _resolve_model_args(model, model_args)
    strategy = init_to_value(values=dict(init)) if init is not None else init_to_value()
    kernel = NUTS(
        model,
        init_strategy=strategy,
        target_accept_prob=target_accept,
        dense_mass=dense_mass,
        inverse_mass_matrix=(
            jnp.asarray(inverse_mass_matrix) if inverse_mass_matrix is not None else None
        ),
        adapt_mass_matrix=adapt_mass_matrix,
        max_tree_depth=max_tree_depth,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="sequential",
        progress_bar=progress_bar,
        jit_model_args=bool(model_args),
    )
    mcmc.run(rng_key, *model_args, extra_fields=("num_steps", "diverging"))
    return mcmc


def posterior_spectra(
    model: MarginalOrbitModel,
    samples: Mapping,
    key,
    *,
    num_draws: int = 32,
    extra: Mapping | None = None,
):
    """Spectra draws from the *joint* posterior, shape ``(num_draws, n_comp, n_pix)``.

    Each draw picks a posterior ``theta`` sample at random and draws once from the
    conditional Gaussian over the spectra — so the returned scatter includes both the
    conditional spectral uncertainty and the orbital uncertainty. ``extra`` supplies
    sites missing from ``samples`` (e.g. ``log_tau``/``log_eta`` when they were fixed
    during sampling).
    """
    extra = {name: jnp.asarray(v) for name, v in (extra or {}).items()}
    site_names = [s for s in _THETA_SITES if s in samples and s not in extra]
    n_samples = np.asarray(samples[site_names[0]]).shape[0]
    key_idx, key_draw = jax.random.split(jnp.asarray(key))
    idx = np.asarray(jax.random.randint(key_idx, (num_draws,), 0, n_samples))
    out = []
    for j, i in enumerate(idx):
        theta = {name: jnp.asarray(samples[name])[i] for name in site_names}
        theta.update(extra)
        result = model.marginal(theta)
        out.append(draw_spectra(result, jax.random.fold_in(key_draw, j), 1)[0])
    return jnp.stack(out)

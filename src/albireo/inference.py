"""Joint Bayesian inference of the orbit with the spectra marginalized (M3).

Implements ``docs/math.md`` §7. The nonlinear parameter vector ``theta`` is a dict of
JAX arrays; conditional on ``theta`` the component spectra are marginalized analytically
(:mod:`albireo.likelihood`), so inference runs over ``theta`` alone. The sites of a
Keplerian model are

- ``period``: orbital period [d].
- ``t_conj``: time of conjunction of component 1 (``nu + omega = pi/2``) [d].
- ``secosw``, ``sesinw``: ``sqrt(e) cos(omega)`` and ``sqrt(e) sin(omega)``
  [dimensionless].
- ``k``: RV semi-amplitudes ``(K_1, K_2, ...)`` [km/s], one per inner stellar
  component; even-indexed components use ``omega``, odd-indexed ``omega + pi``.

The optional sites are the following. The instrumental, continuum and noise sites among
them are the realism extensions of ``docs/math.md`` §7.5.

- ``log_tau``, ``log_eta``: log spectral-prior hyperparameters [dimensionless], one per
  model component, including the telluric and nebular components when enabled; both
  sites or neither.
- ``period_out`` [d], ``t_conj_out`` [d], ``secosw_out``, ``sesinw_out``
  [dimensionless], ``k_out`` [km/s]: a hierarchical outer orbit (SB3), all five
  together. The inner components' center of mass moves with semi-amplitude ``k_out[0]``
  and argument ``omega_out``; one tertiary component is appended, moving with
  ``k_out[1]`` and ``omega_out + pi``.
- ``velocity``: free per-epoch radial velocities [km/s], ``(n_stellar, n_epochs)``,
  replacing the Keplerian sites entirely (D42; ``docs/math.md`` §7.6). With this site
  present the orbital sites must be absent, and each epoch's velocity is its own
  parameter. Each component's zero point is removed before use, in pixel space, where
  the removal is exact; an uncentered table would have its absolute level set by
  shift-interpolation error rather than by the data. The identified table is the
  ``velocity_rel`` deterministic (:func:`relative_velocities`), and
  :func:`keplerian_residuals` is the model check this mode supports.
- ``light``: stellar light fractions [dimensionless], ``(n_stellar,)`` constant or
  ``(n_epochs, n_stellar)`` per epoch, rows on the simplex (Dirichlet priors).
- ``lsf_sigma``: Gaussian LSF widths [km/s], one entry per LSF anchor for instruments
  built with ``lsf_anchors_angstrom`` (wavelength-dependent LSF, D37) and one entry per
  un-anchored instrument, concatenated in instrument order. The construction-time
  ``lsf_sigma_v`` values are per-entry upper bounds, since they fix the kernel radii.
- ``lsf_h3``: Gauss-Hermite LSF skewness [dimensionless] (D38), one entry per LSF
  anchor of each anchored instrument, concatenated in instrument order, ``|h3| <= 0.2``.
  Un-anchored instruments have no entry: a stationary asymmetric LSF is absorbed by the
  free spectra (``docs/math.md`` §1.3). May appear with or without ``lsf_sigma``;
  without it the widths stay at the build values.
- ``response``: multiplicative per-epoch Chebyshev response coefficients
  [dimensionless], ``(n_coef,)`` shared or ``(n_epochs, n_coef)`` per epoch
  (:func:`albireo.forward.with_response`, D7/D33). ``r = 1 + sum_m c_m T_m``, so
  all-zero coefficients give the unit response. This is the per-epoch continuum
  treatment: a low-order response trades against the components' broad features
  (``internal/design.md`` §5), so the order should be low and the priors tight and
  zero-centered. When the site is present the construction-time ``response_coeffs``
  are replaced.
- ``log_jitter``: log noise-inflation factor [dimensionless], scalar (shared) or one
  per epoch; the weights become ``w_j / exp(2 log_jitter_j)``
  (:func:`albireo.forward.with_jitter`, D15). A jitter fitted against systematics
  widens the uncertainties around a biased point estimate; see that function's notes.
- ``ar1_phi``: AR(1) correlation of the standardized noise [dimensionless], scalar
  (shared) or one per epoch, ``|phi| < 1`` (:func:`albireo.forward.with_ar1`, D34).
  Requires ``ar1=True`` at construction, because the correlated coupling widens the
  static solver bandwidth; the marginal stays on the band assembly path (D35). Composes
  with ``log_jitter``: the jitter scales the noise, ``phi`` correlates it.
- ``log_nebular_amp``: log per-epoch amplitude of the nebular component
  [dimensionless], ``(n_epochs,)`` (D40; requires ``nebular=True`` at construction).
  The site is centered before use, ``a_j = exp(u_j - mean(u))``, because only the
  products ``a_j d_neb`` are observable and the overall scale is degenerate with the
  component spectrum. A prior on this site is therefore a prior on the epoch-to-epoch
  variation (a zero-mean Normal with sigma 0.2 allows roughly 20% night to night), and
  shifting every entry by a constant has no effect. The common mode is an exactly flat
  direction of the likelihood, held only by the site's prior; the posterior stays
  proper and well conditioned (curvature ``1/sigma^2`` along it), but
  ``mean(log_nebular_amp)`` in the samples reproduces the prior. The applied
  amplitudes are the ``nebular_amp`` deterministic (:func:`nebular_amplitudes`).

``gamma`` is identically zero (D14): a systemic velocity is exactly degenerate with a
common shift of the component spectra. The ``(secosw, sesinw)`` parameterization is
smooth through ``e = 0``, where ``omega`` and a time of periastron are undefined, and
maps a uniform prior on the unit disk to a uniform prior on ``e`` (``docs/math.md``
§7.2). The disk constraint ``e < 1`` enters the model as a ``-inf`` factor, with ``e``
clipped to ``ecc_max`` before the Kepler solve so that the likelihood stays finite, and
rejectable, outside it. The map is non-differentiable only at ``secosw = sesinw = 0``;
circular orbits should be initialized slightly off the origin.

Inference proceeds in three stages (``docs/math.md`` §7.1-7.3):

1. :func:`run_map` maximizes the marginal posterior over ``theta`` by L-BFGS in
   numpyro's unconstrained space. With ``log_tau``/``log_eta`` among the sampled sites
   this is the ML-II (empirical Bayes) hyperparameter fit, since the marginal likelihood
   is already integrated over the spectra (§7.3). The prior scales control the part of
   spectrum space that the data cannot constrain below the LSF scale (§5.1), so by
   default they are estimated from the data in this stage rather than fixed a priori.
2. :func:`laplace_inverse_mass` evaluates the Hessian of the potential at the MAP and
   returns its inverse as the NUTS mass matrix.
3. :func:`run_nuts` samples ``theta`` with the No-U-Turn Sampler (Hoffman & Gelman 2014;
   Betancourt 2017) as implemented in numpyro (Phan et al. 2019), with the
   hyperparameters either held at their ML-II values (passed through ``fixed=``) or
   sampled.

:func:`posterior_spectra` then draws spectra from the joint posterior (§7.4), and the
free-velocity utilities (:func:`relative_velocities`, :func:`relative_velocity_errors`,
:func:`keplerian_residuals`) implement §7.6.

References
----------
Betancourt, M. 2017, arXiv:1701.02434
Hoffman, M. D. & Gelman, A. 2014, Journal of Machine Learning Research, 15, 1593
Phan, D., Pradhan, N. & Jankowiak, M. 2019, arXiv:1912.11554
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
# assembly.py's gradient budget, and the budget can be considerably larger.
_SWEEP_BATCH_BYTES = 1 << 30
# Gauss-Hermite skewness bound: beyond about 0.2 the truncated series dips measurably
# negative in the tail, and real instrument profiles lie well below it (D38).
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

    Same two-regime rule as :func:`albireo.assembly._epoch_chunk_default`: the whole
    sweep runs as one batch while it stays under ``_SWEEP_BATCH_BYTES``, otherwise the
    batch is cut to fit. The estimate counts the ``(n, bandwidth)`` float64 arrays a
    marginal solve keeps live; the band tensor and the block-tridiagonal factor
    dominate, and the smaller arrays are absorbed into the constant.
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

    The centering is the identifiability convention of the free-velocity table (D42;
    ``docs/math.md`` §7.6). It is done in pixel space because it is exact there:
    ``xi = artanh(v/c)`` turns relativistic velocity addition into ordinary addition, so
    subtracting a constant pixel shift is exactly a translation of the component's
    spectrum, whereas subtracting a constant velocity is only a first-order
    approximation of one.
    """
    pix = grid.velocity_to_pixels(jnp.atleast_2d(jnp.asarray(velocity)))
    return pix - jnp.mean(pix, axis=1, keepdims=True)


def relative_velocities(velocity, grid):
    """The identified part of a free-velocity table: per-component velocities, zero-pointed.

    A table of free per-epoch velocities has one arbitrary zero point per stellar
    component, not one in total. Each component's deviation spectrum is a free vector,
    so translating it absorbs a constant added to that component's shifts and the
    likelihood is unchanged. This generalizes ``gamma = 0`` (D14): with no Keplerian
    tying the components together, each component has its own zero point
    (``docs/math.md`` §7.6).

    The degeneracy is exact only for whole-pixel translations, because the model shifts
    spectra by linear interpolation and a fractional shift blurs slightly as well as
    translating. Measured on a 10-epoch SB2 at SNR 200, a one-pixel common shift of one
    component changes the log-likelihood by 4e-9 in relative terms (boundary effects
    only), while a 0.1-pixel shift costs 7.3 nats. An uncentered table would therefore
    have its absolute zero point set by interpolation error: a number that resembles a
    systemic velocity, changes when the model grid is resampled, and carries no
    information. The zero points are removed instead and the remainder is reported.

    The remainder is fully identified: each component's velocity variation (hence its
    semi-amplitude), the epoch-to-epoch differences, and the slope of component 1
    against component 2 (the Wilson mass ratio), which is a slope and therefore
    independent of both zero points. The systemic velocity and the absolute velocity of
    either star are not recoverable from this table; they are measured afterwards from
    the disentangled spectra, as D14 prescribes for ``gamma``.

    Parameters
    ----------
    velocity
        Free per-epoch velocities [km/s], ``(n_stellar, n_epochs)``: the raw
        ``velocity`` theta site, or a posterior sample of it.
    grid
        The model :class:`~albireo.grids.LogGrid` on which the shifts are taken. The
        centering is grid-dependent by construction, since it is done in pixel space.

    Returns
    -------
    jax.Array
        ``(n_stellar, n_epochs)`` velocities [km/s], each row relativistically
        zero-pointed to that component's mean epoch. Rows sum to zero in pixel space and
        therefore not exactly to zero in km/s.
    """
    return grid.pixels_to_velocity(_centered_shifts(velocity, grid))


def relative_velocity_errors(covariance, unconstrained: Mapping, *, site: str = "velocity"):
    """Per-epoch standard errors of the free-velocity table, zero points projected out.

    The diagonal of the Laplace covariance is not a usable error bar for this site. Each
    component's zero point is an exactly flat direction of the likelihood
    (:func:`relative_velocities`), so its posterior width equals the prior width, and
    every epoch's marginal variance inherits it. Measured on the D42 fixture with a
    ``Normal(0, 120)`` prior over 10 epochs, every raw marginal sigma is
    37.95 km/s = 120/sqrt(10), identical to four digits across both components and all
    epochs, while the identified per-epoch error is 0.059 km/s: a factor of 640, and
    insensitive to the data, so the same value would appear on a good dataset and on an
    uninformative one.

    Projecting each component's mean out of the covariance removes exactly those
    directions and leaves the identified ones; as a check on the count, the projected
    block has exactly ``n_stellar`` zero eigenvalues.

    The projection is applied in velocity units, where it equals the pixel-space
    centering to ``O(v^2/c^2)``: the Jacobian of the pixel-space centering differs from
    unity by about ``1e-8`` at stellar velocities, far below the Gaussian approximation
    already made by using a Laplace covariance.

    Posterior samples are preferable when available. The ``velocity_rel`` deterministic
    recorded by :meth:`MarginalOrbitModel.model` is already the identified table, so
    ``samples["velocity_rel"].std(axis=0)`` needs neither a projection nor a Gaussian
    assumption. This function serves the MAP-plus-Laplace route.

    Parameters
    ----------
    covariance
        Full unconstrained-space covariance from :func:`laplace_inverse_mass`.
    unconstrained
        :attr:`MAPResult.unconstrained` from the same fit; it supplies both the site
        ordering within the flattened vector and the table's shape.
    site
        Name of the free-velocity site.

    Returns
    -------
    numpy.ndarray
        ``(n_stellar, n_epochs)`` standard errors [km/s].

    Raises
    ------
    ValueError
        If ``site`` is absent from the fit, is not two-dimensional, or the covariance
        does not match the fit's parameter count.

    Notes
    -----
    A Laplace covariance is a local Gaussian approximation with the hyperparameters held
    at their MAP values, so it omits the widening that marginalizing over
    ``log_tau``/``log_eta`` would add. On the D42 fixture the resulting errors are about
    1.4x optimistic against the realized errors. They are a fast estimate; NUTS gives
    the posterior.
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
            "parameters: they must come from the same model"
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

    This is the model check the free-velocity mode supports (``docs/math.md`` §7.6):
    fit per-epoch velocities with no orbit imposed, then test whether a Keplerian passes
    through them. A slightly wrong period, an unmodelled third body, or line-profile
    variability that the Keplerian absorbs into ``e`` appear as structured residuals
    (phase-correlated, or one epoch far out) where noise alone would not.

    Both tables are zero-pointed the same way before subtraction, and the subtraction is
    done in pixel space so that the result is an exact relativistic velocity difference
    rather than a first-order one. The two tables' arbitrary zero points
    (:func:`relative_velocities`) then cancel exactly; otherwise the residual would
    carry a constant offset with no meaning.

    Parameters
    ----------
    velocity
        Free per-epoch velocities [km/s], ``(n_stellar, n_epochs)``: the ``velocity``
        site from a fit, or one posterior sample of it.
    theta
        Keplerian parameters (``period``, ``t_conj``, ``secosw``, ``sesinw``, ``k``, and
        the outer-orbit sites if any), as :func:`orbit_velocities` takes them.
    bjd
        Epoch times [d], matching the columns of ``velocity``.
    grid
        The model :class:`~albireo.grids.LogGrid`.
    ecc_max
        Eccentricity clip, matching the model the Keplerian came from.

    Returns
    -------
    jax.Array
        ``(n_stellar, n_epochs)`` residuals [km/s]. They are to be compared against the
        per-epoch uncertainties of the free table, not against zero.

    Raises
    ------
    ValueError
        If the free table and the Keplerian disagree on the component or epoch count.
    """
    free = _centered_shifts(velocity, grid)
    kep = _centered_shifts(orbit_velocities(theta, bjd, ecc_max=ecc_max), grid)
    if free.shape != kep.shape:
        raise ValueError(
            f"the free table is {free.shape} but the Keplerian implies {kep.shape}: "
            "they must agree on both the component count and the epoch count"
        )
    return grid.pixels_to_velocity(free - kep)


def nebular_amplitudes(theta: Mapping):
    """Per-epoch nebular amplitudes from ``theta['log_nebular_amp']`` (differentiable).

    ``a_j = exp(u_j - mean(u))``: the geometric mean is pinned to 1 by convention. The
    model sees only the products ``a_j d_neb``, so without a pinned scale the pair
    ``(c a_j, d_neb / c)`` gives the same fit for every ``c > 0``, and only the spectral
    prior separates them. That direction is nearly flat and unbounded in one coordinate,
    and it would dominate the sampler's step size. Centering removes it exactly, at the
    cost of the one degree of freedom that was never identified.

    Two consequences follow for reading a fit. The recovered ``d_neb`` is on the scale of
    a typical epoch, so its line strengths are comparable to injected or published values
    only up to that convention. The posterior for ``a`` describes relative variation: it
    supports the statement that epoch 7 is 1.4 times the typical epoch, but not a
    statement about the absolute strength of the nebular emission.

    Returns
    -------
    jax.Array
        ``(n_epochs,)`` positive amplitudes with geometric mean 1.
    """
    u = jnp.atleast_1d(jnp.asarray(theta["log_nebular_amp"]))
    return jnp.exp(u - jnp.mean(u))


class MarginalOrbitModel:
    """The marginal posterior over orbital parameters for one dataset.

    Bundles the static problem structure (rebin operators, kernels, weights, built once)
    with the ``theta``-dependent path (Kepler velocities, shifts, marginal likelihood) so
    that :meth:`log_likelihood` is a single jit-compiled, differentiable function of
    ``theta`` (``docs/math.md`` §7.1). The solver bandwidth is fixed by
    ``v_rel_max_kms`` (see :meth:`albireo.forward.Problem.half_bandwidth_bound`) so that
    the computation graph is static. The numpyro model rejects, with a ``-inf`` factor,
    any configuration whose realized relative shifts exceed that budget, so a prior
    wider than ``v_rel_max_kms`` slows mixing near the bound but cannot corrupt the
    result. The direct :meth:`log_likelihood` entry point has no such guard; explicit
    calls must stay within the bound.

    Parameters
    ----------
    grid, dataset, light_fractions, lsf_sigma_v, lsf_anchors_angstrom, response_coeffs
        As in :func:`albireo.forward.build_problem`. ``light_fractions`` and
        ``lsf_sigma_v`` are the build-time values, used whenever ``theta`` carries no
        ``light`` / ``lsf_sigma`` site. When those sites are inferred, the build-time
        light fractions only set ``n_stellar``, and the build-time LSF widths become
        strict upper bounds: they fix the kernel radii, and the model rejects wider
        widths, which the fixed radii would otherwise truncate.
        ``lsf_anchors_angstrom`` makes an instrument's LSF wavelength-dependent (D37)
        and gives it one ``lsf_sigma`` entry per anchor rather than one in total.
        ``response_coeffs`` is the fixed response used whenever ``theta`` carries no
        ``response`` site, and is replaced when it does
        (:func:`albireo.forward.with_response`).
    telluric, nebular, nebular_v_kms
        Extra non-stellar components, as in :func:`albireo.forward.build_problem`. Each
        enabled component adds a trailing row to the recovered spectra and a trailing
        entry to ``prior`` (order: stellar, telluric, nebular), and must be accounted
        for in ``v_rel_max_kms``. ``nebular=True`` also enables the ``log_nebular_amp``
        site; without the site the amplitudes stay at 1 and the component is static.
    v_rel_max_kms
        Bound on the largest relative velocity between any two model components at any
        epoch [km/s]: for an SB2 ``(K_1 + K_2)(1 + e)``; for an SB3 add the outer
        orbit's ``(K_AB + K_C)(1 + e_out)``; plus the barycentric motion if a telluric
        component is enabled. The priors must not allow configurations that exceed it,
        or mixing stalls at the guard, so the value should include headroom.
    prior
        Fixed :class:`SmoothnessPrior`, used whenever ``theta`` carries no
        ``log_tau``/``log_eta`` sites. Optional if the hyperparameters are always in
        ``theta``, with one exception: its per-pixel profiles are kept even when the
        scalars are inferred (D40), because a profile is structure rather than a
        hyperparameter. A windowed component therefore needs its prior passed here even
        in a pure ML-II run.
    ecc_max
        Eccentricity clip and constraint (default 0.95, the Kepler solver's verified
        range).
    block_size
        Solver block size passed through to the marginal likelihood.
    ar1
        Allow an ``ar1_phi`` site (correlated noise, D34). The AR coupling widens
        ``A^T W A`` by a static amount (:attr:`albireo.forward.Problem.ar_bandwidth_extra`)
        that must be reserved in the solver bandwidth at construction, so the site is a
        construction-time choice like the bandwidth itself (D21). It costs a few pixels
        of bandwidth whether or not ``theta`` carries the site.
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
        # The shift budget inside half_bandwidth (inverse of half_bandwidth_bound). The
        # numpyro model rejects any configuration whose realized relative shifts exceed
        # it, so a prior wider than v_rel_max cannot corrupt the assembled band.
        # Computed from the base bandwidth: the AR extra below is reserved for the noise
        # coupling's reach and must not be spent on shifts.
        support = max(g.row_support for g in self.problem.groups)
        self._shift_bound = hb - 1 - 2 * self.problem.kernel_radius - support
        self.ar1 = bool(ar1)
        self.half_bandwidth = hb + (self.problem.ar_bandwidth_extra if self.ar1 else 0)
        self.block_size = block_size
        self.ecc_max = float(ecc_max)
        self.fixed_prior = prior
        # Instrument order for the optional "lsf_sigma" theta site; the widths the
        # kernels were built with are the upper bounds (they fix the kernel radii).
        # Deduplicated in first-seen order: one instrument owns several groups whenever
        # its epochs sit on different native grids (albireo.forward._epoch_groups), and
        # the LSF width is a property of the instrument, not of the grid. One sampled
        # width per group would mis-shape the site and, through dict(zip(...)), keep
        # only the last group's value.
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
        # The problem is passed as a jit argument (Problem is a registered pytree), so
        # its arrays enter the graph as runtime parameters. Capturing them as closure
        # constants instead triggers XLA constant folding that allocates tens of GB at
        # survey scale (D27).
        self._marginal_jit = jax.jit(self._marginal_at)
        self._sweep_jit = jax.jit(self._sweep_at, static_argnums=3)

    @property
    def n_stellar(self) -> int:
        return self.problem.n_stellar

    def _prior(self, theta: Mapping) -> SmoothnessPrior:
        if "log_tau" in theta or "log_eta" in theta:
            if "log_tau" not in theta or "log_eta" not in theta:
                raise ValueError("theta must carry both log_tau and log_eta, or neither")
            # The per-pixel profiles are structure, not hyperparameters (D40): they set
            # where a component may deviate from the continuum, the sampled scalars set
            # how much. An inferred (tau, eta) therefore replaces the scalars and keeps
            # the construction-time profiles; dropping them here would un-confine a
            # windowed component as soon as ML-II was switched on, without any error.
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
        """Problem at ``theta``: velocities always; the other sites when present."""
        base = self.problem if base is None else base
        if "velocity" in theta:
            # Free per-epoch velocities: no Keplerian (D42). The zero point is removed
            # here, in pixel space, where the removal is exact.
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
                    f"velocity must have shape ({self.n_stellar}, {base.n_epochs}); one "
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
                        f"lsf_sigma must have {self._lsf_sigma_max.shape[0]} entries: one per "
                        f"LSF anchor of an anchored instrument, one per un-anchored instrument "
                        f"(instruments {self.instruments}, sizes {self._lsf_sizes}); "
                        f"got shape {sig.shape}"
                    )
                # Clip into the range valid for the kernel radii so the likelihood stays
                # finite (and rejectable, through the model's lsf_bound guard) outside it.
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
                        f"lsf_h3 must have {n_h3} entries: one per LSF anchor of an "
                        f"anchored instrument, skipping un-anchored instruments "
                        f"(instruments {self.instruments}, sizes {self._lsf_sizes}, "
                        f"anchored {self._lsf_anchored}); got shape {h3.shape}"
                    )
                # Clip into the range where the truncated series is valid; the model's
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
                    "cannot be switched on by a θ site: rebuild the "
                    "MarginalOrbitModel with nebular=True (and one more (tau, eta))."
                )
            problem = with_nebular_amplitudes(problem, nebular_amplitudes(theta))
        if "log_jitter" in theta:
            problem = with_jitter(problem, jnp.exp(jnp.asarray(theta["log_jitter"])))
        if "ar1_phi" in theta:
            if not self.ar1:
                raise ValueError(
                    "theta carries an ar1_phi site but the model was built without "
                    "ar1=True: the AR coupling's bandwidth was not reserved, so the "
                    "probed marginal would be silently wrong. Rebuild the "
                    "MarginalOrbitModel with ar1=True."
                )
            problem = with_ar1(problem, jnp.asarray(theta["ar1_phi"]))
        return problem

    def problem_at(self, theta: Mapping):
        """The :class:`albireo.forward.Problem` at ``theta``: data, weights and operators.

        This is the entry point for residual diagnostics, which on real data are
        required: the inverse variances of an archival spectrum are usually estimated
        rather than measured (:func:`albireo.preprocess.estimate_ivar`), and a
        factor-of-two error there rescales every uncertainty the run reports. The
        recommended order is to run without a ``log_jitter`` site first and measure the
        discrepancy, then decide whether a jitter should absorb it
        (:func:`albireo.forward.with_jitter` states when that is legitimate and when it
        only widens a biased answer).

        Parameters
        ----------
        theta
            Parameter dict, as passed to :meth:`log_likelihood`.

        Returns
        -------
        Problem
            The problem with ``theta``'s velocities (and light fractions, LSF widths,
            response, nebular amplitudes, jitter or AR(1) coefficient, if those sites
            are present) substituted in.

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
        """Un-jitted marginal at ``theta`` (for gradient composition and tests)."""
        return self._marginal_at(self.problem, theta)

    def marginal(self, theta: Mapping) -> MarginalResult:
        """Jit-compiled marginal result (log-likelihood and conditional spectra) at ``theta``."""
        return self._marginal_jit(self.problem, dict(theta))

    def log_likelihood(self, theta: Mapping):
        """Jit-compiled marginal log-likelihood at ``theta`` (differentiable)."""
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
        """Marginal log-likelihood over a grid of ``theta`` values in one compiled graph.

        The trial points are independent and each returns a single number; nothing is
        sampled. A Python loop over
        :meth:`log_likelihood` pays a device synchronization per point and dispatches
        every point's linear algebra alone. This method runs the trials as a single
        ``lax.map``, so they share one compiled graph and batch into the same kernels,
        which makes a 2-D ``(K_1, K_2)`` grid, and the thousands of scans an
        injection-recovery calibration runs on top of it (:mod:`albireo.calibrate`),
        affordable.

        Parameters
        ----------
        theta
            The fixed part of the parameter dict, exactly as :meth:`log_likelihood`
            takes it.
        sweep
            The varying part: a mapping from site name to an array whose leading axis is
            the trial axis. Every entry must share that leading length, and each trailing
            shape must be the site's shape at a single ``theta`` (sweeping ``k`` on a
            two-component model takes ``(n_trials, 2)``). Entries override ``theta``.
        batch_size
            Trials per vmapped batch. ``None`` (default) applies the size-adaptive rule
            of :func:`_sweep_batch_default`; 1 is a purely sequential scan (least
            memory); ``n_trials`` forces one wide batch (fastest, most memory).
        problem
            Alternative base :class:`~albireo.forward.Problem` with the same structure
            and other numbers, such as a resimulated dataset
            (:func:`albireo.simulate.resimulate`). It lets a bootstrap reuse this
            model's operators instead of rebuilding them per trial.

        Returns
        -------
        jax.Array
            ``(n_trials,)`` marginal log-likelihoods, in ``sweep`` order.

        Raises
        ------
        ValueError
            If ``sweep`` is empty, names an unknown site, contains scalar entries, or has
            entries that disagree on the trial-axis length.

        Examples
        --------
        >>> k = jnp.stack([jnp.full_like(k2s, 60.0), k2s], axis=1)  # doctest: +SKIP
        >>> ll = model.log_likelihood_sweep(orbit, {"k": k})  # doctest: +SKIP
        """
        sweep = dict(sweep)
        if not sweep:
            raise ValueError("sweep is empty: pass at least one site to vary")
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
        """Build a numpyro model that samples ``priors`` and adds the marginal likelihood.

        Parameters
        ----------
        priors
            Distribution per sampled site (``period``, ``t_conj``, ``secosw``,
            ``sesinw``, ``k``; optionally ``log_tau``/``log_eta`` and the other sites
            listed in the module docstring). The ``k`` site is vector-valued: use a
            distribution with batch shape ``(n_stellar,)``.
        fixed
            Values injected as constants instead of sampled. This is the empirical-Bayes
            route: ``log_tau``/``log_eta`` fixed at their :func:`run_map` values for the
            NUTS run. Keys must not also appear in ``priors``.

        Returns
        -------
        callable
            A numpyro model for :func:`run_map` / :func:`run_nuts`. It records ``ecc``
            and ``omega`` (and ``ecc_out``, ``omega_out``, ``velocity_rel`` and
            ``nebular_amp`` when the corresponding sites are present) as deterministic
            sites, and adds ``-inf`` factors for the eccentricity disk, the LSF width and
            skewness bounds, the AR(1) stationarity bound and the bandwidth guard
            (``docs/math.md`` §7.1). The model takes the base
            :class:`~albireo.forward.Problem` as an optional argument and advertises it
            through a ``model_args`` attribute; the runners pass it through numpyro as a
            traced jit argument, the same contract as :meth:`marginal` (D27). Captured
            as a closure constant instead, the problem's arrays are baked into the jitted
            potential as XLA constants, whose compile-time folding allocates multi-GB
            temporaries at survey scale. Calling the model with no argument (as any plain
            numpyro utility such as ``log_density`` does) falls back to the closure,
            which is correct but not compile-safe at scale.

        Raises
        ------
        ValueError
            If a site is unknown or both fixed and sampled, if a free-velocity model also
            carries Keplerian sites, or if the orbital sites are incomplete.
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
                    f"a free-velocity model cannot also carry Keplerian sites {clash}: "
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
                # The construction-time widths fix the kernel radii: wider LSFs would be
                # truncated by the fixed radii, so they are rejected (see with_lsf).
                numpyro.factor(
                    "lsf_bound",
                    jnp.where(jnp.all(sig <= self._lsf_sigma_max), 0.0, -jnp.inf),
                )
            if "lsf_h3" in theta:
                h3 = jnp.atleast_1d(theta["lsf_h3"])
                # Beyond the bound the truncated Gauss-Hermite series is no longer a
                # credible line-spread profile (see _H3_MAX); clipped and rejected.
                numpyro.factor(
                    "lsf_h3_bound",
                    jnp.where(jnp.all(jnp.abs(h3) <= _H3_MAX), 0.0, -jnp.inf),
                )
            if "ar1_phi" in theta:
                phi = jnp.atleast_1d(theta["ar1_phi"])
                # with_ar1 clips at +-0.999 so the likelihood stays finite (and
                # rejectable, through this factor) outside the stationary region.
                numpyro.factor("ar1_bound", jnp.where(jnp.all(jnp.abs(phi) < 1.0), 0.0, -jnp.inf))
            if "log_nebular_amp" in theta:
                # Record the amplitudes the model applied: the site itself is identified
                # only up to an additive constant (nebular_amplitudes centers it), so the
                # raw samples are not the applied amplitudes.
                numpyro.deterministic("nebular_amp", nebular_amplitudes(theta))
            problem = self._theta_problem(theta, base=base)
            # Reject configurations whose relative shifts exceed the static bandwidth;
            # the assembled marginal likelihood is wrong beyond it (docs/math.md 7.1).
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
    """The model's traced arguments: an explicit ``model_args`` takes precedence, then
    the model's own ``model_args`` attribute (:meth:`MarginalOrbitModel.model`), else
    none."""
    if model_args is None:
        model_args = getattr(model, "model_args", ())
    return tuple(model_args)


@dataclass(frozen=True)
class MAPResult:
    """Result of :func:`run_map`.

    Attributes
    ----------
    params
        Constrained values of all sites, including deterministics.
    unconstrained
        Values in numpyro's unconstrained space.
    potential
        Potential energy (negative log joint, up to constants).
    grad_norm
        Unconstrained-space gradient norm.
    converged
        Whether ``grad_norm`` fell below ``tol``.
    num_steps
        Number of L-BFGS steps taken.

    Notes
    -----
    ``potential`` and ``grad_norm`` are evaluated at the convergence-check point, which,
    by the check-then-step loop, is one accepted L-BFGS step behind ``params``. The
    difference is irrelevant at convergence.
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
    """Maximum a posteriori fit over all sampled sites of ``model`` by L-BFGS.

    Runs L-BFGS (optax, with a zoom line search) on numpyro's potential in the
    unconstrained space, so constrained priors are handled by numpyro's standard
    transforms. With ``log_tau``/``log_eta`` among the sampled sites this is the ML-II
    (empirical Bayes) hyperparameter fit, since the spectra are already marginalized out
    of the likelihood (``docs/math.md`` §7.3).

    Parameters
    ----------
    model
        A numpyro model (from :meth:`MarginalOrbitModel.model`).
    init
        Constrained initial values for every sampled site. Circular orbits should start
        at small nonzero ``(secosw, sesinw)``; the origin is the one non-smooth point.
    rng_key
        JAX PRNG key for numpyro's model initialization; default ``PRNGKey(0)``.
    max_steps
        Maximum number of L-BFGS steps.
    tol
        Convergence threshold on the unconstrained-space gradient norm. This is an
        absolute threshold on a potential whose scale grows with the number of good
        pixels: on a survey-sized dataset the gradient norm at the true parameters is
        already in the hundreds, so the default is unreachable and ``converged`` is
        ``False`` regardless of the fit quality. The parameters, observed through
        ``callback``, are the better convergence indicator.
    callback
        Called after every accepted step as ``callback(step, potential, grad_norm, params)``
        with ``params`` the constrained site values. Without it the function produces no
        output while it runs, and a real-data fit can run for hours, so a first MAP on a
        new dataset should pass one. Returning ``True`` stops the fit early.
    model_args
        Positional arguments for ``model``, passed through numpyro as traced jit
        arguments rather than closure constants, which XLA constant-folds into multi-GB
        temporaries at scale (D27). Default: the model's own ``model_args`` attribute
        when it has one (:meth:`MarginalOrbitModel.model` advertises its base problem
        there), else ``()``. Passing ``()`` explicitly forces the closure path.

    Returns
    -------
    MAPResult
        Constrained and unconstrained parameters at the last step, with the potential,
        gradient norm, convergence flag and step count.

    Raises
    ------
    FloatingPointError
        If the gradient becomes non-finite.

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
    """Unconstrained-space Laplace covariance at ``params``, for use as a NUTS mass matrix.

    Evaluates the Hessian of the model potential at the given constrained site values
    (typically :attr:`MAPResult.params`; extra keys are ignored), symmetrizes it, floors
    the eigenvalues at ``floor * max_eig``, and returns the inverse as a dense array. The
    result is passed as ``inverse_mass_matrix`` to :func:`run_nuts` built from the same
    model. With the mass matrix preset to an approximation of the posterior covariance,
    warmup only tunes the step size; without it, parameter scales spanning many orders
    of magnitude drive early trajectories to the tree-depth cap, and warmup costs more
    than sampling (``docs/benchmarks.md``).

    Parameters
    ----------
    model
        A numpyro model (from :meth:`MarginalOrbitModel.model`).
    params
        Constrained site values at which the Hessian is evaluated.
    rng_key
        JAX PRNG key for numpyro's model initialization; default ``PRNGKey(0)``.
    floor
        Eigenvalue floor as a fraction of the largest eigenvalue.
    model_args
        As in :func:`run_map` (default: the model's own ``model_args`` attribute).

    Returns
    -------
    numpy.ndarray
        Dense inverse Hessian in the unconstrained space.
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
    # Reverse-over-reverse, not jax.hessian (forward-over-reverse). Forward-over-reverse
    # runs here, since it differentiates the custom rule's backward pass, which consists
    # of plain operations, but it was measured to return an appreciably asymmetric
    # Hessian on this stack, on the plain-autodiff path as well, so the cause is the
    # solver scans rather than the custom VJP. jacrev(jacrev(...)) matches central
    # finite differences of the gradient to 8 digits where forward-over-reverse does not
    # (D28). Forward mode applied directly to the marginal is not possible: JAX rejects
    # jvp of a custom_vjp function.
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
    """Sample the sites of ``model`` with NUTS; the spectra stay marginalized.

    The No-U-Turn Sampler (Hoffman & Gelman 2014) is the adaptive Hamiltonian Monte
    Carlo method reviewed by Betancourt (2017), here in its numpyro implementation (Phan
    et al. 2019). The marginal likelihood enters through a ``numpyro.factor`` site
    (D11), and numpyro supplies the priors, transforms and summaries.

    Parameters
    ----------
    model
        A numpyro model (from :meth:`MarginalOrbitModel.model`).
    rng_key
        JAX PRNG key.
    init
        Constrained initial values, typically the :attr:`MAPResult.params` dict (extra
        keys are ignored).
    num_warmup, num_samples, num_chains
        Warmup and sampling lengths per chain, and the number of chains (run
        sequentially).
    target_accept
        Target acceptance probability for the step-size adaptation.
    dense_mass
        Use a dense mass matrix. ``True`` is the appropriate default for the
        low-dimensional, correlated orbital posterior.
    inverse_mass_matrix
        Starting inverse mass matrix, typically from :func:`laplace_inverse_mass`, which
        makes warmup cheap.
    adapt_mass_matrix
        Whether warmup adapts the mass matrix. When an explicit ``inverse_mass_matrix``
        is supplied the default is ``False``: the early adaptation windows would replace
        the Laplace matrix with a poor few-sample estimate and restore the slow,
        deep-tree warmup that the matrix avoids. ``True`` overrides this.
    max_tree_depth
        NUTS tree-depth cap.
    progress_bar
        Show numpyro's progress bar.
    model_args
        As in :func:`run_map` (default: the model's own ``model_args`` attribute). With
        arguments present the MCMC runs with ``jit_model_args=True``, so they are traced
        through the jitted sample loop rather than baked into it as XLA constants.

    Returns
    -------
    MCMC
        The numpyro ``MCMC`` object (``.get_samples()``, ``.print_summary()``).
        Divergences and tree depths are collected as the extra fields ``diverging`` and
        ``num_steps``.

    References
    ----------
    Betancourt, M. 2017, arXiv:1701.02434
    Hoffman, M. D. & Gelman, A. 2014, Journal of Machine Learning Research, 15, 1593
    Phan, D., Pradhan, N. & Jankowiak, M. 2019, arXiv:1912.11554
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
    """Draw spectra from the joint posterior, shape ``(num_draws, n_comp, n_pix)``.

    Each draw selects a posterior ``theta`` sample at random and draws once from the
    conditional Gaussian over the spectra (``docs/math.md`` §7.4), so the returned
    scatter includes both the conditional spectral uncertainty and the orbital
    uncertainty.

    Parameters
    ----------
    model
        The :class:`MarginalOrbitModel` the samples were drawn from.
    samples
        Posterior samples by site name, as returned by ``MCMC.get_samples()``.
    key
        JAX PRNG key.
    num_draws
        Number of spectra draws.
    extra
        Sites missing from ``samples`` (for example ``log_tau``/``log_eta`` when they
        were fixed during sampling).

    Returns
    -------
    jax.Array
        ``(num_draws, n_comp, n_pix)`` deviation spectra.
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

"""Declarative interface to a disentangling fit.

**Experimental.** The names defined here may change.
:class:`~albireo.inference.MarginalOrbitModel` and the functions around it are the
supported interface, and :meth:`Disentangler.expert` gives direct access to it.

A declaration names the components (one :class:`Star` per stellar component, optionally a
:class:`Telluric` column and a :class:`Nebular` component), either an :class:`Orbit` of
priors or a measured per-epoch velocity table, the line-spread function of each
instrument, and the dataset. From that declaration the module derives what the low-level
interface otherwise requires: the velocity budget, the model grid and its margins, the
conjunction-phase scan, the matched ``priors``/``init`` pair, the empirical-Bayes (ML-II)
smoothness step, and the Laplace mass matrix. :meth:`Disentangler.explain` prints every
derivation, and :meth:`Disentangler.expert` returns the ``(model, priors, init, fixed)``
that the declaration compiles to. ``docs/method.md`` describes the method in the terms of
a regularized least-squares fit and defines the statistical terms used in the printed
reports.

Quantities that are scientific claims have no defaults. Light fractions are required:
with constant light fractions the likelihood depends only on the products ``l_i * d_i``
(``docs/math.md`` §5.2), so every recovered line depth scales as ``1 / l_i`` and the fit
does not constrain the assumed value. The same applies to the period prior, to the
wavelength medium wherever absolute line positions matter, and to the nebular velocity.
Each of these is either required, rejected, or listed in the ``Assumed, not measured``
block printed by :meth:`Fit.summary`.
"""

from __future__ import annotations

import math
import textwrap
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro.diagnostics import effective_sample_size, hpdi, split_gelman_rubin
from numpyro.distributions import constraints
from numpyro.distributions.transforms import Transform, biject_to

from albireo.data import Dataset
from albireo.forward import PER_EPOCH, data_residual_zscores, declared_lsf_widths
from albireo.grids import C_KMS, LogGrid
from albireo.inference import (
    _SWEEP_BATCH_BYTES,
    LaplaceCheck,
    MarginalOrbitModel,
    keplerian_residuals,
    laplace_check,
    orbit_parameters,
    orbit_velocities,
    posterior_spectra,
    relative_velocities,
    relative_velocity_errors,
    run_map,
    run_nuts,
)
from albireo.likelihood import spectra_std
from albireo.priors import SmoothnessPrior, nebular_windows, window_profile
from albireo.specs import Between, Fixed, Known, Sampled, Scanned, Spec

__all__ = [
    "LSF",
    "Between",
    "Disentangler",
    "EpochRenormalization",
    "Fit",
    "Fixed",
    "Known",
    "Nebular",
    "Orbit",
    "Posterior",
    "Sampled",
    "Scanned",
    "Smoothness",
    "Spec",
    "Star",
    "Telluric",
]

# Barycentric motion, one way, in km/s. It is reserved in the velocity budget whenever the
# model has a component whose velocity law includes v_bary: a telluric column, or
# topocentric data.
_V_BARY_MAX = 30.0

# Numerical headroom. The terms it multiplies are already a worst case (the sum of the
# priors' upper bounds at the largest eccentricity they allow), so a larger slack would
# double count. Slack also has a cost: the solver bandwidth grows with the budget and the
# solve cost grows with the bandwidth.
_BUDGET_SLACK = 1.05
# Trials per compiled batch of the semi-amplitude scan (D62), and the bytes one trial of
# the scan's sweep holds per component for each entry of the (n, half_bandwidth) band,
# with n the number of components times the grid size. XLA's buffer assignment of the
# compiled sweep gave 100-108 bytes per band entry at two components and 147-164 at three
# (a 1,809-pixel coarse grid at half-bandwidth 156), so the constant is about a factor of
# the component count above the 48 of albireo.inference._sweep_batch_default. It sizes
# the batch on large problems only: below about 1 GiB per batch the scan keeps 16.
_SCAN_BATCH = 16
_SCAN_BYTES_PER_BAND_ENTRY = 52
# Headroom factor for the free table of a `velocities=` declaration. It is a convention,
# not a measurement: the declared velocities are a starting point, and the fit must be able
# to move without reaching the bandwidth guard. It is reported as its own budget term.
_FREE_VELOCITY_HEADROOM = 2.0

# The meaning of each numpyro site, printed under the site list of `Disentangler.explain`.
# The site names are those of `albireo.inference` (docs/method.md has the glossary).
_SITE_MEANINGS = {
    "period": "orbital period [d]",
    "t_conj": "time of conjunction of the first star [d]",
    "secosw": "sqrt(e) cos(omega), eccentricity e and argument of periastron omega",
    "sesinw": "sqrt(e) sin(omega)",
    "k": "radial-velocity semi-amplitude of each star [km/s]",
    "velocity": "free radial velocity of each star at each epoch [km/s]",
    "ar1_phi": "lag-one correlation of the noise along each epoch's pixels",
    "log_nebular_amp": "ln of the nebular line strength at each epoch",
    "log_tau": "ln of each component's curvature weight (regularization strength)",
    "log_eta": "ln of each component's continuum weight (regularization strength)",
}


# -- parameter specifications -------------------------------------------------
#
# The specifications (Spec, Fixed, Known, Between, Scanned, Sampled) are defined in
# albireo.specs, which albireo.match imports without importing this module. They are
# re-exported here, and this module adds the coercion of a bare number.


def _coerce_spec(value, what: str) -> Spec:
    """Accept a bare number as ``Fixed``; everything else must be explicit."""
    if isinstance(value, Spec):
        return value
    if isinstance(value, int | float):
        return Fixed(float(value))
    raise TypeError(
        f"{what} must be a Fixed/Known/Between/Scanned/Sampled, or a plain number meaning "
        f"Fixed; got {type(value).__name__}. A bare tuple is deliberately not accepted: "
        "(5.5, 6.5) reads as either a range or a two-component vector."
    )


# -- what the spectrograph does -----------------------------------------------


@dataclass(frozen=True)
class Smoothness:
    """Starting point of one component's smoothness hyperparameters.

    ``tau`` and ``eta`` are fitted by empirical Bayes (ML-II) in
    :meth:`Disentangler.fit`, so these values are the centre and width of the hyperprior,
    not the values used in the final model. The starting point still matters:
    on HR 6819 the rotationally broadened Be star requires a ``tau0`` five orders of
    magnitude larger than its sharp-lined companion. A hyperparameter that does not move
    from its start was not constrained by the data, and :meth:`Fit.summary` flags it.

    Parameters
    ----------
    tau0
        Centre of the curvature-penalty hyperprior. Larger values impose more smoothing.
    eta0
        Centre of the ridge hyperprior, which shrinks the component toward its continuum.
    sigma
        Width of both hyperpriors, in natural logarithms.
    eta_min
        Lower bound on the fitted ``eta``, or ``None`` (default) for no bound. The ridge
        is the only term of the prior that acts on the constant and the slope of a
        component, which the curvature penalty leaves free. Where the continuum
        normalization of the epochs differs from epoch to epoch, ML-II can lower ``eta``
        by orders of magnitude, and the components then take up those errors as broad
        drifts in opposite directions, which their light-weighted sum does not show. On a
        simulated binary with a ripple of 1.3% rms on scales of 40-80 A in every epoch,
        ``eta`` fell from 5 to 0.4 and 0.025 and the components drifted by 0.28 and 0.85
        continuum units rms; on PEPSI spectra of a binary it fell to 0.015 and 0.0014.
        With a bound, the hyperprior of ``ln eta`` is the normal of ``eta0`` and
        ``sigma`` truncated at ``ln eta_min``. :meth:`Fit.summary` flags a fall of ``eta``
        below its centre and a fitted value at the bound.
    """

    tau0: float = 300.0
    eta0: float = 5.0
    sigma: float = 3.0
    eta_min: float | None = None

    def __post_init__(self) -> None:
        for name in ("tau0", "eta0", "sigma"):
            value = float(getattr(self, name))
            if not (np.isfinite(value) and value > 0.0):
                raise ValueError(
                    f"Smoothness {name} must be a finite positive number; got {value}. The "
                    "hyperpriors are normal in ln tau and ln eta, centred on ln tau0 and "
                    "ln eta0 with width sigma."
                )
        if self.eta_min is not None and not (
            np.isfinite(float(self.eta_min)) and float(self.eta_min) > 0.0
        ):
            raise ValueError(
                f"eta_min must be a positive number or None; got {self.eta_min}. It bounds "
                "the continuum weight eta from below."
            )


@dataclass(frozen=True)
class LSF:
    """An instrument's line-spread function, declared rather than inferred.

    Parameters
    ----------
    sigma_kms
        Gaussian sigma in km/s. A sequence gives one width per entry of
        ``anchors_angstrom`` and declares a wavelength-dependent LSF. With
        :data:`albireo.PER_EPOCH` (see :meth:`per_epoch`) the width is read from each
        epoch's file instead.
    anchors_angstrom
        Wavelengths in angstrom at which ``sigma_kms`` is specified. The width is
        interpolated between them.

    Notes
    -----
    This width also fixes the radius of the convolution kernel, so it is an upper bound as
    well as a value: a fit that later infers a wider LSF is rejected. A width to be
    inferred through the low-level API needs some margin.

    Gauss-Hermite skewness (``h3``) has no field here. It reaches the kernel only
    through :func:`albireo.build_problem`, not through the model class built by this
    module, so a field would be accepted and then discarded. It is declared through
    :meth:`Disentangler.expert`.
    """

    sigma_kms: Any
    anchors_angstrom: Sequence[float] | None = None

    @classmethod
    def from_resolution(cls, resolving_power: float, **kwargs) -> LSF:
        """Build from a resolving power ``R = lambda / dlambda``, taken as a FWHM.

        The conversion is ``sigma = c / (R * 2 sqrt(2 ln 2))``. Using ``c / R`` instead
        overstates the kernel radius by a factor of 2.35.

        Parameters
        ----------
        resolving_power
            Resolving power ``R``, interpreted as the FWHM of the line-spread function.
            Must be positive.
        **kwargs
            Passed to the :class:`LSF` constructor, e.g. ``anchors_angstrom``.

        Returns
        -------
        LSF

        Raises
        ------
        ValueError
            If ``resolving_power`` is not positive.

        Examples
        --------
        >>> round(LSF.from_resolution(48_000).sigma_kms, 3)
        2.652
        """
        if not resolving_power > 0:
            raise ValueError(f"resolving power must be positive; got {resolving_power}")
        sigma = C_KMS / (float(resolving_power) * 2.0 * math.sqrt(2.0 * math.log(2.0)))
        return cls(sigma_kms=sigma, **kwargs)

    @classmethod
    def per_epoch(cls) -> LSF:
        """Take the width from each epoch's own file.

        The reader records a file's resolving power on the epoch
        (:attr:`albireo.EpochData.lsf_sigma_kms`), and this declaration models every
        epoch at that width. Exposures at two resolving powers filed under one
        instrument name (HARPS's high-accuracy and high-efficiency modes, for instance)
        then each get their own kernel. Every epoch of the instrument must have a width;
        otherwise the declaration raises, naming the epochs. The string ``"per-epoch"``
        (the TOML form) is accepted in its place.

        Returns
        -------
        LSF
        """
        return cls(sigma_kms=PER_EPOCH)

    @property
    def is_per_epoch(self) -> bool:
        """Whether this declaration takes its widths from the epochs."""
        return isinstance(self.sigma_kms, str) and self.sigma_kms == PER_EPOCH

    @property
    def max_sigma_kms(self) -> float:
        """Widest declared sigma in km/s, which sets the model-grid margin.

        Raises
        ------
        ValueError
            For a per-epoch declaration, whose widths are stored on the dataset.
            :meth:`Disentangler._widest_lsf` resolves them.
        """
        if self.is_per_epoch:
            raise ValueError(
                "a per-epoch LSF has no width of its own; its widths are the epochs' "
                "(albireo.forward.declared_lsf_widths)"
            )
        return float(np.max(np.atleast_1d(np.asarray(self.sigma_kms, dtype=float))))

    def widths(self, dataset, key: str) -> np.ndarray:
        """Every width this declaration assigns to ``dataset``'s epochs of instrument ``key``.

        Parameters
        ----------
        dataset
            The epochs the declaration applies to.
        key
            The instrument key it is declared under.

        Returns
        -------
        numpy.ndarray
            The declared widths in km/s: the anchors' widths, one scalar, or the
            distinct per-epoch widths.
        """
        if self.is_per_epoch:
            return np.array(list(declared_lsf_widths(dataset, key)), dtype=float)
        return np.atleast_1d(np.asarray(self.sigma_kms, dtype=float))


def _coerce_lsf(value, key: str) -> LSF:
    if isinstance(value, LSF):
        return value
    if isinstance(value, int | float):
        return LSF(sigma_kms=float(value))
    if isinstance(value, str) and value == PER_EPOCH:
        return LSF.per_epoch()
    raise TypeError(
        f"lsf[{key!r}] must be an LSF, a sigma in km/s, or {PER_EPOCH!r}; "
        f"got {type(value).__name__}"
    )


# -- the components -----------------------------------------------------------


@dataclass(frozen=True)
class Star:
    """One stellar component: a name, a light fraction, and a smoothness declaration.

    Parameters
    ----------
    name
        Identifier used wherever the component would otherwise be a row index:
        ``fit.star("Be")``, plot legends, FITS headers, and the ML-II report. Component
        order is a convention that the data do not determine, so the name is the only
        guard against reading row 0 as the wrong star.
    light
        Fraction of the total light contributed by this star. It is required and is an
        assumption: with constant light fractions only the products ``l_i * d_i`` are
        observable (``docs/math.md`` §5.2), so every recovered depth scales as ``1 / l_i``
        and the fit does not constrain this value. It should be quoted beside any
        result derived from the spectra. The star light fractions must sum to 1.
    smoothness
        Starting point of this component's ML-II hyperparameters. A rotationally
        broadened star requires a much larger ``tau0`` than a sharp-lined one.
    """

    name: str
    light: float
    smoothness: Smoothness = field(default_factory=Smoothness)


@dataclass(frozen=True)
class Telluric:
    """A telluric component: static in the topocentric frame, with no light fraction.

    The telluric column has light fraction 1 and lies outside the stellar simplex: it
    multiplies the composite spectrum rather than contributing to it. Declaring one
    reserves barycentric motion, both signs, in the velocity budget.

    Parameters
    ----------
    smoothness
        Starting point of this component's ML-II hyperparameters. The default ``tau0`` is
        50, lower than a star's, because telluric absorption is comparatively sharp.
    """

    smoothness: Smoothness = field(default_factory=lambda: Smoothness(tau0=50.0))


@dataclass(frozen=True)
class Nebular:
    """A nebular emission component: static in the barycentric frame, free amplitude.

    This is the counterpart of :class:`Telluric`. Nebular flux is added on top of the total
    continuum and takes no light from the stars, so its per-epoch amplitude is a free
    parameter rather than a light fraction.

    Declaring one assembles several coupled pieces: the component column, the
    ``log_nebular_amp`` site with its prior and starting value, the per-pixel
    ``eta_profile`` that confines the component to its lines, the agreement between that
    profile and ``nebular_v_kms`` (the component is shifted by ``v_kms``, so its spectrum
    holds the lines at rest, where the profile's windows are built), the smoothness prior
    object passed at construction so that ML-II keeps the profile, and the additional
    velocity budget.
    Omitting the profile makes the fit 250 nats worse and gives a 2.6% error in K₂.
    Omitting the component gives errors of 11.5% in the Hβ equivalent width and 59% in K₂.

    Parameters
    ----------
    v_kms
        Velocity of the nebula in km/s, in the model grid's frame. The component is
        shifted by this velocity (``nebular_v_kms`` of
        :func:`albireo.forward.build_problem`), so its spectrum holds the lines at their
        rest wavelengths, where the windows of the profile are built. The velocity is not
        identified, since a constant shift of a free spectrum is absorbed by the spectrum,
        and is reported as a placement convention. It must agree with the nebula's
        velocity to well within ``halfwidth_kms``: a line displaced by more than the
        half-width falls outside its window.
    lines
        Rest wavelengths in air angstrom. Default :data:`albireo.NEBULAR_LINES`.
    halfwidth_kms
        Half-width of each window in km/s. A generous value is preferable: with too
        narrow a window real emission is absorbed into the stellar components, while too
        wide a window only restores some of the freedom the profile removes.
    smoothness
        Starting point of this component's ML-II hyperparameters. The default ``tau0`` is
        8, appropriate for narrow emission lines.
    """

    v_kms: float = 0.0
    lines: Mapping[str, float] | Sequence[float] | None = None
    halfwidth_kms: float = 300.0
    smoothness: Smoothness = field(default_factory=lambda: Smoothness(tau0=8.0))


Component = Star | Telluric | Nebular


# -- the orbit ----------------------------------------------------------------


@dataclass(frozen=True)
class Orbit:
    """The orbital model, declared in terms of what is known rather than as sample sites.

    Parameters
    ----------
    period
        Orbital period in days (required). This declaration does not search in period: it
        scans conjunction phase at a single period, so the prior must be narrow enough for
        a phase scan to be meaningful (or a periodogram run first).
    k
        Velocity semi-amplitude in km/s per :class:`Star`, in the order the stars are
        declared. Either one specification with one entry per star, such as
        ``Between([10.0, 10.0], [90.0, 90.0])``, or a sequence of one specification per
        star. A bare number is ``Fixed``. A single-valued specification is accepted only
        for a single star.
    t_conj
        Time of conjunction in the dataset's time system. The default ``"scan"`` locates
        it by scanning one period before optimization. The marginal likelihood is sharply
        multimodal in phase, and L-BFGS started in the wrong trough converges to the
        wrong solution without indication of failure.
    ecc
        Eccentricity. ``Between(lo, hi)`` samples it through the
        ``(sqrt(e) cos w, sqrt(e) sin w)`` parameterization. ``Fixed(0.0)`` declares a
        circular orbit and is handled exactly, by not sampling those sites. The
        parameterization is singular at ``e = 0``, where the gradient is NaN and numpyro
        reports only "Cannot find valid initial parameters".
    omega
        Argument of periastron in radians. Required when ``ecc`` is ``Fixed`` and
        nonzero. With a free eccentricity, a value given here (and a ``start_at`` on the
        eccentricity's range) sets the start of the fit, for instance from a velocity
        table's orbit. The default start is ``e = 0.05`` at 0.5 rad.
    outer
        Outer orbit of a hierarchical triple. :class:`Disentangler` does not support it
        in this version and raises ``NotImplementedError`` when it is set; the model
        supports triples through :class:`~albireo.inference.MarginalOrbitModel`.
    """

    period: Any
    k: Any
    t_conj: Any = "scan"
    ecc: Any = field(default_factory=lambda: Between(0.0, 0.95))
    omega: Any = None
    outer: Orbit | None = None


# -- the velocity budget ------------------------------------------------------


@dataclass(frozen=True)
class VelocityBudget:
    """An itemized bound, in km/s, on the largest relative velocity between components.

    The solver bandwidth is built from this total. With too small a bound the sampler
    stalls at the bandwidth guard, and a direct call to
    :meth:`albireo.inference.MarginalOrbitModel.log_likelihood` returns a wrong value
    without raising. The bound is derived from the support of the semi-amplitude priors.

    Attributes
    ----------
    terms
        ``(name, value)`` pairs, one per contribution, in the order they are applied.
        Values are in km/s and the multiplicative terms report the running total.
    total
        The bound passed to the model as ``v_rel_max_kms``, in km/s.
    """

    terms: tuple[tuple[str, float], ...]
    total: float

    def __str__(self) -> str:
        rows = "\n".join(f"    {name:<34s} {value:8.1f}" for name, value in self.terms)
        return f"velocity budget (km/s)\n{rows}\n    {'total':<34s} {self.total:8.1f}"


def _velocity_budget(orbit: Orbit | None, velocities, components, frame: str, explicit):
    """Bound the relative velocity from the declaration, itemizing every term.

    The total must bound the largest relative velocity between any two model components
    at any epoch, over the support of the prior, not over the fitted solution.

    From an :class:`Orbit` the bound is the support of the semi-amplitude priors. From a
    ``velocities=`` declaration it is the measured table centred per component, with a
    factor of two of headroom because the table is free. The table is centred because the
    absolute level is unidentified and must not add to the bandwidth (see
    :func:`_centred_velocities`).
    """
    n_stellar = sum(isinstance(c, Star) for c in components)
    terms: list[tuple[str, float]] = []

    if orbit is None:
        centred = _centred_velocities(velocities)
        reach = float(np.sum(np.max(np.abs(centred), axis=1)))
        terms.append(("sum of declared per-star |v| excursions", reach))
        total = reach * _FREE_VELOCITY_HEADROOM
        terms.append((f"x {_FREE_VELOCITY_HEADROOM:g} headroom (the table is free)", total))
        ecc_max = 0.0
    else:
        specs = _k_specs(orbit, n_stellar)
        reach = sum(spec.upper() for spec in specs)
        terms.append(("sum of stellar |K| bounds", reach))

        ecc_max = 0.95
        if isinstance(orbit.ecc, Between):
            ecc_max = float(np.max(np.asarray(orbit.ecc.hi, dtype=float)))
        elif isinstance(orbit.ecc, Fixed):
            ecc_max = float(np.max(np.abs(np.asarray(orbit.ecc.value, dtype=float))))
        total = reach * (1.0 + ecc_max)
        terms.append((f"x (1 + e_max = {1.0 + ecc_max:.2f})", total))

    if frame == "topocentric" or any(isinstance(c, Telluric) for c in components):
        total += 2.0 * _V_BARY_MAX
        terms.append(("barycentric motion (both signs)", 2.0 * _V_BARY_MAX))

    nebular = next((c for c in components if isinstance(c, Nebular)), None)
    if nebular is not None:
        extra = abs(float(nebular.v_kms)) + reach
        total += extra
        terms.append(("nebular offset + stellar reach", extra))

    total *= _BUDGET_SLACK
    terms.append((f"x {_BUDGET_SLACK} slack", total))

    if explicit is not None:
        if float(explicit) < total:
            raise ValueError(
                f"velocity_budget_kms={explicit} is smaller than the {total:.1f} km/s the "
                "declared priors can reach, so the solver bandwidth would not cover every "
                "configuration the prior allows. Sampling stalls against that guard rather "
                f"than failing.\n{VelocityBudget(tuple(terms), total)}"
            )
        terms.append(("caller's override", float(explicit)))
        total = float(explicit)
    return VelocityBudget(tuple(terms), float(total))


def _centred_velocities(velocities) -> np.ndarray:
    """Declared velocities with each component's own mean removed, ``(n_stellar, n_ep)``.

    Only the centred table is identified and enters the model:
    :func:`albireo.inference.relative_velocities` removes one zero point per component,
    because with no orbit coupling the stars each free spectrum absorbs a constant added
    to its own shifts. The declared absolute level, such as a systemic velocity of
    +150 km/s for the SMC, has no solver cost and does not add to the bandwidth.

    The centring is done here in velocity space. The model does it exactly, in pixels,
    where ``xi = artanh(v/c)`` makes the offset a translation. This function computes only
    a bound, and the two differ by ``O(v^2/c^2)``, of order 1e-8 at stellar velocities,
    compared with the factor-of-two headroom in the budget.
    """
    v = np.asarray(velocities, dtype=float)
    return v - v.mean(axis=1, keepdims=True)


def _place_hyperparameters(dis, priors: dict, init: dict) -> None:
    """Place the sites common to every declaration, whichever velocity model it uses.

    Smoothness is always fitted by ML-II. A fixed ``(tau, eta)`` pair is not used
    anywhere in this package on real data, and defensible centres span five orders of
    magnitude between a sharp-lined and a rotationally broadened star. The nebular
    amplitudes are placed here for the same reason as in :meth:`Fit._velocity_priors`:
    omitting them would hold the component static at amplitude 1 without reporting it.

    A declared ``eta_min`` truncates the hyperprior of ``log_eta`` (:class:`_FlooredNormal`),
    and the start is moved above the bound. Without one the site keeps its normal
    hyperprior, so a declaration without bounds builds the same model as before they
    existed.
    """
    ordered = dis.ordered_components
    centres = np.array([_smoothness_of(c).tau0 for c in ordered], dtype=float)
    etas = np.array([_smoothness_of(c).eta0 for c in ordered], dtype=float)
    widths = np.array([_smoothness_of(c).sigma for c in ordered], dtype=float)
    priors["log_tau"] = dist.Normal(jnp.log(jnp.asarray(centres)), jnp.asarray(widths))
    init["log_tau"] = jnp.log(jnp.asarray(centres))
    floor = dis._log_eta_floor()
    if np.all(np.isinf(floor)):
        priors["log_eta"] = dist.Normal(jnp.log(jnp.asarray(etas)), jnp.asarray(widths))
        init["log_eta"] = jnp.log(jnp.asarray(etas))
    else:
        priors["log_eta"] = _FlooredNormal(
            jnp.log(jnp.asarray(etas)), jnp.asarray(widths), jnp.asarray(floor)
        )
        init["log_eta"] = jnp.asarray(_above_floor(np.log(etas), floor))

    if dis._nebular is not None:
        n_epochs = dis.dataset.n_epochs
        priors["log_nebular_amp"] = dist.Normal(jnp.zeros(n_epochs), 0.3).to_event(1)
        init["log_nebular_amp"] = jnp.zeros(n_epochs)


# A start this far above a declared bound, in ln eta. The unconstrained coordinate of a
# bounded entry is ln(ln eta - ln eta_min), which is infinite at the bound itself.
_START_ABOVE_FLOOR = 0.1
# Fit.summary reports eta at its bound within this distance in ln eta. At a bound that
# holds against the data the fitted value lies 1/|d ln p / d ln eta| above it, a few
# percent where the data pull hard.
_AT_BOUND = 0.05
# Fit.summary flags an eta more than this many hyperprior standard deviations below its
# centre, in a model of two or more components (with one, the data determine its broad
# structure directly). In a blind test on seven two-star fits of APF and PEPSI spectra,
# the binary whose components drifted had eta at -1.9 and -2.7 sigma of the default
# hyperprior, and the other twelve components stayed above -0.6 sigma (eta 1.1 to 297).
_ETA_FALL_SIGMA = 1.0
# Fit.summary flags an optimization stopped at its step cap whose potential still fell by
# more than this over its last steps (MAPResult.potential_fall), in nats. On the same
# blind test the fit whose components drifted fell by 60 nats over its last 25 steps (its
# regularization weights were still moving), and the others stopped at the cap by at most
# 0.6, two of them at kinks of the potential where the gradient stays large and no step
# lowers the potential. The fall is an indication; laplace_check is the test.
_STILL_FALLING_NATS = 1.0


def _indented(text: str) -> str:
    """``text`` wrapped at 90 columns under a flag line of a report."""
    return textwrap.fill(text, width=90, initial_indent="    ", subsequent_indent="    ")


def _above_floor(log_eta: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """``log_eta`` moved to at least ``_START_ABOVE_FLOOR`` above each finite bound."""
    log_eta = np.asarray(log_eta, dtype=float)
    floor = np.asarray(floor, dtype=float)
    return np.where(np.isfinite(floor), np.maximum(log_eta, floor + _START_ABOVE_FLOOR), log_eta)


class _Floor(constraints.Constraint):
    """``x > lower_bound``, entry by entry; an entry of ``-inf`` imposes no bound."""

    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def __call__(self, x):
        return x > self.lower_bound

    def feasible_like(self, prototype):
        lower = jnp.asarray(self.lower_bound)
        return jnp.broadcast_to(
            jnp.where(jnp.isfinite(lower), lower + 1.0, 0.0), jnp.shape(prototype)
        )

    def tree_flatten(self):
        return (self.lower_bound,), (("lower_bound",), {})


class _FloorTransform(Transform):
    """``z -> lower + exp(z)`` where the bound is finite, the identity where it is not.

    numpyro's own transform to ``greater_than`` returns ``-inf`` for an infinite bound, so
    a vector site whose entries are bounded only in part needs this one. An unbounded entry
    keeps the unconstrained coordinate it has under a plain normal prior.
    """

    sign = 1

    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    @property
    def codomain(self):
        return _Floor(self.lower_bound)

    def _bound(self):
        lower = jnp.asarray(self.lower_bound)
        finite = jnp.isfinite(lower)
        return finite, jnp.where(finite, lower, 0.0)

    def __call__(self, z):
        finite, lower = self._bound()
        return jnp.where(finite, lower + jnp.exp(z), z)

    def _inverse(self, x):
        finite, lower = self._bound()
        gap = jnp.where(finite, x - lower, 1.0)
        return jnp.where(finite, jnp.log(gap), x)

    def log_abs_det_jacobian(self, z, x, intermediates=None):
        finite, _ = self._bound()
        return jnp.where(finite, z, 0.0)

    def tree_flatten(self):
        return (self.lower_bound,), (("lower_bound",), {})


biject_to.register(_Floor, lambda constraint: _FloorTransform(constraint.lower_bound))


class _FlooredNormal(dist.Distribution):
    """A normal distribution truncated below at ``low``, entry by entry.

    The density is the normal's divided by its mass above ``low``; an entry with
    ``low = -inf`` is the normal itself. It is the hyperprior of ``log_eta`` when a
    component declares ``Smoothness(eta_min=...)``.
    """

    arg_constraints: ClassVar[dict] = {"loc": constraints.real, "scale": constraints.positive}
    pytree_data_fields = ("loc", "scale", "low")

    def __init__(self, loc, scale, low, *, validate_args=None):
        shape = jax.lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale), jnp.shape(low))
        self.loc = jnp.broadcast_to(jnp.asarray(loc, dtype=jnp.float64), shape)
        self.scale = jnp.broadcast_to(jnp.asarray(scale, dtype=jnp.float64), shape)
        self.low = jnp.broadcast_to(jnp.asarray(low, dtype=jnp.float64), shape)
        super().__init__(batch_shape=shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return _Floor(self.low)

    def log_prob(self, value):
        normal = dist.Normal(self.loc, self.scale).log_prob(value)
        # log P(X > low) for X ~ N(loc, scale); zero where low = -inf
        mass = jax.scipy.special.log_ndtr((self.loc - self.low) / self.scale)
        return jnp.where(value > self.low, normal - mass, -jnp.inf)

    def sample(self, key, sample_shape=()):
        lower = jax.scipy.special.ndtr((self.low - self.loc) / self.scale)
        u = jax.random.uniform(
            key, tuple(sample_shape) + self.batch_shape, minval=lower, maxval=1.0
        )
        return self.loc + self.scale * jax.scipy.special.ndtri(u)


def _range_bounds(spec) -> tuple[float, float] | None:
    """The ``(lo, hi)`` of a semi-amplitude declared as a range, or ``None``.

    A range is a :class:`Between`, either one per star or one component of a vector
    :class:`Between` (``k=Between([10, 10], [90, 90])``), which :func:`_k_specs` returns
    as a :class:`_ScalarView`.
    """
    index = 0
    if isinstance(spec, _ScalarView):
        index, spec = spec.index, spec.spec
    if not isinstance(spec, Between):
        return None
    lo = np.atleast_1d(np.asarray(spec.lo, dtype=float))
    hi = np.atleast_1d(np.asarray(spec.hi, dtype=float))
    return float(lo[min(index, lo.size - 1)]), float(hi[min(index, hi.size - 1)])


def _check_noise_correlation(dis) -> None:
    """Reject a noise correlation that omits an instrument or is not a correlation."""
    declared = dis.noise_correlation
    if declared is None:
        return
    if isinstance(declared, Spec):
        if isinstance(declared, Scanned | Sampled):
            raise TypeError("noise_correlation takes Fixed, Known or Between, not a scan")
        values = [float(v) for v in np.atleast_1d(np.asarray(declared.start(), dtype=float))]
        if len(values) != 1:
            raise ValueError("a fitted noise_correlation is one value shared by every epoch")
        if isinstance(declared, Between):
            bounds = np.concatenate(
                [np.atleast_1d(np.asarray(b, dtype=float)) for b in (declared.lo, declared.hi)]
            )
            if not (np.all(np.isfinite(bounds)) and np.all(np.abs(bounds) <= 1.0)):
                raise ValueError(
                    f"a fitted noise_correlation must range within [-1, 1]; got {declared}. "
                    "The model rejects |phi| >= 1, so a wider range would be cut there "
                    "rather than sampled."
                )
    elif isinstance(declared, Mapping):
        missing = sorted(set(dis.dataset.instruments) - set(declared))
        if missing:
            raise ValueError(
                f"noise_correlation has no entry for instrument(s) {missing}; give one per "
                "instrument of the dataset (0 for independent pixels)"
            )
        values = [float(v) for v in declared.values()]
    else:
        values = [float(declared)]
    if any(not (np.isfinite(v) and -1.0 < v < 1.0) for v in values):
        raise ValueError(
            f"noise_correlation must lie in (-1, 1); got {declared}. It is the lag-one "
            "correlation of each epoch's noise along its pixel index, not a variance."
        )


def _place_noise_correlation(dis, priors: dict, init: dict, fixed: dict) -> None:
    """Place the ``ar1_phi`` site: fixed per epoch from a declaration, or one sampled value."""
    declared = dis.noise_correlation
    if declared is None:
        return
    if isinstance(declared, Spec):
        distribution = declared.distribution()
        if distribution is None:
            fixed["ar1_phi"] = declared.start()
        else:
            priors["ar1_phi"] = distribution
            init["ar1_phi"] = declared.start()
        return
    fixed["ar1_phi"] = jnp.asarray(dis.noise_correlation_per_epoch())


def _check_velocities(dis, stars) -> np.ndarray:
    """Validate a ``velocities=`` declaration, rejecting a table that cannot warm-start.

    Returns a read-only copy, so that a later change to the caller's array cannot
    desynchronize the declaration from the budget, grid and model built from it.
    """
    v = np.array(np.atleast_2d(np.asarray(dis.velocities, dtype=float)), dtype=float, copy=True)
    v.setflags(write=False)
    want = (len(stars), dis.dataset.n_epochs)
    if v.shape != want:
        raise ValueError(
            f"velocities must have shape {want}: one row per star, one column per epoch, "
            f"in the dataset's own epoch order; got {v.shape}"
        )
    if not np.all(np.isfinite(v)):
        raise ValueError("velocities must all be finite")

    # A cold start raises an error. With every component at the same velocity at every
    # epoch the stars are indistinguishable, and the fit converges to a solution
    # 122,000 nats below the warm-started one (D42). The separation is measured on the
    # centred table, which is what the model uses: it removes one zero point per
    # component, so a constant offset between two stars separates nothing. Removing the
    # mean of a constant that is not exactly representable leaves a rounding residue of a
    # few units in the last place (5.7e-14 km/s for a row of 152.3 km/s), so the
    # separation is compared with a tolerance of 1e-9 of the table's largest velocity, at
    # least 1e-9 km/s. The tolerance is far below any LSF width.
    centred = _centred_velocities(v)
    separation = (
        float(np.max(np.ptp(centred, axis=0))) if v.shape[0] > 1 else float(np.ptp(centred))
    )
    if separation <= 1e-9 * max(1.0, float(np.max(np.abs(v)))):
        raise ValueError(
            "these velocities never separate the components: once each star's own zero "
            "point is removed (the model does not identify it), every star has the same "
            "velocity at every epoch, which is exactly the cold start the free-velocity "
            "mode is measured to fail from (122,000 nats worse than a warm one). "
            "Supply the measured per-epoch velocities "
            "(cross-correlation lags, or line splitting read off the two most separated "
            "epochs) rather than a placeholder."
        )
    widest = dis._widest_lsf()
    if separation < widest:
        warnings.warn(
            f"the declared velocities, each star's zero point removed, separate the "
            f"components by at most {separation:.2f} km/s, which is below the widest LSF sigma "
            f"({widest:.2f} km/s); at no epoch are the two resolved. The free-velocity "
            "fit is warm-started from these, and a warm start inside the unresolved "
            "regime is close to the cold one measured failing. Check the sign "
            "convention and the epoch ordering before trusting the result.",
            RuntimeWarning,
            stacklevel=4,
        )
    return v


_NUMERIC_FIELDS = ("lo", "hi", "start_at", "value", "sigma", "values", "upper_bound")


def _spec_size(spec: Spec) -> int:
    """The number of entries of a spec: the largest size among its numeric fields.

    For a :class:`Sampled` spec the size of its distribution (batch and event shape)
    also counts, since that distribution sets the shape of the site.
    """
    sizes = [
        np.atleast_1d(np.asarray(getattr(spec, attr), dtype=float)).size
        for attr in _NUMERIC_FIELDS
        if getattr(spec, attr, None) is not None
    ]
    if isinstance(spec, Sampled):
        prior = spec.distribution_
        shape = tuple(prior.batch_shape) + tuple(prior.event_shape)
        sizes.append(math.prod(shape))
    return max(sizes, default=1)


def _as_k(value) -> Any:
    """A declared ``k``, with a bare number taken as ``Fixed`` as everywhere else."""
    if isinstance(value, int | float):
        return _coerce_spec(value, "orbit.k")
    return value


def _check_k_entries(value, n: int) -> None:
    """Reject one single-valued ``k`` specification declared for two or more stars.

    The ``k`` site has one entry per star. A specification whose fields are all scalars,
    and for :class:`Sampled` whose distribution is also scalar, would build a scalar
    site, and the model would then fail at its first evaluation with a message about
    ``len(k)``.
    """
    spec = _as_k(value)
    if n < 2 or not isinstance(spec, Spec) or isinstance(spec, Scanned):
        return
    if _spec_size(spec) == 1:
        raise ValueError(
            f"orbit.k is one specification with a single value, {spec}, for {n} stars, "
            "and the semi-amplitude site has one entry per star. Declare one specification "
            f"with {n} entries, e.g. Between([10.0] * {n}, [90.0] * {n}), or a sequence of "
            "one specification per star."
        )


def _k_specs(orbit: Orbit, n: int) -> list[Spec]:
    """The per-component ``k`` specifications, given either as one spec or a sequence."""
    k = _as_k(orbit.k)
    if isinstance(k, Spec):
        spec = k
        # A single spec may still be vector-valued, e.g. Between([10, 5], [90, 70]). Its
        # size is the largest among all of its numeric fields, not that of the first one
        # found: Between([10, 5], 90) has a scalar `hi` and two entries, and a vector Known
        # or Fixed has no `hi` and would otherwise be counted at its largest entry n times,
        # inflating the budget.
        if _spec_size(spec) == n:
            return [_ScalarView(spec, i) for i in range(n)]
        return [spec] * n
    specs = [_coerce_spec(item, "each entry of orbit.k") for item in k]
    if len(specs) != n:
        raise ValueError(f"orbit.k has {len(specs)} entries for {n} stars")
    return specs


@dataclass(frozen=True)
class _ScalarView(Spec):
    """One component of a vector-valued spec, for per-star budget accounting."""

    spec: Spec
    index: int

    def distribution(self):
        return self.spec.distribution()

    def start(self):
        return self.spec.start()

    def upper(self) -> float:
        bounds = _range_bounds(self)
        if bounds is not None:
            # The largest magnitude of this star's range: |lo| for a range below zero, as
            # in Between.upper, rather than |hi|.
            return max(abs(bounds[0]), abs(bounds[1]))
        for attr in ("hi", "value", "values"):
            values = getattr(self.spec, attr, None)
            if values is None:
                continue
            array = np.atleast_1d(np.asarray(values, dtype=float))
            if array.size > self.index:
                extra = 0.0
                if isinstance(self.spec, Known):
                    sigma = np.atleast_1d(np.asarray(self.spec.sigma, dtype=float))
                    extra = 5.0 * float(sigma[min(self.index, sigma.size - 1)])
                return float(abs(array[self.index]) + extra)
        return self.spec.upper()


# -- the declaration ----------------------------------------------------------


@dataclass(frozen=True)
class Disentangler:
    """A declared disentangling problem: components, orbit, instrument, and data.

    Construction derives the quantities the low-level interface otherwise requires,
    including the model grid and its margins, the solver's velocity budget, and the matched
    ``priors`` and ``init`` dictionaries. Quantities that are scientific claims are not
    derived. :meth:`explain` prints every derivation and :meth:`expert` returns the
    ``(model, priors, init, fixed)`` for direct use. The method that :meth:`fit` and
    :meth:`Fit.sample` run is described in ``docs/method.md``.

    Parameters
    ----------
    dataset
        From :func:`albireo.read_dataset`, :func:`albireo.load_example`, or built manually.
    components
        One :class:`Star` per stellar component, optionally a :class:`Telluric` and a
        :class:`Nebular`. The declared order is the model's component order, and results
        are retrieved by name afterwards.
    orbit
        An :class:`Orbit`. Exactly one of ``orbit`` and ``velocities`` is required.
    velocities
        The alternative declaration, for a binary whose orbit is not known: measured
        per-epoch velocities, ``(n_stellar, n_epochs)`` in km/s, from cross-correlation, a
        shift-and-add pipeline, or line splitting measured manually. The fit is then the
        free per-epoch RV table (``docs/math.md`` §7.6) rather than a Keplerian. No
        orbital elements are sampled, and :meth:`fit` returns a velocity-mode
        :class:`Fit` whose table can be searched for a period.

        The free table requires a warm start (a cold start was measured 122,000 nats
        worse), and the other warm start, :meth:`Fit.free_velocities`, requires a
        Keplerian fit and hence a period. For a system with no published period the
        sequence is: declare the measured velocities, obtain the table, derive the period,
        then declare an :class:`Orbit`.

        The declared velocities are a starting point, not a constraint. The per-component
        zero point is unidentified in either mode, so only their epoch-to-epoch pattern
        must be correct, not their absolute scale.
    lsf
        Per-instrument :class:`LSF`, or a bare sigma in km/s. Every instrument in the
        dataset must appear.
    dv_kms
        Model-grid pixel size in km/s. Defaults to the finest native sampling in the
        dataset.
    velocity_budget_kms
        Override for the derived bound, in km/s. An override smaller than the derived
        value raises, naming the terms that exceed it.
    ecc_max
        Eccentricity clip, the solver's verified range.
    block_size
        Solver block size, passed through to
        :class:`~albireo.inference.MarginalOrbitModel`.
    noise_correlation
        Lag-one correlation of each epoch's noise along its pixel index, produced by a
        pipeline that resampled the spectra onto a common step (Gaia's RVS grids have
        0.27 and 0.81; :mod:`albireo.gaia` measures it). ``None`` (default) is the
        diagonal noise model. A number, or ``{instrument: number}`` covering every
        instrument, declares the correlation and the noise model becomes AR(1) along the
        pixel index (:func:`albireo.forward.with_ar1`, ``docs/math.md`` §1.4a) with those
        values held. A :class:`Between` or :class:`Known` fits one shared value instead.
        The declared value is an assumption and is listed as one. It also enters
        :meth:`Fit.measure_velocities`, whose errors include it.

    Raises
    ------
    ValueError
        If no :class:`Star` is declared, if star names are not unique, if a light fraction
        is non-positive or the fractions do not sum to 1, if neither or both of ``orbit``
        and ``velocities`` are given, if more than one :class:`Telluric` or
        :class:`Nebular` component is declared, if an instrument in the dataset has no
        LSF, if a :class:`Telluric` or :class:`Nebular` component is declared for a
        dataset whose wavelength medium is undeclared, if a noise correlation omits
        an instrument or lies outside ``(-1, 1)``, if ``orbit.k`` is one single-valued
        specification for two or more stars, or if the declared velocities do not
        separate the stars once each star's zero point is removed.
    NotImplementedError
        If ``orbit.outer`` is set. Hierarchical triples are supported by
        :class:`~albireo.inference.MarginalOrbitModel` but not by this interface.

    Examples
    --------
    >>> import albireo as ab  # doctest: +SKIP
    >>> dis = ab.Disentangler(  # doctest: +SKIP
    ...     dataset,
    ...     components=[ab.Star("primary", light=0.62), ab.Star("secondary", light=0.38)],
    ...     orbit=ab.Orbit(period=ab.Between(5.5, 6.5), k=ab.Between([10.0, 10.0], [90.0, 90.0])),
    ...     lsf={"DEMO": 6.5},
    ... )
    >>> fit = dis.fit()  # doctest: +SKIP

    With no known orbit, the measured velocities are declared instead:

    >>> dis = ab.Disentangler(  # doctest: +SKIP
    ...     dataset,
    ...     components=[ab.Star("A", light=0.6), ab.Star("B", light=0.4)],
    ...     velocities=ccf_velocities,   # (2, n_epochs) km/s
    ...     lsf={"GIRAFFE": ab.LSF.from_resolution(6300)},
    ... )
    >>> table = dis.fit()               # a free-velocity Fit  # doctest: +SKIP
    >>> table.velocities(), table.velocity_errors()  # doctest: +SKIP
    """

    dataset: Dataset
    components: Sequence[Component]
    orbit: Orbit | None = None
    velocities: Any = None
    lsf: Mapping[str, Any] = field(default_factory=dict)
    dv_kms: float | None = None
    velocity_budget_kms: float | None = None
    ecc_max: float = 0.95
    block_size: int | None = None
    noise_correlation: Any = None

    # init=False so that dataclasses.replace() cannot copy a stale grid, budget or model
    # into a new declaration: replace() copies declared fields, and a cache is not one.
    _built: dict = field(default_factory=dict, repr=False, compare=False, init=False)

    def __post_init__(self) -> None:
        # Copy the caller's containers. They are declared as Sequence/Mapping, and a list
        # mutated after construction would desynchronize the cached model from the
        # assumptions this object reports.
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(self, "lsf", dict(self.lsf))
        stars = [c for c in self.components if isinstance(c, Star)]
        if not stars:
            raise ValueError("a Disentangler needs at least one Star component")
        names = [s.name for s in stars]
        if len(set(names)) != len(names):
            raise ValueError(f"star names must be unique; got {names}")
        bad = [s for s in stars if not (np.isfinite(s.light) and s.light > 0.0)]
        if bad:
            listed = ", ".join(f"{s.name}={s.light}" for s in bad)
            raise ValueError(
                f"every star's light fraction must be finite and positive; got {listed}. "
                "A component contributing no light is not a component: remove it."
            )
        total = sum(float(s.light) for s in stars)
        if abs(total - 1.0) > 1e-6:
            listed = ", ".join(f"{s.name}={s.light:g}" for s in stars)
            raise ValueError(
                f"the star light fractions must sum to 1; {listed} sums to {total:g}. "
                "This is an assumption the data cannot check: with constant light "
                "fractions the likelihood sees only l_i * d_i, so every recovered depth "
                "scales as 1/l_i."
            )
        if (self.orbit is None) == (self.velocities is None):
            raise ValueError(
                "declare exactly one of orbit= (a Keplerian to fit) and velocities= (the "
                "per-epoch velocities you measured, for a system whose orbit is not known "
                "yet). They are alternatives: a free velocity table replaces the orbit "
                "entirely, and the model rejects Keplerian sites alongside it."
            )
        if self.orbit is not None:
            _check_k_entries(self.orbit.k, len(stars))
        if self.orbit is not None and self.orbit.outer is not None:
            raise NotImplementedError(
                "hierarchical triples are not in the façade's v1 vocabulary. The model "
                "supports them (the period_out/t_conj_out/k_out sites), so build one "
                "through albireo.MarginalOrbitModel directly: Disentangler.expert() on a "
                "two-star declaration gives you the model, priors, init and fixed sites to "
                "start from."
            )
        if sum(isinstance(c, Telluric) for c in self.components) > 1:
            raise ValueError("at most one Telluric component")
        if sum(isinstance(c, Nebular) for c in self.components) > 1:
            raise ValueError("at most one Nebular component")

        missing = sorted(set(self.dataset.instruments) - set(self.lsf))
        if missing:
            raise ValueError(
                f"no LSF declared for instrument(s) {missing}. The dataset has "
                f"{sorted(self.dataset.instruments)}; pass one entry per instrument, e.g. "
                "lsf={'FEROS': ab.LSF.from_resolution(48_000)}."
            )
        for key in self.dataset.instruments:
            if _coerce_lsf(self.lsf[key], key).is_per_epoch:
                declared_lsf_widths(self.dataset, key)  # raises, naming undeclared epochs
        # The velocity check compares the table with the widest LSF, so it follows the LSF
        # checks above.
        if self.velocities is not None:
            object.__setattr__(self, "velocities", _check_velocities(self, stars))
        _check_noise_correlation(self)
        # A telluric or nebular component depends on absolute line positions, which an
        # undeclared wavelength medium leaves uncertain by a nearly constant 83 km/s.
        needs_medium = any(isinstance(c, Telluric | Nebular) for c in self.components)
        if needs_medium and self.dataset.epochs[0].medium is None:
            raise ValueError(
                "this dataset does not declare whether its wavelengths are air or vacuum, "
                "and a Telluric or Nebular component is keyed to absolute line positions. "
                "The difference is a nearly constant 83 km/s. Declare it with "
                "read_dataset(medium=...) or EpochData(medium=...)."
            )

    # -- derived quantities ---------------------------------------------------

    @property
    def stars(self) -> tuple[Star, ...]:
        """The stellar components, in model row order."""
        return tuple(c for c in self.components if isinstance(c, Star))

    @property
    def n_stellar(self) -> int:
        """The number of stellar components in the model."""
        return len(self.stars)

    @property
    def _nebular(self) -> Nebular | None:
        """The declared :class:`Nebular` component, or ``None``."""
        return next((c for c in self.components if isinstance(c, Nebular)), None)

    @property
    def _has_telluric(self) -> bool:
        """Whether a :class:`Telluric` component is declared."""
        return any(isinstance(c, Telluric) for c in self.components)

    @property
    def ordered_components(self) -> tuple[Component, ...]:
        """The components in model row order: stars, then telluric, then nebular.

        The model fixes this order regardless of the declaration, so every per-component
        array (the smoothness rows, the hyperprior centres, the assumptions report) is
        assembled through this property. Iterating the declaration instead misassigns rows
        without raising: the vectors have the correct length, so the length check in
        :func:`albireo.marginal_loglikelihood` passes, and a rotationally broadened star
        is regularized with a sharp-lined star's curvature penalty.
        """
        return (
            *self.stars,
            *(c for c in self.components if isinstance(c, Telluric)),
            *(c for c in self.components if isinstance(c, Nebular)),
        )

    @property
    def effective_ecc_max(self) -> float:
        """The eccentricity at which the model clips: the declared bound or the solver's.

        The ``(secosw, sesinw)`` sites are bounded independently, so their box reaches
        ``e = 2 * hi`` at the corner. The model's disk factor enforces the bound, so a
        declared ``ecc=Between(0, hi)`` must become the model's ``ecc_max``; otherwise the
        fit can return an eccentricity above the declared bound.
        """
        declared = self.ecc_max
        if self.orbit is None:  # a free-velocity declaration has no eccentricity
            return float(declared)
        if isinstance(self.orbit.ecc, Between):
            declared = min(declared, float(np.asarray(self.orbit.ecc.hi, dtype=float)))
        return float(declared)

    @property
    def component_names(self) -> tuple[str, ...]:
        """One name per model row, in row order: stars, then telluric, then nebular."""
        return tuple(
            c.name
            if isinstance(c, Star)
            else ("telluric" if isinstance(c, Telluric) else "nebular")
            for c in self.ordered_components
        )

    @property
    def velocity_budget(self) -> VelocityBudget:
        """The itemized relative-velocity bound from which the solver bandwidth is built."""
        return self._cached("budget", self._make_budget)

    @property
    def grid(self) -> LogGrid:
        """The model grid, wide enough for the largest shift plus the LSF kernel radius."""
        return self._cached("grid", self._make_grid)

    @property
    def smoothness_prior(self) -> SmoothnessPrior:
        """The smoothness prior, including any per-pixel confinement profile."""
        return self._cached("prior", self._make_prior)

    @property
    def model(self) -> MarginalOrbitModel:
        """The underlying :class:`~albireo.inference.MarginalOrbitModel`."""
        return self._cached("model", self._make_model)

    @property
    def priors(self) -> dict:
        """The numpyro prior per sampled site."""
        return self._cached("specs", self._make_specs)[0]

    @property
    def init(self) -> dict:
        """Starting values, with the same keys as :attr:`priors` by construction."""
        return self._cached("specs", self._make_specs)[1]

    @property
    def fixed(self) -> dict:
        """Sites injected as constants rather than sampled."""
        return self._cached("specs", self._make_specs)[2]

    def _cached(self, key, build):
        if key not in self._built:
            self._built[key] = build()
        return self._built[key]

    def _make_budget(self) -> VelocityBudget:
        return _velocity_budget(
            self.orbit,
            self.velocities,
            self.components,
            self.dataset.frame,
            self.velocity_budget_kms,
        )

    def _lsf_widths(self) -> dict[str, np.ndarray]:
        """Every LSF width in use, per instrument of the dataset, resolved."""
        return {
            key: _coerce_lsf(self.lsf[key], key).widths(self.dataset, key)
            for key in self.dataset.instruments
        }

    def _widest_lsf(self) -> float:
        return max(float(np.max(w)) for w in self._lsf_widths().values())

    def _narrowest_lsf(self) -> float:
        return min(float(np.min(w)) for w in self._lsf_widths().values())

    def _lsf_lines(self) -> list[str]:
        """One line per instrument for :meth:`explain`, giving the source of its width."""
        lines = []
        for key in self.dataset.instruments:
            spec = _coerce_lsf(self.lsf[key], key)
            if spec.is_per_epoch:
                counts = {
                    sigma: len(idx) for sigma, idx in declared_lsf_widths(self.dataset, key).items()
                }
                detail = ", ".join(f"{s:.3f} km/s x{n}" for s, n in counts.items())
                lines.append(f"  LSF        {key}: per epoch, from the files ({detail})")
            elif spec.anchors_angstrom is not None:
                lines.append(
                    f"  LSF        {key}: {len(spec.anchors_angstrom)} anchors, "
                    f"{float(np.min(spec.widths(self.dataset, key))):.3f}-"
                    f"{spec.max_sigma_kms:.3f} km/s"
                )
            else:
                lines.append(f"  LSF        {key}: {spec.max_sigma_kms:.3f} km/s, declared")
        return lines

    def _noise_line(self) -> str:
        """One line for :meth:`explain`: the noise model and the source of its correlation."""
        if self.noise_correlation is None:
            return "  noise      diagonal: independent pixels"
        by_instrument = self._declared_noise_correlation()
        if by_instrument is None:
            return (
                "  noise      AR(1) along the pixel index, one lag-one correlation fitted "
                "(site ar1_phi)"
            )
        listed = ", ".join(f"{k} {v:.3f}" for k, v in by_instrument.items())
        return f"  noise      AR(1) along the pixel index, lag-one correlation {listed} (declared)"

    def _native_dv_kms(self) -> float:
        """The finest native pixel size across the dataset, in km/s.

        The finest is used rather than the median: a model grid coarser than any
        contributing epoch discards that epoch's resolution, and on a mixed-instrument
        dataset the median follows the instrument with more epochs rather than the one
        that resolves more.
        """
        steps = []
        for epoch in self.dataset:
            wave = np.asarray(epoch.wave)
            steps.append(float(np.median(np.diff(wave) / wave[:-1]) * C_KMS))
        return float(np.min(steps))

    def _make_grid(self) -> LogGrid:
        dv = float(self.dv_kms) if self.dv_kms is not None else self._native_dv_kms()
        return LogGrid.covering(
            self.dataset,
            dv,
            v_margin_kms=self.velocity_budget.total,
            lsf_sigma_kms=self._widest_lsf(),
        )

    def _log_eta_floor(self) -> np.ndarray:
        """``ln eta_min`` of every component in model row order, ``-inf`` where none is declared."""
        bounds = [_smoothness_of(c).eta_min for c in self.ordered_components]
        return np.array([-np.inf if b is None else math.log(float(b)) for b in bounds])

    def _make_prior(self) -> SmoothnessPrior:
        tau = np.array([_smoothness_of(c).tau0 for c in self.ordered_components], dtype=float)
        eta = np.array([_smoothness_of(c).eta0 for c in self.ordered_components], dtype=float)
        eta_profile = None
        nebular = self._nebular
        if nebular is not None:
            wave = np.asarray(self.grid.wave)
            # nebular_windows takes rest wavelengths in air. On a vacuum grid the lines lie
            # 0.87-2.74 A redward of those values, so an unconverted window is offset by a
            # nearly constant 83 km/s. That is enough to clip the line it is meant to
            # contain, and the emission is absorbed into the stellar components. The
            # medium is therefore required before a Nebular component is accepted.
            lines = nebular.lines
            if self.dataset.epochs[0].medium == "vacuum":
                from albireo.grids import air_to_vacuum
                from albireo.priors import NEBULAR_LINES

                source = NEBULAR_LINES if lines is None else lines
                values = list(source.values()) if isinstance(source, Mapping) else list(source)
                lines = [float(v) for v in np.asarray(air_to_vacuum(np.asarray(values)))]
            # The model shifts the nebular row by nebular.v_kms (nebular_v_kms of
            # build_problem), so the component's own spectrum holds the lines at rest. The
            # windows confine that spectrum and are therefore built at rest, not at the
            # nebula's velocity; at a velocity beyond the half-width a window built there
            # misses the line altogether.
            windows = nebular_windows(
                lines=lines,
                halfwidth_kms=nebular.halfwidth_kms,
                v_kms=0.0,
                wave_range=(float(wave[0]), float(wave[-1])),
            )
            if not windows:
                raise ValueError(
                    f"the Nebular component's lines all fall outside the model grid "
                    f"({wave[0]:.1f}-{wave[-1]:.1f} A), so there is nowhere for it to have "
                    "structure and it would be pinned to the continuum everywhere: a free "
                    "component that can do nothing, costing solve time and one more "
                    "degeneracy. Either drop it, or pass Nebular(lines=[...]) with lines "
                    "that are in this range."
                )
            # Only the nebular row is confined; every other row keeps a flat profile.
            profile = np.ones((len(self.ordered_components), wave.size))
            profile[-1] = window_profile(wave, windows)
            eta_profile = profile
        return SmoothnessPrior(tau=tau, eta=eta, eta_profile=eta_profile)

    def _make_model(self) -> MarginalOrbitModel:
        lsf_sigma, anchors = {}, {}
        for key, value in self.lsf.items():
            spec = _coerce_lsf(value, key)
            lsf_sigma[key] = spec.sigma_kms
            if spec.anchors_angstrom is not None:
                anchors[key] = spec.anchors_angstrom
        nebular = self._nebular
        return MarginalOrbitModel(
            self.grid,
            self.dataset,
            light_fractions=[s.light for s in self.stars],
            lsf_sigma_v=lsf_sigma,
            lsf_anchors_angstrom=anchors or None,
            v_rel_max_kms=self.velocity_budget.total,
            telluric=self._has_telluric,
            nebular=nebular is not None,
            nebular_v_kms=0.0 if nebular is None else float(nebular.v_kms),
            prior=self.smoothness_prior,
            ecc_max=self.effective_ecc_max,
            block_size=self.block_size,
            ar1=self.noise_correlation is not None,
        )

    def noise_correlation_per_epoch(self) -> np.ndarray | None:
        """The declared lag-one noise correlation of every epoch, ``(n_epochs,)``.

        Returns ``None`` when no correlation is declared, or when it is a fitted site. A
        :class:`Fixed` correlation is held, as a bare number is, and is returned.
        """
        declared = self.noise_correlation
        if declared is None:
            return None
        if isinstance(declared, Spec):
            if declared.distribution() is not None:
                return None
            return np.full(self.dataset.n_epochs, float(np.asarray(declared.start())))
        if isinstance(declared, Mapping):
            return np.array([float(declared[epoch.instrument]) for epoch in self.dataset])
        return np.full(self.dataset.n_epochs, float(declared))

    def _declared_noise_correlation(self) -> dict[str, float] | None:
        """The held lag-one correlation per instrument, or ``None`` when none is held.

        ``None`` means either the diagonal noise model or a fitted correlation.
        """
        values = self.noise_correlation_per_epoch()
        if values is None:
            return None
        by_instrument: dict[str, float] = {}
        for epoch, phi in zip(self.dataset, values, strict=True):
            by_instrument.setdefault(epoch.instrument, float(phi))
        return by_instrument

    def _make_specs(self):
        """``(priors, init, fixed)``, built together so their key sets cannot diverge."""
        priors: dict[str, Any] = {}
        init: dict[str, Any] = {}
        fixed: dict[str, Any] = {}

        def place(name, spec):
            distribution = spec.distribution()
            if distribution is None:
                fixed[name] = spec.start()
            else:
                priors[name] = distribution
                init[name] = spec.start()

        if self.orbit is None:
            # There are no orbital sites: `velocity` replaces them, and the model rejects the
            # two together. The prior is centred on zero rather than on the declared table
            # because the absolute level is unidentified in either case. The declaration
            # supplies the starting value.
            sigma = self.velocity_budget.total / 2.0
            priors["velocity"] = (
                dist.Normal(0.0, sigma).expand(list(self.velocities.shape)).to_event(2)
            )
            init["velocity"] = jnp.asarray(self.velocities)
            _place_hyperparameters(self, priors, init)
            _place_noise_correlation(self, priors, init, fixed)
            return priors, init, fixed

        place("period", _coerce_spec(self.orbit.period, "orbit.period"))
        period = _largest(_start_of(self.orbit.period))
        if self.orbit.t_conj == "scan":
            # The site is always sampled, so fit() replaces this start with the scan result
            # and narrows the prior around it. expert() returns both unchanged. The prior
            # here starts at the first epoch rather than at zero: real epochs lie near BJD
            # 2.46e6, and a prior centred on the origin excludes every conjunction in the
            # span of the data.
            first = float(np.min(np.asarray(self.dataset.bjd)))
            priors["t_conj"] = dist.Uniform(first, first + period)
            init["t_conj"] = first + 0.5 * period
        else:
            spec = _coerce_spec(self.orbit.t_conj, "orbit.t_conj")
            if isinstance(spec, Between):
                width = float(np.max(np.asarray(spec.hi)) - np.min(np.asarray(spec.lo)))
                if width > 0.2 * period:
                    warnings.warn(
                        f"orbit.t_conj was declared as a range {width:g} d wide, which is "
                        f"{width / period:.0%} of a period, so the conjunction scan is "
                        "skipped and L-BFGS starts from its midpoint. The likelihood is "
                        "sharply multimodal in phase, and the neighbouring trough is the "
                        "component-swapped mirror orbit. Leave t_conj at its default "
                        "'scan' to locate it first.",
                        RuntimeWarning,
                        stacklevel=4,
                    )
            place("t_conj", spec)
        if _has_scanned(self.orbit.k):
            raise ValueError(
                "this declaration has a Scanned semi-amplitude, which is the axis of a K2 "
                "scan rather than a sampled site: there is no posterior over a grid. Call "
                "dis.scan() or dis.detection_limit(), or replace Scanned(...) with "
                "Between(...) to fit it."
            )
        place("k", _vector_spec(self.orbit.k, self.n_stellar, "orbit.k"))
        # An outer orbit is refused at construction, so there are no outer-orbit sites.
        for name, spec in _ecc_sites(self.orbit, self.ecc_max):
            place(name, spec)

        _place_hyperparameters(self, priors, init)
        _place_noise_correlation(self, priors, init, fixed)
        return priors, init, fixed

    # -- inspection -----------------------------------------------------------

    def expert(self):
        """The low-level ``(model, priors, init, fixed)`` this declaration compiles to.

        Features the declarative interface does not expose (per-epoch jitter, a per-epoch
        response polynomial, inferred light fractions, inferred LSF widths) are reached by
        modifying the returned dictionaries and calling
        :class:`~albireo.inference.MarginalOrbitModel` directly. Correlated (AR(1)) noise
        is exposed, through ``noise_correlation``.

        ``fixed`` holds the sites the declaration holds constant, such as ``secosw`` and
        ``sesinw`` of a circular orbit, and is empty when every site is sampled. A model
        rebuilt from the four takes it as :meth:`fit` does,
        ``model.model(priors, fixed=fixed)``. A site moved from ``fixed`` into ``priors``
        also needs a starting value in ``init``. ``init`` is the declared start, before the
        scans that :meth:`fit` runs to locate the conjunction phase and the
        semi-amplitudes, so an optimization started from it can converge in the wrong
        phase.

        Returns
        -------
        tuple
            ``(model, priors, init, fixed)``: the
            :class:`~albireo.inference.MarginalOrbitModel`, and copies of :attr:`priors`,
            :attr:`init` and :attr:`fixed`.
        """
        return self.model, dict(self.priors), dict(self.init), dict(self.fixed)

    def explain(self) -> str:
        """Report every derived quantity: budget, grid, operators, and sampled sites.

        Returns
        -------
        str
            A multi-line report ending with :meth:`assumptions`.
        """
        grid = self.grid
        derived = "" if self.dv_kms is not None else "  (derived from the native sampling)"
        return "\n".join(
            [
                f"Disentangler: {self.n_stellar} star(s), {self.dataset.n_epochs} epochs, "
                f"frame={self.dataset.frame!r}",
                "  components (model row order): " + ", ".join(self.component_names),
                "",
                str(self.velocity_budget),
                "",
                f"model grid   {grid.n} px, {grid.wave[0]:.2f}-{grid.wave[-1]:.2f} A, "
                f"dv={grid.dv_kms:.3f} km/s{derived}",
                f"  margin     {self.velocity_budget.total:.1f} km/s of shift + "
                f"{self._widest_lsf():.2f} km/s of LSF sigma",
                *self._lsf_lines(),
                self._noise_line(),
                f"  operators  {len(self.model.problem.groups)} group(s), "
                f"half-bandwidth {self.model.half_bandwidth}",
                "  (the bandwidth follows the budget, and the solve cost follows the "
                "bandwidth: narrowing the k or ecc priors is what makes a fit cheaper)",
                "",
                *self._site_lines(),
                "",
                self.assumptions(),
            ]
        )

    def _site_lines(self) -> list[str]:
        """The sampled and fixed site lines, or a note that this is a scan declaration."""
        try:
            priors, fixed = self.priors, self.fixed
        except ValueError as exc:
            return ["this is a scan declaration, not a fit declaration:", f"  {exc}"]
        lines = ["sampled sites: " + ", ".join(sorted(priors))]
        if fixed:
            lines.append("fixed sites:   " + ", ".join(sorted(fixed)))
        named = sorted(set(priors) | set(fixed))
        width = max(len(name) for name in named)
        lines += [
            f"  {name:<{width}s}  {_SITE_MEANINGS[name]}"
            for name in named
            if name in _SITE_MEANINGS
        ]
        return lines

    def _refuse_anchored_lsf(self, what: str) -> None:
        """Reject a wavelength-dependent LSF: ``k2_scan`` takes one width per instrument."""
        anchored = []
        for key, value in self.lsf.items():
            anchors = _coerce_lsf(value, key).anchors_angstrom
            if anchors is not None and len(anchors) > 0:
                anchored.append(key)
        if anchored:
            raise ValueError(
                f"{what}() takes one line-spread width per instrument, but {anchored} "
                "declared wavelength-dependent widths. Passing them would silently use only "
                "the first anchor. Declare a single representative sigma for the scan, or "
                "use albireo.k2_scan directly."
            )

    def _refuse_correlated_noise(self, what: str) -> None:
        """Reject a correlated noise model: ``k2_scan`` evaluates the diagonal one only.

        A declared correlation of zero is the diagonal model and is accepted.
        """
        if self.noise_correlation is None:
            return
        held = self._declared_noise_correlation()
        if held is not None and all(value == 0.0 for value in held.values()):
            return
        raise ValueError(
            f"{what}() evaluates the likelihood with independent pixel noise "
            "(albireo.k2_scan has no AR(1) noise model), but this declaration has "
            f"noise_correlation={self.noise_correlation!r}. Under correlated noise a "
            "white-noise scan over-counts the residuals, and a threshold calibrated with it "
            "understates the false-alarm rate. Declare noise_correlation=None to scan "
            "under the diagonal model."
        )

    def assumptions(self) -> str:
        """Report every declared quantity that the data cannot constrain.

        Returns
        -------
        str
            The ``Assumed, not measured`` block: the light fractions, the declared
            velocities in a free-velocity declaration, the nebular velocity where one is
            declared, and the smoothness starting values.
        """
        rows = ["Assumed, not measured:"]
        lights = "  ".join(f"{s.name}={s.light:g}" for s in self.stars)
        rows.append(
            f"  light fractions    {lights}\n"
            "      only l_i * d_i is observable, so every recovered depth scales as 1/l_i."
        )
        if self.orbit is None:
            centred = _centred_velocities(self.velocities)
            rows.append(
                "  declared velocities  "
                + "  ".join(
                    f"{s.name}: +/-{float(np.max(np.abs(row))):.1f} km/s"
                    for s, row in zip(self.stars, centred, strict=True)
                )
                + "\n      a warm start, not a constraint. The per-component zero point is "
                "unidentified,\n      so what these have to be right about is the "
                "epoch-to-epoch pattern, not the level."
            )
        nebular = self._nebular
        if nebular is not None:
            rows.append(
                f"  nebular velocity   {nebular.v_kms:g} km/s\n"
                "      unidentified: the shift from the component's lines at rest, where the "
                "window profile\n      confines them, to the observed ones; a placement "
                "convention, not a measurement."
            )
        by_instrument = self._declared_noise_correlation()
        if by_instrument is not None:
            listed = "  ".join(f"{k}={v:.3f}" for k, v in by_instrument.items())
            rows.append(
                f"  noise correlation  {listed}\n"
                "      the lag-one correlation of each epoch's noise along its pixels, "
                "declared from the\n      resampling that produced the spectra rather than "
                "measured here."
            )
        starts = "  ".join(
            f"{name}={_smoothness_of(c).tau0:g}"
            for name, c in zip(self.component_names, self.ordered_components, strict=True)
        )
        rows.append(
            f"  smoothness starts  {starts}\n"
            "      the starting curvature weight tau of each component's regularization. "
            "ML-II fits the\n      weights, but which optimum it finds depends on where "
            "tau starts."
        )
        return "\n".join(rows)

    # -- running it -----------------------------------------------------------

    def fit(
        self,
        *,
        max_steps: int = 300,
        tol: float | None = None,
        progress=None,
        k_scan: bool | str = "auto",
    ) -> Fit:
        """Locate the orbit and fit it: phase scan, semi-amplitude scan, then MAP with ML-II.

        The fit starts with a conjunction-phase scan over one period (unless ``t_conj`` was
        declared) and a coarse scan over every semi-amplitude declared as a range (see
        ``k_scan``). Each star's prior amplitude is then profiled at the located orbit, and
        the semi-amplitude scan is repeated where the profile moved a start by a factor of
        two or more. The phase scan is repeated at the chosen semi-amplitudes, and
        :func:`albireo.run_map` then runs over the orbital sites and the smoothness
        hyperparameters. With the spectra marginalized out, optimizing the hyperparameters
        is the ML-II step. The fitted hyperparameters are returned on :attr:`Fit.hyper`,
        keyed by component name, and :meth:`Fit.sample` holds them fixed.

        A ``velocities=`` declaration has no orbit and no phase to locate. The scan is
        skipped and the free per-epoch table is fitted directly, warm-started from the
        declared velocities. The result is a :class:`Fit` in ``"velocity"`` mode, read with
        :meth:`Fit.velocities` and :meth:`Fit.velocity_errors`. The implied period can
        then be used in a second declaration with an :class:`Orbit`.

        Parameters
        ----------
        max_steps
            L-BFGS iteration cap.
        tol
            Gradient-norm threshold. Defaults to a value scaled to the number of good
            pixels, because the potential's scale grows with that number and a fixed
            threshold is unreachable on a large dataset however good the fit is.
        progress
            ``callback(step, potential, grad_norm, params)``. Without one the fit produces
            no output, and a fit to real data can run for hours.
        k_scan
            ``"auto"`` (default) scans the marginal likelihood over a geometric grid of
            every semi-amplitude declared as a :class:`Between` range before optimizing,
            holding the others at their starting values. L-BFGS starts from the best trial
            when its likelihood exceeds that of the declared start. ``False`` skips the
            scan. The marginal likelihood is multimodal in the semi-amplitudes as it is in
            phase (a start at half the true value converges at half, and a start far above
            it in the static-component minimum), and L-BFGS does not cross between the
            basins. The grid is crossed with a grid of conjunction phases when the
            conjunction is being scanned, since the two are coupled, and refined around
            the best trial twice at half the spacing. Among trials within a few nats of
            the best, the one in the declared order (the first star moving least) is
            taken. The scans run on a copy of the declaration with the model grid at twice
            the pixel size (:meth:`_scan_declaration`), which makes the joint search
            affordable. The fit runs on the full grid. The scan is retained on
            :attr:`Fit.k_scan`.

        Returns
        -------
        Fit
            In ``"keplerian"`` mode when an :class:`Orbit` was declared, otherwise in
            ``"velocity"`` mode.

        Warns
        -----
        RuntimeWarning
            If the period prior is wide enough to constitute a period search, which a
            conjunction-phase scan is not.
        """
        if k_scan not in (False, True, "auto"):
            raise ValueError(f"k_scan must be 'auto', True or False; got {k_scan!r}")
        priors, init = dict(self.priors), dict(self.init)
        scan = None
        amplitude_scan = None
        # A free-velocity declaration has no orbit and therefore no phase to locate, so the
        # scan and the period warning are skipped.
        if self.orbit is not None:
            self._warn_if_the_period_prior_is_a_search()
            # With a semi-amplitude to scan, every scan runs on a coarser copy of this
            # declaration. The scans only select the basin, which the coarse grid does at
            # a fraction of the cost. L-BFGS then runs on the full model.
            ranged = bool(k_scan) and self._has_ranged_k()
            scanner = self._scan_declaration() if ranged else self
            if self.orbit.t_conj == "scan":
                scan = scanner._scan_phase(init)
                _require_a_finite_trial(scan)
                init["t_conj"] = scan.best
            if ranged:
                amplitude_scan = scanner._scan_semi_amplitudes(init)
                # Whether either pass of the scan moved the start. The second pass is
                # measured from the first pass's best, so its gain alone does not say.
                moved = False
                if amplitude_scan is not None and amplitude_scan.gain > 0.0:
                    moved = True
                    init["k"] = jnp.asarray(amplitude_scan.best)
                    if self.orbit.t_conj == "scan":
                        init["t_conj"] = amplitude_scan.best_t_conj
                # The prior amplitude of each component's spectrum is profiled at the
                # orbit located so far. Where the profile maximum is a factor of two or
                # more from the declared start, the hyperparameter starts are moved and
                # the scan is repeated, because a scan at the wrong amplitude has its
                # maximum at a static companion (see _profile_prior_amplitudes).
                if amplitude_scan is not None:
                    profile = scanner._profile_prior_amplitudes(init)
                    adapted = {
                        j: c for j, c in profile.items() if abs(math.log(c)) >= math.log(2.0)
                    }
                    if adapted:
                        log_tau = np.array(init["log_tau"], dtype=float)
                        log_eta = np.array(init["log_eta"], dtype=float)
                        for j, c in adapted.items():
                            # Star j is model row j: the stars occupy the first rows of
                            # ordered_components, ahead of a telluric or nebular row.
                            log_tau[j] -= 2.0 * math.log(c)
                            log_eta[j] -= 2.0 * math.log(c)
                        init["log_tau"] = jnp.asarray(log_tau)
                        # A declared bound on eta holds for the adapted start as well.
                        init["log_eta"] = jnp.asarray(_above_floor(log_eta, self._log_eta_floor()))
                        first_best = ", ".join(
                            f"{s.name} {k:.1f}"
                            for s, k in zip(self.stars, amplitude_scan.best, strict=True)
                        )
                        words = "; ".join(
                            f"{self.stars[j].name} by {1.0 / c**2:.3g} (light-equivalent "
                            f"{c * self.stars[j].light:.3f} against the declared "
                            f"{self.stars[j].light:.3f})"
                            for j, c in adapted.items()
                        )
                        second = scanner._scan_semi_amplitudes(init)
                        if second is not None:
                            amplitude_scan = replace(
                                second,
                                prior_scales={
                                    self.stars[j].name: float(1.0 / c**2)
                                    for j, c in adapted.items()
                                },
                                notes=(
                                    f"prior amplitude precision adapted, {words}; the first "
                                    f"scan had ended at {first_best} km/s and was repeated",
                                    *second.notes,
                                ),
                            )
                            if second.gain > 0.0:
                                moved = True
                                init["k"] = jnp.asarray(second.best)
                                if self.orbit.t_conj == "scan":
                                    init["t_conj"] = second.best_t_conj
                if moved:
                    if self.orbit.t_conj == "scan":
                        # The joint scan located the phase coarsely. The fine phase scan
                        # at the chosen semi-amplitudes refines it.
                        scan = scanner._scan_phase(init)
                        _require_a_finite_trial(scan)
                        init["t_conj"] = scan.best
            if scan is not None:
                # The prior is one period wide and centred on the conjunction the fit starts
                # from, not on the phase scan's best. The semi-amplitude scan may have moved
                # the start by half a period, at semi-amplitudes the phase scan did not use,
                # with no fine phase scan after it. A start on the edge of the window has an
                # infinite unconstrained coordinate (D63).
                centre = float(np.asarray(init["t_conj"], dtype=float))
                priors["t_conj"] = dist.Uniform(
                    centre - 0.5 * scan.period, centre + 0.5 * scan.period
                )
        if tol is None:
            # The potential's scale grows with the number of good pixels, so a fixed
            # threshold is unreachable on a large dataset however good the fit is.
            n_good = sum(int(np.asarray(epoch.good).sum()) for epoch in self.dataset)
            tol = max(1e-2, 1e-6 * n_good)

        result = run_map(
            self.model.model(priors, fixed=self.fixed or None),
            init=init,
            max_steps=max_steps,
            tol=tol,
            callback=progress,
            model_args=(self.model.problem,),
        )
        hyper = _hyper_of(result.params, self.component_names)
        return Fit(
            dis=self,
            result=result,
            hyper=hyper,
            phase_scan=scan,
            mode="keplerian" if self.orbit is not None else "velocity",
            priors_used=priors,
            k_scan=amplitude_scan,
            max_steps=max_steps,
        )

    def _warn_if_the_period_prior_is_a_search(self) -> None:
        """Warn when the period prior is wide enough to constitute a period search.

        The conjunction scan resolves phase at one period, the period prior's midpoint.
        With a prior that wide, the scan locates a phase for a period that may be far from
        the true one, leaving L-BFGS to cross the multimodal structure the scan is meant to
        avoid. The degradation is gradual, so this warns rather than raises.
        """
        if self.orbit is None:  # no period prior, so nothing here can be a search
            return
        spec = _coerce_spec(self.orbit.period, "orbit.period")
        if not isinstance(spec, Between):
            return
        lo = float(np.min(np.asarray(spec.lo, dtype=float)))
        hi = float(np.max(np.asarray(spec.hi, dtype=float)))
        if lo > 0 and (hi - lo) / lo > 0.2:
            warnings.warn(
                f"the period prior spans {lo:g} to {hi:g} d, which is {(hi - lo) / lo:.0%} of "
                "its own lower bound. The conjunction scan resolves *phase* at one period; it "
                "is not a period search, and it runs at the prior's midpoint. Narrow the prior "
                "or run a periodogram first, or the fit starts from a phase located for the "
                "wrong period.",
                RuntimeWarning,
                stacklevel=3,
            )

    def _profile_prior_amplitudes(
        self, init, *, ratio: float = 2.0, span: float = 16.0
    ) -> dict[int, float]:
        """Profile the marginal likelihood over each star's light-equivalent amplitude.

        The observed spectrum depends on a component only through the product of its
        light fraction and its deviation spectrum, and the deviation has a Gaussian
        prior of precision ``tau C + eta I``. The marginal likelihood is therefore exactly
        invariant under scaling the light by ``c`` and both hyperparameters by ``c``
        squared (D63: to 5e-10 nats). A profile over the light at fixed hyperparameters
        is thus a profile over that component's prior amplitude, not a measurement of its
        flux fraction (on four benchmark systems its maximum was at 0.3 to 0.7 of the
        injected fraction). A scan requires this profile. With a companion declared at
        six times its light and the hyperparameters at their starts, a static secondary
        was 43 nats higher in the coarse scan, and at the same orbit the declared
        amplitude was 150 nats below the profile maximum.

        For each star in turn the light-equivalent amplitude is scaled by ``c`` over a
        geometric grid from ``1/span`` to ``span`` at ``ratio`` between neighbours (the
        scaled fraction ``c`` times the declared light kept below 0.98), with the other
        stars and every other site at their ``init`` values. The ``c`` with the highest
        marginal log-likelihood is returned. Through the invariance the scaling is applied
        to the star's ``log_tau`` and ``log_eta``, each lowered by ``2 ln c``, rather than
        to its light: ``theta`` then keeps the sites of the scans, and the marginal
        compiled for them is reused. The two forms agree to about 1e-10 nats. A star takes
        one evaluation per trial, nine at most: five at a light of 0.62 and six at 0.38
        with the defaults.
        """
        theta = self._theta_at(init)
        lights = np.array([s.light for s in self.stars], dtype=float)
        log_tau = np.array(theta["log_tau"], dtype=float)
        log_eta = np.array(theta["log_eta"], dtype=float)
        n = max(3, math.ceil(2.0 * math.log(span) / math.log(ratio)) + 1)
        grid = np.geomspace(1.0 / span, span, n)
        grid = np.unique(np.concatenate([grid, [1.0]]))
        out: dict[int, float] = {}
        for j in range(self.n_stellar):
            trials = grid[grid * lights[j] <= 0.98]
            values = []
            for c in trials:
                # Star j is model row j (ordered_components puts the stars first).
                tau_c, eta_c = log_tau.copy(), log_eta.copy()
                tau_c[j] -= 2.0 * math.log(c)
                eta_c[j] -= 2.0 * math.log(c)
                at = {**theta, "log_tau": jnp.asarray(tau_c), "log_eta": jnp.asarray(eta_c)}
                values.append(float(self.model.log_likelihood(at)))
            values = np.asarray(values)
            if not np.isfinite(values).any():
                out[j] = 1.0
                continue
            out[j] = float(trials[int(np.argmax(np.where(np.isfinite(values), values, -np.inf)))])
        return out

    def _theta_at(self, init) -> dict:
        """``init`` with the fixed sites added, as JAX arrays: the ``theta`` of the scans."""
        theta = {k: jnp.asarray(v) for k, v in init.items()}
        theta.update({k: jnp.asarray(v) for k, v in self.fixed.items()})
        return theta

    def _start_period(self, init) -> float:
        """The period at which the scans run: the start in ``init``, or the fixed value."""
        return _largest(init.get("period", self.fixed.get("period")))

    def _scan_phase(self, init) -> PhaseScan:
        """Locate conjunction on a 42-point grid over one period, before optimizing.

        The trials are ``min(bjd) + linspace(0, P, 42, endpoint=False)``, one period long
        and equally spaced. The count is even so that the grid contains the antipode of
        every trial: trial ``i`` and trial ``i + 21`` are half a period apart. For a
        near-equal pair the marginal likelihood in phase is near-mirror-symmetric under a
        shift of half a period, so the scan distinguishes the two mirrors only if both
        are on the grid. An odd count leaves every antipode midway between two trials, and
        on one benchmark star the unsampled antipode was higher by 672 nats (D63).

        The marginal likelihood is sharply multimodal in phase, and L-BFGS does not cross
        between the troughs. A trial whose likelihood is not finite is chosen only when no
        trial is finite.
        """
        period = self._start_period(init)
        trials = float(np.min(self.dataset.bjd)) + np.linspace(0.0, period, 42, endpoint=False)
        theta = self._theta_at(init)
        # The trials are evaluated one at a time through the already-compiled marginal
        # rather than in a batched sweep. A 42-wide sweep is one more large XLA compile
        # per model shape, and on Windows a long session of compiles has ended in a heap
        # corruption inside the compiler (D62). The loop compiles nothing new.
        values = []
        for t in trials:
            theta["t_conj"] = jnp.asarray(float(t))
            values.append(float(self.model.log_likelihood(theta)))
        scanned = np.asarray(values)
        # np.argmax takes the first NaN as the maximum, so a trial whose likelihood is not
        # finite would become the start. Such trials rank below every finite one. With no
        # finite trial the first is returned; fit() then raises (_require_a_finite_trial),
        # and the pipeline's period comparison ranks the candidate last.
        finite = np.where(np.isfinite(scanned), scanned, -np.inf)
        best = float(trials[int(np.argmax(finite))])
        return PhaseScan(period=period, trials=trials, values=scanned, best=best)

    def _scan_declaration(self) -> Disentangler:
        """This declaration on a model grid twice as coarse, for the scans that precede a fit.

        A scan only selects the basin that contains the maximum. A grid at twice the pixel
        size halves the pixel count and the solver bandwidth, and a marginal solve costs a
        quarter to an eighth of the full model's, which makes a joint scan over the
        semi-amplitudes and the phase affordable.

        The copy is built once and kept with this declaration, so a repeated :meth:`fit`
        reuses its model and the programs compiled for it.
        """
        return self._cached(
            "scan_declaration", lambda: replace(self, dv_kms=2.0 * float(self.grid.dv_kms))
        )

    def _scan_batch(self) -> int:
        """Trials per compiled batch of the semi-amplitude scan: 16, or fewer above 1 GiB.

        One trial of the sweep holds about ``_SCAN_BYTES_PER_BAND_ENTRY`` bytes for each
        entry of the ``(n, half_bandwidth)`` band per component, with ``n`` the number of
        components times the grid size (measured from XLA's buffer assignment). The batch
        is halved from 16 until its temporaries fit the sweep budget of
        :mod:`albireo.inference`. A divisor of 16 pads the last chunk of a sweep to no more
        trials than 16 would: on a 2,509-pixel coarse grid the 13 trials that would fit
        evaluated 273 trials for the scan's 208, and 8 evaluate 224, as 16 do. On a problem
        of a few thousand model pixels the batch is 16, the size the scan has always used;
        on a coarse grid of 4,309 pixels at half-bandwidth 156 it is 4.
        """
        problem = self.model.problem
        n_components = int(problem.n_components)
        band = n_components * int(problem.grid.n) * max(int(self.model.half_bandwidth), 1)
        per_trial = _SCAN_BYTES_PER_BAND_ENTRY * n_components * band
        batch = _SCAN_BATCH
        while batch > 1 and batch * per_trial > _SWEEP_BATCH_BYTES:
            batch //= 2
        return batch

    def _has_ranged_k(self) -> bool:
        if self.orbit is None:
            return False
        return any(_range_bounds(s) is not None for s in _k_specs(self.orbit, self.n_stellar))

    def _scan_semi_amplitudes(
        self,
        init,
        *,
        ratio: float = 1.25,
        n_phases: int = 8,
        levels: int = 2,
        max_trials: int = 1024,
        tie_nats: float = 5.0,
        hold_nats: float = 25.0,
        indifferent_nats: float = 5.0,
    ) -> SemiAmplitudeScan | None:
        """Scan the marginal likelihood over the ranged semi-amplitudes, and the phase.

        The axes are scanned one at a time rather than as a product. The first ranged
        component (declared first, so the brighter or heavier star) is scanned on a
        geometric grid over its range (2 percent inside either bound, ``ratio`` between
        neighbours, coarsened until the level fits ``max_trials``). The grid is crossed
        with ``n_phases`` conjunctions over one period (or the single declared conjunction
        in ``init``), with the other components at their starting values. Each further
        ranged component is then scanned over its own grid at the semi-amplitudes and
        conjunction located so far. Each of ``levels`` refinements halves the spacings
        around the best trial, three values per axis, jointly over the ranged components
        and the phase. A final pass scans each component once more over its whole grid at
        the refined best, and two more refinements follow if the best trial moved. Among
        the trials within ``tie_nats`` of the best, the one in the declared order
        (semi-amplitudes non-decreasing) is taken. With two ranged components the
        exchange-symmetric twin of the best (the semi-amplitudes swapped, the conjunction
        half a period later) is tried for the same reason. Returns ``None`` when no
        semi-amplitude is a range.

        The order follows from a measurement (D63). A companion with a few percent of the
        light changes the log-likelihood by about 25 nats over its whole semi-amplitude
        range, and its maximum is at the true semi-amplitude only when the primary's
        semi-amplitude is within about 10 percent of the true value. The primary's own
        maximum drops by 50 nats within 20 percent. On a product grid with a ratio of 2
        between trials, no trial places the primary close enough for the companion's
        likelihood to peak at the true value, and a local refinement does not move the
        companion out of the basin in which it starts. The phase is scanned with the first
        component because the two are coupled: a phase located at a semi-amplitude wrong
        by a factor of three can be a quarter of a period from the true phase.

        Two checks limit the harm of scanning a coarse model of the wrong shape (a free
        eccentricity starts near circular, the grid has twice the pixel size). A best
        trial with every ranged semi-amplitude at the floor of its grid is the
        static-component minimum, which a wrong phase or eccentricity makes more likely
        than any moving pair. It is not a start, and the declared start is kept. The start
        also moves only when the best trial exceeds it by more than ``hold_nats`` jointly,
        because the moves are joint: tested one component at a time, the same companion's
        move separated into two holds of 15 nats each and was rejected. A component whose
        own move gains less than ``indifferent_nats``, with the others at the scan's best,
        returns to its start, since the data do not distinguish the two and a template
        table is the better guess. The hold losses are recorded in either case.
        """
        specs = _k_specs(self._require_orbit(), self.n_stellar)
        ranges = {i: _range_bounds(s) for i, s in enumerate(specs)}
        ranged = [i for i, r in ranges.items() if r is not None]
        if not ranged:
            return None
        start = np.atleast_1d(np.asarray(init["k"], dtype=float))
        bounds: dict[int, tuple[float, float]] = {}
        for i in ranged:
            lo, hi = ranges[i]
            inner = (lo + 0.02 * (hi - lo), hi - 0.02 * (hi - lo))
            bounds[i] = (max(inner[0], 1e-3 * hi), inner[1])
        period = self._start_period(init)
        t_conj = float(np.asarray(init.get("t_conj", self.fixed.get("t_conj")), dtype=float))
        scanning_phase = self.orbit.t_conj == "scan"
        theta = self._theta_at(init)
        theta["t_conj"] = jnp.asarray(t_conj)
        history: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        batch = self._scan_batch()

        def evaluate(k_trials, t_trials):
            # The trials go through the compiled sweep in chunks of exactly `batch`, the
            # last one padded with copies of its final trial. A sweep is compiled once per
            # trial count, and the calls below have five different counts; with one chunk
            # size, one program serves the whole scan.
            k_trials = np.asarray(k_trials, dtype=float)
            t_trials = np.asarray(t_trials, dtype=float)
            values = np.empty(t_trials.size)
            for lo in range(0, t_trials.size, batch):
                k_part, t_part = k_trials[lo : lo + batch], t_trials[lo : lo + batch]
                m = t_part.size
                if m < batch:
                    k_part = np.concatenate([k_part, np.repeat(k_part[-1:], batch - m, axis=0)])
                    t_part = np.concatenate([t_part, np.repeat(t_part[-1:], batch - m)])
                sweep = {"k": jnp.asarray(k_part), "t_conj": jnp.asarray(t_part)}
                part = self.model.log_likelihood_sweep(theta, sweep, batch_size=batch)
                values[lo : lo + m] = np.asarray(part, dtype=float)[:m]
            return values

        def value_at(k, t):
            at = {**theta, "k": jnp.asarray(k), "t_conj": jnp.asarray(float(t))}
            return float(self.model.log_likelihood(at))

        def ordered(k_trials):
            # Components are declared in order of decreasing mass, so the first moves
            # least. A trial follows that order when its semi-amplitudes do not decrease.
            return np.all(np.diff(k_trials, axis=1) >= 0.0, axis=1)

        def choose(k_trials, values):
            # Among the trials within tie_nats of the best, take the one in the declared order.
            finite = np.isfinite(values)
            top = float(np.max(values[finite]))
            near = finite & (values >= top - tie_nats)
            candidates = np.flatnonzero(near & ordered(k_trials))
            if candidates.size == 0:
                candidates = np.flatnonzero(near)
            return int(candidates[np.argmax(values[candidates])])

        def trials(k_grid, phases):
            k_all = np.repeat(k_grid, len(phases), axis=0)
            t_all = np.tile(np.asarray(phases, dtype=float), k_grid.shape[0])
            return k_all, t_all

        def axis_of(i, r):
            lo, hi = bounds[i]
            return np.geomspace(lo, hi, max(3, math.ceil(math.log(hi / lo) / math.log(r)) + 1))

        n_first = n_phases if scanning_phase else 1
        while True:
            axes = {i: axis_of(i, ratio) for i in ranged}
            n_level = len(axes[ranged[0]]) * n_first + sum(len(axes[i]) for i in ranged[1:])
            if n_level <= max_trials:
                break
            ratio *= 1.2

        def sweep(i, at_k, phases):
            # One component is scanned over its whole grid, the others at their current values.
            k_grid = np.repeat(at_k[None, :], len(axes[i]), axis=0)
            k_grid[:, i] = axes[i]
            k_all, t_all = trials(k_grid, phases)
            values = evaluate(k_all, t_all)
            history.append((k_all, t_all, values))
            return k_all, t_all, values

        def argmax(values):
            return int(np.argmax(np.where(np.isfinite(values), values, -np.inf)))

        # Level 0: the first ranged component is scanned with the phase, then each further
        # one at the values located so far.
        best_k, best_t, best_value = start.copy(), t_conj, -math.inf
        for n, i in enumerate(ranged):
            phases = (
                t_conj + np.arange(n_phases) * period / n_phases
                if n == 0 and scanning_phase
                else np.array([best_t])
            )
            k_all, t_all, values = sweep(i, best_k, phases)
            if not np.isfinite(values).any():
                if n == 0:
                    return None
                continue
            chosen = argmax(values)
            best_k, best_t = k_all[chosen].copy(), float(t_all[chosen])
            best_value = float(values[chosen])

        t_step = period / n_phases if scanning_phase else 0.0

        def refine(k_step, t_step):
            nonlocal best_k, best_t, best_value
            axes_fine = []
            for i in ranged:
                lo, hi = bounds[i]
                centre = math.log(best_k[i])
                offsets = centre + k_step * np.array([-1.0, 0.0, 1.0])
                axes_fine.append(np.exp(np.clip(offsets, math.log(lo), math.log(hi))))
            mesh = np.meshgrid(*axes_fine, indexing="ij")
            k_grid = np.repeat(best_k[None, :], mesh[0].size, axis=0)
            k_grid[:, ranged] = np.stack([m.ravel() for m in mesh], axis=1)
            phases = (
                best_t + t_step * np.array([-1.0, 0.0, 1.0])
                if scanning_phase
                else np.array([best_t])
            )
            k_all, t_all = trials(k_grid, phases)
            values = evaluate(k_all, t_all)
            history.append((k_all, t_all, values))
            if np.isfinite(values).any():
                chosen = choose(k_all, values)
                if float(values[chosen]) >= best_value:
                    best_k, best_t = k_all[chosen].copy(), float(t_all[chosen])
                    best_value = float(values[chosen])

        k_step = math.log(ratio)
        for _ in range(levels):
            k_step /= 2.0
            t_step /= 2.0
            refine(k_step, t_step)

        # The full-axis pass: a component placed in the wrong basin at level 0, while
        # another was still far from its value, is scanned over its whole grid once more
        # at the refined best. A move is followed by two further refinements.
        moved = False
        for i in ranged:
            k_all, _, values = sweep(i, best_k, np.array([best_t]))
            if np.isfinite(values).any():
                chosen = argmax(values)
                if float(values[chosen]) > best_value:
                    best_k, best_value = k_all[chosen].copy(), float(values[chosen])
                    moved = True
        if moved:
            refine(math.log(ratio) / 2.0, t_step)
            refine(math.log(ratio) / 4.0, t_step)

        if len(ranged) == 2 and scanning_phase:
            # The exchange-symmetric twin of the best: with the two semi-amplitudes swapped
            # and the conjunction half a period later, similar components are fitted
            # equally well, and the declared order breaks the tie.
            a, b = ranged
            swapped = best_k.copy()
            swapped[a], swapped[b] = best_k[b], best_k[a]
            t_swapped = best_t + 0.5 * period
            value = value_at(swapped, t_swapped)
            history.append((swapped[None, :], np.array([t_swapped]), np.array([value])))
            if (
                np.isfinite(value)
                and value >= best_value - tie_nats
                and bool(ordered(swapped[None, :])[0])
                and not bool(ordered(best_k[None, :])[0])
            ):
                best_k, best_t, best_value = swapped, t_swapped, float(value)

        start_value = float(self.model.log_likelihood({**theta, "k": jnp.asarray(start)}))
        notes: list[str] = []
        hold_losses: dict[int, float] = {}
        floor = np.array([bounds[i][0] for i in ranged])
        if np.all(np.isclose(best_k[ranged], floor)):
            notes.append(
                "every ranged semi-amplitude of the best trial sits at the floor of its grid, "
                "the static-component minimum; the declared start is kept"
            )
            best_k, best_t, best_value = start.copy(), t_conj, start_value
        elif best_value - start_value < hold_nats:
            notes.append(
                f"the best trial beats the start by {best_value - start_value:.1f} nats, "
                f"below {hold_nats:g}; the declared start is kept"
            )
            best_k, best_t, best_value = start.copy(), t_conj, start_value
        else:
            # The move is taken jointly. A component whose own move gains less than
            # indifferent_nats, with the others at the scan's best, returns to its start.
            held_theta = {**theta, "t_conj": jnp.asarray(best_t)}
            for i in ranged:
                if np.isclose(best_k[i], start[i]) or not (
                    bounds[i][0] <= start[i] <= bounds[i][1]
                ):
                    continue
                held = best_k.copy()
                held[i] = start[i]
                value = float(self.model.log_likelihood({**held_theta, "k": jnp.asarray(held)}))
                hold_losses[i] = best_value - value
                if hold_losses[i] < indifferent_nats:
                    best_k = held
                    best_value = value
                    notes.append(
                        f"component {i} kept at its start of {start[i]:.1f} km/s: holding it "
                        f"there costs {hold_losses[i]:.1f} nats, below {indifferent_nats:g}"
                    )
        return SemiAmplitudeScan(
            k=np.concatenate([h[0] for h in history]),
            t_conj=np.concatenate([h[1] for h in history]),
            values=np.concatenate([h[2] for h in history]),
            best=best_k,
            best_t_conj=best_t,
            best_value=best_value,
            start=start.copy(),
            start_value=start_value,
            dv_kms=float(self.grid.dv_kms),
            hold_losses={int(i): float(v) for i, v in hold_losses.items()},
            notes=tuple(notes),
        )

    # -- the SB1 workflow -----------------------------------------------------

    def _scan_orbit(self) -> dict:
        """The fixed SB1 orbit held by the scan, taken from the declared specifications."""
        orbit = self._require_orbit()
        missing = []
        for name in ("period", "t_conj"):
            declared = getattr(orbit, name)
            # t_conj defaults to the string "scan", which is a phase search rather than a
            # value, and a scan holds the primary's orbit fixed by definition.
            if isinstance(declared, str) or not isinstance(_coerce_spec(declared, name), Fixed):
                missing.append(name)
        if missing:
            raise ValueError(
                f"a K2 scan holds the primary's orbit fixed, so {missing} must be declared "
                "Fixed(...). The scan asks 'is there a companion at this K2', which is only "
                "meaningful at one orbit; fit the SB1 first, then scan at its solution."
            )
        values = {
            "period": float(np.asarray(_coerce_spec(orbit.period, "period").start())),
            "t_conj": float(np.asarray(_coerce_spec(orbit.t_conj, "t_conj").start())),
        }
        for name, spec in _ecc_sites(orbit, self.ecc_max):
            if not isinstance(spec, Fixed):
                raise ValueError(
                    "a K2 scan needs a fixed eccentricity: pass ecc=ab.Fixed(e) with "
                    "omega=ab.Fixed(w), or ecc=ab.Fixed(0.0) for a circular orbit."
                )
            values[name] = float(np.asarray(spec.start()))
        return values

    def _require_orbit(self) -> Orbit:
        """The declared :class:`Orbit`, or an error, for the paths that require one.

        It is shared by the scan entry points so that the message is written once and the
        type narrowing is explicit.
        """
        if self.orbit is None:
            raise ValueError(
                "a K2 scan searches over a companion's semi-amplitude within a *known* "
                "SB1 orbit, so it needs orbit=Orbit(period=..., k=(Fixed(k1), "
                "Scanned(grid))). This declaration supplied velocities= instead, which "
                "replaces the orbit rather than constraining it. Fit the free table "
                "first, get a period from it, then declare the orbit and scan."
            )
        return self.orbit

    def _scan_k(self):
        """``(k1_spec, k2_spec)`` from a two-star declaration, checked for shape and kind.

        It is shared by :meth:`scan` and :meth:`detection_limit`, so the no-orbit error is
        raised here.
        """
        specs = _k_specs(self._require_orbit(), self.n_stellar)
        if self.n_stellar != 2:
            raise ValueError(
                f"a K2 scan is the two-component workflow; this declaration has "
                f"{self.n_stellar} stars."
            )
        k1, k2 = specs
        if not isinstance(k2, Scanned):
            raise ValueError(
                "declare the companion's semi-amplitude as ab.Scanned(grid); that grid is "
                "the scan's axis. The primary's is Fixed(k) to hold it, or Known(k, sigma) "
                "to marginalize over it, which is the only thing that catches a wrong K1 "
                "inflating the detection statistic."
            )
        if not isinstance(k1, Fixed | Known):
            raise ValueError("the primary's semi-amplitude must be Fixed(k) or Known(k, sigma)")
        return k1, k2

    def scan(self, *, block_size: int | None = None, sweep_batch: int | None = None):
        """Scan trial companion semi-amplitudes against the no-companion model.

        This SB1 faint-companion search profiles the marginal likelihood over the
        companion's velocity semi-amplitude, which detects a companion that never appears
        as a second set of lines. Declaring the primary's ``k`` as :class:`Known` rather
        than :class:`Fixed` marginalizes over it. A K₁ 10% high reduces the correlation of
        the recovered companion with the injected one from 0.96 to 0.49 while tripling the
        detection statistic, so the artifact appears as a stronger detection that no
        calibrated threshold identifies.

        Parameters
        ----------
        block_size
            Solver block size for the scan. Defaults to the declaration's ``block_size``.
        sweep_batch
            Number of trial semi-amplitudes evaluated per batch. Passed to
            :func:`albireo.k2_scan`.

        Returns
        -------
        albireo.scan.K2ScanResult

        Raises
        ------
        ValueError
            If the declaration has no :class:`Orbit`, does not have exactly two stars,
            does not declare the companion's ``k`` as :class:`Scanned` and the primary's
            as :class:`Fixed` or :class:`Known`, declares a wavelength-dependent LSF, or
            declares a nonzero ``noise_correlation``. The scan evaluates the diagonal
            noise model only.
        """
        from albireo.scan import k2_scan

        return k2_scan(
            self.grid,
            self.dataset,
            **self._k2_scan_kwargs("scan"),
            block_size=block_size if block_size is not None else self.block_size,
            sweep_batch=sweep_batch,
        )

    def detection_limit(self, **kwargs):
        """Injection-recovery calibration for :meth:`scan`, on this same declaration.

        Companions are injected into resimulations of this dataset, through its own
        operators, and recovered with :meth:`scan`, giving the light fraction above which
        a companion would have been detected at a stated confidence. Sharing the
        :class:`Disentangler` with :meth:`scan` guarantees identical scan arguments.

        Parameters
        ----------
        **kwargs
            Passed to :func:`albireo.detection_limit`.

        Returns
        -------
        albireo.calibrate.DetectionLimit

        Raises
        ------
        ValueError
            Under the same conditions as :meth:`scan`.
        TypeError
            If ``kwargs`` names an argument that the declaration supplies, such as
            ``orbit`` or ``k1``.
        """
        from albireo.calibrate import detection_limit

        scan_kwargs = self._k2_scan_kwargs("detection_limit")
        clash = sorted(set(scan_kwargs) & set(kwargs))
        if clash:
            raise TypeError(
                f"detection_limit() takes {clash} from the declaration, so that the "
                "calibration scans with the arguments of scan(); change the declaration "
                "instead"
            )
        # A block size passed by the caller takes precedence over the declaration's.
        if self.block_size is not None:
            kwargs.setdefault("block_size", self.block_size)
        return detection_limit(self.grid, self.dataset, **scan_kwargs, **kwargs)

    def _k2_scan_kwargs(self, what: str) -> dict[str, Any]:
        """The arguments that :meth:`scan` and :meth:`detection_limit` pass to the K2 scan.

        Both methods take them from here, so the calibration scans with the arguments of
        the scan it calibrates. ``what`` names the calling method in the error messages.
        """
        k1, k2 = self._scan_k()
        self._refuse_anchored_lsf(what)
        self._refuse_correlated_noise(what)
        nebular = self._nebular
        return {
            "orbit": self._scan_orbit(),
            "k1": float(np.asarray(k1.start())),
            "k2_grid": np.asarray(k2.values, dtype=float),
            "light_fractions": [s.light for s in self.stars],
            "lsf_sigma_v": {k: _coerce_lsf(v, k).sigma_kms for k, v in self.lsf.items()},
            "prior": self.smoothness_prior,
            "v_rel_max_kms": self.velocity_budget.total,
            "k1_sigma": float(np.asarray(k1.sigma)) if isinstance(k1, Known) else None,
            "telluric": self._has_telluric,
            "nebular": nebular is not None,
            "nebular_v_kms": 0.0 if nebular is None else float(nebular.v_kms),
        }


# -- results ------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseScan:
    """The conjunction-phase scan run before optimizing, retained for inspection.

    Attributes
    ----------
    period
        Period in days at which the scan was run.
    trials
        Trial conjunction times, one period long, in the dataset's time system.
    values
        Marginal log-likelihood at each trial, in nats.
    best
        Trial time with the highest finite marginal likelihood, or the first trial when
        none is finite.
    """

    period: float
    trials: np.ndarray
    values: np.ndarray
    best: float

    @property
    def contrast(self) -> float:
        """Marginal log-likelihood difference in nats between the best and worst phase.

        Only finite trials enter (a failed solve is ``-inf``); NaN when there are none.
        """
        return _finite_range(self.values)


def _require_a_finite_trial(scan: PhaseScan) -> None:
    """Reject a conjunction scan none of whose trials has a finite marginal likelihood."""
    if not np.any(np.isfinite(scan.values)):
        raise ValueError(
            f"the conjunction scan at period {scan.period:g} d has no finite marginal "
            f"likelihood at any of its {np.size(scan.trials)} trials, so it cannot choose a "
            "starting conjunction. The starting values of the other sites, the declared LSF "
            "or the data give a likelihood that is not finite at every phase."
        )


@dataclass(frozen=True)
class SemiAmplitudeScan:
    """The coarse semi-amplitude scan run before optimizing, retained for inspection.

    Attributes
    ----------
    k
        Trial semi-amplitudes, ``(n_trials, n_stellar)`` km/s: each component declared as
        a range over its geometric grid in turn, the others at their current values, then
        the joint refinements, the full-axis pass and the exchange check, in that order.
    t_conj
        The conjunction each trial was evaluated at, ``(n_trials,)``.
    values
        Marginal log-likelihood at each trial, in nats.
    best, best_t_conj, best_value
        The chosen trial: semi-amplitudes ``(n_stellar,)`` in km/s, its conjunction, and
        its log-likelihood.
    start, start_value
        The semi-amplitudes the declaration started from, and the log-likelihood there at
        the located conjunction.
    dv_kms
        The model-grid pixel size of the declaration the scan ran on, which is coarser
        than the fit's.
    hold_losses
        Per ranged component (by index), the nats lost by holding it at its start with the
        others at the scan's best. This is the gain of each component's own move, once
        the joint move has been taken.
    prior_scales
        Per star (by name), the factor applied to the starting precision of its smoothness
        prior (``tau`` and ``eta``) before the scan was repeated, when the profile over the
        light-equivalent amplitude gave a factor of two or more; empty otherwise.
    notes
        Outcomes of the checks: a best trial at the static minimum rejected, a joint gain
        too small to move the start, a component with too small a gain kept at its start.
        If the scan was repeated, the first note is the prior amplitude adaptation.
    """

    k: np.ndarray
    t_conj: np.ndarray
    values: np.ndarray
    best: np.ndarray
    best_t_conj: float
    best_value: float
    start: np.ndarray
    start_value: float
    dv_kms: float | None = None
    hold_losses: dict = field(default_factory=dict)
    notes: tuple = ()
    prior_scales: dict = field(default_factory=dict)

    @property
    def n_trials(self) -> int:
        """The number of trials evaluated."""
        return int(self.values.size)

    @property
    def gain(self) -> float:
        """Nats gained over the declared start; the fit keeps the start unless this is positive."""
        return float(self.best_value - self.start_value)

    @property
    def contrast(self) -> float:
        """Marginal log-likelihood difference in nats between the best and worst trial.

        Only finite trials enter (a failed solve is ``-inf``); NaN when there are none.
        """
        return _finite_range(self.values)


class _ZeroPointWindowError(ValueError):
    """No default search window holds every component, because of the templates' zero points.

    Raised by :func:`_default_v_range`. Dropping the zero points (``v_zero_kms=None``)
    removes the cause, which is the remedy the pipeline applies to this error and to no
    other.
    """


def _default_v_range(fitted: np.ndarray, templates, margin: float = 40.0):
    """One correlation search window per template, all covering the same velocities.

    :func:`albireo.todcor` searches the shift of each template's own rest frame and
    reports it composed with that template's ``v_zero_kms``. A single ``(lo, hi)`` pair
    therefore searches a different interval of reported velocity for every component whose
    zero point differs. The windows are offset by each template's zero point relative to
    their median, which leaves one common interval of reported velocity: the span of the
    fitted velocities widened by ``margin`` at each end, moved to the median zero point.
    Templates whose zero point is unknown are placed at zero, the frame of the fitted
    velocities.

    The zero points come from a label match, which measures each component's frame
    separately (``docs/math.md`` §9). When two disagree by more than the fitted velocities
    span, no interval of reported velocity contains every component's own velocities, and
    a search would return a table of velocities pinned to the edge of the search. This
    case raises :class:`_ZeroPointWindowError`, a ``ValueError``.
    """
    fitted = np.asarray(fitted, dtype=np.float64)
    zeros = np.array([0.0 if t.v_zero_kms is None else float(t.v_zero_kms) for t in templates])
    offsets = zeros - float(np.median(zeros))
    lo, hi = float(fitted.min()) - margin, float(fitted.max()) + margin
    paired = fitted.ndim == 2 and fitted.shape[0] == len(templates)
    ranges = []
    for i, template in enumerate(templates):
        own = fitted[i] if paired else fitted
        low, high = lo - offsets[i], hi - offsets[i]
        short = max(low - float(own.min()), float(own.max()) - high)
        if short > 0.0:
            declared = ", ".join(
                f"{t.name}={z:+.3f}" for t, z in zip(templates, zeros, strict=True)
            )
            raise _ZeroPointWindowError(
                f"component {template.name!r}: the default search window "
                f"({low:+.3f}, {high:+.3f}) km/s misses its own fitted velocities "
                f"({float(own.min()):+.3f} to {float(own.max()):+.3f} km/s) by "
                f"{short:.3f} km/s. The windows are one common interval of reported "
                f"velocity, offset by each template's zero point ({declared} km/s), and "
                "these zero points disagree by more than the fitted velocities span, so "
                "no such interval holds every component. Pass v_range=[(lo, hi), ...], "
                "one pair per template in that template's own frame, if the zero points "
                "are right; if they are not (a label fit pinned at the edge of its "
                "frame-offset scan is the usual cause), drop them with "
                "Template(..., v_zero_kms=None) and read the velocities as differential."
            )
        ranges.append((low, high))
    return ranges


@dataclass(frozen=True)
class EpochRenormalization:
    """The epochs of a fit divided by smooth corrections against its model.

    Returned by :meth:`Fit.renormalize_epochs`.

    Attributes
    ----------
    dataset
        The epochs divided by their corrections, with the inverse variances multiplied by
        the square of the correction. Wavelengths, times, masks and every other field are
        unchanged.
    corrections
        Per epoch, on its own pixels: the product of the corrections of all passes, so
        that ``dataset[j].flux == original[j].flux / corrections[j]``.
    rms
        ``(passes, n_epochs)``: the root mean square deviation of each pass's correction
        from 1 over the epoch's good pixels. The first pass measures the normalization
        errors; later passes should be much smaller.
    scale_angstrom
        The scale used: the knot spacing of the corrections and the width of the shared
        low-frequency part of the model.
    """

    dataset: Dataset
    corrections: tuple[np.ndarray, ...]
    rms: np.ndarray
    scale_angstrom: float

    def summary(self) -> str:
        """The size of the corrections, per pass."""
        lines = [
            f"Epochs renormalized against the model: {self.rms.shape[1]} epochs, "
            f"{self.rms.shape[0]} passes, scale {self.scale_angstrom:g} A"
        ]
        for i, row in enumerate(self.rms):
            lines.append(
                f"  pass {i + 1}: correction rms median {np.median(row):.4f}, largest "
                f"{np.max(row):.4f} (epoch {int(np.argmax(row))})"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class Fit:
    """The best fit (MAP), its ML-II regularization weights, and the quantities derived.

    The orbit is the maximum of the posterior with the component spectra marginalized, and
    the regularization weights ``tau`` and ``eta`` of each component are those that
    maximize the marginal likelihood (ML-II). Every downstream call requires the fitted
    weights with the parameters, so :meth:`spectra`, :meth:`std`, :meth:`composite` and
    :meth:`sample` supply them automatically. ``docs/method.md`` describes both steps.

    Attributes
    ----------
    dis
        The :class:`Disentangler` this fit was run from.
    result
        The optimizer result returned by :func:`albireo.run_map`.
    hyper
        Fitted ``{"tau": ..., "eta": ...}`` per component name, in linear units.
    phase_scan
        The :class:`PhaseScan` run before optimization, or ``None`` when ``t_conj`` was
        declared or the fit is in ``"velocity"`` mode.
    mode
        ``"keplerian"`` when orbital elements were fitted, ``"velocity"`` when a free
        per-epoch table was fitted instead.
    priors_used
        The priors the fit was run under. Retained because they are not recoverable from
        the declaration once the Keplerian is replaced by a free velocity table, and a
        Laplace covariance is meaningful only for the model that produced it.
    k_scan
        The :class:`SemiAmplitudeScan` run before optimization, or ``None`` when no
        semi-amplitude was a range, the scan was switched off, or the fit is in
        ``"velocity"`` mode.
    smoothing
        ``None`` for a fit to the epochs as observed. For a fit returned by
        :meth:`smoothed`, one entry per smoothed instrument: the instrument, its widths
        before and after, and the diagnostics of the convolution
        (:func:`albireo.preprocess.smooth_epoch`).
    max_steps
        The L-BFGS iteration cap the fit was run with, or ``None`` where it was not
        recorded. :meth:`summary` uses it to distinguish a fit stopped at the cap from one
        stopped by its progress callback.
    """

    dis: Disentangler
    result: Any
    hyper: dict[str, dict[str, float]]
    phase_scan: PhaseScan | None = None
    mode: str = "keplerian"
    priors_used: dict = field(default_factory=dict, repr=False)
    k_scan: SemiAmplitudeScan | None = None
    smoothing: tuple | None = None
    max_steps: int | None = None

    # The conditional mean and standard deviation at the MAP, and the residual z-score
    # RMS, computed once (_solved). init=False so that dataclasses.replace() starts the new
    # fit with an empty cache: it may change the declaration or the parameters.
    _cache: dict = field(default_factory=dict, repr=False, compare=False, init=False)

    @property
    def params(self) -> dict:
        """The MAP site values, as returned by :func:`albireo.run_map`."""
        return dict(self.result.params)

    @property
    def theta(self) -> dict:
        """Site values together with any fixed sites, as the forward model takes them.

        A free velocity table replaces the orbit, and the model rejects Keplerian sites
        alongside it, so in ``"velocity"`` mode the orbital sites that the declaration
        fixed are left out.
        """
        theta = {k: jnp.asarray(v) for k, v in self.result.params.items()}
        fixed = self.dis.fixed if self.mode == "keplerian" else _without_orbit(self.dis.fixed)
        theta.update({k: jnp.asarray(v) for k, v in fixed.items()})
        return theta

    def star(self, name: str) -> dict:
        """Everything fitted for one named star, so that no row index is read manually.

        Parameters
        ----------
        name
            Name of a :class:`Star` in this declaration.

        Returns
        -------
        dict
            ``name``, ``light``, the fitted ``tau`` and ``eta``, and either ``k`` in km/s
            (``"keplerian"`` mode) or the per-epoch ``velocity`` array in km/s
            (``"velocity"`` mode).

        Raises
        ------
        KeyError
            If ``name`` is not a star of this declaration.
        """
        index = _star_index(self.dis, name)
        out = {"name": name, "light": self.dis.stars[index].light, **self.hyper[name]}
        if self.mode == "keplerian":
            # theta rather than params: a semi-amplitude the declaration fixed is a
            # constant of the model, which the optimizer does not return.
            out["k"] = float(np.atleast_1d(np.asarray(self.theta["k"], dtype=float))[index])
        else:
            out["velocity"] = np.asarray(self.velocities()[index])
        return out

    def orbit(self) -> dict:
        """Period, eccentricity, omega and the semi-amplitudes, in physical units.

        Returns
        -------
        dict
            ``period`` in days, ``ecc``, ``omega`` in radians, and ``k`` in km/s per
            component, as produced by :func:`albireo.orbit_parameters`.

        Raises
        ------
        ValueError
            If this fit is in ``"velocity"`` mode and has no orbital elements.
        """
        if self.mode != "keplerian":
            raise ValueError(
                "this fit replaced the Keplerian with a free velocity table, so it has no "
                "orbital elements. Use .velocities(), or .keplerian_residuals(kep)."
            )
        return orbit_parameters(self.theta, ecc_max=self.dis.effective_ecc_max)

    def velocities(self) -> np.ndarray:
        """Per-epoch velocities in km/s, ``(n_stellar, n_epochs)``, zero point removed.

        A free table has one arbitrary zero point per component, not one in total: with no
        orbit coupling the stars, each free spectrum absorbs a constant added to its own
        shifts. The removal is performed in pixel space, where a constant offset is exact
        rather than first order.
        """
        if self.mode == "keplerian":
            return np.asarray(
                orbit_velocities(
                    self.theta, self.dis.dataset.bjd, ecc_max=self.dis.effective_ecc_max
                )
            )
        return np.asarray(relative_velocities(self.params["velocity"], self.dis.grid))

    def velocity_errors(self) -> np.ndarray:
        """Per-epoch velocity uncertainties, projected as in :meth:`velocities`.

        These are not the raw Laplace diagonal, which returns the prior width: 37.95 km/s
        on every entry against an actual 0.059 km/s in the reference case, and identical
        on a good dataset and an uninformative one.

        Returns
        -------
        numpy.ndarray
            Uncertainties in km/s, shape ``(n_stellar, n_epochs)``.

        Raises
        ------
        ValueError
            If this fit is not in ``"velocity"`` mode.
        """
        if self.mode != "velocity":
            raise ValueError(
                "velocity_errors() applies to a free-velocity fit; call free_velocities() "
                "on this fit first."
            )
        # The covariance must come from the model this fit was run against, with the same
        # sampled sites in the same order, or the projection reads the wrong block. The
        # priors are stored on the Fit for this reason.
        fixed = _without_orbit(self.dis.fixed)
        model = self.dis.model.model(self.priors_used, fixed=fixed or None)
        check = laplace_check(model, self.result.params, model_args=(self.dis.model.problem,))
        if not check.valid:
            warnings.warn(check.message(), RuntimeWarning, stacklevel=2)
        return np.asarray(relative_velocity_errors(check.covariance, self.result.unconstrained))

    def noise_correlation(self) -> dict[str, float] | None:
        """The lag-one noise correlation per instrument, as declared or as fitted.

        Returns ``None`` under the diagonal noise model. A fitted correlation is the MAP
        value of the ``ar1_phi`` site, shared by every instrument.
        """
        declared = self.dis.noise_correlation
        if declared is None:
            return None
        instruments = list(self.dis.dataset.instruments)
        if isinstance(declared, Spec):
            return dict.fromkeys(instruments, float(np.asarray(self.theta["ar1_phi"])))
        if isinstance(declared, Mapping):
            return {name: float(declared[name]) for name in instruments}
        return dict.fromkeys(instruments, float(declared))

    def marginal(self):
        """The conditional solve at the MAP: spectra, precision, and log-likelihood.

        Every call solves again. :meth:`spectra`, :meth:`std` and :attr:`z_rms` keep their
        results instead.
        """
        return self.dis.model.marginal(self.theta)

    def _solved(self, *, std: bool = False) -> dict:
        """The cache, holding ``d_hat`` (and ``std`` if asked) from one solve at the MAP.

        The accessors of a fit all read the solve at one ``theta``, and the pipeline calls
        them about ten times per star. The arrays are kept, read-only; the
        :class:`~albireo.inference.MarginalResult`, whose precision band is large, is not.
        Repeated solves at one ``theta`` give the same arrays bit for bit.
        """
        cache = self._cache
        if "d_hat" not in cache or (std and "std" not in cache):
            marginal = self.marginal()
            cache.setdefault("d_hat", np.asarray(marginal.d_hat))
            if std:
                cache["std"] = np.asarray(spectra_std(marginal))
        return cache

    def spectra(self) -> np.ndarray:
        """Conditional-mean component spectra ``d``, shape ``(n_comp, n_pix)``.

        These are deviations from a unit continuum, so the component spectrum is ``1 + d``
        and the observable is ``l_i * d_i``; see :meth:`Disentangler.assumptions`. The
        array is computed once per fit and is read-only.
        """
        return self._solved()["d_hat"]

    def std(self) -> np.ndarray:
        """Pointwise standard deviation of :meth:`spectra`, of the same shape.

        It is large wherever the prior rather than the data sets the spectrum, which
        covers most of the continuum, and is therefore exported alongside the spectrum.
        The array is computed once per fit and is read-only.
        """
        return self._solved(std=True)["std"]

    def _spectra_and_std(self) -> tuple[np.ndarray, np.ndarray]:
        """:meth:`spectra` and :meth:`std` from one solve of the marginal.

        Two solves in succession allocate the second's buffers before those of the first
        are released, and the selected inverse of :meth:`std` adds its own. At 36,085 model
        pixels a pair of solves reached 7.1 GB of commit, against 5.1 GB for one solve with
        its selected inverse. When neither array is cached yet, both come from one solve.
        """
        cache = self._solved(std=True)
        return cache["d_hat"], cache["std"]

    def composite(self) -> np.ndarray:
        """The light-weighted stellar sum ``sum_i l_i d_i``, the observed quantity.

        The individual components are a decomposition of this sum; only the weighted sum
        is constrained by the data at every pixel.
        """
        d = self.spectra()[: self.dis.n_stellar]
        lights = np.array([s.light for s in self.dis.stars], dtype=float)
        return np.tensordot(lights, d, axes=(0, 0))

    def residual_zscores(self) -> np.ndarray:
        """Whitened data residuals. Their RMS is close to 1 when the noise model is correct."""
        return np.asarray(
            data_residual_zscores(
                self.dis.model.problem_at(self.theta), jnp.asarray(self.spectra())
            )
        )

    @property
    def z_rms(self) -> float:
        """RMS of :meth:`residual_zscores`, the check to make before adding jitter."""
        if "z_rms" not in self._cache:
            z = self.residual_zscores()
            self._cache["z_rms"] = float(np.sqrt(np.mean(np.square(z[np.isfinite(z)]))))
        return self._cache["z_rms"]

    def summary(self) -> str:
        """Report the fit: convergence, orbit or velocities, ML-II, residuals, assumptions.

        The first line reports the optimizer: the negative log posterior it minimized, in
        numpyro's unconstrained coordinates and up to a constant, and the norm of its
        gradient. Differences of log likelihood are in nats (natural-log units; a
        difference of 1 nat is a chi-square difference of 2).

        Returns
        -------
        str
            A multi-line report ending with :meth:`Disentangler.assumptions`.
        """
        # The convergence flag is an absolute threshold on a potential whose scale grows
        # with the number of good pixels, so on real data it is routinely False at a good
        # optimum. The outcome is reported without a pass/fail label.
        if self.result.converged:
            stopped = "converged"
        elif self.max_steps is not None and self.result.num_steps < self.max_steps:
            stopped = "stopped by the progress callback"
        else:
            stopped = "stopped at the step cap"
        what = "orbit" if self.mode == "keplerian" else "free velocity table"
        lines = [
            f"Best fit (maximum a posteriori) of the {what}, {self.result.num_steps} steps "
            f"({stopped})",
            f"  -ln posterior {self.result.potential:.6g} (unconstrained coordinates, up to a "
            f"constant), |gradient| {self.result.grad_norm:.3g}",
        ]
        fall, steps = (
            self.result.potential_fall()
            if hasattr(self.result, "potential_fall")
            else (math.nan, 0)
        )
        if not self.result.converged and steps:
            if fall > _STILL_FALLING_NATS:
                lines.append(
                    f"  ! the potential fell by {fall:.3g} nats over the last {steps} steps"
                )
                lines.append(
                    _indented(
                        "The optimizer was still descending, and the orbit or the "
                        "regularization weights are not at the maximum. Continue with a "
                        "larger max_steps. laplace_check() tests whether the Laplace "
                        "approximation of the orbit at these weights is valid."
                    )
                )
            else:
                # The threshold is absolute and is not reached on large datasets, and an
                # optimizer that creeps along a valley gains little per step far from the
                # maximum, so the change is reported with the test that settles it.
                lines.append(
                    f"  the potential changed by {fall:.2g} nats over the last {steps} steps; "
                    "laplace_check() tests the maximum"
                )
        if self.phase_scan is not None:
            lines.append(
                f"  conjunction scan: t_conj = {self.phase_scan.best:.5f}, "
                f"{self.phase_scan.contrast:.3g} nats between the best and worst phase"
            )
        if self.k_scan is not None:
            names = [s.name for s in self.dis.stars]
            chosen = ", ".join(f"{n} {k:.1f}" for n, k in zip(names, self.k_scan.best, strict=True))
            started = ", ".join(f"{k:.1f}" for k in self.k_scan.start)
            outcome = (
                f"started from it, {self.k_scan.gain:+.3g} nats over the declared start ({started})"
                if self.k_scan.gain > 0.0
                else f"the declared start ({started}) was better by {-self.k_scan.gain:.3g} "
                "nats and was kept"
            )
            lines.append(
                f"  semi-amplitude scan: best of {self.k_scan.n_trials} trials at {chosen} km/s; "
                + outcome
            )
            for note in self.k_scan.notes:
                lines.append(f"    {note}")
        lines.append("")
        if self.mode == "keplerian":
            params = self.orbit()
            lines.append(f"  period    {float(params['period']):.6f} d")
            lines.append(
                f"  ecc       {float(params['ecc']):.4f}    omega {float(params['omega']):.4f} rad"
            )
            for star in self.dis.stars:
                row = self.star(star.name)
                lines.append(f"  K({star.name})  {row['k']:8.3f} km/s   (light {star.light:g})")
        else:
            velocities = self.velocities()
            for i, star in enumerate(self.dis.stars):
                span = float(np.max(velocities[i]) - np.min(velocities[i]))
                lines.append(
                    f"  {star.name:<12s} {velocities.shape[1]} epochs, peak-to-peak "
                    f"{span:.3f} km/s (zero point removed per component)"
                )
        lines += ["", self._hyper_report(), ""]
        z_rms = self.z_rms
        lines.append(
            f"  residual z-score RMS {z_rms:.3f}"
            + (
                ""
                if abs(z_rms - 1.0) < 0.2
                else "   <- the noise model is not describing these data; read "
                "docs/benchmarks.md before adding a jitter term"
            )
        )
        lines += ["", self.dis.assumptions()]
        return "\n".join(lines)

    def _hyper_report(self) -> str:
        """The ML-II table, flagging a weight left at its start, a fallen eta, and a bound."""
        rows = [
            "Regularization weights of the component spectra, fitted by maximizing the marginal",
            "likelihood (ML-II) and held fixed for sampling: curvature weight tau and "
            "continuum weight",
            "eta, centre -> fitted, with the moves in ln tau and ln eta in hyperprior "
            "standard deviations",
        ]
        stale, fallen, bounded = [], [], []
        for name, component in zip(
            self.dis.component_names, self.dis.ordered_components, strict=True
        ):
            smooth = _smoothness_of(component)
            fitted = self.hyper[name]
            drift = (math.log(fitted["tau"]) - math.log(smooth.tau0)) / smooth.sigma
            eta_drift = (math.log(fitted["eta"]) - math.log(smooth.eta0)) / smooth.sigma
            bound = "" if smooth.eta_min is None else f", bound {smooth.eta_min:g}"
            rows.append(
                f"  {name:<14s} tau {smooth.tau0:9.3g} -> {fitted['tau']:9.3g} "
                f"({drift:+.2f} sigma)   eta {smooth.eta0:8.3g} -> {fitted['eta']:8.3g} "
                f"({eta_drift:+.2f} sigma{bound})"
            )
            if abs(drift) < 0.02:
                stale.append(name)
            at_bound = (
                smooth.eta_min is not None and math.log(fitted["eta"] / smooth.eta_min) < _AT_BOUND
            )
            if at_bound:
                bounded.append((name, smooth.eta_min))
            elif eta_drift < -_ETA_FALL_SIGMA and len(self.dis.component_names) > 1:
                fallen.append((name, smooth.eta0, fitted["eta"]))
        rows += [
            f"  ! {name!r} tau did not move from its start: the hyperprior, not the data, "
            "is setting this component's smoothness."
            for name in stale
        ]
        for name, centre, eta in fallen:
            rows.append(
                f"  ! {name!r} eta fell more than one hyperprior standard deviation below its "
                f"centre ({centre:g} -> {eta:.3g})."
            )
            rows.append(
                _indented(
                    f"The continuum weight allows each pixel a spread of "
                    f"{1.0 / math.sqrt(eta):.2g} continuum units, and it is the only term that "
                    "holds the constant and the slope of the component: continuum errors "
                    "that differ between epochs can appear as broad drifts of the components "
                    "in opposite directions, which their light-weighted sum does not show. "
                    "Bound it (Smoothness(eta_min=...)) or renormalize the epochs "
                    "(Fit.renormalize_epochs); docs/method.md, section 10."
                )
            )
        rows += [
            f"  ! {name!r} eta is at its declared bound {bound:g}: the data would take it lower."
            for name, bound in bounded
        ]
        return "\n".join(rows)

    def _fixed_hyper(self) -> dict:
        """The ML-II values as constants, plus whatever the declaration already fixed."""
        names = self.dis.component_names
        return {
            **self.dis.fixed,
            "log_tau": jnp.asarray(np.log([self.hyper[n]["tau"] for n in names])),
            "log_eta": jnp.asarray(np.log([self.hyper[n]["eta"] for n in names])),
        }

    def _orbit_model(self, *, hold_weights: bool = True):
        """``(model, start)``: the orbital posterior with the ML-II weights held, at the MAP.

        With ``hold_weights=False`` the weights keep their hyperpriors, as in the best fit,
        and are sites of the model (:meth:`_plug_in_widening`).
        """
        fixed = self._fixed_hyper() if hold_weights else dict(self.dis.fixed)
        priors = {k: v for k, v in self.dis.priors.items() if k not in fixed}
        period = None
        if self.phase_scan is not None:
            period = self.phase_scan.period
        elif (
            self.dis.orbit is not None
            and self.dis.orbit.t_conj == "scan"
            and isinstance(self.priors_used.get("t_conj"), dist.Uniform)
        ):
            # A fit returned by smoothed() has no scan record, but it keeps the window its
            # parameters were fitted under. The declaration's window starts at the first
            # epoch, and a fitted conjunction before it has no unconstrained coordinate.
            window = self.priors_used["t_conj"]
            period = float(np.asarray(window.high)) - float(np.asarray(window.low))
        if period is not None:
            # The prior is one period wide, centred on the fitted conjunction (the MAP may be
            # half a period from the phase scan's best when the semi-amplitude scan moved it).
            centre = float(np.asarray(self.result.params["t_conj"], dtype=float))
            priors["t_conj"] = dist.Uniform(centre - 0.5 * period, centre + 0.5 * period)
        model = self.dis.model.model(priors, fixed=fixed)
        start = {k: v for k, v in self.result.params.items() if k in priors}
        return model, start

    def laplace_check(self) -> LaplaceCheck:
        """The Laplace approximation of the orbit at this fit, and whether it is valid.

        The approximation is that of :meth:`sample`, which takes its covariance as the
        mass matrix: the orbital sites in numpyro's unconstrained coordinates, with the
        regularization weights held at their ML-II values. Its covariance describes the
        posterior only at a maximum, with a positive-definite Hessian, around which the
        potential is quadratic on the scale of one standard deviation
        (:class:`albireo.LaplaceCheck`). A fit stopped at its step cap short of the maximum
        fails the second condition, and a fit stopped at a kink of the potential, which
        the linear shift operator produces at whole pixels of the model grid, the third.
        :meth:`summary` reports the fall of the potential over the last steps, an
        indication; this method evaluates the potential around the best fit. It costs one
        Hessian, as :meth:`sample` does, and two evaluations of the potential per
        orbital parameter.

        Returns
        -------
        albireo.LaplaceCheck

        Raises
        ------
        ValueError
            If this fit is in ``"velocity"`` mode (:meth:`velocity_errors` checks its own
            approximation).
        """
        if self.mode != "keplerian":
            raise ValueError(
                "laplace_check() is the approximation of the orbital posterior, and this fit "
                "replaced the Keplerian with a free velocity table. velocity_errors() checks "
                "the approximation it uses and warns."
            )
        model, start = self._orbit_model()
        return laplace_check(model, start, model_args=(self.dis.model.problem,))

    def sample(
        self,
        *,
        seed: int = 0,
        num_warmup: int = 500,
        num_samples: int = 500,
        num_chains: int = 2,
        max_tree_depth: int = 8,
        progress_bar: bool = False,
    ) -> Posterior:
        """Sample the orbital sites with NUTS, holding the ML-II hyperparameters fixed.

        The method rebuilds the model with ``log_tau`` and ``log_eta`` injected as constants,
        takes the Laplace covariance at the MAP as the starting mass matrix, and samples.

        Fixing the hyperparameters is a plug-in approximation: the orbital credible
        intervals do not include smoothness uncertainty. The method also takes the Laplace
        approximation with the weights free, at the cost of one more Hessian, and
        :attr:`Posterior.plug_in_widening` gives how much that widens the standard deviation
        of each sampled parameter. Marginalizing over the weights in the sampling instead
        means leaving them in the priors dictionary of :meth:`Disentangler.expert`, at a
        higher sampling cost. ``docs/method.md`` describes the stages (best fit, Laplace
        approximation, NUTS) and the convergence diagnostics that
        :meth:`Posterior.summary` reports.

        Parameters
        ----------
        seed
            Seed for the numpyro PRNG key.
        num_warmup, num_samples
            Warmup and sampling draws per chain.
        num_chains
            Number of chains.
        max_tree_depth
            NUTS tree-depth cap.
        progress_bar
            Whether numpyro prints its progress bar.

        Returns
        -------
        Posterior

        Raises
        ------
        ValueError
            If this fit is in ``"velocity"`` mode and has no orbital sites to sample.
        """
        if self.mode != "keplerian":
            raise ValueError(
                "sample() draws the orbital posterior, and this fit replaced the Keplerian "
                "with a free velocity table: there are no orbital sites to sample. Sample "
                "the Keplerian fit instead, and use velocity_errors() here."
            )
        nuts_model, start = self._orbit_model()
        check = laplace_check(nuts_model, start, model_args=(self.dis.model.problem,))
        if not check.valid:
            warnings.warn(
                check.message() + " NUTS starts here, with this covariance as its mass matrix.",
                RuntimeWarning,
                stacklevel=2,
            )
        mass = check.covariance
        widening, note = self._plug_in_widening(check)
        mcmc = run_nuts(
            nuts_model,
            rng_key=jax.random.PRNGKey(int(seed)),
            init=start,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            inverse_mass_matrix=mass,
            max_tree_depth=max_tree_depth,
            progress_bar=progress_bar,
            model_args=(self.dis.model.problem,),
        )
        return Posterior(
            dis=self.dis, mcmc=mcmc, map=self, plug_in_widening=widening, plug_in_note=note
        )

    def _plug_in_widening(self, held: LaplaceCheck) -> tuple[dict[str, float] | None, str]:
        """How much freeing the ML-II weights widens each Laplace standard deviation.

        The best fit is joint over the orbit and the regularization weights, so the Hessian
        of the potential there can be taken with the weights held, as for the mass matrix of
        :meth:`sample` (``held``), or with the weights free under their hyperpriors. The
        orbital block of the free covariance is the Laplace approximation marginalized over
        the weights, and the held covariance is the plug-in. The ratio of their standard
        deviations is taken in numpyro's unconstrained coordinates, which transform each
        site entry by entry, so to first order it is the ratio for the site itself. It is at
        least 1, since conditioning on the weights cannot widen a Gaussian.

        Returns
        -------
        tuple
            ``(factors, note)``: the ratio for each coordinate of ``held``, keyed by its
            label, and an empty note; or ``None`` and the reason it was not computed.
        """
        if held.n_floored:
            return None, "the Hessian with the weights held is not positive definite"
        model, start = self._orbit_model(hold_weights=False)
        if not {"log_tau", "log_eta"} & set(start):
            return None, "the regularization weights are not sites of this model"
        free = laplace_check(model, start, model_args=(self.dis.model.problem,), probe=False)
        if free.n_floored:
            return None, "the Hessian with the weights free is not positive definite"
        index = {label: i for i, label in enumerate(free.labels)}
        sd_free = np.sqrt(np.diag(free.covariance))
        sd_held = np.sqrt(np.diag(held.covariance))
        factors = {
            label: float(sd_free[index[label]] / sd_held[i]) for i, label in enumerate(held.labels)
        }
        return factors, ""

    def _velocity_priors(self, sigma_kms: float | None = None):
        """``(priors, init)`` for the free-velocity mode, warm-started from this fit."""
        start = np.asarray(
            orbit_velocities(self.theta, self.dis.dataset.bjd, ecc_max=self.dis.effective_ecc_max)
        )
        sigma = sigma_kms if sigma_kms is not None else self.dis.velocity_budget.total / 2.0
        priors = {"velocity": dist.Normal(0.0, float(sigma)).expand(list(start.shape)).to_event(2)}
        init = {"velocity": jnp.asarray(start)}
        # Every sampled site that is not part of the orbit is kept unchanged, with its
        # declared prior and start: the smoothness hyperparameters, the per-epoch nebular
        # amplitudes where the declaration has a nebular component, and a fitted noise
        # correlation. Dropping the amplitudes would hold the component static at
        # amplitude 1, and dropping the correlation would fit the table under the diagonal
        # noise model, in either case without reporting it.
        for site, prior in _without_orbit(self.dis.priors).items():
            priors[site] = prior
            init[site] = self.dis.init[site]
        return priors, init

    def free_velocities(
        self,
        *,
        sigma_kms: float | None = None,
        max_steps: int = 300,
        progress=None,
        tol: float = 1e-2,
    ) -> Fit:
        """Refit with one free velocity per component per epoch, warm-started from this fit.

        The per-epoch radial-velocity table is the customary product of a spectroscopic
        binary analysis, and the model check for the Keplerian mode: free velocities are
        fitted, then tested against a Keplerian (:meth:`keplerian_residuals`).

        The method is defined on :class:`Fit` because a cold start was measured 122,000
        nats worse than the warm-started solution. Warm-starting is the only mode shown to
        succeed. The velocities start at this fit's Keplerian. The other sampled sites
        (the smoothness hyperparameters, nebular amplitudes, a fitted noise correlation)
        keep their declared priors and start at the declaration's starting values.

        This is the entry point when a Keplerian fit exists and the table is wanted as a
        model check. For an unsolved system, where the table yields the period,
        ``Disentangler(velocities=...)`` and :meth:`Disentangler.fit` warm-start from
        measured velocities instead.

        Parameters
        ----------
        sigma_kms
            Width in km/s of the Gaussian prior on each free velocity. Defaults to half
            the velocity budget.
        max_steps
            L-BFGS iteration cap.
        progress
            ``callback(step, potential, grad_norm, params)``.
        tol
            Gradient-norm threshold of :func:`albireo.run_map`. Unlike that of
            :meth:`Disentangler.fit`, it is not scaled to the number of good pixels.

        Returns
        -------
        Fit
            A fit in ``"velocity"`` mode, on the same epochs as this fit (smoothed ones
            included, with :attr:`smoothing` carried over).

        Raises
        ------
        ValueError
            If this fit is in ``"velocity"`` mode, which has no Keplerian to start from.
        """
        if self.mode != "keplerian":
            raise ValueError(
                "free_velocities() starts the table from this fit's Keplerian, and this fit "
                "is already a free velocity table. Call it on the Keplerian fit."
            )
        priors, init = self._velocity_priors(sigma_kms)
        # Whatever the declaration fixed about the orbit no longer applies, and the model
        # rejects Keplerian sites alongside a velocity table, so they are dropped.
        fixed = _without_orbit(self.dis.fixed)
        result = run_map(
            self.dis.model.model(priors, fixed=fixed or None),
            init=init,
            max_steps=max_steps,
            tol=tol,
            callback=progress,
            model_args=(self.dis.model.problem,),
        )
        hyper = _hyper_of(result.params, self.dis.component_names)
        return Fit(
            dis=self.dis,
            result=result,
            hyper=hyper,
            mode="velocity",
            priors_used=priors,
            smoothing=self.smoothing,
            max_steps=max_steps,
        )

    def keplerian_residuals(self, keplerian: Fit) -> np.ndarray:
        """This table's velocities minus those of a Keplerian fit, the model check.

        Parameters
        ----------
        keplerian
            A :class:`Fit` in ``"keplerian"`` mode, from the same declaration.

        Returns
        -------
        numpy.ndarray
            Residuals in km/s, shape ``(n_stellar, n_epochs)``.

        Raises
        ------
        ValueError
            If this fit is not in ``"velocity"`` mode.
        """
        if self.mode != "velocity":
            raise ValueError("keplerian_residuals() applies to a free-velocity fit")
        return np.asarray(
            keplerian_residuals(
                self.params["velocity"],
                keplerian.theta,
                self.dis.dataset.bjd,
                self.dis.grid,
                ecc_max=self.dis.effective_ecc_max,
            )
        )

    def _stellar_lsf_widths(self, group) -> np.ndarray:
        """The declared Gaussian widths of one epoch group, in km/s (one per anchor)."""
        spec = _coerce_lsf(self.dis.lsf[group.instrument], group.instrument)
        if spec.is_per_epoch:
            index = int(group.epoch_indices[0])
            for sigma, epochs in declared_lsf_widths(self.dis.dataset, group.instrument).items():
                if index in epochs:
                    return np.atleast_1d(np.asarray(sigma, dtype=float))
            raise ValueError(  # pragma: no cover - the build grouped by these widths
                f"no declared width covers epoch {index} of instrument {group.instrument!r}"
            )
        return np.atleast_1d(np.asarray(spec.sigma_kms, dtype=float))

    def epoch_statistics(
        self, resolving_power: float | None = None, *, grid_compensation: bool = True
    ):
        """The sufficient statistics of the epoch comparison, at this fit's MAP.

        The method evaluates the model problem at the fitted parameters and returns
        ``h = A^T W z``, ``z^T W z`` and the band of ``G = A^T W A`` for the stellar rows.
        The epoch chi-square of any template composite ``m`` then costs one band product
        (:class:`albireo.EpochStatistics`, ``docs/math.md`` §9.2a). The result is passed to
        :func:`albireo.match_labels` with ``compare="epochs"``. :meth:`match_labels` builds
        it itself.

        Three things differ from the disentangling problem. First, every instrument's LSF
        in the stellar operator is replaced by the width not already present in the
        library, ``sigma_q = sqrt(sigma_inst^2 - sigma_lib^2)`` per declared width (per
        anchor, or per epoch for a per-epoch declaration), through
        :func:`albireo.forward.with_lsf`. A template drawn from a library at its own
        resolving power would otherwise be broadened twice. Second, with
        ``grid_compensation`` (the default) that width is reduced further by the
        smoothing that the model grid adds (see below). Third, a telluric or nebular
        component is held at its posterior mean: its predicted contribution, through the
        declared LSF under which that mean was solved, is subtracted from the data, so the
        chi-square is ``||z - A_e d_hat_e - A_s m||_W^2``. The noise model (weights,
        jitter, AR(1) correlation) and the orbit are this fit's, unchanged.

        **The grid compensation.** Between a library spectrum and the epoch pixels a
        template passes through five discrete steps on the model grid of pixel ``dv``, and
        each adds a variance that a continuous spectrum does not have. A box average of
        width ``dv`` has variance ``dv^2 / 12``. A linear-interpolation shift by a fraction
        ``f`` of a pixel is the two-tap kernel with weights ``1 - f`` at ``-f dv`` and
        ``f`` at ``(1 - f) dv``, of mean zero and variance ``f (1 - f) dv^2``, which
        averages to ``dv^2 / 6`` over ``f``. The steps are the library's box average onto
        the model grid (``dv^2 / 12``), the pixel integration of the rotation kernel
        (``dv^2 / 12``), the label rows' shift by the frame velocity (``dv^2 / 6``), the
        operator's shift of each epoch (``dv^2 / 6``), and the model pixel held constant
        across the rebin onto the native pixels (``dv^2 / 12``), so that

            sigma_grid^2 = (1/12 + 1/12 + 1/6 + 1/6 + 1/12) dv^2 = (7/12) dv^2,

        independent of the line width. The epoch data have none of this variance, and
        ``v sin i``, whose limb-darkened profile has variance ``0.225 (v sin i)^2``,
        absorbs the difference: uncompensated, ``v_fit^2 = v^2 - (7/12) dv^2 / 0.225``.
        The stellar operator therefore applies

            sigma_op^2 = sigma_inst^2 - sigma_lib^2 - (7/12) dv^2.

        On the closed-loop fixture of ``tests/test_pipeline.py`` (native pixel 4.63 km/s,
        LSF sigma 5.5 km/s, ``v sin i`` injected at 11 km/s) the uncompensated comparison
        returned 7.37 km/s with the orbit fixed at the injected values, and the compensated
        one 10.08. Over five noise draws the mean error was -3.02 km/s uncompensated and
        -0.24 compensated (``docs/math.md`` §9.2a). Temperatures, gravities, metallicities
        and light fractions did not change. The formula is a mean: the frame shift's phase
        is a fitted parameter, the rotation kernel's pixel term oscillates for ``v sin i``
        near ``dv``, and computing the epoch shifts' own phases changed ``v sin i`` by at
        most 0.12 km/s.

        Where ``sigma_op`` would fall below half a model pixel, which happens when
        ``dv > sqrt(6/5) sigma_q``, about ``1.10 sigma_q``, it is floored there (or at
        ``sigma_q``, when that is narrower). Only part of the bias is then removed, and a
        ``UserWarning`` names the grid spacing that avoids the floor. When
        ``sigma_q < dv`` without a floor, the compensated kernel is below 0.65 pixel and a
        warning reports it. Both are recorded in :attr:`albireo.EpochStatistics.notes`. A
        finer model grid is not a substitute for the compensation: uncompensated, a 1.5
        km/s grid still returned 10.28 km/s for 11 on that fixture.

        Assembling the band takes a few seconds on a survey-sized problem (4.7 s of a 6.2 s
        total on the D65 benchmark products).

        Parameters
        ----------
        resolving_power
            The resolving power the template library is already broadened to
            (:attr:`albireo.SpectralLibrary.resolving_power`). ``None`` declares an
            intrinsic library and keeps the full declared LSF.
        grid_compensation
            Remove ``(7/12) dv^2`` from every operator width (default). ``False`` builds the
            operator of ``sigma_q`` alone, which the D65 benchmark measured to bias ``v sin
            i`` low by ``(7/12) dv^2 / 0.225`` in its square.

        Returns
        -------
        albireo.EpochStatistics

        Raises
        ------
        ValueError
            If the library is at or below an instrument's resolving power (which no
            convolution can undo), if the dataset does not declare its wavelength medium, or
            if the fit's light fractions vary between epochs.

        Warns
        -----
        UserWarning
            If the compensated width was floored at half a model pixel, or if the model grid
            is coarser than ``sigma_q``.
        """
        from albireo.forward import apply_model, with_data, with_lsf
        from albireo.match import (
            EpochStatistics,
            _grid_compensated_widths,
            _grid_compensation_notes,
            _grid_variance,
            _quadrature_width,
        )

        medium = _declared_medium(self.dis.dataset)
        if resolving_power is not None and not (
            np.isfinite(float(resolving_power)) and float(resolving_power) > 0.0
        ):
            raise ValueError(f"resolving_power must be positive or None; got {resolving_power}")
        model = self.dis.model
        theta = self.theta
        problem = model.problem_at(theta)
        n_stellar = self.dis.n_stellar
        lights = np.array([s.light for s in self.dis.stars], dtype=float)
        for group in problem.groups:
            carried = np.asarray(group.light)[:, :n_stellar]
            if not np.allclose(carried, lights[None, :], rtol=1e-9, atol=1e-12):
                raise ValueError(
                    "the fitted problem carries light fractions that differ from the "
                    f"declaration {lights.tolist()} (per-epoch or fitted light); the epoch "
                    "comparison needs one declared light fraction per star"
                )
        names = self.dis.component_names
        conditioned = tuple(names[n_stellar:])
        if conditioned:
            d_hat = self.spectra()
            others = np.zeros_like(d_hat)
            others[n_stellar:] = d_hat[n_stellar:]
            predicted = apply_model(problem, jnp.asarray(others))
            problem = with_data(
                problem,
                [g.z - g.r * p for g, p in zip(problem.groups, predicted, strict=True)],
            )
        dv = float(problem.grid.dv_kms)
        groups, widths, applied_widths = [], [], []
        floored_at: dict[str, float] = {}
        narrow_at: dict[str, tuple[float, float]] = {}
        for group in problem.groups:
            declared = self._stellar_lsf_widths(group)
            instrument = group.instrument
            try:
                reduced = _quadrature_width(declared, resolving_power, f"instrument {instrument!r}")
            except ValueError as exc:
                raise ValueError(
                    f"{exc} Alternatively, Fit.match_labels(..., smoothing='auto') smooths "
                    "the epochs to the library's resolution and compares them there "
                    "(Fit.smoothed)."
                ) from exc
            widths.append((instrument, reduced))
            applied = reduced
            if grid_compensation:
                applied, floored = _grid_compensated_widths(reduced, dv)
                if np.any(floored):
                    lowest = float(np.min(reduced[floored]))
                    floored_at[instrument] = min(floored_at.get(instrument, np.inf), lowest)
                wide = ~floored & (reduced < dv)
                if np.any(wide):
                    k = int(np.argmin(np.where(wide, applied, np.inf)))
                    if instrument not in narrow_at or applied[k] < narrow_at[instrument][1]:
                        narrow_at[instrument] = (float(reduced[k]), float(applied[k]))
            applied_widths.append((instrument, applied))
            if resolving_power is None and not grid_compensation:
                groups.append(group)
                continue
            width = float(applied[0]) if applied.size == 1 else jnp.asarray(applied)
            single = with_lsf(replace(problem, groups=(group,)), {instrument: width})
            groups.append(single.groups[0])
        problem = replace(problem, groups=tuple(groups))
        notes = _grid_compensation_notes(floored_at, narrow_at, dv)
        for note in notes:
            warnings.warn(note, UserWarning, stacklevel=2)
        # A fit to smoothed epochs (Fit.smoothed) states so where its labels are reported;
        # the caller asked for it, so it is recorded rather than warned.
        notes = [*notes, *_smoothing_notes(self.smoothing, resolving_power)]
        return EpochStatistics.from_problem(
            problem,
            names=[s.name for s in self.dis.stars],
            medium=medium,
            light_fractions=lights,
            half_bandwidth=model.half_bandwidth,
            block_size=model.block_size,
            lsf_sigma_kms=widths,
            library_resolving_power=resolving_power,
            conditioned=conditioned,
            grid_variance_kms2=_grid_variance(dv) if grid_compensation else 0.0,
            operator_sigma_kms=applied_widths,
            notes=notes,
            smoothed_lsf_sigma_kms=tuple(
                (entry["instrument"], float(entry["sigma_kms"])) for entry in self.smoothing or ()
            ),
        )

    def smoothing_widths(self, resolving_power, sigma_q="auto") -> dict[str, float]:
        """The widths to which the epochs must be smoothed for a library at ``resolving_power``.

        An instrument is listed when the library is at or below its resolving power, the
        case in which :meth:`epoch_statistics` raises: some declared width is not broader
        than the library's ``sigma_lib``. Its target is ``max(sqrt(sigma_lib^2 +
        sigma_q^2), w_max)``, where ``w_max`` is the widest width the instrument declares,
        so that every epoch can be brought to it and the epoch operator still applies
        ``sigma_q`` to the templates. An instrument broader than the library at every
        epoch is compared as observed and is not listed. :meth:`match_labels` with
        ``smoothing=`` uses these widths (:meth:`smoothed`).

        Parameters
        ----------
        resolving_power
            The library's resolving power (:attr:`albireo.SpectralLibrary.resolving_power`).
            ``None``, an intrinsic library, needs no smoothing.
        sigma_q
            The width in km/s the operator applies to the templates after the smoothing, or
            ``"auto"`` (default) for ``max(3 km/s, 1.2 dv)``. Three km/s was used for the
            measurement of D74, where 5 km/s moved the systemic velocity by 0.02 km/s. The
            grid compensation is floored below ``sigma_q = 0.91 dv`` and approximate below
            ``sigma_q = dv`` (:meth:`epoch_statistics`), so ``"auto"`` stays above 1.2
            model pixels.

        Returns
        -------
        dict
            Instrument to target Gaussian sigma in km/s; empty when nothing needs smoothing.

        Raises
        ------
        ValueError
            If ``sigma_q`` is not positive, or if an instrument to be smoothed declares a
            wavelength-dependent LSF, which the convolution does not reproduce.
        """
        from albireo.match import _resolution_sigma_kms

        if resolving_power is None:
            return {}
        if isinstance(sigma_q, str):
            if sigma_q != "auto":
                raise ValueError(f"sigma_q must be 'auto' or a width in km/s; got {sigma_q!r}")
            q = max(_AUTO_SMOOTHING_KMS, _AUTO_SMOOTHING_PIXELS * float(self.dis.grid.dv_kms))
        else:
            q = float(sigma_q)
            if not (np.isfinite(q) and q > 0.0):
                raise ValueError(f"sigma_q must be a positive width in km/s; got {sigma_q!r}")
        sigma_lib = _resolution_sigma_kms(float(resolving_power))
        targets = {}
        for key, widths in self.dis._lsf_widths().items():
            if float(np.min(widths)) > sigma_lib:
                continue
            if _coerce_lsf(self.dis.lsf[key], key).anchors_angstrom is not None:
                raise ValueError(
                    f"instrument {key!r} declares a wavelength-dependent LSF, and the library "
                    f"(R = {float(resolving_power):g}) is broader than part of it. Smoothing "
                    "to one width does not reproduce a profile that varies along the order; "
                    "declare one width for the instrument, or use a library sharper than the "
                    "data."
                )
            targets[key] = float(max(np.hypot(sigma_lib, q), float(np.max(widths))))
        return targets

    def smoothed(self, sigma_kms, *, fill_max: int = 3) -> Fit:
        """This fit on its epochs convolved to broader Gaussian line-spread functions.

        Every epoch of each instrument in ``sigma_kms`` is convolved to that width with
        :func:`albireo.preprocess.smooth_epoch`, which keeps the input inverse variances,
        and the instrument's LSF is declared at it. The orbit, the smoothness
        hyperparameters and the other instruments are this fit's. The epoch statistics of
        the result are exact for the smoothed epochs at this fit's MAP, and the label fit's
        frame offsets are in this fit's frame, so they serve as zero points of this fit's
        templates. The result is not a new MAP: its components are the conditional mean of
        the smoothed epochs at this fit's parameters.

        Parameters
        ----------
        sigma_kms
            Mapping of instrument name to target Gaussian sigma in km/s, as returned by
            :meth:`smoothing_widths`.
        fill_max
            Passed to :func:`albireo.preprocess.smooth_epoch`.

        Returns
        -------
        Fit
            A fit on the smoothed epochs, with :attr:`smoothing` recording what was done.

        Raises
        ------
        ValueError
            If an instrument is not in the dataset, if a target is narrower than an epoch's
            width, if an instrument declares a wavelength-dependent LSF or a noise
            correlation, or if an epoch keeps no pixel with a whole kernel.
        """
        from albireo.preprocess import _smoothed

        targets = {str(k): float(v) for k, v in dict(sigma_kms).items()}
        if not targets:
            return self
        dataset = self.dis.dataset
        unknown = sorted(set(targets) - set(dataset.instruments))
        if unknown:
            raise ValueError(
                f"no epochs of instrument(s) {unknown}; the dataset has "
                f"{sorted(dataset.instruments)}"
            )
        declared = self.dis.noise_correlation
        for key in targets:
            correlation = declared.get(key, 0.0) if isinstance(declared, Mapping) else declared
            if correlation is not None and not (
                isinstance(correlation, int | float) and float(correlation) == 0.0
            ):
                raise ValueError(
                    f"instrument {key!r} declares a noise correlation ({correlation!r}). A "
                    "convolution changes the correlation of the noise, and the AR(1) model "
                    "of the declaration would no longer describe it."
                )
        widths = {}
        for key in targets:
            spec = _coerce_lsf(self.dis.lsf[key], key)
            if spec.anchors_angstrom is not None:
                raise ValueError(
                    f"instrument {key!r} declares a wavelength-dependent LSF, which smoothing "
                    "to one width does not reproduce; declare one width for the instrument"
                )
            widths[key] = spec
        epochs, diagnostics = [], {key: [] for key in targets}
        for epoch in dataset:
            key = epoch.instrument
            if key not in targets:
                epochs.append(epoch)
                continue
            spec = widths[key]
            own = epoch.lsf_sigma_kms if spec.is_per_epoch else float(spec.sigma_kms)
            smoothed_epoch, info = _smoothed(epoch, targets[key], own, fill_max)
            epochs.append(smoothed_epoch)
            diagnostics[key].append({**info, "from_sigma_kms": float(own)})
        lsf = dict(self.dis.lsf)
        for key, target in targets.items():
            lsf[key] = LSF(sigma_kms=target)
        dis = replace(self.dis, dataset=Dataset(tuple(epochs), frame=dataset.frame), lsf=lsf)
        entries = []
        for key, target in targets.items():
            rows = diagnostics[key]
            entries.append(
                {
                    "instrument": key,
                    "sigma_kms": target,
                    "from_sigma_kms": (
                        min(r["from_sigma_kms"] for r in rows),
                        max(r["from_sigma_kms"] for r in rows),
                    ),
                    "kernel_sigma_kms": (
                        min(r["kernel_sigma_kms"] for r in rows),
                        max(r["kernel_sigma_kms"] for r in rows),
                    ),
                    "sum_k2": float(np.median([r["sum_k2"] for r in rows])),
                    "kept_fraction": sum(r["kept"] for r in rows) / sum(r["good"] for r in rows),
                    "filled": int(sum(r["filled"] for r in rows)),
                    "n_epochs": len(rows),
                }
            )
        return replace(
            self,
            dis=dis,
            phase_scan=None,
            k_scan=None,
            smoothing=(*(self.smoothing or ()), *entries),
        )

    def match_labels(self, stars, *, smoothing=None, **kwargs):
        """Fit Teff, log g, [M/H] and v sin i to the stellar components of this fit.

        This declarative form of :func:`albireo.match_labels` takes the grid, the recovered
        spectra, their uncertainty band, the assumed light fractions, the instrument width
        and the dataset's wavelength medium from this fit, so they are consistent with
        what was solved.

        The comparison defaults to ``compare="epochs"``: the template composite is compared
        with the epoch spectra through :meth:`epoch_statistics`. The statistics are built
        here at the resolving power the stars' libraries declare, with the model grid's
        smoothing, ``(7/12) dv^2``, removed from the operator width.
        ``statistics=fit.epoch_statistics(..., grid_compensation=False)`` leaves it in.
        The epoch comparison is exact for the fit's noise model and does not depend on the
        declared light fractions. On simulated Gaia RVS binaries it returned temperatures
        twice as close to the injected values as the diagonal comparison against ``d_hat``
        converged the same way (``docs/math.md`` §9.2a). ``compare="native"`` and
        ``compare="matched"`` compare with the disentangled components instead.

        A library spectrum cannot be sharpened, so the epoch comparison raises where the
        library is at or below an instrument's resolving power. With ``smoothing`` the
        epochs of such an instrument are smoothed to the library's resolution instead
        (:meth:`smoothing_widths`, :meth:`smoothed`), and the labels and frame offsets are
        measured there. The statistics record it in their notes, the labels then carry the
        library's resolution, and a v sin i below the smoothed width is not resolved
        (``docs/math.md`` §9.2b). On two binaries observed at R = 44,000-56,000 the frame
        offsets agreed with those from libraries sharper than the data to 0.02-0.07 km/s in
        the systemic velocity (D74).

        Only the stellar rows are passed on: a telluric or nebular component has no
        atmospheric parameters, and the epoch comparison holds it at its posterior mean.
        The light fractions are passed with the stars because the label fit's dilution
        model is defined in terms of the assumed fractions.

        Parameters
        ----------
        stars
            Mapping of star name to :class:`albireo.StarLabels`, one per stellar component
            of this declaration. Names must match :attr:`Disentangler.stars`.
        smoothing
            ``None`` (default) compares the epochs as observed. ``"auto"``, or a width
            ``sigma_q`` in km/s, smooths the epochs of every instrument at or below whose
            resolving power the library is (:meth:`smoothing_widths`); instruments the
            library resolves are compared as observed. Only for the epoch comparison, and
            not with ``statistics``.
        **kwargs
            Passed to :func:`albireo.match_labels`. ``statistics`` may be passed to reuse
            statistics built once with :meth:`epoch_statistics`.

        Returns
        -------
        LabelMatch

        Raises
        ------
        ValueError
            If ``stars`` names a component that is not a star of this declaration, if it
            omits one, if the dataset does not declare its wavelength medium, or, for the
            epoch comparison, if the stars' libraries declare different resolving powers,
            if the library is at or below an instrument's resolving power and ``smoothing``
            is not given, or if ``smoothing`` is given with another comparison or with
            ``statistics``.

        Notes
        -----
        The dataset must declare its wavelength medium: air and vacuum wavelengths differ
        by a nearly constant 83 km/s, and there is no safe default (``docs/math.md`` §9).
        """
        from albireo.match import match_labels

        names = [component.name for component in self.dis.stars]
        unknown = set(stars) - set(names)
        if unknown:
            raise ValueError(
                f"unknown star(s) {sorted(unknown)}; this declaration has {names}. "
                "Telluric and nebular components are not stars and have no labels."
            )
        if set(stars) != set(names):
            raise ValueError(
                f"declare labels for every star ({names}); the dilution model fits the "
                "components jointly, so a partial declaration is not well posed."
            )
        medium = _declared_medium(self.dis.dataset)
        ordered = {name: stars[name] for name in names}
        compare = kwargs.setdefault("compare", "epochs")
        if smoothing is not None and compare != "epochs":
            raise ValueError(
                f"smoothing= applies to the epoch comparison, and compare={compare!r} compares "
                "the templates with the disentangled components"
            )
        if smoothing is not None and kwargs.get("statistics") is not None:
            raise ValueError(
                "pass either smoothing= or statistics=: statistics built elsewhere fix the "
                "epochs they were built from (Fit.smoothed(...).epoch_statistics(...) builds "
                "them from smoothed epochs)"
            )
        if compare == "epochs" and kwargs.get("statistics") is None:
            declared = {name: ordered[name].library.resolving_power for name in names}
            if len(set(declared.values())) > 1:
                raise ValueError(
                    f"the stars' libraries declare different resolving powers {declared}; the "
                    "epoch comparison applies one operator width per instrument, so every "
                    "star's template must start at the same resolution. Use libraries at one "
                    "resolving power, or pass compare='native'."
                )
            resolving = next(iter(declared.values()))
            if smoothing is not None:
                targets = self.smoothing_widths(resolving, smoothing)
                if targets:
                    return self.smoothed(targets).match_labels(stars, **kwargs)
            if len(self.dis.component_names) > self.dis.n_stellar:
                # The statistics hold a telluric or nebular row at its posterior mean, and
                # the components passed on below need the same solve with its errors, so
                # one solve serves both.
                self._solved(std=True)
            kwargs["statistics"] = self.epoch_statistics(resolving_power=resolving)
        spectra, std = self._spectra_and_std()
        rows = [self.dis.component_names.index(name) for name in names]
        kwargs.setdefault("lsf_sigma_kms", self.dis._widest_lsf())
        return match_labels(
            self.dis.grid,
            spectra[rows],
            stars=ordered,
            medium=medium,
            light_fractions=[component.light for component in self.dis.stars],
            std=std[rows],
            **kwargs,
        )

    def _epoch_models(self, spectra) -> list[np.ndarray]:
        """The model flux of every epoch on its own pixels, from ``spectra`` at this fit."""
        from albireo.forward import apply_model

        problem = self.dis.model.problem_at(self.theta)
        out: list[np.ndarray] = [np.empty(0)] * self.dis.dataset.n_epochs
        for group, model in zip(
            problem.groups, apply_model(problem, jnp.asarray(spectra)), strict=True
        ):
            flux = np.asarray(group.r) * (np.asarray(group.base)[None, :] + np.asarray(model))
            for local, index in enumerate(group.epoch_indices):
                out[int(index)] = flux[local]
        return out

    def _shared_low_frequency(self, scale_angstrom: float) -> np.ndarray:
        """The spectra with the low-frequency part of every star replaced by the composite's.

        Each stellar row keeps its structure on scales below ``scale_angstrom`` (a Gaussian
        of that width separates the two) and takes the low-frequency part of the
        light-weighted composite, which the data determine. The composite is unchanged,
        because the light fractions sum to one. Telluric and nebular rows are unchanged.

        The Gaussian is applied in model pixels, with ``scale_angstrom`` converted at the
        median wavelength step of the grid. The grid is uniform in log wavelength, so the
        width is ``scale_angstrom`` at the grid's middle wavelength ``lambda_m`` (the
        geometric mean of its ends) and ``scale_angstrom * lambda / lambda_m`` elsewhere: a
        constant width in velocity. On a grid from 3700 to 6900 A, a scale of 12 A is
        8.8 A at the blue end and 16.4 A at the red end.
        """
        from scipy.ndimage import gaussian_filter1d

        spectra = np.array(self.spectra(), dtype=np.float64)
        n_stellar = self.dis.n_stellar
        if n_stellar < 2:
            return spectra
        lights = np.array([s.light for s in self.dis.stars], dtype=np.float64)
        step = float(np.median(np.diff(np.asarray(self.dis.grid.wave))))
        width = float(scale_angstrom) / step
        composite = np.tensordot(lights, spectra[:n_stellar], axes=(0, 0))
        shared = gaussian_filter1d(composite, width, mode="nearest")
        for i in range(n_stellar):
            spectra[i] = spectra[i] - gaussian_filter1d(spectra[i], width, mode="nearest") + shared
        return spectra

    def renormalize_epochs(
        self,
        *,
        scale_angstrom: float = 12.0,
        passes: int = 3,
        clip: float = 3.0,
        exclude=(),
    ) -> EpochRenormalization:
        """Divide every epoch by a smooth correction fitted to its ratio to this fit's model.

        The continuum normalization of an echelle spectrum leaves errors of 1-3% on the
        scale of the orders (a residual blaze), and they differ between epochs. The
        component spectra are static, so the model cannot follow them, except through the
        small Doppler shifts between epochs. Two stars whose low-frequency parts are
        shifted against each other can follow part of them, with opposite drifts of
        several continuum units in the two components that their light-weighted sum does
        not show, and ML-II lowers the continuum weight ``eta`` to allow it
        (``docs/method.md``, section 10). This method removes the errors from the epochs.

        Each epoch is divided by a cubic spline with knots every ``scale_angstrom``,
        fitted to the ratio of its flux to its model (:func:`albireo.continuum_correction`).
        The model is computed at this fit's parameters from the component spectra with the
        low-frequency part of every star replaced by that of the light-weighted composite:
        structure on scales above ``scale_angstrom`` is common to the stars in it. With the
        spectra as fitted, the drifts would be part of the model, which would follow the
        errors it is meant to remove, and the iteration converges slowly. The spectra are
        then solved again on the corrected epochs at the same parameters, and the step is
        repeated ``passes`` times.

        The result is data: refit them, ``dataclasses.replace(fit.dis,
        dataset=renorm.dataset).fit()``, or, at the parameters of this fit without a new
        optimization, ``dataclasses.replace(fit, dis=dataclasses.replace(fit.dis,
        dataset=renorm.dataset))``. On a simulated binary with a ripple of 1.3% rms on
        scales of 40-80 A in every epoch, ML-II had lowered ``eta`` from 5 to 0.4 and
        0.025 and the components drifted by 0.28 and 0.85 continuum units rms. After three
        passes and a refit, ``eta`` was 6.9 and 7.8, the drifts 0.008 and 0.028, and the
        residual z-score RMS 1.01 instead of 1.55, as in a fit to the same binary without
        the ripple (6.7 and 13.3; 0.008 and 0.025; 1.01).

        The correction cannot separate a normalization error common to every epoch from
        the spectra, and leaves it in the composite. Lines broader than ``scale_angstrom``
        whose shapes differ between the stars (Balmer wings) are represented in the model
        by the composite's shape, which differs from the stars' own by about the relative
        shift times the slope of the wings: ``exclude`` keeps them out of the corrections.

        Parameters
        ----------
        scale_angstrom
            The knot spacing of the corrections, and the width of the low-frequency part
            the stars share in the model [A]. It must be well below the scale of the
            normalization errors and well above the width of the lines. On PEPSI spectra
            with a ripple of 40-80 A, 12 A removed it and 30 A did not. The knots are
            spaced in Angstrom; the width of the shared part is this value at the middle
            wavelength of the model grid and is constant in velocity, so it scales with
            wavelength across a wide grid.
        passes
            Number of passes. The corrections of the second pass are typically a tenth of
            the first's.
        clip
            Rejection threshold of the correction fit, in robust standard deviations.
        exclude
            Wavelength ranges ``(lo, hi)`` [A] left out of the correction fits.

        Returns
        -------
        EpochRenormalization

        Raises
        ------
        ValueError
            If ``scale_angstrom`` is not positive, ``passes`` is below 1, or a range of
            ``exclude`` is not a pair ``(lo, hi)`` with ``lo < hi``.

        Warns
        -----
        RuntimeWarning
            If the correction of an epoch cannot be fitted (too few usable pixels); the
            epoch is then left unchanged in that pass.
        """
        from albireo.preprocess import _replace, continuum_correction

        if not (np.isfinite(scale_angstrom) and scale_angstrom > 0.0):
            raise ValueError(f"scale_angstrom must be positive; got {scale_angstrom}")
        if int(passes) < 1:
            raise ValueError(f"passes must be at least 1; got {passes}")
        # Read once, so that every epoch of every pass leaves out the same ranges (a
        # generator would be exhausted by the first epoch), and checked here, so that a
        # malformed range fails the call rather than every epoch's correction.
        try:
            ranges = tuple((float(lo), float(hi)) for lo, hi in exclude)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"exclude must be a sequence of (lo, hi) wavelength ranges [A]; got {exclude!r}"
            ) from exc
        for lo, hi in ranges:
            if not hi > lo:
                raise ValueError(
                    f"each excluded range must be (lo, hi) with lo < hi; got {(lo, hi)!r}"
                )
        grid = self.dis.grid
        fit = self
        totals = [np.ones(epoch.wave.size) for epoch in self.dis.dataset]
        rms = []
        for _ in range(int(passes)):
            models = fit._epoch_models(fit._shared_low_frequency(scale_angstrom))
            epochs, row = [], []
            for j, epoch in enumerate(fit.dis.dataset):
                ivar = np.asarray(epoch.effective_ivar, dtype=np.float64)
                try:
                    correction = continuum_correction(
                        epoch.wave,
                        epoch.flux,
                        ivar,
                        models[j],
                        knot_spacing_angstrom=scale_angstrom,
                        clip=clip,
                        exclude=ranges,
                    )
                except ValueError as exc:
                    warnings.warn(
                        f"epoch {j}: {exc}; the epoch is left unchanged in this pass",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    correction = np.ones(epoch.wave.size)
                totals[j] = totals[j] * correction
                good = ivar > 0.0
                deviation = correction[good] - 1.0
                row.append(float(np.sqrt(np.mean(deviation**2))) if deviation.size else 0.0)
                epochs.append(
                    _replace(
                        epoch,
                        flux=np.asarray(epoch.flux) / correction,
                        ivar=np.asarray(epoch.ivar) * correction**2,
                    )
                )
            rms.append(row)
            dis = replace(fit.dis, dataset=Dataset(epochs, frame=fit.dis.dataset.frame))
            if (dis.grid.x0, dis.grid.dx, dis.grid.n) != (grid.x0, grid.dx, grid.n):
                raise RuntimeError(  # pragma: no cover - the wavelengths are unchanged
                    "the model grid changed when the epochs were renormalized"
                )
            fit = replace(fit, dis=dis)
        return EpochRenormalization(
            dataset=fit.dis.dataset,
            corrections=tuple(totals),
            rms=np.asarray(rms, dtype=np.float64),
            scale_angstrom=float(scale_angstrom),
        )

    def write_spectra(self, path, **kwargs):
        """Write the component spectra and their uncertainty band to FITS.

        Requires astropy.

        Parameters
        ----------
        path
            Destination path.
        **kwargs
            Passed to :func:`albireo.write_spectra`.
        """
        spectra, std = self._spectra_and_std()
        return _write_spectra(self.dis, path, spectra, std, **kwargs)

    def templates(self, *, pixels_per_sigma: float = 3.0) -> list:
        """The stellar components of this fit as :class:`albireo.todcor.Template` objects.

        A system's own disentangled components are the closest templates available for
        it: measured from these epochs, they have the correct lines, depths and rotational
        broadening. They have no absolute rest frame, since each component has its own
        unidentified zero point (``docs/math.md`` §5.3, §7.6). Every template returned
        here therefore has ``v_zero_kms=None``, and the velocities measured against it are
        differential. A label match (:meth:`match_labels`, then
        :meth:`albireo.Template.from_labels`) fixes the zero point.

        Telluric and nebular components are not stars and are not returned. The templates
        are intrinsic (``sigma_kms=0``): the fit solved for the spectra under the declared
        LSF, which :meth:`measure_velocities` applies again per instrument.

        The default model grid has one point per native pixel of the most finely sampled
        epoch. This is adequate for the solver but can be too coarse for a correlation
        template: the pixel-locking ripple of the linear shift operator is
        ``~0.1 / sigma_px^2`` pixels (``docs/math.md`` §10.3). The components are
        linearly upsampled by an integer factor onto a grid with at least
        ``pixels_per_sigma`` pixels per narrowest declared LSF sigma. The smoothness prior
        has band-limited them well above that scale, so the upsampling adds no information
        and removes the ripple.

        Parameters
        ----------
        pixels_per_sigma
            Minimum number of pixels of the upsampled grid per narrowest declared LSF
            sigma. Values of 3 or more keep the pixel-locking ripple below the measured
            velocity precision (``docs/math.md`` §10.3).

        Returns
        -------
        list of albireo.todcor.Template
            One template per stellar component, in declaration order.
        """
        from albireo.todcor import Template

        grid = self.dis.grid
        narrowest = self.dis._narrowest_lsf()
        factor = max(1, math.ceil(pixels_per_sigma * grid.dv_kms / narrowest))
        fine = LogGrid(
            x0=grid.x0,
            dx=grid.dx / factor,
            n=(grid.n - 1) * factor + 1,
            relativistic=grid.relativistic,
        )
        spectra = self.spectra()
        out = []
        for star in self.dis.stars:
            row = self.dis.component_names.index(star.name)
            deviation = spectra[row] if factor == 1 else np.interp(fine.x, grid.x, spectra[row])
            out.append(
                Template(
                    name=star.name,
                    grid=fine,
                    deviation=deviation,
                    sigma_kms=0.0,
                    v_zero_kms=None,
                    meta={
                        "source": "disentangling",
                        "light": star.light,
                        "mode": self.mode,
                        "upsampled": factor,
                    },
                )
            )
        return out

    def measure_velocities(self, *, templates=None, light=None, v_range=None, **kwargs):
        """Per-epoch velocities of every star by TODCOR against this fit's own components.

        The disentangling yields the component spectra. Correlating the epochs against
        them yields one velocity per star per epoch (:func:`albireo.todcor`), which the
        joint fit does not produce. The declared light fractions are held fixed by default
        because the components were solved against them (``docs/math.md`` §9.1: what is
        recovered is ``(w/l0) t``, so ``l0`` is the only amplitude consistent with those
        spectra). The declared LSF is applied per instrument as the fit applied it.

        Parameters
        ----------
        templates
            Defaults to :meth:`templates`. Absolute templates, from a library or a label
            match, give absolute rather than differential velocities.
        light
            Defaults to the declared fractions when the templates are this fit's own, and
            to ``"global"`` otherwise.
        v_range
            Search range in km/s, one ``(lo, hi)`` pair or one per template. Defaults to
            the span of the fitted velocities widened by 40 km/s at each end, offset per
            template by that template's zero point relative to their median. Every
            component then searches the same interval of reported velocity: the range is in
            each template's own frame, and a template with a zero point reports its
            velocities composed with it. When the templates' zero points disagree by more
            than the fitted velocities span, no such interval exists and an error is
            raised.

        **kwargs
            Passed to :func:`albireo.todcor`.

        Returns
        -------
        albireo.todcor.VelocityTable

        Raises
        ------
        ValueError
            If ``v_range`` is left to the default and the templates' zero points make the
            implied window miss one component's own fitted velocities.

        References
        ----------
        Zucker, S. & Mazeh, T. 1994, ApJ, 420, 806
        """
        from albireo.todcor import todcor

        own = templates is None
        if own:
            templates = self.templates()
        if light is None:
            light = [s.light for s in self.dis.stars] if own else "global"
        kwargs.setdefault("noise_correlation", self.noise_correlation())
        if v_range is None:
            v_range = _default_v_range(np.asarray(self.velocities()), templates)
        lsf_sigma, anchors = {}, {}
        for key, value in self.dis.lsf.items():
            spec = _coerce_lsf(value, key)
            lsf_sigma[key] = spec.sigma_kms
            if spec.anchors_angstrom is not None:
                anchors[key] = tuple(spec.anchors_angstrom)
        return todcor(
            self.dis.dataset,
            templates,
            v_range=v_range,
            light=light,
            lsf_sigma_v=lsf_sigma,
            lsf_anchors_angstrom=anchors or None,
            **kwargs,
        )


@dataclass(frozen=True)
class Posterior:
    """A NUTS posterior over the orbit, with the fit it was started from.

    The draws are of the orbital parameters only. The component spectra were marginalized
    at every draw, and :meth:`spectra` recovers them, one conditional draw per posterior
    draw (``docs/method.md``).

    Attributes
    ----------
    dis
        The :class:`Disentangler` the posterior was declared from.
    mcmc
        The numpyro ``MCMC`` object returned by :func:`albireo.run_nuts`.
    map
        The :class:`Fit` whose MAP provided the starting point and the ML-II
        hyperparameters.
    plug_in_widening
        The factor by which freeing the regularization weights widens the Laplace standard
        deviation of each sampled coordinate at the best fit, keyed as
        :attr:`LaplaceCheck.labels` (``"k[1]"``, ``"period"``). It is the part of the
        uncertainty that holding the weights leaves out, in the Laplace approximation, and
        :meth:`summary` reports its largest value. ``None`` where :meth:`Fit.sample` did not
        compute it, with the reason in ``plug_in_note``.
    plug_in_note
        Why ``plug_in_widening`` was not computed; empty otherwise.
    """

    dis: Disentangler
    mcmc: Any
    map: Fit
    plug_in_widening: dict[str, float] | None = None
    plug_in_note: str = ""

    @property
    def samples(self) -> dict:
        """The posterior draws, keyed by site."""
        return self.mcmc.get_samples()

    def star(self, name: str) -> dict:
        """Posterior summary for one named star's semi-amplitude.

        Parameters
        ----------
        name
            Name of a :class:`Star` in this declaration.

        Returns
        -------
        dict
            ``name``, ``light``, the posterior mean ``k`` and standard deviation ``k_std``
            in km/s, and ``k_hdi``, the 95% highest-density interval: the narrowest
            interval that contains 95% of the draws. A semi-amplitude the declaration fixed
            is not sampled, and is reported at its value with ``k_std`` 0.

        Raises
        ------
        KeyError
            If ``name`` is not a star of this declaration.
        """
        index = _star_index(self.dis, name)
        out = {"name": name, "light": self.dis.stars[index].light}
        samples = self.samples
        if "k" not in samples:
            value = float(np.atleast_1d(np.asarray(self.dis.fixed["k"], dtype=float))[index])
            return {**out, "k": value, "k_std": 0.0, "k_hdi": (value, value)}
        # One star's k is a scalar site, whose draws have no component axis.
        draws = np.asarray(samples["k"])
        k = draws.reshape(draws.shape[0], -1)[:, index]
        low, high = hpdi(k, prob=0.95)
        return {
            **out,
            "k": float(np.mean(k)),
            "k_std": float(np.std(k)),
            "k_hdi": (float(low), float(high)),
        }

    def spectra(self, *, num_draws: int = 32, seed: int = 0) -> np.ndarray:
        """Spectra drawn from the joint posterior, ``(num_draws, n_comp, n_pix)``.

        The scatter across draws includes the conditional spectral uncertainty and the
        orbital uncertainty. The ML-II hyperparameters are injected automatically;
        omitting them fails silently, because those sites are absent from the chain.

        Parameters
        ----------
        num_draws
            Number of posterior draws.
        seed
            Seed for the numpyro PRNG key.

        Returns
        -------
        numpy.ndarray
            Deviations from a unit continuum, shape ``(num_draws, n_comp, n_pix)``.
        """
        return np.asarray(
            posterior_spectra(
                self.dis.model,
                self.samples,
                jax.random.PRNGKey(int(seed)),
                num_draws=num_draws,
                extra=self.map._fixed_hyper(),
            )
        )

    def summary(self) -> str:
        """Report the posterior intervals, sampler diagnostics, and the assumptions.

        Each row gives the posterior mean, the standard deviation and the 95%
        highest-density interval of one quantity, and two convergence diagnostics: the
        split Gelman-Rubin statistic R-hat, which is close to 1 when the chains and the
        halves of each chain agree, and the effective sample size (ESS), the number of
        independent draws that the correlated chains are equivalent to. Both are the
        classic estimators of :mod:`numpyro.diagnostics`, not the rank-normalized ones.
        ``ecc`` and ``omega`` are derived from the sampled ``secosw`` and ``sesinw``.
        Sampled sites without a row of their own (a noise correlation, nebular amplitudes)
        are summarized by their largest R-hat and smallest ESS. The paragraph on the plug-in
        approximation gives the largest widening of a standard deviation that freeing the
        regularization weights gives in the Laplace approximation (:attr:`plug_in_widening`).

        Returns
        -------
        str
            A multi-line report ending with :meth:`Disentangler.assumptions`.
        """
        grouped = {
            name: np.asarray(values)
            for name, values in self.mcmc.get_samples(group_by_chain=True).items()
        }
        n_chains, n_draws = next(iter(grouped.values())).shape[:2]
        sampled = [name for name in grouped if name not in _DERIVED_SITES]
        n_parameters = sum(int(np.prod(grouped[name].shape[2:])) for name in sampled)
        lines = [
            f"NUTS posterior: {n_chains} chain(s) x {n_draws} draws of {n_parameters} "
            f"parameters ({', '.join(sampled)})"
        ]
        extra = getattr(self.mcmc, "get_extra_fields", lambda: {})()
        if "diverging" in extra:
            lines.append(f"  divergences: {int(np.sum(np.asarray(extra['diverging'])))}")
        lines += [
            "",
            f"  {'':<24s}{'mean':>13s}{'sd':>11s}   {'95% HDI':<27s}{'R-hat':>6s}{'ESS':>7s}",
        ]
        rows = [
            (label, grouped[site], digits)
            for site, label, digits in (("period", "period [d]", 6), ("t_conj", "t_conj [d]", 5))
            if site in grouped
        ]
        if "secosw" in grouped and "sesinw" in grouped:
            secosw, sesinw = grouped["secosw"], grouped["sesinw"]
            omega = grouped.get("omega", np.arctan2(sesinw, secosw))
            rows.append(("ecc", grouped.get("ecc", secosw**2 + sesinw**2), 4))
            rows.append(("omega [rad]", _about_circular_mean(omega), 4))
        if "k" in grouped:
            # One star's k is a scalar site, whose draws have no component axis.
            k = grouped["k"].reshape(n_chains, n_draws, -1)
            rows += [
                (f"K({star.name}) [km/s]", k[..., i], 3) for i, star in enumerate(self.dis.stars)
            ]
        lines += [_posterior_row(label, values, digits) for label, values, digits in rows]
        for site in ("ecc", "omega"):
            if site in grouped and "secosw" not in grouped:
                lines.append(f"  {site:<24s}held at {float(grouped[site].flat[0]):g}")
        others = [name for name in sampled if name not in _ROW_SITES]
        if others:
            pairs = [
                _chain_diagnostics(column)
                for name in others
                for column in grouped[name].reshape(n_chains, n_draws, -1).transpose(2, 0, 1)
            ]
            rhats = [r for r, _ in pairs if np.isfinite(r)]
            sizes = [e for _, e in pairs if np.isfinite(e)]
            worst = (
                f"largest R-hat {max(rhats):.3f}, smallest ESS {min(sizes):.0f}"
                if rhats and sizes
                else "no defined diagnostic"
            )
            lines.append(f"  other sampled sites ({', '.join(others)}): {worst}")
        lines += [
            "",
            "  The regularization weights were held at their ML-II values, so these intervals",
            "  do not include their uncertainty (a plug-in approximation, not a marginalization).",
        ]
        if self.plug_in_widening:
            label, factor = max(self.plug_in_widening.items(), key=lambda item: item[1])
            if label.startswith("k[") and label.endswith("]"):
                label = f"K({self.dis.stars[int(label[2:-1])].name})"
            sentence = (
                "In the Laplace approximation at the best fit, freeing the weights would widen "
                "the standard deviations of the sampled parameters by at most "
                f"{_percent(factor - 1.0)} ({label})."
            )
        elif self.plug_in_note:
            sentence = (
                f"The widening from freeing the weights was not computed: {self.plug_in_note}."
            )
        else:
            sentence = ""
        lines += textwrap.wrap(sentence, width=88, initial_indent="  ", subsequent_indent="  ")
        lines += ["", self.dis.assumptions()]
        return "\n".join(lines)

    def to_inference_data(self, **kwargs):
        """Convert to an arviz ``InferenceData``, with the component names attached.

        Parameters
        ----------
        **kwargs
            Passed to :func:`albireo.to_inference_data`.
        """
        from albireo.results import to_inference_data

        return to_inference_data(
            self.mcmc, component_names=list(self.dis.component_names), **kwargs
        )

    def write_spectra(self, path, *, num_draws: int = 32, seed: int = 0, **kwargs):
        """Write the posterior-mean spectra and their scatter across draws to FITS.

        Parameters
        ----------
        path
            Destination path.
        num_draws
            Number of posterior draws to average.
        seed
            Seed for the numpyro PRNG key.
        **kwargs
            Passed to :func:`albireo.write_spectra`.
        """
        draws = self.spectra(num_draws=num_draws, seed=seed)
        return _write_spectra(self.dis, path, draws.mean(axis=0), draws.std(axis=0), **kwargs)


# -- helpers ------------------------------------------------------------------

# Deterministic sites of `MarginalOrbitModel.model`: recorded with the draws, not sampled.
_DERIVED_SITES = frozenset({"ecc", "omega", "ecc_out", "omega_out", "velocity_rel", "nebular_amp"})
# The sites of the Keplerian. A free velocity table replaces them, and the model rejects
# them alongside it (`MarginalOrbitModel._theta_problem`).
_KEPLERIAN_SITES = frozenset({"period", "t_conj", "secosw", "sesinw", "k"})
# Sampled sites that `Posterior.summary` reports as rows of their own: the Keplerian's.
_ROW_SITES = _KEPLERIAN_SITES


def _without_orbit(sites: Mapping) -> dict:
    """``sites`` without the Keplerian ones, for a model with a free velocity table."""
    return {name: value for name, value in sites.items() if name not in _KEPLERIAN_SITES}


def _star_index(dis: Disentangler, name: str) -> int:
    """The model row of the star called ``name``; ``KeyError`` naming the stars otherwise."""
    names = [s.name for s in dis.stars]
    if name not in names:
        raise KeyError(f"no star called {name!r}; this fit has {names}")
    return names.index(name)


def _declared_medium(dataset: Dataset) -> str:
    """The wavelength medium of ``dataset``, which a comparison with a synthetic grid needs."""
    medium = dataset[0].medium
    if medium is None:
        raise ValueError(
            "this dataset does not declare whether its wavelengths are air or vacuum, "
            "so it cannot be matched against a synthetic grid: the two differ by ~83 "
            "km/s. Set medium= on the epochs (albireo.air_to_vacuum and "
            "albireo.vacuum_to_air convert)."
        )
    return medium


def _write_spectra(dis: Disentangler, path, spectra, std, **kwargs):
    """Write ``spectra`` and ``std`` with the grid, lights, prior and names of ``dis``."""
    from albireo.io import write_spectra

    return write_spectra(
        path,
        dis.grid,
        spectra,
        std,
        light_fractions=[s.light for s in dis.stars],
        prior=dis.smoothness_prior,
        meta={"COMPNAME": ",".join(dis.component_names)},
        **kwargs,
    )


def _finite_range(values) -> float:
    """The largest minus the smallest finite entry of ``values``; NaN if none is finite."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.max(finite) - np.min(finite)) if finite.size else math.nan


def _chain_diagnostics(values: np.ndarray) -> tuple[float, float]:
    """Split R-hat and effective sample size of one scalar quantity, NaN where undefined.

    ``values`` has shape ``(n_chains, n_draws)``. Split R-hat requires at least four
    draws per chain, and neither statistic is defined for a constant quantity.
    """
    if values.shape[1] < 4 or not np.std(values) > 0.0:
        return math.nan, math.nan
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(split_gelman_rubin(values)), float(effective_sample_size(values))


def _posterior_row(label: str, values: np.ndarray, digits: int) -> str:
    """One row of :meth:`Posterior.summary`: mean, sd, 95% HDI, split R-hat and ESS."""
    flat = values.reshape(-1)
    low, high = hpdi(flat, prob=0.95)
    rhat, ess = _chain_diagnostics(values)
    interval = f"[{float(low):.{digits}f}, {float(high):.{digits}f}]"
    rhat_text = f"{rhat:.3f}" if np.isfinite(rhat) else "n/a"
    ess_text = f"{ess:.0f}" if np.isfinite(ess) else "n/a"
    return (
        f"  {label:<24s}{np.mean(flat):13.{digits}f}{np.std(flat):11.{digits}f}   "
        f"{interval:<27s}{rhat_text:>6s}{ess_text:>7s}"
    )


def _percent(fraction: float) -> str:
    """A non-negative fraction as a percentage with two significant digits."""
    value = 100.0 * max(float(fraction), 0.0)
    if value >= 10.0:
        return f"{value:.0f}%"
    if value >= 0.001:
        return f"{value:.2g}%"
    return "less than 0.001%"


def _about_circular_mean(angle: np.ndarray) -> np.ndarray:
    """Angles in radians, wrapped into ``(m - pi, m + pi]`` about their circular mean ``m``.

    The mean, standard deviation and interval of an angle are then those of a draw set
    that does not straddle the branch cut of ``arctan2``: a set centred near ``pi`` keeps
    values above ``pi`` rather than wrapping them to near ``-pi``.
    """
    centre = math.atan2(float(np.mean(np.sin(angle))), float(np.mean(np.cos(angle))))
    return centre + np.angle(np.exp(1j * (angle - centre)))


def _smoothness_of(component: Component) -> Smoothness:
    return component.smoothness


def _start_of(spec) -> Any:
    return _coerce_spec(spec, "spec").start()


def _largest(value) -> float:
    """The largest entry of a scalar or array value, as a float."""
    return float(np.max(np.atleast_1d(np.asarray(value))))


def _hyper_of(params: Mapping, names: Sequence[str]) -> dict[str, dict[str, float]]:
    """The fitted hyperparameters, keyed by component name and in linear units."""
    tau = np.exp(np.atleast_1d(np.asarray(params["log_tau"], dtype=float)))
    eta = np.exp(np.atleast_1d(np.asarray(params["log_eta"], dtype=float)))
    return {name: {"tau": float(tau[i]), "eta": float(eta[i])} for i, name in enumerate(names)}


_AUTO_SMOOTHING_KMS = 3.0
"""The default width, in km/s, that the epoch operator applies to the templates after the
epochs are smoothed to a library broader than them (:meth:`Fit.smoothing_widths`). It is the
value of the D74 measurement on two binaries observed at R = 44,000-56,000 against BOSZ at
R = 20,000; 5 km/s moved the systemic velocity by 0.02 km/s there."""

_AUTO_SMOOTHING_PIXELS = 1.2
"""The least default width in model pixels. The grid compensation of the epoch operator is
floored below 0.91 pixel and approximate below one (:meth:`Fit.epoch_statistics`)."""


def _smoothing_notes(smoothing, resolving_power) -> list[str]:
    """One note per instrument whose epochs a fit was smoothed (:meth:`Fit.smoothed`)."""
    notes = []
    for entry in smoothing or ():
        sigma = float(entry["sigma_kms"])
        lo, hi = entry["from_sigma_kms"]
        own = f"{lo:.3f} km/s" if lo == hi else f"{lo:.3f}-{hi:.3f} km/s"
        resolving = C_KMS / (sigma * 2.0 * math.sqrt(2.0 * math.log(2.0)))
        library = (
            ""
            if resolving_power is None
            else f" the library (R = {float(resolving_power):,.0f}) is at least as broad as "
            f"their line-spread function ({own}), so"
        )
        notes.append(
            f"epochs of {entry['instrument']!r} smoothed to sigma {sigma:.3f} km/s "
            f"(R = {resolving:,.0f}) before the comparison:{library} the labels and the frame "
            f"offsets are measured at that resolution, and v sin i below about {sigma:.0f} "
            "km/s is not resolved. The epochs keep their own inverse variances on pixels the "
            "convolution correlates, so the formal errors are upper bounds on the noise part "
            f"and a perfect model has a chi-square of about sum(k^2) = {entry['sum_k2']:.3f} "
            f"per pixel ({entry['kept_fraction']:.1%} of the good pixels kept; "
            "docs/math.md 9.2b)"
        )
    return notes


def _has_scanned(value) -> bool:
    """Whether a declared semi-amplitude includes a scan axis."""
    value = _as_k(value)
    if isinstance(value, Scanned):
        return True
    if isinstance(value, Spec):
        return False
    return any(isinstance(item, Scanned) for item in value)


def _vector_spec(value, n: int, what: str) -> Spec:
    """A spec for a vector-valued site, declared either as one spec or as a sequence.

    A bare number is ``Fixed``; for a single star it gives a scalar site.
    """
    if isinstance(value, int | float):
        return _coerce_spec(value, what)
    if isinstance(value, Spec):
        return value
    specs = [_coerce_spec(item, f"each entry of {what}") for item in value]
    if len(specs) != n:
        raise ValueError(f"{what} has {len(specs)} entries but the model has {n} components")
    kinds = {type(s) for s in specs}
    if kinds == {Fixed}:
        return Fixed(jnp.stack([jnp.asarray(s.value, dtype=jnp.float64) for s in specs]))
    if kinds == {Between}:
        # Each entry's starting value is kept. A declaration such as
        # k=[Between(10, 90, start_at=30), Between(10, 90, start_at=60)] breaks the
        # component-swap degeneracy of a symmetric prior: the conjunction scan runs at
        # these starting values, and equal starts do not distinguish the stars.
        return Between(
            jnp.stack([jnp.asarray(s.lo, dtype=jnp.float64) for s in specs]),
            jnp.stack([jnp.asarray(s.hi, dtype=jnp.float64) for s in specs]),
            jnp.stack([jnp.asarray(s.start(), dtype=jnp.float64) for s in specs]),
        )
    if kinds == {Known}:
        return Known(
            jnp.stack([jnp.asarray(s.value, dtype=jnp.float64) for s in specs]),
            jnp.stack([jnp.asarray(s.sigma, dtype=jnp.float64) for s in specs]),
        )
    raise TypeError(
        f"{what} mixes {sorted(k.__name__ for k in kinds)}. A vector-valued site takes one "
        "distribution family, so give every entry the same kind: or one spec with "
        "sequence-valued arguments, e.g. Between([10.0, 5.0], [90.0, 70.0])."
    )


def _ecc_sites(orbit: Orbit, ecc_max: float, suffix: str = "") -> list[tuple[str, Spec]]:
    """The ``(secosw, sesinw)`` sites, handling a circular orbit exactly.

    The parameterization is singular at ``e = 0``, where the gradient is NaN and numpyro
    reports only "Cannot find valid initial parameters". A free eccentricity is therefore
    never started at the origin, and a declared circular orbit does not sample these sites,
    so no gradient is taken at the singular point.
    """
    ecc = orbit.ecc
    if isinstance(ecc, Fixed):
        e = float(np.asarray(ecc.value, dtype=float))
        if e == 0.0:
            return [(f"secosw{suffix}", Fixed(0.0)), (f"sesinw{suffix}", Fixed(0.0))]
        if orbit.omega is None:
            raise ValueError(
                "a Fixed non-zero eccentricity needs omega as well: e and omega together "
                "are one point in the (sqrt(e)cos w, sqrt(e)sin w) plane. Pass "
                "omega=ab.Fixed(radians)."
            )
        omega = float(np.asarray(_coerce_spec(orbit.omega, "orbit.omega").start(), dtype=float))
        root = math.sqrt(e)
        return [
            (f"secosw{suffix}", Fixed(root * math.cos(omega))),
            (f"sesinw{suffix}", Fixed(root * math.sin(omega))),
        ]
    if isinstance(ecc, Between):
        hi = float(np.asarray(ecc.hi, dtype=float))
        lo = float(np.asarray(ecc.lo, dtype=float))
        if hi > ecc_max:
            raise ValueError(f"orbit.ecc upper bound {hi} exceeds ecc_max={ecc_max}")
        if lo != 0.0:
            # e is not a coordinate here: the sampled pair is (sqrt(e)cos w, sqrt(e)sin w),
            # in which a lower bound on e is an annulus that a box prior cannot express.
            # Dropping the bound without notice would return an eccentricity below the
            # declared one, measured at half of it, so the declaration is rejected.
            raise ValueError(
                f"orbit.ecc lower bound must be 0; got {lo}. The sampled parameters are "
                "(sqrt(e)cos w, sqrt(e)sin w), in which a lower bound on e is an annulus "
                "rather than a box, so it cannot be expressed as a prior here. Use "
                "ecc=ab.Between(0.0, hi) and read the posterior, or ecc=ab.Fixed(e) with "
                "omega to hold it."
            )
        # The two sites are bounded independently, so the box corner reaches e = 2 * hi.
        # The declared bound is enforced by the model's disk factor instead, through
        # effective_ecc_max: without it, ecc=Between(0, 0.08) allows e = 0.16.
        limit = math.sqrt(hi)
        # The start is e = 0.05, omega = 0.5 rad unless the declaration gives one: small,
        # but not at the singular origin. Starting from a velocity table's orbit avoids
        # scanning an eccentric orbit with a circular model.
        e_start = 0.05 if ecc.start_at is None else float(np.asarray(ecc.start_at, dtype=float))
        e_start = min(max(e_start, 0.01), 0.95 * hi) if hi > 0.0 else 0.0
        omega = 0.5
        if orbit.omega is not None:
            omega = float(np.asarray(_coerce_spec(orbit.omega, "orbit.omega").start(), dtype=float))
        start = math.sqrt(e_start)
        return [
            (f"secosw{suffix}", Between(-limit, limit, start * math.cos(omega))),
            (f"sesinw{suffix}", Between(-limit, limit, start * math.sin(omega))),
        ]
    raise TypeError(f"orbit.ecc must be Fixed or Between; got {type(ecc).__name__}")

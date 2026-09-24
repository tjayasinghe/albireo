"""Declarative interface to a disentangling fit: describe the system, obtain the model.

**Experimental.** The vocabulary defined here may change;
:class:`~albireo.inference.MarginalOrbitModel` and the functions around it are the
supported surface, and :meth:`Disentangler.expert` returns that surface directly.

A declaration names the components (one :class:`Star` per stellar component, optionally a
:class:`Telluric` column and a :class:`Nebular` component), either an :class:`Orbit` of
priors or a measured per-epoch velocity table, the line-spread function of each
instrument, and the dataset. From that declaration the module derives what the low-level
path otherwise requires: the velocity budget, the model grid and its margins, the
conjunction-phase scan, the matched ``priors``/``init`` pair, the empirical-Bayes (ML-II)
smoothness step, and the Laplace mass matrix. :meth:`Disentangler.explain` prints every
derivation, and :meth:`Disentangler.expert` returns the ``(model, priors, init)`` triple
that the declaration compiles to.

Quantities that are scientific claims are not defaulted. Light fractions are required:
with constant light fractions the likelihood depends only on the products ``l_i * d_i``
(``docs/math.md`` §5.2), so every recovered line depth scales as ``1 / l_i`` and no part
of the fit can contradict the assumed value. The same applies to the period prior, to the
wavelength medium wherever absolute line positions matter, and to the nebular velocity.
Each of these is either required, refused, or listed in the ``Assumed, not measured``
block printed by :meth:`Fit.summary`.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from albireo.data import Dataset
from albireo.forward import PER_EPOCH, data_residual_zscores, declared_lsf_widths
from albireo.grids import C_KMS, LogGrid
from albireo.inference import (
    MarginalOrbitModel,
    keplerian_residuals,
    laplace_inverse_mass,
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

__all__ = [
    "LSF",
    "Between",
    "Disentangler",
    "Fit",
    "Fixed",
    "Known",
    "Nebular",
    "Orbit",
    "Posterior",
    "Sampled",
    "Scanned",
    "Smoothness",
    "Star",
    "Telluric",
]

# Barycentric motion, one way, in km/s. Reserved in the velocity budget whenever the model
# carries a component whose velocity law includes v_bary: a telluric column, or
# topocentric data.
_V_BARY_MAX = 30.0

# Numerical headroom. The terms it multiplies are already a worst case (the sum of the
# priors' own upper bounds at the largest eccentricity they allow), so a larger slack would
# double count, and it has a cost: the solver bandwidth grows with the budget and the solve
# cost grows with the bandwidth.
_BUDGET_SLACK = 1.05
# Room a `velocities=` declaration leaves the free table to move. A convention, not a
# measurement: the declared velocities are a starting point, and the fit must be able to
# travel without reaching the bandwidth guard. Reported as its own budget term.
_FREE_VELOCITY_HEADROOM = 2.0


# -- parameter specifications -------------------------------------------------
#
# Each specification is a (distribution, starting value) pair rather than a distribution
# alone, so that the prior and the initial value are declared together. This makes
# `set(priors) == set(init)` hold by construction.


class Spec:
    """Base class for a declared parameter. Not instantiated directly."""

    def distribution(self) -> dist.Distribution | None:
        """The numpyro prior, or ``None`` when this parameter is not sampled."""
        raise NotImplementedError

    def start(self):
        """The constrained starting value handed to :func:`albireo.run_map`."""
        raise NotImplementedError

    def upper(self) -> float:
        """The largest magnitude this parameter can plausibly take, used for sizing."""
        raise NotImplementedError


@dataclass(frozen=True)
class Fixed(Spec):
    """A value held constant: no sample site, no gradient, and no posterior width.

    Parameters
    ----------
    value
        The constant, in the units of the site it is declared for.
    """

    value: Any

    def distribution(self):
        return None

    def start(self):
        return _as_array(self.value)

    def upper(self) -> float:
        return float(np.max(np.abs(np.atleast_1d(np.asarray(self.value, dtype=float)))))


@dataclass(frozen=True)
class Known(Spec):
    """A measurement with an uncertainty: ``Normal(value, sigma)``, started at ``value``.

    Appropriate for a literature period or a semi-amplitude from a cross-correlation
    study. In :meth:`Disentangler.scan` a ``Known`` primary semi-amplitude is marginalized
    over rather than held fixed, which prevents a 10% error in K₁ from inflating the
    detection statistic.

    Parameters
    ----------
    value
        Central value, in the units of the site it is declared for.
    sigma
        Standard deviation of the Gaussian prior, in the same units.
    """

    value: Any
    sigma: Any

    def distribution(self):
        return dist.Normal(_as_array(self.value), _as_array(self.sigma))

    def start(self):
        return _as_array(self.value)

    def upper(self) -> float:
        value = np.atleast_1d(np.asarray(self.value, dtype=float))
        sigma = np.atleast_1d(np.asarray(self.sigma, dtype=float))
        return float(np.max(np.abs(value) + 5.0 * sigma))


@dataclass(frozen=True)
class Between(Spec):
    """A bounded uniform prior, ``Uniform(lo, hi)``.

    Parameters
    ----------
    lo, hi
        Bounds of the support, in the units of the site.
    start_at
        Starting value. Defaults to the midpoint of ``lo`` and ``hi``.
    """

    lo: Any
    hi: Any
    start_at: Any = None

    def distribution(self):
        return dist.Uniform(_as_array(self.lo), _as_array(self.hi))

    def start(self):
        if self.start_at is not None:
            start = _as_array(self.start_at)
            lo, hi = _as_array(self.lo), _as_array(self.hi)
            if not bool(np.all((start > lo) & (start < hi))):
                raise ValueError(
                    f"start_at must lie strictly inside (lo, hi); got {self.start_at} for "
                    f"bounds ({self.lo}, {self.hi}). A start on a bound has no valid initial "
                    "parameters, and numpyro says only that."
                )
            return start
        return 0.5 * (_as_array(self.lo) + _as_array(self.hi))

    def upper(self) -> float:
        bounds = np.abs(np.atleast_1d(np.asarray(self.hi, dtype=float)))
        return float(np.max(bounds))


@dataclass(frozen=True)
class Scanned(Spec):
    """A grid of trial values, not a prior. Meaningful only for the companion's ``k``.

    The grid is the axis of the faint-companion search run by
    :meth:`Disentangler.scan` and :meth:`Disentangler.detection_limit`; there is no
    posterior over it.

    Parameters
    ----------
    values
        Trial semi-amplitudes in km/s.
    """

    values: Any

    def distribution(self):
        return None

    def start(self):
        values = np.asarray(self.values, dtype=float)
        return float(values[len(values) // 2])

    def upper(self) -> float:
        return float(np.max(np.abs(np.asarray(self.values, dtype=float))))


@dataclass(frozen=True)
class Sampled(Spec):
    """Any numpyro distribution, with a starting value and a declared upper bound.

    The one specification that cannot derive its own bound: a distribution object need not
    have finite support, and the starting value is not a bound. The velocity budget is
    derived from the reach of the semi-amplitude priors, so ``upper_bound`` is required
    wherever the specification is used for a quantity that must be bounded. Without it, a
    ``Sampled`` ``k`` would size the solver from the optimizer's starting point.

    Parameters
    ----------
    distribution_
        The numpyro distribution used as the prior.
    start_at
        Starting value, in the units of the site.
    upper_bound
        Largest magnitude the prior can realistically reach, in the units of the site.
        Required for a semi-amplitude; :meth:`upper` raises without it.
    """

    distribution_: dist.Distribution
    start_at: Any
    upper_bound: float | None = None

    def distribution(self):
        return self.distribution_

    def start(self):
        return _as_array(self.start_at)

    def upper(self) -> float:
        if self.upper_bound is None:
            raise ValueError(
                "Sampled(...) needs upper_bound= when it is used for a quantity the "
                "velocity budget is derived from (a semi-amplitude). Give the largest value "
                "the prior can realistically reach; the starting value is not that, and "
                "sizing the solver from it silently truncates the prior against a guard."
            )
        return float(np.max(np.abs(np.atleast_1d(np.asarray(self.upper_bound, dtype=float)))))


def _as_array(value):
    """A float or a jnp array, preserving scalars as scalars."""
    array = jnp.asarray(value, dtype=jnp.float64)
    return array


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
        Centre of the ridge hyperprior, which pulls the component toward its continuum.
    sigma
        Width of both hyperpriors, in natural logarithms.
    """

    tau0: float = 300.0
    eta0: float = 5.0
    sigma: float = 3.0


@dataclass(frozen=True)
class LSF:
    """An instrument's line-spread function, declared rather than inferred.

    Parameters
    ----------
    sigma_kms
        Gaussian sigma in km/s. A sequence gives one width per entry of
        ``anchors_angstrom``, which is how a wavelength-dependent LSF is declared.
        :data:`albireo.PER_EPOCH` (see :meth:`per_epoch`) reads the width from each
        epoch's own file instead.
    anchors_angstrom
        Wavelengths in angstrom at which ``sigma_kms`` is specified. The width is
        interpolated between them.

    Notes
    -----
    This width also fixes the radius of the convolution kernel, so it is an upper bound as
    well as a value: a fit that later infers a wider LSF is rejected rather than silently
    truncated. A width to be inferred through the low-level API needs some margin.

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
        2.653
        """
        if not resolving_power > 0:
            raise ValueError(f"resolving power must be positive; got {resolving_power}")
        sigma = C_KMS / (float(resolving_power) * 2.0 * math.sqrt(2.0 * math.log(2.0)))
        return cls(sigma_kms=sigma, **kwargs)

    @classmethod
    def per_epoch(cls) -> LSF:
        """Take the width from each epoch's own file rather than from one number.

        The reader records a file's resolving power on the epoch
        (:attr:`albireo.EpochData.lsf_sigma_kms`), and this declaration models every
        epoch at that width, so exposures at two resolving powers filed under one
        instrument name (HARPS's high-accuracy and high-efficiency modes, for instance)
        each get their own kernel. Every epoch of the instrument must carry a width;
        otherwise the declaration is refused, naming the epochs. The string
        ``"per-epoch"`` (the TOML form) is accepted in its place.

        Returns
        -------
        LSF
        """
        return cls(sigma_kms=PER_EPOCH)

    @property
    def is_per_epoch(self) -> bool:
        """Whether this declaration defers to the epochs' own widths."""
        return isinstance(self.sigma_kms, str) and self.sigma_kms == PER_EPOCH

    @property
    def max_sigma_kms(self) -> float:
        """Widest declared sigma in km/s, which sets the model-grid margin.

        Raises
        ------
        ValueError
            For a per-epoch declaration, whose widths live on the dataset; use
            :meth:`Disentangler._widest_lsf`, which resolves them.
        """
        if self.is_per_epoch:
            raise ValueError(
                "a per-epoch LSF has no width of its own; its widths are the epochs' "
                "(albireo.forward.declared_lsf_widths)"
            )
        return float(np.max(np.atleast_1d(np.asarray(self.sigma_kms, dtype=float))))

    def widths(self, dataset, key: str) -> np.ndarray:
        """Every width this declaration puts on ``dataset``'s epochs of instrument ``key``.

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
        order is a convention that the data cannot check, so the name is the only guard
        against reading row 0 as the wrong star.
    light
        Fraction of the total light contributed by this star. Required, and an
        assumption: with constant light fractions only the products ``l_i * d_i`` are
        observable (``docs/math.md`` §5.2), so every recovered depth scales as ``1 / l_i``
        and no part of the fit constrains this value. It should be quoted beside any
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

    The counterpart of :class:`Telluric`. Nebular flux is added on top of the total
    continuum and takes no light from the stars, so its per-epoch amplitude is a free
    parameter rather than a light fraction.

    Declaring one assembles several coupled pieces: the component column, the
    ``log_nebular_amp`` site with its prior and starting value, the per-pixel
    ``eta_profile`` that confines the component to its lines, the agreement between that
    profile's velocity and ``nebular_v_kms``, the smoothness prior object passed at
    construction so that the profile survives ML-II, and the additional velocity budget.
    Omitting the profile costs 250 nats and 2.6% in K₂; omitting the component entirely
    costs 11.5% of the Hβ equivalent width and 59% of K₂.

    Parameters
    ----------
    v_kms
        Velocity of the nebula in km/s, in the model grid's frame. Not identified: it sets
        only where the component's lines fall on the grid, which the window profile must
        match. It is a placement convention and is reported as one.
    lines
        Rest wavelengths in air angstrom. Default :data:`albireo.NEBULAR_LINES`.
    halfwidth_kms
        Half-width of each window in km/s. A generous value is preferable: too narrow a
        window pushes real emission back into the stellar components, while too wide a
        window only restores some of the freedom the profile removes.
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
        Orbital period in days. Required. This declaration does not search in period: it
        scans conjunction phase at a single period, so the prior must be narrow enough for
        a phase scan to be meaningful (or a periodogram run first).
    k
        Velocity semi-amplitude in km/s per :class:`Star`, in the order the stars are
        declared. Either one specification covering all of them, or one per star.
    t_conj
        Time of conjunction in the dataset's time system. The default ``"scan"`` locates
        it by scanning one period before optimization. The marginal likelihood is sharply
        multimodal in phase, and L-BFGS started in the wrong trough converges to the
        wrong solution without any indication of failure.
    ecc
        Eccentricity. ``Between(lo, hi)`` samples it through the
        ``(sqrt(e) cos w, sqrt(e) sin w)`` parameterization; ``Fixed(0.0)`` declares a
        circular orbit and is handled exactly, by not sampling those sites at all. The
        parameterization is singular at ``e = 0``, where the gradient is NaN and numpyro
        reports only "Cannot find valid initial parameters".
    omega
        Argument of periastron in radians. Required when ``ecc`` is ``Fixed`` and
        nonzero. With a free eccentricity, a value given here (and a ``start_at`` on the
        eccentricity's range) is where the fit starts, as a velocity table's orbit
        supplies it; the default start is ``e = 0.05`` at 0.5 rad.
    outer
        Outer orbit of a hierarchical triple. Its ``k`` must have two entries, for the
        inner pair and the tertiary.
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

    The solver bandwidth is built from this total. A bound that is too small makes the
    sampler stall against a guard, and, when the model is reached through
    :meth:`albireo.inference.MarginalOrbitModel.log_likelihood` directly, returns a wrong
    answer without raising. It is derived from the support of the semi-amplitude priors.

    Attributes
    ----------
    terms
        ``(name, value)`` pairs, one per contribution, in the order they are applied.
        Values are in km/s and the multiplicative terms report the running total.
    total
        The bound handed to the model as ``v_rel_max_kms``, in km/s.
    """

    terms: tuple[tuple[str, float], ...]
    total: float

    def __str__(self) -> str:
        rows = "\n".join(f"    {name:<34s} {value:8.1f}" for name, value in self.terms)
        return f"velocity budget (km/s)\n{rows}\n    {'total':<34s} {self.total:8.1f}"


def _velocity_budget(orbit: Orbit | None, velocities, components, frame: str, explicit):
    """Bound the relative velocity from the declaration, itemizing every term.

    The total must bound the largest relative velocity between any two model components
    at any epoch, over everything the prior allows, not over the fitted solution.

    From an :class:`Orbit` the bound is the support of the semi-amplitude priors. From a
    ``velocities=`` declaration it is the measured table centred per component (see
    :func:`_centred_velocities`: the absolute level is unidentified and must not consume
    bandwidth), with a factor of two of headroom because the fit is free to move the
    table.
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

    if orbit is not None and orbit.outer is not None:
        outer = sum(spec.upper() for spec in _k_specs(orbit.outer, 2)) * (1.0 + ecc_max)
        total += outer
        terms.append(("outer orbit", outer))

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

    Only the centred table is identified, and only the centred table reaches the model:
    :func:`albireo.inference.relative_velocities` removes one zero point per component,
    because with no orbit tying the stars together each free spectrum absorbs a constant
    added to its own shifts. The declared absolute level, such as a systemic velocity of
    +150 km/s for the SMC, costs the solver nothing and is not charged to the bandwidth.

    The centring is done here in velocity space. The model does it exactly, in pixels,
    where ``xi = artanh(v/c)`` makes the offset a translation. This function computes only
    a bound, and the two differ by ``O(v^2/c^2)``, of order 1e-8 at stellar velocities,
    against the factor-of-two headroom the budget adds on top.
    """
    v = np.asarray(velocities, dtype=float)
    return v - v.mean(axis=1, keepdims=True)


def _place_hyperparameters(dis, priors: dict, init: dict) -> None:
    """Place the sites every declaration carries, whichever velocity model it uses.

    Smoothness is always fitted by ML-II. A fixed ``(tau, eta)`` pair is not used
    anywhere in this package on real data, and defensible centres span five orders of
    magnitude between a sharp-lined and a rotationally broadened star. The nebular
    amplitudes are placed here for the same reason as in :meth:`Fit._velocity_priors`:
    omitting them would hold the component static at amplitude 1 without reporting it.
    """
    ordered = dis.ordered_components
    centres = np.array([_smoothness_of(c).tau0 for c in ordered], dtype=float)
    etas = np.array([_smoothness_of(c).eta0 for c in ordered], dtype=float)
    widths = np.array([_smoothness_of(c).sigma for c in ordered], dtype=float)
    priors["log_tau"] = dist.Normal(jnp.log(jnp.asarray(centres)), jnp.asarray(widths))
    priors["log_eta"] = dist.Normal(jnp.log(jnp.asarray(etas)), jnp.asarray(widths))
    init["log_tau"] = jnp.log(jnp.asarray(centres))
    init["log_eta"] = jnp.log(jnp.asarray(etas))

    if any(isinstance(c, Nebular) for c in dis.components):
        n_epochs = dis.dataset.n_epochs
        priors["log_nebular_amp"] = dist.Normal(jnp.zeros(n_epochs), 0.3).to_event(1)
        init["log_nebular_amp"] = jnp.zeros(n_epochs)


def _range_bounds(spec) -> tuple[float, float] | None:
    """The ``(lo, hi)`` of a semi-amplitude declared as a range, or ``None``.

    A range is a :class:`Between`, either one per star or one component of a vector
    :class:`Between` (``k=Between([10, 10], [90, 90])``), which :func:`_k_specs` hands out
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
    """Refuse a noise correlation that misses an instrument or is not a correlation."""
    declared = dis.noise_correlation
    if declared is None:
        return
    if isinstance(declared, Spec):
        if isinstance(declared, Scanned | Sampled):
            raise TypeError("noise_correlation takes Fixed, Known or Between, not a scan")
        values = [float(v) for v in np.atleast_1d(np.asarray(declared.start(), dtype=float))]
        if len(values) != 1:
            raise ValueError("a fitted noise_correlation is one value shared by every epoch")
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
    """Validate a ``velocities=`` declaration, refusing a table that cannot warm-start."""
    v = np.atleast_2d(np.asarray(dis.velocities, dtype=float))
    want = (len(stars), dis.dataset.n_epochs)
    if v.shape != want:
        raise ValueError(
            f"velocities must have shape {want}: one row per star, one column per epoch, "
            f"in the dataset's own epoch order; got {v.shape}"
        )
    if not np.all(np.isfinite(v)):
        raise ValueError("velocities must all be finite")

    # A cold start is refused here rather than left to be discovered. With every component
    # at the same velocity at every epoch the stars are indistinguishable, and the fit
    # lands 122,000 nats worse rather than merely converging slowly (D42).
    separation = float(np.max(np.ptp(v, axis=0))) if v.shape[0] > 1 else float(np.ptp(v))
    if separation <= 0.0:
        raise ValueError(
            "these velocities never separate the components: every star has the same "
            "velocity at every epoch, which is exactly the cold start the free-velocity "
            "mode is measured to fail from (122,000 nats worse than a warm one). "
            "Supply the measured per-epoch velocities "
            "(cross-correlation lags, or line splitting read off the two most separated "
            "epochs) rather than a placeholder."
        )
    widest = dis._widest_lsf()
    if separation < widest:
        warnings.warn(
            f"the declared velocities separate the components by at most "
            f"{separation:.2f} km/s, which is below the widest LSF sigma "
            f"({widest:.2f} km/s); at no epoch are the two resolved. The free-velocity "
            "fit is warm-started from these, and a warm start inside the unresolved "
            "regime is close to the cold one measured failing. Check the sign "
            "convention and the epoch ordering before trusting the result.",
            RuntimeWarning,
            stacklevel=4,
        )
    return v


def _k_specs(orbit: Orbit, n: int) -> list[Spec]:
    """The per-component ``k`` specifications, given either as one spec or a sequence."""
    if isinstance(orbit.k, Spec):
        spec = orbit.k
        # A single spec may still be vector-valued, e.g. Between([10, 5], [90, 70]). Detect
        # that from whichever field holds the numbers rather than from `hi` alone: a vector
        # Known or Fixed has no `hi`, and would otherwise be counted at its largest entry n
        # times, inflating the budget.
        for attr in ("hi", "value", "values", "start_at"):
            declared = getattr(spec, attr, None)
            if declared is None:
                continue
            if np.atleast_1d(np.asarray(declared, dtype=float)).size == n:
                return [_ScalarView(spec, i) for i in range(n)]
            break
        return [spec] * n
    specs = [_coerce_spec(item, "each entry of orbit.k") for item in orbit.k]
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

    Construction derives the quantities the low-level path otherwise requires, including
    the model grid and its margins, the solver's velocity budget, and the matched
    ``priors`` and ``init`` dictionaries. Quantities that are scientific claims are not
    derived. :meth:`explain` prints every derivation and :meth:`expert` returns the
    ``(model, priors, init)`` triple for direct use.

    Parameters
    ----------
    dataset
        From :func:`albireo.read_dataset`, :func:`albireo.load_example`, or built by hand.
    components
        One :class:`Star` per stellar component, optionally a :class:`Telluric` and a
        :class:`Nebular`. The declared order is the model's component order, and the
        names are how results are retrieved afterwards.
    orbit
        An :class:`Orbit`. Exactly one of ``orbit`` and ``velocities`` is required.
    velocities
        The alternative declaration, for a binary whose orbit is not known: measured
        per-epoch velocities, ``(n_stellar, n_epochs)`` in km/s, from cross-correlation, a
        shift-and-add pipeline, or line splitting measured by hand. The fit is then the
        free per-epoch RV table (``docs/math.md`` §7.6) rather than a Keplerian. No
        orbital elements are sampled, and :meth:`fit` returns a velocity-mode
        :class:`Fit` whose table can be searched for a period.

        The free table requires a warm start (a cold one was measured at 122,000 nats
        worse), and the other warm start, :meth:`Fit.free_velocities`, requires a
        Keplerian fit and hence a period. For a system with no published period the
        sequence is: declare the measured velocities, obtain the table, derive the period,
        then declare an :class:`Orbit`.

        The declared velocities are a starting point, not a constraint. The per-component
        zero point is unidentified in either mode, so they must be right about the
        epoch-to-epoch pattern, not the absolute scale.
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
        Lag-one correlation of each epoch's noise along its pixel index, the signature of
        a pipeline that resampled the spectra onto a common step (Gaia's RVS grids carry
        0.27 and 0.81; :mod:`albireo.gaia` measures it). ``None`` (default) is the
        diagonal noise model. A number, or ``{instrument: number}`` covering every
        instrument, declares the correlation and the noise model becomes AR(1) along the
        pixel index (:func:`albireo.forward.with_ar1`, ``docs/math.md`` §1.4a) with those
        values held; a :class:`Between` or :class:`Known` fits one shared value instead.
        The declared value is an assumption and is listed as one. It also reaches
        :meth:`Fit.measure_velocities`, whose errors carry it.

    Raises
    ------
    ValueError
        If no :class:`Star` is declared, if star names are not unique, if a light fraction
        is non-positive or the fractions do not sum to 1, if neither or both of ``orbit``
        and ``velocities`` are given, if more than one :class:`Telluric` or
        :class:`Nebular` component is declared, if an instrument in the dataset has no
        LSF, if a :class:`Telluric` or :class:`Nebular` component is declared for a
        dataset whose wavelength medium is undeclared, or if a noise correlation misses
        an instrument or lies outside ``(-1, 1)``.
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

    # init=False so that dataclasses.replace() cannot carry a stale grid, budget or model
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
        if self.velocities is not None:
            object.__setattr__(self, "velocities", _check_velocities(self, stars))
        if self.orbit is not None and self.orbit.outer is not None:
            raise NotImplementedError(
                "hierarchical triples are not in the façade's v1 vocabulary. The model "
                "supports them (the period_out/t_conj_out/k_out sites), so build one "
                "through albireo.MarginalOrbitModel directly: Disentangler.expert() on a "
                "two-star declaration gives you the triple to start from."
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
                declared_lsf_widths(self.dataset, key)  # refuses, naming undeclared epochs
        _check_noise_correlation(self)
        # A telluric or nebular component is keyed to absolute line positions, so an
        # undeclared wavelength medium is worth a nearly constant 83 km/s.
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
        """How many stellar components the model carries."""
        return len(self.stars)

    @property
    def ordered_components(self) -> tuple[Component, ...]:
        """The components in model row order: stars, then telluric, then nebular.

        The model fixes this order whatever the order of the declaration, so every
        per-component array (the smoothness rows, the hyperprior centres, the assumptions
        report) is assembled through this property. Iterating the declaration instead
        misassigns rows without raising: the vectors still have the correct length, so the
        length check in :func:`albireo.marginal_loglikelihood` passes, and a rotationally
        broadened star is regularized with a sharp-lined star's curvature penalty.
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
        if self.orbit is None:  # a free-velocity declaration has no eccentricity at all
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
        """The smoothness prior, carrying any per-pixel confinement profile."""
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
        """Every LSF width in play, per instrument of the dataset, resolved."""
        return {
            key: _coerce_lsf(self.lsf[key], key).widths(self.dataset, key)
            for key in self.dataset.instruments
        }

    def _widest_lsf(self) -> float:
        return max(float(np.max(w)) for w in self._lsf_widths().values())

    def _narrowest_lsf(self) -> float:
        return min(float(np.min(w)) for w in self._lsf_widths().values())

    def _lsf_lines(self) -> list[str]:
        """One line per instrument for :meth:`explain`, saying where its width came from."""
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
        """One line for :meth:`explain`: the noise model and where its correlation came from."""
        declared = self.noise_correlation
        if declared is None:
            return "  noise      diagonal: independent pixels"
        if isinstance(declared, Spec):
            return (
                "  noise      AR(1) along the pixel index, one lag-one correlation fitted "
                "(site ar1_phi)"
            )
        values = self.noise_correlation_per_epoch()
        by_instrument = {}
        for epoch, phi in zip(self.dataset, values, strict=True):
            by_instrument.setdefault(epoch.instrument, float(phi))
        listed = ", ".join(f"{k} {v:.3f}" for k, v in by_instrument.items())
        return f"  noise      AR(1) along the pixel index, lag-one correlation {listed} (declared)"

    def _native_dv_kms(self) -> float:
        """The finest native pixel size across the dataset, in km/s.

        The finest rather than the median: a model grid coarser than any contributing
        epoch discards that epoch's resolution, and on a mixed-instrument dataset the
        median follows whichever instrument contributed more epochs rather than whichever
        resolves more.
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

    def _make_prior(self) -> SmoothnessPrior:
        tau = np.array([_smoothness_of(c).tau0 for c in self.ordered_components], dtype=float)
        eta = np.array([_smoothness_of(c).eta0 for c in self.ordered_components], dtype=float)
        eta_profile = None
        nebular = next((c for c in self.components if isinstance(c, Nebular)), None)
        if nebular is not None:
            wave = np.asarray(self.grid.wave)
            # nebular_windows takes rest wavelengths in air. On a vacuum grid the lines lie
            # 0.87-2.74 A redward of those values, so an unconverted window is offset by a
            # nearly constant 83 km/s, enough to clip the line it is meant to contain and
            # push the emission back into the stellar components. This is why the medium is
            # required before a Nebular component is accepted.
            lines = nebular.lines
            if self.dataset.epochs[0].medium == "vacuum":
                from albireo.grids import air_to_vacuum
                from albireo.priors import NEBULAR_LINES

                source = NEBULAR_LINES if lines is None else lines
                values = list(source.values()) if isinstance(source, Mapping) else list(source)
                lines = [float(v) for v in np.asarray(air_to_vacuum(np.asarray(values)))]
            windows = nebular_windows(
                lines=lines,
                halfwidth_kms=nebular.halfwidth_kms,
                v_kms=nebular.v_kms,
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
        nebular = next((c for c in self.components if isinstance(c, Nebular)), None)
        return MarginalOrbitModel(
            self.grid,
            self.dataset,
            light_fractions=[s.light for s in self.stars],
            lsf_sigma_v=lsf_sigma,
            lsf_anchors_angstrom=anchors or None,
            v_rel_max_kms=self.velocity_budget.total,
            telluric=any(isinstance(c, Telluric) for c in self.components),
            nebular=nebular is not None,
            nebular_v_kms=0.0 if nebular is None else float(nebular.v_kms),
            prior=self.smoothness_prior,
            ecc_max=self.effective_ecc_max,
            block_size=self.block_size,
            ar1=self.noise_correlation is not None,
        )

    def noise_correlation_per_epoch(self) -> np.ndarray | None:
        """The declared lag-one noise correlation of every epoch, ``(n_epochs,)``.

        ``None`` when no correlation is declared, or when it is a fitted site rather than
        a declared value.
        """
        declared = self.noise_correlation
        if declared is None or isinstance(declared, Spec):
            return None
        if isinstance(declared, Mapping):
            return np.array([float(declared[epoch.instrument]) for epoch in self.dataset])
        return np.full(self.dataset.n_epochs, float(declared))

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
            # No orbital sites: `velocity` replaces them, and the model refuses the two
            # together. The prior is centred on zero rather than on the declared table
            # because the absolute level is unidentified either way; the declaration
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
        if self.orbit.t_conj == "scan":
            # Replaced by the scan result in fit(); the site is always sampled, so the
            # narrow prior around the located phase is set there rather than here. The
            # prior is anchored on the data rather than on zero: real epochs lie near BJD
            # 2.46e6, and a prior centred on the origin excludes every conjunction the data
            # can have. fit() narrows this around the scan result; expert() returns it
            # unchanged.
            period = float(np.max(np.atleast_1d(np.asarray(_start_of(self.orbit.period)))))
            first = float(np.min(np.asarray(self.dataset.bjd)))
            priors["t_conj"] = dist.Uniform(first, first + period)
            init["t_conj"] = first + 0.5 * period
        else:
            spec = _coerce_spec(self.orbit.t_conj, "orbit.t_conj")
            if isinstance(spec, Between):
                period = float(np.max(np.atleast_1d(np.asarray(_start_of(self.orbit.period)))))
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
        for name, spec in _ecc_sites(self.orbit, self.ecc_max):
            place(name, spec)
        if self.orbit.outer is not None:
            place("period_out", _coerce_spec(self.orbit.outer.period, "outer period"))
            place("t_conj_out", _coerce_spec(self.orbit.outer.t_conj, "outer t_conj"))
            place("k_out", _vector_spec(self.orbit.outer.k, 2, "outer k"))
            for name, spec in _ecc_sites(self.orbit.outer, self.ecc_max, suffix="_out"):
                place(name, spec)

        _place_hyperparameters(self, priors, init)
        _place_noise_correlation(self, priors, init, fixed)
        return priors, init, fixed

    # -- inspection -----------------------------------------------------------

    def expert(self):
        """The low-level ``(model, priors, init)`` triple this declaration compiles to.

        Features the declarative interface does not expose (jitter, AR(1) noise, inferred
        light fractions, inferred LSF widths) are reached by modifying the returned
        dictionaries and calling :class:`~albireo.inference.MarginalOrbitModel` directly.

        Returns
        -------
        tuple
            ``(model, priors, init)``: the
            :class:`~albireo.inference.MarginalOrbitModel`, a copy of :attr:`priors`, and
            a copy of :attr:`init`.
        """
        return self.model, dict(self.priors), dict(self.init)

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
        return lines

    def _refuse_anchored_lsf(self, what: str) -> None:
        """Refuse a wavelength-dependent LSF: ``k2_scan`` takes one width per instrument."""
        anchored = [
            key for key, value in self.lsf.items() if _coerce_lsf(value, key).anchors_angstrom
        ]
        if anchored:
            raise ValueError(
                f"{what}() takes one line-spread width per instrument, but {anchored} "
                "declared wavelength-dependent widths. Passing them would silently use only "
                "the first anchor. Declare a single representative sigma for the scan, or "
                "use albireo.k2_scan directly."
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
        nebular = next((c for c in self.components if isinstance(c, Nebular)), None)
        if nebular is not None:
            rows.append(
                f"  nebular velocity   {nebular.v_kms:g} km/s\n"
                "      unidentified: a placement convention for the window profile, not a "
                "measurement."
            )
        if self.noise_correlation is not None and not isinstance(self.noise_correlation, Spec):
            values = self.noise_correlation_per_epoch()
            by_instrument: dict[str, float] = {}
            for epoch, phi in zip(self.dataset, values, strict=True):
                by_instrument.setdefault(epoch.instrument, float(phi))
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
            "      ML-II fits these, but which basin it finds depends on where tau starts."
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

        Runs, in order, a conjunction-phase scan over one period (unless ``t_conj`` was
        declared), a coarse scan over every semi-amplitude declared as a range (see
        ``k_scan``), a profile of each star's prior amplitude at the orbit located with the
        scan repeated where that moved a start by a factor of two or more, the phase scan
        again at the chosen semi-amplitudes, and then :func:`albireo.run_map` over the
        orbital sites and the smoothness hyperparameters. With the spectra already
        marginalized out, optimizing the hyperparameters is the ML-II step. The fitted
        hyperparameters are returned on :attr:`Fit.hyper`, keyed by component name, and
        :meth:`Fit.sample` holds them fixed.

        A ``velocities=`` declaration has no orbit and no phase to locate: the scan is
        skipped and the free per-epoch table is fitted directly, warm-started from the
        declared velocities, returning a :class:`Fit` in ``"velocity"`` mode, read with
        :meth:`Fit.velocities` and :meth:`Fit.velocity_errors`. The implied period can
        then be used in a second declaration carrying an :class:`Orbit`.

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
            holding the others at their starting values, and starts L-BFGS from the best
            trial when it beats the declared start; ``False`` skips the scan. The
            marginal likelihood is multimodal in the semi-amplitudes as it is in phase
            (a start at half the true value settles at half, and a start far above it in
            the static-component minimum), and L-BFGS does not cross between the basins.
            The grid is crossed with a grid of conjunction phases when the conjunction
            is being scanned, since the two are coupled, and refined around the best trial
            twice at half the spacing; among trials within a few nats of the best the one
            honouring the declared order (the first star moving least) is taken. The
            scans run on a copy of the declaration with the model grid at twice the pixel
            (:meth:`_scan_declaration`), which makes the joint search affordable; the fit
            itself runs on the full grid. The scan is retained on :attr:`Fit.k_scan`.

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
            # declaration: the basin is the question, and the coarse grid answers it at a
            # fraction of the cost. L-BFGS then runs on the full model.
            ranged = bool(k_scan) and self._has_ranged_k()
            scanner = self._scan_declaration() if ranged else self
            if self.orbit.t_conj == "scan":
                scan = scanner._scan_phase(init)
                init["t_conj"] = scan.best
            if ranged:
                amplitude_scan = scanner._scan_semi_amplitudes(init)
                if amplitude_scan is not None and amplitude_scan.gain > 0.0:
                    init["k"] = jnp.asarray(amplitude_scan.best)
                    if self.orbit.t_conj == "scan":
                        init["t_conj"] = amplitude_scan.best_t_conj
                # The prior amplitude each component's spectrum is allowed, profiled at
                # the orbit located so far: where the data want a factor of two or more
                # from the declared start, the hyperparameter starts are moved and the
                # scan is repeated, because a scan at the wrong amplitude prefers a
                # static companion (see _profile_prior_amplitudes).
                if amplitude_scan is not None:
                    profile = scanner._profile_prior_amplitudes(init)
                    adapted = {
                        j: c for j, c in profile.items() if abs(math.log(c)) >= math.log(2.0)
                    }
                    if adapted:
                        names = [c.name for c in self.ordered_components]
                        log_tau = np.array(init["log_tau"], dtype=float)
                        log_eta = np.array(init["log_eta"], dtype=float)
                        for j, c in adapted.items():
                            row = names.index(self.stars[j].name)
                            log_tau[row] -= 2.0 * math.log(c)
                            log_eta[row] -= 2.0 * math.log(c)
                        init["log_tau"] = jnp.asarray(log_tau)
                        init["log_eta"] = jnp.asarray(log_eta)
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
                                init["k"] = jnp.asarray(second.best)
                                if self.orbit.t_conj == "scan":
                                    init["t_conj"] = second.best_t_conj
                if amplitude_scan is not None and amplitude_scan.gain > 0.0:
                    if self.orbit.t_conj == "scan":
                        # The joint scan located the phase coarsely; the fine phase scan
                        # at the chosen semi-amplitudes settles it.
                        scan = scanner._scan_phase(init)
                        init["t_conj"] = scan.best
            if scan is not None:
                # One period wide, centred on the conjunction the fit starts from rather than
                # on the phase scan's best: the semi-amplitude scan may have moved the start
                # half a period on, at semi-amplitudes the phase scan did not have, without
                # the fine phase scan running after it, and a start on the window's edge has
                # an infinite unconstrained coordinate (D63).
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
        )

    def _warn_if_the_period_prior_is_a_search(self) -> None:
        """Warn when the period prior is wide enough to constitute a period search.

        The conjunction scan resolves phase at one period, the period prior's midpoint. A
        prior wide enough to be a search locates a phase for a period that may be badly
        wrong, leaving L-BFGS to cross the multimodal structure the scan exists to avoid.
        The degradation is gradual, so this warns rather than raises.
        """
        if self.orbit is None:  # no period prior at all, so nothing here can be a search
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
        light fraction and its deviation spectrum, and the deviation carries a Gaussian
        prior of precision ``tau C + eta I``, so the marginal likelihood is exactly
        invariant under scaling the light by ``c`` and both hyperparameters by ``c``
        squared (D63: to 5e-10 nats). A profile over the light at fixed hyperparameters
        is therefore a profile over the prior amplitude the data want for that component,
        not a measurement of its flux fraction (on four benchmark systems it sat at 0.3 to
        0.7 of the injected fraction). It is what a scan needs: a companion
        declared at six times its light, with the hyperparameters at their starts, made
        the coarse scan prefer a static secondary by 43 nats, and the profile rejected the
        declared amplitude by 150 nats at that very orbit.

        For each star in turn the light in ``theta`` is scaled by ``c`` over a geometric
        grid from ``1/span`` to ``span`` at ``ratio`` between neighbours (the scaled
        fraction kept below 0.98), the other stars and every other site where ``init``
        puts them, and the ``c`` with the highest marginal log-likelihood is returned.
        Nine evaluations per star through the compiled marginal.
        """
        theta = {k: jnp.asarray(v) for k, v in init.items()}
        theta.update({k: jnp.asarray(v) for k, v in self.fixed.items()})
        lights = np.array([s.light for s in self.stars], dtype=float)
        n = max(3, math.ceil(2.0 * math.log(span) / math.log(ratio)) + 1)
        grid = np.geomspace(1.0 / span, span, n)
        grid = np.unique(np.concatenate([grid, [1.0]]))
        out: dict[int, float] = {}
        for j in range(self.n_stellar):
            trials = grid[grid * lights[j] <= 0.98]
            values = []
            for c in trials:
                scaled = lights.copy()
                scaled[j] = c * lights[j]
                values.append(
                    float(self.model.log_likelihood({**theta, "light": jnp.asarray(scaled)}))
                )
            values = np.asarray(values)
            if not np.isfinite(values).any():
                out[j] = 1.0
                continue
            out[j] = float(trials[int(np.argmax(np.where(np.isfinite(values), values, -np.inf)))])
        return out

    def _scan_phase(self, init) -> PhaseScan:
        """Locate conjunction on a 42-point grid over one period, before optimizing.

        The trials are ``min(bjd) + linspace(0, P, 42, endpoint=False)``, one period long
        and equally spaced. The count is even so that the grid holds the antipode of every
        trial it samples: trial ``i`` and trial ``i + 21`` are half a period apart. For a
        near-equal pair the marginal likelihood in phase is near-mirror-symmetric under a
        shift of half a period, so the scan can only choose between the two mirrors if both
        are on the grid. An odd count leaves every antipode midway between two trials, and
        on one benchmark star the unsampled antipode was better by 672 nats (D63).

        The marginal likelihood is sharply multimodal in phase, and L-BFGS does not cross
        between the troughs.
        """
        declared = init.get("period", self.fixed.get("period"))
        period = float(np.max(np.atleast_1d(np.asarray(declared))))
        trials = float(np.min(self.dataset.bjd)) + np.linspace(0.0, period, 42, endpoint=False)
        theta = {k: jnp.asarray(v) for k, v in init.items()}
        theta.update({k: jnp.asarray(v) for k, v in self.fixed.items()})
        # One evaluation per trial through the already-compiled marginal rather than a
        # batched sweep: a 42-wide sweep is one more large XLA compile per model shape,
        # and on Windows a long session of compiles has ended in a heap corruption inside
        # the compiler (D62); the loop compiles nothing new.
        values = []
        for t in trials:
            theta["t_conj"] = jnp.asarray(float(t))
            values.append(float(self.model.log_likelihood(theta)))
        best = float(trials[int(np.argmax(values))])
        return PhaseScan(period=period, trials=trials, values=np.asarray(values), best=best)

    def _scan_declaration(self) -> Disentangler:
        """This declaration on a model grid twice as coarse, for the scans that precede a fit.

        A scan asks which basin holds the maximum, not where within it. The grid at twice
        the pixel halves the pixel count and the solver bandwidth, and a marginal solve
        costs a quarter to an eighth of the full model's, which makes a joint scan over the
        semi-amplitudes and the phase affordable.
        """
        return replace(self, dv_kms=2.0 * float(self.grid.dv_kms))

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
        component (declared first, so the brighter or heavier star) crosses a geometric
        grid over its range (2 percent inside either bound, ``ratio`` between neighbours,
        coarsened until the level fits ``max_trials``) with ``n_phases`` conjunctions over
        one period, or with the single conjunction ``init`` carries when it was declared,
        the other components held at their starting values; each further ranged
        component then crosses its own grid at the semi-amplitudes and conjunction located
        so far. Each of ``levels`` refinements halves the spacings around the best trial,
        three values per axis, jointly over the ranged components and the phase. A final
        pass takes each component once more over its whole grid at the refined best, and
        two more refinements follow if that moved anything. Among the trials within
        ``tie_nats`` of the best, the one honouring the declared order (semi-amplitudes
        non-decreasing) is taken, and with two ranged components the exchange-symmetric
        twin of the best (the semi-amplitudes swapped, the conjunction half a period on)
        is tried for the same reason. ``None`` when no semi-amplitude is a range.

        The sequence follows from a measurement (D63). A companion of a few percent of the
        light commands some 25 nats over its whole range and prefers its true
        semi-amplitude only while the primary's is within about 10 percent of the truth,
        whereas the primary's own peak falls by 50 nats within 20 percent: on a product
        grid at ratio 2 no trial holds the primary close enough for the companion's
        evidence to point the right way, and a local refinement cannot carry the companion
        out of the basin it was placed in. The phase is scanned with the first component
        because the two are coupled: a phase located at a semi-amplitude a factor of three
        off can sit a quarter of a period from the truth.

        Two guards keep a scan on a coarse model of the wrong shape (a free eccentricity
        starts near circular, the grid is twice the pixel) from doing harm. A best trial
        with every ranged semi-amplitude at the floor of its grid is the static-component
        minimum, which a wrong phase or eccentricity makes preferable to any moving pair;
        it is not a start, and the declared start is kept. And the start moves only when
        the best trial beats it by more than ``hold_nats`` jointly, because the moves are
        joint: the same companion's move, tested one component at a time, decomposed into
        two holds of 15 nats each and was refused. A component whose own move is worth
        less than ``indifferent_nats``, with the others at the scan's best, returns to its
        start, the data being indifferent and a template table the better guess. The hold
        losses are recorded either way.
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
        declared = init.get("period", self.fixed.get("period"))
        period = float(np.max(np.atleast_1d(np.asarray(declared))))
        t_conj = float(np.asarray(init.get("t_conj", self.fixed.get("t_conj")), dtype=float))
        scanning_phase = self.orbit.t_conj == "scan"
        theta = {k: jnp.asarray(v) for k, v in init.items()}
        theta.update({k: jnp.asarray(v) for k, v in self.fixed.items()})
        theta["t_conj"] = jnp.asarray(t_conj)
        history: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        def evaluate(k_trials, t_trials):
            # Batches of sixteen keep the compiled sweep small; the trials are hundreds.
            sweep = {"k": jnp.asarray(k_trials), "t_conj": jnp.asarray(t_trials)}
            values = self.model.log_likelihood_sweep(theta, sweep, batch_size=16)
            return np.asarray(values, dtype=float)

        def value_at(k, t):
            at = {**theta, "k": jnp.asarray(k), "t_conj": jnp.asarray(float(t))}
            return float(self.model.log_likelihood(at))

        def ordered(k_trials):
            # Components are declared in order of decreasing mass, so the first moves
            # least; a trial honours that order when its semi-amplitudes do not decrease.
            return np.all(np.diff(k_trials, axis=1) >= 0.0, axis=1)

        def choose(k_trials, values):
            # Among the trials the data cannot tell apart, the one honouring the order.
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
            # One component over its whole grid, the others where they stand.
            k_grid = np.repeat(at_k[None, :], len(axes[i]), axis=0)
            k_grid[:, i] = axes[i]
            k_all, t_all = trials(k_grid, phases)
            values = evaluate(k_all, t_all)
            history.append((k_all, t_all, values))
            return k_all, t_all, values

        def argmax(values):
            return int(np.argmax(np.where(np.isfinite(values), values, -np.inf)))

        # Level 0: the first ranged component with the phase, then each further one at
        # what has been located so far.
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
        # another was still far from its value, gets its whole grid once more at the
        # refined best; a move is followed by two refinements of its own.
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
            # The exchange-symmetric twin of the best: the two semi-amplitudes swapped and
            # the conjunction half a period on fit alike components equally well, and the
            # tie goes to the declared order.
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
            # The move is taken jointly; a component the data are indifferent to, with the
            # others at the scan's best, goes back to its start.
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
        """The declared :class:`Orbit`, or a refusal, for the paths that require one.

        Shared by the scan entry points so that the message is written once and the type
        narrowing is explicit rather than implied by call order.
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

        Shared by :meth:`scan` and :meth:`detection_limit`, so the no-orbit refusal is
        raised here rather than in each of them.
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

        The SB1 faint-companion search: the marginal likelihood is profiled over the
        companion's velocity semi-amplitude, which detects a companion that never appears
        as a second set of lines. Declaring the primary's ``k`` as :class:`Known` rather
        than :class:`Fixed` marginalizes over it. A K₁ 10% high reduces the correlation of
        the recovered companion with the truth from 0.96 to 0.49 while tripling the
        detection statistic, so the artifact presents as a stronger detection that no
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
            as :class:`Fixed` or :class:`Known`, or declares a wavelength-dependent LSF.
        """
        from albireo.scan import k2_scan

        k1, k2 = self._scan_k()
        self._refuse_anchored_lsf("scan")
        lsf_sigma = {k: _coerce_lsf(v, k).sigma_kms for k, v in self.lsf.items()}
        return k2_scan(
            self.grid,
            self.dataset,
            orbit=self._scan_orbit(),
            k1=float(np.asarray(k1.start())),
            k2_grid=np.asarray(k2.values, dtype=float),
            light_fractions=[s.light for s in self.stars],
            lsf_sigma_v=lsf_sigma,
            prior=self.smoothness_prior,
            v_rel_max_kms=self.velocity_budget.total,
            k1_sigma=float(np.asarray(k1.sigma)) if isinstance(k1, Known) else None,
            telluric=any(isinstance(c, Telluric) for c in self.components),
            nebular=any(isinstance(c, Nebular) for c in self.components),
            nebular_v_kms=next(
                (float(c.v_kms) for c in self.components if isinstance(c, Nebular)), 0.0
            ),
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
        """
        from albireo.calibrate import detection_limit

        k1, k2 = self._scan_k()
        self._refuse_anchored_lsf("detection_limit")
        scan_kwargs = {
            "orbit": self._scan_orbit(),
            "k1": float(np.asarray(k1.start())),
            "k2_grid": np.asarray(k2.values, dtype=float),
            "light_fractions": [s.light for s in self.stars],
            "lsf_sigma_v": {k: _coerce_lsf(v, k).sigma_kms for k, v in self.lsf.items()},
            "prior": self.smoothness_prior,
            "v_rel_max_kms": self.velocity_budget.total,
        }
        if isinstance(k1, Known):
            scan_kwargs["k1_sigma"] = float(np.asarray(k1.sigma))
        scan_kwargs["telluric"] = any(isinstance(c, Telluric) for c in self.components)
        scan_kwargs["nebular"] = any(isinstance(c, Nebular) for c in self.components)
        scan_kwargs["nebular_v_kms"] = next(
            (float(c.v_kms) for c in self.components if isinstance(c, Nebular)), 0.0
        )
        if self.block_size is not None:
            scan_kwargs.setdefault("block_size", self.block_size)
        return detection_limit(self.grid, self.dataset, **scan_kwargs, **kwargs)


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
        Trial time with the highest marginal likelihood.
    """

    period: float
    trials: np.ndarray
    values: np.ndarray
    best: float

    @property
    def contrast(self) -> float:
        """Marginal log-likelihood difference in nats between the best and worst phase."""
        return float(np.max(self.values) - np.min(self.values))


@dataclass(frozen=True)
class SemiAmplitudeScan:
    """The coarse semi-amplitude scan run before optimizing, retained for inspection.

    Attributes
    ----------
    k
        Trial semi-amplitudes, ``(n_trials, n_stellar)`` km/s: each component declared as
        a range over its geometric grid in turn, the others where they stood, then the
        joint refinements, the full-axis pass and the exchange check, in that order.
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
        The model-grid pixel of the declaration the scan ran on, which is coarser than
        the fit's.
    hold_losses
        Per ranged component (by index), the nats lost by holding it at its start with the
        others at the scan's best: the evidence each component's own move rests on, once
        the joint move has been taken.
    prior_scales
        Per star (by name), the factor applied to the starting precision of its smoothness
        prior (``tau`` and ``eta``) before the scan was repeated, when the profile over the
        light-equivalent amplitude asked for a factor of two or more; empty otherwise.
    notes
        What the guards did: a best trial at the static minimum refused, a joint gain too
        small to move the start, a component the data are indifferent to kept at its start;
        and, first, the prior amplitude adaptation when the scan was repeated.
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
        """How many trials were evaluated."""
        return int(self.values.size)

    @property
    def gain(self) -> float:
        """Nats gained over the declared start; the fit keeps the start unless this is positive."""
        return float(self.best_value - self.start_value)

    @property
    def contrast(self) -> float:
        """Marginal log-likelihood difference in nats between the best and worst trial."""
        finite = self.values[np.isfinite(self.values)]
        return float(np.max(finite) - np.min(finite))


def _default_v_range(fitted: np.ndarray, templates, margin: float = 40.0):
    """One correlation search window per template, all covering the same velocities.

    :func:`albireo.todcor` searches the shift of each template's own rest frame and
    reports it composed with that template's ``v_zero_kms``, so a single ``(lo, hi)``
    pair searches a different interval of *reported* velocity for every component whose
    zero point differs. The windows are offset by each template's zero point relative to
    their median, which leaves one common interval of reported velocity: the span of the
    fitted velocities widened by ``margin`` at each end, moved to the median zero point.
    Templates whose zero point is unknown are placed at zero, the frame the fitted
    velocities are already in.

    The zero points come from a label match, which measures each component's frame
    separately (``docs/math.md`` §9). When two disagree by more than the fitted velocities
    span, no interval of reported velocity holds every component's own velocities, and a
    search measures nothing. That is refused here rather than reported as a table of
    velocities pinned to the edge of the search.
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
            raise ValueError(
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
class Fit:
    """A MAP fit, its ML-II hyperparameters, and the quantities derived from them.

    The fitted smoothness hyperparameters must accompany the parameters into every
    downstream call, so :meth:`spectra`, :meth:`std`, :meth:`composite` and
    :meth:`sample` supply them automatically.

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
        Laplace covariance is meaningful only against the model it came from.
    k_scan
        The :class:`SemiAmplitudeScan` run before optimization, or ``None`` when no
        semi-amplitude was a range, the scan was switched off, or the fit is in
        ``"velocity"`` mode.
    """

    dis: Disentangler
    result: Any
    hyper: dict[str, dict[str, float]]
    phase_scan: PhaseScan | None = None
    mode: str = "keplerian"
    priors_used: dict = field(default_factory=dict, repr=False)
    k_scan: SemiAmplitudeScan | None = None

    @property
    def params(self) -> dict:
        """The MAP site values, as returned by :func:`albireo.run_map`."""
        return dict(self.result.params)

    @property
    def theta(self) -> dict:
        """Site values together with any fixed sites, as the forward model consumes them."""
        theta = {k: jnp.asarray(v) for k, v in self.result.params.items()}
        theta.update({k: jnp.asarray(v) for k, v in self.dis.fixed.items()})
        return theta

    def star(self, name: str) -> dict:
        """Everything fitted for one named star, so that no row index is read by hand.

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
        names = [s.name for s in self.dis.stars]
        if name not in names:
            raise KeyError(f"no star called {name!r}; this fit has {names}")
        index = names.index(name)
        out = {"name": name, "light": self.dis.stars[index].light, **self.hyper[name]}
        if self.mode == "keplerian":
            out["k"] = float(np.atleast_1d(np.asarray(self.params["k"], dtype=float))[index])
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

        A free table carries one arbitrary zero point per component, not one in total:
        with no orbit tying the stars together, each free spectrum absorbs a constant
        added to its own shifts. The removal is performed in pixel space, where a constant
        offset is exact rather than first order.
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

        Not the raw Laplace diagonal, which returns the prior width: 37.95 km/s on every
        entry against an actual 0.059 km/s in the reference case, and identical on a good
        dataset and an uninformative one.

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
        # sampled sites in the same order, or the projection reads the wrong block. This is
        # why the priors are carried on the Fit.
        keplerian = {"period", "t_conj", "secosw", "sesinw", "k"}
        fixed = {k: v for k, v in self.dis.fixed.items() if k not in keplerian}
        model = self.dis.model.model(self.priors_used, fixed=fixed or None)
        covariance = laplace_inverse_mass(
            model, self.result.params, model_args=(self.dis.model.problem,)
        )
        return np.asarray(relative_velocity_errors(covariance, self.result.unconstrained))

    def noise_correlation(self) -> dict[str, float] | None:
        """The lag-one noise correlation per instrument, as declared or as fitted.

        ``None`` under the diagonal noise model. A fitted correlation is the MAP value of
        the ``ar1_phi`` site, shared by every instrument.
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
        """The conditional solve at the MAP: spectra, precision, and log-likelihood."""
        return self.dis.model.marginal(self.theta)

    def spectra(self) -> np.ndarray:
        """Conditional-mean component spectra ``d``, shape ``(n_comp, n_pix)``.

        These are deviations from a unit continuum, so the component spectrum is ``1 + d``
        and the observable is ``l_i * d_i``; see :meth:`Disentangler.assumptions`.
        """
        return np.asarray(self.marginal().d_hat)

    def std(self) -> np.ndarray:
        """Pointwise standard deviation of :meth:`spectra`, of the same shape.

        It is large wherever the prior rather than the data sets the spectrum, which
        covers most of the continuum, and is therefore exported alongside the spectrum.
        """
        return np.asarray(spectra_std(self.marginal()))

    def composite(self) -> np.ndarray:
        """The light-weighted stellar sum ``sum_i l_i d_i``, the observed quantity.

        The individual components are a decomposition of this sum; only the weighted sum
        is constrained by the data at every pixel.
        """
        d = self.spectra()[: self.dis.n_stellar]
        lights = np.array([s.light for s in self.dis.stars], dtype=float)
        return np.tensordot(lights, d, axes=(0, 0))

    def residual_zscores(self) -> np.ndarray:
        """Whitened data residuals. Their RMS is close to 1 when the noise model holds."""
        return np.asarray(
            data_residual_zscores(self.dis.model.problem_at(self.theta), self.marginal().d_hat)
        )

    @property
    def z_rms(self) -> float:
        """RMS of :meth:`residual_zscores`, the check to make before adding jitter."""
        z = self.residual_zscores()
        return float(np.sqrt(np.mean(np.square(z[np.isfinite(z)]))))

    def summary(self) -> str:
        """Report the fit: convergence, orbit or velocities, ML-II, residuals, assumptions.

        Returns
        -------
        str
            A multi-line report ending with :meth:`Disentangler.assumptions`.
        """
        # The convergence flag is an absolute threshold on a potential whose scale grows
        # with the number of good pixels, so on real data it is routinely False at a good
        # optimum. Report the outcome rather than grading it.
        stopped = "converged" if self.result.converged else "stopped at the step cap"
        lines = [
            f"{'MAP' if self.mode == 'keplerian' else 'free-velocity'} fit: potential "
            f"{self.result.potential:.6g}, |grad| {self.result.grad_norm:.3g}, "
            f"{self.result.num_steps} steps ({stopped})"
        ]
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
        lines.append(
            f"  residual z-score RMS {self.z_rms:.3f}"
            + (
                ""
                if abs(self.z_rms - 1.0) < 0.2
                else "   <- the noise model is not describing these data; read "
                "docs/benchmarks.md before adding a jitter term"
            )
        )
        lines += ["", self.dis.assumptions()]
        return "\n".join(lines)

    def _hyper_report(self) -> str:
        """The ML-II table, flagging any hyperparameter the data did not move."""
        rows = ["ML-II smoothness (empirical Bayes: fitted here, then frozen for sampling)"]
        stale = []
        for name, component in zip(
            self.dis.component_names, self.dis.ordered_components, strict=True
        ):
            smooth = _smoothness_of(component)
            fitted = self.hyper[name]
            drift = (math.log(fitted["tau"]) - math.log(smooth.tau0)) / smooth.sigma
            rows.append(
                f"  {name:<14s} tau {smooth.tau0:9.3g} -> {fitted['tau']:9.3g} "
                f"({drift:+.2f} sigma)   eta {smooth.eta0:8.3g} -> {fitted['eta']:8.3g}"
            )
            if abs(drift) < 0.02:
                stale.append(name)
        rows += [
            f"  ! {name!r} tau did not move from its start: the hyperprior, not the data, "
            "is setting this component's smoothness."
            for name in stale
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

        Rebuilds the model with ``log_tau`` and ``log_eta`` injected as constants, takes
        the Laplace covariance at the MAP as the starting mass matrix, and samples.

        Fixing the hyperparameters is a plug-in approximation: the orbital credible
        intervals do not include smoothness uncertainty. Marginalizing over them instead
        means leaving them in the priors dictionary of :meth:`Disentangler.expert`, at a
        higher sampling cost.

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
        fixed = self._fixed_hyper()
        priors = {k: v for k, v in self.dis.priors.items() if k not in fixed}
        if self.phase_scan is not None:
            # One period wide, centred on the fitted conjunction (the MAP may sit half a
            # period from the phase scan's best when the semi-amplitude scan moved it).
            centre = float(np.asarray(self.result.params["t_conj"], dtype=float))
            period = self.phase_scan.period
            priors["t_conj"] = dist.Uniform(centre - 0.5 * period, centre + 0.5 * period)
        nuts_model = self.dis.model.model(priors, fixed=fixed)

        start = {k: v for k, v in self.result.params.items() if k in priors}
        mass = laplace_inverse_mass(nuts_model, start, model_args=(self.dis.model.problem,))
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
        return Posterior(dis=self.dis, mcmc=mcmc, map=self)

    def _velocity_priors(self, sigma_kms: float | None = None):
        """``(priors, init)`` for the free-velocity mode, warm-started from this fit."""
        start = np.asarray(
            orbit_velocities(self.theta, self.dis.dataset.bjd, ecc_max=self.dis.effective_ecc_max)
        )
        sigma = sigma_kms if sigma_kms is not None else self.dis.velocity_budget.total / 2.0
        priors = {"velocity": dist.Normal(0.0, float(sigma)).expand(list(start.shape)).to_event(2)}
        init = {"velocity": jnp.asarray(start)}
        # Everything that is not the orbit carries over unchanged: the smoothness
        # hyperparameters, and the per-epoch nebular amplitudes where the declaration has a
        # nebular component. Dropping the latter would hold the component static at
        # amplitude 1 without reporting it.
        for site in ("log_tau", "log_eta", "log_nebular_amp"):
            if site in self.dis.priors:
                priors[site] = self.dis.priors[site]
                init[site] = self.dis.init[site]
        return priors, init

    def free_velocities(
        self, *, sigma_kms: float | None = None, max_steps: int = 300, progress=None
    ) -> Fit:
        """Refit with one free velocity per component per epoch, warm-started from here.

        The per-epoch radial-velocity table is the customary product of a spectroscopic
        binary analysis, and the model check for the Keplerian mode: free velocities are
        fitted, then tested for whether a Keplerian threads them
        (:meth:`keplerian_residuals`).

        The method is defined on :class:`Fit`, not constructible on its own, because a
        cold start is measured at 122,000 nats worse than the warm-started solution;
        warm-starting is the only mode shown to succeed.

        This is the entry point when a Keplerian fit already exists and the table is
        wanted as a model check. For an unsolved system, where the table yields the
        period, ``Disentangler(velocities=...)`` and :meth:`Disentangler.fit` warm-start
        from measured velocities instead.

        Parameters
        ----------
        sigma_kms
            Width in km/s of the Gaussian prior on each free velocity. Defaults to half
            the velocity budget.
        max_steps
            L-BFGS iteration cap.
        progress
            ``callback(step, potential, grad_norm, params)``.

        Returns
        -------
        Fit
            A fit in ``"velocity"`` mode.
        """
        priors, init = self._velocity_priors(sigma_kms)
        # Whatever the declaration fixed about the orbit no longer applies, and the model
        # rejects Keplerian sites alongside a velocity table, so they are dropped.
        keplerian = {"period", "t_conj", "secosw", "sesinw", "k"}
        fixed = {k: v for k, v in self.dis.fixed.items() if k not in keplerian}
        result = run_map(
            self.dis.model.model(priors, fixed=fixed or None),
            init=init,
            max_steps=max_steps,
            callback=progress,
            model_args=(self.dis.model.problem,),
        )
        hyper = _hyper_of(result.params, self.dis.component_names)
        return Fit(dis=self.dis, result=result, hyper=hyper, mode="velocity", priors_used=priors)

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

        Evaluates the model problem at the fitted parameters and returns ``h = A^T W z``,
        ``z^T W z`` and the band of ``G = A^T W A`` for the stellar rows, so that the epoch
        chi-square of any template composite ``m`` costs one band product
        (:class:`albireo.EpochStatistics`, ``docs/math.md`` §9.2a). The result goes to
        :func:`albireo.match_labels` with ``compare="epochs"``; :meth:`match_labels` builds
        it itself.

        Three things differ from the disentangling's own problem. First, every
        instrument's LSF in the stellar operator is replaced by the width the library does
        not already carry, ``sigma_q = sqrt(sigma_inst^2 - sigma_lib^2)`` per declared width
        (per anchor, or per epoch for a per-epoch declaration), through
        :func:`albireo.forward.with_lsf`, because a template drawn from a library at its own
        resolving power would otherwise be broadened twice. Second, with
        ``grid_compensation`` (the default) that width is reduced once more by the
        smoothing the model grid itself adds (below). Third, a telluric or nebular
        component is held at its posterior mean: its predicted contribution, through the
        declared LSF under which that mean was solved, is subtracted from the data, so the
        chi-square is ``||z - A_e d_hat_e - A_s m||_W^2``. The noise model (weights,
        jitter, AR(1) correlation) and the orbit are this fit's, unchanged.

        **The grid compensation.** Between a library spectrum and the epoch pixels a
        template passes five discrete steps on the model grid of pixel ``dv``, and each adds
        a variance that a continuous spectrum does not have. A box average of width ``dv``
        has variance ``dv^2 / 12``; a linear-interpolation shift by a fraction ``f`` of a
        pixel is the two-tap kernel with weights ``1 - f`` at ``-f dv`` and ``f`` at
        ``(1 - f) dv``, of mean zero and variance ``f (1 - f) dv^2``, which averages to
        ``dv^2 / 6`` over ``f``. The steps are the library's box average onto the model grid
        (``dv^2 / 12``), the pixel integration of the rotation kernel (``dv^2 / 12``), the
        label rows' shift by the frame velocity (``dv^2 / 6``), the operator's shift of each
        epoch (``dv^2 / 6``), and the model pixel held constant across the rebin onto the
        native pixels (``dv^2 / 12``), so that

            sigma_grid^2 = (1/12 + 1/12 + 1/6 + 1/6 + 1/12) dv^2 = (7/12) dv^2,

        independent of the line width. The epoch data carry none of it, and ``v sin i``,
        whose limb-darkened profile has variance ``0.225 (v sin i)^2``, gives it back:
        uncompensated, ``v_fit^2 = v^2 - (7/12) dv^2 / 0.225``. The stellar operator
        therefore applies

            sigma_op^2 = sigma_inst^2 - sigma_lib^2 - (7/12) dv^2.

        On the closed-loop fixture of ``tests/test_pipeline.py`` (native pixel 4.63 km/s,
        LSF sigma 5.5 km/s, ``v sin i`` injected at 11 km/s) the uncompensated comparison
        returned 7.37 km/s with the orbit fixed at the truth and the compensated one 10.08;
        over five noise draws the mean error was -3.02 km/s uncompensated and -0.24
        compensated (``docs/math.md`` §9.2a). Temperatures, gravities, metallicities and
        light fractions did not move. The formula is a mean: the frame shift's phase is a
        fitted parameter, the rotation kernel's pixel term oscillates for ``v sin i`` near
        ``dv``, and computing the epoch shifts' own phases changed ``v sin i`` by at most
        0.12 km/s.

        Where ``sigma_op`` would fall below half a model pixel, which happens when
        ``dv > sqrt(6/5) sigma_q``, about ``1.10 sigma_q``, it is floored there (or at
        ``sigma_q``, when that is narrower), only part of the bias is removed, and a
        ``UserWarning`` names the grid spacing that avoids the floor; when
        ``sigma_q < dv`` without a floor the compensated kernel is below 0.65 pixel and a
        warning says so. Both are recorded in :attr:`albireo.EpochStatistics.notes`. A
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
            If the library is at or below an instrument's resolving power, which no
            convolution can undo; if the dataset does not declare its wavelength medium; or
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

        medium = self.dis.dataset[0].medium
        if medium is None:
            raise ValueError(
                "this dataset does not declare whether its wavelengths are air or vacuum, "
                "so it cannot be matched against a synthetic grid: the two differ by ~83 "
                "km/s. Set medium= on the epochs (albireo.air_to_vacuum and "
                "albireo.vacuum_to_air convert)."
            )
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
            d_hat = np.asarray(model.marginal(theta).d_hat)
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
            reduced = _quadrature_width(declared, resolving_power, f"instrument {instrument!r}")
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
        )

    def match_labels(self, stars, **kwargs):
        """Fit Teff, log g, [M/H] and v sin i to the stellar components of this fit.

        The declarative route to :func:`albireo.match_labels`. The grid, the recovered
        spectra, their uncertainty band, the assumed light fractions, the instrument width
        and the dataset's wavelength medium are taken from this fit, so they cannot
        disagree with what was solved.

        The comparison defaults to ``compare="epochs"``: the template composite is compared
        with the epoch spectra through :meth:`epoch_statistics`, built here at the
        resolving power the stars' libraries declare and with the model grid's own
        smoothing, ``(7/12) dv^2``, removed from the operator width;
        ``statistics=fit.epoch_statistics(..., grid_compensation=False)`` leaves it in. It
        is exact for the fit's noise model, does not depend on the declared light
        fractions, and on simulated Gaia RVS binaries returned temperatures twice as close
        to the truth as the diagonal comparison against ``d_hat`` converged the same way
        (``docs/math.md`` §9.2a). ``compare="native"`` and ``compare="matched"`` compare
        with the disentangled components instead.

        Only the stellar rows are passed on: a telluric or nebular component has no
        atmospheric parameters, and the epoch comparison holds it at its posterior mean.
        The light fractions travel with the stars because the label fit's dilution model is
        defined in terms of what was assumed.

        Parameters
        ----------
        stars
            Mapping of star name to :class:`albireo.StarLabels`, one per stellar component
            of this declaration. Names must match :attr:`Disentangler.stars`.
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
            epoch comparison, if the stars' libraries declare different resolving powers.

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
        medium = self.dis.dataset[0].medium
        if medium is None:
            raise ValueError(
                "this dataset does not declare whether its wavelengths are air or vacuum, "
                "so it cannot be matched against a synthetic grid: the two differ by ~83 "
                "km/s. Set medium= on the epochs (albireo.air_to_vacuum and "
                "albireo.vacuum_to_air convert)."
            )
        ordered = {name: stars[name] for name in names}
        compare = kwargs.setdefault("compare", "epochs")
        if compare == "epochs" and kwargs.get("statistics") is None:
            declared = {name: ordered[name].library.resolving_power for name in names}
            if len(set(declared.values())) > 1:
                raise ValueError(
                    f"the stars' libraries declare different resolving powers {declared}; the "
                    "epoch comparison applies one operator width per instrument, so every "
                    "star's template must start at the same resolution. Use libraries at one "
                    "resolving power, or pass compare='native'."
                )
            kwargs["statistics"] = self.epoch_statistics(
                resolving_power=next(iter(declared.values()))
            )
        spectra, std = self.spectra(), self.std()
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
        from albireo.io import write_spectra

        return write_spectra(
            path,
            self.dis.grid,
            self.spectra(),
            self.std(),
            light_fractions=[s.light for s in self.dis.stars],
            prior=self.dis.smoothness_prior,
            meta={"COMPNAME": ",".join(self.dis.component_names)},
            **kwargs,
        )

    def templates(self, *, pixels_per_sigma: float = 3.0) -> list:
        """The stellar components of this fit as :class:`albireo.todcor.Template` objects.

        A system's own disentangled components are the closest templates available for
        it: measured from these epochs, they carry the correct lines, depths and rotational
        broadening. They do not carry an absolute rest frame, since each
        component has its own unidentified zero point (``docs/math.md`` §5.3, §7.6), so
        every template returned here has ``v_zero_kms=None`` and the velocities measured
        against it are differential. A label match (:meth:`match_labels`, then
        :meth:`albireo.Template.from_labels`) fixes the zero point.

        Telluric and nebular components are not stars and are not returned. The templates
        are intrinsic (``sigma_kms=0``): the fit solved for the spectra under the declared
        LSF, which :meth:`measure_velocities` applies again per instrument.

        The fit's grid samples the native pixel at about two points, which suits the
        solver but is too coarse for a correlation template: the pixel-locking ripple of
        the linear shift operator is ``~0.1 / sigma_px^2`` pixels (``docs/math.md``
        §10.3). The components are linearly upsampled by an integer factor onto a grid
        carrying at least ``pixels_per_sigma`` pixels per narrowest declared LSF sigma.
        The smoothness prior has already band-limited them well above that scale, so the
        upsampling adds no information and removes the ripple.

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

        The disentangling yields the component spectra; correlating the epochs against
        them yields one velocity per star per epoch (:func:`albireo.todcor`), which the
        joint fit does not produce. The declared light fractions are held fixed by default
        because the components were solved against them (``docs/math.md`` §9.1: what is
        recovered is ``(w/l0) t``, so ``l0`` is the only amplitude consistent with those
        spectra), and the declared LSF is applied per instrument as the fit applied it.

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
            template by that template's zero point relative to their median, so that every
            component searches the same interval of *reported* velocity: the range is in
            each template's own frame, and a template carrying a zero point reports its
            velocities composed with it. Templates whose zero points disagree by more than
            the fitted velocities span admit no such interval, which raises rather than
            being searched.

        **kwargs
            Passed to :func:`albireo.todcor`.

        Returns
        -------
        albireo.todcor.VelocityTable

        Raises
        ------
        ValueError
            If ``v_range`` is left to the default and the templates' zero points put the
            implied window off one component's own fitted velocities.

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
    """A NUTS posterior over the orbit, carrying the fit it was started from.

    Attributes
    ----------
    dis
        The :class:`Disentangler` the posterior was declared from.
    mcmc
        The numpyro ``MCMC`` object returned by :func:`albireo.run_nuts`.
    map
        The :class:`Fit` whose MAP provided the starting point and the ML-II
        hyperparameters.
    """

    dis: Disentangler
    mcmc: Any
    map: Fit

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
            in km/s, and the 95% interval ``k_hdi``.

        Raises
        ------
        KeyError
            If ``name`` is not a star of this declaration.
        """
        names = [s.name for s in self.dis.stars]
        if name not in names:
            raise KeyError(f"no star called {name!r}; this fit has {names}")
        k = np.asarray(self.samples["k"])[:, names.index(name)]
        return {
            "name": name,
            "light": self.dis.stars[names.index(name)].light,
            "k": float(np.mean(k)),
            "k_std": float(np.std(k)),
            "k_hdi": (float(np.percentile(k, 2.5)), float(np.percentile(k, 97.5))),
        }

    def spectra(self, *, num_draws: int = 32, seed: int = 0) -> np.ndarray:
        """Spectra drawn from the joint posterior, ``(num_draws, n_comp, n_pix)``.

        The scatter across draws includes both the conditional spectral uncertainty and
        the orbital uncertainty. The ML-II hyperparameters are injected automatically;
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

        Returns
        -------
        str
            A multi-line report ending with :meth:`Disentangler.assumptions`.
        """
        samples = self.samples
        lines = [
            f"NUTS posterior: {len(next(iter(samples.values())))} draws over {len(samples)} sites"
        ]
        extra = getattr(self.mcmc, "get_extra_fields", lambda: {})()
        if "diverging" in extra:
            lines.append(f"  divergences: {int(np.sum(np.asarray(extra['diverging'])))}")
        lines.append("")
        for site in ("period", "t_conj"):
            if site in samples:
                values = np.asarray(samples[site]).ravel()
                lines.append(f"  {site:<9s} {np.mean(values):.6f} +/- {np.std(values):.6f}")
        for star in self.dis.stars:
            row = self.star(star.name)
            lines.append(
                f"  K({star.name})  {row['k']:8.3f} +/- {row['k_std']:.3f} km/s   "
                f"95% [{row['k_hdi'][0]:.3f}, {row['k_hdi'][1]:.3f}]"
            )
        lines += [
            "",
            "  Smoothness was fixed at its ML-II values, so these intervals do not include",
            "  smoothness uncertainty (a plug-in approximation, not a marginalization).",
            "",
            self.dis.assumptions(),
        ]
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
        from albireo.io import write_spectra

        draws = self.spectra(num_draws=num_draws, seed=seed)
        return write_spectra(
            path,
            self.dis.grid,
            draws.mean(axis=0),
            draws.std(axis=0),
            light_fractions=[s.light for s in self.dis.stars],
            prior=self.dis.smoothness_prior,
            meta={"COMPNAME": ",".join(self.dis.component_names)},
            **kwargs,
        )


# -- helpers ------------------------------------------------------------------


def _smoothness_of(component: Component) -> Smoothness:
    return component.smoothness


def _start_of(spec) -> Any:
    return _coerce_spec(spec, "spec").start()


def _hyper_of(params: Mapping, names: Sequence[str]) -> dict[str, dict[str, float]]:
    """The fitted hyperparameters, keyed by component name and in linear units."""
    tau = np.exp(np.atleast_1d(np.asarray(params["log_tau"], dtype=float)))
    eta = np.exp(np.atleast_1d(np.asarray(params["log_eta"], dtype=float)))
    return {name: {"tau": float(tau[i]), "eta": float(eta[i])} for i, name in enumerate(names)}


def _has_scanned(value) -> bool:
    """Whether a declared semi-amplitude carries a scan axis rather than a prior."""
    if isinstance(value, Scanned):
        return True
    if isinstance(value, Spec):
        return False
    return any(isinstance(item, Scanned) for item in value)


def _vector_spec(value, n: int, what: str) -> Spec:
    """A spec for a vector-valued site, declared either as one spec or as a sequence."""
    if isinstance(value, Spec):
        return value
    specs = [_coerce_spec(item, f"each entry of {what}") for item in value]
    if len(specs) != n:
        raise ValueError(f"{what} has {len(specs)} entries but the model has {n} components")
    kinds = {type(s) for s in specs}
    if kinds == {Fixed}:
        return Fixed(jnp.stack([jnp.asarray(s.value, dtype=jnp.float64) for s in specs]))
    if kinds == {Between}:
        # Each entry's own starting value travels with it. A declaration such as
        # k=[Between(10, 90, start_at=30), Between(10, 90, start_at=60)] is how the
        # component-swap degeneracy of a symmetric prior is broken: the conjunction scan
        # runs at these starting values, and equal starts do not distinguish the stars.
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
    never started at the origin, and a declared circular orbit does not sample these sites
    at all, so no gradient is taken at the singular point.
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
            # declared one, measured at half of it, so it is refused instead.
            raise ValueError(
                f"orbit.ecc lower bound must be 0; got {lo}. The sampled parameters are "
                "(sqrt(e)cos w, sqrt(e)sin w), in which a lower bound on e is an annulus "
                "rather than a box, so it cannot be expressed as a prior here. Use "
                "ecc=ab.Between(0.0, hi) and read the posterior, or ecc=ab.Fixed(e) with "
                "omega to hold it."
            )
        # The two sites are bounded independently, so the box corner reaches e = 2 * hi.
        # The declared bound is enforced by the model's disk factor instead, which is what
        # effective_ecc_max provides: without it, ecc=Between(0, 0.08) admits e = 0.16.
        limit = math.sqrt(hi)
        # Started at e = 0.05, omega = 0.5 rad unless the declaration says where: small,
        # but not the singular origin. A start from a velocity table's orbit spares the
        # scans a circular model of an eccentric orbit.
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


def _keplerian_velocities(theta, dis):  # pragma: no cover - convenience alias
    return np.asarray(orbit_velocities(theta, dis.dataset.bjd, ecc_max=dis.ecc_max))

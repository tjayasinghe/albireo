"""A declarative front end: describe the system, and let albireo assemble the fit.

**Experimental.** This module is a vocabulary, and a vocabulary is expensive to change
once people depend on it, so it is marked experimental until it has been used on
somebody else's problem. :class:`~albireo.inference.MarginalOrbitModel` and the functions
around it are the supported surface and are not going anywhere;
:meth:`Disentangler.expert` hands you exactly that surface, so dropping down costs three
lines rather than a rewrite.

The façade is not a shortcut — it is a **compiler**. You declare the system: which
components exist, how bright each one is, what the spectrograph does, and what is already
known about the orbit. It emits the expert path: the model grid and its margins, the
solver's velocity budget, the conjunction-phase scan, the matched ``priors``/``init``
pair, the empirical-Bayes two-step, the Laplace mass matrix. Four of those are *derived*
rather than defaulted, and :meth:`Disentangler.explain` prints every derivation.

What it deliberately will not do is guess at a number that is a scientific claim. Light
fractions have no default and never will: with constant light fractions the likelihood
sees only the products ``l_i * d_i`` (``docs/math.md`` §5.2), so every recovered line
depth scales as ``1 / l_i`` and the value you pass is an assumption the data cannot
contradict. The same goes for the period prior, the wavelength scale where it matters,
and the nebular velocity. Each of those is required, or refused, or reported in the
``Assumed, not measured`` block that :meth:`Fit.summary` always prints.

One warning about the shape of this interface, stated here because an argument list is
not neutral. ``Star(light=0.62)`` sits in the same constructor as
``period=Known(40.335, 0.5)`` and is formatted identically, but the two are not the same
kind of thing: the period is a measurement with an uncertainty, and the light fraction is
a choice. A constructor slot teaches people that everything in it is a measurement. The
mitigation here is prose — the assumptions block, and the FITS headers — and prose in an
output nobody must read is a partial mitigation. Quote your light fractions next to your
semi-amplitudes.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from albireo.data import Dataset
from albireo.forward import data_residual_zscores
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

# Barycentric motion, one way. Reserved in the velocity budget whenever the model has a
# component whose velocity law carries v_bary — a telluric column, or topocentric data.
_V_BARY_MAX = 30.0

# Numerical headroom only. The terms it multiplies are already a worst case — the sum of
# the priors' own upper bounds at the largest eccentricity they allow — so a large slack
# here would be double counting, and it is not free: the solver's bandwidth grows with the
# budget and the solve cost grows with the bandwidth.
_BUDGET_SLACK = 1.05
# How much room a `velocities=` declaration leaves the free table to move. A convention,
# not a measurement: the declared velocities are a starting point, and the fit must be
# able to travel without hitting the bandwidth guard. Reported as its own budget term.
_FREE_VELOCITY_HEADROOM = 2.0


# -- parameter specifications -------------------------------------------------
#
# Every one of these is a (distribution, starting value) pair rather than a distribution,
# because the two have to agree and keeping them in separate dicts is how they stop
# agreeing. It also lets the façade guarantee `set(priors) == set(init)` by construction
# instead of by an assertion the user has to remember to write.


class Spec:
    """Base class for a declared parameter. Not instantiated directly."""

    def distribution(self) -> dist.Distribution | None:
        """The numpyro prior, or ``None`` when this parameter is not sampled."""
        raise NotImplementedError

    def start(self):
        """The constrained starting value handed to :func:`albireo.run_map`."""
        raise NotImplementedError

    def upper(self) -> float:
        """The largest magnitude this parameter can plausibly take, for sizing."""
        raise NotImplementedError


@dataclass(frozen=True)
class Fixed(Spec):
    """A value held constant: no sample site, no gradient, no posterior width."""

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

    Use this for a literature period or a semi-amplitude from a cross-correlation study.
    In :meth:`Disentangler.scan` a ``Known`` primary semi-amplitude means something
    stronger — it is *marginalized* over rather than assumed, which is the difference
    between a detection statistic that can be trusted and one a 10% error in K₁ inflates
    (``docs/design.md`` D41).
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
    """A bounded uniform prior. ``start`` defaults to the midpoint."""

    lo: Any
    hi: Any
    start_at: Any = None

    def distribution(self):
        return dist.Uniform(_as_array(self.lo), _as_array(self.hi))

    def start(self):
        if self.start_at is not None:
            return _as_array(self.start_at)
        return 0.5 * (_as_array(self.lo) + _as_array(self.hi))

    def upper(self) -> float:
        bounds = np.abs(np.atleast_1d(np.asarray(self.hi, dtype=float)))
        return float(np.max(bounds))


@dataclass(frozen=True)
class Scanned(Spec):
    """A grid of trial values. Only meaningful for the companion's ``k`` in a scan."""

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
    """Any numpyro distribution, with a starting value and a declared support.

    The escape hatch, and the one spec that cannot work out its own bound: a distribution
    object does not have to have finite support, and the starting value is not one. Since
    the velocity budget is derived from the semi-amplitude priors' reach, ``upper`` is
    required wherever the spec is used for a quantity that has to be bounded — without it
    a ``Sampled`` ``k`` sizes the solver from wherever L-BFGS happened to start.
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
        f"Fixed; got {type(value).__name__}. A bare tuple is deliberately not accepted — "
        "(5.5, 6.5) reads as either a range or a two-component vector."
    )


# -- what the spectrograph does -----------------------------------------------


@dataclass(frozen=True)
class Smoothness:
    """Where a component's smoothness hyperparameters *start*, per component.

    ``tau`` and ``eta`` are fitted by empirical Bayes (ML-II) inside :meth:`Fit`, so
    these are the centre and width of the hyperprior, never the values used. They matter
    anyway, and by more than the word "starting value" suggests: on HR 6819 the
    rotationally broadened Be star wants ``tau0`` five orders of magnitude larger than its
    sharp-lined companion, and a hyperparameter that does not move from its start is one
    the data did not constrain — which :meth:`Fit.summary` flags.

    Parameters
    ----------
    tau0
        Centre of the curvature-penalty hyperprior. Large means smooth.
    eta0
        Centre of the ridge hyperprior, which pulls the component toward its continuum.
    sigma
        Width of both hyperpriors, in natural logs.
    """

    tau0: float = 300.0
    eta0: float = 5.0
    sigma: float = 3.0


@dataclass(frozen=True)
class LSF:
    """An instrument's line-spread function, as a declaration rather than a parameter.

    Parameters
    ----------
    sigma_kms
        Gaussian sigma in km/s. A sequence gives one width per entry of
        ``anchors_angstrom``, which is how a wavelength-dependent LSF is declared (D37).
    anchors_angstrom
        Wavelengths at which ``sigma_kms`` is specified; interpolated between.

    Notes
    -----
    This width also fixes the convolution kernel's radius, so it is an upper bound as
    well as a value: a fit that later infers a *wider* LSF is rejected rather than
    silently truncated. Give it a little room if you intend to drop down to the low-level
    API and infer the width.

    Gauss-Hermite skewness (``h3``, D38) is deliberately absent. It reaches the kernel only
    through :func:`albireo.build_problem`, not through the model class this builds, so a
    field here would have been accepted and then silently discarded — declaring an
    instrument property that never reaches the instrument. Use :meth:`Disentangler.expert`.
    """

    sigma_kms: Any
    anchors_angstrom: Sequence[float] | None = None

    @classmethod
    def from_resolution(cls, resolving_power: float, **kwargs) -> LSF:
        """Build from ``R = lambda / dlambda``, treating ``R`` as a **FWHM**.

        The conversion is ``sigma = c / (R * 2 sqrt(2 ln 2))``. Getting this wrong by
        using ``c / R`` directly is a factor of 2.35 in the kernel radius, and it is the
        single most common way to mis-specify an instrument.

        Examples
        --------
        >>> round(LSF.from_resolution(48_000).sigma_kms, 3)
        2.653
        """
        if not resolving_power > 0:
            raise ValueError(f"resolving power must be positive; got {resolving_power}")
        sigma = C_KMS / (float(resolving_power) * 2.0 * math.sqrt(2.0 * math.log(2.0)))
        return cls(sigma_kms=sigma, **kwargs)

    @property
    def max_sigma_kms(self) -> float:
        """The widest sigma declared, which is what sets the grid margin."""
        return float(np.max(np.atleast_1d(np.asarray(self.sigma_kms, dtype=float))))


def _coerce_lsf(value, key: str) -> LSF:
    if isinstance(value, LSF):
        return value
    if isinstance(value, int | float):
        return LSF(sigma_kms=float(value))
    raise TypeError(f"lsf[{key!r}] must be an LSF or a sigma in km/s; got {type(value).__name__}")


# -- the components -----------------------------------------------------------


@dataclass(frozen=True)
class Star:
    """One stellar component: a name, a light fraction, and how smooth it is.

    Parameters
    ----------
    name
        Used everywhere the component would otherwise be a row index — ``fit.star("Be")``,
        the plot legend, the FITS header, the ML-II report. Component order is a
        convention that nothing in the data can check, so naming it is the only defence
        against reading row 0 as the wrong star.
    light
        Fraction of the total light this star contributes. **Required, and an
        assumption**: with constant light fractions only the products ``l_i * d_i`` are
        observable, so every recovered depth scales as ``1 / l_i`` and no part of the fit
        can tell you that this number is wrong. Quote it beside any result derived from
        the spectra. The star lights must sum to 1.
    smoothness
        Where this component's ML-II hyperparameters start. A rotationally broadened star
        wants a much larger ``tau0`` than a sharp-lined one.
    """

    name: str
    light: float
    smoothness: Smoothness = field(default_factory=Smoothness)


@dataclass(frozen=True)
class Telluric:
    """A telluric component: static in the topocentric frame, no light fraction.

    Structurally it is a column whose light fraction is 1 and which therefore sits
    outside the stellar simplex — it multiplies the composite rather than contributing to
    it. Adding one costs barycentric motion in the velocity budget, which is reserved
    automatically.
    """

    smoothness: Smoothness = field(default_factory=lambda: Smoothness(tau0=50.0))


@dataclass(frozen=True)
class Nebular:
    """A nebular emission component: static in the **barycentric** frame, free amplitude.

    The mirror of :class:`Telluric` — see ``docs/design.md`` D40. Nebular flux is added on
    top of the total continuum and takes no light from the stars, so its per-epoch
    amplitude is a free parameter rather than a light fraction.

    Declaring one wires up nine coupled things that are individually easy to get wrong:
    the component column, the ``log_nebular_amp`` site and its prior and init, the
    per-pixel ``eta_profile`` that confines the component to its lines, the agreement
    between that profile's velocity and ``nebular_v_kms``, the smoothness prior object
    being passed at *construction* so the profile survives ML-II, and the extra velocity
    budget. Leaving the profile off costs 250 nats and 2.6% in K₂; leaving the whole
    component off costs 11.5% of the Hβ equivalent width and 59% of K₂.

    Parameters
    ----------
    v_kms
        Velocity of the nebula in the model grid's frame. **Not identified** — it decides
        only where the component's lines land on the grid, which the window profile then
        has to agree with. It is a placement convention, and is reported as one.
    lines
        Rest wavelengths in air angstrom. Default :data:`albireo.NEBULAR_LINES`.
    halfwidth_kms
        Half-width of each window. Generous is right: too narrow pushes real emission
        back into the stars, too wide only returns some of the freedom being removed.
    """

    v_kms: float = 0.0
    lines: Mapping[str, float] | Sequence[float] | None = None
    halfwidth_kms: float = 300.0
    smoothness: Smoothness = field(default_factory=lambda: Smoothness(tau0=8.0))


Component = Star | Telluric | Nebular


# -- the orbit ----------------------------------------------------------------


@dataclass(frozen=True)
class Orbit:
    """The orbital model, declared as what is known rather than as sample sites.

    Parameters
    ----------
    period
        Required, and there is no period *search*: the façade scans conjunction phase at
        one period, which is a different problem. Give a prior narrow enough that a phase
        scan is meaningful, or run a periodogram first.
    k
        Velocity semi-amplitude per :class:`Star`, in the order the stars are declared.
        One spec covering all of them, or one per star.
    t_conj
        Time of conjunction. The default ``"scan"`` locates it by scanning one period
        before optimizing anything — the marginal likelihood is sharply multimodal in
        phase, and L-BFGS started in the wrong trough converges confidently to the wrong
        answer.
    ecc
        Eccentricity. ``Between(lo, hi)`` samples it through the
        ``(sqrt(e) cos w, sqrt(e) sin w)`` parameterization; ``Fixed(0.0)`` is a genuinely
        circular orbit and is handled *exactly*, by not sampling those sites at all.
        That matters: the parameterization is singular at exactly ``e = 0``, where the
        gradient is NaN and numpyro reports only "Cannot find valid initial parameters".
    omega
        Argument of periastron. Required only when ``ecc`` is ``Fixed`` and nonzero.
    outer
        The outer orbit of a hierarchical triple, whose ``k`` must have two entries
        (inner pair, tertiary).
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
    """An itemized bound on the largest relative velocity between any two components.

    This is the number the solver's bandwidth is built from, and the one users get wrong.
    Too small and the sampler stalls against a guard it cannot see; too small *and*
    reached through :meth:`albireo.inference.MarginalOrbitModel.log_likelihood` directly
    and the answer is quietly wrong instead. It is derived from the semi-amplitude priors'
    own support, which is where the user already put the information.
    """

    terms: tuple[tuple[str, float], ...]
    total: float

    def __str__(self) -> str:
        rows = "\n".join(f"    {name:<34s} {value:8.1f}" for name, value in self.terms)
        return f"velocity budget (km/s)\n{rows}\n    {'total':<34s} {self.total:8.1f}"


def _velocity_budget(orbit: Orbit | None, velocities, components, frame: str, explicit):
    """Bound the relative velocity from the declaration, and explain every term.

    Two sources, one contract: the number has to bound the largest relative velocity
    between any two model components at any epoch, over everything the *prior* allows —
    not over what the answer turns out to be.

    From an :class:`Orbit` that is the semi-amplitude priors' own support. From a
    ``velocities=`` declaration it is the measured table, centred per component (see
    :func:`_centred_velocities` — the absolute level is unidentified and must not be paid
    for in bandwidth), with an explicit factor of two of headroom because the table is
    what the fit is free to move.
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

    Only the centred table is identified, and it is the only part that reaches the model:
    :func:`albireo.inference.relative_velocities` removes one zero point *per component*,
    because with no orbit tying the stars together each free spectrum absorbs a constant
    added to its own shifts. So the declared absolute level — the systemic velocity, +150
    km/s for the SMC — costs the solver nothing and must not be paid for in bandwidth.

    Centred here in velocity space rather than in pixel space. The model does it exactly,
    in pixels, where ``xi = artanh(v/c)`` makes the offset a translation; this is only a
    *bound*, and the two differ by ``O(v^2/c^2)`` — 1e-8 at stellar velocities, against
    the factor-of-two headroom the budget adds on top.
    """
    v = np.asarray(velocities, dtype=float)
    return v - v.mean(axis=1, keepdims=True)


def _place_hyperparameters(dis, priors: dict, init: dict) -> None:
    """The sites every declaration carries, whichever velocity model it uses.

    Smoothness is always fitted by ML-II: a fixed ``(tau, eta)`` pair is not something any
    part of this package does on real data, and defensible centres span five orders of
    magnitude between a sharp-lined and a rotationally broadened star. The nebular
    amplitudes come along for the same reason they do in
    :meth:`Fit._velocity_priors` — leaving them out would pin the component static at
    amplitude 1 without saying so.
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


def _check_velocities(dis, stars) -> np.ndarray:
    """Validate a ``velocities=`` declaration, and refuse the one that cannot work."""
    v = np.atleast_2d(np.asarray(dis.velocities, dtype=float))
    want = (len(stars), dis.dataset.n_epochs)
    if v.shape != want:
        raise ValueError(
            f"velocities must have shape {want} — one row per star, one column per epoch, "
            f"in the dataset's own epoch order; got {v.shape}"
        )
    if not np.all(np.isfinite(v)):
        raise ValueError("velocities must all be finite")

    # The cold start, refused rather than discovered. With every component at the same
    # velocity at every epoch the two stars are indistinguishable, and the fit does not
    # merely converge slowly — it lands 122,000 nats worse (D42). A declaration that says
    # nothing about their separation is that start with extra steps.
    separation = float(np.max(np.ptp(v, axis=0))) if v.shape[0] > 1 else float(np.ptp(v))
    if separation <= 0.0:
        raise ValueError(
            "these velocities never separate the components: every star has the same "
            "velocity at every epoch, which is exactly the cold start the free-velocity "
            "mode is measured to fail from (122,000 nats worse than a warm one, "
            "docs/design.md D42). Supply the per-epoch velocities you actually measured "
            "— cross-correlation lags, or line splitting read off the two most separated "
            "epochs — rather than a placeholder."
        )
    widest = max(_coerce_lsf(value, key).max_sigma_kms for key, value in dis.lsf.items())
    if separation < widest:
        warnings.warn(
            f"the declared velocities separate the components by at most "
            f"{separation:.2f} km/s, which is below the widest LSF sigma "
            f"({widest:.2f} km/s) — at no epoch are the two resolved. The free-velocity "
            "fit is warm-started from these, and a warm start inside the unresolved "
            "regime is close to the cold one that D42 measured failing. Check the sign "
            "convention and the epoch ordering before trusting the result.",
            RuntimeWarning,
            stacklevel=4,
        )
    return v


def _k_specs(orbit: Orbit, n: int) -> list[Spec]:
    """The per-component ``k`` specs, whether given as one spec or a sequence."""
    if isinstance(orbit.k, Spec):
        spec = orbit.k
        # A single spec may still be vector-valued (Between([10, 5], [90, 70])). Find that
        # from whichever field actually holds the numbers, not from `hi` alone — a vector
        # Known or Fixed has none, and would otherwise be counted at its largest entry n
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


# -- the façade ---------------------------------------------------------------


@dataclass(frozen=True)
class Disentangler:
    """A declared disentangling problem: components, orbit, instrument, data.

    Construction derives everything the low-level path makes you supply — the model grid
    and its margins, the solver's velocity budget, the matched ``priors`` and ``init``
    dicts — and refuses to derive the things that are scientific claims. Nothing is
    hidden: :meth:`explain` prints every derivation and :meth:`expert` hands back the
    ``(model, priors, init)`` triple to use directly.

    Parameters
    ----------
    dataset
        From :func:`albireo.read_dataset`, :func:`albireo.load_example`, or built by hand.
    components
        :class:`Star` per stellar component, optionally a :class:`Telluric` and a
        :class:`Nebular`. Order is the order the model uses; the names are how you get at
        the results afterwards.
    orbit
        An :class:`Orbit`. Exactly one of ``orbit`` and ``velocities`` is required.
    velocities
        The alternative declaration, for a binary whose orbit is **not known**: measured
        per-epoch velocities, ``(n_stellar, n_epochs)`` in km/s, from cross-correlation,
        a shift-and-add pipeline, or line splitting measured by hand. The fit is then the
        free per-epoch RV table (``docs/math.md`` §7.6) rather than a Keplerian — no
        orbital elements are sampled, and :meth:`fit` returns a velocity-mode
        :class:`Fit` whose table you can run a periodogram on.

        This exists because the ordering an unsolved system forces cannot be met the
        other way round. The free table needs a warm start — a cold one is measured at
        122,000 nats worse (``docs/design.md`` D42) — but the only warm start the façade
        used to offer was :meth:`Fit.free_velocities`, which needs a Keplerian fit, which
        needs a period. For a system with no published period that is a circle, and this
        breaks it: bring the velocities you have, get the table, get the period, then
        declare an :class:`Orbit`.

        These are a *starting point*, not a constraint: the per-component zero point is
        unidentified either way, so what they have to be right about is the epoch-to-epoch
        *pattern*, not the absolute scale.
    lsf
        Per-instrument :class:`LSF`, or a bare sigma in km/s. Every instrument in the
        dataset must appear.
    dv_kms
        Model-grid pixel size. Default: the dataset's own median native sampling.
    velocity_budget_kms
        Override the derived bound. An override *smaller* than the derived value raises,
        naming the terms that overflowed it.
    ecc_max
        Eccentricity clip, the solver's verified range.
    block_size
        Solver block size, passed through.

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

    With no known orbit, declare what you measured instead:

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

    # init=False so dataclasses.replace() cannot carry a stale grid, budget or model
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
                "A component contributing no light is not a component — remove it."
            )
        total = sum(float(s.light) for s in stars)
        if abs(total - 1.0) > 1e-6:
            listed = ", ".join(f"{s.name}={s.light:g}" for s in stars)
            raise ValueError(
                f"the star light fractions must sum to 1; {listed} sums to {total:g}. "
                "This is an assumption the data cannot check — with constant light "
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
                "through albireo.MarginalOrbitModel directly — Disentangler.expert() on a "
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
        # A telluric or nebular component is keyed to absolute line positions, so an
        # undeclared wavelength scale is an 83 km/s question rather than a formality.
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
        """The components in **model row order**: stars, then telluric, then nebular.

        The model fixes that order regardless of how the declaration was written, so every
        per-component array — the smoothness rows, the hyperprior centres, the assumptions
        report — has to be assembled through here. Iterating the declaration instead is a
        silent misassignment: the vectors still have the right length, so the length check
        in :func:`albireo.marginal_loglikelihood` passes, and a rotationally broadened star
        is then regularized with a sharp-lined star's curvature penalty.
        """
        return (
            *self.stars,
            *(c for c in self.components if isinstance(c, Telluric)),
            *(c for c in self.components if isinstance(c, Nebular)),
        )

    @property
    def effective_ecc_max(self) -> float:
        """The eccentricity the model actually clips at: the declared bound, or the solver's.

        The ``(secosw, sesinw)`` sites are bounded independently, so their box reaches
        ``e = 2 * hi`` at the corner. The model's own disk factor is what enforces a bound,
        so a declared ``ecc=Between(0, hi)`` has to become the model's ``ecc_max`` — or the
        declaration is decoration and the fit returns an eccentricity above it.
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
        """The itemized relative-velocity bound the solver bandwidth is built from."""
        return self._cached("budget", self._make_budget)

    @property
    def grid(self) -> LogGrid:
        """The model grid, wide enough for the largest shift plus the LSF kernel radius."""
        return self._cached("grid", self._make_grid)

    @property
    def smoothness_prior(self) -> SmoothnessPrior:
        """The prior object, carrying any per-pixel confinement profiles."""
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

    def _widest_lsf(self) -> float:
        return max(_coerce_lsf(v, k).max_sigma_kms for k, v in self.lsf.items())

    def _native_dv_kms(self) -> float:
        """The finest native pixel size in km/s across the dataset.

        The *finest*, not the median: a model grid coarser than any contributing epoch
        throws away that epoch's resolution, and on a mixed-instrument dataset the median
        follows whichever instrument brought more epochs rather than which resolves more.
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
            # nebular_windows takes rest wavelengths in *air*. On a vacuum grid the lines
            # are 0.87-2.74 A redward of that, so an unconverted window is offset by a
            # nearly constant 83 km/s — enough to clip the line it exists to contain, which
            # pushes the emission back into the stars. This is the whole reason the medium
            # is required before a Nebular component is accepted.
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
                    "structure and it would be pinned to the continuum everywhere — a free "
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
        )

    def _make_specs(self):
        """``(priors, init, fixed)`` — built together so their key sets cannot diverge."""
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
            # No orbital sites at all: `velocity` replaces them, and the model refuses
            # the two together. The prior is centred on zero rather than on the declared
            # table because the absolute level is unidentified either way — what the
            # declaration supplies is the *start*, which is the part that matters.
            sigma = self.velocity_budget.total / 2.0
            priors["velocity"] = (
                dist.Normal(0.0, sigma).expand(list(self.velocities.shape)).to_event(2)
            )
            init["velocity"] = jnp.asarray(self.velocities)
            _place_hyperparameters(self, priors, init)
            return priors, init, fixed

        place("period", _coerce_spec(self.orbit.period, "orbit.period"))
        if self.orbit.t_conj == "scan":
            # Replaced by the scan's answer in fit(); the site is always sampled, so a
            # narrow prior around the located phase is set there rather than here.
            # Anchored on the data, not on zero: real epochs are near BJD 2.46e6, and a
            # prior centred on the origin excludes every conjunction the data can have.
            # fit() narrows this around the scan's answer; expert() hands it out as is.
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
                "scan rather than a sampled site — there is no posterior over a grid. Call "
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
        return priors, init, fixed

    # -- inspection -----------------------------------------------------------

    def expert(self):
        """``(model, priors, init)`` — the low-level triple this declaration compiles to.

        The façade is a generator of the supported API, not a wall in front of it.
        Anything it does not expose — jitter, AR(1), inferred light fractions, inferred
        LSF widths — is three lines away from here.
        """
        return self.model, dict(self.priors), dict(self.init)

    def explain(self) -> str:
        """Every derived quantity, spelled out. Worth reading before trusting a fit."""
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
        """The sampled/fixed-site lines, or a note that this is a scan declaration."""
        try:
            priors, fixed = self.priors, self.fixed
        except ValueError as exc:
            return ["this is a scan declaration, not a fit declaration:", f"  {exc}"]
        lines = ["sampled sites: " + ", ".join(sorted(priors))]
        if fixed:
            lines.append("fixed sites:   " + ", ".join(sorted(fixed)))
        return lines

    def _refuse_anchored_lsf(self, what: str) -> None:
        """``k2_scan`` takes one width per instrument, so an anchored LSF cannot go in."""
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
        """The block naming every number the data cannot contradict."""
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

    def fit(self, *, max_steps: int = 300, tol: float | None = None, progress=None) -> Fit:
        """Locate the orbit and fit it: phase scan, then MAP with empirical Bayes.

        Runs, in order: a conjunction-phase scan over one period (unless ``t_conj`` was
        declared), then :func:`albireo.run_map` over the orbital sites *and* the
        smoothness hyperparameters — which, with the spectra already marginalized out, is
        the ML-II step. The fitted hyperparameters come back on :attr:`Fit.hyper`, keyed
        by component name, and :meth:`Fit.sample` freezes them.

        On a ``velocities=`` declaration there is no orbit and therefore no phase to
        locate: the scan is skipped and this fits the free per-epoch table directly,
        warm-started from the velocities you declared, returning a :class:`Fit` in
        ``"velocity"`` mode. Read it with :meth:`Fit.velocities` and
        :meth:`Fit.velocity_errors`, and take the period it implies into a second
        declaration carrying an :class:`Orbit`.

        Parameters
        ----------
        max_steps
            L-BFGS iteration cap.
        tol
            Gradient-norm threshold. Default: scaled to the number of good pixels, because
            the potential's scale grows with it and a fixed threshold is unreachable on a
            large dataset however good the fit is.
        progress
            ``callback(step, potential, grad_norm, params)``. Without one this is silent,
            and a real-data fit can run for hours.

        Returns
        -------
        Fit
        """
        priors, init = dict(self.priors), dict(self.init)
        scan = None
        # A free-velocity declaration has no phase to locate — there is no orbit, which is
        # the whole reason for that mode — so the scan and the period warning are skipped
        # rather than made to cope with a missing Orbit.
        if self.orbit is not None:
            self._warn_if_the_period_prior_is_a_search()
            if self.orbit.t_conj == "scan":
                scan = self._scan_phase(init)
                init["t_conj"] = scan.best
                priors["t_conj"] = dist.Uniform(
                    scan.best - 0.5 * scan.period, scan.best + 0.5 * scan.period
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
        )

    def _warn_if_the_period_prior_is_a_search(self) -> None:
        """A phase scan resolves phase at *one* period. It is not a period search.

        The scan runs at the period prior's midpoint, so a prior wide enough to be a search
        gets a phase located for a period that may be badly wrong, and L-BFGS then has to
        cross the same multimodal structure the scan exists to avoid. There is no threshold
        at which this becomes an error — it degrades — so it warns and names the fix.
        """
        if self.orbit is None:  # no period prior at all — nothing here can be a search
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

    def _scan_phase(self, init) -> PhaseScan:
        """Locate conjunction by scanning one period; L-BFGS cannot cross these troughs."""
        declared = init.get("period", self.fixed.get("period"))
        period = float(np.max(np.atleast_1d(np.asarray(declared))))
        trials = float(np.min(self.dataset.bjd)) + np.linspace(0.0, period, 41, endpoint=False)
        theta = {k: jnp.asarray(v) for k, v in init.items()}
        theta.update({k: jnp.asarray(v) for k, v in self.fixed.items()})
        values = []
        for t in trials:
            theta["t_conj"] = jnp.asarray(float(t))
            values.append(float(self.model.log_likelihood(theta)))
        best = float(trials[int(np.argmax(values))])
        return PhaseScan(period=period, trials=trials, values=np.asarray(values), best=best)

    # -- the SB1 workflow -----------------------------------------------------

    def _scan_orbit(self) -> dict:
        """The fixed SB1 orbit the scan holds, from the declared specs."""
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
        """The declared :class:`Orbit`, or the refusal — for the paths that need one.

        Shared by the scan entry points so the message is written once, and so the
        narrowing is visible to a type checker rather than implied by call order.
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
        """``(k1_spec, k2_spec)`` from a two-star declaration, checked for shape.

        The gate for both :meth:`scan` and :meth:`detection_limit`, which is why the
        no-orbit refusal is reached here rather than in each of them.
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
                "declare the companion's semi-amplitude as ab.Scanned(grid) — that grid is "
                "the scan's axis. The primary's is Fixed(k) to hold it, or Known(k, sigma) "
                "to marginalize over it, which is the only thing that catches a wrong K1 "
                "inflating the detection statistic (docs/design.md D41)."
            )
        if not isinstance(k1, Fixed | Known):
            raise ValueError("the primary's semi-amplitude must be Fixed(k) or Known(k, sigma)")
        return k1, k2

    def scan(self, *, block_size: int | None = None, sweep_batch: int | None = None):
        """Scan trial companion semi-amplitudes against the no-companion model.

        The SB1 faint-companion search: profile the marginal likelihood over the
        companion's velocity semi-amplitude, which finds a companion that never shows up
        as a second set of lines. Declare the primary's ``k`` as :class:`Known` rather than
        :class:`Fixed` to marginalize over it — a K₁ 10% high takes the recovered companion
        from 0.96 to 0.49 correlation with truth while *tripling* the detection statistic,
        so the artifact reads as a stronger detection and no calibrated threshold can catch
        it (``docs/design.md`` D41).

        Returns
        -------
        albireo.scan.K2ScanResult
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

        Produces the sentence a referee asks for — *any companion contributing more than
        X% of the light would have been detected at 95% confidence* — by resimulating this
        dataset through its own operators. Sharing the :class:`Disentangler` with
        :meth:`scan` is the point: every scan-shaped argument matching between the two is
        a requirement, and here it is structural rather than a documented obligation.

        Keyword arguments are passed to :func:`albireo.detection_limit`.
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
    """The conjunction-phase scan run before optimizing, kept so it can be inspected."""

    period: float
    trials: np.ndarray
    values: np.ndarray
    best: float

    @property
    def contrast(self) -> float:
        """Log-likelihood between the best and worst phase — how sharp the trough is."""
        return float(np.max(self.values) - np.min(self.values))


@dataclass(frozen=True)
class Fit:
    """A MAP fit, its ML-II hyperparameters, and everything derived from them.

    The container matters as much as the numbers. The smoothness hyperparameters fitted
    here have to accompany the parameters into every downstream call, and passing them by
    hand is exactly the step that gets skipped — so :meth:`spectra`, :meth:`std`,
    :meth:`composite` and :meth:`sample` all carry them for you.
    """

    dis: Disentangler
    result: Any
    hyper: dict[str, dict[str, float]]
    phase_scan: PhaseScan | None = None
    mode: str = "keplerian"
    # The priors this was actually fitted under. Kept because they are not recoverable from
    # the declaration once the Keplerian has been replaced by a free velocity table, and a
    # Laplace covariance is only meaningful against the model it came from.
    priors_used: dict = field(default_factory=dict, repr=False)

    @property
    def params(self) -> dict:
        """The MAP site values, as :func:`albireo.run_map` returned them."""
        return dict(self.result.params)

    @property
    def theta(self) -> dict:
        """Site values plus any fixed sites — what the forward model consumes."""
        theta = {k: jnp.asarray(v) for k, v in self.result.params.items()}
        theta.update({k: jnp.asarray(v) for k, v in self.dis.fixed.items()})
        return theta

    def star(self, name: str) -> dict:
        """Everything fitted for one named star, so no row index is ever read by hand."""
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
        """Period, eccentricity, omega and the semi-amplitudes, in physical units."""
        if self.mode != "keplerian":
            raise ValueError(
                "this fit replaced the Keplerian with a free velocity table, so it has no "
                "orbital elements. Use .velocities(), or .keplerian_residuals(kep)."
            )
        return orbit_parameters(self.theta, ecc_max=self.dis.effective_ecc_max)

    def velocities(self) -> np.ndarray:
        """Per-epoch velocities, ``(n_stellar, n_epochs)``, with the zero point removed.

        A free table carries **one arbitrary zero point per component**, not one in
        total: with no orbit tying the stars together, each free spectrum absorbs a
        constant added to its own shifts. The removal is done in pixel space, where a
        constant offset is exact rather than first-order.
        """
        if self.mode == "keplerian":
            return np.asarray(
                orbit_velocities(
                    self.theta, self.dis.dataset.bjd, ecc_max=self.dis.effective_ecc_max
                )
            )
        return np.asarray(relative_velocities(self.params["velocity"], self.dis.grid))

    def velocity_errors(self) -> np.ndarray:
        """Per-epoch velocity uncertainties, projected the same way as :meth:`velocities`.

        Never the raw Laplace diagonal: that returns the prior and nothing else — 37.95
        km/s on every entry against a real 0.059 — which is a number that looks equally
        convincing on a good dataset and a useless one.
        """
        if self.mode != "velocity":
            raise ValueError(
                "velocity_errors() applies to a free-velocity fit; call free_velocities() "
                "on this fit first."
            )
        # The covariance has to come from the model this fit was actually run against —
        # the same sampled sites, in the same order — or the projection reads the wrong
        # block. The priors are carried on the Fit for exactly this reason.
        keplerian = {"period", "t_conj", "secosw", "sesinw", "k"}
        fixed = {k: v for k, v in self.dis.fixed.items() if k not in keplerian}
        model = self.dis.model.model(self.priors_used, fixed=fixed or None)
        covariance = laplace_inverse_mass(
            model, self.result.params, model_args=(self.dis.model.problem,)
        )
        return np.asarray(relative_velocity_errors(covariance, self.result.unconstrained))

    def marginal(self):
        """The conditional solve at the MAP: spectra, precision, log-likelihood."""
        return self.dis.model.marginal(self.theta)

    def spectra(self) -> np.ndarray:
        """Conditional-mean component spectra ``d``, shape ``(n_comp, n_pix)``.

        Deviations from a unit continuum, so the component spectrum is ``1 + d`` and the
        observable is ``l_i * d_i`` — see :meth:`Disentangler.assumptions`.
        """
        return np.asarray(self.marginal().d_hat)

    def std(self) -> np.ndarray:
        """Pointwise standard deviation of :meth:`spectra`, same shape.

        Large wherever the prior rather than the data sets the spectrum, which is most of
        the continuum. It is the reason to export the band alongside the spectrum.
        """
        return np.asarray(spectra_std(self.marginal()))

    def composite(self) -> np.ndarray:
        """The light-weighted stellar sum ``sum_i l_i d_i`` — the quantity measured.

        The individual components are a *decomposition* of this; only their weighted sum
        is constrained by the data at every pixel.
        """
        d = self.spectra()[: self.dis.n_stellar]
        lights = np.array([s.light for s in self.dis.stars], dtype=float)
        return np.tensordot(lights, d, axes=(0, 0))

    def residual_zscores(self) -> np.ndarray:
        """Whitened data residuals. Their RMS is ~1 when the noise model is right."""
        return np.asarray(
            data_residual_zscores(self.dis.model.problem_at(self.theta), self.marginal().d_hat)
        )

    @property
    def z_rms(self) -> float:
        """RMS of :meth:`residual_zscores`. Read it before reaching for a jitter term."""
        z = self.residual_zscores()
        return float(np.sqrt(np.mean(np.square(z[np.isfinite(z)]))))

    def summary(self) -> str:
        """Everything worth reading after a fit, including what was assumed."""
        # The convergence flag is an absolute threshold on a potential whose scale grows
        # with the number of good pixels, so on real data it is routinely False at a
        # perfectly good optimum. Report what actually happened instead of grading it.
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
        """NUTS over the orbital sites, with the ML-II hyperparameters held fixed.

        Rebuilds the model with ``log_tau``/``log_eta`` injected as constants, takes the
        Laplace covariance at the MAP as the starting mass matrix, and samples.

        Fixing the hyperparameters is a plug-in approximation: the orbital credible
        intervals do **not** include smoothness uncertainty. To marginalize instead, take
        :meth:`Disentangler.expert` and leave them in the priors dict — more honest, and
        more expensive.
        """
        if self.mode != "keplerian":
            raise ValueError(
                "sample() draws the orbital posterior, and this fit replaced the Keplerian "
                "with a free velocity table — there are no orbital sites to sample. Sample "
                "the Keplerian fit instead, and use velocity_errors() here."
            )
        fixed = self._fixed_hyper()
        priors = {k: v for k, v in self.dis.priors.items() if k not in fixed}
        if self.phase_scan is not None:
            best, period = self.phase_scan.best, self.phase_scan.period
            priors["t_conj"] = dist.Uniform(best - 0.5 * period, best + 0.5 * period)
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
        # Everything that is not the orbit carries over unchanged — the smoothness
        # hyperparameters, and the per-epoch nebular amplitudes if this declaration has a
        # nebular component. Dropping the latter would leave the component static at
        # amplitude 1 without saying so.
        for site in ("log_tau", "log_eta", "log_nebular_amp"):
            if site in self.dis.priors:
                priors[site] = self.dis.priors[site]
                init[site] = self.dis.init[site]
        return priors, init

    def free_velocities(
        self, *, sigma_kms: float | None = None, max_steps: int = 300, progress=None
    ) -> Fit:
        """Refit with one free velocity per component per epoch, warm-started from here.

        The per-epoch RV table is the artifact the binary-star community expects from a
        spectroscopic analysis, and the model check the Keplerian mode exists for: fit
        free velocities, then ask whether a Keplerian threads them
        (:meth:`keplerian_residuals`).

        It hangs off a :class:`Fit` rather than being constructible on its own because a
        cold start does not work — measured at 122,000 nats worse than the warm-started
        answer (``docs/design.md`` D42). Warm-starting is therefore not a convenience
        here; it is the only mode that has been shown to succeed.

        This is the entry point when you *have* a Keplerian and want the table as a model
        check. When you do not — an unsolved system, where the table is what produces the
        period in the first place — declare ``Disentangler(velocities=...)`` and call
        :meth:`Disentangler.fit`, which warm-starts from measured velocities instead.
        """
        priors, init = self._velocity_priors(sigma_kms)
        # Whatever the declaration fixed about the *orbit* is meaningless now, and the
        # model rejects Keplerian sites alongside a velocity table, so they are dropped.
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
        """This table's velocities minus those of a Keplerian fit — the model check."""
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

    def match_labels(self, stars, **kwargs):
        """Fit Teff, log g, [M/H] and *v* sin *i* to the stellar components of this fit.

        The declarative route to :func:`albireo.match_labels`: the grid, the recovered
        spectra, their uncertainty band, the assumed light fractions, the instrument width
        and the dataset's wavelength medium all come from this fit rather than being passed
        again, so they cannot disagree with what was actually solved.

        Only the *stellar* rows are handed over — a telluric or nebular component is not a
        star and has no atmospheric parameters — and the light fractions travel with them,
        because the label fit's whole dilution model is built on knowing what was assumed.

        Parameters
        ----------
        stars
            Mapping of star name to :class:`albireo.StarLabels`, one per stellar component
            of this declaration. Names must match :attr:`Disentangler.stars`.
        **kwargs
            Passed to :func:`albireo.match_labels`.

        Returns
        -------
        LabelMatch

        Notes
        -----
        The dataset must declare its wavelength medium. That is not bureaucracy: matching
        against a synthetic grid is an 83 km/s question, and there is no safe default
        (``docs/math.md`` §9).
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
        """Export the component spectra and their uncertainty band. Needs astropy."""
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


@dataclass(frozen=True)
class Posterior:
    """A NUTS posterior over the orbit, carrying the fit it came from."""

    dis: Disentangler
    mcmc: Any
    map: Fit

    @property
    def samples(self) -> dict:
        """The posterior draws, keyed by site."""
        return self.mcmc.get_samples()

    def star(self, name: str) -> dict:
        """Posterior summary for one named star's semi-amplitude."""
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
        """Spectra drawn from the *joint* posterior, ``(num_draws, n_comp, n_pix)``.

        The scatter includes both the conditional spectral uncertainty and the orbital
        uncertainty. The ML-II hyperparameters are injected automatically — leaving them
        out is a silent error, since the sites are simply missing from the chain.
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
        """Posterior intervals, sampler diagnostics, and what was assumed."""
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
        """Convert to an arviz ``InferenceData``, with the component names attached."""
        from albireo.results import to_inference_data

        return to_inference_data(
            self.mcmc, component_names=list(self.dis.component_names), **kwargs
        )

    def write_spectra(self, path, *, num_draws: int = 32, seed: int = 0, **kwargs):
        """Export the posterior-mean spectra and their scatter across draws."""
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
    """A spec for a vector-valued site, whether declared as one spec or a sequence."""
    if isinstance(value, Spec):
        return value
    specs = [_coerce_spec(item, f"each entry of {what}") for item in value]
    if len(specs) != n:
        raise ValueError(f"{what} has {len(specs)} entries but the model has {n} components")
    kinds = {type(s) for s in specs}
    if kinds == {Fixed}:
        return Fixed(jnp.stack([jnp.asarray(s.value, dtype=jnp.float64) for s in specs]))
    if kinds == {Between}:
        return Between(
            jnp.stack([jnp.asarray(s.lo, dtype=jnp.float64) for s in specs]),
            jnp.stack([jnp.asarray(s.hi, dtype=jnp.float64) for s in specs]),
        )
    if kinds == {Known}:
        return Known(
            jnp.stack([jnp.asarray(s.value, dtype=jnp.float64) for s in specs]),
            jnp.stack([jnp.asarray(s.sigma, dtype=jnp.float64) for s in specs]),
        )
    raise TypeError(
        f"{what} mixes {sorted(k.__name__ for k in kinds)}. A vector-valued site takes one "
        "distribution family, so give every entry the same kind — or one spec with "
        "sequence-valued arguments, e.g. Between([10.0, 5.0], [90.0, 70.0])."
    )


def _ecc_sites(orbit: Orbit, ecc_max: float, suffix: str = "") -> list[tuple[str, Spec]]:
    """The ``(secosw, sesinw)`` sites, handling a circular orbit exactly.

    The parameterization is singular at exactly ``e = 0``: the gradient there is NaN and
    numpyro reports only "Cannot find valid initial parameters". So a free eccentricity is
    never started at the origin, and a *declared* circular orbit does not sample these
    sites at all — with no gradient taken, the singular point is simply not visited.
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
            # and a lower bound on e is an annulus in that plane, which this box cannot
            # express. Silently dropping it would report an eccentricity below the bound
            # the user declared — measured at half of it — so it is refused instead.
            raise ValueError(
                f"orbit.ecc lower bound must be 0; got {lo}. The sampled parameters are "
                "(sqrt(e)cos w, sqrt(e)sin w), in which a lower bound on e is an annulus "
                "rather than a box, so it cannot be expressed as a prior here. Use "
                "ecc=ab.Between(0.0, hi) and read the posterior, or ecc=ab.Fixed(e) with "
                "omega to hold it."
            )
        # The two sites are bounded independently, so the box corner reaches e = 2 * hi.
        # The declared bound is enforced by the model's own disk factor instead, which is
        # why _effective_ecc_max exists — without it, ecc=Between(0, 0.08) admits e = 0.16.
        limit = math.sqrt(hi)
        # Started at e = 0.05, omega = 0.5 rad: small, but never the singular origin.
        start = math.sqrt(0.05)
        return [
            (f"secosw{suffix}", Between(-limit, limit, start * math.cos(0.5))),
            (f"sesinw{suffix}", Between(-limit, limit, start * math.sin(0.5))),
        ]
    raise TypeError(f"orbit.ecc must be Fixed or Between; got {type(ecc).__name__}")


def _keplerian_velocities(theta, dis):  # pragma: no cover - convenience alias
    return np.asarray(orbit_velocities(theta, dis.dataset.bjd, ecc_max=dis.ecc_max))

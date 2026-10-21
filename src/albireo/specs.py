"""Parameter specifications: a prior and a starting value declared together.

**Experimental**, as :mod:`albireo.facade` is: these classes are its declaration
vocabulary, and the names may change.

A specification is a (distribution, starting value) pair, so that the prior and the
initial value of a site are declared in one object and ``set(priors) == set(init)`` holds
by construction. :class:`Fixed` holds a site constant, :class:`Known` is a measurement
with a Gaussian uncertainty, :class:`Between` is a bounded uniform prior,
:class:`Scanned` is the trial grid of a semi-amplitude search, and :class:`Sampled` takes
any numpyro distribution.

:mod:`albireo.facade` and the top-level ``albireo`` namespace re-export the classes, and
``ab.Between`` is the form used in the documentation. They are defined in this module so
that the label fit (:mod:`albireo.match`) can construct them without importing
:mod:`albireo.facade`, which imports the label fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

__all__ = ["Between", "Fixed", "Known", "Sampled", "Scanned", "Spec"]


class Spec:
    """Base class for a declared parameter. Not instantiated directly."""

    def distribution(self) -> dist.Distribution | None:
        """The numpyro prior, or ``None`` when this parameter is not sampled."""
        raise NotImplementedError

    def start(self):
        """The constrained starting value passed to :func:`albireo.run_map`."""
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

    It suits a literature period or a semi-amplitude from a cross-correlation study. In
    :meth:`Disentangler.scan` a ``Known`` primary semi-amplitude is marginalized over
    rather than held fixed, which prevents a 10% error in K₁ from inflating the detection
    statistic.

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
        sigma = _concrete(self.sigma)
        if sigma is not None and not np.all(sigma > 0.0):
            raise ValueError(
                f"Known({self.value}, {self.sigma}): sigma must be positive. A zero or "
                "negative width gives a nan prior density; declare Fixed(value) to hold the "
                "parameter constant."
            )
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
        self._check_bounds()
        return dist.Uniform(_as_array(self.lo), _as_array(self.hi))

    def start(self):
        self._check_bounds()
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
        lo = np.abs(np.atleast_1d(np.asarray(self.lo, dtype=float)))
        hi = np.abs(np.atleast_1d(np.asarray(self.hi, dtype=float)))
        return float(np.max(np.maximum(lo, hi)))

    def _check_bounds(self) -> None:
        # Checked where the prior is used rather than at construction: the label fits
        # build library-derived default ranges eagerly, and a default that the
        # declaration replaces may be empty without consequence.
        lo, hi = _concrete(self.lo), _concrete(self.hi)
        if lo is not None and hi is not None and not np.all(lo < hi):
            raise ValueError(
                f"Between({self.lo}, {self.hi}): the lower bound must lie below the upper "
                "bound. Uniform(lo, hi) with lo >= hi has no support, and its log density "
                "is nan."
            )


@dataclass(frozen=True)
class Scanned(Spec):
    """A grid of trial values, not a prior. Meaningful only for the companion's ``k``.

    The grid is the axis of the faint-companion search run by
    :meth:`Disentangler.scan` and :meth:`Disentangler.detection_limit`. There is no
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

    This is the only specification that cannot derive its own bound: a distribution object
    need not have finite support, and the starting value is not a bound. The velocity
    budget is derived from the upper bounds of the semi-amplitude priors, so ``upper_bound``
    is required wherever the specification is used for a quantity that must be bounded.
    Without it, the solver would be sized from the starting value of a ``Sampled`` ``k``.

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
    """``value`` as a float64 JAX array (0-d for a scalar)."""
    return jnp.asarray(value, dtype=jnp.float64)


def _concrete(value) -> np.ndarray | None:
    """``value`` as a float NumPy array, or ``None`` when it is a traced JAX value."""
    try:
        return np.asarray(value, dtype=float)
    except TypeError:  # jax.errors.TracerArrayConversionError is a TypeError
        return None

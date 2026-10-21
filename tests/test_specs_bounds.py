"""Bounds of the parameter specifications (``albireo.specs``).

``Spec.upper`` is the largest magnitude a parameter can take. The label fit sizes its
warm-start velocity grid from it, so a bound that reads only ``hi`` places that grid
outside a negative prior. A ``Between`` with ``lo >= hi`` or a ``Known`` with a
non-positive width has a nan log density, and the fit then fails far from the cause.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from albireo.specs import Between, Known


def test_between_upper_is_the_largest_magnitude_of_either_bound():
    # A velocity prior entirely below zero: the largest magnitude is |lo|, not |hi|.
    assert Between(-120.0, -40.0).upper() == 120.0
    assert Between([-120.0, 10.0], [-40.0, 90.0]).upper() == 120.0
    # Symmetric and non-negative ranges, the forms every caller uses, are unchanged.
    assert Between(-50.0, 50.0).upper() == 50.0
    assert Between(0.0, 300.0).upper() == 300.0
    assert Between([10.0, 5.0], [90.0, 70.0]).upper() == 90.0


@pytest.mark.parametrize(
    ("lo", "hi"),
    [(50.0, 10.0), (5.0, 5.0), ([10.0, 50.0], [90.0, 20.0])],
    ids=["reversed", "empty", "one_entry_reversed"],
)
def test_between_with_an_empty_range_raises_where_it_is_used(lo, hi):
    spec = Between(lo, hi)  # construction is allowed: an unused default may be empty
    with pytest.raises(ValueError, match="lower bound must lie below the upper bound"):
        spec.distribution()
    with pytest.raises(ValueError, match="lower bound must lie below the upper bound"):
        spec.start()
    # A starting value names the bounds as the cause, not the start.
    with pytest.raises(ValueError, match="lower bound must lie below the upper bound"):
        Between(lo, hi, start_at=30.0).start()


def test_known_with_a_non_positive_width_raises():
    with pytest.raises(ValueError, match="sigma must be positive"):
        Known(5.0, 0.0).distribution()
    with pytest.raises(ValueError, match="sigma must be positive"):
        Known([5.0, 6.0], [1.0, -1.0]).distribution()
    assert np.isfinite(float(Known(5.0, 1.0).distribution().log_prob(5.0)))


def test_valid_specs_keep_their_densities_and_traced_bounds_pass():
    spec = Between(-120.0, -40.0)
    assert float(spec.start()) == -80.0
    assert np.isfinite(float(spec.distribution().log_prob(-80.0)))

    # Bounds that are traced values cannot be checked, and are not rejected.
    def log_prob(lo):
        return Between(lo, lo + 1.0).distribution().log_prob(lo + 0.5)

    assert np.isfinite(float(jax.jit(log_prob)(jnp.asarray(0.0))))

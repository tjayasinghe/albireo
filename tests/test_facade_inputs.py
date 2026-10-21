"""Declarations and inputs of the `Disentangler` interface that must not be misread.

A per-star bound of a semi-amplitude range below zero is its largest magnitude, not its
upper end. A conjunction trial whose marginal likelihood is not finite is never the start
of a fit. The excluded ranges of an epoch renormalization are read once and checked
before any epoch is corrected. A default search window that the templates' zero points
leave empty raises its own subclass of ``ValueError``, which the pipeline alone retries.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import albireo as ab


@pytest.fixture(scope="module")
def dataset():
    return ab.load_example("sb2_sim")


@pytest.fixture(scope="module")
def truth():
    return ab.load_example("sb2_sim", with_truth=True)[1]


def _stars():
    return [ab.Star("primary", light=0.62), ab.Star("secondary", light=0.38)]


def _orbit(**kwargs):
    defaults = {"period": ab.Between(5.5, 6.5), "k": ab.Between([10.0, 10.0], [90.0, 90.0])}
    return ab.Orbit(**{**defaults, **kwargs})


def _dis(dataset, **kwargs):
    options = {"components": _stars(), "orbit": _orbit(), "lsf": {"DEMO": 6.5}}
    return ab.Disentangler(dataset, **{**options, **kwargs})


# -- the per-star bound of a vector range ---------------------------------------------


def test_each_star_of_a_vector_range_is_bounded_by_its_largest_magnitude():
    """A range below zero has its largest magnitude at ``lo``, as ``Between.upper`` takes it."""
    from albireo.facade import _k_specs, _velocity_budget

    views = _k_specs(_orbit(k=ab.Between([-120.0, 10.0], [-40.0, 90.0])), 2)
    assert [view.upper() for view in views] == [120.0, 90.0]
    budget = _velocity_budget(
        _orbit(k=ab.Between([-120.0, -90.0], [-40.0, -30.0]), ecc=ab.Fixed(0.0)),
        None,
        _stars(),
        "barycentric",
        None,
    )
    assert budget.terms[0] == ("sum of stellar |K| bounds", 210.0)

    # Ranges above zero, the form of every recorded run, keep their per-star bounds.
    for k, bounds in (
        (ab.Between([10.0, 5.0], [90.0, 70.0]), [90.0, 70.0]),
        (ab.Between([10.0, 5.0], 90.0), [90.0, 90.0]),
        (ab.Between([10.0, 10.0], [90.0, 90.0]), [90.0, 90.0]),
    ):
        assert [view.upper() for view in _k_specs(_orbit(k=k), 2)] == bounds


# -- the conjunction scan ------------------------------------------------------------


def _scripted_model(dis, values):
    """Replace the declaration's model by one returning ``values`` at the 42 phase trials."""
    period = float(np.asarray(dis.fixed["period"]))
    trials = float(np.min(dis.dataset.bjd)) + np.linspace(0.0, period, 42, endpoint=False)
    lookup = dict(zip(trials.tolist(), values, strict=True))
    dis._built["model"] = SimpleNamespace(
        log_likelihood=lambda theta: lookup[float(theta["t_conj"])]
    )
    return trials


def test_a_trial_whose_likelihood_is_not_finite_never_starts_the_fit(dataset):
    """``np.argmax`` takes the first NaN as the maximum; such trials rank below every finite one."""
    dis = _dis(dataset, orbit=_orbit(period=ab.Fixed(6.0)))
    values = -np.abs(np.arange(42.0) - 30.0)
    values[[0, 5]] = np.nan
    values[7] = -np.inf
    trials = _scripted_model(dis, values)
    scan = dis._scan_phase(dict(dis.init))
    assert scan.best == float(trials[30])
    np.testing.assert_array_equal(scan.values, values)  # recorded as evaluated, NaN included


def test_a_fit_whose_conjunction_scan_has_no_finite_trial_raises(dataset):
    dis = _dis(dataset, orbit=_orbit(period=ab.Fixed(6.0)))
    _scripted_model(dis, np.full(42, np.nan))
    assert dis._scan_phase(dict(dis.init)).values.size == 42  # the scan itself returns
    with pytest.raises(ValueError, match="no finite marginal likelihood at any of its 42 trials"):
        dis.fit(max_steps=1, k_scan=False)


# -- the excluded ranges of an epoch renormalization -----------------------------------


def _fit_at_injected(dis, truth):
    """A fit at the injected orbit, without an optimization."""
    import jax.numpy as jnp

    from albireo.inference import MAPResult

    t_conj = ab.t_conj_from_t_peri(
        truth["t_peri"], period=truth["period"], ecc=truth["ecc"], omega=truth["omega"]
    )
    root = np.sqrt(truth["ecc"])
    params = {
        "period": jnp.asarray(truth["period"]),
        "t_conj": jnp.asarray(float(t_conj)),
        "secosw": jnp.asarray(root * np.cos(truth["omega"])),
        "sesinw": jnp.asarray(root * np.sin(truth["omega"])),
        "k": jnp.asarray(truth["k"]),
        "log_tau": jnp.log(jnp.array([600.0, 1100.0])),
        "log_eta": jnp.log(jnp.array([10.0, 3.0])),
    }
    result = MAPResult(
        params=params,
        unconstrained={},
        potential=0.0,
        grad_norm=0.0,
        converged=True,
        num_steps=1,
        potential_trace=(0.0,),
    )
    hyper = {
        "primary": {"tau": 600.0, "eta": 10.0},
        "secondary": {"tau": 1100.0, "eta": 3.0},
    }
    return ab.Fit(dis=dis, result=result, hyper=hyper)


def test_every_epoch_and_pass_leaves_out_the_same_ranges(dataset, truth, monkeypatch):
    """A generator of ranges was exhausted by the first epoch of the first pass."""
    import albireo.preprocess as preprocess

    seen = []

    def correction(wave, flux, ivar, model, **kwargs):
        seen.append(tuple(tuple(r) for r in kwargs["exclude"]))
        return np.ones(np.size(wave))

    monkeypatch.setattr(preprocess, "continuum_correction", correction)
    fit = _fit_at_injected(_dis(dataset), truth)
    ranges = ((4510.0, 4515.0), (4530.0, 4532.5))
    fit.renormalize_epochs(passes=2, exclude=(r for r in ranges))
    assert len(seen) == 2 * dataset.n_epochs
    assert set(seen) == {ranges}


@pytest.mark.parametrize(
    ("exclude", "match"),
    [
        ([(4515.0, 4510.0)], "lo < hi"),
        ([(4510.0, 4510.0)], "lo < hi"),
        ([(4510.0, float("nan"))], "lo < hi"),
        ([4510.0], r"sequence of \(lo, hi\)"),
        ([(4510.0, 4515.0, 4520.0)], r"sequence of \(lo, hi\)"),
    ],
    ids=["reversed", "empty", "nan", "not_a_pair", "three_values"],
)
def test_a_malformed_excluded_range_fails_the_call_before_any_epoch(dataset, truth, exclude, match):
    """Raised by every epoch's correction, it was one warning per epoch and no correction."""
    fit = _fit_at_injected(_dis(dataset), truth)
    with pytest.raises(ValueError, match=match):
        fit.renormalize_epochs(exclude=exclude)


# -- the default search window of the epoch velocities ---------------------------------


def test_a_window_the_zero_points_leave_empty_raises_its_own_value_error():
    """The pipeline drops the zero points and measures again on this error, and on no other."""
    from albireo.facade import _default_v_range, _ZeroPointWindowError

    templates = [
        SimpleNamespace(name="A", v_zero_kms=-60.0),
        SimpleNamespace(name="B", v_zero_kms=35.0),
    ]
    fitted = np.array([[-30.0, 0.0, 30.0], [30.0, 0.0, -30.0]])
    with pytest.raises(_ZeroPointWindowError, match="misses its own fitted velocities"):
        _default_v_range(fitted, templates)
    assert issubclass(_ZeroPointWindowError, ValueError)
    dropped = [SimpleNamespace(name=t.name, v_zero_kms=None) for t in templates]
    assert _default_v_range(fitted, dropped) == [(-70.0, 70.0), (-70.0, 70.0)]

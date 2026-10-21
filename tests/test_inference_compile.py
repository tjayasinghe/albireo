"""The start and the compiled programs of `run_map` and `laplace_check`.

numpyro's `initialize_model` evaluates the model eagerly, one operation at a time, to find
the start, and the map to constrained values replays the whole model, the marginal
likelihood included. On a marginal-likelihood model both cost hundreds of compilations
and about a gigabyte. The start is instead found by an abstract trace, and the constrained
values come from the model without its likelihood. These tests hold both to numpyro's
results bit for bit, and count the compilations that remain. The programs are compiled
once per call and are not kept between calls, so a later call sees a model that changed.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.infer import init_to_value
from numpyro.infer.util import initialize_model

from albireo.inference import (
    MarginalOrbitModel,
    _initialize,
    laplace_check,
    run_map,
)

PRIORS = {
    "period": dist.Normal(6.0, 0.01),
    "t_conj": dist.Uniform(-3.0, 3.0),
    "secosw": dist.Uniform(-1.0, 1.0),
    "sesinw": dist.Uniform(-1.0, 1.0),
    "k": dist.Uniform(jnp.array([10.0, 10.0]), jnp.array([90.0, 90.0])),
    "log_tau": dist.Normal(jnp.full(2, np.log(300.0)), 3.0),
    "log_eta": dist.Normal(jnp.full(2, np.log(5.0)), 3.0),
}
INIT = {
    "period": 6.001,
    "t_conj": 0.05,
    "secosw": np.sqrt(0.02) * np.cos(0.5),
    "sesinw": np.sqrt(0.02) * np.sin(0.5),
    "k": jnp.array([42.0, 57.0]),
    "log_tau": jnp.full(2, np.log(300.0)),
    "log_eta": jnp.full(2, np.log(5.0)),
}


@pytest.fixture(scope="module")
def marginal_model(small_grid, small_dataset):
    return MarginalOrbitModel(
        small_grid,
        small_dataset,
        light_fractions=[0.6, 0.4],
        lsf_sigma_v={"A": 8.25},
        v_rel_max_kms=150.0,
    )


def _same(a, b) -> bool:
    return (
        type(a) is type(b)
        and np.asarray(a).dtype == np.asarray(b).dtype
        and np.array_equal(np.asarray(a), np.asarray(b))
    )


def test_the_start_and_the_potential_are_numpyros(marginal_model):
    model = marginal_model.model(PRIORS)
    args = model.model_args
    key = jax.random.PRNGKey(0)
    info = initialize_model(
        key, model, init_strategy=init_to_value(values=INIT), dynamic_args=True, model_args=args
    )
    start = _initialize(model, INIT, key, args)
    assert start.sites is not None, "every site has a value, so numpyro is not needed"
    assert start.reduced, "the model without the likelihood records the same sites"
    assert list(start.z) == list(info.param_info.z)
    for site, value in info.param_info.z.items():
        assert _same(start.z[site], value), site
    z = jax.tree.map(jnp.asarray, info.param_info.z)
    potential = jax.jit(lambda zz, a, fn: fn(*a)(zz), static_argnums=2)
    assert float(potential(z, args, start.potential_fn)) == float(
        potential(z, args, info.potential_fn)
    )
    # The constrained values come from the model without its likelihood, and are those of
    # numpyro's replay of the whole model, deterministic sites included.
    moved = jax.tree.map(lambda x: x + 0.003, z)
    full = info.postprocess_fn(*args)(moved)
    reduced = start.postprocess_fn(*args)(moved)
    assert list(reduced) == list(full) and "ecc" in reduced and "omega" in reduced
    for site, value in full.items():
        assert _same(reduced[site], value), site


def test_a_prior_built_inside_the_model_gives_numpyros_start():
    """Bounds computed in the model are tracers of the abstract trace, then evaluated."""

    def model(scale):
        low = jnp.asarray(1.0) * 2.0
        numpyro.sample("x", dist.Uniform(low, jnp.asarray(9.0)))
        numpyro.sample("y", dist.Normal(0.0, 1.0))
        numpyro.factor("data", -0.5 * jnp.sum(scale))

    args = (jnp.ones(3),)
    values = {"x": 3.0, "y": 0.2}
    key = jax.random.PRNGKey(0)
    start = _initialize(model, values, key, args)
    info = initialize_model(
        key, model, init_strategy=init_to_value(values=values), dynamic_args=True, model_args=args
    )
    assert start.sites is not None
    for site, value in info.param_info.z.items():
        assert _same(start.z[site], value), site
    z = jax.tree.map(jnp.asarray, info.param_info.z)
    full = info.postprocess_fn(*args)(z)
    reduced = start.postprocess_fn(*args)(z)
    assert list(reduced) == list(full)
    for site, value in full.items():
        assert _same(reduced[site], value), site


def test_a_site_without_a_start_is_left_to_numpyro():
    def model():
        numpyro.sample("x", dist.Normal(0.0, 1.0))
        numpyro.sample("y", dist.Normal(0.0, 1.0))

    key = jax.random.PRNGKey(4)
    start = _initialize(model, {"x": 0.5}, key, ())
    info = initialize_model(key, model, init_strategy=init_to_value(values={"x": 0.5}))
    assert start.sites is None
    for site in ("x", "y"):
        assert _same(start.z[site], info.param_info.z[site]), site
    result = run_map(model, init={"x": 0.5}, max_steps=20)
    assert abs(float(result.params["y"])) < 1e-6


def test_an_invalid_start_raises_numpyros_error():
    """A start outside a bound of the model: the potential there is infinite."""

    def model():
        x = numpyro.sample("x", dist.Normal(0.0, 1.0))
        numpyro.factor("bound", jnp.where(x > 0.0, 0.0, -jnp.inf))

    for max_steps in (5, 0):
        with pytest.raises(RuntimeError, match="Cannot find valid initial parameters"):
            run_map(model, init={"x": -1.0}, max_steps=max_steps)
    with pytest.raises(RuntimeError, match="Cannot find valid initial parameters"):
        laplace_check(model, {"x": -1.0})
    assert run_map(model, init={"x": 1.0}, max_steps=5).num_steps >= 1


def test_run_map_compiles_its_step_once_per_call(marginal_model, compilations):
    """Weakly typed starts used to compile the step at the first three iterations."""
    warm = run_map(marginal_model.model(PRIORS), init=INIT, max_steps=5, tol=0.0)
    model = marginal_model.model(PRIORS)
    assert compilations(lambda: run_map(model, init=INIT, max_steps=5, tol=0.0)) == 1
    again = []
    assert compilations(lambda: again.append(run_map(model, init=INIT, max_steps=5))) == 1
    for site, value in warm.unconstrained.items():
        assert _same(again[0].unconstrained[site], value), site
    assert again[0].potential_trace == warm.potential_trace


def test_laplace_check_compiles_two_programs_per_call(marginal_model, compilations):
    """The derivatives in one program and the potential of the probes in another."""
    model = marginal_model.model(PRIORS)
    first = laplace_check(model, INIT)
    second = []
    assert compilations(lambda: second.append(laplace_check(model, INIT))) == 2
    np.testing.assert_array_equal(second[0].covariance, first.covariance)
    assert second[0].drop == first.drop


def test_run_map_sees_a_prior_changed_after_a_first_call(marginal_model):
    """The model reads the caller's priors when it is traced, as numpyro's models do."""
    priors = dict(PRIORS)
    model = marginal_model.model(priors)
    first = run_map(model, init=INIT, max_steps=10)
    priors["period"] = dist.Normal(6.2, 0.0005)  # the same support, a different density
    again = run_map(model, init=INIT, max_steps=10)
    fresh = run_map(marginal_model.model(dict(priors)), init=INIT, max_steps=10)
    assert again.potential_trace != first.potential_trace
    assert again.potential_trace == fresh.potential_trace
    for site, value in fresh.params.items():
        assert _same(again.params[site], value), site


def test_a_wrapper_of_a_fitted_model_is_fitted_as_itself():
    """`functools.wraps` copies the attributes of the wrapped model to the wrapper."""

    def base():
        return numpyro.sample("x", dist.Normal(0.0, 1.0))

    assert abs(float(run_map(base, init={"x": 0.5}, max_steps=50, tol=1e-8).params["x"])) < 1e-8
    laplace_check(base, {"x": 0.0})

    @functools.wraps(base)
    def wrapper():
        x = base()
        numpyro.factor("extra", -0.5 * ((x - 3.0) / 0.1) ** 2)

    # The posterior is normal with precision 1 + 100 and mean 300 / 101.
    result = run_map(wrapper, init={"x": 0.5}, max_steps=50, tol=1e-8)
    assert float(result.params["x"]) == pytest.approx(300.0 / 101.0, abs=1e-8)
    check = laplace_check(wrapper, {"x": 300.0 / 101.0})
    np.testing.assert_allclose(check.covariance, [[1.0 / 101.0]], rtol=1e-10)


def test_a_postprocess_model_with_other_sites_is_not_used():
    """A wrapper inherits the wrapped model's `postprocess_model`, which lacks its sites."""

    def base():
        x = numpyro.sample("x", dist.Normal(0.0, 1.0))
        numpyro.deterministic("x2", 2.0 * x)
        return x

    base.postprocess_model = base

    @functools.wraps(base)
    def with_site():
        y = numpyro.sample("y", dist.Normal(base(), 1.0))
        numpyro.deterministic("y2", 2.0 * y)

    @functools.wraps(base)
    def with_deterministic():
        numpyro.deterministic("x3", 3.0 * base())

    @functools.wraps(base)
    def with_factor():
        numpyro.factor("prior", -0.5 * base() ** 2)

    key = jax.random.PRNGKey(0)
    for model, values, reduced in (
        (base, {"x": 0.5}, True),
        (with_site, {"x": 0.5, "y": 0.2}, False),
        (with_deterministic, {"x": 0.5}, False),
        (with_factor, {"x": 0.5}, True),
    ):
        assert _initialize(model, values, key, ()).reduced is reduced, model.__name__
        result = run_map(model, init=values, max_steps=20)
        info = initialize_model(
            key, model, init_strategy=init_to_value(values=values), dynamic_args=True
        )
        expected = info.postprocess_fn()(jax.tree.map(jnp.asarray, result.unconstrained))
        assert list(result.params) == list(expected), model.__name__
        for site, value in expected.items():
            assert _same(result.params[site], value), (model.__name__, site)


def test_an_integer_start_is_left_to_numpyro():
    """An integer start on the real line has no gradient; inside a bound it is a float."""

    def model():
        numpyro.sample("x", dist.Normal(0.0, 1.0))
        numpyro.sample("u", dist.Uniform(0.0, 4.0))

    with pytest.raises(TypeError, match="grad requires real- or complex-valued inputs"):
        run_map(model, init={"x": 1, "u": 2.0}, max_steps=5)
    key = jax.random.PRNGKey(0)
    values = {"x": 0.5, "u": 1}
    start = _initialize(model, values, key, ())
    info = initialize_model(key, model, init_strategy=init_to_value(values=values))
    assert start.sites is not None
    for site, value in info.param_info.z.items():
        assert _same(start.z[site], value), site


def test_the_log_likelihood_does_not_recompile_for_python_floats(marginal_model):
    """A theta of Python floats and one of arrays share one compiled program."""
    arrays = {k: jnp.asarray(np.asarray(v, dtype=np.float64)) for k, v in INIT.items()}
    reference = float(marginal_model.log_likelihood(arrays))
    programs = marginal_model._marginal_jit._cache_size()
    assert float(marginal_model.log_likelihood(dict(INIT))) == reference
    assert marginal_model._marginal_jit._cache_size() == programs

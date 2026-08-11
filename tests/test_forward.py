"""Tests for the fixed-parameter forward model (adjoints, grouping, frames, masks)."""

import jax.numpy as jnp
import numpy as np
import pytest

import albireo as ab
from albireo.forward import (
    apply_model,
    apply_model_adjoint,
    build_problem,
    normal_matvec,
)
from albireo.simulate import InstrumentSpec, simulate_dataset, synthetic_deviation_spectrum

RNG = np.random.default_rng(21)

GRID = ab.LogGrid.from_wavelength_range(4500.0, 4530.0, dv_kms=3.0)
VEL = np.array([[20.0, -35.0, 5.0], [-30.0, 50.0, -8.0]])
LIGHT = [0.6, 0.4]
LSF = {"A": 5.0, "B": 12.0}


def make_dataset(**kwargs):
    defaults = dict(
        bjd=np.arange(3.0),
        velocities=VEL,
        light_fractions=LIGHT,
        instruments={
            "A": InstrumentSpec(wave=np.arange(4504.0, 4526.0, 0.05), sigma_v_lsf=5.0, snr=80.0),
            "B": InstrumentSpec(wave=np.arange(4506.0, 4524.0, 0.10), sigma_v_lsf=12.0, snr=40.0),
        },
        epoch_instruments=["A", "B", "A"],
        v_bary=np.array([10.0, -15.0, 0.0]),
        seed=2,
    )
    defaults.update(kwargs)
    comps = [synthetic_deviation_spectrum(GRID, seed=s, margin=0.1) for s in (1, 2)]
    return simulate_dataset(GRID, comps, **defaults)


def make_problem(ds, **kwargs):
    defaults = dict(velocities=VEL, light_fractions=LIGHT, lsf_sigma_v=LSF)
    defaults.update(kwargs)
    return build_problem(GRID, ds, **defaults)


def test_apply_model_adjoint_identity():
    ds, _ = make_dataset()
    problem = make_problem(ds)
    d = jnp.asarray(RNG.standard_normal((2, GRID.n)))
    v = [jnp.asarray(RNG.standard_normal(np.asarray(g.z).shape)) for g in problem.groups]
    lhs = sum(float(jnp.vdot(m, vv)) for m, vv in zip(apply_model(problem, d), v, strict=True))
    rhs_val = float(jnp.vdot(d, apply_model_adjoint(problem, v)))
    np.testing.assert_allclose(lhs, rhs_val, rtol=1e-12)


def test_normal_matvec_is_symmetric():
    ds, _ = make_dataset()
    problem = make_problem(ds)
    u = jnp.asarray(RNG.standard_normal((2, GRID.n)))
    v = jnp.asarray(RNG.standard_normal((2, GRID.n)))
    lhs = float(jnp.vdot(u, normal_matvec(problem, v)))
    rhs_val = float(jnp.vdot(normal_matvec(problem, u), v))
    np.testing.assert_allclose(lhs, rhs_val, rtol=1e-12)


def test_grouping_by_instrument():
    ds, _ = make_dataset()
    problem = make_problem(ds)
    by_name = {g.instrument: g for g in problem.groups}
    assert set(by_name) == {"A", "B"}
    assert by_name["A"].epoch_indices == (0, 2)
    assert by_name["B"].epoch_indices == (1,)
    assert problem.natural_half_bandwidth > 0


def test_topocentric_shift_composition():
    ds, _ = make_dataset()
    problem = make_problem(ds)
    g = {gr.instrument: gr for gr in problem.groups}["A"]
    # epoch 0: shift = xi(v_star) - xi(v_bary) in pixels
    expected = np.asarray(GRID.velocity_to_pixels(VEL[:, 0])) - np.asarray(
        GRID.velocity_to_pixels(10.0)
    )
    np.testing.assert_allclose(np.asarray(g.shifts)[0], expected, rtol=1e-12)


def test_telluric_component_appended():
    ds, _ = make_dataset()
    problem = make_problem(ds, telluric=True)
    assert problem.n_components == 3
    g = {gr.instrument: gr for gr in problem.groups}["A"]
    np.testing.assert_allclose(np.asarray(g.shifts)[:, 2], 0.0, atol=1e-15)  # static tellurics
    np.testing.assert_allclose(np.asarray(g.light)[:, 2], 1.0)


def test_masked_pixels_have_zero_weight():
    ds, _ = make_dataset(gap_fraction=0.1)
    problem = make_problem(ds)
    for g in problem.groups:
        for row, j in zip(np.asarray(g.w), g.epoch_indices, strict=True):
            np.testing.assert_array_equal(row == 0.0, ~ds[j].good)


def test_validation_errors():
    ds, _ = make_dataset()
    with pytest.raises(ValueError, match="LSF"):
        make_problem(ds, lsf_sigma_v={"A": 5.0})
    with pytest.raises(ValueError, match="sum to 1"):
        make_problem(ds, light_fractions=[0.7, 0.7])
    with pytest.raises(ValueError, match="epochs"):
        make_problem(ds, velocities=VEL[:, :2])


@pytest.mark.parametrize("telluric", [False, True])
def test_with_light_fractions_matches_build(telluric):
    ds, _ = make_dataset()
    ell = np.array([[0.55, 0.7, 0.62], [0.45, 0.3, 0.38]])
    reference = make_problem(ds, light_fractions=ell, telluric=telluric)
    swapped = ab.with_light_fractions(make_problem(ds, telluric=telluric), jnp.asarray(ell))
    d = RNG.standard_normal((reference.n_components, GRID.n))
    for m_ref, m_new in zip(apply_model(reference, d), apply_model(swapped, d), strict=True):
        np.testing.assert_allclose(np.asarray(m_new), np.asarray(m_ref), rtol=1e-14)
    if telluric:
        for g in swapped.groups:
            np.testing.assert_allclose(np.asarray(g.light)[:, 2], 1.0)


def test_with_light_fractions_constant_broadcast():
    ds, _ = make_dataset()
    reference = make_problem(ds, light_fractions=[0.7, 0.3])
    swapped = ab.with_light_fractions(make_problem(ds), jnp.asarray([0.7, 0.3]))
    for g_ref, g_new in zip(reference.groups, swapped.groups, strict=True):
        np.testing.assert_allclose(np.asarray(g_new.light), np.asarray(g_ref.light), rtol=0)


def test_with_lsf_matches_build_at_same_radius():
    # sigma chosen so ceil(4 * sigma_px) is unchanged: kernels must be identical.
    ds, _ = make_dataset()
    target = {"A": 5.1, "B": 11.8}
    reference = make_problem(ds, lsf_sigma_v=target)
    swapped = ab.with_lsf(make_problem(ds), {k: jnp.asarray(v) for k, v in target.items()})
    for g_ref, g_new in zip(reference.groups, swapped.groups, strict=True):
        assert g_ref.kernel.shape == g_new.kernel.shape
        np.testing.assert_allclose(np.asarray(g_new.kernel), np.asarray(g_ref.kernel), atol=1e-15)
        np.testing.assert_allclose(
            np.asarray(g_new.kernel_rev), np.asarray(g_ref.kernel)[::-1], atol=1e-15
        )


def test_with_lsf_narrower_width_agrees_to_truncation():
    # A narrower sigma at the (larger) build radius differs from a fresh build only
    # by the tail mass beyond the smaller natural radius (~1e-6 relative).
    ds, _ = make_dataset()
    target = {"A": 4.0, "B": 9.5}
    reference = make_problem(ds, lsf_sigma_v=target)
    swapped = ab.with_lsf(make_problem(ds), target)
    d = RNG.standard_normal((2, GRID.n))
    for m_ref, m_new in zip(apply_model(reference, d), apply_model(swapped, d), strict=True):
        np.testing.assert_allclose(np.asarray(m_new), np.asarray(m_ref), atol=2e-5)


def test_gaussian_kernel_traced_matches_static():
    from albireo.operators import gaussian_kernel, gaussian_kernel_traced

    static = np.asarray(gaussian_kernel(1.7))
    radius = (static.size - 1) // 2
    traced = np.asarray(gaussian_kernel_traced(jnp.asarray(1.7), radius))
    np.testing.assert_allclose(traced, static, atol=1e-15)
    assert abs(traced.sum() - 1.0) < 1e-14

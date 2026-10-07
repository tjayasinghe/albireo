"""Tests for the TODCOR mode: per-epoch velocities by N-dimensional correlation.

Six kinds of claim are tested here.

1. **The estimator is Zucker & Mazeh's.** On a uniform grid with uniform weights the
   weighted-least-squares surface that albireo evaluates equals the two-dimensional
   correlation ``R(s_1, s_2)`` to 1e-10. This is tested for the symmetric expression with
   the light ratio maximized out and for the original one with the ratio held, against an
   independent NumPy transcription of the published formulae.
2. **The closed loop recovers the injected velocities with calibrated errors.** Simulated
   SB2s pass through the real operator stack (LSF, rebin, cosmic-ray hits, gaps,
   barycentric motion), in both frames, with mixed instruments and one to three components.
3. **The diagnostics are triggered when they should be**: blending, the search edge, the
   unidentified zero point of a disentangled template, the continuum offset the nuisance
   absorbs.
4. **The search window bounds what is reported.** No shift outside it is evaluated. A
   component whose minimum the window does not bracket is returned as NaN rather than as
   the edge value. The reported shift is one at which the chi-square was evaluated, which
   is checked against a brute-force scan of every integer shift.
5. **The search returns the lowest minimum.** Where the lines of two components overlap
   the chi-square has a second minimum, at the pair exchanged about its light-weighted
   mean velocity. The reported chi-square is checked against a lattice of fractional
   shifts built from an independent NumPy transcription, and the pieces of the refinement
   (the cell quadratic, its minimum over the cell, the choice of starting points) are
   checked against direct evaluations.
6. **An epoch with a second solution is flagged.** ``margin`` is the rise in chi-square to
   the other minimum, checked against the same lattice. Below the threshold it raises
   ``blended``, and the epochs returned exchanged are among those flagged. The same pair
   of velocities in the other order is not a second solution. Where a second solution
   flags an epoch, the table has its velocities and their covariance.

All data are generated in the tests. Nothing is downloaded.
"""

from __future__ import annotations

import numpy as np
import pytest

import albireo as ab
from albireo.data import Dataset, EpochData
from albireo.todcor import Template, VelocityTable, todcor, todcor_batch, todcor_surface

GRID = ab.LogGrid.from_wavelength_range(5000.0, 5060.0, dv_kms=1.5)
LIGHT = (0.6, 0.4)
COMMON = {"v_range": (-150.0, 150.0), "lsf_sigma_v": {"a": 5.0}}


def _col(values, table):
    """Per-component values broadcast to the table's ``(n_comp, n_epochs)`` shape."""
    return np.repeat(np.asarray(values, dtype=float)[:, None], table.n_epochs, axis=1)


def components(grid=GRID, seeds=(21, 22)):
    return [
        ab.synthetic_deviation_spectrum(grid, seed=s, sigma_v_range=(4.0, 12.0), margin=0.12)
        for s in seeds
    ]


def simulate(frame="topocentric", seed=5, **kwargs):
    rng = np.random.default_rng(3)
    bjd = np.sort(rng.uniform(0.0, 21.0, size=8))
    c1, c2 = components()
    inst = {
        "a": ab.InstrumentSpec(wave=np.arange(5008.0, 5052.0, 0.05), sigma_v_lsf=5.0, snr=150.0)
    }
    orbit = ab.OrbitParams(period=6.31, t_peri=2.0, ecc=0.15, omega=0.7, k=(30.0, 55.0), gamma=12.0)
    options = {
        "instruments": inst,
        "light_fractions": LIGHT,
        "orbit": orbit,
        "seed": seed,
        "cosmic_fraction": 0.002,
        "gap_fraction": 0.03,
        "frame": frame,
    }
    options.update(kwargs)
    dataset, truth = ab.simulate_dataset(GRID, [c1, c2], bjd=bjd, **options)
    templates = [Template("A", GRID, c1, v_zero_kms=0.0), Template("B", GRID, c2, v_zero_kms=0.0)]
    return dataset, truth, templates


@pytest.fixture(scope="module")
def sb2():
    return simulate()


@pytest.fixture(scope="module")
def fixed_table(sb2):
    dataset, _, templates = sb2
    return todcor(dataset, templates, light=LIGHT, **COMMON)


# ---------------------------------------------------------------------------
# 1. the identity with the published two-dimensional correlation
# ---------------------------------------------------------------------------


def _shifted(t, n):
    """The integer shift operator: ``out[p] = t[p - n]`` with zero fill."""
    out = np.zeros_like(t)
    if n >= 0:
        out[n:] = t[: t.size - n]
    else:
        out[: t.size + n] = t[-n:]
    return out


def _grid_epoch(c1, c2, s1, s2, noise=0.005, seed=0):
    """A composite on the model grid itself, so the rebin is the identity and weights uniform."""
    rng = np.random.default_rng(seed)
    flux = 1.0 + LIGHT[0] * _shifted(c1, s1) + LIGHT[1] * _shifted(c2, s2)
    flux = flux + rng.normal(0.0, noise, flux.shape)
    epoch = EpochData(wave=GRID.wave, flux=flux, ivar=np.full(GRID.n, noise**-2), bjd=0.0)
    return Dataset([epoch], frame="barycentric")


def _classic_terms(z, c1, c2, shifts1, shifts2):
    """Zucker & Mazeh's one-dimensional ingredients, with the norms of the shifted templates."""
    a1 = np.stack([_shifted(c1, n) for n in shifts1], axis=1)
    a2 = np.stack([_shifted(c2, n) for n in shifts2], axis=1)
    norm_z = np.sqrt(z @ z)
    n1 = np.sqrt(np.sum(a1**2, axis=0))
    n2 = np.sqrt(np.sum(a2**2, axis=0))
    corr1 = (a1.T @ z) / (norm_z * n1)
    corr2 = (a2.T @ z) / (norm_z * n2)
    corr12 = (a1.T @ a2) / np.outer(n1, n2)
    return corr1, corr2, corr12, n1, n2


def test_free_light_surface_is_the_symmetric_todcor_expression():
    """R^2(s1, s2) = [c1^2 - 2 c1 c2 c12 + c2^2] / [1 - c12^2], Zucker & Mazeh (1994)."""
    c1, c2 = components()
    dataset = _grid_epoch(c1, c2, 17, -23)
    templates = [Template("A", GRID, c1), Template("B", GRID, c2)]
    surface = todcor_surface(
        dataset, 0, templates, v_range=(-60.0, 60.0), light="free", nuisance_order=None
    )
    shifts1 = np.rint(np.asarray(GRID.velocity_to_pixels(surface.v1))).astype(int)
    shifts2 = np.rint(np.asarray(GRID.velocity_to_pixels(surface.v2))).astype(int)
    z = dataset[0].flux - 1.0
    corr1, corr2, corr12, _, _ = _classic_terms(z, c1, c2, shifts1, shifts2)
    classic = (
        corr1[:, None] ** 2 - 2 * corr1[:, None] * corr2[None, :] * corr12 + corr2[None, :] ** 2
    ) / (1.0 - corr12**2)
    np.testing.assert_allclose(surface.r_squared, classic, rtol=0, atol=1e-10)
    assert surface.peak == (
        pytest.approx(float(GRID.pixels_to_velocity(17))),
        pytest.approx(float(GRID.pixels_to_velocity(-23))),
    )


def test_fixed_ratio_free_scale_surface_is_the_original_todcor_expression():
    """R(s1, s2; alpha) = (c1 + a' c2) / sqrt(1 + 2 a' c12 + a'^2), a' = alpha sigma_2 / sigma_1."""
    c1, c2 = components()
    dataset = _grid_epoch(c1, c2, 17, -23)
    templates = [Template("A", GRID, c1), Template("B", GRID, c2)]
    surface = todcor_surface(
        dataset, 0, templates, v_range=(-60.0, 60.0), light=LIGHT, scale="free", nuisance_order=None
    )
    shifts1 = np.rint(np.asarray(GRID.velocity_to_pixels(surface.v1))).astype(int)
    shifts2 = np.rint(np.asarray(GRID.velocity_to_pixels(surface.v2))).astype(int)
    z = dataset[0].flux - 1.0
    corr1, corr2, corr12, n1, n2 = _classic_terms(z, c1, c2, shifts1, shifts2)
    alpha = LIGHT[1] / LIGHT[0]
    a_prime = alpha * n2[None, :] / n1[:, None]
    classic = (corr1[:, None] + a_prime * corr2[None, :]) / np.sqrt(
        1.0 + 2.0 * a_prime * corr12 + a_prime**2
    )
    np.testing.assert_allclose(surface.r_squared, classic**2, rtol=0, atol=1e-10)


def test_fixed_light_surface_is_the_least_squares_with_the_scale_pinned():
    """With the fractions held exactly, chi2 = |z|^2 - 2 l.b + l.G.l, with no scale freedom."""
    c1, c2 = components()
    dataset = _grid_epoch(c1, c2, 17, -23)
    templates = [Template("A", GRID, c1), Template("B", GRID, c2)]
    surface = todcor_surface(
        dataset, 0, templates, v_range=(-60.0, 60.0), light=LIGHT, nuisance_order=None
    )
    shifts1 = np.rint(np.asarray(GRID.velocity_to_pixels(surface.v1))).astype(int)
    shifts2 = np.rint(np.asarray(GRID.velocity_to_pixels(surface.v2))).astype(int)
    z = dataset[0].flux - 1.0
    a1 = np.stack([_shifted(c1, n) for n in shifts1], axis=1)
    a2 = np.stack([_shifted(c2, n) for n in shifts2], axis=1)
    l1, l2 = LIGHT
    chi2 = (
        z @ z
        - 2 * (l1 * (a1.T @ z)[:, None] + l2 * (a2.T @ z)[None, :])
        + l1**2 * np.sum(a1**2, axis=0)[:, None]
        + l2**2 * np.sum(a2**2, axis=0)[None, :]
        + 2 * l1 * l2 * (a1.T @ a2)
    ) * dataset[0].ivar[0]
    np.testing.assert_allclose(surface.chi2, chi2, rtol=1e-10)


# ---------------------------------------------------------------------------
# 2. the closed loop
# ---------------------------------------------------------------------------


def test_fixed_light_recovers_the_injected_velocities(sb2, fixed_table):
    _, truth, _ = sb2
    table = fixed_table
    error = table.velocity - truth.velocities
    assert table.velocity.shape == (2, 8)
    assert np.all(np.abs(error) < 5.0 * table.sigma)
    assert np.sqrt(np.mean(error**2)) < 0.1
    pull = np.sqrt(np.mean((error / table.sigma) ** 2))
    assert 0.6 < pull < 1.6, pull
    assert table.refined.all()
    assert not table.blended.any()
    assert not table.at_edge.any()
    assert np.all((table.reduced_chi2 > 0.8) & (table.reduced_chi2 < 1.25))
    assert table.absolute == (True, True)
    assert table.good.all()
    assert table.light_mode == "fixed"
    np.testing.assert_allclose(table.light, _col(LIGHT, table), rtol=0, atol=0)


def test_the_detection_statistic_is_large_for_both_stars(fixed_table):
    assert np.all(fixed_table.delta_chi2 > 1e3)
    assert np.all(fixed_table.r_squared > 0.9)


def test_free_light_measures_the_fractions(sb2):
    dataset, truth, templates = sb2
    table = todcor(dataset, templates, light="free", **COMMON)
    assert table.light_mode == "free per epoch"
    np.testing.assert_allclose(table.light, _col(LIGHT, table), atol=0.01)
    np.testing.assert_allclose(table.light.sum(axis=0), 1.0, atol=0.01)
    assert np.all(np.abs(table.velocity - truth.velocities) < 5.0 * table.sigma)


def test_global_light_is_a_median_of_the_free_pass_and_is_then_held(sb2):
    dataset, truth, templates = sb2
    table = todcor(dataset, templates, light="global", **COMMON)
    assert table.light_mode == "global median"
    held = table.settings["global_light"]["a"]
    np.testing.assert_allclose(table.light, _col(held, table), rtol=0, atol=0)
    np.testing.assert_allclose(held, LIGHT, atol=0.005)
    first = table.settings["first_pass_light"]
    assert first.shape == table.light.shape
    assert np.all(np.abs(table.velocity - truth.velocities) < 5.0 * table.sigma)


def test_global_light_is_measured_at_the_epochs_with_positive_amplitudes(fixed_table):
    """A held fraction is a median of positive measured amplitudes, or an error (D68)."""
    from dataclasses import replace

    from albireo.todcor import _global_light, _LightNotMeasured

    light = np.array(fixed_table.light)
    light[:, :5] = np.array([[1.4], [-0.5]])  # a difference of the two templates
    light[:, 5:] = np.array([[0.7], [0.3]])
    np.testing.assert_allclose(_global_light(replace(fixed_table, light=light))["a"], [0.7, 0.3])
    # Where no epoch with positive amplitudes is usable, those epochs are taken as they are.
    flagged = replace(fixed_table, light=light, blended=np.ones(8, dtype=bool))
    np.testing.assert_allclose(_global_light(flagged)["a"], [0.7, 0.3])
    light[:, 5:] = np.array([[1.2], [-0.2]])
    assert issubclass(_LightNotMeasured, ValueError)
    with pytest.raises(_LightNotMeasured, match="no epoch gives a positive amplitude"):
        _global_light(replace(fixed_table, light=light))


def test_global_light_is_an_error_where_the_templates_fit_as_a_difference():
    """Templates that describe neither star measure no light, and ``"global"`` says so.

    Both stars have the spectrum ``l1`` and are never resolved. The templates are ``l1 +
    l2`` and ``l1 + 3 l2``, and the data are ``1.5`` of the first less ``0.5`` of the
    second, so the free pass gives a negative amplitude at every epoch. Before D68 the
    median of those amplitudes was held, and no epoch of the table was usable.
    """
    from albireo.todcor import _LightNotMeasured

    l1, l2 = components()
    bjd = np.sort(np.random.default_rng(3).uniform(0.0, 21.0, size=4))
    inst = {
        "a": ab.InstrumentSpec(wave=np.arange(5008.0, 5052.0, 0.05), sigma_v_lsf=5.0, snr=150.0)
    }
    orbit = ab.OrbitParams(period=6.31, t_peri=2.0, ecc=0.0, omega=0.0, k=(3.0, 4.0), gamma=12.0)
    dataset, _ = ab.simulate_dataset(
        GRID, [l1, l1], bjd=bjd, instruments=inst, light_fractions=LIGHT, orbit=orbit, seed=5
    )
    templates = [
        Template("A", GRID, l1 + l2, v_zero_kms=0.0),
        Template("B", GRID, l1 + 3.0 * l2, v_zero_kms=0.0),
    ]
    free = todcor(dataset, templates, light="free", **COMMON)
    np.testing.assert_allclose(free.light, _col([1.5, -0.5], free), atol=0.05)
    with pytest.raises(_LightNotMeasured, match="templates closer to the components"):
        todcor(dataset, templates, light="global", **COMMON)


def test_free_scale_reports_the_normalization_and_the_same_velocities(sb2, fixed_table):
    dataset, _, templates = sb2
    table = todcor(dataset, templates, light=LIGHT, scale="free", **COMMON)
    # The composite's scale is returned as the sum of the light row, close to one.
    np.testing.assert_allclose(table.light.sum(axis=0), 1.0, atol=0.02)
    np.testing.assert_allclose(table.light[1] / table.light[0], LIGHT[1] / LIGHT[0], rtol=1e-12)
    assert np.all(np.abs(table.velocity - fixed_table.velocity) < 3.0 * fixed_table.sigma)
    assert table.settings["n_parameters"] == fixed_table.settings["n_parameters"] + 1


def test_profiled_errors_are_the_ivar_errors_times_the_reduced_chi_square(sb2):
    dataset, _, templates = sb2
    profiled = todcor(dataset, templates, light=LIGHT, **COMMON)
    trusted = todcor(dataset, templates, light=LIGHT, errors="ivar", **COMMON)
    np.testing.assert_allclose(profiled.sigma_ivar, trusted.sigma, rtol=1e-12)
    np.testing.assert_allclose(
        profiled.sigma, profiled.sigma_ivar * np.sqrt(profiled.reduced_chi2)[None, :], rtol=1e-12
    )
    np.testing.assert_allclose(trusted.sigma, trusted.sigma_ivar, rtol=1e-12)
    np.testing.assert_array_equal(profiled.velocity, trusted.velocity)


def _correlated_sb2(phi: float, n_epochs: int = 40, seed: int = 9):
    """An SB2 whose pixel noise is AR(1) with the given lag-one correlation."""
    rng = np.random.default_rng(seed)
    bjd = np.sort(rng.uniform(0.0, 40.0, size=n_epochs))
    c1, c2 = components()
    inst = {
        "a": ab.InstrumentSpec(wave=np.arange(5008.0, 5052.0, 0.05), sigma_v_lsf=5.0, snr=150.0)
    }
    orbit = ab.OrbitParams(period=6.31, t_peri=2.0, ecc=0.15, omega=0.7, k=(30.0, 55.0), gamma=12.0)
    dataset, truth = ab.simulate_dataset(
        GRID,
        [c1, c2],
        bjd=bjd,
        instruments=inst,
        light_fractions=LIGHT,
        orbit=orbit,
        seed=seed,
        frame="barycentric",
        ar1_phi=phi,
    )
    templates = [Template("A", GRID, c1, v_zero_kms=0.0), Template("B", GRID, c2, v_zero_kms=0.0)]
    return dataset, truth, templates


def test_a_vanishing_noise_correlation_reproduces_the_curvature_errors(sb2):
    """The sandwich at phi -> 0 is the white-noise covariance the stencil measures."""
    dataset, _, templates = sb2
    white = todcor(dataset, templates, light=LIGHT, errors="ivar", **COMMON)
    tiny = todcor(dataset, templates, light=LIGHT, errors="ivar", noise_correlation=1e-9, **COMMON)
    np.testing.assert_array_equal(tiny.velocity, white.velocity)
    # The stencil differentiates the exact profiled chi-square; the Jacobian sandwich is
    # its Gauss-Newton form. They agree to the residual's share of the curvature.
    np.testing.assert_allclose(tiny.sigma, white.sigma, rtol=0.02)
    assert tiny.settings["noise_correlation"] == {"a": 1e-9}
    assert white.settings["noise_correlation"] == {"a": 0.0}


def test_correlated_noise_widens_the_errors_and_calibrates_the_pulls():
    """AR(1) noise in the pixels does not change the estimator and inflates its error.

    With the correlation declared, the pull rms returns to one. With it ignored, the
    diagonal curvature error is too small by the factor the sandwich measures.
    """
    phi = 0.6
    dataset, truth, templates = _correlated_sb2(phi)
    ignored = todcor(dataset, templates, light=LIGHT, errors="ivar", **COMMON)
    declared = todcor(
        dataset, templates, light=LIGHT, errors="ivar", noise_correlation={"a": phi}, **COMMON
    )
    np.testing.assert_array_equal(declared.velocity, ignored.velocity)
    good = ignored.good & declared.good
    assert good.sum() >= 30
    ratio = declared.sigma[:, good] / ignored.sigma[:, good]
    assert np.all(ratio > 1.15) and np.all(ratio < 2.5)
    diff = declared.velocity[:, good] - np.asarray(truth.velocities)[:, good]
    pull_declared = np.sqrt(np.mean((diff / declared.sigma[:, good]) ** 2))
    pull_ignored = np.sqrt(np.mean((diff / ignored.sigma[:, good]) ** 2))
    assert 0.75 < pull_declared < 1.3, pull_declared
    assert pull_ignored > 1.2 * pull_declared, (pull_ignored, pull_declared)
    # The blending and detection diagnostics are unchanged by the noise model.
    np.testing.assert_array_equal(declared.delta_chi2, ignored.delta_chi2)


def test_the_noise_correlation_is_checked(sb2):
    dataset, _, templates = sb2
    with pytest.raises(ValueError, match="does not have"):
        todcor(dataset, templates, light=LIGHT, noise_correlation={"b": 0.3}, **COMMON)
    with pytest.raises(ValueError, match=r"lie in \(-1, 1\)"):
        todcor(dataset, templates, light=LIGHT, noise_correlation=1.0, **COMMON)


def test_both_frames_recover_the_same_barycentric_velocities():
    topo, truth_t, templates = simulate(frame="topocentric")
    bary, truth_b, _ = simulate(frame="barycentric")
    np.testing.assert_allclose(truth_t.velocities, truth_b.velocities)
    table_t = todcor(topo, templates, light=LIGHT, **COMMON)
    table_b = todcor(bary, templates, light=LIGHT, **COMMON)
    assert np.all(np.abs(table_t.velocity - truth_t.velocities) < 5.0 * table_t.sigma)
    assert np.all(np.abs(table_b.velocity - truth_b.velocities) < 5.0 * table_b.sigma)
    # The barycentric corrections are tens of km/s; the two tables must agree to the noise.
    assert np.max(np.abs(topo.v_bary)) > 5.0
    assert np.all(np.abs(table_t.velocity - table_b.velocity) < 0.3)


def test_mixed_instruments_get_their_own_lsf_and_light_fractions():
    rng = np.random.default_rng(11)
    bjd = np.sort(rng.uniform(0.0, 21.0, size=6))
    c1, c2 = components()
    inst = {
        "a": ab.InstrumentSpec(wave=np.arange(5008.0, 5052.0, 0.05), sigma_v_lsf=5.0, snr=150.0),
        "b": ab.InstrumentSpec(wave=np.arange(5010.0, 5050.0, 0.08), sigma_v_lsf=9.0, snr=80.0),
    }
    orbit = ab.OrbitParams(period=6.31, t_peri=2.0, ecc=0.15, omega=0.7, k=(30.0, 55.0))
    dataset, truth = ab.simulate_dataset(
        GRID,
        [c1, c2],
        bjd=bjd,
        instruments=inst,
        epoch_instruments=["a", "b"] * 3,
        light_fractions=LIGHT,
        orbit=orbit,
        seed=2,
    )
    templates = [Template("A", GRID, c1, v_zero_kms=0.0), Template("B", GRID, c2, v_zero_kms=0.0)]
    table = todcor(
        dataset,
        templates,
        v_range=(-150.0, 150.0),
        light="global",
        lsf_sigma_v={"a": 5.0, "b": 9.0},
    )
    assert set(table.settings["global_light"]) == {"a", "b"}
    assert np.all(np.abs(table.velocity - truth.velocities) < 5.0 * table.sigma)
    # The lower-resolution, noisier instrument must have larger errors.
    is_b = np.array([i == "b" for i in table.instrument])
    assert np.median(table.sigma[:, is_b]) > np.median(table.sigma[:, ~is_b])


def test_single_template_is_the_one_dimensional_correlation():
    rng = np.random.default_rng(4)
    bjd = np.sort(rng.uniform(0.0, 21.0, size=4))
    (c1,) = components(seeds=(21,))
    inst = {
        "a": ab.InstrumentSpec(wave=np.arange(5008.0, 5052.0, 0.05), sigma_v_lsf=5.0, snr=100.0)
    }
    velocities = np.array([[-40.0, 12.5, 33.3, 71.0]])
    dataset, truth = ab.simulate_dataset(
        GRID, [c1], bjd=bjd, instruments=inst, light_fractions=[1.0], velocities=velocities, seed=1
    )
    table = todcor(dataset, [Template("A", GRID, c1, v_zero_kms=0.0)], light=[1.0], **COMMON)
    assert table.velocity.shape == (1, 4)
    assert np.all(np.abs(table.velocity - truth.velocities) < 5.0 * table.sigma)
    assert table.wilson() is None
    assert np.all(table.delta_chi2 > 1e3)


def test_three_templates_generalize_the_method():
    rng = np.random.default_rng(6)
    bjd = np.sort(rng.uniform(0.0, 21.0, size=3))
    c1, c2, c3 = components(seeds=(21, 22, 23))
    inst = {
        "a": ab.InstrumentSpec(wave=np.arange(5008.0, 5052.0, 0.05), sigma_v_lsf=5.0, snr=200.0)
    }
    velocities = np.array([[-30.0, 20.0, 45.0], [40.0, -35.0, -60.0], [5.0, 8.0, 3.0]])
    light = (0.5, 0.3, 0.2)
    dataset, truth = ab.simulate_dataset(
        GRID,
        [c1, c2, c3],
        bjd=bjd,
        instruments=inst,
        light_fractions=light,
        velocities=velocities,
        seed=3,
    )
    templates = [
        Template(n, GRID, c, v_zero_kms=0.0) for n, c in zip("ABC", (c1, c2, c3), strict=True)
    ]
    table = todcor(dataset, templates, v_range=(-80.0, 80.0), light="free", lsf_sigma_v={"a": 5.0})
    assert table.velocity.shape == (3, 3)
    assert np.all(np.abs(table.velocity - truth.velocities) < 5.0 * table.sigma)
    np.testing.assert_allclose(table.light, _col(light, table), atol=0.02)


# ---------------------------------------------------------------------------
# 3. the diagnostics
# ---------------------------------------------------------------------------


def test_a_continuum_offset_is_absorbed_by_the_nuisance_and_biases_the_light_without_it(sb2):
    dataset, _, templates = sb2
    shifted = Dataset(
        [
            EpochData(
                wave=e.wave,
                flux=e.flux + 0.02,
                ivar=e.ivar,
                bjd=e.bjd,
                v_bary=e.v_bary,
                instrument=e.instrument,
            )
            for e in dataset
        ],
        frame=dataset.frame,
    )
    with_nuisance = todcor(shifted, templates, light="free", nuisance_order=0, **COMMON)
    without = todcor(shifted, templates, light="free", nuisance_order=None, **COMMON)
    np.testing.assert_allclose(with_nuisance.light, _col(LIGHT, with_nuisance), atol=0.01)
    assert np.max(np.abs(without.light - np.array(LIGHT)[:, None])) > 0.02
    # The nuisance absorbs the constant, so the velocities are unaffected.
    clean = todcor(dataset, templates, light="free", nuisance_order=0, **COMMON)
    assert np.all(np.abs(with_nuisance.velocity - clean.velocity) < 1.0 * clean.sigma)


def test_the_template_zero_point_composes_relativistically(sb2, fixed_table):
    dataset, _, templates = sb2
    moved = [
        Template("A", GRID, templates[0].deviation, v_zero_kms=40.0),
        Template("B", GRID, templates[1].deviation, v_zero_kms=-25.0),
    ]
    table = todcor(dataset, moved, light=LIGHT, **COMMON)
    for i, offset in enumerate((40.0, -25.0)):
        b1 = fixed_table.velocity[i] / ab.C_KMS
        b2 = offset / ab.C_KMS
        expected = ab.C_KMS * (b1 + b2) / (1.0 + b1 * b2)
        np.testing.assert_allclose(table.velocity[i], expected, rtol=0, atol=1e-9)
    np.testing.assert_allclose(table.sigma, fixed_table.sigma, rtol=1e-6)


def test_an_unknown_zero_point_is_reported_as_differential(sb2, tmp_path):
    dataset, _, templates = sb2
    unknown = [
        Template("A", GRID, templates[0].deviation),
        Template("B", GRID, templates[1].deviation),
    ]
    table = todcor(dataset, unknown, light=LIGHT, **COMMON)
    assert table.absolute == (False, False)
    assert "differential" in table.summary()
    np.testing.assert_allclose(
        table.velocity, todcor(dataset, templates, light=LIGHT, **COMMON).velocity, rtol=1e-12
    )
    text = table.write(tmp_path / "table.rv").read_text(encoding="utf-8")
    assert "unidentified zero point" in text


def test_twin_stars_at_the_same_velocity_are_flagged_blended_and_separated_ones_are_not():
    rng = np.random.default_rng(8)
    bjd = np.sort(rng.uniform(0.0, 3.0, size=2))
    (c1,) = components(seeds=(21,))
    inst = {
        "a": ab.InstrumentSpec(wave=np.arange(5008.0, 5052.0, 0.05), sigma_v_lsf=5.0, snr=150.0)
    }
    velocities = np.array([[10.0, 60.0], [10.0, -60.0]])  # coincident, then 120 km/s apart
    dataset, truth = ab.simulate_dataset(
        GRID,
        [c1, c1],
        bjd=bjd,
        instruments=inst,
        light_fractions=LIGHT,
        velocities=velocities,
        seed=9,
    )
    templates = [Template("A", GRID, c1, v_zero_kms=0.0), Template("B", GRID, c1, v_zero_kms=0.0)]
    table = todcor(dataset, templates, light=LIGHT, **COMMON)
    assert bool(table.blended[0]) is True
    assert bool(table.blended[1]) is False
    assert not table.good[0] and table.good[1]
    assert np.all(np.abs(table.velocity[:, 1] - truth.velocities[:, 1]) < 5.0 * table.sigma[:, 1])


def test_a_minimum_at_the_search_edge_is_flagged(sb2):
    dataset, truth, templates = sb2
    # Star A never goes below -20 km/s in this orbit, so a range cut there has its minimum
    # at the edge.
    table = todcor(
        dataset,
        templates,
        v_range=[(-150.0, -20.0), (-150.0, 150.0)],
        light=LIGHT,
        lsf_sigma_v={"a": 5.0},
    )
    assert truth.velocities[0].min() > -20.0
    assert table.at_edge[0].all()
    assert not table.good.any()
    assert "at the search edge" in table.summary()


def test_a_search_window_may_be_declared_per_template(sb2, fixed_table):
    """One ``(lo, hi)`` per template, each in that template's own frame.

    Two components at different zero points cover different intervals of reported
    velocity, so the search window is per template rather than shared. The Disentangler
    interface builds the windows that way, and a window narrowed around each component's
    own velocities must give the same velocities as the shared one.
    """
    dataset, truth, templates = sb2
    repeated = todcor(
        dataset,
        templates,
        v_range=[(-150.0, 150.0), (-150.0, 150.0)],
        light=LIGHT,
        lsf_sigma_v={"a": 5.0},
    )
    np.testing.assert_array_equal(repeated.velocity, fixed_table.velocity)
    assert repeated.settings["v_range"] == [[-150.0, 150.0], [-150.0, 150.0]]
    per_component = [
        (float(v.min()) - 20.0, float(v.max()) + 20.0) for v in np.asarray(truth.velocities)
    ]
    assert per_component[0] != per_component[1]
    narrow = todcor(dataset, templates, v_range=per_component, light=LIGHT, lsf_sigma_v={"a": 5.0})
    assert not narrow.at_edge.any()
    np.testing.assert_allclose(narrow.velocity, fixed_table.velocity, rtol=0, atol=1e-6)
    with pytest.raises(ValueError, match="per template"):
        todcor(dataset, templates, v_range=[(-150.0, 150.0)], light=LIGHT, lsf_sigma_v={"a": 5.0})


def test_surface_peak_matches_the_table(sb2, fixed_table):
    dataset, _, templates = sb2
    surface = todcor_surface(dataset, 2, templates, light=LIGHT, step=2, **COMMON)
    assert surface.chi2.shape == (surface.v1.size, surface.v2.size)
    assert abs(surface.peak[0] - fixed_table.velocity[0, 2]) < 2.5 * GRID.dv_kms
    assert abs(surface.peak[1] - fixed_table.velocity[1, 2]) < 2.5 * GRID.dv_kms
    assert surface.names == ("A", "B")
    assert np.nanmax(surface.r_squared) < 1.0


def test_write_to_dict_and_summary(sb2, fixed_table, tmp_path):
    columns = fixed_table.to_dict()
    assert set(columns) >= {
        "bjd",
        "instrument",
        "v_A",
        "sigma_A",
        "v_B",
        "sigma_B",
        "chi2_red",
        "r2",
    }
    path = fixed_table.write(tmp_path / "sb2.rv", header="a test")
    lines = path.read_text(encoding="utf-8").splitlines()
    header = [line for line in lines if line.startswith("#")]
    rows = [line.split() for line in lines if not line.startswith("#")]
    assert "# a test" in header
    assert len(rows) == 8
    names = header[-1][2:].split()
    assert len(names) == len(rows[0])
    v_a = np.array([float(r[names.index("v_A")]) for r in rows])
    np.testing.assert_allclose(v_a, fixed_table.velocity[0], atol=1e-6)
    summary = fixed_table.summary()
    assert "Wilson slope" in summary and "absolute" in summary
    wilson = fixed_table.wilson()
    assert wilson is not None and abs(wilson[0] - (-55.0 / 30.0)) < 0.05
    comp = fixed_table.component("B")
    np.testing.assert_array_equal(comp["velocity"], fixed_table.velocity[1])
    with pytest.raises(KeyError):
        fixed_table.component("C")


# ---------------------------------------------------------------------------
# 4. the search window, and what is not measured at its edge
#
# These epochs are built on the model grid itself, so the rebin is the identity and the
# weights uniform, and they are correlated against a single unbroadened template with no
# nuisance term. The chi-square albireo minimizes is then exactly the sum `_brute_chi2`
# evaluates, and the shift it reports can be checked against a scan of every integer
# shift in the range. The copies are placed manually, so that the surface has the minima
# the test requires rather than the ones a simulated orbit produces.
# ---------------------------------------------------------------------------

NARROW = ab.synthetic_deviation_spectrum(GRID, seed=31, sigma_v_range=(1.6, 2.2), margin=0.12)
EDGE_MARGIN = 100  # template pixels kept free at each end, so no shift extends past the grid
EDGE_NOISE = 0.004
ONE_TEMPLATE = {"light": [1.0], "lsf_sigma_v": None, "nuisance_order": None}


def _copies_epochs(per_epoch, seed):
    """One epoch per entry, each a superposition of shifted copies of ``NARROW``."""
    keep = slice(EDGE_MARGIN, GRID.n - EDGE_MARGIN)
    rng = np.random.default_rng(seed)
    epochs = []
    for j, copies in enumerate(per_epoch):
        flux = 1.0 + sum(amp * _shifted(NARROW, shift) for shift, amp in copies)
        noisy = flux[keep] + rng.normal(0.0, EDGE_NOISE, GRID.n - 2 * EDGE_MARGIN)
        epochs.append(
            EpochData(
                wave=GRID.wave[keep],
                flux=noisy,
                ivar=np.full(noisy.size, EDGE_NOISE**-2),
                bjd=float(j),
            )
        )
    return Dataset(epochs, frame="barycentric")


def _brute_chi2(dataset, j, shifts):
    """The chi-square of one unit-amplitude template at each integer shift, by direct summation."""
    keep = slice(EDGE_MARGIN, GRID.n - EDGE_MARGIN)
    z = dataset[j].flux - 1.0
    w = dataset[j].ivar
    return np.array([np.sum(w * (z - _shifted(NARROW, int(d))[keep]) ** 2) for d in shifts])


def _pixels(velocity):
    return float(np.asarray(GRID.velocity_to_pixels(velocity)))


def test_a_shift_beyond_the_requested_range_is_not_measured(tmp_path):
    """Nothing is measured where the chi-square is still falling as the range ends.

    The last point evaluated is not a minimum, and writing it out would put a number in the
    table that reads as a measurement. The component is flagged and left NaN instead. The
    diagnostics of that point are kept, because they show that the epoch was at the edge
    rather than at a peak.
    """
    lo_pix = int(np.ceil(_pixels(-90.0)))
    top = float(GRID.pixels_to_velocity(20))
    dataset = _copies_epochs([[(23, 1.0)], [(-10, 1.0)]], seed=2)
    inside = np.arange(lo_pix, 21)
    assert inside[int(np.argmin(_brute_chi2(dataset, 0, inside)))] == 20  # no interior minimum
    assert inside[int(np.argmin(_brute_chi2(dataset, 1, inside)))] == -10

    table = todcor(
        dataset,
        [Template("A", GRID, NARROW, v_zero_kms=0.0)],
        v_range=(-90.0, top),
        coarse_step=1,
        **ONE_TEMPLATE,
    )
    np.testing.assert_array_equal(table.at_edge[0], [True, False])
    assert np.isnan(table.velocity[0, 0])
    assert np.isnan(table.sigma[0, 0]) and np.isnan(table.sigma_ivar[0, 0])
    assert not table.refined[0] and not table.good[0]
    # The epoch whose minimum the range does contain is measured, and the flagged epoch
    # keeps every diagnostic of the point that was evaluated.
    assert table.good[1] and abs(_pixels(table.velocity[0, 1]) + 10.0) < 0.05
    assert np.all(np.isfinite(table.chi2)) and np.all(np.isfinite(table.chi2_null))
    assert np.all(np.isfinite(table.light)) and np.all(np.isfinite(table.delta_chi2))
    assert np.all(np.isfinite(table.r_squared))
    assert "1 at the search edge, not measured" in table.summary()
    row = table.write(tmp_path / "edge.rv").read_text(encoding="utf-8").splitlines()[-2]
    assert row.split()[2:4] == ["nan", "nan"]  # v_A and sigma_A of the flagged epoch


def test_an_advancing_fine_window_reports_a_shift_it_evaluated():
    """The refinement window advances when its minimum is on its edge.

    The velocity and the diagnostics beside it must describe the same point, so the shift
    reported must come from the window in which the chi-square was last evaluated. Here the
    coarse pass selects the weaker of two copies and the deeper one is exactly one window
    away, so the window must move before anything is refined.
    """
    step = 6
    radius = step + 2
    lo_pix = int(np.ceil(_pixels(-90.0)))
    node = lo_pix + step * 6
    dataset = _copies_epochs([[(node, 1.0), (node + radius, 1.05)]], seed=1)

    shifts = np.arange(lo_pix, int(np.floor(_pixels(90.0))) + 1)
    curve = _brute_chi2(dataset, 0, shifts)
    best = int(shifts[int(np.argmin(curve))])
    nodes = shifts[(shifts - lo_pix) % step == 0]
    coarse_best = int(nodes[int(np.argmin(_brute_chi2(dataset, 0, nodes)))])
    assert coarse_best == node and best == node + radius
    assert abs(best - coarse_best) >= radius  # the first window ends on the minimum

    table = todcor(
        dataset,
        [Template("A", GRID, NARROW, v_zero_kms=0.0)],
        v_range=(-90.0, 90.0),
        coarse_step=step,
        **ONE_TEMPLATE,
    )
    assert table.refined[0] and table.good[0] and not table.at_edge[0, 0]
    assert abs(_pixels(table.velocity[0, 0]) - best) <= 1.0


# ---------------------------------------------------------------------------
# 5. the lowest minimum at blended epochs
#
# Two copies of one spectrum at light fractions 0.6 and 0.4, with lines 5 to 9 pixels wide
# (sigma) and 4 to 16 pixels apart, give a surface with two minima: the pair injected, and
# the pair with the same weighted mean and the opposite difference. Both lie on a valley
# along which the weighted mean is constant, much narrower than it is long, so that neither
# the coarse stride nor the integer shifts show which is the lower. The noise is low enough
# that the injected pair is the lower by hundreds in chi-square at every epoch.
# ---------------------------------------------------------------------------

BLEND_NOISE = 0.002


def _fractional(t, shift):
    """``t`` shifted by a fractional number of pixels, by the two-tap interpolation."""
    n = int(np.floor(shift))
    f = float(shift) - n
    return (1.0 - f) * _shifted(t, n) + f * _shifted(t, n + 1)


def _blend_dataset(shifts, seed, noise=BLEND_NOISE, light=LIGHT):
    """Epochs on the model grid: one spectrum at two fractional shifts per epoch."""
    c = ab.synthetic_deviation_spectrum(GRID, seed=21, sigma_v_range=(8.0, 14.0), margin=0.12)
    keep = slice(EDGE_MARGIN, GRID.n - EDGE_MARGIN)
    rng = np.random.default_rng(seed)
    epochs = []
    for j, (s1, s2) in enumerate(shifts):
        flux = 1.0 + light[0] * _fractional(c, s1) + light[1] * _fractional(c, s2)
        noisy = flux[keep] + rng.normal(0.0, noise, GRID.n - 2 * EDGE_MARGIN)
        epochs.append(
            EpochData(
                wave=GRID.wave[keep],
                flux=noisy,
                ivar=np.full(noisy.size, noise**-2),
                bjd=float(j),
            )
        )
    templates = [Template("A", GRID, c, v_zero_kms=0.0), Template("B", GRID, c, v_zero_kms=0.0)]
    return Dataset(epochs, frame="barycentric"), templates, c


def _blend_shifts(seed=4, n=16):
    """Pairs of shifts 3 to 16 pixels apart about a mean within two pixels of zero."""
    rng = np.random.default_rng(seed)
    separation = rng.uniform(3.0, 16.0, n) * rng.choice([-1.0, 1.0], n)
    mean = rng.uniform(-2.0, 2.0, n)
    return np.stack([mean + LIGHT[1] * separation, mean - LIGHT[0] * separation], axis=1)


def _lattice(dataset, j, c, shifts1, shifts2, half=40):
    """The chi-square at every pair of the fractional shifts given, by direct summation.

    The inner products are formed at the integer shifts and combined with the weights of
    the two-tap interpolation, which is exact for the shift operator of the model. The
    lowest lattice value is an upper bound on the minimum of the surface.
    """
    keep = slice(EDGE_MARGIN, GRID.n - EDGE_MARGIN)
    z = dataset[j].flux - 1.0
    w = float(dataset[j].ivar[0])
    integers = np.arange(-half, half + 1)
    columns = np.stack([_shifted(c, int(n))[keep] for n in integers], axis=1)
    b = columns.T @ z
    gram = columns.T @ columns

    def interpolation(shifts):
        position = np.asarray(shifts, dtype=float) + half
        n = np.clip(np.floor(position).astype(int), 0, integers.size - 2)
        f = position - n
        weights = np.zeros((position.size, integers.size))
        weights[np.arange(position.size), n] = 1.0 - f
        weights[np.arange(position.size), n + 1] += f
        return weights

    w1, w2 = interpolation(shifts1), interpolation(shifts2)
    l1, l2 = LIGHT
    return w * (
        z @ z
        - 2.0 * (l1 * (w1 @ b)[:, None] + l2 * (w2 @ b)[None, :])
        + l1**2 * np.einsum("ik,kl,il->i", w1, gram, w1)[:, None]
        + l2**2 * np.einsum("ik,kl,il->i", w2, gram, w2)[None, :]
        + 2.0 * l1 * l2 * (w1 @ gram @ w2.T)
    )


def _lattice_minimum(dataset, j, c, half=40, step=0.1):
    """The lowest chi-square on a lattice of ``step`` pixels over ``half`` on either side."""
    shifts = np.arange(0.0, 2 * half + step / 2, step) - half
    return float(_lattice(dataset, j, c, shifts, shifts, half=half).min())


@pytest.mark.parametrize("coarse_step", [None, 3, 6])
def test_the_lowest_minimum_is_found_where_the_lines_overlap(coarse_step):
    """The solution is the injected pair, at a chi-square no lattice point is below.

    Before D66 the search refined the lowest coarse sample alone, within the cells touching
    the lowest integer shift. On these 16 epochs it returned the exchanged pair at 1, 4 and
    5 of them for strides of 1 (the default here, with no line-spread width declared), 3
    and 6 pixels, up to 3e5 above the lowest chi-square, with a quoted error of 0.006 pixel.
    """
    shifts = _blend_shifts()
    dataset, templates, c = _blend_dataset(shifts, seed=11)
    table = todcor(
        dataset,
        templates,
        v_range=(-45.0, 45.0),
        light=LIGHT,
        lsf_sigma_v=None,
        nuisance_order=None,
        coarse_step=coarse_step,
    )
    assert table.refined.all() and not table.at_edge.any()
    measured = np.asarray(GRID.velocity_to_pixels(table.velocity))
    np.testing.assert_allclose(measured, shifts.T, rtol=0, atol=0.1)
    assert np.all((table.reduced_chi2 > 0.9) & (table.reduced_chi2 < 1.1))
    for j in range(dataset.n_epochs):
        assert table.chi2[j] <= _lattice_minimum(dataset, j, c) * (1.0 + 1e-9)


def test_the_cell_quadratic_is_the_chi_square_inside_a_cell():
    """``c + g.f + f.H.f / 2`` against the chi-square evaluated at fractional shifts."""
    from albireo.todcor import _cell_quadratic, _minimize_box_quadratic, _Terms

    rng = np.random.default_rng(6)
    n_pix, n_shift, n_tmpl = 60, 7, 3
    for m in (0, 2):
        columns = rng.normal(size=(n_tmpl, n_pix, n_shift))
        weight = rng.uniform(0.5, 2.0, n_pix)
        z = rng.normal(size=n_pix)
        basis = rng.normal(size=(n_pix, m))
        weighted = columns * weight[None, :, None]
        terms = _Terms(
            b=np.einsum("ins,n->is", weighted, z),
            gram=np.einsum("ins,knt->ikst", columns, weighted),
            pwa=np.einsum("nm,ins->ims", basis, weighted),
            pwp=basis.T @ (weight[:, None] * basis),
            pwz=basis.T @ (weight * z),
            zwz=float(z @ (weight * z)),
        )
        for _ in range(10):
            corner = rng.integers(0, n_shift - 1, n_tmpl)
            amps = rng.uniform(0.1, 0.9, n_tmpl)
            const, grad, hess = _cell_quadratic(terms, corner, amps)
            assert np.all(np.linalg.eigvalsh(hess) > -1e-9 * np.abs(hess).max())
            for f in rng.uniform(0.0, 1.0, (5, n_tmpl)):
                exact = terms.chi2(corner + f, amps)[0]
                assert const + grad @ f + 0.5 * f @ hess @ f == pytest.approx(exact, rel=1e-10)
            x, value = _minimize_box_quadratic(grad, hess)
            assert np.all((x >= 0.0) & (x <= 1.0))
            axes = np.meshgrid(*[np.linspace(0.0, 1.0, 21)] * n_tmpl, indexing="ij")
            lattice = np.stack(axes, axis=-1).reshape(-1, n_tmpl)
            on_lattice = lattice @ grad + 0.5 * np.einsum("ni,ik,nk->n", lattice, hess, lattice)
            assert value <= on_lattice.min() + 1e-9 * np.abs(on_lattice).max()
            assert value == pytest.approx(grad @ x + 0.5 * x @ hess @ x)


def test_the_box_minimum_of_a_singular_quadratic_is_on_the_boundary():
    """With no curvature along one direction the minimum is on a face of the cell."""
    from albireo.todcor import _minimize_box_quadratic

    direction = np.array([1.0, 1.0]) / np.sqrt(2.0)
    hess = 50.0 * np.outer(direction, direction)  # flat along (1, -1)
    grad = -hess @ np.array([0.4, 0.4]) + np.array([0.3, -0.3])  # a slope along the flat direction
    x, value = _minimize_box_quadratic(grad, hess)
    axes = np.meshgrid(*[np.linspace(0.0, 1.0, 401)] * 2, indexing="ij")
    lattice = np.stack(axes, axis=-1).reshape(-1, 2)
    on_lattice = lattice @ grad + 0.5 * np.einsum("ni,ik,nk->n", lattice, hess, lattice)
    assert value <= on_lattice.min() + 1e-12
    assert np.any((x == 0.0) | (x == 1.0))


def test_candidate_minima_are_those_that_can_be_the_lowest():
    """A basin whose lowest sample is the higher is kept, and a shallow basin is not."""
    from albireo.todcor import _candidate_minima

    x, y = np.meshgrid(np.arange(40.0), np.arange(40.0), indexing="ij")

    def basin(x0, y0, depth, curvature=0.6):
        return -depth * np.exp(-0.5 * curvature * ((x - x0) ** 2 + (y - y0) ** 2))

    def indices(surface, n_max, **kwargs):
        return [start.tolist() for start in _candidate_minima(surface, n_max, **kwargs)[0]]

    # The deeper basin is sampled half a pixel from its minimum along both axes and the
    # shallower one at its minimum, so the lowest sample is in the shallower. A third basin
    # of a fifth of the depth is sampled at its minimum.
    surface = basin(10.5, 10.5, 104.0) + basin(25.0, 25.0, 100.0) + basin(32.0, 8.0, 20.0)
    starts, floors = _candidate_minima(surface, 6)
    assert starts[0].tolist() == [25, 25]
    assert len(starts) > 1
    assert {tuple(start.tolist()) for start in starts[1:]} <= {
        (10, 10),
        (10, 11),
        (11, 10),
        (11, 11),
    }
    # The floor of a candidate is below the minimum of its basin, here -104 between samples.
    for start, floor in zip(starts[1:], floors[1:], strict=True):
        assert floor < -104.0 < surface[tuple(start)]
    assert floors[0] <= surface[25, 25]
    assert indices(surface, 1) == [[25, 25]]
    # A minimum on the boundary is left out of an interior search unless it is the lowest.
    edge = basin(0.0, 20.0, 50.0) + basin(20.0, 20.0, 49.5)
    assert indices(edge, 6) == [[0, 20], [20, 20]]
    assert indices(edge, 6, interior=True) == [[0, 20], [20, 20]]
    lower_inside = basin(0.0, 20.0, 49.5) + basin(20.0, 20.0, 50.0)
    assert indices(lower_inside, 6) == [[20, 20], [0, 20]]
    assert indices(lower_inside, 6, interior=True) == [[20, 20]]
    # With a slack the basins that can lie within it of the lowest are appended. The third
    # basin is 80 above and its floor 75 above, and the first entries are unchanged.
    assert indices(surface, 6, slack=70.0) == indices(surface, 6)
    near = indices(surface, 6, slack=90.0)
    assert near[:-1] == indices(surface, 6) and near[-1] == [32, 8]
    assert indices(surface, 1, slack=90.0) == [[25, 25], [32, 8]]
    assert indices(lower_inside, 6, interior=True, slack=5.0) == [[20, 20]]


# ---------------------------------------------------------------------------
# 6. the margin to a second minimum, and the blend flag
#
# The epochs are those of section 5 with more noise, so that the two minima of an epoch
# are comparable in depth. The pair exchanged about the light-weighted mean has the
# difference of the two shifts of the opposite sign, so the second minimum is the lowest
# point of the half-plane in which that sign is the other one.
# ---------------------------------------------------------------------------

BLEND_SEARCH = {
    "v_range": (-45.0, 45.0),
    "light": LIGHT,
    "lsf_sigma_v": None,
    "nuisance_order": None,
}


def _well(dataset, j, c, sign):
    """The lowest chi-square at which ``s1 - s2`` has the given sign, by direct summation.

    A lattice of 0.1 pixel locates the lowest point more than a pixel inside the half-plane,
    and a lattice of 0.005 pixel around that point gives the value.
    """
    coarse = np.arange(-30.0, 30.05, 0.1)
    chi2 = _lattice(dataset, j, c, coarse, coarse)
    inside = sign * (coarse[:, None] - coarse[None, :]) > 1.0
    i1, i2 = np.unravel_index(int(np.argmin(np.where(inside, chi2, np.inf))), chi2.shape)
    assert sign * (coarse[i1] - coarse[i2]) > 1.25  # a minimum, not the edge of the half-plane
    fine = np.arange(-0.1, 0.1025, 0.005)
    return float(_lattice(dataset, j, c, coarse[i1] + fine, coarse[i2] + fine).min())


def _correlation(table):
    c = table.covariance
    return c[:, 0, 1] / np.sqrt(c[:, 0, 0] * c[:, 1, 1])


def test_the_margin_is_the_rise_to_the_exchanged_minimum():
    """``margin`` equals the rise to the lowest point of the other half-plane.

    Where it is infinite the search refined no second minimum, and the other half-plane is
    then more than a thousand above the solution, against a threshold of 9.
    """
    shifts = _blend_shifts()
    dataset, templates, c = _blend_dataset(shifts, seed=11, noise=0.01)
    table = todcor(dataset, templates, **BLEND_SEARCH)
    measured = np.asarray(GRID.velocity_to_pixels(table.velocity))
    np.testing.assert_allclose(measured, shifts.T, rtol=0, atol=0.2)
    assert not table.blended.any() and not table.second_minimum.any()
    finite = np.isfinite(table.margin)
    assert 6 <= finite.sum() < dataset.n_epochs
    for j in range(dataset.n_epochs):
        sign = np.sign(measured[0, j] - measured[1, j])
        rise = _well(dataset, j, c, -sign) - _well(dataset, j, c, sign)
        if finite[j]:
            assert table.margin[j] == pytest.approx(rise, rel=1e-4, abs=0.05)
        else:
            assert rise > 1000.0


def test_an_epoch_with_a_second_minimum_as_deep_is_flagged():
    """The two epochs returned exchanged are flagged, and the curvature does not show them.

    At this noise the two closest pairs, 4 and 5 pixels apart, are returned at the
    exchanged pair, with the injected one 0.5 and 0.1 above in chi-square. The curvature
    there is regular (correlations of -0.74 and -0.61), so before D67 the two epochs were
    reported as usable, 25 and 39 quoted errors from the injected shifts.
    """
    shifts = _blend_shifts()
    dataset, templates, _ = _blend_dataset(shifts, seed=21, noise=0.04)
    table = todcor(dataset, templates, **BLEND_SEARCH)
    measured = np.asarray(GRID.velocity_to_pixels(table.velocity))
    wrong = np.any(np.abs(measured - shifts.T) > 1.0, axis=0)
    assert np.flatnonzero(wrong).tolist() == [3, 7]
    injected_sign = np.sign(shifts[:, 0] - shifts[:, 1])
    np.testing.assert_array_equal(np.sign(measured[0] - measured[1])[wrong], -injected_sign[wrong])
    np.testing.assert_array_equal(np.sign(measured[0] - measured[1])[~wrong], injected_sign[~wrong])
    np.testing.assert_array_equal(table.second_minimum, table.margin < 9.0 * table.reduced_chi2)
    np.testing.assert_array_equal(table.second_minimum, wrong)
    assert np.all(table.margin[wrong] < 1.0)
    assert np.all(np.abs(_correlation(table)) < 0.9)
    np.testing.assert_array_equal(table.blended, wrong)
    np.testing.assert_array_equal(table.good, ~wrong)
    assert "2 blended (2 by a second minimum)" in table.summary()
    np.testing.assert_array_equal(table.to_dict()["margin"], table.margin)

    # The table has the other minimum where it flags the epoch, and nowhere else. At these
    # two epochs it is the injected pair, with errors and a correlation like those of the
    # minimum returned, since the two lie on one valley.
    other = np.asarray(GRID.velocity_to_pixels(table.alternative))
    assert np.all(np.isfinite(table.alternative[:, wrong]))
    assert np.isnan(table.alternative[:, ~wrong]).all()
    assert np.isnan(table.alternative_covariance[~wrong]).all()
    other_sigma = table.alternative_sigma / GRID.dv_kms
    assert np.all(np.abs(other - shifts.T)[:, wrong] < 3.0 * other_sigma[:, wrong])
    np.testing.assert_allclose(table.alternative_sigma[:, wrong], table.sigma[:, wrong], rtol=0.1)
    variance = np.diagonal(table.alternative_covariance, axis1=1, axis2=2).T
    np.testing.assert_allclose(table.alternative_sigma[:, wrong], np.sqrt(variance[:, wrong]))
    c = table.alternative_covariance[wrong]
    assert np.all(c[:, 0, 1] / np.sqrt(c[:, 0, 0] * c[:, 1, 1]) < -0.5)
    columns = table.to_dict()
    np.testing.assert_array_equal(columns["alt_A"], table.alternative[0])
    np.testing.assert_array_equal(columns["alt_sigma_B"], table.alternative_sigma[1])

    # The threshold is in units of the noise level the quoted errors use. With the variances
    # declared four times too large the rises are a quarter, and so is the reduced
    # chi-square, so the profiled flag is unchanged. Taking the weights as given flags two
    # more epochs, whose rises of 4.5 and 8.1 are 18 and 32 on the scale of the residuals.
    loose = Dataset(
        [EpochData(wave=e.wave, flux=e.flux, ivar=0.25 * e.ivar, bjd=e.bjd) for e in dataset],
        frame="barycentric",
    )
    profiled = todcor(loose, templates, **BLEND_SEARCH)
    np.testing.assert_allclose(profiled.margin, 0.25 * table.margin, rtol=1e-9)
    np.testing.assert_array_equal(profiled.second_minimum, wrong)
    np.testing.assert_allclose(profiled.alternative, table.alternative, atol=1e-6)
    np.testing.assert_allclose(
        profiled.alternative_covariance, table.alternative_covariance, rtol=1e-6
    )
    as_given = todcor(loose, templates, errors="ivar", **BLEND_SEARCH)
    np.testing.assert_array_equal(as_given.second_minimum, as_given.margin < 9.0)
    assert np.flatnonzero(as_given.second_minimum).tolist() == [3, 5, 7, 15]
    np.testing.assert_array_equal(as_given.blended, as_given.second_minimum)
    np.testing.assert_array_equal(
        np.all(np.isfinite(as_given.alternative), axis=0), as_given.second_minimum
    )


def test_the_same_pair_in_the_other_order_is_not_a_second_minimum():
    """Two identical spectra at equal light: the interchanged minimum is not counted.

    The surface is symmetric under the exchange of the two shifts, so every epoch has a
    second minimum exactly as deep, with the two velocities interchanged. It gives the same
    pair of velocities, so the margin is infinite, no epoch is flagged, and the pair is
    measured at every epoch, in either order.
    """
    shifts = _blend_shifts()
    equal = (0.5, 0.5)
    dataset, templates, _ = _blend_dataset(shifts, seed=21, noise=0.04, light=equal)
    table = todcor(dataset, templates, **{**BLEND_SEARCH, "light": equal})
    assert np.all(np.isinf(table.margin)) and np.isnan(table.alternative).all()
    assert not table.second_minimum.any() and not table.blended.any() and table.good.all()
    measured = np.asarray(GRID.velocity_to_pixels(table.velocity))
    sigma = table.sigma / GRID.dv_kms
    assert np.all(np.abs(np.sort(measured, axis=0) - np.sort(shifts.T, axis=0)) < 5.0 * sigma)
    interchanged = np.sign(measured[0] - measured[1]) != np.sign(shifts[:, 0] - shifts[:, 1])
    assert 0 < interchanged.sum() < dataset.n_epochs


def test_the_margin_counts_the_minima_at_which_the_velocities_differ():
    """``_margin`` on hand-made minima: the order of the components is free.

    It returns the rise and the place of that minimum in the list it was given.
    """
    from albireo.todcor import _margin

    velocity, sigma = np.array([10.0, -20.0]), np.array([1.0, 2.0])
    returned = (100.0, velocity)
    assert _margin(100.0, velocity, sigma, [returned]) == (np.inf, None)
    # Within three quoted errors, as labelled and with the two velocities interchanged.
    close = (101.0, np.array([12.9, -14.1]))
    interchanged = (100.5, np.array([-17.5, 12.5]))
    assert _margin(100.0, velocity, sigma, [returned, close, interchanged]) == (np.inf, None)
    # One velocity more than three of its errors away in both orders: a different pair.
    other = (104.0, np.array([13.5, -20.0]))
    far = (102.5, np.array([-40.0, 60.0]))
    assert _margin(100.0, velocity, sigma, [returned, close, other]) == (pytest.approx(4.0), 2)
    minima = [returned, other, interchanged, far]
    assert _margin(100.0, velocity, sigma, minima) == (pytest.approx(2.5), 3)
    # The tolerance is each component's own error: 6.5 is beyond three errors of 2, 5.5 is not.
    beyond, within = (103.0, np.array([10.0, -26.5])), (103.0, np.array([10.0, -25.5]))
    assert _margin(100.0, velocity, sigma, [beyond]) == (pytest.approx(3.0), 0)
    assert _margin(100.0, velocity, sigma, [within]) == (np.inf, None)
    # Three components: any order of the same three velocities is the same solution.
    three, errors = np.array([0.0, 50.0, -50.0]), np.ones(3)
    cycled = (7.0, np.array([50.2, -50.1, 0.3]))
    assert _margin(0.0, three, errors, [cycled]) == (np.inf, None)
    assert _margin(0.0, three, errors, [cycled, (8.0, np.array([50.0, -50.0, 6.0]))]) == (8.0, 1)


# ---------------------------------------------------------------------------
# templates, validation, batch
# ---------------------------------------------------------------------------


def test_template_from_library_renders_at_labels_with_rotation():
    from test_library import build_library

    library = build_library()
    grid = ab.LogGrid.from_wavelength_range(5165.0, 5235.0, dv_kms=2.0)
    labels = {"teff": 4830.0, "logg": 4.1, "mh": -0.2}
    template = Template.from_library(
        "A", library, labels, grid=grid, medium="air", vsini_kms=12.0, resolving_power=20_000
    )
    interpolator = ab.library_interpolator(library.resampled_to(grid, medium="air"))
    expected = np.asarray(interpolator(np.array([4830.0, 4.1, -0.2]))[0]) - 1.0
    expected = np.convolve(
        expected, np.asarray(ab.rotational_kernel(12.0 / grid.dv_kms)), mode="same"
    )
    np.testing.assert_allclose(template.deviation, expected, atol=1e-12)
    assert template.absolute and template.v_zero_kms == 0.0
    assert template.sigma_kms == pytest.approx(ab.C_KMS / (20_000 * 2.354820045), rel=1e-6)
    assert template.meta["labels"] == labels
    with pytest.raises(ValueError, match="missing the library axes"):
        Template.from_library("A", library, {"teff": 4800.0}, grid=grid, medium="air")


@pytest.mark.parametrize("compare", ["native", "matched"])
def test_a_label_template_records_the_width_it_already_carries(compare, monkeypatch):
    """TODCOR broadens a label template only by what it lacks of the instrument profile.

    A ``"native"`` (or ``"epochs"``) match renders its template from the library at the
    library's own resolving power, so the template has the width ``sigma_lib``. A
    ``"matched"`` match convolves it further with ``sqrt(sigma_inst^2 - sigma_lib^2)``, so
    it has the declared instrument width. If ``from_labels`` records neither, TODCOR applies
    the whole instrument profile again (D65). The optimiser and the Laplace step are
    replaced by the scan's start, because the test checks the recorded width, not the fit.
    """
    from test_library import build_library

    import albireo.match as match_module
    from albireo.inference import MAPResult
    from albireo.match import StarLabels, match_labels
    from albireo.todcor import _convolved_templates, _effective_sigma

    def start_only(model, *, init, **kwargs):
        return MAPResult(
            params=dict(init),
            unconstrained={},
            potential=0.0,
            grad_norm=0.0,
            converged=True,
            num_steps=0,
        )

    monkeypatch.setattr(match_module, "run_map", start_only)
    monkeypatch.setattr(match_module, "_laplace", lambda model, result, seed: (None, ()))

    grid = ab.LogGrid.from_wavelength_range(5165.0, 5235.0, dv_kms=2.0)
    sigma_inst = 12.0
    intrinsic = build_library()
    for resolving_power in (None, 20_000.0):
        library = (
            intrinsic
            if resolving_power is None
            else intrinsic.replace(meta={**intrinsic.meta, "resolving_power": resolving_power})
        )
        sigma_lib = 0.0 if resolving_power is None else ab.C_KMS / (resolving_power * 2.354820045)
        labels = {"teff": ab.Between(4200.0, 5300.0), "logg": ab.Between(3.5, 4.5)}
        match = match_labels(
            grid,
            np.zeros((2, grid.n)),
            stars={n: StarLabels(library=library, **labels) for n in ("A", "B")},
            medium="air",
            light_fractions=[0.6, 0.4],
            lsf_sigma_kms=sigma_inst,
            compare=compare,
            mh=ab.Fixed(0.0),
            top_k=1,
        )
        template = Template.from_labels(match, "B")
        expected = sigma_inst if compare == "matched" else sigma_lib
        assert template.sigma_kms == pytest.approx(expected, rel=1e-9), (compare, resolving_power)
        assert template.meta["compare"] == compare
        assert template.meta["library_resolving_power"] == resolving_power

        # TODCOR then applies the quadrature remainder, which is zero for matched.
        remainder = _effective_sigma("a", sigma_inst, template)
        np.testing.assert_allclose(remainder, [np.sqrt(sigma_inst**2 - expected**2)], atol=1e-6)
        rows, _ = _convolved_templates([template], grid, "a", {"a": sigma_inst}, None)
        if compare == "matched":
            np.testing.assert_array_equal(rows[0], template.deviation)
        else:
            kernel = ab.gaussian_kernel(float(remainder[0]) / grid.dv_kms)
            np.testing.assert_allclose(
                rows[0],
                np.convolve(template.deviation, np.asarray(kernel), mode="same"),
                atol=1e-12,
            )


def test_template_and_argument_validation(sb2):
    dataset, _, templates = sb2
    c1, c2 = components()
    with pytest.raises(ValueError, match="shape"):
        Template("A", GRID, c1[:-1])
    other = ab.LogGrid.from_wavelength_range(5000.0, 5060.0, dv_kms=2.0)
    with pytest.raises(ValueError, match="different grids"):
        todcor(
            dataset,
            [
                templates[0],
                Template("B", other, c2[: other.n] if other.n <= c2.size else np.zeros(other.n)),
            ],
        )
    with pytest.raises(ValueError, match="distinct"):
        todcor(dataset, [templates[0], Template("A", GRID, c2)])
    with pytest.raises(ValueError, match="sum to 1"):
        todcor(dataset, templates, light=(0.6, 0.6), **COMMON)
    with pytest.raises(ValueError, match="v_range"):
        todcor(dataset, templates, v_range=(10.0, -10.0), lsf_sigma_v={"a": 5.0})
    with pytest.raises(ValueError, match="errors must be"):
        todcor(dataset, templates, errors="both", **COMMON)
    with pytest.raises(ValueError, match="scale must be"):
        todcor(dataset, templates, scale="maybe", **COMMON)
    with pytest.raises(ValueError, match="no LSF width"):
        todcor(dataset, templates, v_range=(-150.0, 150.0), lsf_sigma_v={"b": 5.0})
    with pytest.raises(ValueError, match="two templates"):
        todcor_surface(dataset, 0, templates[:1], **COMMON)


def test_a_template_broader_than_the_instrument_warns_and_is_used_unbroadened(sb2):
    dataset, _, templates = sb2
    broad = [Template("A", GRID, t.deviation, sigma_kms=5.0, v_zero_kms=0.0) for t in templates[:1]]
    broad.append(Template("B", GRID, templates[1].deviation, sigma_kms=8.0, v_zero_kms=0.0))
    with pytest.warns(UserWarning, match="broader"):
        table = todcor(dataset, broad, light=LIGHT, **COMMON)
    assert np.all(np.isfinite(table.velocity))


def test_a_narrow_grid_warns_about_its_margins(sb2):
    dataset, _, _ = sb2
    tight = ab.LogGrid.from_wavelength_range(5007.5, 5052.5, dv_kms=1.5)
    c1, c2 = components(grid=tight)
    templates = [Template("A", tight, c1), Template("B", tight, c2)]
    with pytest.warns(UserWarning, match="too narrow"):
        todcor(dataset, templates, light=LIGHT, **COMMON)


def test_batch_records_failures_and_writes_one_table_per_star(sb2, tmp_path):
    dataset, _, templates = sb2
    other = ab.LogGrid.from_wavelength_range(5000.0, 5060.0, dv_kms=2.0)
    broken = [Template("A", other, np.zeros(other.n)), Template("B", GRID, templates[1].deviation)]
    batch = todcor_batch(
        {"s1": dataset, "s2": dataset, "s3": dataset},
        {"s1": templates, "s2": templates, "s3": broken},
        progress=False,
        light=LIGHT,
        **COMMON,
    )
    assert set(batch.tables) == {"s1", "s2"}
    assert set(batch.failures) == {"s3"}
    assert "different grids" in batch.failures["s3"]
    written = batch.write(tmp_path / "rv")
    assert {p.name for p in written} == {"s1.rv", "s2.rv"}
    assert (tmp_path / "rv" / "failures.txt").read_text(encoding="utf-8").startswith("s3:")
    assert "FAILED" in batch.summary()
    with pytest.raises(ValueError, match="different grids"):
        todcor_batch(
            {"s3": dataset}, broken, on_error="raise", progress=False, light=LIGHT, **COMMON
        )


def test_velocity_table_is_a_plain_dataclass_of_arrays(fixed_table):
    assert isinstance(fixed_table, VelocityTable)
    assert fixed_table.n_epochs == 8 and fixed_table.n_components == 2
    assert fixed_table.covariance.shape == (8, 2, 2)
    np.testing.assert_allclose(
        np.sqrt(np.diagonal(fixed_table.covariance, axis1=1, axis2=2)).T,
        fixed_table.sigma,
        rtol=1e-10,
    )
    # A table built without a margin has none: no second minimum is declared.
    assert fixed_table.margin.shape == (8,) and not np.any(np.isnan(fixed_table.margin))
    assert fixed_table.alternative.shape == (2, 8)
    assert fixed_table.alternative_covariance.shape == (8, 2, 2)
    fields = {
        name: getattr(fixed_table, name)
        for name in fixed_table.__dataclass_fields__
        if name not in ("margin", "alternative", "alternative_covariance")
    }
    by_hand = VelocityTable(**fields)
    assert np.all(np.isinf(by_hand.margin)) and not by_hand.second_minimum.any()
    assert by_hand.alternative.shape == (2, 8) and np.isnan(by_hand.alternative).all()
    assert np.isnan(by_hand.alternative_covariance).all()
    assert np.isnan(by_hand.alternative_sigma).all()
    np.testing.assert_array_equal(by_hand.good, fixed_table.good)


# ---------------------------------------------------------------------------
# the loop through the Disentangler interface: disentangle, then measure against the components
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_fit_templates_close_the_loop_on_the_packaged_example():
    dataset, truth = ab.load_example("sb2_sim", with_truth=True)
    dis = ab.Disentangler(
        dataset,
        components=[
            ab.Star("primary", light=float(truth["light_fractions"][0])),
            ab.Star("secondary", light=float(truth["light_fractions"][1])),
        ],
        orbit=ab.Orbit(period=ab.Between(5.5, 6.5), k=ab.Between([10.0, 10.0], [90.0, 90.0])),
        lsf={"DEMO": 6.5},
    )
    fit = dis.fit(max_steps=150)
    templates = fit.templates()
    assert [t.name for t in templates] == ["primary", "secondary"]
    assert all(not t.absolute for t in templates)
    table = fit.measure_velocities()
    assert table.velocity.shape == (2, 12)
    # Differential velocities: compare after removing each component's own zero point.
    for i in range(2):
        residual = (table.velocity[i] - truth["velocities"][i]) - np.mean(
            table.velocity[i] - truth["velocities"][i]
        )
        assert np.sqrt(np.mean(residual**2)) < 0.5, residual

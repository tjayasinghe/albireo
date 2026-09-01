"""Tests for the TODCOR mode: per-epoch velocities by N-dimensional correlation.

Three kinds of claim are pinned here.

1. **The estimator is Zucker & Mazeh's.** On a uniform grid with uniform weights the
   weighted-least-squares surface albireo evaluates *is* the two-dimensional correlation
   ``R(s_1, s_2)`` — the symmetric expression with the light ratio maximized out, and the
   original one with the ratio held — to 1e-10, against an independent NumPy transcription
   of the published formulae.
2. **The closed loop recovers the injected velocities with calibrated errors.** Simulated
   SB2s through the real operator stack (LSF, rebin, cosmics, gaps, barycentric motion),
   in both frames, with mixed instruments, one to three components.
3. **The diagnostics fire when they should**: blending, the search edge, the unidentified
   zero point of a disentangled template, the continuum offset the nuisance absorbs.

Everything is offline and generated in-test.
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
    """``out[p] = t[p - n]`` with zero fill — the integer shift operator."""
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
    """With the fractions held exactly, chi2 = |z|^2 - 2 l.b + l.G.l — no scale freedom."""
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


def test_free_scale_reports_the_normalization_and_the_same_velocities(sb2, fixed_table):
    dataset, _, templates = sb2
    table = todcor(dataset, templates, light=LIGHT, scale="free", **COMMON)
    # The composite's scale comes back as the sum of the light row, close to one.
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
    # The lower-resolution, noisier instrument must carry larger errors.
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
    # The constant is what the nuisance absorbs, so the velocities are unaffected.
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
    # Star A never goes below -20 km/s in this orbit, so cutting the range there hits an edge.
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


# ---------------------------------------------------------------------------
# the loop through the façade: disentangle, then measure against the components
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

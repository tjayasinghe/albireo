"""Writers for the atmosphere codes downstream (roadmap Tier 2 item 8).

These tests are format transcriptions, not behaviour checks, and they are written that way
on purpose: every assertion here corresponds to a sentence in GSSP's or iSpec's own
documentation, because the failure mode of an adapter is not a crash but a file the other
code reads *differently* than intended. The two that would cost a user the most are pinned
hardest — iSpec's wavelengths are nanometres (an Angstrom value lands a factor of ten off
every model grid and still fits *something*), and GSSP infers its synthetic step from the
observation's spacing, so a log-wavelength grid must be resampled rather than dumped.
"""

from __future__ import annotations

import numpy as np
import pytest

import albireo as ab
from albireo.handoff import export_draws, write_gssp, write_ispec

GRID = ab.LogGrid.from_wavelength_range(4400.0, 4420.0, dv_kms=4.0)


def _spectra(n_comp=2, seed=3):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.02, (n_comp, GRID.n))


# ----------------------------------------------------------------- GSSP


def test_gssp_writes_two_columns_and_nothing_else(tmp_path):
    """Appendix B.2: "a two-column ASCII file". No header, no error, no third column."""
    paths = write_gssp(tmp_path / "c.dat", GRID, _spectra())
    assert len(paths) == 2
    text = paths[0].read_text()
    assert not text.lstrip().startswith("#"), "GSSP's reader takes numbers, not a comment header"
    rows = [ln.split() for ln in text.strip().splitlines()]
    assert {len(r) for r in rows} == {2}, "exactly two columns"


def test_gssp_grid_is_equidistant_even_though_albireos_is_not(tmp_path):
    """The load-bearing one.

    GSSP: "the step width in wavelength that will be used for the calculation of synthetic
    spectra is computed from the observations", so an equidistant scale is required. A
    LogGrid is equidistant in *log* wavelength, so its linear spacing drifts across the
    window and GSSP would take the first pair as the step for the whole spectrum.
    """
    source = np.asarray(GRID.wave)
    src_steps = np.diff(source)
    src_drift = src_steps.max() / src_steps.min() - 1.0
    assert src_drift > 1e-3, f"the source grid really is non-uniform ({src_drift:.2%})"

    path = write_gssp(tmp_path / "one.dat", GRID, _spectra(n_comp=1))
    steps = np.diff(np.loadtxt(path)[:, 0])
    # Equidistant to the precision the file is written at, and no better: `%.8f` rounds
    # each wavelength independently, so consecutive differences jitter by up to ~2e-8 A.
    # That is 0.7 mm/s at 4400 A — four orders below anything albireo models — whereas the
    # source grid's own drift is a third of a per-cent. The comparison of those two numbers
    # is the point of the resampling.
    assert np.ptp(steps) < 3e-8, np.ptp(steps)
    assert (np.ptp(steps) / steps.mean()) < src_drift / 1000.0


def test_gssp_wavelengths_stay_in_angstrom(tmp_path):
    """GSSP wants Angstrom; iSpec wants nm. Writing one for the other is the whole risk."""
    path = write_gssp(tmp_path / "a.dat", GRID, _spectra(n_comp=1))
    wave = np.loadtxt(path)[:, 0]
    assert 4400.0 <= wave[0] <= 4420.0, wave[0]


def test_gssp_writes_flux_not_deviation(tmp_path):
    """``1 + d``: an atmosphere code expects a normalized spectrum near unity."""
    d = np.full((1, GRID.n), -0.3)
    path = write_gssp(tmp_path / "f.dat", GRID, d)
    flux = np.loadtxt(path)[:, 1]
    np.testing.assert_allclose(flux, 0.7, atol=1e-6)


def test_gssp_step_can_be_set_and_is_honoured(tmp_path):
    path = write_gssp(tmp_path / "s.dat", GRID, _spectra(n_comp=1), step_angstrom=0.05)
    wave = np.loadtxt(path)[:, 0]
    np.testing.assert_allclose(np.diff(wave), 0.05, rtol=1e-9)


def test_gssp_refuses_a_step_that_leaves_no_spectrum(tmp_path):
    with pytest.raises(ValueError, match="would not be a spectrum"):
        write_gssp(tmp_path / "bad.dat", GRID, _spectra(n_comp=1), step_angstrom=500.0)


# ----------------------------------------------------------------- iSpec


def test_ispec_wavelengths_are_nanometres(tmp_path):
    """The single most likely silent failure in the module.

    iSpec does no unit conversion on the text path and its whole internal scale, line lists
    included, is nm. 4410 A written as 4410 lands ten times outside every model grid.
    """
    path = write_ispec(tmp_path / "i.txt", GRID, _spectra(n_comp=1))
    first = path.read_text().splitlines()[1].split("\t")
    assert 440.0 <= float(first[0]) <= 442.0, f"expected nm, got {first[0]}"


def test_ispec_header_and_three_tab_separated_columns(tmp_path):
    """iSpec drops line 1 positionally and fixes the column order via its dtype."""
    path = write_ispec(tmp_path / "i.txt", GRID, _spectra(n_comp=1))
    lines = path.read_text().splitlines()
    assert lines[0] == "waveobs\tflux\terr"
    assert all(len(ln.split("\t")) == 3 for ln in lines[1:])
    assert len(lines) == GRID.n + 1, "one header line plus one row per pixel"


def test_ispec_file_has_no_trailing_newline(tmp_path):
    """A final empty line splits to a 1-tuple, which drops the file into a legacy parser
    that mangles it silently rather than refusing it."""
    path = write_ispec(tmp_path / "i.txt", GRID, _spectra(n_comp=1))
    assert not path.read_text().endswith("\n")


def test_ispec_error_column_is_written_even_without_a_band(tmp_path):
    """There is no two-column mode: the dtype has three fields."""
    path = write_ispec(tmp_path / "i.txt", GRID, _spectra(n_comp=1))
    err = np.array([float(ln.split("\t")[2]) for ln in path.read_text().splitlines()[1:]])
    assert np.all(err > 0.0)


def test_ispec_floors_the_error_because_ispec_deletes_nonpositive_pixels(tmp_path):
    """iSpec discards ``err <= 0`` rather than down-weighting, so a posterior sd that has
    relaxed to zero would remove exactly those pixels from the fit."""
    d = _spectra(n_comp=1)
    std = np.zeros_like(d)
    std[0, 5] = 0.01
    path = write_ispec(tmp_path / "i.txt", GRID, d, std)
    err = np.array([float(ln.split("\t")[2]) for ln in path.read_text().splitlines()[1:]])
    assert np.all(err > 0.0), "no pixel may be written with a zero error"
    np.testing.assert_allclose(err[5], 0.01, rtol=1e-6)


def test_ispec_error_is_absolute_not_relative(tmp_path):
    """iSpec's S/N is ``flux / err``, so ``err`` is a 1-sigma in flux units."""
    d = np.zeros((1, GRID.n))
    std = np.full((1, GRID.n), 0.004)
    path = write_ispec(tmp_path / "i.txt", GRID, d, std)
    rows = [ln.split("\t") for ln in path.read_text().splitlines()[1:]]
    np.testing.assert_allclose([float(r[2]) for r in rows], 0.004, rtol=1e-6)
    np.testing.assert_allclose([float(r[1]) for r in rows], 1.0, rtol=1e-6)


def test_ispec_grid_is_not_resampled(tmp_path):
    """Unlike GSSP, iSpec imposes no equidistance, so resampling would correlate the noise
    for nothing."""
    path = write_ispec(tmp_path / "i.txt", GRID, _spectra(n_comp=1))
    wave = np.array([float(ln.split("\t")[0]) for ln in path.read_text().splitlines()[1:]])
    np.testing.assert_allclose(wave, np.asarray(GRID.wave) / 10.0, rtol=1e-9)


# ----------------------------------------------------------------- draws


def test_export_draws_keeps_the_draw_index_paired_across_components(tmp_path):
    """The jointness is the product. Draw i of A and draw i of B are one posterior sample,
    so they must stay identifiable as a pair."""
    rng = np.random.default_rng(0)
    draws = rng.normal(0.0, 0.02, (5, 2, GRID.n))
    out = export_draws(tmp_path, GRID, draws, format="gssp")
    assert len(out) == 5 and all(len(p) == 2 for p in out)
    assert out[3][0].name == "draw_0003_1.dat"
    assert out[3][1].name == "draw_0003_2.dat"


def test_export_draws_writes_distinct_spectra(tmp_path):
    """A silently-constant export would pass every format check and report zero spread."""
    rng = np.random.default_rng(1)
    draws = rng.normal(0.0, 0.05, (3, 1, GRID.n))
    out = export_draws(tmp_path, GRID, draws, format="ispec")
    fluxes = [
        np.array([float(ln.split("\t")[1]) for ln in p[0].read_text().splitlines()[1:]])
        for p in out
    ]
    assert not np.allclose(fluxes[0], fluxes[1])
    assert not np.allclose(fluxes[1], fluxes[2])


def test_export_draws_resamples_every_draw_onto_the_same_grid(tmp_path):
    """GSSP's step comes from the file, so draws that disagreed on the grid would be fitted
    against different synthetic samplings and the spread would carry that."""
    rng = np.random.default_rng(2)
    draws = rng.normal(0.0, 0.02, (4, 1, GRID.n))
    out = export_draws(tmp_path, GRID, draws, format="gssp")
    waves = [np.loadtxt(p[0])[:, 0] for p in out]
    for w in waves[1:]:
        np.testing.assert_array_equal(w, waves[0])


def test_export_draws_rejects_the_wrong_shape(tmp_path):
    with pytest.raises(ValueError, match="n_draws, n_comp, n_pix"):
        export_draws(tmp_path, GRID, np.zeros((2, GRID.n)))


def test_export_draws_rejects_an_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="format must be one of"):
        export_draws(tmp_path, GRID, np.zeros((2, 1, GRID.n)), format="korg")


def test_draws_from_a_real_fit_round_trip_to_disk(tmp_path):
    """End to end on the packaged example: joint draws in, fittable files out."""
    import jax

    from albireo.likelihood import draw_spectra, marginal_loglikelihood

    dataset = ab.load_example()
    grid = ab.LogGrid.covering(dataset, dv_kms=6.0, v_margin_kms=140.0)
    problem = ab.build_problem(
        grid,
        dataset,
        velocities=np.zeros((2, len(list(dataset)))),
        light_fractions=[0.6, 0.4],
        lsf_sigma_v={name: 6.0 for name in dataset.instruments},
    )
    result = marginal_loglikelihood(problem, ab.SmoothnessPrior(tau=[200.0, 200.0], eta=[5.0, 5.0]))
    draws = np.asarray(draw_spectra(result, jax.random.key(0), 4))
    assert draws.shape == (4, 2, grid.n)

    out = export_draws(tmp_path, grid, draws, format="ispec")
    assert len(out) == 4
    wave = np.array([float(ln.split("\t")[0]) for ln in out[0][0].read_text().splitlines()[1:]])
    np.testing.assert_allclose(wave, np.asarray(grid.wave) / 10.0, rtol=1e-9)

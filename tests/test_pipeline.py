"""Tests for the pipeline driver: one declaration in, structured products out.

What is pinned here is the *driver*, not the stages it calls -- each of those has its own
closed-loop tests. Three kinds of claim:

1. **The declaration is honest.** Light fractions are required and must sum to one, the
   wavelength medium is never guessed, unknown settings are refused by name, and a TOML
   file round-trips through the loader with every path resolved against its own
   directory.
2. **The products are structured and complete.** One star run writes the velocity table
   (twice: the commented ASCII and a CSV), the component spectra with their bands, the
   orbit, the labels, ``result.json`` with the keys a survey table needs, and the figures;
   a batch writes ``results.csv`` with one row per star and records a failure without
   stopping.
3. **The loop closes.** A star whose components are drawn from a toy library at known
   labels comes back with those labels, with the velocities *absolute* because the label
   fit pinned each component's zero point, and with the systemic velocity -- which the
   disentangling alone can never see -- recovered by the orbit fitted to the table.

Everything is offline and generated in-test; the worker-process test spawns two real
processes, each importing JAX, which is the point of it.
"""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pytest

import albireo as ab
from albireo.pipeline import (
    Analysis,
    ComponentConfig,
    PipelineConfig,
    StarConfig,
    _with_medium,
    config_from_dict,
    demo_config,
    load_config,
    run_pipeline,
    run_star,
    write_config_template,
)
from albireo.simulate import (
    InstrumentSpec,
    OrbitParams,
    library_component,
    simulate_dataset,
    synthetic_library,
)

HAS_MPL = importlib.util.find_spec("matplotlib") is not None

TRUE_LABELS = {
    "A": {"teff": 5180.0, "logg": 4.05, "mh": -0.15, "vsini": 11.0},
    "B": {"teff": 4460.0, "logg": 4.55, "mh": -0.15, "vsini": 27.0},
}
LIGHT = (0.62, 0.38)
GAMMA = 12.0
ORBIT = OrbitParams(period=6.31, t_peri=2.0, ecc=0.15, omega=0.7, k=(30.0, 55.0), gamma=GAMMA)


@pytest.fixture(scope="module")
def library():
    return synthetic_library((5140.0, 5230.0), n_pix=900)


def _simulate(library, *, n_epochs=8, snr=120.0, seed=7):
    """An SB2 whose components are library spectra at TRUE_LABELS, with a +12 km/s gamma."""
    grid = ab.LogGrid.from_wavelength_range(5150.0, 5220.0, dv_kms=2.0)
    components = [
        library_component(
            library,
            {k: v for k, v in labels.items() if k != "vsini"},
            grid,
            medium="air",
            vsini_kms=labels["vsini"],
        )
        for labels in TRUE_LABELS.values()
    ]
    bjd = np.sort(np.random.default_rng(3).uniform(0.0, 21.0, size=n_epochs))
    dataset, truth = simulate_dataset(
        grid,
        components,
        bjd=bjd,
        instruments={
            "TOY": InstrumentSpec(wave=np.arange(5156.0, 5214.0, 0.08), sigma_v_lsf=5.5, snr=snr)
        },
        light_fractions=LIGHT,
        orbit=ORBIT,
        seed=seed,
    )
    return _with_medium(dataset, "air"), truth, grid


def _star(dataset, truth, grid, name="toy", **kwargs):
    options = {
        "period": (6.0, 6.6),
        "components": [
            ComponentConfig(
                "A", LIGHT[0], teff=(4200.0, 5700.0), logg=(3.2, 4.9), vsini=(1.0, 60.0)
            ),
            ComponentConfig(
                "B", LIGHT[1], teff=(4100.0, 5200.0), logg=(3.2, 4.9), vsini=(1.0, 60.0)
            ),
        ],
        "lsf": {"TOY": 5.5},
        "truth": {
            "k": list(ORBIT.k),
            "period": ORBIT.period,
            "gamma": GAMMA,
            "velocities": np.asarray(truth.velocities),
            "components": np.asarray(truth.components),
            "grid": grid,
            "labels": TRUE_LABELS,
        },
        "overrides": {"k_max": 90.0},
    }
    options.update(kwargs)
    return StarConfig(name=name, dataset=dataset, **options)


@pytest.fixture(scope="module")
def toy(library):
    return _simulate(library)


# ---------------------------------------------------------------------------
# 1. the declaration
# ---------------------------------------------------------------------------


def test_light_fractions_are_required_and_must_sum_to_one(toy):
    dataset, truth, grid = toy
    with pytest.raises(TypeError):
        ComponentConfig("A")  # no light
    with pytest.raises(ValueError, match="sum to 1"):
        _star(
            dataset, truth, grid, components=[ComponentConfig("A", 0.7), ComponentConfig("B", 0.7)]
        )


def test_exactly_one_orbit_declaration(toy):
    dataset, truth, grid = toy
    with pytest.raises(ValueError, match="exactly one of period= and velocities="):
        _star(dataset, truth, grid, period=None)
    with pytest.raises(ValueError, match="exactly one of period= and velocities="):
        _star(dataset, truth, grid, velocities=np.zeros((2, dataset.n_epochs)))


def test_exactly_one_data_source(toy):
    dataset, truth, grid = toy
    with pytest.raises(ValueError, match="exactly one of spectra=, dataset= and bloem="):
        _star(dataset, truth, grid, spectra="nowhere/*.fits")


def test_unknown_settings_are_refused_by_name(toy):
    dataset, truth, grid = toy
    with pytest.raises(ValueError, match="unknown setting"):
        _star(dataset, truth, grid, overrides={"kmax": 90.0})
    with pytest.raises(ValueError, match="unknown \\[analysis\\] key"):
        config_from_dict({"analysis": {"steps": 3}, "stars": []})


def test_a_period_search_needs_a_library(toy):
    dataset, truth, grid = toy
    with pytest.raises(ValueError, match="library is required"):
        PipelineConfig(stars=[_star(dataset, truth, grid, period="search")])


def test_star_names_must_be_unique(toy):
    dataset, truth, grid = toy
    star = _star(dataset, truth, grid)
    with pytest.raises(ValueError, match="unique"):
        PipelineConfig(stars=[star, star])


def test_the_template_round_trips_through_the_loader(tmp_path):
    path = write_config_template(tmp_path / "albireo.toml")
    config = load_config(path)
    assert [s.name for s in config.stars] == ["AI Phe"]
    star = config.stars[0]
    # relative paths resolve against the file's own directory
    assert os.path.normcase(star.spectra).startswith(os.path.normcase(str(tmp_path)))
    assert os.path.normcase(config.output) == os.path.normcase(str(tmp_path / "albireo_results"))
    assert config.library == "bosz2024-fgk-r20000"
    assert [c.name for c in star.components] == ["primary", "secondary"]
    assert sum(c.light for c in star.components) == pytest.approx(1.0)
    assert config.analysis.region == (5000.0, 5300.0)
    with pytest.raises(FileExistsError):
        write_config_template(path)


def test_per_star_overrides_and_fast_mode(toy):
    dataset, truth, grid = toy
    star = _star(dataset, truth, grid, overrides={"max_steps": 500, "circular": True})
    settings = star.settings(Analysis(fast=True))
    assert settings.circular is True
    assert settings.max_steps == 40, "fast mode trims the optimizer budget"
    assert star.settings(Analysis()).max_steps == 500


def test_bad_toml_is_reported_as_a_value_error(tmp_path):
    path = tmp_path / "broken.toml"
    path.write_text("[stars\nname = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid TOML"):
        load_config(path)


# ---------------------------------------------------------------------------
# 2. the products
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def quick_run(toy, library, tmp_path_factory):
    """One fast in-process run with the label stage on: the products fixture."""
    dataset, truth, grid = toy
    star = _star(dataset, truth, grid)
    config = PipelineConfig(
        stars=[star],
        output=tmp_path_factory.mktemp("quick"),
        library=library,
        mh=(-0.9, 0.4),
        analysis=Analysis(fast=True, v_zero_range=40.0, plots=HAS_MPL),
    )
    return run_star(star, config, progress=False)


def test_a_run_writes_every_product(quick_run):
    result = quick_run
    assert result.ok and result.live is not None
    for kind in (
        "velocities",
        "velocities_csv",
        "spectrum_A",
        "spectrum_B",
        "fit",
        "orbit",
        "labels",
        "template_A",
        "report",
        "summary",
        "log",
    ):
        assert kind in result.files, kind
        assert os.path.isfile(result.files[kind]), result.files[kind]
    if HAS_MPL:
        for figure in ("spectra", "residuals", "velocities", "phase_scan", "todcor_surface"):
            assert os.path.isfile(result.files[f"plot_{figure}"])


def test_the_report_is_json_and_carries_the_survey_columns(quick_run):
    with open(quick_run.files["report"], encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["status"] == "ok"
    assert set(report) >= {
        "dataset",
        "declaration",
        "disentangling",
        "labels",
        "velocities",
        "orbit",
        "truth",
        "seconds",
        "flags",
        "files",
    }
    assert report["velocities"]["absolute"] == {"A": True, "B": True}, (
        "the label fit measured each component's frame offset, so the table is absolute"
    )
    assert report["orbit"]["gamma_mode"] == "shared"
    assert set(report["orbit"]["k"]) == {"A", "B"}
    assert report["labels"]["components"]["A"]["teff_err"] > 0.0
    assert "total" in report["seconds"] and report["seconds"]["disentangle"] > 0.0


def test_the_velocity_tables_agree_with_each_other(quick_run):
    table = quick_run.live["velocities"]
    rows = np.loadtxt(quick_run.files["velocities"], comments="#", usecols=(0, 2, 4))
    np.testing.assert_allclose(rows[:, 1], table.velocity[0], atol=1e-6)
    with open(quick_run.files["velocities_csv"], encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    assert header[:4] == ["bjd", "instrument", "v_A", "sigma_A"]


def test_the_summary_states_the_zero_point_and_the_truth(quick_run):
    text = quick_run.summary
    assert "absolute" in text and "Against the injected truth" in text
    assert "Assumed, not measured" in text, "the assumptions block travels into the report"


def test_the_written_spectra_can_be_read_back(quick_run):
    wave, flux, err = np.loadtxt(quick_run.files["spectrum_A"], unpack=True)
    fit = quick_run.live["fit"]
    np.testing.assert_allclose(wave, fit.dis.grid.wave)
    np.testing.assert_allclose(flux, 1.0 + fit.spectra()[0], atol=1e-5)
    assert np.all(err > 0)


def test_a_batch_records_a_failure_without_stopping(toy, library, tmp_path):
    dataset, truth, grid = toy
    good = _star(dataset, truth, grid, name="good", labels=False)
    # A declaration the façade refuses at fit time: an eccentricity bound above the
    # solver's clip is caught when the model is built, not when the config is parsed.
    bad = _star(
        dataset, truth, grid, name="bad", labels=False, overrides={"ecc_max": 0.95, "dv_kms": -1.0}
    )
    config = PipelineConfig(
        stars=[good, bad], output=tmp_path, analysis=Analysis(fast=True, plots=False, max_steps=3)
    )
    run = run_pipeline(config, progress=False)
    assert set(run.results) == {"good", "bad"}
    assert run.results["good"].ok
    assert not run.results["bad"].ok and "bad" in run.failures
    assert (tmp_path / "failures.txt").is_file()
    assert (tmp_path / "bad" / "error.txt").is_file()
    rows = run.rows()
    assert [r["star"] for r in rows] == ["good", "bad"]
    assert rows[1]["status"] == "failed" and rows[0]["status"] == "ok"
    with open(tmp_path / "results.csv", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    assert {"star", "status", "period", "K_1", "K_2", "flags"} <= set(header)
    assert "differential" in " ".join(run.results["good"].flags)


def test_a_batch_runs_in_worker_processes(toy, tmp_path):
    dataset, truth, grid = toy
    stars = [
        _star(dataset, truth, grid, name=f"star{i}", labels=False, overrides={"max_steps": 3})
        for i in range(2)
    ]
    config = PipelineConfig(stars=stars, output=tmp_path, analysis=Analysis(fast=True, plots=False))
    run = run_pipeline(config, jobs=2, progress=False)
    assert run.jobs == 2
    assert all(r.ok for r in run.results.values()), run.failures
    assert all(r.live is None for r in run.results.values()), "live objects do not cross a pipe"
    assert list(run.results) == ["star0", "star1"], "results keep the declaration order"
    assert (tmp_path / "results.json").is_file()
    payload = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert payload["jobs"] == 2 and set(payload["stars"]) == {"star0", "star1"}


def test_selecting_stars_by_name(toy, tmp_path):
    dataset, truth, grid = toy
    stars = [
        _star(dataset, truth, grid, name=n, labels=False, overrides={"max_steps": 2})
        for n in ("one", "two")
    ]
    config = PipelineConfig(stars=stars, output=tmp_path, analysis=Analysis(fast=True, plots=False))
    run = run_pipeline(config, stars=["two"], progress=False)
    assert list(run.results) == ["two"]
    with pytest.raises(KeyError, match="unknown star"):
        run_pipeline(config, stars=["three"], progress=False)


def test_a_velocity_file_declares_the_free_table(toy, tmp_path):
    """Measured velocities instead of a period: the free per-epoch table is fitted."""
    dataset, truth, grid = toy
    v = np.asarray(truth.velocities) + np.random.default_rng(1).normal(
        0.0, 2.0, (2, dataset.n_epochs)
    )
    path = tmp_path / "rv.txt"
    np.savetxt(path, np.column_stack([dataset.bjd, v.T]), header="bjd v_A v_B")
    star = _star(dataset, truth, grid, period=None, velocities=str(path), labels=False)
    config = PipelineConfig(
        stars=[star], output=tmp_path, analysis=Analysis(fast=True, plots=False, max_steps=5)
    )
    result = run_star(star, config, progress=False)
    assert result.report["disentangling"]["mode"] == "velocity"
    assert result.report["orbit"] is not None, result.flags
    assert "periodogram" in result.report["orbit"]["period_source"]
    assert "sampling" not in " ".join(result.flags)


def test_the_demo_declares_two_known_stars():
    config = demo_config("unused", fast=True)
    assert [s.name for s in config.stars] == ["sb2_sim", "toy_library_sb2"]
    assert config.stars[0].labels is False and config.stars[1].labels is True
    assert config.stars[1].truth["gamma"] == 12.0
    assert config.library is not None


# ---------------------------------------------------------------------------
# 3. the loop closes
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_the_pipeline_recovers_the_injected_system(library, tmp_path):
    dataset, truth, grid = _simulate(library, n_epochs=10, snr=150.0)
    star = _star(dataset, truth, grid)
    config = PipelineConfig(
        stars=[star],
        output=tmp_path,
        library=library,
        mh=(-0.9, 0.4),
        analysis=Analysis(max_steps=150, label_steps=300, v_zero_range=40.0, plots=False),
    )
    result = run_star(star, config, progress=False)
    orbit = result.report["orbit"]
    for name, k_true in zip(("A", "B"), ORBIT.k, strict=True):
        assert abs(orbit["k"][name] - k_true) < 0.03 * k_true, (name, orbit["k"])
        assert abs(orbit["gamma"][name] - GAMMA) < 1.0, (
            "absolute velocities: the systemic velocity the disentangling cannot see"
        )
    assert abs(orbit["period"] - ORBIT.period) < 0.01 * ORBIT.period
    labels = result.report["labels"]["components"]
    # 5%, not the label mode's 2-3% template-selection target: what is pinned here is the
    # driver, and on components that came through a real disentangling the fainter star's
    # Teff lands about 4% off -- on this fixture as on AI Phoenicis (D55), where the
    # secondary missed by 4.3% while the primary met the target. The formal errors below
    # are 5-10x smaller than that, which is the documented behaviour of the label mode.
    for name, true in TRUE_LABELS.items():
        assert abs(labels[name]["teff"] - true["teff"]) < 0.05 * true["teff"], name
        assert abs(labels[name]["logg"] - true["logg"]) < 0.15, name
        assert abs(labels[name]["vsini"] - true["vsini"]) < 0.25 * true["vsini"], name
    assert result.report["velocities"]["absolute_all"] is True


@pytest.mark.slow
def test_a_period_search_bootstraps_from_library_templates(library, tmp_path):
    """No period declared: library templates measure a first table, and the orbit follows."""
    dataset, truth, grid = _simulate(library, n_epochs=10, snr=150.0)
    star = _star(dataset, truth, grid, period="search")
    config = PipelineConfig(
        stars=[star],
        output=tmp_path,
        library=library,
        mh=(-0.9, 0.4),
        analysis=Analysis(
            max_steps=120, label_steps=200, v_zero_range=40.0, plots=False, v_range=150.0
        ),
    )
    result = run_star(star, config, progress=False)
    bootstrap = result.report["bootstrap"]
    assert abs(bootstrap["period"] - ORBIT.period) < 0.02 * ORBIT.period, bootstrap
    orbit = result.report["orbit"]
    # The bootstrap warm-starts a Keplerian disentangling, so the final table's orbit
    # starts from the disentangling's period, as on any other Keplerian run.
    assert orbit["period_source"] == "the disentangling"
    assert result.report["disentangling"]["mode"] == "keplerian"
    for name, k_true in zip(("A", "B"), ORBIT.k, strict=True):
        assert abs(orbit["k"][name] - k_true) < 0.05 * k_true, (name, orbit["k"])


@pytest.mark.slow
def test_the_demo_runs_end_to_end(tmp_path):
    config = demo_config(tmp_path, fast=True)
    run = run_pipeline(config, progress=False)
    assert not run.failures, run.failures
    packaged = run.results["sb2_sim"].report
    assert packaged["velocities"]["absolute_all"] is False
    assert packaged["orbit"]["gamma_mode"] == "one per component"
    toy = run.results["toy_library_sb2"].report
    assert toy["velocities"]["absolute_all"] is True
    assert abs(toy["orbit"]["gamma"]["A"] - 12.0) < 1.5

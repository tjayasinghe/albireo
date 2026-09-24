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
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

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
            "ecc": ORBIT.ecc,
            "omega": ORBIT.omega,
            "t_conj": float(
                ab.t_conj_from_t_peri(
                    ORBIT.t_peri, period=ORBIT.period, ecc=ORBIT.ecc, omega=ORBIT.omega
                )
            ),
            "gamma": GAMMA,
            "light_fractions": list(LIGHT),
            "velocities": np.asarray(truth.velocities),
            "components": np.asarray(truth.components),
            "grid": grid,
            "labels": TRUE_LABELS,
            "windows": {"blue": [(5150.0, 5185.0)], "red": [(5185.0, 5220.0)]},
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


def test_known_elements_are_declared_and_described(toy):
    dataset, truth, grid = toy
    star = _star(
        dataset,
        truth,
        grid,
        t_conj={"value": 2457001.0, "sigma": 0.02},
        ecc=0.15,
        omega=0.7,
    )
    described = PipelineConfig(stars=[star]).to_dict()["stars"][0]
    assert described["t_conj"] == {"value": 2457001.0, "sigma": 0.02}
    assert described["ecc"] == {"fixed": 0.15}
    assert described["omega"] == {"fixed": 0.7}
    scan = _star(dataset, truth, grid)
    assert PipelineConfig(stars=[scan]).to_dict()["stars"][0]["t_conj"] == "scan"
    with pytest.raises(ValueError, match="needs omega"):
        _star(dataset, truth, grid, ecc=0.2)
    with pytest.raises(ValueError, match="Gaussian"):
        _star(dataset, truth, grid, ecc={"value": 0.2, "sigma": 0.05})
    with pytest.raises(ValueError, match="must start at 0"):
        _star(dataset, truth, grid, ecc=(0.1, 0.4))
    # A TOML round trip carries the new keys.
    data = {
        "stars": [
            {
                "name": "x",
                "spectra": "a/*.fits",
                "period": [1.0, 2.0],
                "t_conj": {"value": 1.5, "sigma": 0.1},
                "ecc": [0.0, 0.3],
                "components": [{"name": "A", "light": 0.5}, {"name": "B", "light": 0.5}],
            }
        ]
    }
    config = config_from_dict(data)
    assert config.stars[0].ecc == [0.0, 0.3]
    assert config.stars[0].t_conj == {"value": 1.5, "sigma": 0.1}


def test_measured_light_needs_a_library(toy):
    dataset, truth, grid = toy
    measured = [ComponentConfig("A", "measure"), ComponentConfig("B", "measure")]
    declared = _star(dataset, truth, grid, components=measured)
    assert declared.measures_light and not declared.searching
    with pytest.raises(ValueError, match="light = 'measure'"):
        PipelineConfig(stars=[declared])
    with pytest.raises(ValueError, match="every component or to none"):
        _star(
            dataset,
            truth,
            grid,
            period="search",
            components=[ComponentConfig("A", "measure"), ComponentConfig("B", 0.4)],
        )
    with pytest.raises(ValueError, match="light must be a fraction"):
        ComponentConfig("A", "guess")
    star = _star(dataset, truth, grid, period="search", components=measured)
    assert star.measures_light
    assert (
        PipelineConfig(stars=[star], library="bosz2024-fgk-r20000").to_dict()["stars"][0][
            "components"
        ][0]["light"]
        == "measure"
    )


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
    # The label fit's comparison, and the light it measured beside the declared one (D65).
    labels = report["labels"]
    assert labels["compare"] == "epochs"
    assert set(labels["flux_ratio_errors"]) == {"A", "B"}
    assert all(np.isfinite(v) and v >= 0.0 for v in labels["flux_ratio_errors"].values())
    assert labels["light_declared"] == {"A": LIGHT[0], "B": LIGHT[1]}
    assert labels["epoch_fit"]["lm_runs"] > 0 and isinstance(labels["notes"], list)
    assert isinstance(labels["at_bounds"], dict)
    assert report["velocities"]["light_source"] in ("declared", "global re-measure")
    # The operator the epoch comparison ran with: the declared 5.5 km/s (an intrinsic toy
    # library) less the model grid's own (7/12) dv^2 on the native 4.63 km/s pixel.
    operator = labels["epoch_operator"]
    dv = report["declaration"]["grid"]["dv_kms"]
    assert operator["dv_kms"] == pytest.approx(dv)
    assert operator["library_resolving_power"] is None
    assert operator["grid_variance_kms2"] == pytest.approx(7.0 / 12.0 * dv**2)
    assert operator["quadrature_sigma_kms"] == [["TOY", [5.5]]]
    ((instrument, (width,)),) = operator["operator_sigma_kms"]
    assert instrument == "TOY" and width == pytest.approx(np.sqrt(5.5**2 - 7.0 / 12.0 * dv**2))
    assert operator["notes"] == [], "dv = 4.63 km/s is inside sigma_q = 5.5 km/s"


def test_the_truth_block_carries_elements_pulls_and_spectra(quick_run):
    truth = quick_run.report["truth"]
    for key in (
        "k_disentangling",
        "k_table",
        "k_table_pull",
        "elements_disentangling",
        "elements_table",
        "elements_table_pull",
        "gamma",
        "gamma_pull",
        "velocity_rms",
        "velocity_pull_rms",
        "velocity_rms_keplerian",
        "labels",
        "light_declared",
        "light_label_fit",
        "spectra",
    ):
        assert key in truth, key
    assert set(truth["elements_table"]) == {"period", "ecc", "omega_deg", "t_conj"}
    # The conjunction is compared modulo the period, so the difference is a fraction of it.
    assert abs(truth["elements_table"]["t_conj"]) < 0.5 * ORBIT.period
    assert abs(truth["elements_table"]["omega_deg"]) <= 180.0
    for name in ("A", "B"):
        spectra = truth["spectra"][name]
        # A fast-mode fit (40 steps) is not converged; what is pinned is that the block
        # exists and reads sensibly, not the recovery, which the slow closed loops pin.
        assert 0.0 < spectra["rms"] < spectra["rms_truth"], spectra
        assert spectra["corr"] > 0.7, spectra
        assert spectra["pull_rms"] > 0.0
        assert set(spectra["windows"]) == {"blue", "red"}
        for window in spectra["windows"].values():
            assert window["ew_true_angstrom"] > 0.0 and np.isfinite(window["ratio"])
        assert truth["light_declared"][name] == pytest.approx(0.0)
        assert truth["labels"][name]["teff_pull"] is not None
    assert 0 < truth["velocity_n_used"]["A"] <= quick_run.report["dataset"]["n_epochs"]


def test_the_report_carries_every_epoch_velocity(quick_run):
    """The measured table, the disentangling's Keplerian and the injected velocities, per epoch."""
    report = quick_run.report
    n = report["dataset"]["n_epochs"]
    table = report["velocities"]
    assert table["names"] == ["A", "B"] and len(table["bjd"]) == n
    assert all(len(table["velocity"][c]) == n and len(table["sigma"][c]) == n for c in ("A", "B"))
    assert table["noise_correlation"] == {"TOY": 0.0}
    keplerian = report["disentangling"]["velocities"]
    assert set(keplerian) == {"A", "B"} and len(keplerian["A"]) == n
    injected = report["truth"]["epoch_velocities_true"]
    assert set(injected) == {"A", "B"} and len(injected["B"]) == n
    assert report["declaration"]["noise_model"] == {"kind": "diagonal"}
    assert report["disentangling"].get("k_scan") is not None  # the period is a range: K scanned
    assert report["disentangling"]["k_scan"]["n_trials"] > 0


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
            max_steps=120,
            label_steps=200,
            v_zero_range=40.0,
            plots=False,
            v_range=150.0,
            # Two candidates exercise the decision by the disentangling at a fraction
            # of the default's wall on this toy.
            period_decision_candidates=2,
        ),
    )
    result = run_star(star, config, progress=False)
    bootstrap = result.report["bootstrap"]
    assert abs(bootstrap["period"] - ORBIT.period) < 0.02 * ORBIT.period, bootstrap
    # The template table the bootstrap measured is a product of its own, written before
    # the disentangling, so the period decision can be examined from disk.
    template_table = Path(result.report["files"]["template_velocities"])
    assert template_table.name == "template_velocities.rv" and template_table.exists()
    assert "purpose: bootstrap" in template_table.read_text(encoding="utf-8")
    bjd = np.loadtxt(template_table, comments="#", usecols=(0,), ndmin=1)
    assert bjd.size == dataset.n_epochs
    assert Path(result.report["files"]["template_velocities_csv"]).exists()
    # The bootstrap also carries the table the period search itself ran on, which the
    # winning orbit's re-assignment would otherwise leave nowhere on disk.
    assert result.live["bootstrap"]["table_unexchanged"].n_epochs == dataset.n_epochs
    orbit = result.report["orbit"]
    # The bootstrap warm-starts a Keplerian disentangling, so the final table's orbit
    # starts from the disentangling's period, as on any other Keplerian run.
    assert orbit["period_source"] == "the disentangling"
    assert result.report["disentangling"]["mode"] == "keplerian"
    for name, k_true in zip(("A", "B"), ORBIT.k, strict=True):
        assert abs(orbit["k"][name] - k_true) < 0.05 * k_true, (name, orbit["k"])


def test_the_noise_correlation_setting_is_checked_and_declared(toy):
    """A number, a table per instrument or "fit"; anything else is refused by name."""
    from albireo.facade import Between as _Between
    from albireo.pipeline import _noise_declaration, _noise_values

    dataset, truth, grid = toy
    assert Analysis().noise_correlation is None
    with pytest.raises(ValueError, match="noise_correlation must be a number"):
        Analysis(noise_correlation="later")
    with pytest.raises(ValueError, match=r"lie in \(-1, 1\)"):
        Analysis(noise_correlation={"TOY": 1.0})
    assert _noise_declaration(Analysis(), dataset) is None
    assert _noise_declaration(Analysis(noise_correlation=0.3), dataset) == 0.3
    per = _noise_declaration(Analysis(noise_correlation={"OTHER": 0.5}), dataset)
    assert per == {"TOY": 0.0}, "an instrument left out of the table is independent"
    fitted = _noise_declaration(Analysis(noise_correlation="fit"), dataset)
    assert isinstance(fitted, _Between) and float(np.asarray(fitted.start())) == 0.0
    assert _noise_values(Analysis(noise_correlation="fit"), dataset) is None
    assert _noise_values(Analysis(noise_correlation=0.3), dataset) == {"TOY": 0.3}
    star = _star(dataset, truth, grid, overrides={"noise_correlation": {"TOY": 0.27}})
    assert star.overrides["noise_correlation"] == {"TOY": 0.27}


def test_a_free_eccentricity_starts_from_the_table_orbit_when_it_is_usable():
    from albireo.facade import Between as _Between
    from albireo.pipeline import _declared_orbit, _ecc_omega_specs, _element_starts

    star = StarConfig(
        name="starts",
        spectra=["a.fits", "b.fits"],
        period=6.31,
        components=[ComponentConfig("A", 0.6), ComponentConfig("B", 0.4)],
    )
    settings = Analysis(k_min=2.0, k_max=250.0, ecc_max=0.9)

    class _Orbit:
        def __init__(self, ecc, omega, error=0.02):
            self.ecc, self.omega = ecc, omega
            self.errors = {"ecc": error}

    class _Ctx:
        def __init__(self):
            self.lines = []

        def log(self, text):
            self.lines.append(text)

    ctx = _Ctx()
    assert _element_starts(ctx, None, star, settings) is None
    assert _element_starts(ctx, _Orbit(0.005, 1.0), star, settings) is None  # as good as default
    assert _element_starts(ctx, _Orbit(0.95, 1.0), star, settings) is None  # outside the prior
    assert _element_starts(ctx, _Orbit(float("nan"), 1.0), star, settings) is None
    assert _element_starts(ctx, _Orbit(0.4, 1.2), star, settings) == (0.4, 1.2)
    assert "started from the template table" in ctx.lines[-1]
    # An eccentricity the table has not detected is not a start: 0.2 +- 0.15 stays default.
    assert _element_starts(ctx, _Orbit(0.2, 1.2, error=0.15), star, settings) is None
    assert "not detected at three sigma" in ctx.lines[-1]
    assert _element_starts(ctx, _Orbit(0.2, 1.2, error=float("nan")), star, settings) is None
    ecc, omega = _ecc_omega_specs(star, settings, (0.4, 1.2))
    assert isinstance(ecc, _Between) and float(np.asarray(ecc.start())) == 0.4
    assert float(np.asarray(omega.start())) == 1.2
    ecc, omega = _ecc_omega_specs(star, settings, None)
    assert isinstance(ecc, _Between) and ecc.start_at is None and omega is None
    # A declared eccentricity is never overridden by the table.
    held = replace(star, ecc=0.1, omega=0.3)
    ecc, omega = _ecc_omega_specs(held, settings, (0.4, 1.2))
    assert float(np.asarray(ecc.value)) == 0.1 and float(np.asarray(omega.value)) == 0.3
    assert _element_starts(ctx, _Orbit(0.4, 1.2), held, settings) is None
    circular = Analysis(circular=True)
    assert _element_starts(ctx, _Orbit(0.4, 1.2), star, circular) is None
    orbit = _declared_orbit(star, settings, starts=[23.0, 34.0], elements=(0.4, 1.2))
    assert float(np.asarray(orbit.ecc.start())) == 0.4


def test_an_exchanged_pair_is_recognised_and_the_truth_reordered():
    """Velocities that match the truth in the other order are judged in that order."""
    from albireo.pipeline import _exchange_rms, _exchanged_truth

    phase = np.linspace(0.0, 1.0, 12, endpoint=False)
    v_true = np.stack([30.0 * np.sin(2 * np.pi * phase), -55.0 * np.sin(2 * np.pi * phase)])
    good = np.ones(12, dtype=bool)
    named, exchanged = _exchange_rms(v_true + 0.1, v_true, good, (True, True))
    assert named == pytest.approx(0.1) and exchanged > 10.0 * named
    named, exchanged = _exchange_rms(v_true[::-1], v_true, good, (True, True))
    assert exchanged == pytest.approx(0.0, abs=1e-12) and named > 30.0
    # A differential table is compared after removing each zero point.
    named, _ = _exchange_rms(v_true + np.array([[7.0], [-3.0]]), v_true, good, (False, False))
    assert named == pytest.approx(0.0, abs=1e-12)
    truth = {
        "k": [30.0, 55.0],
        "light_fractions": [0.6, 0.4],
        "velocities": v_true,
        "components": np.array([[1.0, 0.9], [1.0, 0.8]]),
        "labels": {"A": {"teff": 6000.0}, "B": {"teff": 5000.0}},
        "period": 6.31,
    }
    swapped = _exchanged_truth(truth, ["A", "B"])
    assert swapped["k"] == [55.0, 30.0] and swapped["light_fractions"] == [0.4, 0.6]
    np.testing.assert_array_equal(swapped["velocities"], v_true[::-1])
    np.testing.assert_array_equal(swapped["components"][0], [1.0, 0.8])
    assert swapped["labels"] == {"A": {"teff": 5000.0}, "B": {"teff": 6000.0}}
    assert swapped["period"] == 6.31 and truth["k"] == [30.0, 55.0]


def test_semi_amplitude_starts_follow_the_table_and_fill_in_the_rest():
    """Measured starts are taken as they are; a missing one follows its nearest neighbour."""
    from albireo.facade import Between
    from albireo.pipeline import _k_prior

    star = StarConfig(
        name="starts",
        spectra=["a.fits", "b.fits"],
        period=6.31,
        components=[ComponentConfig("A", 0.6), ComponentConfig("B", 0.4)],
    )
    settings = Analysis(k_min=2.0, k_max=250.0)

    def starts(seeds=None):
        return [float(np.asarray(s.start())) for s in _k_prior(star, settings, starts=seeds)]

    spread = starts()
    assert spread == pytest.approx([2.0 + 248.0 / 3.0, 2.0 + 2.0 * 248.0 / 3.0])
    assert starts([None, None]) == pytest.approx(spread)
    assert starts([23.0, None]) == pytest.approx([23.0, 34.5])  # the lighter star moves more
    assert starts([900.0, 30.0]) == pytest.approx([20.0, 30.0])  # the heavier one, less
    assert starts([None, 240.0]) == pytest.approx([160.0, 240.0])
    # Every start stays strictly inside the range: 2% in from either bound.
    assert starts([249.0, None]) == pytest.approx([245.04, 245.04])
    assert starts([2.0, None]) == pytest.approx([6.96, 6.96])
    for spec in _k_prior(star, settings, starts=[900.0, 30.0]):
        assert isinstance(spec, Between) and (spec.lo, spec.hi) == (2.0, 250.0)
    with pytest.raises(ValueError, match="starts for 2 components"):
        _k_prior(star, settings, starts=[1.0])
    with pytest.raises(ValueError, match="strictly inside"):
        Between(2.0, 250.0, start_at=250.0).start()


def test_the_candidate_periods_are_the_union_of_three_searches(monkeypatch):
    """One list from the sinusoid, one from the second harmonic, one swap-invariant doubled."""
    import albireo.rvorbit as rvorbit_module
    from albireo.pipeline import _period_candidates

    calls = []

    def stub(table, *, swap_invariant=False, n_harmonics=1, **kwargs):
        first = bool(kwargs.get("components"))
        calls.append((swap_invariant, n_harmonics, first))
        if first:
            assert kwargs["components"] == ["A"]
            return {"period": 15.0, "aliases": [30.0, 10.0]}
        if swap_invariant:
            return {"period": 3.0, "aliases": [4.0, 5.0, 6.0, 7.0]}
        if n_harmonics == 2:
            return {"period": 11.0, "aliases": [10.05, 13.0]}
        return {"period": 10.0, "aliases": [20.0, 6.0]}

    monkeypatch.setattr(rvorbit_module, "find_period", stub)

    class _Table:
        n_components = 2
        names = ("A", "B")

    candidates, searches = _period_candidates(_Table(), swap_invariant=True)
    assert calls == [(False, 1, False), (False, 2, False), (False, 1, True), (True, 1, False)]
    # The four sources (the sinusoid, its two-harmonic form, the first component alone, the
    # swap-invariant peaks each followed by their double) merged round robin by rank: every
    # source's best peak, then every source's second, and so on. Everything within 2% of an
    # earlier candidate is dropped: 10.05 duplicates 10.0, the first component's 10.0 is
    # already there, and the invariant branch's 6.0, 10.0 and 6.0 (the double of 3.0 and of
    # 5.0, and the peak 6.0 itself) are all already there.
    assert candidates == [10.0, 11.0, 15.0, 3.0, 20.0, 30.0, 6.0, 13.0, 4.0, 8.0, 5.0, 12.0]
    assert searches["single"]["period"] == 10.0
    assert searches["harmonic"]["period"] == 11.0
    assert searches["first"]["period"] == 15.0
    assert searches["invariant"]["period"] == 3.0
    # Without the swap-invariant branch, and with a table too short for five parameters.
    calls.clear()
    plain, searches = _period_candidates(_Table(), swap_invariant=False)
    assert calls == [(False, 1, False), (False, 2, False), (False, 1, True)]
    assert searches["invariant"] is None
    assert plain == [10.0, 11.0, 15.0, 20.0, 30.0, 6.0, 13.0]

    def short(table, *, swap_invariant=False, n_harmonics=1, **kwargs):
        if n_harmonics == 2:
            raise ValueError("6 usable epochs cannot support the 5 parameters")
        return stub(table, swap_invariant=swap_invariant, n_harmonics=n_harmonics, **kwargs)

    monkeypatch.setattr(rvorbit_module, "find_period", short)
    candidates, searches = _period_candidates(_Table(), swap_invariant=False)
    assert candidates == [10.0, 15.0, 20.0, 30.0, 6.0] and searches["harmonic"] is None


def _peak_list_stub(lists):
    """A ``find_period`` standing in for four searches whose peak lists are given."""

    def stub(table, *, swap_invariant=False, n_harmonics=1, **kwargs):
        if kwargs.get("components"):
            peaks = lists["first"]
        elif swap_invariant:
            peaks = lists["invariant"]
        elif n_harmonics == 2:
            peaks = lists["harmonic"]
        else:
            peaks = lists["single"]
        return {"period": peaks[0], "aliases": list(peaks[1:])}

    return stub


def test_a_deeper_peak_list_only_adds_candidates_and_never_removes_one(monkeypatch):
    """Merging by rank makes the proposal monotone in the number of peaks each search reports.

    ``rvorbit._distinct_peaks`` walks the local maxima in descending power, so the list it
    returns for a given ``n_peaks`` is the first entries of the list it returns for any
    larger one: a deeper search extends each source rather than rewriting it. Merged round
    robin by rank, a deeper setting therefore only appends, and a period proposed at the
    shallower setting cannot be deduplicated away by a peak that the deeper setting reached
    first. Concatenation had no such property, and over the 33 blind-tier stars of D64 it
    lost 233 of the 1296 starts proposed at twenty peaks when the count went to fifty.
    """
    import albireo.rvorbit as rvorbit_module
    from albireo.pipeline import _period_candidates

    class _Table:
        n_components = 2
        names = ("A", "B")

    deep = {
        "single": [10.0, 21.0, 6.3, 33.0, 44.0, 57.0, 66.0, 77.0, 88.0, 4.05, 99.0, 111.0],
        "harmonic": [11.0, 10.05, 13.0, 17.0, 23.0, 29.0, 37.0, 43.0, 53.0, 61.0, 71.0, 83.0],
        "first": [15.0, 30.0, 10.0, 19.0, 25.0, 35.0, 46.0, 64.0, 75.0, 95.0, 105.0, 125.0],
        "invariant": [3.0, 4.0, 5.0, 7.0],  # four peaks whatever the count, plus their doubles
    }
    shallow = dict(deep, **{k: deep[k][:8] for k in ("single", "harmonic", "first")})

    monkeypatch.setattr(rvorbit_module, "find_period", _peak_list_stub(shallow))
    at_eight, _ = _period_candidates(_Table(), swap_invariant=True)
    monkeypatch.setattr(rvorbit_module, "find_period", _peak_list_stub(deep))
    at_twelve, _ = _period_candidates(_Table(), swap_invariant=True)

    assert set(at_eight) <= set(at_twelve) and len(at_twelve) > len(at_eight)
    # It is not only a superset: the shallow proposal is the head of the deeper one, since
    # the extra ranks fall after every rank the shallow setting reached.
    assert at_twelve[: len(at_eight)] == at_eight
    # The deeper setting's own new peak at 4.05 is the one dropped, not the invariant
    # search's 4.0, which outranks it.
    assert 4.0 in at_twelve and 4.05 not in at_twelve


def test_the_best_peak_of_the_last_search_is_not_displaced_by_a_deep_earlier_one(monkeypatch):
    """A lower-ranked peak of an earlier source no longer pre-empts a better peak of a later.

    The case is the blind-tier star gaia-37608387208382848 of D64. The start within 0.03% of
    its true 6.1037 d period is the double of the swap-invariant search's top peak, and a
    one-harmonic peak at 6.079648 d, 0.4% away and deep in that search's list, displaced it
    under the old concatenation: the orbit started at 6.103977 d reaches chi-square 2028 and
    the one started at 6.079648 d reaches 10072.
    """
    import albireo.rvorbit as rvorbit_module
    from albireo.pipeline import _period_candidates

    class _Table:
        n_components = 2
        names = ("A", "B")

    monkeypatch.setattr(
        rvorbit_module,
        "find_period",
        _peak_list_stub(
            {
                "single": [1.0134, 2.0271, 0.5067, 12.3305, 6.079648],
                "harmonic": [1.0133, 3.4021, 0.6712, 24.7712, 8.8814],
                "first": [1.0135, 2.0273, 0.5068, 40.1122, 15.7781],
                "invariant": [3.0519885, 1.2231, 0.8114, 4.4457],
            }
        ),
    )
    candidates, _ = _period_candidates(_Table(), swap_invariant=True)
    assert 6.103977 in candidates and 6.079648 not in candidates
    # The double of the invariant search's top peak is rank two of its source, so it is
    # proposed in the second round, well before the earlier source's fifth peak.
    assert candidates.index(6.103977) < candidates.index(12.3305)


def test_an_orbit_outside_the_declared_ranges_cannot_win(monkeypatch):
    """A lower chi-square at an absurd semi-amplitude or eccentricity is set aside.

    A companion the templates could not follow leaves velocities that a wrong period fits
    with a semi-amplitude of thousands of km/s at a lower chi-square than the truth; the
    declared ranges say such an orbit is not a solution, so the decision skips it.
    """
    import types

    import albireo.rvorbit as rvorbit_module
    from albireo.pipeline import _orbit_over_candidates

    fits = {
        0.34: dict(chi2=10.0, k=[1537.0, 0.0], ecc=0.94),
        0.58: dict(chi2=40.0, k=[103.0, 190.0], ecc=0.01),
        1.20: dict(chi2=90.0, k=[80.0, 20.0], ecc=0.95),
    }

    def fake_fit(table, *, period, circular):
        spec = fits[min(fits, key=lambda p: abs(p - period))]
        return types.SimpleNamespace(
            period=period, chi2=spec["chi2"], k=np.array(spec["k"]), ecc=spec["ecc"]
        )

    monkeypatch.setattr(rvorbit_module, "fit_rv_orbit", fake_fit)
    flags: list[str] = []
    ctx = types.SimpleNamespace(
        settings=Analysis(k_min=2.0, k_max=250.0, ecc_max=0.9), flag=flags.append
    )
    table = types.SimpleNamespace(n_components=1, bjd=np.arange(3.0))
    best, _, record = _orbit_over_candidates(ctx, table, [0.34, 0.58, 1.20], circular=False)
    assert best.period == 0.58 and record["n_candidates"] == 3
    assert any("lies outside the declared ranges" in f and "1537" in f for f in flags)
    # With nothing inside the ranges the lowest chi-square stands, and says so.
    flags.clear()
    best, _, _ = _orbit_over_candidates(ctx, table, [0.34, 1.20], circular=False)
    assert best.period == 0.34
    assert any("no candidate orbit lies inside the declared ranges" in f for f in flags)


def test_the_orbit_loop_merges_fits_and_names_every_ambiguity(monkeypatch):
    """Deduplication is on the fitted period; every rival within delta chi2 25 is flagged."""
    import albireo.rvorbit as rvorbit_module
    from albireo.pipeline import _orbit_over_candidates

    class _Fit:
        def __init__(self, period, chi2):
            self.period, self.chi2 = period, chi2
            self.k, self.ecc = np.array([40.0, 50.0]), 0.1  # inside the default ranges

        def predict(self, t):
            return np.zeros((2, np.size(t)))

    # starting period -> the period it converges to, and its chi-square
    outcome = {6.0: (6.31, 100.0), 6.4: (6.30, 90.0), 12.6: (12.62, 110.0), 3.1: (3.15, 400.0)}

    def stub_fit(table, *, period, circular=False, **kwargs):
        return _Fit(*outcome[round(float(period), 4)])

    monkeypatch.setattr(rvorbit_module, "fit_rv_orbit", stub_fit)
    monkeypatch.setattr(
        rvorbit_module,
        "reassign_by_orbit",
        lambda table, predicted: (table, np.zeros(table.bjd.size, dtype=bool)),
    )

    class _Table:
        n_components = 2
        bjd = np.arange(10.0)

    class _Ctx:
        def __init__(self):
            self.flags = []
            self.settings = Analysis()

        def flag(self, text):
            self.flags.append(text)

    ctx = _Ctx()
    # 6.0 is repeated and 6.05 is within 2% of it, so neither is fitted twice.
    best, _, decision = _orbit_over_candidates(
        ctx, _Table(), [6.0, 6.4, 6.05, 12.6, 6.0, 3.1], circular=False
    )
    assert decision["n_candidates"] == 4
    assert best.period == 6.30 and best.chi2 == 90.0
    # 6.31 converged to the winner's period and is not an ambiguity; 12.62 is, at +20;
    # 3.15 is 310 away and is not named.
    assert decision["ambiguous"] == [{"period": 12.62, "delta_chi2": 20.0}]
    assert len(ctx.flags) == 1 and "12.6200 d at +20.0" in ctx.flags[0]
    assert "6.31" not in ctx.flags[0] and "3.15" not in ctx.flags[0]
    # The ranking the period decision takes the top few of: the distinct orbits in
    # chi-square order, the winner first, each with the table it was fitted to. 6.31 is
    # the same period as the winner and was merged into it, as it is for the ambiguity.
    assert [round(o.period, 2) for o, _ in decision["ranked"]] == [6.30, 12.62, 3.15]


def _orbit_table(bjd, *, sigma=0.3, seed=0):
    """A two-component ``VelocityTable`` of ``ORBIT`` at ``bjd``, every velocity detected."""
    from albireo.todcor import VelocityTable

    bjd = np.asarray(bjd, dtype=float)
    n = bjd.size
    truth = ORBIT.component_velocities(bjd)
    velocity = truth + np.random.default_rng(seed).normal(0.0, sigma, truth.shape)
    errors = np.full(truth.shape, sigma)
    return VelocityTable(
        names=("A", "B"),
        bjd=bjd,
        instrument=("TOY",) * n,
        velocity=velocity,
        sigma=errors,
        sigma_ivar=errors,
        covariance=np.tile(np.diag([sigma**2, sigma**2]), (n, 1, 1)),
        light=np.repeat([[0.62], [0.38]], n, axis=1),
        light_mode="global median",
        chi2=np.full(n, 1000.0),
        chi2_null=np.full(n, 1e5),
        n_pixels=np.full(n, 1000),
        delta_chi2=np.full(truth.shape, 1e4),
        blended=np.zeros(n, dtype=bool),
        at_edge=np.zeros(truth.shape, dtype=bool),
        refined=np.ones(n, dtype=bool),
        absolute=(True, True),
        frame="barycentric",
        settings={"n_parameters": 3},
    )


def test_the_detection_gate_takes_the_weight_off_undetected_velocities_only(monkeypatch):
    """A companion the templates lost leaves draws the gate removes, and the primary stays (D65).

    On the benchmark system that motivated it the 7% secondary was undetected at 24 of 26
    epochs, its velocities spread across the search window at detection statistics of 1.4 to
    61, and the joint fit from the true period ended at 511 d with semi-amplitudes of 49 and
    66 km/s. The gate sets those velocities to nan in a copy of the table that only the
    search and the candidate fits see.
    """
    from types import SimpleNamespace

    from albireo.pipeline import _detection_gate, _orbit_over_candidates, _period_candidates
    from albireo.rvorbit import fit_rv_orbit

    rng = np.random.default_rng(4)
    table = _orbit_table(np.sort(rng.uniform(0.0, 60.0, size=24)), seed=4)
    lost = np.zeros(24, dtype=bool)
    lost[rng.choice(24, size=18, replace=False)] = True
    velocity = np.array(table.velocity)
    statistic = np.array(table.delta_chi2)
    velocity[1, lost] = rng.uniform(-300.0, 300.0, size=18)
    statistic[1, lost] = rng.uniform(1.0, 60.0, size=18)
    weak = replace(table, velocity=velocity, delta_chi2=statistic)

    gated, mask = _detection_gate(weak, 100.0)
    np.testing.assert_array_equal(mask, np.vstack([np.zeros(24, dtype=bool), lost]))
    assert np.isnan(gated.velocity[1, lost]).all() and np.isnan(gated.sigma[1, lost]).all()
    np.testing.assert_array_equal(gated.velocity[:, ~lost], weak.velocity[:, ~lost])
    np.testing.assert_array_equal(gated.velocity[0], weak.velocity[0])
    same, none = _detection_gate(weak, 0.0)
    assert same is weak and not none.any()
    # The first component is never gated, however weak: that is the rule measured, and
    # gating it took one benchmark system's true period from rank 2 to absent.
    faint_first = np.array(statistic)
    faint_first[0, :5] = 12.0
    primary_weak = replace(weak, delta_chi2=faint_first)
    kept, first_mask = _detection_gate(primary_weak, 100.0)
    assert not first_mask[0].any()
    np.testing.assert_array_equal(first_mask, mask)
    np.testing.assert_array_equal(kept.velocity[0], weak.velocity[0])
    weak_first_only = np.array(table.delta_chi2)
    weak_first_only[0, :5] = 12.0
    only_first = replace(table, delta_chi2=weak_first_only)
    alone, alone_mask = _detection_gate(only_first, 100.0)
    assert alone is only_first and not alone_mask.any()

    # The search: the first component's source keeps every epoch, the relative velocity
    # only the six at which both were detected.
    _, searches = _period_candidates(weak, swap_invariant=True, detection_min=100.0)
    assert searches["first"]["used"].all()
    np.testing.assert_array_equal(searches["single"]["used"], ~lost)
    # A companion detected nowhere leaves the first component's source alone, not a failure.
    nowhere = np.array(statistic)
    nowhere[1] = 10.0
    blind = replace(weak, delta_chi2=nowhere)
    candidates, searches = _period_candidates(blind, swap_invariant=True, detection_min=100.0)
    assert searches["single"] is None and searches["invariant"] is None
    assert searches["first"] is not None and candidates

    # The candidate fits see the gated copy and hand back the table as measured.
    flags: list[str] = []
    ctx = SimpleNamespace(settings=Analysis(k_min=2.0, k_max=250.0, ecc_max=0.9), flag=flags.append)
    best, best_table, _ = _orbit_over_candidates(
        ctx, weak, [6.35], circular=False, detection_min=100.0
    )
    assert abs(best.period - ORBIT.period) < 0.02 and abs(best.k[0] - 30.0) < 1.0
    assert best.chi2 == pytest.approx(fit_rv_orbit(gated, period=6.35).chi2, rel=1e-6)
    np.testing.assert_array_equal(best_table.velocity, weak.velocity)
    ungated, _, _ = _orbit_over_candidates(ctx, weak, [6.35], circular=False)
    assert ungated.chi2 > 100.0 * best.chi2

    # The exchange is decided on the gated copy: the refit is on the exchanged gated copy,
    # so a gated velocity still carries no weight, and the table handed back is the table
    # as measured with the same epochs exchanged.
    import albireo.rvorbit as rvorbit_module
    from albireo.rvorbit import _exchanged

    swap = np.zeros(24, dtype=bool)
    swap[np.flatnonzero(~lost)[0]] = True
    decided_on = []
    monkeypatch.setattr(
        rvorbit_module,
        "reassign_by_orbit",
        lambda t, p: (decided_on.append(t) or _exchanged(t, swap), swap),
    )
    fitted_to = []

    def spy_fit(t, **kwargs):
        fitted_to.append(t)
        return fit_rv_orbit(t, **kwargs)

    monkeypatch.setattr(rvorbit_module, "fit_rv_orbit", spy_fit)
    _, exchanged, _ = _orbit_over_candidates(ctx, weak, [6.35], circular=False, detection_min=100.0)
    assert len(decided_on) == 1 and np.isnan(decided_on[0].velocity[1, lost]).all()
    assert len(fitted_to) == 2
    np.testing.assert_array_equal(fitted_to[0].velocity, gated.velocity)
    np.testing.assert_array_equal(fitted_to[1].velocity, _exchanged(gated, swap).velocity)
    assert np.isnan(fitted_to[1].velocity[1, lost]).all()
    np.testing.assert_array_equal(exchanged.velocity, _exchanged(weak, swap).velocity)
    np.testing.assert_array_equal(exchanged.velocity[:, swap], weak.velocity[::-1, swap])
    np.testing.assert_array_equal(exchanged.velocity[:, ~swap], weak.velocity[:, ~swap])
    assert exchanged.settings["reassigned_by_orbit"] == 1
    monkeypatch.setattr(rvorbit_module, "fit_rv_orbit", fit_rv_orbit)

    # Without the exchange nothing is re-assigned at all.
    def refuse(*args, **kwargs):
        raise AssertionError("the exchange was not allowed")

    monkeypatch.setattr(rvorbit_module, "reassign_by_orbit", refuse)
    _, kept, _ = _orbit_over_candidates(ctx, weak, [6.35], circular=False, exchange=False)
    assert kept is weak

    assert Analysis().detection_min == 100.0
    for bad in (-1.0, float("nan"), True):
        with pytest.raises(ValueError, match="detection_min must be a non-negative"):
            Analysis(detection_min=bad)


def test_leave_one_epoch_out_peaks_follow_the_merge_on_a_short_table(monkeypatch):
    """The fifth source: three peaks per left-out epoch, after the round robin (D65).

    On the benchmark system that motivated it one epoch of twelve had its twin components
    exchanged, which put the true period at peak 108 of the one-harmonic search; leaving each
    epoch out in turn and adding the three highest peaks of each put it at rank 1 of the
    chi-square ranking. The block is appended after the round robin, in epoch order, and a
    peak within 2% of anything already listed is dropped, which is the form that was measured.
    """
    import albireo.rvorbit as rvorbit_module
    from albireo.pipeline import _LEAVE_ONE_OUT_MAX_EPOCHS, _period_candidates

    base = {
        "single": [10.0, 21.0, 6.3],
        "harmonic": [11.0, 10.05, 13.0],
        "first": [15.0, 30.0, 10.0],
        "invariant": [3.0, 4.0, 5.0, 7.0],
    }
    left_out = {
        0: [5.356, 10.1, 2.6],  # 10.1 is within 2% of 10.0, already listed
        1: [5.36, 1.8, 12.5],  # 5.36 within 2% of 5.356 from the epoch before
        2: [40.0, 1.81, 0.9],
    }
    calls = []

    def stub(table, *, swap_invariant=False, n_harmonics=1, n_peaks=20, components=None):
        missing = np.flatnonzero(np.all(np.isnan(np.asarray(table.velocity)), axis=0))
        if missing.size:
            assert n_peaks == 3 and n_harmonics == 1 and not swap_invariant and components is None
            calls.append(int(missing[0]))
            peaks = left_out.get(int(missing[0]), [])
            if not peaks:
                raise ValueError("stands for a search with too few epochs")
            return {"period": peaks[0], "aliases": peaks[1:]}
        if components:
            peaks = base["first"]
        elif swap_invariant:
            peaks = base["invariant"]
        elif n_harmonics == 2:
            peaks = base["harmonic"]
        else:
            peaks = base["single"]
        return {"period": peaks[0], "aliases": list(peaks[1:])}

    monkeypatch.setattr(rvorbit_module, "find_period", stub)
    table = _velocity_table(n_epochs=5)
    # Only on request: the bootstrap asks, the route that fits declared velocities does not.
    plain, searches = _period_candidates(table, swap_invariant=True)
    assert searches["leave_one_out"] is None and calls == []
    candidates, searches = _period_candidates(table, swap_invariant=True, leave_one_out=True)
    head = [10.0, 11.0, 15.0, 3.0, 21.0, 30.0, 6.0, 6.3, 13.0, 4.0, 8.0, 5.0, 7.0, 14.0]
    assert plain == head
    assert candidates[: len(head)] == head  # the round robin, as without the source
    assert candidates[len(head) :] == [5.356, 2.6, 1.8, 12.5, 40.0, 0.9]
    assert searches["leave_one_out"] == [5.356, 2.6, 1.8, 12.5, 40.0, 0.9]
    assert calls == [0, 1, 2, 3, 4]
    # A leave-one-out start is never listed within 2% of a periodogram start, so a deeper
    # periodogram list can take its place: 5.3 from the one-harmonic search replaces 5.356.
    base["single"] = [10.0, 21.0, 6.3, 5.3]
    deeper, _ = _period_candidates(table, swap_invariant=True, leave_one_out=True)
    assert 5.3 in deeper and 5.356 not in deeper and 5.36 not in deeper

    # Above the limit the source does not run.
    calls.clear()
    long_table = _velocity_table(n_epochs=_LEAVE_ONE_OUT_MAX_EPOCHS + 1)
    _, searches = _period_candidates(long_table, swap_invariant=True, leave_one_out=True)
    assert searches["leave_one_out"] is None and calls == []

    # The limit is on the usable epochs as measured, before the detection gate: gating the
    # companion at six of 26 epochs leaves 20 relative velocities, and still no source.
    statistic = np.array(long_table.delta_chi2)
    statistic[1, :6] = 10.0
    gated_long = replace(long_table, delta_chi2=statistic)
    _, searches = _period_candidates(
        gated_long, swap_invariant=True, detection_min=100.0, leave_one_out=True
    )
    assert searches["leave_one_out"] is None and calls == []
    # At 25 measured epochs it runs, leaving out each of the 25, gated or not.
    at_limit = _velocity_table(n_epochs=_LEAVE_ONE_OUT_MAX_EPOCHS)
    statistic = np.array(at_limit.delta_chi2)
    statistic[1, :6] = 10.0
    _period_candidates(
        replace(at_limit, delta_chi2=statistic),
        swap_invariant=True,
        detection_min=100.0,
        leave_one_out=True,
    )
    assert calls == list(range(_LEAVE_ONE_OUT_MAX_EPOCHS))


def test_the_known_period_and_declared_velocity_routes_never_gate_or_leave_epochs_out(
    toy, tmp_path, monkeypatch
):
    """The gate and the leave-one-out source belong to the bootstrap alone (D65).

    Both were measured on the bootstrap's library-template tables. The template table of the
    known-period route (``_table_orbit``) and the table measured after a free-velocity
    disentangling (``_orbit``) are searched and fitted as measured.
    """
    import types

    import albireo.pipeline as pipeline_module
    from albireo.pipeline import _orbit, _table_orbit

    dataset, truth, grid = toy
    star = _star(dataset, truth, grid, name="routes")
    ctx = _context(star, tmp_path / "routes", Analysis())
    seen: dict[str, list] = {"candidates": [], "fits": []}
    real_candidates = pipeline_module._period_candidates

    def spy_candidates(t, **kwargs):
        seen["candidates"].append(kwargs)
        candidates, searches = real_candidates(t, **kwargs)
        seen["searches"] = searches
        return candidates, searches

    stub_orbit = types.SimpleNamespace(period=6.31, chi2=1.0, k=np.array([30.0, 55.0]), ecc=0.1)

    def spy_fits(ctx_, t, periods, **kwargs):
        seen["fits"].append(kwargs)
        return stub_orbit, t, {"n_candidates": 1, "ambiguous": [], "ranked": [(stub_orbit, t)]}

    monkeypatch.setattr(pipeline_module, "_period_candidates", spy_candidates)
    monkeypatch.setattr(pipeline_module, "_orbit_over_candidates", spy_fits)
    table = _velocity_table(n_epochs=12)
    statistic = np.array(table.delta_chi2)
    statistic[1, :4] = 10.0  # weak companion velocities a gate would have removed
    table = replace(table, delta_chi2=statistic)

    orbit, source = _orbit(ctx, _StubFit(("A", "B"), mode="free"), table)
    assert orbit is stub_orbit and "periodogram" in source
    assert seen["candidates"] == [{"swap_invariant": True}]
    assert seen["searches"]["leave_one_out"] is None
    assert seen["fits"] == [{"circular": False}]

    seen["fits"].clear()
    assert _table_orbit(ctx, table, star, Analysis()) is stub_orbit
    assert seen["fits"] == [{"circular": False}]


def test_the_bootstrap_flags_a_table_on_few_nights_and_honours_the_exchange_rule(
    toy, tmp_path, monkeypatch
):
    """Two rules of D65 on the bootstrap: the few-nights flag and the light ratio of the exchange.

    The flag changes nothing; three of the 33 blind tables of the third run had their usable
    epochs on fewer than eight nights and the search recovered none of them. The exchange
    in the candidate fits now follows the rule the velocities measured after the
    disentangling follow, on the light fractions of the table's own amplitudes.
    """
    from types import SimpleNamespace

    import albireo.pipeline as pipeline_module
    from albireo.pipeline import _bootstrap

    dataset, truth, grid = toy
    star = _star(dataset, truth, grid, name="sparse", period="search")
    orbit = SimpleNamespace(
        period=6.31,
        chi2=12.0,
        k=np.array([30.0, 55.0]),
        ecc=0.12,
        omega=0.7,
        t_conj=2455000.5,
        names=("A", "B"),
        errors={"k": np.array([1.2, 1.6]), "ecc": 0.01},
        summary=lambda: "  orbit",
    )
    seen: dict = {}

    def table_with(bjd, light):
        n = len(bjd)
        statistic = np.full((2, n), 1e4)
        statistic[1, :3] = [5.0, 40.0, 99.0]
        statistic[0, 3] = 5.0  # a weak first component is never gated
        return SimpleNamespace(
            names=("A", "B"),
            n_components=2,
            settings={},
            n_epochs=n,
            bjd=np.asarray(bjd, dtype=float),
            velocity=np.zeros((2, n)),
            delta_chi2=statistic,
            light=np.repeat(np.asarray(light, dtype=float)[:, None], n, axis=1),
            good=np.ones(n, dtype=bool),
            summary=lambda: "  table summary",
        )

    def run(table, name):
        monkeypatch.setattr(pipeline_module, "_library_table", lambda *a, **k: (table, []))

        def candidates(t, **kwargs):
            seen["search"] = kwargs
            return [6.3], {"single": None, "harmonic": None, "invariant": None, "first": None}

        def fits(ctx_, t, periods, **kwargs):
            seen["fits"] = kwargs
            return orbit, t, {"n_candidates": 1, "ambiguous": [], "ranked": [(orbit, t)]}

        monkeypatch.setattr(pipeline_module, "_period_candidates", candidates)
        monkeypatch.setattr(pipeline_module, "_orbit_over_candidates", fits)
        ctx = _context(
            star, tmp_path / name, Analysis(period_decision_candidates=0), library="stub"
        )
        _spec, block = _bootstrap(ctx, dataset, {"TOY": 5.5}, None)
        return ctx, block["report"]

    # Nine epochs on five nights, light fractions 0.9 and 0.1.
    sparse = 2455000.2 + np.array([0.0, 0.07, 0.14, 3.0, 3.07, 40.0, 41.0, 90.0, 90.05])
    ctx, report = run(table_with(sparse, [0.9, 0.1]), "sparse")
    assert seen["search"]["detection_min"] == 100.0 and seen["fits"]["detection_min"] == 100.0
    assert seen["search"]["leave_one_out"] is True
    assert seen["fits"]["exchange"] is False and report["exchange_allowed"] is False
    assert report["n_nights"] == 5
    assert report["detection_gate"] == {"threshold": 100.0, "gated": {"A": 0, "B": 3}}
    assert any("usable epochs fall on 5 night(s), fewer than 8" in f for f in ctx.flags)
    assert any(
        "exchange of the two components in the bootstrap's candidate fits was skipped" in f
        for f in ctx.flags
    )
    assert report["periodogram_peak"] is None and report["aliases"] == []

    # Eight nights and alike light: neither flag, and the exchange runs.
    spread = 2455000.2 + 5.0 * np.arange(8.0)
    ctx, report = run(table_with(spread, [0.55, 0.45]), "spread")
    assert seen["fits"]["exchange"] is True and report["n_nights"] == 8
    assert not any("night(s)" in f or "was skipped" in f for f in ctx.flags)


def test_the_velocity_products_carry_every_digit_of_the_epoch_time(tmp_path):
    """Six decimals of a BJD could not reproduce a period search; the products round-trip (D65)."""
    import csv

    from albireo.pipeline import _write_velocity_csv

    table = _velocity_table()
    table = replace(
        table, bjd=2457946.280993 + np.array([0.0, 1.23e-7, 0.0739612345, 3.1, 4.2, 5.3])
    )
    written = np.loadtxt(table.write(tmp_path / "table.rv"), comments="#", usecols=(0,))
    np.testing.assert_array_equal(written, table.bjd)
    with _write_velocity_csv(tmp_path / "table.csv", table).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    np.testing.assert_array_equal([float(r["bjd"]) for r in rows], table.bjd)
    assert rows[0]["v_A"] == f"{table.velocity[0, 0]:.6f}"  # everything else as before

    # The edge flag: merged in the .rv file and the CSV's at_edge, per component in the CSV.
    at_edge = np.zeros((2, 6), dtype=bool)
    at_edge[1, 2] = True
    velocity = np.array(table.velocity)
    velocity[1, 2] = np.nan
    edge = replace(table, at_edge=at_edge, velocity=velocity)
    with _write_velocity_csv(tmp_path / "edge.csv", edge).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["at_edge"] for r in rows] == ["0", "0", "1", "0", "0", "0"]
    assert [r["at_edge_A"] for r in rows] == ["0"] * 6
    assert [r["at_edge_B"] for r in rows] == ["0", "0", "1", "0", "0", "0"]
    header = edge.write(tmp_path / "edge.rv").read_text(encoding="utf-8").splitlines()
    assert "at_edge_A" not in " ".join(header)


def test_the_disentangling_decides_among_the_best_candidate_periods(toy, tmp_path, monkeypatch):
    """The table ranks the candidates by chi-square; the disentangling chooses among the best.

    The velocity table cannot separate a period from its aliases when the table is poor.
    Two blind systems of the D62 Gaia population make the case: on fifteen epochs an
    eccentric Keplerian at 0.2485 d with semi-amplitudes of 233 and 239 km/s at e = 0.73
    fitted better than the true 6.104 d, and on twelve epochs the same happened at 1.129 d
    at e = 0.72 against a true 5.356 d. Both lie inside the declared ranges, so the range
    filter does not reach them. The three candidate orbits here stand for that case: the
    chi-square prefers the first, the marginal likelihood of the disentangling prefers the
    second, and the second is the one the fit is started from.
    """
    from types import SimpleNamespace

    import albireo.pipeline as pipeline_module
    from albireo.pipeline import _bootstrap, _jsonable

    dataset, truth, grid = toy
    star = _star(dataset, truth, grid, name="decide", period="search")

    def orbit_at(period, chi2, k):
        return SimpleNamespace(
            period=period,
            chi2=chi2,
            k=np.asarray(k, dtype=float),
            ecc=0.12,
            omega=0.7,
            t_conj=2455000.5,
            names=("A", "B"),
            errors={"k": np.array([1.2, 1.6]), "ecc": 0.01},
            summary=lambda: f"  orbit at P {period:.5f} d",
        )

    table = SimpleNamespace(
        names=("A", "B"),
        n_components=2,
        settings={},
        n_epochs=12,
        bjd=2455000.25 + 3.0 * np.arange(12.0),
        velocity=np.zeros((2, 12)),
        delta_chi2=np.full((2, 12), 1e4),
        light=np.repeat([[0.6], [0.4]], 12, axis=1),
        good=np.ones(12, dtype=bool),
        summary=lambda: "  table summary",
    )
    templates = [SimpleNamespace(meta={"source": "stub"}) for _ in range(2)]
    # The chi-square order: the absurd alias first, the truth second. The truth's secondary
    # is one the templates could not follow, so its semi-amplitude is degenerate and the
    # declaration falls back to the range, which is where the scan's best becomes a start.
    ranked = [
        (orbit_at(0.24847, 8.0, (233.0, 239.0)), table),
        (orbit_at(6.10400, 31.0, (40.0, 300.0)), table),
        (orbit_at(1.12943, 45.0, (182.0, 185.0)), table),
    ]
    # period -> (best marginal over phase at the start, the semi-amplitude scan's best
    # value, and the semi-amplitudes it found). The first and third scans end below their
    # start and are refused, so those candidates are worth their phase scan's maximum.
    scanned = {
        0.24847: (-1000.0, -1010.0, [231.0, 237.0]),
        6.10400: (-850.0, -840.0, [38.0, 61.0]),
        1.12943: (-980.0, -990.0, [180.0, 186.0]),
    }

    class _Phase:
        def __init__(self, best, values):
            self.best, self.values = best, np.asarray(values, dtype=float)

    class _Amp:
        def __init__(self, best, best_t_conj, best_value, start_value):
            self.best = np.asarray(best, dtype=float)
            self.best_t_conj, self.best_value, self.start_value = (
                best_t_conj,
                best_value,
                start_value,
            )
            self.n_trials, self.notes = 317, ()

        @property
        def gain(self):
            return self.best_value - self.start_value

    class _FakeDis:
        """Only what the comparison touches: a coarse copy, a phase scan, a K scan."""

        def __init__(self, spec):
            centre = 0.5 * (float(spec.period.lo) + float(spec.period.hi))
            self.period = min(scanned, key=lambda p: abs(p - centre))
            self.init = {"k": np.array([10.0, 20.0]), "t_conj": 2455000.0}
            self.fixed = {}

        def _scan_declaration(self):
            return self

        def _scan_phase(self, init):
            return _Phase(2455000.25, [scanned[self.period][0] - 30.0, scanned[self.period][0]])

        def _has_ranged_k(self):
            return True

        def _scan_semi_amplitudes(self, init):
            start, best, k_best = scanned[self.period]
            return _Amp(k_best, 2455000.3, best, start)

    declared = []

    def fake_declare(ctx_, dataset_, lsf_, spec, velocities):
        assert velocities is None
        # The comparison scans the conjunction, which is as uncertain as the period it
        # came from, and declares every semi-amplitude on the settings' own bounds, so
        # that every candidate is measured on one velocity budget and one model grid.
        assert spec.t_conj == "scan"
        declared.append([(float(s.lo), float(s.hi)) for s in spec.k])
        return _FakeDis(spec)

    searches = {
        "single": {"period": 0.18652, "aliases": [0.2657, 0.1765, 0.1991, 0.4084]},
        "harmonic": {"period": 0.22311, "aliases": [0.3]},
        "invariant": None,
        "first": None,
    }
    monkeypatch.setattr(pipeline_module, "_library_table", lambda *a, **k: (table, templates))
    monkeypatch.setattr(pipeline_module, "_period_candidates", lambda t, **k: ([1.0], searches))
    monkeypatch.setattr(
        pipeline_module,
        "_orbit_over_candidates",
        lambda ctx_, t, periods, **k: (
            ranked[0][0],
            ranked[0][1],
            {"n_candidates": 7, "ambiguous": [], "ranked": ranked},
        ),
    )
    monkeypatch.setattr(pipeline_module, "_declare", fake_declare)

    ctx = _context(star, tmp_path / "decide", Analysis(), library="stub")
    spec, block = _bootstrap(ctx, dataset, {"TOY": 5.5}, None)

    # The period the fit is started from is the disentangling's choice, not the table's.
    assert 0.5 * (spec.period.lo + spec.period.hi) == pytest.approx(6.10400)
    assert spec.period.lo == pytest.approx(0.97 * 6.10400)
    decision = block["report"]["decision"]
    assert decision["by"] == "disentangling"
    assert decision["chosen_period"] == pytest.approx(6.10400)
    assert decision["table_period"] == pytest.approx(0.24847)
    assert decision["margin_nats"] == pytest.approx(140.0)  # over the runner-up, 1.129 d
    assert decision["over_table_nats"] == pytest.approx(160.0)
    assert [c["value_nats"] for c in decision["candidates"]] == pytest.approx(
        [-1000.0, -840.0, -980.0]
    )
    assert [c["chosen"] for c in decision["candidates"]] == [False, True, False]
    assert [c["scan_moved"] for c in decision["candidates"]] == [False, True, False]
    assert decision["candidates"][1]["k"] == pytest.approx([38.0, 61.0])
    assert decision["candidates"][0]["k"] == pytest.approx([10.0, 20.0])  # the refused scan
    assert all(c["n_trials"] == 319 for c in decision["candidates"])  # 317 K trials, 2 phases
    assert all(c["seconds"] >= 0.0 for c in decision["candidates"])
    assert [c["table_chi2"] for c in decision["candidates"]] == [8.0, 31.0, 45.0]
    # Three candidates were declared, on identical semi-amplitude bounds, and nothing that
    # holds a compiled model survived into the report.
    assert len(declared) == 3 and declared[0] == declared[1] == declared[2]
    json.dumps(_jsonable(block["report"]))
    # The disagreement with the table is flagged, with both periods and the margin.
    flag = next(f for f in ctx.flags if "the table preferred" in f)
    assert "P 0.24847 d at chi2 8.0" in flag and "P 6.10400 d by 160 nats" in flag
    assert "coarse grid" in block["text"] and "6.10400 d, 140 nats" in block["text"]
    # The chosen candidate's scan moved, so its semi-amplitudes start the disentangling
    # where the range is searched (this orbit's secondary is outside the declared range).
    assert [float(s.start_at) for s in spec.k] == pytest.approx([38.0, 61.0])

    # With the setting at zero the table's choice stands and no model is ever built.
    declared.clear()
    plain = _context(
        star, tmp_path / "table", Analysis(period_decision_candidates=0), library="stub"
    )
    spec, block = _bootstrap(plain, dataset, {"TOY": 5.5}, None)
    assert declared == []
    assert 0.5 * (spec.period.lo + spec.period.hi) == pytest.approx(0.24847)
    decision = block["report"]["decision"]
    assert decision["by"] == "table" and decision["candidates"] == []
    assert decision["chosen_period"] == pytest.approx(0.24847)
    assert decision["margin_nats"] is None and "disabled" in decision["reason"]
    assert not any("the disentangling prefers" in f for f in plain.flags)
    with pytest.raises(ValueError, match="period_decision_candidates must be a non-negative"):
        Analysis(period_decision_candidates=-1)


def test_a_degenerate_bootstrap_falls_back_to_the_declared_range(library, tmp_path, monkeypatch):
    """A bootstrap semi-amplitude outside the range must not pin the disentangling to it.

    The first guard clipped such a value into the range and declared it known, which held
    a faint secondary's K at the floor; the range search is what the known-period route
    does, and it recovers the orbit from the bootstrap's period alone.
    """
    import dataclasses

    import albireo.pipeline as pipeline_module

    dataset, truth, grid = _simulate(library, n_epochs=10, snr=150.0)
    star = _star(dataset, truth, grid, period="search")
    real = pipeline_module._orbit_over_candidates

    def degenerate(ctx, table, periods, *, circular, **kwargs):
        orbit, table, decision = real(ctx, table, periods, circular=circular, **kwargs)
        # The decision by the disentangling takes its candidates from the ranked list, so
        # every candidate is made degenerate too, whichever it chooses.
        broken = dataclasses.replace(orbit, k=np.array([0.0, 5000.0]))
        decision = {
            **decision,
            "ranked": [
                (dataclasses.replace(o, k=np.array([0.0, 5000.0])), t)
                for o, t in decision.get("ranked", [])
            ],
        }
        return broken, table, decision

    monkeypatch.setattr(pipeline_module, "_orbit_over_candidates", degenerate)
    config = PipelineConfig(
        stars=[star],
        output=tmp_path,
        library=library,
        mh=(-0.9, 0.4),
        analysis=Analysis(
            max_steps=120,
            label_steps=200,
            v_zero_range=40.0,
            plots=False,
            v_range=150.0,
            period_decision_candidates=2,
        ),
    )
    result = run_star(star, config, progress=False)
    flags = result.report["flags"]
    assert any("searches the range instead" in f for f in flags), flags
    orbit = result.report["orbit"]
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


def test_a_semi_amplitude_start_no_medium_allows_is_skipped_and_not_a_failure(
    toy, library, tmp_path
):
    """A stage nobody asked for must be skipped, not raise, when the medium is missing.

    A ranged semi-amplitude with a library at hand buys a head start: a velocity table
    from library templates at the declared period, whose orbit seeds the starting values.
    Rendering a template needs the wavelength medium, which the files need not declare.
    Nobody asked for that table, so its unavailability is not a reason to fail the star:
    the skip is recorded as a flag and the declared range's evenly spaced starts are used,
    which is where they start when no library is declared at all. The refusal stays a
    refusal on the routes the user did ask for (``period = "search"``, ``light =
    "measure"``). This is what the packaged demo's first star hit.
    """
    dataset, truth, grid = toy
    undeclared = ab.Dataset(tuple(replace(e, medium=None) for e in dataset), frame=dataset.frame)
    star = _star(
        undeclared, truth, grid, name="no_medium", labels=False, overrides={"max_steps": 2}
    )
    assert all(c.k is None for c in star.components), "the branch needs a ranged semi-amplitude"
    config = PipelineConfig(
        stars=[star], output=tmp_path, library=library, analysis=Analysis(fast=True, plots=False)
    )
    result = run_star(star, config, progress=False)
    assert result.ok, result.error
    assert any("semi-amplitude starts skipped" in f for f in result.flags), result.flags
    assert "k-start" not in result.seconds, "the stage was skipped, not run"


# ---------------------------------------------------------------------------
# 4. the guards: a stage that failed must not be read as a measurement
# ---------------------------------------------------------------------------
#
# One archived star of the D62 benchmark wrote fifteen rows of plausible velocities out of
# a diverged disentangling, a label fit that beat neither of its nulls, and a correlation
# that reported a position it never evaluated
# (internal/research/2026-09-09-gaia-rvs-benchmark/d63_edge_pinned_table.md). Each guard
# below refuses one of the steps by which that file was written. Three of them refuse a
# table a healthy fit can still write: a component held at a light fraction the shrunken
# spectrum no longer has, an exchange of two components the correlation's peaks cannot
# support, and a frame offset the label fit never learned.


def _context(star, directory, settings=None, library=None):
    """A ``_Context`` for the stage helpers, logging into ``directory``."""
    from albireo.pipeline import _Context, _Log

    directory.mkdir(parents=True, exist_ok=True)
    config = PipelineConfig(stars=[star], output=directory, library=library)
    return _Context(
        star=star,
        config=config,
        settings=settings or Analysis(),
        directory=directory,
        log=_Log(star.name, directory / "log.txt", progress=False),
    )


class _StubFit:
    """Enough of a ``Fit`` for ``_templates``: templates, mode, budget, narrowest LSF.

    ``lights`` and ``taus`` add what the velocity stage reads besides: the declared
    components as real :class:`albireo.Star` objects, so that the smoothness starting
    point is the library's own default rather than a number this file chose, and the
    hyperparameters ML-II came back with. Given a ``table``, ``measure_velocities``
    returns it and the stub also answers the reporting and assessment helpers that
    ``_run_stages`` calls around the stage.
    """

    def __init__(
        self,
        names,
        *,
        mode="keplerian",
        budget=300.0,
        narrowest=5.0,
        lights=None,
        taus=None,
        table=None,
    ):
        from types import SimpleNamespace

        from albireo.todcor import Template

        grid = ab.LogGrid.from_wavelength_range(5150.0, 5160.0, dv_kms=5.0)
        self._templates = [
            Template(name=n, grid=grid, deviation=np.zeros(grid.n), meta={"source": "stub"})
            for n in names
        ]
        self.mode = mode
        self._table = table
        lights = [1.0 / len(names)] * len(names) if lights is None else list(lights)
        taus = [300.0] * len(names) if taus is None else list(taus)
        stars = [ab.Star(name=n, light=float(f)) for n, f in zip(names, lights, strict=True)]
        self.hyper = {n: {"tau": float(t), "eta": 5.0} for n, t in zip(names, taus, strict=True)}
        self.z_rms = 1.0
        self.phase_scan = None
        self.k_scan = None
        self.result = SimpleNamespace(
            potential=-1.0e4, grad_norm=1.0e-4, num_steps=12, converged=True
        )
        self.dis = SimpleNamespace(
            velocity_budget=SimpleNamespace(total=budget),
            _narrowest_lsf=lambda: narrowest,
            stars=stars,
            ordered_components=stars,
            component_names=tuple(names),
            n_stellar=len(stars),
            grid=grid,
            model=SimpleNamespace(half_bandwidth=3),
            noise_correlation=None,
            orbit=None,
            explain=lambda: "  stub declaration",
            fit=lambda **kwargs: self,
        )

    def templates(self):
        return list(self._templates)

    def summary(self):
        return "  stub fit"

    def noise_correlation(self):
        return None

    def velocities(self):
        """The Keplerian at the epochs: here the measured table itself, which is a fit."""
        if self._table is None:
            return np.zeros((len(self._templates), 1))
        return np.asarray(self._table.velocity, dtype=float)

    def orbit(self):
        return {"period": 6.31, "t_conj": 2455000.5, "ecc": 0.15, "omega": 0.7}

    def star(self, name):
        return {"k": 30.0 if name == self.dis.component_names[0] else 55.0}

    def measure_velocities(self, *, templates, light):
        self.measured = (list(templates), light)
        return self._table


class _StubMatch:
    """Enough of a label match: the fitted frame offsets, the two nulls, the posterior widths.

    ``widths`` are the ``posterior_over_prior`` ratios, one per site. Left out, the
    attribute is absent, as it is on a match written by an older release.
    """

    def __init__(self, offsets, *, chi2=1.0e6, nearest=2.0e6, continuum=3.0e6, widths=None):
        self.labels = {n: {"teff": 5000.0, "v_kms": v} for n, v in offsets.items()}
        self.chi2 = chi2
        self.chi2_nearest_node = nearest
        self.chi2_continuum = continuum
        if widths is not None:
            self.posterior_over_prior = dict(widths)


def _velocity_table(*, n_epochs=6, chi2=1000.0, chi2_null=2000.0, blended=False):
    """A hand-built ``VelocityTable``; ``chi2 > chi2_null`` makes its R-squared negative."""
    from albireo.todcor import VelocityTable

    ones = np.ones(n_epochs)
    phase = np.linspace(0.0, 1.0, n_epochs, endpoint=False)
    return VelocityTable(
        names=("A", "B"),
        bjd=2455000.0 + np.arange(n_epochs, dtype=float),
        instrument=("TOY",) * n_epochs,
        velocity=np.vstack([30.0 * np.sin(2 * np.pi * phase), -55.0 * np.sin(2 * np.pi * phase)]),
        sigma=np.vstack([ones, ones]),
        sigma_ivar=np.vstack([ones, ones]),
        covariance=np.tile(np.eye(2), (n_epochs, 1, 1)),
        light=np.vstack([0.62 * ones, 0.38 * ones]),
        light_mode="fixed",
        chi2=chi2 * ones,
        chi2_null=chi2_null * ones,
        n_pixels=1000.0 * ones,
        delta_chi2=np.vstack([400.0 * ones, 400.0 * ones]),
        blended=np.full(n_epochs, blended),
        at_edge=np.zeros((2, n_epochs), dtype=bool),
        refined=np.ones(n_epochs, dtype=bool),
        absolute=(False, False),
        frame="barycentric",
        settings={"n_parameters": 3},
    )


def test_a_zero_point_the_label_fit_disowned_is_refused(toy, tmp_path):
    """Three ways for the label fit to report a frame offset it did not measure."""
    from albireo.pipeline import _templates

    dataset, truth, grid = toy
    star = _star(dataset, truth, grid, name="zeros")
    settings = Analysis(v_zero_range=150.0)  # trial step max(2 x 5, 5, 2 x 150 / 120) = 10 km/s

    cases = []

    def run(match, **kwargs):
        ctx = _context(star, tmp_path / f"case{len(cases)}", settings)
        cases.append(ctx)
        return ctx, _templates(ctx, _StubFit(("A", "B"), **kwargs), match)

    # A fit that beat both nulls, with both offsets far from the scan's bounds, is adopted.
    ctx, templates = run(_StubMatch({"A": 12.4, "B": 11.6}))
    assert [t.v_zero_kms for t in templates] == [12.4, 11.6]
    assert ctx.zero_points["adopted"] is True and not ctx.zero_points["refused"]
    assert ctx.zero_points["v_zero_kms"] == {"A": 12.4, "B": 11.6}
    assert not ctx.flags

    # (i) the fit beats neither null: chi2 above both, as in the archived star.
    ctx, templates = run(_StubMatch({"A": 12.4, "B": 11.6}, chi2=1.15e7, nearest=7.0e6))
    assert [t.v_zero_kms for t in templates] == [None, None]
    assert ctx.zero_points["adopted"] is False
    assert any("beats neither null" in f for f in ctx.flags), ctx.flags

    # (ii) an offset pinned on the bound of its own scan (the archived star: 3e-4 km/s from
    # -150 over a +-150 km/s scan).
    ctx, templates = run(_StubMatch({"A": -149.9997, "B": 11.6}))
    assert [t.v_zero_kms for t in templates] == [None, None]
    reason = " ".join(ctx.zero_points["refused"])
    assert "'A'" in reason and "-149.9997" in reason and "bound of its scan" in reason

    # (iii) two zero points that no single systemic velocity can hold, on a Keplerian fit.
    ctx, templates = run(_StubMatch({"A": -60.0, "B": 60.0}), budget=100.0)
    assert [t.v_zero_kms for t in templates] == [None, None]
    assert any("velocity budget" in f for f in ctx.flags), ctx.flags
    # The same pair on a free-velocity fit is not a contradiction: there is no shared gamma.
    ctx, templates = run(_StubMatch({"A": -60.0, "B": 60.0}), budget=100.0, mode="velocity")
    assert [t.v_zero_kms for t in templates] == [-60.0, 60.0]

    # Every refusal is in the log as well as in the flags.
    text = (cases[1].directory / "log.txt").read_text(encoding="utf-8")
    assert "flag: template zero points refused" in text


def test_a_frame_offset_the_label_fit_never_learned_is_refused_for_that_component(toy, tmp_path):
    """A posterior as wide as its prior: that component alone keeps its unidentified zero point.

    The three refusals above are properties of the fit as a whole. This one is per
    component: the label fit of a third benchmark run kept 11 percent of the secondary's
    equivalent width, its ``v`` site came back as wide as the prior it started from, the
    offset it reported sat 46 km/s from the systemic velocity, and every velocity of that
    component carried it. The other component's offset was measured and is still applied.
    """
    from albireo.pipeline import _templates

    dataset, truth, grid = toy
    star = _star(dataset, truth, grid, name="unlearned")
    settings = Analysis(v_zero_range=150.0)

    # B learned nothing (0.93 of its prior), A did (0.20): B stays differential, A is pinned.
    ctx = _context(star, tmp_path / "one", settings)
    offsets = {"A": 12.4, "B": -46.0}
    templates = _templates(
        ctx, _StubFit(("A", "B")), _StubMatch(offsets, widths={"v_A": 0.2, "v_B": 0.93})
    )
    assert [t.v_zero_kms for t in templates] == [12.4, None]
    assert templates[0].meta["zero_point"] == "label match"
    assert "zero_point" not in templates[1].meta
    (flag,) = ctx.flags
    assert flag.startswith("zero point refused for B (frame offset posterior 0.93 of the prior)")
    assert "that component's velocities stay differential" in flag
    assert ctx.zero_points["refused"] == ["B: the label fit learned nothing about the frame offset"]
    # One component pinned is still an adopted zero point, and B's offset is recorded,
    # unapplied, beside the reason it was refused.
    assert ctx.zero_points["adopted"] is True
    assert ctx.zero_points["v_zero_kms"] == offsets
    log = (ctx.directory / "log.txt").read_text(encoding="utf-8")
    assert "template zero points from the label fit: A +12.40 km/s, B none" in log

    # Neither component learned anything: nothing is pinned and nothing is adopted.
    ctx = _context(star, tmp_path / "both", settings)
    templates = _templates(
        ctx, _StubFit(("A", "B")), _StubMatch(offsets, widths={"v_A": 0.95, "v_B": 0.93})
    )
    assert [t.v_zero_kms for t in templates] == [None, None]
    assert ctx.zero_points["adopted"] is False
    assert len(ctx.zero_points["refused"]) == 2
    assert "A (frame offset posterior 0.95 of the prior)" in ctx.flags[0]

    # A match that reports no widths at all pins both, as before the test existed.
    ctx = _context(star, tmp_path / "silent", settings)
    templates = _templates(ctx, _StubFit(("A", "B")), _StubMatch(offsets))
    assert [t.v_zero_kms for t in templates] == [12.4, -46.0]
    assert ctx.zero_points["adopted"] is True and not ctx.zero_points["refused"]
    assert not ctx.flags


def test_zero_points_no_search_window_can_hold_are_dropped_and_the_epochs_measured(toy, tmp_path):
    """Two zero points 95 km/s apart on velocities spanning 60: the zero points go, not the star.

    The default search window of a template is its own zero point away from the fitted
    velocities, so zero points that disagree by more than those velocities span leave no
    window that holds every component, and the correlation refuses rather than report a
    table pinned to the edge of its search. The velocities then stay differential, as when
    the label fit disowns its own offsets, and the orbit fit carries one systemic velocity
    per component.
    """
    from albireo.pipeline import _measure_epoch_velocities

    dataset, truth, grid = toy
    star = _star(dataset, truth, grid, name="dropped")
    ctx = _context(star, tmp_path / "dropped")
    ctx.zero_points = {
        "source": "label match",
        "adopted": True,
        "v_zero_kms": {"A": -60.0, "B": 35.0},
        "refused": [],
    }
    bare = _StubFit(("A", "B")).templates()
    pinned = [replace(t, v_zero_kms=z) for t, z in zip(bare, (-60.0, 35.0), strict=True)]

    class _Fit:
        """A fit whose correlation refuses a window while any template carries a zero point."""

        def __init__(self):
            self.calls = []

        def measure_velocities(self, *, templates, light):
            self.calls.append([t.v_zero_kms for t in templates])
            if any(t.v_zero_kms is not None for t in templates):
                raise ValueError(
                    "component 'B': the default search window (-95.000, +25.000) km/s "
                    "misses its own fitted velocities"
                )
            return _velocity_table()

    fit = _Fit()
    table, used = _measure_epoch_velocities(ctx, fit, pinned, [0.62, 0.38])
    assert fit.calls == [[-60.0, 35.0], [None, None]]  # refused, then measured again
    assert [t.v_zero_kms for t in used] == [None, None]
    assert all(t.meta["zero_point"] == "dropped" for t in used)
    assert table.n_epochs == 6 and not all(table.absolute)
    assert ctx.zero_points["adopted"] is False
    assert ctx.zero_points["v_zero_kms"] == {"A": -60.0, "B": 35.0}  # what was dropped is kept
    assert "search window" in " ".join(ctx.zero_points["refused"])
    assert any(f.startswith("template zero points dropped") for f in ctx.flags), ctx.flags
    assert "one gamma per component" in ctx.flags[0]

    # A failure with no zero point to drop is a real one, and is raised as it stands.
    class _Fails:
        def measure_velocities(self, *, templates, light):
            raise ValueError("the correlation found no usable epoch")

    other = _context(star, tmp_path / "raised")
    with pytest.raises(ValueError, match="no usable epoch"):
        _measure_epoch_velocities(other, _Fails(), bare, [0.62, 0.38])
    assert not other.flags and other.zero_points is None


def test_the_declared_fractions_are_held_only_while_the_smoothness_stayed_near_its_start(
    toy, tmp_path
):
    """A smoothness ML-II moved an order of magnitude: the amplitudes are fitted, not held.

    The disentangling recovers a component at the declared light fraction only while its
    posterior mean is not shrunk, and a smoothness precision far from its start says it
    is. A secondary of a third benchmark run lost a quarter of its line depth that way and
    the table then lost it under the declared fraction, at -89 percent against the
    injected semi-amplitude, where a freely fitted amplitude left -25.
    """
    from albireo.facade import _smoothness_of
    from albireo.pipeline import _template_light

    dataset, truth, grid = toy
    star = _star(dataset, truth, grid, name="smooth")
    declared = list(LIGHT)

    # tau 400 against the 300 the stellar default starts at: the fractions stand.
    ctx = _context(star, tmp_path / "held")
    held = _StubFit(("A", "B"), lights=LIGHT, taus=(400.0, 400.0))
    assert [float(_smoothness_of(s).tau0) for s in held.dis.stars] == [300.0, 300.0]
    assert _template_light(ctx, held, declared) == declared
    assert not ctx.flags

    # One component 470 times its start: the correlation fits both amplitudes instead.
    ctx = _context(star, tmp_path / "moved")
    moved = _StubFit(("A", "B"), lights=LIGHT, taus=(400.0, 141452.0))
    assert _template_light(ctx, moved, declared) == "global"
    (flag,) = ctx.flags
    assert "B 300 -> 1.41e+05" in flag and "A 300" not in flag  # only B moved
    assert "moved from its start by more than a factor 10" in flag
    assert "the table's light column is that scale, not a light fraction" in flag


def test_the_orbit_exchanges_the_components_only_where_they_are_alike(toy, tmp_path):
    """Equal peaks are exchangeable; peaks a factor of several apart are not.

    On a 95/5 pair the exchange swapped 19 of 80 epochs on noise draws of the faint
    component's velocity, and the primary's semi-amplitude went from 5 to 56 percent off,
    because an exchanged row carries the other component's amplitude.
    """
    from albireo.pipeline import _exchange_allowed

    dataset, truth, grid = toy
    star = _star(dataset, truth, grid, name="pairs")

    for i, fractions in enumerate(([0.5, 0.5], [0.7, 0.3])):
        ctx = _context(star, tmp_path / f"alike{i}")
        assert _exchange_allowed(ctx, fractions) is True
        assert not ctx.flags

    ctx = _context(star, tmp_path / "apart")
    assert _exchange_allowed(ctx, [0.95, 0.05]) is False
    (flag,) = ctx.flags
    assert "the exchange of the two components by the orbit was skipped" in flag
    assert "[0.95, 0.05] differ by a factor 19.0, above 3" in flag

    # Nothing to exchange, or nothing to compare: the rule does not apply and does not flag.
    ctx = _context(star, tmp_path / "single")
    assert _exchange_allowed(ctx, [1.0]) is True
    assert _exchange_allowed(ctx, [0.95, float("nan")]) is True
    assert not ctx.flags


def test_the_table_as_measured_is_kept_where_the_orbit_exchanged_an_epoch(
    toy, tmp_path, monkeypatch
):
    """The velocity stage, with the fit and the products stubbed: the exchange, and its record.

    What separates a genuine exchange from a swap made on noise is the table the
    correlation actually measured, so the stage writes it beside the delivered one
    whenever an epoch was exchanged.
    """
    import albireo.pipeline as pipeline_module
    import albireo.rvorbit as rvorbit_module
    from albireo.pipeline import _run_stages

    dataset, truth, grid = toy
    measured = _velocity_table()
    swapped_at = 2
    calls = []

    def swap_one(table, predicted, **kwargs):
        """Exchange one epoch, as ``reassign_by_orbit`` would where the orbit says so."""
        calls.append(np.asarray(predicted, dtype=float).shape)
        velocity = np.asarray(table.velocity, dtype=float).copy()
        velocity[:, swapped_at] = velocity[::-1, swapped_at]
        mask = np.zeros(table.n_epochs, dtype=bool)
        mask[swapped_at] = True
        return replace(table, velocity=velocity), mask

    def run(lights, name):
        star = replace(  # no truth block: the stub fit has no spectra to compare
            _star(
                dataset,
                truth,
                grid,
                name=name,
                components=[
                    ComponentConfig("A", lights[0], teff=(4200.0, 5700.0)),
                    ComponentConfig("B", lights[1], teff=(4100.0, 5200.0)),
                ],
            ),
            truth=None,
        )
        ctx = _context(star, tmp_path / name, Analysis(plots=False))
        fit = _StubFit(("A", "B"), lights=lights, taus=(400.0, 400.0), table=measured)
        monkeypatch.setattr(pipeline_module, "_declare", lambda *args, **kwargs: fit.dis)
        monkeypatch.setattr(pipeline_module, "_write_products", lambda *args, **kwargs: None)
        monkeypatch.setattr(pipeline_module, "_orbit", lambda *args, **kwargs: (None, None))
        monkeypatch.setattr(rvorbit_module, "reassign_by_orbit", swap_one)
        report, _text, live = _run_stages(ctx)
        return ctx, report, live

    # Alike components: the exchange runs, and the table as measured is kept beside it.
    ctx, report, live = run([0.5, 0.5], "alike")
    assert calls == [(2, measured.n_epochs)]
    assert live["velocities"].velocity[0][swapped_at] == measured.velocity[1][swapped_at]
    assert any("had their components re-assigned" in f for f in ctx.flags), ctx.flags
    path = Path(ctx.files["velocities_unexchanged"])
    assert path.name == "velocities_unexchanged.rv" and path.parent == ctx.directory
    rows = np.loadtxt(path, comments="#", usecols=(0, 2, 4))
    np.testing.assert_allclose(rows[:, 1], measured.velocity[0], atol=1e-6)
    np.testing.assert_allclose(rows[:, 2], measured.velocity[1], atol=1e-6)
    assert "as measured, before the exchange" in path.read_text(encoding="utf-8")
    assert report["velocities"]["names"] == ["A", "B"]

    # A 95/5 pair: the exchange never runs, so there is no table to keep beside anything.
    calls.clear()
    ctx, _report, live = run([0.95, 0.05], "apart")
    assert calls == []
    np.testing.assert_allclose(live["velocities"].velocity, measured.velocity)
    assert any(
        "the exchange of the two components by the orbit was skipped" in f for f in ctx.flags
    )
    assert "velocities_unexchanged" not in ctx.files
    assert not (ctx.directory / "velocities_unexchanged.rv").exists()


def test_the_bootstrap_table_the_period_search_ran_on_is_kept_beside_the_delivered_one(
    toy, tmp_path
):
    """The bootstrap's template table is written twice where the winning orbit re-assigned.

    ``_orbit_over_candidates`` re-assigns the components at the epochs where a per-epoch
    correlation exchanged two alike spectra, and refits, so the table it returns beside the
    winning orbit is not the table the period search ran on. The delivered file stays the
    winning orbit's and its header says whose assignment it carries; the table as measured
    is written beside it, because only that one reproduces the periodogram.
    """
    from albireo.pipeline import _write_template_table

    dataset, truth, grid = toy
    star = _star(dataset, truth, grid, name="bootstrapped")
    measured = _velocity_table()
    swapped_at = 2
    velocity = np.asarray(measured.velocity, dtype=float).copy()
    velocity[:, swapped_at] = velocity[::-1, swapped_at]
    delivered = replace(
        measured,
        velocity=velocity,
        settings={**dict(measured.settings), "reassigned_by_orbit": 1},
    )

    ctx = _context(star, tmp_path / "exchanged")
    _write_template_table(ctx, delivered, "bootstrap", unexchanged=measured)
    path = Path(ctx.files["template_velocities_unexchanged"])
    assert path.name == "template_velocities_unexchanged.rv" and path.parent == ctx.directory
    text = path.read_text(encoding="utf-8")
    assert "as measured, before the exchange: the table the period search ran on" in text
    assert "purpose: bootstrap" in text

    # The delivered file holds what it always held, and now says whose assignment it is.
    delivered_path = Path(ctx.files["template_velocities"])
    delivered_text = delivered_path.read_text(encoding="utf-8")
    assert "component assignment: the winning orbit's, not the correlation's" in delivered_text
    assert "1 of 6 epochs re-assigned" in delivered_text
    assert Path(ctx.files["template_velocities_csv"]).exists()

    # The two differ at the re-assigned epoch, by the exchange, and nowhere else.
    rows = np.loadtxt(delivered_path, comments="#", usecols=(0, 2, 4))
    kept = np.loadtxt(path, comments="#", usecols=(0, 2, 4))
    np.testing.assert_allclose(rows[:, 0], kept[:, 0])  # the same epochs in the same order
    np.testing.assert_allclose(rows[swapped_at, 1:], kept[swapped_at, 1:][::-1], atol=1e-6)
    others = [j for j in range(measured.n_epochs) if j != swapped_at]
    np.testing.assert_allclose(rows[others, 1:], kept[others, 1:], atol=1e-6)
    np.testing.assert_allclose(kept[:, 1], measured.velocity[0], atol=1e-6)
    np.testing.assert_allclose(kept[:, 2], measured.velocity[1], atol=1e-6)

    # No epoch re-assigned: there is nothing to keep, and no line claiming otherwise.
    ctx = _context(star, tmp_path / "as-measured")
    _write_template_table(ctx, measured, "bootstrap", unexchanged=measured)
    assert "template_velocities_unexchanged" not in ctx.files
    assert not (ctx.directory / "template_velocities_unexchanged.rv").exists()
    held = Path(ctx.files["template_velocities"]).read_text(encoding="utf-8")
    assert "component assignment" not in held and "re-assigned" not in held


def test_a_failed_velocity_table_is_marked_and_no_orbit_is_fitted(toy, tmp_path):
    """A median R-squared below zero: the templates fit worse than no template at all."""
    from albireo.pipeline import _assess_table, _describe_table, _orbit, _velocity_header

    dataset, truth, grid = toy
    star = _star(dataset, truth, grid, name="failed")

    healthy = _velocity_table(chi2=1000.0, chi2_null=2000.0)
    described = _describe_table(healthy)
    assert described["status"] == "ok" and described["failure"] is None
    assert described["r_squared_median"] == pytest.approx(0.5)

    failed = _velocity_table(chi2=2000.0, chi2_null=1000.0)
    described = _describe_table(failed)
    assert described["status"] == "failed"
    assert "median R-squared -1.00" in described["failure"]

    ctx = _context(star, tmp_path / "failed")
    _assess_table(ctx, failed)
    assert any("the velocity table failed" in f for f in ctx.flags), ctx.flags
    assert any("velocities are differential" in f for f in ctx.flags), (
        "the existing guards still run"
    )

    class _NoFit:
        mode = "keplerian"

        def orbit(self):
            raise AssertionError("the orbit stage must not read a failed table")

    assert _orbit(ctx, _NoFit(), failed) == (None, None)
    assert any("orbit from the table skipped" in f for f in ctx.flags), ctx.flags

    # The written file says so on the line under the format line, before any row.
    path = failed.write(ctx.directory / "velocities.rv", header=_velocity_header(ctx, failed))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# albireo.todcor velocity table"
    assert lines[1].startswith("# FAILED: median R-squared -1.00")
    assert lines[2] == "# star: failed"
    # A usable table carries no such line.
    ok_path = healthy.write(ctx.directory / "ok.rv", header=_velocity_header(ctx, healthy))
    assert "FAILED" not in ok_path.read_text(encoding="utf-8")

    # A table with no usable epoch fails on that count instead.
    empty = _velocity_table(chi2=1000.0, chi2_null=2000.0, blended=True)
    assert _describe_table(empty)["failure"] == "no usable epoch of 6"


def test_an_unmeasured_epoch_in_a_declared_table_is_refused(toy, tmp_path):
    """`velocities = "file"` with a nan row: the free fit has no site for an unmeasured epoch."""
    from albireo.pipeline import _read_velocities

    dataset, truth, grid = toy
    rows = np.column_stack([dataset.bjd, np.asarray(truth.velocities).T])
    path = tmp_path / "rv.txt"
    np.savetxt(path, rows, header="bjd v_A v_B")
    star = _star(dataset, truth, grid, period=None, velocities=str(path), labels=False)
    np.testing.assert_allclose(_read_velocities(star, dataset), np.asarray(truth.velocities))

    rows[3, 2] = np.nan  # the correlation stage measured nothing for B at that epoch
    np.savetxt(path, rows, header="bjd v_A v_B")
    with pytest.raises(ValueError, match=r"component 'B' at epoch 3") as excinfo:
        _read_velocities(star, dataset)
    assert "no site for an unmeasured one" in str(excinfo.value)
    assert f"{dataset.bjd[3]:.5f}" in str(excinfo.value)


def test_a_diverged_disentangling_stops_the_star(toy, library, tmp_path, monkeypatch):
    """The one number that separates a fit from a failure of the optimizer stops the star."""
    from albireo.facade import Fit

    with pytest.raises(ValueError, match="z_rms_max must be positive"):
        Analysis(z_rms_max=0.0)
    assert Analysis().z_rms_max == 10.0

    dataset, truth, grid = toy
    monkeypatch.setattr(Fit, "z_rms", property(lambda self: 45.0))
    star = _star(dataset, truth, grid, name="diverged", overrides={"k_max": 90.0, "max_steps": 2})
    config = PipelineConfig(
        stars=[star],
        output=tmp_path,
        library=library,
        mh=(-0.9, 0.4),
        analysis=Analysis(fast=True, plots=False, v_zero_range=40.0, v_range=60.0),
    )
    run = run_pipeline(config, progress=False)
    result = run.results["diverged"]
    assert result.status == "failed" and "diverged" in run.failures
    assert "RuntimeError" in result.error
    for phrase in ("z-score rms 45.0", "z_rms_max ceiling of 10", "were not run", "noise model"):
        assert phrase in result.error, result.error

    directory = Path(result.directory)
    assert (directory / "error.txt").is_file()
    assert not (directory / "velocities.rv").exists(), "the velocity stage did not run"
    assert not (directory / "labels.txt").exists(), "the label stage did not run"
    # The table measured before the disentangling stays on disk, as its own product.
    assert (directory / "template_velocities.rv").is_file()
    log = (directory / "log.txt").read_text(encoding="utf-8")
    assert "flag: the disentangling diverged" in log
    assert "FAILED: RuntimeError" in log


# ---------------------------------------------------------------------------
# The label stage and the epoch comparison (D65)
# ---------------------------------------------------------------------------


def test_the_label_comparison_is_declared_checked_and_passed_to_the_fit(toy, library, tmp_path):
    """``label_compare`` defaults to the epochs, is refused by name, and reaches the fit."""
    from albireo.pipeline import _labels, config_template

    assert Analysis().label_compare == "epochs"
    with pytest.raises(ValueError, match="label_compare must be one of 'epochs', 'native'"):
        Analysis(label_compare="diagonal")
    one_star = {
        "stars": [
            {
                "name": "x",
                "spectra": "x/*.fits",
                "period": 6.3,
                "components": [{"name": "A", "light": 0.6}, {"name": "B", "light": 0.4}],
            }
        ]
    }
    loaded = config_from_dict({**one_star, "labels": {"compare": "native", "steps": 40}})
    assert loaded.analysis.label_compare == "native" and loaded.analysis.label_steps == 40
    direct = config_from_dict({**one_star, "analysis": {"label_compare": "matched"}})
    assert direct.analysis.label_compare == "matched"
    with pytest.raises(ValueError, match="label_compare must be one of"):
        config_from_dict({**one_star, "labels": {"compare": "diagonal"}})
    assert '# compare = "epochs"' in config_template()

    dataset, truth, grid = toy
    star = _star(dataset, truth, grid, name="compare")
    seen = {}

    class _Fit(_StubFit):
        def match_labels(self, stars, **options):
            seen.update(options)
            return _EpochMatch({"A": LIGHT[0], "B": LIGHT[1]})

    for compare, budget in (("epochs", "70 Levenberg-Marquardt iterations per run"),
                            ("matched", "70 steps")):  # fmt: skip
        fit = _Fit(("A", "B"), lights=LIGHT)
        fit.dis.dataset = [SimpleNamespace(medium="air")]
        ctx = _context(
            star, tmp_path / compare, Analysis(label_compare=compare, label_steps=70), library
        )
        assert _labels(ctx, fit, library) is not None
        assert seen["compare"] == compare and seen["max_steps"] == 70
        log = (ctx.directory / "log.txt").read_text(encoding="utf-8")
        assert f"label fit ({compare} comparison)" in log and budget in log


class _EpochMatch(_StubMatch):
    """A label match that ran the epoch comparison: bounds, notes, light and its errors."""

    def __init__(self, flux, *, errors=None, at_bounds=None, notes=(), widths=None):
        super().__init__({name: 12.0 for name in flux}, widths=widths or {})
        self.names = tuple(flux)
        self.compare = "epochs"
        self.flux_ratio = dict(flux)
        self.flux_ratio_errors = dict(errors or {})
        self.at_bounds = dict(at_bounds or {})
        self.epoch_fit = SimpleNamespace(notes=tuple(notes))
        self.multimodal = False


def test_the_label_stage_flags_what_the_epoch_fit_did_not_measure(toy, tmp_path):
    """A label on a bound has no posterior width, so the width test cannot see it.

    In the epochs comparison a site on a bound, or a v sin i on the plateau below half a
    model pixel, is left out of the formal covariance, and ``posterior_over_prior`` has no
    entry for it; the flag names it with the reason. A collapse the restarts could not
    resolve is a note on the fit, and it travels into the flags as it stands.
    """
    from albireo.pipeline import _assess_labels

    dataset, truth, grid = toy
    star = _star(dataset, truth, grid, name="bounds")
    declared = {"A": 0.62, "B": 0.38}
    note = "'B' looks collapsed after the joint restart (its light fraction is 0.0016)"

    ctx = _context(star, tmp_path / "held")
    match = _EpochMatch(
        {"A": 0.62, "B": 0.38},
        errors={"A": 0.004, "B": 0.004},
        at_bounds={
            "vsini_B": "rotation plateau",
            "logg_B": "lower bound",
            "offset_A[0]": "no curvature",
        },
        notes=(note,),
        widths={"teff_B": 0.9, "offset_B": 0.95},
    )
    _assess_labels(ctx, match, declared)
    assert "labels learned nothing about teff_B (posterior width >= 80% of the prior)" in ctx.flags
    (held,) = [f for f in ctx.flags if f.startswith("labels with no formal error")]
    assert "logg_B (lower bound), vsini_B (rotation plateau)" in held
    assert "offset" not in held.split(".")[0], "the additive offsets are not labels"
    assert f"labels: {note}" in ctx.flags
    assert not any("light fractions" in f for f in ctx.flags), "the light agrees"

    # A d_hat comparison carries none of the epoch fit's attributes and raises none of these.
    ctx = _context(star, tmp_path / "native")
    plain = _StubMatch({"A": 12.0, "B": 12.0}, widths={})
    plain.flux_ratio, plain.multimodal = {"A": 0.6, "B": 0.4}, False
    _assess_labels(ctx, plain, declared)
    assert not ctx.flags


def test_the_label_stage_flags_and_records_the_grid_compensation(toy, tmp_path):
    """A floored compensation, or a model grid coarser than sigma_q, is flagged as it stands,
    and the operator widths and the grid variance are recorded for result.json."""
    from albireo.pipeline import _assess_labels, _describe_operator

    dataset, truth, grid = toy
    star = _star(dataset, truth, grid, name="grid")
    floored = (
        "grid compensation floored for TOY: the model grid's discretisation variance (7/12) "
        "dv^2 = 12.48 km^2/s^2 at dv = 4.626 km/s leaves 0.000 km/s of sigma_q = 3.000 km/s"
    )
    statistics = SimpleNamespace(
        grid=SimpleNamespace(dv_kms=4.626),
        library_resolving_power=20_000.0,
        grid_variance_kms2=12.48,
        lsf_sigma_kms=(("TOY", (3.0,)),),
        operator_sigma_kms=(("TOY", (2.313,)),),
        notes=(floored,),
    )
    match = _EpochMatch({"A": 0.62, "B": 0.38}, errors={"A": 0.004, "B": 0.004})
    match.statistics = statistics
    ctx = _context(star, tmp_path / "floored")
    _assess_labels(ctx, match, {"A": 0.62, "B": 0.38})
    assert f"labels: {floored}" in ctx.flags
    described = _describe_operator(statistics)
    assert described == {
        "dv_kms": 4.626,
        "library_resolving_power": 20_000.0,
        "grid_variance_kms2": 12.48,
        "quadrature_sigma_kms": [["TOY", [3.0]]],
        "operator_sigma_kms": [["TOY", [2.313]]],
        "notes": [floored],
    }
    json.dumps(described)  # result.json takes it as it stands

    # Without notes nothing is flagged, and without the compensation the widths coincide.
    statistics = SimpleNamespace(
        grid=SimpleNamespace(dv_kms=2.0),
        library_resolving_power=None,
        grid_variance_kms2=0.0,
        lsf_sigma_kms=(("TOY", (5.5,)),),
        operator_sigma_kms=(),
        notes=(),
    )
    match.statistics = statistics
    ctx = _context(star, tmp_path / "quiet")
    _assess_labels(ctx, match, {"A": 0.62, "B": 0.38})
    assert not any("grid compensation" in f for f in ctx.flags), ctx.flags
    assert _describe_operator(statistics)["operator_sigma_kms"] == [["TOY", [5.5]]]


@pytest.mark.parametrize(
    ("declared", "measured", "off"),
    [
        ((0.62, 0.38), (0.60, 0.40), ""),  # inside both limits
        ((0.62, 0.38), (0.67, 0.33), ""),  # factors 1.08 and 1.15, differences of 0.05
        ((0.90, 0.10), (0.84, 0.16), "B off by a factor 1.60"),  # the factor alone
        ((0.45, 0.55), (0.62, 0.38), "A off by a factor 1.38, B off by a factor 1.45"),  # 0.17
        ((0.62, 0.38), (0.93, 0.07), "A off by a factor 1.50, B off by a factor 5.43"),
    ],
)
def test_a_measured_light_far_from_the_declared_one_is_flagged(
    toy, tmp_path, declared, measured, off
):
    """A factor of 1.5 catches a faint component; a difference of 0.15 a bright one."""
    from albireo.pipeline import _assess_labels

    dataset, truth, grid = toy
    star = _star(dataset, truth, grid, name="light")
    ctx = _context(star, tmp_path / "x")
    match = _EpochMatch(dict(zip("AB", measured, strict=True)), errors={"A": 0.01, "B": 0.01})
    _assess_labels(ctx, match, dict(zip("AB", declared, strict=True)))
    flags = [f for f in ctx.flags if f.startswith("the label fit measures light fractions")]
    assert bool(flags) is bool(off), ctx.flags
    if off:
        (flag,) = flags
        assert f"A {measured[0]:.3f} +- 0.010, B {measured[1]:.3f} +- 0.010" in flag
        assert f"against the declared {declared[0]:.3f}, {declared[1]:.3f} ({off};" in flag
        assert "flagged above a factor 1.5 or a difference of 0.15" in flag
        assert "declared in the wrong order" in flag
        assert "measures the light without reference to the declaration" in flag


def test_the_velocity_stage_records_where_the_amplitudes_came_from(toy, tmp_path, monkeypatch):
    """The declared fractions, or the global re-measure; never the label fit's light.

    The disentangled components are ``(w / l0) t``, so the amplitude that reproduces the
    epochs with them is the declared ``l0`` (``test_match_epochs`` measures what holding
    the measured light instead does to the velocities).
    """
    import albireo.pipeline as pipeline_module
    from albireo.pipeline import _run_stages

    dataset, truth, grid = toy
    measured = _velocity_table()

    def run(taus, name):
        star = replace(_star(dataset, truth, grid, name=name), truth=None)
        ctx = _context(star, tmp_path / name, Analysis(plots=False))
        fit = _StubFit(("A", "B"), lights=LIGHT, taus=taus, table=measured)
        monkeypatch.setattr(pipeline_module, "_declare", lambda *args, **kwargs: fit.dis)
        monkeypatch.setattr(pipeline_module, "_write_products", lambda *args, **kwargs: None)
        monkeypatch.setattr(pipeline_module, "_orbit", lambda *args, **kwargs: (None, None))
        report, _text, _live = _run_stages(ctx)
        return ctx, report, fit

    ctx, report, fit = run((400.0, 400.0), "held")
    assert report["velocities"]["light_source"] == "declared"
    assert fit.measured[1] == list(LIGHT)
    assert "amplitudes declared" in (ctx.directory / "log.txt").read_text(encoding="utf-8")

    ctx, report, fit = run((400.0, 141452.0), "moved")
    assert report["velocities"]["light_source"] == "global re-measure"
    assert fit.measured[1] == "global"

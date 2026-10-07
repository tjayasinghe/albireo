"""Tests for the Gaia RVS benchmark harness (``albireo.benchmark``).

The fast tests assert three properties: a system under each tier becomes the pipeline
declaration the tier describes, the collector and the summary read what the pipeline
writes, and the report is produced from rows alone. The slow test runs two systems through
the whole chain on the toy library.
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

from albireo.benchmark import (
    TIERS,
    BenchmarkConfig,
    Tier,
    build_star,
    build_stars,
    collect,
    collect_velocities,
    resolve_tier,
    run_benchmark,
    simulate_system,
    summarize,
    summarize_velocities,
    system_transit_times,
    write_report,
)
from albireo.facade import Between, Fixed, Known
from albireo.gaia import (
    RVS_DR4_EPOCH,
    RVS_SPANS,
    GostTransits,
    rvs_delivered_sigma_kms,
    rvs_lsf_sigma_kms,
    rvs_transit_times_from_gost,
)
from albireo.grids import C_KMS
from albireo.pipeline import _lsf, _spec
from albireo.population import BinarySystem, draw_population, write_population
from albireo.simulate import synthetic_library

HAS_MPL = importlib.util.find_spec("matplotlib") is not None


@pytest.fixture(scope="module")
def library():
    return synthetic_library(
        (8400.0, 8760.0),
        n_pix=1600,
        teff=(4000.0, 4500.0, 5000.0, 5500.0, 6000.0, 6500.0, 7000.0),
        logg=(3.5, 4.0, 4.5, 5.0),
        mh=(-0.5, 0.0, 0.5),
    )


@pytest.fixture(scope="module")
def systems(library):
    drawn = draw_population(3, kind="mixed", seed=11, library=library, period_range=(2.0, 30.0))
    # Force one eclipsing and one not, and enough transits for the minimum.
    out = []
    for i, s in enumerate(drawn):
        out.append(
            BinarySystem.from_dict(
                {**s.to_dict(), "eclipsing": i == 0, "n_transits": 14 + 4 * i, "grvs": 8.0 + i}
            )
        )
    return out


def test_tiers_are_declared_consistently():
    assert set(TIERS) == {"oracle", "eclipsing", "orbit", "blind"}
    assert resolve_tier("blind") is TIERS["blind"]
    assert resolve_tier(TIERS["orbit"]) is TIERS["orbit"]
    with pytest.raises(ValueError, match="unknown tier"):
        resolve_tier("guess")
    with pytest.raises(ValueError, match="searched period has no known conjunction"):
        Tier("bad", period="search", t_conj="known")
    with pytest.raises(ValueError, match="light must be"):
        Tier("bad", light="guess")


def test_the_config_resolves_its_arguments(systems):
    config = BenchmarkConfig(output="x", systems=systems, tiers=("blind", "oracle"), product="dr3")
    assert [t.name for t in config.tiers] == ["blind", "oracle"]
    assert config.product.name == "dr3-mean" and config.product_name == "dr3-mean"
    # A GOST cadence cuts its epochs to the span of the release the product belongs to.
    assert config.release == "dr3"
    assert BenchmarkConfig(output="x", systems=systems).release == "dr4"
    with pytest.raises(ValueError, match="product"):
        BenchmarkConfig(output="x", systems=systems, product="dr5")
    with pytest.raises(ValueError, match="cadence"):
        BenchmarkConfig(output="x", systems=systems, cadence="daily")
    with pytest.raises(ValueError, match="unique"):
        BenchmarkConfig(output="x", systems=[systems[0], systems[0]])


def test_each_tier_becomes_the_declaration_it_describes(systems, library):
    system = systems[0]
    dataset, truth, grid, components = simulate_system(
        system, library=library, product=RVS_DR4_EPOCH, seed=1
    )
    assert dataset.n_epochs == system.n_transits and dataset.instruments == ("RVS",)
    assert truth.velocities.shape == (2, system.n_transits)
    config = BenchmarkConfig(output="x", systems=systems, library=library)
    bounds = dict(library.bounds)
    kwargs = dict(
        dataset=dataset, truth=truth, grid=grid, components=components, config=config, bounds=bounds
    )

    oracle = build_star(system, TIERS["oracle"], **kwargs)
    assert oracle.name == f"{system.name}__oracle"
    # The analysis declares the width of the simulated, delivered epochs. This is the nominal
    # line-spread function with the resampling's Delta_det^2 / 6, the detector pixel's width
    # in place of the delivered one, and the simulation's (5/12) dv^2 on its 2 km/s grid
    # added: 11.67 km/s against 11.07 (D65).
    assert grid.dv_kms == pytest.approx(2.0)
    delivered = rvs_delivered_sigma_kms(RVS_DR4_EPOCH, simulation_dv_kms=grid.dv_kms)
    assert oracle.lsf == {"RVS": {"sigma_kms": delivered}}
    assert _lsf(oracle.lsf["RVS"], "RVS").sigma_kms == pytest.approx(11.670, abs=5e-4)
    det = C_KMS * RVS_DR4_EPOCH.detector_step / 8580.0
    prod = C_KMS * RVS_DR4_EPOCH.step / 8580.0
    assert delivered**2 - rvs_lsf_sigma_kms() ** 2 == pytest.approx(
        det**2 / 6.0 + (det**2 - prod**2) / 12.0 + (5.0 / 12.0) * grid.dv_kms**2, rel=1e-12
    )
    assert isinstance(_spec(oracle.period, "p"), Known)
    assert isinstance(_spec(oracle.t_conj, "t"), Known)
    assert isinstance(_spec(oracle.ecc, "e"), Fixed) and oracle.ecc == system.ecc
    assert (oracle.omega is None) == (system.ecc == 0.0)
    assert oracle.components[0].light == pytest.approx(system.light_fractions[0])
    assert isinstance(_spec(oracle.components[0].k, "k"), Known)
    assert isinstance(_spec(oracle.components[0].teff, "teff"), Between)
    assert oracle.truth["k"] == [system.k1, system.k2]
    assert set(oracle.truth["windows"]) == {"ca_triplet", "metal"}
    assert oracle.truth["labels"]["B"]["vsini"] == system.vsini2

    eclipsing = build_star(system, TIERS["eclipsing"], **kwargs)
    assert eclipsing.ecc is None and eclipsing.components[0].k is None
    assert isinstance(_spec(eclipsing.t_conj, "t"), Known)
    assert eclipsing.components[1].light == pytest.approx(system.light_fractions[1])

    orbit = build_star(system, TIERS["orbit"], **kwargs)
    assert orbit.t_conj is None and orbit.measures_light and not orbit.searching
    assert orbit.components[0].teff is None  # the library box

    blind = build_star(system, TIERS["blind"], **kwargs)
    assert blind.searching and blind.measures_light and blind.period == "search"


def test_build_stars_skips_what_does_not_apply(systems, library, tmp_path):
    few = BinarySystem.from_dict({**systems[2].to_dict(), "n_transits": 4, "name": "few"})
    config = BenchmarkConfig(
        output=tmp_path, systems=[*systems, few], library=library, tiers=("eclipsing", "blind")
    )
    stars, records = build_stars(config, progress=False)
    names = [s.name for s in stars]
    # The eclipsing tier applies only to the eclipsing system; the few-transit system is left out.
    assert f"{systems[0].name}__eclipsing" in names
    assert f"{systems[1].name}__eclipsing" not in names
    assert all("few__" not in n for n in names)
    assert set(records) == {s.name for s in systems}
    assert all(
        r["n_epochs"] == s.n_transits
        for r, s in zip((records[s.name] for s in systems), systems, strict=True)
    )


def _canned_forecast(counts):
    """A GOST forecast of a set number of transits per position, all on an RVS CCD row."""

    def forecast(ra_deg, dec_deg, **kwargs):
        n = counts[(round(float(ra_deg), 6), round(float(dec_deg), 6))]
        start, span = RVS_SPANS["dr4"]
        return GostTransits(
            bjd=start + np.linspace(0.0, span, n),
            ra_deg=float(ra_deg),
            dec_deg=float(dec_deg),
            source="test",
            ccd_row=np.full(n, 4),
        )

    return forecast


def test_the_gost_cadence_takes_its_epochs_from_the_forecast(systems, monkeypatch):
    system = systems[0]
    without = BinarySystem.from_dict({**system.to_dict(), "ra_deg": None, "dec_deg": None})
    with pytest.raises(ValueError) as error:
        system_transit_times(without, cadence="gost")
    assert without.name in str(error.value) and "ra_deg" in str(error.value)
    with pytest.raises(ValueError, match="cadence"):
        system_transit_times(system, cadence="nightly")

    key = (round(system.ra_deg, 6), round(system.dec_deg, 6))
    forecast = _canned_forecast({key: 60})
    monkeypatch.setattr("albireo.benchmark.gost_transits", forecast)
    bjd = system_transit_times(system, cadence="gost", seed=2)
    expected = rvs_transit_times_from_gost(forecast(*key), seed=2)
    np.testing.assert_array_equal(bjd, expected)
    # The count is the service's, not the record's, and 0.78 of the forecast is kept.
    assert bjd.size != system.n_transits
    assert 40 <= bjd.size <= 55
    start, span = RVS_SPANS["dr4"]
    assert np.all((bjd >= start) & (bjd <= start + span))
    # The other cadences still place exactly the record's transits.
    assert system_transit_times(system, cadence="scanning-law").size == system.n_transits
    assert system_transit_times(system, cadence="uniform-phase").size == system.n_transits


def test_the_minimum_transit_rule_uses_the_gost_count(systems, library, tmp_path, monkeypatch):
    counts = {
        (round(s.ra_deg, 6), round(s.dec_deg, 6)): n
        for s, n in zip(systems, (40, 40, 6), strict=True)
    }
    monkeypatch.setattr("albireo.benchmark.gost_transits", _canned_forecast(counts))
    config = BenchmarkConfig(
        output=tmp_path, systems=systems, library=library, tiers=("blind",), cadence="gost"
    )
    stars, records = build_stars(config, progress=False)
    # The third system has 14+ transits in its record but only six in the forecast, and
    # about 0.78 of those are kept, so it is below min_transits = 10 and is not simulated.
    assert [s.name for s in stars] == [f"{s.name}__blind" for s in systems[:2]]
    assert records[systems[2].name]["skipped"].endswith("below min_transits = 10")
    assert "snr_epoch" not in records[systems[2].name]
    for system in systems[:2]:
        assert records[system.name]["n_epochs"] != system.n_transits
        assert 25 <= records[system.name]["n_epochs"] <= 40


def _fake_epochs(system: BinarySystem, seed: int = 0):
    """Per-epoch columns: the injected Keplerian plus noise, with one flagged epoch."""
    from albireo.kepler import radial_velocity

    rng = np.random.default_rng(seed)
    n = system.n_transits
    bjd = system.t_conj + np.sort(rng.uniform(0.0, 3.0, n)) * system.period
    truth = {}
    for comp, k, extra in (("A", system.k1, 0.0), ("B", system.k2, np.pi)):
        truth[comp] = np.asarray(
            radial_velocity(
                bjd,
                period=system.period,
                t_peri=system.t_peri,
                ecc=system.ecc,
                omega=system.omega + extra,
                k=k,
                gamma=system.gamma,
            )
        )
    sigma = {"A": np.full(n, 0.8), "B": np.full(n, 1.3)}
    measured = {c: truth[c] + rng.normal(0.0, 1.0, n) * sigma[c] for c in ("A", "B")}
    good = np.ones(n, dtype=bool)
    good[0] = False
    return {
        "names": ["A", "B"],
        "bjd": bjd.tolist(),
        "instrument": ["RVS"] * n,
        "velocity": {c: measured[c].tolist() for c in ("A", "B")},
        "sigma": {c: sigma[c].tolist() for c in ("A", "B")},
        "light": {c: [0.6 if c == "A" else 0.4] * n for c in ("A", "B")},
        "delta_chi2": {c: [30.0] * n for c in ("A", "B")},
        "good": good.tolist(),
        "blended": (~good).tolist(),
        "absolute": {"A": True, "B": True},
        "absolute_all": True,
        "n_usable": int(good.sum()),
    }, {c: truth[c].tolist() for c in ("A", "B")}


def _fake_result(system: BinarySystem, tier: str, *, k_scale=1.01, status="ok"):
    epochs, injected = _fake_epochs(system)
    return {
        "status": status,
        "seconds": {"total": 12.5},
        "flags": ["labels: the scan found a second basin", "velocities are differential: x"],
        "declaration": {"noise_model": {"kind": "ar1", "correlation": {"RVS": 0.3}}},
        "disentangling": {
            "z_rms": 0.98,
            "k_scan": {"n_trials": 500, "gain_nats": 12.0, "used": True},
            "velocities": {c: [0.0] * system.n_transits for c in ("A", "B")},
        },
        "velocities": epochs,
        "bootstrap": {"period": system.period * 1.001} if tier == "blind" else None,
        "truth": {
            "epoch_velocities_true": injected,
            "components_exchanged": False,
            "k_table": {"A": (k_scale - 1) * system.k1, "B": (k_scale - 1) * system.k2},
            "k_table_pull": {"A": 0.5, "B": -1.5},
            "k_disentangling": {"A": 0.1, "B": -0.2},
            "period": 1e-4,
            "elements_table": {
                "period": 1e-4,
                "ecc": 0.01,
                "omega_deg": -3.0,
                "t_conj": 0.02 * system.period,
            },
            "elements_table_pull": {"period": 0.3, "ecc": 0.8, "omega_deg": -0.5, "t_conj": 1.2},
            "elements_disentangling": {"ecc": 0.005},
            "gamma": {"A": 0.2, "B": 0.3},
            "gamma_pull": {"A": 0.4, "B": 0.6},
            "velocity_rms": {"A": 0.8, "B": 1.4},
            "velocity_pull_rms": {"A": 1.05, "B": 1.1},
            "velocity_rms_keplerian": {"A": 0.5, "B": 0.9},
            "labels": {
                "A": {"teff": 40.0, "logg": 0.05, "vsini": 2.0, "mh": 0.03, "teff_pull": 0.7},
                "B": {"teff": -120.0, "logg": -0.1, "vsini": -3.0, "mh": 0.03},
            },
            "light_label_fit": {"A": 0.01, "B": -0.01},
            "light_declared": {"A": 0.0, "B": 0.0},
            "spectra": {
                "A": {
                    "rms": 0.01,
                    "rms_truth": 0.05,
                    "corr": 0.97,
                    "pull_rms": 1.2,
                    "windows": {"ca_triplet": {"ratio": 0.99}, "metal": {"ratio": 1.02}},
                },
                "B": {
                    "rms": 0.02,
                    "rms_truth": 0.04,
                    "corr": 0.85,
                    "pull_rms": 1.5,
                    "windows": {"ca_triplet": {"ratio": 0.9}, "metal": {"ratio": 1.1}},
                },
            },
        },
    }


def test_collect_summarize_and_report_from_written_results(systems, tmp_path):
    config = BenchmarkConfig(
        output=tmp_path, systems=systems, tiers=("orbit", "blind"), library="unused"
    )
    write_population(tmp_path / "population.json", systems)
    plan = [(s, t) for s in systems for t in config.tiers]
    manifest = {
        "albireo": "test",
        "machine": "test",
        "title": "test benchmark",
        "product": "dr4-epoch",
        "product_description": "test",
        "library": "toy",
        "cadence": "scanning-law",
        "resolving_power": "nominal",
        "tiers": [t.__dict__ for t in config.tiers],
        "settings": {
            "dv_kms": 3.0,
            "k_min": 2.0,
            "k_max": 250.0,
            "ecc_max": 0.9,
            "max_steps": 300,
            "label_steps": 300,
            "min_transits": 10,
            "fast": False,
            "seed": 0,
        },
        "windows": {},
        "n_systems": len(systems),
        "n_planned": len(plan),
        "stars": [f"{s.name}__{t.name}" for s, t in plan],
        "skipped_transits": [],
        "simulation": {
            s.name: {"snr_epoch": 20.0, "baseline_days": 1500.0, "lag1": 0.3} for s in systems
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for s, t in plan:
        if s is systems[2] and t.name == "blind":
            continue  # one star left unrun
        directory = tmp_path / f"{s.name}__{t.name}"
        directory.mkdir()
        status = "failed" if (s is systems[1] and t.name == "orbit") else "ok"
        (directory / "result.json").write_text(
            json.dumps(_fake_result(s, t.name, status=status)), encoding="utf-8"
        )
    rows = collect(tmp_path)
    assert len(rows) == len(plan)
    by_star = {r["star"]: r for r in rows}
    missing = by_star[f"{systems[2].name}__blind"]
    assert missing["status"] == "missing"
    good = by_star[f"{systems[0].name}__orbit"]
    assert good["status"] == "ok" and good["k_A_rel"] == pytest.approx(0.01)
    assert good["k_B_pull"] == -1.5 and good["v_rms_B"] == 1.4
    assert good["ew_metal_A"] == 1.02 and good["teff_err_B"] == -120.0
    assert good["t_conj_err_phase"] == pytest.approx(0.02)
    assert good["period_rel"] == pytest.approx(1.0 + 1e-4 / systems[0].period)
    assert good["detected_B"] == 1.0 and good["bootstrap_period_rel"] is None
    blind = by_star[f"{systems[0].name}__blind"]
    assert (
        blind["bootstrap_period_rel"] == pytest.approx(0.001) and blind["period_recovered"] is True
    )
    assert good["snr_total"] == pytest.approx(20.0 * np.sqrt(systems[0].n_transits))

    summary = summarize(rows)
    orbit = summary["tiers"]["orbit"]
    assert orbit["n"] == 3 and orbit["n_ok"] == 2 and orbit["n_failed"] == 1
    assert orbit["k_A_abs_rel"]["median"] == pytest.approx(0.01)
    assert orbit["k_A_within_5pct"] == 1.0 and orbit["k_A_within_1pct"] == 0.0
    assert orbit["k_B_pull_rms"] == pytest.approx(1.5)
    assert orbit["period_recovered_fraction"] is None
    assert summary["tiers"]["blind"]["period_recovered_fraction"] == 1.0
    assert "labels" in orbit["flags"]

    assert good["k_scan_used"] is True and good["noise_model"] == "ar1"

    velocity_rows = collect_velocities(tmp_path)
    ok_stars = [r["star"] for r in rows if r["status"] == "ok"]
    expected = sum(2 * s.n_transits for s, t in plan if f"{s.name}__{t.name}" in ok_stars)
    assert len(velocity_rows) == expected
    first = [r for r in velocity_rows if r["star"] == f"{systems[0].name}__orbit"]
    assert {r["component"] for r in first} == {"A", "B"}
    assert all(0.0 <= r["phase"] < 1.0 for r in first)
    assert sum(not r["good"] for r in first) == 2  # one flagged epoch, both components
    residuals = np.array([r["residual_kms"] for r in first if r["good"]])
    pulls = np.array([r["pull"] for r in first if r["good"]])
    assert np.all(np.isfinite(residuals)) and np.sqrt(np.mean(pulls**2)) < 2.0
    assert all(r["v_keplerian_dis_kms"] == 0.0 for r in first)
    per_tier = summarize_velocities(velocity_rows)
    assert set(per_tier) == {"orbit", "blind"}
    assert per_tier["orbit"]["n_systems"] == 2 and 0.5 < per_tier["orbit"]["pull_rms_A"] < 1.6
    assert per_tier["orbit"]["usable_fraction"] == pytest.approx(
        1.0 - 2 / (systems[0].n_transits + systems[2].n_transits)
    )

    report = write_report(tmp_path)
    text = report.read_text(encoding="utf-8")
    assert text.startswith("# test benchmark")
    assert "## Recovery by tier" in text and "| orbit | blind |" in text
    assert "abs(dK1 / K1) [%]" in text and "period search recovered" in text
    assert "## Epoch velocities" in text and "| pull rms, A |" in text
    assert (tmp_path / "rows.csv").is_file() and (tmp_path / "summary.json").is_file()
    assert (tmp_path / "velocities.csv").is_file()
    summary_json = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert "epoch_velocities" in summary_json and "orbit" in summary_json["epoch_velocities"]
    if HAS_MPL:
        assert (tmp_path / "figures" / "rv_curves_orbit.png").is_file()
        assert "rv_curves_orbit" in text
    assert "Failed stars" in text


@pytest.mark.slow
def test_two_systems_run_end_to_end(systems, library, tmp_path):
    config = BenchmarkConfig(
        output=tmp_path / "bench",
        systems=systems[:2],
        tiers=("oracle", "blind"),
        library=library,
        fast=True,
        plots=False,
        max_steps=40,
        label_steps=40,
        k_max=200.0,
    )
    run = run_benchmark(config, progress=False)
    assert run.report is not None and run.report.is_file()
    assert len(run.rows) == 4
    assert all(r["status"] == "ok" for r in run.rows), [r.get("error") for r in run.rows]
    oracle = [r for r in run.rows if r["tier"] == "oracle"]
    assert all(abs(r["k_A_rel"]) < 0.25 for r in oracle), oracle
    # Resuming runs nothing and reproduces the rows.
    again = run_benchmark(config, progress=False)
    assert [r["star"] for r in again.rows] == [r["star"] for r in run.rows]

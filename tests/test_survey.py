"""Tests for the correlation survey of simulated eclipsing binaries (``albireo.survey``).

The chain is run on a toy library in the RVS band, registered under a name of its own, so
the tests need no download. They test what the tables are used for: the injected
semi-amplitudes are recovered from bright systems, the transits in eclipse are kept out of
the orbit, the variants change what they describe, and the tables do not depend on how the
systems are divided into chunks.
"""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest

import albireo.survey as survey
from albireo.eclipsing import LibrarySet
from albireo.population import BinarySystem, semi_amplitudes
from albireo.simulate import synthetic_library
from albireo.survey import SurveyConfig, collect, declare, read_tables, run_survey, run_system

TOY = ("toy-rvs",)


@pytest.fixture(scope="module")
def libraries():
    library = synthetic_library(
        (8400.0, 8760.0),
        n_pix=1800,
        n_lines=36,
        teff=tuple(float(t) for t in range(4000, 7001, 500)),
        logg=(3.5, 4.0, 4.5, 5.0),
        mh=(-0.5, 0.0, 0.5),
        medium="vacuum",
    )
    boxes = LibrarySet([library])
    survey._LIBRARY_SETS[TOY] = boxes
    yield boxes
    survey._LIBRARY_SETS.pop(TOY, None)


def _system(name: str, seed: int, **changes) -> BinarySystem:
    m1, m2, period, incl = 1.25, 1.0, 2.9, math.radians(88.0)
    values = {
        "name": name,
        "m1": m1,
        "m2": m2,
        "r1": 1.35,
        "r2": 1.0,
        "teff1": 6300.0,
        "teff2": 5600.0,
        "logg1": 4.25,
        "logg2": 4.4,
        "mh": 0.0,
        "vsini1": 23.0,
        "vsini2": 17.0,
        "period": period,
        "ecc": 0.0,
        "omega": 0.0,
        "t_peri": 2456864.5,
        "incl": incl,
        "gamma": -8.0,
        "eclipsing": True,
        "light_ratio": 0.4,
        "g_mag": 8.0,
        "g_rp": 0.42,
        "grvs": 7.4,
        "ecl_lat_deg": 40.0,
        "n_transits": 16,
        "meta": {"seed": seed},
    }
    values.update(changes)
    k1, k2 = semi_amplitudes(
        values["m1"], values["m2"], values["period"], values["ecc"], values["incl"]
    )
    return BinarySystem(k1=float(k1), k2=float(k2), **values)


@pytest.fixture(scope="module")
def systems():
    return [
        _system("a", 11),
        _system("b", 12, period=4.7, n_transits=14, grvs=7.9),
        _system("c", 13, m2=1.2, teff2=6200.0, light_ratio=0.85, r2=1.3, n_transits=12),
    ]


CONFIG = SurveyConfig(libraries=TOY, v_search_kms=250.0, v_max_kms=300.0, vsini_max_kms=60.0)


def test_the_configuration_validates_and_round_trips():
    assert SurveyConfig.from_dict(json.loads(json.dumps(CONFIG.to_dict()))) == CONFIG
    with pytest.raises(ValueError, match="declaration"):
        SurveyConfig(declarations=("guessed",))
    with pytest.raises(ValueError, match="variant"):
        SurveyConfig(variant="spots")
    with pytest.raises(ValueError, match="simulation"):
        SurveyConfig(simulation="other")


def test_the_three_declarations(libraries, systems):
    system = systems[0]
    injected = declare(system, "injected", libraries, CONFIG)
    assert injected["labels"][0] == {"teff": 6300.0, "logg": 4.25, "mh": 0.0}
    assert injected["vsini"] == [23.0, 17.0] and not injected["identical"]
    assert sum(injected["light"]) == pytest.approx(1.0)
    assert injected["light"][1] / injected["light"][0] == pytest.approx(0.4)
    classified = declare(system, "classified", libraries, CONFIG)
    assert classified["light"] == "global"
    # The errors are drawn once per system, and both stars have the metallicity error.
    assert classified == declare(system, "classified", libraries, CONFIG)
    assert classified["labels"][0]["teff"] != 6300.0
    assert classified["labels"][0]["mh"] == classified["labels"][1]["mh"]
    assert abs(classified["labels"][0]["teff"] - 6300.0) < 5 * 0.03 * 6300.0
    catalogue = declare(system, "catalogue", libraries, CONFIG)
    assert catalogue["identical"] and catalogue["labels"][0] == catalogue["labels"][1]
    assert catalogue["labels"][0]["teff"] % 500.0 == 0.0
    assert catalogue["labels"][0]["logg"] == 4.5 and catalogue["labels"][0]["mh"] == 0.0
    assert catalogue["vsini"][0] == catalogue["vsini"][1] > 0.0
    with pytest.raises(ValueError, match="unknown"):
        declare(system, "guessed", libraries, CONFIG)


@pytest.fixture(scope="module")
def measured(libraries, systems):
    return run_system(systems[0], CONFIG)


def test_one_system_gives_its_epochs_and_its_orbits(measured, systems):
    system = systems[0]
    epochs, record = measured["epochs"], measured["system"]
    n = system.n_transits
    assert measured["name"] == "a" and record["n_epochs"] == n
    assert epochs["bjd"].shape == (n,) and epochs["v1"].shape == (3, n)
    assert np.all(np.diff(epochs["bjd"]) >= 0.0)
    # The injected velocities are those of the orbit, and the light fractions sum to one.
    expected = system.orbit().component_velocities(epochs["bjd"])
    np.testing.assert_allclose(epochs["v1_true"], expected[0])
    np.testing.assert_allclose(epochs["l1_true"] + epochs["l2_true"], 1.0)
    assert record["k1_true"] == system.k1 and record["light3_true"] == 0.0
    # The predicted errors follow the S/N and the light fractions.
    out = epochs["in_eclipse"] == 0
    assert np.all(epochs["pred1"] > 0.0) and np.all(epochs["pred2"][out] > epochs["pred1"][out])
    assert record["snr_k1"] > record["snr_k2"] > 5.0
    for row in range(3):
        assert record["n_out"][row] == n - record["n_in_eclipse"]
        assert record["e_ok"][row] == 1.0 and record["s_ok"][row] == 1.0
        # A circular orbit: the shape is held, and three parameters are fitted.
        assert record["e_npar"][row] == 3.0 and record["e_ecc"][row] == 0.0
    # With the injected templates both semi-amplitudes are recovered at this S/N.
    assert record["e_k1"][0] == pytest.approx(system.k1, rel=0.03)
    assert record["e_k2"][0] == pytest.approx(system.k2, rel=0.03)
    assert record["p_k1"][0] == pytest.approx(system.k1, rel=0.03)
    assert record["e_gamma"][0] == pytest.approx(system.gamma, abs=1.5)
    assert record["light1"][0] == pytest.approx(1.0 / 1.4, abs=1e-6)
    good = epochs["ge"][0] == 1.0
    assert good.sum() >= 0.6 * record["n_out"][0]
    error = np.abs(epochs["ve1"][0] - epochs["v1_true"])[good]
    assert np.median(error) < 2.0


def test_the_transits_in_eclipse_are_measured_and_kept_out_of_the_orbit(libraries):
    # An edge-on pair of short period, with many transits: some fall in an eclipse.
    system = _system("eclipsed", 21, period=1.3, incl=math.radians(89.9), n_transits=40)
    out = run_system(system, dataclasses.replace(CONFIG, declarations=("injected",)))
    epochs, record = out["epochs"], out["system"]
    in_eclipse = epochs["in_eclipse"].astype(bool)
    assert 0 < in_eclipse.sum() < system.n_transits
    assert record["n_in_eclipse"] == in_eclipse.sum()
    assert np.all(epochs["flux"][in_eclipse] < 1.0) and np.all(epochs["flux"][~in_eclipse] == 1.0)
    # In eclipse the S/N falls with the light, and the velocities do not enter the orbit.
    assert epochs["snr"][in_eclipse].max() < epochs["snr"][~in_eclipse].min()
    assert np.all(np.isnan(epochs["ve1"][0][in_eclipse]))
    assert np.all(np.isnan(epochs["vs"][0][in_eclipse]))
    assert np.isfinite(epochs["chi2"][0]).all()
    assert record["e_npts"][0] <= 2 * (~in_eclipse).sum()


def test_the_variants_change_what_they_describe(libraries, systems):
    one = ("injected",)
    system = dataclasses.replace(
        systems[0],
        meta={
            "seed": 11,
            "activity_ew": [0.8, 0.5],
            "tertiary": {
                "m": 0.8,
                "teff": 5200.0,
                "logg": 4.5,
                "radius": 0.8,
                "vsini": 4.0,
                "light": 0.2,
                "dv": 1.5,
            },
        },
    )
    baseline = run_system(system, dataclasses.replace(CONFIG, declarations=one))
    null = run_system(system, dataclasses.replace(CONFIG, declarations=one, variant="null"))
    assert np.all(null["epochs"]["l2_true"] == 0.0) and np.isnan(null["system"]["quality2"])
    np.testing.assert_allclose(null["epochs"]["bjd"], baseline["epochs"]["bjd"])
    third = run_system(system, dataclasses.replace(CONFIG, declarations=one, variant="third-light"))
    assert third["system"]["light3_true"] == pytest.approx(0.2)
    total = third["epochs"]["l1_true"] + third["epochs"]["l2_true"]
    np.testing.assert_allclose(total[third["epochs"]["in_eclipse"] == 0], 0.8)
    # The pair is fainter by the third light, and its predicted errors are larger.
    assert np.all(third["epochs"]["pred1"] > baseline["epochs"]["pred1"])
    active = run_system(system, dataclasses.replace(CONFIG, declarations=one, variant="activity"))
    assert not np.allclose(active["epochs"]["chi2"][0], baseline["epochs"]["chi2"][0])
    instrument = run_system(
        system, dataclasses.replace(CONFIG, declarations=one, variant="instrument")
    )
    # A background that varies between transits gives the transits different S/N.
    assert np.ptp(instrument["epochs"]["snr"]) > np.ptp(baseline["epochs"]["snr"])
    # Without the tertiary or the emission in its record, a system is the baseline.
    plain = dataclasses.replace(CONFIG, declarations=one, variant="third-light")
    same = run_system(systems[0], plain)
    reference = run_system(systems[0], dataclasses.replace(CONFIG, declarations=one))
    np.testing.assert_array_equal(same["epochs"]["v1"], reference["epochs"]["v1"])


def test_a_run_is_collected_and_does_not_depend_on_its_chunks(libraries, systems, tmp_path):
    config = dataclasses.replace(CONFIG, declarations=("injected", "catalogue"))
    first = run_survey(systems, config, tmp_path / "two", workers=0, chunk_size=2, progress=False)
    assert sorted(p.name for p in (first / "chunks").iterdir()) == [
        "chunk_00000.npz",
        "chunk_00001.npz",
    ]
    manifest = collect(first)
    assert manifest["n_systems"] == 3 and manifest["n_system_rows"] == 6
    assert manifest["failures"] == [] and manifest["declarations"] == ["injected", "catalogue"]
    table, epochs = read_tables(first)
    assert table["system"].tolist() == [0, 0, 1, 1, 2, 2] and table["decl"].tolist() == [0, 1] * 3
    assert epochs["system"].size == 2 * sum(s.n_transits for s in systems)
    assert manifest["n_epoch_rows"] == epochs["system"].size
    # The epochs of a system are those of its transits, once per declaration.
    rows = (epochs["system"] == 1) & (epochs["decl"] == 1)
    assert epochs["epoch"][rows].tolist() == list(range(systems[1].n_transits))
    assert np.all(table["e_ok"] == 1.0)

    # One chunk per system gives the same tables.
    second = run_survey(systems, config, tmp_path / "one", workers=0, chunk_size=1, progress=False)
    assert collect(second)["content_sha256"] == manifest["content_sha256"]
    # A run is continued: a chunk that is missing is run again, and the others are kept.
    kept = (first / "chunks" / "chunk_00000.npz").stat().st_mtime_ns
    (first / "chunks" / "chunk_00001.npz").unlink()
    run_survey(systems, config, first, workers=0, chunk_size=2, progress=False)
    assert (first / "chunks" / "chunk_00000.npz").stat().st_mtime_ns == kept
    assert collect(first)["content_sha256"] == manifest["content_sha256"]
    # A directory holds one run: the same systems, configuration and chunk size. The
    # systems are compared by their content, since drawn populations share their names.
    with pytest.raises(ValueError, match="another population"):
        run_survey(systems[:2], config, first, workers=0, chunk_size=2, progress=False)
    renamed = [dataclasses.replace(s, period=s.period * 1.01) for s in systems]
    with pytest.raises(ValueError, match="another population"):
        run_survey(renamed, config, first, workers=0, chunk_size=2, progress=False)
    other = dataclasses.replace(config, variant="null")
    with pytest.raises(ValueError, match="another configuration"):
        run_survey(systems, other, first, workers=0, chunk_size=2, progress=False)
    with pytest.raises(ValueError, match="another chunk size"):
        run_survey(systems, config, first, workers=0, chunk_size=3, progress=False)
    assert "uncommitted_changes" in manifest and len(manifest["commit"]) >= 7
    with pytest.raises(FileNotFoundError):
        collect(tmp_path / "empty")


def test_a_failing_system_is_recorded_and_does_not_lose_its_chunk(libraries, systems, tmp_path):
    broken = dataclasses.replace(systems[1], n_transits=0, name="broken")
    config = dataclasses.replace(CONFIG, declarations=("injected",))
    directory = run_survey(
        [systems[0], broken], config, tmp_path / "run", workers=0, progress=False
    )
    manifest = collect(directory)
    assert manifest["n_systems"] == 1 and len(manifest["failures"]) == 1
    assert manifest["failures"][0]["name"] == "broken"
    assert "n_transits" in manifest["failures"][0]["error"]

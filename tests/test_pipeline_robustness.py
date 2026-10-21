"""Tests for the pipeline driver's handling of declarations, reports and re-runs.

Each test covers one defect found in a review of :mod:`albireo.pipeline` and fails on the
code before its correction:

1. **A declaration the stages would reject is refused when it is read**, naming the star
   and the setting: per-star settings, mixed semi-amplitude kinds, a held eccentricity
   outside ``[0, 1)``, ``light = "measure"`` without a period, a reversed or degenerate
   prior table, a ``~`` path. A declared eccentricity above the settings' bound fails the
   star before its data are read.
2. **The products state the run that wrote them.** A report is strict JSON, a failed
   re-run leaves no earlier result beside its traceback, and the measured light is
   flagged once with the fractions the fit used.
3. **The period search keeps what the star declares**: a declared semi-amplitude prior
   survives the bootstrap, and a non-finite marginal likelihood does not decide the
   comparison of candidate periods. The candidates share one compiled coarse model, with
   values identical to those of a model built for each.

No test here runs a disentangling to convergence. The one that builds a model compiles
it on a small simulated dataset.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import albireo as ab
import albireo.pipeline as pipeline_module
from albireo.data import Dataset, EpochData
from albireo.facade import Between, Fixed, Known
from albireo.pipeline import (
    Analysis,
    ComponentConfig,
    PipelineConfig,
    PipelineRun,
    StarConfig,
    StarResult,
    _Context,
    _Log,
)


def _components(**k):
    return [
        ComponentConfig("A", 0.6, k=k.get("A")),
        ComponentConfig("B", 0.4, k=k.get("B")),
    ]


def _files_star(name="x", **kwargs):
    options = {"spectra": ["a.fits"], "period": (6.0, 6.6), "components": _components()}
    options.update(kwargs)
    return StarConfig(name=name, **options)


def _ctx(star, directory, *, settings=None, library=None):
    directory.mkdir(parents=True, exist_ok=True)
    config = PipelineConfig(stars=[star], output=directory, library=library)
    return _Context(
        star=star,
        config=config,
        settings=settings or Analysis(),
        directory=directory,
        log=_Log(star.name, directory / "log.txt", progress=False),
    )


def _orbit(period, k, *, ecc=0.12, chi2=10.0):
    return SimpleNamespace(
        period=period,
        chi2=chi2,
        k=np.asarray(k, dtype=float),
        ecc=ecc,
        omega=0.7,
        t_conj=2455000.5,
        names=("A", "B"),
        errors={"k": np.array([1.2, 1.6]), "ecc": 0.01},
        summary=lambda: f"  orbit at P {period:.5f} d",
    )


# ---------------------------------------------------------------------------
# 1. declarations
# ---------------------------------------------------------------------------


def test_a_per_star_setting_the_settings_reject_is_refused_when_the_batch_is_read():
    good = _files_star("good")
    reversed_region = _files_star("reversed", overrides={"region": (5300.0, 5000.0)})
    with pytest.raises(ValueError, match="star 'reversed': region needs lo < hi"):
        PipelineConfig(stars=[good, reversed_region])
    swapped_k = _files_star("swapped", overrides={"k_min": 50.0, "k_max": 20.0})
    with pytest.raises(ValueError, match="star 'swapped': need 0 < k_min < k_max"):
        PipelineConfig(stars=[good, swapped_k])


def test_a_star_run_outside_its_configuration_is_recorded_as_failed(tmp_path):
    """The batch guard records a failure of the settings too, rather than raising it."""
    config = PipelineConfig(stars=[_files_star("good")], output=tmp_path)
    bad = _files_star("bad", overrides={"region": (5300.0, 5000.0)})
    result = pipeline_module._run_star_guarded(
        bad, config, tmp_path / "bad", progress=False, keep_live=False
    )
    assert result.status == "failed" and "region needs lo < hi" in result.error
    assert (tmp_path / "bad" / "error.txt").is_file()
    assert not (tmp_path / "bad" / "result.json").exists()


def test_semi_amplitude_priors_of_different_kinds_are_refused_by_name():
    with pytest.raises(ValueError, match=r"mix kinds \(A a held value, B a range"):
        _files_star(components=_components(A=30.0))
    with pytest.raises(ValueError, match="mix kinds"):
        _files_star(components=_components(A=30.0, B={"value": 50.0, "sigma": 5.0}))
    # One kind for every component, an undeclared one counting as a range, is accepted.
    _files_star(components=_components(A=30.0, B=55.0))
    _files_star(components=_components(A=[5.0, 40.0]))
    _files_star(
        components=_components(A={"value": 30.0, "sigma": 3.0}, B={"value": 50.0, "sigma": 5.0})
    )
    # The velocity route fits no semi-amplitude, so its k is not checked.
    _files_star(period=None, velocities="rv.txt", components=_components(A=30.0))


def test_a_measured_light_without_a_period_is_refused():
    measured = [ComponentConfig("A", "measure"), ComponentConfig("B", "measure")]
    with pytest.raises(ValueError, match="light = 'measure' needs a declared period"):
        _files_star(period=None, velocities="rv.txt", components=measured)


def test_a_held_eccentricity_outside_zero_to_one_is_refused():
    with pytest.raises(ValueError, match=r"a held ecc must lie in \[0, 1\)"):
        _files_star(ecc=-0.1, omega=0.3)
    with pytest.raises(ValueError, match=r"a held ecc must lie in \[0, 1\)"):
        _files_star(ecc=1.2, omega=0.3)


def test_a_declared_eccentricity_above_the_bound_fails_the_star_before_its_data_are_read(
    tmp_path,
):
    from albireo.pipeline import _check_declared_ecc

    wide = _files_star("wide", ecc=[0.0, 0.8])
    with pytest.raises(ValueError, match=r"reaches 0.8, above the bound 0.5.*ecc_max"):
        _check_declared_ecc(wide, Analysis())
    _check_declared_ecc(wide, Analysis(ecc_max=0.8))
    _check_declared_ecc(wide, Analysis(circular=True))  # the bound is then 0.95
    with pytest.raises(ValueError, match=r"reaches 0.6"):
        _check_declared_ecc(_files_star(ecc=0.6, omega=1.0), Analysis())
    # The star fails at once: its spectra (which do not exist) are never read.
    config = PipelineConfig(stars=[wide], output=tmp_path)
    run = pipeline_module.run_pipeline(config, progress=False)
    assert "reaches 0.8" in run.results["wide"].error
    assert "read" not in run.results["wide"].seconds


def test_prior_tables_are_checked_like_prior_lists():
    from albireo.pipeline import _spec

    with pytest.raises(ValueError, match="a range needs lo < hi"):
        _spec({"lo": 5.0, "hi": 1.0}, "k")
    with pytest.raises(ValueError, match="finite positive sigma"):
        _spec({"value": 1.0, "sigma": 0.0}, "k")
    with pytest.raises(ValueError, match="finite positive sigma"):
        _spec({"value": 1.0, "sigma": -2.0}, "k")
    with pytest.raises(ValueError, match="finite value"):
        _spec({"value": float("nan"), "sigma": 1.0}, "k")
    assert _spec({"lo": 1, "hi": 5}, "k") == Between(1.0, 5.0)
    held = _spec(np.int64(3), "k")
    assert isinstance(held, Fixed) and float(held.value) == 3.0
    with pytest.raises(TypeError, match="boolean"):
        _spec(np.True_, "k")


def test_a_path_under_the_home_directory_is_expanded(tmp_path):
    from albireo.pipeline import _resolve_path

    resolved = _resolve_path("~/data/*.fits", tmp_path)
    assert resolved == os.fspath(Path("~/data/*.fits").expanduser())
    assert not resolved.startswith(os.fspath(tmp_path))
    assert _resolve_path("data/*.fits", tmp_path) == os.fspath(tmp_path / "data/*.fits")
    absolute = os.fspath(tmp_path / "x.fits")
    assert _resolve_path(absolute, Path("elsewhere")) == absolute


def test_the_workers_receive_the_shared_settings_without_a_stars_data():
    dataset = Dataset(
        tuple(
            EpochData(
                wave=np.linspace(5000.0, 5010.0, 50),
                flux=np.ones(50),
                ivar=np.ones(50),
                bjd=2455000.0 + j,
                instrument="X",
            )
            for j in range(3)
        )
    )
    star = StarConfig(
        name="memory",
        dataset=dataset,
        period=(6.0, 6.6),
        components=_components(),
        truth={"k": [30.0, 55.0]},
    )
    shared = PipelineConfig(stars=[star], lsf={"X": 5.0}).without_stars()
    assert len(shared.stars) == 1
    placeholder = shared.stars[0]
    assert placeholder.dataset is None and placeholder.truth is None
    assert placeholder.components == star.components and shared.lsf == {"X": 5.0}


def _epochs(instrument, starts, *, n=200, step=0.05, lsf=None):
    """Epochs of one instrument whose grids start at ``starts`` with one step."""
    return [
        EpochData(
            wave=start + step * np.arange(n),
            flux=np.ones(n),
            ivar=np.ones(n),
            bjd=2455000.0 + j,
            instrument=instrument,
            lsf_sigma_kms=lsf,
        )
        for j, start in enumerate(starts)
    ]


def test_a_medium_declared_on_the_star_keeps_each_epochs_line_spread_width(tmp_path):
    """A per-epoch width survives the medium override, and declares the instrument's LSF.

    The override rebuilt every epoch without its width, after which a per-epoch LSF found
    none. An in-memory dataset whose epochs all declare a width is modelled per epoch, as
    files that declare a resolving power are.
    """
    from albireo.pipeline import _resolve_lsf, _with_medium

    dataset = Dataset(tuple(_epochs("PEPSI", [5000.0, 5000.0, 5000.0], lsf=2.9)))
    vacuum = _with_medium(dataset, "vacuum")
    assert [e.medium for e in vacuum] == ["vacuum"] * 3
    assert np.array_equal(vacuum.lsf_sigma_kms, [2.9, 2.9, 2.9])
    star = StarConfig(
        name="pepsi",
        dataset=vacuum,
        period=(6.0, 6.6),
        components=_components(),
        lsf={"PEPSI": "per-epoch"},
    )
    lsf = _resolve_lsf(_ctx(star, tmp_path / "declared"), vacuum, {})
    assert lsf["PEPSI"].is_per_epoch
    undeclared = replace(star, lsf={})
    ctx = _ctx(undeclared, tmp_path / "undeclared")
    assert _resolve_lsf(ctx, vacuum, {})["PEPSI"].is_per_epoch
    assert "from the widths the epochs declare" in (ctx.directory / "log.txt").read_text(
        encoding="utf-8"
    )
    bare = Dataset(tuple(_epochs("PEPSI", [5000.0, 5000.0])))
    with pytest.raises(ValueError, match="no line-spread function for instrument 'PEPSI'"):
        _resolve_lsf(_ctx(undeclared, tmp_path / "bare"), bare, {})


def test_the_wavelength_grids_are_shared_per_instrument(tmp_path):
    """An instrument whose exposures are distinct solutions does not stop the others' sharing.

    The ``SHIFTED`` exposures differ by sub-pixel offsets and are put on one grid. The
    ``DRIFTED`` ones differ by about 1 km/s, as APF exposures do, and keep their own.
    """
    from albireo.pipeline import _try_share_grid

    shifted = _epochs("SHIFTED", [5000.0, 5000.0 + 1e-5, 5000.0 + 2e-5])
    drifted = _epochs("DRIFTED", [6000.0, 6000.0])
    drifted[1] = EpochData(
        wave=drifted[1].wave * (1.0 + 1.0 / 299_792.458),
        flux=drifted[1].flux,
        ivar=drifted[1].ivar,
        bjd=drifted[1].bjd,
        instrument="DRIFTED",
    )
    dataset = Dataset(tuple([shifted[0], drifted[0], shifted[1], drifted[1], shifted[2]]))
    log = _Log("grids", tmp_path / "log.txt", progress=False)
    shared = _try_share_grid(dataset, log)
    assert shared[0].wave is shared[2].wave is shared[4].wave
    assert shared[1] is dataset[1] and shared[3] is dataset[3]
    assert [e.bjd for e in shared] == [e.bjd for e in dataset]
    text = (tmp_path / "log.txt").read_text(encoding="utf-8")
    assert "relabelled 3 per-exposure wavelength grids of 'SHIFTED' onto 1" in text
    assert "instrument 'DRIFTED': per-exposure wavelength grids kept" in text


def test_the_library_cache_holds_one_library_and_reloads_a_rebuilt_file(tmp_path, monkeypatch):
    import albireo.library as library_module
    from albireo.pipeline import _resolve_library

    loads = []

    def load(path):
        loads.append(Path(path).name)
        return SimpleNamespace(path=path)

    monkeypatch.setattr(library_module, "load_library", load)
    monkeypatch.setattr(pipeline_module, "_LIBRARY_CACHE", {})
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    log = _Log("library", tmp_path / "log.txt", progress=False)
    a = _resolve_library(os.fspath(first), log)
    assert _resolve_library(os.fspath(first), log) is a and loads == ["first.npz"]
    _resolve_library(os.fspath(second), log)
    assert len(pipeline_module._LIBRARY_CACHE) == 1, "the first library was released"
    first.write_bytes(b"rebuilt")
    _resolve_library(os.fspath(first), log)
    assert loads == ["first.npz", "second.npz", "first.npz"]


# ---------------------------------------------------------------------------
# 2. products
# ---------------------------------------------------------------------------


def _reject_constant(name):
    raise ValueError(f"not strict JSON: {name}")


def test_non_finite_values_are_null_in_the_reports_whatever_their_type():
    import jax.numpy as jnp

    from albireo.pipeline import _jsonable

    report = {
        "python": float("nan"),
        "numpy_scalar": np.float64("nan"),
        "float32": np.float32("inf"),
        "array": np.array([1.0, np.nan, -np.inf]),
        "zero_d": np.array(np.inf),
        "jax": jnp.asarray([2.0, jnp.nan]),
        "nested": {"values": [np.float64(3.0), np.float64("nan")]},
    }
    clean = _jsonable(report)
    text = json.dumps(clean, allow_nan=False)
    assert json.loads(text, parse_constant=_reject_constant) == {
        "python": None,
        "numpy_scalar": None,
        "float32": None,
        "array": [1.0, None, None],
        "zero_d": None,
        "jax": [2.0, None],
        "nested": {"values": [3.0, None]},
    }
    # The batch summary reads such a null semi-amplitude.
    result = StarResult(
        name="s",
        status="ok",
        directory=".",
        report={"orbit": {"k": {"A": 30.0, "B": None}, "period": 6.3}},
    )
    run = PipelineRun(results={"s": result}, directory=Path("."), seconds=1.0, jobs=1)
    assert "K_A 30.00, K_B nan" in run.summary()


def test_a_failed_rerun_leaves_no_earlier_result_beside_its_traceback(tmp_path):
    """A re-run removes the status files and the products the earlier run listed.

    ``benchmark.collect`` reads a star's ``result.json`` as its outcome, so a failed re-run
    beside an earlier result was scored as a success. The BLOeM download directory, files
    the earlier run did not list, and anything outside the star's directory are kept.
    """
    directory = tmp_path / "star"
    (directory / "spectra").mkdir(parents=True)
    outside = tmp_path / "elsewhere.txt"
    listed = {
        "orbit": directory / "orbit.txt",
        "velocities": directory / "velocities.rv",
        "summary": directory / "summary.txt",
        "outside": outside,
        "download": directory / "spectra" / "epoch.fits",
    }
    for path in [*listed.values(), directory / "notes.txt"]:
        path.write_text("earlier run\n", encoding="utf-8")
    (directory / "result.json").write_text(
        json.dumps({"status": "ok", "files": {k: os.fspath(v) for k, v in listed.items()}}),
        encoding="utf-8",
    )
    # The star fails when its velocity table is read, before any model is built.
    star = _files_star("star", period=None, velocities=np.zeros(3))
    dataset = Dataset(
        tuple(
            EpochData(
                wave=np.linspace(5000.0, 5010.0, 50),
                flux=np.ones(50),
                ivar=np.ones(50),
                bjd=2455000.0 + j,
                instrument="X",
            )
            for j in range(3)
        )
    )
    star = replace(star, spectra=None, dataset=dataset, lsf={"X": 5.0}, labels=False)
    config = PipelineConfig(stars=[star], output=tmp_path)
    result = pipeline_module._run_star_guarded(
        star, config, directory, progress=False, keep_live=False
    )
    assert not result.ok and "two-dimensional" in result.error
    remaining = sorted(p.relative_to(directory).as_posix() for p in directory.rglob("*"))
    assert remaining == ["error.txt", "log.txt", "notes.txt", "spectra", "spectra/epoch.fits"]
    assert outside.is_file()
    assert "removed 4 file(s) written by an earlier run" in (directory / "log.txt").read_text(
        encoding="utf-8"
    )
    # The next run of the star starts by removing the traceback.
    log = _Log("star", directory / "log.txt", progress=False)
    pipeline_module._remove_previous_products(star, directory, log)
    assert not (directory / "error.txt").exists()


def test_the_light_measured_by_the_bootstrap_is_flagged_once_with_the_fractions_used(
    tmp_path, monkeypatch
):
    """The comparison may choose a candidate with another table, and other fractions.

    The fractions measured from the lowest chi-square candidate's table declare the
    comparison, and those of the chosen candidate's table declare the fit. One flag gives
    both, where a flag per measurement listed two values as if both were used.
    """
    from albireo.pipeline import _bootstrap

    def table(light):
        return SimpleNamespace(
            names=("A", "B"),
            n_components=2,
            settings={},
            n_epochs=12,
            bjd=2455000.25 + 3.0 * np.arange(12.0),
            velocity=np.zeros((2, 12)),
            delta_chi2=np.full((2, 12), 1e4),
            chi2=np.full(12, 1000.0),
            chi2_null=np.full(12, 3e4),
            light=np.repeat([[light], [1.0 - light]], 12, axis=1),
            good=np.ones(12, dtype=bool),
            summary=lambda: "  table summary",
        )

    first, second = table(0.6), table(0.7)
    ranked = [(_orbit(6.31, (30.0, 55.0)), first), (_orbit(4.71, (33.0, 50.0)), second)]
    searches = {"single": None, "harmonic": None, "invariant": None, "first": None}
    entries = [
        {"period": 6.31, "scan_moved": False, "k": [30.0, 55.0]},
        {"period": 4.71, "scan_moved": False, "k": [33.0, 50.0]},
    ]
    templates = [SimpleNamespace(meta={"source": "stub"}) for _ in range(2)]
    monkeypatch.setattr(pipeline_module, "_library_table", lambda *a, **k: (first, templates))
    monkeypatch.setattr(pipeline_module, "_period_candidates", lambda t, **k: ([1.0], searches))
    monkeypatch.setattr(
        pipeline_module,
        "_orbit_over_candidates",
        lambda ctx_, t, periods, **k: (
            ranked[0][0],
            first,
            {"n_candidates": 2, "ambiguous": [], "ranked": ranked},
        ),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_decide_period_by_disentangling",
        lambda *a, **k: {
            "index": 1,
            "entries": entries,
            "margin_nats": 12.0,
            "over_table_nats": 12.0,
        },
    )
    measured = [ComponentConfig("A", "measure"), ComponentConfig("B", "measure")]
    star = _files_star("light", period="search", components=measured)
    ctx = _ctx(star, tmp_path / "light", library="stub")
    _spec, block = _bootstrap(ctx, None, {"X": ab.LSF(sigma_kms=5.0)}, None)
    flags = [f for f in ctx.flags if f.startswith("light fractions measured")]
    assert len(flags) == 1, ctx.flags
    assert "declared: A 0.700, B 0.300" in flags[0]
    assert "compared under A 0.600, B 0.400" in flags[0]
    assert block["report"]["light_measured"] == pytest.approx({"A": 0.7, "B": 0.3})
    assert [c.light for c in ctx.star.components] == pytest.approx([0.7, 0.3])


# ---------------------------------------------------------------------------
# 3. the period search
# ---------------------------------------------------------------------------


def test_the_bootstrap_keeps_a_declared_semi_amplitude_prior(tmp_path):
    from albireo.pipeline import _bootstrap_spec

    ranged = _files_star(
        "ranged", period="search", components=_components(A=[5.0, 40.0], B=[5.0, 40.0])
    )
    ctx = _ctx(ranged, tmp_path / "ranged", library="stub")
    spec, text = _bootstrap_spec(ctx, _orbit(6.31, (30.0, 80.0)))
    assert all(isinstance(s, Between) for s in spec.k), spec.k
    assert [(float(s.lo), float(s.hi)) for s in spec.k] == [(5.0, 40.0), (5.0, 40.0)]
    assert "K as declared for ['A', 'B']" in text
    assert any("outside the range declared for ['B']" in f for f in ctx.flags), ctx.flags

    held = _files_star("held", period="search", components=_components(A=30.0, B=55.0))
    spec, _ = _bootstrap_spec(
        _ctx(held, tmp_path / "held", library="stub"), _orbit(6.31, (31.0, 80.0))
    )
    assert all(isinstance(s, Fixed) for s in spec.k)
    assert [float(s.value) for s in spec.k] == [30.0, 55.0]

    # A star that declares nothing is given the Gaussian around the table's values.
    free = _files_star("free", period="search")
    spec, _ = _bootstrap_spec(
        _ctx(free, tmp_path / "free", library="stub"), _orbit(6.31, (31.0, 56.0))
    )
    assert isinstance(spec.k, Known)
    assert np.asarray(spec.k.value).tolist() == [31.0, 56.0]


class _Scanner:
    """A coarse declaration whose phase scan and table-orbit value are given."""

    def __init__(self, phase, at_table):
        self.phase, self.at_table = list(phase), float(at_table)
        self.init = {"k": np.array([10.0, 20.0]), "t_conj": 2455000.0}
        self.fixed = {}
        self.model = SimpleNamespace(log_likelihood=lambda theta: self.at_table)
        self.grid = SimpleNamespace(x0=8.5, dx=1e-5, n=500)
        self.velocity_budget = SimpleNamespace(total=200.0)
        self.effective_ecc_max = 0.5
        self.stars = (SimpleNamespace(light=0.6), SimpleNamespace(light=0.4))
        self._built = {}

    def _scan_declaration(self):
        return self

    def _scan_phase(self, init):
        values = np.asarray(self.phase, dtype=float)
        return SimpleNamespace(best=2455000.25, values=values)

    def _has_ranged_k(self):
        return False


def test_a_non_finite_marginal_likelihood_does_not_decide_the_period(tmp_path, monkeypatch):
    """A NaN trial is left out of a candidate's value, and a NaN value ranks last.

    Before, one NaN phase trial made the candidate's value NaN, which the table orbit could
    not replace (``at_table > nan`` is false), and a NaN sort key left the ranking in the
    order given whatever the other values were.
    """
    from albireo.pipeline import _decide_period_by_disentangling

    scanners = {
        6.31: _Scanner([-840.0, np.nan], -2000.0),
        4.71: _Scanner([-900.0, -905.0], -2000.0),
        2.95: _Scanner([np.nan, np.nan], np.nan),
        9.47: _Scanner([np.nan, np.nan], -870.0),
    }

    def declare(ctx_, dataset_, lsf_, spec, velocities):
        centre = 0.5 * (float(spec.period.lo) + float(spec.period.hi))
        return scanners[min(scanners, key=lambda p: abs(p - centre))]

    monkeypatch.setattr(pipeline_module, "_declare", declare)
    star = _files_star("nan", period="search")
    ctx = _ctx(star, tmp_path / "nan", library="stub")
    ranked = [
        (_orbit(4.71, (33.0, 50.0)), None),
        (_orbit(2.95, (40.0, 70.0)), None),
        (_orbit(6.31, (30.0, 55.0)), None),
        (_orbit(9.47, (25.0, 60.0)), None),
    ]
    decision = _decide_period_by_disentangling(ctx, None, {}, ranked)
    values = [e["value_nats"] for e in decision["entries"]]
    assert values[0] == -900.0 and values[2] == -840.0
    assert np.isnan(values[1]), "no finite trial at all"
    assert values[3] == -870.0 and decision["entries"][3]["scored_at"] == "table orbit"
    assert decision["index"] == 2, values
    assert decision["margin_nats"] == pytest.approx(30.0)
    flag = next(f for f in ctx.flags if "not finite" in f)
    assert "[2.95]" in flag


def test_the_candidate_periods_share_one_coarse_model_with_identical_values(tmp_path, compilations):
    """The coarse declarations of two candidates differ in their starting values only.

    The second candidate is given the first one's compiled model, which compiles nothing
    for it, and every value the comparison takes from it is the value a model of its own
    returns, bit for bit.
    """
    from albireo.pipeline import (
        _bootstrap_spec,
        _coarse_marginal,
        _declare,
        _share_coarse_model,
        _table_orbit_trial,
    )
    from albireo.simulate import InstrumentSpec, OrbitParams, simulate_dataset

    grid = ab.LogGrid.from_wavelength_range(5000.0, 5040.0, dv_kms=2.0)
    components = [
        ab.synthetic_deviation_spectrum(grid, n_lines=14, seed=s, margin=0.1) for s in (1, 2)
    ]
    bjd = 2459000.0 + np.sort(np.random.default_rng(5).uniform(0.0, 21.0, size=6))
    dataset, _truth = simulate_dataset(
        grid,
        components,
        bjd=bjd,
        instruments={
            "SIM": InstrumentSpec(wave=np.arange(5005.0, 5035.0, 0.1), sigma_v_lsf=5.5, snr=150.0)
        },
        light_fractions=(0.6, 0.4),
        orbit=OrbitParams(period=6.31, t_peri=2.0, ecc=0.1, omega=0.7, k=(30.0, 50.0)),
        frame="barycentric",
        seed=11,
    )
    star = StarConfig(
        name="shared",
        dataset=dataset,
        period="search",
        components=_components(),
        lsf={"SIM": 5.5},
        overrides={"k_max": 90.0},
    )
    ctx = _ctx(star, tmp_path / "shared", settings=star.settings(Analysis()), library="stub")
    lsf = {"SIM": ab.LSF(sigma_kms=5.5)}
    candidates = []
    shared = None
    for orbit in (_orbit(6.31, (30.0, 50.0), ecc=0.1), _orbit(4.70, (35.0, 45.0), ecc=0.1)):
        spec, _ = _bootstrap_spec(ctx, orbit, comparison=True)
        dis = _declare(ctx, dataset, lsf, spec, None)
        scanner = dis._scan_declaration()
        shared = _share_coarse_model(scanner, shared)
        candidates.append((orbit, spec, dis, scanner))
    (orbit1, spec1, dis1, first), (orbit2, spec2, dis2, second) = candidates
    assert second.model is first.model
    _coarse_marginal(first, _table_orbit_trial(dict(dis1.init), orbit1, spec1, ctx.settings))

    trial = _table_orbit_trial(dict(dis2.init), orbit2, spec2, ctx.settings)
    values = []
    assert compilations(lambda: values.append(_coarse_marginal(second, trial))) == 0
    own = dis2._scan_declaration()  # the same declaration with a model of its own
    assert own.model is not first.model
    assert values[0] == _coarse_marginal(own, trial)
    init = dict(dis2.init)
    assert np.array_equal(second._scan_phase(init).values, own._scan_phase(init).values)

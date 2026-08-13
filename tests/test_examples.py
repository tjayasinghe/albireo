"""The packaged example dataset, and the download/cache machinery around it.

The packaged example is load-bearing for the quickstart, so these tests check the property
that actually matters: it loads with no network, no astropy, and no matplotlib, and what
comes back is a usable :class:`~albireo.data.Dataset` whose injected truth is self-consistent.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import albireo as ab
from albireo import examples


def test_the_packaged_example_loads_offline():
    dataset = ab.load_example("sb2_sim")

    epochs = list(dataset)
    assert len(epochs) == 12
    assert dataset.frame in {"topocentric", "barycentric"}
    assert {e.instrument for e in epochs} == {"DEMO"}
    for epoch in epochs:
        assert epoch.wave.ndim == 1
        assert epoch.wave.shape == epoch.flux.shape == epoch.ivar.shape
        assert np.all(np.diff(epoch.wave) > 0)
        assert np.all(epoch.ivar >= 0)
        assert np.all(np.isfinite(epoch.flux[epoch.ivar > 0]))
    # Continuum-normalized data sits around 1, which is what the model assumes.
    assert 0.5 < float(np.median(epochs[0].flux)) < 1.5


def test_the_packaged_example_is_the_default():
    default, named = list(ab.load_example()), list(ab.load_example("sb2_sim"))

    assert len(default) == len(named)
    for a, b in zip(default, named, strict=True):
        assert (a.bjd, a.instrument) == (b.bjd, b.instrument)
        np.testing.assert_array_equal(a.flux, b.flux)


def test_the_truth_is_self_consistent():
    dataset, truth = ab.load_example("sb2_sim", with_truth=True)

    n_epochs = len(list(dataset))
    assert truth["components"].shape[0] == 2
    assert truth["velocities"].shape == (2, n_epochs)
    np.testing.assert_allclose(np.sum(truth["light_fractions"]), 1.0, rtol=1e-12)
    assert truth["period"] == pytest.approx(6.0)
    assert truth["k"] == [42.0, 63.0]
    # A circular orbit with these semi-amplitudes: the two components move in opposition,
    # so K_1 * v_2 = -K_2 * v_1 at every epoch. Stated multiplicatively rather than as the
    # ratio v_2 / v_1, which is a 0/0 at the quarter phases where both velocities cross
    # zero — one of these twelve epochs lands exactly there.
    v1, v2 = truth["velocities"]
    np.testing.assert_allclose(42.0 * v2, -63.0 * v1, atol=1e-10)


def test_the_truth_grid_matches_the_component_spectra():
    _, truth = ab.load_example("sb2_sim", with_truth=True)
    grid = ab.LogGrid(x0=truth["grid_x0"], dx=truth["grid_dx"], n=int(truth["grid_n"]))

    assert truth["components"].shape[1] == grid.n
    # The simulated data lie inside the model grid, which is what build_problem requires.
    dataset = ab.load_example("sb2_sim")
    for epoch in dataset:
        assert epoch.wave[0] >= grid.wave[0]
        assert epoch.wave[-1] <= grid.wave[-1]


def test_example_metadata_is_discoverable():
    assert "sb2_sim" in ab.example_names()

    info = ab.example_info("sb2_sim")
    assert info["packaged"] is True
    assert info["cached"] is True
    assert info["url"] is None
    assert "12 epochs" in info["description"] or "simulated" in info["description"].lower()


def test_unknown_example_names_are_rejected_with_the_list():
    with pytest.raises(ValueError, match="unknown example"):
        ab.load_example("no_such_star")
    with pytest.raises(ValueError, match="sb2_sim"):
        ab.example_info("no_such_star")


def test_truth_is_refused_for_data_that_has_none(monkeypatch, tmp_path):
    # Rewrite the packaged file into a temporary copy with the truth stripped, which is
    # what an observed example looks like on disk.
    source = examples._PACKAGED_DIR / "sb2_sim.npz"
    with np.load(source, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files if not k.startswith("truth/")}
    header = json.loads(str(arrays["__albireo_example__"]))
    header["has_truth"] = False
    arrays["__albireo_example__"] = np.array(json.dumps(header))
    np.savez_compressed(tmp_path / "sb2_sim.npz", **arrays)

    monkeypatch.setattr(examples, "_PACKAGED_DIR", tmp_path)
    with pytest.raises(ValueError, match="no injected truth"):
        ab.load_example("sb2_sim", with_truth=True)


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------


def test_cache_dir_honours_the_environment_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ALBIREO_DATA_DIR", str(tmp_path / "elsewhere"))
    assert examples.cache_dir() == tmp_path / "elsewhere"


def test_cache_dir_falls_back_to_a_platform_location(monkeypatch):
    monkeypatch.delenv("ALBIREO_DATA_DIR", raising=False)
    assert examples.cache_dir().name in {"albireo", "Cache"}


def test_clearing_the_cache_never_deletes_a_packaged_example(monkeypatch, tmp_path):
    monkeypatch.setenv("ALBIREO_DATA_DIR", str(tmp_path))

    assert ab.clear_example_cache() == []
    # The packaged file is part of the installation and must survive.
    assert (examples._PACKAGED_DIR / "sb2_sim.npz").is_file()


def test_a_corrupted_download_is_rejected_rather_than_cached(monkeypatch, tmp_path):
    monkeypatch.setenv("ALBIREO_DATA_DIR", str(tmp_path))
    example = examples._Example(
        name="fake",
        description="fixture",
        packaged=False,
        filename="fake.npz",
        citation="none",
        url="https://example.invalid/fake.npz",
        sha256="0" * 64,
        n_bytes=10,
    )

    def fake_download(url, destination, attempts=4):
        destination.write_bytes(b"not the real bytes")

    monkeypatch.setattr(examples, "_download_with_retries", fake_download)

    with pytest.raises(RuntimeError, match="SHA-256"):
        examples._fetch(example, progress=False)
    # Nothing is left behind that a later run could mistake for a good cache entry.
    assert not (tmp_path / "examples" / "fake.npz").exists()
    assert not (tmp_path / "examples" / "fake.npz.part").exists()

"""The packaged example dataset, and the download/cache machinery around it.

The quickstart depends on the packaged example, so these tests check that it loads with no
network, no astropy, and no matplotlib, and that the result is a usable
:class:`~albireo.data.Dataset` whose injected truth is self-consistent.
"""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import threading
import time
import urllib.request

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
    # Continuum-normalized data are near 1, as the model assumes.
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
    # In a circular orbit with these semi-amplitudes the two components move in opposition,
    # so K_1 * v_2 = -K_2 * v_1 at every epoch. The relation is tested as a product rather
    # than as the ratio v_2 / v_1, which is 0/0 at the quarter phases where both velocities
    # cross zero. One of these twelve epochs is exactly at such a phase.
    v1, v2 = truth["velocities"]
    np.testing.assert_allclose(42.0 * v2, -63.0 * v1, atol=1e-10)


def test_the_truth_grid_matches_the_component_spectra():
    _, truth = ab.load_example("sb2_sim", with_truth=True)
    grid = ab.LogGrid(x0=truth["grid_x0"], dx=truth["grid_dx"], n=int(truth["grid_n"]))

    assert truth["components"].shape[1] == grid.n
    # The simulated data lie inside the model grid, as build_problem requires.
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
    # the on-disk form of an observed example.
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
    # The packaged file is part of the installation and must not be deleted.
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
    # No file remains that a later run would load as a valid cache entry.
    assert not (tmp_path / "examples" / "fake.npz").exists()
    assert not list((tmp_path / "examples").glob("*.part"))


def _fixture_example(payload: bytes, sha256: str | None) -> examples._Example:
    return examples._Example(
        name="fake",
        description="fixture",
        packaged=False,
        filename="fake.npz",
        citation="none",
        url="https://example.invalid/fake.npz",
        sha256=sha256,
        n_bytes=len(payload),
    )


def test_two_writers_of_one_example_do_not_share_a_temporary_file(monkeypatch, tmp_path):
    """Pipeline workers fetch the same uncached download at the same time.

    With one fixed ``.part`` name the second writer held the file the first one was
    renaming: on Windows the rename failed, and elsewhere the second writer lost its file.
    The second writer here keeps its temporary file open until the first has finished.
    """
    monkeypatch.setenv("ALBIREO_DATA_DIR", str(tmp_path))
    payload = b"the example bytes"
    example = _fixture_example(payload, hashlib.sha256(payload).hexdigest())
    holding, release = threading.Event(), threading.Event()

    def fake_download(url, destination, attempts=4):
        if threading.current_thread().name == "second":
            with destination.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                holding.set()
                release.wait(10.0)
        else:
            holding.wait(10.0)
            destination.write_bytes(payload)

    monkeypatch.setattr(examples, "_download_with_retries", fake_download)
    outcomes: dict[str, object] = {}

    def fetch():
        name = threading.current_thread().name
        try:
            outcomes[name] = examples._fetch(example, progress=False)
        except Exception as exc:
            outcomes[name] = exc
        finally:
            if name == "first":
                release.set()

    threads = [threading.Thread(target=fetch, name=name) for name in ("first", "second")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(20.0)

    path = tmp_path / "examples" / "fake.npz"
    assert outcomes == {"first": path, "second": path}, outcomes
    assert path.read_bytes() == payload
    assert not list(path.parent.glob("*.part"))


def test_an_example_without_a_digest_is_downloaded_once(monkeypatch, tmp_path):
    monkeypatch.setenv("ALBIREO_DATA_DIR", str(tmp_path))
    calls = []

    def fake_download(url, destination, attempts=4):
        calls.append(url)
        destination.write_bytes(b"unverified")

    monkeypatch.setattr(examples, "_download_with_retries", fake_download)
    example = _fixture_example(b"unverified", None)
    first = examples._fetch(example, progress=False)
    assert examples._fetch(example, progress=False) == first
    assert len(calls) == 1


@pytest.mark.parametrize("in_place", ["verified", "corrupt"])
def test_a_file_that_cannot_be_replaced_is_kept_only_if_it_verifies(
    monkeypatch, tmp_path, in_place
):
    """On Windows the final rename fails while another process holds the file in place.

    That file was kept on the assumption that another writer had put a verified copy
    there. A reader holding a corrupt copy open fails the same rename, and that copy was
    returned as the example.
    """
    monkeypatch.setenv("ALBIREO_DATA_DIR", str(tmp_path))
    payload = b"the example bytes"
    example = _fixture_example(payload, hashlib.sha256(payload).hexdigest())
    path = tmp_path / "examples" / "fake.npz"

    def fake_download(url, destination, attempts=4):
        destination.write_bytes(payload)
        # The file another process put in place during this download.
        path.write_bytes(payload if in_place == "verified" else b"a corrupt copy")

    def held_open(src, dst):
        raise PermissionError(13, "the file is held open by another process", str(dst))

    monkeypatch.setattr(examples, "_download_with_retries", fake_download)
    monkeypatch.setattr(examples.os, "replace", held_open)
    if in_place == "verified":
        assert examples._fetch(example, progress=False) == path
    else:
        with pytest.raises(PermissionError):
            examples._fetch(example, progress=False)
    assert not list(path.parent.glob("*.part"))


# ---------------------------------------------------------------------------
# the download helper shared with albireo.library
# ---------------------------------------------------------------------------


class _Response(io.BytesIO):
    """Minimal stand-in for urlopen's context manager."""

    def __init__(self, payload: bytes, headers: dict, error: Exception | None = None):
        super().__init__(payload)
        self.headers = headers
        self.error = error

    def read(self, *args):
        if self.error is not None:
            raise self.error
        return super().read(*args)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


@pytest.fixture
def no_backoff(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)


def test_a_truncated_download_is_retried_and_never_written(monkeypatch, tmp_path, no_backoff):
    """http.client ends a cut body with an empty read, not an error.

    A wavelength-grid file cut this way loaded as a shorter grid, and the library cache
    kept it on every later run.
    """
    calls = []

    def fake(*a, **k):
        calls.append(1)
        return _Response(b"16 bytes of data", {"Content-Length": "100000"})

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    destination = tmp_path / "grid.txt"
    with pytest.raises(RuntimeError, match="truncated download"):
        examples._download_with_retries("https://example.invalid/grid.txt", destination)
    assert len(calls) == 4
    assert not destination.exists()
    assert not list(tmp_path.iterdir()), "no temporary file may be left behind"


def test_a_cut_chunked_download_is_retried(monkeypatch, tmp_path, no_backoff):
    """A cut chunked body raises http.client.IncompleteRead, which is not an OSError."""
    calls = []

    def fake(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            return _Response(b"", {}, error=http.client.IncompleteRead(b"partial"))
        return _Response(b"complete", {"Content-Length": "8"})

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    destination = tmp_path / "shard.txt.gz"
    examples._download_with_retries("https://example.invalid/shard.txt.gz", destination)
    assert destination.read_bytes() == b"complete"
    assert [p.name for p in tmp_path.iterdir()] == ["shard.txt.gz"]


def test_a_local_file_error_is_not_retried(monkeypatch, tmp_path, no_backoff):
    """A local error (here a missing directory) is reported at once, not after backoff."""
    calls = []

    def fake(*a, **k):
        calls.append(1)
        return _Response(b"complete", {"Content-Length": "8"})

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    with pytest.raises(FileNotFoundError):
        examples._download_with_retries("https://example.invalid/x", tmp_path / "no" / "x")
    assert len(calls) == 1

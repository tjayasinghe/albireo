"""Tests for the batch driver of :mod:`albireo.pipeline`: directories, workers, interruption.

Each test covers one defect found in a review of :func:`albireo.pipeline.run_pipeline` and
fails on the code before its correction:

1. **A star's directory does not depend on the selection.** The directories are assigned
   over the whole batch, so ``stars=[...]`` writes each star where the full batch would,
   and names that differ only in case get distinct directories, as they must on the
   case-insensitive filesystems of Windows and macOS. An in-process batch that keeps no
   live objects also releases the programs compiled for each star.
2. **The worker environment sets only what takes effect.** The bare token
   ``intra_op_parallelism_threads`` is not an XLA flag and is not set.
3. **An interrupted worker batch cancels the stars still queued** instead of running
   every one of them before the interruption is raised.

The worker pool is replaced by a stub, so no test here starts a process for a star.
"""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import Future
from pathlib import Path

import pytest

import albireo.pipeline as pipeline_module
from albireo.pipeline import (
    ComponentConfig,
    PipelineConfig,
    StarConfig,
    _star_directories,
    _thread_environment,
    run_pipeline,
)


def _files_star(name, **kwargs):
    options = {
        "spectra": ["missing/*.fits"],
        "period": (6.0, 6.6),
        "components": [ComponentConfig("A", 0.6), ComponentConfig("B", 0.4)],
    }
    options.update(kwargs)
    return StarConfig(name=name, **options)


# ---------------------------------------------------------------------------
# 1. directories and memory
# ---------------------------------------------------------------------------


def test_a_selection_writes_each_star_where_the_whole_batch_would(tmp_path):
    """``HD 1`` and ``HD_1`` share a sanitized name; the second is ``HD_1_2`` in any run.

    Before, a run of ``HD_1`` alone wrote into ``HD_1``, over the products of ``HD 1``.
    """
    config = PipelineConfig(stars=[_files_star("HD 1"), _files_star("HD_1")], output=tmp_path)
    assert _star_directories(tmp_path, config.stars) == {
        "HD 1": tmp_path / "HD_1",
        "HD_1": tmp_path / "HD_1_2",
    }
    run = run_pipeline(config, stars=["HD_1"], progress=False)
    result = run.results["HD_1"]
    assert result.status == "failed", "the star has no epochs; only its directory matters"
    assert Path(result.directory) == tmp_path / "HD_1_2"
    assert (tmp_path / "HD_1_2" / "error.txt").is_file()
    assert not (tmp_path / "HD_1").exists()


def test_star_names_that_differ_only_in_case_get_distinct_directories(tmp_path):
    stars = [_files_star("Alpha"), _files_star("alpha"), _files_star("ALPHA_2")]
    assert {k: v.name for k, v in _star_directories(tmp_path, stars).items()} == {
        "Alpha": "Alpha",
        "alpha": "alpha_2",
        "ALPHA_2": "ALPHA_2_2",
    }


def test_a_batch_that_keeps_no_live_objects_releases_the_compiled_programs(tmp_path, monkeypatch):
    """Between the stars of an in-process batch run with ``keep_live=False``.

    JAX keeps every program it compiles for the life of the process. On five simulated
    stars the resident memory rose by about 0.5 GB per star with the live objects released,
    and stayed near 0.65 GB with the compiled programs released too.
    """
    import jax

    cleared = []
    monkeypatch.setattr(jax, "clear_caches", lambda: cleared.append(True))
    stars = [_files_star(n) for n in ("one", "two", "three")]
    config = PipelineConfig(stars=stars, output=tmp_path)
    run_pipeline(config, progress=False, keep_live=False)
    assert len(cleared) == 2, "after every star but the last"
    cleared.clear()
    run_pipeline(config, progress=False)
    assert not cleared, "a batch that keeps the live objects keeps the programs"
    # The release can be chosen independently of the live objects.
    run_pipeline(config, progress=False, keep_live=False, release_compiled=False)
    assert not cleared
    run_pipeline(config, progress=False, release_compiled=True)
    assert len(cleared) == 2


# ---------------------------------------------------------------------------
# 2. the worker environment
# ---------------------------------------------------------------------------


def test_the_worker_environment_sets_no_flag_that_xla_ignores(monkeypatch):
    """BLAS and OpenMP are capped; the XLA flags are ones XLA reads.

    ``intra_op_parallelism_threads=N`` is a TensorFlow session option. XLA's flag parser
    passes the bare token over without effect, and the dashed form is an unknown flag
    that aborts the process.
    """
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    env = _thread_environment(4)
    threads = str(max(1, (os.cpu_count() or 1) // 4))
    assert env["OMP_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == threads
    assert env["MKL_NUM_THREADS"] == threads
    assert env["XLA_FLAGS"] == "--xla_cpu_multi_thread_eigen=false"
    # A flag the user set is kept, and one naming the Eigen threads is left as it is.
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=2")
    assert _thread_environment(4)["XLA_FLAGS"] == (
        "--xla_force_host_platform_device_count=2 --xla_cpu_multi_thread_eigen=false"
    )
    monkeypatch.setenv("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=true")
    assert "XLA_FLAGS" not in _thread_environment(4)
    # The flags are accepted by the XLA of the installed JAX: an unknown one aborts.
    process = subprocess.run(
        [sys.executable, "-c", "import jax.numpy as jnp; print(float(jnp.ones(3).sum()))"],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert process.returncode == 0 and process.stdout.strip() == "3.0", process.stderr


# ---------------------------------------------------------------------------
# 3. interruption
# ---------------------------------------------------------------------------


def test_an_interrupted_worker_batch_cancels_the_stars_still_queued(tmp_path, monkeypatch):
    """The pool's exit waits for every submitted star unless the queued ones are cancelled.

    A Ctrl-C in the parent would otherwise run every queued star to completion, which
    for a large batch is hours, before the interruption is raised.
    """
    shutdowns = []

    class Pool:
        """A pool whose stars never finish, recording how it is shut down."""

        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.shutdown(wait=True)
            return False

        def submit(self, fn, *args):
            return Future()

        def shutdown(self, wait=True, *, cancel_futures=False):
            shutdowns.append({"wait": wait, "cancel_futures": cancel_futures})

    def interrupted(futures):
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline_module, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(pipeline_module, "as_completed", interrupted)
    config = PipelineConfig(
        stars=[_files_star(n) for n in ("one", "two", "three")], output=tmp_path
    )
    with pytest.raises(KeyboardInterrupt):
        run_pipeline(config, jobs=2, progress=False)
    assert shutdowns[0] == {"wait": False, "cancel_futures": True}

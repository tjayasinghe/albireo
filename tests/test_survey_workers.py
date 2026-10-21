"""The environment of the survey's worker processes.

XLA's flag parser ignores a bare ``intra_op_parallelism_threads=1`` token (a TensorFlow
session option), so the survey's workers are given only the flags XLA reads, added to
the caller's own.
"""

from __future__ import annotations

from albireo.survey import _single_thread_environment


def test_the_workers_get_one_thread_of_each_pool_and_only_flags_xla_reads(monkeypatch):
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    env = _single_thread_environment()
    assert env == {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
    }
    assert "intra_op" not in env["XLA_FLAGS"]


def test_the_callers_xla_flags_are_kept(monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=2")
    assert _single_thread_environment()["XLA_FLAGS"] == (
        "--xla_force_host_platform_device_count=2 --xla_cpu_multi_thread_eigen=false"
    )
    monkeypatch.setenv("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=true")
    assert "XLA_FLAGS" not in _single_thread_environment(), "the caller's choice is kept"

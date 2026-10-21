"""Wall-clock time, XLA compilations and peak memory of the quickstart fit (2026-10-21).

Measures, in one fresh process: the import of the package; a first `Disentangler.fit()` of
the packaged example with the quickstart declaration (`docs/quickstart.md`); a second
`fit()` of the same declaration; `Fit.summary()`; and, with ``--laplace``,
`Fit.laplace_check()`. For each stage it records the wall-clock time, the number of XLA
backend compilations and their total time (from a `jax.monitoring` listener on
``/jax/core/compile/backend_compile_duration``), the resident set and the process's peak
working set (Windows; 0 elsewhere). It also writes the best fit's potential, gradient norm
and parameters as ``repr`` strings, so that two checkouts can be compared bit for bit.

Run: ``python scripts/compile_bench.py out.json --laplace`` (about two and a half minutes on
the desktop CPU). Compare two checkouts by running the script under each, one after the
other on an otherwise idle machine. The record is in docs/benchmarks.md, "The optimization
pass of 2026-10-21".
"""

from __future__ import annotations

import json
import os
import sys
import time

_START = time.perf_counter()

import jax  # noqa: E402
import jax.monitoring  # noqa: E402
import numpy as np  # noqa: E402
import psutil  # noqa: E402

import albireo as ab  # noqa: E402

IMPORT_S = time.perf_counter() - _START
PROCESS = psutil.Process()
COMPILES: list[float] = []


def _listener(event, duration, **kwargs):
    if event == "/jax/core/compile/backend_compile_duration":
        COMPILES.append(duration)


jax.monitoring.register_event_duration_secs_listener(_listener)


def _stage(name, function, record):
    """Run ``function`` and record its wall-clock time, compilations and memory."""
    count, seconds = len(COMPILES), sum(COMPILES)
    start = time.perf_counter()
    out = function()
    wall = time.perf_counter() - start
    memory = PROCESS.memory_info()
    record[name] = {
        "wall_s": round(wall, 2),
        "compiles": len(COMPILES) - count,
        "compile_s": round(sum(COMPILES) - seconds, 2),
        "rss_gb": round(memory.rss / 1e9, 3),
        "peak_wset_gb": round(getattr(memory, "peak_wset", 0) / 1e9, 3),
    }
    print(name, record[name], flush=True)
    return out


def main() -> None:
    out_path = sys.argv[1]
    laplace = "--laplace" in sys.argv
    steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 300
    record: dict = {"albireo": ab.__file__, "import_s": round(IMPORT_S, 2), "max_steps": steps}
    dataset = ab.load_example("sb2_sim")
    dis = ab.Disentangler(
        dataset,
        components=[ab.Star("primary", light=0.62), ab.Star("secondary", light=0.38)],
        orbit=ab.Orbit(period=ab.Between(5.5, 6.5), k=ab.Between([10.0, 10.0], [90.0, 90.0])),
        lsf={"DEMO": 6.5},
    )
    fit = _stage("fit_cold", lambda: dis.fit(max_steps=steps), record)
    _stage("fit_again_same_declaration", lambda: dis.fit(max_steps=steps), record)
    _stage("summary", fit.summary, record)
    if laplace:
        check = _stage("laplace_check", fit.laplace_check, record)
        record["laplace"] = {
            "valid": bool(check.valid),
            "eigenvalues": [float(x) for x in check.eigenvalues],
            "cov_diag": [float(x) for x in np.diag(check.covariance)],
        }
    result = fit.result
    record["map"] = {
        "potential": repr(float(result.potential)),
        "grad_norm": repr(float(result.grad_norm)),
        "num_steps": int(result.num_steps),
        "params": {
            name: [repr(float(x)) for x in np.ravel(np.asarray(value, dtype=np.float64))]
            for name, value in sorted(result.params.items())
        },
    }
    record["peak_wset_gb_total"] = round(getattr(PROCESS.memory_info(), "peak_wset", 0) / 1e9, 3)
    record["pid"] = os.getpid()
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1)


if __name__ == "__main__":
    main()

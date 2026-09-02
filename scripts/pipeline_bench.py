"""Batch scaling with worker processes, and what the thread cap is worth (D58).

The pipeline runs stars in a spawn-based process pool with each worker's XLA and BLAS
threads capped at ``cpu_count // jobs``. This script measures the resulting speedup: the
same batch of simulated stars, in-process and with 2, 4 and 8 workers, with the cap on and,
for one point, off, so that the record carries a measured cost of oversubscription on this
machine.

The fixture is the pipeline's own toy star (a two-component SB2 drawn from the synthetic
library, 8 epochs, 725 native pixels) with the label stage off, so that what is timed is
the disentangling, the velocity table and the orbit, the stages every star pays for.
Every star is identical up to its noise seed.

    python scripts/pipeline_bench.py                 # 8 stars; jobs 1, 2, 4, 8
    python scripts/pipeline_bench.py --stars 4 --jobs 1 2

Writes a Markdown table to stdout for ``docs/benchmarks.md``.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

import albireo as ab
from albireo.pipeline import (
    Analysis,
    ComponentConfig,
    PipelineConfig,
    StarConfig,
    _environment,
    _with_medium,
    run_pipeline,
)
from albireo.simulate import (
    InstrumentSpec,
    OrbitParams,
    library_component,
    simulate_dataset,
    synthetic_library,
)

LABELS = {
    "A": {"teff": 5180.0, "logg": 4.05, "mh": -0.15, "vsini": 11.0},
    "B": {"teff": 4460.0, "logg": 4.55, "mh": -0.15, "vsini": 27.0},
}


def machine() -> str:
    cpu = platform.processor() or platform.machine()
    return f"{platform.system()} {platform.release()}, {cpu}, {os.cpu_count()} threads"


def make_stars(n: int, max_steps: int) -> list[StarConfig]:
    library = synthetic_library((5140.0, 5230.0), n_pix=900)
    grid = ab.LogGrid.from_wavelength_range(5150.0, 5220.0, dv_kms=2.0)
    components = [
        library_component(
            library,
            {k: v for k, v in val.items() if k != "vsini"},
            grid,
            medium="air",
            vsini_kms=val["vsini"],
        )
        for val in LABELS.values()
    ]
    orbit = OrbitParams(period=6.31, t_peri=2.0, ecc=0.15, omega=0.7, k=(30.0, 55.0), gamma=12.0)
    bjd = np.sort(np.random.default_rng(3).uniform(0.0, 21.0, size=8))
    stars = []
    for i in range(n):
        dataset, _ = simulate_dataset(
            grid,
            components,
            bjd=bjd,
            instruments={
                "TOY": InstrumentSpec(
                    wave=np.arange(5156.0, 5214.0, 0.08), sigma_v_lsf=5.5, snr=120.0
                )
            },
            light_fractions=(0.62, 0.38),
            orbit=orbit,
            seed=100 + i,
        )
        stars.append(
            StarConfig(
                name=f"star{i:02d}",
                dataset=_with_medium(dataset, "air"),
                period=(6.0, 6.6),
                components=[ComponentConfig("A", 0.62), ComponentConfig("B", 0.38)],
                lsf={"TOY": 5.5},
                labels=False,
                overrides={"k_max": 90.0, "max_steps": max_steps},
            )
        )
    return stars


def time_batch(stars, jobs: int, *, cap: bool) -> tuple[float, float]:
    out = Path(tempfile.mkdtemp(prefix="albireo_bench_"))
    config = PipelineConfig(stars=stars, output=out, analysis=Analysis(plots=False))
    # With the cap off, the worker environment is pre-seeded with a flag string that
    # already contains the key the pipeline looks for, so it adds nothing and each
    # worker sizes its pools to the whole machine. The bare `intra_op_parallelism_threads`
    # token is the form XLA accepts (the double-dash form is rejected as unknown), and the
    # string has to start with `--` or XLA reads it as a file name.
    env = (
        {}
        if cap
        else {"XLA_FLAGS": "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=0"}
    )
    t0 = time.perf_counter()
    with _environment(env):
        run = run_pipeline(config, jobs=jobs, progress=False)
    wall = time.perf_counter() - t0
    assert not run.failures, run.failures
    per_star = float(np.mean([r.seconds["total"] for r in run.results.values()]))
    shutil.rmtree(out, ignore_errors=True)
    return wall, per_star


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stars", type=int, default=8)
    ap.add_argument("--jobs", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--no-uncapped", action="store_true", help="skip the oversubscribed run")
    args = ap.parse_args()

    print(f"albireo {ab.__version__} pipeline batch benchmark -- {machine()}")
    stars = make_stars(args.stars, args.steps)
    print(f"{len(stars)} stars, {args.steps} L-BFGS steps each, labels off, plots off\n")

    # One warm-up star in-process so the compile is paid before the in-process timing,
    # as every worker pays it once too.
    time_batch(stars[:1], 1, cap=True)

    rows = []
    baseline = None
    for jobs in args.jobs:
        wall, per_star = time_batch(stars, jobs, cap=True)
        baseline = wall if baseline is None else baseline
        rows.append((jobs, "capped", wall, per_star, baseline / wall))
        speedup = baseline / wall
        print(f"jobs={jobs:2d} capped   wall {wall:7.1f} s, star {per_star:6.1f} s, x{speedup:.2f}")
    if not args.no_uncapped:
        jobs = max(args.jobs)
        if jobs > 1:
            wall, per_star = time_batch(stars, jobs, cap=False)
            rows.append((jobs, "uncapped", wall, per_star, baseline / wall))
            speedup = baseline / wall
            line = f"jobs={jobs:2d} uncapped wall {wall:7.1f} s, star {per_star:5.1f} s"
            print(f"{line}, x{speedup:.2f}")

    print("\n| workers | threads | batch wall [s] | per star [s] | speedup |")
    print("|---|---|---|---|---|")
    cores = os.cpu_count() or 1
    for jobs, mode, wall, per_star, speedup in rows:
        threads = (
            f"{max(1, cores // jobs)} each" if mode == "capped" else f"{cores} each (uncapped)"
        )
        print(f"| {jobs} | {threads} | {wall:.1f} | {per_star:.1f} | {speedup:.2f}x |")


if __name__ == "__main__":
    main()

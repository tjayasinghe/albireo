"""Simulate a population of Gaia RVS double-lined binaries, run albireo on them, and report.

    python scripts/gaia_rvs_benchmark.py --n 40 --kind mixed --jobs 8 --out bench/rvs
    python scripts/gaia_rvs_benchmark.py --debcat debs.dat --tiers eclipsing blind --out bench/deb
    python scripts/gaia_rvs_benchmark.py --gaia-sb2 200 --tiers orbit blind --out bench/gaia
    python scripts/gaia_rvs_benchmark.py --report-only --out bench/rvs      # regenerate the report

The population comes from the parametric generator (``--n``), from DEBCat's ``debs.dat``
(``--debcat``), from the Gaia DR3 double-lined orbits (``--gaia-sb2``, network), or from a
population written by an earlier run (``--population``). Every system is simulated with
albireo's Gaia RVS model and run through the pipeline under the requested knowledge
tiers. The report (``report.md`` with figures, ``rows.csv``, ``summary.json``) is written
to ``--out``. An interrupted run resumes when started again with the same directory.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import albireo as ab
from albireo.benchmark import CADENCES, TIERS, BenchmarkConfig, run_benchmark, write_report
from albireo.population import (
    draw_population,
    from_debcat,
    from_gaia_sb2,
    population_summary,
    query_gaia_sb2,
    read_population,
)


def inside_library(systems, library):
    """Split the systems into those the library can render and the names of the rest.

    A catalogue row can describe a star hotter than the library box (Gaia masses of 1.9
    solar give a primary at 8600 K). The parametric draw never does. The population module
    does not clamp a temperature, so the benchmark leaves such systems out and reports
    their names.
    """
    bounds = library.bounds
    kept, outside = [], []
    for system in systems:
        ok = all(
            bounds[key][0] <= labels[key] <= bounds[key][1]
            for labels in system.labels
            for key in ("teff", "logg", "mh")
            if key in bounds
        )
        (kept if ok else outside).append(system if ok else system.name)
    return kept, outside


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--n", type=int, default=40, help="systems to draw parametrically")
    source.add_argument("--population", help="a population.json from an earlier run")
    source.add_argument("--debcat", help="DEBCat's debs.dat")
    source.add_argument("--gaia-sb2", type=int, metavar="N", help="query N Gaia DR3 SB2 orbits")
    parser.add_argument("--kind", default="mixed", choices=["mixed", "eclipsing", "spectroscopic"])
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=["oracle", "eclipsing", "orbit", "blind"],
        choices=list(TIERS),
    )
    parser.add_argument("--product", default="dr4", choices=["dr3", "dr4"])
    parser.add_argument(
        "--library",
        default="bosz2024-fgk-rvs",
        help=(
            "a registered name or a saved .npz; one run uses one library, so "
            "'bosz2024-hot-rvs' is the box for catalogue systems above 7000 K and it "
            "leaves the cool ones out"
        ),
    )
    parser.add_argument(
        "--cadence",
        default="scanning-law",
        choices=list(CADENCES),
        help=(
            "epoch times: the scanning law's structure without its phase (default), evenly "
            "spaced phases, or the real forecast of the Gaia Observation Forecast Tool for "
            "each system's position (network on the first use of a position, then cached)"
        ),
    )
    parser.add_argument("--resolving-power", default="nominal", choices=["nominal", "per-transit"])
    parser.add_argument(
        "--period-range", nargs=2, type=float, default=[0.5, 1000.0], metavar=("LO", "HI")
    )
    parser.add_argument(
        "--min-separation",
        type=float,
        default=40.0,
        metavar="KMS",
        help="the double-lined selection: smallest (K1 + K2)(1 + e) kept, km/s",
    )
    parser.add_argument("--mass-weighting", default="magnitude", choices=["magnitude", "volume"])
    parser.add_argument(
        "--grvs",
        nargs=2,
        type=float,
        default=[9.8, 1.3],
        metavar=("MEAN", "SIGMA"),
        help="G mean and sigma of the draw",
    )
    parser.add_argument("--min-transits", type=int, default=10)
    parser.add_argument("--jobs", default="1", help="worker processes, an integer or 'auto'")
    parser.add_argument("--steps", type=int, default=300, help="disentangling steps")
    parser.add_argument("--label-steps", type=int, default=300)
    parser.add_argument("--k-max", type=float, default=250.0)
    parser.add_argument("--dv", type=float, default=3.0, help="model grid pixel, km/s")
    parser.add_argument("--mask-ca", action="store_true", help="zero-weight the Ca II triplet")
    parser.add_argument(
        "--noise-model",
        default="correlated",
        choices=["correlated", "diagonal"],
        help="declare the delivered grid's noise correlation to the analysis, or not",
    )
    parser.add_argument("--fast", action="store_true", help="trim every optimizer budget")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="benchmark_rvs")
    parser.add_argument("--title", default=None)
    parser.add_argument("--report-only", action="store_true", help="only regenerate the report")
    args = parser.parse_args(argv)

    out = Path(args.out)
    if args.report_only:
        path = write_report(out, title=args.title)
        print(f"report written to {path}")
        return 0

    registered = args.library in ab.library_names()
    library = ab.fetch_library(args.library) if registered else ab.load_library(args.library)
    release = "dr4" if args.product == "dr4" else "dr3"
    t0 = time.perf_counter()
    if args.population:
        systems = read_population(args.population)
    elif args.debcat:
        systems = from_debcat(args.debcat, seed=args.seed, library=library, release=release)
    elif args.gaia_sb2:
        rows = query_gaia_sb2(args.gaia_sb2)
        systems = from_gaia_sb2(rows, seed=args.seed, library=library, release="dr3")
    else:
        systems = draw_population(
            args.n,
            kind=args.kind,
            seed=args.seed,
            library=library,
            period_range=tuple(args.period_range),
            mass_weighting=args.mass_weighting,
            min_separation_kms=args.min_separation,
            g_mag=tuple(args.grvs),
            release=release,
        )
    systems, outside = inside_library(systems, library)
    if outside:
        print(
            f"{len(outside)} catalogue system(s) left out: a temperature or gravity outside "
            f"the library box ({', '.join(outside[:6])}{', ...' if len(outside) > 6 else ''})"
        )
    print(population_summary(systems))
    print(f"population ready in {time.perf_counter() - t0:.1f} s\n", flush=True)

    jobs = args.jobs if args.jobs == "auto" else int(args.jobs)
    title = args.title or (
        f"albireo on Gaia RVS double-lined binaries ({args.product}, {len(systems)} systems)"
    )
    config = BenchmarkConfig(
        output=out,
        systems=systems,
        tiers=args.tiers,
        product=args.product,
        library=args.library if registered else library,
        cadence=args.cadence,
        resolving_power=args.resolving_power,
        dv_kms=args.dv,
        k_max=args.k_max,
        max_steps=args.steps,
        label_steps=args.label_steps,
        mask_ca=args.mask_ca,
        noise_model=args.noise_model,
        min_transits=args.min_transits,
        jobs=jobs,
        fast=args.fast,
        plots=not args.no_plots,
        resume=not args.no_resume,
        seed=args.seed,
        title=title,
    )
    run = run_benchmark(config)
    print(f"\nbenchmark finished in {run.seconds / 60:.1f} min; report at {run.report}")
    for tier, entry in run.summary["tiers"].items():
        k = entry.get("k_A_abs_rel") or {}
        pull = entry.get("k_A_pull_rms")
        print(
            f"  {tier}: {entry['n_ok']}/{entry['n']} ok, median |dK1/K1| "
            f"{100 * k.get('median', float('nan')):.2f}%, K1 pull rms "
            f"{float('nan') if pull is None else pull:.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

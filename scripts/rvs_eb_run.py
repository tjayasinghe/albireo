"""Draw and run the eclipsing-binary survey experiment on simulated Gaia RVS epochs.

The experiment measures both velocities of 10^4 simulated detached eclipsing binaries at
every Gaia RVS transit by correlation against library templates, with no disentangling
(``albireo.survey``), and is reported in ``docs/reports/gaia-rvs-eclipsing-binaries.md``
(``scripts/rvs_eb_report.py``). This script draws the population and makes the runs::

    python scripts/rvs_eb_run.py draw                      # the design sample and its weights
    python scripts/rvs_eb_run.py run pilot                 # 300 systems, 5 per cell
    python scripts/rvs_eb_run.py run design                # 7800 systems, 130 per cell
    python scripts/rvs_eb_run.py run null                  # 900 primaries alone
    python scripts/rvs_eb_run.py run third-light           # 1200 systems, 20 per cell
    python scripts/rvs_eb_run.py run activity
    python scripts/rvs_eb_run.py run instrument
    python scripts/rvs_eb_run.py run recorded              # the 33 systems of the D66 notebook

The three runs with one effect added take the first 20 systems of every cell, with the
transit times and the seeds they have in the design run, which is their baseline. A run is
kept in ``<root>/<name>``, one file per chunk of 50 systems, and is continued by the same
command if it is interrupted. Every run is collected into its two tables and its
manifest when its last chunk is written. The chunks and the tables are not kept in the
repository: the population is drawn from its seed and every system is simulated from its
own, so the manifest reproduces them.

The three library boxes of ``albireo.eclipsing.DEFAULT_LIBRARIES`` must be in the albireo
cache (``albireo.fetch_library``; 190 to 530 MB of downloads each).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from albireo.eclipsing import (
    GRVS_EDGES,
    TEMPERATURE_CLASSES,
    catalogue_weights,
    draw_eclipsing_population,
    population_columns,
)
from albireo.population import read_population, write_population
from albireo.survey import SurveyConfig, collect, run_survey

REPO = Path(__file__).resolve().parent.parent
RESEARCH = REPO / "internal" / "research" / "2026-10-07-rvs-eclipsing-todcor"
ROOT = RESEARCH / "runs"
QUERIES = RESEARCH / "A_queries"
RECORDED = REPO / "internal" / "research" / "2026-09-09-gaia-rvs-benchmark"
SEED = 20261007
PER_CELL = 130

# name of the run: (variant, systems kept per cell, simulation)
RUNS = {
    "pilot": ("baseline", 5),
    "design": ("baseline", PER_CELL),
    "null": ("null", 15),
    "third-light": ("third-light", 20),
    "activity": ("activity", 20),
    "instrument": ("instrument", 20),
}
_CLASS_CODES = {1: "K", 2: "G", 3: "F", 4: "A"}


def catalogue_counts() -> tuple[dict[tuple[int, str, int], float], dict[str, float]]:
    """Counts of the Gaia DR3 candidates of the detached classes in the cells of the design.

    From two aggregate archive queries of 2026-10-07: ``q11a`` counts the candidates of
    the light-curve classes 2G-A, 2G-D, 2GE-A and 1G of Mowlavi et al. (2023) that have a
    ``teff_gspphot``, in bins of 0.5 mag of predicted G_RVS, 0.2 dex of period and the
    temperature classes of the design, and ``q11b`` those without. The sources without a
    temperature are spread over the classes of their magnitude bin in proportion to those
    that have one.

    Returns
    -------
    (dict, dict)
        ``{(G_RVS bin, class, period bin): count}`` for the four classes of the design,
        and the totals: every candidate of the range, and those outside the four classes.
    """
    known: dict[tuple[int, int, int], float] = {}
    by_bin: dict[int, float] = {}
    missing: dict[int, float] = {}
    first = round(GRVS_EDGES[0] / 0.5)
    n_bins = len(GRVS_EDGES) - 1
    with open(QUERIES / "q11a_eb_wide_grvs_logp_teffclass.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            m = int(float(row["mbin"])) - first
            if 0 <= m < n_bins:
                key = (m, int(float(row["tcls"])), int(float(row["pbin"])))
                known[key] = known.get(key, 0.0) + float(row["n"])
                by_bin[m] = by_bin.get(m, 0.0) + float(row["n"])
    with open(QUERIES / "q11b_eb_wide_grvs_logp_no_teff.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            m = int(float(row["mbin"])) - first
            if 0 <= m < n_bins:
                missing[m] = missing.get(m, 0.0) + float(row["n"])
    counts: dict[tuple[int, str, int], float] = {}
    outside = 0.0
    for (m, code, p), value in known.items():
        scaled = value * (1.0 + missing.get(m, 0.0) / by_bin[m])
        if code in _CLASS_CODES:
            counts[(m, _CLASS_CODES[code], p)] = scaled
        else:
            outside += scaled
    totals = {"all": sum(by_bin.values()) + sum(missing.values()), "outside_classes": outside}
    return counts, totals


def draw(args) -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    target = ROOT / "population.json"
    if target.exists() and not args.force:
        print(f"{target} exists; pass --force to draw it again")
        return 1
    t0 = time.perf_counter()
    systems = draw_eclipsing_population(args.per_cell, seed=args.seed)
    counts, totals = catalogue_counts()
    systems, summary = catalogue_weights(systems, counts)
    write_population(target, systems)
    summary.update(
        {
            "seed": args.seed,
            "per_cell": args.per_cell,
            "n_systems": len(systems),
            "catalogue_all": totals["all"],
            "catalogue_outside_classes": totals["outside_classes"],
            "seconds": time.perf_counter() - t0,
        }
    )
    (ROOT / "population_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


def _subset(systems, per_cell: int):
    return [s for s in systems if int(s.name.rsplit("-", 1)[1]) < per_cell]


def _recorded_systems():
    """The 33 systems of the recorded disentangling runs, with their recorded seeds."""
    import dataclasses

    out = []
    for run in ("field_d63", "gaia_d63"):
        manifest = json.loads((RECORDED / run / "manifest.json").read_text(encoding="utf-8"))
        recorded = manifest["simulation"]
        for system in read_population(RECORDED / run / "population.json"):
            if system.name in recorded:
                meta = {**system.meta, "seed": int(recorded[system.name]["seed"])}
                out.append(dataclasses.replace(system, meta=meta))
    return out


def run(args) -> int:
    name = args.name
    if name == "recorded":
        systems = _recorded_systems()
        config = SurveyConfig(
            simulation="recorded", v_search_kms=350.0, libraries=("bosz2024-fgk-rvs",)
        )
    else:
        variant, per_cell = RUNS[name]
        systems = _subset(read_population(ROOT / "population.json"), per_cell)
        config = SurveyConfig(variant=variant)
    if args.limit:
        systems = systems[: args.limit]
    directory = ROOT / (args.tag or name)
    print(f"{name}: {len(systems)} systems, variant {config.variant}, into {directory}")
    # Chunks kept from an earlier, interrupted invocation: the wall-clock time recorded
    # below is then that of this invocation alone.
    kept = len(list((directory / "chunks").glob("chunk_*.npz"))) if directory.exists() else 0
    t0 = time.perf_counter()
    run_survey(
        systems,
        config,
        directory,
        workers=args.workers,
        chunk_size=args.chunk,
        chunks_per_worker=args.chunks_per_worker,
        min_free_gb=args.min_free_gb,
    )
    manifest = collect(directory)
    manifest["run"] = name
    manifest["wall_clock_seconds"] = time.perf_counter() - t0
    manifest["chunks_kept_from_an_earlier_invocation"] = kept
    manifest["workers"] = args.workers
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(
        f"{name}: {manifest['n_systems']} systems, {manifest['n_epoch_rows']} epoch rows, "
        f"{len(manifest['failures'])} failures, {time.perf_counter() - t0:.0f} s"
    )
    for failure in manifest["failures"][:3]:
        print(failure["name"], failure["error"].strip().splitlines()[-1])
    return 0


def summary(args) -> int:
    systems = read_population(ROOT / "population.json")
    cols = population_columns(systems)
    print(f"{len(systems)} systems; classes {list(TEMPERATURE_CLASSES)}")
    for name in TEMPERATURE_CLASSES:
        sel = cols["klass"] == name
        quantiles = np.round(np.percentile(cols["period"][sel], [16, 50, 84]), 2).tolist()
        print(
            f"  {name}: {int(sel.sum())} systems, period 16/50/84 percent {quantiles}, "
            f"median q {np.median(cols['q'][sel]):.2f}, weight sum {cols['weight'][sel].sum():.0f}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("draw", help="draw the design sample and its catalogue weights")
    p.add_argument("--per-cell", type=int, default=PER_CELL)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=draw)
    p = sub.add_parser("run", help="make one run")
    p.add_argument("name", choices=[*RUNS, "recorded"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunk", type=int, default=50)
    p.add_argument("--chunks-per-worker", type=int, default=3)
    p.add_argument("--min-free-gb", type=float, default=3.0)
    p.add_argument("--limit", type=int, default=0, help="run the first systems only")
    p.add_argument("--tag", default="", help="directory name, if not that of the run")
    p.set_defaults(func=run)
    p = sub.add_parser("summary", help="print the drawn population by class")
    p.set_defaults(func=summary)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

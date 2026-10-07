"""The pipeline's period search on the benchmark's blind tables: where the injected period ranks.

    python scripts/period_decision_bench.py [--systems N] [--only PROCEDURE]

The blind tier of the Gaia RVS benchmark gives the pipeline no period. Library templates at
the declared starting labels measure a velocity table, four periodograms propose candidate
periods, an orbit is fitted from every candidate, and the chi-square ranks the orbits
(``docs/api/pipeline.md``). This script measures that table again for the 33 systems of the
recorded runs, with the pipeline's own functions and settings, and ranks the candidates
with four procedures for the fit at each candidate:

- one exchange: a Keplerian fit, one call of ``reassign_by_orbit`` and a second fit, with the
  epochs flagged for a second minimum left out, which is what the pipeline did before D68;
- assign: ``albireo.rvorbit.assign_components`` as the pipeline calls it now. The exchange
  is repeated from the table as measured and from assignments made from the period alone,
  at the candidate period and at the periods within 1/T of it, T being the time span of
  the usable epochs, and every flagged epoch takes the minimum the orbit fits better;
- left out: the same on the table without its recorded second minima, so that the flagged
  epochs have no weight, which isolates what their decision contributes;
- no window: the same as assign with every assignment made at the candidate period itself.

The populations and the simulation seeds are read from
``internal/research/2026-09-09-gaia-rvs-benchmark/``, so the script is run from a checkout
of the repository, and it needs the RVS box of the BOSZ grid (``albireo.fetch_library``).
Nothing is disentangled: the decision among the best candidates by the disentangling's
likelihood, which follows the ranking in the pipeline, is not run. The rank of the injected
period is the place, in the ranking by chi-square of the distinct fitted orbits, of the
first orbit within 2 percent of it. The results are recorded in ``docs/benchmarks.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

import albireo as ab
from albireo import benchmark, pipeline, rvorbit
from albireo.population import read_population

RUNS = Path("internal/research/2026-09-09-gaia-rvs-benchmark")


def one_exchange(table, *, period, circular=False, exchange=True, **_):
    """The candidate fit of the pipeline before D68, in the form of ``assign_components``."""
    orbit = rvorbit.fit_rv_orbit(table, period=period, circular=circular)
    decided = rvorbit.Assignment.none(table.n_epochs)
    if table.n_components == 2 and exchange:
        reassigned, moved = rvorbit.reassign_by_orbit(table, orbit.predict(table.bjd))
        if moved.any():
            orbit = rvorbit.fit_rv_orbit(reassigned, period=orbit.period, circular=circular)
            decided = rvorbit.Assignment(moved, decided.alternative, decided.resolved)
            table = reassigned
    return table, orbit, decided


ASSIGN = rvorbit.assign_components  # the name is replaced while a procedure is measured


def at_the_candidate(table, **options):
    """``assign_components`` with every assignment made at the candidate period itself."""
    return ASSIGN(table, **{**options, "period_window": 0.0})


def flagged_left_out(table, **options):
    """``assign_components`` on the table without its recorded second minima."""
    closed = replace(table, alternative=np.full(np.shape(table.velocity), np.nan))
    return ASSIGN(closed, **options)


PROCEDURES = {
    "one-exchange": one_exchange,
    "left out": flagged_left_out,
    "no window": at_the_candidate,
    "assign": ASSIGN,
}


def bootstrap_table(system, record, library, directory):
    """The table the pipeline's bootstrap measures for a system, with its context."""
    dataset, truth, grid, components = benchmark.simulate_system(
        system, library=library, seed=record["seed"]
    )
    config = benchmark.BenchmarkConfig(output=directory, systems=[system], tiers=("blind",))
    bounds = dict(library.bounds)
    star = benchmark.build_star(
        system,
        benchmark.TIERS["blind"],
        dataset=dataset,
        truth=truth,
        grid=grid,
        components=components,
        config=config,
        bounds=bounds,
    )
    run = pipeline.PipelineConfig(
        stars=[star],
        output=directory,
        library=library,
        mh=tuple(bounds["mh"]),
        analysis=benchmark._analysis(config),
    )
    where = Path(directory) / star.name
    where.mkdir(parents=True, exist_ok=True)
    ctx = pipeline._Context(
        star=star,
        config=run,
        settings=star.settings(run.analysis),
        directory=where,
        log=pipeline._Log(star.name, where / "log.txt", False),
    )
    data, header_lsf = pipeline._load_dataset(ctx)
    lsf = pipeline._resolve_lsf(ctx, data, header_lsf)
    table, _ = pipeline._library_table(ctx, data, lsf, library, "bootstrap")
    return ctx, table


def ranked_orbits(ctx, table, procedure):
    """The pipeline's ranking of the candidate orbits, with ``procedure`` as the fit."""
    settings = ctx.settings
    detection_min = float(settings.detection_min)
    lights = pipeline._table_light_fractions(table)
    exchange = (
        table.n_components != 2
        or lights is None
        or pipeline._exchange_allowed(ctx, lights, where="in the bootstrap's candidate fits")
    )
    candidates, _ = pipeline._period_candidates(
        table, swap_invariant=True, detection_min=detection_min, leave_one_out=True
    )
    fit, rvorbit.assign_components = rvorbit.assign_components, procedure
    try:
        t0 = time.perf_counter()
        _, _, found = pipeline._orbit_over_candidates(
            ctx,
            table,
            candidates,
            circular=settings.circular,
            detection_min=detection_min,
            exchange=exchange,
        )
        seconds = time.perf_counter() - t0
    finally:
        rvorbit.assign_components = fit
    return found["ranked"], len(candidates), seconds


def place(ranked, system):
    """The rank of the injected period, and the larger semi-amplitude error of its orbit."""
    for rank, (orbit, _) in enumerate(ranked, start=1):
        if abs(orbit.period / system.period - 1.0) < 0.02:
            k = np.asarray(orbit.k, dtype=float)
            injected = np.array([system.k1, system.k2])
            error = min(np.max(np.abs(order / injected - 1.0)) for order in (k, k[::-1]))
            return rank, float(error)
    return 0, float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--systems", type=int, help="measure the first N systems only")
    parser.add_argument("--only", choices=sorted(PROCEDURES), help="one procedure only")
    args = parser.parse_args(argv)
    warnings.simplefilter("ignore")  # the fits at wrong periods hit their bounds

    library = ab.fetch_library("bosz2024-fgk-rvs")
    population = []
    for run in ("field_d63", "gaia_d63"):
        recorded = json.loads((RUNS / run / "manifest.json").read_text(encoding="utf-8"))
        for system in read_population(RUNS / run / "population.json"):
            if system.name in recorded["simulation"]:
                population.append((system, recorded["simulation"][system.name]))
    if args.systems:
        population = population[: args.systems]
    names = [args.only] if args.only else list(PROCEDURES)

    print(
        "| system | epochs | usable | second minimum | candidates | "
        + " | ".join(f"rank, {n} | K error [%] | seconds" for n in names)
        + " |"
    )
    print("|---|---|---|---|---|" + "---|---|---|" * len(names))
    ranks = {name: [] for name in names}
    errors = {name: [] for name in names}
    total = dict.fromkeys(names, 0.0)
    no_table = []
    with tempfile.TemporaryDirectory() as directory:
        for system, record in population:
            try:
                ctx, table = bootstrap_table(system, record, library, directory)
            except ValueError as error:
                # The correlation measured no light fractions: the pipeline stops here too.
                no_table.append(system.name)
                print(f"| {system.name} | no table: {str(error)[:90]}... |", flush=True)
                continue
            cells = []
            n_candidates = 0
            for name in names:
                ranked, n_candidates, seconds = ranked_orbits(ctx, table, PROCEDURES[name])
                rank, error = place(ranked, system)
                ranks[name].append(rank)
                errors[name].append(error)
                total[name] += seconds
                shown = "-" if not np.isfinite(error) else f"{100 * error:.1f}"
                cells.append(f"{rank or 'absent'} | {shown} | {seconds:.0f}")
            print(
                f"| {system.name} | {table.n_epochs} | {int(table.good.sum())} "
                f"| {int(table.second_minimum.sum())} | {n_candidates} | "
                + " | ".join(cells)
                + " |",
                flush=True,
            )

    print(
        "\n| procedure | rank 1 | within the first 4 | within the first 8 | absent "
        "| rank 1 with both K within 5 % | seconds per system |"
    )
    print("|---|---|---|---|---|---|---|")
    n_tables = len(population) - len(no_table)
    for name in names:
        rank, error = np.array(ranks[name]), np.array(errors[name])
        good = (rank == 1) & (error < 0.05)
        print(
            f"| {name} | {int((rank == 1).sum())} | {int(((rank >= 1) & (rank <= 4)).sum())} "
            f"| {int(((rank >= 1) & (rank <= 8)).sum())} | {int((rank == 0).sum())} "
            f"| {int(good.sum())} | {total[name] / max(n_tables, 1):.0f} |"
        )
    print(f"\n{len(population)} systems, {n_tables} with a table")
    if no_table:
        print("no table: " + ", ".join(no_table))
    return 0


if __name__ == "__main__":
    sys.exit(main())

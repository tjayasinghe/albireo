"""Templates fitted to the epochs, on the simulated survey of eclipsing binaries.

    python scripts/template_fit_bench.py [RUN ...]

Reads collected runs of ``scripts/rvs_eb_run.py`` (default: ``fitted``) and tabulates, for
every declaration of the templates each run measured, on the same systems:

- the share of systems with both semi-amplitudes within 10 and within 5 percent of the
  injected ones, in ranges of G_RVS, weighted to the Gaia DR3 catalogue of eclipsing
  binaries as in ``docs/reports/gaia-rvs-eclipsing-binaries.md`` (the weight of a system
  is that of its cell of the design);
- the robust widths of the pulls of the two semi-amplitudes over the recovered systems;
- for the declarations whose templates are fitted (``albireo.fit_templates``), the errors
  of the fitted labels, rotation and light fraction, and the time of the fit;
- each fitted declaration against the one it starts from, system by system: the systems
  gained and lost, the difference of the weighted shares with a bootstrap interval, the
  pull widths and the errors of K on the systems recovered under both, and the share
  recovered among the fits whose alternation converged and among the others;
- with the three runs named below, the systems with Ca II emission: the share recovered
  without the emission, with it, and with it fitted beside the labels, and the fitted
  equivalent widths against the injected ones.

The numbers are recorded in ``docs/benchmarks.md``. The runs are made by::

    python scripts/rvs_eb_run.py run fitted
    python scripts/rvs_eb_run.py run activity-fitted
    python scripts/rvs_eb_run.py run activity-emission
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from albireo.eclipsing import population_columns
from albireo.population import read_population
from albireo.survey import read_tables

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "internal" / "research" / "2026-10-07-rvs-eclipsing-todcor" / "runs"
RANGES = ((6.0, 9.0), (9.0, 10.0), (10.0, 11.0), (11.0, 12.0), (12.0, 13.0), (13.0, 13.5))


def robust_sigma(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else float("nan")


def wshare(flag, keep, weights) -> float:
    total = float(np.sum(weights[keep]))
    return float(np.sum(weights[keep & flag]) / total) if total > 0.0 else float("nan")


def load(directory: Path):
    systems, _ = read_tables(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    cols = population_columns(read_population(directory / "population.json"))
    return systems, manifest, cols


def table(directory: Path) -> None:
    systems, manifest, cols = load(directory)
    declarations = list(manifest["declarations"])
    index = systems["system"]
    grvs, weight = cols["grvs"][index], cols["weight_cell"][index]
    with np.errstate(invalid="ignore", divide="ignore"):
        r1 = systems["e_k1"] / systems["k1_true"] - 1.0
        r2 = systems["e_k2"] / systems["k2_true"] - 1.0
    worst = np.where(np.isfinite(r1) & np.isfinite(r2), np.fmax(np.abs(r1), np.abs(r2)), np.inf)
    n_systems = int(np.unique(index).size)
    variant = manifest.get("config", {}).get("variant", "?")
    print(
        f"\n## {directory.name}: {n_systems} systems, variant {variant}, "
        f"{manifest.get('wall_clock_seconds', float('nan')):.0f} s on "
        f"{manifest.get('workers', '?')} processes"
    )
    header = " | ".join(f"{lo:g}-{hi:g}" for lo, hi in RANGES)
    for limit, label in ((0.10, "10"), (0.05, "5")):
        print(f"\nBoth K within {label} percent, weighted share [%] by G_RVS\n")
        print(f"| declaration | {header} | all |")
        print("|---|" + "---|" * (len(RANGES) + 1))
        for d, name in enumerate(declarations):
            rows = systems["decl"] == d
            cells = []
            for lo, hi in (*RANGES, (RANGES[0][0], RANGES[-1][1])):
                keep = rows & (grvs >= lo) & (grvs < hi)
                cells.append(f"{100.0 * wshare(worst < limit, keep, weight):.0f}")
            print(f"| {name} | " + " | ".join(cells) + " |")
    print(
        "\nSystems per range: "
        + ", ".join(
            f"{lo:g}-{hi:g}: {int(np.sum((systems['decl'] == 0) & (grvs >= lo) & (grvs < hi)))}"
            for lo, hi in RANGES
        )
    )

    print(
        "\n| declaration | pull width K1 | pull width K2 | recovered | fit made | converged "
        "| median fit [s] |"
    )
    print("|---|---|---|---|---|---|---|")
    for d, name in enumerate(declarations):
        rows = systems["decl"] == d
        ok = rows & (systems["e_ok"] == 1) & (worst < 0.10)
        with np.errstate(invalid="ignore", divide="ignore"):
            pulls = [
                robust_sigma(
                    (systems[f"e_k{k}"] - systems[f"k{k}_true"])[ok] / systems[f"e_k{k}_err"][ok]
                )
                for k in (1, 2)
            ]
        fitted = rows & np.isfinite(systems["f_ok"]) if "f_ok" in systems else rows & False
        made = converged = seconds = "-"
        if fitted.any():
            made = f"{100.0 * np.mean(systems['f_ok'][fitted] == 1):.0f}%"
            done = fitted & (systems["f_ok"] == 1)
            converged = f"{100.0 * np.mean(systems['f_converged'][done] == 1):.0f}%"
            seconds = f"{np.median(systems['t_fit'][fitted]):.1f}"
        print(
            f"| {name} | {pulls[0]:.2f} | {pulls[1]:.2f} | {int(ok.sum())} | {made} | "
            f"{converged} | {seconds} |"
        )

    if "f_ok" not in systems:
        return
    print(
        "\nFitted labels against the injected ones, for the systems with both K within 10 "
        "percent (median absolute error). The stars are named by the orbit at the ephemeris: "
        "the fit's first star is the second where most transits were exchanged against the "
        "table as measured.\n"
    )
    print(
        "| declaration | G_RVS | n | Teff 1 [K] | Teff 2 [K] | log g 1 | log g 2 | "
        "v sin i 1 [km/s] | v sin i 2 [km/s] | light fraction 2 | fits in the other order [%] |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    labels = ("teff", "logg", "vsini")
    truth = {f"{label}{k}": cols[f"{label}{k}"][index] for label in labels for k in (1, 2)}
    light2 = systems["light2_true"]
    with np.errstate(invalid="ignore"):
        swap = systems["e_nx"] > 0.5 * systems["n_out"]
    for d, name in enumerate(declarations):
        rows = (systems["decl"] == d) & (systems["f_ok"] == 1) & (worst < 0.10)
        if not rows.any():
            continue
        for lo, hi in ((6.0, 10.0), (10.0, 12.0), (12.0, 13.0)):
            keep = rows & (grvs >= lo) & (grvs < hi)
            if not keep.any():
                continue
            cells = []
            for label in labels:
                first, second = systems[f"f_{label}1"], systems[f"f_{label}2"]
                named = ((1, np.where(swap, second, first)), (2, np.where(swap, first, second)))
                for k, fitted in named:
                    error = np.nanmedian(np.abs(fitted - truth[f"{label}{k}"])[keep])
                    cells.append(f"{error:.0f}" if label == "teff" else f"{error:.2f}")
            fraction = np.where(swap, 1.0 - systems["f_light2"], systems["f_light2"])
            print(
                f"| {name} | {lo:g}-{hi:g} | {int(keep.sum())} | " + " | ".join(cells) + " | "
                f"{np.nanmedian(np.abs(fraction - light2)[keep]):.3f} | "
                f"{100.0 * np.mean(swap[keep]):.0f} |"
            )


def paired(directory: Path, limit: float = 0.10, n_boot: int = 2000) -> None:
    """A fitted declaration against the declaration it starts from, system by system."""
    systems, manifest, cols = load(directory)
    declarations = list(manifest["declarations"])
    if "f_ok" not in systems:
        return
    index = systems["system"]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = [systems[f"e_k{k}"] / systems[f"k{k}_true"] - 1.0 for k in (1, 2)]
    finite = np.isfinite(ratio[0]) & np.isfinite(ratio[1])
    worst = np.where(finite, np.fmax(np.abs(ratio[0]), np.abs(ratio[1])), np.inf)
    rng = np.random.default_rng(20261013)

    def rows_of(name: str) -> np.ndarray:
        rows = np.flatnonzero(systems["decl"] == declarations.index(name))
        return rows[np.argsort(index[rows])]

    pairs = [
        (base, base + "-fitted")
        for base in ("classified", "catalogue")
        if base in declarations and base + "-fitted" in declarations
    ]
    if not pairs:
        return
    print(
        f"\nFitted against the declaration it starts from, on the same systems (both K within "
        f"{100.0 * limit:.0f} percent; the difference of the weighted shares with the 16th and "
        "84th percentiles of a bootstrap over systems)\n"
    )
    print(
        "| pair | G_RVS | n | both | gained | lost | base [%] | fitted [%] | difference [points] |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for base, fitted in pairs:
        rb, rf = rows_of(base), rows_of(fitted)
        ids = index[rb]
        grvs, weight = cols["grvs"][ids], cols["weight_cell"][ids]
        ok_b, ok_f = worst[rb] < limit, worst[rf] < limit
        for lo, hi in (*RANGES, (RANGES[0][0], RANGES[-1][1])):
            keep = np.flatnonzero((grvs >= lo) & (grvs < hi))
            w = weight[keep]
            b, f = ok_b[keep].astype(float), ok_f[keep].astype(float)
            draws = rng.integers(0, keep.size, size=(n_boot, keep.size))
            spread = np.sum(w[draws] * (f[draws] - b[draws]), axis=1) / np.sum(w[draws], axis=1)
            low, high = np.percentile(spread, [16.0, 84.0])
            print(
                f"| {base} to {fitted} | {lo:g}-{hi:g} | {keep.size} | {int(np.sum(b * f))} | "
                f"{int(np.sum((1 - b) * f))} | {int(np.sum(b * (1 - f)))} | "
                f"{100.0 * np.sum(w * b) / np.sum(w):.1f} | "
                f"{100.0 * np.sum(w * f) / np.sum(w):.1f} | "
                f"{100.0 * np.sum(w * (f - b)) / np.sum(w):+.1f} ({100.0 * low:+.1f}, "
                f"{100.0 * high:+.1f}) |"
            )

    print(
        "\nOn the systems recovered under both members of a pair: the robust widths of the K "
        "pulls and the median errors of K, base to fitted\n"
    )
    print(
        "| pair | n | pull width K1 | pull width K2 | median error K1 [%] | median error K2 [%] |"
    )
    print("|---|---|---|---|---|---|")
    for base, fitted in pairs:
        rb, rf = rows_of(base), rows_of(fitted)
        common = (worst[rb] < limit) & (worst[rf] < limit)
        cells = []
        for k in (1, 2):
            widths, errors = [], []
            for rows in (rb, rf):
                pick = rows[common]
                with np.errstate(invalid="ignore", divide="ignore"):
                    pull = (systems[f"e_k{k}"] - systems[f"k{k}_true"])[pick] / systems[
                        f"e_k{k}_err"
                    ][pick]
                widths.append(robust_sigma(pull))
                errors.append(100.0 * np.nanmedian(np.abs(ratio[k - 1][pick])))
            cells.append((widths, errors))
        print(
            f"| {base} to {fitted} | {int(common.sum())} | "
            f"{cells[0][0][0]:.2f} to {cells[0][0][1]:.2f} | "
            f"{cells[1][0][0]:.2f} to {cells[1][0][1]:.2f} | "
            f"{cells[0][1][0]:.2f} to {cells[0][1][1]:.2f} | "
            f"{cells[1][1][0]:.2f} to {cells[1][1][1]:.2f} |"
        )

    print(
        "\nThe convergence of the alternation as a flag: systems with both K within "
        f"{100.0 * limit:.0f} percent among the fits that converged and among those that did "
        "not (unweighted)\n"
    )
    print(
        "| declaration | G_RVS | fits | converged | recovered of converged [%] "
        "| recovered of the others [%] | ephemeris order exchanged [%] |"
    )
    print("|---|---|---|---|---|---|---|")
    for _, fitted in pairs:
        rf = rows_of(fitted)
        ids = index[rf]
        grvs = cols["grvs"][ids]
        made = systems["f_ok"][rf] == 1
        converged = made & (systems["f_converged"][rf] == 1)
        ok = worst[rf] < limit
        swapped = systems["swapped"][rf] == 1
        for lo, hi in ((6.0, 10.0), (10.0, 12.0), (12.0, 13.0)):
            keep = (grvs >= lo) & (grvs < hi) & made
            yes, no = keep & converged, keep & ~converged
            share = [100.0 * ok[m].mean() if m.any() else float("nan") for m in (yes, no)]
            print(
                f"| {fitted} | {lo:g}-{hi:g} | {int(keep.sum())} | {int(yes.sum())} | "
                f"{share[0]:.0f} | {share[1]:.0f} | {100.0 * swapped[keep].mean():.0f} |"
            )


EMISSION_RUNS = ("fitted", "activity-fitted", "activity-emission")


def emission() -> None:
    """The systems with Ca II emission: recovered without it, with it, and with it fitted.

    The three runs are compared on the systems they share, by name: those of
    ``activity-emission`` whose simulated spectra have emission in the Ca II cores. The
    shares are unweighted, as in the table of effects outside the baseline of
    ``docs/reports/gaia-rvs-eclipsing-binaries.md``.
    """
    if not all((ROOT / name / "manifest.json").exists() for name in EMISSION_RUNS):
        return
    runs = {}
    for name in EMISSION_RUNS:
        systems, manifest, _ = load(ROOT / name)
        population = read_population(ROOT / name / "population.json")
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = [systems[f"e_k{k}"] / systems[f"k{k}_true"] - 1.0 for k in (1, 2)]
        finite = np.isfinite(ratio[0]) & np.isfinite(ratio[1])
        worst = np.where(finite, np.fmax(np.abs(ratio[0]), np.abs(ratio[1])), np.inf)
        runs[name] = (systems, list(manifest["declarations"]), population, worst)

    def rows_by_name(run: str, declaration: str) -> dict[str, int]:
        systems, declarations, population, _ = runs[run]
        if declaration not in declarations:
            return {}
        rows = np.flatnonzero(systems["decl"] == declarations.index(declaration))
        return {population[int(systems["system"][row])].name: int(row) for row in rows}

    population = runs["activity-emission"][2]
    injected = {s.name: s.meta.get("activity_ew", [0.0, 0.0]) for s in population}
    cases = [
        ("without emission", "fitted", "classified"),
        ("without emission", "fitted", "classified-fitted"),
        ("with emission", "activity-fitted", "classified"),
        ("with emission", "activity-fitted", "classified-fitted"),
        ("with emission, fitted with the labels", "activity-emission", "classified-fitted"),
    ]
    found = [(label, run, decl, rows_by_name(run, decl)) for label, run, decl in cases]
    names = [
        name
        for name, ew in injected.items()
        if max(ew) > 0.0 and all(name in rows for _, _, _, rows in found)
    ]
    if not names:
        return
    print(
        f"\n## Emission in the Ca II cores: the {len(names)} systems with emission that the "
        "three runs share (unweighted)\n"
    )
    print("| simulation | templates | both K within 10 percent [%] | within 5 percent [%] |")
    print("|---|---|---|---|")
    flags = {}
    for label, run, decl, rows in found:
        worst = runs[run][3]
        errors = np.array([worst[rows[name]] for name in names])
        flags[(run, decl)] = errors < 0.10
        print(
            f"| {label} | {decl} | {100.0 * np.mean(errors < 0.10):.0f} | "
            f"{100.0 * np.mean(errors < 0.05):.0f} |"
        )
    without, with_it = flags[("activity-fitted", "classified-fitted")], flags[cases[4][1:]]
    print(
        f"\nFitting the emission: {int(np.sum(~without & with_it))} systems gained and "
        f"{int(np.sum(without & ~with_it))} lost against the fit without it, of {len(names)}."
    )
    systems = runs["activity-emission"][0]
    rows = rows_by_name("activity-emission", "classified-fitted")
    active, quiet = [], []
    for s in population:
        if s.name not in rows:
            continue
        for k, ew in enumerate(injected[s.name], start=1):
            fitted = float(systems[f"f_emission{k}"][rows[s.name]])
            if np.isfinite(fitted):
                (active if ew > 0.0 else quiet).append((fitted, ew, s.grvs))
    for label, sample in (("with emission", active), ("without", quiet)):
        if not sample:
            continue
        fitted, ew, grvs = (np.array(x) for x in zip(*sample, strict=True))
        for lo, hi in ((6.0, 10.0), (10.0, 12.0), (12.0, 13.5)):
            keep = (grvs >= lo) & (grvs < hi)
            if keep.any():
                print(
                    f"stars {label}, G_RVS {lo:g}-{hi:g}: {int(keep.sum())}; injected equivalent "
                    f"width, median {np.median(ew[keep]):.2f}; fitted, median "
                    f"{np.median(fitted[keep]):.2f}; median absolute error "
                    f"{np.median(np.abs(fitted - ew)[keep]):.2f} [angstrom]"
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="*", default=["fitted"])
    args = parser.parse_args(argv)
    for name in args.runs:
        directory = ROOT / name
        if not (directory / "manifest.json").exists():
            print(f"{name}: no collected run in {directory}")
            continue
        table(directory)
        paired(directory)
    if set(EMISSION_RUNS) <= set(args.runs):
        emission()
    return 0


if __name__ == "__main__":
    sys.exit(main())

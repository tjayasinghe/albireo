"""Run every stage for a list of stars from one declaration (``internal/design.md`` D58).

The other examples apply one stage to one star. This one runs the whole chain
(disentangle, fit labels to the components, measure one velocity per component per epoch
against them, fit the orbit to the table, write the products and the figures) for a batch
of stars from a single declaration. It is what ``albireo run config.toml`` does from the
command line.

    python examples/13_pipeline.py            # in-process, two simulated stars
    python examples/13_pipeline.py --jobs 2   # the same batch in two worker processes

Both stars are simulated with known answers, so the reports carry an "against the
injected truth" block. The first is the packaged example, measured against its own
disentangled components: its velocities are differential, because a disentangled
component's rest frame is not identified and no synthetic grid is consulted, so the orbit
fitted to the table gives each component its own systemic velocity. The second star's
components are drawn from a toy synthetic library at known labels. The label stage fits
them back and measures each component's frame offset, so its velocities are absolute and
the orbit recovers the +12 km/s systemic velocity that the disentangling alone cannot
determine.

Three results to check in the output.

1. The flags. Every caveat a run records is printed at the end of each star's report and
   stored in ``result.json``. On the packaged star the flag states that the velocities are
   differential and why; on the toy star there should be none.
2. The zero point. The toy star's ``gamma`` against the injected +12 km/s, a quantity a
   disentangling cannot produce on its own.
3. The batch table. ``results.csv`` has one row per star (period, eccentricity,
   semi-amplitudes and systemic velocities with errors, labels, flags), and
   ``failures.txt`` lists any star that did not complete, without stopping the others.

Environment
-----------
``ALBIREO_EXAMPLE_FAST=1`` reduces every optimizer budget for CI (the same as ``--fast``).
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

import albireo as ab

FAST = bool(os.environ.get("ALBIREO_EXAMPLE_FAST"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", default="1", help="worker processes (an integer or 'auto')")
    parser.add_argument("--out", default="pipeline_example", help="output directory")
    parser.add_argument("--fast", action="store_true", help="trim the optimizer budgets")
    args = parser.parse_args(argv)

    fast = FAST or args.fast
    config = ab.demo_config(args.out, fast=fast)
    jobs = args.jobs if args.jobs == "auto" else int(args.jobs)
    print(f"{len(config.stars)} stars, jobs={jobs}, fast={fast}, output {args.out}\n")

    t0 = time.perf_counter()
    run = ab.run_pipeline(config, jobs=jobs)
    print(f"\nbatch wall {time.perf_counter() - t0:.1f} s")

    # ---- the products ---------------------------------------------------------
    packaged = run.results["sb2_sim"]
    toy = run.results["toy_library_sb2"]
    print("\nsb2_sim (packaged example, no library):")
    print("  " + "\n  ".join(packaged.flags))
    orbit = packaged.report["orbit"]
    print(
        f"  K from the table {orbit['k']['primary']:.3f} / {orbit['k']['secondary']:.3f} km/s "
        f"(injected 42 / 63); gamma mode: {orbit['gamma_mode']}"
    )
    print("\ntoy_library_sb2 (toy library, labels fitted):")
    orbit = toy.report["orbit"]
    labels = toy.report["labels"]["components"]
    print(
        f"  K from the table {orbit['k']['A']:.3f} / {orbit['k']['B']:.3f} km/s "
        f"(injected 30 / 55); gamma {orbit['gamma']['A']:+.3f} +- {orbit['gamma_err']['A']:.3f} "
        f"km/s (injected +12); absolute: {toy.report['velocities']['absolute_all']}"
    )
    for name, offsets in toy.report["truth"]["labels"].items():
        print(
            f"  {name}: Teff {labels[name]['teff']:.0f} K ({offsets['teff']:+.0f} from truth), "
            f"log g {labels[name]['logg']:.2f} ({offsets['logg']:+.2f}), "
            f"v sin i {labels[name]['vsini']:.1f} ({offsets['vsini']:+.1f})"
        )
    print(f"\n  flags: {toy.flags or 'none'}")
    print(f"\nproducts: {run.directory / 'results.csv'} and one directory per star")

    # ---- the gate ---------------------------------------------------------------
    assert not run.failures, run.failures
    assert packaged.report["velocities"]["absolute_all"] is False
    assert packaged.report["orbit"]["gamma_mode"] == "one per component"
    k = packaged.report["orbit"]["k"]
    assert abs(k["primary"] - 42.0) < 1.0 and abs(k["secondary"] - 63.0) < 1.5, k
    assert toy.report["velocities"]["absolute_all"] is True
    k = toy.report["orbit"]["k"]
    assert abs(k["A"] - 30.0) < 1.5 and abs(k["B"] - 55.0) < 2.5, k
    assert abs(toy.report["orbit"]["gamma"]["A"] - 12.0) < 1.5, toy.report["orbit"]["gamma"]
    # The truth block records differences (result minus injected), so these are offsets.
    for name, offsets in toy.report["truth"]["labels"].items():
        assert abs(offsets["teff"]) < 0.08 * labels[name]["teff"], (name, offsets)
    assert np.isfinite(list(toy.report["seconds"].values())).all()
    print(
        "\nOK - both stars through every stage from one declaration, the differential and "
        "absolute cases distinguished, and the systemic velocity recovered where the label "
        "fit measured the frame."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

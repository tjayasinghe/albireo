"""The ``albireo`` command: ``init``, ``run``, ``demo`` and ``fetch``.

A command line was deferred until the façade existed, because the façade is the
configuration schema and a CLI built first would have frozen a second one
(``docs/roadmap.md``). Now that it does, the command is a thin rendering of it:
``albireo init`` writes an annotated TOML whose tables are the façade's own vocabulary
plus the label and velocity stages, ``albireo run`` hands that file to
:func:`albireo.pipeline.run_pipeline`, ``albireo demo`` runs the same pipeline on two
simulated stars whose answers are known, and ``albireo fetch`` downloads a BLOeM star's
epochs and prints the ``[[stars]]`` entry that would analyse them.

Nothing scientific lives here. Every default the pipeline applies is stated in the
template it writes, and every claim it cannot make -- a light fraction, a wavelength
medium -- is required by the schema rather than defaulted by the command.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="albireo",
        description=(
            "Spectral disentangling, template identification and epoch velocities for "
            "spectroscopic binaries, in one command."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="write an annotated albireo.toml to edit")
    init.add_argument("path", nargs="?", default="albireo.toml", help="where to write it")
    init.add_argument("--force", action="store_true", help="overwrite an existing file")

    run = sub.add_parser("run", help="run every star of a configuration")
    run.add_argument("config", help="the TOML configuration (see `albireo init`)")
    _add_run_options(run)
    run.add_argument("--stars", nargs="+", metavar="NAME", help="run only these stars")

    demo = sub.add_parser(
        "demo", help="run the pipeline on two simulated stars with known answers (offline)"
    )
    _add_run_options(demo)
    demo.set_defaults(out="albireo_demo")

    fetch = sub.add_parser("fetch", help="download a BLOeM star's public epochs (network)")
    fetch.add_argument("targets", nargs="+", metavar="ID", help="BLOeM identifiers, e.g. 1-037")
    fetch.add_argument("--out", default="data", help="directory; one sub-directory per star")
    return parser


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", default=None, help="output directory (overrides the file)")
    parser.add_argument(
        "--jobs",
        default="1",
        help="worker processes: an integer, or 'auto' for cpu_count // 4 (default 1)",
    )
    parser.add_argument(
        "--fast", action="store_true", help="trim every optimizer budget for a smoke run"
    )
    parser.add_argument("--sample", action="store_true", help="run NUTS after the MAP (slow)")
    parser.add_argument("--no-plots", action="store_true", help="skip the figures")
    parser.add_argument("--quiet", action="store_true", help="no per-stage progress lines")


def _jobs(value: str) -> int | str:
    if value.lower() == "auto":
        return "auto"
    try:
        return int(value)
    except ValueError:
        raise SystemExit(f"--jobs must be an integer or 'auto'; got {value!r}") from None


def _apply_run_options(config, args):
    from dataclasses import replace

    analysis = config.analysis
    changes = {}
    if args.fast:
        changes["fast"] = True
    if args.sample:
        changes["sample"] = True
    if args.no_plots:
        changes["plots"] = False
    if changes:
        analysis = replace(analysis, **changes)
    output = config.output if args.out is None else args.out
    return replace(config, analysis=analysis, output=output)


def _cmd_init(args) -> int:
    from albireo.pipeline import write_config_template

    try:
        path = write_config_template(args.path, overwrite=args.force)
    except FileExistsError as exc:
        print(f"{exc} (or pass --force)", file=sys.stderr)
        return 1
    print(f"wrote {path}. Edit the stars, then: albireo run {path}")
    return 0


def _cmd_run(args) -> int:
    from albireo.pipeline import load_config, run_pipeline

    path = Path(args.config)
    if not path.is_file():
        print(f"no such configuration: {path}", file=sys.stderr)
        return 1
    try:
        config = load_config(path)
    except ValueError as exc:
        print(f"{path}: {exc}", file=sys.stderr)
        return 1
    config = _apply_run_options(config, args)
    run = run_pipeline(config, jobs=_jobs(args.jobs), stars=args.stars, progress=not args.quiet)
    if args.quiet:
        print(run.summary())
    return 0 if not run.failures else 2


def _cmd_demo(args) -> int:
    from albireo.pipeline import demo_config, run_pipeline

    config = demo_config(args.out or "albireo_demo", fast=args.fast, sample=args.sample)
    config = _apply_run_options(config, args)
    print(
        "albireo demo: two simulated stars with known answers. The packaged example is "
        "measured against its own components (differential velocities); the toy-library "
        "star has its labels fitted, so its velocities come out absolute and its systemic "
        "velocity is recovered.",
        flush=True,
    )
    run = run_pipeline(config, jobs=_jobs(args.jobs), progress=not args.quiet)
    if args.quiet:
        print(run.summary())
    for result in run.results.values():
        if result.ok and "truth" in result.report:
            print(f"\n{result.name}: {result.report['truth']}")
    print(f"\nRead {run.directory / 'summary.txt'} and each star's summary.txt and figures.")
    return 0 if not run.failures else 2


def _cmd_fetch(args) -> int:
    from albireo import archive

    out = Path(args.out)
    entries = []
    status = 0
    for target in args.targets:
        try:
            star = archive.resolve_bloem(target)
            records = archive.bloem_spectra(star, public_only=True)
        except Exception as exc:
            print(f"{target}: {type(exc).__name__}: {exc}", file=sys.stderr)
            status = 1
            continue
        if not records:
            print(f"{target}: no public spectra", file=sys.stderr)
            status = 1
            continue
        directory = out / f"bloem-{star.bloem_id}"
        statuses = archive.download(records, directory)
        failed = [s for s in statuses if s.startswith("FAIL")]
        print(
            f"BLOeM {star.bloem_id} (Gaia DR3 {star.gaia_dr3}, {star.spectral_type or '?'}, "
            f"{star.binary_class or 'unclassified'}): {len(records) - len(failed)} of "
            f"{len(records)} epochs in {directory}"
        )
        if failed:
            status = 1
        entries.append(
            "\n".join(
                [
                    "[[stars]]",
                    f'name = "BLOeM {star.bloem_id}"',
                    f'spectra = "{os.fspath(directory / "*.fits").replace(os.sep, "/")}"',
                    'period = "search"        # or [lo, hi] once a period is known',
                    "region = [4120.0, 4300.0]",
                    'medium = "air"',
                    "",
                    "[[stars.components]]",
                    'name = "primary"',
                    "light = 0.6              # an assumption: quote it",
                    "",
                    "[[stars.components]]",
                    'name = "secondary"',
                    "light = 0.4",
                ]
            )
        )
    if entries:
        print("\nAdd to your albireo.toml (an [instrument.GIRAFFE] table with R = 6300 too):\n")
        print("\n\n".join(entries))
    return status


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``albireo`` console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers = {"init": _cmd_init, "run": _cmd_run, "demo": _cmd_demo, "fetch": _cmd_fetch}
    return handlers[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""The ``albireo`` command: ``init``, ``run``, ``demo``, ``fetch`` and ``apf-wavelengths``.

The command is a thin layer over :mod:`albireo.pipeline`, whose configuration schema is
that of the ``Disentangler`` interface plus the label and velocity stages. ``albireo init``
writes an annotated TOML configuration. ``albireo run`` passes such a file to
:func:`albireo.pipeline.run_pipeline`. ``albireo demo`` runs the same pipeline on two
simulated stars with known injected values. ``albireo fetch`` downloads a BLOeM star's
public epochs and prints the ``[[stars]]`` entry that would analyse them.
``albireo apf-wavelengths`` refits an APF wavelength solution to the stellar and telluric
lines of one exposure (:func:`albireo.apf.refine_wavelengths`), for the ``[read.apf]``
table of a configuration.

No scientific decision is made here. Every default the pipeline applies is stated in the
template it writes, and the quantities the pipeline cannot infer (a light fraction, a
wavelength medium) are required by the schema and have no default.
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

    wavecal = sub.add_parser(
        "apf-wavelengths",
        help="refit an APF wavelength solution to one exposure's stellar and telluric lines",
    )
    wavecal.add_argument("reference", help="a CPS r*.fits exposure of a line-rich star (G or K)")
    wavecal.add_argument(
        "--base", required=True, help="the solution to refine (apf_wave_2022.fits)"
    )
    wavecal.add_argument("--out", required=True, help="where to write the refined solution (FITS)")
    wavecal.add_argument("--teff", type=float, default=4750.0, help="template Teff, K (4750)")
    wavecal.add_argument("--logg", type=float, default=4.5, help="template log g (4.5)")
    wavecal.add_argument("--mh", type=float, default=0.0, help="template [M/H] (0.0)")
    wavecal.add_argument(
        "--library",
        default="bosz2024-fgk-r20000",
        help="the synthetic grid (albireo.library_names(); downloaded once)",
    )
    wavecal.add_argument("--force", action="store_true", help="overwrite --out")
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
    if args.stars:
        names = [s.name for s in config.stars]
        unknown = sorted(set(args.stars) - set(names))
        if unknown:
            print(f"{path}: unknown star(s) {unknown}; the batch has {names}", file=sys.stderr)
            return 1
    # The command reads only the written products, so no star's live objects are kept, and
    # the programs compiled for one star are released before the next.
    run = run_pipeline(
        config,
        jobs=_jobs(args.jobs),
        stars=args.stars,
        progress=not args.quiet,
        keep_live=False,
    )
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
    run = run_pipeline(config, jobs=_jobs(args.jobs), progress=not args.quiet, keep_live=False)
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


def _cmd_apf_wavelengths(args) -> int:
    import numpy as np

    import albireo as ab
    from albireo import apf

    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"{out} exists (pass --force to overwrite)", file=sys.stderr)
        return 1
    try:
        exposure = apf.read_apf_exposure(args.reference)
        library = ab.fetch_library(args.library)
        lo = max(float(np.min(library.wave)) + 1.0, 3800.0)
        hi = min(float(np.max(library.wave)) - 1.0, 7400.0)
        grid = ab.LogGrid.from_wavelength_range(lo, hi, dv_kms=0.25)
        labels = {"teff": args.teff, "logg": args.logg, "mh": args.mh}
        template = ab.Template.from_library("reference", library, labels, grid=grid, medium="air")
        result = apf.refine_wavelengths(
            args.base,
            exposure,
            template_wave=np.asarray(grid.wave),
            template_flux=1.0 + np.asarray(template.deviation),
            template_resolving_power=library.resolving_power,
        )
    except (OSError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    result.wavelengths.save(out, overwrite=args.force)
    print(result.summary())
    print(f'wrote {out}. In a configuration: [read.apf] wavelengths = "{out.as_posix()}"')
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``albireo`` console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "init": _cmd_init,
        "run": _cmd_run,
        "demo": _cmd_demo,
        "fetch": _cmd_fetch,
        "apf-wavelengths": _cmd_apf_wavelengths,
    }
    return handlers[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

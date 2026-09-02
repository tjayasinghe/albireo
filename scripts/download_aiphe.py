"""Download the AI Phoenicis HARPS spectra from the ESO Science Archive.

Fetches the 36 Phase-3 (``SCIENCE.SPECTRUM``) 1-D spectra of AI Phe (HD 6980) taken with
HARPS on the ESO 3.6 m. They are public, so no ESO login is required.

AI Phe is the validation target with an external answer. It is a detached eclipsing binary
whose orbit is published to 0.02 per cent (Maxted et al. 2020, MNRAS 498, 332) and whose
component temperatures are known independently, 6310 K and 5010 K, so that
``scripts/aiphe_labels_bench.py`` can score a label fit against values external to the fit.

The files land in ``data/aiphe/`` as ``ADP.<id>.fits`` (~5 MB each, ~194 MB total) with a
``manifest.json`` recording the archive metadata and each download's outcome. Re-running
skips files already present at the expected size, so an interrupted download resumes.

Like ``download_hr6819.py``, this is a thin wrapper over :mod:`albireo.archive`, kept as a
worked invocation of it.

Usage
-----
    python scripts/download_aiphe.py [--outdir DIR] [--force] [--jobs N]

See ``scripts/aiphe_bench.py`` for the disentangling benchmark and
``scripts/aiphe_labels_bench.py`` for the label validation.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from albireo.archive import download, query, spectra_query

# AI Phe = HD 6980 = AI Phoenicis (ICRS, J2000). Resolved to coordinates rather than looked
# up by name for the same reason as HR 6819: ESO's target_name is PI free text, and these
# 36 files are filed under several spellings across more than one programme.
TARGET_RA_DEG = 17.392458
TARGET_DEC_DEG = -46.265583
SEARCH_RADIUS_DEG = 0.05
INSTRUMENT = "HARPS"
EXPECTED_N = 36


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_out = pathlib.Path(__file__).resolve().parent.parent / "data" / "aiphe"
    parser.add_argument("--outdir", type=pathlib.Path, default=default_out)
    parser.add_argument("--force", action="store_true", help="re-download files already present")
    parser.add_argument("--jobs", type=int, default=4, help="parallel downloads (default 4)")
    args = parser.parse_args(argv)

    print(f"querying the ESO archive for {INSTRUMENT} spectra of AI Phe ...")
    records = query(
        spectra_query(
            ra_deg=TARGET_RA_DEG,
            dec_deg=TARGET_DEC_DEG,
            radius_deg=SEARCH_RADIUS_DEG,
            instrument=INSTRUMENT,
        ),
        maxrec=1000,
    )
    total_mb = sum((r.size_bytes or 0) for r in records) / 1e6
    print(f"  {len(records)} spectra, {total_mb:.0f} MB total")
    if len(records) != EXPECTED_N:
        print(
            f"  note: expected {EXPECTED_N} spectra, the archive returned {len(records)}; "
            "the holdings may have grown. Proceeding.",
            file=sys.stderr,
        )

    results = download(
        records,
        args.outdir,
        force=args.force,
        jobs=args.jobs,
        progress=lambda done, total, line: print(f"  [{done:3d}/{total}] {line}", flush=True),
    )

    failures = [line for line in results if line.startswith("FAIL")]
    print(f"\n{len(results) - len(failures)}/{len(results)} files in {args.outdir}")
    if failures:
        print("re-run the script to retry the failures", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Download the HR 6819 FEROS spectra from the ESO Science Archive.

Fetches the 51 Phase-3 (``SCIENCE.SPECTRUM``) 1-D spectra of HR 6819 taken under ESO
programme 073.D-0274(A) (PI Rivinius) with FEROS on the MPG/ESO 2.2 m. These are the
spectra behind Rivinius et al. (2020) and the reanalyses that followed; they are public,
so no ESO login is required.

The files land in ``data/hr6819/`` as ``ADP.<id>.fits`` (~3 MB each, ~153 MB total) with a
``manifest.json`` recording the archive metadata and each download's outcome. Re-running
skips files already present at the expected size, so an interrupted download resumes.

This script is a thin wrapper over :mod:`albireo.archive`, the same client generalized to
any ESO programme, instrument or sky position (``docs/api/archive.md``). It is kept because
the HR 6819 dataset is the one the benchmark record and
``examples/03_hr6819_real_data.py`` are built on, and as a worked invocation.

Usage
-----
    python scripts/download_hr6819.py [--outdir DIR] [--force] [--jobs N]

See ``examples/03_hr6819_real_data.py`` for what to do with the result.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from albireo.archive import download, query, spectra_query

# HR 6819 = HD 167128 = QV Tel (ICRS, J2000). Resolved to coordinates rather than looked up
# by name because ESO's target_name is PI free text and is not resolver-normalized: this
# object is filed as 'HR-6819', hyphenated.
TARGET_RA_DEG = 274.28139
TARGET_DEC_DEG = -56.02337
SEARCH_RADIUS_DEG = 0.05
PROGRAMME = "073.D-0274(A)"
INSTRUMENT = "FEROS"
EXPECTED_N = 51


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_out = pathlib.Path(__file__).resolve().parent.parent / "data" / "hr6819"
    parser.add_argument("--outdir", type=pathlib.Path, default=default_out)
    parser.add_argument("--force", action="store_true", help="re-download files already present")
    parser.add_argument("--jobs", type=int, default=4, help="parallel downloads (default 4)")
    args = parser.parse_args(argv)

    print(f"querying the ESO archive for {INSTRUMENT} spectra of HR 6819 ({PROGRAMME}) ...")
    records = query(
        spectra_query(
            ra_deg=TARGET_RA_DEG,
            dec_deg=TARGET_DEC_DEG,
            radius_deg=SEARCH_RADIUS_DEG,
            instrument=INSTRUMENT,
            programme=PROGRAMME,
        ),
        maxrec=1000,
    )
    total_mb = sum((r.size_bytes or 0) for r in records) / 1e6
    print(f"  {len(records)} spectra, {total_mb:.0f} MB total")
    if len(records) != EXPECTED_N:
        print(
            f"  note: expected {EXPECTED_N} spectra, the archive returned {len(records)}; "
            "the programme may have been re-released. Proceeding.",
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

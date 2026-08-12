"""Download the HR 6819 FEROS spectra from the ESO Science Archive.

Fetches the 51 Phase-3 (``SCIENCE.SPECTRUM``, calib_level 2) 1-D spectra of HR 6819
taken under ESO programme 073.D-0274(A) (PI Rivinius) with FEROS on the MPG/ESO 2.2 m.
These are the spectra behind Rivinius et al. (2020) and the reanalyses that followed;
they are public, so no ESO login is required.

The files land in ``data/hr6819/`` as ``ADP.<id>.fits`` (~3 MB each, ~153 MB total) with
a ``manifest.json`` recording the archive metadata. Re-running the script skips files
that are already present with the expected size, so an interrupted download resumes.

Usage
-----
    python scripts/download_hr6819.py [--outdir DIR] [--force] [--jobs N]

See ``examples/03_hr6819_real_data.py`` for what to do with the result.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

TAP_URL = "https://archive.eso.org/tap_obs/sync"
FILE_URL = "https://dataportal.eso.org/dataPortal/file/{dp_id}"
USER_AGENT = "albireo/0.1 (https://github.com/tjayasinghe/albireo)"

# HR 6819 = HD 167128 = QV Tel (ICRS, J2000).
TARGET_RA_DEG = 274.28139
TARGET_DEC_DEG = -56.02337
SEARCH_RADIUS_DEG = 0.05
PROGRAMME = "073.D-0274(A)"
INSTRUMENT = "FEROS"
EXPECTED_N = 51

_QUERY = f"""
SELECT dp_id, target_name, proposal_id, instrument_name, t_min, t_max, t_exptime,
       em_min, em_max, em_res_power, snr, access_estsize, obs_creator_did
FROM ivoa.ObsCore
WHERE INTERSECTS(s_region, CIRCLE('ICRS',{TARGET_RA_DEG},{TARGET_DEC_DEG},{SEARCH_RADIUS_DEG}))=1
  AND instrument_name='{INSTRUMENT}'
  AND proposal_id='{PROGRAMME}'
  AND dataproduct_type='spectrum'
ORDER BY t_min
"""


# The ESO endpoints refuse or drop connections under load often enough that a bare
# urlopen fails a few times per 51-file run; retrying is the difference between a
# script the user can trust and one they have to babysit.
_TRANSIENT = (urllib.error.URLError, TimeoutError, ConnectionError, OSError)


def _with_retries[T](what: str, call: Callable[[], T], attempts: int = 5) -> T:
    """Run ``call``, retrying transient network failures with exponential backoff."""
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except urllib.error.HTTPError as exc:
            # 4xx other than 429 will not fix themselves; fail fast with a clear message.
            if exc.code != 429 and exc.code < 500:
                raise
            last: Exception = exc
        except _TRANSIENT as exc:
            last = exc
        if attempt == attempts:
            raise RuntimeError(f"{what} failed after {attempts} attempts: {last!r}") from last
        print(f"    retry {attempt}/{attempts - 1} for {what}: {last}", file=sys.stderr)
        time.sleep(delay)
        delay *= 2
    raise AssertionError("unreachable")


def query_archive(timeout: float = 300.0) -> list[dict]:
    """Return one metadata dict per archive spectrum, sorted by observation time."""
    payload = urllib.parse.urlencode(
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json", "MAXREC": "1000", "QUERY": _QUERY}
    ).encode()

    def fetch() -> dict:
        request = urllib.request.Request(TAP_URL, data=payload, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", "replace"))

    table = _with_retries("TAP query", fetch)
    columns = [meta["name"] for meta in table["metadata"]]
    return [dict(zip(columns, row, strict=True)) for row in table["data"]]


def local_filename(dp_id: str) -> str:
    """Filesystem-safe name for an ESO dataset id.

    ESO ids embed an ISO timestamp (``ADP.2016-09-20T09:32:35.364``), and a colon is not a
    legal character in a Windows filename — NTFS reads ``name:stream`` as an alternate data
    stream, so the open fails with a bare ``EINVAL``. Colons become hyphens on every platform
    so a data directory copied between machines keeps the same names.
    """
    return dp_id.replace(":", "-") + ".fits"


def download_one(record: dict, outdir: pathlib.Path, force: bool, timeout: float = 600.0) -> str:
    """Download one spectrum; return a one-line status string."""
    dp_id = record["dp_id"]
    path = outdir / local_filename(dp_id)
    expected_bytes = int(record["access_estsize"]) * 1024
    if path.exists() and not force:
        size = path.stat().st_size
        # access_estsize is rounded to whole KB, so allow a kilobyte of slack.
        if abs(size - expected_bytes) <= 1024:
            return f"skip  {dp_id}  ({size / 1e6:.1f} MB already present)"

    url = FILE_URL.format(dp_id=urllib.parse.quote(dp_id))
    tmp = path.with_name(path.name + ".part")

    def fetch() -> None:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)

    try:
        _with_retries(dp_id, fetch)
    except (RuntimeError, urllib.error.HTTPError) as exc:
        tmp.unlink(missing_ok=True)
        return f"FAIL  {dp_id}  {type(exc).__name__}: {exc}"
    size = tmp.stat().st_size
    if size < 1024:
        tmp.unlink(missing_ok=True)
        return f"FAIL  {dp_id}  suspiciously small response ({size} bytes)"
    tmp.replace(path)
    return f"got   {dp_id}  {size / 1e6:.1f} MB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_out = pathlib.Path(__file__).resolve().parent.parent / "data" / "hr6819"
    parser.add_argument("--outdir", type=pathlib.Path, default=default_out)
    parser.add_argument("--force", action="store_true", help="re-download files already present")
    parser.add_argument("--jobs", type=int, default=4, help="parallel downloads (default 4)")
    args = parser.parse_args(argv)

    print(f"querying {TAP_URL} for {INSTRUMENT} spectra of HR 6819 ({PROGRAMME}) ...")
    records = query_archive()
    total_mb = sum(int(r["access_estsize"]) for r in records) / 1024
    print(f"  {len(records)} spectra, {total_mb:.0f} MB total")
    if len(records) != EXPECTED_N:
        print(
            f"  note: expected {EXPECTED_N} spectra, the archive returned {len(records)}; "
            "the programme may have been re-released. Proceeding.",
            file=sys.stderr,
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = [dict(r, local_file=local_filename(r["dp_id"])) for r in records]
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        results = list(pool.map(lambda r: download_one(r, args.outdir, args.force), records))
    for line in results:
        print(" ", line)

    failures = [line for line in results if line.startswith("FAIL")]
    print(f"\n{len(results) - len(failures)}/{len(results)} files in {args.outdir}")
    if failures:
        print("re-run the script to retry the failures", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

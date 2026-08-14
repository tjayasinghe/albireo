"""Query and download reduced spectra from the ESO Science Archive.

One query language, one file layout, one proprietary period: the ESO archive publishes
Phase 3 one-dimensional spectra for every instrument albireo's users are likely to have
data from — FEROS, HARPS, UVES, X-shooter, GIRAFFE and ESPRESSO all deliver
``SCIENCE.SPECTRUM`` products, and everything becomes public a year after observation.
That makes a single loader worth more than six.

This module is the fetch half: an ObsCore/TAP client and a downloader. Reading what comes
back is :mod:`albireo.io`'s job. It is deliberately **stdlib only** — ``urllib``, ``json``,
``concurrent.futures`` — so that finding data costs no dependency; astropy is needed only
once you open a file.

Three things about the archive drive the design, all measured rather than assumed:

**Cone-search, do not name-search.** ``target_name`` is PI free text and is not
resolver-normalized: HR 6819 is filed as ``HR-6819``. Resolve to coordinates elsewhere and
search a circle. And for one-dimensional spectra ``s_region`` is a bare
``POSITION J2000 ra dec`` — a point, not a footprint — so the form ESO's own documentation
shows for images, ``CONTAINS(POINT(...), s_region)=1``, matches *nothing*. Use
``INTERSECTS(s_region, CIRCLE(...))``.

**Truncation is silent.** The sync endpoint caps output at 20,000 rows by default, and the
JSON and CSV serializations carry no overflow marker — only VOTable does. A query that hits
the cap looks exactly like a query that did not. :func:`query` therefore compares the row
count against ``maxrec`` and raises rather than returning a quietly short list.

**One row is not one epoch.** GIRAFFE writes one file per science fibre (up to 130 per raw
frame) and multi-epoch stacks exist with ``M_EPOCH=True`` spanning weeks. Filtering to the
products you actually want is the caller's job, and :func:`query` returns the metadata that
makes it possible.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

__all__ = [
    "ArchiveRecord",
    "download",
    "local_filename",
    "query",
    "spectra_query",
]

TAP_SYNC_URL = "https://archive.eso.org/tap_obs/sync"
FILE_URL = "https://dataportal.eso.org/dataPortal/file/{dp_id}"

# The columns that ESO populates for essentially every spectrum. Measured over 2.4 million
# rows: target_name, s_ra/s_dec, t_min/t_max, t_exptime, em_min/em_max, em_res_power,
# obs_collection, proposal_id, calib_level, access_estsize, obs_release_date and
# dataproduct_subtype are 100% non-null; snr is 99.9996%. Deliberately absent:
# s_resolution, abmaglim and filter are *always* null for spectra, so asking for them
# would only invite someone to depend on them.
DEFAULT_COLUMNS = (
    "dp_id",
    "target_name",
    "proposal_id",
    "instrument_name",
    "obs_collection",
    "t_min",
    "t_max",
    "t_exptime",
    "em_min",
    "em_max",
    "em_res_power",
    "snr",
    "calib_level",
    "dataproduct_subtype",
    "access_estsize",
    "obs_release_date",
    "obs_creator_did",
)


def _user_agent() -> str:
    """Identify the client, with the real version rather than a hardcoded one."""
    from albireo import __version__

    return f"albireo/{__version__} (https://github.com/tjayasinghe/albireo)"


# The ESO endpoints refuse or drop connections under load often enough that a bare urlopen
# fails a few times in a 50-file run; retrying is the difference between a fetch the user
# can trust and one they have to babysit. OSError is deliberately NOT in this tuple: it
# would swallow local disk errors (a full filesystem on the .part write) and retry them
# five times before reporting them as network trouble.
_TRANSIENT = (urllib.error.URLError, TimeoutError, ConnectionError)


def _with_retries[T](what: str, call: Callable[[], T], attempts: int = 5, quiet: bool = False) -> T:
    """Run ``call``, retrying transient network failures with exponential backoff."""
    delay = 2.0
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except urllib.error.HTTPError as exc:
            # 4xx other than 429 will not fix themselves; fail fast with a clear message.
            if exc.code != 429 and exc.code < 500:
                raise
            last = exc
        except _TRANSIENT as exc:
            last = exc
        if attempt == attempts:
            raise RuntimeError(f"{what} failed after {attempts} attempts: {last!r}") from last
        if not quiet:
            print(f"    retry {attempt}/{attempts - 1} for {what}: {last}", file=sys.stderr)
        time.sleep(delay)
        delay *= 2
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class ArchiveRecord:
    """One archive product, as ObsCore describes it.

    A thin typed view over the TAP row rather than a re-modelling of it: :attr:`row` keeps
    every column the query asked for, so a caller that needs something this class does not
    name is never blocked.
    """

    dp_id: str
    row: dict

    @property
    def target_name(self) -> str:
        return str(self.row.get("target_name", ""))

    @property
    def instrument(self) -> str:
        return str(self.row.get("instrument_name", ""))

    @property
    def size_bytes(self) -> int | None:
        """Estimated size. ``access_estsize`` is in kilobytes, and may be null."""
        value = self.row.get("access_estsize")
        return None if value is None else int(value) * 1024

    @property
    def filename(self) -> str:
        return local_filename(self.dp_id)

    @property
    def url(self) -> str:
        return FILE_URL.format(dp_id=urllib.parse.quote(self.dp_id))


def local_filename(dp_id: str) -> str:
    """Filesystem-safe name for an ESO dataset id.

    ESO ids embed an ISO timestamp (``ADP.2016-09-20T09:32:35.364``), and a colon is not a
    legal character in a Windows filename — NTFS reads ``name:stream`` as an alternate data
    stream, so the open fails with a bare ``EINVAL``. Colons become hyphens on every
    platform so a data directory copied between machines keeps the same names.
    """
    return dp_id.replace(":", "-") + ".fits"


def spectra_query(
    *,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    radius_deg: float = 0.05,
    instrument: str | None = None,
    programme: str | None = None,
    collection: str | None = None,
    calib_level: int | None = None,
    columns: Sequence[str] = DEFAULT_COLUMNS,
) -> str:
    """Build the ADQL for a Phase 3 spectrum search. Every constraint is optional but one.

    Parameters
    ----------
    ra_deg, dec_deg, radius_deg
        Cone search in ICRS degrees. Give both coordinates or neither. Expressed as
        ``INTERSECTS(s_region, CIRCLE(...))``, which is the form that works for spectra —
        ``s_region`` is a bare point for 1-D products, so the ``CONTAINS(POINT(...), ...)``
        form in ESO's own examples silently returns nothing.
    instrument
        ``instrument_name``, uppercase and unhyphenated as the archive stores it:
        ``FEROS``, ``HARPS``, ``UVES``, ``XSHOOTER``, ``GIRAFFE``, ``ESPRESSO``.
    programme
        ``proposal_id``, exact match including the parenthesised run letter, e.g.
        ``073.D-0274(A)``. ESO Large Programmes split into sub-runs, so a prefix match is
        usually what you want — pass ``112.25R7%`` and it is issued as ``LIKE``.
    collection
        ``obs_collection``. Worth setting when an instrument's products are split between
        ESO's own pipeline (e.g. ``GIRAFFE``) and a community release (``GAIAESO``), which
        differ in header content.
    calib_level
        2 for ordinary Phase 3 products, 3 for stacked/community ones.
    columns
        ObsCore columns to select.

    Returns
    -------
    str
        The ADQL query.
    """
    if (ra_deg is None) != (dec_deg is None):
        raise ValueError("give both ra_deg and dec_deg, or neither")
    where = ["dataproduct_type='spectrum'"]
    if ra_deg is not None and dec_deg is not None:
        if not radius_deg > 0:
            raise ValueError(f"radius_deg must be positive; got {radius_deg}")
        where.append(
            f"INTERSECTS(s_region, CIRCLE('ICRS',{ra_deg!r},{dec_deg!r},{radius_deg!r}))=1"
        )
    if instrument is not None:
        where.append(f"instrument_name='{_quote(instrument)}'")
    if programme is not None:
        op = "LIKE" if "%" in programme else "="
        where.append(f"proposal_id {op} '{_quote(programme)}'")
    if collection is not None:
        where.append(f"obs_collection='{_quote(collection)}'")
    if calib_level is not None:
        where.append(f"calib_level={int(calib_level)}")
    if len(where) == 1:
        raise ValueError(
            "a query needs at least one constraint besides dataproduct_type — an "
            "unconstrained search would ask the archive for every spectrum it holds "
            "(2.4 million and counting). Give coordinates, an instrument, or a programme."
        )
    return (
        f"SELECT {', '.join(columns)}\nFROM ivoa.ObsCore\nWHERE "
        + "\n  AND ".join(where)
        + "\nORDER BY t_min"
    )


def _quote(value: str) -> str:
    """Escape a string for an ADQL literal, and refuse anything that looks like injection."""
    text = str(value)
    if "'" in text:
        raise ValueError(f"ADQL string literals may not contain a single quote; got {text!r}")
    return text


def query(
    adql: str,
    *,
    maxrec: int = 20000,
    timeout: float = 300.0,
    url: str = TAP_SYNC_URL,
) -> list[ArchiveRecord]:
    """Run an ADQL query against the ESO TAP service and return the rows.

    Parameters
    ----------
    adql
        The query, e.g. from :func:`spectra_query`.
    maxrec
        Row cap. **Reaching it raises**, because the archive does not say when it
        truncates: the sync endpoint's JSON and CSV serializations carry no overflow
        marker (only VOTable does), so a capped result is indistinguishable from a
        complete one. Silently returning half a programme's epochs is exactly the kind of
        quiet wrongness that produces a confident wrong orbit.
    timeout
        Socket timeout in seconds. The service's own default execution limit is 60 s.
    url
        TAP sync endpoint.

    Returns
    -------
    list of ArchiveRecord
        In the order the query asked for.
    """
    payload = urllib.parse.urlencode(
        {
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "json",
            "MAXREC": str(int(maxrec)),
            "QUERY": adql,
        }
    ).encode()

    def fetch() -> dict:
        request = urllib.request.Request(url, data=payload, headers={"User-Agent": _user_agent()})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", "replace"))

    table = _with_retries("TAP query", fetch)
    try:
        columns = [meta["name"] for meta in table["metadata"]]
        data = table["data"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"the TAP service returned something that is not an ESO result table "
            f"(keys: {sorted(table) if isinstance(table, dict) else type(table).__name__}). "
            "This is usually an ADQL syntax error reported as a document rather than an "
            f"HTTP error. The query was:\n{adql}"
        ) from exc

    if len(data) >= maxrec:
        raise RuntimeError(
            f"the query returned {len(data)} rows, which is the MAXREC cap — the result is "
            "almost certainly truncated, and ESO's JSON output carries no overflow marker "
            "to confirm it either way. Raise maxrec or narrow the query. Returning a "
            "silently short list would be worse than this error."
        )
    rows = [dict(zip(columns, row, strict=True)) for row in data]
    return [ArchiveRecord(dp_id=str(row["dp_id"]), row=row) for row in rows]


def _download_one(record: ArchiveRecord, outdir: pathlib.Path, force: bool, timeout: float) -> str:
    """Download one product; return a one-line status string starting FAIL, skip or got."""
    path = outdir / record.filename
    expected = record.size_bytes
    if path.exists() and not force:
        size = path.stat().st_size
        # access_estsize is rounded to whole kB, so allow a kilobyte of slack. With no
        # estimate at all, any existing non-empty file is taken as complete — the
        # alternative is re-downloading it on every run.
        if expected is None or abs(size - expected) <= 1024:
            if size > 0:
                return f"skip  {record.dp_id}  ({size / 1e6:.1f} MB already present)"

    tmp = path.with_name(path.name + ".part")

    def fetch() -> None:
        request = urllib.request.Request(record.url, headers={"User-Agent": _user_agent()})
        with (
            urllib.request.urlopen(request, timeout=timeout) as response,
            tmp.open("wb") as handle,
        ):
            declared = response.headers.get("Content-Length")
            written = 0
            while chunk := response.read(1 << 20):
                handle.write(chunk)
                written += len(chunk)
            # A transfer cut cleanly in the middle is still a valid HTTP response; without
            # this check a truncated FITS file lands on disk looking complete.
            if declared is not None and written != int(declared):
                raise ConnectionError(
                    f"truncated download: got {written} bytes, Content-Length said {declared}"
                )

    try:
        _with_retries(record.dp_id, fetch)
    except (RuntimeError, urllib.error.HTTPError) as exc:
        tmp.unlink(missing_ok=True)
        return f"FAIL  {record.dp_id}  {type(exc).__name__}: {exc}"
    size = tmp.stat().st_size
    if size < 1024:
        tmp.unlink(missing_ok=True)
        return f"FAIL  {record.dp_id}  suspiciously small response ({size} bytes)"
    tmp.replace(path)
    return f"got   {record.dp_id}  {size / 1e6:.1f} MB"


def download(
    records: Sequence[ArchiveRecord],
    outdir: str | pathlib.Path,
    *,
    force: bool = False,
    jobs: int = 4,
    timeout: float = 600.0,
    manifest: bool = True,
    progress: Callable[[int, int, str], None] | None = None,
) -> list[str]:
    """Download products into ``outdir``, resumably, and write a manifest.

    Each file lands via a ``.part`` temporary and an atomic rename, so an interrupted run
    never leaves a half-written FITS file that looks complete; re-running skips what is
    already there. Downloads are checked against ``Content-Length`` where the server sends
    it, because a cleanly-truncated transfer is otherwise indistinguishable from success.

    Parameters
    ----------
    records
        From :func:`query`.
    outdir
        Created if missing.
    force
        Re-download files already present.
    jobs
        Parallel downloads. ESO publishes no rate limit; 4 is a deliberately polite
        default rather than a tuned one.
    timeout
        Per-file socket timeout in seconds.
    manifest
        Write ``manifest.json`` recording every product's archive metadata *and the
        outcome of its download*. Written after the transfers rather than before, so it
        cannot claim a file that failed.
    progress
        Optional ``callback(done, total, status_line)``.

    Returns
    -------
    list of str
        One status line per record, in input order, each starting ``got``, ``skip`` or
        ``FAIL``.
    """
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    records = list(records)
    if not records:
        return []

    results: list[str] = [""] * len(records)
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, int(jobs))) as pool:
        futures = {
            pool.submit(_download_one, record, outdir, force, timeout): index
            for index, record in enumerate(records)
        }
        # Completion order, not submission order, so `progress` reports each file as it
        # actually lands; `results` is still indexed back into input order.
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
            done += 1
            if progress is not None:
                progress(done, len(records), results[index])

    if manifest:
        payload = {
            "n_records": len(records),
            "records": [
                dict(record.row, local_file=record.filename, status=status.split()[0])
                for record, status in zip(records, results, strict=True)
            ],
        }
        (outdir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return results

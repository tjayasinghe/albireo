"""Query and download reduced spectra from the ESO Science Archive.

**Experimental.** This is a thin layer over an external service; its query surface
follows the archive's, not a stability promise of albireo's.

The ESO archive publishes Phase 3 one-dimensional spectra under one query language, one
file layout and one proprietary period: FEROS, HARPS, UVES, X-shooter, GIRAFFE and
ESPRESSO all deliver ``SCIENCE.SPECTRUM`` products, and every product becomes public a
year after observation. One loader therefore serves all of them.

This module is the fetch half, an ObsCore/TAP client and a downloader; reading the files
is the job of :mod:`albireo.io`. It uses the standard library only (``urllib``, ``json``,
``concurrent.futures``), so locating data adds no dependency; astropy is needed only to
open a file.

Three properties of the archive determine the design. Searches are by cone, not by name:
``target_name`` is PI free text and is not resolver-normalized (HR 6819 is filed as
``HR-6819``), and for one-dimensional spectra ``s_region`` is a bare
``POSITION J2000 ra dec`` (a point, not a footprint), so the form ESO's own documentation
shows for images, ``CONTAINS(POINT(...), s_region)=1``, matches nothing;
``INTERSECTS(s_region, CIRCLE(...))`` is used instead. Truncation is silent: the sync
endpoint caps output at 20,000 rows by default, and the JSON and CSV serializations carry
no overflow marker (only VOTable does), so :func:`query` compares the row count against
``maxrec`` and raises at the cap. An archive row is not necessarily one epoch: GIRAFFE
delivers one file per science fibre (up to 130 per raw frame), and multi-epoch stacks
spanning weeks are flagged with ``M_EPOCH=True``; :func:`query` returns the metadata
needed to select products, and the selection itself is left to the caller.

The BLOeM functions (:func:`bloem_spectra`, :class:`BloemTarget`) apply all three to one
survey: a BLOeM identifier such as ``"1-002"`` is resolved to that star's ~25 epochs. The
survey's spectra are filed under the Gaia DR3 source id rather than the survey's own name,
and the cross-match is a VizieR table; VizieR serves the same TAP dialect, so the join
needs no additional dependency.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

__all__ = [
    "ArchiveRecord",
    "BloemTarget",
    "bloem_catalogue",
    "bloem_spectra",
    "download",
    "local_filename",
    "query",
    "resolve_bloem",
    "spectra_query",
]

TAP_SYNC_URL = "https://archive.eso.org/tap_obs/sync"
FILE_URL = "https://dataportal.eso.org/dataPortal/file/{dp_id}"

# VizieR serves the same IVOA TAP dialect as ESO, so a survey's published catalogue and its
# archived spectra can be joined with one client. The u-strasbg.fr host is the same service
# under its older name.
VIZIER_SYNC_URL = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"

# The columns ESO populates for essentially every spectrum, measured over 2.4 million rows:
# target_name, s_ra/s_dec, t_min/t_max, t_exptime, em_min/em_max, em_res_power,
# obs_collection, proposal_id, calib_level, access_estsize, obs_release_date and
# dataproduct_subtype are 100% non-null; snr is 99.9996%. s_resolution, abmaglim and
# filter are always null for spectra and are omitted.
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
    """The User-Agent string, carrying the installed albireo version."""
    from albireo import __version__

    return f"albireo/{__version__} (https://github.com/tjayasinghe/albireo)"


# The ESO endpoints refuse or drop connections under load often enough that a bare urlopen
# fails a few times in a 50-file run, so transfers are retried. OSError is excluded from
# this tuple: it would cover local disk errors (a full filesystem on the .part write) and
# retry them five times before reporting them as network trouble.
_TRANSIENT = (urllib.error.URLError, TimeoutError, ConnectionError)


class _TapQueryError(Exception):
    """A 400 from a TAP service, carrying the VOTable body that says why."""

    def __init__(self, body: str):
        super().__init__(body[:200])
        self.body = body


def _with_retries[T](what: str, call: Callable[[], T], attempts: int = 5, quiet: bool = False) -> T:
    """Run ``call``, retrying transient network failures with exponential backoff."""
    delay = 2.0
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except urllib.error.HTTPError as exc:
            # A 4xx other than 429 is not transient; fail immediately.
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

    A typed view over the TAP row: :attr:`row` keeps every column the query selected, so
    a caller can reach columns this class does not name.
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
    legal character in a Windows filename (NTFS reads ``name:stream`` as an alternate data
    stream, and the open fails with ``EINVAL``). Colons become hyphens on every platform
    so that a data directory copied between machines keeps the same names.
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
    """Build the ADQL for a Phase 3 spectrum search; at least one constraint is required.

    Parameters
    ----------
    ra_deg, dec_deg, radius_deg
        Cone search in ICRS degrees. Give both coordinates or neither. Expressed as
        ``INTERSECTS(s_region, CIRCLE(...))``: ``s_region`` is a bare point for 1-D
        products, so the ``CONTAINS(POINT(...), ...)`` form in ESO's own examples returns
        nothing.
    instrument
        ``instrument_name``, uppercase and unhyphenated as the archive stores it:
        ``FEROS``, ``HARPS``, ``UVES``, ``XSHOOTER``, ``GIRAFFE``, ``ESPRESSO``.
    programme
        ``proposal_id``, exact match including the parenthesised run letter, e.g.
        ``073.D-0274(A)``. ESO Large Programmes are split into sub-runs, so a prefix match
        is usually wanted; a value containing ``%`` (``112.25R7%``) is issued as ``LIKE``.
    collection
        ``obs_collection``. Useful when an instrument's products are split between ESO's
        own pipeline (e.g. ``GIRAFFE``) and a community release (``GAIAESO``), which
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
            "a query needs at least one constraint besides dataproduct_type: an "
            "unconstrained search would ask the archive for every spectrum it holds "
            "(2.4 million and counting). Give coordinates, an instrument, or a programme."
        )
    return (
        f"SELECT {', '.join(columns)}\nFROM ivoa.ObsCore\nWHERE "
        + "\n  AND ".join(where)
        + "\nORDER BY t_min"
    )


def _like_prefix(value: str, what: str) -> str:
    """A literal prefix for a ``LIKE`` pattern, refusing characters that are wildcards.

    ESO's ADQL parser rejects ``ESCAPE`` (verified against the live service), so a ``_``
    cannot be neutralized and would match any single character. It is refused; no ESO
    programme id contains one.
    """
    if "_" in value or "%" in value:
        raise ValueError(
            f"{what} may not contain '_' or '%': both are SQL LIKE wildcards, and this "
            f"service does not support escaping them. Got {value!r}."
        )
    return value


def _quote(value: str) -> str:
    """Validate a string for an ADQL literal, refusing one that contains a single quote."""
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
        Row cap. Reaching it raises, because the archive does not signal truncation: the
        sync endpoint's JSON and CSV serializations carry no overflow marker (only VOTable
        does), so a capped result is indistinguishable from a complete one, and a partial
        list of a programme's epochs would yield a wrong orbit.
    timeout
        Socket timeout in seconds. The service's own default execution limit is 60 s.
    url
        TAP sync endpoint.

    Returns
    -------
    list of ArchiveRecord
        In the order the query asked for.
    """
    rows = _tap_rows(adql, url=url, maxrec=maxrec, timeout=timeout)
    return [ArchiveRecord(dp_id=str(row["dp_id"]), row=row) for row in rows]


def _tap_rows(
    adql: str,
    *,
    url: str,
    maxrec: int,
    timeout: float,
    service: str = "ESO",
) -> list[dict]:
    """Run an ADQL query against an IVOA TAP sync endpoint and return rows as dicts.

    Shared by the ESO and VizieR paths: a silent MAXREC truncation, an error document
    served with a 200 status, and a dropped connection are properties of the protocol
    rather than of either archive.
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

    def fetch() -> bytes:
        request = urllib.request.Request(url, data=payload, headers={"User-Agent": _user_agent()})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            # A TAP service reports an ADQL error as a 400 whose body is a VOTable that
            # explains it. Propagating the HTTPError would discard that body, so it is
            # carried in the exception and unpacked below.
            if exc.code != 400:
                raise
            raise _TapQueryError(exc.read().decode("utf-8", "replace")) from exc

    try:
        raw = _with_retries(f"{service} TAP query", fetch).decode("utf-8", "replace")
    except _TapQueryError as exc:
        raise RuntimeError(
            f"the {service} TAP service rejected the query. Its message was:\n"
            f"{_votable_message(exc.body)}\nThe query was:\n{adql}"
        ) from exc
    # VizieR honours FORMAT=json only when the query succeeds: a syntax error comes back as
    # a VOTable document, sometimes with a 200 status. The payload is inspected rather than
    # the status so that the service's message is reported instead of a JSONDecodeError.
    if raw.lstrip().startswith("<"):
        raise RuntimeError(
            f"the {service} TAP service returned an XML document rather than JSON, which is "
            "how it reports an ADQL error. Its message was:\n"
            f"{_votable_message(raw)}\nThe query was:\n{adql}"
        )
    table: object = None
    try:
        table = json.loads(raw)
        columns = [meta["name"] for meta in table["metadata"]]  # type: ignore[index]
        data = table["data"]  # type: ignore[index]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        described = sorted(table) if isinstance(table, dict) else repr(raw[:200])
        raise RuntimeError(
            f"the TAP service returned something that is not an {service} result table "
            f"(keys: {described}). This is usually an ADQL syntax error reported as a "
            f"document rather than an HTTP error. The query was:\n{adql}"
        ) from exc

    if len(data) >= maxrec:
        raise RuntimeError(
            f"the query returned {len(data)} rows, which is the MAXREC cap: the result is "
            f"almost certainly truncated, and {service}'s JSON output carries no overflow "
            "marker to confirm it either way. Raise maxrec or narrow the query. Returning a "
            "silently short list would be worse than this error."
        )
    return [dict(zip(columns, row, strict=True)) for row in data]


def _votable_message(document: str) -> str:
    """The human-readable part of a VOTable error document, or a truncated dump of it."""
    for tag in ("INFO", "DESCRIPTION"):
        start = document.find(f"<{tag}")
        while start != -1:
            open_end = document.find(">", start)
            close = document.find(f"</{tag}>", open_end)
            if open_end != -1 and close != -1:
                text = document[open_end + 1 : close].strip()
                if text:
                    return text
            start = document.find(f"<{tag}", open_end)
    return document[:400]


def _download_one(record: ArchiveRecord, outdir: pathlib.Path, force: bool, timeout: float) -> str:
    """Download one product; return a one-line status string starting FAIL, skip or got."""
    path = outdir / record.filename
    expected = record.size_bytes
    if path.exists() and not force:
        size = path.stat().st_size
        # access_estsize is rounded to whole kB, so a kilobyte of slack is allowed. With no
        # estimate, any existing non-empty file is taken as complete rather than
        # re-downloaded on every run.
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
            # A transfer cut cleanly is still a valid HTTP response; without this check a
            # truncated FITS file would be written as if complete.
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

    Each file is written to a ``.part`` temporary and renamed atomically, so an
    interrupted run leaves no half-written FITS file under the final name; re-running
    skips files already present. Downloads are checked against ``Content-Length`` where
    the server sends it, since a cleanly truncated transfer is otherwise
    indistinguishable from success.

    Parameters
    ----------
    records
        From :func:`query`.
    outdir
        Created if missing.
    force
        Re-download files already present.
    jobs
        Parallel downloads. ESO publishes no rate limit; the default of 4 is conservative
        rather than tuned.
    timeout
        Per-file socket timeout in seconds.
    manifest
        Write ``manifest.json`` recording every product's archive metadata and the
        outcome of its download. It is written after the transfers, so it cannot list a
        failed file as present.
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
        # Completion order, so `progress` reports each file as it lands; `results` is
        # indexed back into input order.
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


# -- BLOeM -------------------------------------------------------------------
#
# The Binarity at LOw Metallicity survey (Shenar et al. 2024, A&A, 690, A289): 929 OBAF
# stars in the Small Magellanic Cloud, about 25 epochs each, with an intrinsic binary
# fraction above 70%. The constants below record how the survey is filed in the archive.

# The reduced spectra sit under obs_collection='GIRAFFE'. There is no BLOeM Phase 3
# collection, and target_name is the Gaia DR3 source id rather than the survey's own name,
# which is what the VizieR cross-match supplies. The programme id is 112.25R7, not the
# 112.25W2 printed in arXiv v1 of 2407.14593; in the archive that id belongs to an ERIS
# programme.
BLOEM_COLLECTION = "GIRAFFE"
BLOEM_PROGRAMME = "112.25R7"

# A second programme observes the same 929 stars and is excluded by default. 115.28A9 uses
# a different setup: 4537-4761 A at R=23000 and 6442-6822 A at R=17000, against the
# survey's LR02 at 3960-4571 A and R=6300. Merging them without distinct instrument keys
# would present the model with three disjoint windows under one LSF.
BLOEM_FOLLOWUP_PROGRAMME = "115.28A9"

# VizieR table names contain slashes and must be double-quoted in ADQL.
_BLOEM_CATALOGUE_TABLE = '"J/A+A/690/A289/tablea2"'  # 929 rows: the identifier cross-match
_BLOEM_BINARY_TABLE = '"J/A+A/698/A41/tabled1"'  # 309 rows: the B-star binary classes


@dataclass(frozen=True)
class BloemTarget:
    """One BLOeM star, resolved from its survey identifier to something the archive knows.

    Attributes
    ----------
    bloem_id : str
        The survey identifier, normalized to ``field-star`` with the star zero-padded to
        three digits (``"1-001"``). The published tables spell it three different ways.
    gaia_dr3 : str
        The Gaia DR3 source id, and the archive's ``target_name`` for this star. Kept as a
        string: the value exceeds 2^53, so any conversion through a float (JavaScript, a
        numpy ``float64`` column, ``pandas.read_json``) loses digits without raising.
    ra_deg, dec_deg : float
        Gaia DR3 position, ICRS degrees at epoch 2016.0.
    spectral_type : str
        As published, whitespace stripped.
    binary_class : str or None
        ``"SB1"``, ``"SB2"``, ``"SB3"``, ``"RVvar"`` or ``"RVcst"`` where the star appears
        in the B-star multiplicity table, else ``None``. ``None`` means unclassified in
        that table, not single: it covers all 620 stars the table does not list, including
        every O star.
    row : dict
        The full catalogue row, so a caller can reach a column this class does not name.

    References
    ----------
    .. [1] Shenar, T. et al. 2024, A&A, 690, A289
    """

    bloem_id: str
    gaia_dr3: str
    ra_deg: float
    dec_deg: float
    spectral_type: str = ""
    binary_class: str | None = None
    row: dict = field(default_factory=dict, repr=False)


def normalize_bloem_id(target: str) -> str:
    """Normalize a BLOeM identifier to the ``field-star`` form the catalogue uses.

    The published tables spell the identifier three ways: the cross-match table writes
    ``1-001``, the multiplicity table writes ``1-002``, and the O-star table writes
    ``BLOeM_1-006``. Those spellings and ``BLOeM 1-1`` all reduce to the same identifier.

    References
    ----------
    .. [1] Shenar, T. et al. 2024, A&A, 690, A289

    Examples
    --------
    >>> normalize_bloem_id("BLOeM_1-6")
    '1-006'
    >>> normalize_bloem_id("8-100")
    '8-100'
    """
    text = str(target).strip()
    lowered = text.lower()
    if lowered.startswith("bloem"):
        text = text[5:].lstrip(" _-")
    if "-" not in text:
        raise ValueError(
            f"{target!r} is not a BLOeM identifier. They look like '1-001': a field number "
            "1-8, a hyphen, and a star number. albireo.archive.bloem_catalogue() lists them."
        )
    field, _, star = text.partition("-")
    try:
        return f"{int(field)}-{int(star):03d}"
    except ValueError as exc:
        raise ValueError(
            f"{target!r} is not a BLOeM identifier: expected 'field-star' with both numeric, "
            f"e.g. '1-001'. ({exc})"
        ) from exc


def _clean(value: object) -> str:
    """A VizieR CHAR cell as a plain string. Every one of them is space-padded."""
    return "" if value is None else str(value).strip()


def _gaia_id(value: object, context: str) -> str:
    """A Gaia source id as an exact decimal string, refusing anything lossy.

    VizieR sends it as a bare JSON integer literal, which Python's ``json`` decodes to an
    arbitrary-precision ``int`` without loss. A ``float`` arriving here means some layer has
    already rounded it: 809 of the 929 BLOeM ids do not survive a float64 round trip, and
    the rounded value remains a plausible-looking source id.
    """
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise RuntimeError(
            f"the Gaia DR3 id for {context} arrived as {type(value).__name__} ({value!r}), "
            "not an integer or string. A Gaia source id exceeds 2^53, so a float has already "
            "lost digits and the id cannot be trusted."
        )
    text = str(value).strip()
    if not text.isdigit():
        raise RuntimeError(f"the Gaia DR3 id for {context} is not a decimal number: {text!r}")
    return text


def _vizier_rows(adql: str, *, maxrec: int = 5000, timeout: float = 180.0) -> list[dict]:
    """Rows from VizieR's TAP service, with the same guards as the ESO path."""
    return _tap_rows(adql, url=VIZIER_SYNC_URL, maxrec=maxrec, timeout=timeout, service="VizieR")


_BINARY_CLASS_CACHE: dict[str, str] = {}


def _binary_classes(timeout: float) -> dict[str, str]:
    """``{bloem_id: class}`` from the B-star multiplicity table, fetched at most once.

    Cached for the process: it is a 309-row published table that does not change, and
    without this every :func:`resolve_bloem` in a loop over targets pays for it again.
    """
    if _BINARY_CLASS_CACHE:
        return _BINARY_CLASS_CACHE
    rows = _vizier_rows(
        f"SELECT BLOeM, Bin FROM {_BLOEM_BINARY_TABLE}", maxrec=5000, timeout=timeout
    )
    out: dict[str, str] = {}
    for row in rows:
        identifier = _clean(row.get("BLOeM"))
        label = _clean(row.get("Bin"))
        if identifier and label:
            out[normalize_bloem_id(identifier)] = label
    _BINARY_CLASS_CACHE.update(out)
    return out


def bloem_catalogue(
    *,
    binary_class: str | None = None,
    with_classification: bool = True,
    timeout: float = 180.0,
) -> list[BloemTarget]:
    """The BLOeM target list, resolved to Gaia DR3 source ids.

    One VizieR query for the 929-row identifier cross-match and, unless
    ``with_classification`` is disabled, a second for the published multiplicity classes,
    joined on the survey identifier.

    Parameters
    ----------
    binary_class
        Keep only stars with this class, e.g. ``"SB2"``. Filtered client-side, because
        VizieR stores the column space-padded and ``WHERE Bin='SB2'`` matches nothing.
        Implies ``with_classification``.
    with_classification
        Fetch and attach :attr:`BloemTarget.binary_class`. One extra query.
    timeout
        Socket timeout in seconds.

    Returns
    -------
    list of BloemTarget
        Ordered by identifier.

    Notes
    -----
    The classification covers only the 309 B-type stars of Villasenor et al. (2025):
    91 SB1, 59 SB2, 3 SB3, 110 RV-variable and 46 RV-constant. The O stars are classified
    in a separate, unharmonized catalogue that this function does not join, so a ``None``
    class means not in that table, never single.

    References
    ----------
    .. [1] Shenar, T. et al. 2024, A&A, 690, A289
    .. [2] Villasenor, J. I. et al. 2025, A&A, 698, A41

    Examples
    --------
    >>> sb2 = bloem_catalogue(binary_class="SB2")  # doctest: +SKIP
    >>> len(sb2)  # doctest: +SKIP
    59
    """
    rows = _vizier_rows(
        "SELECT ID, GaiaDR3, RA_ICRS, DE_ICRS, SpTypenew "
        f"FROM {_BLOEM_CATALOGUE_TABLE} ORDER BY ID",
        maxrec=5000,
        timeout=timeout,
    )
    if not rows:
        raise RuntimeError(
            f"VizieR returned no rows for {_BLOEM_CATALOGUE_TABLE}. The catalogue has 929; "
            "an empty result means the table was renamed or the service is degraded."
        )
    classes = _binary_classes(timeout) if (with_classification or binary_class is not None) else {}

    targets = []
    for row in rows:
        identifier = normalize_bloem_id(_clean(row.get("ID")))
        targets.append(
            BloemTarget(
                bloem_id=identifier,
                gaia_dr3=_gaia_id(row.get("GaiaDR3"), f"BLOeM {identifier}"),
                ra_deg=float(row["RA_ICRS"]),
                dec_deg=float(row["DE_ICRS"]),
                spectral_type=_clean(row.get("SpTypenew")),
                binary_class=classes.get(identifier),
                row=row,
            )
        )
    if binary_class is not None:
        wanted = str(binary_class).strip().lower()
        targets = [t for t in targets if (t.binary_class or "").lower() == wanted]
        if not targets:
            available = sorted({c for c in classes.values()})
            raise ValueError(f"no BLOeM target has binary class {binary_class!r}; have {available}")
    return targets


def resolve_bloem(
    target: str, *, with_classification: bool = True, timeout: float = 180.0
) -> BloemTarget:
    """Resolve one BLOeM identifier to its Gaia DR3 source id and position.

    Parameters
    ----------
    target
        The survey identifier in any of its published spellings: ``"1-001"``,
        ``"BLOeM_1-1"``, ``"BLOeM 1-001"``.
    with_classification
        Also fill :attr:`BloemTarget.binary_class`. This takes one extra query the first
        time in a process; the table is small, published and cached thereafter. Set it to
        ``False`` when only the source id is needed, as :func:`bloem_spectra` does.
    timeout
        Socket timeout in seconds.

    Returns
    -------
    BloemTarget

    Raises
    ------
    ValueError
        If the identifier is malformed, or names no star in the catalogue.

    References
    ----------
    .. [1] Shenar, T. et al. 2024, A&A, 690, A289

    Examples
    --------
    >>> star = resolve_bloem("1-001")  # doctest: +SKIP
    >>> star.gaia_dr3, star.spectral_type  # doctest: +SKIP
    ('4690503998385774848', 'B9 Iab')
    """
    identifier = normalize_bloem_id(target)
    rows = _vizier_rows(
        "SELECT ID, GaiaDR3, RA_ICRS, DE_ICRS, SpTypenew "
        f"FROM {_BLOEM_CATALOGUE_TABLE} WHERE ID = '{_quote(identifier)}'",
        maxrec=16,
        timeout=timeout,
    )
    if not rows:
        raise ValueError(
            f"no BLOeM target is called {identifier!r}. The survey has 929 stars in 8 fields, "
            "numbered like '1-001' to '8-117'; bloem_catalogue() lists them."
        )
    row = rows[0]
    classes = _binary_classes(timeout) if with_classification else {}
    return BloemTarget(
        bloem_id=identifier,
        gaia_dr3=_gaia_id(row.get("GaiaDR3"), f"BLOeM {identifier}"),
        ra_deg=float(row["RA_ICRS"]),
        dec_deg=float(row["DE_ICRS"]),
        spectral_type=_clean(row.get("SpTypenew")),
        binary_class=classes.get(identifier),
        row=row,
    )


def bloem_spectra(
    target: str | BloemTarget,
    *,
    programme: str | None = BLOEM_PROGRAMME,
    public_only: bool = False,
    maxrec: int = 2000,
    timeout: float = 300.0,
) -> list[ArchiveRecord]:
    """Every archived BLOeM spectrum of one star, from its survey identifier.

    Resolves the identifier through VizieR, then queries the ESO archive by
    ``target_name``, which for this survey is the Gaia DR3 source id. The records returned
    are accepted directly by :func:`download`.

    Parameters
    ----------
    target
        A BLOeM identifier, or an already-resolved :class:`BloemTarget` (which saves the
        VizieR round trip when looping over many stars).
    programme
        ``proposal_id`` prefix. Defaults to the survey programme ``112.25R7``, whose four
        sub-runs are the LR02 epochs the BLOeM papers analyse. ``None`` returns every
        GIRAFFE spectrum of the star regardless of programme, which currently includes the
        ``115.28A9`` follow-up: a different setup at R = 17000 and 23000 in two other
        wavelength windows. Those are usable only as separate instruments with their own
        line-spread functions, never pooled with LR02.
    public_only
        Drop rows still inside their proprietary period. Sub-run ``.004`` releases through
        2027-01-15, so a fetch before then is partial by construction; without this flag
        the proprietary rows are returned and will fail to download.
    maxrec
        Row cap. One star has ~25-32 epochs; the default leaves room without disabling the
        truncation guard.
    timeout
        Socket timeout in seconds.

    Returns
    -------
    list of ArchiveRecord
        Ordered by observation time.

    Examples
    --------
    >>> records = bloem_spectra("1-002")  # doctest: +SKIP
    >>> download(records, "data/bloem-1-002")  # doctest: +SKIP

    Notes
    -----
    BLOeM spectra are neither continuum-normalized nor flux-calibrated (``CONTNORM=F``,
    ``FLUXCAL='UNCALIBRATED'``), are on an air wavelength scale in nanometres, and are
    heliocentric. :func:`albireo.io.read_dataset` handles all four. At about 178 kB per
    spectrum, one star amounts to roughly 5 MB.

    References
    ----------
    .. [1] Shenar, T. et al. 2024, A&A, 690, A289
    """
    star = (
        target
        if isinstance(target, BloemTarget)
        else resolve_bloem(target, with_classification=False, timeout=timeout)
    )
    where = [
        "dataproduct_type='spectrum'",
        f"obs_collection='{_quote(BLOEM_COLLECTION)}'",
        f"target_name='{_quote(star.gaia_dr3)}'",
    ]
    if programme is not None:
        prefix = _like_prefix(programme.rstrip("%"), "a programme prefix")
        where.append(f"proposal_id LIKE '{_quote(prefix)}%'")
    adql = (
        f"SELECT {', '.join(DEFAULT_COLUMNS)}\nFROM ivoa.ObsCore\nWHERE "
        + "\n  AND ".join(where)
        + "\nORDER BY t_min"
    )
    records = query(adql, maxrec=maxrec, timeout=timeout)
    if public_only:
        records = [r for r in records if not _is_proprietary(r)]
    return records


def _is_proprietary(record: ArchiveRecord) -> bool:
    """Whether a record is still inside its proprietary period, by its own release date."""
    released = record.row.get("obs_release_date")
    if not released:
        return False
    try:
        when = datetime.datetime.fromisoformat(str(released).replace("Z", "+00:00"))
    except ValueError:
        return False
    if when.tzinfo is None:
        when = when.replace(tzinfo=datetime.UTC)
    return when > datetime.datetime.now(datetime.UTC)

"""The ESO Science Archive client (D44).

The tests are in two layers. Everything that can be checked offline is checked offline
(query construction, the truncation guard, filename safety, resume and atomicity), using
a fake TAP response and a local HTTP handler. The few assertions that need the archive
are marked ``network``.

The two most important behaviours concern silent failures: a TAP query that hits MAXREC
returns a short list with no overflow marker, and a download cut cleanly in the middle
returns a valid HTTP response. Both would give a wrong result downstream with no error,
so both raise.
"""

import http.client
import io
import json
import ssl
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

import albireo as ab
from albireo import archive


def _tap_json(rows, columns=("dp_id", "target_name", "access_estsize")):
    return json.dumps(
        {"metadata": [{"name": name} for name in columns], "data": [list(r) for r in rows]}
    ).encode()


class _FakeResponse(io.BytesIO):
    """Minimal stand-in for urlopen's context manager."""

    def __init__(self, payload: bytes, headers: dict | None = None):
        super().__init__(payload)
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# -- query construction ------------------------------------------------------


def test_spectra_query_uses_the_form_that_works_for_spectra():
    """s_region is a bare point for 1-D products, so CONTAINS(POINT(...)) matches nothing."""
    adql = archive.spectra_query(ra_deg=274.28139, dec_deg=-56.02337, radius_deg=0.05)
    assert "INTERSECTS(s_region, CIRCLE('ICRS'" in adql
    assert "CONTAINS(POINT(" not in adql
    assert "dataproduct_type='spectrum'" in adql
    assert adql.strip().endswith("ORDER BY t_min")


def test_spectra_query_composes_every_constraint():
    adql = archive.spectra_query(
        ra_deg=1.0,
        dec_deg=2.0,
        instrument="FEROS",
        programme="073.D-0274(A)",
        collection="GIRAFFE",
        calib_level=2,
    )
    assert "instrument_name='FEROS'" in adql
    assert "proposal_id = '073.D-0274(A)'" in adql
    assert "obs_collection='GIRAFFE'" in adql
    assert "calib_level=2" in adql


def test_a_wildcard_programme_becomes_LIKE():
    """ESO Large Programmes are split into sub-runs, so a prefix match is needed."""
    adql = archive.spectra_query(programme="112.25R7%")
    assert "proposal_id LIKE '112.25R7%'" in adql
    assert archive.spectra_query(programme="112.25R7.001").count("LIKE") == 0


def test_query_refuses_to_ask_for_the_whole_archive():
    with pytest.raises(ValueError, match="at least one constraint"):
        archive.spectra_query()


def test_query_requires_both_coordinates_or_neither():
    with pytest.raises(ValueError, match="both ra_deg and dec_deg"):
        archive.spectra_query(ra_deg=1.0)
    with pytest.raises(ValueError, match="both ra_deg and dec_deg"):
        archive.spectra_query(dec_deg=1.0)
    with pytest.raises(ValueError, match="radius_deg must be positive"):
        archive.spectra_query(ra_deg=1.0, dec_deg=2.0, radius_deg=0.0)


def test_adql_string_literals_reject_embedded_quotes():
    """This is not a security boundary; it makes a mangled target name raise an error."""
    with pytest.raises(ValueError, match="may not contain a single quote"):
        archive.spectra_query(instrument="FER'OS")


def test_numpy_coordinates_are_written_as_numbers():
    """Under NumPy 2 the repr of np.float64(1.5) is 'np.float64(1.5)', which is not ADQL.

    Coordinates from astropy (``SkyCoord.ra.deg``) or from a table are NumPy scalars, so
    every such cone search was rejected by the service. Python floats keep their exact form.
    """
    expected = "INTERSECTS(s_region, CIRCLE('ICRS',274.28139,-56.02337,0.05))=1"
    from_floats = archive.spectra_query(ra_deg=274.28139, dec_deg=-56.02337, radius_deg=0.05)
    from_numpy = archive.spectra_query(
        ra_deg=np.float64(274.28139), dec_deg=np.float64(-56.02337), radius_deg=np.float64(0.05)
    )
    assert expected in from_floats
    assert from_numpy == from_floats
    assert "np." not in from_numpy


@pytest.mark.parametrize(
    ("ra", "dec", "radius", "message"),
    [
        (float("nan"), 2.0, 0.05, "must be finite"),
        (1.0, float("inf"), 0.05, "must be finite"),
        (1.0, 95.0, 0.05, r"\[-90, 90\]"),
        (1.0, 2.0, float("inf"), "radius_deg must be positive"),
    ],
)
def test_a_cone_outside_the_sky_is_refused(ra, dec, radius, message):
    with pytest.raises(ValueError, match=message):
        archive.spectra_query(ra_deg=ra, dec_deg=dec, radius_deg=radius)


# -- the truncation guard ----------------------------------------------------


def test_query_raises_when_the_result_hits_maxrec(monkeypatch):
    """ESO's JSON has no overflow marker, so a result that reaches MAXREC must raise."""
    rows = [(f"ADP.{i}", "TGT", 3000) for i in range(5)]
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _FakeResponse(_tap_json(rows)))
    with pytest.raises(RuntimeError, match="MAXREC cap"):
        archive.query("SELECT 1", maxrec=5)

    # One row short of the cap is a complete result and must be returned normally.
    got = archive.query("SELECT 1", maxrec=6)
    assert [r.dp_id for r in got] == [f"ADP.{i}" for i in range(5)]


def test_query_reports_a_non_table_response_with_the_query(monkeypatch):
    """ESO returns ADQL syntax errors as a document, not an HTTP error."""
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **k: _FakeResponse(json.dumps({"error": "syntax"}).encode()),
    )
    with pytest.raises(RuntimeError, match="not an ESO result table") as exc:
        archive.query("SELECT nonsense FROM nowhere")
    assert "SELECT nonsense FROM nowhere" in str(exc.value)


def test_the_error_message_is_read_from_the_query_status():
    """A self-closing INFO before the error must not put the next element's markup in it."""
    document = (
        '<VOTABLE><RESOURCE type="results"><INFO name="provider" value="ESO"/>\n'
        '<INFO name="QUERY_STATUS" value="ERROR">Column foo unknown</INFO>'
        "</RESOURCE></VOTABLE>"
    )
    assert archive._votable_message(document) == "Column foo unknown"
    # Without a QUERY_STATUS element, the first INFO or DESCRIPTION with text is used.
    other = "<VOTABLE><DESCRIPTION>service is down</DESCRIPTION></VOTABLE>"
    assert archive._votable_message(other) == "service is down"
    assert archive._votable_message("<html>plain</html>") == "<html>plain</html>"


def test_query_returns_typed_records(monkeypatch):
    rows = [("ADP.2016-09-20T09:32:35.364", "HR-6819", 3000)]
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _FakeResponse(_tap_json(rows)))
    (record,) = archive.query("SELECT 1")
    assert isinstance(record, ab.ArchiveRecord)
    assert record.target_name == "HR-6819"
    assert record.size_bytes == 3000 * 1024
    assert record.filename == "ADP.2016-09-20T09-32-35.364.fits"
    assert record.row["access_estsize"] == 3000


# -- filenames ---------------------------------------------------------------


def test_local_filename_is_safe_on_windows():
    """NTFS interprets a colon in the name as an alternate data stream, and the open fails."""
    name = archive.local_filename("ADP.2016-09-20T09:32:35.364")
    assert ":" not in name
    assert name == "ADP.2016-09-20T09-32-35.364.fits"


def test_records_without_a_size_estimate_are_handled():
    record = ab.ArchiveRecord(dp_id="ADP.x", row={"dp_id": "ADP.x", "access_estsize": None})
    assert record.size_bytes is None
    assert record.instrument == ""


# -- downloading -------------------------------------------------------------


def _record(dp_id="ADP.test", size_kb=2):
    return ab.ArchiveRecord(dp_id=dp_id, row={"dp_id": dp_id, "access_estsize": size_kb})


def test_download_writes_atomically_and_records_a_manifest(tmp_path, monkeypatch):
    payload = b"x" * 2048
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **k: _FakeResponse(payload, {"Content-Length": str(len(payload))}),
    )
    records = [_record("ADP.a"), _record("ADP.b")]
    statuses = archive.download(records, tmp_path, jobs=1)

    assert all(s.startswith("got") for s in statuses), statuses
    assert (tmp_path / "ADP.a.fits").read_bytes() == payload
    assert not list(tmp_path.glob("*.part")), "no temporary files may survive a clean run"

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_records"] == 2
    assert {r["local_file"] for r in manifest["records"]} == {"ADP.a.fits", "ADP.b.fits"}
    assert all(r["status"] == "got" for r in manifest["records"])


def test_download_skips_what_is_already_there(tmp_path, monkeypatch):
    payload = b"x" * 2048
    calls = []

    def fake(*a, **k):
        calls.append(1)
        return _FakeResponse(payload, {"Content-Length": str(len(payload))})

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    records = [_record("ADP.a")]
    assert archive.download(records, tmp_path, jobs=1)[0].startswith("got")
    assert len(calls) == 1
    assert archive.download(records, tmp_path, jobs=1)[0].startswith("skip")
    assert len(calls) == 1, "a present, complete file must not be fetched again"
    assert archive.download(records, tmp_path, jobs=1, force=True)[0].startswith("got")
    assert len(calls) == 2


def test_a_truncated_transfer_is_not_accepted(tmp_path, monkeypatch):
    """A transfer cut cleanly in the middle is still a valid HTTP response."""
    # The truncation guard raises ConnectionError, which is in `_TRANSIENT` because a real
    # truncated transfer is usually worth retrying. The test therefore goes through the
    # full backoff sequence (2 + 4 + 8 + 16 s) before the asserted failure is reported.
    # Stubbing the sleep keeps the retry logic under test and saves 30 s of wall-clock
    # time in every run of the suite, locally and in CI.
    monkeypatch.setattr(archive.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **k: _FakeResponse(b"x" * 1500, {"Content-Length": "999999"}),
    )
    status = archive.download([_record("ADP.a")], tmp_path, jobs=1)[0]
    assert status.startswith("FAIL")
    assert "truncated" in status
    assert not (tmp_path / "ADP.a.fits").exists()
    assert not list(tmp_path.glob("*.part")), "the partial file must be cleaned up"


def test_a_suspiciously_small_response_is_not_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: _FakeResponse(b"<html>error</html>")
    )
    status = archive.download([_record("ADP.a")], tmp_path, jobs=1)[0]
    assert status.startswith("FAIL")
    assert "suspiciously small" in status
    assert not (tmp_path / "ADP.a.fits").exists()


def test_a_client_error_fails_fast_without_retrying(tmp_path, monkeypatch):
    calls = []

    def fake(*a, **k):
        calls.append(1)
        raise urllib.error.HTTPError("u", 404, "Not Found", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    status = archive.download([_record("ADP.a")], tmp_path, jobs=1)[0]
    assert status.startswith("FAIL")
    assert len(calls) == 1, "a 404 will not fix itself; retrying it wastes 30 s"


def test_progress_is_reported_per_file(tmp_path, monkeypatch):
    payload = b"x" * 2048
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **k: _FakeResponse(payload, {"Content-Length": str(len(payload))}),
    )
    seen = []
    archive.download(
        [_record(f"ADP.{i}") for i in range(3)],
        tmp_path,
        jobs=1,
        progress=lambda done, total, line: seen.append((done, total)),
    )
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_download_of_nothing_is_not_an_error(tmp_path):
    assert archive.download([], tmp_path) == []


class _BrokenResponse(_FakeResponse):
    """A response whose body cannot be read: the connection fails after the headers."""

    def __init__(self, error: Exception):
        super().__init__(b"", {"Content-Length": "4096"})
        self.error = error

    def read(self, *args):
        raise self.error


def test_a_dropped_tls_connection_fails_its_record_not_the_batch(tmp_path, monkeypatch):
    """ssl.SSLError is an OSError, not a URLError, and escaped the per-record handling.

    The whole call then raised: no status lines, no manifest, and every .part file left.
    """
    monkeypatch.setattr(archive.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **k: _BrokenResponse(ssl.SSLEOFError(8, "EOF occurred in violation")),
    )
    statuses = archive.download([_record("ADP.a"), _record("ADP.b")], tmp_path, jobs=1)

    assert [s.split()[0] for s in statuses] == ["FAIL", "FAIL"]
    assert "5 attempts" in statuses[0], "a TLS error is transient and is retried"
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert [r["status"] for r in manifest["records"]] == ["FAIL", "FAIL"]
    assert not list(tmp_path.glob("*.part"))


def test_a_cut_chunked_body_is_retried(tmp_path, monkeypatch):
    """A chunked body cut short raises http.client.IncompleteRead, which is transient."""
    monkeypatch.setattr(archive.time, "sleep", lambda _seconds: None)
    payload = b"SIMPLE  =" + b" " * 2039
    calls = []

    def fake(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            return _BrokenResponse(http.client.IncompleteRead(b"SIMPLE"))
        return _FakeResponse(payload, {"Content-Length": str(len(payload))})

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    (status,) = archive.download([_record("ADP.a")], tmp_path, jobs=1)
    assert status.startswith("got"), status
    assert len(calls) == 2
    assert (tmp_path / "ADP.a.fits").read_bytes() == payload


def test_a_record_listed_twice_is_downloaded_once(tmp_path, monkeypatch):
    """Two threads writing one product raced on its .part file and aborted the batch."""
    payload = b"x" * 2048
    calls = []

    def fake(*a, **k):
        calls.append(1)
        return _FakeResponse(payload, {"Content-Length": str(len(payload))})

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    seen = []
    records = [_record("ADP.d"), _record("ADP.e"), _record("ADP.d")]
    statuses = archive.download(
        records, tmp_path, jobs=2, progress=lambda done, total, line: seen.append(done)
    )

    assert len(calls) == 2
    assert [s.split()[1] for s in statuses] == ["ADP.d", "ADP.e", "ADP.d"]
    assert statuses[0] == statuses[2] and statuses[0].startswith("got")
    assert sorted(seen) == [1, 2, 3], "progress still counts every record"
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_records"] == 3


def test_an_error_page_is_not_saved_as_a_product(tmp_path, monkeypatch):
    """A login or error page served with status 200 can exceed the 1 kB size threshold."""
    page = b"\n<!DOCTYPE html><html><body>" + b"Please log in. " * 200 + b"</body></html>"
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **k: _FakeResponse(page, {"Content-Length": str(len(page))}),
    )
    (status,) = archive.download([_record("ADP.a")], tmp_path, jobs=1)
    assert status.startswith("FAIL") and "not a data product" in status, status
    assert not (tmp_path / "ADP.a.fits").exists()
    assert not list(tmp_path.glob("*.part"))


def test_an_interrupt_cancels_the_queued_files(tmp_path, monkeypatch):
    """The executor's exit waited for every queued file before an interrupt took effect."""
    payload = b"x" * 2048
    calls = []
    gate = threading.Event()

    def fake(*a, **k):
        calls.append(1)
        if len(calls) > 1:
            gate.wait(10.0)
        return _FakeResponse(payload, {"Content-Length": str(len(payload))})

    class Stop(Exception):
        pass

    def progress(done, total, line):
        raise Stop

    class GatedPool(archive.ThreadPoolExecutor):
        """An executor that opens the gate at shutdown.

        The worker may have taken the second file, which waits on the gate. The gate opens
        after a shutdown without waiting, which cancels the queued files, and before a
        waiting shutdown, the exit of the ``with`` block. The test therefore does not
        depend on timing.
        """

        def shutdown(self, wait=True, *, cancel_futures=False):
            if wait:
                gate.set()
            super().shutdown(wait=wait, cancel_futures=cancel_futures)
            gate.set()

    monkeypatch.setattr(archive, "ThreadPoolExecutor", GatedPool)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    records = [_record(f"ADP.{i}") for i in range(10)]
    with pytest.raises(Stop):
        archive.download(records, tmp_path, jobs=1, progress=progress)
    # The first file, and at most the one the worker had already taken.
    assert len(calls) <= 2
    assert not (tmp_path / "manifest.json").exists()


# -- BLOeM, offline ----------------------------------------------------------


def test_bloem_spectra_queries_the_survey_collection_by_gaia_id(monkeypatch):
    seen = []
    monkeypatch.setattr(archive, "query", lambda adql, **kwargs: seen.append(adql) or [])
    star = ab.BloemTarget(bloem_id="1-002", gaia_dr3="4690503998385774848", ra_deg=1, dec_deg=2)
    assert archive.bloem_spectra(star) == []
    assert seen == [
        f"SELECT {', '.join(archive.DEFAULT_COLUMNS)}\nFROM ivoa.ObsCore\nWHERE "
        "dataproduct_type='spectrum'\n  AND obs_collection='GIRAFFE'\n"
        "  AND target_name='4690503998385774848'\n  AND proposal_id LIKE '112.25R7%'\n"
        "ORDER BY t_min"
    ]


def _fake_vizier(classification_rows):
    """A VizieR stand-in: two catalogue rows, and the given classification rows."""
    issued = []

    def rows(adql, **kwargs):
        issued.append(adql)
        if archive._BLOEM_BINARY_TABLE in adql:
            return list(classification_rows)
        return [
            {"ID": "1-001 ", "GaiaDR3": 1, "RA_ICRS": 1.0, "DE_ICRS": -72.0, "SpTypenew": "B1"},
            {"ID": "1-002 ", "GaiaDR3": 2, "RA_ICRS": 1.1, "DE_ICRS": -72.1, "SpTypenew": "B2"},
        ]

    return rows, issued


def test_an_empty_classification_table_is_reported_and_not_cached(monkeypatch):
    """An empty table blamed the caller's class ("have []") and was fetched on every call."""
    monkeypatch.setattr(archive, "_BINARY_CLASS_CACHE", None)
    rows, issued = _fake_vizier([])
    monkeypatch.setattr(archive, "_vizier_rows", rows)
    for _ in range(2):
        with pytest.raises(RuntimeError, match=r"returned no rows.*with_classification=False"):
            archive.bloem_catalogue(binary_class="SB2")
    assert sum(archive._BLOEM_BINARY_TABLE in q for q in issued) == 2
    # The workaround the message names does not query the class table.
    star = archive.resolve_bloem("1-001", with_classification=False)
    assert (star.gaia_dr3, star.binary_class) == ("1", None)
    assert sum(archive._BLOEM_BINARY_TABLE in q for q in issued) == 2


def test_the_classification_table_is_fetched_once(monkeypatch):
    monkeypatch.setattr(archive, "_BINARY_CLASS_CACHE", None)
    rows, issued = _fake_vizier([{"BLOeM": "1-2", "Bin": "SB2 "}])
    monkeypatch.setattr(archive, "_vizier_rows", rows)
    (sb2,) = archive.bloem_catalogue(binary_class="SB2")
    assert (sb2.bloem_id, sb2.gaia_dr3, sb2.binary_class) == ("1-002", "2", "SB2")
    assert archive.bloem_catalogue()[0].binary_class is None
    assert sum(archive._BLOEM_BINARY_TABLE in q for q in issued) == 1


# -- the live archive --------------------------------------------------------


@pytest.mark.network
def test_the_hr6819_programme_is_still_where_the_query_says_it_is():
    """The query form is checked against the real service.

    HR 6819's FEROS programme has been public since 2005, so this is about as stable as
    an archive query can be. The test is marked ``network`` and deselected by
    ``--no-network``.
    """
    adql = archive.spectra_query(
        ra_deg=274.28139,
        dec_deg=-56.02337,
        radius_deg=0.05,
        instrument="FEROS",
        programme="073.D-0274(A)",
    )
    records = archive.query(adql, maxrec=500)
    assert len(records) >= 50, f"expected ~51 FEROS spectra, got {len(records)}"
    assert all(r.instrument == "FEROS" for r in records)
    assert all(r.dp_id.startswith("ADP.") for r in records)
    # Sorted by observation time, as the query specifies.
    times = [r.row["t_min"] for r in records]
    assert times == sorted(times)


@pytest.mark.network
def test_the_cone_search_form_is_the_one_that_matches():
    """CONTAINS(POINT(...), s_region) is ESO's documented form and returns nothing here."""
    common = "FROM ivoa.ObsCore WHERE dataproduct_type='spectrum' AND instrument_name='FEROS'"
    circle = "CIRCLE('ICRS',274.28139,-56.02337,0.05)"
    working = archive.query(
        f"SELECT dp_id {common} AND INTERSECTS(s_region, {circle})=1", maxrec=500
    )
    documented = archive.query(
        f"SELECT dp_id {common} AND CONTAINS(POINT('ICRS',274.28139,-56.02337), s_region)=1",
        maxrec=500,
    )
    assert len(working) > 0
    assert len(documented) == 0, (
        "if this ever starts matching, ESO has changed s_region for spectra and the "
        "module docstring's advice needs revisiting"
    )


def test_a_temporary_file_left_by_a_stopped_write_is_removed(tmp_path):
    """Each write has a temporary name of its own, so a write that was stopped leaves a
    file that no later write reuses. Such a file of the same target, not modified for a
    day, is removed when the next write of that target starts; a recent one, which may
    belong to a write in progress, and those of other targets are kept."""
    import os
    import time

    target = tmp_path / "library.npz"
    old = tmp_path / "library.npz.123-0123456789ab.part"
    recent = tmp_path / "library.npz.456-ba9876543210.part"
    other = tmp_path / "other.npz.123-0123456789ab.part"
    for path in (old, recent, other):
        path.write_bytes(b"partial")
    two_days_ago = time.time() - 2 * 86400.0
    os.utime(old, (two_days_ago, two_days_ago))
    os.utime(other, (two_days_ago, two_days_ago))

    part = archive._part_path(target)

    assert not old.exists()
    assert recent.exists() and other.exists()
    assert part.parent == tmp_path and part.name.startswith("library.npz.")
    assert part.name.endswith(".part") and not part.exists()

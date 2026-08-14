"""The ESO Science Archive client (D44).

Two layers. Everything that can be checked without the internet is checked without it —
query construction, the truncation guard, filename safety, resume and atomicity — using a
fake TAP response and a local HTTP handler. The handful of assertions that genuinely need
the archive are marked ``network`` and are the first tests in the suite to use that marker.

The two behaviours worth the most here are both about *silence*: a TAP query that hits
MAXREC returns a short list with no marker saying so, and a download cut cleanly in the
middle returns a valid HTTP response. Both would produce a confident wrong answer
downstream, so both raise.
"""

import io
import json
import urllib.error
import urllib.request

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
    """ESO Large Programmes split into sub-runs, so a prefix match is what is wanted."""
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
    """Not a security boundary so much as a way to fail loudly on a mangled target name."""
    with pytest.raises(ValueError, match="may not contain a single quote"):
        archive.spectra_query(instrument="FER'OS")


# -- the truncation guard ----------------------------------------------------


def test_query_raises_when_the_result_hits_maxrec(monkeypatch):
    """The failure this guard exists for: ESO's JSON carries no overflow marker."""
    rows = [(f"ADP.{i}", "TGT", 3000) for i in range(5)]
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _FakeResponse(_tap_json(rows)))
    with pytest.raises(RuntimeError, match="MAXREC cap"):
        archive.query("SELECT 1", maxrec=5)

    # One row short of the cap is a complete result and must come back normally.
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
    """A colon makes NTFS read the name as an alternate data stream and the open fails."""
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


# -- the live archive --------------------------------------------------------


@pytest.mark.network
def test_the_hr6819_programme_is_still_where_the_query_says_it_is():
    """The one assertion that proves the query shape against the real service.

    HR 6819's FEROS programme is public and has been since 2005, so this is about as
    stable as an archive query gets. It is marked ``network`` and deselected by
    ``--no-network``; it is also the reason that marker exists.
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
    # Sorted by observation time, as the query asks.
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

"""``albireo fetch`` and the BLOeM multiplicity table.

The binary class is only printed. It is read from the multiplicity table apart from the
identifier query, once for the command. An empty or renamed class table, or a failed query
for it, leaves the stars unclassified and their spectra are fetched. A failure to resolve
an identifier fails that target alone, with no query of the class table on its account.
The archive is stubbed, because the real one requires the network.
"""

from __future__ import annotations

import pytest

from albireo import archive, cli

TARGET = archive.BloemTarget(
    bloem_id="1-037",
    gaia_dr3="4685929099908473856",
    ra_deg=12.0,
    dec_deg=-73.0,
    spectral_type="B0 V",
    binary_class=None,
)


@pytest.fixture
def stubbed(monkeypatch):
    """The archive with every query recorded; ``classes`` sets the class-table outcome."""
    calls: dict[str, list] = {"resolve": [], "classes": []}
    state: dict[str, object] = {"classes": {"1-037": "SB2"}, "unresolved": set()}

    def resolve(name, *, with_classification=True, timeout=180.0):
        calls["resolve"].append((name, with_classification))
        if name in state["unresolved"]:
            raise RuntimeError("VizieR TAP query failed after 5 attempts: HTTP 503")
        return TARGET

    def binary_classes(timeout):
        calls["classes"].append(timeout)
        if isinstance(state["classes"], Exception):
            raise state["classes"]
        return state["classes"]

    monkeypatch.setattr(archive, "resolve_bloem", resolve)
    monkeypatch.setattr(archive, "_binary_classes", binary_classes)
    monkeypatch.setattr(archive, "bloem_spectra", lambda star, **kw: ["r1", "r2"])
    monkeypatch.setattr(archive, "download", lambda records, outdir, **kw: ["OK", "OK"])
    return calls, state


def test_the_class_is_read_apart_from_the_identifier_query(tmp_path, stubbed, capsys):
    calls, _ = stubbed
    assert cli.main(["fetch", "1-037", "1-37", "--out", str(tmp_path)]) == 0
    assert calls["resolve"] == [("1-037", False), ("1-37", False)]
    assert len(calls["classes"]) == 1, "the class table is read once for the command"
    captured = capsys.readouterr()
    assert captured.out.count("B0 V, SB2): 2 of 2 epochs") == 2
    assert "binary class not available" not in captured.err


def test_a_star_whose_class_table_is_unreadable_is_fetched_unclassified(tmp_path, stubbed, capsys):
    calls, state = stubbed
    state["classes"] = RuntimeError("VizieR returned no rows for J/A+A/698/A41/tablea1")
    assert cli.main(["fetch", "1-037", "--out", str(tmp_path)]) == 0
    assert calls["resolve"] == [("1-037", False)]
    captured = capsys.readouterr()
    assert "B0 V, unclassified): 2 of 2 epochs" in captured.out
    assert "binary class not available" in captured.err and "no rows" in captured.err


def test_a_target_that_cannot_be_resolved_fails_alone(tmp_path, stubbed, capsys):
    """The identifier query is not repeated, and the class note does not name its error."""
    calls, state = stubbed
    state["unresolved"] = {"1-999"}
    assert cli.main(["fetch", "1-999", "1-037", "--out", str(tmp_path)]) == 1
    assert calls["resolve"] == [("1-999", False), ("1-037", False)]
    captured = capsys.readouterr()
    assert "1-999: RuntimeError: VizieR TAP query failed" in captured.err
    assert "binary class not available" not in captured.err
    assert "B0 V, SB2): 2 of 2 epochs" in captured.out


def test_a_command_whose_targets_all_fail_does_not_read_the_class_table(tmp_path, stubbed, capsys):
    calls, state = stubbed
    state["unresolved"] = {"1-999"}
    assert cli.main(["fetch", "1-999", "--out", str(tmp_path)]) == 1
    assert calls["classes"] == []
    assert "[[stars]]" not in capsys.readouterr().out

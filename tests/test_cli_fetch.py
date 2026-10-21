"""``albireo fetch`` when the BLOeM multiplicity table cannot be read.

The binary class is only printed. An empty or renamed class table, or a failed query for
it, leaves the star unclassified and its spectra are fetched; a failure to resolve the
identifier still fails the target. The archive is stubbed, because the real one requires
the network.
"""

from __future__ import annotations

from albireo import archive, cli

TARGET = archive.BloemTarget(
    bloem_id="1-037",
    gaia_dr3="4685929099908473856",
    ra_deg=12.0,
    dec_deg=-73.0,
    spectral_type="B0 V",
    binary_class=None,
)


def _stub(monkeypatch, resolve):
    monkeypatch.setattr(archive, "resolve_bloem", resolve)
    monkeypatch.setattr(archive, "bloem_spectra", lambda star, **kw: ["r1", "r2"])
    monkeypatch.setattr(archive, "download", lambda records, outdir, **kw: ["OK", "OK"])


def test_a_star_whose_class_table_is_unreadable_is_fetched_unclassified(
    tmp_path, monkeypatch, capsys
):
    calls = []

    def resolve(name, *, with_classification=True, timeout=180.0):
        calls.append(with_classification)
        if with_classification:
            raise RuntimeError("VizieR returned no rows for J/A+A/698/A41/tablea1")
        return TARGET

    _stub(monkeypatch, resolve)
    assert cli.main(["fetch", "1-037", "--out", str(tmp_path)]) == 0
    assert calls == [True, False]
    captured = capsys.readouterr()
    assert "B0 V, unclassified): 2 of 2 epochs" in captured.out
    assert "binary class not available" in captured.err and "no rows" in captured.err


def test_a_target_that_cannot_be_resolved_still_fails(tmp_path, monkeypatch, capsys):
    def resolve(name, *, with_classification=True, timeout=180.0):
        raise RuntimeError("the VizieR TAP service returned HTTP 503")

    _stub(monkeypatch, resolve)
    assert cli.main(["fetch", "1-037", "--out", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert "1-037: RuntimeError: the VizieR TAP service returned HTTP 503" in captured.err
    assert "binary class not available" not in captured.err
    assert "[[stars]]" not in captured.out

"""The report of the eclipsing-binary survey experiment is what its builder wrote.

``scripts/rvs_eb_report.py`` computes every number of
``docs/reports/gaia-rvs-eclipsing-binaries.md`` from the tables of the runs, which are not
kept in the repository. What is kept is the page, the numbers it quotes
(``numbers.json``), the binned table behind every figure and the figures. These tests hold
the kept parts together: the page is the template rendered with the numbers, the numbers
agree with the binned tables, and every file the page refers to exists. The statistics of
the builder are tested on small inputs.
"""

from __future__ import annotations

import csv
import importlib.util
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
REPORT = REPO / "docs" / "reports" / "gaia-rvs-eclipsing-binaries"
PAGE = REPO / "docs" / "reports" / "gaia-rvs-eclipsing-binaries.md"


@pytest.fixture(scope="module")
def builder():
    path = REPO / "scripts" / "rvs_eb_report.py"
    spec = importlib.util.spec_from_file_location("rvs_eb_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def numbers():
    return json.loads((REPORT / "numbers.json").read_text(encoding="utf-8"))


def _table(name: str) -> list[dict[str, str]]:
    with open(REPORT / "tables" / f"{name}.csv", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_the_page_is_the_template_rendered_with_the_numbers(builder, numbers):
    page = PAGE.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert builder.render(numbers) == page
    # Nothing is left to fill, and a number that is missing is an error and not a blank.
    assert "[[" not in page
    with pytest.raises(KeyError):
        builder.render({key: value for key, value in numbers.items() if key != "n_design"})


def test_every_file_the_page_refers_to_exists(builder):
    page = PAGE.read_text(encoding="utf-8")
    targets = re.findall(r"\]\((gaia-rvs-eclipsing-binaries/[^)#]+)", page)
    assert len(targets) >= 3 * len(builder.FIGURES)
    for target in targets:
        assert (PAGE.parent / target).exists(), target
    for name, _ in builder.FIGURES:
        for theme in builder.THEMES:
            assert (REPORT / "figures" / f"{name}-{theme}.png").stat().st_size > 10_000
    assert (REPORT / "tables" / "population.csv.gz").exists()
    for run in ("design", "null", "third-light", "activity", "instrument", "recorded"):
        manifest = json.loads((REPORT / "manifests" / f"{run}.json").read_text(encoding="utf-8"))
        assert manifest["n_systems"] > 0 and len(manifest["content_sha256"]) == 2


def test_the_recovery_numbers_are_those_of_the_table(numbers):
    """A share over a range of magnitudes is the cells' counts weighted to the catalogue."""
    rows = _table("fig03-recovery")
    for declaration in ("injected", "classified", "catalogue"):
        for lo, hi, tag in ((10.0, 11.0, "10_11"), (12.0, 13.0, "12_13"), (6.0, 9.0, "6_9")):
            keep = [
                row
                for row in rows
                if row["declaration"] == declaration
                and row["systems"] == "all"
                and lo <= float(row["grvs_low"]) < hi
            ]
            weight = [float(row["catalogue_weight_per_system"]) for row in keep]
            n = sum(w * int(row["n"]) for w, row in zip(weight, keep, strict=True))
            k = sum(w * int(row["both_within_10pct"]) for w, row in zip(weight, keep, strict=True))
            assert numbers[f"rec10_{declaration}_{tag}"] == f"{100 * k / n:.0f}"
            # More systems are within 10 percent than within 5, in every cell.
            assert all(
                int(row["both_within_10pct"]) >= int(row["both_within_5pct"]) for row in keep
            )
            assert all(int(row["n"]) >= int(row["both_within_10pct"]) for row in keep)


def test_the_yield_numbers_are_those_of_the_table(numbers):
    rows = {float(row["grvs_limit"]): row for row in _table("fig15-yield")}
    for limit, tag in ((12.0, "12"), (13.5, "13p5")):
        for key in ("orbit", "mass10", "mass3"):
            assert numbers[f"{key}_{tag}"] == f"{float(rows[limit][key]):,.0f}"
        assert numbers[f"cand_{tag}"] == f"{float(rows[limit]['candidates']):,.0f}"
    # Cumulative counts rise with the limit, and an orbit is needed for a mass.
    limits = sorted(rows)
    for key in ("candidates", "orbit", "mass10", "mass3"):
        values = [float(rows[limit][key]) for limit in limits]
        assert all(b >= a - 1e-6 for a, b in itertools.pairwise(values))
    assert float(rows[13.5]["mass3"]) <= float(rows[13.5]["mass10"]) <= float(rows[13.5]["orbit"])
    assert float(rows[13.5]["orbit"]) <= float(rows[13.5]["candidates"])


def test_the_single_star_numbers_are_those_of_the_table(numbers):
    """The comparison with the scatter that Gaia DR3 publishes, the first limit of the page."""
    rows = _table("check-single-stars")
    assert [float(row["grvs_low"]) for row in rows] == [6.0, 7.5, 9.5, 11.5]
    for tag, row in zip(("6", "8", "10", "12"), rows, strict=True):
        assert int(row["n_transits"]) >= 100
        sigma = float(row["robust_sigma_kms"])
        assert numbers[f"single_sigma_{tag}"] == f"{sigma:.2f}"
        assert numbers[f"single_ratio_{tag}"] == f"{float(row['gaia_dr3_kms']) / sigma:.1f}"
        assert float(row["gaia_over_simulated"]) > 0.9
    # A fainter transit is noisier.
    scatter = [float(row["robust_sigma_kms"]) for row in rows]
    assert scatter == sorted(scatter)


def test_the_recorded_systems_reproduce_the_notebook(numbers):
    # The numbers of the TODCOR notebook for the same 33 systems and seeds.
    assert numbers["rec_n"] == "33" and numbers["rec_usable"] == "93.4"
    assert numbers["rec_wide"] == "21 of 22" and numbers["rec_close"] == "9 of 11"


def test_the_wilson_interval(builder):
    p, lo, hi = builder.wilson(5, 10)
    assert p == pytest.approx(0.5) and lo == pytest.approx(0.5 - 0.5 / np.sqrt(11.0))
    assert hi == pytest.approx(1.0 - lo)
    # At the ends the interval starts or ends at the fraction itself.
    p, lo, hi = builder.wilson([0, 20], [20, 20])
    assert lo[0] == pytest.approx(0.0, abs=1e-12) and hi[1] == pytest.approx(1.0, abs=1e-12)
    assert hi[0] == pytest.approx(1.0 / 21.0) and lo[1] == pytest.approx(20.0 / 21.0)
    assert np.isnan(builder.wilson(0, 0)[0])


def test_the_binned_fractions_and_the_quantiles(builder):
    x = np.array([0.1, 0.2, 1.1, 1.2, 1.3, 2.5])
    flag = np.array([True, False, True, True, False, True])
    n, k, p, _, _ = builder.fraction_by_bin(x, flag, np.array([0.0, 1.0, 2.0, 3.0]))
    assert n.tolist() == [2, 3, 1] and k.tolist() == [1, 2, 1]
    np.testing.assert_allclose(p, [0.5, 2 / 3, 1.0])
    n, k, _, _, _ = builder.fraction_by_bin(x, flag, np.array([0.0, 1.0, 2.0, 3.0]), x > 1.15)
    assert n.tolist() == [0, 2, 1] and k.tolist() == [0, 1, 1]
    values = np.arange(1.0, 102.0)
    assert builder.weighted_quantile(values, [0.5])[0] == pytest.approx(51.0)
    # A weight of two is the value twice.
    doubled = builder.weighted_quantile([1.0, 2.0, 3.0], [0.5], [1.0, 1.0, 2.0])[0]
    assert doubled == pytest.approx(builder.weighted_quantile([1.0, 2.0, 3.0, 3.0], [0.5])[0])
    assert builder.robust_sigma(np.random.default_rng(0).normal(0, 2.0, 20000)) == pytest.approx(
        2.0, rel=0.03
    )
    assert np.isnan(builder.robust_sigma([np.nan]))


def test_the_recovery_function_is_recovered_from_binary_outcomes(builder):
    rng = np.random.default_rng(3)
    x = 10 ** rng.uniform(0.0, 3.0, size=6000)
    truth = builder._logistic(x, 0.9, 1.6, 0.2)
    top, mid, width = builder.fit_recovery(x, rng.uniform(size=x.size) < truth)
    assert top == pytest.approx(0.9, abs=0.03)
    assert mid == pytest.approx(1.6, abs=0.05) and width == pytest.approx(0.2, abs=0.04)

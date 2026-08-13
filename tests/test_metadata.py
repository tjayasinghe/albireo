"""The version string lives in one place; these tests keep the copies honest.

``src/albireo/__init__.py`` is the single source of truth. ``pyproject.toml`` reads it via
``[tool.hatch.version]``, so the built distribution cannot disagree. ``CITATION.cff`` is the
one copy no build backend can reach — it is plain data, read by GitHub and Zenodo rather
than by Python — so it is checked here instead.
"""

from __future__ import annotations

import re
from importlib import metadata
from pathlib import Path

import pytest

import albireo

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_version_is_a_pep440_release_or_prerelease():
    # Not a full PEP 440 grammar — just enough to catch a hand-edit that would make the
    # distribution unbuildable or sort strangely on PyPI.
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:\.?(?:a|b|rc|dev)\d+)?", albireo.__version__), (
        f"{albireo.__version__!r} is not a version string hatchling and PyPI will accept"
    )


def test_installed_distribution_matches_dunder_version():
    try:
        installed = metadata.version("albireo")
    except metadata.PackageNotFoundError:  # pragma: no cover - running from a bare checkout
        pytest.skip("albireo is not installed; nothing to compare against")
    assert installed == albireo.__version__, (
        "the installed distribution and albireo.__version__ disagree — reinstall "
        "(`pip install -e .`) after a version bump"
    )


def test_citation_cff_version_matches():
    cff = REPO_ROOT / "CITATION.cff"
    if not cff.is_file():  # pragma: no cover - wheels do not ship CITATION.cff
        pytest.skip("CITATION.cff is not present in this tree")
    # Deliberately a regex rather than a YAML parse: pyyaml is not a dependency, and the
    # one field being checked is unambiguous on its own line.
    match = re.search(r"^version:\s*(\S+)\s*$", cff.read_text(encoding="utf-8"), re.MULTILINE)
    assert match is not None, "CITATION.cff has no `version:` field"
    assert match.group(1) == albireo.__version__, (
        "CITATION.cff records a different version from albireo.__version__ — bump both, "
        "and remember Zenodo reads this file when it archives a release"
    )

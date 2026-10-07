"""The notebook pages link to other pages by source path; the site needs page URLs.

A notebook under ``docs/tutorials/`` writes ``[science overview](../science.md)``, which
GitHub's notebook viewer resolves. mkdocs-jupyter publishes the ``href`` as written, and
on the site it requests a path that does not exist. ``scripts/mkdocs_notebook_links_hook.py``
rewrites these links after rendering. These tests check the rewriting against a stub of
the MkDocs file collection, and check that every such link in the committed notebooks
names an existing page.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import posixpath
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS = REPO_ROOT / "docs"

MARKDOWN_LINK = re.compile(r"\]\(([^)\s]+)\)")


def _load_hook():
    path = REPO_ROOT / "scripts" / "mkdocs_notebook_links_hook.py"
    spec = importlib.util.spec_from_file_location("mkdocs_notebook_links_hook", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _File:
    """The part of ``mkdocs.structure.files.File`` the hook reads, with directory URLs."""

    def __init__(self, src_uri: str):
        self.src_uri = src_uri
        stem = src_uri.rsplit(".", 1)[0]
        self.url = "./" if stem == "index" else stem + "/"

    def is_documentation_page(self) -> bool:
        return self.src_uri.endswith((".md", ".ipynb"))

    def url_relative_to(self, other: _File) -> str:
        return posixpath.relpath(self.url, other.url) + "/"


class _Files:
    def __init__(self, *src_uris: str):
        self._files = {uri: _File(uri) for uri in src_uris}

    def get_file_from_path(self, path: str):
        return self._files.get(path)


class _Page:
    def __init__(self, file: _File):
        self.file = file


FILES = _Files(
    "science.md",
    "benchmarks.md",
    "tutorials/gaia-rvs.md",
    "tutorials/showcase.ipynb",
    "tutorials/gaia-rvs-benchmark.ipynb",
)
NOTEBOOK = _Page(FILES.get_file_from_path("tutorials/gaia-rvs-benchmark.ipynb"))


@pytest.mark.parametrize(
    ("href", "expected"),
    [
        ("../science.md", "../../science/"),
        ("../benchmarks.md#gaia-rvs", "../../benchmarks/#gaia-rvs"),
        ("gaia-rvs.md", "../gaia-rvs/"),
        ("showcase.ipynb", "../showcase/"),
        # Not a relative page link: left as written.
        ("https://example.org/page.md", "https://example.org/page.md"),
        ("#section", "#section"),
        ("/albireo/science.md", "/albireo/science.md"),
        ("../figure.png", "../figure.png"),
    ],
)
def test_relative_page_link_becomes_page_url(href, expected):
    hook = _load_hook()
    html = f'<p>See <a class="x" href="{href}">the page</a>.</p>'
    out = hook.on_page_content(html, page=NOTEBOOK, files=FILES, config=None)
    assert out == html.replace(f'href="{href}"', f'href="{expected}"')


def test_markdown_page_is_left_unchanged():
    hook = _load_hook()
    html = '<a href="../science.md">x</a>'
    page = _Page(FILES.get_file_from_path("tutorials/gaia-rvs.md"))
    assert hook.on_page_content(html, page=page, files=FILES, config=None) == html


def test_escaped_anchor_in_cell_output_is_left_unchanged():
    hook = _load_hook()
    html = "<pre>&lt;a href=&quot;../science.md&quot;&gt;</pre>"
    assert hook.on_page_content(html, page=NOTEBOOK, files=FILES, config=None) == html


def test_missing_target_is_a_warning(caplog):
    # A warning fails `mkdocs build --strict`.
    hook = _load_hook()
    html = '<a href="../absent.md">x</a>'
    with caplog.at_level(logging.WARNING, logger="mkdocs.hooks.notebook_links"):
        out = hook.on_page_content(html, page=NOTEBOOK, files=FILES, config=None)
    assert out == html
    assert "absent.md" in caplog.text


@pytest.mark.parametrize("notebook", sorted(DOCS.rglob("*.ipynb")), ids=lambda p: p.name)
def test_committed_notebook_links_name_existing_pages(notebook):
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    for cell in cells:
        if cell["cell_type"] != "markdown":
            continue
        for target in MARKDOWN_LINK.findall("".join(cell["source"])):
            path = target.split("#", 1)[0]
            if "://" in target or not path.endswith((".md", ".ipynb")):
                continue
            assert (notebook.parent / path).resolve().is_file(), f"{notebook.name}: {target}"

"""Resolve relative ``.md`` links in the rendered notebook pages.

The executed notebooks under ``docs/tutorials/`` link to other pages by relative source
path, for example ``[science overview](../science.md)``. GitHub's notebook viewer
resolves such a link against ``docs/tutorials/``, so it works there. MkDocs rewrites the
same link in a Markdown page to the URL of the built page, but mkdocs-jupyter renders a
notebook through nbconvert, outside the MkDocs Markdown pipeline, and the ``href`` is
published as written. With directory URLs the notebook is served at
``/tutorials/<name>/``, so ``../science.md`` requests ``/tutorials/science.md``, which
does not exist. ``mkdocs build --strict`` does not report this, because the link
validation is part of the pipeline the notebook bypasses. mkdocs-jupyter 0.26 has no
setting for it.

This hook applies the MkDocs rule to the notebook pages after rendering. Each relative
``href`` whose path ends in ``.md`` or ``.ipynb`` is resolved against the notebook's
source path and replaced by the relative URL of the target page. A target that is not a
documentation page is logged as a warning, which fails a strict build, so that a notebook
link is checked in the same way as a link in a Markdown page. The notebook sources are
not modified and remain valid on GitHub.

``tests/test_docs_notebook_links.py`` checks the rewriting.
"""

from __future__ import annotations

import logging
import posixpath
import re
from html import escape, unescape
from urllib.parse import urlsplit, urlunsplit

log = logging.getLogger("mkdocs.hooks.notebook_links")

# The href of an anchor element. Text in a cell output is HTML-escaped by nbconvert, so
# a printed `<a href="...">` does not match.
ANCHOR_HREF = re.compile(r'(<a\b[^>]*?\bhref=")([^"]*)(")')

PAGE_SUFFIXES = (".md", ".ipynb")


def _rewrite(href: str, page, files) -> str:
    parts = urlsplit(unescape(href))
    if parts.scheme or parts.netloc or parts.path.startswith("/"):
        return href
    if not parts.path.endswith(PAGE_SUFFIXES):
        return href
    source = page.file.src_uri
    target_uri = posixpath.normpath(posixpath.join(posixpath.dirname(source), parts.path))
    target = files.get_file_from_path(target_uri)
    if target is None or not target.is_documentation_page():
        log.warning(
            "Notebook '%s' contains a link '%s', but the target '%s' is not found among "
            "documentation files.",
            source,
            href,
            target_uri,
        )
        return href
    url = target.url_relative_to(page.file)
    return escape(urlunsplit(("", "", url, parts.query, parts.fragment)), quote=True)


def on_page_content(html: str, page, files, **_kwargs) -> str:
    # A Markdown page has already been through the MkDocs link rewriting.
    if not page.file.src_uri.endswith(".ipynb"):
        return html
    return ANCHOR_HREF.sub(
        lambda m: m.group(1) + _rewrite(m.group(2), page, files) + m.group(3), html
    )

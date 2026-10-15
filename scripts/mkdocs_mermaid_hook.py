"""Render Mermaid fences as diagrams on the MkDocs site.

GitHub draws a ``` ```mermaid ``` fence as a diagram, and Python-Markdown renders the
same fence as a code block. Material for MkDocs draws a diagram from every
``<pre class="mermaid">`` element on a page, loading Mermaid itself and following the
light or dark palette. Its documented route to that element is a superfences custom
fence, which needs a ``!!python/name`` tag in ``mkdocs.yml`` that the check-yaml
pre-commit hook cannot load; ``scripts/mkdocs_math_hook.py`` converts the math fences
for the same reason. This hook writes the element directly, before the Markdown parser
runs. The fence must start in the first column, outside lists and admonitions.

``tests/test_docs_mermaid.py`` checks the conversion and the pages that use it.
"""

from __future__ import annotations

import html
import re

MERMAID = re.compile(r"^```mermaid[ \t]*\n(.*?)\n```[ \t]*$", re.DOTALL | re.MULTILINE)


def on_page_markdown(markdown: str, **_kwargs) -> str:
    # The blank lines make the element a raw HTML block, which Python-Markdown passes
    # through unchanged. The diagram source is escaped so that an arrow such as -->
    # stays text; Mermaid reads the element's text content, which undoes the escaping.
    return MERMAID.sub(
        lambda m: f'\n<pre class="mermaid"><code>{html.escape(m.group(1))}</code></pre>\n',
        markdown,
    )

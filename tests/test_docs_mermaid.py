"""Mermaid diagrams render on both of the documentation's renderers.

GitHub draws a ``` ```mermaid ``` fence itself. On the MkDocs site
``scripts/mkdocs_mermaid_hook.py`` turns the fence into the ``<pre class="mermaid">``
element that Material for MkDocs draws. The conversion only sees a fence that starts in
the first column, so a fence inside a list or an admonition would reach the site as a
code block, without an error. Math is not rendered inside a diagram on either renderer.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INDENTED_FENCE = re.compile(r"^[ \t]+```mermaid", re.MULTILINE)
FENCE = re.compile(r"^```mermaid[ \t]*\n(.*?)\n```[ \t]*$", re.DOTALL | re.MULTILINE)


def _pages() -> list[Path]:
    pages = [*sorted((REPO_ROOT / "docs").rglob("*.md")), REPO_ROOT / "README.md"]
    return [page for page in pages if "```mermaid" in page.read_text(encoding="utf-8")]


def _load_hook():
    path = REPO_ROOT / "scripts" / "mkdocs_mermaid_hook.py"
    spec = importlib.util.spec_from_file_location("mkdocs_mermaid_hook", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_hook_writes_the_element_material_draws():
    hook = _load_hook()
    out = hook.on_page_markdown("before\n\n```mermaid\nflowchart LR\n  a --> b\n```\n\nafter\n")
    assert '<pre class="mermaid"><code>flowchart LR\n  a --&gt; b</code></pre>' in out
    assert "```" not in out
    assert out.startswith("before") and out.rstrip().endswith("after")


def test_the_hook_leaves_other_fences_alone():
    hook = _load_hook()
    source = "```python\nprint('a --> b')\n```\n\n```math\na = b\n```\n"
    assert hook.on_page_markdown(source) == source


def test_the_hook_runs_after_the_math_hook():
    config = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    hooks = re.findall(r"^\s*-\s*(scripts/mkdocs_\w+_hook\.py)", config, re.MULTILINE)
    assert "scripts/mkdocs_mermaid_hook.py" in hooks, "the hook is not listed in mkdocs.yml"
    assert hooks.index("scripts/mkdocs_math_hook.py") < hooks.index(
        "scripts/mkdocs_mermaid_hook.py"
    )


@pytest.mark.parametrize("page", _pages(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix())
def test_diagrams_start_in_the_first_column_and_hold_no_math(page: Path):
    text = page.read_text(encoding="utf-8").replace("\r\n", "\n")
    name = page.relative_to(REPO_ROOT).as_posix()
    assert not INDENTED_FENCE.search(text), (
        f"{name} has an indented ```mermaid fence, which the site renders as a code block"
    )
    for block in FENCE.finditer(text):
        assert "$" not in block.group(1), f"{name} has math inside a Mermaid diagram"

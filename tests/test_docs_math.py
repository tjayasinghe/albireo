"""The documentation is read on two renderers; these tests keep it legible on both.

Every page is Markdown that GitHub shows directly and that MkDocs also builds into the
site. The two disagree about how to spell math, and only one spelling survives both.

GitHub runs its Markdown parser over a bare ``$...$`` before its math extension sees it,
so the expression is quietly mangled rather than rejected: a backslash escape (``\\,``
``\\!`` ``\\;`` ``\\%`` ``\\{``) is eaten as a Markdown escape and renders as the bare
punctuation, a subscript underscore that follows a bracket can pair with a later one and
turn the middle of the formula into italics, an opening ``$`` preceded by anything but a
space or ``(`` does not start math at all, and a ``$$`` block containing a line that
begins ``- `` or ``+ `` becomes a bullet list. None of this errors; it just renders
wrongly, which is why it needs a test rather than a build step.

The forms GitHub documents for this -- ``$`x`$`` inline and a ``` ```math ``` fence for
display -- are opaque to its Markdown parser. ``scripts/mkdocs_math_hook.py`` and the
``custom_fences`` entry in ``mkdocs.yml`` map both back onto arithmatex for the site.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

FENCE_OPEN = re.compile(r"^[ \t]*(`{3,}|~{3,})")
BACKTICKS = re.compile(r"`+")


def _markdown_pages() -> list[Path]:
    """The pages this convention governs: the site's sources, and the front page.

    ``paper/paper.md`` is deliberately not among them. Open Journals compiles it with
    pandoc, which reads ``$...$`` and not the backtick form, so it has to stay in the
    LaTeX spelling; its handful of expressions happen to satisfy GitHub's rules anyway
    (each opens after a space, none carries a backslash escape).
    """
    pages = sorted((REPO_ROOT / "docs").rglob("*.md"))
    readme = REPO_ROOT / "README.md"
    if readme.is_file():
        pages.append(readme)
    return pages


def _page_ids() -> list[str]:
    return [p.relative_to(REPO_ROOT).as_posix() for p in _markdown_pages()]


def _bare_dollars(text: str) -> list[tuple[int, str]]:
    """Line numbers of every ``$`` that is neither ``$`x`$`` math nor inside code.

    Scans left to right in the same order the writers' converter did, so a ``$`` is
    reported only when no legitimate construct can claim it.
    """
    lines = text.split("\n")
    offsets = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1

    def line_of(index: int) -> int:
        lo, hi = 0, len(offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if offsets[mid] <= index:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    found: list[tuple[int, str]] = []
    i = 0
    in_fence: str | None = None
    at_line_start = True

    while i < len(text):
        if at_line_start:
            line = text[i : text.find("\n", i) if "\n" in text[i:] else len(text)]
            marker = FENCE_OPEN.match(line)
            if in_fence is not None:
                if marker and line.strip().startswith(in_fence):
                    in_fence = None
                i += len(line) + 1
                continue
            if marker:
                in_fence = marker.group(1)
                i += len(line) + 1
                continue

        char = text[i]
        if char == "\n":
            at_line_start = True
            i += 1
            continue
        at_line_start = False

        if char == "$":
            # $`x`$ -- the run of backticks after the dollar must close the same way.
            ticks = BACKTICKS.match(text, i + 1)
            if ticks:
                close = text.find(ticks.group(0) + "$", ticks.end())
                if close != -1:
                    i = close + len(ticks.group(0)) + 1
                    continue
            found.append((line_of(i), lines[line_of(i) - 1].strip()))
            i += 1
            continue

        if char == "`":
            ticks = BACKTICKS.match(text, i).group(0)
            close = text.find(ticks, i + len(ticks))
            if close != -1:
                i = close + len(ticks)
                continue

        i += 1

    return found


@pytest.mark.parametrize("page", _markdown_pages(), ids=_page_ids())
def test_no_bare_dollar_math(page: Path):
    text = page.read_text(encoding="utf-8").replace("\r\n", "\n")
    offenders = _bare_dollars(text)
    assert not offenders, (
        f"{page.relative_to(REPO_ROOT).as_posix()} has math GitHub will render wrongly. "
        "Inline math must be written $`x`$, not $x$ — a bare $ lets GitHub's Markdown "
        "parser eat the backslash escapes and pair the subscript underscores. First "
        f"offender, line {offenders[0][0]}: {offenders[0][1][:90]!r}"
    )


@pytest.mark.parametrize("page", _markdown_pages(), ids=_page_ids())
def test_display_math_uses_a_fence(page: Path):
    text = page.read_text(encoding="utf-8").replace("\r\n", "\n")
    # A $$ delimiter would already have been reported as two bare dollars above; this
    # names the display case specifically so the failure says what to write instead.
    lines = [n for n, line in enumerate(text.split("\n"), 1) if "$$" in line]
    assert not lines, (
        f"{page.relative_to(REPO_ROOT).as_posix()} uses $$ for display math at line "
        f"{lines[0] if lines else '?'}. Write a ```math fence instead — GitHub parses the "
        "block's contents as Markdown, so a line beginning '- ' or '+ ' becomes a list."
    )


@pytest.mark.parametrize("page", _markdown_pages(), ids=_page_ids())
def test_math_fences_do_not_end_a_line_with_a_tex_break(page: Path):
    text = page.read_text(encoding="utf-8").replace("\r\n", "\n")
    offenders = []
    for block in re.finditer(r"^```math\n(.*?)\n```$", text, re.DOTALL | re.MULTILINE):
        first = text[: block.start()].count("\n") + 2
        for n, line in enumerate(block.group(1).split("\n")):
            if re.search(r"\\\\[ \t]*$", line):
                offenders.append((first + n, line.strip()))
    assert not offenders, (
        f"{page.relative_to(REPO_ROOT).as_posix()} ends a line inside a ```math fence with "
        f"a TeX line break (line {offenders[0][0] if offenders else '?'}). GitHub emits a "
        "spurious third backslash for that; begin the following line with '\\\\ ' instead."
    )


def _load_hook():
    path = REPO_ROOT / "scripts" / "mkdocs_math_hook.py"
    spec = importlib.util.spec_from_file_location("mkdocs_math_hook", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hook_restores_the_dollar_form_arithmatex_reads():
    hook = _load_hook()
    assert hook.on_page_markdown(r"a $`\alpha \, \beta`$ b") == r"a $\alpha \, \beta$ b"
    # An expression may be wrapped across two source lines.
    assert hook.on_page_markdown("x $`a +\nb`$ y") == "x $a +\nb$ y"
    # The padding the writer adds around a body that starts with a backtick comes back off.
    assert hook.on_page_markdown("$`` `a ``$") == "$`a$"


def test_hook_turns_a_math_fence_into_a_display_block():
    hook = _load_hook()
    out = hook.on_page_markdown("before\n\n```math\na \\, b\n  - c\n```\n\nafter\n")
    assert "$$\na \\, b\n  - c\n$$" in out
    assert "```" not in out


def test_hook_separates_back_to_back_math_fences():
    # Python-Markdown reads two $$ blocks with no blank line between them as one
    # paragraph and hands arithmatex a single malformed expression.
    hook = _load_hook()
    out = hook.on_page_markdown("```math\nA = 1\n```\n```math\nB = 2\n```\n")
    assert re.search(r"\$\$\nA = 1\n\$\$\n\s*\n\s*\$\$\nB = 2\n\$\$", out), out


def test_hook_leaves_fenced_blocks_alone():
    hook = _load_hook()
    source = "before $`x`$\n\n```python\nprint"
    source += '("$`not math`$")\n```\n\nafter $`y`$\n'
    out = hook.on_page_markdown(source)
    assert '("$`not math`$")' in out, "a code sample must survive the hook verbatim"
    assert out.startswith("before $x$")
    assert out.endswith("after $y$\n")


def test_hook_leaves_shell_variables_alone():
    hook = _load_hook()
    source = 'On Windows, `$env:ALBIREO_EXAMPLE_FAST = "1"` sets it; `$ALBIREO_DATA_DIR` moves it.'
    assert hook.on_page_markdown(source) == source

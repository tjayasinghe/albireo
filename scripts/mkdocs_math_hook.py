"""Rewrite GitHub's inline math spelling into the one arithmatex reads.

The documentation is read in two places: on this site, and as plain Markdown on the
repository page. GitHub parses a bare ``$...$`` as ordinary Markdown before its math
extension sees it, which silently destroys the expression -- every backslash escape
(``\\,`` ``\\!`` ``\\;`` ``\\%`` ``\\{``) is eaten as a Markdown escape, a pair of
subscript underscores turns the middle of a formula into italics, and an opening ``$``
that follows anything but a space or ``(`` does not start math at all. The two forms
GitHub documents for this, ``$`x`$`` inline and a ``` ```math ``` fence for display, are
opaque to its Markdown parser and come through intact.

Python-Markdown makes the opposite trade: the backticks in ``$`x`$`` become a code span
before arithmatex can claim the expression, and a ``` ```math ``` fence is a code block.
So the sources are written in GitHub's spelling and this hook converts both back on the
way in, before the Markdown parser runs.

``tests/test_docs_math.py`` holds the source side to the same convention.
"""

from __future__ import annotations

import re

# $ <backtick run> body <same backtick run> $, the body being anything at all, including
# newlines: an inline expression may be wrapped across two source lines.
INLINE_MATH = re.compile(r"\$(`+)(.+?)\1\$", re.DOTALL)

# A fenced block of any kind, so that code samples are left exactly as written.
FENCED_BLOCK = re.compile(
    r"^(?P<indent>[ \t]*)(?P<marker>`{3,}|~{3,})[^\n]*\n.*?(?:^(?P=indent)(?P=marker)[ \t]*$|\Z)",
    re.DOTALL | re.MULTILINE,
)

# The display form. Handled before anything else, so the block is already $$...$$ by the
# time the inline pass walks the page and never looks like a code fence to it.
DISPLAY_MATH = re.compile(r"^```math[ \t]*\n(.*?)\n```[ \t]*$", re.DOTALL | re.MULTILINE)


def _unwrap(match: re.Match[str]) -> str:
    body = match.group(2)
    # CommonMark strips one leading and one trailing space from a code span when both are
    # present; the writer side adds that padding only to protect a body that itself starts
    # or ends with a backtick. Undo exactly that much and no more.
    if len(body) > 1 and body.startswith(" ") and body.endswith(" "):
        body = body[1:-1]
    return f"${body}$"


def on_page_markdown(markdown: str, **_kwargs) -> str:
    # The blank lines matter. Two ```math fences may sit back to back, which GitHub reads
    # as two blocks; without the separation Python-Markdown would take the pair for one
    # paragraph and hand arithmatex a single malformed \[ A $$ $$ B \].
    markdown = DISPLAY_MATH.sub(lambda m: f"\n$$\n{m.group(1)}\n$$\n", markdown)
    out = []
    pos = 0
    for block in FENCED_BLOCK.finditer(markdown):
        out.append(INLINE_MATH.sub(_unwrap, markdown[pos : block.start()]))
        out.append(block.group(0))
        pos = block.end()
    out.append(INLINE_MATH.sub(_unwrap, markdown[pos:]))
    return "".join(out)

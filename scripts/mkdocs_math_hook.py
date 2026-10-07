"""Rewrite GitHub's inline math spelling into the one arithmatex reads.

The documentation is read in two places: on this site, and as plain Markdown on the
repository page. GitHub parses a bare ``$...$`` as ordinary Markdown before its math
extension processes it, which silently destroys the expression. Every backslash escape
(``\\,`` ``\\!`` ``\\;`` ``\\%`` ``\\{``) is consumed as a Markdown escape. A pair of
subscript underscores turns the middle of a formula into italics. An opening ``$``
that follows anything but a space or ``(`` does not start math. The two forms GitHub
documents for this, ``$`x`$`` inline and a ``` ```math ``` fence for display, are opaque
to its Markdown parser and are left intact.

Python-Markdown has the opposite problem: the backticks in ``$`x`$`` become a code span
before arithmatex can process the expression, and a ``` ```math ``` fence is a code
block. The sources are therefore written in GitHub's spelling, and this hook converts
both back before the Markdown parser runs.

``tests/test_docs_math.py`` checks that the sources follow the same convention.
"""

from __future__ import annotations

import re

# $ <backtick run> body <same backtick run> $, the body being anything, including
# newlines: an inline expression may be wrapped across two source lines.
INLINE_MATH = re.compile(r"\$(`+)(.+?)\1\$", re.DOTALL)

# A fenced block of any kind, so that code samples are left exactly as written.
FENCED_BLOCK = re.compile(
    r"^(?P<indent>[ \t]*)(?P<marker>`{3,}|~{3,})[^\n]*\n.*?(?:^(?P=indent)(?P=marker)[ \t]*$|\Z)",
    re.DOTALL | re.MULTILINE,
)

# The display form. Handled before anything else, so the block is already $$...$$ when
# the inline pass scans the page and is not matched as a code fence.
DISPLAY_MATH = re.compile(r"^```math[ \t]*\n(.*?)\n```[ \t]*$", re.DOTALL | re.MULTILINE)


def _unwrap(match: re.Match[str]) -> str:
    body = match.group(2)
    # CommonMark strips one leading and one trailing space from a code span when both are
    # present. The writer side adds that padding only to protect a body that itself starts
    # or ends with a backtick. Remove only that padding.
    if len(body) > 1 and body.startswith(" ") and body.endswith(" "):
        body = body[1:-1]
    return f"${body}$"


def on_page_markdown(markdown: str, **_kwargs) -> str:
    # The blank lines are required. Two ```math fences may be adjacent, which GitHub reads
    # as two blocks. Without the separation Python-Markdown would treat the pair as one
    # paragraph and pass arithmatex a single malformed \[ A $$ $$ B \].
    markdown = DISPLAY_MATH.sub(lambda m: f"\n$$\n{m.group(1)}\n$$\n", markdown)
    out = []
    pos = 0
    for block in FENCED_BLOCK.finditer(markdown):
        out.append(INLINE_MATH.sub(_unwrap, markdown[pos : block.start()]))
        out.append(block.group(0))
        pos = block.end()
    out.append(INLINE_MATH.sub(_unwrap, markdown[pos:]))
    return "".join(out)

"""The modules of the package import one another without a cycle.

A cycle between two modules runs only while one of them defers its import into a
function, and it ties the two together: neither can be read, tested or moved alone. Two
were removed in October 2026. The forward model imported the simulator for its response
polynomial, and `albireo.facade` and the label fit (`albireo.match`) imported each other.
The test reads the source rather than importing it, so that imports made inside functions
are counted.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "src" / "albireo"
MODULES = {path.stem for path in PACKAGE.glob("*.py")} - {"__init__", "__main__"}


def _imported_modules(path: Path) -> set[str]:
    """The package modules that one module imports, at any depth of its source."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            parts = node.module.split(".")
            if parts[0] == "albireo":
                found.update(parts[1:2] if len(parts) > 1 else (a.name for a in node.names))
        elif isinstance(node, ast.Import):
            found.update(a.name.split(".")[1] for a in node.names if a.name.startswith("albireo."))
    return (found & MODULES) - {path.stem}


def _cycle(graph: dict[str, set[str]]) -> list[str] | None:
    """One cycle of the graph as a list of modules, or None when the graph is acyclic."""
    state: dict[str, int] = {}

    def visit(module: str, path: list[str]) -> list[str] | None:
        state[module] = 1
        for target in sorted(graph[module]):
            if state.get(target) == 1:
                return [*path[path.index(target) :], target]
            if target not in state and (found := visit(target, [*path, target])):
                return found
        state[module] = 2
        return None

    for module in sorted(graph):
        if module not in state and (found := visit(module, [module])):
            return found
    return None


def test_the_import_graph_has_no_cycle():
    graph = {module: _imported_modules(PACKAGE / f"{module}.py") for module in MODULES}
    cycle = _cycle(graph)
    assert cycle is None, "import cycle: " + " -> ".join(cycle or [])


def test_the_cycle_finder_reports_a_cycle():
    graph = {"a": {"b"}, "b": {"c"}, "c": {"a"}, "d": set()}
    assert _cycle(graph) == ["a", "b", "c", "a"]
    assert _cycle({"a": {"b"}, "b": set()}) is None


def test_the_core_model_does_not_import_the_simulator():
    assert "simulate" not in _imported_modules(PACKAGE / "forward.py")
    assert "facade" not in _imported_modules(PACKAGE / "match.py")

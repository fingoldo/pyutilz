"""E4 — meta-test that the package's internal import graph is acyclic.

A circular import can lurk for years in a package because Python's
import machinery resolves cycles at module-level if the offending name
is accessed lazily. Then a refactor moves a top-level access of one of
the cycle members and ``ImportError: cannot import name X from
partially initialized module Y`` ships to users.

Builds a dep graph by AST-walking every production .py for ``import`` /
``from`` statements; restricts to imports whose target is inside the
package itself (third-party deps are out of scope). Runs Tarjan's SCC
to find any cycle of size > 1.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import pytest

import pyutilz

PYUTILZ_DIR = Path(pyutilz.__file__).resolve().parent
PKG_NAME = "pyutilz"


def _module_name_from_path(path: Path) -> str:
    """``src/pyutilz/llm/factory.py`` → ``pyutilz.llm.factory``."""
    rel = path.relative_to(PYUTILZ_DIR)
    parts = list(rel.parts)
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][: -len(".py")]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join([PKG_NAME] + parts)


def _internal_imports(tree: ast.AST, current: str) -> set[str]:
    """Yield fully-qualified names this module imports from inside the
    same package, considering ONLY top-level imports — lazy imports
    inside function bodies don't participate in the module-load
    dependency graph (they fire after both modules have finished
    loading), so a "cycle" that only closes via lazy imports isn't a
    runtime ImportError waiting to happen.
    """
    out: set[str] = set()
    current_parts = current.split(".")
    # Walk ONLY top-level statements (and ``if`` / ``try`` blocks at
    # module scope, e.g. ``try: import optdep / except ImportError``).
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(PKG_NAME):
                    out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module and node.module.startswith(PKG_NAME):
                    out.add(node.module)
            else:
                base_parts = current_parts[: -node.level]
                if node.module:
                    base_parts.append(node.module)
                if base_parts and base_parts[0] == PKG_NAME:
                    out.add(".".join(base_parts))
        elif isinstance(node, (ast.If, ast.Try)):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Import):
                    for alias in sub.names:
                        if alias.name.startswith(PKG_NAME):
                            out.add(alias.name)
                elif isinstance(sub, ast.ImportFrom):
                    if sub.level == 0 and sub.module and sub.module.startswith(PKG_NAME):
                        out.add(sub.module)
    return out


def _build_graph() -> dict[str, set[str]]:
    """``{module_name: set_of_imported_internal_module_names}``."""
    graph: dict[str, set[str]] = defaultdict(set)
    for py in PYUTILZ_DIR.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        if py.name.endswith(".py.old"):
            continue
        try:
            src = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        mod_name = _module_name_from_path(py)
        graph[mod_name].update(_internal_imports(tree, mod_name))
    return graph


def _strongly_connected_components(graph: dict[str, set[str]]) -> list[list[str]]:
    """Tarjan's SCC. Returns list of components (each ≥ 1 node).
    Cycles are SCCs with > 1 node OR self-loops."""
    index_counter = [0]
    stack: list[str] = []
    lowlinks: dict[str, int] = {}
    index: dict[str, int] = {}
    on_stack: dict[str, bool] = {}
    result: list[list[str]] = []

    def strongconnect(v: str):
        index[v] = index_counter[0]
        lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True

        for w in graph.get(v, ()):
            if w not in index:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif on_stack.get(w, False):
                lowlinks[v] = min(lowlinks[v], index[w])

        if lowlinks[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                comp.append(w)
                if w == v:
                    break
            result.append(comp)

    import sys
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(10_000)
    try:
        for v in list(graph):
            if v not in index:
                strongconnect(v)
    finally:
        sys.setrecursionlimit(old_limit)
    return result


def test_no_import_cycles_in_package():
    graph = _build_graph()
    assert graph, "no modules discovered — package layout broken?"

    sccs = _strongly_connected_components(graph)
    cycles: list[list[str]] = []
    for comp in sccs:
        # Multi-node SCCs are unambiguous import cycles. Single-node
        # self-loops are typically a package ``__init__.py`` doing
        # ``from .submodule import X`` — Python's relative-import machinery
        # resolves that as ``pyutilz.foo.foo`` (the submodule), but our
        # AST-level resolver collapses both to ``pyutilz.foo``. We skip
        # those as false positives; if a true direct self-import ever
        # ships, ``ImportError`` at runtime is the louder backstop.
        if len(comp) > 1:
            cycles.append(comp)

    if cycles:
        details = []
        for cyc in cycles:
            details.append(" → ".join(cyc + [cyc[0]]))
        pytest.fail(
            f"{len(cycles)} import cycle(s) detected in {PKG_NAME}:\n  "
            + "\n  ".join(details)
        )

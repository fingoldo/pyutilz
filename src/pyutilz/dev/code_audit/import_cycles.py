"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path
from typing import FrozenSet

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _safe_parse

# --- internal import-graph cycle detection ----------------------------------


def _module_name_from_path(path: Path, root: Path, package_name: str) -> str:
    """``<root>/llm/factory.py`` -> ``<package_name>.llm.factory``."""
    rel = path.relative_to(root)
    parts = list(rel.parts)
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][: -len(".py")]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join([package_name, *parts])


def _internal_imports(tree: ast.AST, current: str, package_name: str) -> set[str]:
    """Yield fully-qualified names ``current`` imports from inside the same package, considering
    ONLY top-level imports -- lazy imports inside function bodies don't participate in the
    module-load dependency graph (they fire after both modules have finished loading), so a
    "cycle" that only closes via lazy imports isn't a runtime ImportError waiting to happen.
    """
    out: set[str] = set()
    current_parts = current.split(".")
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(package_name):
                    out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module and node.module.startswith(package_name):
                    out.add(node.module)
            else:
                base_parts = current_parts[: -node.level]
                if node.module:
                    base_parts.append(node.module)
                if base_parts and base_parts[0] == package_name:
                    out.add(".".join(base_parts))
        elif isinstance(node, ast.If):
            # `if TYPE_CHECKING: ... import X` is not at runtime, already excluded by iterating
            # top-level statements without walking into other If bodies -- but `try: import
            # optdep / except ...` IS top-level and contributes, so descend into If bodies here.
            for sub in ast.walk(node):
                if isinstance(sub, ast.Import):
                    for alias in sub.names:
                        if alias.name.startswith(package_name):
                            out.add(alias.name)
                elif isinstance(sub, ast.ImportFrom):
                    if sub.level == 0 and sub.module and sub.module.startswith(package_name):
                        out.add(sub.module)
    return out


def _build_graph(root: Path, package_name: str, exclude_dirs: FrozenSet[str]) -> dict:
    """``{module_name: set_of_imported_internal_module_names}``."""
    graph: dict = defaultdict(set)
    for py in root.rglob("*.py"):
        if any(part in exclude_dirs for part in py.parts):
            continue
        tree = _safe_parse(py)
        if tree is None:
            continue
        mod_name = _module_name_from_path(py, root, package_name)
        graph[mod_name].update(_internal_imports(tree, mod_name, package_name))
    return graph


def _strongly_connected_components(graph: dict) -> list:
    """Tarjan's SCC. Returns a list of components (each >= 1 node); cycles are SCCs with > 1 node."""
    index_counter = [0]
    stack: list = []
    lowlinks: dict = {}
    index: dict = {}
    on_stack: dict = {}
    result: list = []

    def strongconnect(v: str) -> None:
        """Tarjan's DFS step: assign ``v`` an index/lowlink and pop its completed SCC off the stack."""
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

    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(10_000)
    try:
        for v in list(graph):
            if v not in index:
                strongconnect(v)
    finally:
        sys.setrecursionlimit(old_limit)
    return result


def scan_import_cycles(
    root: Path,
    exclude_dirs: FrozenSet[str] = _DEFAULT_EXCLUDE_DIRS,
    *,
    package_name: str = "",
    deferred_cycles: FrozenSet[str] = frozenset(),
) -> list[Finding]:
    """Find multi-node cycles in the package's internal top-level import graph.

    A circular import can lurk for years because Python's import machinery resolves cycles at
    module-load if the offending name is accessed lazily -- then a refactor moves a top-level
    access of one of the cycle members and ``ImportError: cannot import name X from partially
    initialized module Y`` ships to users. Builds a dependency graph by AST-walking every .py file
    for top-level ``import``/``from`` statements (lazy imports inside function bodies are
    excluded -- they don't participate in the module-load cycle), restricted to imports whose
    target is inside the package itself, then runs Tarjan's SCC to find any cycle of size > 1.

    ``package_name`` defaults to ``root.name`` when not given (the usual convention: ``root`` is
    the package's own source directory, e.g. ``src/mypackage``, so its directory name IS the
    importable package name).

    ``deferred_cycles``: a set of ``" -> ".join(sorted(component))`` strings for cycles the project
    has reviewed and confirmed benign (e.g. a sibling-file monolith split where both halves
    top-level-import each other but the cycle resolves at runtime because the parent binds its
    exports before importing the sibling) -- these are not flagged. Each project maintains its own
    list; nothing is baked into this scanner.
    """
    pkg = package_name or root.name
    graph = _build_graph(root, pkg, exclude_dirs)
    if not graph:
        return []
    sccs = _strongly_connected_components(graph)
    findings: list[Finding] = []
    for comp in sccs:
        if len(comp) <= 1:
            continue
        comp_key = " -> ".join(sorted(comp))
        if comp_key in deferred_cycles:
            continue
        cycle_display = " -> ".join([*comp, comp[0]])
        findings.append(Finding(
            check="import_cycle",
            severity="P1",
            file=comp[0].replace(f"{pkg}.", "", 1).replace(".", "/") + ".py",
            line=1,
            snippet=cycle_display,
            detail=(
                f"import cycle of {len(comp)} module(s) in the internal import graph: "
                f"{cycle_display}. Resolves at runtime only if every member happens to bind its "
                f"top-level exports before the cycle's back-edge executes -- a later refactor can "
                f"silently break this into `ImportError: cannot import name X from partially "
                f"initialized module Y`."
            ),
        ))
    return findings

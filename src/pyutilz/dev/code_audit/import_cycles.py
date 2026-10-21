"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs.

Shared-gate counterpart: py_ci_shared.import_cycles is the one implementation of the rule, the one that also judges order-dependent
from-imports, TYPE_CHECKING and main guards, and failed-import guards. This module is the pyutilz face of it: it translates the
findings into pyutilz's ``Finding`` type and keeps the cycle-finding API that consumers call. py-ci-shared is a development tool
here, not a runtime dependency, so it is imported only when the scanner runs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import FrozenSet

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _read_src_lines, is_line_exempt

_CHECK = "import_cycle"


def _component_of(key: str) -> str:
    """The `` -> `` joined, sorted member list of a cycle finding, as ``deferred_cycles`` spells it."""
    members = key.split("::", 1)[1].split(" | ")
    return " -> ".join(sorted(members))


def scan_import_cycles(
    root: Path,
    exclude_dirs: FrozenSet[str] = _DEFAULT_EXCLUDE_DIRS,
    *,
    package_name: str = "",
    deferred_cycles: FrozenSet[str] = frozenset(),
) -> list[Finding]:
    """Multi-node import cycles and order-dependent from-imports in the package at *root*.

    ``deferred_cycles`` names cycles the project has accepted for now, each as the `` -> `` joined sorted members.
    """
    from py_ci_shared.import_cycles import find_import_cycles  # py-ci-shared is a dev tool for pyutilz, imported only here

    found, _scan = find_import_cycles(root, package=package_name or None, exclude_parts=tuple(exclude_dirs))
    out: list[Finding] = []
    lines_by_file: dict[str, list[str]] = {}
    for f in found:
        if f.rule == "import-cycle" and _component_of(f.key) in deferred_cycles:
            continue
        if f.path not in lines_by_file:
            lines_by_file[f.path] = _read_src_lines(Path(root) / f.path)
        if is_line_exempt(lines_by_file[f.path], f.line, _CHECK):
            continue
        # The snippet is the cycle itself, as the pyutilz scanner always wrote it; the message adds the module count.
        snippet = f.message.split(": ", 1)[1] if f.rule == "import-cycle" and ": " in f.message else f.message
        out.append(
            Finding(
                check=_CHECK,
                severity="P1" if f.rule == "import-cycle" else "Low",
                file=f.path,
                line=f.line,
                snippet=snippet,
                detail=f.message,
            )
        )
    return out


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

"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs.

Shared-gate counterpart: py_ci_shared.import_cycles is the one implementation of the rule, the one that also judges order-dependent
from-imports, TYPE_CHECKING and main guards, and failed-import guards. This module is the pyutilz face of it: it translates the
findings into pyutilz's ``Finding`` type and keeps the cycle-finding API that consumers call. py-ci-shared is a development tool
here, not a runtime dependency, so it is imported only when the scanner runs.
"""

from __future__ import annotations

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

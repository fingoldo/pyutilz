"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

# --- resource-acquisition calls not wrapped in `with` ----------------------


def _is_resource_call(call: ast.Call) -> bool:
    """Match the resource-acquisition calls this scanner audits:
    * bare ``open(...)``
    * ``tempfile.NamedTemporaryFile(...)`` / ``tempfile.TemporaryFile(...)`` /
      ``tempfile.SpooledTemporaryFile(...)``
    * ``subprocess.Popen(...)``
    """
    func = call.func
    if isinstance(func, ast.Name):
        return func.id == "open"
    if isinstance(func, ast.Attribute):
        if func.attr in {"NamedTemporaryFile", "TemporaryFile", "SpooledTemporaryFile"}:
            return True
        if func.attr == "Popen":
            return True
    return False


def _build_parent_map(tree: ast.AST) -> dict:
    """Return ``{id(child): parent_node}`` for every node in ``tree``."""
    parent_map: dict = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_map[id(child)] = parent
    return parent_map


def _ancestor_chain(node: ast.AST, parent_map: dict) -> list:
    """Return ``node``'s ancestor chain (root last excluded implicitly by the map running out)."""
    chain: list = []
    cur: Optional[ast.AST] = node
    while id(cur) in parent_map:
        cur = parent_map[id(cur)]
        chain.append(cur)
    return chain


def _is_under_with_statement(call: ast.Call, chain: list) -> bool:
    """True if any ancestor of ``call`` is a ``with`` statement and the call appears inside one
    of its ``items`` (the context-manager position)."""
    for ancestor in chain:
        if isinstance(ancestor, ast.With):
            for item in ancestor.items:
                for sub in ast.walk(item.context_expr):
                    if sub is call:
                        return True
    return False


def scan_resource_handle_safety(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find resource-acquisition calls (``open()``/``tempfile.NamedTemporaryFile()``/
    ``tempfile.TemporaryFile()``/``tempfile.SpooledTemporaryFile()``/``subprocess.Popen()``) used
    outside a ``with`` block, i.e. as a bare expression or plain assignment.

    The classic failure modes: ``f = open(path)`` -> the caller forgets ``f.close()``, and on
    Windows the file lock survives until GC, so downstream code that tries to delete/overwrite the
    path fails with ``PermissionError``. ``tmp = tempfile.NamedTemporaryFile()`` -> the temp file
    persists past the function until GC. ``proc = subprocess.Popen(...)`` outside ``with`` ->
    zombie process risk on parent crash.

    False-negative-biased by design (matches this package's other heuristic scanners): a handle
    returned through a ``return`` statement as part of a public API (legitimate) is still flagged,
    since this scanner can't distinguish "caller now owns the handle" from "forgotten cleanup" --
    snapshot/baseline usage is expected to accumulate a few such known-safe sites.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        parent_map = _build_parent_map(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_resource_call(node):
                continue
            chain = _ancestor_chain(node, parent_map)
            if _is_under_with_statement(node, chain):
                continue
            findings.append(Finding(
                check="resource_handle_safety",
                severity="P2",
                file=rel,
                line=node.lineno,
                snippet=_line_text(src_lines, node.lineno),
                detail=(
                    "resource-acquisition call (open()/Popen()/NamedTemporaryFile()) outside a "
                    "`with` block; handle close/cleanup on exception is not guaranteed. Use "
                    "`with open(...) as f:` form, OR this is a confirmed intentional case (e.g. "
                    "handle returned as part of a public API)."
                ),
            ))
    return findings

"""H2 — meta-test that every resource-acquisition call (file open,
tempfile creation, subprocess) is used inside a ``with`` block, NOT
left dangling as a bare expression / assignment.

The classic failure modes:

  * ``f = open(path, encoding="utf-8")`` → user forgets ``f.close()``,
    on Windows the file lock survives until GC, downstream code that
    tries to delete / overwrite the path fails with PermissionError.
  * ``tmp = tempfile.NamedTemporaryFile()`` → temp file persists past
    the function until GC; ``delete=True`` semantics defeat the test.
  * ``proc = subprocess.Popen(...)`` outside ``with`` → zombie process
    on parent crash.

This test walks the AST and flags `open()` / `tempfile.NamedTemporaryFile()`
/ `subprocess.Popen()` calls whose return value is bound to a name OR
discarded, AND whose enclosing statement is not a ``with``-block
context manager.

Snapshot-style — the legacy code has dozens of safe-but-unidiomatic
sites (e.g. file handles passed back through return statements as part
of a public API). New violations fail unless added to the baseline.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

import pyutilz

PYUTILZ_DIR = Path(pyutilz.__file__).resolve().parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_resource_handle_baseline.json"

_EXEMPT_PATH_FRAGMENTS = ("__pycache__", "tests")


def _refresh_requested() -> bool:
    return "--refresh-resource-handle-baseline" in sys.argv


def _is_resource_call(call: ast.Call) -> bool:
    """Match the resource-acquisition calls we audit:
      * bare ``open(...)``
      * ``tempfile.NamedTemporaryFile(...)`` /
        ``tempfile.TemporaryFile(...)`` / ``tempfile.SpooledTemporaryFile(...)``
      * ``subprocess.Popen(...)``
    """
    func = call.func
    if isinstance(func, ast.Name):
        return func.id == "open"
    if isinstance(func, ast.Attribute):
        if func.attr in {"NamedTemporaryFile", "TemporaryFile",
                         "SpooledTemporaryFile"}:
            return True
        if func.attr == "Popen":
            return True
    return False


def _walk_with_parents(tree: ast.AST):
    """Generator yielding (node, parent_node) — for context analysis."""
    parent_map: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent_map[id(child)] = node
    for node in ast.walk(tree):
        yield node, parent_map.get(id(node))


def _is_under_with_statement(call: ast.Call, ancestors_by_id: dict[int, ast.AST],
                             call_to_chain: dict[int, list[ast.AST]]) -> bool:
    """True if any ancestor of ``call`` is a ``with`` statement and the
    call appears inside one of its ``items`` (the context-manager position).
    """
    chain = call_to_chain.get(id(call), [])
    for ancestor in chain:
        if isinstance(ancestor, ast.With):
            for item in ancestor.items:
                # Walk the context-expr subtree of THIS with-item.
                for sub in ast.walk(item.context_expr):
                    if sub is call:
                        return True
    return False


def _build_offending_set() -> set[str]:
    out: set[str] = set()
    for py in PYUTILZ_DIR.rglob("*.py"):
        if any(frag in py.parts for frag in _EXEMPT_PATH_FRAGMENTS):
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
        # Build parent chain map for ancestor lookup.
        parent_map: dict[int, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parent_map[id(child)] = parent
        # For each Call, build its ancestor chain to root.
        chain_map: dict[int, list[ast.AST]] = {}
        rel = py.relative_to(PYUTILZ_DIR).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not _is_resource_call(node):
                continue
            chain: list[ast.AST] = []
            cur: ast.AST | None = node
            while id(cur) in parent_map:
                cur = parent_map[id(cur)]
                chain.append(cur)
            chain_map[id(node)] = chain
            if not _is_under_with_statement(node, parent_map, chain_map):
                out.add(f"{rel}:{node.lineno}")
    return out


def test_no_new_unmanaged_resource_acquisition():
    current = _build_offending_set()

    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(
            json.dumps(sorted(current), indent=2), encoding="utf-8"
        )
        pytest.skip(
            f"resource-handle baseline refreshed at "
            f"{_BASELINE_PATH.name} ({len(current)} bare-acquisition site(s))"
        )

    baseline = set(json.loads(_BASELINE_PATH.read_text(encoding="utf-8")))
    new = sorted(current - baseline)
    fixed = sorted(baseline - current)

    if fixed:
        sys.stderr.write(
            f"\n[test_no_new_unmanaged_resource_acquisition] "
            f"{len(fixed)} call site(s) DRAINED:\n  "
            + "\n  ".join(fixed[:15])
            + (f"\n  ... and {len(fixed) - 15} more" if len(fixed) > 15 else "")
            + "\n  Refresh baseline to lock in.\n"
        )

    if new:
        pytest.fail(
            f"{len(new)} new resource-acquisition call(s) outside a "
            f"``with`` block. ``open()``/``Popen()``/``NamedTemporaryFile()`` "
            f"should be context-managed so handles close on exception. "
            f"Replace with ``with open(...) as f:`` form, OR refresh the "
            f"baseline if intentional (e.g. handle is returned as part of "
            f"the public API):\n  "
            + "\n  ".join(new[:30])
            + (f"\n  ... and {len(new) - 30} more" if len(new) > 30 else "")
        )

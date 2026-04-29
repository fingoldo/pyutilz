"""E1 — meta-test that the set of public symbols WITHOUT a docstring
doesn't grow.

Snapshot-style: the baseline captures the current state on first run.
A future commit that adds a new public ``def`` / ``class`` without a
docstring fails the test. Removing a symbol from the missing-docstring
list (i.e. adding a docstring) is silent (and a refresh prompt is
printed for the user to commit). This lets the codebase grow toward
full coverage without blocking unrelated work on a 100+-item cleanup.

Refresh after intentionally accepting more missing docstrings (rare):

  pytest tests/test_meta/test_public_docstrings.py --refresh-docstring-baseline
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

import pyutilz

PYUTILZ_DIR = Path(pyutilz.__file__).resolve().parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_docstring_baseline.json"

_EXEMPT_FILES = {"version.py", "__init__.py", "__main__.py"}
_EXEMPT_PATH_FRAGMENTS = ("__pycache__", "tests")


def _refresh_requested() -> bool:
    return "--refresh-docstring-baseline" in sys.argv


def _has_docstring(node) -> bool:
    if not getattr(node, "body", None):
        return False
    first = node.body[0]
    return (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    )


def _public_top_level_nodes(path: Path):
    try:
        src = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                yield node


def _build_missing_set() -> set[str]:
    bare: set[str] = set()
    for py in PYUTILZ_DIR.rglob("*.py"):
        if py.name in _EXEMPT_FILES:
            continue
        if any(frag in py.parts for frag in _EXEMPT_PATH_FRAGMENTS):
            continue
        if py.name.endswith(".py.old"):
            continue
        rel = py.relative_to(PYUTILZ_DIR).as_posix()
        for node in _public_top_level_nodes(py):
            if not _has_docstring(node):
                bare.add(f"{rel}::{node.name}")
    return bare


def test_no_new_undocumented_public_symbols():
    current = _build_missing_set()

    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(
            json.dumps(sorted(current), indent=2),
            encoding="utf-8",
        )
        pytest.skip(
            f"docstring baseline refreshed at {_BASELINE_PATH.name} "
            f"({len(current)} undocumented symbols)"
        )

    baseline = set(json.loads(_BASELINE_PATH.read_text(encoding="utf-8")))
    new = sorted(current - baseline)
    fixed = sorted(baseline - current)

    if fixed:
        sys.stderr.write(
            f"\n[test_no_new_undocumented_public_symbols] "
            f"{len(fixed)} symbol(s) DRAINED — they now have docstrings:\n  "
            + "\n  ".join(fixed[:15])
            + (f"\n  ... and {len(fixed) - 15} more" if len(fixed) > 15 else "")
            + f"\n  Refresh baseline to lock in: pytest "
            f"tests/test_meta/test_public_docstrings.py "
            f"--refresh-docstring-baseline\n"
        )

    if new:
        pytest.fail(
            f"{len(new)} new public symbol(s) WITHOUT a docstring — every "
            f"public def/class should have at least a one-line description. "
            f"Add a docstring, or refresh the baseline if intentional:\n  "
            + "\n  ".join(new[:30])
            + (f"\n  ... and {len(new) - 30} more" if len(new) > 30 else "")
        )

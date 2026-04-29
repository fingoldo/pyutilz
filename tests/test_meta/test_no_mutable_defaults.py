"""H4 — meta-test that no production function uses a mutable default
argument value.

The classic Python footgun::

    def append_to(item, lst=[]):
        lst.append(item)
        return lst

The ``lst=[]`` default is created ONCE at function-definition time and
shared across every call that doesn't supply ``lst`` explicitly. The
second caller sees the first caller's mutations.

The safe pattern is the sentinel-then-init dance::

    def append_to(item, lst=None):
        if lst is None:
            lst = []
        lst.append(item)
        return lst

This test flags every function whose parameter default is one of:
  * ``[]`` / ``{}`` / ``set()`` / ``[...literal items...]`` / ``{key: ...}``
  * ``list(...)`` / ``dict(...)`` / ``set(...)`` calls
  * Any other mutable container constructor (``collections.OrderedDict()``,
    ``defaultdict()``, ``Counter()``)

Snapshot-style — first run captures any existing offenders for incremental cleanup.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

import pyutilz

PYUTILZ_DIR = Path(pyutilz.__file__).resolve().parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_mutable_defaults_baseline.json"

_EXEMPT_PATH_FRAGMENTS = ("__pycache__", "tests")

# Constructors known to produce mutable instances. ``frozenset`` and
# ``tuple`` are intentionally absent — those are immutable.
_MUTABLE_CONSTRUCTORS = {
    "list", "dict", "set",
    "OrderedDict", "defaultdict", "Counter", "ChainMap", "deque",
    # NumPy / Pandas: any of these as a default is a code smell.
    "ndarray", "DataFrame", "Series",
}


def _refresh_requested() -> bool:
    return "--refresh-mutable-defaults-baseline" in sys.argv


def _is_mutable_default(default: ast.expr) -> bool:
    """True if ``default`` evaluates to a fresh mutable object."""
    # Empty / non-empty list/dict/set/tuple-of-mutables literals.
    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
        return True
    # Constructor calls: ``list()``, ``dict(...)``, etc.
    if isinstance(default, ast.Call):
        func = default.func
        if isinstance(func, ast.Name) and func.id in _MUTABLE_CONSTRUCTORS:
            return True
        if isinstance(func, ast.Attribute) and func.attr in _MUTABLE_CONSTRUCTORS:
            return True
    return False


def _audit_function(fn: ast.FunctionDef | ast.AsyncFunctionDef,
                    rel: str) -> list[str]:
    out: list[str] = []
    args = fn.args
    # Walk every parameter slot that can have a default.
    all_params = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    # ``args.defaults`` covers posonly + args (right-aligned with the
    # positional list); ``args.kw_defaults`` matches kwonlyargs slot
    # for slot (None => no default).
    n_pos_with_default = len(args.defaults)
    pos_defaults_start = len(args.posonlyargs) + len(args.args) - n_pos_with_default
    for i, default in enumerate(args.defaults):
        param = (list(args.posonlyargs) + list(args.args))[pos_defaults_start + i]
        if _is_mutable_default(default):
            out.append(f"{rel}:{fn.lineno}::{fn.name}({param.arg})")
    for param, default in zip(args.kwonlyargs, args.kw_defaults):
        if default is not None and _is_mutable_default(default):
            out.append(f"{rel}:{fn.lineno}::{fn.name}({param.arg})")
    return out


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
        rel = py.relative_to(PYUTILZ_DIR).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for entry in _audit_function(node, rel):
                    out.add(entry)
    return out


def test_no_new_mutable_default_arguments():
    current = _build_offending_set()

    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(
            json.dumps(sorted(current), indent=2), encoding="utf-8"
        )
        pytest.skip(
            f"mutable-defaults baseline refreshed at {_BASELINE_PATH.name} "
            f"({len(current)} site(s))"
        )

    baseline = set(json.loads(_BASELINE_PATH.read_text(encoding="utf-8")))
    new = sorted(current - baseline)
    fixed = sorted(baseline - current)

    if fixed:
        sys.stderr.write(
            f"\n[test_no_new_mutable_default_arguments] {len(fixed)} site(s) "
            f"DRAINED:\n  " + "\n  ".join(fixed[:15])
            + (f"\n  ... and {len(fixed) - 15} more" if len(fixed) > 15 else "")
            + "\n  Refresh: pytest ... --refresh-mutable-defaults-baseline\n"
        )

    if new:
        pytest.fail(
            f"{len(new)} new function(s) with a mutable default argument. "
            f"Replace with the sentinel pattern: ``def f(x=None): if x is None: "
            f"x = []`` — mutable defaults are shared across all calls and "
            f"silently accumulate state:\n  "
            + "\n  ".join(new[:30])
            + (f"\n  ... and {len(new) - 30} more" if len(new) > 30 else "")
        )

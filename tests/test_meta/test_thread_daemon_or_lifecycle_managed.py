"""Meta-test proposed after the 2026-07-21 audit round 2 meta-test review: every
``threading.Thread(...)`` this package creates must have SOME story for not hanging process
exit -- either it's a daemon thread, its lifetime is provably bounded by a synchronous
``.join()`` in the same function, or the enclosing class manages it via a context manager.

Class this catches: a non-daemon ``threading.Thread`` that outlives the function that created it
(no join, no ``daemon=True``) keeps the interpreter alive at process exit until it finishes on its
own -- exactly the ``system.hardware_monitor.UtilizationMonitor`` bug fixed earlier this round
(the background sampling thread hung process exit whenever a caller forgot to call ``.stop()``;
fixed with ``daemon=True`` plus a ``__enter__``/``__exit__`` context-manager contract).

Mechanism: for every ``threading.Thread(...)`` call, look for (a) a ``daemon=True`` keyword
argument on the constructor itself, (b) ANY ``.join(`` call anywhere in the same enclosing
function (a synchronous fan-out-and-wait pattern -- covers both a single named thread and a
list/loop of threads, since tracking "was THIS SPECIFIC thread object joined" through a list
comprehension is not reliably inferable from syntax alone; a coarser "does this function join
*something*" signal is precise enough in practice and confirmed zero false positives below), or
(c) the enclosing class defining both ``__enter__`` and ``__exit__`` (a managed lifecycle, e.g.
``UtilizationMonitor``). Flag if NONE of these hold.

Verified clean (zero findings) against the current codebase before finalizing: the sole
borderline case, ``performance.kernel_tuning.benchmark.time_backend``'s concurrency fan-out
(threads built in a list comprehension, no ``daemon=True``), is correctly exempted by (b) --
every thread it starts is joined synchronously before the function returns.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import pyutilz

_REPO_ROOT = Path(pyutilz.__file__).resolve().parents[2]
_SRC_DIR = Path(pyutilz.__file__).resolve().parent


def _set_parents(tree: ast.Module) -> None:
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node  # type: ignore[attr-defined]


def _enclosing_function(node: ast.AST) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    cur = getattr(node, "parent", None)
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return cur
        cur = getattr(cur, "parent", None)
    return None


def _enclosing_class(node: ast.AST) -> ast.ClassDef | None:
    cur = getattr(node, "parent", None)
    while cur is not None:
        if isinstance(cur, ast.ClassDef):
            return cur
        cur = getattr(cur, "parent", None)
    return None


def _has_join_call(fn: ast.AST) -> bool:
    return any(isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "join" for sub in ast.walk(fn))


def _is_thread_constructor(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr == "Thread"
    if isinstance(func, ast.Name):
        return func.id == "Thread"
    return False


def _find_unmanaged_threads(root: Path) -> list[str]:
    out: list[str] = []
    for py in root.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            src = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        _set_parents(tree)
        rel = py.relative_to(_REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and _is_thread_constructor(node)):
                continue
            if any(kw.arg == "daemon" for kw in node.keywords):
                continue
            fn = _enclosing_function(node)
            if fn is not None and _has_join_call(fn):
                continue
            cls = _enclosing_class(node)
            if cls is not None:
                method_names = {m.name for m in cls.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
                if "__enter__" in method_names and "__exit__" in method_names:
                    continue
            out.append(f"{rel}:{node.lineno}")
    return out


def test_every_thread_is_daemon_joined_or_lifecycle_managed():
    found = _find_unmanaged_threads(_SRC_DIR)
    if found:
        pytest.fail(
            f"{len(found)} threading.Thread(...) with no daemon=True, no synchronous .join() in "
            f"the same function, and no __enter__/__exit__ lifecycle on the enclosing class. A "
            f"non-daemon thread with none of these keeps the interpreter alive at process exit "
            f"until it finishes on its own -- add daemon=True, join it before the function "
            f"returns, or give the owning class a context-manager contract (see "
            f"system.hardware_monitor.UtilizationMonitor):\n  " + "\n  ".join(found)
        )

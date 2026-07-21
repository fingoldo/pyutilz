"""Meta-test proposed after the 2026-07-21 audit round 2 meta-test review: a module-level global
(or ``self.<attr>``) that's provably resource-holding (``.close()`` is called on it somewhere in
the module/class) must be closed BEFORE being reassigned to a freshly constructed replacement, not
just dropped.

Class this catches: this generalizes ``web.web``'s session-rotation leak fixed earlier this round
-- ``get_new_session()``'s ``sess = requests.Session()`` used to drop the previous
``requests.Session`` (owning an open urllib3 connection pool) with no ``.close()``, leaking
sockets every time a new session was rotated in. Implementing and running this check surfaced one
more real instance beyond the original audit: ``database.redislib.rconnect()`` reassigning the
module-level ``rc`` to a fresh ``redis.Redis`` client without closing the previous one -- now
fixed (mirrors ``web.init_vars()``'s ``old_sess.close()`` pattern).

Mechanism: for each name proven resource-holding (a bare global reassigned via ``global X`` with a
``X.close()`` call somewhere in the SAME module; or a ``self.<attr>`` with a
``self.<attr>.close()`` call somewhere in the SAME class), find every ``X = <Call>(...)``
reassignment to a FRESH constructed value (excluding ``__init__``/``__new__``, which are the
initial construction, not a rotation) and require SOME ``.close()`` call to exist in the same
function body -- proving the old value gets a chance to be released, even if via an intermediate
``old = X`` rename (the common "capture-then-null-out-then-close" idiom).

Deliberately restricted to bare module globals and ``self.<attr>`` (NOT arbitrary local
variables): an earlier, broader design also flagged local-variable reassignments sharing a name
with some OTHER function's ``.close()``-called local (e.g. ``llm.factory._schedule_provider_close``'s
purely-local ``loop = asyncio.get_running_loop()`` got flagged purely because a DIFFERENT function
in the same file, ``_close_cached_providers``, happens to have an unrelated local also named
``loop`` that it does close) -- a false positive from conflating same-named locals across
unrelated functions. Scoping to declared globals / instance attributes (persistent state that
genuinely gets "rotated" over a value's lifetime, unlike a fresh-each-call local) eliminates that
false positive and matches the actual bug class this test exists to catch.
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


def _has_close_call_on_name(fn: ast.AST) -> bool:
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "close" and isinstance(sub.func.value, ast.Name):
            return True
    return False


def _module_scope_findings(tree: ast.Module, rel: str) -> list[str]:
    module_close_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "close":
            recv = node.func.value
            if isinstance(recv, ast.Name):
                module_close_names.add(recv.id)

    out: list[str] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        globals_here: set[str] = set()
        for stmt in ast.walk(fn):
            if isinstance(stmt, ast.Global):
                globals_here.update(stmt.names)
        relevant = globals_here & module_close_names
        if not relevant:
            continue
        for stmt in ast.walk(fn):
            if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)):
                continue
            name = stmt.targets[0].id
            if name not in relevant or not isinstance(stmt.value, ast.Call):
                continue
            if not _has_close_call_on_name(fn):
                out.append(f"{rel}:{stmt.lineno} global {name}")
    return out


def _class_scope_findings(tree: ast.Module, rel: str) -> list[str]:
    out: list[str] = []
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        close_attrs: set[str] = set()
        for node in ast.walk(cls):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "close":
                recv = node.func.value
                if isinstance(recv, ast.Attribute) and isinstance(recv.value, ast.Name) and recv.value.id == "self":
                    close_attrs.add(recv.attr)
        if not close_attrs:
            continue
        for method in cls.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) or method.name in ("__init__", "__new__"):
                continue
            for stmt in ast.walk(method):
                if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
                    continue
                t = stmt.targets[0]
                if not (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self" and t.attr in close_attrs):
                    continue
                if not isinstance(stmt.value, ast.Call):
                    continue
                has_close = any(isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "close" for sub in ast.walk(method))
                if not has_close:
                    out.append(f"{rel}:{stmt.lineno} {cls.name}.self.{t.attr}")
    return out


def _find_leaked_reassignments(root: Path) -> list[str]:
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
        out.extend(_module_scope_findings(tree, rel))
        out.extend(_class_scope_findings(tree, rel))
    return out


def test_no_resource_reassignment_without_close():
    found = _find_leaked_reassignments(_SRC_DIR)
    if found:
        pytest.fail(
            f"{len(found)} resource-holding global/self.attr reassigned to a fresh constructed "
            f"value with no .close() call anywhere in the same function -- the previous value's "
            f"underlying resource (socket, connection pool, file handle) leaks every time this "
            f"runs. Capture the old value before reassigning and close it (see "
            f"web.init_vars()'s old_sess.close() / database.redislib.rconnect()'s "
            f"old_rc.close()):\n  " + "\n  ".join(found)
        )

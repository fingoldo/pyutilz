"""Meta-test proposed after the 2026-07-21 audit round 2 meta-test review: if a module-level
global (or a ``self.<attr>``) is assigned at least once INSIDE a ``with <lock>:`` block, proving
it's meant to be lock-protected, every OTHER assignment to that same name in the same
module/class must also be inside a ``with`` on that lock.

Class this catches: this generalizes the ``web.web`` proxy-state race fixed earlier this round --
``get_url()``'s snapshot read and ``set_proxy()``'s write of ``proxy_user``/``proxy_pass``/etc.
both took ``_state_lock``, but ``init_vars()`` and ``connect()`` (writing the SAME globals) did
not, so a concurrent ``connect()``/``init_vars()`` call could race past the very lock its sibling
functions relied on for consistency. Implementing and running this check surfaced that exact gap
(now fixed: both functions' writes are wrapped in ``with _state_lock:``) -- proving the proactive
premise before this test was even finalized.

Mechanism, two variants:

1. Module scope: find module-level ``name = threading.Lock()/RLock()``. For each function with a
   ``global X`` declaration that also assigns ``X = ...``/``X += ...``, check whether that
   assignment is lexically nested inside a ``with <lock_name>:`` before the enclosing function
   boundary. Group by ``X``; if some sites are protected and others aren't, flag the unprotected
   ones.
2. Class scope: same idea for ``self.<lockattr> = threading.Lock()/RLock()`` and
   ``self.<attr> = ...`` assignments across the class's methods.

FP risk (confirmed empirically before finalizing): a private helper method that mutates the
attribute WITHOUT its own lock, but is only ever called by callers that already hold the lock
(e.g. ``system.config.TomlLiveConfig._reload()``, called from ``__init__`` -- no concurrency
exposure yet -- and from ``_maybe_reload()``, which holds ``self._lock`` for the whole call) is
lexically indistinguishable from a genuine miss without call-graph analysis this AST walk doesn't
do. Baseline-JSON snapshot (same idiom as ``test_no_bare_except.py``), not a hand-curated
whitelist, since new instances of "helper safe only because its caller already holds the lock" are
a legitimate, common pattern -- unlike ``test_no_module_reload.py``'s whitelist, where each new
entry needs a specific, reasoned justification, a drained/refreshed baseline here just means "this
specific site was reviewed and is a call-graph-safe helper, not a lexical miss."
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

import pyutilz

_REPO_ROOT = Path(pyutilz.__file__).resolve().parents[2]
_SRC_DIR = Path(pyutilz.__file__).resolve().parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_lock_discipline_baseline.json"


def _refresh_requested() -> bool:
    return "--refresh-lock-discipline-baseline" in sys.argv


def _set_parents(tree: ast.Module) -> None:
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node  # type: ignore[attr-defined]


def _is_lock_call(value: ast.expr) -> bool:
    if not isinstance(value, ast.Call):
        return False
    func = value.func
    name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
    return name in ("Lock", "RLock")


def _is_inside_with_lock(node: ast.AST, matches_lock, boundary: ast.AST) -> bool:
    cur = getattr(node, "parent", None)
    while cur is not None and cur is not boundary:
        if isinstance(cur, (ast.With, ast.AsyncWith)):
            for item in cur.items:
                if matches_lock(item.context_expr):
                    return True
        cur = getattr(cur, "parent", None)
    return False


def _module_level_lock_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        targets: list[ast.expr] = []
        value = None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = ([node.target] if node.target else []), node.value
        if value is not None and _is_lock_call(value):
            for t in targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
    return names


def _find_module_scope_findings(tree: ast.Module, rel: str) -> list[str]:
    lock_names = _module_level_lock_names(tree)
    if not lock_names:
        return []

    def matcher(expr: ast.expr) -> bool:
        return isinstance(expr, ast.Name) and expr.id in lock_names

    sites: dict[str, list[tuple[int, bool]]] = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        globals_here: set[str] = set()
        for stmt in ast.walk(fn):
            if isinstance(stmt, ast.Global):
                globals_here.update(stmt.names)
        if not globals_here:
            continue
        for stmt in ast.walk(fn):
            target_name = None
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                target_name = stmt.targets[0].id
            elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
                target_name = stmt.target.id
            if target_name is None or target_name not in globals_here:
                continue
            protected = _is_inside_with_lock(stmt, matcher, fn)
            sites.setdefault(target_name, []).append((stmt.lineno, protected))

    out: list[str] = []
    for name, entries in sites.items():
        flags = {p for _, p in entries}
        if True in flags and False in flags:
            for lineno, protected in entries:
                if not protected:
                    out.append(f"{rel}:{lineno} global {name}")
    return out


def _find_class_scope_findings(tree: ast.Module, rel: str) -> list[str]:
    out: list[str] = []
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        lock_attrs: set[str] = set()
        for node in ast.walk(cls):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                t = node.targets[0]
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self" and _is_lock_call(node.value):
                    lock_attrs.add(t.attr)
        if not lock_attrs:
            continue

        def matcher(expr: ast.expr, _lock_attrs: set[str] = lock_attrs) -> bool:
            return isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name) and expr.value.id == "self" and expr.attr in _lock_attrs

        sites: dict[str, list[tuple[int, bool]]] = {}
        for method in cls.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for stmt in ast.walk(method):
                if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
                    continue
                t = stmt.targets[0]
                if not (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self"):
                    continue
                if t.attr in lock_attrs:
                    continue
                protected = _is_inside_with_lock(stmt, matcher, method)
                sites.setdefault(t.attr, []).append((stmt.lineno, protected))

        for attr, entries in sites.items():
            flags = {p for _, p in entries}
            if True in flags and False in flags:
                for lineno, protected in entries:
                    if not protected:
                        out.append(f"{rel}:{lineno} {cls.name}.self.{attr}")
    return out


def _build_findings() -> set[str]:
    out: set[str] = set()
    for py in _SRC_DIR.rglob("*.py"):
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
        out.update(_find_module_scope_findings(tree, rel))
        out.update(_find_class_scope_findings(tree, rel))
    return out


def test_no_new_inconsistently_locked_names():
    current = _build_findings()

    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(json.dumps(sorted(current), indent=2), encoding="utf-8")
        pytest.skip(f"lock-discipline baseline refreshed at {_BASELINE_PATH.name} ({len(current)} finding(s))")

    baseline = set(json.loads(_BASELINE_PATH.read_text(encoding="utf-8")))
    new = sorted(current - baseline)
    fixed = sorted(baseline - current)

    if fixed:
        sys.stderr.write(
            f"\n[test_no_new_inconsistently_locked_names] {len(fixed)} site(s) DRAINED:\n  " + "\n  ".join(fixed[:15]) + (f"\n  ... and {len(fixed) - 15} more" if len(fixed) > 15 else "")
            + "\n  Refresh: pytest ... --refresh-lock-discipline-baseline\n"
        )

    if new:
        pytest.fail(
            f"{len(new)} name(s) with inconsistent lock discipline -- assigned to under a lock "
            f"in at least one place but NOT in another, in the same module/class. Either wrap "
            f"the unprotected site in the same `with <lock>:`, OR if it's a private helper "
            f"safe-by-call-graph (only ever called while the caller already holds the lock, e.g. "
            f"TomlLiveConfig._reload()), refresh the baseline after reviewing:\n  " + "\n  ".join(new[:30]) + (f"\n  ... and {len(new) - 30} more" if len(new) > 30 else "")
        )

"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path

from ._base import _DEFAULT_EXCLUDE_DIRS, Finding, _iter_py_files, _line_text, _safe_parse

# --- one sibling's except handler weaker than its twin's ----------------------------------------
#
# Two methods of one class catch the same exception around the same operation, and one of them
# guards the recovery call while the other does not. The unguarded one then re-raises out of the
# handler that was supposed to contain the failure.
#
# Confirmed instances in one audited codebase:
#
# * `already_in_db` called `self.db.rollback()` bare inside `except psycopg2.Error`, while its
#   sibling `already_in_db_batch` wrapped the identical call in `try/except Exception: pass`. A
#   dropped connection therefore aborted the whole batch from inside the handler meant to survive
#   it.
# * Three checkpoint `unlink()` sites in one file: two swallow `OSError` silently, the third logs
#   at ERROR. Sibling drift inside a single function family.
#
# The rule compares handlers that catch THE SAME exception type and call THE SAME recovery
# function, and reports only where one wraps that call and the other does not. That pairing is what
# makes it specific: two handlers doing genuinely different jobs share neither.

_RECOVERY_HINTS = ("rollback", "close", "unlink", "remove", "release", "reconnect", "cleanup", "flush")


def _handler_exception_names(handler: ast.ExceptHandler) -> frozenset[str]:
    """The exception names this handler catches, as written -- `<bare>` for a bare except."""
    if handler.type is None:
        return frozenset({"<bare>"})
    if isinstance(handler.type, ast.Name):
        return frozenset({handler.type.id})
    if isinstance(handler.type, ast.Attribute):
        return frozenset({handler.type.attr})
    if isinstance(handler.type, ast.Tuple):
        names = set()
        for element in handler.type.elts:
            if isinstance(element, ast.Name):
                names.add(element.id)
            elif isinstance(element, ast.Attribute):
                names.add(element.attr)
        return frozenset(names)
    return frozenset()


def _recovery_calls(handler: ast.ExceptHandler) -> dict[str, bool]:
    """{recovery function name: is it wrapped in its own try} for calls inside this handler."""
    wrapped_ids: set[int] = set()
    for node in ast.walk(handler):
        if isinstance(node, ast.Try):
            for stmt in node.body:
                for sub in ast.walk(stmt):
                    wrapped_ids.add(id(sub))

    calls: dict[str, bool] = {}
    for node in ast.walk(handler):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", None)
        if not name or not any(hint in name.lower() for hint in _RECOVERY_HINTS):
            continue
        # A call seen both wrapped and bare stays recorded as wrapped: the guarded site is what
        # the sibling comparison is about.
        calls[name] = calls.get(name, False) or id(node) in wrapped_ids
    return calls


def scan_asymmetric_except_siblings(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find sibling methods whose identical except handlers guard a recovery call differently.

    Two methods of one class catch the same exception type and both call the same recovery
    function -- `rollback`, `close`, `unlink` -- but only one wraps that call. The unwrapped one
    re-raises out of the handler that exists to contain the failure, which is how a dropped
    connection during rollback aborted an entire batch.

    Both halves of the pairing are required: the same exception type AND the same recovery call.
    Two handlers doing genuinely different jobs share neither, so they are not compared.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()

        for cls in ast.walk(tree):
            if not isinstance(cls, ast.ClassDef):
                continue
            # (exception names, recovery call) -> [(method, wrapped, line)]
            seen: dict[tuple[frozenset[str], str], list[tuple[str, bool, int]]] = {}
            for method in cls.body:
                if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for node in ast.walk(method):
                    if not isinstance(node, ast.ExceptHandler):
                        continue
                    exceptions = _handler_exception_names(node)
                    if not exceptions:
                        continue
                    for call, wrapped in _recovery_calls(node).items():
                        seen.setdefault((exceptions, call), []).append((method.name, wrapped, node.lineno))

            for (exceptions, call), sites in sorted(seen.items(), key=lambda kv: sorted(kv[0][0])):
                if len(sites) < 2:
                    continue
                guarded = [m for m, wrapped, _ in sites if wrapped]
                bare = [(m, line) for m, wrapped, line in sites if not wrapped]
                if not guarded or not bare:
                    continue
                bare_method, bare_line = bare[0]
                findings.append(
                    Finding(
                        check="asymmetric_except_siblings",
                        severity="P2",
                        file=rel,
                        line=bare_line,
                        snippet=_line_text(src_lines, bare_line),
                        detail=(
                            f"`{cls.name}.{bare_method}` calls `{call}` bare inside "
                            f"`except {'/'.join(sorted(exceptions))}`, while its sibling "
                            f"`{guarded[0]}` wraps the identical call. A failure in the recovery "
                            "then escapes the handler that exists to contain it."
                        ),
                    )
                )
    return findings

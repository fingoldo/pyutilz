"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _line_text, _read_src_lines, _safe_parse

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


_CATCH_ALL = frozenset({"<bare>", "Exception", "BaseException"})


def _guards_compatibly(inner: ast.Try, outer_exceptions: frozenset[str]) -> bool:
    """Does this inner ``try`` catch something that would contain the OUTER handler's failure?

    An inner `try/except ValueError` around `rollback()` does not contain the `OSError` the sibling
    comparison is about, so counting any enclosing `try` as "wrapped" silenced genuine asymmetries.
    """
    for handler in inner.handlers:
        caught = _handler_exception_names(handler)
        if caught & _CATCH_ALL or caught & outer_exceptions:
            return True
    return False


def _recovery_calls(handler: ast.ExceptHandler, exceptions: frozenset[str]) -> dict[str, tuple[bool, int]]:
    """{recovery function name: (is it wrapped in a compatible try, line of the call)}."""
    wrapped_ids: set[int] = set()
    for node in ast.walk(handler):
        if isinstance(node, ast.Try) and _guards_compatibly(node, exceptions):
            for stmt in node.body:
                for sub in ast.walk(stmt):
                    wrapped_ids.add(id(sub))

    calls: dict[str, tuple[bool, int]] = {}
    for node in ast.walk(handler):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", None)
        if not name or not any(hint in name.lower() for hint in _RECOVERY_HINTS):
            continue
        # A call seen both wrapped and bare stays recorded as wrapped: the guarded site is what
        # the sibling comparison is about.
        was_wrapped, line = calls.get(name, (False, node.lineno))
        calls[name] = (was_wrapped or id(node) in wrapped_ids, line if was_wrapped else node.lineno)
    return calls


def _methods_of(cls: ast.ClassDef) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Every method of this class, including ones nested in a class-body `if`/`try`.

    Walking only direct children of the class body missed a correctly-reported asymmetric pair the
    moment it sat under `if TYPE_CHECKING:`-style conditional definition.
    """
    out: list[ast.FunctionDef | ast.AsyncFunctionDef] = []

    def descend(statements: list[ast.stmt]) -> None:
        """Collect FunctionDefs from these statements, not entering nested classes or functions."""
        for stmt in statements:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append(stmt)
            elif isinstance(stmt, ast.ClassDef):
                continue
            elif isinstance(stmt, (ast.If, ast.Try, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith)):
                descend(list(stmt.body))
                descend(list(getattr(stmt, "orelse", [])))
                descend(list(getattr(stmt, "finalbody", [])))
                for inner in getattr(stmt, "handlers", []):
                    descend(list(inner.body))

    descend(list(cls.body))
    return out


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
        src_lines = _read_src_lines(py)
        rel = py.relative_to(root).as_posix()

        for cls in ast.walk(tree):
            if not isinstance(cls, ast.ClassDef):
                continue
            # (exception names, recovery call) -> [(method, wrapped, line)]
            seen: dict[tuple[frozenset[str], str], list[tuple[str, bool, int]]] = {}
            for method in _methods_of(cls):
                for node in ast.walk(method):
                    if not isinstance(node, ast.ExceptHandler):
                        continue
                    exceptions = _handler_exception_names(node)
                    if not exceptions:
                        continue
                    for call, (wrapped, call_line) in _recovery_calls(node, exceptions).items():
                        seen.setdefault((exceptions, call), []).append((method.name, wrapped, call_line))

            for (exceptions, call), sites in sorted(seen.items(), key=lambda kv: sorted(kv[0][0])):
                # At least two DISTINCT methods: two handlers inside one method are not siblings,
                # and the finding text ("while its sibling `run` wraps ...") named the method itself.
                if len({name for name, _wrapped, _line in sites}) < 2:
                    continue
                guarded = [m for m, wrapped, _ in sites if wrapped]
                bare = [(m, line) for m, wrapped, line in sites if not wrapped]
                if not guarded or not bare:
                    continue
                bare_method, bare_line = bare[0]
                # The sibling named in the detail has to be a DIFFERENT method.
                sibling = next((name for name in guarded if name != bare_method), None)
                if sibling is None:
                    continue
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
                            f"`{sibling}` wraps the identical call. A failure in the recovery "
                            "then escapes the handler that exists to contain it."
                        ),
                    )
                )
    return findings

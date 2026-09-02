"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from ._base import _DEFAULT_EXCLUDE_DIRS, Finding, _iter_py_files, _line_text, _module_sql_constants, _safe_parse, _sql_text

# --- a SELECT list wider than the rows anyone reads --------------------------------------------
#
# `SELECT id, uid, payload, updated_at FROM ...` unpacked as `for id_, uid, payload in cur:` is a
# ValueError, so that is not the shape. The shape that ships is the quiet one: the query names four
# columns and the row is unpacked into four, one of which is then never referenced -- or the query
# grew a column when a caller stopped needing it and nobody narrowed the SELECT.
#
# It costs on every row: the server materialises the value, the driver decodes it, and it crosses
# the wire. On a table whose `payload` is a JSONB document that is the whole cost of the query.
#
# Detected within ONE function, and only when every piece is unambiguous:
#
# * the SQL is a literal or a module-level constant, with a SELECT list this module can parse --
#   no `*`, no subquery or function call in the list (a `COUNT(*)` or a `CASE` ends the attempt).
# * the row is unpacked by an explicit tuple of plain names with no starred element, so the arity
#   is known exactly.
# * the unused name is not `_`-prefixed, which is how a deliberately-ignored column is spelled.
#
# MEASURED REACH, so silence is not mistaken for coverage. Across pyutilz, mlframe and the
# scraper codebase this rule was designed from, it reports nothing -- and the gating above is not
# why. In the scrapers: 60 SELECTs it can reach, 19 with a plainly parseable column list, 16 in a
# function holding exactly one query, and zero rows unpacked into a flat tuple. That codebase
# reads rows through dict cursors and passes them on, which this rule deliberately declines to
# follow. It stays on by default because it is precise, not because it is productive: on a
# codebase that does unpack rows positionally it fires, and it has never yet produced a false
# positive.
#
# Anything else -- a row passed to another function, indexed dynamically, turned into a dict, or
# unpacked with a star -- is left alone: the column may well be read somewhere this cannot see.

_SELECT = re.compile(r"\bSELECT\b(?P<cols>.+?)\bFROM\b", re.IGNORECASE | re.DOTALL)
_UNPARSEABLE = re.compile(r"[()*]")


def _select_columns(sql: str) -> list[str] | None:
    """The column names a SELECT list names, or None when the list is not plainly parseable."""
    match = _SELECT.search(sql)
    if match is None:
        return None
    cols = match.group("cols").strip()
    # A call, a star or a parenthesised expression means the comma split below would be wrong.
    if _UNPARSEABLE.search(cols):
        return None
    names = []
    for part in cols.split(","):
        token = part.strip().split()[-1] if part.strip() else ""
        token = token.rsplit(".", 1)[-1].strip('"')
        if not token.isidentifier():
            return None
        names.append(token)
    return names or None


def _plain_targets(target: ast.expr) -> list[str] | None:
    """The names this unpacking binds, or None if it is not a flat tuple of plain names."""
    if not isinstance(target, (ast.Tuple, ast.List)):
        return None
    names = []
    for element in target.elts:
        if not isinstance(element, ast.Name):
            return None
        names.append(element.id)
    return names or None


def _read_names(scope: ast.AST, ignore: ast.AST) -> set[str]:
    """Every name LOADED in this scope, skipping the subtree that binds them."""
    skip = set(map(id, ast.walk(ignore)))
    return {node.id for node in ast.walk(scope) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and id(node) not in skip}


def scan_sql_selects_unread_column(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find a SELECT that fetches a column the code unpacking its rows never reads.

    The server materialises the value, the driver decodes it, and it crosses the wire on every
    row -- which on a JSONB payload column is the entire cost of the query. It happens when a
    caller stops needing a column and nobody narrows the SELECT.

    Only unambiguous cases: a parseable SELECT list (no `*`, no calls), an explicit flat tuple
    unpacking with no star, and a name that is not `_`-prefixed. A row passed elsewhere, indexed
    dynamically or turned into a dict is left alone -- the column may be read out of sight.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        constants = _module_sql_constants(tree)
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()

        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # The SELECT lists this function executes, in order.
            executed: list[tuple[list[str], int]] = []
            for node in ast.walk(func):
                if not isinstance(node, ast.Call):
                    continue
                callee = node.func
                if not (isinstance(callee, ast.Attribute) and callee.attr in {"execute", "execute_query", "fetch", "fetchall", "query"}):
                    continue
                if not node.args:
                    continue
                sql = _sql_text(node.args[0], constants)
                columns = _select_columns(sql) if sql else None
                if columns:
                    executed.append((columns, node.lineno))
            if len(executed) != 1:
                # More than one SELECT in a function, and this cannot say which unpacking belongs
                # to which query without following the cursor. Silence beats a coin flip.
                continue
            columns, sql_line = executed[0]

            for node in ast.walk(func):
                if isinstance(node, (ast.For, ast.AsyncFor)):
                    target: ast.expr = node.target
                elif isinstance(node, ast.Assign) and node.targets:
                    target = node.targets[0]
                else:
                    continue
                names = _plain_targets(target)
                if names is None or len(names) != len(columns):
                    continue
                used = _read_names(func, target)
                unread = [(name, column) for name, column in zip(names, columns) if name not in used and not name.startswith("_")]
                if not unread:
                    continue
                fetched = ", ".join(f"`{column}` (bound to `{name}`)" for name, column in unread)
                findings.append(
                    Finding(
                        check="sql_selects_unread_column",
                        severity="P2",
                        file=rel,
                        line=sql_line,
                        snippet=_line_text(src_lines, sql_line),
                        detail=(
                            f"this query fetches {fetched}, which the unpacking at line "
                            f"{node.lineno} binds and nothing in the function reads. The server "
                            "materialises it, the driver decodes it and it crosses the wire on "
                            "every row. Drop it from the SELECT list, or rename the binding to "
                            "`_` to say the waste is deliberate."
                        ),
                    )
                )
    return findings

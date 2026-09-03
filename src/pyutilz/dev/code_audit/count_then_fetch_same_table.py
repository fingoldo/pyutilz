"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _line_text, _module_sql_constants, _read_src_lines, _safe_parse, _sql_table_of, _sql_text

# --- a COUNT round trip that the fetch beside it already answers -------------------------------
#
# `SELECT COUNT(*) FROM jobs WHERE ...` followed by `SELECT id, uid FROM jobs WHERE ...` in the
# same function is two full round trips for one answer: `len(rows)` is the count, exactly, whenever
# the second query is not paginated. On the codebase this rule comes from a round trip measured
# ~310 ms regardless of what the query did, so the statement COUNT was the cost model -- and this
# shape doubles it on a path that runs per scan tick.
#
# It is also a correctness trap when the two run outside one transaction: rows can be inserted
# between them, and the count then describes a table state the fetched rows never had.
#
# Reported only when the pair is unambiguous:
#
# * both queries name the SAME table, matched on the token after FROM.
# * the second is NOT paginated -- a LIMIT or OFFSET means the count genuinely cannot come from
#   the rows, which is the one legitimate reason to ask twice.
# * the count query is a bare COUNT with no GROUP BY, so its answer is a single number rather than
#   a breakdown the fetch does not contain.

_COUNT = re.compile(r"\bCOUNT\s*\(", re.IGNORECASE)
_PAGINATED = re.compile(r"\b(LIMIT|OFFSET|FETCH\s+FIRST)\b", re.IGNORECASE)
_GROUPED = re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE)
_EXECUTORS = frozenset({"execute", "execute_query", "fetch", "fetchall", "fetchone", "query", "scalar"})


def _outer_query(sql: str) -> str:
    """``sql`` with every balanced parenthesised group removed.

    The pagination test has to speak about the OUTER statement only: a `LIMIT` inside a subquery
    (`SELECT * FROM (SELECT id FROM t LIMIT 10) x`) bounds the subquery, not the rows this
    statement returns, so it is no reason for a separate COUNT round trip.
    """
    out: list[str] = []
    depth = 0
    for ch in sql:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def _queries_in(func: ast.AST, constants: dict[str, str]) -> list[tuple[str, int]]:
    """(sql, line) for every query this function executes, in source order."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        callee = node.func
        if not (isinstance(callee, ast.Attribute) and callee.attr in _EXECUTORS):
            continue
        sql = _sql_text(node.args[0], constants)
        if sql and "SELECT" in sql.upper():
            found.append((sql, node.lineno))
    return sorted(found, key=lambda pair: pair[1])


def scan_count_then_fetch_same_table(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find a COUNT query paired with an unpaginated fetch of the same table in one function.

    Two round trips for one answer: `len(rows)` already is the count. On the codebase this rule
    comes from, a round trip cost ~310 ms whatever the query did, so the statement count was the
    cost model outright. It is a correctness trap as well -- run outside a transaction, rows can
    arrive between the two, and the count then describes a state the fetched rows never had.

    Silent when the fetch is paginated (LIMIT/OFFSET), which is the one legitimate reason to ask
    twice, and when the COUNT carries a GROUP BY, whose answer the rows do not contain.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        constants = _module_sql_constants(tree)
        src_lines = _read_src_lines(py)
        rel = py.relative_to(root).as_posix()

        # Nested functions are walked as part of their parent too, so the same site is visited
        # more than once; one finding per (file, count line, fetch line, table).
        reported: set[tuple[str, int, int, str]] = set()
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            queries = _queries_in(func, constants)
            for index, (sql, line) in enumerate(queries):
                if not _COUNT.search(sql) or _GROUPED.search(sql):
                    continue
                table = _sql_table_of(sql)
                if table is None:
                    continue
                # The whole list, not only the suffix: the redundancy and the TOCTOU race are the
                # same when the fetch is written BEFORE the COUNT.
                for other_index, (other_sql, other_line) in enumerate(queries):
                    if other_index == index:
                        continue
                    if _COUNT.search(other_sql) or _sql_table_of(other_sql) != table:
                        continue
                    if _PAGINATED.search(_outer_query(other_sql)):
                        continue
                    key = (rel, line, other_line, table)
                    if key in reported:
                        break
                    reported.add(key)
                    findings.append(
                        Finding(
                            check="count_then_fetch_same_table",
                            severity="P2",
                            file=rel,
                            line=line,
                            snippet=_line_text(src_lines, line),
                            detail=(
                                f"this COUNT over `{table}` is answered by the unpaginated fetch of "
                                f"the same table at line {other_line}: `len(rows)` is the count "
                                "exactly. Two round trips for one answer, and outside a "
                                "transaction the count can describe a state the rows never had."
                            ),
                        )
                    )
                    break
    return findings

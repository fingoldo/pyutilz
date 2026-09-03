"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _line_text, _safe_parse, _sql_table_of, _sql_text

# --- one query over a growing table with no time bound its siblings all have -------------------
#
# Four module-level statements read the same scan table, and three of them end `WHERE ts > now() -
# interval '7 days'`. The fourth does not, and on a table that grows forever it is the one that
# reads every row ever written. It was fast when it was written -- every query on a young table is
# -- and it degrades silently, in proportion to how well the system is working.
#
# This is what the audited codebase's most expensive query looked like: correct, tested, and
# unbounded, on a table at 24.5 million rows. The siblings around it were bounded, which is what
# makes the omission findable at all: the convention is right there in the same module.
#
# The rule needs the family to be unambiguous:
#
# * at least TWO sibling constants in one module read the SAME table, and every one of them but
#   the candidate carries a time bound.
# * the candidate is a SELECT. An INSERT or an UPDATE keyed by primary key is bounded by its key,
#   not by time.
# * the candidate has no LIMIT and no predicate against a caller-supplied parameter either. A
#   bounded fetch is bounded, whatever bounds it -- `WHERE cl_uid IN %s` reads only what the
#   caller named, however large the table gets.
#
# "Carries a time bound" is deliberately broad -- an interval, a NOW()/CURRENT_TIMESTAMP
# comparison, a BETWEEN, or a placeholder compared against a column whose name looks temporal.
# Being wrong in the generous direction here costs a missed finding; being wrong in the strict
# direction reports every query in the module.

_SELECT = re.compile(r"\bSELECT\b", re.I)
_LIMITED = re.compile(r"\b(LIMIT|FETCH\s+FIRST)\b", re.I)
# A predicate against a caller-supplied parameter bounds the scan to whatever the caller passed,
# which is a bound even though it is not a time bound. This rule's only hit on real code was
# `WHERE fj.client_team_uid IN %s` -- a fallback that retries a handful of named clients, and
# reads nothing else however large the table gets.
_KEY_BOUND = re.compile(r"\b[\w.]+\s*(?:=|IN|<@|@>)\s*(?:%\(?\w*\)?s|\$\d+|ANY\s*\(|\?)", re.I)

_TIME_BOUND = re.compile(
    r"\b(INTERVAL\s*'|NOW\s*\(\s*\)|CURRENT_TIMESTAMP|CURRENT_DATE|AGE\s*\(|BETWEEN\b)"
    r"|\b\w*(ts|time|date|_at|seen|updated|created|expires)\w*\s*(>=|<=|>|<)",
    re.I,
)


def _module_sql(tree: ast.Module) -> list[tuple[str, str, int]]:
    """(constant name, sql, line) for every module-level string constant holding a SELECT."""
    found: list[tuple[str, str, int]] = []
    for stmt in tree.body:
        targets = list(stmt.targets) if isinstance(stmt, ast.Assign) else ([stmt.target] if isinstance(stmt, ast.AnnAssign) else [])
        value = stmt.value if isinstance(stmt, (ast.Assign, ast.AnnAssign)) else None
        if value is None:
            continue
        sql = _sql_text(value, {})
        if not sql or not _SELECT.search(sql):
            continue
        found.extend((target.id, sql, stmt.lineno) for target in targets if isinstance(target, ast.Name))
    return found


def scan_sql_sibling_missing_time_bound(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find a query over a growing table whose sibling queries all carry a time bound.

    Three statements over one scan table end `WHERE ts > now() - interval '7 days'` and the fourth
    does not. On a table that grows forever, the fourth reads every row ever written -- and it
    degrades in proportion to how well the system is working, which is why nobody notices. That is
    what an audited codebase's most expensive query looked like at 24.5 million rows.

    The convention has to be visible in the same module: at least two siblings reading the same
    table, all bounded but the candidate. A LIMIT counts as a bound, and only SELECTs are
    considered -- an INSERT or a keyed UPDATE is bounded by its key.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        statements = _module_sql(tree)
        if len(statements) < 3:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()

        by_table: dict[str, list[tuple[str, str, int]]] = {}
        for name, sql, line in statements:
            table = _sql_table_of(sql)
            if table:
                by_table.setdefault(table, []).append((name, sql, line))

        for table, family in sorted(by_table.items()):
            bounded = [item for item in family if _TIME_BOUND.search(item[1]) or _LIMITED.search(item[1]) or _KEY_BOUND.search(item[1])]
            unbounded = [item for item in family if item not in bounded]
            if len(bounded) < 2 or len(unbounded) != 1:
                continue
            name, _sql, line = unbounded[0]
            findings.append(
                Finding(
                    check="sql_sibling_missing_time_bound",
                    severity="P2",
                    file=rel,
                    line=line,
                    snippet=_line_text(src_lines, line),
                    detail=(
                        f"`{name}` reads `{table}` with no time bound, no LIMIT and no key "
                        f"predicate, while "
                        f"{len(bounded)} sibling statements over the same table in this module all "
                        "carry one. On a table that grows forever this reads every row ever "
                        "written, and it degrades in proportion to how well the system is working "
                        "-- which is why it is never noticed at the time."
                    ),
                )
            )
    return findings

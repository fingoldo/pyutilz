"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
import re
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

# --- SQL-literal heuristics -----------------------------------------------

# A string literal "looks like SQL" if it contains a standalone SELECT
# keyword. Deliberately narrow (one keyword, word-boundaried) to keep the
# false-positive rate on unrelated string literals low.
_SELECT_RE = re.compile(r"\bSELECT\b", re.IGNORECASE)

# ``LIMIT`` followed by either a bind-param placeholder (``:name`` / ``%s`` /
# ``$1`` / ``?``) or an integer literal. Captures the literal value (if any)
# so ``LIMIT 1`` can be exempted as the ubiquitous "fetch one row" idiom.
_LIMIT_RE = re.compile(r"\bLIMIT\s+(?::\w+|%\(?\w*\)?s|\$\d+|\?|(\d+))", re.IGNORECASE)
_ORDER_BY_RE = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)
_OFFSET_RE = re.compile(r"\bOFFSET\b", re.IGNORECASE)

# --- MIN()/MAX() over a raw JSON-text extraction, before casting ----------

_MIN_MAX_CALL_RE = re.compile(r"\b(MIN|MAX)\s*\(", re.IGNORECASE)
# Postgres/MySQL/SQLite JSON text-extraction operators: ``->>'key'`` (Postgres,
# text result) and ``->'key'`` (Postgres, jsonb result -- still needs a cast
# to compare/aggregate as anything but jsonb).
_JSON_EXTRACT_RE = re.compile(r"->>?\s*'")
_CAST_RE = re.compile(r"::\s*\w+")


def _matching_close_paren(text: str, open_paren_idx: int) -> int:
    """Index of the ``)`` that closes the ``(`` at ``open_paren_idx``
    (balanced-paren scan, so nested calls inside the aggregate's argument
    don't confuse it), or -1 if unbalanced."""
    depth = 0
    for i in range(open_paren_idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def scan_sql_aggregate_before_cast(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find ``MIN()``/``MAX()`` wrapping a raw JSON-text extraction
    (``col->>'key'``) with no ``::type`` cast anywhere in the aggregate's
    argument.

    Aggregating (or comparing) the *text* representation of a JSON value
    sorts lexicographically, not by the value's real type -- correct only by
    coincidence when the source data happens to already be in a
    lexicographically-sortable format (e.g. ISO-8601 timestamps), an
    unstated invariant that silently breaks the moment the upstream producer
    changes format or a differently-formatted row slips in. Cast inside the
    aggregate's own parens (``MIN((col->>'key')::timestamptz)``) so the
    aggregate operates on the real type.

    Severity: P2 (silently wrong ordering/selection under a data-shape
    assumption that isn't enforced anywhere, not an immediate crash).
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        for node, sql in _string_constants(tree):
            for m in _MIN_MAX_CALL_RE.finditer(sql):
                open_idx = m.end() - 1
                close_idx = _matching_close_paren(sql, open_idx)
                if close_idx == -1:
                    continue
                arg = sql[open_idx + 1 : close_idx]
                if not _JSON_EXTRACT_RE.search(arg):
                    continue
                if _CAST_RE.search(arg):
                    continue
                findings.append(Finding(
                    check="sql_aggregate_before_cast",
                    severity="P2",
                    file=rel,
                    line=node.lineno,
                    snippet=_line_text(src_lines, node.lineno),
                    detail=(
                        f"{m.group(1).upper()}() wraps a raw JSON-text extraction "
                        "(->>'...' / ->'...') with no ::type cast anywhere in its "
                        "argument -- aggregates/compares the TEXT representation, "
                        "which sorts lexicographically rather than by the real "
                        "value's type. Cast inside the aggregate's own parens, "
                        "e.g. MIN((col->>'key')::timestamptz)."
                    ),
                ))
    return findings


def _string_constants(tree: ast.Module) -> list[tuple[ast.Constant, str]]:
    """Every ``ast.Constant`` string literal in the module, in traversal
    order, paired with its string value (mypy can't narrow
    ``Constant.value``'s union type through an isinstance filter inside
    a comprehension, so the str is captured alongside the node)."""
    out: list[tuple[ast.Constant, str]] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            out.append((n, n.value))  # noqa: PERF401 -- a comprehension loses mypy's isinstance narrowing here
    return out


def scan_sql_limit_without_order_by(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find SQL string literals with a ``LIMIT`` clause but no ``ORDER BY``.

    Without an explicit ordering, which rows survive the cap is whatever
    physical/plan order the database happens to return that run --
    arbitrary and non-reproducible, and any code that assumes a
    "top N by some criterion" semantic is silently wrong. ``LIMIT 1``
    (the ubiquitous "does at least one row exist" / "fetch a single
    row" idiom) is exempted -- flagging every occurrence of that pattern
    would be pure noise.

    Severity: P2 (silently non-deterministic results, not a crash).
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        for node, sql in _string_constants(tree):
            if not _SELECT_RE.search(sql):
                continue
            limit_match = _LIMIT_RE.search(sql)
            if not limit_match:
                continue
            if limit_match.group(1) == "1":
                continue
            if _ORDER_BY_RE.search(sql):
                continue
            findings.append(Finding(
                check="sql_limit_without_order_by",
                severity="P2",
                file=rel,
                line=node.lineno,
                snippet=_line_text(src_lines, node.lineno),
                detail=(
                    "SQL literal has a LIMIT clause with no ORDER BY -- which "
                    "rows survive the cap is arbitrary DB physical order, not "
                    "reproducible across runs. Add an ORDER BY matching the "
                    "intended selection criterion, or LIMIT 1 if any single "
                    "row is acceptable."
                ),
            ))
    return findings


def scan_sql_offset_pagination(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find SQL string literals combining ``LIMIT`` and ``OFFSET``.

    OFFSET pagination is only correct when the filtered result set is
    stable across batches. A common bug: the same query loop both SELECTs
    rows via ``WHERE some_flag IS NULL`` and UPDATEs that flag for
    processed rows -- each batch's UPDATE removes rows from the very set
    OFFSET is walking, so the next OFFSET skips over rows that shifted
    into the just-vacated positions. Keyset pagination (``WHERE id > :last_id
    ORDER BY id LIMIT :n``) is immune to this because it never depends on
    the filtered set's size staying constant between batches.

    This is advisory only -- OFFSET pagination over a genuinely static
    result set (no concurrent mutation of the filter columns) is fine.
    Severity: Low.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        for node, sql in _string_constants(tree):
            if not _SELECT_RE.search(sql):
                continue
            if not (_LIMIT_RE.search(sql) and _OFFSET_RE.search(sql)):
                continue
            findings.append(Finding(
                check="sql_offset_pagination",
                severity="Low",
                file=rel,
                line=node.lineno,
                snippet=_line_text(src_lines, node.lineno),
                detail=(
                    "SQL literal paginates via LIMIT+OFFSET. Verify the "
                    "filtered result set is stable across batches -- if this "
                    "query's own loop also mutates a column referenced in "
                    "its WHERE clause between batches, OFFSET will silently "
                    "skip rows. Prefer keyset pagination (WHERE id > "
                    ":last_id ORDER BY id LIMIT :n) when in doubt."
                ),
            ))
    return findings

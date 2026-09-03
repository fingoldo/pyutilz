"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import re
from pathlib import Path

from ._base import _DEFAULT_EXCLUDE_DIRS, Finding, _iter_py_files, _line_text

# --- a column the schema declares that no statement anywhere can fill --------------------------
#
# A telemetry column is added to a table, the dashboard learns to read it, and no INSERT or UPDATE
# in the codebase ever names it. It reads NULL forever, and a NULL in a telemetry column is
# indistinguishable from "this never happened" -- so the panel shows a flat line and everyone
# believes the thing being measured does not occur.
#
# The audited codebase had several of these, and the reason is always the same: the DDL lands in a
# migration, the writing code is a separate change, and one of the two is forgotten. Nothing fails
# and nothing is logged, because writing fewer columns than a table has is legal.
#
# Excluded, because each fills itself and an absent write is correct:
#
# * a column with a DEFAULT, including `generated always` and `serial`/`identity` -- the database
#   is the writing path.
# * a primary key, which is either supplied by a default or by a conflict target this rule cannot
#   see through.
# * a column named anywhere in a `SELECT`, because a table this codebase only reads from is not
#   something it is expected to write.
#
# MEASURED REACH, so silence is not read as coverage. On the scraper codebase this comes from, it
# parses 20 tables and 85 columns that need a writing path of their own, and reports none: every
# one of the 85 is either written somewhere or read nowhere. That is the answer for that tree, not
# a gap -- the worked example in the tests is a column with exactly this defect and it is found.
#
# The search is textual and deliberately generous: any mention of the column name inside a
# statement that writes -- an INSERT column list, a SET clause, an `EXCLUDED.col` in an upsert, a
# parameter dict key -- counts as a write path. A rule that demanded a parsed INSERT would miss
# every ORM and every string-built statement, which is most of them.

_CREATE_TABLE = re.compile(r"CREATE\s+(?:UNLOGGED\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?P<name>[\w.\"]+)\s*\((?P<body>.*?)\n\s*\)\s*;", re.I | re.S)
_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_COLUMN = re.compile(r"^\s*(?P<name>[a-z_][\w]*)\s+(?P<rest>.+)$", re.I)
_SELF_FILLING = re.compile(r"\b(DEFAULT|GENERATED|SERIAL|BIGSERIAL|SMALLSERIAL|IDENTITY)\b", re.I)
_NOT_A_COLUMN = re.compile(r"^\s*(PRIMARY|FOREIGN|UNIQUE|CHECK|CONSTRAINT|EXCLUDE|LIKE|INHERITS)\b", re.I)
_PRIMARY_KEY = re.compile(r"\bPRIMARY\s+KEY\b", re.I)
_WRITES = re.compile(r"\b(INSERT\s+INTO|UPDATE\s+\w|SET\s+\w|DO\s+UPDATE|EXCLUDED\.)", re.I)


def _strip_comments(text: str) -> str:
    """SQL with comments removed -- a commented-out column is not a declaration."""
    return _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", text))


def _declared_columns(sql: str) -> dict[str, list[tuple[str, int]]]:
    """`{table: [(column, line)]}` for columns that need a writing path of their own."""
    tables: dict[str, list[tuple[str, int]]] = {}
    for match in _CREATE_TABLE.finditer(sql):
        table = match.group("name").strip('"').rpartition(".")[2].lower()
        body_start = sql.count("\n", 0, match.start("body"))
        for offset, line in enumerate(match.group("body").splitlines()):
            if _NOT_A_COLUMN.match(line) or _SELF_FILLING.search(line) or _PRIMARY_KEY.search(line):
                continue
            column = _COLUMN.match(line)
            if column:
                tables.setdefault(table, []).append((column.group("name").lower(), body_start + offset + 1))
    return tables


def _written_names(texts: list[str]) -> set[str]:
    """Every word appearing in a statement that writes, across the whole tree.

    Textual and generous on purpose: an INSERT column list, a SET clause, an `EXCLUDED.col` and a
    parameter dict key all count. Demanding a parsed INSERT would miss every ORM and every
    string-built statement, which between them are most of the writes in any codebase.
    """
    written: set[str] = set()
    for text in texts:
        for chunk in re.split(r";|\n\s*\n", text):
            if _WRITES.search(chunk):
                written.update(word.lower() for word in re.findall(r"[A-Za-z_][\w]*", chunk))
    return written


def scan_column_no_write_path(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
    sql_dir: str = "sql",
) -> list[Finding]:
    """Find a declared column that no INSERT or UPDATE in the tree can fill.

    It reads NULL forever, and a NULL in a telemetry column cannot be told apart from "this never
    happened" -- so the panel built on it shows a flat line and everyone believes the thing being
    measured does not occur. It happens because the DDL lands in one change and the writing code
    in another, and nothing fails when a table has more columns than a statement names.

    Columns that fill themselves (DEFAULT, GENERATED, serial, identity, primary keys) are skipped,
    as is any column named in a SELECT -- a table this codebase only reads is not one it is
    expected to write.
    """
    sql_root = root / sql_dir
    if not sql_root.is_dir():
        return []

    declarations: dict[str, list[tuple[str, int, str]]] = {}
    sql_texts: list[str] = []
    for path in sorted(sql_root.rglob("*.sql")):
        text = _strip_comments(path.read_text(encoding="utf-8", errors="replace"))
        sql_texts.append(text)
        for table, columns in _declared_columns(text).items():
            for column, line in columns:
                declarations.setdefault(table, []).append((column, line, path.relative_to(root).as_posix()))

    if not declarations:
        return []

    corpus = list(sql_texts)
    read_names: set[str] = set()
    for py in _iter_py_files(root, exclude_dirs):
        text = py.read_text(encoding="utf-8", errors="replace")
        corpus.append(text)
        for chunk in re.split(r";|\n\s*\n", text):
            if re.search(r"\bSELECT\b", chunk, re.I):
                read_names.update(word.lower() for word in re.findall(r"[A-Za-z_][\w]*", chunk))
    for text in sql_texts:
        for chunk in re.split(r";|\n\s*\n", text):
            if re.search(r"\bSELECT\b", chunk, re.I):
                read_names.update(word.lower() for word in re.findall(r"[A-Za-z_][\w]*", chunk))

    written = _written_names(corpus)
    findings: list[Finding] = []
    for table, sites in sorted(declarations.items()):
        for column, line, rel in sorted(set(sites)):
            if column in written or column not in read_names:
                continue
            src_lines = (root / rel).read_text(encoding="utf-8", errors="replace").splitlines()
            findings.append(
                Finding(
                    check="column_no_write_path",
                    severity="P1",
                    file=rel,
                    line=line,
                    snippet=_line_text(src_lines, line),
                    detail=(
                        f"`{table}.{column}` is declared with no default and no statement anywhere "
                        "in this tree writes it, while something reads it. It is NULL forever, and "
                        "in a telemetry column NULL cannot be told apart from `this never "
                        "happened` -- the panel built on it shows a flat line. Either add the "
                        "write, or drop the column."
                    ),
                )
            )
    return findings

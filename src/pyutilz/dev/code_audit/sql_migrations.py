"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import re
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS

# --- non-idempotent SQL migrations ------------------------------------------

# Statements that unconditionally fail on a second run because the guarded
# form (``IF EXISTS`` / ``IF NOT EXISTS``) is missing. Each entry is
# (compiled pattern matching the UNGUARDED form, human description).
_UNGUARDED_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r"\bDROP\s+CONSTRAINT\s+(?!IF\s+EXISTS\b)", re.IGNORECASE),
        "DROP CONSTRAINT without IF EXISTS -- fails with " '"constraint ... does not exist" on a second run once the ' "constraint has already been dropped.",
    ),
    (
        re.compile(r"\bDROP\s+COLUMN\s+(?!IF\s+EXISTS\b)", re.IGNORECASE),
        "DROP COLUMN without IF EXISTS -- fails on a second run once " "the column is already gone.",
    ),
    (
        re.compile(r"\bDROP\s+TABLE\s+(?!IF\s+EXISTS\b)", re.IGNORECASE),
        "DROP TABLE without IF EXISTS -- fails on a second run once " "the table is already gone.",
    ),
    (
        re.compile(r"\bADD\s+COLUMN\s+(?!IF\s+NOT\s+EXISTS\b)", re.IGNORECASE),
        "ADD COLUMN without IF NOT EXISTS -- fails with a duplicate-column " "error on a second run.",
    ),
]

# ``ADD PRIMARY KEY`` / bare ``ADD CONSTRAINT`` have no ``IF NOT EXISTS``
# form in standard SQL at all -- the idiomatic fix is a ``DO $$ ... END $$``
# guard block that checks the current state first, not a keyword. Flagged
# separately (advisory, not "add this keyword").
_UNGUARDED_ADD_PK_RE = re.compile(r"\bADD\s+PRIMARY\s+KEY\b", re.IGNORECASE)
_DO_BLOCK_RE = re.compile(r"\bDO\s*\$\$", re.IGNORECASE)

# A statement inside a ``DO $$ ... END $$`` block that itself contains an
# ``IF [NOT] EXISTS (...) THEN`` existence check is already conditional at
# the block level, even though the guarded statement's own line has no
# ``IF EXISTS``/``IF NOT EXISTS`` keyword -- the classic
# information_schema-probe idempotency idiom. Matches the whole block
# (DOTALL) so a multi-line ``IF NOT EXISTS (\n  SELECT ...\n)`` is found.
_DO_BLOCK_SPAN_RE = re.compile(r"\bDO\s*\$\$(.*?)END\s*\$\$", re.IGNORECASE | re.DOTALL)
_EXISTENCE_GUARD_RE = re.compile(r"\bIF\s+(?:NOT\s+)?EXISTS\s*\(", re.IGNORECASE)


def _existence_guarded_line_ranges(text: str) -> list[tuple[int, int]]:
    """Line ranges (1-indexed, inclusive) of DO-blocks that contain an
    ``IF [NOT] EXISTS (...)`` existence check anywhere in the block --
    any statement on a line within such a range is conditionally guarded
    at the block level, regardless of its own keyword form."""
    ranges = []
    for m in _DO_BLOCK_SPAN_RE.finditer(text):
        if not _EXISTENCE_GUARD_RE.search(m.group(1)):
            continue
        start_line = text.count("\n", 0, m.start()) + 1
        end_line = text.count("\n", 0, m.end()) + 1
        ranges.append((start_line, end_line))
    return ranges


def scan_sql_migration_idempotency(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
    sql_glob: str = "**/*.sql",
) -> list[Finding]:
    """Find SQL migration statements that fail (rather than no-op) when
    the migration is re-run against a database where it already applied.

    A migration re-run is not a hypothetical: deploy retries, a manual
    re-apply after a partial failure, and multi-environment rollouts
    (dev/staging/prod applying the same file at different times) are all
    ordinary operational events. Each finding here corresponds to a
    statement that raises an error on the SECOND run instead of being a
    no-op, turning a routine re-run into an incident.

    Operates on raw ``.sql`` files (line-based regex, not Python AST --
    SQL is not parseable by the ``ast`` module). ``sql_glob`` defaults to
    every ``.sql`` file under ``root``; narrow it (e.g.
    ``"migrations/*.sql"``) to scope to a migrations directory only and
    skip a full schema-definition file that's expected to run once on an
    empty database.

    Severity: P1 (a routine re-run aborts the whole migration batch).
    """
    findings: list[Finding] = []
    excluded = set(exclude_dirs)
    for sql_path in root.glob(sql_glob):
        if any(part in excluded for part in sql_path.parts):
            continue
        try:
            text = sql_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = text.splitlines()
        rel = sql_path.relative_to(root).as_posix()
        has_do_block = bool(_DO_BLOCK_RE.search(text))
        guarded_ranges = _existence_guarded_line_ranges(text)
        for lineno, line in enumerate(lines, start=1):
            if any(start <= lineno <= end for start, end in guarded_ranges):
                continue
            for pattern, description in _UNGUARDED_PATTERNS:
                if pattern.search(line):
                    findings.append(
                        Finding(
                            check="sql_migration_not_idempotent",
                            severity="P1",
                            file=rel,
                            line=lineno,
                            snippet=line.strip(),
                            detail=description,
                        )
                    )
            if _UNGUARDED_ADD_PK_RE.search(line) and not has_do_block:
                findings.append(
                    Finding(
                        check="sql_migration_not_idempotent",
                        severity="P1",
                        file=rel,
                        line=lineno,
                        snippet=line.strip(),
                        detail=(
                            "ADD PRIMARY KEY has no IF NOT EXISTS form in "
                            "standard SQL and this file has no DO $$ ... END $$ "
                            "guard block -- wrap the ADD in a DO block that "
                            "checks the current primary-key columns first "
                            "(see pg_index / information_schema) before adding."
                        ),
                    )
                )
    return findings

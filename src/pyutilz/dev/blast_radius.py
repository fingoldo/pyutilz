"""Blast-radius check: one write changes exactly the rows and columns it should, and nothing it should not.

WHY
---
glossum's verdict savers shipped five data defects that a green suite of 14,000+ tests and every
persistence gate missed, because each gate asked whether a value ARRIVED, never what ELSE changed:

* a WHERE clause ``a AND b AND c OR d`` wrote one verdict onto every sibling row (AND binds first);
* an UPDATE matched a lemma in every synset instead of the one being judged;
* a stored ``False`` was set back to NULL a few lines later;
* a fix wrote the verdict over the producer's own columns, erasing the note the producer left.

Fake sessions returned "1 row" for any SQL, and the runtime sweeps seeded one row per class, so there was no
sibling to overwrite. The only check that sees this class runs the real writer with neighbouring rows
present and compares the whole store before and after.

WHAT
----
Take a snapshot, run the writer, take another, and assert on the diff:

* :func:`sql_snapshot` (async, any SQLAlchemy-style session) or :func:`records_snapshot` (records keyed by a
  field, for JSONL/JSON stores) -- ``{table: {key: {column: value}}}``.
* :func:`diff_snapshots` -- every changed, added and removed row, with the columns that changed.
* :func:`blast_radius_violations` / :func:`assert_only_target_changed` -- fail when a row outside ``targets``
  changed, when a column outside ``allowed_columns`` changed on a target, or when a column that held a value
  became NULL (unless listed in ``may_clear``). A wider write is declared, never hidden.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from pyutilz.dev.persistence_sweep import _quote

__all__ = [
    "Change",
    "assert_only_target_changed",
    "blast_radius_violations",
    "diff_snapshots",
    "records_snapshot",
    "sql_snapshot",
]

Key = Tuple[Any, ...]
Snapshot = Dict[str, Dict[Key, Dict[str, Any]]]


async def sql_snapshot(session: Any, tables: Mapping[str, Sequence[str]], *, where: Optional[Mapping[str, str]] = None) -> Snapshot:
    """``{table: {key: row}}`` for each ``"schema.table"`` keyed by its given key columns.

    ``where`` narrows a large table to the rows a test can touch (a ``WHERE`` fragment with no user input);
    keep it wide enough to include every sibling the writer could hit, or the check has nothing to see.
    """
    from sqlalchemy import text

    out: Snapshot = {}
    for table, key_columns in tables.items():
        qualified = ".".join(_quote(part) for part in table.split("."))
        clause = f" WHERE {where[table]}" if where and table in where else ""
        result = await session.execute(text(f"SELECT * FROM {qualified}{clause}"))  # nosec B608 -- identifiers quoted, fragment is the test's own
        rows = result.mappings().all()
        out[table] = {tuple(row[c] for c in key_columns): dict(row) for row in rows}
    return out


def records_snapshot(tables: Mapping[str, Iterable[Mapping[str, Any]]], key_fields: Mapping[str, Sequence[str]]) -> Snapshot:
    """The file-store twin of :func:`sql_snapshot`: ``{table: {key: record}}`` from already-read records."""
    out: Snapshot = {}
    for table, records in tables.items():
        keys = key_fields[table]
        out[table] = {tuple(r.get(k) for k in keys): dict(r) for r in records}
    return out


class Change:
    """One row that differs between two snapshots."""

    __slots__ = ("after", "before", "columns", "key", "kind", "table")

    def __init__(self, table: str, key: Key, kind: str, columns: Set[str], before: Optional[Dict[str, Any]], after: Optional[Dict[str, Any]]) -> None:
        """One row: its table, key, kind of change, the changed columns, and both versions."""
        self.table, self.key, self.kind, self.columns, self.before, self.after = table, key, kind, columns, before, after

    def __repr__(self) -> str:
        """The change with each changed column's before and after values."""
        detail = {c: ((self.before or {}).get(c), (self.after or {}).get(c)) for c in sorted(self.columns)}
        return f"Change({self.table} {self.key} {self.kind} {detail})"


def diff_snapshots(before: Snapshot, after: Snapshot, *, ignore_columns: Iterable[str] = ()) -> List[Change]:
    """Every row changed, added or removed between the two snapshots, ignoring ``ignore_columns``."""
    ignored = set(ignore_columns)
    changes: List[Change] = []
    for table in sorted(set(before) | set(after)):
        old, new = before.get(table, {}), after.get(table, {})
        for key in sorted(set(old) | set(new), key=repr):
            a, b = old.get(key), new.get(key)
            if a is None or b is None:
                present = b if a is None else a
                cols = set(present or {}) - ignored
                changes.append(Change(table, key, "added" if a is None else "removed", cols, a, b))
                continue
            cols = {c for c in set(a) | set(b) if c not in ignored and a.get(c) != b.get(c)}
            if cols:
                changes.append(Change(table, key, "changed", cols, a, b))
    return changes


def blast_radius_violations(
    changes: Sequence[Change],
    *,
    targets: Mapping[str, Iterable[Key]],
    allowed_columns: Mapping[str, Iterable[str]],
    may_clear: Optional[Mapping[str, Iterable[str]]] = None,
    may_add: Iterable[str] = (),
    may_remove: Iterable[str] = (),
) -> List[str]:
    """Why each change is outside the write's declared reach; empty when it stayed inside it.

    ``targets``: the rows the write is meant to touch, per table. ``allowed_columns``: the columns it may set
    on them. ``may_clear``: columns it may set to NULL (a real "clear" is declared, never assumed).
    ``may_add`` / ``may_remove``: tables where it may insert or delete rows.
    """
    target_keys = {t: set(keys) for t, keys in targets.items()}
    allowed = {t: set(cols) for t, cols in allowed_columns.items()}
    clearable = {t: set(cols) for t, cols in (may_clear if may_clear is not None else {}).items()}
    add_ok, remove_ok = set(may_add), set(may_remove)
    problems: List[str] = []
    for ch in changes:
        if ch.kind == "added":
            if ch.table not in add_ok:
                problems.append(f"{ch.table} {ch.key}: a row was inserted where the write declared none")
            continue
        if ch.kind == "removed":
            if ch.table not in remove_ok:
                problems.append(f"{ch.table} {ch.key}: a row was deleted where the write declared none")
            continue
        if ch.key not in target_keys.get(ch.table, set()):
            problems.append(f"{ch.table} {ch.key}: a row the write did not target changed {sorted(ch.columns)}")
            continue
        outside = ch.columns - allowed.get(ch.table, set())
        if outside:
            problems.append(f"{ch.table} {ch.key}: columns outside the write's declared set changed {sorted(outside)}")
        cleared = {c for c in ch.columns if (ch.before or {}).get(c) is not None and (ch.after or {}).get(c) is None}
        cleared -= clearable.get(ch.table, set())
        if cleared:
            problems.append(f"{ch.table} {ch.key}: values that were set became NULL {sorted(cleared)}")
    return problems


def assert_only_target_changed(
    before: Snapshot,
    after: Snapshot,
    *,
    targets: Mapping[str, Iterable[Key]],
    allowed_columns: Mapping[str, Iterable[str]],
    ignore_columns: Iterable[str] = ("updated_at",),
    may_clear: Optional[Mapping[str, Iterable[str]]] = None,
    may_add: Iterable[str] = (),
    may_remove: Iterable[str] = (),
    require_change: bool = True,
) -> List[Change]:
    """Raise ``AssertionError`` naming every out-of-reach change; return the changes otherwise.

    ``require_change`` also fails when nothing changed at all, so a writer that silently matched no row
    cannot pass by having done nothing.
    """
    changes = diff_snapshots(before, after, ignore_columns=ignore_columns)
    problems = blast_radius_violations(changes, targets=targets, allowed_columns=allowed_columns, may_clear=may_clear, may_add=may_add, may_remove=may_remove)
    if require_change and not changes:
        problems.append("the write changed nothing at all")
    if problems:
        raise AssertionError("the write reached beyond what it declared:\n  " + "\n  ".join(problems))
    return changes

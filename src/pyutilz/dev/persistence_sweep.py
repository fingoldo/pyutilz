"""Runtime sentinel sweep: prove that every field handed to a saver reaches the database.

Static checks can show that a column exists and that a field's name appears in some saver. They cannot
show that THIS code path writes it: a value persisted by one branch (one language, one part of speech)
and silently dropped by another looks fully covered to any name-based analysis. The only check that sees
per-path persistence runs the real saver and then looks for the values in storage.

The recipe, as used by glossum's ``test_ol_sense_fields_are_persisted.py``:

1. Give every free-text field a unique, findable string (:func:`field_sentinels`), and every constrained
   or typed field a legal non-default value.
2. Run the production saver. Build its preconditions by running production code too -- a hand-inserted
   row that production never creates is how a saver test passes against a broken saver -- and assert
   the rows production should have created exist before looking for anything.
3. Find each string in any text/JSON column (:func:`text_columns`, :func:`sentinel_locations`) and each
   typed value on the rows keyed by the record under test (:func:`rows_holding`). The column list comes
   from ``information_schema``, so a table added later is covered the day it exists.
4. Fail by name for every field stored nowhere, unless exempted with a reason, and fail when a field is
   added to the dataclass without being swept or exempted (:func:`unaccounted_fields`).

The database helpers take an SQLAlchemy ``AsyncSession`` (or anything with the same ``execute``) and use
PostgreSQL's ``information_schema`` and ``::text`` casts. SQLAlchemy is imported lazily; it is in the
``database`` extra.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

__all__ = [
    "TEXT_DATA_TYPES",
    "field_sentinels",
    "unaccounted_fields",
    "text_columns",
    "sentinel_locations",
    "rows_holding",
    "written_files",
    "file_sentinel_locations",
    "jsonl_records",
    "records_holding",
]

#: ``information_schema.columns.data_type`` values a string sentinel can land in.
TEXT_DATA_TYPES: Tuple[str, ...] = ("text", "character varying", "jsonb", "json")

Column = Tuple[str, str, str]  # (schema, table, column)


def field_sentinels(names: Iterable[str], prefix: str = "ZZSENT") -> Dict[str, str]:
    """``{name: sentinel}`` with one unique string per field name.

    Every sentinel has the same digit width and a closing ``ZZ``, so none is a substring of another: a
    LIKE search for one can never be satisfied by a different field's value. Duplicate names raise,
    since two fields sharing a sentinel would each be reported as stored when only one was.
    """
    ordered = list(names)
    duplicates = sorted({n for n in ordered if ordered.count(n) > 1})
    if duplicates:
        raise ValueError(f"duplicate field names: {duplicates}")
    width = max(4, len(str(len(ordered))))
    return {name: f"{prefix}{i:0{width}d}ZZ" for i, name in enumerate(ordered)}


def unaccounted_fields(cls: type, accounted: Iterable[str]) -> Set[str]:
    """Dataclass fields of ``cls`` that are neither swept nor exempted.

    Assert this is empty so a field added to the dataclass later cannot be silently skipped.
    """
    return {f.name for f in dataclasses.fields(cls)} - set(accounted)


# ---------------------------------------------------------------------------------------------------------
# File-backed stores: the same sweep when storage is JSONL rows and JSON documents rather than a database.
#
# The recipe is unchanged -- unique sentinels, the production writer, then a search of everything written --
# only "every text column" becomes "every file under the store's roots". Keep the negative control: search for
# a sentinel nothing wrote and assert it is found nowhere, or a search that matches everything passes having
# checked nothing.
# ---------------------------------------------------------------------------------------------------------

#: File suffixes a file-backed store writes text into.
TEXT_FILE_SUFFIXES: Tuple[str, ...] = (".jsonl", ".json", ".txt", ".ndjson", ".csv", ".md")


def written_files(roots: Iterable[Any], suffixes: Sequence[str] = TEXT_FILE_SUFFIXES) -> List[Any]:
    """Every file with one of ``suffixes`` under each root (a directory or a single file), sorted."""
    from pathlib import Path

    out: List[Path] = []
    for root in roots:
        path = Path(root)
        if path.is_file():
            out.append(path)
        elif path.is_dir():
            out.extend(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)
    return sorted(set(out))


def _decoded_json(text: str) -> Optional[str]:
    """``text`` re-serialised with non-ASCII left literal when it parses as JSON, else ``None``."""
    import json

    try:
        return json.dumps(json.loads(text), ensure_ascii=False)
    except ValueError:
        return None


def file_sentinel_locations(sentinels: Mapping[str, str], files: Iterable[Any]) -> Dict[str, List[str]]:
    """``{name: [file, ...]}`` for every sentinel, empty where no file holds it.

    Each file is read once and every sentinel tested against it. JSON escapes non-ASCII characters by default
    (``\\u00e9``), so a file is also searched in its decoded form when it parses as JSON or JSONL; the
    ``ZZSENT`` sentinels are ASCII either way, but a caller's own sentinels need not be.
    """
    found: Dict[str, List[str]] = {name: [] for name in sentinels}
    for f in files:
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                raw = fh.read()
        except OSError:
            continue
        haystacks = [raw]
        whole = _decoded_json(raw)
        if whole is not None:
            haystacks.append(whole)
        else:
            decoded = [d for d in (_decoded_json(line) for line in raw.splitlines()) if d is not None]
            if decoded:
                haystacks.append("\n".join(decoded))
        for name, sentinel in sentinels.items():
            if any(sentinel in h for h in haystacks):
                found[name].append(str(f))
    return found


def jsonl_records(path: Any) -> List[Dict[str, Any]]:
    """Every JSON object in a JSONL file, skipping blank lines; a missing file is no records."""
    import json
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _walk_values(obj: Any, field: str) -> List[Any]:
    """Values under key ``field`` anywhere inside ``obj`` (nested dicts and lists included)."""
    out: List[Any] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == field:
                out.append(v)
            out.extend(_walk_values(v, field))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(_walk_values(item, field))
    return out


def records_holding(
    records: Iterable[Mapping[str, Any]],
    value: Any,
    *,
    fields: Sequence[str],
    key_field: Optional[str] = None,
    key_value: Any = None,
    tolerance: float = 1e-9,
) -> List[str]:
    """``"#index.field"`` for every record (optionally only those with ``key_field == key_value``) holding
    ``value`` under one of ``fields``, at any nesting depth.

    The file-store twin of :func:`rows_holding`: typed values (booleans, numbers, enum strings) cannot be made
    unique the way a string sentinel can, so they are looked for only on the records about the item under
    test, never anywhere in the store. Floats compare within ``tolerance``; ``True`` never matches ``1``.
    """
    hits: List[str] = []
    for i, record in enumerate(records):
        if key_field is not None and record.get(key_field) != key_value:
            continue
        for field in fields:
            for held in _walk_values(record, field):
                same_kind = isinstance(held, bool) == isinstance(value, bool)
                if isinstance(value, float) and isinstance(held, (int, float)) and same_kind:
                    match = abs(float(held) - value) <= tolerance
                else:
                    match = same_kind and held == value
                if match:
                    hits.append(f"#{i}.{field}")
                    break
    return hits


def _quote(identifier: str) -> str:
    """A PostgreSQL quoted identifier, doubling any embedded quote."""
    return '"' + identifier.replace('"', '""') + '"'


def _like_contains(value: str) -> str:
    """A LIKE pattern matching ``value`` anywhere, with its own ``%``, ``_`` and ``\\`` taken literally."""
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


async def text_columns(session: Any, schemas: Sequence[str], data_types: Sequence[str] = TEXT_DATA_TYPES) -> List[Column]:
    """Every ``(schema, table, column)`` of the given data types in the given schemas."""
    from sqlalchemy import text

    rows = await session.execute(
        text(
            "SELECT table_schema, table_name, column_name FROM information_schema.columns "
            "WHERE table_schema = ANY(:schemas) AND data_type = ANY(:types) "
            "ORDER BY table_schema, table_name, column_name"
        ),
        {"schemas": list(schemas), "types": list(data_types)},
    )
    return [(r[0], r[1], r[2]) for r in rows.fetchall()]


async def sentinel_locations(session: Any, sentinels: Mapping[str, str], columns: Iterable[Column]) -> Dict[str, List[str]]:
    """``{name: ["schema.table.column", ...]}`` for every sentinel, empty where it is stored nowhere.

    One query per column tests all sentinels at once, rather than one query per column per sentinel.
    """
    from sqlalchemy import text

    names = list(sentinels)
    patterns = [_like_contains(sentinels[n]) for n in names]
    found: Dict[str, List[str]] = {n: [] for n in names}
    if not names:
        return found
    for schema, table, column in columns:
        rows = await session.execute(
            text(
                "SELECT p.i FROM unnest(CAST(:patterns AS text[])) WITH ORDINALITY AS p(pattern, i) "  # nosec B608 -- identifiers quoted, values bound
                f"WHERE EXISTS (SELECT 1 FROM {_quote(schema)}.{_quote(table)} t WHERE t.{_quote(column)}::text LIKE p.pattern)"
            ),
            {"patterns": patterns},
        )
        for (ordinal,) in rows.fetchall():
            found[names[int(ordinal) - 1]].append(f"{schema}.{table}.{column}")
    return found


async def rows_holding(
    session: Any,
    value: object,
    *,
    columns: Sequence[str],
    key_value: object,
    keys: Sequence[Tuple[str, Optional[str], str]],
) -> List[str]:
    """``"schema.table.column"`` for every row about ``key_value`` that holds ``value`` in one of ``columns``.

    ``keys`` says which rows are "about" the record: each entry is ``(schema, table, key_column)``, with
    ``table=None`` meaning every table of that schema having ``key_column``. For example
    ``[("llm", None, "sense_id"), ("oewn", "senses", "id")]`` covers every per-sense table and the sense
    row itself -- probing only the first shape once reported a field as stored nowhere while it sat on
    the sense row.
    """
    from sqlalchemy import text

    if not keys or not columns:
        return []
    params: Dict[str, object] = {"cols": list(columns)}
    clauses = []
    for i, (schema, table, key_column) in enumerate(keys):
        clause = f"(c.table_schema = :s{i} AND k.column_name = :k{i}"
        params[f"s{i}"], params[f"k{i}"] = schema, key_column
        if table is not None:
            clause += f" AND c.table_name = :t{i}"
            params[f"t{i}"] = table
        clauses.append(clause + ")")
    candidates = await session.execute(
        text(
            "SELECT c.table_schema, c.table_name, c.column_name, k.column_name FROM information_schema.columns c "  # nosec B608 -- only bind placeholders are interpolated
            "JOIN information_schema.columns k ON k.table_schema = c.table_schema AND k.table_name = c.table_name "
            f"WHERE c.column_name = ANY(:cols) AND ({' OR '.join(clauses)}) "
            "ORDER BY 1, 2, 3"
        ),
        params,
    )
    hits = []
    for schema, table, column, key_column in candidates.fetchall():
        held = await session.execute(
            text(f"SELECT 1 FROM {_quote(schema)}.{_quote(table)} WHERE {_quote(key_column)} = :key AND {_quote(column)} = :value LIMIT 1"),  # nosec B608 -- identifiers quoted, values bound
            {"key": key_value, "value": value},
        )
        if held.fetchone() is not None:
            hits.append(f"{schema}.{table}.{column}")
    return hits

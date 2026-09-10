"""pyutilz.dev.persistence_sweep: the pure parts, and the SQL each database helper sends.

The statements are also run against a real PostgreSQL by glossum's
tests/test_integration/test_ol_sense_fields_are_persisted.py, which is where "Postgres accepts this"
is proven; a mocked session can only show what is sent and how results are read back.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("sqlalchemy")

from pyutilz.dev.persistence_sweep import (
    _like_contains,
    _quote,
    field_sentinels,
    rows_holding,
    sentinel_locations,
    text_columns,
    unaccounted_fields,
)


class _Result:
    def __init__(self, rows: List[tuple]) -> None:
        self._rows = rows

    def fetchall(self) -> List[tuple]:
        return self._rows

    def fetchone(self) -> Optional[tuple]:
        return self._rows[0] if self._rows else None


def _session(*results: List[tuple]) -> MagicMock:
    """A session whose ``execute`` returns the queued row lists in order, then empty results."""
    queue = list(results)
    session = MagicMock()
    session.execute = AsyncMock(side_effect=lambda *a, **k: _Result(queue.pop(0) if queue else []))
    return session


def _sent(session: MagicMock) -> List[Tuple[str, Any]]:
    """``(sql, params)`` for each ``execute`` the helper awaited."""
    return [(str(c.args[0]), c.args[1] if len(c.args) > 1 else None) for c in session.execute.await_args_list]


class TestFieldSentinels:
    def test_one_distinct_value_per_name(self) -> None:
        s = field_sentinels(["a", "b", "c"])

        assert s == {"a": "ZZSENT0000ZZ", "b": "ZZSENT0001ZZ", "c": "ZZSENT0002ZZ"}

    def test_no_sentinel_is_a_substring_of_another(self) -> None:
        values = list(field_sentinels([f"f{i}" for i in range(12000)]).values())
        sample = values[:50] + values[-50:]

        assert not [(a, b) for a in sample for b in sample if a != b and a in b]

    def test_duplicate_names_are_refused(self) -> None:
        with pytest.raises(ValueError, match=r"\['x'\]"):
            field_sentinels(["x", "y", "x"])


def test_unaccounted_fields_names_what_is_neither_swept_nor_exempted() -> None:
    @dataclasses.dataclass
    class Record:
        a: str = ""
        b: int = 0
        c: bool = False

    assert unaccounted_fields(Record, {"a", "c"}) == {"b"}
    assert unaccounted_fields(Record, {"a", "b", "c", "gone"}) == set()


def test_identifiers_are_quoted_and_like_metacharacters_escaped() -> None:
    assert _quote('we"ird') == '"we""ird"'
    assert _like_contains("50%_a\\b") == "%50\\%\\_a\\\\b%"


def test_text_columns_asks_for_the_given_schemas_and_types() -> None:
    session = _session([("llm", "t", "c")])

    cols = asyncio.run(text_columns(session, ("llm", "oewn")))

    assert cols == [("llm", "t", "c")]
    assert session.execute.await_count == 1
    assert _sent(session)[0][1] == {"schemas": ["llm", "oewn"], "types": ["text", "character varying", "jsonb", "json"]}


def test_sentinel_locations_maps_ordinals_back_to_names_one_query_per_column() -> None:
    session = _session([(2,)], [(1,), (2,)])
    sentinels = {"x": "ZZSENT0000ZZ", "y": "ZZSENT0001ZZ", "z": "ZZSENT0002ZZ"}

    found = asyncio.run(sentinel_locations(session, sentinels, [("llm", "a", "c1"), ("oewn", 'q"t', "c2")]))

    assert found == {"x": ['oewn.q"t.c2'], "y": ["llm.a.c1", 'oewn.q"t.c2'], "z": []}
    sent = _sent(session)
    assert len(sent) == 2
    assert 'FROM "oewn"."q""t" t WHERE t."c2"::text LIKE p.pattern' in sent[1][0]
    assert sent[0][1] == {"patterns": ["%ZZSENT0000ZZ%", "%ZZSENT0001ZZ%", "%ZZSENT0002ZZ%"]}


def test_rows_holding_filters_by_each_key_and_probes_each_candidate() -> None:
    session = _session([("llm", "meta", "gender", "sense_id"), ("oewn", "senses", "gender", "id")], [(1,)], [])

    hits = asyncio.run(rows_holding(session, "f", columns=["gender"], key_value="s1", keys=[("llm", None, "sense_id"), ("oewn", "senses", "id")]))

    assert hits == ["llm.meta.gender"]
    sent = _sent(session)
    assert "(c.table_schema = :s0 AND k.column_name = :k0)" in sent[0][0]
    assert "(c.table_schema = :s1 AND k.column_name = :k1 AND c.table_name = :t1)" in sent[0][0]
    assert sent[0][1] == {"cols": ["gender"], "s0": "llm", "k0": "sense_id", "s1": "oewn", "k1": "id", "t1": "senses"}
    assert 'FROM "oewn"."senses" WHERE "id" = :key AND "gender" = :value' in sent[2][0]
    assert sent[2][1] == {"key": "s1", "value": "f"}


def test_rows_holding_with_nothing_to_look_for_sends_nothing() -> None:
    session = _session()

    assert asyncio.run(rows_holding(session, 1, columns=[], key_value="s", keys=[("llm", None, "sense_id")])) == []
    session.execute.assert_not_awaited()

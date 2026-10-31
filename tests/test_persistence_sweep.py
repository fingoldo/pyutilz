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

from pyutilz.dev import persistence_sweep as ps
from pyutilz.dev.persistence_sweep import (
    _like_contains,
    _quote,
    field_sentinels,
    file_sentinel_locations,
    jsonl_records,
    records_holding,
    rows_holding,
    sentinel_locations,
    text_columns,
    unaccounted_fields,
    written_files,
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
    assert sent[0][1] == {"patterns": ["%ZZSENT0000ZZ%", "%ZZSENT0001ZZ%", "%ZZSENT0002ZZ%"]}


def test_rows_holding_filters_by_each_key_and_probes_each_candidate() -> None:
    session = _session([("llm", "meta", "gender", "sense_id"), ("oewn", "senses", "gender", "id")], [(1,)], [])

    hits = asyncio.run(rows_holding(session, "f", columns=["gender"], key_value="s1", keys=[("llm", None, "sense_id"), ("oewn", "senses", "id")]))

    assert hits == ["llm.meta.gender"]
    sent = _sent(session)
    assert "(c.table_schema = :s0 AND k.column_name = :k0)" in sent[0][0]
    assert "(c.table_schema = :s1 AND k.column_name = :k1 AND c.table_name = :t1)" in sent[0][0]
    assert sent[0][1] == {"cols": ["gender"], "s0": "llm", "k0": "sense_id", "s1": "oewn", "k1": "id", "t1": "senses"}
    assert sent[2][1] == {"key": "s1", "value": "f"}


def test_rows_holding_with_nothing_to_look_for_sends_nothing() -> None:
    session = _session()

    assert asyncio.run(rows_holding(session, 1, columns=[], key_value="s", keys=[("llm", None, "sense_id")])) == []
    session.execute.assert_not_awaited()


class TestFileStoreSweep:
    def test_written_files_keeps_only_the_text_suffixes_in_any_case(self, tmp_path):
        (tmp_path / "a.JSON").write_text("{}")
        (tmp_path / "b.csv").write_text("x")
        (tmp_path / "c.py").write_text("x")
        assert [p.name for p in written_files([tmp_path])] == ["a.JSON", "b.csv"]

    def test_written_files_recurses_into_directories_and_accepts_a_single_file(self, tmp_path):
        sub = tmp_path / "deep"
        sub.mkdir()
        (sub / "x.jsonl").write_text("{}")
        single = tmp_path / "y.md"
        single.write_text("#")
        found = written_files([tmp_path / "deep", single])
        assert [p.name for p in found] == ["x.jsonl", "y.md"]

    def test_written_files_skips_a_path_that_does_not_exist(self, tmp_path):
        assert written_files([tmp_path / "missing"]) == []

    def test_a_sentinel_in_the_raw_text_is_located(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("row ZZSENT0001ZZ here")
        assert file_sentinel_locations({"name": "ZZSENT0001ZZ"}, [f]) == {"name": [str(f)]}

    def test_a_non_ascii_value_is_found_in_its_decoded_json_form(self, tmp_path):
        f = tmp_path / "a.json"
        f.write_text('{"city": "\u00e9t\u00e9"}')
        assert file_sentinel_locations({"city": "été"}, [f]) == {"city": [str(f)]}

    def test_a_jsonl_line_is_searched_in_its_decoded_form(self, tmp_path):
        f = tmp_path / "a.jsonl"
        f.write_text('{"city": "\u00e9"}\n{"city": "x"}\n')
        assert file_sentinel_locations({"city": "é"}, [f]) == {"city": [str(f)]}

    def test_a_sentinel_nowhere_is_reported_with_an_empty_list(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("nothing here")
        assert file_sentinel_locations({"ghost": "ZZGHOST"}, [f]) == {"ghost": []}

    def test_an_unreadable_file_is_logged_by_name_and_the_scan_continues(self, tmp_path, monkeypatch, caplog):
        import logging

        good = tmp_path / "good.txt"
        good.write_text("ZZHIT")
        bad = tmp_path / "bad.txt"
        bad.write_text("x")
        real_open = open

        def _open(path, *args, **kwargs):
            if str(path) == str(bad):
                raise OSError("denied")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", _open)
        with caplog.at_level(logging.WARNING, logger=ps.logger.name):
            found = file_sentinel_locations({"h": "ZZHIT"}, [bad, good])
        assert found == {"h": [str(good)]}
        assert [r.getMessage() for r in caplog.records] == [f"1 file(s) could not be read and were not scanned: {bad}: denied"]

    def test_invalid_json_text_is_not_decoded_and_logged_at_debug(self, caplog):
        import logging

        with caplog.at_level(logging.DEBUG, logger=ps.logger.name):
            assert ps._decoded_json("{not json") is None
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
        assert len(messages) == 1
        assert messages[0].startswith("_decoded_json: text is not valid JSON: ")

    def test_jsonl_records_skips_blank_lines_and_a_missing_file_is_no_records(self, tmp_path):
        f = tmp_path / "r.jsonl"
        f.write_text('{"a": 1}\n\n   \n{"a": 2}\n')
        assert jsonl_records(f) == [{"a": 1}, {"a": 2}]
        assert jsonl_records(tmp_path / "missing.jsonl") == []


class TestRecordsHolding:
    def test_a_value_is_found_at_any_depth_under_the_field(self):
        records = [{"outer": {"inner": [{"stored": "v"}]}}]
        assert records_holding(records, "v", fields=["stored"]) == ["#0.stored"]

    def test_the_key_filter_keeps_only_the_matching_records(self):
        records = [{"k": "a", "f": 1}, {"k": "b", "f": 1}]
        assert records_holding(records, 1, fields=["f"], key_field="k", key_value="b") == ["#1.f"]

    def test_a_boolean_never_matches_an_integer(self):
        assert records_holding([{"f": 1}], True, fields=["f"]) == []
        assert records_holding([{"f": True}], 1, fields=["f"]) == []

    def test_a_float_matches_within_the_tolerance_and_not_beyond_it(self):
        assert records_holding([{"f": 0.5}], 0.5 + 1e-10, fields=["f"]) == ["#0.f"]
        assert records_holding([{"f": 0.5}], 0.5 + 1e-3, fields=["f"]) == []

    def test_a_value_absent_from_every_record_gives_no_hits(self):
        assert records_holding([{"f": "x"}], "y", fields=["f"]) == []


def _sql_of(call_args) -> str:
    return str(call_args.args[0])


def test_text_columns_sends_exactly_this_information_schema_query() -> None:
    session = MagicMock()
    session.execute = AsyncMock(return_value=_Result([]))
    asyncio.run(text_columns(session, ["public"]))
    assert _sql_of(session.execute.call_args) == (
        "SELECT table_schema, table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = ANY(:schemas) AND data_type = ANY(:types) "
        "ORDER BY table_schema, table_name, column_name"
    )


def test_sentinel_locations_probes_each_column_with_its_quoted_identifiers() -> None:
    session = MagicMock()
    session.execute = AsyncMock(return_value=_Result([]))
    asyncio.run(sentinel_locations(session, {"a": "ZZ0000ZZ"}, [("public", "t", "c")]))
    assert _sql_of(session.execute.call_args) == (
        "SELECT p.i FROM unnest(CAST(:patterns AS text[])) WITH ORDINALITY AS p(pattern, i) "
        'WHERE EXISTS (SELECT 1 FROM "public"."t" t WHERE t."c"::text LIKE p.pattern)'
    )


def test_rows_holding_finds_the_key_columns_of_every_table_in_the_schema() -> None:
    session = MagicMock()
    session.execute = AsyncMock(return_value=_Result([]))
    asyncio.run(rows_holding(session, "v", columns=["c"], key_value=1, keys=[("llm", None, "sense_id")]))
    assert _sql_of(session.execute.call_args) == (
        "SELECT c.table_schema, c.table_name, c.column_name, k.column_name FROM information_schema.columns c "
        "JOIN information_schema.columns k ON k.table_schema = c.table_schema AND k.table_name = c.table_name "
        "WHERE c.column_name = ANY(:cols) AND ((c.table_schema = :s0 AND k.column_name = :k0)) "
        "ORDER BY 1, 2, 3"
    )


def test_rows_holding_narrows_to_a_named_table_when_one_is_given() -> None:
    session = MagicMock()
    session.execute = AsyncMock(return_value=_Result([]))
    asyncio.run(rows_holding(session, "v", columns=["c"], key_value=1, keys=[("oewn", "senses", "id")]))
    assert "((c.table_schema = :s0 AND k.column_name = :k0 AND c.table_name = :t0))" in _sql_of(session.execute.call_args)

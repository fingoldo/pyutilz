"""Behaviour of ``pyutilz.database.db.legacy``: each PascalCase alias emits a DeprecationWarning naming its replacement
and forwards to the snake_case facade function with the matching keyword arguments, and the result comes back
unchanged. The facade functions are replaced by recorders, so no database is needed."""

from __future__ import annotations

import pytest

import pyutilz.database.db as facade
from pyutilz.database.db import legacy


@pytest.fixture
def recorder(monkeypatch):
    calls: list[tuple[str, dict]] = []

    def make(name: str, result=None):
        def fake(**kwargs):
            calls.append((name, kwargs))
            return result

        return fake

    for name in (
        "ensure_pg_table_exists",
        "read_table_into_dict",
        "read_table_into_dict_reversed",
        "get_id_by_key_field_and_insert_if_needed",
    ):
        monkeypatch.setattr(facade, name, make(name, result="id-1" if name.startswith("get_id") else None))
    return calls


class TestEnsurePgTableExists:
    def test_warns_and_forwards_the_renamed_arguments(self, recorder):
        with pytest.warns(DeprecationWarning, match="ensure_pg_table_exists"):
            legacy.EnsurePgTableExists("items", "code", "item_id", "serial")
        assert recorder == [
            ("ensure_pg_table_exists", {"table": "items", "key_field_name": "code", "id_field_name": "item_id", "autocreate_id_type_name": "serial"})
        ]


class TestReadTableIntoDic:
    def test_warns_and_forwards_with_the_condition_renamed(self, recorder):
        with pytest.warns(DeprecationWarning, match="read_table_into_dict"):
            legacy.ReadTableIntoDic({}, "items", "code", "active", "item_id", None)
        name, kwargs = recorder[0]
        assert name == "read_table_into_dict"
        assert kwargs["condition"] == "active"
        assert kwargs["dict_enums"] == {}
        assert kwargs["table"] == "items"


class TestReadTableIntoDicReversed:
    def test_warns_and_forwards_to_the_reversed_reader(self, recorder):
        with pytest.warns(DeprecationWarning, match="read_table_into_dict_reversed"):
            legacy.ReadTableIntoDicReversed({}, "items")
        assert recorder[0][0] == "read_table_into_dict_reversed"
        assert recorder[0][1]["table"] == "items"


class TestGetIdByKeyFieldAndInsertIfNeeded:
    def test_the_result_is_returned_unchanged(self, recorder):
        with pytest.warns(DeprecationWarning, match="get_id_by_key_field_and_insert_if_needed"):
            result = legacy.GetIdByKeyFieldAndInsertIfNeeded({}, "items", "widget")
        assert result == "id-1"

    def test_the_legacy_api_always_passed_a_raw_fragment(self, recorder):
        with pytest.warns(DeprecationWarning):
            legacy.GetIdByKeyFieldAndInsertIfNeeded({}, "items", "widget", bAddUpdatedAtTimestamp="updated_at")
        kwargs = recorder[0][1]
        assert kwargs["raw_sql"] is True
        assert kwargs["key_field_value"] == "widget"
        assert kwargs["add_updated_at_timestamp"] == "updated_at"

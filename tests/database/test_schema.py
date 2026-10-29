"""Behaviour of ``pyutilz.database.db.schema``: the partition walk steps by day, week, month or year and rejects any other size,
the partition name follows its convention, bounds are written as timestamps or as scaled UTC epochs, an alternate value
clause binds a sequence and refuses a raw SQL string unless asked, a JSON navigation path must match the allowed shape, and
the value distribution reports the most frequent value and whether a null occurs. The database call is replaced by a
recorder, so no connection is needed."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from pyutilz.database.db import schema
from pyutilz.database.db.schema import _facade
from pyutilz.database.exceptions import SQLValidationError


@pytest.fixture
def executed(monkeypatch):
    statements = []
    monkeypatch.setattr(_facade, "safe_execute", statements.append)
    return statements


class TestPartitionWalk:
    def test_a_day_walk_steps_one_day_at_a_time(self):
        pairs = list(schema._iter_partition_dates(date(2026, 1, 1), date(2026, 1, 3), "day"))
        assert pairs == [(date(2026, 1, 1), date(2026, 1, 2)), (date(2026, 1, 2), date(2026, 1, 3)), (date(2026, 1, 3), date(2026, 1, 4))]

    def test_a_month_walk_steps_one_month_at_a_time(self):
        pairs = list(schema._iter_partition_dates(date(2026, 1, 31), date(2026, 3, 1), "month"))
        assert pairs[0] == (date(2026, 1, 31), date(2026, 2, 28))
        assert pairs[-1][0] <= date(2026, 3, 1)

    def test_an_unknown_size_is_refused(self):
        with pytest.raises(AssertionError, match="day/week/month/year"):
            list(schema._iter_partition_dates(date(2026, 1, 1), date(2026, 1, 2), "fortnight"))


class TestPartitionNames:
    def test_the_name_encodes_year_month_week_and_day(self):
        assert schema._partition_table_name("ev", date(2026, 1, 5)) == "z_ev_y2026m01w01d05"

    def test_a_date_is_midnight_utc_and_a_datetime_keeps_its_zone(self):
        assert schema._as_utc_datetime(date(2026, 1, 5)) == datetime(2026, 1, 5, tzinfo=timezone.utc)
        zoned = datetime(2026, 1, 5, 3, tzinfo=timezone.utc)
        assert schema._as_utc_datetime(zoned) is zoned


class TestCreatePartitions:
    def test_timestamp_bounds_are_written_by_default(self, executed):
        schema.create_postgres_range_partitions("ev", date(2026, 1, 1), date(2026, 1, 1), "day")
        assert executed == ["CREATE TABLE z_ev_y2026m01w01d01 PARTITION OF ev FOR VALUES FROM ('2026-01-01 00:00:00') TO ('2026-01-02 00:00:00')"]

    def test_an_invalid_table_name_is_refused_before_any_statement(self, executed):
        with pytest.raises(SQLValidationError):
            schema.create_postgres_range_partitions("ev; drop table x", date(2026, 1, 1), date(2026, 1, 1), "day")
        assert executed == []


class TestAlternateValues:
    def test_a_sequence_becomes_one_placeholder_per_value(self):
        assert schema._alternate_values_clause("a,b", [1, 2], False) == ("%s,%s", [1, 2])

    def test_a_raw_fragment_is_refused_unless_asked(self):
        with pytest.raises(TypeError, match="SQL fragment"):
            schema._alternate_values_clause("a", "1", False)

    def test_a_raw_fragment_is_spliced_when_asked(self):
        assert schema._alternate_values_clause("a", "1", True) == ("1", [])

    def test_a_name_and_value_count_mismatch_is_refused(self):
        with pytest.raises(ValueError, match="2 name"):
            schema._alternate_values_clause("a,b", [1], False)


class TestJsonPath:
    def test_no_path_gives_the_bare_column(self):
        assert schema._json_field_path("col", "") == "col"

    def test_a_valid_path_navigates_from_the_column(self):
        assert schema._json_field_path("col", "'a'") == "col->'a'"

    def test_an_invalid_path_is_refused(self):
        with pytest.raises(ValueError, match="Invalid JSON navigation path"):
            schema._json_field_path("col", "'a'); drop table x; --")


class TestValueDistribution:
    def test_the_most_frequent_value_and_its_total_are_reported(self):
        assert schema._value_distribution([("x", 3), (None, 1), ("y", 5)]) == (True, "y", 5, 9)

    def test_no_null_is_reported_as_such(self):
        assert schema._value_distribution([("x", 2), ("y", 1)]) == (False, "x", 2, 3)


class _Recorder:
    def __init__(self, returned_id=7):
        self.calls = []
        self.returned_id = returned_id

    def __call__(self, sql, params=None, *args, **kwargs):
        self.calls.append((sql, tuple(params) if params is not None else None))
        return [[self.returned_id]]


@pytest.fixture
def db(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(_facade, "safe_execute", recorder)
    return recorder


class TestGetIdByKeyField:
    def test_the_null_sentinel_is_returned_without_a_query(self, db):
        assert schema.get_id_by_key_field_and_insert_if_needed({}, "t", "null") == "null"
        assert db.calls == []

    def test_a_known_key_returns_its_stored_id_without_a_query(self, db):
        assert schema.get_id_by_key_field_and_insert_if_needed({"red": 3}, "t", "red") == 3
        assert db.calls == []

    def test_a_new_key_is_inserted_with_the_default_name_column_and_remembered(self, db):
        enums = {}
        assert schema.get_id_by_key_field_and_insert_if_needed(enums, "colors", "red") == 7
        sql, params = db.calls[0]
        assert sql.startswith("insert into colors (name) values (%s) on conflict (name)")
        assert "returning id" in sql
        assert params == ("red",)
        assert enums == {"red": 7}

    def test_none_defaults_fall_back_to_name_and_id_columns(self, db):
        schema.get_id_by_key_field_and_insert_if_needed({}, "colors", "red", key_field_name=None, id_field_name=None, alternate_fields_names=None, unique_constraint_fields=None)
        sql, _params = db.calls[0]
        assert "insert into colors (name)" in sql
        assert "returning id" in sql

    def test_a_string_key_is_passed_as_text_and_a_numeric_key_as_is(self, db):
        schema.get_id_by_key_field_and_insert_if_needed({}, "t", 5)
        schema.get_id_by_key_field_and_insert_if_needed({}, "t", 5, key_is_not_string=True)
        assert db.calls[0][1] == ("5",)
        assert db.calls[1][1] == (5,)

    def test_an_explicit_unique_constraint_names_the_conflict_target(self, db):
        schema.get_id_by_key_field_and_insert_if_needed({}, "t", "x", key_field_name="code", unique_constraint_fields="code,region")
        assert "on conflict (code,region)" in db.calls[0][0]

    def test_alternate_fields_are_inserted_with_their_values(self, db):
        schema.get_id_by_key_field_and_insert_if_needed({}, "colors", "red", alternate_fields_names="hex", alternate_fields_values=["#f00"])
        sql, params = db.calls[0]
        assert "insert into colors (name , hex) values (%s,%s)" in sql
        assert params == ("red", "#f00")

    def test_alternate_fields_only_inserts_without_the_key_column(self, db):
        schema.get_id_by_key_field_and_insert_if_needed({}, "colors", "red", alternate_fields_names="hex", alternate_fields_values=["#f00"], use_alternate_fields_only=True)
        sql, params = db.calls[0]
        assert "insert into colors (hex) values (%s)" in sql
        assert params == ("#f00",)

    def test_an_update_timestamp_is_added_to_the_conflict_update(self, db):
        schema.get_id_by_key_field_and_insert_if_needed({}, "colors", "red", add_updated_at_timestamp="updated_at")
        assert "updated_at" in db.calls[0][0]

    def test_an_invalid_table_name_is_refused_before_any_query(self, db):
        with pytest.raises(SQLValidationError):
            schema.get_id_by_key_field_and_insert_if_needed({}, "colors; drop table x", "red")
        assert db.calls == []


class TestGetIdExactStatements:
    def test_the_default_statement_is_pinned_word_for_word(self, db):
        schema.get_id_by_key_field_and_insert_if_needed({}, "colors", "red")
        assert db.calls[0][0] == "insert into colors (name) values (%s) on conflict (name) do update set name=excluded.name returning id"

    def test_the_alternate_statement_with_a_timestamp_is_pinned_word_for_word(self, db):
        schema.get_id_by_key_field_and_insert_if_needed({}, "colors", "red", alternate_fields_names="hex", alternate_fields_values=["#f00"], add_updated_at_timestamp="updated_at")
        assert db.calls[0][0] == (
            "insert into colors (name , hex) values (%s,%s) on conflict (name) do update set name=excluded.name,updated_at=(now() at time zone 'utc') returning id"
        )


class TestGetIdIdentifierValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"key_field_name": "name; drop table x"},
            {"id_field_name": "id; drop table x"},
            {"alternate_fields_names": "hex; drop table x", "alternate_fields_values": ["#f00"]},
            {"unique_constraint_fields": "name; drop table x"},
        ],
    )
    def test_an_invalid_identifier_is_refused_before_any_query(self, db, kwargs):
        with pytest.raises(SQLValidationError):
            schema.get_id_by_key_field_and_insert_if_needed({}, "colors", "red", **kwargs)
        assert db.calls == []


class _Rows:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def __call__(self, sql, params=None, *args, **kwargs):
        self.calls.append(sql)
        return self.rows


def _suggest(monkeypatch, rows, **kwargs):
    fake = _Rows(rows)
    monkeypatch.setattr(_facade, "safe_execute", fake)
    result = schema.suggest_json_optimization("items", "data", fields=["color"], **kwargs)
    return result, fake


class TestSuggestJsonOptimization:
    def test_the_query_groups_the_field_values_by_frequency(self, monkeypatch):
        _result, fake = _suggest(monkeypatch, [("x", 6), ("y", 4)])
        assert " ".join(fake.calls[0].split()) == (
            "select data->>'color' as val,count(*) as qty from items where data is not null group by data->>'color' order by qty desc"
        )

    def test_a_value_holding_exactly_the_share_is_suggested(self, monkeypatch):
        result, _fake = _suggest(monkeypatch, [("x", 5), ("y", 5)])
        assert result == {"color": "x"}

    def test_a_value_below_the_share_is_not_suggested(self, monkeypatch):
        result, _fake = _suggest(monkeypatch, [("x", 6), ("y", 4)], min_occurence_percent=0.7)
        assert result == {}

    def test_a_tie_keeps_the_first_most_frequent_value(self, monkeypatch):
        result, _fake = _suggest(monkeypatch, [("x", 5), ("y", 5)])
        assert result["color"] == "x"

    def test_a_frequent_value_is_not_suggested_when_nulls_also_occur(self, monkeypatch):
        result, _fake = _suggest(monkeypatch, [("x", 6), (None, 4)])
        assert result == {}

    def test_more_values_than_the_display_limit_are_logged_with_an_ellipsis(self, monkeypatch, caplog):
        rows = [(f"v{i}", 10 - i) for i in range(6)]
        with caplog.at_level("INFO"):
            _suggest(monkeypatch, rows, max_vals=5)
        assert any(rec.getMessage().startswith("color ") and rec.getMessage().endswith("...") for rec in caplog.records)

    def test_an_invalid_table_is_refused_before_any_query(self, monkeypatch):
        fake = _Rows([])
        monkeypatch.setattr(_facade, "safe_execute", fake)
        with pytest.raises(SQLValidationError):
            schema.suggest_json_optimization("items; drop table x", "data", fields=["color"])
        assert fake.calls == []

    def test_an_invalid_json_column_is_refused_before_any_query(self, monkeypatch):
        fake = _Rows([])
        monkeypatch.setattr(_facade, "safe_execute", fake)
        with pytest.raises(SQLValidationError):
            schema.suggest_json_optimization("items", "data; drop", fields=["color"])
        assert fake.calls == []

    def test_an_invalid_json_key_is_refused_before_any_query(self, monkeypatch):
        fake = _Rows([])
        monkeypatch.setattr(_facade, "safe_execute", fake)
        with pytest.raises(SQLValidationError):
            schema.suggest_json_optimization("items", "data", fields=["color'; drop"])
        assert fake.calls == []

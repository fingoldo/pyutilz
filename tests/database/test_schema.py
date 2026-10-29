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

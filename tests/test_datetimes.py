"""Behaviour of ``pyutilz.core.pythonlib.datetimes``: timezone-offset normalisation, unix timestamps (naive
datetimes are UTC), the 7-day week-of-month buckets, and the whole-years age. Expected values follow the
functions' docstrings and are checked against the epoch and fixed dates."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from pyutilz.core.pythonlib.datetimes import (
    age,
    datetime_to_unix_ts,
    datetime_to_utc_timestamp,
    get_utc_unix_ts_milliseconds,
    get_utc_unix_ts_seconds,
    read_timezoned_ts,
    weekofmonth,
)


class TestTimezoneOffset:
    def test_the_colon_is_dropped_from_a_trailing_offset(self):
        assert read_timezoned_ts("2020-02-20T11:54:00.000-07:00") == "2020-02-20T11:54:00.000-0700"

    def test_a_string_without_an_offset_is_unchanged(self):
        assert read_timezoned_ts("2020-02-20T11:54:00") == "2020-02-20T11:54:00"


class TestUnixTimestamps:
    def test_the_epoch_is_zero(self):
        assert datetime_to_unix_ts(datetime(1970, 1, 1)) == 0

    def test_a_naive_datetime_is_read_as_utc(self):
        assert datetime_to_unix_ts(datetime(1970, 1, 2)) == 86400

    def test_utc_timestamp_of_a_naive_datetime_matches_an_aware_utc_one(self):
        assert datetime_to_utc_timestamp(datetime(2024, 1, 1, 12, 0, 0)) == 1704110400
        assert datetime_to_utc_timestamp(datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)) == 1704110400

    def test_current_timestamps_have_the_right_units(self):
        seconds = get_utc_unix_ts_seconds()
        millis = get_utc_unix_ts_milliseconds()
        assert isinstance(seconds, int) and isinstance(millis, int)
        assert abs(millis - seconds * 1000) < 5000


class TestWeekOfMonth:
    @pytest.mark.parametrize(
        "day,expected",
        [(date(2024, 1, 1), 1), (date(2024, 1, 7), 1), (date(2024, 1, 8), 2), (date(2024, 1, 31), 5)],
    )
    def test_seven_day_buckets_starting_at_day_one(self, day, expected):
        assert weekofmonth(day) == expected


class TestAge:
    def test_whole_years_before_the_birthday_this_year(self):
        today = date.today()
        dob = date(today.year - 30, today.month, today.day)
        assert age(dob) == 30

    def test_the_birthday_not_yet_reached_counts_one_year_less(self):
        today = date.today()
        if (today.month, today.day) == (12, 31):
            pytest.skip("the last day of the year: every birthday this year has been reached")
        assert age(date(today.year - 30, 12, 31)) == 29

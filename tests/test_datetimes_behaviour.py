"""Behaviour of ``pyutilz.core.pythonlib.datetimes``: the UTC-to-local formatting with its "time passed" suffix, the
timezone-offset normalisation, Unix timestamps, the week-of-month and age arithmetic, and the imitated delay. Values
that depend on the current time are checked with a pattern or against a reference taken at the same moment."""

import logging
import re
from datetime import date, datetime, timedelta, timezone

import pytest

from pyutilz.core.pythonlib import datetimes as dt_mod
from pyutilz.core.pythonlib.datetimes import (
    age,
    datetime_to_unix_ts,
    datetime_to_utc_timestamp,
    imitate_delay,
    read_timezoned_ts,
    utc_to_local,
    utc_ts_2_locstr,
    weekofmonth,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class TestUtcTsToLocStr:
    def test_default_output_has_seconds_and_no_suffix(self):
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", utc_ts_2_locstr("2021-01-02T03:04:05.000006"))

    def test_empty_or_missing_input_gives_an_empty_string(self):
        assert utc_ts_2_locstr(None) == ""
        assert utc_ts_2_locstr("") == ""

    def test_a_custom_output_format_is_used(self):
        assert re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", utc_ts_2_locstr("2021-01-02T03:04:05.000006", output_date_format="%d.%m.%Y"))

    def test_a_custom_input_format_is_parsed(self):
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", utc_ts_2_locstr("2021-01-02", input_date_format="%Y-%m-%d"))

    def test_elapsed_minutes_are_counted_whole_and_named(self):
        stamp = (_utc_now() - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S.%f")
        assert utc_ts_2_locstr(stamp, dst=60).endswith(" (120 мин. тому назад)")

    @pytest.mark.parametrize("dst, unit", [(1, "сек"), (60 * 60, "ч"), (60 * 60 * 24, "дн")])
    def test_the_default_units_are_named_in_russian(self, dst, unit):
        stamp = (_utc_now() - timedelta(seconds=dst * 3)).strftime("%Y-%m-%dT%H:%M:%S.%f")
        assert re.search(rf" \(\d+ {unit}\. тому назад\)$", utc_ts_2_locstr(stamp, dst=dst))

    def test_a_custom_unit_table_is_used(self):
        stamp = (_utc_now() - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%S.%f")
        assert utc_ts_2_locstr(stamp, dst=60, dst_names={60: "m"}).endswith(" m. тому назад)")

    @pytest.mark.parametrize("dst", [0, -60, 7, None])
    def test_a_non_positive_or_unknown_unit_adds_no_suffix(self, dst):
        assert "тому назад" not in utc_ts_2_locstr("2021-01-02T03:04:05.000006", dst=dst)


class TestReadTimezonedTs:
    def test_a_trailing_colon_in_the_offset_is_dropped(self):
        assert read_timezoned_ts("2020-02-20T11:54:00.000-07:00") == "2020-02-20T11:54:00.000-0700"

    def test_a_string_without_an_offset_is_unchanged(self):
        assert read_timezoned_ts("2020-02-20T11:54:00") == "2020-02-20T11:54:00"

    def test_a_plus_offset_is_normalised_too(self):
        assert read_timezoned_ts("2020-02-20T11:54:00+05:30") == "2020-02-20T11:54:00+0530"


class TestUnixTimestamps:
    def test_the_epoch_plus_one_day_is_86400_seconds(self):
        assert datetime_to_unix_ts(datetime(1970, 1, 2)) == 86400

    def test_fractions_of_a_second_are_truncated(self):
        assert datetime_to_unix_ts(datetime(1970, 1, 1, 0, 0, 1, 999999)) == 1

    def test_a_utc_aware_datetime_gives_its_utc_timestamp(self):
        assert datetime_to_utc_timestamp(datetime(1970, 1, 1, 0, 0, 10)) == 10
        assert datetime_to_utc_timestamp(datetime(1970, 1, 1, 1, tzinfo=timezone(timedelta(hours=1)))) == 0


class TestCalendarArithmetic:
    @pytest.mark.parametrize("day, week", [(1, 1), (7, 1), (8, 2), (14, 2), (15, 3), (29, 5), (31, 5)])
    def test_week_of_month_uses_seven_day_buckets(self, day, week):
        assert weekofmonth(date(2021, 1, day)) == week

    def test_age_counts_a_passed_birthday(self):
        today = date.today()
        assert age(date(today.year - 30, today.month, today.day)) == 30

    def test_age_does_not_count_a_birthday_not_yet_reached(self):
        today = date.today()
        if (today.month, today.day) == (12, 31):
            pytest.skip("on 31 December every birthday this year has passed")
        assert age(date(today.year - 30, 12, 31)) == 29

    def test_utc_to_local_is_the_same_instant(self):
        utc_dt = datetime(2021, 9, 22, 15, 14, 34)
        local = utc_to_local(utc_dt)
        assert local.tzinfo is not None
        assert local.astimezone(timezone.utc).replace(tzinfo=None) == utc_dt


class TestImitatedDelay:
    def test_an_inverted_range_is_warned_about_and_doubled(self, monkeypatch, caplog):
        calls = []
        monkeypatch.setattr("random.uniform", lambda lo, hi: calls.append((lo, hi)) or lo)
        monkeypatch.setattr("time.sleep", lambda seconds: None)
        with caplog.at_level(logging.WARNING, logger=dt_mod.logger.name):
            imitate_delay(5, 5, b_force=True)
        assert [r.getMessage() for r in caplog.records] == ["min_delay_seconds of 5 >= max_delay_seconds of 5!"]
        assert calls == [(5, 10)]

    def test_a_big_delay_multiplies_the_drawn_delay(self, monkeypatch):
        slept = []
        monkeypatch.setattr("random.uniform", lambda lo, hi: 2.0)
        monkeypatch.setattr("random.random", lambda: 0.0)
        monkeypatch.setattr("time.sleep", lambda seconds: slept.append(seconds))
        imitate_delay(1, 3, b_force=True, big_delay_prob=0.5, big_delay_multiplier=3)
        assert slept == [6.0]

    def test_without_a_big_delay_probability_the_drawn_delay_is_kept(self, monkeypatch):
        slept = []
        monkeypatch.setattr("random.uniform", lambda lo, hi: 2.0)
        monkeypatch.setattr("time.sleep", lambda seconds: slept.append(seconds))
        imitate_delay(1, 3, b_force=True, big_delay_prob=None)
        assert slept == [2.0]

    def test_an_elapsed_part_of_the_delay_is_not_slept_again(self, monkeypatch):
        slept = []
        monkeypatch.setattr("random.uniform", lambda lo, hi: 15.0)
        monkeypatch.setattr("time.sleep", lambda seconds: slept.append(seconds))
        imitate_delay(1, 20, last_call_ts=_utc_now() - timedelta(seconds=10), b_force=False)
        assert slept == [pytest.approx(5.0, abs=0.5)]

    def test_no_force_and_no_previous_call_neither_sleeps_nor_is_forced(self, monkeypatch):
        slept = []
        monkeypatch.setattr("time.sleep", lambda seconds: slept.append(seconds))
        before = _utc_now()
        stamp = imitate_delay(1, 2, b_force=False)
        assert slept == []
        assert before <= stamp <= _utc_now()


class TestSurvivorBoundaries:
    def test_the_suffix_is_appended_after_the_date_not_instead_of_it(self):
        stamp = (_utc_now() - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S.%f")
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \(120 мин\. тому назад\)", utc_ts_2_locstr(stamp, dst=60))

    def test_the_elapsed_count_is_rounded_down_not_to_the_nearest(self):
        stamp = (_utc_now() - timedelta(minutes=120, seconds=45)).strftime("%Y-%m-%dT%H:%M:%S.%f")
        assert utc_ts_2_locstr(stamp, dst=60).endswith(" (120 мин. тому назад)")

    def test_an_offset_is_normalised_only_at_the_end_of_the_string(self):
        assert read_timezoned_ts("2020-02-20T11:54:00+05:30x") == "2020-02-20T11:54:00+05:30x"

    def test_the_millisecond_clock_is_the_second_clock_times_a_thousand(self):
        assert abs(dt_mod.get_utc_unix_ts_milliseconds() - dt_mod.get_utc_unix_ts_seconds() * 1000) < 1000

    def test_the_default_big_delay_multiplier_is_ten(self, monkeypatch):
        slept = []
        monkeypatch.setattr("random.uniform", lambda lo, hi: 2.0)
        monkeypatch.setattr("random.random", lambda: 0.5)
        monkeypatch.setattr("time.sleep", lambda seconds: slept.append(seconds))
        imitate_delay(1, 3, b_force=True, big_delay_prob=1.0)
        assert slept == [20.0]

    def test_a_forced_call_with_no_default_force_still_sleeps(self, monkeypatch):
        slept = []
        monkeypatch.setattr("random.uniform", lambda lo, hi: 2.0)
        monkeypatch.setattr("time.sleep", lambda seconds: slept.append(seconds))
        imitate_delay(1, 3)
        assert slept == [2.0]

    def test_a_forced_call_with_a_previous_call_sleeps_only_the_rest_of_the_delay(self, monkeypatch):
        slept = []
        monkeypatch.setattr("random.uniform", lambda lo, hi: 15.0)
        monkeypatch.setattr("time.sleep", lambda seconds: slept.append(seconds))
        imitate_delay(1, 20, last_call_ts=_utc_now() - timedelta(seconds=10), b_force=True)
        assert slept == [pytest.approx(5.0, abs=0.5)]

    def test_a_delay_equal_to_the_elapsed_time_is_not_slept(self, monkeypatch):
        slept = []
        monkeypatch.setattr("random.uniform", lambda lo, hi: 0.0)
        monkeypatch.setattr("time.sleep", lambda seconds: slept.append(seconds))
        imitate_delay(0, 1, b_force=True)
        assert slept == []

    def test_the_sleep_is_logged_at_debug_with_its_length(self, monkeypatch, caplog):
        import logging

        monkeypatch.setattr("random.uniform", lambda lo, hi: 2.5)
        monkeypatch.setattr("time.sleep", lambda seconds: None)
        with caplog.at_level(logging.DEBUG, logger=dt_mod.logger.name):
            imitate_delay(1, 3, b_force=True)
        assert [r.getMessage() for r in caplog.records] == ["Sleeping 2.50 sec."]

    def test_an_inverted_range_names_the_values_in_their_own_order(self, monkeypatch, caplog):
        import logging

        monkeypatch.setattr("random.uniform", lambda lo, hi: lo)
        monkeypatch.setattr("time.sleep", lambda seconds: None)
        with caplog.at_level(logging.WARNING, logger=dt_mod.logger.name):
            imitate_delay(7, 3, b_force=True)
        assert [r.getMessage() for r in caplog.records] == ["min_delay_seconds of 7 >= max_delay_seconds of 3!"]

    def test_age_is_one_less_when_the_birthday_is_tomorrow_in_the_same_month(self):
        today = date.today()
        if (today.month, today.day) == (12, 31):
            pytest.skip("no later day in the year")
        dob = date(today.year - 30, today.month, today.day) + timedelta(days=1)
        assert age(dob) == 29


class TestSecondSurvivorRound:
    def test_a_zero_unit_gives_no_suffix_even_when_it_has_a_name(self):
        stamp = (_utc_now() - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S.%f")
        assert "тому назад" not in utc_ts_2_locstr(stamp, dst=0, dst_names={0: "zero"})

    def test_a_big_delay_probability_equal_to_the_draw_is_not_applied(self, monkeypatch):
        slept = []
        monkeypatch.setattr("random.uniform", lambda lo, hi: 2.0)
        monkeypatch.setattr("random.random", lambda: 0.5)
        monkeypatch.setattr("time.sleep", lambda seconds: slept.append(seconds))
        imitate_delay(1, 3, b_force=True, big_delay_prob=0.5)
        assert slept == [2.0]

    def test_a_missing_multiplier_falls_back_to_ten(self, monkeypatch):
        slept = []
        monkeypatch.setattr("random.uniform", lambda lo, hi: 2.0)
        monkeypatch.setattr("random.random", lambda: 0.0)
        monkeypatch.setattr("time.sleep", lambda seconds: slept.append(seconds))
        imitate_delay(1, 3, b_force=True, big_delay_prob=1.0, big_delay_multiplier=None)
        assert slept == [20.0]

    def test_the_debug_log_gives_the_part_of_the_delay_still_to_wait(self, monkeypatch, caplog):
        monkeypatch.setattr("random.uniform", lambda lo, hi: 5.0)
        monkeypatch.setattr("time.sleep", lambda seconds: None)
        with caplog.at_level(logging.DEBUG, logger=dt_mod.logger.name):
            imitate_delay(1, 20, last_call_ts=_utc_now() - timedelta(seconds=1), b_force=True)
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
        assert len(messages) == 1
        assert re.fullmatch(r"Sleeping 4\.\d\d sec\.", messages[0])

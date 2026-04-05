"""Extra tests for pythonlib.py — covers functions with no or minimal existing tests."""

import pytest
import os
import sys
from datetime import date, datetime, timezone


# ---------------------------------------------------------------------------
# show_methods
# ---------------------------------------------------------------------------

class TestShowMethods:
    def test_basic(self):
        from pyutilz.pythonlib import show_methods

        class Foo:
            def bar(self): ...
            def _private(self): ...
            def __dunder__(self): ...
            Baz = 1

        result = show_methods(Foo())
        assert "bar" in result
        assert "_private" in result
        # dunder excluded
        assert "__dunder__" not in result

    def test_uppercased_filter(self):
        from pyutilz.pythonlib import show_methods

        class Foo:
            def lower(self): ...
            Upper = 1

        result = show_methods(Foo(), uppercased=True)
        assert "Upper" in result
        assert "lower" not in result

    def test_empty_object(self):
        from pyutilz.pythonlib import show_methods

        result = show_methods(object())
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# keys_changed_enough
# ---------------------------------------------------------------------------

class TestKeysChangedEnough:
    def test_change_above_threshold(self):
        from pyutilz.pythonlib import keys_changed_enough

        assert keys_changed_enough(
            {"b": 180}, {"b": 200}, min_change_percent=10.0, key_contains="b"
        ) is True

    def test_change_below_threshold(self):
        from pyutilz.pythonlib import keys_changed_enough

        assert keys_changed_enough(
            {"b": 195}, {"b": 200}, min_change_percent=10.0, key_contains="b"
        ) is False

    def test_prev_zero_returns_true(self):
        from pyutilz.pythonlib import keys_changed_enough

        assert keys_changed_enough(
            {"x": 5}, {"x": 0}, min_change_percent=1.0, key_contains="x"
        ) is True

    def test_key_filter(self):
        from pyutilz.pythonlib import keys_changed_enough

        # "a" changed a lot but key_contains="b" so ignored
        assert keys_changed_enough(
            {"a": 999, "b": 200}, {"a": 1, "b": 200}, min_change_percent=10.0, key_contains="b"
        ) is False


# ---------------------------------------------------------------------------
# unpack_counter
# ---------------------------------------------------------------------------

class TestUnpackCounter:
    def test_basic(self):
        from pyutilz.pythonlib import unpack_counter

        assert unpack_counter([("a", 3), ("b", 2)]) == ["a", "b"]

    def test_empty(self):
        from pyutilz.pythonlib import unpack_counter

        assert unpack_counter([]) == []


# ---------------------------------------------------------------------------
# flatten_keys_to_set
# ---------------------------------------------------------------------------

class TestFlattenKeysToSet:
    def test_dict(self):
        from pyutilz.pythonlib import flatten_keys_to_set

        result = flatten_keys_to_set({"a": 1, "b": 2})
        assert "a:1" in result
        assert "b:2" in result

    def test_nested(self):
        from pyutilz.pythonlib import flatten_keys_to_set

        result = flatten_keys_to_set({"outer": {"inner": 5}})
        assert "inner:5" in result

    def test_list_of_strings(self):
        from pyutilz.pythonlib import flatten_keys_to_set

        result = flatten_keys_to_set(["hello", "world"])
        assert "hello" in result
        assert "world" in result

    def test_no_merge_symbol(self):
        from pyutilz.pythonlib import flatten_keys_to_set

        # Use numeric value (non-iterable) so both key and value are added separately
        result = flatten_keys_to_set({"k": 7}, dict_merge_symbol=None)
        assert "k" in result
        assert "7" in result

    def test_number_stringify(self):
        from pyutilz.pythonlib import flatten_keys_to_set

        result = flatten_keys_to_set(42, stringify=True)
        assert "42" in result


# ---------------------------------------------------------------------------
# get_or_warn
# ---------------------------------------------------------------------------

class TestGetOrWarn:
    def test_field_present(self):
        from pyutilz.pythonlib import get_or_warn

        assert get_or_warn({"x": 10}, "x", "test") == 10

    def test_field_missing(self):
        from pyutilz.pythonlib import get_or_warn

        assert get_or_warn({"x": 10}, "y", "test") is None


# ---------------------------------------------------------------------------
# get_partitioned_filepath
# ---------------------------------------------------------------------------

class TestGetPartitionedFilepath:
    def test_alpha(self):
        from pyutilz.pythonlib import get_partitioned_filepath

        result = get_partitioned_filepath("hello.txt", depth=2)
        assert result == os.sep.join(["h", "e"]) + os.sep

    def test_special_chars_replaced(self):
        from pyutilz.pythonlib import get_partitioned_filepath

        result = get_partitioned_filepath("!@file", depth=2)
        assert result == os.sep.join(["_", "_"]) + os.sep

    def test_depth_1(self):
        from pyutilz.pythonlib import get_partitioned_filepath

        result = get_partitioned_filepath("abc", depth=1)
        assert result == "a" + os.sep


# ---------------------------------------------------------------------------
# ensure_valid_filename
# ---------------------------------------------------------------------------

class TestEnsureValidFilename:
    @pytest.mark.parametrize("inp,expected_substr", [
        ("normal.txt", "normal.txt"),
        ("file/with:bad*chars", "file_with_bad_chars"),
        ("COM1.txt", "_.txt"),
    ])
    def test_cases(self, inp, expected_substr):
        from pyutilz.pythonlib import ensure_valid_filename

        result = ensure_valid_filename(inp)
        assert result == expected_substr

    def test_control_chars(self):
        from pyutilz.pythonlib import ensure_valid_filename

        for i in range(32):
            assert ensure_valid_filename(chr(i)) == "_"

    def test_max_length(self):
        from pyutilz.pythonlib import ensure_valid_filename

        result = ensure_valid_filename("a" * 500, max_length=10)
        assert len(result) == 10


# ---------------------------------------------------------------------------
# get_human_readable_set_size
# ---------------------------------------------------------------------------

class TestGetHumanReadableSetSize:
    @pytest.mark.parametrize("size,expected", [
        (100500, "100.5K"),
        (5_000_000, "5.0M"),
        (2_000_000_000, "2.0B"),
        (500, "500"),
    ])
    def test_sizes(self, size, expected):
        from pyutilz.pythonlib import get_human_readable_set_size

        assert get_human_readable_set_size(size) == expected


# ---------------------------------------------------------------------------
# suppress_stdout_stderr
# ---------------------------------------------------------------------------

class TestSuppressStdoutStderr:
    def test_suppresses_output(self, capsys):
        from pyutilz.pythonlib import suppress_stdout_stderr

        with suppress_stdout_stderr():
            print("should not appear")

        # After context manager, stdout restored
        print("visible")
        captured = capsys.readouterr()
        assert "should not appear" not in captured.out
        assert "visible" in captured.out

    def test_restores_on_exception(self):
        from pyutilz.pythonlib import suppress_stdout_stderr

        old_stdout = sys.stdout
        try:
            with suppress_stdout_stderr():
                raise ValueError("test")
        except ValueError:
            pass
        assert sys.stdout is old_stdout


# ---------------------------------------------------------------------------
# is_jupyter_notebook
# ---------------------------------------------------------------------------

class TestIsJupyterNotebook:
    def test_not_in_notebook(self):
        from pyutilz.pythonlib import is_jupyter_notebook

        assert is_jupyter_notebook() is False


# ---------------------------------------------------------------------------
# is_cuda_available
# ---------------------------------------------------------------------------

class TestIsCudaAvailable:
    def test_returns_bool(self):
        from pyutilz.pythonlib import is_cuda_available

        assert isinstance(is_cuda_available(), bool)


# ---------------------------------------------------------------------------
# read_timezoned_ts
# ---------------------------------------------------------------------------

class TestReadTimezonedTs:
    def test_removes_colon_from_tz(self):
        from pyutilz.pythonlib import read_timezoned_ts

        result = read_timezoned_ts("2020-02-20T11:54:00.000-07:00")
        assert result == "2020-02-20T11:54:00.000-0700"

    def test_positive_offset(self):
        from pyutilz.pythonlib import read_timezoned_ts

        result = read_timezoned_ts("2020-02-20T11:54:00.000+05:30")
        assert result == "2020-02-20T11:54:00.000+0530"

    def test_no_timezone(self):
        from pyutilz.pythonlib import read_timezoned_ts

        # String with no +/- should return as-is
        # Note: the function splits on - which exists in dates, so this tests the len<2 branch
        result = read_timezoned_ts("20200220")
        # No +/- at all → parts is None check fails because - is not in "20200220"
        # Actually "20200220" has no + or -, so parts stays None
        assert result == "20200220"


# ---------------------------------------------------------------------------
# utc_ts_2_locstr
# ---------------------------------------------------------------------------

class TestUtcTs2Locstr:
    def test_empty_input(self):
        from pyutilz.pythonlib import utc_ts_2_locstr

        assert utc_ts_2_locstr(None) == ""
        assert utc_ts_2_locstr("") == ""

    def test_formats_date(self):
        from pyutilz.pythonlib import utc_ts_2_locstr

        result = utc_ts_2_locstr("2021-09-22T15:14:34.532707")
        assert "2021" in result
        assert "09" in result

    def test_with_dst(self):
        from pyutilz.pythonlib import utc_ts_2_locstr

        result = utc_ts_2_locstr("2021-09-22T15:14:34.532707", dst=60)
        assert "мин" in result


# ---------------------------------------------------------------------------
# age
# ---------------------------------------------------------------------------

class TestAge:
    def test_known_age(self):
        from pyutilz.pythonlib import age

        today = date.today()
        dob = date(today.year - 30, today.month, today.day)
        assert age(dob) == 30

    def test_birthday_not_yet(self):
        from pyutilz.pythonlib import age

        today = date.today()
        # Birthday is tomorrow (roughly)
        future_month = today.month + 1 if today.month < 12 else 1
        future_year = today.year - 25 if today.month < 12 else today.year - 24
        dob = date(future_year, future_month, 1)
        result = age(dob)
        assert result >= 23  # reasonable range
        assert result <= 25


# ---------------------------------------------------------------------------
# weekofmonth (deeper tests)
# ---------------------------------------------------------------------------

class TestWeekofmonthExtra:
    @pytest.mark.parametrize("day,expected", [
        (1, 1),
        (6, 1),
        (7, 2),
        (8, 2),
        (14, 3),
        (15, 3),
        (28, 5),
        (29, 5),
    ])
    def test_various_days(self, day, expected):
        from pyutilz.pythonlib import weekofmonth

        assert weekofmonth(date(2024, 1, day)) == expected


# ---------------------------------------------------------------------------
# Additional parametrized tests for already-covered but shallow functions
# ---------------------------------------------------------------------------

class TestIsFloatExtra:
    @pytest.mark.parametrize("val,expected", [
        ("1,000", True),
        ("1,234.56", True),
        (None, False),
        ("inf", True),
        ("nan", True),
    ])
    def test_edge_cases(self, val, expected):
        from pyutilz.pythonlib import is_float

        assert is_float(val) is expected


class TestToFloatExtra:
    def test_comma_separated(self):
        from pyutilz.pythonlib import to_float

        assert to_float("1,000") == 1000.0
        assert to_float("1,234.5") == 1234.5


class TestCountTrailingZerosExtra:
    @pytest.mark.parametrize("number,precision,expected", [
        (1.30e-6, 8, 1),
        (1.00000, 5, 5),
        (1.23000, 5, 3),
    ])
    def test_cases(self, number, precision, expected):
        from pyutilz.pythonlib import count_trailing_zeros

        assert count_trailing_zeros(number, precision=precision) == expected


class TestBatchExtra:
    def test_uneven(self):
        from pyutilz.pythonlib import batch

        result = list(batch([1, 2, 3, 4, 5], n=2))
        assert len(result) == 3
        assert list(result[-1]) == [5]

    def test_n_larger_than_list(self):
        from pyutilz.pythonlib import batch

        result = list(batch([1, 2], n=10))
        assert len(result) == 1
        assert list(result[0]) == [1, 2]


class TestEnsureDictElemExtra:
    def test_does_not_overwrite(self):
        from pyutilz.pythonlib import ensure_dict_elem

        d = {"k": "original"}
        ensure_dict_elem(d, "k", "new")
        assert d["k"] == "original"


class TestSortDictByKeyExtra:
    def test_reverse(self):
        from pyutilz.pythonlib import sort_dict_by_key

        result = sort_dict_by_key({"a": 1, "c": 2, "b": 3}, reverse=True)
        assert list(result.keys()) == ["c", "b", "a"]


class TestFlattenKeysToDict:
    def test_nested_dict(self):
        from pyutilz.pythonlib import flatten_keys_to_dict

        result = flatten_keys_to_dict({"a": {"b": 1, "c": {"d": 2}}})
        assert result == {"b": 1, "d": 2}

    def test_list_of_dicts(self):
        from pyutilz.pythonlib import flatten_keys_to_dict

        result = flatten_keys_to_dict([{"x": 1}, {"y": 2}])
        assert result == {"x": 1, "y": 2}


class TestPopulateObjectFromDictExtra:
    def test_overwrites(self):
        from pyutilz.pythonlib import populate_object_from_dict

        class Obj:
            x = 0

        o = Obj()
        populate_object_from_dict(o, {"x": 99, "y": "new"})
        assert o.x == 99
        assert o.y == "new"


class TestFilterElementsByTypeExtra:
    def test_empty_dict(self):
        from pyutilz.pythonlib import filter_elements_by_type

        assert filter_elements_by_type({}, allowed_types=(int,)) == {}

    def test_empty_list(self):
        from pyutilz.pythonlib import filter_elements_by_type

        assert filter_elements_by_type([], allowed_types=(int,)) == []


class TestImitatedelayExtra:
    def test_returns_datetime(self):
        from pyutilz.pythonlib import imitate_delay

        result = imitate_delay(0, 0)
        assert isinstance(result, datetime)

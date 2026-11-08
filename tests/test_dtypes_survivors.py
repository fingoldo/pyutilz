"""Identifier-like text and column promotion pins for data.pandaslib.dtypes (mutation survivors, first cluster)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyutilz.data.pandaslib import dtypes as dt


class TestIdentifierLikeText:
    @pytest.mark.parametrize("value", [" 12", "12 ", "+5", "01234", "-0501"])
    def test_text_a_number_would_not_reproduce_is_flagged(self, value):
        assert dt._has_identifier_like_text(pd.Series([value])) is True

    @pytest.mark.parametrize("value", ["0", "10", "-0", "1 2", "7"])
    def test_plain_numbers_are_not_flagged(self, value):
        assert dt._has_identifier_like_text(pd.Series([value])) is False

    def test_nulls_are_ignored(self):
        assert dt._has_identifier_like_text(pd.Series([None, "5"])) is False


class TestPromoteColumnToInt64:
    def test_integer_text_becomes_int64(self):
        df = pd.DataFrame({"n": ["12", "7"]})
        old, ints = {"n": "object"}, []
        dt._promote_column_to_int64(df, "n", old, ints)
        assert df["n"].dtype == "int64"
        assert old["n"] == "int64"
        assert ints == ["n"]

    def test_fractional_values_are_rejected_and_the_column_is_untouched(self):
        df = pd.DataFrame({"n": ["3.5", "2"]})
        old, ints = {"n": "object"}, []
        with pytest.raises(ValueError):
            dt._promote_column_to_int64(df, "n", old, ints)
        assert df["n"].dtype == object
        assert old == {"n": "object"}
        assert ints == []

    def test_identifier_like_text_is_rejected_even_when_the_values_round_trip(self):
        df = pd.DataFrame({"zip": ["007", "42"]})
        old, ints = {"zip": "object"}, []
        with pytest.raises(ValueError, match="identifier-like"):
            dt._promote_column_to_int64(df, "zip", old, ints)
        assert df["zip"].dtype == object
        assert ints == []


class TestPromoteColumnToFloat64:
    def test_decimal_text_becomes_float64(self):
        df = pd.DataFrame({"x": ["1.5", "2"]})
        old, floats = {"x": "object"}, []
        dt._promote_column_to_float64(df, "x", old, floats)
        assert df["x"].dtype == "float64"
        assert old["x"] == "float64"
        assert floats == ["x"]

    def test_identifier_like_text_is_rejected_and_the_column_is_untouched(self):
        df = pd.DataFrame({"code": ["+1.5", "2"]})
        old, floats = {"code": "object"}, []
        with pytest.raises(ValueError, match="identifier-like"):
            dt._promote_column_to_float64(df, "code", old, floats)
        assert df["code"].dtype == object
        assert old == {"code": "object"}
        assert floats == []


class TestColumnsOfType:
    def test_a_column_matches_when_its_dtype_contains_a_listed_substring(self):
        df = pd.DataFrame({"a": [1], "b": [1.5], "c": ["x"]})
        assert dt.get_columns_of_type(df, ["int"]) == ["a"]
        assert dt.get_columns_of_type(df, ["float", "object"]) == ["b", "c"]

    def test_no_matching_type_gives_an_empty_list(self):
        df = pd.DataFrame({"a": [1]})
        assert dt.get_columns_of_type(df, ["bool"]) == []


def _promote(df, old, new=None, ints=None, floats=None, skip=(), max_categories=10, inplace=True, verbose=False):
    new = {} if new is None else new
    ints = [] if ints is None else ints
    floats = [] if floats is None else floats
    dt._promote_object_columns(df, old, new, ints, floats, list(skip), max_categories, inplace, verbose)
    return new, ints, floats


class TestPromoteObjectColumns:
    def test_integer_text_is_promoted_to_int64_first(self):
        df = pd.DataFrame({"n": ["12", "7"]})
        old = {"n": "object"}
        _, ints, floats = _promote(df, old)
        assert df["n"].dtype == "int64"
        assert ints == ["n"]
        assert floats == []

    def test_decimal_text_falls_back_to_float64(self):
        df = pd.DataFrame({"x": ["1.5", "2.25"]})
        old = {"x": "object"}
        _, ints, floats = _promote(df, old)
        assert df["x"].dtype == "float64"
        assert floats == ["x"]
        assert ints == []

    def test_fractional_objects_are_not_truncated_to_int64(self):
        df = pd.DataFrame({"v": pd.Series([2.5, 3.0], dtype=object)})
        old = {"v": "object"}
        _promote(df, old)
        assert df["v"].dtype == "float64"
        assert list(df["v"]) == [2.5, 3.0]

    def test_a_skipped_column_does_not_stop_the_later_columns(self):
        df = pd.DataFrame({"a": ["1"], "b": ["2"]})
        old = {"a": "object", "b": "object"}
        _promote(df, old, skip=["a"])
        assert df["a"].dtype == object
        assert df["b"].dtype == "int64"

    def test_an_object_dtype_entry_is_promoted_on_its_own(self):
        df = pd.DataFrame({"n": ["5"]})
        old = {"n": "object"}
        _promote(df, old)
        assert old["n"] == "int64"

    def test_a_non_text_dtype_is_not_touched(self):
        df = pd.DataFrame({"n": [1, 2]})
        old = {"n": "int64"}
        _promote(df, old)
        assert df["n"].dtype == "int64"
        assert old == {"n": "int64"}

    def test_high_cardinality_text_is_left_as_text_when_categories_exceed_the_cap(self):
        df = pd.DataFrame({"s": ["a", "b", "c"]})
        old = {"s": "object"}
        new, _, _ = _promote(df, old, max_categories=2)
        assert "s" not in new
        assert df["s"].dtype == object

    def test_low_cardinality_text_becomes_a_category(self):
        df = pd.DataFrame({"s": ["a", "b", "a"]})
        old = {"s": "object"}
        new, _, _ = _promote(df, old, max_categories=5)
        assert df["s"].dtype.name == "category"
        assert new["s"] == "category"


class TestPromotionMessages:
    def test_a_fractional_column_error_names_the_column(self):
        df = pd.DataFrame({"price": pd.Series([2.5, 3.0], dtype=object)})
        with pytest.raises(ValueError, match="Column price contains fractional values; cannot be int64"):
            dt._promote_column_to_int64(df, "price", {"price": "object"}, [])

    def test_an_identifier_error_names_the_column(self):
        df = pd.DataFrame({"zip": ["007"]})
        with pytest.raises(ValueError, match="Column zip holds identifier-like text"):
            dt._promote_column_to_int64(df, "zip", {"zip": "object"}, [])

    def test_a_float_identifier_error_names_the_column(self):
        df = pd.DataFrame({"code": ["+1.5"]})
        with pytest.raises(ValueError, match="Column code holds identifier-like text"):
            dt._promote_column_to_float64(df, "code", {"code": "object"}, [])


class TestTextDtypeNames:
    def test_a_str_dtype_name_alone_is_promoted(self):
        df = pd.DataFrame({"n": ["5"]})
        old = {"n": "str"}
        _promote(df, old)
        assert old["n"] == "int64"


class _RecordingLogger:
    def __init__(self):
        self.infos: list = []
        self.warnings: list = []

    def info(self, *args):
        self.infos.append(args)

    def warning(self, message):
        self.warnings.append(message)


@pytest.fixture
def recording_logger(monkeypatch):
    fake = _RecordingLogger()
    monkeypatch.setattr(dt, "logger", fake)
    return fake


def _float_df(values):
    return pd.DataFrame({"x": np.array(values, dtype=np.float64)})


class TestPromoteColumnToCategory:
    def test_few_distinct_values_are_marked_category_and_converted_in_place(self):
        df = pd.DataFrame({"s": ["a", "b", "a"]})
        new = {}
        dt._promote_column_to_category(df, "s", "object", new, 5, True, False)
        assert new == {"s": "category"}
        assert df["s"].dtype.name == "category"

    def test_not_inplace_fills_the_dtype_map_and_keeps_the_frame(self):
        df = pd.DataFrame({"s": ["a", "b", "a"]})
        new = {}
        dt._promote_column_to_category(df, "s", "object", new, 5, False, False)
        assert new == {"s": "category"}
        assert df["s"].dtype == object

    def test_too_many_distinct_values_leave_everything_alone(self):
        df = pd.DataFrame({"s": ["a", "b", "c"]})
        new = {}
        dt._promote_column_to_category(df, "s", "object", new, 2, True, False)
        assert new == {}
        assert df["s"].dtype == object

    def test_verbose_logs_the_conversion_with_column_and_type(self, recording_logger):
        df = pd.DataFrame({"s": ["a", "a"]})
        dt._promote_column_to_category(df, "s", "object", {}, 5, False, True)
        assert recording_logger.infos == [("%s %s->category", "s", "object")]

    def test_quiet_by_default(self, recording_logger):
        df = pd.DataFrame({"s": ["a", "a"]})
        dt._promote_column_to_category(df, "s", "object", {}, 5, False, False)
        assert recording_logger.infos == []

    def test_unhashable_values_are_warned_about_and_the_column_is_kept(self, recording_logger):
        df = pd.DataFrame({"s": [[1], [2]]})
        new = {}
        dt._promote_column_to_category(df, "s", "object", new, 5, True, False)
        assert new == {}
        assert len(recording_logger.warnings) == 1
        assert recording_logger.warnings[0].startswith("Could not convert to category column s, leaving its dtype unchanged:")


class TestFindPossiblyIntegerFloats:
    def test_whole_numbers_without_nans_are_selected(self):
        df = pd.DataFrame({"w": [1.0, 2.0], "f": [1.5, 2.0], "n": [1.0, np.nan]})
        assert dt._find_possibly_integer_floats(df, ["w", "f", "n"]) == ["w"]

    def test_a_column_with_some_nans_is_excluded_even_when_its_values_are_whole(self):
        df = pd.DataFrame({"n": [1.0, np.nan]})
        assert dt._find_possibly_integer_floats(df, ["n"]) == []

    def test_the_progress_bar_is_labelled_and_transient(self, monkeypatch):
        seen: list = []

        def fake_tqdmu(iterable, **kwargs):
            seen.append(kwargs)
            return iterable

        monkeypatch.setattr(dt, "tqdmu", fake_tqdmu)
        df = pd.DataFrame({"w": [1.0]})
        dt._find_possibly_integer_floats(df, ["w"])
        assert seen == [{"desc": "checking float2int", "leave": False}]


class TestBuildConversions:
    def test_uint_pass_comes_before_the_signed_pass(self):
        df = pd.DataFrame({"w": [1.0]})
        conv = dt._build_conversions(df, ["a"], [], True, False, False)
        assert conv == [(["a"], "uint"), (["a"], "int")]

    def test_signed_only_pass_when_uint_is_off(self):
        df = pd.DataFrame({"w": [1.0]})
        conv = dt._build_conversions(df, ["a"], [], False, False, False)
        assert conv == [(["a"], "int")]

    def test_whole_float_columns_get_int_conversions_appended(self):
        df = pd.DataFrame({"w": [1.0, 2.0], "f": [1.5, 2.0]})
        conv = dt._build_conversions(df, [], ["w", "f"], False, True, False)
        assert conv == [([], "int"), (["w"], "int")]

    def test_float_to_float_appends_the_full_float_list(self):
        df = pd.DataFrame({"f": [1.5]})
        conv = dt._build_conversions(df, [], ["f"], False, False, True)
        assert conv == [([], "int"), (["f"], "float")]


class TestFloatPrecisionOk:
    def test_a_value_within_float32_range_and_precision_passes(self):
        assert dt._float_precision_ok(_float_df([1.5, 2.25]), "x", 32, "float64", {}, False, False) is True

    def test_a_value_losing_float32_precision_is_rejected_by_the_mantissa_check(self):
        assert dt._float_precision_ok(_float_df([123456.789]), "x", 32, "float64", {}, False, False) is False

    def test_the_mantissa_check_rejects_large_identifier_like_floats(self):
        assert dt._float_precision_ok(_float_df([2.205001270000e9]), "x", 32, "float64", {}, False, False) is False

    def test_the_exact_round_trip_rejects_a_value_the_mantissa_check_accepts(self):
        assert dt._float_precision_ok(_float_df([0.1]), "x", 32, "float64", {}, False, False) is True
        assert dt._float_precision_ok(_float_df([0.1]), "x", 32, "float64", {}, False, True) is False

    def test_an_exact_value_passes_the_round_trip(self):
        assert dt._float_precision_ok(_float_df([3.0]), "x", 32, "float64", {}, False, True) is True

    def test_the_precision_loss_is_logged_when_verbose(self, recording_logger):
        dt._float_precision_ok(_float_df([123456.789]), "x", 32, "float64", {}, True, False)
        assert recording_logger.infos == [("Column %s can't be converted to float%s due to precision loss.", "x", 32)]

    def test_the_round_trip_failure_is_logged_when_verbose(self, recording_logger):
        dt._float_precision_ok(_float_df([0.1]), "x", 32, "float64", {}, True, True)
        assert recording_logger.infos == [("Column %s can't be converted to float%s: float64 round-trip is not exact.", "x", 32)]

    def test_the_precision_loss_is_silent_by_default(self, recording_logger):
        dt._float_precision_ok(_float_df([123456.789]), "x", 32, "float64", {}, False, False)
        assert recording_logger.infos == []


class TestPrecisionMantissaKillers:
    def test_a_tiny_value_survives_the_float32_mantissa_rounding(self):
        assert dt._float_precision_ok(_float_df([1e-20]), "x", 32, "float64", {}, False, False) is True

    def test_a_float16_mantissa_check_accepts_a_value_with_a_long_fraction(self):
        assert dt._float_precision_ok(_float_df([268139.9586948864]), "x", 16, "float64", {}, False, False) is True

    def test_nan_entries_do_not_fail_the_exact_round_trip(self):
        assert dt._float_precision_ok(_float_df([1.5, np.nan]), "x", 32, "float64", {}, False, True) is True

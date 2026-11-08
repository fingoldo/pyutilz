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

    def warning(self, message, *args):
        self.warnings.append(message if not args else (message, args))


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


class _Range:
    def __init__(self, lo, hi):
        self.min = lo
        self.max = hi


def _pick(values, type_name, cur_power, col="x", float_fields=(), ensure=True, exact=False, verbose=False, src="int64"):
    df = pd.DataFrame({col: values})
    powers = [16, 32, 64] if type_name == "float" else [8, 16, 32, 64]
    if type_name == "float":
        topvals = [np.finfo("float" + str(p)) for p in powers]
    else:
        topvals = [np.iinfo(type_name + str(p)) for p in powers]
    return dt._select_target_power(df, col, _Range(min(values), max(values)), type_name, powers, topvals, cur_power, src, list(float_fields), {}, ensure, exact, verbose)


class TestSelectTargetPower:
    def test_the_smallest_fitting_power_is_chosen(self):
        assert _pick([0, 100], "int", 64) == 8

    def test_a_range_beyond_the_smaller_type_moves_up(self):
        assert _pick([0, 200], "int", 64) == 16

    def test_a_range_on_the_upper_limit_still_fits_the_power(self):
        assert _pick([-128, 127], "int", 64) == 8
        assert _pick([-128, 128], "int", 64) == 16

    def test_a_range_below_the_lower_limit_does_not_fit(self):
        assert _pick([-129, 0], "int", 64) == 16

    def test_no_power_smaller_than_the_current_one_is_offered(self):
        assert _pick([0, 100], "int", 8) is None

    def test_negative_values_do_not_fit_an_unsigned_type(self):
        assert _pick([-1, 100], "uint", 64) is None

    def test_a_float_field_is_considered_for_an_int_conversion_at_its_own_power(self):
        assert _pick([0, 100], "int", 8, col="x", float_fields=["x"]) == 8
        assert _pick([0, 100], "int", 8, col="x") is None
        assert _pick([0.0, 100.0], "int", 64, col="x", float_fields=["x"]) == 8

    def test_the_float_precision_check_rejects_a_value_that_does_not_round_trip(self):
        assert _pick([0.1], "float", 64, src="float64", exact=True) is None

    def test_the_exact_flag_is_not_confused_with_verbose(self):
        assert _pick([0.1], "float", 64, src="float64", exact=True, verbose=False) is None
        assert _pick([0.1], "float", 64, src="float64", exact=False, verbose=True) == 16


class TestRecordConversion:
    def test_an_int_conversion_is_recorded_and_tracked_as_converted(self):
        df = pd.DataFrame({"x": [1, 2]})
        new, uint_fields = {}, []
        dt._record_conversion(df, "x", "int", 8, {"x": "int64"}, new, uint_fields, False, False)
        assert new == {"x": "int8"}
        assert uint_fields == ["x"]

    def test_a_uint_conversion_is_tracked_as_converted(self):
        df = pd.DataFrame({"x": [1, 2]})
        new, uint_fields = {}, []
        dt._record_conversion(df, "x", "uint", 16, {"x": "int64"}, new, uint_fields, False, False)
        assert new == {"x": "uint16"}
        assert uint_fields == ["x"]

    def test_a_float_conversion_is_not_tracked_as_converted(self):
        df = pd.DataFrame({"x": [1.5]})
        new, uint_fields = {}, []
        dt._record_conversion(df, "x", "float", 32, {"x": "float64"}, new, uint_fields, False, False)
        assert new == {"x": "float32"}
        assert uint_fields == []

    def test_inplace_casts_the_column_now(self):
        df = pd.DataFrame({"x": [1, 2]})
        dt._record_conversion(df, "x", "int", 8, {"x": "int64"}, {}, [], False, True)
        assert df["x"].dtype == "int8"

    def test_not_inplace_leaves_the_frame_alone(self):
        df = pd.DataFrame({"x": [1, 2]})
        dt._record_conversion(df, "x", "int", 8, {"x": "int64"}, {}, [], False, False)
        assert df["x"].dtype == "int64"

    def test_verbose_logs_the_column_its_source_type_the_target_and_the_power(self, recording_logger):
        df = pd.DataFrame({"x": [1, 2]})
        dt._record_conversion(df, "x", "int", 8, {"x": "int64"}, {}, [], True, False)
        assert recording_logger.infos == [("%s [%s]->[%s%s]", "x", "int64", "int", 8)]

    def test_quiet_by_default(self, recording_logger):
        df = pd.DataFrame({"x": [1, 2]})
        dt._record_conversion(df, "x", "int", 8, {"x": "int64"}, {}, [], False, False)
        assert recording_logger.infos == []


class TestFloatFieldStopsAtItsOwnPower:
    def test_a_float_column_out_of_range_for_every_smaller_float_type_stops_at_its_current_power(self):
        assert _pick([1e300], "float", 64, src="float64", float_fields=["x"], ensure=False) is None


class TestOptimizeDtypesDefaults:
    def test_the_default_call_optimizes_the_frame_in_place(self):
        df = pd.DataFrame({"n": [1, 2, 3]})
        dt.optimize_dtypes(df)
        assert df["n"].dtype == "uint8"

    def test_the_default_call_is_quiet(self, recording_logger):
        df = pd.DataFrame({"n": [1, 2, 3]})
        dt.optimize_dtypes(df, inplace=False)
        assert recording_logger.infos == []

    def test_the_default_precision_check_accepts_a_decimal_that_float32_holds_approximately(self):
        df = pd.DataFrame({"x": [0.1, 0.2]})
        out = dt.optimize_dtypes(df, inplace=False)
        assert out["x"].dtype == "float32"

    def test_the_exact_round_trip_rejects_the_decimal_when_requested(self):
        df = pd.DataFrame({"x": [0.1, 0.2]})
        out = dt.optimize_dtypes(df, inplace=False, exact_float_roundtrip=True)
        assert out["x"].dtype == "float64"


class TestOptimizeDtypesPipeline:
    def test_an_unsigned_source_column_is_reduced_without_a_crash(self):
        df = pd.DataFrame({"u": np.array([1, 2], dtype=np.uint64)})
        out = dt.optimize_dtypes(df, inplace=False)
        assert out["u"].dtype == "uint8"

    def test_a_numeric_text_column_is_promoted_and_then_reduced(self):
        df = pd.DataFrame({"n": ["12", "7"]})
        out = dt.optimize_dtypes(df, inplace=False)
        assert out["n"].dtype == "uint8"

    def test_a_skipped_text_column_stays_text(self):
        df = pd.DataFrame({"s": ["12", "7"]})
        out = dt.optimize_dtypes(df, inplace=False, skip_columns=["s"])
        assert out["s"].dtype == object

    def test_a_low_cardinality_text_column_becomes_a_category_in_place(self):
        df = pd.DataFrame({"c": ["a", "b", "a"]})
        dt.optimize_dtypes(df)
        assert df["c"].dtype.name == "category"

    def test_the_exact_flag_reaches_the_precision_check_as_passed(self):
        df = pd.DataFrame({"x": [0.1, 0.2]})
        out = dt.optimize_dtypes(df, inplace=False, exact_float_roundtrip=True)
        assert out["x"].dtype == "float64"

    def test_the_size_reduction_pass_is_labelled_and_transient(self, monkeypatch):
        seen: list = []

        def fake_tqdmu(iterable, **kwargs):
            seen.append(kwargs)
            return iterable

        monkeypatch.setattr(dt, "tqdmu", fake_tqdmu)
        dt.optimize_dtypes(pd.DataFrame({"n": [1, 2]}), inplace=False)
        assert {"desc": "size reduction", "leave": False} in seen


class TestDecimalTextColumns:
    def test_decimal_text_is_promoted_to_float_and_downcast_to_float32(self):
        df = pd.DataFrame({"x": ["1.5", "2.25"]})
        out = dt.optimize_dtypes(df, inplace=False)
        assert out["x"].dtype == "float32"
        assert list(out["x"]) == [1.5, 2.25]


class TestClassifyColumnTypes:
    def test_an_explicit_dtype_or_a_frame_and_column_is_required(self):
        with pytest.raises(AssertionError):
            dt.classify_column_types(col="x")

    @pytest.mark.parametrize(
        "series, numeric",
        [
            (pd.Series([1, 2], dtype="int64"), True),
            (pd.Series([1.5], dtype="float64"), True),
            (pd.Series([True]), False),
            (pd.Series(["a"]), False),
            (pd.Series(pd.to_datetime(["2020-01-01"])), False),
            (pd.Series(["a"], dtype="category"), False),
        ],
    )
    def test_only_plain_numeric_dtypes_are_numeric(self, series, numeric):
        df = pd.DataFrame({"x": series})
        assert dt.classify_column_types(df, "x")[4] is numeric

    def test_a_boolean_column_is_flagged_boolean_and_not_numeric(self):
        df = pd.DataFrame({"x": [True, False]})
        assert dt.classify_column_types(df, "x") == (True, False, False, False, False)


class TestGetDfMemoryConsumption:
    def test_an_unsupported_type_raises_with_its_name(self):
        with pytest.raises(TypeError, match="Unsupported dataframe type: "):
            dt.get_df_memory_consumption([1, 2])

    def test_a_pandas_frame_reports_its_bytes(self):
        df = pd.DataFrame({"x": np.arange(10, dtype=np.int64)})
        assert dt.get_df_memory_consumption(df, deep=False) == float(df.memory_usage(deep=False).sum())


class TestFloat32IntegerLimit:
    def test_the_limit_is_two_to_the_twenty_fourth(self):
        assert dt.FLOAT32_EXACT_INT_LIMIT == 16777216


class TestLossyIntegerColumns:
    def test_values_above_the_limit_are_lossy(self):
        df = pd.DataFrame({"x": [0, 2**25]})
        assert dt._int_cols_beyond_float32_mantissa(df, ["x"]) == ["x"]

    def test_values_exactly_at_the_limit_are_not_lossy(self):
        df = pd.DataFrame({"x": [0, 2**24], "y": [-(2**24), 0]})
        assert dt._int_cols_beyond_float32_mantissa(df, ["x", "y"]) == []

    def test_a_large_negative_value_is_lossy_and_a_small_negative_is_not(self):
        assert dt._int_cols_beyond_float32_mantissa(pd.DataFrame({"x": [-(2**25), 1]}), ["x"]) == ["x"]
        assert dt._int_cols_beyond_float32_mantissa(pd.DataFrame({"x": [-1, 0]}), ["x"]) == []

    def test_missing_values_are_ignored(self):
        df = pd.DataFrame({"x": pd.array([pd.NA, 5], dtype="Int64")})
        assert dt._int_cols_beyond_float32_mantissa(df, ["x"]) == []

    def test_the_warning_names_the_lossy_columns(self, recording_logger):
        dt._warn_lossy_float32_int_columns(pd.DataFrame({"id": [16777217, 1]}), ["id"])
        assert recording_logger.warnings == [
            (
                "ensure_dataframe_float32_convertability: integer column(s) %s have values beyond 2**24 -- Float32 cast will lose exact-integer precision.",
                (["id"],),
            )
        ]

    def test_no_warning_when_every_column_is_exact(self, recording_logger):
        dt._warn_lossy_float32_int_columns(pd.DataFrame({"id": [1, 2]}), ["id"])
        assert recording_logger.warnings == []


class TestArrowTypeHelpers:
    def test_an_arrow_dtype_exposes_its_pyarrow_type(self):
        pa = pytest.importorskip("pyarrow")
        assert dt._arrow_type(pd.ArrowDtype(pa.int64())) == pa.int64()

    def test_a_numpy_dtype_has_no_arrow_type(self):
        assert dt._arrow_type(np.dtype("int64")) is None

    def test_integer_and_float_arrow_types_are_told_apart(self):
        pa = pytest.importorskip("pyarrow")
        assert dt._arrow_is_integer(pd.ArrowDtype(pa.int64())) is True
        assert dt._arrow_is_float(pd.ArrowDtype(pa.float64())) is True
        assert dt._arrow_is_integer(pd.ArrowDtype(pa.float64())) is False
        assert dt._arrow_is_float(pd.ArrowDtype(pa.int64())) is False

    def test_numpy_dtypes_are_neither_arrow_integers_nor_arrow_floats(self):
        assert dt._arrow_is_integer(np.dtype("int64")) is False
        assert dt._arrow_is_float(np.dtype("float64")) is False


class TestEnsureFloat32Convertability:
    def test_each_numpy_numeric_dtype_is_cast_to_float32(self):
        df = pd.DataFrame(
            {
                "a": np.array([1, 2], dtype=np.uint32),
                "b": np.array([1, 2], dtype=np.int32),
                "c": np.array([1, 2], dtype=np.int64),
                "d": np.array([1, 2], dtype=np.uint64),
                "e": np.array([1.5, 2.5], dtype=np.float64),
            }
        )
        out = dt.ensure_dataframe_float32_convertability(df)
        assert set(out.dtypes.astype(str)) == {"float32"}

    def test_a_single_numeric_column_is_converted(self):
        out = dt.ensure_dataframe_float32_convertability(pd.DataFrame({"x": np.array([1.5], dtype=np.float64)}))
        assert out["x"].dtype == "float32"

    def test_the_input_frame_is_left_unchanged(self):
        df = pd.DataFrame({"x": np.array([1.5], dtype=np.float64)})
        dt.ensure_dataframe_float32_convertability(df)
        assert df["x"].dtype == "float64"

    def test_a_frame_without_numeric_columns_is_returned_unchanged_and_quiet(self, recording_logger):
        df = pd.DataFrame({"s": ["a"]})
        out = dt.ensure_dataframe_float32_convertability(df, verbose=True)
        assert list(out["s"]) == ["a"]
        assert recording_logger.infos == []

    def test_verbose_reports_the_number_of_numeric_columns(self, recording_logger):
        df = pd.DataFrame({"a": [1.5], "b": [2.5]})
        dt.ensure_dataframe_float32_convertability(df, verbose=True)
        assert ("Converting %s numeric columns to float32", 2) in recording_logger.infos

    def test_a_large_integer_column_warns_and_is_still_converted(self, recording_logger):
        df = pd.DataFrame({"id": np.array([16777217], dtype=np.int64)})
        out = dt.ensure_dataframe_float32_convertability(df)
        assert out["id"].dtype == "float32"
        assert recording_logger.warnings and recording_logger.warnings[0][1] == (["id"],)

    def test_a_polars_frame_is_cast_through_the_polars_helper(self):
        pl = pytest.importorskip("polars")
        out = dt.ensure_dataframe_float32_convertability(pl.DataFrame({"x": [1.5]}))
        assert out["x"].dtype == pl.Float32

    def test_an_arrow_integer_column_becomes_float_pyarrow(self):
        pa = pytest.importorskip("pyarrow")
        df = pd.DataFrame({"x": pd.array([1, 2], dtype=pd.ArrowDtype(pa.int16()))})
        out = dt.ensure_dataframe_float32_convertability(df)
        assert out["x"].dtype == pd.ArrowDtype(pa.float32())

    def test_an_arrow_string_column_is_left_alone(self):
        pa = pytest.importorskip("pyarrow")
        df = pd.DataFrame({"s": pd.array(["a"], dtype=pd.ArrowDtype(pa.string()))})
        out = dt.ensure_dataframe_float32_convertability(df)
        assert list(out["s"]) == ["a"]
        assert out["s"].dtype == pd.ArrowDtype(pa.string())

    def test_a_mixed_frame_converts_its_arrow_integer_and_warns_about_it(self, recording_logger):
        pa = pytest.importorskip("pyarrow")
        df = pd.DataFrame(
            {
                "n": np.array([1.5], dtype=np.float64),
                "big": pd.array([16777217], dtype=pd.ArrowDtype(pa.int64())),
            }
        )
        out = dt.ensure_dataframe_float32_convertability(df)
        assert out["n"].dtype == "float32"
        assert out["big"].dtype == "float32"
        assert recording_logger.warnings and recording_logger.warnings[0][1] == (["big"],)

    def test_verbose_logs_each_arrow_conversion(self, recording_logger):
        pa = pytest.importorskip("pyarrow")
        df = pd.DataFrame({"x": pd.array([1, 2], dtype=pd.ArrowDtype(pa.int16()))})
        dt.ensure_dataframe_float32_convertability(df, verbose=True)
        assert any(info[0] == "Converting PyArrow column '%s' (%s) -> float32[pyarrow]" and info[1] == "x" for info in recording_logger.infos)


class TestFloat32ConversionDefaults:
    def test_the_default_call_is_quiet(self, recording_logger):
        df = pd.DataFrame({"a": [1.5], "b": [2.5]})
        dt.ensure_dataframe_float32_convertability(df)
        assert recording_logger.infos == []


class TestMixedArrowAndNumpyFrames:
    def test_an_arrow_int16_column_is_converted_alongside_a_numpy_float_column(self):
        pa = pytest.importorskip("pyarrow")
        df = pd.DataFrame(
            {
                "n": np.array([1.5], dtype=np.float64),
                "s": pd.array([7], dtype=pd.ArrowDtype(pa.int16())),
            }
        )
        out = dt.ensure_dataframe_float32_convertability(df)
        assert out["n"].dtype == "float32"
        assert out["s"].dtype == pd.ArrowDtype(pa.float32())

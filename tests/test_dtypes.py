"""Behaviour of ``pyutilz.data.pandaslib.dtypes``: columns are grouped and classified by dtype, categorical columns are
located by position, columns are cast from a mapping, float64 columns become float32 on a copy, and ``optimize_dtypes``
shrinks numeric columns without touching the caller's frame when ``inplace=False``. Identifier-like text (leading zeros,
surrounding whitespace, a leading sign) is never rewritten as a number. Frames are small and built in memory."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pyutilz.data.pandaslib import dtypes


def _frame():
    return pd.DataFrame(
        {
            "i": np.array([1, 2, 3], dtype=np.int64),
            "f": np.array([1.5, 2.5, 3.5]),
            "s": pd.Series(["a", "b", "a"], dtype=object),
            "c": pd.Categorical(["x", "y", "x"]),
            "b": [True, False, True],
        }
    )


class TestGrouping:
    def test_columns_are_grouped_by_their_dtype_name(self):
        groups = dtypes.group_columns_by_dtype(_frame())
        assert groups["int64"] == {"i"}
        assert groups["float64"] == {"f"}
        assert groups["object"] == {"s"}
        assert groups["category"] == {"c"}
        assert groups["bool"] == {"b"}

    def test_columns_of_a_named_type_are_found_by_substring(self):
        assert dtypes.get_columns_of_type(_frame(), ["int"]) == ["i"]
        assert dtypes.get_columns_of_type(_frame(), ["category"]) == ["c"]

    def test_no_columns_of_a_missing_type_gives_an_empty_list(self):
        assert dtypes.get_columns_of_type(_frame(), ["datetime"]) == []

    def test_categorical_columns_are_split_by_position_with_their_categories(self):
        non_cat, cat, values = dtypes.get_categorical_columns_indices(_frame())
        assert cat == [3]
        assert non_cat == [0, 1, 2, 4]
        assert values == {"c": ["x", "y"]}


class TestClassify:
    def test_a_float_column_is_numeric(self):
        assert dtypes.classify_column_types(df=_frame(), col="f") == (False, False, False, False, True)

    def test_a_string_column_is_object_like(self):
        assert dtypes.classify_column_types(df=_frame(), col="s") == (False, True, False, False, False)

    def test_a_categorical_column_is_categorical(self):
        assert dtypes.classify_column_types(df=_frame(), col="c") == (False, False, False, True, False)

    def test_a_boolean_column_is_boolean(self):
        assert dtypes.classify_column_types(df=_frame(), col="b") == (True, False, False, False, False)

    def test_an_explicit_dtype_is_classified_without_a_frame(self):
        assert dtypes.classify_column_types(dtype=np.dtype("datetime64[ns]")) == (False, False, True, False, False)


class TestSetColumnTypes:
    def test_listed_columns_are_cast_in_place(self):
        df = _frame()
        dtypes.set_df_columns_types(df, {np.int32: ["i"]})
        assert df["i"].dtype == np.int32

    def test_a_listed_column_missing_from_the_frame_is_skipped(self):
        df = _frame()
        dtypes.set_df_columns_types(df, {np.float32: ["absent", "f"]})
        assert df["f"].dtype == np.float32
        assert "absent" not in df.columns


class TestFloatToFloat32:
    def test_float64_columns_become_float32_on_a_new_frame(self):
        df = pd.DataFrame({"f": [1.5]})
        out = dtypes.convert_float64_to_float32(df)
        assert out["f"].dtype == np.float32
        assert df["f"].dtype == np.float64

    def test_integer_columns_are_left_alone(self):
        out = dtypes.convert_float64_to_float32(pd.DataFrame({"f": [1.5], "i": [2]}))
        assert out["i"].dtype == np.int64


class TestMemory:
    def test_deep_measurement_counts_object_contents_too(self):
        df = pd.DataFrame({"s": pd.Series(["x" * 100] * 3, dtype=object)})
        assert dtypes.get_df_memory_consumption(df) > dtypes.get_df_memory_consumption(df, deep=False)


class TestIdentifierLikeText:
    def test_a_leading_zero_marks_an_identifier(self):
        assert dtypes._has_identifier_like_text(pd.Series(["007", "123"])) is True

    def test_surrounding_whitespace_marks_an_identifier(self):
        assert dtypes._has_identifier_like_text(pd.Series([" 12"])) is True

    def test_plain_numbers_are_not_identifiers(self):
        assert dtypes._has_identifier_like_text(pd.Series([1, 2])) is False
        assert dtypes._has_identifier_like_text(pd.Series(["1.5", "2"])) is False


class TestOptimizeDtypes:
    def test_small_integers_are_shrunk_without_touching_the_input(self):
        df = pd.DataFrame({"i": np.array([1, 2, 3], dtype=np.int64)})
        out = dtypes.optimize_dtypes(df, verbose=False, inplace=False)
        assert out["i"].dtype.itemsize < 8
        assert df["i"].dtype == np.int64

    def test_a_leading_zero_column_keeps_its_text(self):
        df = pd.DataFrame({"zip": pd.Series(["01234", "02345", "01234"], dtype=object)})
        out = dtypes.optimize_dtypes(df, verbose=False, inplace=False, max_categories=None)
        assert out["zip"].dtype == object
        assert out["zip"].tolist() == ["01234", "02345", "01234"]


class TestOptimizeDtypeSwitches:
    @staticmethod
    def _frame():
        return pd.DataFrame(
            {
                "i": np.array([1, 2, 3], dtype=np.int64),
                "f": np.array([1.0, 2.0, 3.0]),
                "s": pd.Series(["a", "a", "b"], dtype=object),
                "g": np.array([1.5, 2.25, 3.125]),
            }
        )

    @staticmethod
    def _dtypes(df):
        return {col: str(df[col].dtype) for col in df.columns}

    def test_the_default_shrinks_integers_floats_and_low_cardinality_text(self):
        out = dtypes.optimize_dtypes(self._frame(), verbose=False, inplace=False)
        assert self._dtypes(out) == {"i": "uint8", "f": "uint8", "s": "category", "g": "float32"}

    def test_reduce_size_off_leaves_every_column_as_it_was(self):
        out = dtypes.optimize_dtypes(self._frame(), verbose=False, inplace=False, reduce_size=False)
        assert self._dtypes(out) == {"i": "int64", "f": "float64", "s": "category", "g": "float64"}

    def test_float_to_int_off_keeps_integral_floats_as_floats(self):
        out = dtypes.optimize_dtypes(self._frame(), verbose=False, inplace=False, float_to_int=False)
        assert out["f"].dtype == np.float32

    def test_float_to_float_off_keeps_fractional_floats_as_float64(self):
        out = dtypes.optimize_dtypes(self._frame(), verbose=False, inplace=False, float_to_float=False)
        assert out["g"].dtype == np.float64

    def test_text_above_the_category_limit_stays_object(self):
        out = dtypes.optimize_dtypes(self._frame(), verbose=False, inplace=False, max_categories=1)
        assert out["s"].dtype == object

    def test_no_category_limit_leaves_text_alone(self):
        out = dtypes.optimize_dtypes(self._frame(), verbose=False, inplace=False, max_categories=None)
        assert out["s"].dtype == object

    def test_a_skipped_column_keeps_its_dtype(self):
        out = dtypes.optimize_dtypes(self._frame(), verbose=False, inplace=False, skip_columns=["f"])
        assert out["f"].dtype == np.float64

    def test_use_uint_off_picks_signed_integers(self):
        out = dtypes.optimize_dtypes(self._frame(), verbose=False, inplace=False, use_uint=False)
        assert out["i"].dtype == np.int8

    def test_inplace_true_changes_the_callers_frame_and_returns_it(self):
        df = self._frame()
        returned = dtypes.optimize_dtypes(df, verbose=False, inplace=True)
        assert returned is df
        assert df["i"].dtype == np.uint8

    def test_inplace_false_leaves_the_callers_frame_untouched(self):
        df = self._frame()
        dtypes.optimize_dtypes(df, verbose=False, inplace=False)
        assert df["i"].dtype == np.int64

    def test_verbose_reports_the_new_dtypes_at_info(self, caplog):
        with caplog.at_level("INFO"):
            dtypes.optimize_dtypes(self._frame(), verbose=True, inplace=False)
        assert any(rec.levelname == "INFO" and rec.getMessage().startswith("Going to use the following new dtypes") for rec in caplog.records)

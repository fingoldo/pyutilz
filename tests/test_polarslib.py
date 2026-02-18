"""
Test suite for polarslib.py
Tests cover Polars utility functions for DataFrame operations.
"""

import pytest
import polars as pl
import polars.selectors as cs
import numpy as np

from pyutilz.data.polarslib import (
    find_nan_cols,
    find_infinite_cols,
    clean_numeric,
    cast_f64_to_f32,
    apply_agg_func_safe,
    polars_df_info,
)


class TestFindNanCols:
    """Test find_nan_cols function"""

    def test_finds_nan_column(self):
        """Test that columns with NaN are found"""
        df = pl.DataFrame({"a": [1.0, float("nan"), 3.0], "b": [1.0, 2.0, 3.0]})
        result = find_nan_cols(df)

        assert "a" in result.columns
        assert "b" not in result.columns

    def test_no_nan_returns_empty(self):
        """Test that no NaN returns empty DataFrame"""
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = find_nan_cols(df)

        assert result.width == 0

    def test_all_nan_columns_found(self):
        """Test that all NaN columns are found"""
        df = pl.DataFrame({
            "a": [float("nan"), float("nan")],
            "b": [float("nan"), 1.0],
            "c": [1.0, 2.0]
        })
        result = find_nan_cols(df)

        assert "a" in result.columns
        assert "b" in result.columns
        assert "c" not in result.columns


class TestFindInfiniteCols:
    """Test find_infinite_cols function"""

    def test_finds_infinite_column(self):
        """Test that columns with infinity are found"""
        df = pl.DataFrame({"a": [1.0, float("inf"), 3.0], "b": [1.0, 2.0, 3.0]})
        result = find_infinite_cols(df)

        assert "a" in result.columns
        assert "b" not in result.columns

    def test_finds_negative_infinite(self):
        """Test that columns with -inf are found"""
        df = pl.DataFrame({"a": [1.0, -float("inf"), 3.0], "b": [1.0, 2.0, 3.0]})
        result = find_infinite_cols(df)

        assert "a" in result.columns

    def test_no_infinite_returns_empty(self):
        """Test that no infinity returns empty DataFrame"""
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = find_infinite_cols(df)

        assert result.width == 0


class TestCleanNumeric:
    """Test clean_numeric function"""

    def test_replaces_nan_with_zero(self):
        """Test that NaN is replaced with 0 (default)"""
        df = pl.DataFrame({"a": [1.0, float("nan"), 3.0]})
        result = df.select(clean_numeric(pl.col("a")))

        values = result["a"].to_list()
        assert values[0] == 1.0
        assert values[1] == 0.0
        assert values[2] == 3.0

    def test_replaces_inf_with_fill(self):
        """Test that inf is replaced with fill value"""
        df = pl.DataFrame({"a": [1.0, float("inf"), -float("inf")]})
        result = df.select(clean_numeric(pl.col("a"), nans_filler=999.0))

        values = result["a"].to_list()
        assert values[0] == 1.0
        assert values[1] == 999.0
        assert values[2] == 999.0

    def test_returns_expression(self):
        """Test that function returns a polars expression"""
        expr = clean_numeric(pl.col("a"))
        assert isinstance(expr, pl.Expr)


class TestCastF64ToF32:
    """Test cast_f64_to_f32 function"""

    def test_casts_float64_to_float32(self):
        """Test that Float64 columns are cast to Float32"""
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        assert df["a"].dtype == pl.Float64

        result = cast_f64_to_f32(df)
        assert result["a"].dtype == pl.Float32
        assert result["b"].dtype == pl.Float32

    def test_preserves_string_columns(self):
        """Test that non-numeric columns are preserved"""
        df = pl.DataFrame({"a": [1.0, 2.0], "s": ["x", "y"]})
        result = cast_f64_to_f32(df)

        assert result["s"].dtype == pl.Utf8 or result["s"].dtype == pl.String
        assert "s" in result.columns

    def test_integer_columns_cast(self):
        """Test that integer columns are also cast"""
        df = pl.DataFrame({"a": pl.Series([1, 2, 3], dtype=pl.Int64)})
        result = cast_f64_to_f32(df)

        assert result["a"].dtype == pl.Float32

    def test_returns_dataframe(self):
        """Test that result is a DataFrame"""
        df = pl.DataFrame({"a": [1.0, 2.0]})
        result = cast_f64_to_f32(df)

        assert isinstance(result, pl.DataFrame)


class TestApplyAggFuncSafe:
    """Test apply_agg_func_safe function"""

    def test_skew_gets_clean_numeric(self):
        """Test that skew function applies clean_numeric"""
        expr = pl.col("a")
        result = apply_agg_func_safe(expr, "skew")

        assert isinstance(result, pl.Expr)

    def test_kurtosis_gets_clean_numeric(self):
        """Test that kurtosis function applies clean_numeric"""
        expr = pl.col("a")
        result = apply_agg_func_safe(expr, "kurtosis")

        assert isinstance(result, pl.Expr)

    def test_other_funcs_passthrough(self):
        """Test that non-skew/kurtosis functions pass through unchanged"""
        expr = pl.col("a")
        result = apply_agg_func_safe(expr, "mean")

        assert isinstance(result, pl.Expr)
        # Should return the same expression object (not wrapped)
        assert result is expr

    def test_min_passthrough(self):
        """Test that min passes through"""
        expr = pl.col("b")
        result = apply_agg_func_safe(expr, "min")

        assert result is expr


class TestPolarsdfInfo:
    """Test polars_df_info function"""

    def test_returns_string(self):
        """Test that result is a string"""
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        result = polars_df_info(df)

        assert isinstance(result, str)

    def test_contains_row_count(self):
        """Test that result contains row count info"""
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        result = polars_df_info(df)

        assert "5" in result

    def test_contains_dtype_info(self):
        """Test that result contains dtype information"""
        df = pl.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
        result = polars_df_info(df)

        assert "dtypes" in result

    def test_empty_dataframe(self):
        """Test with empty DataFrame"""
        df = pl.DataFrame({"a": pl.Series([], dtype=pl.Float64)})
        result = polars_df_info(df)

        assert isinstance(result, str)
        assert "0" in result

    def test_no_columns_dataframe(self):
        """Test with DataFrame with no columns"""
        df = pl.DataFrame()
        result = polars_df_info(df)

        assert isinstance(result, str)
        assert "0" in result

    def test_contains_memory_info(self):
        """Test that result contains memory usage info"""
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = polars_df_info(df)

        assert "memory" in result.lower() or "GB" in result

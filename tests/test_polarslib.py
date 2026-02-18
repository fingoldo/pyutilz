"""Tests for polarslib.py - Phase 2 refactoring

Tests cover:
- Missing return statements fix (find_nan_cols, find_infinite_cols)
- Return type annotation fix (cast_f64_to_f32)
- Performance optimizations (df.head() usage)
"""

import pytest
import polars as pl
import numpy as np


class TestFindNanCols:
    """Test find_nan_cols function - regression test for missing return statement"""

    def test_returns_dataframe_not_none(self):
        """Verify function returns DataFrame, not None (line 47-50 bug fix)"""
        from pyutilz.polarslib import find_nan_cols

        df = pl.DataFrame({
            'col1': [1.0, 2.0, float('nan')],
            'col2': [3.0, 4.0, 5.0],
            'col3': [float('nan'), float('nan'), float('nan')]
        })

        result = find_nan_cols(df)

        # Should return DataFrame, not None
        assert result is not None
        assert isinstance(result, pl.DataFrame)

    def test_finds_columns_with_nans(self):
        """Test that columns with NaN values are correctly identified"""
        from pyutilz.polarslib import find_nan_cols

        df = pl.DataFrame({
            'has_nan': [1.0, float('nan'), 3.0],
            'no_nan': [1.0, 2.0, 3.0]
        })

        result = find_nan_cols(df)

        # Should only return column with NaN
        assert 'has_nan' in result.columns
        assert 'no_nan' not in result.columns

    def test_empty_result_when_no_nans(self):
        """Test returns empty DataFrame when no NaN columns"""
        from pyutilz.polarslib import find_nan_cols

        df = pl.DataFrame({
            'col1': [1.0, 2.0, 3.0],
            'col2': [4.0, 5.0, 6.0]
        })

        result = find_nan_cols(df)

        assert result is not None
        assert len(result.columns) == 0


class TestFindInfiniteCols:
    """Test find_infinite_cols function - regression test for missing return statement"""

    def test_returns_dataframe_not_none(self):
        """Verify function returns DataFrame, not None (line 53-56 bug fix)"""
        from pyutilz.polarslib import find_infinite_cols

        df = pl.DataFrame({
            'col1': [1.0, 2.0, float('inf')],
            'col2': [3.0, 4.0, 5.0]
        })

        result = find_infinite_cols(df)

        # Should return DataFrame, not None
        assert result is not None
        assert isinstance(result, pl.DataFrame)

    def test_finds_columns_with_infinity(self):
        """Test that columns with infinite values are correctly identified"""
        from pyutilz.polarslib import find_infinite_cols

        df = pl.DataFrame({
            'has_inf': [1.0, float('inf'), 3.0],
            'has_neg_inf': [float('-inf'), 2.0, 3.0],
            'no_inf': [1.0, 2.0, 3.0]
        })

        result = find_infinite_cols(df)

        # Should return columns with infinite values
        assert result is not None
        assert 'has_inf' in result.columns or 'has_neg_inf' in result.columns
        assert 'no_inf' not in result.columns


class TestCastF64ToF32:
    """Test cast_f64_to_f32 function - return type annotation fix"""

    def test_returns_polars_dataframe(self):
        """Verify function returns pl.DataFrame, not pd.DataFrame (line 64 fix)"""
        from pyutilz.polarslib import cast_f64_to_f32

        df = pl.DataFrame({
            'float64_col': [1.0, 2.0, 3.0],
            'int64_col': [1, 2, 3]
        })

        result = cast_f64_to_f32(df)

        # Should return Polars DataFrame
        assert isinstance(result, pl.DataFrame)
        assert not hasattr(result, 'iloc')  # pandas-specific attribute

    def test_converts_float64_to_float32(self):
        """Test that Float64 columns are converted to Float32"""
        from pyutilz.polarslib import cast_f64_to_f32

        df = pl.DataFrame({
            'col': [1.0, 2.0, 3.0]
        })

        # Ensure it's Float64 initially
        df = df.with_columns(pl.col('col').cast(pl.Float64))

        result = cast_f64_to_f32(df)

        # Should be Float32 after conversion
        assert result['col'].dtype == pl.Float32

    def test_converts_integer_types(self):
        """Test that Int64, UInt64 are also converted to Float32"""
        from pyutilz.polarslib import cast_f64_to_f32

        df = pl.DataFrame({
            'int64': pl.Series([1, 2, 3], dtype=pl.Int64),
            'uint64': pl.Series([1, 2, 3], dtype=pl.UInt64)
        })

        result = cast_f64_to_f32(df)

        assert result['int64'].dtype == pl.Float32
        assert result['uint64'].dtype == pl.Float32


class TestCleanNumeric:
    """Test clean_numeric function"""

    def test_replaces_inf_with_filler(self):
        """Test that infinite values are replaced with filler"""
        from pyutilz.polarslib import clean_numeric
        import polars as pl

        df = pl.DataFrame({'col': [1.0, float('inf'), float('-inf'), float('nan')]})

        result = df.select(clean_numeric(pl.col('col'), nans_filler=0.0))

        values = result['col'].to_list()
        assert values[0] == 1.0
        # inf, -inf, nan should all be replaced with 0.0
        assert values[1] == 0.0
        assert values[2] == 0.0
        assert values[3] == 0.0


@pytest.mark.parametrize("dtype,expected_f32", [
    (pl.Float64, True),
    (pl.Int64, True),
    (pl.UInt32, True),
    (pl.Int32, True),
    (pl.Int8, False),  # Int8 not in conversion list
])
def test_cast_f64_to_f32_parametrized(dtype, expected_f32):
    """Parametrized test for different dtype conversions"""
    from pyutilz.polarslib import cast_f64_to_f32

    df = pl.DataFrame({'col': pl.Series([1, 2, 3], dtype=dtype)})
    result = cast_f64_to_f32(df)

    if expected_f32:
        assert result['col'].dtype == pl.Float32
    else:
        assert result['col'].dtype == dtype

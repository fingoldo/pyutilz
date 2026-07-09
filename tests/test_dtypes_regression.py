"""Regression tests for pyutilz.data.pandaslib.dtypes bug fixes.

1. optimize_dtypes must not silently truncate object columns containing genuine
   fractional float values when trying to reclassify them as int64.
2. classify_column_types(df=None, dtype=...) must still work at runtime
   (the Optional[pd.DataFrame] annotation fix is a static-typing concern).
"""

import numpy as np
import pandas as pd

from pyutilz.data.pandaslib.dtypes import optimize_dtypes, classify_column_types


def test_optimize_dtypes_does_not_truncate_fractional_object_column():
    df = pd.DataFrame({"col": pd.Series([1, 2, 3.5], dtype=object)})

    result = optimize_dtypes(df, max_categories=100)

    # Must NOT have been silently cast to int64 (which would truncate 3.5 -> 3).
    assert "int" not in result["col"].dtype.name, f"Column was reclassified as int: {result['col'].dtype}"

    values = result["col"].astype(np.float64).tolist()
    assert values == [1.0, 2.0, 3.5], f"Values corrupted after optimize_dtypes: {values}"


def test_optimize_dtypes_still_converts_genuine_integer_object_column():
    df = pd.DataFrame({"col": pd.Series([1, 2, 3], dtype=object)})

    result = optimize_dtypes(df, max_categories=100)

    assert "int" in result["col"].dtype.name, f"Genuine integer column was not converted: {result['col'].dtype}"
    assert result["col"].tolist() == [1, 2, 3]


def test_classify_column_types_accepts_df_none_with_dtype():
    col_is_boolean, col_is_object, col_is_datetime, col_is_categorical, col_is_numeric = classify_column_types(df=None, dtype=np.dtype("float64"))
    assert col_is_numeric is True
    assert col_is_boolean is False
    assert col_is_object is False
    assert col_is_datetime is False
    assert col_is_categorical is False

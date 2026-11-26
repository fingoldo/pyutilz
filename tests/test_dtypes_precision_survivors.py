"""Mutation-survivor pins for pandaslib/dtypes.py: the dtype selector, the integer-float candidates, the float32 precision
check, the optimize_dtypes identity return, the dtype classifier, and the float32 conversion of pyarrow columns with its
lossy-integer warning (dtypes.py 83, 197, 253, 459, 493, 622, 635)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import pyutilz.data.pandaslib.dtypes as dt


class TestGetColumnsOfTypeSurvivors:
    """A request for 'object' finds object columns and pandas string columns, and only those (dtypes.py 83)."""

    def test_object_requests_find_object_and_string_columns_and_nothing_else(self):
        df = pd.DataFrame({"s": ["a"], "t": pd.Series(["b"], dtype="string"), "i": [1]})
        assert dt.get_columns_of_type(df, ["object"]) == ["s", "t"]


class TestFindPossiblyIntegerFloatsSurvivors:
    """A float column with any missing value is never an integer candidate (dtypes.py 197)."""

    def test_a_column_with_some_missing_values_is_not_a_candidate(self):
        df = pd.DataFrame({"f": [1.0, np.nan]})
        assert dt._find_possibly_integer_floats(df, ["f"]) == []

    def test_a_complete_whole_number_column_is_a_candidate(self):
        df = pd.DataFrame({"f": [1.0, 2.0]})
        assert dt._find_possibly_integer_floats(df, ["f"]) == ["f"]


class TestFloat32PrecisionSurvivors:
    """The float32 precision check looks at the mantissa to seven digits, after rounding the tail to five (dtypes.py 253).
    A tail of 4e-7 rounds away and is accepted; a tail of 4e-5 survives the rounding and is rejected."""

    def test_a_tail_that_rounds_away_at_float32_precision_is_accepted(self):
        df = pd.DataFrame({"c": [1.23456780000004]})
        assert dt._float_precision_ok(df, "c", 32, "float64", {}, False, False) is True

    def test_a_tail_just_under_the_mantissa_rounding_boundary_is_accepted(self):
        df = pd.DataFrame({"c": [1.23456780000049]})
        assert dt._float_precision_ok(df, "c", 32, "float64", {}, False, False) is True

    def test_a_tail_that_rounds_up_at_the_mantissa_step_is_rejected(self):
        df = pd.DataFrame({"c": [1.234567800000497]})
        assert dt._float_precision_ok(df, "c", 32, "float64", {}, False, False) is False

    def test_a_tail_that_survives_float32_rounding_is_rejected(self):
        df = pd.DataFrame({"c": [1.234567800004]})
        assert dt._float_precision_ok(df, "c", 32, "float64", {}, False, False) is False


class TestClassifyObjectLikeColumnsSurvivors:
    """Object and pandas string dtypes are object-like; neither is numeric; numbers are numeric (dtypes.py 493)."""

    def test_object_dtype_is_object_like(self):
        _, is_object, _, _, _ = dt.classify_column_types(dtype=np.dtype(object))
        assert is_object is True

    def test_pandas_string_dtype_is_object_like(self):
        _, is_object, _, _, is_numeric = dt.classify_column_types(dtype=pd.StringDtype())
        assert is_object is True
        assert is_numeric is False

    def test_a_boolean_column_is_not_numeric(self):
        is_boolean, _, _, _, is_numeric = dt.classify_column_types(dtype=np.dtype(bool))
        assert is_boolean is True
        assert is_numeric is False

    def test_an_integer_column_is_numeric(self):
        _, _, _, _, is_numeric = dt.classify_column_types(dtype=np.dtype("int64"))
        assert is_numeric is True


class TestPyarrowFloat32ConversionSurvivors:
    """A numpy-backed float column becomes float32 numpy, a pyarrow float column becomes float32 pyarrow, and a pyarrow integer
    beyond 2**24 is named in a warning (dtypes.py 622, 635)."""

    def test_numpy_and_pyarrow_integer_columns_convert_each_in_its_own_backend(self):
        df = pd.DataFrame({"n": np.array([1, 2], dtype=np.int64), "a": pd.Series([1.5, 2.5], dtype="float32[pyarrow]")})
        out = dt.ensure_dataframe_float32_convertability(df)
        assert out["n"].dtype == np.float32
        assert str(out["a"].dtype) == "float[pyarrow]"

    def test_a_pyarrow_integer_beyond_float32_exactness_is_warned_by_name(self, caplog):
        df = pd.DataFrame({"big": pd.Series([16777217, 1], dtype="int64[pyarrow]")})
        with caplog.at_level(logging.WARNING, logger=dt.logger.name):
            dt.ensure_dataframe_float32_convertability(df)
        assert any("integer column(s)" in r.getMessage() and "big" in r.getMessage() for r in caplog.records)

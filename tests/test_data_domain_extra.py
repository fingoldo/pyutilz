"""Regression tests for pyutilz.data.{polarslib,numpylib,numbalib,pandaslib} / stats.normality (2026-07-21 audit fixes)."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from pyutilz.data.polarslib import find_nan_cols, cast_f64_to_f32
from pyutilz.data.pandaslib.dtypes import get_columns_of_type
from pyutilz.data.pandaslib.frames import remove_stale_columns, get_non_stale_columns, showcase_df_columns
from pyutilz.data.numpylib import smart_ratios
from pyutilz.data.numbalib import generate_combinations_recursive_njit
from pyutilz.stats.normality import normality_verdict


class TestFindNanCols:
    def test_detects_all_null_column(self):
        df = pl.DataFrame({
            "allnull": pl.Series([None, None, None], dtype=pl.Float64),
            "hasnan": pl.Series([1.0, float("nan"), 3.0]),
            "clean": pl.Series([1.0, 2.0, 3.0]),
        })
        cols = find_nan_cols(df).columns
        assert set(cols) == {"allnull", "hasnan"}


class TestCastF64ToF32:
    def test_still_casts_ints_to_float32(self):
        df = pl.DataFrame({"id": pl.Series([1, 2, 3], dtype=pl.Int64)})
        result = cast_f64_to_f32(df)
        assert result["id"].dtype == pl.Float32

    def test_warns_on_precision_loss(self, caplog):
        import logging

        df = pl.DataFrame({"id": pl.Series([16777217, 16777219], dtype=pl.Int64)})
        with caplog.at_level(logging.WARNING, logger="pyutilz.data.polarslib"):
            cast_f64_to_f32(df)
        assert any("precision" in r.message for r in caplog.records)


class TestGetColumnsOfType:
    def test_no_duplicate_columns_on_multi_substring_match(self):
        df = pd.DataFrame({
            "a": pd.array([1, 2, 3], dtype="int64"),
            "b": pd.array([1, 2, 3], dtype="uint32"),
            "c": [1.0, 2.0, 3.0],
        })
        result = get_columns_of_type(df, ["int", "uint"])
        assert result == ["a", "b"]


class TestRemoveStaleColumns:
    def test_does_not_mutate_caller_df_but_returns_non_stale_names(self):
        df = pd.DataFrame({"const": [1, 1, 1], "var": [1, 2, 3]})
        result = remove_stale_columns(df)
        assert result == ["var"]
        assert list(df.columns) == ["const", "var"]

    def test_get_non_stale_columns_alias_equivalence(self):
        df = pd.DataFrame({"const": [1, 1, 1], "var": [1, 2, 3]})
        assert get_non_stale_columns(df) == remove_stale_columns(df)


class TestShowcaseDfColumnsDropnaGate:
    def test_pandas_dropna_true_respects_gate(self):
        vals = list(range(50)) * 2 + [None] * 100
        df = pd.DataFrame({"x": vals})
        r, u = showcase_df_columns(df, cols=["x"], dropna=True, max_cat_uniq_qty=50, max_unique_percent=0.9, use_print=False)
        assert "x" in r
        assert r["x"]
        assert u.get("x") == 1.0

    def test_polars_dropna_true_respects_gate(self):
        vals = list(range(50)) * 2 + [None] * 100
        df = pl.DataFrame({"x": pl.Series(vals, dtype=pl.Int64)})
        r, u = showcase_df_columns(df, cols=["x"], dropna=True, max_cat_uniq_qty=50, max_unique_percent=0.9, use_print=False)
        assert "x" in r
        assert r["x"]
        assert u.get("x") == 1.0


class TestSmartRatiosDocstring:
    def test_docstring_no_longer_claims_automatic_shift(self):
        assert "automatic" not in smart_ratios.__doc__.lower() or "NOT" in smart_ratios.__doc__
        assert "caller-supplied" in smart_ratios.__doc__


class TestGenerateCombinationsNegativeR:
    def test_negative_r_raises_clear_value_error(self):
        with pytest.raises(ValueError, match="r must be >= 0"):
            generate_combinations_recursive_njit(np.array([1, 2, 3]), -1)

    def test_positive_r_still_works(self):
        result = generate_combinations_recursive_njit(np.array([1, 2, 3]), 2)
        assert result.shape == (3, 2)


class TestNormalityVerdictDegenerate:
    def test_constant_sample_reported_as_degenerate(self):
        result = normality_verdict(np.array([5.0] * 30))
        assert result["verdict"] == "degenerate (zero variance)"
        assert result["reject_normal"] is False

    def test_non_constant_sample_unaffected(self):
        rng = np.random.default_rng(0)
        result = normality_verdict(rng.normal(size=200))
        assert result["verdict"] != "degenerate (zero variance)"

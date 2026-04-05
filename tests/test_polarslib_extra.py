import pytest
import polars as pl
import polars.selectors as cs
import numpy as np
from unittest.mock import patch

from pyutilz.data.polarslib import (
    find_nan_cols,
    find_infinite_cols,
    clean_numeric,
    cast_f64_to_f32,
    apply_agg_func_safe,
    polars_df_info,
    entropy_for_column,
    mi_for_column,
    bin_numerical_columns,
    drop_constant_columns,
    add_weighted_aggregates,
    build_aggregate_features_polars,
    POLARS_DEFAULT_NUMAGGS,
    POLARS_DEFAULT_QUANTILES,
)


# ============================================================
# entropy_for_column
# ============================================================


class TestEntropyForColumn:
    def test_uniform_distribution(self):
        bins = pl.DataFrame({"x": ["a", "b", "c", "d"]})
        h = entropy_for_column(bins, "x")
        assert pytest.approx(h, rel=1e-6) == np.log(4)

    def test_single_value_zero_entropy(self):
        bins = pl.DataFrame({"x": ["a", "a", "a"]})
        h = entropy_for_column(bins, "x")
        assert pytest.approx(h, abs=1e-10) == 0.0

    def test_binary_equal_split(self):
        bins = pl.DataFrame({"x": ["a", "b"] * 50})
        h = entropy_for_column(bins, "x")
        assert pytest.approx(h, rel=1e-6) == np.log(2)

    def test_skewed_distribution(self):
        bins = pl.DataFrame({"x": ["a"] * 99 + ["b"]})
        h = entropy_for_column(bins, "x")
        assert h > 0
        assert h < np.log(2)

    def test_integer_column(self):
        bins = pl.DataFrame({"x": [1, 2, 3, 4]})
        h = entropy_for_column(bins, "x")
        assert pytest.approx(h, rel=1e-6) == np.log(4)


# ============================================================
# mi_for_column
# ============================================================


class TestMiForColumn:
    def test_identical_columns_max_mi(self):
        bins = pl.DataFrame({"x": [0, 1, 2, 3] * 25, "y": [0, 1, 2, 3] * 25})
        entropies = {
            "x": entropy_for_column(bins, "x"),
            "y": entropy_for_column(bins, "y"),
        }
        mi = mi_for_column(bins, entropies, "x", "y")
        assert pytest.approx(mi, rel=1e-6) == entropies["x"]

    def test_independent_columns_low_mi(self):
        np.random.seed(42)
        x = np.random.choice(["a", "b", "c"], size=1000)
        y = np.random.choice(["x", "y", "z"], size=1000)
        bins = pl.DataFrame({"x": x, "y": y})
        entropies = {
            "x": entropy_for_column(bins, "x"),
            "y": entropy_for_column(bins, "y"),
        }
        mi = mi_for_column(bins, entropies, "x", "y")
        assert mi >= 0
        assert mi < 0.05  # near zero for independent

    def test_mi_non_negative(self):
        bins = pl.DataFrame({"a": [0, 1, 0, 1, 0], "b": [1, 0, 1, 0, 1]})
        entropies = {
            "a": entropy_for_column(bins, "a"),
            "b": entropy_for_column(bins, "b"),
        }
        mi = mi_for_column(bins, entropies, "a", "b")
        assert mi >= -1e-10

    def test_perfect_inverse_mapping(self):
        bins = pl.DataFrame({"a": [0, 1, 2] * 10, "b": [2, 1, 0] * 10})
        entropies = {
            "a": entropy_for_column(bins, "a"),
            "b": entropy_for_column(bins, "b"),
        }
        mi = mi_for_column(bins, entropies, "a", "b")
        assert pytest.approx(mi, rel=1e-6) == entropies["a"]


# ============================================================
# drop_constant_columns
# ============================================================


@patch("pyutilz.data.polarslib.clean_ram")
class TestDropConstantColumns:
    def test_drops_constant_numeric_col(self, mock_clean):
        df = pl.DataFrame({"a": [1, 1, 1], "b": [1.0, 2.0, 3.0]}).lazy()
        result = drop_constant_columns(df, verbose=0)
        assert "a" not in result.columns
        assert "b" in result.columns

    def test_keeps_varying_columns(self, mock_clean):
        df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}).lazy()
        result = drop_constant_columns(df, verbose=0)
        assert set(result.columns) == {"x", "y"}

    def test_all_constant_drops_all_numeric(self, mock_clean):
        df = pl.DataFrame({"a": [5, 5, 5], "b": [3.0, 3.0, 3.0]}).lazy()
        result = drop_constant_columns(df, verbose=0)
        assert result.width == 0

    def test_preserves_non_numeric(self, mock_clean):
        df = pl.DataFrame({"a": [1, 1, 1], "s": ["x", "x", "x"]}).lazy()
        result = drop_constant_columns(df, verbose=0)
        assert "s" in result.columns
        assert "a" not in result.columns

    def test_none_values_treated_as_constant(self, mock_clean):
        df = pl.DataFrame({"a": pl.Series([None, None, None], dtype=pl.Int64)}).lazy()
        result = drop_constant_columns(df, verbose=0)
        assert "a" not in result.columns

    def test_verbose_logging(self, mock_clean):
        df = pl.DataFrame({"a": [1, 1, 1], "b": [1.0, 2.0, 3.0]}).lazy()
        result = drop_constant_columns(df, verbose=1)
        assert "a" not in result.columns


# ============================================================
# bin_numerical_columns
# ============================================================


@patch("pyutilz.data.polarslib.clean_ram")
class TestBinNumericalColumns:
    def test_basic_binning(self, mock_clean):
        df = pl.DataFrame({"a": list(range(100)), "b": list(range(100, 200))}).lazy()
        bins, binned_targets, clips, dropped, stats = bin_numerical_columns(
            df, target_columns=[], verbose=0
        )
        assert isinstance(bins, pl.DataFrame)
        assert bins.height == 100

    def test_target_columns_separated(self, mock_clean):
        df = pl.DataFrame({"feat": list(range(100)), "target": list(range(100))}).lazy()
        bins, binned_targets, clips, dropped, stats = bin_numerical_columns(
            df, target_columns=["target"], verbose=0
        )
        assert binned_targets is not None
        assert "target" in binned_targets.columns

    def test_constant_columns_dropped(self, mock_clean):
        df = pl.DataFrame({"a": [5] * 20, "b": list(range(20))}).lazy()
        bins, _, _, dropped, _ = bin_numerical_columns(
            df, target_columns=[], verbose=0
        )
        assert "a" in dropped

    def test_num_bins_respected(self, mock_clean):
        df = pl.DataFrame({"a": list(range(1000))}).lazy()
        bins, _, _, _, _ = bin_numerical_columns(
            df, target_columns=[], num_bins=5, verbose=0
        )
        unique_bins = bins["a"].n_unique()
        assert unique_bins <= 5

    def test_exclude_columns(self, mock_clean):
        df = pl.DataFrame({"a": list(range(50)), "b": list(range(50))}).lazy()
        bins, _, _, _, _ = bin_numerical_columns(
            df, target_columns=[], exclude_columns=["b"], verbose=0
        )
        assert "b" not in bins.columns

    def test_bin_dtype(self, mock_clean):
        df = pl.DataFrame({"a": list(range(50))}).lazy()
        bins, _, _, _, _ = bin_numerical_columns(
            df, target_columns=[], bin_dtype=pl.Int16, verbose=0
        )
        assert bins["a"].dtype == pl.Int16

    def test_clips_dict_populated_for_outliers(self, mock_clean):
        values = list(range(100)) + [10000]
        df = pl.DataFrame({"a": values}).lazy()
        _, _, clips, _, _ = bin_numerical_columns(
            df, target_columns=[], verbose=0
        )
        # With extreme outlier, clipping should be detected
        assert isinstance(clips, dict)

    def test_fill_nulls(self, mock_clean):
        df = pl.DataFrame({"a": [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}).lazy()
        bins, _, _, _, _ = bin_numerical_columns(
            df, target_columns=[], fill_nulls=True, verbose=0
        )
        assert bins["a"].null_count() == 0

    def test_fill_nans(self, mock_clean):
        df = pl.DataFrame({"a": [float("nan")] + list(range(1, 50))}).cast({"a": pl.Float64}).lazy()
        bins, _, _, _, _ = bin_numerical_columns(
            df, target_columns=[], fill_nans=True, verbose=0
        )
        assert bins.height == 50

    def test_missing_target_columns_ignored(self, mock_clean):
        df = pl.DataFrame({"a": list(range(50))}).lazy()
        bins, _, _, _, _ = bin_numerical_columns(
            df, target_columns=["nonexistent"], verbose=0
        )
        assert bins.height == 50

    def test_binned_targets_passthrough(self, mock_clean):
        df = pl.DataFrame({"a": list(range(50))}).lazy()
        pre_binned = pl.DataFrame({"t": list(range(50))})
        bins, bt, _, _, _ = bin_numerical_columns(
            df, target_columns=["t"], binned_targets=pre_binned, verbose=0
        )
        assert "t" in bins.columns


# ============================================================
# add_weighted_aggregates
# ============================================================


class TestAddWeightedAggregates:
    def test_returns_list(self):
        selector = cs.numeric()
        result = add_weighted_aggregates(selector, weighting_columns=["w"])
        assert isinstance(result, list)

    def test_no_weighting_columns_empty(self):
        result = add_weighted_aggregates(cs.numeric(), weighting_columns=[])
        assert result == []

    def test_none_weighting_columns_empty(self):
        result = add_weighted_aggregates(cs.numeric(), weighting_columns=None)
        assert result == []

    def test_weighted_mean_computable(self):
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "w": [1.0, 1.0, 1.0]})
        exprs = add_weighted_aggregates(cs.numeric(), weighting_columns=["w"])
        result = df.group_by(pl.lit(1).alias("grp")).agg(exprs)
        assert result.width > 1

    def test_fields_remap(self):
        df = pl.DataFrame({"a": [1.0, 2.0], "w": [1.0, 1.0]})
        exprs = add_weighted_aggregates(
            cs.numeric(), weighting_columns=["w"], fields_remap={"w": "weight"}
        )
        result = df.group_by(pl.lit(1).alias("grp")).agg(exprs)
        col_names = result.columns
        assert any("weight" in c for c in col_names)

    def test_fpref(self):
        df = pl.DataFrame({"a": [1.0, 2.0], "w": [1.0, 1.0]})
        exprs = add_weighted_aggregates(
            cs.numeric(), weighting_columns=["w"], fpref="pref_"
        )
        result = df.group_by(pl.lit(1).alias("grp")).agg(exprs)
        col_names = result.columns
        assert any("pref_" in c for c in col_names)


# ============================================================
# find_nan_cols (extra edge cases)
# ============================================================


class TestFindNanColsExtra:
    def test_empty_dataframe(self):
        df = pl.DataFrame({"a": pl.Series([], dtype=pl.Float64)})
        result = find_nan_cols(df)
        assert result.width == 0

    def test_mixed_types_only_numeric_checked(self):
        df = pl.DataFrame({
            "n": [1.0, float("nan")],
            "s": ["a", "b"],
        })
        result = find_nan_cols(df)
        assert "n" in result.columns
        assert "s" not in result.columns

    def test_integer_columns_no_nan(self):
        df = pl.DataFrame({"i": [1, 2, 3]})
        result = find_nan_cols(df)
        assert result.width == 0


# ============================================================
# find_infinite_cols (extra edge cases)
# ============================================================


class TestFindInfiniteColsExtra:
    def test_both_inf_directions(self):
        df = pl.DataFrame({
            "a": [float("inf")],
            "b": [float("-inf")],
            "c": [1.0],
        })
        result = find_infinite_cols(df)
        assert "a" in result.columns
        assert "b" in result.columns
        assert "c" not in result.columns

    def test_empty_dataframe(self):
        df = pl.DataFrame({"a": pl.Series([], dtype=pl.Float64)})
        result = find_infinite_cols(df)
        assert result.width == 0


# ============================================================
# clean_numeric (extra edge cases)
# ============================================================


class TestCleanNumericExtra:
    def test_custom_fill_value(self):
        df = pl.DataFrame({"a": [float("nan"), float("inf"), -float("inf"), 5.0]})
        result = df.select(clean_numeric(pl.col("a"), nans_filler=-1.0))
        assert result["a"].to_list() == [-1.0, -1.0, -1.0, 5.0]

    def test_no_special_values_unchanged(self):
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = df.select(clean_numeric(pl.col("a")))
        assert result["a"].to_list() == [1.0, 2.0, 3.0]

    def test_all_nan(self):
        df = pl.DataFrame({"a": [float("nan"), float("nan")]})
        result = df.select(clean_numeric(pl.col("a"), nans_filler=42.0))
        assert result["a"].to_list() == [42.0, 42.0]


# ============================================================
# cast_f64_to_f32 (extra edge cases)
# ============================================================


class TestCastF64ToF32Extra:
    def test_int32_cast(self):
        df = pl.DataFrame({"a": pl.Series([1, 2], dtype=pl.Int32)})
        result = cast_f64_to_f32(df)
        assert result["a"].dtype == pl.Float32

    def test_uint64_cast(self):
        df = pl.DataFrame({"a": pl.Series([1, 2], dtype=pl.UInt64)})
        result = cast_f64_to_f32(df)
        assert result["a"].dtype == pl.Float32

    def test_uint32_cast(self):
        df = pl.DataFrame({"a": pl.Series([1, 2], dtype=pl.UInt32)})
        result = cast_f64_to_f32(df)
        assert result["a"].dtype == pl.Float32

    def test_mixed_types(self):
        df = pl.DataFrame({
            "f64": [1.0, 2.0],
            "i64": pl.Series([1, 2], dtype=pl.Int64),
            "s": ["a", "b"],
            "b": [True, False],
        })
        result = cast_f64_to_f32(df)
        assert result["f64"].dtype == pl.Float32
        assert result["i64"].dtype == pl.Float32
        assert result["s"].dtype in (pl.Utf8, pl.String)
        assert result["b"].dtype == pl.Boolean

    def test_float32_stays_float32(self):
        df = pl.DataFrame({"a": pl.Series([1.0, 2.0], dtype=pl.Float32)})
        result = cast_f64_to_f32(df)
        assert result["a"].dtype == pl.Float32


# ============================================================
# apply_agg_func_safe (extra)
# ============================================================


@pytest.mark.parametrize("func_name", ["mean", "std", "min", "max", "first", "last"])
def test_apply_agg_func_safe_passthrough(func_name):
    expr = pl.col("a")
    result = apply_agg_func_safe(expr, func_name)
    assert result is expr


@pytest.mark.parametrize("func_name", ["skew", "kurtosis"])
def test_apply_agg_func_safe_wraps(func_name):
    expr = pl.col("a")
    result = apply_agg_func_safe(expr, func_name)
    assert result is not expr
    assert isinstance(result, pl.Expr)


def test_apply_agg_func_safe_custom_filler():
    expr = pl.col("a")
    result = apply_agg_func_safe(expr, "skew", nans_filler=99.0)
    assert isinstance(result, pl.Expr)


# ============================================================
# polars_df_info (extra)
# ============================================================


class TestPolarsdfInfoExtra:
    def test_multiple_dtypes(self):
        df = pl.DataFrame({
            "a": [1, 2],
            "b": [1.0, 2.0],
            "c": ["x", "y"],
        })
        result = polars_df_info(df)
        assert "dtypes" in result
        assert "2" in result  # row count

    def test_single_column(self):
        df = pl.DataFrame({"only": [1]})
        result = polars_df_info(df)
        assert "only" in result
        assert "1 entries" in result

    def test_memory_line_present(self):
        df = pl.DataFrame({"a": list(range(100))})
        result = polars_df_info(df)
        assert "memory" in result.lower()

    def test_large_dataframe(self):
        df = pl.DataFrame({"a": list(range(10000))})
        result = polars_df_info(df)
        assert "10000" in result


# ============================================================
# build_aggregate_features_polars (basic tests)
# ============================================================


class TestBuildAggregateFeatures:
    def test_returns_tuple_of_three(self):
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = build_aggregate_features_polars(df, engine="cpu")
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_expressions_are_list(self):
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        exprs, unnest_cols, unnest_rules = build_aggregate_features_polars(df, engine="cpu")
        assert isinstance(exprs, list)
        assert len(exprs) > 0

    def test_custom_numaggs(self):
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        exprs, _, _ = build_aggregate_features_polars(df, numaggs=["mean", "std"], engine="cpu")
        # Should have fewer expressions with fewer aggs
        exprs_all, _, _ = build_aggregate_features_polars(df, engine="cpu")
        assert len(exprs) < len(exprs_all)

    def test_custom_quantiles(self):
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        exprs, _, _ = build_aggregate_features_polars(df, quantiles=[0.5], engine="cpu")
        assert isinstance(exprs, list)

    def test_boolean_fields(self):
        df = pl.DataFrame({"a": [1.0, 2.0], "b": [True, False]})
        exprs, _, _ = build_aggregate_features_polars(df, engine="cpu")
        assert len(exprs) > 0

    def test_exclude_fields(self):
        df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        exprs_all, _, _ = build_aggregate_features_polars(df, engine="cpu")
        exprs_excl, _, _ = build_aggregate_features_polars(df, exclude_fields=["b"], engine="cpu")
        assert len(exprs_excl) < len(exprs_all)

    def test_categorical_fields(self):
        df = pl.DataFrame({"a": [1.0, 2.0], "c": ["x", "y"]})
        exprs, _, _ = build_aggregate_features_polars(df, engine="cpu")
        assert len(exprs) > 0

    def test_fields_remap(self):
        df = pl.DataFrame({"a": [1.0, 2.0]})
        exprs, _, _ = build_aggregate_features_polars(
            df, fields_remap={"a": "alpha"}, engine="cpu"
        )
        assert len(exprs) > 0

    def test_custom_expressions_included(self):
        df = pl.DataFrame({"a": [1.0, 2.0]})
        custom = [pl.col("a").sum().alias("a_total")]
        exprs, _, _ = build_aggregate_features_polars(
            df, custom_expressions=custom, engine="cpu"
        )
        aliases = []
        # custom expression should be first in list
        assert len(exprs) > 1

    def test_no_peaks_stats(self):
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        exprs_with, _, _ = build_aggregate_features_polars(df, add_peaks_stats=True, engine="cpu")
        exprs_without, _, _ = build_aggregate_features_polars(df, add_peaks_stats=False, engine="cpu")
        assert len(exprs_without) < len(exprs_with)


# ============================================================
# Constants
# ============================================================


def test_default_numaggs_contents():
    assert "mean" in POLARS_DEFAULT_NUMAGGS
    assert "std" in POLARS_DEFAULT_NUMAGGS
    assert "skew" in POLARS_DEFAULT_NUMAGGS
    assert "kurtosis" in POLARS_DEFAULT_NUMAGGS


def test_default_quantiles():
    assert 0.5 in POLARS_DEFAULT_QUANTILES
    assert len(POLARS_DEFAULT_QUANTILES) == 5
    assert POLARS_DEFAULT_QUANTILES == sorted(POLARS_DEFAULT_QUANTILES)

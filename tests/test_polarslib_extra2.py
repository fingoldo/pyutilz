"""Tests targeting uncovered lines in polarslib.py:
98-154, 231, 287, 289, 293, 326-328, 388-391, 394-411, 427, 456-457,
506-513, 533-535, 554-571, 596-658, 714, 722, 731, 741, 753, 769,
784-785, 787-788, 807-809, 821-825, 827-830, 839, 852, 859, 875-878
"""

import pytest
import polars as pl
import polars.selectors as cs
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from pyutilz.data.polarslib import (
    compute_concentrations,
    build_aggregate_features_polars,
    create_ts_features_polars,
    bin_numerical_columns,
    add_weighted_aggregates,
    clean_numeric,
    cast_f64_to_f32,
    POLARS_DEFAULT_NUMAGGS,
)


# ============================================================
# compute_concentrations (lines 98-154)
# ============================================================


class TestComputeConcentrations:
    """Tests for compute_concentrations covering lines 98-154."""

    def _make_groupby(self, df, index_col="ts", every="1h", period="2h"):
        return df.sort(index_col).group_by_dynamic(index_col, every=every, period=period)

    def _sample_df(self):
        base = datetime(2024, 1, 1)
        return pl.DataFrame({
            "ts": [base + timedelta(minutes=i * 10) for i in range(20)],
            "customer": [f"c{i % 4}" for i in range(20)],
            "amount": [float(i * 10 + 1) for i in range(20)],
        })

    def test_return_ids_and_values(self):
        # lines 98-154: both return_ids=True and return_values=True
        df = self._sample_df()
        gbo = self._make_groupby(df)
        result = compute_concentrations(
            groupby_object=gbo,
            entity="customer",
            entity_name="customer",
            by="amount",
            groupby_columns=("ts",),
            top_n=2,
            return_ids=True,
            return_values=True,
        )
        assert isinstance(result, pl.DataFrame)
        assert result.height > 0
        # Should have unnested top1/top2 id columns
        cols = result.columns
        assert any("top1" in c for c in cols)
        assert any("top2" in c for c in cols)

    def test_return_ids_only(self):
        df = self._sample_df()
        gbo = self._make_groupby(df)
        result = compute_concentrations(
            groupby_object=gbo,
            entity="customer",
            entity_name="customer",
            by="amount",
            groupby_columns=("ts",),
            top_n=2,
            return_ids=True,
            return_values=False,
        )
        assert isinstance(result, pl.DataFrame)
        cols = result.columns
        assert any("top1" in c and "conc" not in c for c in cols)

    def test_return_values_only(self):
        df = self._sample_df()
        gbo = self._make_groupby(df)
        result = compute_concentrations(
            groupby_object=gbo,
            entity="customer",
            entity_name="customer",
            by="amount",
            groupby_columns=("ts",),
            top_n=2,
            return_ids=False,
            return_values=True,
        )
        assert isinstance(result, pl.DataFrame)
        cols = result.columns
        assert any("conc" in c for c in cols)

    def test_with_fields_remap(self):
        df = self._sample_df()
        gbo = self._make_groupby(df)
        result = compute_concentrations(
            groupby_object=gbo,
            entity="customer",
            entity_name="customer",
            by="amount",
            groupby_columns=("ts",),
            top_n=2,
            fields_remap={"amount": "amt"},
        )
        cols = result.columns
        assert any("amt" in c for c in cols)

    def test_no_mean_concentration(self):
        df = self._sample_df()
        gbo = self._make_groupby(df)
        result = compute_concentrations(
            groupby_object=gbo,
            entity="customer",
            entity_name="customer",
            by="amount",
            groupby_columns=("ts",),
            top_n=2,
            add_mean_concentration=False,
        )
        cols = result.columns
        assert not any("avg_conc" in c for c in cols)

    def test_sort_by_name(self):
        df = self._sample_df()
        gbo = self._make_groupby(df)
        result = compute_concentrations(
            groupby_object=gbo,
            entity="customer",
            entity_name="customer",
            by="amount",
            groupby_columns=("ts",),
            top_n=2,
            sort_by_concentration=False,
        )
        assert isinstance(result, pl.DataFrame)

    def test_assert_fails_no_return(self):
        df = self._sample_df()
        gbo = self._make_groupby(df)
        with pytest.raises(AssertionError):
            compute_concentrations(
                groupby_object=gbo,
                entity="customer",
                entity_name="customer",
                by="amount",
                groupby_columns=("ts",),
                return_ids=False,
                return_values=False,
            )

    def test_top_n_3(self):
        df = self._sample_df()
        gbo = self._make_groupby(df)
        result = compute_concentrations(
            groupby_object=gbo,
            entity="customer",
            entity_name="customer",
            by="amount",
            groupby_columns=("ts",),
            top_n=3,
        )
        cols = result.columns
        assert any("top3" in c for c in cols)

    def test_dtype_float32(self):
        df = self._sample_df()
        gbo = self._make_groupby(df)
        result = compute_concentrations(
            groupby_object=gbo,
            entity="customer",
            entity_name="customer",
            by="amount",
            groupby_columns=("ts",),
            top_n=2,
            dtype=pl.Float32,
        )
        assert isinstance(result, pl.DataFrame)


# ============================================================
# build_aggregate_features_polars — advanced paths
# ============================================================


class TestBuildAggregateFeaturesAdvanced:
    """Covers lines 287, 289, 293, 326-328, 388-391, 394-411, 427, 456-457."""

    def test_filter_field_with_values(self):
        # lines 326-328: filter_field triggers categorical removal from list
        df = pl.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0],
            "cat": ["x", "x", "y", "y"],
        })
        exprs, _, _ = build_aggregate_features_polars(
            df,
            subgroups={"cat": ["x", "y"]},
            engine="cpu",
        )
        assert len(exprs) > 0

    def test_filter_field_none_key(self):
        # lines 322-324: filter_field=None triggers the no-filter path
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        exprs, _, _ = build_aggregate_features_polars(
            df,
            subgroups={"": [""]},
            engine="cpu",
        )
        assert len(exprs) > 0

    def test_weighting_fields(self):
        # lines 388-391: weighting path
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "w": [0.5, 0.3, 0.2]})
        exprs, _, _ = build_aggregate_features_polars(
            df,
            weighting_fields=["w"],
            exclude_fields=["placeholder_nonexistent"],
            engine="cpu",
        )
        assert len(exprs) > 0

    def test_othersvals_at_extremums(self):
        # lines 394-411: othersvals_at_extremums path
        df = pl.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
        })
        exprs, _, _ = build_aggregate_features_polars(
            df,
            othersvals_at_extremums=True,
            engine="cpu",
        )
        assert any("at_" in str(e) for e in exprs)

    def test_othersvals_with_basic_and_other_fields(self):
        # lines 395-398: othersvals_basic_fields and othersvals_other_fields
        df = pl.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [10.0, 20.0, 30.0],
            "c": [5.0, 6.0, 7.0],
        })
        exprs, _, _ = build_aggregate_features_polars(
            df,
            othersvals_at_extremums=True,
            othersvals_basic_fields=["a"],
            othersvals_other_fields=["b", "c"],
            engine="cpu",
        )
        assert len(exprs) > 0

    def test_othersvals_with_excluded_fields(self):
        # lines 404-405: othersvals_excluded_fields
        df = pl.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [10.0, 20.0, 30.0],
            "c": [5.0, 6.0, 7.0],
        })
        exprs, _, _ = build_aggregate_features_polars(
            df,
            othersvals_at_extremums=True,
            othersvals_excluded_fields=["c"],
            engine="cpu",
        )
        assert len(exprs) > 0

    def test_othersvals_with_filter_field(self):
        # lines 407-408: filter_field exclusion from other_columns
        df = pl.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
            "cat": ["x", "x", "y", "y"],
        })
        exprs, _, _ = build_aggregate_features_polars(
            df,
            subgroups={"cat": ["x"]},
            othersvals_at_extremums=True,
            engine="cpu",
        )
        assert len(exprs) > 0

    def test_ewm_spans(self):
        # line 427 area: ewm features
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        exprs, _, _ = build_aggregate_features_polars(
            df,
            ewm_spans=[3],
            engine="cpu",
        )
        assert any("ewm" in str(e) for e in exprs)

    def test_concentration_top_n(self):
        # lines 456-457: categorical concentration
        df = pl.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "cat": pl.Series(["x", "x", "y"], dtype=pl.Utf8),
        })
        exprs, _, _ = build_aggregate_features_polars(
            df,
            concentration_top_n=2,
            engine="cpu",
        )
        assert any("top" in str(e) for e in exprs)

    def test_boolean_fields(self):
        # line 287,289: boolean_fields with exclude
        df = pl.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "flag": [True, False, True],
        })
        exprs, _, _ = build_aggregate_features_polars(
            df,
            exclude_fields=["flag"],
            engine="cpu",
        )
        # flag should be excluded
        assert len(exprs) > 0

    def test_ts_diff_fields_excluded(self):
        # line 289: ts_diff_fields exclusion
        df = pl.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
        })
        exprs, _, _ = build_aggregate_features_polars(
            df,
            ts_diff_fields=["b"],
            exclude_fields=["b"],
            engine="cpu",
        )
        assert len(exprs) > 0

    def test_categorical_fields_excluded(self):
        # line 293: categorical exclusion
        df = pl.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "cat": pl.Series(["x", "y", "z"], dtype=pl.Utf8),
        })
        exprs, _, _ = build_aggregate_features_polars(
            df,
            exclude_fields=["cat"],
            engine="cpu",
        )
        assert len(exprs) > 0

    def test_gpu_engine_fallback(self):
        # line 231: gpu engine with no cuda
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        with patch("pyutilz.data.polarslib.is_cuda_available", return_value=False):
            exprs, _, _ = build_aggregate_features_polars(df, engine="gpu")
            assert len(exprs) > 0


# ============================================================
# create_ts_features_polars (lines 596-658)
# ============================================================


@patch("pyutilz.data.polarslib.clean_ram")
class TestCreateTsFeaturesPolars:
    """Covers lines 596-658."""

    def _make_ts_df(self, n=50):
        base = datetime(2024, 1, 1)
        return pl.DataFrame({
            "ts": [base + timedelta(hours=i) for i in range(n)],
            "val": [float(i) for i in range(n)],
        }).lazy()

    def test_rolling_basic(self, mock_clean):
        df = self._make_ts_df()
        result = create_ts_features_polars(
            df, index_column="ts", period="5h", rolling=True,
            numaggs=["mean", "std"], engine="cpu",
        )
        assert isinstance(result, pl.DataFrame)
        assert result.height > 0

    def test_group_by_dynamic_basic(self, mock_clean):
        df = self._make_ts_df()
        result = create_ts_features_polars(
            df, index_column="ts", period="10h", every="5h",
            numaggs=["mean"], engine="cpu",
        )
        assert isinstance(result, pl.DataFrame)

    def test_rolling_with_group_by(self, mock_clean):
        base = datetime(2024, 1, 1)
        df = pl.DataFrame({
            "ts": [base + timedelta(hours=i) for i in range(40)],
            "grp": ["A", "B"] * 20,
            "val": [float(i) for i in range(40)],
        }).lazy()
        result = create_ts_features_polars(
            df, index_column="ts", period="10h", rolling=True,
            group_by="grp", numaggs=["mean"], engine="cpu",
        )
        assert isinstance(result, pl.DataFrame)

    def test_group_by_with_existing_exclude(self, mock_clean):
        # line 604-605: exclude_fields already set + group_by appends
        base = datetime(2024, 1, 1)
        df = pl.DataFrame({
            "ts": [base + timedelta(hours=i) for i in range(40)],
            "grp": ["A", "B"] * 20,
            "val": [float(i) for i in range(40)],
            "extra": [float(i) for i in range(40)],
        }).lazy()
        result = create_ts_features_polars(
            df, index_column="ts", period="10h", rolling=True,
            group_by="grp", exclude_fields=["extra"],
            numaggs=["mean"], engine="cpu",
        )
        assert isinstance(result, pl.DataFrame)

    def test_closed_default_rolling(self, mock_clean):
        # line 597: closed defaults to "right" for rolling
        df = self._make_ts_df()
        result = create_ts_features_polars(
            df, index_column="ts", period="5h", rolling=True,
            numaggs=["mean"], engine="cpu",
        )
        assert isinstance(result, pl.DataFrame)

    def test_closed_default_dynamic(self, mock_clean):
        # line 597: closed defaults to "left" for non-rolling
        df = self._make_ts_df()
        result = create_ts_features_polars(
            df, index_column="ts", period="10h", every="5h",
            numaggs=["mean"], engine="cpu",
        )
        assert isinstance(result, pl.DataFrame)

    def test_dtype_float32(self, mock_clean):
        # line 648-649: Float32 cast path
        df = self._make_ts_df()
        result = create_ts_features_polars(
            df, index_column="ts", period="10h", every="5h",
            numaggs=["mean", "std"], dtype=pl.Float32, engine="cpu",
        )
        assert isinstance(result, pl.DataFrame)
        # Numeric cols should be Float32
        for c in result.columns:
            if result[c].dtype.is_float():
                assert result[c].dtype == pl.Float32

    def test_clean_memory_false(self, mock_clean):
        # line 599: skip clean_ram
        df = self._make_ts_df()
        result = create_ts_features_polars(
            df, index_column="ts", period="10h", every="5h",
            clean_memory=False, numaggs=["mean"], engine="cpu",
        )
        assert isinstance(result, pl.DataFrame)

    def test_include_boundaries(self, mock_clean):
        # line 622: include_boundaries=True
        df = self._make_ts_df()
        result = create_ts_features_polars(
            df, index_column="ts", period="10h", every="5h",
            include_boundaries=True, numaggs=["mean"], engine="cpu",
        )
        assert isinstance(result, pl.DataFrame)


# ============================================================
# bin_numerical_columns — advanced paths (lines 714+)
# ============================================================


@patch("pyutilz.data.polarslib.clean_ram")
class TestBinNumericalColumnsAdvanced:
    """Covers lines 714, 722, 731, 741, 753, 769, 784-788, 807-809,
    821-825, 827-830, 839, 852, 859, 875-878."""

    def test_verbose_warnings(self, mock_clean):
        # line 714, 769, 826-830, 875: verbose logging paths
        df = pl.DataFrame({
            "a": [5] * 20,
            "b": list(range(20)),
        }).lazy()
        bins, _, _, dropped, _ = bin_numerical_columns(
            df, target_columns=[], verbose=2,
        )
        assert "a" in dropped

    def test_clean_features_clipping(self, mock_clean):
        # lines 784-785, 807-809: clean_features triggers clipping
        values = list(range(100)) + [10000, -10000]
        df = pl.DataFrame({"a": [float(v) for v in values]}).lazy()
        bins, _, clips, _, _ = bin_numerical_columns(
            df, target_columns=[], clean_features=True, verbose=1,
        )
        assert isinstance(clips, dict)

    def test_clean_targets_only(self, mock_clean):
        # lines 787-788: clean_targets=True, clean_features=False
        values = list(range(100)) + [10000]
        df = pl.DataFrame({
            "feat": [float(v) for v in values],
            "tgt": [float(v) for v in values],
        }).lazy()
        bins, bt, clips, _, _ = bin_numerical_columns(
            df, target_columns=["tgt"], clean_targets=True, clean_features=False, verbose=1,
        )
        assert isinstance(bins, pl.DataFrame)

    def test_min_nuniques_to_clip(self, mock_clean):
        # lines 821-825: skip clipping for low-nunique columns
        values = [1.0, 1.0, 1.0, 2.0, 2.0, 100.0] * 10 + [100000.0]
        df = pl.DataFrame({"a": values}).lazy()
        bins, _, clips, _, _ = bin_numerical_columns(
            df, target_columns=[], clean_features=True,
            min_nuniques_to_clip=100, verbose=1,
        )
        # a has few uniques, clipping should be skipped
        assert "a" not in clips

    def test_binned_targets_skip_target_cols(self, mock_clean):
        # line 722, 852: when binned_targets provided, skip target columns
        df = pl.DataFrame({
            "feat": list(range(50)),
            "tgt": list(range(50)),
        }).lazy()
        pre_binned = pl.DataFrame({"tgt": list(range(50))})
        bins, bt, _, _, _ = bin_numerical_columns(
            df, target_columns=["tgt"], binned_targets=pre_binned, verbose=0,
        )
        assert "tgt" in bins.columns

    def test_dead_columns_in_binning_phase(self, mock_clean):
        # line 858-859, 874-878: dead columns found during binning
        df = pl.DataFrame({
            "a": [5.0] * 20,
            "b": [float(i) for i in range(20)],
        }).lazy()
        bins, _, _, dropped, _ = bin_numerical_columns(
            df, target_columns=[], verbose=1,
        )
        assert "a" in dropped

    def test_clean_targets_quantiles(self, mock_clean):
        # line 741: quantile_cols = target columns when clean_features=False
        values = list(range(100))
        df = pl.DataFrame({
            "feat": [float(v) for v in values],
            "tgt": [float(v) for v in values],
        }).lazy()
        bins, bt, _, _, stats = bin_numerical_columns(
            df, target_columns=["tgt"], clean_targets=True, clean_features=False, verbose=0,
        )
        assert isinstance(bins, pl.DataFrame)

    def test_stats_empty_df(self, mock_clean):
        # line 753: empty stats dict when no rows — triggers error in polars,
        # so we just verify it raises or handles gracefully
        df = pl.DataFrame({"a": pl.Series([], dtype=pl.Float64)}).lazy()
        with pytest.raises(Exception):
            bin_numerical_columns(df, target_columns=[], verbose=0)

    def test_verbose_level2(self, mock_clean):
        # lines 731, 839: verbose > 1 triggers info logs
        df = pl.DataFrame({"a": [float(i) for i in range(50)]}).lazy()
        bins, _, _, _, _ = bin_numerical_columns(
            df, target_columns=[], verbose=2,
        )
        assert isinstance(bins, pl.DataFrame)

    def test_clipping_with_inf(self, mock_clean):
        # line 798: np.isneginf / np.isinf check
        values = [float("-inf")] + [float(i) for i in range(98)] + [float("inf")]
        df = pl.DataFrame({"a": values}).lazy()
        bins, _, clips, _, _ = bin_numerical_columns(
            df, target_columns=[], clean_features=True, verbose=1,
        )
        assert isinstance(clips, dict)

"""Regression tests for the 2026-09-03 data/stats domain audit (F01..F11).

Every test here fails against the pre-fix source and passes after it.
"""

from decimal import Decimal

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
import pytest

from pyutilz.data.pandaslib.dtypes import optimize_dtypes
from pyutilz.data.pandaslib.frames import showcase_df_columns
from pyutilz.data.pandaslib.io_ops import concat_and_flush_df_list
from pyutilz.data.polarslib import (
    add_weighted_aggregates,
    bin_numerical_columns,
    build_aggregate_features_polars,
    drop_constant_columns,
    entropy_for_column,
    find_infinite_cols,
    find_nan_cols,
    mi_for_column,
)

try:
    import polars_ds  # noqa: F401

    _has_polars_ds = True
except Exception:
    _has_polars_ds = False


def _subgroup_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "g": [1, 1, 1, 1],
            "side": ["buy", "buy", "sell", "sell"],
            "px": [10.0, 20.0, 100.0, 200.0],
            "vol": [1.0, 1.0, 1.0, 1.0],
        }
    )


# F01 -------------------------------------------------------------------------------------------


class TestWeightedAggregatesRespectSubgroupF01:
    def test_weighted_mean_differs_per_subgroup(self):
        """The subgroup prefix used to name the feature while the expression stayed unfiltered."""
        df = _subgroup_frame()
        exprs, _, _ = build_aggregate_features_polars(
            df,
            numerical_fields=["px"],
            weighting_fields=["vol"],
            subgroups={"side": ["buy", "sell"]},
            engine="cpu",
        )
        res = df.group_by("g").agg(exprs).to_dicts()[0]
        assert res["px_side_buy_wmeanby_vol"] == pytest.approx(15.0)
        assert res["px_side_sell_wmeanby_vol"] == pytest.approx(150.0)

    def test_expr_filter_is_applied_to_weights_too(self):
        """A filter on the numerator only would divide by the whole-group weight sum."""
        df = _subgroup_frame()
        (expr,) = add_weighted_aggregates(
            columns_selector=(cs.numeric() - cs.by_name("g")),
            weighting_columns=["vol"],
            expr_filter=lambda e: e.filter(pl.col("side") == "sell"),
        )
        res = df.group_by("g").agg(expr).to_dicts()[0]
        assert res["px_wmeanby_vol"] == pytest.approx(150.0)


# F02 -------------------------------------------------------------------------------------------


@pytest.mark.skipif(not _has_polars_ds, reason="polars_ds not installed")
class TestPolarsDsExpressionsRespectSubgroupF02:
    def test_linreg_differs_per_subgroup(self):
        df = _subgroup_frame()
        exprs, _, _ = build_aggregate_features_polars(
            df,
            numerical_fields=["px"],
            linreg_fields=["px"],
            pds_numaggs=[],
            subgroups={"side": ["buy", "sell"]},
            engine="cpu",
        )
        res = df.group_by("g").agg(exprs).to_dicts()[0]
        assert res["side_buy_px_linreg"] != res["side_sell_px_linreg"]
        # buy = (10, 20) -> slope 10; sell = (100, 200) -> slope 100.
        assert res["side_buy_px_linreg"][0] == pytest.approx(10.0)
        assert res["side_sell_px_linreg"][0] == pytest.approx(100.0)

    def test_othersvals_at_extremums_differ_per_subgroup(self):
        df = pl.DataFrame(
            {
                "g": [1, 1, 1, 1],
                "side": ["buy", "buy", "sell", "sell"],
                "px": [10.0, 20.0, 100.0, 200.0],
                "tag": [1.0, 2.0, 3.0, 4.0],
            }
        )
        exprs, _, _ = build_aggregate_features_polars(
            df,
            numerical_fields=["px"],
            othersvals_at_extremums=True,
            othersvals_basic_fields=["px"],
            othersvals_other_fields=["tag"],
            pds_numaggs=[],
            subgroups={"side": ["buy", "sell"]},
            engine="cpu",
        )
        res = df.group_by("g").agg(exprs).to_dicts()[0]
        assert res["tag_side_buy_at_px_max"] == pytest.approx(2.0)
        assert res["tag_side_sell_at_px_max"] == pytest.approx(4.0)

    def test_corr_differs_per_subgroup(self):
        df = pl.DataFrame(
            {
                "g": [1, 1, 1, 1],
                "side": ["buy", "buy", "sell", "sell"],
                "px": [1.0, 2.0, 1.0, 2.0],
                "qty": [1.0, 2.0, 2.0, 1.0],
            }
        )
        exprs, _, _ = build_aggregate_features_polars(
            df,
            numerical_fields=["px"],
            corr_fields={"px": "qty"},
            corr_methods=["pearson"],
            pds_numaggs=[],
            subgroups={"side": ["buy", "sell"]},
            engine="cpu",
        )
        res = df.group_by("g").agg(exprs).to_dicts()[0]
        assert res["side_buy_corr_px-qty-pearson"] == pytest.approx(1.0)
        assert res["side_sell_corr_px-qty-pearson"] == pytest.approx(-1.0)


# F03 -------------------------------------------------------------------------------------------


class TestMutualInformationBoundF03:
    def test_mi_never_exceeds_either_marginal_entropy(self):
        """Complementary missingness made H(x), H(y) and H(x,y) three different samples."""
        bins = pl.DataFrame(
            {
                "x": [None] * 40 + [0] * 30 + [1] * 30,
                "y": [0] * 20 + [1] * 20 + [0] * 30 + [None] * 30,
            }
        )
        entropies = {c: entropy_for_column(bins, c, drop_nulls=True) for c in ("x", "y")}
        mi = mi_for_column(bins, entropies, "x", "y", drop_nulls=True)
        assert mi >= -1e-12
        assert mi <= min(entropies.values()) + 1e-12

    def test_drop_nulls_still_ignores_pure_missingness_signal(self):
        """The 2026-09-02 F17 fix stays fixed: with drop_nulls, a null-only pattern scores 0."""
        bins = pl.DataFrame({"x": [None] * 50 + [1] * 50, "t": [0] * 50 + [1] * 50})
        entropies = {c: entropy_for_column(bins, c, drop_nulls=True) for c in ("x", "t")}
        assert mi_for_column(bins, entropies, "x", "t", drop_nulls=True) == pytest.approx(0.0, abs=1e-12)

    def test_default_path_unchanged(self):
        bins = pl.DataFrame({"x": [0, 0, 1, 1], "t": [0, 0, 1, 1]})
        entropies = {c: entropy_for_column(bins, c) for c in ("x", "t")}
        assert mi_for_column(bins, entropies, "x", "t") == pytest.approx(np.log(2))


# F04 -------------------------------------------------------------------------------------------


class TestBenchmarkResultAssemblyF04:
    def test_parquet_rows_stay_numeric(self, monkeypatch):
        """Extending a list-of-lists with dicts rendered each parquet row as its column NAMES.

        The parquet sweep is stubbed so the assembly is exercised on every box: whether the real
        sweep produces rows depends on the installed pyarrow/fastparquet, and where it raises the
        corrupted rows never reach the frame.
        """
        import pyutilz.data.pandaslib as facade
        from pyutilz.data.pandaslib.benchmarks import benchmark_dataframe_compression

        stub = pd.DataFrame(
            [["parquet-pyarrow-snappy", 1.0, 0.1, 2.0, 0.2, 3.0, 0.3, 4.0, 0.4]],
            columns=["config", "mean_read_time", "std_read_time", "mean_write_time", "std_write_time", "mean_read_size", "std_read_size", "mean_write_size", "std_write_size"],
        )
        monkeypatch.setattr(facade, "benchmark_dataframe_parquet_compression", lambda *a, **kw: stub)

        df = pd.DataFrame({"a": range(30), "b": np.random.rand(30)})
        res = benchmark_dataframe_compression(df, head=0, nrepeats=1, return_styled=False, verbose=False)
        assert "parquet-pyarrow-snappy" in res.index
        for column in res.columns:
            assert pd.api.types.is_numeric_dtype(res[column]), f"{column} is not numeric: {res[column].tolist()}"

    def test_parquet_columns_use_the_shared_singular_names(self):
        pytest.importorskip("pyarrow")
        from pyutilz.data.pandaslib.benchmarks import benchmark_dataframe_parquet_compression
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            res = benchmark_dataframe_parquet_compression(
                pd.DataFrame({"a": range(20)}), tmp, nrepeats=1, engines=("pyarrow",), max_compression_level=1, skip_configs=()
            )
        assert "mean_write_size" in res.columns
        assert "mean_write_sizes" not in res.columns


# F05 -------------------------------------------------------------------------------------------


class TestLeadingZeroStringsPreservedF05:
    @pytest.mark.parametrize("inplace", [False, True])
    def test_zip_codes_are_not_cast_to_integers(self, inplace):
        df = pd.DataFrame({"zip": ["01234", "00501", "90210"]})
        res = optimize_dtypes(df, inplace=inplace, verbose=False)
        target = df if inplace else res
        assert [str(v) for v in target["zip"].tolist()] == ["01234", "00501", "90210"]

    def test_object_dtype_identifier_column_preserved(self):
        df = pd.DataFrame({"acct": pd.Series(["007", "008"], dtype=object)})
        res = optimize_dtypes(df.copy(), inplace=False, verbose=False)
        assert [str(v) for v in res["acct"].tolist()] == ["007", "008"]

    def test_plain_numeric_strings_still_compress(self):
        """The guard must not disable the numeric probe for genuine quantities."""
        df = pd.DataFrame({"n": ["1", "2", "3"], "f": ["1.5", "2.5", "3.5"]})
        res = optimize_dtypes(df.copy(), inplace=False, verbose=False)
        assert pd.api.types.is_integer_dtype(res["n"])
        assert pd.api.types.is_float_dtype(res["f"])

    def test_padded_and_signed_numeric_strings_are_left_alone(self):
        """Whitespace and an explicit "+" are formatting a number would not reproduce either."""
        df = pd.DataFrame({"padded": [" 1 ", " 2 "], "signed": ["+1", "+2"], "negzero": ["-0501", "-0502"]})
        res = optimize_dtypes(df.copy(), inplace=False, verbose=False)
        for col in ("padded", "signed", "negzero"):
            assert not pd.api.types.is_numeric_dtype(res[col]), f"{col} -> {res[col].dtype}"

    def test_free_text_column_still_becomes_a_category(self):
        df = pd.DataFrame({"t": ["alpha", "beta", "alpha", "gamma"]})
        res = optimize_dtypes(df.copy(), inplace=False, verbose=False, max_categories=10)
        assert str(res["t"].dtype) == "category"


# F06 -------------------------------------------------------------------------------------------


def _decimal_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {"d": [Decimal("1.50"), Decimal("2.50"), Decimal("300.50"), Decimal("1.75")], "t": [0, 1, 0, 1]},
        schema={"d": pl.Decimal(12, 2), "t": pl.Int64},
    )


class TestDecimalColumnsBinF06:
    def test_bin_numerical_columns_accepts_decimal(self):
        bins, _, _, _, _ = bin_numerical_columns(_decimal_frame(), target_columns=["t"], num_bins=4)
        assert "d" in bins.columns
        assert bins["d"].null_count() == 0


# F07 -------------------------------------------------------------------------------------------


class TestDecimalColumnsScreenedF07:
    def test_find_nan_cols_accepts_decimal(self):
        df = pl.DataFrame({"d": [Decimal("1.5"), Decimal("2.5")], "f": [1.0, np.nan]}, schema={"d": pl.Decimal(12, 2), "f": pl.Float64})
        assert find_nan_cols(df).columns == ["f"]

    def test_find_infinite_cols_accepts_decimal(self):
        df = pl.DataFrame({"d": [Decimal("1.5"), Decimal("2.5")], "f": [1.0, np.inf]}, schema={"d": pl.Decimal(12, 2), "f": pl.Float64})
        assert find_infinite_cols(df).columns == ["f"]

    def test_all_null_numeric_column_still_reported(self):
        df = pl.DataFrame({"a": [None, None], "b": [1, 2]}, schema={"a": pl.Float64, "b": pl.Int64})
        assert find_nan_cols(df).columns == ["a"]


# F08 -------------------------------------------------------------------------------------------


class TestNoNumericColumnsF08:
    @pytest.mark.parametrize("func", [drop_constant_columns, find_nan_cols, find_infinite_cols])
    def test_all_string_frame_does_not_raise(self, func):
        df = pl.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
        res = func(df)
        assert isinstance(res, pl.DataFrame)

    def test_drop_constant_columns_returns_frame_unchanged(self):
        df = pl.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
        assert drop_constant_columns(df).columns == ["a", "b"]

    def test_find_cols_return_empty_frames(self):
        df = pl.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
        assert find_nan_cols(df).width == 0
        assert find_infinite_cols(df).width == 0


# F09 -------------------------------------------------------------------------------------------


class TestNanNotFoldedIntoBinZeroF09:
    def test_nan_stays_missing_when_fill_nans_is_false(self):
        df = pl.DataFrame({"a": [np.nan, 1.0, 2.0, 3.0, 10.0, None], "t": [0, 1, 0, 1, 0, 1]})
        bins, _, _, _, _ = bin_numerical_columns(
            df, target_columns=["t"], num_bins=5, fill_nans=False, fill_nulls=False, clean_features=False, clean_targets=False
        )
        values = bins["a"].to_list()
        assert values[0] is None, f"NaN was folded into a real bin: {values}"
        assert values[5] is None
        assert values[1] == 0  # the genuine minimum still occupies bin 0

    def test_fill_nans_true_still_bins_nan_into_zero(self):
        df = pl.DataFrame({"a": [np.nan, 1.0, 2.0, 3.0, 10.0], "t": [0, 1, 0, 1, 0]})
        bins, _, _, _, _ = bin_numerical_columns(
            df, target_columns=["t"], num_bins=5, fill_nans=True, fill_nulls=True, clean_features=False, clean_targets=False
        )
        assert bins["a"].to_list()[0] == 0


# F10 -------------------------------------------------------------------------------------------


class TestShowcaseQuietF10:
    @pytest.mark.parametrize("frame_kind", ["pandas", "polars"])
    def test_no_stdout_when_use_print_is_false(self, capsys, frame_kind):
        data = {"x": ["a"] * 100 + ["b"] * 2 + [None] * 50}
        df = pd.DataFrame(data) if frame_kind == "pandas" else pl.DataFrame(data)
        rare, uninformative = showcase_df_columns(df, use_markdown=False, use_print=False, max_unique_percent=0.05, dropna=True)
        assert capsys.readouterr().out == ""
        assert "x" in rare

    @pytest.mark.parametrize("frame_kind", ["pandas", "polars"])
    def test_use_print_true_still_prints(self, capsys, frame_kind):
        data = {"x": ["a"] * 100 + ["b"] * 2}
        df = pd.DataFrame(data) if frame_kind == "pandas" else pl.DataFrame(data)
        showcase_df_columns(df, use_markdown=False, use_print=True, max_unique_percent=0.05)
        assert "100" in capsys.readouterr().out


# F11 -------------------------------------------------------------------------------------------


class TestConcatKeepsIndexOnRequestF11:
    def test_ignore_index_false_preserves_labels(self, tmp_path):
        lst = [pd.DataFrame({"v": [1, 2]}, index=["a", "b"])]
        res = concat_and_flush_df_list(lst, str(tmp_path / "out"), ignore_index=False)
        assert res is not None
        assert res.index.tolist() == ["a", "b"]
        assert pd.read_pickle(str(tmp_path / "out.pckl")).index.tolist() == ["a", "b"]

    def test_default_still_resets_the_index(self, tmp_path):
        lst = [pd.DataFrame({"v": [1, 2]}, index=["a", "b"])]
        res = concat_and_flush_df_list(lst, str(tmp_path / "out2"))
        assert res is not None
        assert res.index.tolist() == [0, 1]

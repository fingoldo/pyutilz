"""Regression tests for the 2026-09-02 data/stats domain audit (F01..F17).

Every test here fails against the pre-fix source and passes after it.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from pyutilz.data.numpylib import div0, get_topk_indices, smart_ratios
from pyutilz.data.pandaslib.dtypes import (
    classify_column_types,
    convert_float64_to_float32,
    optimize_dtypes,
)
from pyutilz.data.pandaslib.frames import (
    get_non_stale_columns,
    get_suspiciously_constant_columns,
    remove_constant_columns,
    share_dataframe,
    showcase_df_columns,
)
from pyutilz.data.polarslib import (
    bin_numerical_columns,
    clean_numeric,
    drop_constant_columns,
    entropy_for_column,
    polars_df_info,
)
from pyutilz.stats.normality import normality_verdict

# F01 -------------------------------------------------------------------------------------------


class TestConstantColumnToleranceF01:
    def test_large_monotone_ids_are_kept(self):
        """np.allclose's relative rtol=1e-5 declared a strictly increasing int64 id column constant."""
        df = pl.DataFrame({"uid": [1_000_000_000, 1_000_000_001, 1_000_000_002]})
        assert drop_constant_columns(df).columns == ["uid"]

    def test_small_relative_spread_is_kept(self):
        df = pl.DataFrame({"big": [1e9, 1_000_002_000.0, 1_000_001_000.0]})
        assert drop_constant_columns(df).columns == ["big"]

    def test_truly_constant_column_still_dropped(self):
        df = pl.DataFrame({"c": [5.0, 5.0, 5.0], "v": [1.0, 2.0, 3.0]})
        assert drop_constant_columns(df).columns == ["v"]

    def test_bin_numerical_columns_keeps_large_ids(self):
        df = pl.DataFrame({"uid": [1_000_000_000, 1_000_000_001, 1_000_000_002], "t": [0, 1, 0]})
        bins, _, _, columns_to_drop, _ = bin_numerical_columns(df, target_columns=["t"], num_bins=4)
        assert "uid" not in columns_to_drop
        assert "uid" in bins.columns


# F02 -------------------------------------------------------------------------------------------


class TestBinNumericalDegenerateFramesF02:
    @pytest.mark.parametrize(
        "df",
        [
            pl.DataFrame(schema={"a": pl.Float64, "b": pl.Float64}),
            pl.DataFrame({"a": [1.0], "b": [2.0]}),
            pl.DataFrame({"a": [1.0, 1.0], "b": [2.0, 2.0]}),
        ],
        ids=["zero-row", "single-row", "all-constant"],
    )
    def test_no_out_of_bounds_crash(self, df):
        bins, _, _, columns_to_drop, _ = bin_numerical_columns(df, target_columns=[])
        assert bins.width == 0
        assert set(columns_to_drop) == {"a", "b"}


# F03 -------------------------------------------------------------------------------------------


class TestCleanNumericNullsF03:
    def test_nulls_are_preserved_by_default(self):
        df = pl.DataFrame({"x": [1.0, None, float("nan"), float("inf")]})
        out = df.select(clean_numeric(pl.col("x"), nans_filler=-1.0))["x"].to_list()
        assert out[0] == 1.0
        assert out[1] is None
        assert out[2] == -1.0
        assert out[3] == -1.0

    def test_fill_nulls_opt_in_restores_old_behavior(self):
        df = pl.DataFrame({"x": [1.0, None]})
        out = df.select(clean_numeric(pl.col("x"), nans_filler=-1.0, fill_nulls=True))["x"].to_list()
        assert out == [1.0, -1.0]

    def test_single_row_group_std_stays_missing(self):
        """A one-observation group has null std -- recording 0.0 fabricates 'measured zero dispersion'."""
        df = pl.DataFrame({"g": ["a", "a", "b"], "v": [1.0, 3.0, 7.0]})
        res = df.group_by("g", maintain_order=True).agg(clean_numeric(pl.col("v").std()).alias("s"))
        assert res.filter(pl.col("g").eq_missing("b"))["s"].to_list() == [None]


# F04 -------------------------------------------------------------------------------------------


class TestOptimizeDtypesInplaceContractF04:
    def test_not_inplace_leaves_string_columns_untouched(self):
        df = pd.DataFrame({"s": ["1", "2", "3"], "t": ["1.5", "2.5", "3.5"]})
        before = df.dtypes.to_dict()
        _ = optimize_dtypes(df, inplace=False)
        assert df.dtypes.to_dict() == before
        assert df["s"].tolist() == ["1", "2", "3"]

    def test_inplace_still_converts(self):
        df = pd.DataFrame({"s": ["1", "2", "3"]})
        optimize_dtypes(df, inplace=True)
        assert "int" in str(df["s"].dtype)


# F05 -------------------------------------------------------------------------------------------


class TestShareDataframeIndexF05:
    def test_index_is_preserved(self):
        d = pd.DataFrame({"a": [1.0, 2.0], "b": [10, 20]}, index=["r1", "r2"])
        shared = share_dataframe(d)
        assert list(shared.index) == ["r1", "r2"]
        assert shared.loc["r2", "b"] == 20


# F06 -------------------------------------------------------------------------------------------


class TestConstantColumnNullSemanticsF06:
    def test_siblings_agree_on_one_value_plus_nulls(self):
        d = pd.DataFrame({"mostly_nan": [1.0, np.nan, np.nan, np.nan], "real": [1.0, 2.0, 3.0, 4.0]})
        kept = get_non_stale_columns(d)
        assert get_suspiciously_constant_columns(d) == []
        remove_constant_columns(d)
        assert d.columns.tolist() == kept

    def test_all_nan_column_still_flagged(self):
        d = pd.DataFrame({"empty": [np.nan, np.nan], "real": [1.0, 2.0]})
        assert get_suspiciously_constant_columns(d) == ["empty"]


# F07 -------------------------------------------------------------------------------------------


class TestNormalityVerdictSmallSamplesF07:
    def test_anderson_darling_runs_from_n_8(self):
        rng = np.random.default_rng(0)
        res = normality_verdict(rng.normal(size=10))
        assert res["verdict"] != "too-few-samples"
        assert np.isfinite(res["ad_stat"])
        assert np.isfinite(res["ad_p"])
        assert res["verdict"].startswith("AD-only (n<20)")

    def test_below_8_still_too_few(self):
        assert normality_verdict(np.arange(7.0) + np.array([0, 1, 0, 1, 0, 1, 0.0]))["verdict"] == "too-few-samples"

    def test_strongly_non_normal_small_sample_is_rejected(self):
        sample = np.array([0.0] * 14 + [1.0, 2.0, 400.0, 900.0])
        assert normality_verdict(sample)["reject_normal"] is True


# F08 -------------------------------------------------------------------------------------------


class TestDiv0DenominatorMaskF08:
    def test_infinite_numerator_survives(self):
        out = div0(np.array([np.inf, 1.0]), np.array([1.0, 1.0]), na_fill=-999.0)
        assert out[0] == np.inf
        assert out[1] == 1.0

    def test_zero_denominator_still_filled(self):
        out = div0(np.array([1.0, 2.0]), np.array([0.0, 2.0]), na_fill=-999.0)
        assert out.tolist() == [-999.0, 1.0]

    def test_overflow_from_denormal_denominator_is_not_masked(self):
        out = smart_ratios(np.array([1.0, 2.0]), np.array([0.0, 1e-320]))
        assert np.isnan(out[0])
        assert out[1] == np.inf


# F09 -------------------------------------------------------------------------------------------


class TestOptimizeDtypesExactRoundtripF09:
    def _eps_frame(self):
        return pd.DataFrame({"eps": [1.0000000000000002, 1.0, 2.0, 3.0, 4.0]})

    def test_exact_roundtrip_rejects_last_bit_difference(self):
        out = optimize_dtypes(self._eps_frame(), inplace=False, float_to_int=False, exact_float_roundtrip=True)
        assert out["eps"].dtype == np.float64

    def test_default_keeps_the_decimal_heuristic(self):
        out = optimize_dtypes(self._eps_frame(), inplace=False, float_to_int=False)
        assert out["eps"].dtype == np.float32


# F10 -------------------------------------------------------------------------------------------


class TestClassifyColumnTypesF10:
    @pytest.mark.parametrize("dtype_str", ["period[D]", "timedelta64[ns]", "interval"])
    def test_non_numeric_dtypes_are_not_reported_numeric(self, dtype_str):
        dtype = pd.Series([], dtype=dtype_str).dtype
        *_, col_is_numeric = classify_column_types(dtype=dtype)
        assert col_is_numeric is False

    @pytest.mark.parametrize("dtype_str", ["int64", "float32", "uint8"])
    def test_real_numerics_unaffected(self, dtype_str):
        dtype = pd.Series([], dtype=dtype_str).dtype
        *_, col_is_numeric = classify_column_types(dtype=dtype)
        assert col_is_numeric is True


# F11 -------------------------------------------------------------------------------------------


class TestGetNonStaleColumnsSingleRowF11:
    def test_single_row_frame_keeps_all_columns(self):
        assert get_non_stale_columns(pd.DataFrame({"a": [1], "b": [2]})) == ["a", "b"]

    def test_zero_row_frame_unchanged(self):
        assert get_non_stale_columns(pd.DataFrame({"a": [], "b": []})) == ["a", "b"]

    def test_two_rows_still_detect_staleness(self):
        assert get_non_stale_columns(pd.DataFrame({"a": [1, 1], "b": [1, 2]})) == ["b"]


# F12 -------------------------------------------------------------------------------------------


class TestShowcaseDenominatorF12:
    def test_dropna_modes_report_different_fractions(self, capsys):
        data = {"c": ["a"] * 97 + ["b"] + [None] * 2}
        _, u_drop = showcase_df_columns(pd.DataFrame(data), use_markdown=False, use_print=True, max_unique_percent=0.02, dropna=True)
        _, u_keep = showcase_df_columns(pd.DataFrame(data), use_markdown=False, use_print=True, max_unique_percent=0.02, dropna=False)
        capsys.readouterr()
        assert u_drop["c"] == pytest.approx(1 - 97 / 98)
        assert u_keep["c"] == pytest.approx(1 - 97 / 100)
        assert u_drop["c"] != u_keep["c"]

    def test_polars_matches_pandas(self, capsys):
        data = {"c": ["a"] * 97 + ["b"] + [None] * 2}
        _, u_pl = showcase_df_columns(pl.DataFrame(data), use_markdown=False, use_print=True, max_unique_percent=0.02, dropna=True)
        capsys.readouterr()
        assert u_pl["c"] == pytest.approx(1 - 97 / 98)


# F13 -------------------------------------------------------------------------------------------


class TestPolarsDfInfoUnitsF13:
    def test_small_frame_does_not_report_gb(self):
        info = polars_df_info(pl.DataFrame({"a": [1, 2]}))
        assert "0.0+ GB" not in info
        assert "memory usage:" in info

    def test_large_frame_reports_mb(self):
        info = polars_df_info(pl.DataFrame({"a": np.zeros(1_000_000, dtype=np.float64)}))
        assert "MB" in info


# F14 -------------------------------------------------------------------------------------------


class TestConvertFloat64ToFloat32PurityF14:
    def test_input_frame_is_not_mutated(self):
        d = pd.DataFrame({"x": [1.0, 2.0]})
        out = convert_float64_to_float32(d)
        assert d["x"].dtype == np.float64
        assert out["x"].dtype == np.float32


# F15 -------------------------------------------------------------------------------------------


class TestSharedIntDtypeListF15:
    def test_polars_branch_survives_an_int128_less_polars(self, monkeypatch):
        """On older polars the hardcoded pl.Int128 in the pandaslib branch raised AttributeError;
        both mirrors now read the same guarded helper, so a dtype list without Int128 still works."""
        import pyutilz.data.polarslib as plib
        from pyutilz.data.pandaslib.dtypes import ensure_dataframe_float32_convertability

        monkeypatch.setattr(plib, "polars_castable_int_dtypes", lambda: [pl.Int32, pl.UInt32, pl.Int64, pl.UInt64])
        out = ensure_dataframe_float32_convertability(pl.DataFrame({"a": pl.Series([1, 2], dtype=pl.Int64)}))
        assert out["a"].dtype == pl.Float32

    def test_ensure_convertability_uses_the_helper(self):
        from pyutilz.data.pandaslib.dtypes import ensure_dataframe_float32_convertability

        out = ensure_dataframe_float32_convertability(pl.DataFrame({"a": pl.Series([1, 2], dtype=pl.Int64)}))
        assert out["a"].dtype == pl.Float32


# F16 -------------------------------------------------------------------------------------------


class TestTopKAllNaNF16:
    def test_require_finite_returns_minus_one(self):
        a = np.array([np.nan, np.nan, np.nan])
        assert get_topk_indices(a, k=1, highest=True, require_finite=True).tolist() == [-1]
        assert get_topk_indices(a, k=1, highest=False, require_finite=True).tolist() == [-1]

    def test_partial_nan_slice_marks_only_missing_slots(self):
        a = np.array([np.nan, 5.0, np.nan])
        assert get_topk_indices(a, k=2, highest=True, require_finite=True).tolist() == [1, -1]

    def test_default_behavior_unchanged(self):
        a = np.array([2.0, 0.0, 3.0])
        assert get_topk_indices(a, k=2, highest=True).tolist() == [2, 0]


# F17 -------------------------------------------------------------------------------------------


class TestEntropyNullHandlingF17:
    def test_drop_nulls_excludes_the_null_bin(self):
        b = pl.DataFrame({"x": [0, None, 1, 1]})
        with_nulls = entropy_for_column(b, "x")
        without = entropy_for_column(b, "x", drop_nulls=True)
        assert with_nulls == pytest.approx(1.0397207708399179)
        assert without == pytest.approx(0.6365141682948128)

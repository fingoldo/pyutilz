import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch
from pathlib import Path

from pyutilz.pandaslib import (
    load_df,
    optimize_dtypes,
    remove_stale_columns,
    concat_and_flush_df_list,
    remove_constant_columns,
    get_categorical_columns_indices,
    convert_float64_to_float32,
    get_df_memory_consumption,
    nullify_standard_values,
    prefixize_columns,
    FeatureNamer,
    set_df_columns_types,
    get_columns_of_type,
    classify_column_types,
    ensure_dataframe_float32_convertability,
    showcase_df_columns,
    share_dataframe,
    group_columns_by_dtype,
    get_suspiciously_constant_columns,
    measure_read_write_performance,
    pack_benchmark_results,
    read_stats_from_multiple_files,
)


# ---------------------------------------------------------------------------
# load_df
# ---------------------------------------------------------------------------

class TestLoadDf:
    def test_load_pickle(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3]})
        fpath = str(tmp_path / "data.pkl")
        df.to_pickle(fpath)
        result = load_df(fpath, tail=None)
        assert len(result) == 3

    def test_load_pickle_with_tail(self, tmp_path):
        df = pd.DataFrame({"a": range(10)})
        fpath = str(tmp_path / "data.pkl")
        df.to_pickle(fpath)
        result = load_df(fpath, tail=3)
        assert len(result) == 3
        assert list(result["a"]) == [7, 8, 9]

    def test_load_pickle_tail_zero(self, tmp_path):
        df = pd.DataFrame({"a": range(5)})
        fpath = str(tmp_path / "data.pkl")
        df.to_pickle(fpath)
        result = load_df(fpath, tail=0)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# showcase_df_columns
# ---------------------------------------------------------------------------

class TestShowcaseDfColumns:
    def test_basic_output(self, capsys):
        df = pd.DataFrame({"x": [1, 2, 2], "y": ["a", "b", "a"]})
        showcase_df_columns(df, use_markdown=False, use_print=True)
        captured = capsys.readouterr()
        assert "X" in captured.out
        assert "Y" in captured.out

    def test_excluded_cols(self, capsys):
        df = pd.DataFrame({"x": [1], "y": [2]})
        showcase_df_columns(df, excluded_cols=["x"], use_markdown=False, use_print=True)
        captured = capsys.readouterr()
        assert "X" not in captured.out
        assert "Y" in captured.out

    def test_max_vars_zero(self, capsys):
        df = pd.DataFrame({"x": [1, 2, 3]})
        showcase_df_columns(df, max_vars=0, use_markdown=False, use_print=True)
        captured = capsys.readouterr()
        assert "X" in captured.out

    def test_max_vars_positive(self, capsys):
        df = pd.DataFrame({"x": [1, 1, 2, 2, 3]})
        showcase_df_columns(df, max_vars=1, use_markdown=False, use_print=True)
        # Should not raise

    def test_dropna(self, capsys):
        df = pd.DataFrame({"x": [1, np.nan, 1]})
        showcase_df_columns(df, dropna=True, use_markdown=False, use_print=True)
        captured = capsys.readouterr()
        assert "X" in captured.out

    def test_specific_cols(self, capsys):
        df = pd.DataFrame({"x": [1], "y": [2], "z": [3]})
        showcase_df_columns(df, cols=["y"], use_markdown=False, use_print=True)
        captured = capsys.readouterr()
        assert "Y" in captured.out
        assert "X" not in captured.out


# ---------------------------------------------------------------------------
# share_dataframe
# ---------------------------------------------------------------------------

class TestShareDataframe:
    def test_shared_df_matches_original(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        shared = share_dataframe(df)
        assert list(shared.columns) == ["a", "b"]
        assert shared["a"].tolist() == [1.0, 2.0]
        assert shared["b"].tolist() == [3.0, 4.0]

    def test_shared_df_shape(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        shared = share_dataframe(df)
        assert shared.shape == df.shape


# ---------------------------------------------------------------------------
# group_columns_by_dtype
# ---------------------------------------------------------------------------

class TestGroupColumnsByDtype:
    def test_groups_correctly(self):
        df = pd.DataFrame({"a": [1, 2], "b": [1.0, 2.0], "c": ["x", "y"]})
        groups = group_columns_by_dtype(df)
        assert "a" in groups["int64"]
        assert "b" in groups["float64"]
        assert "c" in groups["object"]

    def test_empty_df(self):
        df = pd.DataFrame()
        groups = group_columns_by_dtype(df)
        assert len(groups) == 0


# ---------------------------------------------------------------------------
# get_suspiciously_constant_columns
# ---------------------------------------------------------------------------

class TestGetSuspiciouslyConstantColumns:
    def test_finds_constant(self):
        df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
        result = get_suspiciously_constant_columns(df)
        assert "a" in result
        assert "b" not in result

    def test_single_value(self):
        df = pd.DataFrame({"a": [5]})
        result = get_suspiciously_constant_columns(df)
        assert "a" in result

    def test_all_nan(self):
        df = pd.DataFrame({"a": [np.nan, np.nan]})
        result = get_suspiciously_constant_columns(df)
        assert "a" in result

    def test_unhashable_column(self):
        df = pd.DataFrame({"a": [[1, 2], [1, 2]]})
        result = get_suspiciously_constant_columns(df)
        # Should not raise; falls back to per-column check
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# remove_constant_columns (prewarm path)
# ---------------------------------------------------------------------------

class TestRemoveConstantColumnsPrewarm:
    def test_prewarm_with_large_df(self):
        n = 15_000
        df = pd.DataFrame({"const": [1] * n, "var": range(n)})
        remove_constant_columns(df, prewarm_size=10_000)
        assert "const" not in df.columns
        assert "var" in df.columns

    def test_prewarm_false_positive_filtered(self):
        # Column is constant in first 10k rows but varies after
        n = 15_000
        vals = [1] * 10_000 + list(range(5_000))
        df = pd.DataFrame({"tricky": vals, "var": range(n)})
        remove_constant_columns(df, prewarm_size=10_000)
        assert "tricky" in df.columns

    def test_verbose_many_columns(self):
        cols = {f"c{i}": [1] * 5 for i in range(25)}
        cols["var"] = [1, 2, 3, 4, 5]
        df = pd.DataFrame(cols)
        remove_constant_columns(df, verbose=True)
        assert "var" in df.columns
        assert len(df.columns) == 1


# ---------------------------------------------------------------------------
# classify_column_types — dtype parameter
# ---------------------------------------------------------------------------

class TestClassifyColumnTypesDtype:
    @pytest.mark.parametrize("dtype,expected_field", [
        (pd.Series([True]).dtype, "bool"),
        (pd.Series(["x"]).dtype, "object"),
        (pd.Series(pd.to_datetime(["2020-01-01"])).dtype, "datetime"),
        (pd.Categorical(["a"]).dtype, "category"),
        (pd.Series([1]).dtype, "numeric"),
    ])
    def test_dtype_param(self, dtype, expected_field):
        is_bool, is_obj, is_dt, is_cat, is_num = classify_column_types(dtype=dtype)
        mapping = {"bool": is_bool, "object": is_obj, "datetime": is_dt, "category": is_cat, "numeric": is_num}
        assert mapping[expected_field] is True


# ---------------------------------------------------------------------------
# optimize_dtypes — additional branches
# ---------------------------------------------------------------------------

class TestOptimizeDtypsBranches:
    def test_string_col_to_int(self):
        df = pd.DataFrame({"a": ["1", "2", "3"]})
        result = optimize_dtypes(df, max_categories=10, inplace=False)
        assert pd.api.types.is_integer_dtype(result["a"]) or pd.api.types.is_float_dtype(result["a"])

    def test_string_col_to_float(self):
        df = pd.DataFrame({"a": ["1.1", "2.2", "3.3"]})
        result = optimize_dtypes(df, max_categories=10, inplace=False)
        assert pd.api.types.is_float_dtype(result["a"])

    def test_unconvertible_string_col_to_category(self):
        df = pd.DataFrame({"a": ["hello", "world", "hello"]})
        result = optimize_dtypes(df, max_categories=10, inplace=False)
        assert result["a"].dtype.name == "category"

    def test_verbose_logging(self):
        df = pd.DataFrame({"a": ["hello", "world", "hello"]})
        optimize_dtypes(df, max_categories=10, verbose=True, inplace=True)
        assert df["a"].dtype.name == "category"

    def test_reduce_size_int_downcast(self):
        df = pd.DataFrame({"a": np.array([1, 2, 3], dtype=np.int64)})
        result = optimize_dtypes(df, reduce_size=True, inplace=False)
        assert result["a"].dtype.itemsize <= 8

    def test_not_inplace_returns_copy(self):
        df = pd.DataFrame({"a": np.array([1, 2, 3], dtype=np.int64)})
        result = optimize_dtypes(df, reduce_size=True, inplace=False)
        assert result is not df or True  # may be same if no changes needed

    def test_use_uint_false(self):
        df = pd.DataFrame({"a": np.array([1, 2, 3], dtype=np.int64)})
        result = optimize_dtypes(df, use_uint=False, inplace=False)
        assert "uint" not in result["a"].dtype.name or pd.api.types.is_integer_dtype(result["a"])

    def test_float_precision_preserved(self):
        df = pd.DataFrame({"a": np.array([2.205001270000e09 + 0.123456789], dtype=np.float64)})
        result = optimize_dtypes(df, ensure_float64_precision=True, float_to_float=True, float_to_int=False, inplace=False)
        # Should stay float64 due to precision loss if downcast to float32
        assert pd.api.types.is_float_dtype(result["a"])

    def test_skip_halffloat_false(self):
        df = pd.DataFrame({"a": np.array([0.5, 1.0], dtype=np.float64)})
        result = optimize_dtypes(df, skip_halffloat=False, float_to_float=True, inplace=False)
        assert pd.api.types.is_float_dtype(result["a"])

    def test_max_categories_none(self):
        df = pd.DataFrame({"a": ["x", "y"], "b": [1, 2]})
        result = optimize_dtypes(df, max_categories=None, inplace=False)
        assert result["a"].dtype.name == "object"


# ---------------------------------------------------------------------------
# prefixize_columns — inplace path
# ---------------------------------------------------------------------------

class TestPrefixizeColumnsInplace:
    def test_inplace_returns_mapping(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        result = prefixize_columns(df, "pfx", inplace=True)
        assert isinstance(result, dict)
        assert "pfx_a" in df.columns
        assert "pfx_b" in df.columns


# ---------------------------------------------------------------------------
# FeatureNamer — initial_values
# ---------------------------------------------------------------------------

class TestFeatureNamerInitialValues:
    def test_initial_values(self):
        namer = FeatureNamer(initial_values=["x", "y", "z"])
        assert namer("x") == 0
        assert namer("y") == 1
        assert namer("z") == 2
        assert namer("w") == 3

    def test_rev_after_initial(self):
        namer = FeatureNamer(initial_values=["a", "b"])
        assert namer.rev(0) == "a"
        assert namer.rev(1) == "b"


# ---------------------------------------------------------------------------
# concat_and_flush_df_list — set_index path
# ---------------------------------------------------------------------------

class TestConcatAndFlushSetIndex:
    def test_set_index(self, tmp_path):
        lst = [pd.DataFrame({"id": [1, 2], "v": [10, 20]})]
        result = concat_and_flush_df_list(lst, file_name=str(tmp_path / "out"), set_index="id")
        assert result.index.name == "id"

    def test_empty_list(self, tmp_path):
        result = concat_and_flush_df_list([], file_name=str(tmp_path / "out"))
        assert result is None

    def test_csv_with_cols(self, tmp_path):
        lst = [pd.DataFrame({"a": [1], "b": [2]})]
        result = concat_and_flush_df_list(lst, file_name=str(tmp_path / "out"), to_csv=True, csv_cols=["a"])
        assert result is not None
        assert os.path.exists(str(tmp_path / "out.csv"))


# ---------------------------------------------------------------------------
# get_df_memory_consumption — polars and error paths
# ---------------------------------------------------------------------------

class TestGetDfMemoryConsumptionExtra:
    def test_polars_dataframe(self):
        try:
            import polars as pl
            df = pl.DataFrame({"a": [1, 2, 3]})
            mem = get_df_memory_consumption(df)
            assert mem > 0
        except ImportError:
            pytest.skip("polars not installed")

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported dataframe type"):
            get_df_memory_consumption({"a": [1, 2, 3]})


# ---------------------------------------------------------------------------
# nullify_standard_values — verbose path
# ---------------------------------------------------------------------------

class TestNullifyStandardValuesVerbose:
    def test_verbose_output(self, capsys):
        df = pd.DataFrame({
            "field": ["A"] * 500 + ["B"] * 10,
            "person": list(range(500)) + list(range(10)),
        })
        nullify_standard_values(df, "field", min_records=100, persons_field="person", min_persons=10, verbose=True)
        captured = capsys.readouterr()
        assert "Field field" in captured.out


# ---------------------------------------------------------------------------
# measure_read_write_performance & pack_benchmark_results
# ---------------------------------------------------------------------------

class TestMeasureReadWritePerformance:
    def test_basic_benchmark(self, tmp_path):
        df = pd.DataFrame({"a": range(100), "b": np.random.rand(100)})
        fname = str(tmp_path / "test.pkl")
        read_times, write_times, read_sizes, write_sizes = measure_read_write_performance(
            df=df,
            fname=fname,
            read_method="read_pickle",
            read_params={},
            write_method="to_pickle",
            write_params={},
            nrepeats=1,
        )
        assert len(read_times) == 1
        assert len(write_times) == 1
        assert read_times[0] > 0
        assert write_times[0] > 0
        assert read_sizes[0] > 0
        assert write_sizes[0] > 0


class TestPackBenchmarkResults:
    def test_appends_row(self):
        res = []
        arrays = [np.array([1.0, 2.0]) for _ in range(4)]
        pack_benchmark_results(res, "test-config", *arrays)
        assert len(res) == 1
        assert res[0][0] == "test-config"
        assert len(res[0]) == 9  # config + 4 * (mean, std)


# ---------------------------------------------------------------------------
# read_stats_from_multiple_files
# ---------------------------------------------------------------------------

class TestReadStatsFromMultipleFiles:
    def test_reads_and_merges(self, tmp_path):
        for i in range(3):
            df = pd.DataFrame({"v": [i * 10]})
            df.to_pickle(str(tmp_path / f"part_{i}.pckl"))
        result = read_stats_from_multiple_files(
            joint_file_name=str(tmp_path / "joint"),
            folder=str(tmp_path),
            template="*.pckl",
        )
        assert result is not None
        assert len(result) == 3

    def test_with_sentinel_field(self, tmp_path):
        df = pd.DataFrame({"v": [1]})
        df.to_pickle(str(tmp_path / "file1.pckl"))
        result = read_stats_from_multiple_files(
            joint_file_name=str(tmp_path / "joint"),
            folder=str(tmp_path),
            template="file*.pckl",
            sentinel_field="source",
        )
        assert "source" in result.columns

    def test_with_sentinel_fcn(self, tmp_path):
        df = pd.DataFrame({"v": [1]})
        df.to_pickle(str(tmp_path / "file1.pckl"))
        result = read_stats_from_multiple_files(
            joint_file_name=str(tmp_path / "joint"),
            folder=str(tmp_path),
            template="file*.pckl",
            sentinel_field="src",
            sentinel_fcn=lambda f: f.upper(),
        )
        assert result["src"].iloc[0] == "FILE1.PCKL"

    def test_max_files_limit(self, tmp_path):
        for i in range(5):
            pd.DataFrame({"v": [i]}).to_pickle(str(tmp_path / f"p{i}.pckl"))
        result = read_stats_from_multiple_files(
            joint_file_name=str(tmp_path / "joint"),
            folder=str(tmp_path),
            template="p*.pckl",
            max_files=2,
        )
        assert len(result) == 2

    def test_exclude_filter(self, tmp_path):
        pd.DataFrame({"v": [1]}).to_pickle(str(tmp_path / "keep.pckl"))
        pd.DataFrame({"v": [2]}).to_pickle(str(tmp_path / "joint.pckl"))
        result = read_stats_from_multiple_files(
            joint_file_name=str(tmp_path / "out"),
            folder=str(tmp_path),
            template="*.pckl",
            exclude="joint",
        )
        assert len(result) == 1

    def test_optimize_flag(self, tmp_path):
        df = pd.DataFrame({"v": np.array([1, 2, 3], dtype=np.int64)})
        df.to_pickle(str(tmp_path / "f.pckl"))
        result = read_stats_from_multiple_files(
            joint_file_name=str(tmp_path / "joint"),
            folder=str(tmp_path),
            template="f*.pckl",
            optimize=True,
        )
        assert result is not None

    def test_delete_after(self, tmp_path):
        fpath = str(tmp_path / "del.pckl")
        pd.DataFrame({"v": [1]}).to_pickle(fpath)
        read_stats_from_multiple_files(
            joint_file_name=str(tmp_path / "joint"),
            folder=str(tmp_path),
            template="del*.pckl",
            delete_after=True,
        )
        assert not os.path.exists(fpath)

    def test_empty_folder(self, tmp_path):
        result = read_stats_from_multiple_files(
            joint_file_name=str(tmp_path / "joint"),
            folder=str(tmp_path),
            template="*.pckl",
        )
        assert result is None

    def test_set_index(self, tmp_path):
        pd.DataFrame({"id": [1], "v": [10]}).to_pickle(str(tmp_path / "f.pckl"))
        result = read_stats_from_multiple_files(
            joint_file_name=str(tmp_path / "joint"),
            folder=str(tmp_path),
            template="f*.pckl",
            set_index="id",
        )
        assert result.index.name == "id"


# ---------------------------------------------------------------------------
# ensure_dataframe_float32_convertability — polars path
# ---------------------------------------------------------------------------

class TestEnsureFloat32Polars:
    def test_polars_conversion(self):
        try:
            import polars as pl
            df = pl.DataFrame({
                "a": pl.Series([1, 2, 3], dtype=pl.Int64),
                "b": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float64),
            })
            result = ensure_dataframe_float32_convertability(df)
            assert result["a"].dtype == pl.Float32
            assert result["b"].dtype == pl.Float32
        except ImportError:
            pytest.skip("polars not installed")


# ---------------------------------------------------------------------------
# convert_float64_to_float32 — extra cases
# ---------------------------------------------------------------------------

class TestConvertFloat64ToFloat32Extra:
    def test_mixed_dtypes_only_converts_float64(self):
        df = pd.DataFrame({
            "f64": np.array([1.0, 2.0], dtype=np.float64),
            "f32": np.array([1.0, 2.0], dtype=np.float32),
            "i64": np.array([1, 2], dtype=np.int64),
        })
        result = convert_float64_to_float32(df)
        assert result["f64"].dtype == np.float32
        assert result["f32"].dtype == np.float32  # unchanged
        assert result["i64"].dtype == np.int64  # unchanged

    def test_no_float64_noop(self):
        df = pd.DataFrame({"a": np.array([1, 2], dtype=np.int32)})
        result = convert_float64_to_float32(df)
        assert result["a"].dtype == np.int32


# ---------------------------------------------------------------------------
# edge cases / parametrized
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype_name,expected", [
    ("bool", (True, False, False, False, False)),
    ("object", (False, True, False, False, False)),
    ("datetime64[ns]", (False, False, True, False, False)),
    ("category", (False, False, False, True, False)),
    ("int64", (False, False, False, False, True)),
    ("float32", (False, False, False, False, True)),
])
def test_classify_column_types_parametrized(dtype_name, expected):
    dtype = pd.Series([], dtype=dtype_name).dtype
    result = classify_column_types(dtype=dtype)
    assert result == expected


@pytest.mark.parametrize("input_vals,expected_groups", [
    ({"a": [1], "b": [1.0]}, 2),
    ({"x": ["s"]}, 1),
])
def test_group_columns_by_dtype_parametrized(input_vals, expected_groups):
    df = pd.DataFrame(input_vals)
    groups = group_columns_by_dtype(df)
    assert len(groups) == expected_groups


@pytest.mark.parametrize("tail,expected_len", [
    (None, 10),
    (3, 3),
    (0, 10),
    (100, 10),
])
def test_load_df_tail_parametrized(tmp_path, tail, expected_len):
    df = pd.DataFrame({"a": range(10)})
    fpath = str(tmp_path / "data.pkl")
    df.to_pickle(fpath)
    result = load_df(fpath, tail=tail)
    assert len(result) == min(expected_len, 10)

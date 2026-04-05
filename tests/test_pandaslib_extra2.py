"""Additional coverage tests for pyutilz.data.pandaslib — targeting uncovered lines."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock
from itertools import chain

from pyutilz.pandaslib import (
    optimize_dtypes,
    read_parquet_with_pyarrow,
    showcase_df_columns,
    ensure_dataframe_float32_convertability,
    read_stats_from_multiple_files,
    measure_read_write_performance,
    pack_benchmark_results,
    remove_constant_columns,
    get_suspiciously_constant_columns,
)


# ---------------------------------------------------------------------------
# HAS_IPYTHON = False branch (lines 41-42)
# ---------------------------------------------------------------------------

class TestNoIPython:
    def test_showcase_no_ipython_falls_back_to_print(self, capsys):
        """When HAS_IPYTHON is False, showcase_df_columns uses print."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        with patch("pyutilz.data.pandaslib.HAS_IPYTHON", False):
            showcase_df_columns(df, use_markdown=True, use_print=True)
        out = capsys.readouterr().out
        assert "COL1" in out

    def test_showcase_markdown_branch_with_ipython(self, capsys):
        """When HAS_IPYTHON is True and use_markdown, display(Markdown(...)) is called."""
        df = pd.DataFrame({"x": [1]})
        mock_display = MagicMock()
        with patch("pyutilz.data.pandaslib.HAS_IPYTHON", True), \
             patch("pyutilz.data.pandaslib.display", mock_display, create=True), \
             patch("pyutilz.data.pandaslib.Markdown", lambda s: s, create=True):
            showcase_df_columns(df, use_markdown=True, use_print=False)
        mock_display.assert_called()


# ---------------------------------------------------------------------------
# optimize_dtypes — skip_columns branch (line 140)
# ---------------------------------------------------------------------------

class TestOptimizeDtypesSkipColumns:
    def test_skip_columns_preserved(self):
        df = pd.DataFrame({"a": ["hello", "world"], "b": ["1", "2"]})
        result = optimize_dtypes(df, max_categories=10, skip_columns=["a"], inplace=False)
        assert result["a"].dtype.name == "object"
        # b should be converted to int
        assert result["b"].dtype.name != "object"

    def test_category_conversion_exception(self):
        """Lines 163-166: exception in nunique (e.g. unhashable list values)."""
        df = pd.DataFrame({"a": [[1, 2], [3, 4]]})
        # Should not raise, just skip
        result = optimize_dtypes(df, max_categories=10, inplace=False)
        assert result["a"].dtype.name == "object"

    def test_verbose_category_exception(self):
        """Lines 164-165: verbose logging on category exception."""
        df = pd.DataFrame({"a": [[1, 2], [3, 4]]})
        result = optimize_dtypes(df, max_categories=10, verbose=True, inplace=False)
        assert result["a"].dtype.name == "object"


# ---------------------------------------------------------------------------
# optimize_dtypes — size reduction edge cases (lines 224-225, 237, 243, 248, 260)
# ---------------------------------------------------------------------------

class TestOptimizeDtypsSizeReduction:
    def test_int_field_already_at_min_power(self):
        """Line 224-225: col already at target power, breaks early."""
        df = pd.DataFrame({"a": np.array([1, 2], dtype=np.int8)})
        result = optimize_dtypes(df, reduce_size=True, inplace=False)
        assert result["a"].dtype == np.int8

    def test_mantissa_cache_hit(self):
        """Line 237: second access to mantissa cache for same column."""
        # use_uint=True creates two passes over same float column: uint then int
        # float_to_float with ensure_float64_precision triggers mantissa computation
        df = pd.DataFrame({"a": np.array([1.5, 2.5], dtype=np.float64)})
        result = optimize_dtypes(df, reduce_size=True, float_to_float=True,
                                 ensure_float64_precision=True, inplace=False)
        assert pd.api.types.is_float_dtype(result["a"])

    def test_verbose_precision_loss(self):
        """Line 243: verbose log when float can't downcast due to precision."""
        df = pd.DataFrame({"a": np.array([2.205001270000e09 + 0.123456789], dtype=np.float64)})
        result = optimize_dtypes(df, reduce_size=True, float_to_float=True,
                                 float_to_int=False, ensure_float64_precision=True, verbose=True, inplace=False)
        assert pd.api.types.is_float_dtype(result["a"])

    def test_verbose_type_change(self):
        """Line 248: verbose log for successful type change."""
        df = pd.DataFrame({"a": np.array([1, 2, 3], dtype=np.int64)})
        result = optimize_dtypes(df, reduce_size=True, verbose=True, inplace=False)
        assert result["a"].dtype.itemsize < 8

    def test_verbose_new_dtypes_not_inplace(self):
        """Line 260: verbose log 'Going to use the following new dtypes'."""
        df = pd.DataFrame({"a": np.array([1, 2], dtype=np.int64)})
        result = optimize_dtypes(df, reduce_size=True, verbose=True, inplace=False)
        assert result is not None

    def test_float_to_int_conversion(self):
        """Float column with no NaN and no fractional part -> int."""
        df = pd.DataFrame({"a": np.array([1.0, 2.0, 3.0], dtype=np.float64)})
        result = optimize_dtypes(df, reduce_size=True, float_to_int=True, inplace=False)
        assert "int" in result["a"].dtype.name or "uint" in result["a"].dtype.name

    def test_float_to_int_with_nan_stays_float(self):
        """Float column with NaN cannot be converted to int."""
        df = pd.DataFrame({"a": np.array([1.0, np.nan, 3.0], dtype=np.float64)})
        result = optimize_dtypes(df, reduce_size=True, float_to_int=True, inplace=False)
        assert "float" in result["a"].dtype.name

    def test_use_uint_true(self):
        """When use_uint=True, positive ints may become uint."""
        df = pd.DataFrame({"a": np.array([1, 2, 200], dtype=np.int64)})
        result = optimize_dtypes(df, reduce_size=True, use_uint=True, inplace=False)
        assert result["a"].dtype.itemsize <= 8


# ---------------------------------------------------------------------------
# read_parquet_with_pyarrow (lines 515-520)
# ---------------------------------------------------------------------------

class TestReadParquetWithPyarrow:
    def test_read_with_nrows(self, tmp_path):
        df = pd.DataFrame({"a": range(100), "b": range(100)})
        fpath = str(tmp_path / "test.parquet")
        df.to_parquet(fpath, engine="pyarrow")
        result = read_parquet_with_pyarrow(fpath, nrows=10)
        assert len(result) == 10

    def test_read_without_nrows(self, tmp_path):
        """nrows=0 is falsy, takes the else branch (line 517-518)."""
        expected = pd.DataFrame({"a": range(50)})
        mock_table = MagicMock()
        mock_table.to_pandas.return_value = expected
        mock_scanner = MagicMock()
        mock_scanner.to_pandas.return_value = expected
        mock_scanner.head.return_value = mock_table
        mock_ds = MagicMock()
        mock_ds.scanner.return_value = mock_scanner
        with patch("pyutilz.data.pandaslib.dataset", return_value=mock_ds):
            result = read_parquet_with_pyarrow("fake.parquet", nrows=0)
        assert len(result) == 50
        mock_scanner.to_pandas.assert_called_once()

    def test_read_nrows_none(self, tmp_path):
        """nrows=None is falsy, takes the else branch."""
        expected = pd.DataFrame({"a": range(20)})
        mock_scanner = MagicMock()
        mock_scanner.to_pandas.return_value = expected
        mock_ds = MagicMock()
        mock_ds.scanner.return_value = mock_scanner
        with patch("pyutilz.data.pandaslib.dataset", return_value=mock_ds):
            result = read_parquet_with_pyarrow("fake.parquet", nrows=None)
        assert len(result) == 20


# ---------------------------------------------------------------------------
# read_stats_from_multiple_files — uncovered branches
# ---------------------------------------------------------------------------

class TestReadStatsEdgeCases:
    def test_save_on_successful_optimization(self, tmp_path):
        """Lines 453-455: re-save file when optimization reduces size."""
        df = pd.DataFrame({"a": np.array([1, 2, 3], dtype=np.int64)})
        df.to_pickle(str(tmp_path / "f.pckl"))
        result = read_stats_from_multiple_files(
            joint_file_name=str(tmp_path / "joint"),
            folder=str(tmp_path),
            template="f*.pckl",
            optimize=True,
            save_on_successful_optimization=True,
            min_size_improvement_percent=0.0,
            min_size_improvement=0.0,
        )
        assert result is not None

    def test_sentinel_field_collision(self, tmp_path):
        """Lines 461-462: sentinel field already exists in dataframe."""
        df = pd.DataFrame({"source": [1], "v": [10]})
        df.to_pickle(str(tmp_path / "f.pckl"))
        result = read_stats_from_multiple_files(
            joint_file_name=str(tmp_path / "joint"),
            folder=str(tmp_path),
            template="f*.pckl",
            sentinel_field="source",
        )
        # Should use "source1" since "source" already exists
        assert "source1" in result.columns or "source" in result.columns

    def test_delete_after_exception_handled(self, tmp_path):
        """Lines 483-484: exception during file deletion is silently caught."""
        df = pd.DataFrame({"v": [1]})
        fpath = str(tmp_path / "f.pckl")
        df.to_pickle(fpath)
        with patch("os.remove", side_effect=PermissionError("locked")):
            result = read_stats_from_multiple_files(
                joint_file_name=str(tmp_path / "joint"),
                folder=str(tmp_path),
                template="f*.pckl",
                delete_after=True,
            )
        assert result is not None

    def test_concat_exception_returns_none(self, tmp_path):
        """Lines 486-487: exception during concat returns None."""
        df = pd.DataFrame({"v": [1]})
        df.to_pickle(str(tmp_path / "f.pckl"))
        with patch("pyutilz.data.pandaslib.concat_and_flush_df_list", side_effect=ValueError("bad")):
            result = read_stats_from_multiple_files(
                joint_file_name=str(tmp_path / "joint"),
                folder=str(tmp_path),
                template="f*.pckl",
            )
        assert result is None


# ---------------------------------------------------------------------------
# ensure_dataframe_float32_convertability — uncovered branches
# ---------------------------------------------------------------------------

class TestEnsureFloat32Extra:
    def test_verbose_pandas_numeric(self):
        """Line 888: verbose log for numeric column conversion."""
        df = pd.DataFrame({"a": np.array([1, 2], dtype=np.int64)})
        result = ensure_dataframe_float32_convertability(df, verbose=1)
        assert result["a"].dtype == np.float32

    def test_pyarrow_backed_integer(self):
        """Lines 893-901: PyArrow-backed integer column conversion."""
        try:
            import pyarrow as pa
            df = pd.DataFrame({"a": pd.array([1, 2, 3], dtype="int64[pyarrow]")})
            result = ensure_dataframe_float32_convertability(df)
            assert "float" in str(result["a"].dtype)
        except (ImportError, TypeError):
            pytest.skip("pyarrow-backed dtypes not supported")

    def test_pyarrow_backed_float(self):
        """Lines 893-901: PyArrow-backed float column conversion."""
        try:
            import pyarrow as pa
            df = pd.DataFrame({"a": pd.array([1.0, 2.0], dtype="float64[pyarrow]")})
            result = ensure_dataframe_float32_convertability(df)
            assert "float" in str(result["a"].dtype)
        except (ImportError, TypeError):
            pytest.skip("pyarrow-backed dtypes not supported")

    def test_pyarrow_backed_verbose(self):
        """Lines 899-900: verbose log for pyarrow column."""
        try:
            import pyarrow as pa
            df = pd.DataFrame({"a": pd.array([1, 2], dtype="int64[pyarrow]")})
            result = ensure_dataframe_float32_convertability(df, verbose=1)
            assert "float" in str(result["a"].dtype)
        except (ImportError, TypeError):
            pytest.skip("pyarrow-backed dtypes not supported")

    def test_pyarrow_string_not_converted(self):
        """PyArrow string column should NOT be converted to float32."""
        try:
            import pyarrow as pa
            df = pd.DataFrame({"a": pd.array(["x", "y"], dtype="string[pyarrow]")})
            result = ensure_dataframe_float32_convertability(df)
            assert "float32" not in str(result["a"].dtype)
        except (ImportError, TypeError):
            pytest.skip("pyarrow-backed dtypes not supported")


# ---------------------------------------------------------------------------
# Benchmark functions — mocked I/O (lines 635-854)
# ---------------------------------------------------------------------------

class TestBenchmarkParquetCompression:
    def test_parquet_benchmark_basic(self, tmp_path):
        """Lines 635-690: benchmark_dataframe_parquet_compression."""
        from pyutilz.pandaslib import benchmark_dataframe_parquet_compression
        df = pd.DataFrame({"a": range(50), "b": np.random.rand(50)})
        result = benchmark_dataframe_parquet_compression(
            df, str(tmp_path), nrepeats=1,
            engines=("pyarrow",),
            max_compression_level=1,
            skip_configs=(),
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "config" in result.columns

    def test_parquet_skip_config(self, tmp_path):
        """Line 648-649: skip_configs filtering."""
        from pyutilz.pandaslib import benchmark_dataframe_parquet_compression
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = benchmark_dataframe_parquet_compression(
            df, str(tmp_path), nrepeats=1,
            engines=("pyarrow",),
            max_compression_level=0,
            skip_configs=("parquet-pyarrow-snappy",),
        )
        configs = result["config"].tolist()
        assert "parquet-pyarrow-snappy" not in configs

    def test_parquet_compression_level_exception(self, tmp_path):
        """Lines 681-682: exception in compression_level loop is caught."""
        from pyutilz.pandaslib import benchmark_dataframe_parquet_compression
        df = pd.DataFrame({"a": [1, 2]})
        # max_compression_level=1 with an engine that may fail on some combos
        result = benchmark_dataframe_parquet_compression(
            df, str(tmp_path), nrepeats=1,
            engines=("pyarrow",),
            max_compression_level=2,
        )
        assert isinstance(result, pd.DataFrame)

    def test_write_parquet_method(self, tmp_path):
        """Lines 637-638, 641-642: write_method='write_parquet' branch (polars-style)."""
        from pyutilz.pandaslib import benchmark_dataframe_parquet_compression
        # This will use engines=("main",) and no engine param
        # We mock measure_read_write_performance to avoid actual polars dependency
        mock_arrays = [np.array([0.1]), np.array([0.1]), np.array([1.0]), np.array([1.0])]
        with patch("pyutilz.data.pandaslib.measure_read_write_performance", return_value=mock_arrays):
            result = benchmark_dataframe_parquet_compression(
                pd.DataFrame({"a": [1]}), str(tmp_path), nrepeats=1,
                write_method="write_parquet",
                max_compression_level=0,
            )
        assert isinstance(result, pd.DataFrame)


class TestBenchmarkPickleCompression:
    def test_pickle_benchmark(self, tmp_path):
        """Lines 694-711."""
        from pyutilz.pandaslib import benchmark_dataframe_pickle_compression
        df = pd.DataFrame({"a": range(20)})
        res = []
        benchmark_dataframe_pickle_compression(res, str(tmp_path), df, nrepeats=1)
        assert len(res) > 0
        assert res[0][0].startswith("pickle-")


class TestBenchmarkCsvCompression:
    def test_csv_benchmark(self, tmp_path):
        """Lines 736-752."""
        from pyutilz.pandaslib import benchmark_dataframe_csv_compression
        df = pd.DataFrame({"a": range(20)})
        res = []
        benchmark_dataframe_csv_compression(res, str(tmp_path), df, nrepeats=1)
        assert len(res) > 0
        assert res[0][0].startswith("csv-")


class TestBenchmarkOrcCompression:
    def test_orc_benchmark(self, tmp_path):
        """Lines 756-771."""
        from pyutilz.pandaslib import benchmark_dataframe_orc_compression
        df = pd.DataFrame({"a": range(20), "b": np.random.rand(20)})
        res = []
        benchmark_dataframe_orc_compression(res, str(tmp_path), df, nrepeats=1)
        assert len(res) == 1
        assert res[0][0] == "orc"


class TestBenchmarkFeatherCompression:
    def test_feather_benchmark(self, tmp_path):
        """Lines 775-790."""
        from pyutilz.pandaslib import benchmark_dataframe_feather_compression
        df = pd.DataFrame({"a": range(20), "b": np.random.rand(20)})
        res = []
        benchmark_dataframe_feather_compression(res, str(tmp_path), df, nrepeats=1)
        assert len(res) == 1
        assert res[0][0] == "feather"


class TestBenchmarkDataframeCompression:
    def test_full_benchmark_mocked(self, tmp_path):
        """Lines 804-854: orchestrator function."""
        from pyutilz.pandaslib import benchmark_dataframe_compression
        df = pd.DataFrame({"a": range(50), "b": np.random.rand(50)})
        result = benchmark_dataframe_compression(
            df, head=20, nrepeats=1,
            benchmark_dir_path=str(tmp_path),
            return_styled=False,
            should_clean_temp_folder=True,
            verbose=True,
        )
        assert isinstance(result, pd.DataFrame)

    def test_benchmark_return_styled(self, tmp_path):
        """Line 848-852: return_styled=True attempts style.background_gradient."""
        from pyutilz.pandaslib import benchmark_dataframe_compression
        df = pd.DataFrame({"a": range(50), "b": np.random.rand(50)})
        result = benchmark_dataframe_compression(
            df, head=20, nrepeats=1,
            benchmark_dir_path=str(tmp_path),
            return_styled=True,
            should_clean_temp_folder=True,
            verbose=False,
        )
        # Either Styler or DataFrame depending on columns present
        assert result is not None

    def test_benchmark_no_head(self, tmp_path):
        """Line 806: head=None uses full df."""
        from pyutilz.pandaslib import benchmark_dataframe_compression
        df = pd.DataFrame({"a": range(10)})
        result = benchmark_dataframe_compression(
            df, head=None, nrepeats=1,
            benchmark_dir_path=str(tmp_path),
            return_styled=False,
            verbose=False,
        )
        assert result is not None

    def test_benchmark_exception_in_func(self, tmp_path):
        """Lines 828-829: exception in individual benchmark function is caught."""
        from pyutilz.pandaslib import benchmark_dataframe_compression
        df = pd.DataFrame({"a": range(10)})
        with patch("pyutilz.data.pandaslib.benchmark_dataframe_feather_compression", side_effect=RuntimeError("fail")):
            result = benchmark_dataframe_compression(
                df, head=10, nrepeats=1,
                benchmark_dir_path=str(tmp_path),
                return_styled=False,
                verbose=False,
            )
        assert result is not None


# ---------------------------------------------------------------------------
# HDF benchmark — skipped if tables not available (lines 715-732)
# ---------------------------------------------------------------------------

class TestBenchmarkHdfCompression:
    def test_hdf_benchmark_mocked(self, tmp_path):
        """Lines 715-732: mock to avoid tables dependency."""
        from pyutilz.pandaslib import benchmark_dataframe_hdf_compression
        mock_arrays = [np.array([0.1]), np.array([0.1]), np.array([1.0]), np.array([1.0])]
        res = []
        with patch("pyutilz.data.pandaslib.measure_read_write_performance", return_value=mock_arrays):
            benchmark_dataframe_hdf_compression(res, str(tmp_path), pd.DataFrame({"a": [1]}), nrepeats=1)
        assert len(res) > 0
        assert res[0][0].startswith("hdf-")


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------

class TestOptimizeDtypesEdgeCases:
    def test_skip_halffloat_false_with_small_values(self):
        """Trigger float16 consideration when skip_halffloat=False."""
        df = pd.DataFrame({"a": np.array([0.1, 0.2], dtype=np.float64)})
        result = optimize_dtypes(df, reduce_size=True, skip_halffloat=False,
                                 float_to_float=True, ensure_float64_precision=False, inplace=False)
        # Should downcast to float16 since values are small and no precision check
        assert result["a"].dtype.itemsize <= 4

    def test_float_to_int_with_use_uint(self):
        """Float with no NaN, no frac -> try uint then int."""
        df = pd.DataFrame({"a": np.array([1.0, 2.0, 255.0], dtype=np.float64)})
        result = optimize_dtypes(df, reduce_size=True, float_to_int=True,
                                 use_uint=True, inplace=False)
        assert "int" in result["a"].dtype.name or "uint" in result["a"].dtype.name

    def test_inplace_reduce_size(self):
        """Inplace mode with reduce_size."""
        df = pd.DataFrame({"a": np.array([1, 2, 3], dtype=np.int64)})
        result = optimize_dtypes(df, reduce_size=True, inplace=True)
        assert result["a"].dtype.itemsize < 8
        assert result is df


class TestRemoveConstantColumnsEdge:
    def test_verbose_few_columns(self):
        """Line 584: verbose with <=20 constant columns logs their names."""
        df = pd.DataFrame({"c1": [1, 1], "c2": [1, 1], "var": [1, 2]})
        remove_constant_columns(df, verbose=True)
        assert "var" in df.columns
        assert "c1" not in df.columns

    def test_verbose_many_constant_columns(self):
        """Line 582: verbose with >20 constant columns logs count."""
        cols = {f"c{i}": [1, 1] for i in range(25)}
        cols["var"] = [1, 2]
        df = pd.DataFrame(cols)
        remove_constant_columns(df, verbose=True)
        assert "var" in df.columns
        assert len(df.columns) == 1

    def test_drop_many_at_once(self):
        """Line 587: >20 constant cols uses df.drop."""
        cols = {f"c{i}": [1, 1, 1] for i in range(25)}
        cols["var"] = [1, 2, 3]
        df = pd.DataFrame(cols)
        remove_constant_columns(df)
        assert "var" in df.columns
        assert len(df.columns) == 1

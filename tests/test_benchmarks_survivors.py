"""Behaviour tests for the pandas benchmark sweeps: config names, file names, call wiring, timings, sizes and cleanup."""

from __future__ import annotations

import logging
import os
import warnings

import numpy as np
import pandas as pd
import pytest
from pandas.io.formats.style import Styler

import pyutilz.data.pandaslib as facade
from pyutilz.data.pandaslib import benchmarks

PARQUET_COLUMNS = [
    "config",
    "mean_read_time",
    "std_read_time",
    "mean_write_time",
    "std_write_time",
    "mean_read_size",
    "std_read_size",
    "mean_write_size",
    "std_write_size",
]


@pytest.fixture
def frame():
    return pd.DataFrame({"a": np.arange(40), "s": [f"x{i}" for i in range(40)]})


@pytest.fixture
def calls(monkeypatch):
    """Replace the facade's measure step with a recorder returning one constant value per repeat."""
    recorded = []

    def fake_measure(df, fname, read_method, read_params, write_method, write_params, nrepeats):
        recorded.append(
            dict(df=df, fname=fname, read_method=read_method, read_params=read_params, write_method=write_method, write_params=write_params, nrepeats=nrepeats)
        )
        values = np.full(nrepeats, 1.0)
        return values, values, values, values

    monkeypatch.setattr(facade, "measure_read_write_performance", fake_measure)
    return recorded


@pytest.fixture
def fake_backends(monkeypatch):
    """Replace every format backend with a recorder that appends one row with varying values."""
    rows = []

    def backend(name):
        def run(res, temp_folder, df, nrepeats):
            rows.append((name, temp_folder, df, nrepeats))
            base = float(len(res))
            res.append([f"{name}-row", *[base + j if j % 2 == 0 else 0.5 for j in range(8)]])

        return run

    def parquet(df, temp_folder, nrepeats):
        rows.append(("parquet", temp_folder, df, nrepeats))
        base = 100.0
        return pd.DataFrame([["parquet-row"] + [base + j if j % 2 == 0 else 0.5 for j in range(8)]], columns=PARQUET_COLUMNS)

    monkeypatch.setattr(facade, "benchmark_dataframe_feather_compression", backend("feather"))
    monkeypatch.setattr(facade, "benchmark_dataframe_orc_compression", backend("orc"))
    monkeypatch.setattr(facade, "benchmark_dataframe_hdf_compression", backend("hdf"))
    monkeypatch.setattr(facade, "benchmark_dataframe_pickle_compression", backend("pickle"))
    monkeypatch.setattr(facade, "benchmark_dataframe_csv_compression", backend("csv"))
    monkeypatch.setattr(facade, "benchmark_dataframe_parquet_compression", parquet)
    return rows


class TestParquetSweep:
    def test_default_sweep_has_every_engine_codec_and_level_except_the_skipped_brotli_block(self, frame, tmp_path, calls):
        out = benchmarks.benchmark_dataframe_parquet_compression(frame, str(tmp_path), nrepeats=3)
        names = out["config"].tolist()
        assert len(names) == 207
        assert "parquet-fastparquet-brotli" not in names
        assert "parquet-fastparquet-brotli-1" not in names
        assert "parquet-pyarrow-brotli-1" in names
        assert list(out.columns) == PARQUET_COLUMNS

    def test_the_first_rows_are_the_base_config_then_its_levels_in_order(self, frame, tmp_path, calls):
        out = benchmarks.benchmark_dataframe_parquet_compression(frame, str(tmp_path), nrepeats=1, max_compression_level=3)
        names = out["config"].tolist()
        assert names[:4] == [
            "parquet-fastparquet-snappy",
            "parquet-fastparquet-snappy-1",
            "parquet-fastparquet-snappy-2",
            "parquet-fastparquet-snappy-3",
        ]
        assert names[4] == "parquet-fastparquet-gzip"
        assert "parquet-pyarrow-snappy" in names

    def test_each_config_writes_to_its_own_named_file_in_the_temp_folder(self, frame, tmp_path, calls):
        benchmarks.benchmark_dataframe_parquet_compression(frame, str(tmp_path), nrepeats=1, max_compression_level=1)
        assert calls[0]["fname"] == os.path.join(str(tmp_path), "parquet-fastparquet-snappy.parquet")
        assert calls[1]["fname"] == os.path.join(str(tmp_path), "parquet-fastparquet-snappy-1.parquet")

    def test_engine_and_compression_options_reach_read_and_write(self, frame, tmp_path, calls):
        benchmarks.benchmark_dataframe_parquet_compression(frame, str(tmp_path), nrepeats=2, max_compression_level=1)
        base, level = calls[0], calls[1]
        assert base["read_method"] == "read_parquet"
        assert base["write_method"] == "to_parquet"
        assert base["read_params"] == {"engine": "fastparquet"}
        assert base["write_params"] == {"engine": "fastparquet", "compression": "snappy"}
        assert base["nrepeats"] == 2
        assert level["write_params"] == {"engine": "fastparquet", "compression": "snappy", "compression_level": 1}

    def test_write_parquet_runs_only_the_main_engine_without_engine_options(self, frame, tmp_path, calls):
        out = benchmarks.benchmark_dataframe_parquet_compression(
            frame, str(tmp_path), nrepeats=1, max_compression_level=1, write_method="write_parquet", skip_configs=()
        )
        assert out["config"].iloc[0] == "parquet-main-snappy"
        assert len(out) == 10
        assert calls[0]["read_params"] == {}
        assert calls[0]["write_params"] == {"compression": "snappy"}
        assert all(c["write_method"] == "write_parquet" for c in calls)

    def test_an_empty_skip_list_keeps_the_brotli_configs(self, frame, tmp_path, calls):
        out = benchmarks.benchmark_dataframe_parquet_compression(frame, str(tmp_path), nrepeats=1, max_compression_level=1, skip_configs=())
        assert "parquet-fastparquet-brotli" in out["config"].tolist()
        assert len(out) == 2 * 5 * 2

    def test_a_failing_level_is_logged_and_skipped_while_later_levels_still_run(self, frame, tmp_path, monkeypatch, caplog):
        def flaky(**kwargs):
            if kwargs["fname"].endswith("parquet-pyarrow-snappy-2.parquet"):
                raise ValueError("bad level")
            values = np.full(kwargs["nrepeats"], 1.0)
            return values, values, values, values

        monkeypatch.setattr(facade, "measure_read_write_performance", flaky)
        caplog.set_level(logging.WARNING)
        out = benchmarks.benchmark_dataframe_parquet_compression(frame, str(tmp_path), nrepeats=1, max_compression_level=3, skip_configs=())
        names = out["config"].tolist()
        assert "parquet-pyarrow-snappy-2" not in names
        assert "parquet-pyarrow-snappy-3" in names
        assert "Skipping config parquet-pyarrow-snappy-2" in caplog.text


class TestRunCompressionSweep:
    def test_an_empty_label_names_the_config_and_file_after_the_format_alone(self, frame, tmp_path, calls):
        res = []
        benchmarks._run_compression_sweep(res, str(tmp_path), frame, 3, "orc", [("", "read_orc", {"a": 1}, "to_orc", {"b": 2})])
        assert res[0][0] == "orc"
        assert calls[0]["fname"] == os.path.join(str(tmp_path), "orc.orc")
        assert calls[0]["nrepeats"] == 3
        assert calls[0]["df"] is frame
        assert (calls[0]["read_method"], calls[0]["write_method"]) == ("read_orc", "to_orc")
        assert calls[0]["read_params"] == {"a": 1}
        assert calls[0]["write_params"] == {"b": 2}

    def test_a_label_becomes_a_suffix_in_both_the_config_and_the_file_name(self, frame, tmp_path, calls):
        res = []
        benchmarks._run_compression_sweep(res, str(tmp_path), frame, 1, "csv", [("gzip", "read_csv", {}, "to_csv", {})])
        assert res[0][0] == "csv-gzip"
        assert calls[0]["fname"] == os.path.join(str(tmp_path), "csv-gzip.csv.gzip")


class TestEachFormat:
    def test_pickle_sweep_uses_pickle_methods_and_protocol_minus_one_on_every_write(self, frame, tmp_path, calls):
        res = []
        benchmarks.benchmark_dataframe_pickle_compression(res, str(tmp_path), frame, 1)
        assert [r[0] for r in res] == ["pickle-zip", "pickle-gzip", "pickle-bz2", "pickle-zstd", "pickle-xz", "pickle-tar"]
        assert all(c["read_method"] == "read_pickle" and c["write_method"] == "to_pickle" for c in calls)
        assert all(c["write_params"]["protocol"] == -1 for c in calls)
        assert calls[0]["read_params"] == {"compression": {"method": "zip"}}

    def test_csv_sweep_has_no_protocol_option(self, frame, tmp_path, calls):
        res = []
        benchmarks.benchmark_dataframe_csv_compression(res, str(tmp_path), frame, 1)
        assert all(c["read_method"] == "read_csv" and c["write_method"] == "to_csv" for c in calls)
        assert all("protocol" not in c["write_params"] for c in calls)

    def test_hdf_sweep_crosses_four_libraries_with_levels_one_to_nine(self, frame, tmp_path, calls):
        res = []
        benchmarks.benchmark_dataframe_hdf_compression(res, str(tmp_path), frame, 1)
        assert len(res) == 36 == len(calls)
        assert {c["write_params"]["complevel"] for c in calls} == set(range(1, 10))
        assert {c["write_params"]["complib"] for c in calls} == {"zlib", "lzo", "bzip2", "blosc"}
        assert all(c["write_params"]["key"] == "test" for c in calls)
        assert all(c["read_method"] == "read_hdf" and c["write_method"] == "to_hdf" for c in calls)
        assert calls[0]["read_params"] == {"complib": "zlib"}

    def test_orc_and_feather_each_run_one_default_config_with_their_own_methods(self, frame, tmp_path, calls):
        res = []
        benchmarks.benchmark_dataframe_orc_compression(res, str(tmp_path), frame, 2)
        benchmarks.benchmark_dataframe_feather_compression(res, str(tmp_path), frame, 2)
        assert [r[0] for r in res] == ["orc", "feather"]
        assert (calls[0]["read_method"], calls[0]["write_method"]) == ("read_orc", "to_orc")
        assert (calls[1]["read_method"], calls[1]["write_method"]) == ("read_feather", "to_feather")
        assert all(c["nrepeats"] == 2 for c in calls)


class TestMeasureArithmetic:
    def test_durations_are_the_clock_difference_for_each_step(self, frame, tmp_path, monkeypatch):
        clock = iter([5.0, 8.0, 20.0, 24.0])
        monkeypatch.setattr(benchmarks, "timer", lambda: next(clock))
        read_times, write_times, _, _ = benchmarks.measure_read_write_performance(frame, str(tmp_path / "f.csv"), "read_csv", {}, "to_csv", {"index": False}, 1)
        assert write_times[0] == 3.0
        assert read_times[0] == 4.0

    def test_sizes_are_megabytes_of_the_file_and_of_the_read_back_frame(self, frame, tmp_path, monkeypatch):
        monkeypatch.setattr(benchmarks, "getsize", lambda path: 3 * 1024**2)
        monkeypatch.setattr(benchmarks, "get_df_memory_consumption", lambda df: 2 * 1024**2)
        _, _, read_sizes, write_sizes = benchmarks.measure_read_write_performance(frame, str(tmp_path / "f.csv"), "read_csv", {}, "to_csv", {"index": False}, 1)
        assert write_sizes[0] == 3.0
        assert read_sizes[0] == 2.0

    def test_garbage_collection_runs_before_and_after_each_read_and_write(self, frame, tmp_path, monkeypatch):
        collected = []
        monkeypatch.setattr(benchmarks.gc, "collect", lambda: collected.append(1))
        benchmarks.measure_read_write_performance(frame, str(tmp_path / "f.csv"), "read_csv", {}, "to_csv", {"index": False}, 2)
        assert len(collected) == 4


class TestFullBenchmarkWiring:
    def test_every_backend_gets_the_frame_temp_folder_and_repeats_in_order(self, frame, fake_backends):
        benchmarks.benchmark_dataframe_compression(frame, head=0, nrepeats=4, return_styled=False, verbose=False)
        assert [name for name, *_ in fake_backends] == ["feather", "orc", "hdf", "pickle", "csv", "parquet"]
        for _name, temp_folder, df, nrepeats in fake_backends:
            assert isinstance(temp_folder, str)
            assert df is frame
            assert nrepeats == 4

    def test_head_limits_the_frame_and_the_default_head_is_one_hundred_thousand(self, fake_backends):
        big = pd.DataFrame({"a": np.arange(100_001)})
        benchmarks.benchmark_dataframe_compression(big, nrepeats=1, return_styled=False, verbose=False)
        assert all(len(df) == 100_000 for _, _, df, _ in fake_backends)

    def test_the_default_repeat_count_is_ten(self, frame, fake_backends):
        benchmarks.benchmark_dataframe_compression(frame, head=0, return_styled=False, verbose=False)
        assert all(nrepeats == 10 for *_, nrepeats in fake_backends)

    def test_the_table_is_indexed_by_config_with_only_the_four_mean_columns(self, frame, fake_backends):
        out = benchmarks.benchmark_dataframe_compression(frame, head=0, return_styled=False, verbose=False)
        assert out.index.name == "config"
        assert list(out.columns) == ["mean_read_time", "mean_write_time", "mean_read_size", "mean_write_size"]
        assert set(out.index) == {"feather-row", "orc-row", "hdf-row", "pickle-row", "csv-row", "parquet-row"}

    def test_the_parquet_rows_are_kept_in_the_table(self, frame, fake_backends):
        out = benchmarks.benchmark_dataframe_compression(frame, head=0, return_styled=False, verbose=False)
        assert "parquet-row" in out.index

    def test_sort_by_orders_the_table_ascending_on_the_chosen_column(self, frame, fake_backends):
        out = benchmarks.benchmark_dataframe_compression(frame, head=0, sort_by="mean_read_time", return_styled=False, verbose=False)
        assert out["mean_read_time"].is_monotonic_increasing

    def test_the_gradient_is_applied_to_the_three_comparison_columns_by_default(self, frame, fake_backends, monkeypatch):
        seen = {}

        def record(self, axis=None, subset=None, **kwargs):
            seen.update(axis=axis, subset=subset)
            return self

        monkeypatch.setattr(Styler, "background_gradient", record)
        out = benchmarks.benchmark_dataframe_compression(frame, head=0, verbose=False)
        assert isinstance(out, Styler)
        assert seen == {"axis": None, "subset": ["mean_write_size", "mean_write_time", "mean_read_time"]}

    def test_a_failing_gradient_is_logged_and_the_plain_table_is_returned(self, frame, fake_backends, monkeypatch, caplog):
        def boom(self, **kwargs):
            raise RuntimeError("gradient broke")

        monkeypatch.setattr(Styler, "background_gradient", boom)
        caplog.set_level(logging.ERROR)
        out = benchmarks.benchmark_dataframe_compression(frame, head=0, verbose=False)
        assert isinstance(out, pd.DataFrame)
        assert "gradient broke" in caplog.text

    def test_verbose_logs_the_pandas_version_and_the_frame_size(self, frame, fake_backends, caplog):
        caplog.set_level(logging.INFO)
        benchmarks.benchmark_dataframe_compression(frame, head=0, return_styled=False, verbose=True)
        expected_mb = benchmarks.get_df_memory_consumption(frame) / (1024**2)
        assert f"Pandas: {pd.__version__}" in caplog.text
        assert f"DF size: {expected_mb:.2f}Mb" in caplog.text

    def test_quiet_run_logs_nothing_about_the_frame(self, frame, fake_backends, caplog):
        caplog.set_level(logging.INFO)
        benchmarks.benchmark_dataframe_compression(frame, head=0, return_styled=False, verbose=False)
        assert "DF size" not in caplog.text

    def test_a_failing_backend_is_logged_and_the_others_still_run(self, frame, fake_backends, monkeypatch, caplog):
        def boom(res, temp_folder, df, nrepeats):
            raise RuntimeError("orc broke")

        monkeypatch.setattr(facade, "benchmark_dataframe_orc_compression", boom)
        caplog.set_level(logging.ERROR)
        out = benchmarks.benchmark_dataframe_compression(frame, head=0, return_styled=False, verbose=False)
        assert "orc broke" in caplog.text
        assert "csv-row" in out.index

    def test_a_failing_parquet_run_is_logged_and_the_other_formats_remain(self, frame, fake_backends, monkeypatch, caplog):
        def boom(df, temp_folder, nrepeats):
            raise RuntimeError("parquet broke")

        monkeypatch.setattr(facade, "benchmark_dataframe_parquet_compression", boom)
        caplog.set_level(logging.ERROR)
        out = benchmarks.benchmark_dataframe_compression(frame, head=0, return_styled=False, verbose=False)
        assert "parquet broke" in caplog.text
        assert "csv-row" in out.index
        assert "parquet-row" not in out.index

    def test_the_temp_folder_is_created_inside_a_missing_benchmark_directory_and_removed(self, frame, fake_backends, tmp_path):
        bench = tmp_path / "bench"
        benchmarks.benchmark_dataframe_compression(frame, head=0, benchmark_dir_path=str(bench), nrepeats=1, return_styled=False, verbose=False)
        assert bench.is_dir()
        assert list(bench.iterdir()) == []

    def test_keeping_the_temp_folder_leaves_it_in_place(self, frame, fake_backends, tmp_path):
        bench = tmp_path / "bench"
        benchmarks.benchmark_dataframe_compression(
            frame, head=0, benchmark_dir_path=str(bench), nrepeats=1, return_styled=False, verbose=False, should_clean_temp_folder=False
        )
        assert len(list(bench.iterdir())) == 1

    def test_performance_warnings_are_ignored_during_the_run(self, frame, fake_backends, monkeypatch):
        monkeypatch.setattr(warnings, "filters", list(warnings.filters))
        benchmarks.benchmark_dataframe_compression(frame, head=0, return_styled=False, verbose=False)
        assert any(f[0] == "ignore" and f[2] is pd.errors.PerformanceWarning for f in warnings.filters)

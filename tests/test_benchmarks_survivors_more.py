"""Behaviour tests for the benchmark survivors that need distinct timings, real reads, default arguments and named configs."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

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
def distinct_measure(monkeypatch):
    """Measure step returning a different constant per metric, so pack_benchmark_results argument order is visible."""
    recorded = []

    def fake_measure(df, fname, read_method, read_params, write_method, write_params, nrepeats):
        recorded.append(dict(fname=fname, read_method=read_method, read_params=read_params, write_params=write_params, nrepeats=nrepeats))
        return np.full(nrepeats, 1.0), np.full(nrepeats, 2.0), np.full(nrepeats, 3.0), np.full(nrepeats, 4.0)

    monkeypatch.setattr(facade, "measure_read_write_performance", fake_measure)
    return recorded


@pytest.fixture
def backend_rows(monkeypatch):
    """Replace every backend with a recorder that appends one row per call, varying only the mean columns."""
    rows = []

    def backend(name):
        def run(res, temp_folder, df, nrepeats):
            rows.append((name, temp_folder, df, nrepeats))
            res.append([f"{name}-row", float(len(res)), 0.5, float(len(res)) * 2, 0.5, float(len(res)) * 3, 0.5, float(len(res)) * 4, 0.5])

        return run

    def parquet(df, temp_folder, nrepeats):
        rows.append(("parquet", temp_folder, df, nrepeats))
        return pd.DataFrame([["parquet-row", 1.0, 0.5, 2.0, 0.5, 3.0, 0.5, 4.0, 0.5]], columns=PARQUET_COLUMNS)

    monkeypatch.setattr(facade, "benchmark_dataframe_feather_compression", backend("feather"))
    monkeypatch.setattr(facade, "benchmark_dataframe_orc_compression", backend("orc"))
    monkeypatch.setattr(facade, "benchmark_dataframe_hdf_compression", backend("hdf"))
    monkeypatch.setattr(facade, "benchmark_dataframe_pickle_compression", backend("pickle"))
    monkeypatch.setattr(facade, "benchmark_dataframe_csv_compression", backend("csv"))
    monkeypatch.setattr(facade, "benchmark_dataframe_parquet_compression", parquet)
    return rows


class TestPackArgumentOrder:
    def test_each_summary_row_carries_the_metrics_in_read_write_size_order(self, frame, tmp_path, distinct_measure):
        res = []
        benchmarks._run_compression_sweep(res, str(tmp_path), frame, 2, "csv", [("", "read_csv", {}, "to_csv", {})])
        assert res[0] == ["csv", 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0]


class TestParquetDetails:
    def test_the_default_repeat_count_for_parquet_is_three(self, frame, tmp_path, distinct_measure):
        benchmarks.benchmark_dataframe_parquet_compression(frame, str(tmp_path), max_compression_level=1, skip_configs=())
        assert all(c["nrepeats"] == 3 for c in distinct_measure)

    def test_every_compression_level_reads_back_with_read_parquet(self, frame, tmp_path, distinct_measure):
        benchmarks.benchmark_dataframe_parquet_compression(frame, str(tmp_path), nrepeats=1, max_compression_level=2, skip_configs=())
        assert all(c["read_method"] == "read_parquet" for c in distinct_measure)

    def test_a_failing_level_config_is_named_in_the_log_and_the_base_still_lands(self, frame, tmp_path, monkeypatch, caplog):
        def flaky(**kwargs):
            if kwargs["fname"].endswith("parquet-fastparquet-snappy-1.parquet"):
                raise ValueError("bad")
            values = np.full(kwargs["nrepeats"], 1.0)
            return values, values, values, values

        monkeypatch.setattr(facade, "measure_read_write_performance", flaky)
        caplog.set_level(logging.WARNING)
        out = benchmarks.benchmark_dataframe_parquet_compression(frame, str(tmp_path), nrepeats=1, max_compression_level=1, skip_configs=())
        assert "parquet-fastparquet-snappy" in out["config"].tolist()
        assert "Skipping config parquet-fastparquet-snappy-1" in caplog.text


class TestNamedConfigs:
    def test_hdf_rows_are_named_by_library(self, frame, tmp_path, distinct_measure):
        res = []
        benchmarks.benchmark_dataframe_hdf_compression(res, str(tmp_path), frame, 1)
        assert {r[0] for r in res} == {"hdf-zlib", "hdf-lzo", "hdf-bzip2", "hdf-blosc"}

    def test_csv_rows_are_named_and_write_the_compression_method_only(self, frame, tmp_path, distinct_measure):
        res = []
        benchmarks.benchmark_dataframe_csv_compression(res, str(tmp_path), frame, 1)
        assert [r[0] for r in res] == ["csv-zip", "csv-gzip", "csv-bz2", "csv-zstd", "csv-xz", "csv-tar"]
        assert distinct_measure[0]["write_params"] == {"compression": {"method": "zip"}}


class TestMeasureReadParams:
    def test_read_parameters_are_passed_as_keywords_to_the_reader(self, frame, tmp_path):
        path = tmp_path / "f.csv"
        frame.to_csv(path, index=False)
        _, _, read_sizes, _ = benchmarks.measure_read_write_performance(frame, str(path), "read_csv", {"usecols": ["a"]}, "to_csv", {"index": False}, 1)
        expected = benchmarks.get_df_memory_consumption(pd.read_csv(path, usecols=["a"])) / (1024**2)
        assert read_sizes[0] == pytest.approx(expected)


class TestFullBenchmarkDetails:
    def test_the_default_repeat_count_is_ten_and_the_default_verbose_logs(self, frame, backend_rows, caplog):
        caplog.set_level(logging.INFO)
        benchmarks.benchmark_dataframe_compression(frame, head=0, return_styled=False)
        assert all(nrepeats == 10 for *_, nrepeats in backend_rows)
        assert "Pandas:" in caplog.text

    def test_the_default_head_keeps_the_original_columns_when_it_trims(self, backend_rows):
        big = pd.DataFrame({"a": np.arange(100_005), "s": ["x"] * 100_005})
        benchmarks.benchmark_dataframe_compression(big, nrepeats=1, return_styled=False, verbose=False)
        assert backend_rows
        for _, _, df, _ in backend_rows:
            assert list(df.columns) == ["a", "s"]
            assert list(df.index) == list(range(100_000))

    def test_the_logged_frame_size_has_two_decimals_of_the_real_megabytes(self, backend_rows, caplog):
        big = pd.DataFrame({"a": np.arange(2_500_000, dtype=np.float64)})
        expected = benchmarks.get_df_memory_consumption(big) / (1024**2)
        caplog.set_level(logging.INFO)
        benchmarks.benchmark_dataframe_compression(big, head=0, nrepeats=1, return_styled=False, verbose=True)
        assert f"DF size: {expected:.2f}Mb" in caplog.text

    def test_the_temp_folder_is_removed_tolerating_errors(self, frame, backend_rows, monkeypatch):
        seen = {}

        def record(path, ignore_errors=False):
            seen["ignore_errors"] = ignore_errors

        monkeypatch.setattr(benchmarks.shutil, "rmtree", record)
        benchmarks.benchmark_dataframe_compression(frame, head=0, nrepeats=1, return_styled=False, verbose=False)
        assert seen == {"ignore_errors": True}

    def test_varying_std_columns_are_kept_with_their_names(self, frame, monkeypatch):
        def varying(res, temp_folder, df, nrepeats):
            res.append(["row-a", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            res.append(["row-b", 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])

        for name in ("benchmark_dataframe_feather_compression", "benchmark_dataframe_orc_compression", "benchmark_dataframe_hdf_compression", "benchmark_dataframe_pickle_compression", "benchmark_dataframe_csv_compression"):
            monkeypatch.setattr(facade, name, lambda res, temp_folder, df, nrepeats: None)
        monkeypatch.setattr(facade, "benchmark_dataframe_orc_compression", varying)
        monkeypatch.setattr(facade, "benchmark_dataframe_parquet_compression", lambda df, temp_folder, nrepeats: pd.DataFrame(columns=PARQUET_COLUMNS))
        out = benchmarks.benchmark_dataframe_compression(frame, head=0, return_styled=False, verbose=False)
        assert {"std_read_time", "std_write_time", "std_read_size", "std_write_size"} <= set(out.columns)


class TestParquetPackOrder:
    def test_parquet_rows_carry_the_metrics_in_read_write_size_order(self, frame, tmp_path, distinct_measure):
        out = benchmarks.benchmark_dataframe_parquet_compression(frame, str(tmp_path), nrepeats=1, max_compression_level=1, skip_configs=())
        assert out.iloc[0].tolist()[1:] == [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0]

    def test_parquet_level_rows_carry_the_metrics_in_read_write_size_order(self, frame, tmp_path, distinct_measure):
        out = benchmarks.benchmark_dataframe_parquet_compression(frame, str(tmp_path), nrepeats=1, max_compression_level=1, skip_configs=())
        assert out.iloc[1].tolist()[1:] == [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0]

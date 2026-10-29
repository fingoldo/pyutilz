"""Behaviour of ``pyutilz.data.pandaslib.benchmarks``: a single read/write measurement returns four per-repeat arrays,
a summary row is packed with a config name and mean/std pairs, the compression sweeps name each config after its format
and compression method, and the full benchmark returns a table sorted by the requested column. Frames are tiny and every
file is written under ``tmp_path``. Optional engines that are not installed (parquet, hdf) are not asserted on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyutilz.data.pandaslib import benchmarks


@pytest.fixture
def small_frame():
    return pd.DataFrame({"a": np.arange(40), "s": [f"x{i}" for i in range(40)]})


class TestMeasureReadWrite:
    def test_one_array_per_metric_with_one_value_per_repeat(self, small_frame, tmp_path):
        out = benchmarks.measure_read_write_performance(small_frame, str(tmp_path / "f.csv"), "read_csv", {}, "to_csv", {"index": False}, 2)
        assert len(out) == 4
        assert all(len(arr) == 2 for arr in out)

    def test_timings_and_sizes_are_non_negative(self, small_frame, tmp_path):
        read_times, write_times, read_sizes, write_sizes = benchmarks.measure_read_write_performance(
            small_frame, str(tmp_path / "f.csv"), "read_csv", {}, "to_csv", {"index": False}, 1
        )
        assert (read_times >= 0).all() and (write_times >= 0).all()
        assert (read_sizes > 0).all() and (write_sizes > 0).all()

    def test_the_write_size_is_the_on_disk_size_in_megabytes(self, small_frame, tmp_path):
        fname = tmp_path / "f.csv"
        _, _, _, write_sizes = benchmarks.measure_read_write_performance(small_frame, str(fname), "read_csv", {}, "to_csv", {"index": False}, 1)
        assert write_sizes[0] == pytest.approx(fname.stat().st_size / (1024**2))


class TestPackResults:
    def test_a_summary_row_holds_the_config_and_mean_std_pairs(self):
        res = []
        arrays = [np.array([1.0, 3.0]) for _ in range(4)]
        benchmarks.pack_benchmark_results(res, "csv-zip", *arrays)
        assert len(res) == 1
        assert res[0][0] == "csv-zip"
        assert res[0][1:] == [2.0, 1.0] * 4


class TestCompressionSweeps:
    def test_the_standard_sweep_covers_six_methods_with_matching_read_and_write_options(self):
        sweep = list(benchmarks._standard_method_compression_sweep("csv", "read_csv", "to_csv"))
        assert [entry[0] for entry in sweep] == ["zip", "gzip", "bz2", "zstd", "xz", "tar"]
        compr, read_method, read_params, write_method, write_params = sweep[0]
        assert (read_method, write_method) == ("read_csv", "to_csv")
        assert read_params == {"compression": {"method": "zip"}}
        assert write_params == {"compression": {"method": "zip"}}

    def test_the_extra_write_options_are_added_to_every_write(self):
        sweep = list(benchmarks._standard_method_compression_sweep("pickle", "read_pickle", "to_pickle", {"protocol": -1}))
        assert len(sweep) == 6
        assert all(entry[4]["protocol"] == -1 for entry in sweep)

    def test_the_csv_benchmark_adds_one_named_row_per_method(self, small_frame, tmp_path):
        res = []
        benchmarks.benchmark_dataframe_csv_compression(res, str(tmp_path), small_frame, 1)
        assert [row[0] for row in res] == ["csv-zip", "csv-gzip", "csv-bz2", "csv-zstd", "csv-xz", "csv-tar"]

    def test_the_feather_benchmark_adds_one_default_row(self, small_frame, tmp_path):
        res = []
        benchmarks.benchmark_dataframe_feather_compression(res, str(tmp_path), small_frame, 1)
        assert [row[0] for row in res] == ["feather"]


class TestFullBenchmark:
    def test_the_table_is_indexed_by_config_with_the_four_mean_columns(self, small_frame):
        out = benchmarks.benchmark_dataframe_compression(small_frame, head=0, benchmark_dir_path=None, nrepeats=1, return_styled=False, verbose=False)
        assert out.index.name == "config"
        assert list(out.columns) == ["mean_read_time", "mean_write_time", "mean_read_size", "mean_write_size"]
        assert {"csv-zip", "pickle-zip", "feather", "orc"} <= set(out.index)

    def test_rows_are_sorted_by_the_write_size_by_default(self, small_frame):
        out = benchmarks.benchmark_dataframe_compression(small_frame, head=0, nrepeats=1, return_styled=False, verbose=False)
        assert out["mean_write_size"].is_monotonic_increasing

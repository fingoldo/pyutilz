"""Mutation-survivor pins for pandaslib/benchmarks.py: the progress-bar labels and ``leave`` flags of the compression sweeps
(benchmarks.py 114, 121, 204, 221, 222).

The sweeps report progress through ``tqdmu``; each call's ``desc`` names the level being swept and ``leave=False`` clears the
bar when the level ends. Recording the keyword arguments pins both without running the real compressors.
"""

from __future__ import annotations

import pandas as pd
import pytest

import pyutilz.data.pandaslib.benchmarks as bm


@pytest.fixture
def tqdm_calls(monkeypatch):
    calls: list = []

    def recording_tqdmu(iterable, **kwargs):
        calls.append(kwargs)
        return iterable

    monkeypatch.setattr(bm, "tqdmu", recording_tqdmu)
    return calls


def _has_bar(calls, desc):
    return any(kw.get("desc") == desc and kw.get("leave") is False for kw in calls)


class TestStandardSweepProgressSurvivors:
    """The shared pickle/csv generator labels its compression level and clears the bar (benchmarks.py 204)."""

    def test_the_pickle_compression_bar_is_labelled_and_cleared(self, tqdm_calls):
        list(bm._standard_method_compression_sweep("pickle", "read_pickle", "to_pickle"))
        assert _has_bar(tqdm_calls, "pickle compression method")


class TestHdfSweepProgressSurvivors:
    """The hdf sweep labels its engine and compression levels and clears both bars (benchmarks.py 221, 222)."""

    def test_the_hdf_engine_and_compression_bars_are_labelled_and_cleared(self, monkeypatch, tqdm_calls):
        monkeypatch.setattr(bm, "_run_compression_sweep", lambda res, temp_folder, df, nrepeats, file_format, sweep: list(sweep))
        bm.benchmark_dataframe_hdf_compression([], "", pd.DataFrame(), 1)
        assert _has_bar(tqdm_calls, "hdf engine")
        assert _has_bar(tqdm_calls, "hdf compression method")


class TestParquetSweepProgressSurvivors:
    """The parquet sweep labels its engine and compression levels and clears both bars (benchmarks.py 114, 121)."""

    def test_the_parquet_engine_and_compression_bars_are_labelled_and_cleared(self, tqdm_calls, tmp_path):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4, 5, 6]})
        bm.benchmark_dataframe_parquet_compression(df, str(tmp_path), nrepeats=1, engines=("pyarrow",))
        assert _has_bar(tqdm_calls, "parquet engine")
        assert _has_bar(tqdm_calls, "parquet compression method")

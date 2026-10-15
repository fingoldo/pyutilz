"""Regression tests for audit findings C-04 (mmap unlink race) and R-04 (benchmark temp folder on interrupt)."""

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pyutilz.data.pandaslib import benchmarks
from pyutilz.system import parallel


def test_mem_map_array_tolerates_the_file_vanishing_before_unlink():
    """C-04: a file removed between a separate exists() check and unlink() used to raise FileNotFoundError."""
    arr = np.arange(10, dtype=np.float64)
    real_unlink = os.unlink

    def vanish_then_unlink(path, *args, **kwargs):
        if os.path.exists(path):
            real_unlink(path)  # another caller removes it first ...
        return real_unlink(path, *args, **kwargs)  # ... so this call finds nothing

    with patch.object(parallel.os, "unlink", side_effect=vanish_then_unlink):
        parallel.mem_map_array(arr, "race_probe")
        parallel.mem_map_array(arr, "race_probe")  # the second call finds the first file and removes it


def test_benchmark_cleans_its_temp_folder_when_interrupted(tmp_path, monkeypatch):
    """R-04: a KeyboardInterrupt inside a benchmark must not leave the temp folder behind."""
    created = []
    real_mkdtemp = benchmarks.tempfile.mkdtemp

    def recording_mkdtemp(**kwargs):
        path = real_mkdtemp(dir=tmp_path)
        created.append(path)
        return path

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(benchmarks.tempfile, "mkdtemp", recording_mkdtemp)
    monkeypatch.setattr(benchmarks._facade, "benchmark_dataframe_feather_compression", interrupt)
    df = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(KeyboardInterrupt):
        benchmarks.benchmark_dataframe_compression(df, nrepeats=1)
    assert created, "the benchmark never created its temp folder, so the test proves nothing"
    assert not any(os.path.exists(p) for p in created)

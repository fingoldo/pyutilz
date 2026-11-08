"""Shared-memory dtype routing and constant-column pins for data.pandaslib.frames (mutation survivors)."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from pyutilz.data.pandaslib import frames


class _RecordingTqdm:
    def __init__(self):
        self.calls: list = []

    def __call__(self, iterable, **kwargs):
        self.calls.append(kwargs)
        return iterable


class TestShareDataframeDtypes:
    def test_each_integer_dtype_keeps_its_dtype_and_values(self):
        df = pd.DataFrame(
            {
                "i8": np.array([1, -2], dtype=np.int8),
                "i16": np.array([300, -4], dtype=np.int16),
                "i32": np.array([70000, -5], dtype=np.int32),
                "i64": np.array([1, -2], dtype=np.int64),
                "u8": np.array([1, 2], dtype=np.uint8),
                "u16": np.array([60000, 2], dtype=np.uint16),
                "u32": np.array([4_000_000_000, 2], dtype=np.uint32),
                "u64": np.array([1, 2], dtype=np.uint64),
            }
        )
        out = frames.share_dataframe(df)
        for col in df.columns:
            assert out[col].dtype == df[col].dtype, col
            assert list(out[col]) == list(df[col]), col

    def test_large_int64_values_survive_exactly(self):
        big = 2**53 + 1
        df = pd.DataFrame({"id": np.array([big, 7], dtype=np.int64)})
        out = frames.share_dataframe(df)
        assert int(out["id"].iloc[0]) == big

    def test_integer_and_float_columns_together_keep_their_own_dtypes(self):
        df = pd.DataFrame({"n": np.array([1, 2], dtype=np.int32), "f": np.array([1.5, 2.5], dtype=np.float64)})
        out = frames.share_dataframe(df)
        assert out["n"].dtype == np.int32
        assert out["f"].dtype == np.float64
        assert list(out["f"]) == [1.5, 2.5]


class TestConstantColumnPins:
    def test_the_prewarm_default_is_ten_thousand_rows(self):
        params = inspect.signature(frames.remove_constant_columns).parameters
        assert params["prewarm_size"].default == 10_000

    def test_the_prewarm_progress_bar_is_labelled_and_transient(self, monkeypatch):
        spy = _RecordingTqdm()
        monkeypatch.setattr(frames, "tqdmu", spy)
        df = pd.DataFrame({"a": [1, 1, 1, 1], "b": [1, 2, 3, 4]})
        frames.remove_constant_columns(df, prewarm_size=2)
        assert {"desc": "cnst col", "leave": False} in spy.calls

    def test_a_column_constant_in_the_head_but_not_overall_is_kept(self):
        df = pd.DataFrame({"a": [5] * 3 + [6]})
        frames.remove_constant_columns(df, prewarm_size=2)
        assert "a" in df.columns

    def test_a_truly_constant_column_is_dropped_on_the_prewarm_path(self):
        df = pd.DataFrame({"a": [5, 5, 5, 5], "b": [1, 2, 3, 4]})
        frames.remove_constant_columns(df, prewarm_size=2)
        assert list(df.columns) == ["b"]

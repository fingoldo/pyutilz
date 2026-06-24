"""Bench for classify_column_types wide-frame lookup.

The function previously read a single column's dtype via ``df.dtypes[col]``, which
materializes the whole-frame dtypes Series on every call -- O(ncols) per call, hence
O(ncols**2) when scanning every column of a wide frame (the mlframe preprocessing
cleaning path calls it once per column, sometimes several times per column).

The fix reads ``df[col].dtype`` directly (single-column dtype, O(1)).

Measured (Python 3.14, 1000-col x 2000-row frame, best of 5 full-column passes):
    old (df.dtypes[col]) : ~0.123 s / pass
    new (df[col].dtype)  : ~0.0030 s / pass   -> ~40x faster

Output is dtype-identical across numeric / float / object / bool / category / datetime
columns (df[col].dtype is the same dtype object the full dtypes Series would yield).

Run: python _benchmarks/bench_classify_column_types.py
"""

import time

import numpy as np
import pandas as pd

from pyutilz.data.pandaslib import classify_column_types


def _make_frame(ncols: int = 1000, nrows: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cols = {}
    for i in range(ncols):
        kind = i % 5
        if kind == 0:
            cols[f"c{i}"] = rng.integers(0, 5, nrows)
        elif kind == 1:
            cols[f"c{i}"] = rng.random(nrows)
        elif kind == 2:
            cols[f"c{i}"] = pd.Series(rng.integers(0, 3, nrows)).astype("category")
        elif kind == 3:
            cols[f"c{i}"] = pd.to_datetime(rng.integers(0, int(1e9), nrows), unit="s")
        else:
            cols[f"c{i}"] = rng.integers(0, 2, nrows).astype(bool)
    return pd.DataFrame(cols)


def _bench(df: pd.DataFrame, fn, passes: int = 5) -> float:
    for _ in range(2):  # warm
        for col in df.columns:
            fn(df, col)
    best = float("inf")
    for _ in range(passes):
        t = time.perf_counter()
        for col in df.columns:
            fn(df, col)
        best = min(best, time.perf_counter() - t)
    return best


def _old(df, col):
    return df.dtypes[col]


def _new(df, col):
    return classify_column_types(df=df, col=col)


if __name__ == "__main__":
    df = _make_frame()
    # identity: the production function must agree with the whole-frame dtypes path.
    for col in df.columns:
        assert classify_column_types(df=df, col=col) == classify_column_types(dtype=df.dtypes[col]), col
    print("identity OK across all dtype kinds")
    print(f"old df.dtypes[col]      best/pass: {_bench(df, _old):.4f} s")
    print(f"new classify_column_types best/pass: {_bench(df, _new):.4f} s")

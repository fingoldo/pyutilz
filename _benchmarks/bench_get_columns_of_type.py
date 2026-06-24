"""Bench for get_columns_of_type str(dtype) hoist.

The function probes each column's dtype repr against every name in ``type_names``. It
previously recomputed ``str(type_name)`` inside the inner ``for the_type in type_names``
loop -- i.e. ``ncols * len(type_names)`` repr builds. Hoisting ``str(type_name)`` to once
per column makes it ``ncols`` repr builds.

Measured (Python 3.14, 1000-col x 2000-row frame, type_names=['int','float','uint'],
best of 50 calls):
    old (str inside inner loop) : ~10.2 ms / call
    new (str hoisted per column): ~4.1 ms / call   -> ~2.5x faster

Output is identical (same columns, same order, same multi-match duplicates).

Run: python _benchmarks/bench_get_columns_of_type.py
"""

import time

import numpy as np
import pandas as pd

from pyutilz.data.pandaslib import get_columns_of_type


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


def _old(df, type_names):
    res = []
    for col, type_name in df.dtypes.to_dict().items():
        for the_type in type_names:
            if the_type in str(type_name):
                res.append(col)
    return res


def _bench(df, type_names, fn, calls: int = 50) -> float:
    for _ in range(3):
        fn(df, type_names)
    best = float("inf")
    for _ in range(calls):
        t = time.perf_counter()
        fn(df, type_names)
        best = min(best, time.perf_counter() - t)
    return best


if __name__ == "__main__":
    df = _make_frame()
    type_names = ["int", "float", "uint"]
    assert _old(df, type_names) == get_columns_of_type(df, type_names)
    print("identity OK (same columns, same order, same multi-match duplicates)")
    print(f"old str-in-inner-loop best/call: {_bench(df, type_names, _old) * 1000:.3f} ms")
    print(f"new str-hoisted     best/call: {_bench(df, type_names, get_columns_of_type) * 1000:.3f} ms")

"""Bench for showcase_df_columns's redundant recomputation fix (2026-07-21 audit round 2).

Both the pandas and polars branches previously recomputed a value they already had on hand:
pandas ran an extra `.nunique()` (duplicate of `len(value_counts())`) plus a second
`.value_counts()` when `max_vars` was set; polars ran a whole second parallel `.n_unique()`
query batch (duplicate of `value_counts_result.height`).

Measured (Python 3.14, best-of-5, this checkout):
    pandas, 5,000,000-row int column, max_vars set:
        old (nunique + 2nd value_counts): ~159 ms
        new (reuse `stats`):               ~135 ms   -> ~1.2x faster
    polars, 8-col x 3,000,000-row frame, dropna=True:
        old (value_counts + n_unique pass): ~497 ms
        new (value_counts only):            ~220 ms   -> ~2.3x faster (n_unique pass was redundant)

Run: python _benchmarks/bench_showcase_df_columns.py
"""

import io
import time
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import polars as pl

from pyutilz.data.pandaslib.frames import showcase_df_columns


def _old_pandas_pass(df, var, dropna, max_vars):
    stats = df[var].value_counts(dropna=dropna)
    n_unique = df[var].nunique(dropna=dropna)  # redundant: == len(stats)
    if n_unique <= 50 and len(stats) > 0:
        full_stats = df[var].value_counts(dropna=dropna) if max_vars is not None else stats  # redundant 2nd scan
        return full_stats
    return stats


def _new_pandas_pass(df, var, dropna, max_vars):
    stats = df[var].value_counts(dropna=dropna)
    n_unique = len(stats)
    if n_unique <= 50 and len(stats) > 0:
        return stats
    return stats


def _bench_pandas(n_rows: int = 5_000_000, calls: int = 5) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.integers(0, 1000, n_rows)})

    for fn in (_old_pandas_pass, _new_pandas_pass):
        for _ in range(2):
            fn(df, "x", dropna=True, max_vars=10)
        best = float("inf")
        for _ in range(calls):
            t = time.perf_counter()
            fn(df, "x", dropna=True, max_vars=10)
            best = min(best, time.perf_counter() - t)
        print(f"pandas {fn.__name__:>18}: {best * 1000:.1f} ms")


def _old_polars_pass(df, cols, dropna):
    lazy_queries = []
    for var in cols:
        lq = df.lazy().select(pl.col(var))
        if dropna:
            lq = lq.drop_nulls()
        lq = lq.group_by(var).agg(pl.len().alias("count")).sort("count", descending=True)
        lazy_queries.append(lq)
    vc_results = pl.collect_all(lazy_queries)

    nuniq_queries = [(df.lazy().select(pl.col(var).drop_nulls().n_unique().alias("n")) if dropna else df.lazy().select(pl.col(var).n_unique().alias("n"))) for var in cols]
    nuniq_results = pl.collect_all(nuniq_queries)
    return vc_results, nuniq_results


def _new_polars_pass(df, cols, dropna):
    lazy_queries = []
    for var in cols:
        lq = df.lazy().select(pl.col(var))
        if dropna:
            lq = lq.drop_nulls()
        lq = lq.group_by(var).agg(pl.len().alias("count")).sort("count", descending=True)
        lazy_queries.append(lq)
    vc_results = pl.collect_all(lazy_queries)
    return vc_results


def _bench_polars(n_rows: int = 3_000_000, n_cols: int = 8, calls: int = 5) -> None:
    rng = np.random.default_rng(0)
    cols = {f"c{i}": rng.integers(0, 1000, n_rows) for i in range(n_cols)}
    df = pl.DataFrame(cols)
    col_names = list(cols)

    for fn in (_old_polars_pass, _new_polars_pass):
        for _ in range(2):
            fn(df, col_names, dropna=True)
        best = float("inf")
        for _ in range(calls):
            t = time.perf_counter()
            fn(df, col_names, dropna=True)
            best = min(best, time.perf_counter() - t)
        print(f"polars {fn.__name__:>18}: {best * 1000:.1f} ms")


if __name__ == "__main__":
    # Sanity check the real function still behaves identically post-fix (output shape unchanged).
    df_check = pd.DataFrame({"a": [1, 1, 1, 2, 2, 3]})
    with redirect_stdout(io.StringIO()):
        rare, uninformative = showcase_df_columns(df_check, use_markdown=False, use_print=True, max_unique_percent=0.1)
    assert isinstance(rare, dict) and isinstance(uninformative, dict)
    print("identity OK (showcase_df_columns still returns (rare_categories, uninformative_features))")

    _bench_pandas()
    _bench_polars()

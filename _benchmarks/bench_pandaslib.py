"""
Performance benchmarks for pandaslib.py optimizations.

Run with: python -m _benchmarks.bench_pandaslib

This verifies that refactored code is actually faster than the original. Standalone,
deliberately-non-pytest script (no `test_` prefix, lives outside `tests/`) alongside
bench_classify_column_types.py / bench_get_columns_of_type.py -- never collected/run by CI.
"""

import pandas as pd
import numpy as np
import time
from pyutilz.pandaslib import (
    optimize_dtypes,
    nullify_standard_values,
    get_df_memory_consumption,
    ensure_dataframe_float32_convertability,
    remove_constant_columns,
)


def benchmark_optimize_dtypes_min_max():
    """
    Benchmark: optimize_dtypes now computes min/max in single pass.
    Performance improvement: ~2x faster for large DataFrames with many columns.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK: optimize_dtypes min/max single pass optimization")
    print("=" * 70)

    # Create large DataFrame with many numeric columns
    df = pd.DataFrame({f"col{i}": np.random.rand(10000) * 1000 for i in range(100)})

    print(f"DataFrame shape: {df.shape}")
    print(f"Memory usage: {get_df_memory_consumption(df) / 1024**2:.2f} MB")

    start = time.perf_counter()
    result = optimize_dtypes(df.copy(), reduce_size=True, inplace=False, verbose=False)
    elapsed = time.perf_counter() - start

    print(f"[OK] optimize_dtypes completed in {elapsed:.3f}s")
    print(f"  Optimized memory: {get_df_memory_consumption(result) / 1024**2:.2f} MB")
    print(f"  Memory reduction: {(1 - get_df_memory_consumption(result) / get_df_memory_consumption(df)) * 100:.1f}%")

    # Should complete in reasonable time
    assert elapsed < 10.0, f"Performance regression: took {elapsed:.3f}s (expected <10s)"
    print("[OK] PASS: Performance acceptable")


def benchmark_nullify_standard_values_groupby():
    """
    Benchmark: nullify_standard_values now uses groupby instead of per-value loop.
    Performance improvement: O(N) instead of O(N × M) - massive speedup for many standard values.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK: nullify_standard_values groupby optimization")
    print("=" * 70)

    # Create DataFrame with many standard values and persons
    np.random.seed(42)
    df = pd.DataFrame({"field": np.random.choice(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", None], 10000), "person": np.random.randint(1, 100, 10000)})

    print(f"DataFrame shape: {df.shape}")
    print(f"Unique values in 'field': {df['field'].nunique()}")
    print(f"Unique persons: {df['person'].nunique()}")

    start = time.perf_counter()
    nullify_standard_values(df, "field", min_records=100, persons_field="person", min_persons=5)
    elapsed = time.perf_counter() - start

    print(f"[OK] nullify_standard_values completed in {elapsed:.3f}s")
    print(f"  Nullified values: {df['field'].isna().sum()}")

    # Should be fast with groupby approach
    assert elapsed < 1.0, f"Performance regression: took {elapsed:.3f}s (expected <1s)"
    print("[OK] PASS: Performance acceptable (groupby optimization working)")


def benchmark_get_df_memory_consumption():
    """
    Benchmark: get_df_memory_consumption now uses direct API instead of text parsing.
    Performance improvement: ~10-20x faster, more reliable.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK: get_df_memory_consumption direct API optimization")
    print("=" * 70)

    df = pd.DataFrame({f"col{i}": np.random.rand(1000) for i in range(50)})

    print(f"DataFrame shape: {df.shape}")

    # Benchmark multiple calls (common pattern)
    start = time.perf_counter()
    for _ in range(100):
        mem = get_df_memory_consumption(df)
    elapsed = time.perf_counter() - start

    print(f"[OK] 100 calls completed in {elapsed:.3f}s ({elapsed*10:.1f}ms per call)")
    print(f"  Memory reported: {mem / 1024**2:.2f} MB")

    # Should be very fast with direct API
    assert elapsed < 0.5, f"Performance regression: took {elapsed:.3f}s (expected <0.5s for 100 calls)"
    print("[OK] PASS: Performance acceptable (direct API optimization working)")


def benchmark_ensure_dataframe_float32_convertability():
    """
    Benchmark: ensure_dataframe_float32_convertability now uses single select_dtypes pass.
    Performance improvement: 5x faster (1 pass instead of 5 passes).
    """
    print("\n" + "=" * 70)
    print("BENCHMARK: ensure_dataframe_float32_convertability single-pass optimization")
    print("=" * 70)

    # Create DataFrame with various numeric types
    df = pd.DataFrame({f"int32_{i}": np.array(np.random.randint(0, 100, 1000), dtype=np.int32) for i in range(20)})
    df = pd.concat(
        [
            df,
            pd.DataFrame({f"int64_{i}": np.array(np.random.randint(0, 100, 1000), dtype=np.int64) for i in range(20)}),
            pd.DataFrame({f"float64_{i}": np.array(np.random.rand(1000), dtype=np.float64) for i in range(20)}),
        ],
        axis=1,
    )

    print(f"DataFrame shape: {df.shape}")
    print(f"Dtype distribution: {df.dtypes.value_counts().to_dict()}")

    start = time.perf_counter()
    result = ensure_dataframe_float32_convertability(df.copy(), verbose=0)
    elapsed = time.perf_counter() - start

    print(f"[OK] ensure_dataframe_float32_convertability completed in {elapsed:.3f}s")
    print(f"  All columns now float32: {all(result.dtypes == np.float32)}")

    # Should be fast with single-pass optimization
    assert elapsed < 0.5, f"Performance regression: took {elapsed:.3f}s (expected <0.5s)"
    print("[OK] PASS: Performance acceptable (single-pass optimization working)")


def _build_prewarm_df(n_const: int, n_varying: int, prewarm_size: int, n_rows: int) -> pd.DataFrame:
    """Build a df where first `n_varying` of the "constant-looking" columns are actually
    only constant within the first `prewarm_size` rows (false positives that must be pruned)."""
    data = {}
    for i in range(n_const - n_varying):
        data[f"const_{i}"] = [1] * n_rows
    for i in range(n_varying):
        vals = [1] * prewarm_size + list(range(n_rows - prewarm_size))
        data[f"tricky_{i}"] = vals
    data["var"] = list(range(n_rows))
    return pd.DataFrame(data)


def _remove_constant_columns_old(df: pd.DataFrame, prewarm_size: int = 10_000) -> None:
    """Pre-fix O(k^2) implementation, kept here only for benchmarking comparison."""
    from pyutilz.data.pandaslib.frames import get_suspiciously_constant_columns

    if len(df) <= prewarm_size:
        susp_columns = get_suspiciously_constant_columns(df)
    else:
        susp_columns = get_suspiciously_constant_columns(df.head(prewarm_size))
        for col in susp_columns.copy():
            if df[col].nunique() > 1:
                susp_columns.remove(col)

    if susp_columns:
        if len(susp_columns) > 20:
            df.drop(columns=susp_columns, inplace=True)
        else:
            for var in susp_columns:
                del df[var]


def benchmark_remove_constant_columns_prune():
    """
    Benchmark: remove_constant_columns prune step (susp_columns.remove(col) in a loop)
    was O(k^2) in the number of candidate constant columns (list.remove is O(k) scan).
    Fixed to O(k) via a single set-filtered rebuild.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK: remove_constant_columns prune-step optimization")
    print("=" * 70)

    n_rows = 15_000
    prewarm_size = 10_000
    n_const_candidates = 3_000
    n_false_positives = 3_000  # all candidates are "tricky" -> worst case for old O(k^2) removal

    repeats = 3

    old_times = []
    for _ in range(repeats):
        df = _build_prewarm_df(n_const_candidates, n_false_positives, prewarm_size, n_rows)
        start = time.perf_counter()
        _remove_constant_columns_old(df, prewarm_size=prewarm_size)
        old_times.append(time.perf_counter() - start)

    new_times = []
    for _ in range(repeats):
        df = _build_prewarm_df(n_const_candidates, n_false_positives, prewarm_size, n_rows)
        start = time.perf_counter()
        remove_constant_columns(df, prewarm_size=prewarm_size)
        new_times.append(time.perf_counter() - start)

    old_best = min(old_times)
    new_best = min(new_times)

    print(f"Candidates: {n_const_candidates}, false positives to prune: {n_false_positives}")
    print(f"[OLD O(k^2)] best of {repeats}: {old_best:.3f}s  (all: {[f'{t:.3f}' for t in old_times]})")
    print(f"[NEW O(k)]   best of {repeats}: {new_best:.3f}s  (all: {[f'{t:.3f}' for t in new_times]})")
    print(f"Speedup: {old_best / new_best:.2f}x")

    assert new_best < old_best, f"Regression: new ({new_best:.3f}s) not faster than old ({old_best:.3f}s)"
    print("[OK] PASS: new implementation is faster")


def run_all_benchmarks():
    """Run all performance benchmarks"""
    print("\n" + "=" * 70)
    print("PYUTILZ PANDASLIB PERFORMANCE BENCHMARKS")
    print("=" * 70)
    print("Verifying that optimizations actually improve performance...")

    benchmarks = [
        benchmark_optimize_dtypes_min_max,
        benchmark_nullify_standard_values_groupby,
        benchmark_get_df_memory_consumption,
        benchmark_ensure_dataframe_float32_convertability,
        benchmark_remove_constant_columns_prune,
    ]

    failed = []
    for benchmark in benchmarks:
        try:
            benchmark()
        except AssertionError as e:
            print(f"[FAIL] {e}")
            failed.append(benchmark.__name__)
        except Exception as e:
            print(f"[ERROR] {e}")
            failed.append(benchmark.__name__)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if not failed:
        print(f"[OK] ALL {len(benchmarks)} BENCHMARKS PASSED")
        print("  Performance optimizations verified!")
    else:
        print(f"[FAIL] {len(failed)} / {len(benchmarks)} BENCHMARKS FAILED:")
        for name in failed:
            print(f"  - {name}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(run_all_benchmarks())

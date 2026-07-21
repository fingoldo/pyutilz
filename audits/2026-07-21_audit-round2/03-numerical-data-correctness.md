# Numerical & Data Correctness Audit

## Summary

I read in full: `src/pyutilz/data/numpylib.py`, `src/pyutilz/data/numbalib.py`, `src/pyutilz/data/polarslib.py`, `src/pyutilz/data/pandaslib/{dtypes,frames,io_ops,benchmarks,_common}.py`, `src/pyutilz/stats/normality.py`, `src/pyutilz/text/similarity.py`, `src/pyutilz/text/strings/textentropy.py`, `src/pyutilz/text/humanizer.py`, `src/pyutilz/text/strings/basics.py`, `src/pyutilz/core/pythonlib.py` (full file, including `datetime_to_unix_ts`, `keys_changed_enough`, `float_distinct_digits_percent`, `count_trailing_zeros`, `get_human_readable_set_size`), and `src/pyutilz/performance/kernel_tuning/{benchmark,cache/cache_class,cache/region_matching}.py`. Every finding below was empirically verified with a standalone repro against the actual installed packages (numpy/pandas/polars) in this checkout, not just read-and-assumed. Cross-referenced all findings against `audits/2026-07-21_full-audit/*.md` to avoid duplicating round-1 findings; two look-alike issues were round-1-covered and are listed under "Investigated, not an issue" instead.

Findings: 1 CRITICAL, 3 HIGH, 1 MEDIUM, 1 LOW.

## Findings

### [CRITICAL] `get_topk_indices` silently returns NaN as a "highest" value — src/pyutilz/data/numpylib.py:47-60

- **Category**: NaN-propagation / ranking correctness
- **Problem**: The `highest=True` branch does `part = np.argpartition(arr, n - k, axis=axis)` followed by `order = np.argsort(cand_vals, ...)` with no NaN handling anywhere. Both `np.argpartition` and `np.argsort` treat `NaN` as **greater than every real number** in their default sort order, so a single `NaN` anywhere in `arr` is picked as the top-ranked "highest" element instead of being excluded.
- **Failure scenario**: Verified directly:
  ```python
  arr = np.array([5.0, np.nan, 3.0, 10.0, 1.0])
  get_topk_indices(arr, k=1, highest=True)   # -> [1]  (the NaN!), value nan
  get_topk_indices(arr, k=2, highest=True)   # -> [1, 3]  (NaN + the real max 10.0)
  ```
  The true top-1 is index 3 (value 10.0); the function returns index 1 (`NaN`) instead. This is a general-purpose ranking utility with no documented NaN-exclusion contract, and NaN-containing score/feature arrays (missing data, upstream `0/0`, failed computations) are an everyday occurrence, not an adversarial edge case — any caller doing feature/candidate ranking gets a silently wrong answer with no exception, no warning.
- **Suggested fix**: Before partitioning, either reject NaN explicitly (`if np.isnan(arr).any(): raise/warn`) or make ranking NaN-aware, e.g. substitute `-inf`/`+inf` for NaN depending on `highest` (mirroring `np.nanargpartition`-style handling) so NaNs always sort to the "worst" end regardless of direction, then restore/mask them from the result.

### [HIGH] `share_dataframe` silently corrupts int64 values above 2**53 via forced float64 round-trip — src/pyutilz/data/pandaslib/frames.py:329-344

- **Category**: dtype-narrowing precision loss
- **Problem**: The shared-memory clone forces the array through `ctypes.c_double` regardless of the original column dtypes:
  ```python
  mparr = Array(ctypes.c_double, df.values.reshape(-1), lock=True)
  df_shared = pd.DataFrame(np.frombuffer(mparr.get_obj()).reshape(df.shape), columns=df.columns).astype(df_dtypes_dict)
  ```
  `df.values` upcasts a mixed-dtype frame to `float64` (52-bit mantissa) before being written into the double-precision `Array`, then the result is cast back to the original per-column dtype (e.g. back to `int64`) via `.astype(df_dtypes_dict)`. Any `int64` value whose magnitude exceeds `2**53` (9,007,199,254,740,992) — common for nanosecond-epoch timestamps, database auto-increment IDs, or hashed keys — is not exactly representable in float64 and comes back altered.
- **Failure scenario**: Verified directly:
  ```python
  df = pd.DataFrame({'id': np.array([9007199254740993, 9007199254740995], dtype=np.int64)})
  out = share_dataframe(df)
  out['id'].tolist()  # -> [9007199254740992, 9007199254740996]  (both silently wrong, still int64)
  ```
  Two distinct IDs get corrupted to different (and possibly colliding) values with no error, no warning, and the output dtype still claims `int64` — a downstream consumer has no signal anything went wrong.
- **Suggested fix**: Either document the precision ceiling loudly (the existing docstring only warns about datetime dtype, not integer magnitude) and assert `df.select_dtypes(include="int64").abs().max().max() < 2**53` before sharing, or route integer columns through a separate `ctypes` buffer typed to their actual width instead of unconditionally widening everything through `c_double`.

### [HIGH] `keys_changed_enough` computes percent-change with the wrong sign for negative baseline values — src/pyutilz/core/pythonlib.py:194-223

- **Category**: sign-error in percent-change formula (division correctness)
- **Problem**:
  ```python
  change = abs(new_value - prev_value) * 100 / prev_value
  if change >= min_change_percent:
      return True
  ```
  The numerator is `abs(...)` (always ≥ 0) but the denominator `prev_value` is used un-abs'd. When `prev_value` is negative, `change` comes out **negative**, so `change >= min_change_percent` (a positive threshold) is false even for an enormous real change — the function reports "not changed enough" no matter how large the actual swing was.
- **Failure scenario**: Verified directly:
  ```python
  keys_changed_enough(obj={'a': 200}, prev_obj={'a': -100}, min_change_percent=10.0)   # -> False
  keys_changed_enough(obj={'a': -200}, prev_obj={'a': -100}, min_change_percent=10.0)  # -> False
  ```
  Both are 100%+ swings from a negative baseline (e.g. a P&L/exposure/residual metric crossing zero), and both are misreported as "not changed enough." For any metric that legitimately takes negative values, this monitoring/change-detection function silently fails at its one job.
- **Suggested fix**: `change = abs(new_value - prev_value) * 100 / abs(prev_value)`.

### [HIGH] Weighted-mean aggregates in `build_aggregate_features_polars` leak unguarded Inf/NaN on zero-sum-weight groups — src/pyutilz/data/polarslib.py:193-206, 433-437, 679-693

- **Category**: division-by-zero propagation not covered by the module's own NaN-cleanup pass
- **Problem**: `add_weighted_aggregates` builds `weighted_mean = ((all_other_num_cols * pl.col(wcol)).sum() / pl.col(wcol).sum())...` with no zero-denominator guard, and the resulting expressions are appended straight to `feature_expressions` (`wcols.append(weighted_mean)` / `feature_expressions.extend(wcols)`) — never wrapped in `clean_numeric`. Downstream, `create_ts_features_polars`'s NaN-cleanup pass only targets a fixed set of suffixes:
  ```python
  fragile_cols = (cs.contains("_skew") | cs.contains("_kurtosis") | cs.contains("_entropy")
                  | cs.contains("_c3_stats") | cs.contains("_cid_ce") | cs.contains("corr_")
                  | cs.contains("_linreg") | cs.contains("_lempel_ziv"))
  ```
  The weighted-mean column suffix (`_wmeanby_`, set at `polarslib.py:201`) is not in this list, so it is never routed through `clean_numeric`. This is despite `clean_numeric`'s own docstring (line 70) explicitly naming "zero-weight groups in weighted-mean" as the motivating scenario for the function's existence — the protection it describes doesn't actually reach the weighted-mean feature.
- **Failure scenario**: Verified directly (a weighting column that sums to zero within a window/group — e.g. hedged buy/sell volumes, net-zero flow):
  ```python
  df = pl.DataFrame({'x': [1.0, 2.0, 3.0], 'w': [1.0, -1.0, 0.0]})
  wcols = add_weighted_aggregates(cs.by_name('x'), ['w'])
  df.select(wcols).to_series(0)[0]   # -> -inf
  ```
  `create_ts_features_polars` would emit this `-inf` straight into the output feature matrix; any downstream `.cast(int)`, ML training, or aggregation over that column silently inherits the Inf, exactly the class of failure `clean_numeric` was written to prevent for other stats.
- **Suggested fix**: In `add_weighted_aggregates`, wrap `weighted_mean` in `clean_numeric(weighted_mean, nans_filler=nans_filler)` at construction time (it already has access to nothing else it needs), or add the `_wmeanby_` suffix to `fragile_cols` in `create_ts_features_polars`.

### [MEDIUM] `bin_numerical_columns` default `bin_dtype=pl.Int8` is unvalidated against `num_bins`, crashing instead of binning — src/pyutilz/data/polarslib.py:728-741, 938

- **Category**: unvalidated dtype/parameter interaction
- **Problem**: `bin_dtype: Any = pl.Int8` is the default, and the binning expression is `... .clip(0, num_bins - 1).cast(bin_dtype)` with no check that `num_bins - 1` fits in `bin_dtype`'s range (signed 8-bit max is 127). `num_bins` has no documented upper bound.
- **Failure scenario**: Verified directly:
  ```python
  df = pl.DataFrame({'x': [float(i) for i in range(300)]})
  bin_numerical_columns(df, target_columns=[], num_bins=200)
  # polars.exceptions.InvalidOperationError: conversion from `f64` to `i8` failed ... for 108 out of 300 values: [128.0, 129.0, ...]
  ```
  With strict polars casting this currently raises rather than silently wrapping/corrupting (good), but it is still a reliability bug: a perfectly reasonable, undocumented-as-forbidden `num_bins=200` (or any value > 128) crashes the whole call with no guidance from the function's own signature about the 128-bin ceiling implied by its own default dtype.
- **Suggested fix**: Validate `num_bins <= np.iinfo(bin_dtype).max + 1` (or the polars-dtype equivalent) up front and raise a clear `ValueError` naming the actual limit, or auto-widen `bin_dtype` (e.g. to `pl.Int16`) when `num_bins` exceeds Int8's range.

### [LOW] `float_distinct_digits_percent` rounding carry can push the fractional-digit count past the declared `precision` — src/pyutilz/core/pythonlib.py:358-384

- **Category**: rounding-carry edge case
- **Problem**: `frac_part = round(abs(number - int_part), precision)` can round up to exactly `1.0` when the true fractional part is very close to 1 (e.g. `0.99999996` at `precision=5`). The subsequent `frac_digits = int(frac_part * (10**precision))` then produces `10**precision` (e.g. `100000` for `precision=5`), which has `precision + 1` digits, not `precision` — silently inflating `ntotal` and skewing the returned "distinct digits / total digits" ratio for values that round-trip through a `.999...→1.0` carry.
- **Failure scenario**: Verified directly: `float_distinct_digits_percent(0.99999996, precision=5)` returns `0.333...` computed against a 6-digit `ntotal`, inconsistent with the function's own precision contract (which normally bounds the fractional-part digit count to exactly `precision`).
- **Suggested fix**: After rounding, re-clamp: `frac_part = min(frac_part, 1 - 10**-precision)` (or re-round `frac_digits` modulo `10**precision`) before computing `ntotal`.

## Things done well

- `optimize_dtypes` (pandaslib/dtypes.py) is unusually careful about float→smaller-float precision loss: it explicitly reconstructs each candidate's decimal mantissa and checks it survives a round-trip at the target dtype's actual mantissa precision before downcasting, rather than trusting a naive `min/max`-fits-in-range check.
- `clean_numeric`/`cast_f64_to_f32` (polarslib.py) both document *why* naive `.replace([inf,-inf,nan], ...)` fails for polars (NaN != NaN under float equality) and `cast_f64_to_f32` proactively warns when an integer column's magnitude will lose exact-integer precision under the cast — exactly the kind of check the sibling weighted-mean path (finding above) is missing.
- `_pack_words`/numba sentence-similarity path (text/similarity.py) has clearly been through a real differential-testing pass: the in-file comments document a discovered-and-fixed ULP-collapsing bug in an earlier quantized-key sort attempt, with the fix (exact float64-bit-pattern sort) explained in detail.
- `normality.py`'s Anderson-Darling `log_phi_cdf` uses a documented asymptotic expansion specifically to avoid `log(0)` underflow to `-inf` for `z < -5`, rather than letting the naive formula silently degrade.

## Investigated, not an issue

- **`entropy_for_column`/`mi_for_column` (polarslib.py) division by `len(bins)`**: for a zero-row input, `np.array([]) / 0` doesn't raise and `-np.sum(...)` on the empty result is `0.0` — confirms round-1's finding (08-domain-data.md) that this is not a crash risk.
- **`ensure_dataframe_float32_convertability` (pandaslib/dtypes.py:316-358)**: casts int32/int64/uint32/uint64 columns straight to float32 with no magnitude guard (verified: `16777217 -> 16777216.0`, precision silently lost above 2**24). This is the same class of issue as finding 2 above, but round 1 (08-domain-data.md) already investigated this exact function and explicitly dispositioned it as "working as designed" (matches its own docstring, unlike the polars sibling it was compared against) — not re-flagging.
- **`dagostino_k2` (stats/normality.py)**: stress-tested 2000 random samples (n=20-40, exponential-power-transformed for heavy skew) looking for the `sqrt(negative)` / `log(negative)` NaN paths in the `W2`/`delta` computation the code's own edge-case math suggests could exist for small `n` and pathological skew; zero NaN/Inf outputs observed. Consistent with the module's claim of matching `scipy.stats.normaltest` on `n >= 20`.
- **`optimize_dtypes` object-column int64 promotion (pandaslib/dtypes.py:112-118)**: verified that out-of-int64-range numeric strings (`"99999999999999999999999999"`) raise `OverflowError` on `.astype(np.int64)`, which is caught by the surrounding `except Exception` and correctly falls through to the float64 attempt rather than silently wrapping/corrupting.

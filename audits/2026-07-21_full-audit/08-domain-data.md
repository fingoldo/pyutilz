# Domain audit: data (pandaslib, polarslib, numpylib, numbalib) + stats

## Summary

Covered all 12 files in scope in full: `data/__init__.py`, `data/numbalib.py`, `data/numpylib.py`,
`data/polarslib.py`, `data/pandaslib/{__init__,_common,dtypes,frames,io_ops,benchmarks}.py`,
`stats/__init__.py`, `stats/normality.py`. The angle was computational-efficiency and dtype/null-handling
correctness (iterrows/vectorization, numba object-mode risk, dtype-downcast overflow, pandas/polars null
drift). No `.iterrows()`/`.itertuples()`-over-rows anti-patterns and no chained-indexing
`SettingWithCopyWarning` risk were found in this scope — the code is already careful there. The real yield
was in silent dtype/null-handling correctness gaps, several of them concretely reproduced below with a
runnable snippet. Two are High severity (silent numeric corruption reachable through a public,
documented code path); the rest are narrower-path Medium/Low issues. All numbered findings below were
verified by actually running the code (not inferred from reading alone) except where explicitly marked.

Findings: 2 High, 5 Medium, 3 Low.

## Findings

### [HIGH] `cast_f64_to_f32` silently converts integer columns to Float32, contradicting its own docstring and losing exact-integer precision — src/pyutilz/data/polarslib.py:74-80

- **Category**: correctness / dtype-optimization
- **Problem**: The docstring says `"Downcast Float64 and the common integer dtypes to Float32/Int32 to shrink memory usage"` (i.e. implies int64 -> int32), but the implementation selects `Int32, UInt32, Int64, UInt64, (Int128)` together with `Float64` and casts **all of them to `pl.Float32`**:
  ```python
  int_types = [pl.Int32, pl.UInt32, pl.Int64, pl.UInt64]
  ...
  return df.with_columns(pl.col(*int_types, pl.Float64).cast(pl.Float32))
  ```
  Float32 has a 24-bit mantissa, so integers are only representable exactly up to 2^24 = 16,777,216. Any Int64/UInt64/UInt32/Int128 column with values beyond that silently loses precision (rounds to the nearest representable float32) — and the column's dtype silently changes from an integer type to `Float32`, which downstream code that expects e.g. `pl.Int64` will not catch.
  This is also **auto-invoked** from the public `create_ts_features_polars(..., dtype=pl.Float32)` (line 676-677: `if dtype == pl.Float32: res = cast_f64_to_f32(res)`), so any caller of the main FE-builder with `dtype=pl.Float32` gets this silently applied to the whole result frame, including any surviving integer-typed aggregate columns (e.g. `nrecs` from `.len()`, which is `UInt32`; or `arg_max`/`arg_min` indices).
- **Failure scenario**: verified directly:
  ```python
  df = pl.DataFrame({'id': pl.Series([16777217, 16777219, 9007199254740993], dtype=pl.Int64)})
  cast_f64_to_f32(df)['id'].to_list()  # -> [16777216.0, 16777220.0, 9007199254740992.0]
  ```
  Every value that isn't exactly representable in float32 is silently altered, and the dtype flips from `Int64` to `Float32`. For a time-series aggregation over a window with more than ~16.7M rows (a realistic count for high-frequency event data), the resulting `nrecs`/index columns would be silently wrong.
- **Suggested fix**: Either (a) fix the code to match the docstring — downcast integer columns to the smallest *integer* type that fits (Int32/etc.), independent of the Float64->Float32 cast — or (b) fix the docstring to say "and the common integer dtypes to Float32" (matching the already-correct, differently-documented sibling function `ensure_dataframe_float32_convertability` in `pandaslib/dtypes.py:310-352`, which explicitly documents converting ints to float32 for LightGBM compatibility). Either way, guard against magnitude > 2^24 before doing an unconditional int->float32 cast, or document the precision-loss risk prominently since it is currently silent.

### [HIGH] `get_columns_of_type` returns duplicate column names when more than one requested type-name substring matches the same dtype string — src/pyutilz/data/pandaslib/dtypes.py:56-63

- **Category**: correctness
- **Problem**:
  ```python
  def get_columns_of_type(df: pd.DataFrame, type_names: Sequence) -> list:
      res = []
      for col, type_name in df.dtypes.to_dict().items():
          type_name_str = str(type_name)
          res.extend([col for the_type in type_names if the_type in type_name_str])
      return res
  ```
  For each column, the inner comprehension adds `col` to `res` **once per matching `the_type`**, instead of at most once per column. If two entries in `type_names` are both substrings of the same dtype string (a very natural case — e.g. `"int"` is a substring of `"uint32"`), the column is duplicated in the returned list.
- **Failure scenario**: verified directly:
  ```python
  df = pd.DataFrame({'a': pd.array([1,2,3], dtype='int64'),
                      'b': pd.array([1,2,3], dtype='uint32'),
                      'c': [1.0,2.0,3.0]})
  get_columns_of_type(df, ['int', 'uint'])   # -> ['a', 'b', 'b']
  ```
  Column `'b'` (uint32) appears twice because `"int"` and `"uint"` are both substrings of `"uint32"`. A caller doing `df[get_columns_of_type(df, ['int','uint'])]` gets a DataFrame with a duplicated column label, which silently breaks `.astype()`/`set_index()`/`groupby()`/any code assuming unique labels downstream; a caller doing `len(get_columns_of_type(...))` to count matching columns overcounts.
- **Suggested fix**: `if any(t in type_name_str for t in type_names): res.append(col)`.

### [MEDIUM] `remove_stale_columns` does not remove anything from the caller's DataFrame, despite its name and docstring — src/pyutilz/data/pandaslib/frames.py:338-356

- **Category**: correctness / API-footgun
- **Problem**: The docstring says `"Removes columns with values that do not change"`, and the function's name mirrors its sibling `remove_constant_columns` (which genuinely mutates its input in place via `df.drop(columns=..., inplace=True)` / `del df[var]`, `pandaslib/frames.py:378-408`). But `remove_stale_columns` only rebinds its **local** parameter name:
  ```python
  X = X.loc[:, stale_columns[~stale_columns].index.values]
  all_features_names = X.columns.tolist()
  return all_features_names
  ```
  Reassigning the local `X` does not affect the caller's original DataFrame object at all — the function is really "return the list of non-stale column names," not "remove stale columns."
- **Failure scenario**: verified directly:
  ```python
  df = pd.DataFrame({'const': [1,1,1], 'var': [1,2,3]})
  result = remove_stale_columns(df)
  # result == ['var']
  # but: list(df.columns) == ['const', 'var']  -- df is completely unchanged
  ```
  A caller who (reasonably, given the name and the sibling function's real in-place contract) calls `remove_stale_columns(df)` expecting `df` to shrink, and doesn't use the return value, silently keeps every stale column.
- **Suggested fix**: Either rename the function to reflect what it does (e.g. `get_non_stale_columns`) and update the docstring, or make it genuinely mutate in place (`X.drop(columns=..., inplace=True)`) for parity with `remove_constant_columns`.

### [MEDIUM] `showcase_df_columns`'s rare/uninformative-feature detection gate ignores the caller's `dropna` flag when counting uniques, silently suppressing detection near the threshold — src/pyutilz/data/pandaslib/frames.py:249-250 (polars) and :278-279 (pandas)

- **Category**: correctness / edge-case
- **Problem**: The function accepts a `dropna` parameter that correctly controls whether nulls are excluded from the printed value-counts (`vc`/`stats`). But the **eligibility gate** that decides whether a column is even considered for rare-category/uninformative-feature detection uses a *null-inclusive* unique count regardless of `dropna`:
  - polars branch: `n_unique = nuniq_df.item(0, 0)`, computed via `pl.col(var).n_unique()` (polars always counts null as one of the unique values, independent of `dropna`).
  - pandas branch: `n_unique = df[var].nunique(dropna=False)` (hardcoded `dropna=False`, ignoring the caller's `dropna` argument).
  Both branches then gate on `n_unique <= max_cat_uniq_qty`. If a column has exactly `max_cat_uniq_qty` real (non-null) distinct values *plus* any nulls, and the caller passed `dropna=True` (meaning: "I only care about the non-null values"), the gate still counts the null as +1, so the column is one over the threshold and is **entirely skipped** — even though, under the caller's own `dropna=True` semantics, it should have been analyzed.
- **Failure scenario**: verified directly — a 150-row pandas column with exactly 50 distinct non-null values (2 rows each) plus 100 nulls, called with `dropna=True, max_cat_uniq_qty=50`:
  ```python
  vals = list(range(50)) * 2 + [None]*100
  df = pd.DataFrame({'x': vals})
  r, u = showcase_df_columns(df, cols=['x'], dropna=True, max_cat_uniq_qty=50, max_unique_percent=0.9, ...)
  # r == {}, u == {}   -- nothing flagged
  ```
  Raising `max_cat_uniq_qty` to 51 (to compensate for the silently-counted null) immediately reveals the true result: `r['x']` has all 50 values flagged rare and `u['x'] == 1.0` (i.e. the column is 100% "uninformative" after dropping rares) — exactly the kind of feature this function exists to catch, per its own docstring ("detects low-variability features useful for ML feature selection"). The boundary bug silently hides this from a caller who explicitly asked to ignore nulls.
- **Suggested fix**: compute the gating `n_unique` respecting `dropna` too (`df[var].nunique(dropna=dropna)` for pandas; `pl.col(var).drop_nulls().n_unique()` when `dropna` else `pl.col(var).n_unique()` for polars), so the gate and the display counts agree.

### [MEDIUM] `find_nan_cols` only detects float NaN, not polars null — an all-null column is invisible to it — src/pyutilz/data/polarslib.py:45-49

- **Category**: correctness / polars-vs-pandas API drift (eq_missing-class trap)
- **Problem**: Polars distinguishes `is_nan()` (float NaN) from `is_null()`/missing (`None`), unlike pandas where `isna()`/`isnull()` catch both. `find_nan_cols` uses only `cs.numeric().is_nan().any()`:
  ```python
  def find_nan_cols(df: pl.DataFrame) -> pl.DataFrame:
      meta = df.select(cs.numeric().is_nan().any())
      ...
  ```
  A numeric column that is entirely `None` (missing, not NaN) is not flagged. This is exactly the class of "min()==max() misses all-null columns" trap: someone porting pandas-mental-model code (where `df.isna().any()` catches both) to this helper will silently miss all-null columns.
- **Failure scenario**: verified directly:
  ```python
  df = pl.DataFrame({'allnull': pl.Series([None, None, None], dtype=pl.Float64),
                      'hasnan':  pl.Series([1.0, float('nan'), 3.0])})
  find_nan_cols(df).columns   # -> ['hasnan']   (allnull is NOT returned)
  ```
  Contrast with `bin_numerical_columns`/`drop_constant_columns` elsewhere in the *same file*, which correctly guard against exactly this case with an explicit `if (min_val is None or max_val is None) or np.allclose(...)` check (see "Things done well") — so the codebase clearly knows about this trap in some places but not here.
- **Suggested fix**: Either OR in `.is_null()` (`cs.numeric().is_nan().fill_null(False).any() | cs.numeric().is_null().any()`, being careful that `is_nan()` on a null itself returns null) if the intent is "any missing/invalid value," or rename/document the function explicitly as "NaN-only, not null" so callers don't assume pandas semantics.

### [MEDIUM] `smart_ratios`'s docstring promises automatic zero-avoidance that the implementation does not perform — src/pyutilz/data/numpylib.py:77-83

- **Category**: correctness / docs-vs-implementation mismatch / numeric-stability
- **Problem**: The docstring says:
  > "Returns (a-b)/b, but watches that b is not close to zero by shifting both values up, so that b.min() becomes positive and at least as big as the entire span of a or b."

  This describes an *automatic*, data-derived shift (computed from `b.min()` and the span of `a`/`b`) applied to *both* `a` and `b`. The actual implementation does none of that:
  ```python
  def smart_ratios(a, b, span_correction: float = 0.0, na_fill=np.nan) -> np.ndarray:
      return div0(a - b, b + span_correction, na_fill=na_fill)
  ```
  It adds a caller-supplied constant `span_correction` (default **0.0**, i.e. no shift at all) to `b` only; `a` is never shifted. `div0` only catches an exactly-zero (or exactly non-finite) division result — it does not protect against `b` being merely *close to* zero, which is precisely the scenario the docstring claims is handled. The project's own test suite even has a comment acknowledging the docstring is misleading (`tests/test_numpylib.py:154`: "smart_ratios computes (a-b)/b, not a/b") but does not test the zero-avoidance claim at all.
- **Failure scenario**: verified directly:
  ```python
  a = np.array([100.0, 200.0]); b = np.array([0.0001, 300.0])
  smart_ratios(a, b)   # -> [999999.0, -0.333...]
  ```
  With the documented "smart" zero-avoidance, a `b` value of `0.0001` should not blow the ratio up to ~1e6; with the actual (default) behavior, it does — the function provides no more zero-protection than a bare `div0(a-b, b)` unless the caller manually pre-computes and supplies an appropriate `span_correction`, defeating the "smart"/automatic framing.
- **Suggested fix**: either implement the documented auto-shift (derive a default `span_correction` from `np.abs(b).max() - b.min()` or similar when the caller doesn't supply one), or rewrite the docstring to describe the actual (manual, caller-driven) contract.

### [MEDIUM] `read_stats_from_multiple_files`'s size-based re-save decision uses the pandas `deep=False` memory estimate, undercounting object/string columns by 30x+ in a measured example, instead of the package's own `deep=True`-aware helper — src/pyutilz/data/pandaslib/io_ops.py:109,115 vs pandaslib/dtypes.py:268-307

- **Category**: efficiency / correctness (needs confirmation on real-world decision impact)
- **Problem**: `read_stats_from_multiple_files` computes the before/after sizes used to decide whether to re-save an optimized file via:
  ```python
  old_size = tmp_df.memory_usage(index=True).sum() / 1024**3
  ...
  new_size = tmp_df.memory_usage(index=True).sum() / 1024**3
  ...
  if new_size <= old_size * (1 - min_size_improvement_percent) or old_size - new_size >= min_size_improvement:
  ```
  `.memory_usage()` defaults to `deep=False`, which for `object`-dtype columns (Python strings) reports only pointer size (~8 bytes/cell) rather than the actual string bytes. The *same package* ships `get_df_memory_consumption(df, deep=True)` (`pandaslib/dtypes.py:268-307`) specifically documented to give byte-accurate sizing for this exact purpose, with an explicit docstring warning about the shallow/deep tradeoff — but this call site doesn't use it.
- **Failure scenario**: verified directly (undercount factor, not a hypothetical):
  ```python
  df = pd.DataFrame({'s': ['a very long repeated-ish string value ' * 5] * 100000})
  df.memory_usage(index=True, deep=False).sum() / 1024**3   # -> 0.000745 GB
  df.memory_usage(index=True, deep=True).sum()  / 1024**3   # -> 0.022259 GB   (~30x more)
  ```
  For string/object-heavy dataframes (a common case for the files this function batch-processes), both the logged "size" figures (lines 110, 116) and the resave-trigger comparison against `min_size_improvement`/`min_size_improvement_percent` are computed against a baseline that can be off by more than an order of magnitude, making the percent-based and absolute-GB-based thresholds unreliable for exactly the kind of data (string columns getting converted to category/int by `optimize_dtypes`) this function is meant to optimize. I did not construct a specific input that flips a resave go/no-go decision end-to-end, so labeling the downstream-decision impact "needs confirmation" — but the underlying size figures being wrong by ~30x is directly measured, not speculative.
- **Suggested fix**: use `get_df_memory_consumption(tmp_df, deep=True)` (already imported transitively via `pyutilz.data.pandaslib`) instead of the bare `.memory_usage()` call.

### [LOW] `generate_combinations_recursive_njit` crashes with a raw numpy `ValueError` instead of a clear input-validation error on negative `r` — src/pyutilz/data/numbalib.py:72-94

- **Category**: edge-case / robustness
- **Problem**: The function has no lower-bound check on `r`. For `r == 0` and for `r` positive it behaves correctly (verified by trace: `r>0` monotonically decreases toward 0 or the sequence empties first, both handled). For `r < 0`, `r` never reaches 0 as the recursion progresses (each `with_first` call decrements `r` further negative), so eventually `sequence.size == 0` is hit with `r` still negative and the function executes `np.empty((0, r), dtype=sequence.dtype)` — a negative array dimension.
- **Failure scenario**: verified directly:
  ```python
  generate_combinations_recursive_njit(np.array([1,2,3]), -1)
  # -> ValueError: negative dimensions are not allowed
  ```
  A caller passing a negative `r` (e.g. from an off-by-one in an outer loop) gets an opaque numpy shape error from deep inside a recursive numba kernel rather than a clear "r must be >= 0" message at the call boundary.
- **Suggested fix**: add `if r < 0: raise ValueError(...)` (or an `assert r >= 0`) at the top of the function (or its Python-level `set_random_seed`-style wrapper, since raising a custom exception message from inside `@njit` code requires care).

### [LOW] `optimize_dtypes`'s float-precision-loss check leaks a numpy `RuntimeWarning` for columns containing an exact `0.0` — src/pyutilz/data/pandaslib/dtypes.py:205-207

- **Category**: efficiency / robustness (noise, not wrong output)
- **Problem**: When checking whether a float column can be safely downcast without precision loss, the code computes:
  ```python
  with np.errstate(divide="ignore"):
      _, int_part = np.modf(np.log10(np.abs(values)))
      mantissa = np.round(values / 10**int_part, np.finfo(old_dtypes[col]).precision - 1)
  ```
  For a value of exactly `0.0`: `log10(0)` is `-inf` (suppressed correctly by `divide="ignore"`), `modf(-inf)` gives `int_part = -inf`, so `10**int_part == 0.0`, and `values / 10**int_part` becomes `0.0 / 0.0` — which raises numpy's **"invalid value encountered in divide"** warning, a different warning category than `"divide"`, so the existing `np.errstate(divide="ignore")` does not suppress it.
- **Failure scenario**: verified directly (with warnings promoted to errors to make the point unambiguous):
  ```python
  df = pd.DataFrame({'z': [0.0, 100.5, 200.25]})
  optimize_dtypes(df, ensure_float64_precision=True, skip_halffloat=False, float_to_int=False, inplace=False)
  # under -W error::RuntimeWarning: RuntimeWarning: invalid value encountered in divide
  ```
  Under normal (non-strict) execution this is just an uncontrolled console/log warning bypassing the function's own `verbose`/`logger` mechanism (not a wrong result — the resulting NaN is correctly masked out downstream via `np.ma.array(fract_part, mask=np.isnan(fract_part))`, so the precision-safety verdict is unaffected). Still, it is unwanted noise that the author clearly intended to suppress (they already added `errstate(divide="ignore")` for the adjacent `log10` call) but missed the second source.
- **Suggested fix**: widen the `errstate` context to `np.errstate(divide="ignore", invalid="ignore")`.

### [LOW] `normality_verdict` reports "consistent with Normal" for a perfectly constant (zero-variance) sample — src/pyutilz/stats/normality.py:126-127, 208-209, 299-311

- **Category**: edge-case / stats correctness
- **Problem**: For a zero-variance sample, `dagostino_k2` short-circuits to `(0.0, 1.0, 0.0, 0.0)` (`m2 <= 0.0` branch) and `anderson_darling_normal` short-circuits to `(nan, nan)` (`s2 <= 0.0` branch). `normality_verdict` then computes `reject_k2 = isfinite(1.0) and 1.0 < alpha` -> `False`, and `reject_ad = isfinite(nan) and ...` -> `False`, so `reject_normal = False` and the verdict text says "consistent with Normal". A constant sample is a degenerate distribution, not a Normal one, and the module's own stated use-case ("residual-distribution audits exist to surface non-Normality") makes this a meaningful miss: a pipeline bug that outputs identical residuals (e.g. a broken model always predicting the same value) would be reported by this helper as "consistent with Normal" rather than flagged as suspicious/degenerate.
- **Failure scenario**: verified directly:
  ```python
  normality_verdict(np.array([5.0]*30))
  # -> {'k2_stat': 0.0, 'k2_p': 1.0, 'ad_stat': nan, 'ad_p': nan, 'reject_normal': False,
  #     'verdict': "consistent with Normal (K2 p=1, AD A*=nan p=nan)"}
  ```
- **Suggested fix**: special-case zero variance (`m2 <= 0` / `s2 <= 0`) as an explicit `verdict == "degenerate (zero variance)"` with `reject_normal` left as a separate, clearly-distinct field, rather than folding it into "consistent with Normal".

## Things done well

- **Deliberate, commented avoidance of exactly the null-vs-constant trap this audit targets**, in the same file where `find_nan_cols` misses it: `remove_stale_columns` (`pandaslib/frames.py:345-348`) explicitly uses `nunique(dropna=False)` with an inline comment explaining why (`"an all-NaN column (nunique==1) is correctly flagged stale -- unlike X != X.iloc[0], which is always True for NaN vs NaN"`), and `bin_numerical_columns`/`drop_constant_columns` (`polarslib.py:797-800, 969-972`) both guard `np.allclose(min_val, max_val)` with an explicit `min_val is None or max_val is None` check first, correctly handling all-null polars columns rather than crashing or silently misclassifying them.
- **Careful, direct bounds-checking (not heuristics) for dtype downcasts** in `optimize_dtypes`: every candidate downcast is checked against the real `np.iinfo`/`np.finfo` min/max of the *actual* target dtype before being applied, plus a dedicated mantissa-based float-precision-loss detector (with an inline regression-test-style comment: `"np.array([2.205001270000e09]).astype(np.float32) must not pass here"`) that goes well beyond a naive range check.
- **`clean_numeric` in polarslib.py (59-68)** documents and correctly avoids the classic `NaN != NaN` trap in `.replace([inf,-inf,NaN], ...)`-style cleanup, using `.is_finite()` instead, with an explicit comment explaining the failure mode it's avoiding.
- **Correct closure-capture idiom** in `build_aggregate_features_polars`'s nested `af()` helper (`polarslib.py:358-360`), which captures `filter_field`/`filter_value` as default-argument values rather than by reference — avoiding the classic Python late-binding-closure-in-a-loop bug that would otherwise make every subgroup's filter expression silently reference the *last* loop iteration's values.
- No `.iterrows()`/`.itertuples()`-over-data-rows anti-patterns anywhere in scope (the one `itertuples()` call, `dtypes.py:193`, iterates over a tiny per-*column* min/max summary table, not per-row data).
- No chained-indexing (`df[...][...] = ...`) `SettingWithCopyWarning` risk anywhere in scope — all in-place mutations use `.loc[...]` or direct `df[col] = ...` column assignment.
- `showcase_df_columns` ships runnable doctests exercising both the pandas and polars code paths (including the `dropna=True`/`False` and `max_vars` edge cases), which is genuinely useful executable documentation.

## Investigated, not an issue

- **Numba object-mode silent fallback** (the audit's stated concern): every JIT-decorated function in scope (`numbalib.py`, `stats/normality.py`) uses `@njit`, never bare `@jit`. `@njit` is strict nopython mode — it raises a loud `TypingError` on unsupported types rather than silently degrading to object mode (that legacy fallback behavior applies to `@jit` without `nopython=True`, which is not used anywhere in this scope). So the specific "looks optimized but silently gets no speedup" failure mode does not materialize here.
- **`entropy_for_column`/`mi_for_column` (polarslib.py) division by `len(bins)`**: for a zero-row `bins` input this looks like a potential `ZeroDivisionError`, but `np.array([]) / 0` on an empty numpy array does not raise (there are no elements to divide), so `-np.sum(...)` on the empty result is `0.0`, not a crash.
- **`optimize_dtypes` integer-overflow protection**: extensively traced the size-reduction loop (uint/int passes, possibly-integer float->int promotion, the `p >= cur_power` early-break logic, and the `uint_fields` dedup-across-passes bookkeeping) looking for an overflow/double-conversion bug; every downcast is gated by an explicit `r.max <= topvals[j].max and r.min >= topvals[j].min` check against the real target dtype's bounds, and columns already converted in an earlier pass are correctly excluded from later passes via the `uint_fields` accumulator. No overflow bug found despite the deliberate hunt.
- **`optimize_dtypes` mixed-type / all-NaN object columns**: traced the try/int64 -> try/float64 -> try/category fallback chain for an all-`None` `object` column; it correctly falls through to a `float64` NaN column (via the second `try`), which is then correctly skipped by the later `float_to_int` promotion path (guarded by `df[col].isna().any()`) rather than raising or silently corrupting.
- **`ensure_dataframe_float32_convertability` (dtypes.py:310-352)**: casts Int32/Int64/UInt32/UInt64/Int128 columns straight to Float32, the same pattern flagged as a bug in `cast_f64_to_f32` above — but here it is fully consistent with its own docstring (explicitly documents converting *all* numeric columns to float32 for LightGBM compatibility), so this one is working as designed, not a bug.

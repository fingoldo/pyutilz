# Data / Stats Domain Audit — pyutilz (2026-09-02)

## Summary

Read `CLAUDE.md` (dispositioned items: no community-health files, no repo-wide formatter runs — neither touched here), `audits/2026-07-21_full-audit/08-domain-data.md` and all of `audits/2026-07-21_audit-round2/` (esp. `03-numerical-data-correctness.md`). Every 2026-07-21 finding in this domain has since been fixed in the source and is NOT re-raised: `cast_f64_to_f32` int->Float32 (now documented + magnitude warning), `get_columns_of_type` duplicates (now `any()`), `find_nan_cols` null-blindness (now `is_nan | is_null`), `remove_stale_columns` (now a deprecated alias of `get_non_stale_columns`), `smart_ratios` docstring, `io_ops` `deep=True`, `optimize_dtypes` RuntimeWarning (`np.errstate(invalid="ignore")`), `generate_combinations_recursive_njit` negative `r`, `normality_verdict` zero-variance (now "degenerate"), `get_topk_indices` NaN ranking, `share_dataframe` int64 > 2**53, `bin_dtype=Int8` vs `num_bins` (now auto-widens), `add_weighted_aggregates` zero-weight Inf (now wrapped in `clean_numeric`).

Files read in full: `data/polarslib.py`, `data/numpylib.py`, `data/numbalib.py`, `data/pandaslib/{__init__,_common,dtypes,frames,io_ops}.py`, `stats/normality.py`. Everything below was reproduced by running the actual package code with `D:/ProgramData/anaconda3/python.exe` (scratch scripts under `D:\Temp\audit0902`, nothing written into the repo) on polars 1.33.1 / pandas 3.0.3 / numpy 1.26.4; observed output is quoted verbatim in each finding. `dagostino_k2` and `anderson_darling_normal` were cross-checked against `scipy.stats.normaltest` / `scipy.stats.anderson` and match to ~1e-13 relative on n=5000 Normal and n=2000 exponential samples — no formula errors found there.

Findings: 2 High, 4 Medium, 11 Low (17 total).

## Findings

### F01. [High] `np.allclose`'s default rtol=1e-5 makes the "constant column" test drop columns with real variation — src/pyutilz/data/polarslib.py:1055 (and the identical check at :872)
- **Disposition**: OPEN
- **Category**: statistical-formula-error / silent-data-loss
- **Problem**: Both `drop_constant_columns` (line 1055) and `bin_numerical_columns` (line 872) decide "no change" with `np.allclose(min_val, max_val)`. `np.allclose` is a *relative* comparison (`|a-b| <= atol + rtol*|b|`, atol=1e-8, rtol=1e-5), not an equality test, so any column whose value range is under 1e-5 of its own magnitude is declared constant and dropped. Observed:

  ```
  >>> drop_constant_columns(pl.DataFrame({"uid":[1_000_000_000, 1_000_000_001, 1_000_000_002]})).columns
  []                                   # a strictly increasing int64 id column, silently dropped
  >>> df = pl.DataFrame({"big":[1e9, 1_000_002_000.0, 1_000_001_000.0]})   # std = 1000.0
  >>> drop_constant_columns(df).columns
  []
  ```

  A sweep over magnitudes with a fixed relative range of 2e-6 (`[m, m*(1+1e-6), m*(1+2e-6)]`) drops the column at every magnitude tested (1e2, 1e4, 1e6, 1e8, 1e10) — the failure is scale-free, it is purely about relative spread. Note the 2026-07-21 audit listed this `np.allclose` guard under "Things done well" for its `None` handling; the `None` handling is fine, the tolerance is not.
- **Failure scenario**: A frame with epoch-microsecond timestamps, large monotone ids, unix-nanosecond columns, or any sensor channel whose readings vary in the 6th significant digit (prices around 1e9, cumulative counters) loses those columns entirely. In `bin_numerical_columns` the same columns are added to `columns_to_drop` and never binned, so a downstream MI / feature-selection run silently never sees them; in `drop_constant_columns` they vanish from the returned frame with only a `verbose`-gated warning.
- **Suggested fix**: Use exact equality (`min_val == max_val`) — polars min/max return the actual stored values and the `None` pre-check already covers all-null columns — or, if a tolerance is genuinely wanted, make it explicit and caller-supplied (`np.isclose(min_val, max_val, rtol=rtol, atol=atol)` with `rtol=0.0` default) rather than inheriting numpy's 1e-5.

### F02. [High] `bin_numerical_columns` crashes with a raw polars `OutOfBoundsError` on empty, single-row, or all-constant frames — src/pyutilz/data/polarslib.py:966
- **Disposition**: OPEN
- **Category**: empty-input-edge-case / crash
- **Problem**: After the dead-column pass drops every constant column, `df` can have zero columns. Line 966 (`fill_nulls=True`, the default) then runs `df.lazy().select(pl.all().null_count()).collect().row(0, named=True)`; `pl.all()` over a zero-column frame collects to a 0-row 0-column frame, and `.row(0)` raises. Observed for three separate inputs:

  ```
  0-row    : pl.DataFrame(schema={"a":pl.Float64,"b":pl.Float64})
  1-row    : pl.DataFrame({"a":[1.0],"b":[2.0]})
  all-const: pl.DataFrame({"a":[1.0,1.0],"b":[2.0,2.0]})
  -> File ".../polarslib.py", line 966, in bin_numerical_columns
       cols_with_nulls = [key for key, value in df.lazy().select(pl.all().null_count()).collect().row(0, named=True).items() ...]
     polars.exceptions.OutOfBoundsError: index 0 is out of bounds for sequence of length 0
  ```

  All three are ordinary inputs, not adversarial: a single-row frame is by definition all-constant, so *any* one-row call crashes. The sibling `drop_constant_columns` handles the same 0-row frame fine (`-> []`).
- **Failure scenario**: A feature-selection pipeline calling `bin_numerical_columns` per group/fold hits a fold that filtered down to one row (or whose numeric features are all constant after filtering) and dies with an opaque polars index error instead of returning empty bins; the caller cannot tell this apart from a genuine polars bug.
- **Suggested fix**: Guard the post-drop block on `df.width == 0` (return `(pl.DataFrame(), binned_targets, public_clips, columns_to_drop, stats)` early), and/or width-check before the `.row(0, named=True)` reads at lines 966 and 968.

### F03. [Medium] `clean_numeric` silently converts NULL to `nans_filler`, conflating "not computable" with a real value — src/pyutilz/data/polarslib.py:82
- **Disposition**: OPEN
- **Category**: null-vs-nan-handling
- **Problem**: The docstring promises "Replace non-finite floats (inf, -inf, NaN) with `nans_filler`", but `pl.when(expr.is_finite()).then(expr).otherwise(...)` also catches nulls: `null.is_finite()` is null, and polars treats a null `when` predicate as false, so the `otherwise` branch fires. Observed:

  ```
  >>> pl.DataFrame({"x":[1.0, None, nan, inf]}).select(clean_numeric(pl.col("x"), nans_filler=-1.0))
  [1.0, -1.0, -1.0, -1.0]          # the None became -1.0, undocumented
  ```

  This is not theoretical — it reaches two production paths. (a) `create_ts_features_polars` runs `clean_numeric(fragile_cols.cast(dtype))` over every `_skew/_kurtosis/_entropy/_c3_stats/_cid_ce/corr_/_linreg/_lempel_ziv` column, and polars returns **null** (not NaN) for `std()` of a single-row group. Observed on a group-by where group "b" has one row: raw `{'c': [nan, nan], 's': [0.0, None]}` -> cleaned `{'c': [0.0, 0.0], 's': [0.0, 0.0]}`. (b) `bin_numerical_columns` applies `clean_numeric(col_expr, nans_filler=min_val)` under `fill_nans`, so with `fill_nulls=False, fill_nans=True` nulls are still pushed into the `min_val` bin, defeating the `fill_nulls=False` request.
- **Failure scenario**: A window with a single observation produces `std = null` and `corr = null`; the feature matrix records `0.0`, i.e. "measured zero dispersion / zero correlation", indistinguishable from a real zero. A model trained on this learns from fabricated zeros, and any downstream missingness audit reports 0% nulls.
- **Suggested fix**: Either let nulls pass through explicitly (`pl.when(expr.is_null()).then(expr).when(expr.is_finite()).then(expr).otherwise(lit)`) behind an opt-in `fill_nulls: bool = False` parameter, or keep the behavior but document it and give callers a way to distinguish "was null" — silently filling is the one option that hides it.

### F04. [Medium] `optimize_dtypes(..., inplace=False)` still mutates the caller's DataFrame in the object->numeric branch — src/pyutilz/data/pandaslib/dtypes.py:118, 123
- **Disposition**: OPEN
- **Category**: copy-vs-view / inplace-contract-violation
- **Problem**: The size-reduction branch correctly guards its writes with `if inplace: df[col] = ...` (lines 230-231) and returns `df.astype(new_dtypes)` when `inplace=False`. But the earlier object/string handling writes `df[col] = candidate` (line 118) and `df[col] = df[col].astype(np.float64)` (line 123) unconditionally, with no `inplace` check. Observed:

  ```
  >>> df = pd.DataFrame({"s":["1","2","3"], "t":["1.5","2.5","3.5"]})
  >>> _ = optimize_dtypes(df, inplace=False)
  >>> df.dtypes.to_dict()
  {'s': dtype('int64'), 't': dtype('float64')}      # caller's frame changed; was StringDtype for both
  >>> df["s"].tolist(), df["t"].tolist()
  ([1, 2, 3], [1.5, 2.5, 3.5])                      # values are now numbers, not strings
  ```
- **Failure scenario**: Code that deliberately passes `inplace=False` to probe candidate dtypes without touching the original (e.g. sizing a frame before deciding whether to re-save, as `read_stats_from_multiple_files` does) finds its "untouched" frame's string columns silently retyped; a later `.str` accessor call raises `AttributeError` far from the cause.
- **Suggested fix**: In the `max_categories` block, compute the candidate into a local and assign back only under `if inplace:`; when `inplace=False`, record the target dtype in `new_dtypes` (as the size-reduction branch already does) and let the final `df.astype(new_dtypes)` apply it.

### F05. [Medium] `share_dataframe` silently discards the caller's index — src/pyutilz/data/pandaslib/frames.py:389
- **Disposition**: OPEN
- **Category**: index-alignment
- **Problem**: The shared pieces are built from `sub.values` (positional) and reassembled with `pd.DataFrame({c: pieces[c] for c in df.columns})`, which gives the result a fresh `RangeIndex`. Nothing in the docstring mentions the index. Observed:

  ```
  >>> d = pd.DataFrame({"a":[1.0,2.0],"b":[10,20]}, index=["r1","r2"])
  >>> list(share_dataframe(d).index)
  [0, 1]                                            # original index ["r1","r2"] lost
  ```
- **Failure scenario**: A worker process gets the shared frame as a global and does `shared_df.loc[key]` / `shared_df.join(other)` / `shared_df["x"] - other["x"]` against a frame still carrying the original labels. `.loc[key]` raises `KeyError`; the arithmetic/join silently aligns on mismatched labels and yields all-NaN rows rather than an error.
- **Suggested fix**: Pass the original index through (`pd.DataFrame({...}, index=df.index)`), or document loudly that the returned frame is positionally indexed and callers must reset their own frames to match.

### F06. [Medium] `remove_constant_columns` and `get_non_stale_columns` disagree on null semantics, so one drops a column the other keeps — src/pyutilz/data/pandaslib/frames.py:440 vs :407
- **Disposition**: OPEN
- **Category**: null-handling / api-inconsistency
- **Problem**: `get_non_stale_columns` uses `df.nunique(dropna=False)` (with an inline comment explaining exactly why). Its sibling `get_suspiciously_constant_columns` — the screening step for `remove_constant_columns` — uses bare `df.nunique()`, i.e. `dropna=True`. A column with one real value plus nulls therefore has nunique 2 for one function and 1 for the other. Observed on the same frame:

  ```
  >>> d = pd.DataFrame({"mostly_nan":[1.0, nan, nan, nan], "real":[1.0,2.0,3.0,4.0]})
  >>> get_non_stale_columns(d)          -> ['mostly_nan', 'real']    # keeps it
  >>> remove_constant_columns(d); d.columns.tolist()
  ['real']                                                            # drops it, in place
  ```

  `get_suspiciously_constant_columns`'s own docstring claims it returns columns that are "constant or all-NaN" — a column that is neither is being returned.
- **Failure scenario**: A pipeline that screens features with `get_non_stale_columns` and later cleans the same frame with `remove_constant_columns` loses a rare-event indicator column (one positive observation, rest missing) between the two steps, and the feature list computed earlier no longer matches the frame.
- **Suggested fix**: Make `get_suspiciously_constant_columns` use `nunique(dropna=False)` to match its sibling and its own docstring (the all-NaN case still yields nunique==1 and stays flagged), or add an explicit `dropna` parameter threaded from `remove_constant_columns` with the same default in both.

### F07. [Low] `normality_verdict` reports "too-few-samples" for 8 <= n < 20 although Anderson-Darling is valid there — src/pyutilz/stats/normality.py:274
- **Disposition**: OPEN
- **Category**: degenerate-input / stats
- **Problem**: `anderson_darling_normal` explicitly supports `n >= 8` (lines 195-197), and `dagostino_k2` returns NaN below 20 which the `np.isfinite` guards on `reject_k2`/`reject_ad` already handle. But `normality_verdict` short-circuits the whole function at `n_total < 20`, so A-D never runs on samples of 8..19. Observed: `normality_verdict(rng.normal(size=10))["verdict"] -> 'too-few-samples'`, with `ad_stat`/`ad_p` both NaN even though `anderson_darling_normal` would have produced a finite statistic.
- **Failure scenario**: A per-group residual audit on small groups (n=10..19) reports "too-few-samples" and `reject_normal=False` for every such group, so a strongly non-Normal small group is indistinguishable from an untested one.
- **Suggested fix**: Lower the early-return threshold to `n_total < 8`, letting the existing `np.isfinite(k2_p)` guard handle the still-NaN K2 half, and use a distinct verdict string (e.g. "AD-only (n<20)") so the caller knows only one test ran.

### F08. [Low] `div0` replaces legitimately infinite *inputs*, not just division-by-zero results — src/pyutilz/data/numpylib.py:85
- **Disposition**: OPEN
- **Category**: nan-inf-handling
- **Problem**: The docstring frames the function as "a / b, divide by 0 -> fill", but the implementation masks on the *result* being non-finite, which also catches an inf that arrived in `a` or came from overflow. Observed:

  ```
  >>> div0(np.array([np.inf, 1.0]), np.array([1.0, 1.0]), na_fill=-999)
  array([-999.,    1.])          # inf/1 == inf was a correct result, not a 0-division
  >>> smart_ratios(np.array([1.0,2.0]), np.array([0.0, 1e-320]))
  array([nan, nan])              # 2/1e-320 overflowed to inf -> also replaced
  ```
- **Failure scenario**: A ratio series that legitimately contains `+inf` (an upstream saturation sentinel, or overflow from a denormal denominator) is rewritten to `na_fill`, so downstream code sees "missing" where the true answer was "unbounded" — and imputation treats the two very differently.
- **Suggested fix**: Mask on the denominator (`b == 0` / `~np.isfinite(b)`) rather than on the quotient, or document that any non-finite result — including from non-finite inputs and from overflow — is replaced.

### F09. [Low] `optimize_dtypes`'s float32 precision guard rounds the mantissa to 14 significant digits, so it cannot see losses beyond that — src/pyutilz/data/pandaslib/dtypes.py:213
- **Disposition**: OPEN
- **Category**: dtype-precision
- **Problem**: `ensure_float64_precision=True` (default) is described as ensuring "we are not losing precision", but the check first does `np.round(values / 10**int_part, np.finfo('float64').precision - 1)` — 14 decimals of the mantissa — so any distinction living in digits 15-17 of a float64 is erased *before* the test. Observed with default settings (`float_to_int=False`):

  ```
  column "eps" = [1.0000000000000002, 1.0, 2.0, 3.0, 4.0]  ->  dtype('float32')
  max abs round-trip error 2.220446049250313e-16
  ```

  In float32 the first two entries become bit-identical, so two distinct float64 values collapse into one. (The `[2.205001270000e09]*5` case named in the code comment is correctly rejected, and `[0.1..0.5]` is downcast with ~1.2e-7 relative error, which is the intended decimal-digit heuristic.)
- **Failure scenario**: A column whose values differ only in the last few float64 bits (accumulated sums, tiny residuals, near-tied scores used as ranking keys) is downcast to float32; ties appear where there were none, and an `argmax`/dedup over that column changes answer.
- **Suggested fix**: Add a direct round-trip check alongside the mantissa heuristic — `np.array_equal(values.astype(f'float{p}').astype(np.float64), values, equal_nan=True)` — or at minimum document that the guard protects decimal-repr fidelity to 14 significant digits, not bit-exactness.

### F10. [Low] `classify_column_types` reports period / interval / timedelta dtypes as numeric — src/pyutilz/data/pandaslib/dtypes.py:269
- **Disposition**: OPEN
- **Category**: dtype-classification
- **Problem**: `col_is_numeric` is computed by exclusion (`not (bool or object or datetime or category)`), so every dtype outside those four buckets falls through to "numeric". Observed:

  ```
  period[D]            -> (False, False, False, False, True)
  timedelta64[ns]      -> (False, False, False, False, True)
  Sparse[float64, nan] -> (False, False, False, False, True)
  interval             -> (False, False, False, False, True)
  ```

  `period[D]` and `interval` are not numeric under any pandas API (`pd.api.types.is_numeric_dtype` is False for both).
- **Failure scenario**: A caller routing "numeric" columns into `df[cols].mean()` / a correlation matrix / a scaler hits a `TypeError` from pandas on the period or interval column, or (for Sparse) gets a densification it did not ask for.
- **Suggested fix**: Compute `col_is_numeric` positively via `pd.api.types.is_numeric_dtype(dtype)`, keeping the existing boolean/object/datetime/categorical flags as they are.

### F11. [Low] `get_non_stale_columns` returns ALL columns for a 0-row frame but NONE for a 1-row frame — src/pyutilz/data/pandaslib/frames.py:401, 407
- **Disposition**: OPEN
- **Category**: empty-input-edge-case
- **Problem**: The explicit `len(df) == 0` early return hands back every column, while a single-row frame falls into `nunique(dropna=False) <= 1`, true for every column by construction. Observed:

  ```
  >>> get_non_stale_columns(pd.DataFrame({"a":[],"b":[]}))   -> ['a', 'b']
  >>> get_non_stale_columns(pd.DataFrame({"a":[1],"b":[2]})) -> []      (logs "Found 2 stale columns: a,b")
  ```

  Neither answer is wrong in isolation, but the discontinuity is undocumented and the n=1 case is the more common one in practice.
- **Failure scenario**: A scoring path that reuses this helper to pick features for a single-row inference frame gets an empty feature list and either predicts on nothing or raises on the empty selection.
- **Suggested fix**: Return all columns for `len(df) <= 1` (matching the 0-row convention — a single row carries no evidence of staleness) and say so in the docstring.

### F12. [Low] `showcase_df_columns`'s uninformative-fraction divides by the full row count even when `dropna=True` — src/pyutilz/data/pandaslib/frames.py:266 (polars) and :300 (pandas)
- **Disposition**: OPEN
- **Category**: null-handling / statistic-denominator
- **Problem**: `height` is `len(df)` / `df.height` regardless of `dropna`, while the counts it divides come from `value_counts(dropna=dropna)`, which excluded nulls. `rare_threshold = max_unique_percent * height` has the same mismatch. Observed on `{"c": ["a"]*97 + ["b"]*1 + [None]*2}` with `max_unique_percent=0.02`:

  ```
  dropna=True  -> rare {'c': ['b']}        uninformative {'c': 0.030000000000000027}
  dropna=False -> rare {'c': [nan, 'b']}   uninformative {'c': 0.030000000000000027}
  ```

  With `dropna=True` the non-dominant share among the 98 non-null rows is `1 - 97/98 = 0.0102`, not `0.03`; the two modes report an identical number despite analysing different populations.
- **Failure scenario**: A feature-selection threshold applied to the returned fraction (e.g. "drop if < 2% non-dominant") behaves differently than the caller's `dropna` choice implies, and null-heavy columns are scored as if their nulls were a third category even when the caller asked to ignore them.
- **Suggested fix**: Take the denominator from the value_counts total actually in play (`sum(counts)` / `vc["count"].sum()`) rather than the frame height, and derive `rare_threshold` from that same total.

### F13. [Low] `polars_df_info` always prints GB (local misnamed `size_kb`), showing "0.0+ GB" for anything under ~50 MB — src/pyutilz/data/polarslib.py:1089
- **Disposition**: OPEN
- **Category**: reporting / naming
- **Problem**: `size_kb = df.estimated_size(unit="gb")` then `f"memory usage: {size_kb:.1f}+ GB"` — the unit is hardcoded to gigabytes at one decimal, so a function whose docstring calls itself "pandas-`.info()`-style" reports nothing usable for ordinary frames, unlike pandas which auto-scales bytes/KB/MB. Observed:

  ```
  >>> print(polars_df_info(pl.DataFrame({"a":[1,2]})))
  ...
  memory usage: 0.0+ GB
  ```

  The local name `size_kb` also contradicts the `unit="gb"` argument it holds.
- **Failure scenario**: Any inspection of a sub-50MB frame (i.e. most interactive use) shows `0.0+ GB` and conveys no information; a reader trusting the variable name would read the number as kilobytes.
- **Suggested fix**: Pick the unit by magnitude (bytes/KB/MB/GB) the way `pandas.DataFrame.info()` does, and rename the local to `size`.

### F14. [Low] `convert_float64_to_float32` mutates its input in place while presenting a return-value API — src/pyutilz/data/pandaslib/dtypes.py:373
- **Disposition**: OPEN
- **Category**: copy-vs-view
- **Problem**: The function assigns `df[col] = df[col].astype(np.float32)` on the caller's frame and also returns it, with no `.copy()` and no note in the docstring — while its documented replacement `ensure_dataframe_float32_convertability` explicitly promises "Always returns a NEW object and never mutates the input in place". Observed:

  ```
  >>> d = pd.DataFrame({"x":[1.0,2.0]}); convert_float64_to_float32(d)   # return discarded
  >>> d.dtypes.to_dict()
  {'x': dtype('float32')}
  ```
- **Failure scenario**: Code that keeps `df` as a float64 reference and expects `df32 = convert_float64_to_float32(df)` to be a separate downcast copy silently loses the float64 originals; a later precision-sensitive computation on `df` runs at float32.
- **Suggested fix**: `df = df.copy()` at the top (matching the sibling's documented contract), or document the in-place mutation in the docstring.

### F15. [Low] `ensure_dataframe_float32_convertability` references `pl.Int128` unguarded while the sibling `cast_f64_to_f32` guards it with `hasattr` — src/pyutilz/data/pandaslib/dtypes.py:335
- **Disposition**: OPEN
- **Category**: version-compat-inconsistency
- **Problem**: `polarslib.cast_f64_to_f32` builds its dtype list defensively (`if hasattr(pl, "Int128"): int_types.append(pl.Int128)`, with a comment noting Int128 is absent in older polars), but `dtypes.py:335` hardcodes `pl.Int128` inside the selector list. On the installed polars 1.33.1 both work; in any environment where the `hasattr` guard is actually needed, this line raises `AttributeError` at call time. The two functions are documented as mirrors of each other (see `cast_f64_to_f32`'s docstring), so the divergence is unintentional.
- **Failure scenario**: A consumer pinned to an older polars — the exact case the guard exists for — gets `AttributeError: module 'polars' has no attribute 'Int128'` from `ensure_dataframe_float32_convertability`'s polars branch, while the pandas branch and `cast_f64_to_f32` work fine.
- **Suggested fix**: Build the dtype list the same way in both places — a shared module-level helper returning the int dtype list, used by `cast_f64_to_f32` and by this function.

### F16. [Low] `get_topk_indices` returns arbitrary indices for an all-NaN slice with no signal — src/pyutilz/data/numpylib.py:429-446
- **Disposition**: OPEN
- **Category**: nan-handling / degenerate-input
- **Problem**: The NaN substitution maps every element to the same sentinel (`-inf` for `highest`, `+inf` for `lowest`), so on an all-NaN input the argpartition order is arbitrary and the caller gets a valid-looking index. Observed:

  ```
  >>> a = np.array([np.nan, np.nan, np.nan])
  >>> get_topk_indices(a, k=1, highest=True), get_topk_indices(a, k=1, highest=False)
  (array([2], dtype=int64), array([0], dtype=int64))
  ```

  Two different positions are returned for the same all-NaN data depending only on the direction flag. The docstring covers the "not enough real values to fill k slots" case but says nothing about the fully-degenerate one.
- **Failure scenario**: A candidate-ranking loop over a score matrix hits a row where every score failed to compute (all NaN) and picks candidate 2 as "best" with full confidence, instead of the caller being able to detect that no candidate was rankable.
- **Suggested fix**: Document the behavior precisely, and/or offer an opt-in that returns `-1` (or raises) for slices with fewer than `k` finite values — the information is already available from the `np.isnan` mask that is computed anyway.

### F17. [Low] `entropy_for_column` / `mi_for_column` count null as an ordinary bin category — src/pyutilz/data/polarslib.py:747, 752
- **Disposition**: OPEN
- **Category**: null-handling
- **Problem**: `_group_freqs` uses `bins.group_by(cols)`, and polars `group_by` emits null as its own group, so nulls contribute real probability mass to the Shannon entropy. Observed:

  ```
  >>> b = pl.DataFrame({"x":[0, None, 1, 1], "t":[0,1,0,1]})
  >>> entropy_for_column(b, "x")
  1.0397207708399179            # H over three categories {0, null, 1}; over the 3 non-null rows it is 0.6365
  ```

  Whether that is desirable depends on the caller, and `bin_numerical_columns` only guarantees null-free bins when `fill_nulls=True` — a caller feeding hand-made bins gets missingness silently folded into the information estimate.
- **Failure scenario**: A feature whose only signal is its missingness pattern scores a high MI against the target purely from the null group and is selected as informative; conversely, two features with identical observed distributions but different missingness rank differently for a reason the entropy number does not disclose.
- **Suggested fix**: Add an explicit `drop_nulls: bool` parameter to `_group_freqs`/`entropy_for_column`/`mi_for_column` (defaulting to current behavior to avoid a silent change), and state in the docstrings that null is currently treated as its own bin.

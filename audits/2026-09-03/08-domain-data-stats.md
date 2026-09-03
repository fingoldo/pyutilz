# Data / Stats Domain Audit — pyutilz (2026-09-03)

## Summary

Read `CLAUDE.md` in full (dispositioned: no repo-wide formatter runs, no community-health files, Python 3.8 floor / no PEP 639 migration, no trimming of environment-specific `# type: ignore` codes — none of those areas are touched here) and `audits/implemented/2026-09-02/08-domain-data-stats.md` (all 17 findings COMPLETED). None of the 2026-09-02 findings is re-raised; I re-checked the two adjacent to findings below and both are still fixed (exact `min_val == max_val` comparison in place at `polarslib/frames.py:56` and `polarslib/binning.py:198`; `bin_numerical_columns` returns cleanly on 0-column/1-row frames at `binning.py:290-295`). One finding below (F03) is a *new defect introduced by* the 2026-09-02 fix for that audit's F17 (the `drop_nulls` parameter it added).

Files read in full: `data/polarslib/{_common,columns,frames,binning,aggregations,__init__}.py`, `data/pandaslib/{_common,dtypes,frames,io_ops,benchmarks,__init__}.py`, `data/numpylib.py`, `data/numbalib.py`, `data/git_checkpoint_cache.py`, `stats/{__init__,normality}.py`.

Everything below was reproduced by running the installed package code with `D:/ProgramData/anaconda3/python.exe` (scratch scripts under `D:\Temp\aud0903`, nothing written into the repo) on **polars 1.44.1 / pandas 3.0.3** on Windows; observed output is quoted verbatim in each finding. Checks that came back CLEAN and are therefore not reported: no polars or pandas DeprecationWarning/FutureWarning is emitted by any pyutilz call site exercised here (`concat_horizontal_ragged`, `explode_keeping_empty_as_null`, `bin_numerical_columns`, `entropy_for_column`, `mi_for_column`, `optimize_dtypes`, `showcase_df_columns` all ran under `warnings.simplefilter("always"/"error")` with zero warnings); the `normality.py` formulas (`dagostino_k2` moment transforms, `anderson_darling_normal` ddof=1 + Stephens correction + piecewise-p continuity at the 0.200/0.340 knots) hold up; no Python 3.8-incompatible syntax exists in any file in scope (the only PEP 585 generic, `frames.py:64`, is inside a string annotation).

Findings: 5 High, 4 Medium, 2 Low (11 total).

## Findings

### F01. [High] Weighted aggregates ignore the active `subgroups` filter, so every subgroup gets the same whole-group number under a subgroup-specific name — src/pyutilz/data/polarslib/aggregations.py:369

- **Disposition**: OPEN
- **Category**: groupby-semantics
- **Problem**: Inside `build_aggregate_features_polars`, every other family of expressions is wrapped in the per-subgroup helper `af(...)` (defined at :312) which appends `.filter(pl.col(filter_field) == filter_value)`. The `weighting_fields` branch at :368-375 calls `add_weighted_aggregates(columns_selector=(cs.numeric() - cs.by_name(...)), ...)` and passes `fpref` — the subgroup name prefix — for **naming only**. `add_weighted_aggregates` (:131) then builds `(all_other_num_cols * pl.col(wcol)).sum() / pl.col(wcol).sum()` with no filter anywhere, so the expression is evaluated over the whole group while the alias claims it belongs to one subgroup.
- **Failure scenario**: `df = pl.DataFrame({"g":[1,1,1,1], "side":["buy","buy","sell","sell"], "px":[10.,20.,100.,200.], "vol":[1.,1.,1.,1.]})` with `numerical_fields=["px"]`, `weighting_fields=["vol"]`, `subgroups={"side":["buy","sell"]}`, then `df.group_by("g").agg(exprs)`. Observed:
  ```
  'side_buy_px_mean': 15.0,   'px_side_buy_wmeanby_vol': 82.5
  'side_sell_px_mean': 150.0, 'px_side_sell_wmeanby_vol': 82.5
  ```
  The correct volume-weighted means are 15.0 (buy) and 150.0 (sell); 82.5 is the unfiltered `(10+20+100+200)/4`. Both subgroups get the identical wrong value, so a model fed these features sees a constant column where it was told it has buy-side and sell-side price levels.
- **Suggested fix**: Thread the filter into the expression, not just the name — either pass the `af` callable into `add_weighted_aggregates` and wrap both the numerator selector and `pl.col(wcol)` (`af(all_other_num_cols * pl.col(wcol)).sum() / af(pl.col(wcol)).sum()`), or apply `.filter(...)` to the returned expressions in the caller before extending `feature_expressions`. Add a regression test asserting the two subgroup values differ on the frame above.

### F02. [High] `polars-ds` linreg / corr / othersvals expressions also ignore the `subgroups` filter, emitting identical values under distinct subgroup names — src/pyutilz/data/polarslib/aggregations.py:521

- **Disposition**: OPEN
- **Category**: groupby-semantics
- **Problem**: Same root cause as F01 but three more expression families, each of which builds `pl.col(field)` directly instead of `af(pl.col(field))` while still aliasing with `fpref`:
  - linreg, :521 — `pds.simple_lin_reg(pl.int_range(pl.len()), target=pl.col(field), add_bias=True).alias(alias)` (neither the row index nor the target is filtered);
  - linreg-by-timestamp, :534-538 — same, plus an unfiltered `pl.col(linreg_timestamp_field)`;
  - correlations, :515 — `pds.corr(corr_x, corr_y, method=corr_method)` takes bare column *names*, so no filter can be attached at all;
  - othersvals-at-extremums, :394-395 — `other_columns.get(pl.col(col).arg_max())`, unfiltered on both sides.
- **Failure scenario**: The F01 frame, same call. Observed:
  ```
  'side_buy_px_linreg':  [65.0, -15.0]
  'side_sell_px_linreg': [65.0, -15.0]
  ```
  Both are the OLS fit over all four rows (x=[0,1,2,3], y=[10,20,100,200] -> slope 325/5 = 65.0), not the buy fit (slope 10) or the sell fit (slope 100). Every `_linreg_k` / `_linreg_b` / `corr_*` / `*_at_*_max` feature is therefore duplicated across subgroups with a value belonging to none of them.
- **Suggested fix**: Wrap each target in `af(...)`: `pds.simple_lin_reg(pl.int_range(af(pl.col(field)).len()), target=af(pl.col(field)), ...)` and `af(pl.col(linreg_timestamp_field))`; for `pds.corr`, pass expressions instead of names (`pds.corr(af(pl.col(corr_x)), af(pl.col(corr_y)), method=...)`) or skip corr features entirely when a filter is active rather than mislabelling them; wrap the `othersvals` `arg_max`/`arg_min` and their `other_columns.get(...)` in `af`. If any of these genuinely cannot be filtered, drop the `fpref` from their alias so the name stops asserting a subgroup.

### F03. [High] `mi_for_column(drop_nulls=True)` mixes three different row populations and returns a mutual information larger than either marginal entropy — src/pyutilz/data/polarslib/binning.py:81

- **Disposition**: OPEN
- **Category**: statistical-formula
- **Problem**: `_group_freqs(bins, cols, drop_nulls=True)` (:51-55) calls `bins.drop_nulls(subset=cols)` and normalizes by the length of *that* subset. `entropy_for_column` therefore estimates H(X) on the rows where X is non-null, H(Y) on the rows where Y is non-null, and `mi_for_column`'s joint term on the rows where BOTH are non-null. `mi = H(X) + H(Y) - H(X,Y)` is only valid when all three come from the same sample, so the identity breaks and the bound `MI <= min(H(X), H(Y))` is violated. This code path did not exist before the 2026-09-02 fix for that audit's F17, which added the `drop_nulls` parameter.
- **Failure scenario**:
  ```
  x = [None]*40 + [0]*30 + [1]*30
  y = [0]*20 + [1]*20 + [0]*30 + [None]*30
  bins = pl.DataFrame({"x": x, "y": y})
  e = {c: entropy_for_column(bins, c, drop_nulls=True) for c in ("x", "y")}
  mi_for_column(bins, e, "x", "y", drop_nulls=True)
  ```
  Observed:
  ```
  H: {'x': 0.6931471805599453, 'y': 0.5982695885852573}
  MI(drop_nulls=True): 1.2914167691452025
  ```
  MI = 1.291 nats exceeds H(x) = 0.693 nats, which is mathematically impossible (MI <= min(H)). The joint subset here is 30 rows that are all `(0, 0)`, so the joint entropy is 0 and the two marginals — measured on disjoint populations — simply add. A feature-ranking pass that sorts by this MI puts the columns with the most *complementary* missingness on top.
- **Suggested fix**: Under `drop_nulls=True`, restrict every term to the same complete-case subset: have `mi_for_column` compute `sub = bins.drop_nulls(subset=[col, target_col])` once and derive H(col), H(target) and H(col, target) from `sub`, instead of reading precomputed per-column marginals. Either recompute the two marginals locally, or make `entropies` a `{(col, target): (h_col, h_target)}` structure; at minimum, assert `0 <= mi <= min(h_col, h_target)` so the inconsistency cannot pass silently.

### F04. [High] `benchmark_dataframe_compression` corrupts the parquet rows into a row of column-name strings and then raises `UFuncTypeError` — src/pyutilz/data/pandaslib/benchmarks.py:270

- **Disposition**: OPEN
- **Category**: dataframe-construction
- **Problem**: `res` is accumulated by `pack_benchmark_results` as **lists** (`[config, mean, std, ...]`, :78). The parquet block then does `res.extend(parquet_results.to_dict("records"))` (:270), appending **dicts**. `pd.DataFrame(list_of_mixed_lists_and_dicts, columns=[...])` treats each dict as an iterable of its keys, so each parquet row becomes a literal row of the strings `config, mean_read_times, std_read_times, ...`. Compounding it, `benchmark_dataframe_parquet_compression` names its columns in the plural (`mean_read_times`, :149) while the final frame uses the singular (`mean_read_time`, :278) and the default `sort_by="mean_write_size"`, so even a correct dict-to-row alignment would not have matched.
- **Failure scenario**: One feather config plus one parquet config, reproduced with the exact constructor from :277-281:
  ```
      config   mean_read_time   std_read_time  ...  mean_write_size   std_write_size
  0  feather-x           1.05            0.05  ...              4.0              0.0
  1     config  mean_read_times  std_read_times ...  mean_write_sizes  std_write_sizes
  ```
  and the very next statement, `.set_index("config").sort_values("mean_write_size")`, raises:
  ```
  UFuncTypeError: ufunc 'greater' did not contain a loop with signature matching types (<class 'numpy.dtypes.Float64DType'>, <class 'numpy.dtypes.StrDType'>) -> None
  ```
  So any call to `benchmark_dataframe_compression` on a machine where pyarrow/fastparquet is installed (i.e. where the parquet sweep succeeds) dies with an unrelated-looking numpy error, and if it did not die the parquet numbers would all be strings. The parquet `try/except` at :268-272 does not cover the crash, which happens after it.
- **Suggested fix**: Convert the parquet results to the same list-of-lists shape before extending — e.g. `res.extend(parquet_results.itertuples(index=False, name=None))` — and rename `benchmark_dataframe_parquet_compression`'s columns to the singular set used at :278 (or build the final frame with `pd.concat` of two properly-named frames). Add a test that runs the sweep with only feather+parquet and asserts every cell of the result is numeric.

### F05. [High] `optimize_dtypes` silently converts identifier-like string columns to integers, destroying leading zeros — src/pyutilz/data/pandaslib/dtypes.py:128

- **Disposition**: OPEN
- **Category**: dtype-coercion
- **Problem**: For every object/`str`/`string` column the function first attempts `df[col].astype(np.int64)` (:128) and accepts it if `candidate.astype(np.float64) == df[col].astype(np.float64)` holds (:131). That guard was added to reject fractional truncation and does exactly that, but it compares **numerically**, so any purely-textual information that is not numeric — most importantly a leading zero — round-trips as equal and the cast is accepted. Zip codes, account/customer numbers, phone numbers, ISBN/EAN codes and zero-padded ids are all silently rewritten.
- **Failure scenario**:
  ```
  df = pd.DataFrame({"zip": ["01234", "00501", "90210"]})   # StringDtype under pandas 3
  optimize_dtypes(df.copy(), inplace=False)
  ```
  Observed:
  ```
  {'zip': dtype('uint32')}  [1234, 501, 90210]
  ```
  `"01234"` became `1234` and `"00501"` became `501`. The same happens with `inplace=True` and with an explicitly `dtype=object` column. A join against another table still holding the string keys then matches zero rows, and the memory-optimization pass that was supposed to be lossless has changed the data.
- **Suggested fix**: Make the acceptance guard textual rather than numeric — require `candidate.astype(str).equals(df[col].astype(str))` (or, cheaper, reject up front any column where a non-null value has a leading `"0"` followed by another digit, or leading/trailing whitespace, or a `"+"` sign) before accepting the int64 cast. Consider gating the whole object->numeric probe behind an explicit opt-in parameter, since "compress dtypes" is not obviously licence to reinterpret strings as numbers.

### F06. [Medium] `bin_numerical_columns` raises `InvalidOperationError` on any Decimal column, because `cs.numeric()` selects Decimal but `.fill_nan()` does not support it — src/pyutilz/data/polarslib/binning.py:327

- **Disposition**: OPEN
- **Category**: dtype-support
- **Problem**: `cs.numeric()` in polars 1.44 includes `Decimal` (verified: `cs.expand_selector(df, cs.numeric()) -> ('d', 'f')` for a `Decimal(12,2)` + `Float64` frame), but `cs.float()` does not (`-> ('f',)`). So a Decimal column is selected for binning and reaches the unconditional `.fill_nan(0)` on the binning expression at :327 while never being routed through the float-only `clean_numeric` guard at :324-325. `fill_nan` is not implemented for Decimal.
- **Failure scenario**:
  ```
  df = pl.DataFrame({"d": [Decimal("1.5"), Decimal("2.5"), Decimal("300.5"), Decimal("1.75")], "t": [0,1,0,1]},
                    schema={"d": pl.Decimal(12,2), "t": pl.Int64})
  bin_numerical_columns(df, target_columns=["t"], num_bins=4)
  ```
  Observed:
  ```
  InvalidOperationError: `is_not_nan` operation not supported for dtype `decimal[38,2]`
  ...col("d").is_not_nan()...clip([dyn int: 0, dyn int: 3]).strict_cast(Int8)
  ```
  Decimal columns are the normal result of reading a NUMERIC/DECIMAL SQL column via `pl.read_database`, so an otherwise ordinary frame from a database aborts the whole binning/MI pipeline.
- **Suggested fix**: Either cast Decimal to Float64 up front (`df.with_columns(cs.by_dtype(pl.Decimal).cast(pl.Float64))` before the stats pass) or apply `.fill_nan(0)` only to columns in `cols_with_floats` (Decimal cannot hold NaN, so the call is a no-op there anyway).

### F07. [Medium] `find_nan_cols` / `find_infinite_cols` raise `InvalidOperationError` on Decimal columns — src/pyutilz/data/polarslib/columns.py:51

- **Disposition**: OPEN
- **Category**: dtype-support
- **Problem**: Same `cs.numeric()`-includes-Decimal mismatch as F06, on a different pair of functions: `find_nan_cols` applies `cs.numeric().is_nan()` (:51) and `find_infinite_cols` applies `cs.numeric().is_infinite()` (:56); neither predicate is defined for Decimal.
- **Failure scenario**:
  ```
  df = pl.DataFrame({"d": [Decimal("1.5"), Decimal("2.5")], "f": [1.0, 2.0]},
                    schema={"d": pl.Decimal(12,2), "f": pl.Float64})
  ```
  Observed:
  ```
  find_nan_cols      RAISED InvalidOperationError `is_nan` operation not supported for dtype `decimal[12,2]`
  find_infinite_cols RAISED InvalidOperationError `is_infinite` operation not supported for dtype `decimal[12,2]`
  ```
  A data-quality screen that is supposed to *report* problems instead crashes on the frame it was pointed at, and it crashes for a dtype that can by construction hold neither NaN nor infinity.
- **Suggested fix**: Narrow the NaN/infinity predicates to `cs.float()` and keep the null check on `cs.numeric()` — `(cs.float().is_nan().fill_null(False).any()) | (cs.numeric().is_null().any())` for `find_nan_cols`, and `cs.float().is_infinite().any()` for `find_infinite_cols`. The `polars_castable_int_dtypes` helper next door is the established place to record such dtype-set decisions.

### F08. [Medium] `drop_constant_columns`, `find_nan_cols` and `find_infinite_cols` crash with a raw `OutOfBoundsError` on a frame that has no numeric columns — src/pyutilz/data/polarslib/frames.py:44

- **Disposition**: OPEN
- **Category**: empty-edge-case
- **Problem**: All three call `.row(0)` on the result of a selector-based aggregate: `drop_constant_columns` at `frames.py:44` (`df.lazy().select(stats_expr).collect().row(0, named=True)`) and `_cols_matching` at `columns.py:40` (`meta.row(0)`), which backs both `find_nan_cols` and `find_infinite_cols`. When the selector expands to nothing the collected frame has 0 columns AND 0 rows, so `.row(0)` raises. The sibling `bin_numerical_columns` was given exactly this guard on 2026-09-02 (see `binning.py:290-295`); these three were not.
- **Failure scenario**:
  ```
  df = pl.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
  ```
  Observed (polars 1.44.1):
  ```
  drop_constant_columns RAISED OutOfBoundsError index 0 is out of bounds for sequence of length 0
  find_nan_cols         RAISED OutOfBoundsError index 0 is out of bounds for sequence of length 0
  find_infinite_cols    RAISED OutOfBoundsError index 0 is out of bounds for sequence of length 0
  ```
  An all-categorical / all-string frame (a raw feature table before encoding, a text-only staging frame) is a completely ordinary input, and the caller gets a polars-internal indexing error it cannot distinguish from a polars bug. The correct answers are trivially "return `df` unchanged" and "return an empty frame".
- **Suggested fix**: Guard on the collected frame's width in both places, mirroring `binning.py:290-295`: in `drop_constant_columns`, `stats_df = ...collect(); if stats_df.width == 0: return df` before `.row(0)`; in `_cols_matching`, `if meta.width == 0: return df.select([])`. Add a test parametrized over the three functions with the all-string frame above.

### F09. [Medium] `bin_numerical_columns` still folds NaN into bin 0 when `fill_nans=False`, so the flag does not do what it says — src/pyutilz/data/polarslib/binning.py:327

- **Disposition**: OPEN
- **Category**: nan-handling
- **Problem**: `fill_nans=False` skips the `clean_numeric(col_expr, nans_filler=min_val)` wrapper at :324-325, but the binning expression itself ends with an unconditional `.fill_nan(0)` at :327. Bin 0 is also where the column's true minimum lands, so a NaN and the smallest real observation become indistinguishable. The parallel `fill_nulls=False` path behaves correctly (nulls survive as null), which makes the asymmetry a genuine surprise rather than a documented convention.
- **Failure scenario**:
  ```
  df = pl.DataFrame({"a": [nan, 1.0, 2.0, 3.0, 10.0, None], "t": [0,1,0,1,0,1]})
  bin_numerical_columns(df, target_columns=["t"], num_bins=5,
                        fill_nans=False, fill_nulls=False, clean_features=False, clean_targets=False)
  ```
  Observed:
  ```
  fill_nans=False, fill_nulls=False -> [0, 0, 0, 1, 4, None]
  fill both True                    -> [0, 0, 0, 1, 4, 0]
  ```
  The two runs are identical for the NaN row (both `0`) despite opposite flags, and the NaN shares bin 0 with the genuine minimum `1.0`. Downstream `entropy_for_column` / `mi_for_column` then treat "missing" as "smallest observed value", inflating the apparent informativeness of any column whose NaNs correlate with the target.
- **Suggested fix**: Make the `.fill_nan(0)` conditional on `fill_nans` (and, when it does fire, fill with the same bin the `nans_filler=min_val` path produces rather than a hardcoded `0`, so the two paths agree). When `fill_nans=False`, leave NaN to propagate — the subsequent `.cast(bin_dtype)` then needs `strict=False` or a preceding `.fill_nan(None)` to turn it into a null, matching what the `fill_nulls=False` path already yields.

### F10. [Low] `showcase_df_columns` writes the value-count table to stdout even with `use_print=False` — src/pyutilz/data/pandaslib/frames.py:262

- **Disposition**: OPEN
- **Category**: output-contract
- **Problem**: `use_print` gates only the one-line dtype header (polars :245, pandas :299). The actual `print(stats)` / `print(stats.head(max_vars))` / `print("")` calls that emit the distribution table (polars :263, :267, :272, :274; pandas :305, :307, :309) are unconditional. With IPython installed the header goes through `display(Markdown(...))` and is correctly suppressed, so the caller ends up with the table but no header.
- **Failure scenario**:
  ```
  d2 = pd.DataFrame({"x": ["a"]*100 + ["b"]*2 + [None]*50})
  showcase_df_columns(d2, use_markdown=False, use_print=False, max_unique_percent=0.05, dropna=True)
  ```
  Observed on stdout despite `use_print=False` (one such block from the pandas branch and one from the polars branch on the equivalent `pl.DataFrame`):
  ```
  x
  a    100
  b      2
  Name: count, dtype: int64
  ```
  A caller using this purely for its `(rare_categories, uninformative_features)` return value — e.g. inside a feature-selection loop over hundreds of columns — floods the log it explicitly asked to stay quiet.
- **Suggested fix**: Wrap each of the eight `print(...)` calls listed above in the same `if use_print or not _facade.HAS_IPYTHON:` condition that already guards the header, or compute a `should_print` boolean once per column and reuse it. The doctests all pass `use_print=True`, so they stay green.

### F11. [Low] `concat_and_flush_df_list` discards the index of every input frame without saying so — src/pyutilz/data/pandaslib/io_ops.py:65

- **Disposition**: OPEN
- **Category**: index-alignment
- **Problem**: The concat is hardcoded to `pd.concat(lst, axis=0, ignore_index=True)`, which throws away whatever index the caller's frames carried and replaces it with a fresh `RangeIndex`. The docstring (:57-62) describes concatenation, writing and the `set_index` option but never mentions that an existing index is dropped; `set_index` can only *promote a column*, so an index that was not also a column is unrecoverable after the call. `read_stats_from_multiple_files` (:159) is the in-repo caller, so files whose pickles carry a meaningful index (a timestamp index, an entity id) lose it during the merge.
- **Failure scenario**: `concat_and_flush_df_list([pd.DataFrame({"v":[1,2]}, index=["a","b"])], "out")` returns a frame indexed `0,1`; the labels `"a"`,`"b"` are gone from both the return value and the written pickle, and `set_index=` cannot bring them back because they were never a column.
- **Suggested fix**: Expose the behavior as a parameter (`ignore_index: bool = True`, preserving today's default) and state it in the docstring; alternatively, in `read_stats_from_multiple_files`, reset the index into a column before appending each frame to `lst` when the index is not a plain `RangeIndex`.

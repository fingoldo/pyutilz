# Performance & Algorithmic Complexity Audit

## Summary

I read in full: `src/pyutilz/data/pandaslib/{frames,dtypes,io_ops,benchmarks,_common,__init__}.py`, `src/pyutilz/data/polarslib.py`, `src/pyutilz/data/{numpylib,numbalib}.py`, `src/pyutilz/text/similarity.py`, `src/pyutilz/text/tokenizers.py`, `src/pyutilz/text/humanizer.py`, `src/pyutilz/text/strings/{basics,webtext,textentropy,jsonutils,configfiles}.py`, `src/pyutilz/database/db/{__init__,sqlite,sql_helpers,upsert}.py`, and `src/pyutilz/performance/kernel_tuning/{registry,benchmark}.py` + `cache/{cache_class,region_matching}.py`. Every finding below was empirically measured with a standalone timing script (shown inline) rather than asserted from reading alone.

Three genuinely new bug-class instances were found and verified: an unbounded, per-word **O(word_len³)** hot loop in the morpheme tokenizer with no length cap (reachable on ordinary scraped-web text, not just adversarial input), and two instances of **redundant full-column-scan recomputation** of an already-available value inside `pandaslib.frames.showcase_df_columns` (one in the pandas branch, one in the polars branch). Totals: 1 HIGH, 2 MEDIUM.

Most of the explicitly-named hot-path files (`polarslib.py`, `text/similarity.py`, `database/db/*.py`, `kernel_tuning/*`) were already heavily optimized in round 1 — they carry extensive inline benchmark tables, dispatcher thresholds, and explicit complexity-analysis comments (see "Things done well"), so this pass mostly confirmed those are sound rather than finding new issues there.

## Findings

### [HIGH] Unbounded per-word cubic-time morpheme extraction, no length cap — src/pyutilz/text/tokenizers.py:130-178

- **Category**: algorithmic-complexity / unbounded-cost-per-input
- **Problem**: `AdvancedTokenizer.tokenize()` iterates every substring of every word to build morpheme statistics:
  ```python
  for w, word in enumerate(words):
      word_len = len(word)
      for i in range(word_len):
          for j in range(1, word_len - i + 1):
              morpheme = word[i : i + j]
              ...
              self.NUM_OCCS[base_morpheme] += 1
              ...
  ```
  For a word of length `L` this is `O(L²)` substring positions, and slicing `word[i:i+j]` is itself `O(j)`, so the total work for one word is `O(L³)`. There is no cap on `L` and no gate on word frequency — the docstring at line 112 claims "for every genuine word with number of occurrences>1, count its consecutive symbols of length MIN_MORPHEME_LENGTH=2 to MAX_MORPHEME_LENGTH", but the actual code processes **every** word regardless of occurrence count, starting at `j=1` (single characters), with no `MIN_MORPHEME_LENGTH`/`MAX_MORPHEME_LENGTH` bound anywhere in the method. `tokenize_db_reviews()` (the production entry point, streaming arbitrary DB rows through `fix_html`/`fix_duplicate_tokens`/etc., per its own example SQL comment against `amazon_users_reviews`) feeds arbitrary web-scraped text straight into this with no word-length cap upstream either.
- **Failure scenario**: A single pathological "word" — a long URL, a base64/JS blob that survives `fix_html`, or any run of scraped text with no whitespace — reaches `tokenize()`. Measured directly against the real (uninstrumented) `AdvancedTokenizer.tokenize()` with nltk's real tokenizer, timing one synthetic unbroken token of length `n`:
  ```
  n=100   0.0202s
  n=200   0.0666s   (3.3x)
  n=400   0.3493s   (5.2x)
  n=800   1.3598s   (3.9x)
  ```
  This is clearly super-quadratic (doubling `n` costs far more than 4x), consistent with the `O(L³)` shape confirmed separately via a proxy micro-benchmark of just the `i,j` loop + slicing cost (50→800 chars: 0.0001s → 0.0378s, an ~8x-per-doubling / cubic curve). A single ~2000-4000 character garbled token (entirely plausible in a scraped-HTML corpus after normalization) would take tens of seconds to minutes to process, and `tokenize_db_reviews()` calls this once per DB row inside a `while True: chunk = cur.fetchmany(...)` streaming loop with no per-row timeout — one bad row stalls the whole ingestion job.
- **Suggested fix**: Cap per-word morpheme length (respect the docstring's own `MIN_MORPHEME_LENGTH`/`MAX_MORPHEME_LENGTH`, e.g. `j` bounded to a small constant like 2-8 instead of `word_len - i`), and/or skip words above a configurable max length (log + skip) before entering the `i,j` loop. This also fixes the described-vs-actual behavior mismatch noted above (the "occurrence > 1" gate is likewise absent from the code).

### [MEDIUM] Redundant full-column value_counts/nunique recomputation (pandas branch) — src/pyutilz/data/pandaslib/frames.py:274, 287, 289

- **Category**: redundant-recomputation-in-loop
- **Problem**: Inside the per-column loop of `showcase_df_columns`'s pandas branch:
  ```python
  stats = df[var].value_counts(dropna=dropna)          # line 274 — full column scan
  ...
  n_unique = df[var].nunique(dropna=dropna)             # line 287 — ANOTHER full scan; == len(stats)
  if n_unique <= max_cat_uniq_qty and len(stats) > 0:
      full_stats = df[var].value_counts(dropna=dropna) if max_vars is not None else stats   # line 289 — a THIRD full scan when max_vars is not None; identical to `stats`
  ```
  `stats` (line 274) already IS the complete `value_counts(dropna=dropna)` result — `.head(max_vars)` at lines 277-278 only affects what gets printed, it does not mutate or truncate `stats`. Consequently: (a) `len(stats)` already equals `nunique(dropna=dropna)` exactly (verified below), making line 287 a pure duplicate of information already on hand; (b) when `max_vars is not None`, line 289 recomputes the identical `value_counts` a second time instead of reusing `stats`.
- **Failure scenario**: `showcase_df_columns` is the library's dataframe-EDA/diagnostic entry point, normally called once per column across a whole production dataframe with `max_vars` set (its usual calling pattern, to avoid dumping huge value_counts to stdout). Measured directly against a 5,000,000-row int column:
  ```
  value_counts()  #1 (needed):                    0.0943s
  nunique()       (redundant, == len(vc1)):        0.0753s
  value_counts()  #2 (redundant when max_vars set): 0.0684s
  len(vc1) == nunique?  True
  ```
  So with `max_vars` set (the common case), the function does ~2.5x the necessary work per column (0.237s vs 0.094s needed) purely from re-scanning the same column two extra times for values it already computed. This compounds linearly across every column of a wide dataframe during routine EDA.
- **Suggested fix**: Replace line 287 with `n_unique = len(stats)`; replace line 289 with `full_stats = stats` unconditionally (drop the `max_vars is not None` branch entirely, since `stats` was never truncated).

### [MEDIUM] Redundant full-column n_unique scan duplicating value_counts height (polars branch) — src/pyutilz/data/polarslib... actually src/pyutilz/data/pandaslib/frames.py:208-266

- **Category**: redundant-recomputation-in-loop
- **Problem**: In the polars branch of `showcase_df_columns`, two independent sets of lazy queries are built and run over the same columns:
  ```python
  lazy_queries = [... .group_by(var).agg(pl.len().alias("count")).sort(...) for var in target_cols]   # line 210-216
  vc_results = pl.collect_all(lazy_queries)                                                              # line 219

  nuniq_queries = [... pl.col(var).n_unique() ... for var in target_cols]                                 # line 226-229
  nuniq_results = pl.collect_all(nuniq_queries)                                                            # line 230
  ...
  n_unique = nuniq_df.item(0, 0)                                                                           # line 256
  ```
  `vc` (the value_counts result) has one row per distinct value of the column under the *same* `dropna` treatment as `nuniq_queries` (both apply `.drop_nulls()` when `dropna=True`, both count null as one group otherwise). So `vc.height` is *exactly* `n_unique` for free — verified directly: `df.select(pl.col('x').n_unique()).item()` and `df.lazy().select('x').group_by('x').agg(pl.len()).collect().height` return the identical count (4 == 4) on a null-containing test column. The `nuniq_queries` pass is therefore a complete, unnecessary second full-column scan.
- **Failure scenario**: Measured the isolated cost of the two passes on an 8-column, 3,000,000-row polars frame (matching `showcase_df_columns`'s own realistic dropna=True code path):
  ```
  value_counts pass (needed):        0.8365s
  n_unique pass (redundant):          0.3204s   -- ~24% of total wall time, entirely avoidable
  ```
  `pl.collect_all` does run the two batches in parallel across polars' thread pool, so this shows up as extra CPU/core consumption and wall time rather than a doubling — but on a busy host (concurrent EDA sessions, CI, or a smaller machine) that's ~24% of the function's total cost paid for a value that was already sitting in `vc_results`.
- **Suggested fix**: Drop the `nuniq_queries`/`nuniq_results`/`pl.collect_all(nuniq_queries)` block entirely; replace `n_unique = nuniq_df.item(0, 0)` with `n_unique = vc.height`.

## Things done well

- `text/similarity.py`'s numba-accelerated `sentences_similarity` variants document their own complexity honestly (inline benchmark table for the sort-vs-scan greedy-matcher crossover, with the exact threshold `550*550` justified by measured numbers) rather than assuming an optimization always wins.
- `data/polarslib.py` and `data/pandaslib/dtypes.py` consistently batch per-column stats into single `.agg([...])`/`.select([...])` polars/pandas calls instead of looping calls per column (e.g. `optimize_dtypes`'s min/max-in-one-pass, `bin_numerical_columns`'s single `stats_expr` collect).
- `database/db/sqlite.py::insert_sqllite_data` and `pandaslib/io_ops.py::concat_and_flush_df_list` both follow the correct "accumulate into a list, do one bulk op" pattern (`executemany`, `pd.concat` once) rather than looping single-row inserts / repeated `concat`/`append` calls.
- `dtypes.py::classify_column_types` and `dtypes.py::get_columns_of_type` both carry explicit comments explaining a specific hoisted-out-of-loop micro-optimization (avoiding `df.dtypes[col]` rebuilding the whole dtypes Series per call; hoisting `str(type_name)`), suggesting real prior attention to this exact bug class.
- `performance/kernel_tuning/registry.py`'s `TunerSpec.choose()` memoizes per-dims dispatch decisions in `_choice_cache` (a genuinely hot per-fit-call path) and is careful to only memoize once "tuned" rather than caching a still-pending fallback forever.

## Investigated, not an issue

- `text/strings/basics.py::strip_doubled_characters` / `webtext.py::fix_duplicate_tokens` (`while token*2 in text: text = text.replace(token*2, token)`): looks like it could be quadratic on a long run of the same character, but each `str.replace` call halves the run length in one `O(n)` pass, so the loop only runs `O(log(run_length))` times — `O(n log n)` total, not quadratic. Verified by reasoning through CPython's non-overlapping left-to-right `str.replace` semantics.
- `text/humanizer.py::introduce_typos`/`_apply_char_typo`: rebuilds the word-offset list from scratch on every one of `count` typo injections (`O(count * n)`), but `count` is a small caller-controlled constant (default 1), not data-dependent, so this doesn't scale with input size in a concerning way.
- `performance/kernel_tuning/cache/cache_class.py`: per-kernel directories are actively garbage-collected to the newest 4 files (`_gc_kernel_dir`), so the `_glob.glob` + read-every-file pattern in `_load`/`_read_kernel_newest` stays bounded regardless of how many times a kernel gets re-tuned.
- `database/db/upsert.py::build_upsert_query`: builds a SQL string once per call (not per-row), so its several `for field in fields_names` loops are bounded by column count, not row count — no hot-path concern despite the visual density of the function.

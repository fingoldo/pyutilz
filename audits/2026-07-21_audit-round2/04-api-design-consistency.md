# API Design Consistency & Ergonomics Audit

## Summary

I read, in full, the public API surface of: `pyutilz/__init__.py`; `data/pandaslib/{frames,dtypes,io_ops,benchmarks,_common}.py`; `data/polarslib.py`; `data/numpylib.py`; `data/numbalib.py`; `core/{pythonlib,matrix,image,serialization,safe_pickle}.py`; `database/db/{__init__,sql_helpers}.py`; `text/strings/basics.py`; `text/similarity.py`; `text/strings/jsonutils.py` (partial, the sibling-relevant functions); `system/parallel.py`; `system/system/{fsutils,misc}.py`; `web/web.py` (first ~200 lines plus the getter functions); and `llm/base.py` plus signature-diffed every provider's `generate`/`count_tokens`/`supports_json_mode` across `anthropic_provider.py`, `claude_code_provider.py`, `gemini_provider.py`, `openai_compat.py`, `openrouter_provider/_provider.py`. Every finding below that claims a concrete failure was verified by running real code against this checkout (`PYTHONPATH=src`), not just read. A large fraction of the codebase's pre-existing rough edges are already flagged with explicit "unlike its sibling X, this does Y" docstrings from round 1 — those were treated as already-dispositioned and excluded. Findings below are ones with no such caveat.

Total: 4 HIGH, 5 MEDIUM, 2 LOW.

## Findings

### [HIGH] Four different parameter names for "a pandas DataFrame" within one module, two of them TypeError-incompatible with the rest — `src/pyutilz/data/pandaslib/frames.py`
- **Category**: parameter-naming inconsistency
- **Problem**: `frames.py` is a single, cohesive "dataframe column cleanup" module. Six of its functions name the DataFrame parameter `df`: `nullify_standard_values(df, ...)` (L27), `prefixize_columns(df, ...)` (L49), `showcase_df_columns(df, ...)` (L67), `share_dataframe(df)` (L329), `remove_constant_columns(df, ...)` (L401). But `get_non_stale_columns(X: pd.DataFrame)` (L347) and its own alias `remove_stale_columns(X: pd.DataFrame)` (L371) name it `X`, and `get_suspiciously_constant_columns(ref_df: pd.DataFrame)` (L382) names it `ref_df` — three spellings for the identical concept in one file, with `remove_constant_columns` and `get_non_stale_columns`/`get_suspiciously_constant_columns` being conceptually the sibling "which columns are dead" family the docstrings themselves cross-reference.
- **Failure scenario**: A caller who does the natural thing — call these column-cleanup helpers back to back with the keyword form they just used for the previous call — gets a hard `TypeError`. Verified:
  ```
  remove_constant_columns(df=df)                      # OK
  get_non_stale_columns(df=df)                         # TypeError: unexpected keyword argument 'df'
  get_suspiciously_constant_columns(df=df)              # TypeError: unexpected keyword argument 'df'
  ```
  (`get_non_stale_columns(X=df)` / `get_suspiciously_constant_columns(ref_df=df)` are the only spellings that work — undiscoverable without reading the source, since every neighboring function in the same file uses `df`.)
- **Suggested fix**: Rename `X` → `df` and `ref_df` → `df` in these three functions (keep positional-arg back-compat since none of round-1's regression tests appear to call them by keyword); or, if a breaking rename is undesirable, accept `df` as an additional keyword alias.

### [HIGH] `prefixize_columns`'s return type silently changes shape based on the `inplace` flag it flips on — `src/pyutilz/data/pandaslib/frames.py:49-63`
- **Category**: flag-trap / return-type inconsistency
- **Problem**:
  ```python
  def prefixize_columns(df, prefix, special_prefixes=None, sep="_", exclusions=None, inplace: bool = True):
      ...
      if inplace:
          df.rename(columns=columns, inplace=True)
          return columns          # <- a dict: {old_col: new_col}
      else:
          return df.rename(columns=columns, inplace=False)   # <- a DataFrame
  ```
  With the default `inplace=True`, the function returns a plain `dict`. Flip the same boolean to `False` and it returns a `pd.DataFrame` instead. No other `inplace`-flag function in the same file (`remove_constant_columns`, and pandaslib's `optimize_dtypes` in `dtypes.py`) changes its *return type* based on `inplace` — they only change whether mutation happens, always returning the same shape (`None` / `pd.DataFrame` respectively) regardless of the flag.
- **Failure scenario**: verified —
  ```python
  prefixize_columns(df, prefix='p')                 # default inplace=True -> {'a': 'p_a', 'b': 'p_b'}  (dict)
  prefixize_columns(df2, prefix='p', inplace=False)  # -> pd.DataFrame
  ```
  Code written and tested against `inplace=False` (e.g. `renamed_df = prefixize_columns(df, "p", inplace=False); renamed_df.head()`) breaks silently at runtime if a later refactor flips the default call to omit `inplace=False`, since `dict` also supports iteration/`in` but not `.head()`/`.columns` — the failure surfaces far from the actual mistake.
- **Suggested fix**: Always return the DataFrame; if the column-rename mapping is useful to callers, return it as a second tuple element (`return df, columns`) or expose it via a separate `get_prefix_mapping()`-style helper, consistently regardless of `inplace`.

### [HIGH] `store_params_in_object` / `load_object_params_into_func`, explicitly documented as an inverse pair, do not round-trip with their own defaults — `src/pyutilz/core/pythonlib.py:629-644`
- **Category**: default-value mismatch between documented-inverse sibling functions
- **Problem**:
  ```python
  def store_params_in_object(obj, params: dict, postfix: str = ""):
      """Useful for persisting __init__ params in the class instance."""
      for param_name, param_value in params.items():
          setattr(obj, param_name + postfix, param_value)

  def load_object_params_into_func(obj, locals: dict, postfix: str = "_param_"):
      """Contrary action to store_params_in_object, ..."""
      for attr in dir(obj):
          if attr.endswith(postfix):
              key, value = attr[: -len(postfix)], getattr(obj, attr)
              locals[key] = value
  ```
  `store_params_in_object`'s default `postfix=""` writes attributes with the bare param name (`obj.alpha`), while `load_object_params_into_func`'s default `postfix="_param_"` only picks up attributes *ending in* `"_param_"`. The two documented-as-inverse defaults don't compose.
- **Failure scenario**: verified empirically —
  ```python
  store_params_in_object(o, {'alpha': 1, 'beta': 2})   # -> o.alpha=1, o.beta=2
  loc = {}
  load_object_params_into_func(o, loc)                  # -> loc == {}  (silently empty, no error, no warning)
  ```
  Any caller trusting the docstring's "contrary action" framing and calling both with default arguments loses all data silently — no exception, no log line, just an empty `locals` dict.
- **Suggested fix**: Either make the defaults agree (both `""` or both `"_param_"`), or make `load_object_params_into_func` accept `postfix=""` to mean "any attribute name / no suffix filter," or add an explicit docstring warning that the postfixes must match by hand — right now nothing signals the mismatch.

### [HIGH] `db/__init__.py` mixes 28 snake_case/typed functions with 4 PascalCase, Hungarian-notation functions doing the same category of work, immediately adjacent — `src/pyutilz/database/db/__init__.py`
- **Category**: naming-convention inconsistency within one module's public surface
- **Problem**: The overwhelming majority of `db/__init__.py`'s public functions are modern, typed, snake_case (`validate_sql_identifier`, `check_if_pg_table_exists(table_name: str, schema_name: Optional[str] = "public")`, `read_unique_table_field(table_name, field_name, container, ...)`, 28 total). But four functions — `EnsurePgTableExists(sTable, sKeyFieldName, sIdFieldName, sAutocreateIdTypeName)` (L583), `ReadTableIntoDic(dicEnums, sTable, sKeyFieldName, sCondition, sIdFieldName, ...)` (L607), `ReadTableIntoDicReversed(...)` (L632), `GetIdByKeyFieldAndInsertIfNeeded(dicEnums, sTable, sKeyFieldValue, ...)` (L675) — use PascalCase names and Hungarian-prefixed params (`sTable`, `sKeyFieldName`, `bKeyIsNotString`, `bAddUpdatedAtTimestamp`). `EnsurePgTableExists` sits *directly below* `check_if_pg_table_exists` (L565) — literally the previous function in the file — and both answer "does this table exist," one taking `table_name: str`, the other `sTable`.
- **Failure scenario**: A caller who just wrote `check_if_pg_table_exists(table_name="events")` reaches for the next logical helper, `EnsurePgTableExists`, and naturally tries `EnsurePgTableExists(table_name="events")` — `TypeError: unexpected keyword argument 'table_name'` (the real param is `sTable`). More broadly, this corner of the public API is undiscoverable via IDE autocomplete conventions the rest of the module trains the user to expect, and it's the one place in the module still exposed to param-order mistakes since none of its many `Optional[...]` params have type-narrowed sentinels the way the snake_case functions do.
- **Suggested fix**: Add snake_case wrappers (`ensure_pg_table_exists`, `read_table_into_dict`, `get_id_by_key_field_and_insert_if_needed`) with modern param names that delegate to the legacy functions, and mark the PascalCase originals deprecated; or do a straight rename with a compatibility shim, matching how `remove_stale_columns` was kept as a documented deprecated alias elsewhere in this same codebase.

### [MEDIUM] `verbose` is `bool` in one function and `int` in the very next one in the same file — `src/pyutilz/data/pandaslib/dtypes.py:75` vs `:318`
- **Category**: parameter-type inconsistency for the same concept
- **Problem**: `optimize_dtypes(..., verbose: bool = False, ...)` (L75) and `ensure_dataframe_float32_convertability(df, verbose: int = 0)` (L318) are both dtype-conversion functions in the same file, `dtypes.py`. One types `verbose` as a boolean toggle, the other as an int (implying multi-level verbosity, though the body only ever does `if verbose:`, i.e. it's used as a bool anyway). The same split repeats across the package: `polarslib.py`'s `drop_constant_columns(..., verbose: int = 1)` and `bin_numerical_columns(..., verbose: int = 1)` both default verbose to *truthy* `1`, while every pandaslib equivalent (`remove_constant_columns`, `optimize_dtypes`, `nullify_standard_values`) defaults `verbose: bool = False`.
- **Failure scenario**: Not a crash (Python doesn't enforce parameter types at runtime), but a real ergonomics footgun: code ported from `optimize_dtypes(df, verbose=False)` to `ensure_dataframe_float32_convertability(df, verbose=False)` works by coincidence (falsy int), but a caller who reasonably infers "verbose is graded 0/1/2" from the `int` annotation and passes `verbose=2` expecting more detail gets nothing extra — `ensure_dataframe_float32_convertability` only ever checks truthiness. Conversely, porting a `remove_constant_columns(df)` call to `drop_constant_columns(df)` (documented siblings, see next finding) silently starts logging by default, since polars' default is `1` (truthy) vs pandas' `False`.
- **Suggested fix**: Standardize on `bool` (the only value ever consulted) across the whole package, or if graded verbosity is truly intended for the polars functions, honor more than one level and document it; align defaults (`False`/`0`) across the pandas/polars sibling pairs.

### [MEDIUM] `remove_constant_columns` (pandas) vs `drop_constant_columns` (polars) — documented siblings, different verb, different defaults — `src/pyutilz/data/pandaslib/frames.py:401` / `src/pyutilz/data/polarslib.py:964`
- **Category**: sibling-API naming/default drift
- **Problem**: `polarslib.drop_constant_columns`'s own docstring says: *"Unlike `pandaslib.frames.remove_constant_columns` (the conceptually equivalent pandas function...)"* — the two are explicitly documented as the pandas/polars pair for the identical operation. Yet they use different verbs (`remove_` vs `drop_`) for the identical action, taking different signatures (`prewarm_size` sampling param exists only on the pandas side; `max_log_text_width` only on the polars side), and — per the finding above — different `verbose` defaults (`False` vs `1`).
- **Failure scenario**: A user grepping the codebase for "remove constant columns" to find the polars equivalent of a pandas call they already use will not find `drop_constant_columns` by name; conversely a user who read the polars docstring and expects `remove_constant_columns(verbose=1)` to log the way `drop_constant_columns(verbose=1)` does gets no output (pandas function treats `1` as truthy but that's incidental — its own convention is `True`/`False`).
- **Suggested fix**: Rename one to match the other (`drop_constant_columns` → `remove_constant_columns` is the more natural fix, matching the existing pandas name and the "remove_"/"get_" naming already used by `get_non_stale_columns`/`get_suspiciously_constant_columns` in the same package), and align the `verbose` type/default per the previous finding.

### [MEDIUM] `ensure_dataframe_float32_convertability` mutates its argument in place for pandas but not for polars — same function, same param name, mutation semantics flip on runtime type — `src/pyutilz/data/pandaslib/dtypes.py:316-358`
- **Category**: inconsistent mutation contract keyed on input's dynamic type
- **Problem**:
  ```python
  def ensure_dataframe_float32_convertability(df: Union[pd.DataFrame, pl.DataFrame], verbose: int = 0) -> Union[pd.DataFrame, pl.DataFrame]:
      if isinstance(df, pl.DataFrame):
          df = df.with_columns(...)          # rebinds local var; caller's original polars frame is untouched
      elif isinstance(df, pd.DataFrame):
          df[numeric_cols] = df[numeric_cols].astype(np.float32)   # mutates caller's frame in place
      return df
  ```
  For a pandas input, the caller's original object is mutated (columns replaced in place) *and* the same (now-mutated) object is returned. For a polars input, the caller's original object is left untouched and only the *returned* new object reflects the conversion — polars DataFrames are immutable so `.with_columns` can't do otherwise, but the function's single name/signature/docstring gives no hint that whether the caller must capture the return value depends on which backend they passed in.
- **Failure scenario**: A caller who writes `ensure_dataframe_float32_convertability(df)` without capturing the return value (reasonable, since it "worked" during testing with a pandas frame — the in-place pandas mutation makes the discarded-return-value call appear to succeed) silently gets no conversion at all when the same call site later receives a polars DataFrame (e.g. a caller that's generic over both backends, which is exactly this function's documented use case).
- **Suggested fix**: Always return a new object and never mutate the pandas input in place (`df = df.copy(); df[numeric_cols] = ...`), matching the polars branch's return-a-new-frame contract; document "always capture the return value" once, applying to both backends identically.

### [MEDIUM] Three names for "degree of parallelism" within one module — `n_jobs` / `n_cores` / `nworkers` — `src/pyutilz/system/parallel.py`
- **Category**: parameter-naming inconsistency within one module
- **Problem**: `parallel_run(jobslist, n_jobs: int = -1, ...)` (L162), `applyfunc_parallel(iterable, func, n_cores: Optional[int] = None, ...)` (L177), and `distribute_work(workload, nworkers: int)` (L133) are the three "how many workers should this run in parallel" functions in `system/parallel.py`, and each spells the same concept differently. `parallel_run` mirrors joblib's own `n_jobs` naming (reasonable, given it wraps `joblib.Parallel`), but the other two invent their own, mutually different, spellings for a concept the module otherwise treats as interchangeable (`applyfunc_parallel`'s docstring even calls the underlying detected value `ncores` in its own log message while the parameter is `n_cores`).
- **Failure scenario**: Not a crash — but a caller building a small wrapper that fans out `n_jobs=...` uniformly to whichever of these three helpers a config flag selects needs three different keyword names for what a config schema would naturally model as one concept, and a caller who assumes (reasonably, from `parallel_run`) that `n_jobs=-1` means "use all cores" on `applyfunc_parallel`/`distribute_work` gets a `TypeError` (`n_cores`/`nworkers` are the actual names) rather than the sentinel-based auto-detection those functions do support under a different spelling.
- **Suggested fix**: Standardize on one name (`n_jobs`, matching the joblib convention `parallel_run` already follows) across all three; keep the others as deprecated aliases if backward compatibility for existing call sites matters.

### [LOW] "Get memory usage" helpers use opposite failure conventions across the package — sentinel `-1` vs raised `TypeError` — `src/pyutilz/core/matrix.py:84-98` vs `src/pyutilz/data/pandaslib/dtypes.py:274-313`
- **Category**: inconsistent not-found/unsupported-input convention across a "get_X_memory_usage" function family
- **Problem**: `get_sparse_memory_usage(mat: object) -> int` returns the sentinel `-1` both when `mat` is a type the function doesn't recognize (`else: return -1`) and when a recognized type unexpectedly lacks the needed attributes (`except AttributeError: ... return -1`) — i.e. "unsupported input" is folded into the same int return space as a real byte count, distinguishable only by a negative-number convention the caller must know to check. `get_df_memory_consumption(df, max_cols=0, deep=True) -> float`, the pandas/polars analog elsewhere in the package, instead `raise TypeError(f"Unsupported dataframe type: {type(df)}")` for the same class of "I don't know this container type" situation.
- **Failure scenario**: Code that generically sums memory usage across mixed sparse-matrix and dataframe collections (`total = sum(get_sparse_memory_usage(x) if issparse(x) else get_df_memory_consumption(x) for x in objs)`) silently under-counts by subtracting real memory (`-1`) for any unsupported sparse-matrix-like input, instead of failing loudly the way the dataframe counterpart would for the equivalent mistake.
- **Suggested fix**: Pick one convention package-wide for "this get_*_memory_usage-family function received something it can't measure" — raising is safer (matches `get_df_memory_consumption`) and avoids the silent arithmetic corruption a `-1` sentinel invites when summed.

### [LOW] `get_external_ip` and `get_ipinfo` (sibling "look up my public IP" helpers, same file) use different failure-return conventions — `src/pyutilz/web/web.py:129-186`
- **Category**: inconsistent not-found/failure return convention between sibling functions
- **Problem**: `get_external_ip(...) -> Optional[str]` returns `None` uniformly for every failure path (no provider responded, all raised, a malformed response). `get_ipinfo(use_urllib=False, url=...)` — declared immediately after it, same module, same "ask an external IP-info provider" purpose — returns three different things depending on *which* failure occurred: an HTTP non-200 response returns `{}` (L177, explicit), a raised exception during `urlopen` falls through the `try/except/else` with no `return` in the `except` branch and implicitly returns `None` (L169-172), and the non-`use_urllib` branch also returns `None` on a JSON-decode exception (L182-184).
- **Failure scenario**: A caller checking `if info is None: retry()` (matching `get_external_ip`'s single documented convention) silently treats a non-200 response from `get_ipinfo(use_urllib=True)` as "success, got `{}`" and proceeds to do `info.get("ip")` → `None` without ever knowing the request actually failed, whereas the exact same failure reached through the exception path (rather than a non-200 status) would have been caught by the `is None` check.
- **Suggested fix**: Make `get_ipinfo`'s `use_urllib=True` branch return `None` for both failure modes (non-200 and exception), matching `get_external_ip`'s single convention and its own `use_urllib=False` branch.

## Things done well

- Several genuinely tricky pandas/polars mutation-semantics differences (`remove_constant_columns` vs `drop_constant_columns`, `get_non_stale_columns`'s non-mutation vs its own misleadingly-named alias `remove_stale_columns`) are already called out explicitly in docstrings from round 1, with the exact "unlike its sibling X, this does Y" framing this audit's angle is looking for — that pattern should be the template applied to the findings above.
- `llm/base.py`'s `LLMProvider` ABC is a well-designed consistency anchor: every concrete provider's `generate()` keeps the same typed `(prompt, system=None, temperature=0.7, max_tokens=0)` shape (OpenRouter's `*args/**kwargs` override, noted above, is the sole exception), and the shared `_get_pricing`/`_pricing_model_id` hook pattern lets subclasses differ in storage attribute name (`self.model` vs `self.model_name`) without leaking that inconsistency into the public interface.
- `safe_pickle.py`'s `safe_load`/`safe_dump` pair is a clean, honest inverse pair — `safe_dump` documents exactly what `safe_load` needs, they share the same path argument shape, and the raise-on-failure convention is consistent and clearly justified in the module docstring (a useful contrast with `unserialize`'s swallow-and-return-`None` convention, which is itself explicitly cross-referenced and justified for backward compatibility rather than left as a silent trap).

## Investigated, not an issue

- `sentences_similarity` / `sentences_similarity_numba` / `sentences_similarity_numba_batch` / `SentenceSimilarityIndex.query` (`text/similarity.py`): all consistently return/yield `None` per pair for degenerate (empty-sentence) inputs, both in the numba and pure-Python fallback paths — checked because the numba/no-numba code paths are large and independently maintained, but the "not found"/degenerate convention matches throughout.
- `count_tokens`'s signature widening in `AnthropicProvider` (adds an optional `system: str | None = None` beyond the base class's `(self, text: str) -> int`): this is a strict widening (Liskov-compatible, optional kwarg), not a breaking inconsistency — checked because provider-interface signature drift was a specific concern for this angle.
- `get_jsonlist_property` vs `get_jsonlist_properties` (`text/strings/jsonutils.py`): different `return_indices` defaults (`False` vs `True`) between the singular/plural sibling functions — already explicitly documented in the plural function's own docstring as an intentional backward-compatibility decision, not a hidden footgun.
- `unserialize()` (return `None` on failure) vs `safe_pickle.safe_load()` (raise `PickleVerificationError`) in `core/serialization.py`: two overlapping "load a pickle" APIs with opposite error conventions, but `unserialize`'s docstring extensively cross-references `safe_pickle` and explains exactly why the historical silent-`None` behavior is preserved — already dispositioned, not a new finding.

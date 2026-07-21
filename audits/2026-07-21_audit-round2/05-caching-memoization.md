# Caching & Memoization Correctness Audit

## Summary

I read in full: `llm/factory.py`, `llm/config.py`, `llm/claude_code_provider.py`, `llm/openrouter_provider/__init__.py`, `llm/openrouter_provider/_catalogue.py`, `llm/openrouter_provider/_health.py`, `llm/openrouter_provider/_provider.py`, `llm/token_counter.py`, `performance/kernel_tuning/cache/cache_base.py`, `performance/kernel_tuning/cache/cache_class.py`, `performance/kernel_tuning/cache/region_matching.py`, `performance/kernel_tuning/cache/cache_hooks.py`, `performance/kernel_tuning/code_versioning.py`, `performance/kernel_tuning/registry.py`, `performance/kernel_tuning/remote.py`, `performance/kernel_tuning/benchmark.py`, `system/gpu_dispatch.py`, `dev/code_audit/_base.py`, `dev/code_audit/redundant_test_fit.py`, `text/strings/webtext.py`, and `text/strings/__init__.py`, plus every other `lru_cache`/`@cache` hit repo-wide (`grep -rn "lru_cache|@cache"` across `src/pyutilz`). Two findings were verified with standalone read-only repro scripts (run and captured below); the others are direct code-reading findings with concrete call-site confirmation via grep (e.g. confirming no caller ever refreshes/clears a given cache).

Findings by severity: 1 HIGH, 3 MEDIUM, 2 LOW.

## Findings

### [HIGH] `compute_code_version()` memoizes by function-object IDENTITY, blind to in-place source mutation — `src/pyutilz/performance/kernel_tuning/code_versioning.py:105-128`

- **Category**: cache-key collision (mutable-object identity key)
- **Problem**: `compute_code_version` is decorated `@lru_cache(maxsize=2048)` and takes the kernel functions themselves (`*variant_fns: Callable`) as part of the cache key:
  ```python
  @lru_cache(maxsize=2048)
  def compute_code_version(*variant_fns: Callable, extra_fns: tuple = (), salt: int = 0) -> str:
      sources = sorted(_normalized_source(fn) for fn in list(variant_fns) + list(extra_fns))
      ...
  ```
  Plain Python functions don't override `__eq__`/`__hash__`, so `lru_cache`'s key comparison is `id()`-based: two calls with the *same function object* always hit the cache, regardless of whether that object's actual behavior/source has changed since the first call. The whole point of this function (per its own docstring: "a kernel's cached tuning is keyed partly by a `code_version` so a tuning is invalidated **only** when the function's logic changes") is defeated the moment a function object's code is swapped in place while its identity is preserved — e.g. IPython's `%autoreload 2` (the standard interactive-development reload mechanism in exactly the Jupyter/data-science workflow this kernel-tuning subsystem targets), `jurigged`-style hot-reload, or any test/mocking pattern that reassigns `fn.__code__` rather than rebinding the name to a new function object.
- **Failure scenario**: verified with a standalone repro (`python` against this checkout):
  ```
  hash before code swap: 73099936da9efeaf1e6cfe53026dd58721542c86c1c5b5536789b923ed3e45d2
  hash after code swap : 73099936da9efeaf1e6cfe53026dd58721542c86c1c5b5536789b923ed3e45d2
  STALE (same hash despite different logic)? True
  hash after cache_clear (same object, mutated code): 10caa90ecc70ef3592b19d5154d83f52649053b056326d939746f3679b572776
  Now reflects new logic (proves the swap really changed source)? True
  ```
  Concretely: a developer iterating on a numba/GPU kernel inside a live IPython session with autoreload enabled edits `variant_fns[0]`'s body (e.g. changes the block size / algorithm), re-runs `TunerSpec.choose()` (`registry.py:119-158`, which calls `self.code_version()` → `compute_code_version(*self.variant_fns, ...)` on every dispatch). Because autoreload patches `__code__` on the *same* function object rather than creating a new one, `compute_code_version` returns the pre-edit hash. `KernelTuningCache._code_version_stale()` then finds the stored `code_version` still matches, so the cache is never invalidated — the OLD (now-incorrect for the new kernel logic) region/timing data keeps being served silently, potentially routing to a variant that no longer behaves as measured, or masking a real correctness regression introduced by the edit.
- **Suggested fix**: derive the lru_cache key from something content-based rather than the function object's identity — e.g. hash `_normalized_source(fn)` for each function *before* the `lru_cache` boundary (so the cache key is the source-text hash, not the function object), or drop the `lru_cache` decorator on `compute_code_version` itself and instead memoize per-function `_normalized_source` results keyed by `(fn.__code__, fn.__code__.co_filename, fn.__code__.co_firstlineno)`/content-hash rather than by the enclosing function tuple.

### [MEDIUM] Kernel-tuning cache garbage-collects by file mtime while the reader picks "newest" by `tuned_utc` — can silently delete the logically newest tuning — `src/pyutilz/performance/kernel_tuning/cache/cache_class.py:432-449` vs `:235-275` / `:331-367`

- **Category**: cache-eviction / read-path inconsistency (staleness via mismatched "newest" definitions)
- **Problem**: The reader (`_read_kernel_newest`, `_read_kernel_dir_by_path`) resolves "the current tuning" by sorting candidate files on `(entry["tuned_utc"], mtime)` — i.e. the *declared* logical timestamp wins, mtime is only a tiebreak:
  ```python
  candidates.sort(key=lambda c: (c[0], c[1]))
  return candidates[-1][2]
  ```
  But `_gc_kernel_dir`, which runs on every `_persist_kernel` call (`cache_class.py:394`), evicts purely by **file mtime** (write order), with no knowledge of `tuned_utc`:
  ```python
  files.sort(key=lambda p: os.path.getmtime(p))
  for p in files[:-keep]:
      os.remove(p)
  ```
  `update()`'s public signature accepts an explicit `tuned_utc: Optional[str]` override (`cache_class.py:465-467`). Whenever `tuned_utc` is set non-monotonically with respect to wall-clock write order (e.g. a tool that imports/replays historically-timestamped tunings, or merges tunings recorded on a different host/clock), the entry the reader would select as newest can have an *older* mtime than 4+ other entries and gets garbage-collected before it's ever read — permanently and silently losing the logically-current tuning in favor of an older one.
- **Failure scenario**: verified with a standalone repro against a real `KernelTuningCache` (temp `PYUTILZ_KERNEL_CACHE_DIR`): wrote one entry with `tuned_utc="2030-01-01..."` first, then four more entries with progressively *older* `tuned_utc` values ("2020-01-01" .. "2020-01-04") written *after* it (so they have newer mtimes). Output:
  ```
  files before GC (gc runs automatically inside update(), keep=4): 4
    ... tuned_utc= 2020-01-01T00:00:00+00:00 variant= gen1
    ... tuned_utc= 2020-01-02T00:00:00+00:00 variant= gen2
    ... tuned_utc= 2020-01-03T00:00:00+00:00 variant= gen3
    ... tuned_utc= 2020-01-04T00:00:00+00:00 variant= gen4
  reader (_ensure_loaded -> newest by tuned_utc) picks: [{'variant': 'gen4', ...}]
  Was gen0 (the logically newest tuned_utc) evicted by mtime-based GC? True
  ```
  The `gen0` entry (declared as the newest tuning, `tuned_utc=2030-01-01`) is gone — GC removed it purely because it happened to be written first. No current in-tree caller passes `tuned_utc=` explicitly (all use the default "now"), so this is not reachable through today's call sites, but it's a public, documented parameter of `update()`, and the two code paths (`_gc_kernel_dir` vs. the newest-resolution readers) silently disagree about what "newest" means — a latent data-loss trap for any downstream/migration tool that does pass it.
- **Suggested fix**: make `_gc_kernel_dir` sort by the same key the readers use (`entry["tuned_utc"]`, falling back to mtime), not raw mtime alone — read each candidate's `tuned_utc` the same way `_read_kernel_newest` already does before deciding which `keep` files to retain.

### [MEDIUM] OpenRouter model catalogue cache has no TTL/staleness handling — pricing & context-limit data goes stale forever in a long-running process — `src/pyutilz/llm/openrouter_provider/__init__.py:50` / `_catalogue.py:35-61`

- **Category**: staleness (no invalidation policy)
- **Problem**: `_MODELS_CATALOGUE` is fetched once and cached indefinitely with no time-based expiry:
  ```python
  def _fetch_models_catalogue(timeout: float = 10.0) -> dict[str, dict[str, Any]]:
      if _pkg()._MODELS_CATALOGUE is not None:
          return _pkg()._MODELS_CATALOGUE
      with _pkg()._MODELS_LOCK:
          ...
          _pkg()._MODELS_CATALOGUE = catalogue
          return _pkg()._MODELS_CATALOGUE
  ```
  It is invalidated only via an explicit `list_openrouter_models(refresh=True)` or `clear_openrouter_caches()` call — confirmed by `grep -rn "clear_openrouter_caches(" src/pyutilz` returning only the function's own definition, no auto-invoking call site anywhere in the package. This cache backs `OpenRouterProvider.context_window`, `max_output_tokens`, `supports_json_mode()`, and `_input_cost_per_1m`/`_output_cost_per_1m` (`_provider.py:251-300`, `:481-487`) — i.e. request validation, per-token cost estimates, and JSON-mode capability detection for every call a long-lived `OpenRouterProvider` instance makes. This is a deliberate asymmetry against the *sibling* health cache in the same package, which correctly implements a TTL + sweep (`_health.py:79-104`, default 300s) with an explicit design comment about long-running processes — the catalogue cache has no equivalent despite OpenRouter's `/models` catalogue (pricing, context limits, parameter support) changing on a comparable or faster cadence than endpoint health.
- **Failure scenario**: a long-running service builds one `OpenRouterProvider` per model (cached forever by `factory.py`'s `_provider_cache`, see next finding) and calls `generate()` repeatedly over days/weeks. If OpenRouter raises a model's price, drops/adds `response_format` support, or lowers `top_provider.context_length` for the routed upstream, the process keeps using the day-1 snapshot: `estimate_cost()` under/over-reports spend, `supports_json_mode()` can wrongly assume JSON-mode support and requests silently degrade to prose+JSON, and `context_window` can under- or over-estimate the real upstream cap (the latter risking `HTTP 400: prompt too long` the code's own docstring calls out as the exact failure this field exists to prevent).
- **Suggested fix**: give `_MODELS_CATALOGUE` the same TTL treatment as `_HEALTH_CACHE` — store `(fetched_at, catalogue)` and re-fetch once the age exceeds a configurable TTL (e.g. reuse `health_ttl_seconds`'s 300s default or a separate `PYUTILZ_OR_CATALOGUE_TTL`), rather than relying on callers to remember `refresh=True`.

### [MEDIUM] `get_llm_provider()`'s instance cache grows unbounded for the process lifetime, holding live HTTP-client resources — `src/pyutilz/llm/factory.py:22`, `:143-151`

- **Category**: unbounded growth (no eviction)
- **Problem**:
  ```python
  _provider_cache: dict[tuple, LLMProvider] = {}
  ...
  if cache_key in _provider_cache:
      return _provider_cache[cache_key]
  with _provider_lock:
      if cache_key in _provider_cache:
          return _provider_cache[cache_key]
      instance = constructor(**kwargs)
      _provider_cache[cache_key] = instance
      return instance
  ```
  Every distinct `(canonical_provider, sorted(kwargs))` combination gets a permanent entry — there is no max size, no LRU, no TTL, and no eviction API. Each cached entry is a live `LLMProvider` instance; for the `OpenAICompatibleProvider`-based providers (openai/openrouter/deepseek/xai) this holds an `httpx.AsyncClient` with its own connection pool and TLS state (per `factory.py`'s own module docstring: caching exists specifically "avoiding expensive re-initialization (SSL context loading, Anthropic client creation, etc.)" — i.e. the cached objects are deliberately resource-heavy). The `openrouter` provider in particular is explicitly a "meta-provider exposing 200+ models" (`_provider.py:1`), and `model` is part of the cache key, so a service that lets callers pick from that catalogue (or rotates through many models for A/B/cost-routing) accumulates one live provider + HTTP client per distinct model ever requested, forever, for the life of the process.
- **Failure scenario**: a backend service using `get_llm_provider("openrouter", model=<user-selected model>)` across many users/sessions, where users pick from OpenRouter's 200+ model catalogue over time, accumulates up to hundreds of live `OpenRouterProvider` instances (each with its own `httpx.AsyncClient` connection pool) that are never released — `_close_cached_providers()` (`factory.py:154-184`) only runs once, at `atexit`, so nothing reclaims them mid-run. Memory and open-socket usage grows monotonically with the number of distinct model/kwargs combinations ever seen, not with concurrent usage.
- **Suggested fix**: bound `_provider_cache` (e.g. an LRU with a configurable max size, mirroring `_HEALTH_CACHE_MAX_SIZE`'s pattern in `_health.py`), and on eviction call the evicted provider's `_close()` (the same coroutine `_close_cached_providers` already knows how to invoke) so a mid-run eviction doesn't leak the connection pool either.

### [LOW] `get_llm_settings()` is a permanent, unrefreshable singleton — API-key rotation never takes effect without a process restart — `src/pyutilz/llm/config.py:31-34`

- **Category**: staleness (no invalidation policy)
- **Problem**: `@lru_cache` with no arguments caches the *one* `LLMSettings()` instance forever:
  ```python
  @lru_cache
  def get_llm_settings() -> LLMSettings:
      return LLMSettings()
  ```
  `LLMSettings` reads API keys from environment variables / `.env` at construction time (`config.py:14-28`). `grep -rn "get_llm_settings.cache_clear"` across `src` and `tests` returns no hits — nothing in the codebase ever busts this cache. Every provider constructor (`anthropic_provider.py:64`, `deepseek_provider.py:60`, `gemini_provider.py:103`, `openai_provider.py:131`, `xai_provider.py:104`, `openrouter_provider/_provider.py:104`, `factory.py:78`) calls `get_llm_settings()` and trusts whatever it returns.
- **Failure scenario**: an operator rotates a leaked/expiring API key by updating `.env` or the process environment (a routine ops action, e.g. after a credential-scanning alert) in a long-running service that hasn't been restarted. Every subsequent `get_llm_provider(...)` call for a *new* kwargs combination (a cache miss in `_provider_cache`, e.g. a new model string) still resolves `settings.<provider>_api_key` from the stale, pre-rotation `LLMSettings` snapshot — continuing to authenticate with the revoked/old key until the process is restarted, with no error or warning surfaced anywhere in this path.
- **Suggested fix**: either drop the `@lru_cache` in favor of a short TTL (settings changes are rare enough that even a 60s TTL removes the "stuck forever" failure mode), or expose `get_llm_settings.cache_clear` through a documented "reload credentials" hook that ops tooling can call after a rotation.

### [LOW] `TunerSpec._choice_cache` grows unbounded, keyed on continuous-valued dims — `src/pyutilz/performance/kernel_tuning/registry.py:96-97`, `:126-128`, `:157`

- **Category**: unbounded growth (no eviction)
- **Problem**: `_choice_cache: dict = field(default_factory=dict, ...)` memoizes `choose()`'s resolved backend per exact `tuple(sorted(dims.items()))`. `TunerSpec` instances live in the module-level `_REGISTRY` for the process lifetime (`registry.py:33`, never cleared — see `discover_tuners`'s own docstring explaining why it's intentionally never wiped). If `dims` includes a continuous-valued axis such as `n_samples`/`n_features` (the documented common case, e.g. `{"ndim_eq": [2, 3], "n_max": [100, 1000, 10000]}` in the module docstring, but the actual dispatch dims passed to `choose(**dims)` are the live per-call sizes, not the bucketed axis values), every distinct dataset size seen by a long-running service adds a permanent new entry with no cap.
- **Failure scenario**: a long-running feature-selection service (this is squarely mlframe's MRMR/RFECV use case, the stated consumer of this subsystem) processes a continuous stream of datasets with varying row/column counts; each distinct `(n_samples, n_features, ...)` combination it ever sees creates one more permanent dict entry in `_choice_cache`, for as long as the process runs.
- **Suggested fix**: bound `_choice_cache` with a small LRU (e.g. `maxsize=1024`), since its only purpose is to skip the `get_or_tune` region-match lookup on the hot path, not to remember every distinct input forever.

## Things done well

- `KernelTuningCache._ensure_loaded()`/`_persist_kernel()` (`cache_class.py:369-461`) fully build the in-memory payload / on-disk record before ever publishing it (assignment happens under the lock only after the whole dict/JSON is constructed), so a concurrent reader can never observe a partially-populated cache entry — genuinely correct handling of the "torn read" hazard this angle asks about.
- The sweep-ownership marker (`_try_create_marker`, `cache_class.py:865-922`) was explicitly hardened against a real double-sweep race (documented in its own comment: a two-syscall create-then-write left a window where a concurrent loser could read an empty marker and wrongly judge it stale) by switching to a stage-then-atomic-`link` publish — a careful, verified fix to exactly the "partially-populated entry visible to a concurrent reader" failure mode.
- `_HEALTH_CACHE` (`openrouter_provider/_health.py:79-104`) is the one cache in the codebase that gets TTL + bounded-size eviction right: a two-pass sweep (drop >24h-old entries, then LRU-trim to a cap) explicitly sized for "a long-lived-process ... probing 10k distinct models over hours" — this is the template the catalogue cache and `_provider_cache` findings above are missing.
- `hw_fingerprint()`'s disk-backed cache (`cache_base.py:165-210`) documents its own staleness window (7-day freshness via file mtime) and gives an explicit escape hatch (`PYUTILZ_HW_FP_REFRESH=1`) rather than silently caching forever.
- `select_best_gpu()` (`system/gpu_dispatch.py:90-151`) is cached per `(strategy, pid)` but ships a public, documented `reset_cache()` and its docstring is explicit that the result is memoized — no silent-staleness trap.

## Investigated, not an issue

- `select_best_gpu()`'s `lru_cache(maxsize=16)` on `(strategy, pid)` (`system/gpu_dispatch.py:90`) caches a live GPU-load-based decision for the process lifetime. This looks like classic cache staleness (GPU load changes constantly), but it's a deliberate, documented design with an explicit `reset_cache()` API, and — confirmed via `grep -rln "select_best_gpu"` — it is not currently called from anywhere else in `src/pyutilz`, so it has no live blast radius today.
- `webtext.py`'s lazily-initialized globals (`nlp`, `inflect_engine`, `ascii_emojies`, `unicode_emojies`, `webtext.py:13-26`, `381-414`) use the `if x is None: x = expensive_init()` pattern without a lock. Two threads racing on first use could both construct the (e.g.) spaCy model, wasting work, but since Python only rebinds the module global after the right-hand side fully evaluates, no thread can ever observe a partially-constructed object — this is a benign duplicate-init race, not a cache-correctness bug.
- `token_counter.py`'s `_encoding_cache` dict (`token_counter.py:33-50`) has no lock and no bound, but its key space is the small, fixed set of tiktoken-recognized model name strings — not attacker- or user-controlled in a way that could grow unbounded, and a lost race just re-does a cheap `tiktoken.encoding_for_model()` call.
- `cache_base.py`'s `_gpu_summary_cached(device_id)` (`lru_cache(maxsize=16)`) staying keyed per-device rather than globally was specifically checked against the "mutable-object identity key" failure mode: the key is a plain `int` device id, not a mutable GPU handle, so there's no risk of a stale key surviving a device's state change under a colliding key.

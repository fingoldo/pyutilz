# 2026-09-26 audit round: tracker

One row per finding; the full disposition sits under each finding in its report and in `dispositions/`. The round
stays in the open tree while any row is OPEN, then moves to `audits/implemented/`.

| Status | Sev | ID | Title |
|---|---|---|---|
| **RESOLVED** | High | `OR-1` | Mid-stream errors are swallowed, so a failed stream returns partial text as a success |
| **RESOLVED** | Med | `OR-2` | `provider.require_parameters` is never sent, although the code already works around the failure it prevents |
| **RESOLVED** | Med | `OR-3` | Several routing knobs are missing: only, data_collection, zdr, quantizations, max_price, preferred_min_throughput / preferred_max_latency |
| **RESOLVED** | Low | `OR-4` | Web-search citations are lost on the streaming path |
| **RESOLVED** | Low | `OR-5` | Most per-call OR fields are overwritten, not summed, when one call makes a second POST |
| **RESOLVED** | Low | `OR-6` | The session cost estimate prices cache writes at the plain input rate |
| **RESOLVED** | Low | `OR-7` | Pricing, limits and capability checks use the requested model, even when `models` fallback served another one |
| **RESOLVED** | Low | `OR-8` | Per-block `cache_control` breakpoints are not available |
| **RESOLVED** | Low | `OR-9` | `reasoning.max_tokens` is never sent |
| **RESOLVED** | Low | `OR-10` | The json_schema wrapper does not force `strict` |
| **RESOLVED** | Low | `OR-11` | Tool calling is response-only |
| **RESOLVED** | Low | `OR-12` | `stream_options.include_usage` is deprecated on OpenRouter, but a comment says it is required |
| **RESOLVED** | Low | `OR-13` | `transforms: ["middle-out"]` is not available |
| **RESOLVED** | Low | `OR-14` | The file-parser (PDF) plugin, `:online`, and web plugin options (`max_results`, `engine`) are not available |
| **RESOLVED** | Low | `OR-15` | seed, stop, logprobs/top_logprobs, verbosity, `user` and presets are not available |
| **RESOLVED** | Low | `OR-16` | The buffered path ignores Retry-After on 429 |
| **RESOLVED** | Low | `OR-17` | Batch recovery matches on `metadata.request_hash`, which submit never sends |
| **RESOLVED** | Low | `OR-18` | Batch result parsing reads only `message.reasoning` and treats truncated results as successful |
| **RESOLVED** | Low | `OR-19` | The decisions client does not retry 408 or 504 and ignores Retry-After |
| **RESOLVED** | Low | `OR-20` | A 200 body with a non-numeric error code is never retried |
| **RESOLVED** | High | `PROV-1` | `thinking={"type":"enabled","budget_tokens":N}` is rejected with 400 by every current Claude model |
| **RESOLVED** | High | `PROV-2` | Pricing / limits tables miss every current model and use wrong legacy IDs |
| **RESOLVED** | High | `PROV-3` | `context_window` hardcoded to 200_000 |
| **RESOLVED** | Med | `PROV-4` | Only the FIRST text block is returned |
| **RESOLVED** | Med | `PROV-5` | `stop_reason` `refusal` and `model_context_window_exceeded` not handled |
| **RESOLVED** | Med | `PROV-6` | Thinking tokens estimated as chars//4 though the API reports them |
| **RESOLVED** | Med | `PROV-7` | Cache-read multiplier hardcoded 0.10; cache write always 1.25x |
| **RESOLVED** | Low | `PROV-8` | `count_tokens` failure silently falls back to tiktoken |
| **RESOLVED** | Low | `PROV-9` | Default model `claude-sonnet-4-20250514` and `max_tokens` default `min(max_output, 21000)` |
| **RESOLVED** | High | `PROV-10` | Sends `max_tokens`, which reasoning models do not accept |
| **RESOLVED** | High | `PROV-11` | `thinking=` silently ignored for OpenAI; `reasoning_effort` never sent |
| **RESOLVED** | Med | `PROV-12` | Default `temperature=0.7` sent to reasoning models (unverified) |
| **RESOLVED** | Low | `PROV-13` | `check_account_limits` override discards captured headers |
| **RESOLVED** | Low | `PROV-14` | Unknown-model warning fires even when prefix match succeeds |
| **NOT A DEFECT** | Low | `PROV-15` | `num_tokens_from_messages` uses gpt-3.5-0613 framing |
| **RESOLVED** | High | `PROV-16` | Current model `deepseek-flash` unknown: 8192 max output, 64K window, wrong price |
| **RESOLVED** | Med | `PROV-17` | Pricing stale and off-peak not modeled |
| **RESOLVED** | Low | `PROV-18` | Legacy alias check `startswith("deepseek-v4")` excludes `deepseek-flash` |
| **RESOLVED** | Low | `PROV-19` | HTTP 402 (no balance) retried forever |
| **RESOLVED** | High | `PROV-20` | Live Search `search_parameters` retired; requests now 410 |
| **RESOLVED** | Med | `PROV-21` | `reasoning_effort` not sent; billed-output formula unverified |
| **RESOLVED** | Med | `PROV-22` | Uses `thinking_budget`; Gemini 3 docs describe `thinking_level` |
| **RESOLVED** | Med | `PROV-23` | httpx transport errors not retried; no explicit timeout |
| **RESOLVED** | Low | `PROV-24` | Structured output not used |
| **RESOLVED** | Low | `PROV-25` | Prompt-level block reason not captured |
| **RESOLVED** | High | `PROV-26` | SDK path records zero tokens and zero cost |
| **RESOLVED** | Med | `PROV-27` | `is_error` ignored on both paths |
| **RESOLVED** | Med | `PROV-28` | Thinking passed via MAX_THINKING_TOKENS budgets |
| **RESOLVED** | Med | `PROV-29` | `max_output_tokens=32000`, `context_window=200_000` hardcoded |
| **RESOLVED** | Low | `PROV-30` | `generate_batch` hardcodes `max_tokens=1024` / temperature 0.7 |
| **RESOLVED** | Low | `PROV-31` | `count_tokens` constructs a new `AsyncAnthropic()` per call, never closed |
| **WON'T FIX** | Low | `PROV-32` | `claude_code_sdk` pinned to 0.0.25-0.0.30 and monkey-patched |
| **NOT A DEFECT** | Low | `PROV-33` | Default cl100k for unknown models |
| **RESOLVED** | High | `CORE-1` | `images_on_disk` path traversal: the data-URI media type becomes a filename unsanitised |
| **RESOLVED** | High | `CORE-2` | A factory-cached provider breaks on the second event loop (`LazySemaphore` binds to the first loop) |
| **RESOLVED** | Med | `CORE-3` | `PerCallAttr` store never prunes dead instances: unbounded growth and O(n) copy per set |
| **RESOLVED** | Med | `CORE-4` | Worst-case wall clock per call is roughly 46 h: 50 retries times a 3,000 s derived timeout, with no total deadline |
| **RESOLVED** | Med | `CORE-5` | Unknown model silently priced at `_DEFAULT_PRICING` (0, 0), although the docstring promises a warning |
| **RESOLVED** | Med | `CORE-6` | `extract_json` lets `RecursionError` escape (the retry/refusal handling only catches `JSONDecodeError`) |
| **RESOLVED** | Low | `CORE-7` | `extract_json` step 4 is quadratic in the number of `{` candidates, and step 5 logs the full response at ERROR |
| **RESOLVED** | Low | `CORE-8` | Derived timeout covers output size but not reasoning time-to-first-token for small budgets |
| **RESOLVED** | Low | `CORE-9` | `_progress` counter leaks across concurrent streams started from a tracked context |
| **NOT A DEFECT** | Low | `CORE-10` | `repetition_loop` misses loops until the stream ends on a unit boundary |
| **RESOLVED** | Low | `CORE-11` | `_longest_prefix_pricing` duplicates `longest_prefix_lookup` |
| **RESOLVED** | Low | `CORE-12` | atexit close runs each provider's `aclose()` on a new loop that is not the one that opened the sockets |
| **RESOLVED** | Low | `CORE-13` | `generate_batch` cannot forward `images`/`thinking`/`json_mode` |
| **RESOLVED** | Low | `CORE-14` | Test gaps for important core behaviour |
| **RESOLVED** | High | `CI-1` | Black gate red on every run: one real formatting finding |
| **RESOLVED** | High | `CI-2` | Dependabot auto-merge cannot work: repo setting disabled |
| **RESOLVED** | High | `CI-3` | Every push runs the full 21-leg OS x Python matrix |
| **RESOLVED** | Med | `CI-4` | No paths-ignore on push: docs/audit-only commits run the full CI |
| **RESOLVED** | Med | `CI-5` | Tests run serially: no pytest-xdist |
| **RESOLVED** | Med | `CI-6` | numba-disabled flag badge shows "unknown": no codecov.yml, no carryforward |
| **RESOLVED** | Med | `CI-7` | Codecov CLI regenerates coverage.xml and hits the whole-suite fail_under in the numba job |
| **RESOLVED** | Med | `CI-8` | Chained coverage workflows almost never fire because CI is never green |
| **RESOLVED** | Low | `CI-9` | fetch-depth: 0 on all 21 test legs |
| **RESOLVED** | Low | `CI-10` | pip rather than uv for the heavy install |
| **RESOLVED** | Low | `CI-11` | Missing gates that mlframe has: dependency-review and CodeQL |
| **WON'T FIX** | Low | `CI-12` | ubuntu-latest will move to Ubuntu 26 on 2026-10-19; Node-20 actions inside the reusable workflows |
| **WON'T FIX** | Low | `CI-13` | publish.yml reruns the full suite on tag push with no cache |
| **RESOLVED** | Med | `CI-14` | PyPI version and pyversions badges point at a package that does not exist |
| **RESOLVED** | Low | `CI-15` | Workflow badges use the legacy `/workflows/<name>/badge.svg` form with no branch or event filter |
| **RESOLVED** | Low | `CI-16` | Missing badges that mlframe shows |
| **RESOLVED** | High | `GEN-1` | `hash_array_summary` gives the same hash when numeric data changes in the middle, or anywhere in a column that contains a NaN |
| **RESOLVED** | Med | `GEN-2` | Different kernel names can map to the same cache directory, and one of the two kernels is then lost |
| **RESOLVED** | Med | `GEN-3` | `_region_matches` matches when a dim is NaN or misspelled, and never matches `_eq` tuples after a reload |
| **RESOLVED** | Med | `GEN-4` | `synchronize_gpu_if_available` only waits on cupy's null stream and never on numba.cuda, although its docstring says it does |
| **RESOLVED** | Med | `GEN-5` | `sweep_backend_crossover`: a NaN in the reference output rejects every other backend, and tuple or ragged outputs crash the sweep |
| **RESOLVED** | Med | `GEN-6` | `cuda_memory_guard` raises MemoryError when cupy's pool has the memory, and empties the pool on every exit |
| **RESOLVED** | Med | `GEN-7` | nvidia-smi (GPUtil) device ids are used as CUDA device ids |
| **RESOLVED** | Med | `GEN-8` | `safe_load` checks the hash of one read and unpickles a second read; `write_sidecar` rewrites the sidecar in place |
| **RESOLVED** | Low | `GEN-9` | `hw_fingerprint` writes its disk cache through a fixed temp file name |
| **RESOLVED** | Low | `GEN-10` | `_read_kernel_newest` is unused and duplicates `_read_kernel_dir_by_path` |
| **RESOLVED** | Low | `GEN-11` | `SingleFlightCache.get_or_fetch` counts a hit as a miss, and a second cancellation can leave waiters blocked forever |
| **RESOLVED** | Low | `GEN-12` | `_get_path_lock` does not normalise path case on Windows |
| **RESOLVED** | Low | `GEN-13` | `distribute_work` balances load in input order instead of largest-first |
| **RESOLVED** | Low | `GEN-14` | Four near-identical chunking helpers |
| **RESOLVED** | Low | `GEN-15` | Local code_audit scanners duplicate py_ci_shared gates, and relevant shared gates are not adopted |
| **RESOLVED** | High | `DS-1` | autopsia: Noema complaint parsing lets LLMStreamInterruptedError escape as HTTP 500 |
| **RESOLVED** | High | `DS-2` | autopsia: Noema sends patient complaint text to OpenRouter with no data-collection policy |
| **RESOLVED** | Med | `DS-3` | autopsia: the ingest pipeline treats a mid-stream failure as a crash, not as a recorded outcome |
| **RESOLVED** | Low | `DS-4` | autopsia: KB translation retries a non-retryable interruption three times |
| **RESOLVED** | Med | `DS-5` | autopsia: Track A/B/C json_schema + thinking calls now get require_parameters automatically; pinned routes can 404 |
| **RESOLVED** | Low | `DS-6` | autopsia: attempt archive records a mid-stream failure as "truncated" |
| **RESOLVED** | High | `DS-7` | glossum: LLMStreamInterruptedError bypasses every `except LLMProviderError` in the pipeline |
| **RESOLVED** | Med | `DS-8` | glossum: the response-cache route key misses the new routing attributes |
| **RESOLVED** | Low | `DS-9` | glossum: two measurement scripts call OpenRouter over raw HTTP and miss what the provider now does |
| **RESOLVED** | Med | `DS-10` | llm_bench: no way to pass the new provider kwargs to the default factory, so benchmarks cannot hold quantization and route policy fixed |
| **RESOLVED** | Low | `DS-11` | llm_bench: failure classification has no label for a stream interruption, and a require_parameters 404 is labelled ModelNotFound |
| **OPEN** | Med | `DS-12` | all repos: pins cannot express the new API |

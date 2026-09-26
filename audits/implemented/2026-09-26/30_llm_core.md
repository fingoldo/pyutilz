# pyutilz.llm shared core audit (2026-09-26)

Scope: `src/pyutilz/llm/{__init__,base,factory,config,_unset,_messages,_progress,_retry,_timeouts,_stream_attempts,degeneracy,exceptions,_pricing}.py`,
plus the call sites in `openai_compat.py` / `anthropic_provider.py` that consume the core's budget and timeout helpers.
Read-only. Findings marked "Reproduced" were run against this worktree on Python 3.14.3 (Windows).

### CORE-1 (High) -- `images_on_disk` path traversal: the data-URI media type becomes a filename unsanitised

**Disposition:** RESOLVED -- `images_on_disk` now takes the extension from an allowlist (`_IMAGE_EXTENSIONS`, `src/pyutilz/llm/_messages.py:165`, used at `:215`). An unknown subtype is dropped and counted in the one warning. A `resolve().parent` assertion guards the write, and the `finally: rmtree` cleanup is unchanged. Tests: `TestImagesOnDiskTraversal` (the traversal URI writes nothing outside the temp dir, fixed extensions, unknown subtype dropped, dir removed when the body raises).
Evidence: `_messages.py:197-199` builds `extension = media_type.split("/", 1)[1].split("+")[0]` and writes
`directory / f"attachment_{index + 1}.{extension}"`. `split_data_uri` (`_messages.py:112-125`) only checks that the media type
starts with `image/`. Reproduced: `data:image/x/../../pyutilz_trav_probe;base64,aGk=` wrote
`C:\Users\Admin\AppData\Local\Temp\pyutilz_trav_probe`, outside the mkdtemp dir. The `rmtree` in `finally` (`:210`) does not remove it.
Impact: whoever controls an image data URI passed to `ClaudeCodeProvider` (`claude_code_provider.py:417`) can write arbitrary bytes to
any path relative to the temp dir (on Windows the missing intermediate component does not stop it). The file outlives the call. The CLI
agent is then told to read it.
Fix: take the extension from a whitelist (`{"image/png": "png", "image/jpeg": "jpg", ...}`) or `mimetypes.guess_extension`, else drop
the image. At minimum `re.fullmatch(r"[a-z0-9.+-]{1,16}", ext)`, and assert `path.resolve().parent == directory.resolve()`. Test:
none exists for `images_on_disk`, `split_data_uri` or `build_gemini_parts` (grep over `tests/` returns 0 files).

### CORE-2 (High) -- A factory-cached provider breaks on the second event loop (`LazySemaphore` binds to the first loop)

**Disposition:** RESOLVED -- `LazySemaphore` keeps one semaphore per running loop in a `WeakKeyDictionary` (`base.py:127`). An explicitly assigned semaphore still wins. The new `PerLoopHTTPClient` descriptor (`base.py:222`, declared as `LLMProvider._client` at `:485`) gives the first loop the client assigned in `__init__` and every other loop a clone (`_clone_async_client`, `:175`). The clone copies base_url, headers, timeout, auth and hooks, and rebuilds a network transport from its pool settings. A socket-free transport such as `MockTransport` is shared. Non-httpx values pass through unchanged. `openai_compat.py` did not need an edit. Tests: `TestPerLoopResources` (two `asyncio.run` under contention, a distinct semaphore per loop, the explicit semaphore kept, a client cloned for the second loop with the same config, a network transport rebuilt, non-httpx values untouched). HANDOFF to openai_compat owner: `OpenAICompatibleProvider._close` closes only `self._client` for the current loop. To also close per-loop clones, it should iterate `type(self).__dict__`/`LLMProvider.__dict__["_client"].all_clients(self)`. The factory atexit path already closes clones (CORE-12).
Evidence: `base.py:130-137` builds the `asyncio.Semaphore` once per instance and keeps it forever. `factory.py:181-186` caches
instances process-wide, and the httpx client is also built once (`openai_compat.py:109`). Reproduced: two consecutive
`asyncio.run(batch())` calls with 3 contending tasks on `_max_concurrent=1`. The second one raised
`RuntimeError: <asyncio.locks.Semaphore ... [locked]> is bound to a different event loop`.
Impact: scripts, notebooks, and test suites that call `asyncio.run` more than once (for example a sync wrapper per call) get a hard
failure. It only happens under contention, so it looks flaky. The same pattern holds for the `httpx.AsyncClient` pool, whose sockets
belong to the dead loop.
Fix: record `asyncio.get_running_loop()` next to the semaphore in `LazySemaphore.__get__` and rebuild it when the loop differs. Apply
the same rule to the HTTP client, or key the factory cache on the running loop (weakly). Add a two-`asyncio.run` regression test.

### CORE-3 (Med) -- `PerCallAttr` store never prunes dead instances: unbounded growth and O(n) copy per set

**Disposition:** RESOLVED -- `PerCallAttr.__set__` (`base.py:92`) drops entries whose weakref is dead while doing its copy-on-write copy. The `_StrongRef` fallback, which pinned instances forever, is removed. A type without weakref support now gets a `TypeError` that says to add `__weakref__` (no provider uses `__slots__`). Tests: `TestPerCallAttrPrunes` (2000 dead instances leave 1 entry, a slotted type is refused).
Evidence: `base.py:89-97` copies the whole `{id(instance): (ref, value)}` dict on every set and only adds to it. `__get__` (`:82-87`)
ignores dead refs but never deletes them. Reproduced: 10,000 short-lived instances left 10,000 entries (each holding its value) in the
root context's store for one attribute.
Impact: this is the case the F41 comment (`:60-64`) says it fixed, in a new place. The ContextVar count is bounded, but a service that
builds a provider per request (the unhashable-kwargs path, `factory.py:167-179`) keeps every dead provider's last usage, tool calls and
citations for the life of the context. Each set also costs O(live+dead), so the total cost is quadratic.
Fix: in `__set__`, drop entries whose `ref() is None` while copying, or register a `weakref.finalize` that marks them for pruning. A
`_StrongRef` entry (`:100-109`) pins its instance forever, so it needs the same treatment or should be refused.

### CORE-4 (Med) -- Worst-case wall clock per call is roughly 46 h: 50 retries times a 3,000 s derived timeout, with no total deadline

**Disposition:** RESOLVED -- `_stop_policy` (`_retry.py:160`) adds a total per-call deadline, `PYUTILZ_LLM_MAX_CALL_SECONDS` (default 7200 s, 0 disables it, `:129`). The deadline is checked only between attempts, so an in-flight long stream is never cut. The 7200 s default fits two full 3000 s derived-timeout attempts plus backoff. The policy also adds a consecutive read-timeout cap, `PYUTILZ_LLM_MAX_CONSECUTIVE_TIMEOUTS` (default 3, 0 disables it, `:132`). It matches ReadTimeout, APITimeoutError, DeadlineExceeded and TimeoutError by MRO class name, and any other error resets the streak. Both env vars are parsed by `_env_number`, which warns and falls back on junk or negative values. Tests: `TestCallDeadline` (the boundary, 0 disables, the default bound, the streak and its reset, ConnectTimeout excluded, env parsing, and a real tenacity loop stopping on the deadline).
Evidence: `_retry.py:26` defaults to 50 attempts. `_retry.py:66` waits up to 300 s plus jitter between attempts. `_timeouts.py:40`
allows 3,000 s per attempt. `_openai_compat_http.py:71` treats every `httpx.TransportError`, `ReadTimeout` included, as retryable.
The only elapsed-time stop is the 402 grace (`_retry.py:108-120`).
Impact: a silent route (the case the `_timeouts.py:36-39` comment accepts at "up to 50 minutes") is retried 49 more times. A
non-streamed ReadTimeout on a route that was in fact generating is also billed upstream on every attempt, and none of those attempts
are recorded (only the streaming path records aborted attempts, via `_stream_attempts.py`).
Fix: add `stop_after_delay(PYUTILZ_LLM_MAX_CALL_SECONDS)` OR-ed into `_stop_policy`. Also cap consecutive `ReadTimeout`s at the
derived ceiling (for example 2) separately from 429/5xx.

### CORE-5 (Med) -- Unknown model silently priced at `_DEFAULT_PRICING` (0, 0), although the docstring promises a warning

**Disposition:** RESOLVED -- the default-pricing branch of `_longest_prefix_pricing` (`base.py:332`) now logs the warning its docstring promised, once per (provider, model). The fallback value stays `_DEFAULT_PRICING` rather than NaN, because every `estimate_cost` sum would silently become NaN and subclasses already set real defaults. Tests: `TestPricingFallback::test_default_fallback_warns_once_per_model`.
Evidence: `base.py:159` docstring: "3. `default` fallback (with a single warning)". Code `base.py:205` returns `default` with no log.
Only the prefix-match branch warns (`:197-204`). `_DEFAULT_PRICING = (0.0, 0.0)` (`:557`).
Impact: `estimate_cost` (`:805-806`) reports $0 for any model missing from a subclass table, with nothing in the log. Budget and ledger
figures are understated with no signal.
Fix: log once per (class, model) on the default branch, as the docstring says. Consider `None`/NaN rather than 0 so callers cannot add
it silently.

### CORE-6 (Med) -- `extract_json` lets `RecursionError` escape (the retry/refusal handling only catches `JSONDecodeError`)

**Disposition:** RESOLVED -- `extract_json` catches `(json.JSONDecodeError, RecursionError)` (`base.py:693`) and raises `JSONParsingError` ("nesting too deep to decode"). Tests: `TestExtractJson::test_deep_nesting_is_a_json_parsing_error` (arrays and objects nested 100k deep).
Evidence: `base.py:457-530` catches only `json.JSONDecodeError`. Reproduced: `LLMProvider.extract_json("[" * 100000)` raised
`RecursionError: Stack overflow ... while decoding a JSON array`, not `JSONParsingError`.
Impact: deeply nested or runaway output (the `[[[[` / `{"a":{"a":` loops that degeneracy.py exists to catch) escapes the
`JSONParsingError` contract and every `except LLMProviderError` handler. In `generate_batch` it becomes an opaque per-item error.
Fix: widen the except to `(json.JSONDecodeError, RecursionError)` and map both to `JSONParsingError`.

### CORE-7 (Low) -- `extract_json` step 4 is quadratic in the number of `{` candidates, and step 5 logs the full response at ERROR

**Disposition:** RESOLVED -- profiling showed the quadratic cost is building each `JSONDecodeError`, which counts newlines up to the failure position, and not the scan itself. Stopping at the first failure that reaches end-of-input would also lose `{"a": {"b": 2}`, so the scan stops instead once failed candidates have cost `_JSON_SCAN_BUDGET_CHARS` (20M chars, `base.py:464`). A 180 KB runaway went from 10.3 s to well under 2 s. The ERROR log now shows only `%.2000s` of the response plus its length. Tests: `test_runaway_scan_is_capped` (counts raw_decode calls), `test_runaway_scan_is_fast`, `test_object_nested_in_an_unterminated_one_is_still_found`, `test_object_after_non_runaway_failures_is_found`, `test_failure_log_is_truncated`.
Evidence: `base.py:503-511` calls `raw_decode` from every `{`, and each call can scan to the end of the text. Reproduced: 60 KB with
5,000 unterminated `{"b": 1, ` fragments took 0.54 s. The time grows about quadratically with length. `base.py:529` logs the
entire `text` at ERROR.
Impact: slow on long truncated outputs (65k-token outputs are in scope per `degeneracy.py`), and the log fills with multi-MB lines.
Fix: stop after the first failure that reaches end-of-input (every later candidate nested inside it fails the same way), or cap the
candidates. Truncate the logged response (`%.2000s`).

### CORE-8 (Low) -- Derived timeout covers output size but not reasoning time-to-first-token for small budgets

**Disposition:** RESOLVED -- `_timeout_for` adds a reasoning budget stated separately from `max_tokens` (`reasoning.max_tokens`, `thinking.budget_tokens`, via `_separate_reasoning_budget`, `_timeouts.py:13`) to the derived tokens. It also uses that budget when no `max_tokens` is given. The value is still clamped by `_max_derived_timeout_s`. Tests: `TestReasoningBudgetInTimeout`.
Evidence: `_timeouts.py:78-79` computes `max(base, tokens/30)`. For a 500-token request, `base` (240 s per
`tests/test_llm_openai_compat.py:759`) wins, yet the docstring (`:51-54`) cites a 393 s silent-thinking TTFT. When the provider's
reasoning budget is separate from `max_tokens` (Anthropic `budget_tokens`, some routes), a small `max_tokens` can still hit a long
think.
Impact: ReadTimeout storms on reasoning calls with small answer budgets, retried under CORE-4.
Fix: include the reasoning budget (`reasoning.max_tokens` / `thinking.budget_tokens`) in `requested` when present.

### CORE-9 (Low) -- `_progress` counter leaks across concurrent streams started from a tracked context

**Disposition:** RESOLVED -- `generate_batch`'s per-request task clears `_progress._CURRENT` (`base.py:939`), so batch streams no longer sum into an outer `StreamProgress`. The one-stream-per-block rule is documented on `track_stream_progress`. Test: `TestGenerateBatch::test_batch_requests_do_not_share_an_outer_stream_counter`.
Evidence: `_progress.py:55-62` installs the counter in a ContextVar. `asyncio.create_task` copies the context, so tasks created inside
`track_stream_progress` (for example `generate_batch`'s `create_task` at `base.py:780`) all increment the same `StreamProgress`.
Impact: a watcher sees summed chars and interleaved tails, so `repetition_loop` on `answer_tail` can fire on, or miss, a mixture of
streams.
Fix: document the rule "one tracked stream per block", or reset `_CURRENT` to None inside `generate_batch`'s per-request task.

### CORE-10 (Low) -- `repetition_loop` misses loops until the stream ends on a unit boundary

**Disposition:** NOT A DEFECT -- measured: a tail that ends mid-unit is still periodic, so `_LOOP_AT_END` matches with the rotated unit. For example, `"Hmm. "*400 + "Hm"` gives `'m. Hm'`, and the same holds for every cut. Test pinning this: `TestRepetitionLoopMidUnit` (5 cut points).
Evidence: `degeneracy.py:110` anchors the regex with `\Z`, and the whitespace normalisation plus `strip()` (`:121`) helps only for
whitespace. A tail in the middle of a unit ("Hmm. Hmm. Hm") does not match.
Impact: a watcher polling mid-delta gets intermittent `None`, which delays detection by one or more polls. This is not a correctness
failure.
Fix: also try with the last `len(unit)` characters trimmed, or allow a partial final unit (`(?:\1){12,}.{0,80}\Z` with a prefix check).

### CORE-11 (Low) -- `_longest_prefix_pricing` duplicates `longest_prefix_lookup`

**Disposition:** RESOLVED -- `_longest_prefix_pricing` and `longest_prefix_lookup` now share one `_longest_prefix_match` (`base.py:307`). The prefix-match warning fires once per (provider, model) through `_warn_pricing_once` instead of on every `estimate_cost`. Tests: `test_prefix_warning_is_once_not_per_estimate`, `test_both_lookups_share_one_algorithm` (6 models), `test_exact_match_is_silent`.
Evidence: `base.py:143-205` vs `:208-236`. The algorithm bodies are identical, including the copied F19 comment. They differ in
truthiness (`if exact:` at `:175` vs `is not None` at `:216`) and in the warning.
Impact: the two copies drift. Also, the prefix warning (`:198`) fires on every `estimate_cost` call for a non-pinned model, so the log
is spammed.
Fix: implement `_longest_prefix_pricing` as `longest_prefix_lookup` plus a warn-once set.

### CORE-12 (Low) -- atexit close runs each provider's `aclose()` on a new loop that is not the one that opened the sockets

**Disposition:** RESOLVED -- `factory._close_provider` (`factory.py:216`) closes a provider on its client's home loop through `run_coroutine_threadsafe` when that loop is still running. It skips the close when that loop is closed or garbage-collected, because those sockets cannot be closed from another loop. Otherwise it closes on the shutdown loop with `_NO_REBIND` set, so the original client is closed and no clone is built. It also closes each per-loop clone. The first failure is logged at WARNING (`_warn_shutdown_once`, `:206`). Tests: `TestShutdownClose` (the original is closed rather than a clone, clones are closed, a dead home loop is skipped).
Evidence: `factory.py:211-231` creates a new loop and runs `close()`. Errors are logged at debug.
Impact: on Windows (Proactor), connections bound to the original loop can raise or leave "unclosed transport" warnings. Those are the
exact symptom the handler's docstring (`:206-209`) says it prevents. It is silent at default log levels.
Fix: record each provider's creating loop and close there if it is still running, otherwise close the transports synchronously. Log at
WARNING once.

### CORE-13 (Low) -- `generate_batch` cannot forward `images`/`thinking`/`json_mode`

**Disposition:** RESOLVED -- `generate_batch` forwards `images`, `thinking`, `json_mode` and `json_schema` when a request has them (`_BATCH_FORWARDED_KEYS`, `base.py:460`), and warns once per batch about unknown keys. Tests: `TestGenerateBatch::test_forwards_optional_options_only_when_present`, `test_unknown_keys_are_reported`.
Evidence: `base.py:756-765` passes only prompt/system/temperature/max_tokens.
Impact: a batch of vision or reasoning requests silently runs without them. The request dict's extra keys are ignored with no error.
Fix: forward a whitelisted `**{k: req[k] for k in ("images", "thinking", "json_mode", "json_schema") if k in req}`, or reject unknown
keys.

### CORE-14 (Low) -- Test gaps for important core behaviour

**Disposition:** RESOLVED -- every CORE finding above has a regression test in `tests/test_llm_core_audit_20260926.py`, including `images_on_disk` (CORE-1), two event loops (CORE-2), PerCallAttr growth (CORE-3), the deadline (CORE-4), default-pricing logging (CORE-5) and deep nesting (CORE-6). `split_data_uri`, `build_gemini_parts`, `_note_aborted_stream_attempt` and `_repaired_stream_body` belong to modules outside this owner set (`_stream_attempts.py`, `openai_compat.py`), apart from `split_data_uri`, which the CORE-1 tests exercise through `images_on_disk`. Line references above were taken before the base.py split (follow-up 3). After the split, `PerCallAttr`, `LazySemaphore` and `PerLoopHTTPClient` live in `src/pyutilz/llm/_descriptors.py` (`:85`, `:120`, `:215`). In `base.py`: `_longest_prefix_match` is at `:52`, `_longest_prefix_pricing` at `:77`, `_BATCH_FORWARDED_KEYS` at `:205`, `_JSON_SCAN_BUDGET_CHARS` at `:209`, `LLMProvider._client` at `:230`, the `extract_json` except at `:438`, and the batch progress reset at `:684`.
Evidence (grep over `tests/`): 0 files reference `images_on_disk`, `split_data_uri`, `build_gemini_parts`,
`_note_aborted_stream_attempt` (covered only indirectly via `last_aborted_stream_attempts` in
`tests/llm/test_stream_not_retried_after_generation.py`), or `_repaired_stream_body`. No test uses a provider across two event loops
(CORE-2), PerCallAttr growth (CORE-3), a total-deadline stop (CORE-4), default-pricing fallback logging (CORE-5), or deep-nesting
`extract_json` (CORE-6).
Fix: add regression tests together with each fix above.

## Checked and found sound
- `_timeout_for` max_tokens=0 sentinel: explicit None checks (`_timeouts.py:64-77`), tested at `tests/test_llm_openai_compat.py:805-809`.
  The stream and buffered paths both derive the body budget the same way (`openai_compat.py:543-551` vs `:746-760`) and both pass
  `_timeout_for(body)` (`:419`, `:836`).
- The negative `PYUTILZ_LLM_MAX_RETRIES` guard and the 402 elapsed-time grace (`_retry.py`).
- Factory eviction refcount baseline: 2 on 3.14.3, as the comment claims (measured).
- Settings TTL/env_file sentinel (`config.py`, `_unset.py`), exceptions hierarchy, `Pricing` NamedTuple.

## Counts
High 2, Med 4, Low 8. Total 14.

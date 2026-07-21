# Concurrency & Thread/Async Safety Audit

## Summary

I read in full: `src/pyutilz/llm/factory.py`, `src/pyutilz/llm/base.py`, `src/pyutilz/llm/openai_compat.py`, `src/pyutilz/llm/claude_code_provider.py`, `src/pyutilz/llm/openrouter_provider/{__init__.py,_health.py,_catalogue.py}`, `src/pyutilz/performance/kernel_tuning/cache/{cache_base.py,cache_class.py,__init__.py}`, `src/pyutilz/database/db/__init__.py` (connection/cursor/globals sections), `src/pyutilz/database/redislib.py`, `src/pyutilz/system/parallel.py`, `src/pyutilz/system/monitoring.py`, `src/pyutilz/system/hardware_monitor.py`, `src/pyutilz/system/distributed.py`, `src/pyutilz/system/config.py`, `src/pyutilz/core/safe_pickle.py`, `src/pyutilz/web/proxy/base.py`, and `src/pyutilz/web/web.py`. Grep-scanned `anthropic_provider.py`/`gemini_provider.py` for the same shared-state pattern found in `openai_compat.py`. Two findings were empirically confirmed with standalone repro scripts run against the real code (one against `pyutilz.llm.openai_compat.OpenAICompatibleProvider` directly, showing `generate_batch()` misattributes per-call metadata across concurrent requests); one was confirmed via a behavior-equivalent model of psycopg2's cursor object (a live Postgres server isn't available in this sandbox) after first demonstrating the underlying mechanism against real `sqlite3`. Findings: 1 CRITICAL, 2 HIGH, 1 MEDIUM.

## Findings

### [CRITICAL] Module-global psycopg2 cursor shared across threads with zero locking — src/pyutilz/database/db/__init__.py:245-360

- **Category**: shared-mutable-state race / TOCTOU on a non-thread-safe object
- **Problem**: `connect_to_db()` stores the single connection and a **cache of cursor objects** in module globals (`conn`, `cur`, `cursors: Dict[str, Any] = {}`, line 84-86). `get_cursor()` (line 245-257) returns the **same cached cursor object** out of `cursors[cursor_type]` for every caller using the default (unnamed) cursor — which is the overwhelming majority of calls (`safe_execute`, `db_command`, `log_to_db`, etc. all route through `basic_db_execute`). `basic_db_execute()` (line 266-360) then does, with no lock anywhere in this module:
  ```python
  cur = get_cursor(cursor_type=cursor_type, ...)   # SAME shared object, every caller
  cur.execute(statement, data)                     # line 289
  ...
  return cur.fetchall()                             # line 355, several lines / branches later
  ```
  A psycopg2 cursor is a single mutable object whose `execute()` call populates an internal result buffer that a later, separate `fetchall()` reads. psycopg2 provides **no internal cross-thread synchronization on a cursor** (only the connection is documented thread-safe for issuing concurrent commands from different cursors — a single shared cursor is explicitly not). If two threads call `basic_db_execute()` around the same time, thread B's `cur.execute()` can land between thread A's `cur.execute()` and thread A's `cur.fetchall()`, so **thread A silently receives thread B's query results** — no exception, wrong data returned to the caller.
- **Failure scenario**: Any caller that uses this module (which several other pieces of this same repo are built to run under thread pools — `system/parallel.py`'s `applyfunc_parallel(use_threads=True)`, `system/monitoring.py`'s `ThreadPoolExecutor`, `system/distributed.py`'s heartbeat pattern) issues two concurrent `db.safe_execute("select ... where owner=%s", (thread_id,))` calls on different threads. Thread A's `where owner='A'` query can come back holding thread B's `where owner='B'` rows (or vice versa), silently, with no error — a `db_command("select", ...)` used to look up a record by key could hand back a **completely different row** to the wrong caller. This is exactly the failure-scenario shape "wrong results in a common code path" (nothing here is adversarial — ordinary concurrent use of the module's public entry point).
- **Verification**: A live Postgres server isn't available in this sandbox, so I verified the mechanism two ways: (1) an actual `sqlite3` repro (`check_same_thread=False`) reproduces two threads racing `execute()`/`fetchall()` on one shared cursor object — sqlite3's Python wrapper happens to detect this and raises `ProgrammingError: Recursive use of cursors not allowed` (a safety net psycopg2 does **not** have); (2) a model of psycopg2's actual cursor semantics (execute() overwrites `self._buffer`, fetchall() returns `self._buffer`, no lock — matching psycopg2's documented behavior) run under the identical two-thread execute/sleep/fetchall pattern used in `basic_db_execute` reliably reproduces the corruption: thread B's `fetchall()` call returned thread A's row set instead of its own. Both repros are in the scratchpad (`repro_db_shared_cursor.py`, `repro_db_shared_cursor2.py`).
- **Suggested fix**: Either document this module as strictly single-threaded (and add an assertion / thread-owner check that fails loudly on cross-thread use), or give every cursor-using code path its own `threading.local()` cursor (or a lock serializing `execute()`+`fetchall()` as one critical section) so concurrent callers never share a live result buffer.

### [HIGH] LLM providers' "last_*" per-call metadata is shared **instance** state, silently misattributed across concurrent requests on a cached/shared provider — src/pyutilz/llm/openai_compat.py:657-687, 546-564; src/pyutilz/llm/base.py:337-386; src/pyutilz/llm/anthropic_provider.py:73-263; src/pyutilz/llm/gemini_provider.py:111-290

- **Category**: shared-mutable-instance-state race under concurrent asyncio tasks
- **Problem**: Every provider stores "per-call" results the same class's own docstring says are "read by LLMClient after generate()" as plain **instance** attributes: `self._last_usage`, `self.last_tool_calls`, `self.last_citations`, `self._last_finish_reason` (`openai_compat.py:657-666`, written right before `generate()` returns), and the Anthropic/Gemini equivalents (`self.last_cache_creation_input_tokens`, `self.last_thinking_tokens`, `self.last_safety_ratings`, `self.last_function_calls`, etc.). These are written unconditionally at the end of every `generate()` call — no lock, no per-call scoping. Meanwhile:
  1. `pyutilz.llm.factory.get_llm_provider()` explicitly **caches and shares one provider instance** across all callers with the same `(provider, kwargs)` (`factory.py:22,143-151` — that's the entire point of the cache).
  2. Both `LLMProvider.generate_batch()` (`base.py:373-385`, used by Anthropic/Gemini/OpenRouter's default) and `OpenAICompatibleProvider.generate_batch()` (`openai_compat.py:739-750`, used by DeepSeek/xAI/OpenAI) fire **N concurrent `self.generate()` calls on the same `self`** via `asyncio.create_task` + `asyncio.as_completed`.

  So a batch of concurrent requests on one provider instance write these "last_*" attributes from multiple in-flight tasks with no synchronization; a consumer reading `provider.last_tool_calls` / `provider._last_usage` right after a batch item is yielded can get a **different request's** data. `ClaudeCodeProvider` is the one provider that avoids this — its `generate_batch` (`claude_code_provider.py:734-751`) is deliberately sequential — but every other provider inherits the concurrent implementation.
- **Failure scenario**: A caller does
  ```python
  async for resp in provider.generate_batch(requests):
      tool_calls = provider.last_tool_calls   # or provider._last_usage for cost accounting
      record(resp["id"], tool_calls)
  ```
  When two or three requests complete close together (normal under real network jitter, not a crafted timing), the tool-calls/citations/usage attributed to `resp["id"]` actually belong to a **different, concurrently-completing** request. For `_last_usage`, this means per-request cost/token accounting silently attributes the wrong token counts to the wrong request id.
- **Verification**: Ran a standalone repro against the real `pyutilz.llm.openai_compat.OpenAICompatibleProvider.generate_batch()` (monkeypatching only `httpx.AsyncClient.post`, no other code changed) with 4 synthetic requests completing within ~10ms of each other. Confirmed reproducibly (3/3 runs):
  ```
  generate_batch yielded id='req-0' ...; provider.last_tool_calls right now = [{'id': 'req-2'}]
  generate_batch yielded id='req-1' ...; provider.last_tool_calls right now = [{'id': 'req-2'}]
  generate_batch yielded id='req-2' ...; provider.last_tool_calls right now = [{'id': 'req-2'}]
  RACE CONFIRMED -- last_tool_calls read right after a yielded batch response belonged to a DIFFERENT request
  ```
  (Note: reading `provider.last_tool_calls` immediately after **your own, non-batched** `await provider.generate(...)` call is safe — there's no `await` between the write and the `return` inside `generate()`, so no other task can interleave there. The race is specifically triggered by the batch methods' own internal concurrency.) Repro at scratchpad `repro_batch_race.py` / `repro_llm_race.py`.
- **Suggested fix**: Return per-call metadata (tool_calls, citations, usage, finish_reason) as part of the value yielded by `generate_batch()`'s dict, instead of via shared instance attributes; or have `generate_batch()` construct a fresh, unshared provider/response-context object per request instead of mutating `self`.

### [HIGH] `get_url()`'s error/rate-limit-triggered proxy rotation writes the module-global `proxies` without the lock every other write site uses — src/pyutilz/web/web.py:601-613, 645-657

- **Category**: inconsistent lock discipline — unprotected write to a lock-protected shared global
- **Problem**: This module has an explicit `_state_lock` (`web.py:38`) specifically to guard concurrent access to module-level session/proxy state, and the file's own comments show this was deliberately hardened against races (e.g. the "Regression fix" note at line 548-553 about `proxy_*` fields needing to be read as one locked snapshot). The correct, established pattern for writing the global `proxies` is demonstrated at `get_new_session()`:
  ```python
  new_proxies = get_new_smartproxy(...)   # compute into a LOCAL var
  with _state_lock:
      proxies = new_proxies               # line 745-746
  ```
  and at `set_proxy()` (line 323-324). But inside `get_url()` itself — the very function that reads `proxies` under the lock at line 544-546 (`proxies_snapshot = proxies`) — the two proxy-rotation-on-error paths write straight to the global with **no lock at all**:
  ```python
  proxies = get_new_smartproxy(...)   # line 601, inside the transient-error except-branch
  ...
  proxies = get_new_smartproxy(...)   # line 645, inside the ratelimiting-status branch
  ```
  Additionally, `report_params(url, proxies, ...)` at line 625 reads the global `proxies` directly (not the locked `proxies_snapshot`), another unprotected read in the same function that otherwise takes pains to snapshot everything under `_state_lock`.
- **Failure scenario**: Two threads (a realistic usage pattern for this module — it exists specifically to drive concurrent scraping with proxy rotation) both call `get_url()` against different URLs sharing the same module-level proxy pool. Both hit a transient/blocked-proxy condition around the same time; both compute a fresh `get_new_smartproxy(...)` result and assign it to the shared global with no serialization. Whichever assignment lands last silently discards the other thread's proxy rotation — one thread's deliberate "this proxy is bad, rotate off it" correction is thrown away moments after being applied, exactly under the failure conditions (concurrent blocking/rate-limiting) where correct rotation matters most for evading blocks. This directly undermines the anti-blocking purpose the rest of the file's locking was added to protect.
- **Suggested fix**: Wrap both assignments the same way `get_new_session()` does — compute into a local, then `with _state_lock: proxies = local_result` — and change the `report_params(url, proxies, ...)` call at line 625 to use the already-captured `proxies_snapshot`.

### [MEDIUM] `TomlLiveConfig.data` setter bypasses the class's own lock, breaking its documented "Thread-safe" contract — src/pyutilz/system/config.py:198-201

- **Category**: inconsistent lock discipline on documented thread-safe class
- **Problem**: The module docstring states "Thread-safe via a single `threading.Lock`" (line 5), and every other mutation of `self._data` (`_reload()`, always invoked from inside `_maybe_reload()`'s `with self._lock:` block) honors that. But the public `data` property setter does not:
  ```python
  @data.setter
  def data(self, value: dict[str, Any]) -> None:
      """Replace config data programmatically (e.g. after validation)."""
      self._data = value          # line 201 -- no lock
  ```
  while the background mtime-triggered reload path (`_reload()`, called under `self._lock` from `_maybe_reload()`) can run concurrently on another thread and also reassign `self._data`.
- **Failure scenario**: One thread periodically calls `cfg.get(...)` (any getter triggers `_maybe_reload()`, which — if the file's mtime changed — reloads under the lock and reassigns `self._data`). Concurrently, another thread does `cfg.data = validated_overrides` (the documented "replace config data programmatically" use case) with no lock. Whichever assignment executes last wins non-deterministically: a background file-reload landing after the programmatic override silently discards the validated overrides, or the override landing after a legitimate reload discards fresh file contents — with no error, no log, and no way for either caller to know they lost the race. This is a plain lost-update, not a torn read (dict reassignment is atomic), but it directly violates the class's advertised thread-safety guarantee for one of its four public mutation paths.
- **Suggested fix**: `with self._lock: self._data = value` in the setter, matching every other mutator.

## Things done well

- `llm/factory.py`'s `_provider_cache` uses correct double-checked locking (unlocked fast-path check, `with _provider_lock:` re-check before construct+insert) — a clean, textbook-correct pattern.
- `performance/kernel_tuning/cache/` is unusually rigorous about concurrency: immutable-per-write files (no read-modify-write, hence no lost-update), an atomic `O_EXCL`-then-`os.link` singleton-claim protocol with steal-on-crash semantics, and extensive inline documentation of races that were found and fixed (e.g. the empty-marker steal window at `cache_class.py:865-922`). This is the strongest concurrency engineering in the codebase.
- `core/safe_pickle.py`'s per-path `threading.Lock()` registry (`_get_path_lock`) correctly serializes the "write payload, then write sidecar" pair as one unit, with a documented rationale for why a lock-free "last writer wins" on the payload alone isn't sufficient once a sidecar is added.
- `web/proxy/base.py`'s `PortHealthTracker` is consistently locked — every method that reads or mutates `_ports`/`_banned` does so under `self._lock`, including the ban-decision logic that reads across the whole `_ports` dict.
- `web/web.py` and `system/distributed.py` both show evidence of a real, deliberate pass to fix torn-read races (see the "Regression fix" comments in both files) — the two `web.py` gaps flagged above look like that pass simply missing two call sites, not an absence of effort.

## Investigated, not an issue

- `database/redislib.py`'s module-global `rc` read/write without a lock (`rconnect`/`rclose`/`rexecute`'s `if rc is None` check-then-`getattr`) — already investigated and explicitly dispositioned in the round-1 audit (`audits/2026-07-21_full-audit/09-domain-database.md:169`): the module is documented as a single-connection-per-process design, not flagged as a bug given that stated scope. Not re-raising it.
- `system/hardware_monitor.py`'s `UtilizationMonitor` — background thread `list.append()`s into several lists while `get_average_utilization()` (typically called after `stop()`) reads them via `np.mean()`. `list.append` is atomic under the GIL and the documented usage pattern is `start()` → work → `stop()` → `get_average_utilization()`, so no genuine race in the intended usage.
- `system/monitoring.py`'s `timeout_wrapper` — the background daemon thread writes into a per-call `outcome` dict that the caller may abandon on timeout; this looks racy at first glance but the write and the caller's read are never actually contended (the caller either already returned on timeout, or `thread.join()` guarantees the write happened-before the read).
- `system/distributed.py`'s `_identity_lock` — this module already carries explicit round-1-style comments documenting exactly why `register_scraper`'s check-then-act on `_container.node_id` is inside the lock and why `heartbeat_scraper` must be called after releasing it (non-reentrant `threading.Lock`). Read through the full lock-ordering; no deadlock or gap found.
- `llm/openrouter_provider/_catalogue.py`'s `_fetch_models_catalogue()` double-checked-locking against `_MODELS_LOCK`, and `_health.py`'s `_HEALTH_CACHE_LOCK`-guarded TTL cache + `_sweep_health_cache_locked` eviction — both correctly hold the lock across the full check-then-write, and the "write-time re-check against a newer entry" logic at `_health.py:322-328` correctly handles a slow fetch racing a faster concurrent one.

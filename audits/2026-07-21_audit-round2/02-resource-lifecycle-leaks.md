# Resource Lifecycle & Leaks Audit

## Summary

I read in full: `database/db/__init__.py`, `database/db/sqlite.py`, `database/db/sql_helpers.py`, `database/db/upsert.py`, `database/redislib.py`, `web/web.py`, `web/browser.py`, `web/graphql.py`, `web/proxy/base.py`, `web/proxy/session.py`, `web/proxy/decodo.py`, `web/proxy/ip_check.py`, `llm/openai_compat.py`, `llm/base.py`, `llm/factory.py`, `llm/claude_code_provider.py`, `llm/openrouter_provider/_provider.py`, `llm/openrouter_provider/_catalogue.py`, `llm/openrouter_provider/_health.py`, `core/safe_pickle.py`, `core/filemaker.py`, `system/parallel.py`, `system/distributed.py`, `system/hardware_monitor.py`, `system/monitoring.py`, and `system/system/{fsutils,misc,probing,sysinfo,memory,_common}.py`. The `llm` package's httpx-based providers turned out to be unusually well-instrumented for lifecycle (an `atexit`-registered cache-sweep in `factory.py`, a `weakref.WeakSet` for uncached instances, `try/finally` session context managers in `web/proxy/session.py`), so most findings below are concentrated in the older `database`/`web`/`system` modules that predate that pattern. Found 3 HIGH findings (one confirmed via a strong internal-evidence trail in the code itself, two via direct reading of the reassignment sites) and 1 MEDIUM (lower-confidence, could not verify against real CUDA hardware).

## Findings

### [HIGH] Named/server-side PostgreSQL cursor is never closed — `src/pyutilz/database/db/__init__.py:245-360`

- **Category**: db-cursor-leak
- **Problem**: `get_cursor()` creates a genuine server-side cursor whenever `cursor_name` is given:
  ```python
  cur = conn.cursor(cursor_factory=cursor_factory, name=cursor_name, withhold=(False if cursor_name is None else True))
  if itersize: ...
  if "_named" not in cursor_type:
      cursors[cursor_type] = cur
  return cur
  ```
  Named cursors are deliberately **excluded** from the `cursors` cache (`if "_named" not in cursor_type`), so nothing else in the module holds a reference to them either. `basic_db_execute()` (the only caller) then does, on its success path:
  ```python
  else:
      if cur.description is not None:
          if return_cursor:
              return cur
          else:
              return cur.fetchall()
      ...
  ```
  When `return_cursor` is left at its default (`False`, both on `basic_db_execute` and the public `safe_execute`/`safe_execute_values` wrappers), the rows are fetched and returned, but the cursor object itself — which on Postgres corresponds to a real `DECLARE ... CURSOR ... WITH HOLD` on the server, kept alive across commits precisely because `withhold=True` — is discarded with no `.close()` call anywhere in the file. Nothing closes it in the error paths either (`except OperationalError/InterfaceError`, `except DuplicateTable`, `except InternalError`, `except Exception`).
  There is strong internal evidence this is a live, previously-observed symptom rather than a theoretical concern: the `except Exception` branch a few lines below has special-cased handling for `"cursor" in str(e) and "already exists" in str(e)` — i.e. retry-with-backoff when Postgres refuses to `DECLARE` a cursor whose name collides with one still open from a prior, never-closed call. That collision can only happen because the previous named cursor was never closed; the retry logic works around the leak's symptom instead of fixing its cause.
- **Failure scenario**: Any caller of the public API in the natural way — `db.safe_execute(big_select_sql, cursor_name="page1", itersize=10_000)` (a named/server-side cursor is normally requested specifically to stream a large result set without materializing it all in one round trip) — leaks one open server-side cursor on Postgres per call. Repeated calls with the *same* `cursor_name` (a very natural pattern — e.g. a fixed name reused across pagination pages, or the same name across process restarts against a long-lived connection) eventually hit `DuplicateCursor`, silently retried instead of surfaced, while the connection accumulates open server-side cursors until Postgres refuses new ones or the session is torn down. (The one first-party caller of `cursor_name`, `pyutilz.text.tokenizers.tokenize_db_reviews`, avoids the bug only because it explicitly passes `return_cursor=True` and closes the cursor itself in a `try/finally` — this was fixed in round 1 for that one call site, but the underlying library function that made the mistake easy to make was not fixed.)
- **Suggested fix**: In `basic_db_execute`'s `return_cursor=False` + named-cursor branch, close the cursor after `fetchall()` (`rows = cur.fetchall(); cur.close(); return rows`), and close it in every exception branch too (`if cursor_name is not None: cur.close()` in a `finally`). Longer-term: `return_cursor=False` combined with a named cursor is also a design trap independent of the leak — it forces a full `fetchall()` into memory, defeating the entire point of requesting a server-side/`itersize` cursor in the first place.

### [HIGH] `requests.Session` is replaced without closing the old one, leaking its connection pool on every rotation — `src/pyutilz/web/web.py:114-126, 698-747`

- **Category**: http-session-leak
- **Problem**: The module keeps a single module-level `sess: Optional[Any]` (a `requests.Session`). Two functions overwrite it without ever calling `.close()` on the value being replaced:
  ```python
  def init_vars():
      global sess, ...
      sess = None            # drops whatever Session was there, unclosed
      ...

  def get_new_session(...):
      global sess, proxies, headers
      new_sess = requests.Session()
      ...
      with _state_lock:
          sess = new_sess     # overwrites the old Session, unclosed
          ...
  ```
  A `requests.Session` owns a `urllib3` connection pool (`HTTPAdapter`s) with potentially several open sockets (keep-alive connections). `requests.Session.close()` exists specifically to release them; nothing in this module ever calls it before the reference is dropped.
- **Failure scenario**: `get_url()` calls `get_new_session()` whenever `need_new_session` is true (`sess is None or (max_ip_queries>0 and num_ip_queries > cur_max_ip_queries)`, `web.py:536`) — i.e. routinely, by design, every time the configured per-session request budget is exhausted during a normal scraping run — and also from `handle_blocking()` on every detected block. Each rotation silently leaks the previous Session's open sockets/connection-pool state; over a long-running scraper doing many rotations this accumulates open file descriptors/sockets without bound. Separately, `pyutilz.core.filemaker.get_session_token()` calls `web.connect(...)` (which calls `init_vars()`, dropping `sess` again) **twice per invocation** — once to set Basic-Auth headers, once again after a successful login to set the Bearer token — so every FileMaker re-authentication (which itself retries up to `max_retries` times on transient failures) leaks at least one Session per call.
- **Suggested fix**: Before reassigning the module-level `sess` in both `init_vars()` and `get_new_session()`, capture the old value and call `.close()` on it if not `None` (inside the existing `_state_lock` critical section, guarded by `try/except` so a close failure doesn't block rotation).

### [HIGH] `UtilizationMonitor`'s background thread is non-daemon with no exception-safe stop, hanging process exit if `.stop()` is skipped — `src/pyutilz/system/hardware_monitor.py:29-179`

- **Category**: thread-leak / process-exit-hang
- **Problem**: The monitor's worker thread is created without `daemon=True`:
  ```python
  self.stop_flag = threading.Event()
  self.thread = threading.Thread(target=self.query_utilization)
  ```
  and `query_utilization()` runs an unconditional loop that only exits via the flag:
  ```python
  def query_utilization(self):
      while not self.stop_flag.is_set():
          time.sleep(self.sleep_interval_seconds)
          ...
  ```
  `start()`/`stop()` are two independent public methods with no context-manager pairing (`__enter__`/`__exit__`) and no `try/finally` enforcing that `stop()` runs. The class's own docstring example is exactly the failure shape:
  ```python
  >>> monitor.start()
  >>> # ... run your code ...
  >>> monitor.stop()
  ```
  If "run your code" raises (the entire reason someone wraps a benchmark in a monitor — to profile code that might fail), `stop()` is skipped, `stop_flag` is never set, and the thread runs its infinite loop forever. Because CPython does not exit the interpreter until every **non-daemon** thread finishes, this hangs the whole process at shutdown — not merely "a thread leaks", but the process itself refuses to terminate.
- **Failure scenario**: `monitor = UtilizationMonitor(); monitor.start(); risky_benchmark()  # raises; monitor.stop() never reached`. The exception propagates normally and looks handled/logged by the caller, but the process now never exits on its own (e.g. a CI job or script hangs until an external timeout/SIGKILL, rather than the exception's own exit code being observed).
- **Suggested fix**: Set `daemon=True` on the thread (cheapest fix — a leaked monitor thread no longer blocks interpreter exit, though the reads it triggers between "code raised" and "interpreter actually exits" are harmless), and/or add `__enter__`/`__exit__` so `with UtilizationMonitor(...) as monitor:` guarantees `stop()` runs via `finally` even when the wrapped code raises.

## Investigated, not an issue

- `system/parallel.py`'s `mem_map_array()` creates a fresh `tempfile.mkdtemp()` per call and never removes it inline — but this is a **documented, deliberate** design (module-level `_TEMP_DIRS` list + `atexit.register(_cleanup_temp_dirs)`), not an oversight; each call does leak disk space until process exit if called many times in a long-running loop, but that's the explicit stated contract, not a bug.
- `web/proxy/session.py`'s `curl_session`/`requests_session` context managers, `web/proxy/decodo.py`'s `get_endpoints()`, and `llm/openrouter_provider/_health.py`'s `_enrich_with_health()` all correctly use `with ... as session/client:` around every HTTP client they construct — checked closely because they're exactly the shape that tends to leak, and they don't.
- `llm/factory.py`'s provider cache + `atexit`-registered `_close_cached_providers()` (plus a `weakref.WeakSet` for the unhashable-kwargs bypass path) correctly closes every `httpx.AsyncClient` it's aware of at interpreter exit; `openai_compat.py`'s `_client` and `gemini_provider.py`'s analogous client both expose the `_close()` hook the factory looks for via `getattr`.
- `llm/claude_code_provider.py`'s CLI subprocess path (`run_cli()`) is unusually careful: dedicated stdout-reader thread, `finally`-block `kill()` + bounded `wait()` + reader-thread join, and a code comment explaining exactly why it's not wrapped in `with Popen(...)` (a pre-existing `try/finally` is functionally equivalent and is grandfathered by `test_resource_handle_safety`).
- `core/safe_pickle.py`'s `safe_dump()`/`_replace_with_retry()` — every `open()` is a context manager, the temp file is removed in the `except` path, and the per-path locking is explicit about its (documented) single-process-only scope.

## Things done well

- The `llm` package's shared `OpenAICompatibleProvider` base plus `factory.py`'s cache is a genuinely solid pattern for a resource this hard to get right in an async codebase: every subclass gets `_close()` for free, and the `atexit` handler plus `WeakSet` tracking means even providers built outside the cache (unhashable kwargs) still get a shutdown-time close attempt.
- `web/proxy/session.py` and `decodo.py`'s `get_endpoints()` model the right pattern (`with requests.Session() as s: ... ` / `try/finally: s.close()`) that `web/web.py`'s older module-level-session code (finding #2 above) should have followed.
- `core/safe_pickle.py`'s atomic-write path (temp file + fsync + `os.replace` + explicit cleanup on the exception path) is a good template for "resource acquired, must be released even on the failure branch."
- `llm/base.py` and `llm/openai_compat.py`'s `generate_batch()` both explicitly cancel any still-pending `asyncio.Task`s in a `finally` around the `yield` loop, specifically to avoid orphaned, unobservable, billable background LLM calls if the caller stops consuming early — a resource-lifecycle concern (task/API-call lifetime, not just handles) that's easy to miss and was handled correctly in both places.

# Logging & Observability Quality Audit

## Summary

I read in full: `src/pyutilz/dev/logginglib.py`, `src/pyutilz/database/db/__init__.py`, `src/pyutilz/database/db/sqlite.py`, `src/pyutilz/database/db/sql_helpers.py`, `src/pyutilz/database/db/upsert.py`, `src/pyutilz/database/redislib.py`, `src/pyutilz/database/deltalakes.py`, `src/pyutilz/llm/openrouter_provider/_health.py`, `src/pyutilz/llm/openrouter_provider/_provider.py`, `src/pyutilz/llm/openrouter_provider/_catalogue.py`, `src/pyutilz/web/web.py`, `src/pyutilz/web/browser.py` (relevant sections), `src/pyutilz/web/proxy/decodo.py` (relevant sections), `src/pyutilz/system/monitoring.py`, `src/pyutilz/core/serialization.py`, `src/pyutilz/core/image.py` (relevant sections), `src/pyutilz/text/strings/jsonutils.py`, `src/pyutilz/performance/kernel_tuning/cache/cache_class.py` (relevant sections), and `src/pyutilz/dev/code_audit/broad_except.py` (to determine precisely what the existing `broad_except_swallow`/debug-only scanner does and does not catch). I also grepped repo-wide for `print(`, `logger.exception`, `logger.debug`, `logger.info`/`logger.warning` with f-strings, and cross-checked candidate findings against source. One finding (`system/monitoring.py`) was verified with a standalone repro script reproducing the exact broken-traceback output; the rest are derived from direct, deterministic reading of the control flow (not runtime-ambiguous), so I report them at full confidence with the code cited.

Total findings: **4 HIGH**, **5 MEDIUM**, **1 LOW**.

## Findings

### [HIGH] `logger.exception()` called from the wrong thread loses the traceback entirely — `src/pyutilz/system/monitoring.py:197-214`

- **Category**: exception-outside-except-context (cross-thread variant)
- **Problem**: `timeout_wrapper`'s inner `_run()` executes `func` on a dedicated child thread and catches its exception there:
  ```python
  def _run():
      try:
          outcome["result"] = func(*args, **kwargs)
      except Exception as e:
          outcome["error"] = e
      ...
  thread = threading.Thread(target=_run, ...)
  thread.start(); thread.join(timeout=timeout)
  ...
  if "error" in outcome:
      logger.exception("Error in %s: %s", func.__name__, outcome["error"])
  ```
  `logger.exception()` sets `exc_info=True` implicitly, which pulls the *current thread's* `sys.exc_info()`. By the time control reaches line 214 (back in the **main/wrapper thread**), the exception was already caught and its `except` block already exited **on the child thread** — the main thread was never inside an `except` clause at all, so `sys.exc_info()` there is `(None, None, None)`.
- **Failure scenario**: Any exception raised by a function wrapped with `@timeout_wrapper(...)` produces a log line with the correct message text but a bogus, useless traceback. Verified empirically:
  ```
  ERROR:test:Error in myfunc: division by zero
  NoneType: None
  ```
  instead of the real `ZeroDivisionError` traceback. In production this means every timeout-wrapped function's real exception traceback is silently discarded — the log line looks normal (message text is present) so the loss is easy to miss until someone actually needs the stack trace to debug a failure.
- **Suggested fix**: Log with the actual exception object bound via `exc_info=`, e.g. `logger.error("Error in %s: %s", func.__name__, outcome["error"], exc_info=outcome["error"])`, which works regardless of which thread originally raised it.

### [HIGH] Remote kernel-tuning-cache failures are always logged at DEBUG, with no escalation — `src/pyutilz/performance/kernel_tuning/cache/cache_class.py:308-311,397-401,415-418,589-593`

- **Category**: wrong log level hides a real error
- **Problem**: Every one of the four remote-backend call sites (`read` on load, `write` on persist, `read` on pre-write merge, `write` on evict) wraps the call in `except Exception as e: logger.debug(...)`:
  ```python
  try:
      remote_data = self._remote.read(hw_fingerprint())
  except Exception as e:
      logger.debug("kernel_tuning_cache: remote read failed: %s", e)
      remote_data = None
  ...
  try:
      self._remote.write(hw_fingerprint(), self._remote_payload())
  except Exception as e:
      logger.debug("kernel_tuning_cache: remote write failed: %s", e)
  ```
  There is no counter, no "first failure escalates to WARNING", no periodic re-announcement — I grepped the whole file for `logger.warning`/consecutive-failure tracking near `_remote` and found none (the one `logger.warning` in the file, line 79, is an unrelated local-cache-load path). Python loggers default to WARNING-and-above when unconfigured, so DEBUG is invisible out of the box.
- **Failure scenario**: The remote backend is the entire point of this cache (cross-machine sharing of expensive GPU kernel-tuning results, keyed by `hw_fingerprint()` — see `feedback_use_kernel_tuning_cache_for_gpu`). If credentials expire, a bucket is renamed, or network access to the remote store is blocked (e.g. a firewall change, an IAM policy edit), every single read and write silently fails from that moment on. The local file-based cache keeps working, so nothing crashes and nothing looks wrong — but cross-machine sharing has stopped working completely, forever, with zero default-visible signal. An operator would only discover this by explicitly enabling DEBUG logging and knowing to look for it.
- **Why the existing scanner doesn't catch this**: `broad_except.py`'s `_has_debug_only_log_call` (lines 34-47) explicitly treats *any* `logger.debug(...)` call in the handler as sufficient to avoid the `broad_except_swallow` flag, on the stated rationale that debug is "a materially different signal than TRUE silence." That rationale is reasonable for genuinely optional/best-effort paths, but it has no way to distinguish "a truly optional cosmetic fallback" from "the entire point of this code path just went dark" — this is exactly that latter case, four times in one file.
- **Suggested fix**: Track consecutive remote-op failures per process (or per backend instance) and escalate to `logger.warning` on the first failure after a run of successes (and/or once per some interval), so a broken remote backend surfaces without needing DEBUG enabled.

### [HIGH] `get_url()`'s retry loop logs real fetch errors only when `verbose=True` (default `False`) — `src/pyutilz/web/web.py:492,591-594,693-694`

- **Category**: swallowed exception logged at the wrong level (here: not logged at all by default)
- **Problem**: `get_url` is the package's general-purpose "fetch a URL with retries/proxy-rotation" primitive, default `verbose: bool = False`. Its per-attempt exception handler is:
  ```python
  except Exception as e:  # noqa: PERF203
      se = str(e)
      if verbose:
          logger.exception(e)
      se = se.lower()
      ...
  ```
  With the default `verbose=False`, this is the *only* log statement for the exception — nothing else in the `except` branch logs at any level. After all `max_retries` (default 10) attempts are exhausted this way, the only trace left is the generic:
  ```python
  if res is None:
      logger.warning("Could not get url %s", url)
  ```
  which carries no information about *why* (DNS failure? TLS handshake failure? proxy auth error? timeout?) — the specific `str(e)` that was inspected internally (to decide whether to rotate the proxy) is thrown away.
- **Failure scenario**: A caller uses `get_url(url)` with defaults (as most callers reasonably would — `verbose` reads as a "print extra debug info" toggle, not "enable error logging"). Every network exception across 10 retry attempts is completely silent; only a bare "Could not get url https://..." warning appears at the end. Diagnosing a systematic outage (bad proxy pool, expired proxy credentials, SSL misconfiguration) from logs alone is impossible without re-running with `verbose=True`, which most production callers never think to do since nothing in the function signature/docstring flags `verbose` as controlling whether *errors* get logged at all (vs. just request/response tracing).
- **Suggested fix**: Log the per-attempt exception unconditionally at `WARNING` (not `verbose`-gated), e.g. `logger.warning("get_url attempt %d/%d for %s failed: %s", n_retries, max_retries, url, e)`, and reserve `verbose` for the full `report_params()`-style request dump it already controls elsewhere in this function.

### [HIGH] Unbounded Redis retry loop logs a full traceback on every attempt — `src/pyutilz/database/redislib.py:49,74-96`

- **Category**: retry-loop log flooding
- **Problem**:
  ```python
  def rexecute(method_name: str, *args, max_retries: Any = None, **kwargs) -> Any:
      ...
      while True:
          try:
              res = method(*args, **kwargs)
          ...
          except RedisConnectionError as e:
              attempt += 1
              logger.exception(e)
              if max_retries is not None and attempt >= max_retries:
                  logger.error("rexecute: giving up after %d attempts", attempt)
                  raise
              sleep(1 * random())
  ```
  `max_retries: Any = None` and the docstring is explicit: `"None`, the default, retries indefinitely`. `logger.exception(e)` (full traceback, ERROR level) fires on **every single attempt**, with only ~0-1s of jitter between attempts and no cap.
- **Failure scenario**: Any caller of `rexecute(...)` (the module's documented "safely execute any Redis command" entrypoint) that doesn't pass an explicit `max_retries` during a real Redis outage (server restart, network partition, connection-pool exhaustion) produces on the order of one full ERROR-level traceback log line **per second, forever**, until Redis recovers or the process is killed. This can fill disk-backed log files or blow through a log-aggregator's ingestion quota/cost budget for as long as the outage lasts, and drowns out every other log signal from the same process in the meantime.
- **Suggested fix**: Log the full exception only on the first occurrence of a given failure streak (or every Nth attempt / with exponential log-throttling), and log subsequent identical retries at DEBUG or with just a one-line summary (no traceback) until either success or the final give-up.

### [MEDIUM] Core DB-execution primitive never logs the failing SQL statement — `src/pyutilz/database/db/__init__.py:266-348`

- **Category**: missing context in error logs
- **Problem**: `basic_db_execute(ex_type, statement, data=None, ...)` is the single choke point underlying `safe_execute`/`safe_execute_values`, which in turn underlie essentially every write/read helper in this module (`db_command`, `log_to_db`, `EnsurePgTableExists`, `GetIdByKeyFieldAndInsertIfNeeded`, the `regjobs_*` helpers, etc.). Every exception branch logs only the exception object, never `statement` or `data`, both of which are in scope the whole time:
  ```python
  except (OperationalError, InterfaceError) as e:
      ...
      logger.exception(e)
      logger.info("Retrying database operation (%s/%s)...", retry_count, max_retries)
      ...
  except DuplicateTable as e:
      logger.warning(e)
      return
  except InternalError as e:
      logger.exception(e)
      raise
  except Exception as e:
      logger.exception(e)
      ...
  ```
  psycopg2 exception messages for `OperationalError`/`InterfaceError`/generic connection failures typically do **not** embed the failing SQL text (unlike SQLAlchemy's wrapped `StatementError`, which does).
- **Failure scenario**: A production incident where one specific query starts failing (bad column after a migration, lock contention, a malformed value from an upstream data source) is logged as an anonymous `OperationalError`/`InternalError` traceback with no way to tell, from the log alone, which of the dozens of call sites through this shared primitive actually issued the failing statement — the operator has to correlate timestamps against application code to even guess.
- **Suggested fix**: Include the statement (and a redacted/truncated form of `data`) in every log call in this function, e.g. `logger.exception("DB execute failed (statement=%s)", statement[:500])`.

### [MEDIUM] `insert_sqllite_data` discards the traceback of its own exception — `src/pyutilz/database/db/sqlite.py:85-87`

- **Category**: wrong logging call loses traceback
- **Problem**:
  ```python
  except Exception as e:
      logger.error("Could not insert data into %s table: %s.", table_name, e)
      logger.error("Data sample: %s", values_list[-10:])
  ```
  `logger.error(...)` with `%s` on `e` records only `str(e)`; `logger.exception(...)` (or `logger.error(..., exc_info=True)`) would capture the full traceback at effectively zero extra cost, since this is already inside the live `except` block.
- **Failure scenario**: A bulk SQLite insert fails on a constraint violation or type mismatch triggered deep inside a helper called from `cursor.executemany`; the log records the DB driver's one-line error message and a sample of the offending rows, but not *where* in the call stack the failure actually originated (useful when `insert_sqllite_data` is called from several different pipelines) — an unnecessary loss for a change that costs nothing to fix.
- **Suggested fix**: `logger.exception("Could not insert data into %s table: %s.", table_name, e)`.

### [MEDIUM] `log_to_db()` silently downgrades unrecognized severity levels to INFO — `src/pyutilz/database/db/__init__.py:518,536-548`

- **Category**: wrong log level hides intended severity
- **Problem**:
  ```python
  def log_to_db(message, details=None, more_details=None, level="info", append_severity=False, ...):
      ...
      if level in ["warning", "warn"]:
          logger.warning(s); severity = cWarning; ...
      elif level == "error":
          logger.error(s); severity = cError; ...
      else:
          logger.info(s); severity = cInfo
  ```
  Any `level` value other than exactly `"warning"`/`"warn"`/`"error"` — including `"critical"`, `"fatal"`, a typo like `"Error"` (case mismatch), or any other caller-invented string — falls into the `else` branch and is logged at INFO both in the live Python logger and in the persisted DB `severity` column, with no warning that the requested level wasn't recognized.
- **Failure scenario**: A caller writes `log_to_db("disk full on worker-3", level="critical")` expecting it to stand out; it is silently recorded as `severity=1` (info) in the `logs` table and emitted via `logger.info`, indistinguishable from routine informational messages — the very message meant to be most visible is the one that gets buried.
- **Suggested fix**: Validate `level` against an explicit allow-list up front and raise/log-a-meta-warning on an unrecognized value instead of silently coercing to the least-severe branch; at minimum extend the `elif` to cover `"critical"`/`"fatal"`.

### [MEDIUM] `serialize()`'s failure log carries no identifying context — `src/pyutilz/core/serialization.py:62-89`

- **Category**: missing context in error logs
- **Problem**: `fname` (the target path or file-like object) and `obj`'s type are both known at the point of failure but never included:
  ```python
  try:
      data = pickle.dumps(obj)
      ...
      with open(fname, "wb") as f:
          f.write(data)
      ...
  except Exception as e:
      logger.exception(e)
      return None
  ```
- **Failure scenario**: A batch job calls `serialize(obj, fname=path)` in a loop over many objects/paths; one write fails (disk full, permission error, unpicklable object). The log shows a bare traceback with no `fname`/`obj` info, so identifying *which* of the batch's items failed (to retry or investigate) requires re-running under a debugger rather than reading the log.
- **Suggested fix**: `logger.exception("serialize failed (fname=%r, type=%s): %s", fname, type(obj).__name__, e)`.

### [MEDIUM] OpenRouter health-check fan-out logs one WARNING per model with no dedup for a single shared root cause — `src/pyutilz/llm/openrouter_provider/_health.py:295-313`

- **Category**: retry/fan-out log flooding
- **Problem**: `_enrich_with_health` dispatches up to `max_workers` concurrent `/endpoints` GETs across potentially hundreds of models (`list_openrouter_models` defaults to health-checking the whole catalogue). Each failure is logged independently:
  ```python
  for fut in as_completed(futs):
      ...
      try:
          endpoints = fut.result()
      except Exception as exc:
          logger.warning("Health check failed for %s (%s); excluding from results.", mid, exc)
          continue
  ```
  There is no aggregation across the fan-out — every failed future gets its own WARNING line.
- **Failure scenario**: The configured OR API key expires or is revoked. Every one of the (potentially 200+) `/endpoints` calls fails with the same 401, producing 200+ near-identical WARNING lines in a single `list_openrouter_models()` call, obscuring the actual signal ("your API key is invalid") behind sheer volume, and making it easy to miss that this is one root cause rather than 200 independent per-model problems.
- **Suggested fix**: After the fan-out completes, if failures share a common exception type/message, log one summary WARNING (`"Health check failed for %d/%d models, e.g. %s: %s"`) instead of one per model; keep the per-model detail at DEBUG.

### [LOW] Frame-introspection failures bypass the module's own configured logger — `src/pyutilz/dev/logginglib.py:120-150`

- **Category**: wrong logger instance used
- **Problem**: `initialize_function_log()`'s two `except` blocks call the **module-level `logging.exception(...)` function** (which always logs against the root logger) instead of this file's own `logger` object (which `init_logging()` reconfigures with the caller's handlers/level/format):
  ```python
  except (AttributeError, TypeError) as e:
      logging.exception(e)
  ...
  except (AttributeError, TypeError) as e:
      logging.exception(e)
  ```
  Everywhere else in this same file (`_message`, `finalize_function_log`, `debugged`, etc.) correctly uses the module's `logger`.
- **Failure scenario**: A caller that has run `init_logging(...)` to route everything through a rotating file handler + custom format never sees these two specific error paths (frame/argument introspection failures inside `initialize_function_log`) in that file — they only ever go to the root logger's default handler (console, with default formatting), inconsistent with every other log line the application produces.
- **Suggested fix**: Use `logger.exception(e)` (the module attribute) in both spots, consistent with the rest of the file.

## Things done well

- Proactive, purpose-built redaction helpers (`_redact_credential_shaped_value` in `logginglib.py`, `_redact_proxy_url`/`_redact_headers`/`_redact_proxies_dict` in `web.py`) are applied *before* values reach any log call or persisted `results_log`, not bolted on after the fact, and their docstrings explicitly record the prior vulnerability they closed.
- Retry logic consistently distinguishes permanent vs. transient failure classes for logging/control-flow purposes (e.g. `redislib.py` separates `RedisAuthenticationError` — logged once, no retry — from `RedisConnectionError` — retried; `db/__init__.py` separates `InternalError`/`DuplicateTable` from `OperationalError`/`InterfaceError`), rather than treating every exception identically.
- `web.py`'s `report_params()` and `get_new_smartproxy()`'s unconditional-after-N-attempts logging (explicitly un-gated from `verbose` specifically so a stuck wait is diagnosable "from logs alone" per its own docstring) show real thought about what must be visible by default vs. opt-in.
- The `_logproxy.py` facade-forwarding logger correctly keeps `pyutilz.text.strings.logger` monkeypatching working across the split submodules, including proper `__repr__`/`__eq__`/`__hash__` forwarding that a naive `__getattr__`-only proxy would have missed.

## Investigated, not an issue

- **`print()` statements across the package**: grepped every `print(` outside tests/docstrings. All real (non-commented, non-docstring) instances are either explicit `verbose=True`/`use_print=True` stdout-display contracts already flagged with `# noqa: T201` comments explaining the intentional design (`pandaslib/frames.py`, `text/strings/webtext.py`), or `dev/notebook_init.py`'s interactive-notebook startup banner (appropriate for that context), or `web/proxy/decodo.py`'s documented `print_usage()` CLI-style display method (which *also* logs via `_log.exception(...)` alongside the print). No stray print() silently substituting for real library logging was found.
- **Eager f-string formatting before a level check, in a hot loop**: grepped every `logger.<level>(f"...")` call. The handful found (`pandaslib/frames.py`, `pandaslib/dtypes.py`, `text/strings/jsonutils.py`) are all inside conditional branches already gated by an infrequent condition (stale-column detection, an explicit `verbose`/`if property_name not in elem` per-item branch bounded by realistic collection sizes), not bare per-iteration emissions in a genuinely hot path. No case of unconditional eager formatting burning CPU behind a disabled log level was found.
- **`connect_to_db`'s retry loop** (`db/__init__.py:174-234`): also logs `logger.exception(e)` per attempt, but unlike `redislib.py`'s `rexecute`, `max_retries` here has sane bounded-by-default call sites in practice and a clear final `"giving up after %d attempts"` message — flagged only `redislib.py` as the stronger instance since its `max_retries=None` default makes indefinite flooding the out-of-the-box behavior, not an opt-in.

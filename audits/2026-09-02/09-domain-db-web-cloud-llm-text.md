# Database / Web / Cloud / LLM / Text Domain Audit — pyutilz (2026-09-02)

## Summary

Read `CLAUDE.md` (community-health won't-fix and formatter/mypy conventions respected — nothing here proposes a repo-wide reformat or a SECURITY.md-class file), `audits/2026-07-21_full-audit/09-domain-database.md`, `audits/2026-07-21_full-audit/10-domain-web-cloud.md`, and all ten reports in `audits/2026-07-21_audit-round2/`, then every `.py` under `src/pyutilz/database/`, `src/pyutilz/web/`, `src/pyutilz/cloud/`, `src/pyutilz/llm/` and `src/pyutilz/text/` (~15.5k LOC).

**Prior-round status.** The 21 database findings and all 26 web/cloud findings from 2026-07-21 are fixed in current code (spot-verified: `sql_helpers.py:96-103`, `upsert.py:189`, `redislib.py:88-107`, `sqlite.py:56,70,91,100`, `deltalakes.py:90,105`, `cloud.py:93-98`, `browser.py:119`, `graphql.py:153`, `web.py:162,276,333,344,914`, `proxy/base.py:166,246`, `ip_check.py:398`, `configfiles.py:32`, `tokenizers.py:77-88`). No web/cloud/text finding remains open. **Two prior findings are only partially fixed and are re-raised as F02 and F08 — "still open since 2026-07-21".** Three prior fixes were incomplete in a way that produced *new* defects, filed as F05, F06 and F11.

**Reproduced offline** (`D:/ProgramData/anaconda3/python.exe`, scratch under `D:\Temp`, no real database and no network): the `timestamp_update_fields` SQL emission (F01), psycopg2's `can't adapt type 'dict'` (F03), the pool DSN swap (F04), the single-port `get_new_smartproxy` loop (F05), the 0-byte `download_to_file` (F07), `last_json_schema_applied` under concurrency (F09), `extract_json` returning non-dicts (F27), OpenAI pricing/limit table gaps (F36), the `sentences_similarity` suffix miss (F12), negative `compute_entropy_stats` (F13), the empty-username DSN and Basic-auth leaks (F14), `json_pg_dumps` on literal `\u0000` (F37), `clean_description("")` (F38), `unescape_html` (F39), `MIN_MORPHEME_LENGTH=2` `NameError` (F40), and the one-character prefix score (F41).

Counts: **1 Critical, 13 High, 27 Medium, 22 Low** (63 total).

## Findings

### F01. [Critical] `build_upsert_query` emits an `UPDATE` with no `WHERE` when `timestamp_update_fields` is used without `hash_fields` — src/pyutilz/database/db/upsert.py:220-247
- **Disposition**: OPEN
- **Category**: sql-building
- **Problem**: `join_condtion` is built only inside `if hash_fields:` (line 221); the `else` branch sets it to `None` (line 231). At line 244 `the_join_condtion = f"where {join_condtion}" if join_condtion else ""` therefore collapses to the empty string, and line 247 emits the statement with no join predicate. Reproduced output with `hash_fields` omitted: `... with tmp as (update t AS u set updated_at=c.checked_at from changed_data as c ) select count(*) from changed_data;` — versus the correct `... as c where u.id=c.id)` when `hash_fields='h'` is supplied.
- **Failure scenario**: `build_upsert_query(fields_names=['id','name'], table_name='t', conflict_fields=['id'], on_conflict_update_fields=['name'], timestamp_check_fields=['checked_at'], timestamp_update_fields=['updated_at'], history_table_name='t_hist', history_fields=['id','name'])`. In PostgreSQL an `UPDATE t AS u SET ... FROM changed_data c` with no `WHERE` is a cross join: every row of `t` has `updated_at` overwritten from an arbitrary `changed_data` row. Valid SQL, no error, whole-table silent corruption on every upsert batch — of the function's own documented "when was this element last updated" feature.
- **Suggested fix**: build `join_condtion = " and ".join(f"u.{f}=c.{f}" for f in conflict_fields)` unconditionally before the `if hash_fields:` branch (`conflict_fields` is already asserted non-empty at line 200 and identifier-validated), so the `else` path keeps the same predicate.

### F02. [High] Module-global psycopg2 cursor still shared across threads at two read sites — src/pyutilz/database/db/__init__.py:141-143, 434
- **Disposition**: OPEN
- **Category**: concurrency
- **Problem**: still open since 2026-07-21 (round 2's CRITICAL was only partially fixed). The cursor cache became `threading.local` (`:96-109`) and `basic_db_execute` now works on a genuine local, but `get_table_fields:141-143` does `cur.execute("select * from " + table + " where 0=1")` → `cur.fetchall()` → `cur.description` against the module-global `cur`, and `fetch_db_elements:434` reads `cur.description` off the same global.
- **Failure scenario**: thread A calls `get_table_fields("orders", "o")`; thread B calls `safe_execute(...)` between A's `execute()` and `fetchall()`, rebinding the global `cur`. A reads B's result buffer and returns column names for the wrong table — silently, no exception. Identical shape to the round-2 CRITICAL, via the two call sites the fix did not convert.
- **Suggested fix**: route both through the per-thread `get_cursor(get_cursor_type(None, None))`; for `fetch_db_elements`, have `db_command` pass the executed cursor's description explicitly instead of re-reading a global.

### F03. [High] `regjobs_progress`/`regjobs_finalize` bind a raw `dict` psycopg2 cannot adapt — src/pyutilz/database/db/__init__.py:1196-1212
- **Disposition**: OPEN
- **Category**: parameterization
- **Problem**: `_regjobs_update(job_name, result: dict, ...)` binds `{"result": result}` to `%(result)s` for the `last_result`/`result` columns, declared `jsonb` at `:1144,1147`. psycopg2 does not adapt `dict` without a registered adapter; `grep -rn "register_adapter|register_default_json|extras.Json" src/` returns nothing in the package. Verified against the installed driver (psycopg2 2.9.9): `ProgrammingError: can't adapt type 'dict'`.
- **Failure scenario**: `regjobs_progress("nightly_etl", {"rows": 120})` raises `ProgrammingError` out of `basic_db_execute`, so the heartbeat is never written; `regjobs_poll`'s `last_ping_at` singleton-takeover logic then sees a permanently missed ping and hands the still-running job to a second worker. All four `regjobs_*` helpers are public API and none works with the type its own signature declares.
- **Suggested fix**: wrap with `psycopg2.extras.Json(result)` in `_regjobs_update` (or `json.dumps(result)` plus a `::jsonb` cast).

### F04. [High] Connection pool silently serves a different DSN; the guard against it is unreachable — src/pyutilz/database/psycopg2_pool.py:67-79, 184-187
- **Disposition**: OPEN
- **Category**: connection-lifecycle
- **Problem**: `_ensure_pool` returns the existing `_pool` at lines 70-71 and again at 73-74, *before* any DSN comparison. The `if _pool_dsn is not None and _pool_dsn != dsn: raise ValueError(...)` at 75-76 is reachable only when `_pool is None` **and** `_pool_dsn is not None` — a state the module never produces (`_pool_dsn` is assigned at line 78 immediately after `_pool`; `_reset_pool:91-92` clears both together). `get_connection_from_pool:184-187` ignores `dsn` entirely when a pool exists. Reproduced with a stubbed `ThreadedConnectionPool`: `get_connection(dsn_B)` returns a `dsn_A` connection.
- **Failure scenario**: a process touching two databases calls `get_connection(dsn_prod)` then `get_connection(dsn_staging)` and receives a **prod** connection with no error and no log line, so writes intended for staging land in prod. The module docstring's own contract ("only ONE pool per process; call `close_pool()` before switching DSNs") is exactly what the dead guard was meant to enforce.
- **Suggested fix**: compare before returning — `if _pool is not None: if _pool_dsn != dsn: raise ValueError(...); return _pool` — on both the unlocked fast path and the double-checked path, and apply the same check in `get_connection_from_pool`.

### F05. [High] `get_new_smartproxy()` never re-rolls the port: the search loop retries one single port forever — src/pyutilz/web/web.py:532-533
- **Disposition**: OPEN
- **Category**: proxy-rotation
- **Problem**: `proxy_port` is the function *parameter*, rebound in place at line 533 under `if proxy_port is None:`. After the first iteration the guard is False forever, so the `while True` loop re-hashes the identical proxies dict, gets the identical verdict, and sleeps `delay` again. The docstring's "keeps re-rolling a random port" (line 509) never happens and the port-range / `min_idle_interval_minutes` policy is inert. Reproduced: with a `last_used_dict` reporting every proxy as recently touched, six consecutive iterations produced port `36117` every time — 1 unique port out of a 20001-37960 range.
- **Failure scenario**: a scraper with a ~18k-port pool and `min_idle_interval_minutes=10` calls `set_proxy()`; the first random port happens to be in `last_used_dict`; `get_new_smartproxy` blocks the calling thread until `max_wait_seconds` despite ~17,958 eligible ports. This is the un-fixed core of the round-1 HIGH whose documented caveat understates it.
- **Suggested fix**: use a separate loop variable — capture `fixed_port = proxy_port` before the loop, and inside compute `port = fixed_port if fixed_port is not None else <random draw>` on every iteration.

### F06. [High] `min_failed_idle_interval_minutes` is accepted, documented, threaded through three call sites, and never used — src/pyutilz/web/web.py:499, 542-545
- **Disposition**: OPEN
- **Category**: proxy-rotation
- **Problem**: the identifier does not appear anywhere in the function body after the docstring. Lines 542-545 loop over `(failed_dict, last_used_dict)` and compare **both** against `min_idle_interval_minutes` only. The three internal callers (`web.py:446, 742, 798`) all pass `min_failed_idle_interval_minutes=... else 60*24` alongside `min_idle_interval_minutes=... else 0`.
- **Failure scenario**: under default configuration (`set_params()` leaves `min_idle_interval_minutes` at `0`), `0 < 0` is False, so a port just marked *failed* by `handle_blocking()` is re-selected on the very next rotation. The advertised 24-hour failed-proxy cooldown does not exist; a blocked exit IP is handed straight back to the retry that was rotating away from it.
- **Suggested fix**: select the threshold per dict — `threshold = min_failed_idle_interval_minutes if dict_to_check is failed_dict else min_idle_interval_minutes` — and compare against that.

### F07. [High] `download_to_file()` leaves a silently truncated 0-byte file and reports success — src/pyutilz/web/web.py:974-1004
- **Disposition**: OPEN
- **Category**: http-correctness
- **Problem**: a `requests` response body is a single-use stream. When the body breaks mid-transfer, the inner retry loop calls `request.iter_content()` again (line 981) on an already-consumed urllib3 stream, which yields nothing; `open(filename,"wb")` has already truncated the file, the `else: break` at 990-991 is taken, and the function returns `None` — the same value the success path returns. Reproduced with a response raising `ChunkedEncodingError` after 10 bytes: `exists: True size: 0`.
- **Failure scenario**: a 100 MB dataset download hits a connection reset mid-transfer and leaves a 0-byte file that downstream code loads as a valid (empty) artifact, with no exception and no log distinguishing it from a completed download.
- **Suggested fix**: move `requests.get(...)` inside the retry loop so each attempt re-issues the HTTP request (ideally with a `Range:` header to resume), and delete the partial file on final failure as the `else:` clause at 992-1002 already does.

### F08. [High] OpenRouter's entire per-call metadata set is still the cross-request race round 2 described — src/pyutilz/llm/openrouter_provider/_provider.py:170-192, 410-511, 728-772
- **Disposition**: OPEN
- **Category**: concurrency
- **Problem**: still open since 2026-07-21 (partially fixed). `PerCallAttr` covers only the 14 attributes listed in `base.py:484-499` `_PERCALL_METADATA_ATTRS`. OpenRouter's `last_actual_cost_usd`, `last_generation_id`, `last_upstream_provider`, `last_upstream_model`, `last_native_finish_reason`, `last_cache_write_tokens`, `last_cache_hit_tokens`, `last_audio_tokens`, `last_upstream_inference_cost_usd`, `last_cache_discount_usd`, `last_is_byok`, `last_response_cache_source_id` and `last_web_search_citations` remain plain instance attributes, and `_reset_per_call_state()` runs at the *start* of each call. `LLMProvider.generate_batch` (`base.py:611`) fires N concurrent `self.generate()` tasks on one instance, so request B zeroes request A's in-flight values; none of these names is carried in the yielded batch dict either.
- **Failure scenario**: a cost-accounting loop over `generate_batch()` reading `p.last_actual_cost_usd` / `p.last_call_summary()` attributes the wrong upstream provider, wrong generation id and wrong billed USD to each request id. There is no race-free way to obtain them under batching at all.
- **Suggested fix**: convert these to `PerCallAttr` and add them to `_PERCALL_METADATA_ATTRS`, or have `_capture_percall_metadata` merge a provider hook returning `last_call_summary()`.

### F09. [High] `_last_json_schema_applied` is clobbered across concurrent calls, so the strict-schema guarantee flag lies — src/pyutilz/llm/openai_compat.py:381, 384, 396-398
- **Disposition**: OPEN
- **Category**: concurrency
- **Problem**: a plain instance attribute, unlike its `PerCallAttr` neighbours at `:182-185`. Reproduced with two concurrent tasks (A passing a strict schema, B passing none): `A(with schema): last_json_schema_applied=False`, `B(no schema): last_json_schema_applied=False` — expected `True`/`False`.
- **Failure scenario**: the docstring at `:377-379` promises this flag lets a caller "tell a guaranteed-shape response from a merely-hopeful one". Under concurrency it reports `False` for a call whose enum genuinely was constrained, and can equally report `True` for one that was not — a caller that skips enum validation on `True` accepts unvalidated output.
- **Suggested fix**: make it a `PerCallAttr` and add it to `_PERCALL_METADATA_ATTRS`.

### F10. [High] Streaming re-records usage on every chunk carrying a `usage` block, multiplying reported spend — src/pyutilz/llm/openai_compat.py:556-561, 600-649
- **Disposition**: OPEN
- **Category**: token-accounting
- **Problem**: `_track_streaming_usage` → `_record_usage` performs `total_prompt_tokens += ...`, `_call_count += 1`, and `_track_provider_specific_usage` (which does `total_actual_cost_usd += cost`, `_provider.py:425`). The loop invokes it for every chunk whose `usage` is truthy — the comment at `:556-558` states this explicitly. Several OpenAI-compatible upstreams emit cumulative usage on more than the final chunk, and a stream retried by the `while True` at `:534` after already receiving a usage chunk records it a second time.
- **Failure scenario**: `get_session_cost()["total_cost_usd"]` and OpenRouter's `actual_cost_usd` over-report spend by a multiple, and `calls` exceeds the real request count — silently, since nothing cross-checks against the provider's own billing.
- **Suggested fix**: accumulate `usage` into a local inside the loop and call `_record_usage` once after the stream closes, or guard with a per-stream `usage_recorded` flag.

### F11. [High] LRU eviction closes an httpx client another caller may still be using — src/pyutilz/llm/factory.py:45-73, 196-198
- **Disposition**: OPEN
- **Category**: resource-lifecycle
- **Problem**: eviction is keyed purely on cache size (128), not liveness. `get_llm_provider()` hands out shared instances and keeps no refcount, so `_schedule_provider_close` → `OpenAICompatibleProvider._close` → `await self._client.aclose()` (`openai_compat.py:400-402`) can fire on a provider a coroutine obtained earlier and is still using. Regression introduced by the round-2 unbounded-cache fix.
- **Failure scenario**: a service routing across more than 128 distinct `(provider, kwargs)` combinations raises `RuntimeError: Cannot send a request, as the client has been closed.` mid-batch, on a provider the caller legitimately holds.
- **Suggested fix**: only close on eviction when no strong external reference remains (weakref, or an in-flight counter incremented in `generate`); otherwise drop the LRU entry without closing and let `_uncached_providers` + atexit handle teardown.

### F12. [High] Sliding-window loop misses the last window, so any exact suffix match scores as unrelated — src/pyutilz/text/similarity.py:265 (and numba twins :493, :674, :975, :1067)
- **Disposition**: OPEN
- **Category**: off-by-one
- **Problem**: `for k in range(long_len - short_len)` — there are `L-S+1` windows of length `S`, so the final (suffix) window is never tested. The same off-by-one is present in all four numba matrix-fill kernels, so the numba/pure-Python differential tests agree while both are wrong. Reproduced: `sentences_similarity(["bcd"],["abcd"])` → `0.5417` (only `'abc'` is ever compared; `'bcd'` at distance 0 never is), correct value `0.875`; `sentences_similarity(["MADRID"],["REALMADRID"])` → `0.6333`, should be `0.80`.
- **Failure scenario**: in the function's documented domain (team/club name matching), a word that is an exact *suffix* of another scores as unrelated while the same word as a *prefix* scores ~0.96, so `MADRID` vs `REALMADRID` falls below any sane threshold. It also defeats the `if t_sim == 0: break` perfect-substring fast path for suffixes.
- **Suggested fix**: `range(long_len - short_len + 1)` in all five loops (`short_len == long_len` never reaches this branch, so no degenerate case is added).

### F13. [High] `compute_entropy_stats` returns a large negative "entropy" — src/pyutilz/text/strings/textentropy.py:108
- **Disposition**: OPEN
- **Category**: numerical-correctness
- **Problem**: `entropy(stats, len(stats))` normalizes by the number of *distinct prefixes* instead of the total observation count. Shannon entropy is never negative. Reproduced on `"abcabcabcabc"`: `compute_entropy_stats(t, 0)` → `(-43.0195..., 1.58496...)` (at order 0, `stats` is `Counter({(): 12})` so `len(stats)==1` and the term becomes `-12*log2(12)`); at order 1 it returns `-1.1068` where the correct value is `1.5726`. The entropy *rate* (second element) is correct — only `sample_raw_entropy` is wrong, silently, at every order.
- **Failure scenario**: any caller thresholding on raw entropy (randomness/gibberish detection is the module's purpose) compares a negative number against a positive cutoff and classifies every input the same way. This is public API, re-exported from `pyutilz.text.strings`.
- **Suggested fix**: `entropy(stats, sum(stats.values()))`.

### F14. [High] Two secret shapes pass through `redact_secrets`/`sanitize_dsn` unredacted — src/pyutilz/text/secrets_scrub.py:35, 44, 69
- **Disposition**: OPEN
- **Category**: secret-leak
- **Problem**: (a) `DSN_PASSWORD_RE = r"(://[^:]+:).+(@)"` and `DSN_SCHEME_RE`'s `[^\s:@/]+:` both require at least one username character, so an empty-username DSN is not matched. Reproduced: `redis://:sup3rs3cret@localhost:6379/0` is returned **verbatim** by both functions, as are `postgresql://:sup3rs3cret@db.internal:5432/prod` and `amqp://:pw@rabbit`. Empty-user is the standard Redis/valkey password-only URL form, i.e. the common case for that scheme. (b) `TELEGRAM_TOKEN_RE` knows only `Bearer`, so `Authorization: Basic dXNlcjpwYXNz` falls through to `SECRET_KEY_VALUE_RE`, whose `\S+` consumes only the word `Basic` — output is `Authorization=*** dXNlcjpwYXNz`, printing the base64 `user:pass` in full.
- **Failure scenario**: a connection failure to a password-authenticated Redis, or an HTTP error carrying a Basic-auth header, is passed through `redact_secrets()` and written to the log with the live credential intact — precisely the outcome this module exists to prevent.
- **Suggested fix**: `[^:@/]*` in `DSN_PASSWORD_RE` and `[^\s:@/]*` in `DSN_SCHEME_RE`; add a `\bBasic\s+\S+` alternative and extend the negative lookahead to `(?!(?:Bearer|Basic)\b)`.

### F15. [Medium] `log_to_db(level=None)` / `level=""` silently discards the message entirely — src/pyutilz/database/db/__init__.py:583
- **Disposition**: OPEN
- **Category**: silent-failure
- **Problem**: the whole function body is nested inside `if level:`. A falsy level produces no Python log record, no DB row, no return value and no warning. Verified with `safe_execute` stubbed: `log_to_db("boom", level=None)` and `log_to_db("boom2", level="")` both leave the recorded-call list empty. Round 2's fix added a warning for *unrecognized* levels; it does not cover falsy ones.
- **Failure scenario**: a wrapper forwarding its own optional level (`log_to_db(msg, level=cfg.get("db_log_level"))`) with the key absent silently drops every audit and error message this module exists to persist.
- **Suggested fix**: normalize at entry — `level = level or "info"` — and drop the `if level:` wrapper.

### F16. [Medium] `build_upsert_query` builds invalid SQL when `history_table_name` is given with empty `history_fields` — src/pyutilz/database/db/upsert.py:181-206
- **Disposition**: OPEN
- **Category**: sql-building
- **Problem**: the `returning` clause on the fresh-data CTE is emitted only under `if len(history_fields) > 0:` (line 181), but the history INSERT at 199-206 is emitted whenever `history_table_name` is truthy, with no such guard. Reproduced output contains `insert into t_hist() select  from fresh_data u` — an empty column list and an empty select list, a PostgreSQL syntax error, against a `fresh_data` CTE that has no `RETURNING` and so is not selectable.
- **Failure scenario**: `history_fields` defaults to `[]` (lines 46-47), so passing only `history_table_name` — the natural way to turn history on — produces unrunnable SQL that fails at execution time rather than at build time.
- **Suggested fix**: raise `ValueError` up front when `history_table_name` is set and `history_fields` is empty.

### F17. [Medium] `db_command(mode="insert", set_fields=None)` crashes with an opaque `TypeError` — src/pyutilz/database/db/__init__.py:457-473
- **Disposition**: OPEN
- **Category**: input-validation
- **Problem**: the round-1 fix added a guard requiring non-empty `where_fields` for `select`/`update` (471-473), but `insert` consults only `set_fields` and has no equivalent guard; the entry check at 457 passes as long as one of the two is non-None. Verified: `db_command("insert", "t", where_fields=["id"], set_fields=None, source={"id": 1})` → `TypeError: 'NoneType' object is not iterable` from `construct_templates_and_values`. `set_fields=[]` likewise passes and would build `insert into t () values ()`.
- **Failure scenario**: a caller building `set_fields` from a filtered dict that happens to be empty gets an uninformative `TypeError` from two frames down instead of the module's own diagnostic.
- **Suggested fix**: mirror the existing guard — `if mode == "insert" and not set_fields: logger.error(...); return`.

### F18. [Medium] `release_connection` silently drops the connection when the pool has been closed — src/pyutilz/database/psycopg2_pool.py:238-240
- **Disposition**: OPEN
- **Category**: connection-lifecycle
- **Problem**: `pool = _pool; if pool is not None: pool.putconn(conn)`. When `_pool` is `None` — after `close_pool()`, and transiently inside `get_connection`'s retry path which calls `_reset_pool()` at line 165 — the connection is neither returned nor closed. It is abandoned with its server-side backend still open until GC or process exit.
- **Failure scenario**: shutdown code calls `close_pool()` while a `managed_connection(...)` block is still unwinding; the `finally` at line 257 takes the `pool is None` path and leaks that backend. Repeated across a restart loop this exhausts `max_connections` on the server.
- **Suggested fix**: add `else: try: conn.close() except Exception: logger.warning(...)` so a connection with nowhere to go is closed.

### F19. [Medium] `auto_commit` is accepted by three public functions and does nothing — src/pyutilz/database/db/__init__.py:294, 327-328, 418, 423
- **Disposition**: OPEN
- **Category**: transaction-correctness
- **Problem**: `basic_db_execute(..., auto_commit=True, ...)` never reads the parameter; the only `conn.commit()` calls are commented out at lines 327-328 and 352. `safe_execute`/`safe_execute_values` forward it faithfully to nowhere. The connection is set to `ISOLATION_LEVEL_AUTOCOMMIT` at line 213, so every statement commits regardless of the flag.
- **Failure scenario**: a caller writing `safe_execute(insert_a, auto_commit=False); safe_execute(insert_b, auto_commit=False); conn.commit()` believes it has a two-statement transaction. Both statements have already committed independently; if `insert_b` fails, `insert_a` is permanently in the database with no rollback possible.
- **Suggested fix**: either honour `auto_commit=False` by switching the connection off autocommit for the call, or remove the parameter behind a deprecation shim; at minimum log a warning (or raise `NotImplementedError`) when it is False.

### F20. [Medium] `get_url()`'s rate-limit branch retries with zero delay by default — src/pyutilz/web/web.py:781-806
- **Disposition**: OPEN
- **Category**: rate-limiting
- **Problem**: the exception branch (line 758) and the generic error-status branch (line 820) both received the round-1 `if delay: sleep(delay*random())` fix. The 429 branch did not: with a truthy `proxy_server_snapshot` the only sleep is gated on `ratelimited_proxy_sleep_interval`, whose default is `0` (line 625), after which the loop jumps straight back to the top.
- **Failure scenario**: a 429-ing endpoint behind a configured proxy is hit `max_retries=10` times back to back with no pause — the thundering-herd shape the other two branches were fixed for — and, per F05, the "fresh" proxy pick can be the same port.
- **Suggested fix**: add `if delay: sleep(delay * random())` after the proxy rotation at line 804, and honour a `Retry-After` header when present.

### F21. [Medium] `LoginAndGetCookies()` writes cookies and Bearer tokens into the shared `basic_headers` global — src/pyutilz/web/browser.py:75-76, 525-531
- **Disposition**: OPEN
- **Category**: shared-state
- **Problem**: verified `browser.headers is browser.basic_headers` → `True`. Line 525 rebinds `headers` to the *same dict object*, then lines 527/529/531 mutate it in place.
- **Failure scenario**: `basic_headers`, documented as the neutral default header set, permanently carries the last session's `cookie` and `authorization`, so any other consumer of it leaks them; `default_headers=True` never yields clean headers on a re-login; and if a later login produces no `oauth2_global_js_token`, the previous account's `authorization: Bearer ...` stays in the dict and is sent with the new session's requests — cross-account request attribution.
- **Suggested fix**: `headers = dict(basic_headers)` at line 525, and `headers.pop("authorization", None)` before the conditional at line 530.

### F22. [Medium] `CachedHttpClient`'s cache key ignores payload kind, so `get_json` returns a permanent cached `None` — src/pyutilz/web/cached_client.py:101-104, 160-164, 196-204
- **Disposition**: OPEN
- **Category**: caching
- **Problem**: `_cache_path` hashes only the URL (plus tag). `get_text` writes `{"url":..., "text":...}` while `get_json` reads `payload.get("data")`, which is absent → `None`. Reproduced: `get_text(url, tag)` → `'{"a": 1}'`, then `get_json(url, tag)` on the same url+tag → `None`. The docstring defines `None` as "permanent failure", and with `cache_failures=True` the entry is then rewritten as a cached negative.
- **Failure scenario**: a pipeline that fetches a document as text (to hash it) and later as JSON under the same tag gets a permanent `None` for a URL that answers correctly, and the poison persists on disk across runs.
- **Suggested fix**: include the payload kind in the key — `_cache_path(url, tag, kind)` hashing `f"{kind}:{url}"` — or have each getter fall back to the other key on a shape mismatch.

### F23. [Medium] `CachedHttpClient.get_json()` raises `UnicodeDecodeError` on a non-UTF-8 body instead of returning `None` — src/pyutilz/web/cached_client.py:174-176
- **Disposition**: OPEN
- **Category**: encoding
- **Problem**: only `json.JSONDecodeError` is caught at line 175. `UnicodeDecodeError` is a `ValueError` but not a `JSONDecodeError`, so it escapes. Reproduced with a latin-1 body `b'{"name": "Caf\xe9"}'` → `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe9`. The sibling `get_text` at line 209 correctly uses `errors="replace"`.
- **Failure scenario**: one mis-encoded API response aborts an entire ingestion batch, contradicting the docstring's "returns `None` on a permanent failure ... neither should abort a batch job".
- **Suggested fix**: `raw.decode("utf-8", errors="replace")` at line 174, or catch `(json.JSONDecodeError, UnicodeDecodeError)`.

### F24. [Medium] `PortHealthTracker` never trims on success — unbounded memory growth on a healthy pool — src/pyutilz/web/proxy/base.py:207-211
- **Disposition**: OPEN
- **Category**: resource-lifecycle
- **Problem**: `report_error` calls `_trim_all` (line 203); `report_success` does not, and `_PortStats.record` appends unconditionally. Reproduced with `window=0.001s`: 100,000 `report_success(7)` calls leave `_ports[7].total == 100000` — every sample retained far past the window.
- **Failure scenario**: a long-running scraper whose ports mostly succeed accumulates one tuple per request per port for the process lifetime (100k requests is tens of MB), and the first subsequent `_maybe_ban` computes `error_rate` over a window that was, until that instant, unbounded.
- **Suggested fix**: call `self._trim_all(now)` in `report_success` too, or trim per-port inside `_record_unlocked`.

### F25. [Medium] `DecodoProvider.get_traffic()` fetches only page 1 and reports the truncated sum as the total — src/pyutilz/web/proxy/decodo.py:365-382, 412-429
- **Disposition**: OPEN
- **Category**: pagination
- **Problem**: `"page": 1` is hardcoded (line 371) with `limit=500` (line 339), and `_parse_traffic_response` sums `total_requests`/`total_bytes` over just those rows (lines 426-427). No response-level page count or total is read, and nothing signals truncation to the caller.
- **Failure scenario**: `get_traffic(days=90, group_by="target")` on an account with more than 500 distinct targets returns a `DecodoTrafficReport` whose `total_gb` understates real usage, and `print_usage()` prints it as authoritative — a quota or billing decision made on silently partial data.
- **Suggested fix**: loop `page` until a short page is returned (or the API's reported page count is exhausted), accumulating rows; at minimum log a warning when `len(rows) == limit`.

### F26. [Medium] `ClaudeCodeProvider` re-uses the previous call's `ResultMessage`, double-counting cost and tokens — src/pyutilz/llm/claude_code_provider.py:377-396, 512
- **Disposition**: OPEN
- **Category**: token-accounting
- **Problem**: `_last_result_message` is assigned only at line 512 and, per grep, never reset anywhere.
- **Failure scenario**: call 1 yields `ResultMessage(total_cost_usd=0.42, usage=...)`. Call 2 produces no `ResultMessage` (SDK error path, or the CLI fallback at line 365) — `rm_usage` is still call 1's object, so `total_cost_usd += 0.42` again, `total_cache_*` are re-added, and `_last_usage` reports call 1's token counts as call 2's. `get_session_cost()` over-reports monotonically, and `claude-code` is the factory's default provider.
- **Suggested fix**: `self._last_result_message = None` at the top of `generate()` (or of `_generate_sdk`).

### F27. [Medium] `extract_json` returns non-dict values despite its `-> dict[str, Any]` contract — src/pyutilz/llm/base.py:336, 368, 384, 406
- **Disposition**: OPEN
- **Category**: type-contract
- **Problem**: the fence regex at line 363 explicitly accepts `[.*?]`, and three of the five paths `return json.loads(...)` unchecked; only path 4 has an `isinstance(obj, dict)` guard at line 401. Reproduced: a ```json-fenced `[{"a":1}]` → `[{'a': 1}]`; `'[1,2,3]'` → `[1, 2, 3]`; `'"just a string"'` → `'just a string'`; `'42'` → `42`.
- **Failure scenario**: a model asked for JSON returns a top-level array (very common under "respond with valid JSON only"); `generate_json()` hands back a `list`, and the caller's first `result["field"]` raises `TypeError: list indices must be integers` far from the parse site rather than the `JSONParsingError` its retry layer catches.
- **Suggested fix**: after each parse, `if not isinstance(obj, dict): raise JSONParsingError(...)` — or widen the annotation and document the contract.

### F28. [Medium] `GeminiProvider.generate_json()` never uses Gemini's native JSON mode — src/pyutilz/llm/gemini_provider.py:157-163, 175, 188
- **Disposition**: OPEN
- **Category**: structured-output
- **Problem**: `supports_json_mode()` returns `True` and `generate()` maps `json_mode` to `response_mime_type` (line 188), but Gemini does not override `generate_json`, so `base.generate_json` → `_generate_json_via(prompt, system, temperature, max_tokens)` (`base.py:568`, `:534-553`) forwards no `generate_kwargs` and `json_mode` stays `False`.
- **Failure scenario**: a caller branching on `supports_json_mode()` and calling `generate_json()` believes generation is constrained to JSON; it is prompt-steering only, and prose-wrapped output that `extract_json` mis-parses is blamed on the model.
- **Suggested fix**: override `generate_json` on `GeminiProvider` to pass `json_mode=True`, mirroring `OpenAICompatibleProvider.generate_json` (`openai_compat.py:801-818`).

### F29. [Medium] Gemini reports truncation and empty completions as safety blocks — src/pyutilz/llm/gemini_provider.py:255-268
- **Disposition**: OPEN
- **Category**: error-classification
- **Problem**: `if not text_out: raise LLMSafetyBlockError("likely safety block")` at lines 255-262 executes *before* the `MAX_TOKENS` check at 263-268, making that check unreachable whenever the truncated response contains no text.
- **Failure scenario**: a Gemini model with thinking enabled and a tight `max_tokens` returns `finish_reason=MAX_TOKENS` with empty `text`. The caller gets `LLMSafetyBlockError` — documented at `exceptions.py:35-42` as "do not retry" — instead of `LLMTruncationError`, whose whole contract (`exceptions.py:45-49`) is "double max_tokens and re-issue". The recoverable call is permanently abandoned and logged as a policy refusal.
- **Suggested fix**: move the `"MAX_TOKENS" in _fr` check above the empty-text checks.

### F30. [Medium] `AnthropicProvider.get_session_cost()` omits `total_cost_usd`; Gemini has no `get_session_cost` at all — src/pyutilz/llm/anthropic_provider.py:357-368, src/pyutilz/llm/gemini_provider.py:81-88
- **Disposition**: OPEN
- **Category**: api-consistency
- **Problem**: Anthropic returns `input_cost_usd`/`output_cost_usd` but no `total_cost_usd`, while `OpenAICompatibleProvider.get_session_cost` (`openai_compat.py:843-853`) and `ClaudeCodeProvider.get_session_cost` (`claude_code_provider.py:300-312`) both return it. `GeminiProvider` defines no `get_session_cost`, and its `_CACHE_HIT_COST` table at `gemini_provider.py:81-88` is unreferenced dead code.
- **Failure scenario**: provider-agnostic spend reporting (`p.get_session_cost()["total_cost_usd"]`) raises `KeyError` for Anthropic and `AttributeError` for Gemini, breaking a multi-provider cost dashboard on exactly two of six providers.
- **Suggested fix**: add `"total_cost_usd": input_cost + output_cost` to the Anthropic dict; add a `get_session_cost` to Gemini that uses the existing `_CACHE_HIT_COST` table.

### F31. [Medium] Anthropic drops the partial text on truncation, and `content[0].text` can raise — src/pyutilz/llm/anthropic_provider.py:237-245
- **Disposition**: OPEN
- **Category**: error-handling
- **Problem**: (a) `LLMTruncationError(...)` is raised without `partial_text=`, though `result_text` is in hand at line 237; `exceptions.py:54-58` explains exactly why that field exists ("a caller that catches it to keep what was already paid for ... had nothing to keep") and `openai_compat.py:797` passes it. (b) `result_text = response.content[0].text` raises `IndexError` on an empty `content` list and `AttributeError` when block 0 is a `thinking`/`tool_use` block — reachable precisely when extended thinking consumes the whole budget, i.e. the `max_tokens` case.
- **Failure scenario**: a long extraction truncated at `max_tokens` either discards paid-for output the caller was designed to salvage, or surfaces an opaque `IndexError` instead of the typed `LLMTruncationError` the retry layer catches.
- **Suggested fix**: pass `partial_text=result_text or ""`, and guard the content access with `if not response.content: raise LLMTruncationError(..., partial_text="")`.

### F32. [Medium] `ClaudeCodeProvider` silently ignores `max_tokens` and `temperature` — src/pyutilz/llm/claude_code_provider.py:363, 365, 540-570
- **Disposition**: OPEN
- **Category**: parameter-handling
- **Problem**: `_generate_sdk(prompt, system)` drops both; the CLI argv builder never carries either; `generate()` never calls `fit_max_tokens_to_context`. Every other provider honours both.
- **Failure scenario**: `claude-code` is the factory's **default** provider (`factory.py:101`). A caller setting `max_tokens=32000` for a long extraction, or `temperature=0.0` for determinism, gets neither, with no warning — so a determinism requirement silently does not hold.
- **Suggested fix**: forward both to `ClaudeCodeOptions`/CLI flags where supported, and log a one-time warning where the backend cannot honour them.

### F33. [Medium] `ClaudeCodeProvider.generate_json` bypasses the shared parser and raises the wrong exception type — src/pyutilz/llm/claude_code_provider.py:704-732
- **Disposition**: OPEN
- **Category**: error-classification
- **Problem**: it re-implements parsing with `re.search(r"\{[\s\S]*\}", text)` — greedy, spanning from the first `{` to the **last** `}`, merging two adjacent objects into invalid JSON — never runs `is_llm_refusal` (`base.py:205`), and raises a bare `ValueError` instead of `JSONParsingError`/`LLMRefusalError`.
- **Failure scenario**: a caller with `except JSONParsingError: retry` / `except LLMRefusalError: fallback` (the documented contract at `exceptions.py:21-28`) catches neither for the default provider: refusals are retried forever and parse errors escape as untyped `ValueError`.
- **Suggested fix**: delegate to `self.extract_json(text, self._provider_display_name)` as `base._generate_json_via` does.

### F34. [Medium] OpenRouter catalogue fetch still blocks the event loop on several paths — src/pyutilz/llm/openrouter_provider/_catalogue.py:63-67, 83-102
- **Disposition**: OPEN
- **Category**: async-sync-mixing
- **Problem**: `_catalogue.py:63-67` holds `_MODELS_LOCK` across a blocking `httpx.get(timeout=10)`. `_ensure_catalogue_warm_async` only pre-warms from `_async_prepare` before the max-token computation, but the same sync fetch remains reachable from the event loop via `supports_json_mode()` (`_provider.py:279`), `supports_json_schema()` (`:308`, called from `_response_format` *after* the warm), `_per_token_cost_pair` through `estimate_cost`/`get_session_cost` (`:513-519`), and `_resolve_model_limits` when the TTL expires between the warm and the property read.
- **Failure scenario**: on a 300s TTL rollover, one `estimate_cost()` or `get_session_cost()` call from async code stalls the entire event loop — every other in-flight request — for up to 10 seconds, the exact bug `_ensure_catalogue_warm_async`'s docstring claims to close.
- **Suggested fix**: also warm before `_response_format`, and make `estimate_cost`/`get_session_cost` read only an already-cached catalogue (returning zeros on a miss) rather than fetching.

### F35. [Medium] `AnthropicProvider` layers tenacity on top of the SDK's own retries, with no explicit timeout — src/pyutilz/llm/anthropic_provider.py:83, 144-159
- **Disposition**: OPEN
- **Category**: retry-policy
- **Problem**: `anthropic.AsyncAnthropic(api_key=...)` is constructed with neither `max_retries=` nor `timeout=`, while the `@retry` decorator uses `INFINITE_RETRY_KWARGS` (up to 50 attempts, `_retry.py:24`). Every other provider pins its timeout explicitly (`openai_compat.py:197-205` plus `_get_timeout` overrides).
- **Failure scenario**: each tenacity attempt is itself several SDK-internal retries, so a sustained 529 produces far more upstream calls than `PYUTILZ_LLM_MAX_RETRIES` documents (`_retry.py:7-8`), and the effective per-attempt timeout is the SDK default rather than anything this package controls.
- **Suggested fix**: construct with `max_retries=0` and an explicit per-model `timeout=`, leaving `_retry.py` as the single retry policy.

### F36. [Medium] OpenAI cache-hit pricing and limit tables have gaps that under-report cost and budget — src/pyutilz/llm/openai_provider.py:49-66, 204-208; src/pyutilz/llm/openai_compat.py:347, 356
- **Disposition**: OPEN
- **Category**: cost-accounting
- **Problem**: running the shipped tables: `_PRICING` keys missing from `_CACHE_HIT_COST` are `['gpt-5-codex', 'gpt-5.1-codex', 'o1-pro']`; keys missing from `_MAX_TOKENS`/`_CONTEXT_WINDOW` are `['gpt-5-codex', 'gpt-5.1-codex']`. The fallback resolves `o1-pro` (input $150.0/1M in this same table) to a cache-hit rate of $0.025/1M borrowed from `gpt-5-mini`. Separately, `max_output_tokens`/`context_window` use exact `dict.get` rather than the `longest_prefix_lookup` helper at `base.py:166` (used only by Anthropic), so any dated snapshot id falls back to 16,384 / 128,000. These are the repo's own tables; no claim is made here about the vendor's currently published prices.
- **Failure scenario**: `get_session_cost()` on a heavily cache-hitting `o1-pro` session reports cached input at a rate three orders of magnitude below the model's own input price, and a dated `gpt-5-mini-2026-...` id silently receives an 8x-too-small output budget and a 3x-too-small context window, so `fit_max_tokens_to_context` truncates requests that would have fit.
- **Suggested fix**: add the three missing `_CACHE_HIT_COST` entries and the two missing `_MAX_TOKENS`/`_CONTEXT_WINDOW` entries, and route both properties through `longest_prefix_lookup`.

### F37. [Medium] `json_pg_dumps` raises `JSONDecodeError` on any value containing the literal text `\u0000` — src/pyutilz/text/strings/jsonutils.py:274
- **Disposition**: OPEN
- **Category**: encoding
- **Problem**: `raw.replace("\\u0000", "")` operates on serialized JSON text and cannot distinguish the *escape* `\u0000` from a literal backslash followed by `u0000`. Reproduced: `json_pg_dumps({"a": "path\\u0000literal"})` → `JSONDecodeError: Invalid \escape: line 1 column 11`; a doubled backslash variant → `Unterminated string`.
- **Failure scenario**: any code snippet, regex, or Windows-style path stored in a jsonb payload crashes the serializer on valid input. It fails loudly rather than corrupting, but a scraper writing arbitrary text to jsonb dies on a legal row.
- **Suggested fix**: strip real NULs from the *object* before serialization (extend `_normalize_nonfinite_floats` to also apply `s.replace("\x00","")` to `str` values) instead of string-editing the encoded JSON.

### F38. [Medium] `clean_description("")` and whitespace-only input crash — src/pyutilz/text/strings/webtext.py:29-46, 375
- **Disposition**: OPEN
- **Category**: edge-case
- **Problem**: `remove_videos` guards its whole body with `if text:` and has no `else`, so it returns `None` for `""`/`None` instead of the input. `clean_description` chains that straight into `fix_broken_sentences` → `sentencize_text`, whose first statement dereferences `text`. Reproduced: `remove_videos("")` → `None`; `clean_description("")`, `clean_description("   ")` and `clean_description("\t")` all raise `TypeError: argument of type 'NoneType' is not iterable`.
- **Failure scenario**: whitespace-only reaches this because `fix_html` strips to `""`. This is the module's top-level entry point for scraped text, where empty and whitespace-only rows are routine, so one blank listing aborts the batch.
- **Suggested fix**: `return text` unconditionally at the end of `remove_videos`, plus an `if not text: return text` guard at the top of `sentencize_text` (line 375's `text[-1]` also raises on `""` when called directly).

### F39. [Medium] `unescape_html` decodes only three of the entity set web text actually contains — src/pyutilz/text/strings/webtext.py:63-65
- **Disposition**: OPEN
- **Category**: encoding
- **Problem**: `xml.sax.saxutils.unescape` handles only `&amp;`, `&lt;`, `&gt;`, while the docstring promises "Decode HTML/XML entities". Reproduced: `unescape_html("Caf&eacute; &amp; bar &#39;x&#39; &nbsp;")` → `'Caf&eacute; & bar &#39;x&#39; &nbsp;'`; `unescape_html("&quot;hi&quot;")` → unchanged.
- **Failure scenario**: `&#39;`, `&quot;`, `&nbsp;` and `&eacute;` are among the most frequent entities in scraped HTML, and this module exists specifically for scraped web text — downstream tokenizers then treat `&#39;` as three tokens and accented words never normalize.
- **Suggested fix**: use `html.unescape` (stdlib, full HTML5 named + numeric set).

### F40. [Medium] `MIN_MORPHEME_LENGTH > 1` raises `NameError` — src/pyutilz/text/tokenizers.py:77, 157-166
- **Disposition**: OPEN
- **Category**: edge-case
- **Problem**: `MIN_MORPHEME_LENGTH` is a documented class-attribute knob, and the method's own comment at line 127 states the intended value is 2. But `FIRSTLETTER_CAPITAL`/`ALLLETTERS_CAPITAL` are assigned only inside the `if j == 1:` branch (lines 160-166), which never executes when the loop starts at `j=2`; the `else` branch then reads them. Reproduced against a standalone replica of the exact `(i, j)` loop with `MIN=2`: `NameError: name 'FIRSTLETTER_CAPITAL' is not defined` on the first word.
- **Failure scenario**: only the shipped default `1` works; any caller taking the documented knob at its word crashes on the first token.
- **Suggested fix**: initialize both flags from `word[i]` before the `j` loop rather than inside `j == 1`.

### F41. [Medium] A single-character prefix overlap scores 0.91+ — src/pyutilz/text/similarity.py:245-249 (and :466, :646, :947, :1039)
- **Disposition**: OPEN
- **Category**: algorithm-correctness
- **Problem**: the prefix rule `if a[:lminLen] == b[:lminLen]: sim = 0.9 + 0.1*lminLen/t` is evaluated *before* the `lminLen < cMinLenTHreshold` guard, so it fires on a one-character overlap. Reproduced: `sentences_similarity(["A"],["ANDERSON"])` → `0.9125`; `sentences_similarity(["REAL"],["REALLYBIGCLUB"])` → `0.9308`. The in-code comment on the same branch (the "Almeria B" vs "Al-Budaiya" note) shows this over-scoring is precisely what the rule was meant to avoid.
- **Failure scenario**: an unrelated one-letter token scores higher against any word starting with that letter (0.9125) than a correct suffix match scores (0.633, per F12) — so name matching prefers a spurious pair over the right one.
- **Suggested fix**: move the `lminLen < cMinLenTHreshold` check above the prefix branch, and require `lminLen >= 2` for the 0.9 floor.

### F42. [Low] `suggest_json_optimization`'s `path` argument is spliced into SQL unvalidated and undocumented — src/pyutilz/database/db/__init__.py:1084-1093
- **Disposition**: OPEN
- **Category**: sql-building
- **Problem**: `table`/`table_field` are validated (1070-1071) and each `field` is validated (1081) even though single-quoted, but `path` goes straight into `full_path = table_field + "->" + path` (1087) and then into the query text at 1092, three times. Unlike `showcase_table`/`select`/`read_table_into_dict_reversed`, this function's docstring carries no "trusted input only" warning.
- **Failure scenario**: a `path` value that closes the surrounding quoted expression terminates it, and nothing in the API surface tells a caller that this argument is a raw SQL fragment.
- **Suggested fix**: validate `path` against a restrictive JSON-navigation regex, or add the same explicit trusted-input docstring warning its siblings carry.

### F43. [Low] `db_command(returning=None)` raises an opaque `TypeError` — src/pyutilz/database/db/__init__.py:499, 510
- **Disposition**: OPEN
- **Category**: input-validation
- **Problem**: `returning` defaults to `"*"`, and both `len(returning) > 0` (line 510) and `"select " + returning` (line 499) assume a string. Verified: `db_command("select", "t", where_fields=["id"], source={"id":1}, returning=None)` → `TypeError: can only concatenate str (not "NoneType") to str`.
- **Failure scenario**: `None` is the natural way to express "no RETURNING clause" on an insert/update; the caller gets a `TypeError` rather than the intended behaviour or a diagnostic.
- **Suggested fix**: `returning = returning or ""` at entry, and default the `select`-mode column list to `*` when empty.

### F44. [Low] `connect_to_db` leaves the module-global `conn` pointing at a closed connection after a failed attempt — src/pyutilz/database/db/__init__.py:211, 247-249
- **Disposition**: OPEN
- **Category**: connection-lifecycle
- **Problem**: `conn` is assigned at line 211 before `set_isolation_level`/`create_engine`/`conn.cursor()` can fail. The error handler closes it (247-249) but never resets `conn = None`.
- **Failure scenario**: with `max_retries` set and exhausted, `get_cursor:274` passes its `assert conn is not None` and calls `conn.cursor()` on a closed connection, producing `InterfaceError: connection already closed` instead of the intended "connect_to_db() has not been called" diagnostic.
- **Suggested fix**: set `conn = None` alongside the `conn.close()` in the except branch.

### F45. [Low] `zip(timestamp_update_fields, timestamp_check_fields + timestamp_check_fields)` masks an arity mismatch under `python -O` — src/pyutilz/database/db/upsert.py:238-241
- **Disposition**: OPEN
- **Category**: sql-building
- **Problem**: line 238 asserts the two lists are equal length, making the `+ timestamp_check_fields` doubling a no-op. Under `python -O` asserts are stripped, and the doubling then makes `zip` pair `timestamp_update_fields[i]` against a *second copy* of the check list instead of truncating.
- **Failure scenario**: a mismatched call under `-O` produces a plausible-looking but wrong `SET updated_at=c.<wrong_column>` rather than failing.
- **Suggested fix**: drop the doubling and convert the assert to a real `ValueError`, as was already done for `field_type` (lines 105-109) and `autocreate_id_type_name` (`db/__init__.py:663-666`).

### F46. [Low] `_conn_key` may collapse all closed connections to a single key — src/pyutilz/database/psycopg2_pool.py:53-64
- **Disposition**: OPEN
- **Category**: connection-lifecycle
- **Problem**: the `except AttributeError` fallback assumes an unavailable `backend_pid` raises `AttributeError`. In psycopg2 2.9.x `conn.info.backend_pid` on a closed connection is documented to return `0` rather than raise, so every closed connection would share key `0` in `_conn_last_used`. **Uncertainty flagged**: this was not reproduced against a live server (no database available), so treat the `0` behaviour as read-from-docs, not measured.
- **Failure scenario**: a stale or shared `last_used` timestamp that at worst skips or forces one `SELECT 1` health check — bounded either way.
- **Suggested fix**: also catch `psycopg2.Error`, and treat a falsy `backend_pid` as "unavailable" so it falls back to `id(conn)`.

### F47. [Low] `explain_table` returns `None` for PostgreSQL with no signal — src/pyutilz/database/db/__init__.py:954-960
- **Disposition**: OPEN
- **Category**: silent-failure
- **Problem**: the function is implemented only for `db_flavor == "mysql"`; against PostgreSQL — the module's primary and default flavor — it silently returns `None`.
- **Failure scenario**: `explain_table("orders")["Field"]` raises `TypeError: 'NoneType' object is not subscriptable` with nothing pointing at the flavor as the cause.
- **Suggested fix**: log a warning (or raise `NotImplementedError`) in the non-MySQL branch, naming the flavor.

### F48. [Low] `get_url()` honours a `timeout=None` global, unlike every other fetch path — src/pyutilz/web/web.py:713 (set via `connect(m_timeout=...)`, :394-404)
- **Disposition**: OPEN
- **Category**: http-timeout
- **Problem**: `timeout` is typed `Optional[int]` and passed straight to `method(...)` at line 713; `requests` treats `timeout=None` as "wait forever". The two `urlopen` sites (lines 162, 276) defensively use `timeout if timeout is not None else 10` for exactly this reason.
- **Failure scenario**: `web.connect(..., m_timeout=None)` makes every `get_url()` call hang indefinitely on a stalled server, with the retry loop never advancing.
- **Suggested fix**: mirror the urlopen sites — `timeout=timeout if timeout is not None else 10` at line 713.

### F49. [Low] `start_selenium()` crashes with `TypeError` when `PROXY_PASS` is explicitly `None` — src/pyutilz/web/browser.py:205
- **Disposition**: OPEN
- **Category**: input-validation
- **Problem**: `len(proxy_server.get("PROXY_PASS", ""))` — the `""` default covers only a *missing* key; a present-but-`None` value gives `len(None)`. Sibling checks on the same dict use truthiness (`not proxy_server.get(k)`, lines 210, 288).
- **Failure scenario**: a JSON proxy config expressing "unauthenticated proxy" as `"PROXY_PASS": null` raises `TypeError: object of type 'NoneType' has no len()` at browser startup.
- **Suggested fix**: `if proxy_server.get("PROXY_PASS"):`.

### F50. [Low] `DecodoProvider.from_env()` validates `PROXY_PORT` but not `PROXY_PORT_RANGE` — src/pyutilz/web/proxy/decodo.py:245-250
- **Disposition**: OPEN
- **Category**: config-validation
- **Problem**: `base_port` gets a `try/except ValueError` with a clear message (lines 245-248); `port_range = int(os.environ.get(range_var, default_range))` on the next line raises a bare `ValueError: invalid literal for int()` naming neither the variable nor the value.
- **Failure scenario**: a typo in the `..._PROXY_PORT_RANGE` environment variable produces a stack trace an operator cannot map back to an environment variable.
- **Suggested fix**: wrap it in the same try/except, with the variable name in the message.

### F51. [Low] `get_country_by_ip()`'s `try/except` around `get_ipinfo` is dead code — src/pyutilz/web/web.py:237-244
- **Disposition**: OPEN
- **Category**: dead-code
- **Problem**: `get_ipinfo(use_urllib=True, ...)` catches every exception internally and returns `None` (lines 277-279), so the `except Exception` at line 239 and its "provider failed" debug log can never fire. A failed provider is handled instead by the `isinstance(data, dict)` check at line 245, which logs nothing.
- **Failure scenario**: when a geo-IP provider starts failing, the intended per-provider diagnostic never appears and the fallback chain is silently exhausted.
- **Suggested fix**: log at lines 245-246 when `data` is not a dict, or have `get_ipinfo` re-raise for this caller.

### F52. [Low] `LoginAndGetCookies()` restarts Selenium without quitting the old driver — src/pyutilz/web/browser.py:410
- **Disposition**: OPEN
- **Category**: resource-lifecycle
- **Problem**: `browser = start_selenium()` overwrites the handle directly. On the `"no such window"` half of the line-408 condition the driver process is typically still alive, so it is orphaned.
- **Failure scenario**: a long-running scraper that re-logs in repeatedly accumulates chromedriver processes — the exact leak class fixed at `close_browser()` (line 119) and at the other restart path (line 505).
- **Suggested fix**: call `close_browser()` before `browser = start_selenium()` at line 410.

### F53. [Low] `_reset_per_call_state` is a no-op for every provider except OpenRouter — src/pyutilz/llm/openai_compat.py:263-270
- **Disposition**: OPEN
- **Category**: stale-state
- **Problem**: the base implementation is an empty no-op; grep shows the only override is `openrouter_provider/_provider.py:170`.
- **Failure scenario**: for OpenAI/xAI/DeepSeek, a `generate()` that raises after a prior success leaves the previous call's `last_tool_calls`/`last_citations`/`_last_usage` readable in the same context — "masquerading as the latest one", in the docstring's own words.
- **Suggested fix**: give the base class a real implementation resetting the four `PerCallAttr` fields.

### F54. [Low] xAI mis-prices unknown or dated models silently — src/pyutilz/llm/xai_provider.py:158-168
- **Disposition**: OPEN
- **Category**: cost-accounting
- **Problem**: `_PRICING.get(model, (0.20, 0.50))` with no warning, unlike DeepSeek (`deepseek_provider.py:164-166`) and OpenAI (`openai_provider.py:194-195`), which both call `_warn_unknown_model_once`. Verified that `grok-3`/`grok-3-mini` are in `_PRICING` but absent from `_MAX_TOKENS`.
- **Failure scenario**: a dated snapshot id falls back to the cheapest tariff in the table with nothing in the log, so session cost under-reports by several times and the operator has no signal that the model was unrecognized.
- **Suggested fix**: apply the same `_warn_unknown_model_once` pattern, or route through a longest-prefix pricing lookup.

### F55. [Low] `_normalize_uptime` reads a genuine 1% uptime as 100% — src/pyutilz/llm/openrouter_provider/_health.py:157
- **Disposition**: OPEN
- **Category**: health-check
- **Problem**: `return v / 100.0 if v > 1.0 else v`. The docstring acknowledges both fraction and percentage shapes appear in the wild; `1.0` is the single value where they are indistinguishable, and the code resolves it in the unsafe direction.
- **Failure scenario**: an upstream reporting `uptime_last_30m: 1.0` as a percentage (a nearly-dead backend) becomes `1.0` = perfect, so `is_model_healthy(min_uptime=0.99)` (`_provider.py:578`) returns True and `list_openrouter_models` keeps the row — routing traffic to a dead provider.
- **Suggested fix**: decide the shape per payload (e.g. if any uptime field in the same response exceeds 1.0, treat them all as percentages) rather than per value.

### F56. [Low] `_parse_reset_wait_seconds` can raise `ValueError` from inside the rate-limit handler — src/pyutilz/llm/claude_code_provider.py:210-214
- **Disposition**: OPEN
- **Category**: error-handling
- **Problem**: `now.replace(month=month, day=day, ...)` and `reset_time.replace(year=now.year + 1)` construct dates without validation.
- **Failure scenario**: a reset string naming an impossible date ("resets Feb 30", or Feb 29 rolled into a non-leap year) raises `ValueError: day is out of range for month` at line 434 — *inside* the `except Exception` rate-limit branch — turning a recoverable rate-limit pause into a hard failure of `generate()`.
- **Suggested fix**: wrap the date construction in `try/except ValueError` and fall through to the existing 3600s default at line 436.

### F57. [Low] `last_call_summary()` mixes real values with `PerCallAttr` defaults after a batch — src/pyutilz/llm/openrouter_provider/_provider.py:760-771
- **Disposition**: OPEN
- **Category**: concurrency
- **Problem**: lines 765-767 read `self._last_usage` (a `PerCallAttr`, resolving to the zero-filled default in any context other than the task that set it), while the surrounding lines read plain attributes carrying some request's real values.
- **Failure scenario**: after `generate_batch()`, the summary reports a real `cost_usd` alongside `input_tokens: 0` / `output_tokens: 0` — a self-inconsistent record that reads as "a free call" rather than "unavailable".
- **Suggested fix**: fix F08; until then, have `last_call_summary` return `None` for `PerCallAttr`-backed fields that equal their default.

### F58. [Low] `token_counter._encoding_cache` is mutated without a lock — src/pyutilz/llm/token_counter.py:51-60
- **Disposition**: OPEN
- **Category**: concurrency
- **Problem**: `move_to_end` / `__setitem__` / `popitem(last=False)` on a module-level `OrderedDict` from any thread. `count_tokens` is reached from `fit_max_tokens_to_context` on every `generate()` and from `_health`'s `ThreadPoolExecutor` paths.
- **Failure scenario**: concurrent `popitem` at the cap can raise `KeyError` between the `len()` check and the pop, surfacing as a spurious failure inside a token-budget computation.
- **Suggested fix**: guard the three mutations with a `threading.Lock`, or use `functools.lru_cache` on the resolver.

### F59. [Low] `normalize_sentence` deduplicates tokens and returns them in nondeterministic order — src/pyutilz/text/similarity.py:383
- **Disposition**: OPEN
- **Category**: determinism
- **Problem**: `return list(set(sentence.split(placeholder)))`. Repeated tokens are silently dropped, changing `w_max` and therefore the score (`normalize_sentence("NEW YORK NEW YORK")` → `['YORK','NEW']`), and ordering varies by `PYTHONHASHSEED` (verified differing across four runs). The greedy matcher's `>=` tie-break is order-sensitive. `normalize_sentence("")` also yields `['']`, which scores `0.0` rather than `None`.
- **Failure scenario**: the same two names compared in two processes can produce different similarity scores, so a threshold decision is not reproducible.
- **Suggested fix**: `list(dict.fromkeys(...))` for order-stable dedup (or a plain `.split()` if dedup was unintended), and filter empty tokens.

### F60. [Low] `find_between` ignores `idx2=0` and ignores `idx1` when `start=""` — src/pyutilz/text/strings/basics.py:20, 22-25
- **Disposition**: OPEN
- **Category**: edge-case
- **Problem**: `if not idx2: idx2 = len(s)` conflates an explicit `0` with "unset". Verified: `find_between("abcXdefYghi","X","Y",0,0)` → `'def'` (should be `None`; the requested window is empty), while `idx2=5` correctly gives `None`. Separately `if len(start)==0: p1 = 0` discards `idx1`: `find_between("abcXdef","","X",3)` → `'abc'`, content from before the requested start.
- **Failure scenario**: a caller scanning a buffer with a computed window that shrinks to `[0,0)` gets a match from the whole string instead of nothing.
- **Suggested fix**: `if idx2 is None:`, and `p1 = idx1 or 0` in the empty-start branch.

### F61. [Low] `fix_dashes` collapses every multi-space run in the document — src/pyutilz/text/humanizer.py:140
- **Disposition**: OPEN
- **Category**: over-broad-regex
- **Problem**: `re.sub(r" {2,}", " ", text)` is unconditional, despite the inline comment scoping it to spaces around the replacement dashes and a docstring scoped to dashes and quotes. Verified: `fix_dashes("Column A    Column B")` → `'Column A Column B'` on text containing no dash at all.
- **Failure scenario**: running the `humanize()` pipeline over a document destroys aligned tables, indented code blocks and any ASCII layout, far outside the transformation the caller asked for.
- **Suggested fix**: restrict the collapse to the neighbourhood of the substituted dash.

### F62. [Low] `sanitize_dsn` deletes text between two DSNs and reports the wrong host — src/pyutilz/text/secrets_scrub.py:35, 87
- **Disposition**: OPEN
- **Category**: over-broad-regex
- **Problem**: the greedy `.+` anchoring to the last `@` is documented as deliberate over-masking, but on multi-DSN text the effect is destructive rather than cosmetic. Verified: `sanitize_dsn("a postgres://u:p1@h1/d and postgres://u2:p2@h2/d")` → `'a postgres://u:***@h2/d'` — the second DSN, the connective text and host `h1` are gone, and the surviving line attributes host `h2` to user `u`.  <!-- pragma: allowlist secret -->
- **Failure scenario**: an error message naming both the source and target databases is reduced to one fabricated DSN, defeating the function's own stated purpose ("preserving the rest for operational debugging ... so operators can tell environments apart").
- **Suggested fix**: keep the greedy semantics but bound the match to a non-whitespace run — `r"(://[^:@/\s]*:)[^\s]+(@)"` — so it cannot span two DSNs. (Combines with F14's empty-username fix.)

### F63. [Low] `NUM_FIRSTLETTER_CAPITAL` / `NUM_ALLLETTERS_CAPITAL` count mid-word capitals — src/pyutilz/text/tokenizers.py:160-172
- **Disposition**: OPEN
- **Category**: algorithm-correctness
- **Problem**: both flags are recomputed at every start offset `i`, not once per word, so the name "FIRSTLETTER" does not describe what is counted. Verified with a replica of the loop on `"aBCDE"`: morphemes `('B',True,False)`, `('BC',True,True)`, `('BCD',True,True)`.
- **Failure scenario**: a word that is neither capitalized nor all-caps contributes to both counters, so any downstream feature keyed on capitalization statistics is systematically inflated for camelCase-ish tokens.
- **Suggested fix**: compute both flags once from `word` before the `i` loop.

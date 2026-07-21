# Security Audit — Semantic / Business-Logic (beyond bandit)

## Summary

Covered: `core/safe_pickle.py` and its sibling `core/serialization.py`; `database/db/sql_helpers.py`, `database/db/upsert.py`, `database/db/__init__.py`, `database/db/sqlite.py`; `database/redislib.py`; `database/deltalakes.py`; `dev/logginglib.py`; `web/web.py`, `web/browser.py`, `web/proxy/*.py` (`base.py`, `session.py`, `decodo.py`, `ip_check.py`, `__init__.py`); `core/pythonlib.py`'s `ensure_installed`; a repo-wide grep for `subprocess`/`eval`/`exec`/`yaml.load`/`pickle.loads`; and `.env.example` plus its full git history. All files read in full via the Read tool; every finding below cites a line I actually read, and several were confirmed with a standalone, read-only `python -c`/script reproduction (shown inline).

Overall the SQL-identifier-validation discipline in `database/db/` is unusually strong for a convenience library (see "Things done well"), which made the two places where that discipline silently lapses (`bAddUpdatedAtTimestamp`, the regex anchor) stand out as real, not hypothetical, gaps. The most serious findings are two credential-leak paths (`web/web.py`'s `report_params()` and `web/browser.py`'s on-disk proxy-auth Chrome extension) that write plaintext proxy/API credentials to logs or disk in ordinary, non-debug usage. Findings: 2 Critical, 3 High, 5 Medium, 2 Low.

## Findings

### [CRITICAL] Proxy credentials and Authorization headers logged in cleartext, unconditionally on a common code path — src/pyutilz/web/web.py:382-385,503-506
- **Category**: security / credential-leak
- **Problem**: `report_params()` logs the full `proxies` dict and the full `headers_to_use` dict at INFO level:
  ```python
  382  def report_params(url, proxies, params, data, json, headers_to_use, timeout):
  383      """Log a request's url/proxies/params/data/json/headers/timeout at INFO level, for debugging a fetch."""
  384      logger.info("url=%s, proxies=%s, params=%s, data=%s, json=%s, headers=%s, timeout=%s", url, str(proxies), params, data, json, headers_to_use, timeout)
  ```
  `proxies` is built by `make_proxies_dict()` (web.py:306-313) as `"{proxy_type}://{proxy_user}:{proxy_pass}@{proxy_server}:{proxy_port}"` — the proxy password is embedded in plaintext in the URL string that gets logged. `headers_to_use` can likewise contain an `Authorization` header verbatim (e.g. `pyutilz.core.filemaker.get_session_token()` sets `template_headers={"Authorization": "Basic " + b64encode(username+":"+password)...}` via `web.connect()`, and any Bearer-token API client built on this module would do the same).
  There are two call sites. One is gated by `verbose` (line 463-464). The other, at lines 503-506, is **not gated by `verbose` at all** — it fires automatically whenever the response status is in the caller-supplied `blocking_statuses` tuple, which is the module's advertised mechanism for detecting anti-bot blocks:
  ```python
  503    if res.status_code in blocking_statuses:
  504        logger.info("Error %s while getting %s", res.status_code, url)
  505        report_params(url, proxies, params, data, json, headers_to_use, timeout)
  506        handle_blocking(target, b_random_ua=b_random_ua, b_use_proxy=b_use_proxy)
  ```
  Tellingly, `handle_blocking()` itself (line 627, a few lines below) deliberately strips the credential before logging: `logger.warning("IP %s blocked. ...", proxies["https"].split("@")[1], target)`. The same function, in the same branch, logs the unredacted version one line earlier via `report_params`. This asymmetry is strong evidence the leak in `report_params` is an oversight, not a documented tradeoff (unlike the explicitly-labeled raw-SQL escape hatches elsewhere in this repo).
- **Failure scenario**: A caller of `get_url(url, blocking_statuses=(403,), ...)` (a normal configuration for any scraper that needs to detect being blocked) is using an authenticated proxy and/or an `Authorization` header. The target returns HTTP 403. `report_params` runs unconditionally, writing `proxies=http://user:P@ssw0rd@proxy.host:1234` and `headers={'authorization': 'Bearer sk-...'}` into the application's INFO-level log stream/file — which may be shipped to a log aggregator, retained for months, or readable by other users/processes on a shared box.
- **Suggested fix**: Redact credentials before logging in `report_params`, mirroring `handle_blocking`'s own pattern — e.g. strip `user:pass@` from each proxy URL and mask/omit `Authorization`/`Cookie`/`Set-Cookie` header values before formatting the log line. Also consider degrading this call to DEBUG level even after redaction.

### [CRITICAL] Plaintext proxy credentials written to a predictable, non-temp, never-cleaned-up file — src/pyutilz/web/browser.py:220-255
- **Category**: security / credential-leak
- **Problem**: When a proxy with a password is configured, `start_selenium()` builds a Chrome extension embedding the proxy username/password in cleartext inside `background.js` (`password: "%s"` interpolated with `proxy_server["PROXY_PASS"]`, line 239/249), then writes it to a **hardcoded, predictable, relative filename in the process's current working directory**:
  ```python
  250    pluginfile = "proxy_auth_plugin.zip"
  252    with zipfile.ZipFile(pluginfile, "w") as zp:
  253        zp.writestr("manifest.json", manifest_json)
  254        zp.writestr("background.js", background_js)
  255    options.add_extension(pluginfile)
  ```
  There is no `tempfile.mkstemp`/`NamedTemporaryFile`, no restrictive permission setting, and (confirmed via `grep -n "pluginfile\|os.remove\|os.unlink" src/pyutilz/web/browser.py`) **no cleanup anywhere in the file** — the zip persists on disk indefinitely after the browser session ends, in the CWD rather than an isolated temp directory.
- **Failure scenario**: Any process using `pyutilz.web.browser` with an authenticated proxy leaves `./proxy_auth_plugin.zip` containing the plaintext proxy password sitting in whatever directory the script was launched from. That directory is commonly a project/working folder — easy to accidentally `git add .`/back up/leave world-readable, and trivially readable by any other local process/user on a shared host. Concurrent scraping workers (this repo explicitly supports parallelism, see `system/parallel.py`) racing on the *same* filename can also corrupt each other's extension zip or load a different worker's credentials.
- **Suggested fix**: Write to a `tempfile.TemporaryDirectory()`/`mkstemp`-generated path (unpredictable name, restrictive default permissions), and delete it in a `finally` once `webdriver.Chrome(**kwargs)` has loaded the extension (or register an atexit/`close_browser()` cleanup).

### [HIGH] `bAddUpdatedAtTimestamp` spliced into raw SQL with zero validation — src/pyutilz/database/db/sql_helpers.py:114-131 (reachable via src/pyutilz/database/db/__init__.py:654,658,662)
- **Category**: security / SQL injection
- **Problem**: `MakeSetExcludedClause(sFields, bAddUpdatedAtTimestamp)` validates every field in `sFields` via `validate_sql_identifier`, but splices `bAddUpdatedAtTimestamp` directly into the generated SQL with **no validation and no "accepted raw fragment" warning** (unlike every other intentionally-raw parameter in this same file/module, e.g. `update_if_now`'s `clause` param, which carries an explicit inline comment "`clause` is an accepted raw SQL fragment by design"):
  ```python
  114  def MakeSetExcludedClause(sFields: str, bAddUpdatedAtTimestamp: Optional[str] = None) -> str:
  ...
  127      if bAddUpdatedAtTimestamp:
  128          res = res + f"{bAddUpdatedAtTimestamp}=(now() at time zone 'utc')"  # updated_at
  ```
  It is called from the public `GetIdByKeyFieldAndInsertIfNeeded()` in `db/__init__.py` at three sites (lines 654, 658, 662), all inside an f-string executed via `safe_execute` (e.g. line 654): `f"insert into {sTable} (...) ... do update set {MakeSetExcludedClause(sKeyFieldName, bAddUpdatedAtTimestamp)} returning {sIdFieldName}"`.
  Confirmed with a standalone read-only repro:
  ```
  >>> MakeSetExcludedClause('name', "updated_at=now(); DROP TABLE users;--")
  "name=excluded.name,updated_at=now(); DROP TABLE users;--=(now() at time zone 'utc')"
  ```
  psycopg2's `cursor.execute()` sends the string via the simple query protocol, which supports semicolon-stacked statements, so this is a full injection primitive, not just a syntax break.
- **Failure scenario**: Any caller of `GetIdByKeyFieldAndInsertIfNeeded(..., bAddUpdatedAtTimestamp=<value>)` where `<value>` is influenced by external/config input (plausible for a "which timestamp column to bump" setting sourced from a settings table, admin UI, or generic field-mapping config) gets arbitrary SQL execution, including `DROP TABLE`/data exfiltration via stacked statements.
- **Suggested fix**: Call `validate_sql_identifier(bAddUpdatedAtTimestamp)` when it is not `None`, exactly as is already done for `sKeyFieldName`/`sIdFieldName` in the same function family.

### [HIGH] `unserialize()` defaults to fail-OPEN and silently swallows its own tamper-detection exception — src/pyutilz/core/serialization.py:91-146
- **Category**: security / pickle deserialization
- **Problem**: `pyutilz.core.safe_pickle.safe_load()` is fail-closed by default. Its older sibling in the same package, `pyutilz.core.serialization.unserialize()`, does the opposite: `verify_sidecar: bool = False` (line 91), documented as "preserving historical behaviour for existing callers" (line 97) — i.e. `unserialize(path)` with no extra kwargs performs **zero integrity verification** and unconditionally `pickle.loads()`s whatever is on disk (line 140). A caller who has read `safe_pickle.py`'s security-conscious docstring could easily assume the whole `pyutilz.core` package behaves the same way and reach for `unserialize()` (the older, more likely-referenced name) without realizing it offers no protection by default.
  Worse, even when a caller explicitly opts in with `verify_sidecar=True`, the resulting `PickleVerificationError` (line 120) is raised *inside* the function's own blanket `try/except Exception` (line 110...145), so it is caught, logged via `logger.exception(e)`, and swallowed — the function just returns `None`, indistinguishable from "file not found" or "corrupt zlib data" (lines 111-114, 133-138 use the exact same log-and-return-None pattern). This defeats `PickleVerificationError`'s own stated purpose in `safe_pickle.py`: "Distinguishes verification failures from arbitrary `pickle.UnpicklingError` so callers can surface a security-relevant message in logs without having to inspect the error string" (safe_pickle.py:90-93).
- **Failure scenario**: (a) Default usage — an attacker who can plant a file at a path later passed to `unserialize(path)` gets unauthenticated `pickle.loads()`, i.e. RCE, with no verification and no warning printed at call time (only in the docstring). (b) Opt-in usage — a caller who sets `verify_sidecar=True` specifically to get a security alarm on tampering gets `None` back on both a tampered file and a merely-missing file, with no way to distinguish "possible attack" from "cache miss" without grepping logs for exception text.
- **Suggested fix**: Either flip `unserialize()`'s default to match `safe_load`'s fail-closed posture (a breaking change, but consistent with the rest of the package), or at minimum re-raise `PickleVerificationError` instead of swallowing it into the generic `except Exception` path, so callers can `except PickleVerificationError` distinctly from ordinary I/O failures.

### [MEDIUM] `validate_sql_identifier`'s regex accepts a trailing newline — src/pyutilz/database/db/sql_helpers.py:30,33-43
- **Category**: security / edge-case / input validation
- **Problem**: `_SQL_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")` uses `$`, which in Python's `re` (without `re.MULTILINE`) matches both at the true end of string *and* immediately before a trailing `\n`. Confirmed:
  ```
  >>> import re; pat = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
  >>> bool(pat.match('users\n'))
  True
  ```
  So `validate_sql_identifier("users\n")` returns `"users\n"` as "valid," even though the docstring promises "Valid identifiers must match: alphanumeric, underscore, start with letter/underscore" with no newline allowance.
- **Failure scenario**: Because `[a-zA-Z0-9_]*` cannot itself match `\n`, nothing can follow that trailing newline and still pass (`"a\nb"` correctly fails), so this does not currently enable stacked-query injection through this function alone — the practical impact is a contract violation (an identifier with a trailing `\n` reaches SQL text as e.g. `insert into users\n (...)`, silently accepted where it should be rejected) rather than a demonstrated RCE. It's the well-known Python `$`-vs-`\Z` anchor pitfall on the library's core SQL-identifier gate, and any future loosening of the character class (e.g. to support quoted/mixed-case identifiers) would inherit this gap.
- **Suggested fix**: Use `\Z` instead of `$` (`r"^[a-zA-Z_][a-zA-Z0-9_]*\Z"`), or switch to `_SQL_IDENTIFIER_RE.fullmatch(identifier)`.

### [MEDIUM] `u()`/`nu()` only double the quote character, not backslashes — src/pyutilz/database/db/sql_helpers.py:94-111
- **Category**: security / SQL injection (dialect-dependent)
- **Problem**: `u(str_val, symb="'")` escapes an embedded quote by doubling it (`str_val.replace(symb, symb*2)`) but does nothing about a trailing backslash. Confirmed:
  ```
  >>> u("foo" + chr(92))          # "foo\"
  "'foo\\'"                        # literally: 'foo\'
  ```
  Under a backslash-escaping SQL dialect (MySQL's default `NO_BACKSLASH_ESCAPES=off`, or pre-9.1 PostgreSQL with `standard_conforming_strings=off`), the trailing backslash escapes the closing quote so the literal never actually terminates, letting the rest of the query text be swallowed into/reinterpreted around the string — the classic bypass for quote-doubling-only escaping. `u()`/`nu()` carry no dialect caveat in their docstrings, yet `database/db/__init__.py:149` (`assert db_flavor in ("postgres", "mysql")`) shows this same package explicitly supports MySQL as a target, and `u`/`nu` are re-exported public helpers (`database/db/__init__.py:67-68`) usable by any downstream caller independent of `db_flavor`.
- **Failure scenario**: Not exploitable via any function currently shipped in this repo (the two live call sites — `check_if_pg_table_exists` and `GetIdByKeyFieldAndInsertIfNeeded` — only ever execute through the psycopg2/Postgres cursor path, and modern Postgres defaults `standard_conforming_strings=on`). It is a latent footgun for any downstream/future code that imports `u()`/`nu()` directly against a MySQL connection (or an older Postgres instance), given the module explicitly advertises MySQL support and the docstring gives no warning.
- **Suggested fix**: Either document the Postgres-only/`standard_conforming_strings=on` assumption prominently in `u()`/`nu()`'s docstrings, or additionally escape backslashes (`str_val.replace("\\", "\\\\")`) before quote-doubling so the function is dialect-agnostic, and prefer parameterized queries over `u()`/`nu()` wherever the caller can.

### [MEDIUM] Predictable lock-file name in a shared, world-writable temp directory — src/pyutilz/database/deltalakes.py:75
- **Category**: security / edge-case (local multi-tenant)
- **Problem**: `safe_delta_write()` derives its cross-process lock path as `os.path.join(tempfile.gettempdir(), f"{os.path.basename(path).replace('/', '_')}{lock_suffix}")` — a fully deterministic name (the Delta table's own basename plus `.lock`) placed directly in the shared system temp directory, rather than via `tempfile.mkstemp`, a per-user subdirectory, or a randomized component.
- **Failure scenario**: On a shared multi-tenant host (a common deployment for ETL/cron workloads using Delta Lake), a local unprivileged attacker who knows or guesses the target table's basename (typically a well-known name, e.g. `orders.lock`) can pre-create `<tmp>/orders.lock` before the legitimate job ever runs — no race required, just planting ahead of time. Whether this escalates to arbitrary-file-write/corruption depends on how the third-party `filelock` package opens an already-existing path (not verified here, out of this repo's control), but placing a predictable name in a shared writable directory is the root anti-pattern (CWE-377) regardless of `filelock`'s exact internals, and at minimum lets an attacker deny the lock to the legitimate writer indefinitely.
- **Suggested fix**: Derive the lock path from a per-user or per-invocation-unique subdirectory (e.g. `tempfile.mkdtemp()` once per process, reused across calls) rather than the shared system temp root, or namespace it under a directory created with restrictive permissions.

### [MEDIUM] `rexecute()` is an unrestricted reflection gateway onto the whole Redis command surface — src/pyutilz/database/redislib.py:46-76
- **Category**: security / architecture
- **Problem**: `rexecute(method_name, *args, **kwargs)` resolves and calls `getattr(rc, method_name)(*args, **kwargs)` (line 57, 65) with no allow-list of permitted commands. This is a generic RPC dispatcher onto every method the `redis-py` client exposes, including `eval`/`evalsha` (arbitrary Lua), `config_set`, `flushall`, `flushdb`, `shutdown`, `script_load`, etc. Unlike the analogous raw-SQL escape hatches in `database/db/__init__.py` (`select()`, `execute_alchemy()`), which each carry an explicit "WARNING: must never be built from external/user-controlled input" docstring, `rexecute`'s docstring ("Safely execute any Redis command, not worrying about temporary network/server issues") gives no such warning — "safely" here refers only to retry-on-`ConnectionError` semantics, not command authorization.
- **Failure scenario**: If `method_name` (or `args`/`kwargs`) is ever derived from a request boundary (e.g. a thin API wrapper that lets a caller specify "which Redis op to run"), this is equivalent to unauthenticated arbitrary Redis command execution — including the well-known `CONFIG SET dir <path>` + `SAVE` chain used to write an arbitrary file to disk via Redis, or direct Lua execution via `eval`.
- **Suggested fix**: Add a docstring warning matching the `db/__init__.py` convention ("`method_name`/`args`/`kwargs` must never be built from external/user-controlled input"), and/or offer an explicit allow-list parameter for callers that do need to expose a subset of commands.

### [MEDIUM] Fixed, well-known Chrome DevTools debug port — src/pyutilz/web/browser.py:262
- **Category**: security / edge-case (local multi-tenant)
- **Problem**: `start_selenium()` unconditionally adds `options.add_argument("--remote-debugging-port=9222")` — a hardcoded, non-randomized, well-known port, with no `--remote-debugging-address` restriction specified.
- **Failure scenario**: On a shared host running multiple scraping jobs (this repo explicitly supports parallel workers), any other local process/user can connect to `http://localhost:9222/json` and attach to the Chrome DevTools Protocol, gaining full control of the already-authenticated Selenium session established by `LoginAndGetCookies()` — reading session cookies, injecting arbitrary JS into the logged-in page, effectively hijacking the session `browser.py` just spent effort logging in to. (Modern Chrome's Host-header check limits *remote* exploitation from a malicious web page, but local processes on the same host remain fully exposed, and two concurrent `start_selenium()` calls on one machine would also collide on the same port.)
- **Suggested fix**: Bind to an ephemeral port (`--remote-debugging-port=0` and read back the assigned port from Chrome's stdout, or use Selenium's own driver-managed debugging channel) and/or restrict `--remote-debugging-address=127.0.0.1` explicitly.

### [MEDIUM] `logged()` decorator's `special_vars` bucket doesn't redact credentials, only relocates them — src/pyutilz/dev/logginglib.py:290-352 (esp. 305,328,333-341) and 198-219
- **Category**: security / credential-leak (needs confirmation — no in-repo call site with real data)
- **Problem**: The `logged()` decorator explicitly excludes `special_vars = ("current_proxy", "current_proxy_resolved", "login")` from `results_log["parameters"]` (line 328: `if key not in special_vars`), which reads as an intentional "don't log these directly" design decision. But it then copies the very same raw values, unredacted, into `results_log["session"][var]` (lines 333-341), and that `results_log` (including `["session"]`) is exactly what `finalize_function_log()` either `print()`s (line 217-218, when `verbose=True`) or persists to a DB `logs`-style table via `safe_execute_values` (lines 208-215) when `db_path` is configured. The relocation achieves no confidentiality benefit — it's the same plaintext value, just under a different dict key that is equally logged/persisted.
  I could not find any call site inside this repo that actually passes a `current_proxy`/`login` kwarg to a `@logged(...)`-decorated function, so I cannot demonstrate this end-to-end within pyutilz itself — flagging as "needs confirmation" for the *content* of what lands here. However, this repo's own naming convention (`web/browser.py:60` uses `login` for the username, and both `web/web.py:make_proxies_dict()` and `web/proxy/decodo.py:proxy_url()` embed the proxy password directly inside a "current proxy"-shaped URL string) makes it highly plausible that a `current_proxy`/`current_proxy_resolved` value in a downstream caller is exactly the credential-bearing proxy URL described in the two Critical findings above.
- **Failure scenario**: A downstream function decorated with `@logged(db_path="logging.api_calls")` and a `current_proxy` kwarg holding a `http://user:pass@host:port` string would have that value persisted verbatim into the `logging.api_calls` table's `session` JSON column on every call — a durable, queryable credential leak, not just a transient log line. <!-- pragma: allowlist secret -->
- **Suggested fix**: Actually redact (not just relocate) `special_vars` values before they enter `results_log["session"]` — e.g. strip `user:pass@` from anything URL-shaped, or store only a hash/last-4-chars for correlation purposes.

### [LOW] `verify_sidecar()` uses a plain `==` digest comparison — src/pyutilz/core/safe_pickle.py:159
- **Category**: security / hygiene
- **Problem**: `return expected == actual` (both lowercase hex sha256 strings) is not a constant-time comparison (`hmac.compare_digest` or equivalent).
- **Failure scenario**: Not currently exploitable — both operands are read from local disk (the sidecar file and the payload file), and the module's own documented threat model (safe_pickle.py:10-17) already scopes this to a *corruption* check, not an authenticity control with a remote/network-observable timing oracle. This is flagged purely because the module's own docstring invites callers to layer a keyed `HMAC-SHA256(secret_key, payload)` sidecar on top (line 15) — if a future extension copies this same `==` comparison pattern for that keyed check, *that* would need `hmac.compare_digest`.
- **Suggested fix**: No change needed for the current corruption-only use; if/when a keyed variant is added, use `hmac.compare_digest`.

### [LOW] `ensure_installed()` invokes `pip` by bare name instead of `sys.executable -m pip` — src/pyutilz/core/pythonlib.py:69
- **Category**: security / hygiene
- **Problem**: `subprocess.check_call(["pip", "install", pkg])` resolves `pip` via the OS executable-search order rather than pinning to the current interpreter's `sys.executable -m pip`.
- **Failure scenario**: On Windows, `CreateProcess`'s search order includes the current working directory before `PATH`. A local attacker who can drop a `pip.exe`/`pip.bat` into the process's CWD before `ensure_installed()` runs would have it preferred over the real `pip`. This requires local file-drop access to the CWD already, so the practical bar is high, but the fix is essentially free and is also the officially pip-recommended invocation (guarantees installing into the *running* interpreter's environment, sidestepping this ambiguity entirely).
- **Suggested fix**: `subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])`.

## Things done well

- `database/db/` has unusually disciplined SQL-injection hygiene for a convenience library: `validate_sql_identifier()`/`validate_sql_qualified_identifier()` are called before nearly every identifier splice, and the code is careful about which checks are real `raise ValueError`s vs. which are `assert`s — with explicit inline comments explaining *why* a given check must not be an assert (e.g. `upsert.py:105-109` and `db/__init__.py:531-535` both call out that `python -O` strips asserts, so the field-type/`sAutocreateIdTypeName` whitelist checks are real raises, not asserts).
- Every deliberate raw-SQL "escape hatch" in `db/__init__.py` (`select()`, `execute_alchemy()`, `showcase_table()`'s `condition`, `ReadTableIntoDicReversed()`'s `sCondition`) carries an explicit, consistent "WARNING: this executes raw SQL, never build this from external input" docstring — a good, repeatable pattern that made the two places lacking it (`bAddUpdatedAtTimestamp`, `rexecute`) stand out as real gaps rather than a pervasive style.
- Correct use of `psycopg2.sql.Identifier`/`sql.SQL(...).format()` for driver-level identifier quoting in several functions (`read_unique_table_field`, `create_enum_from_table`, `regjobs_create_table`, `regjobs_poll`) instead of manual string concatenation — the more robust approach where it's used.
- `safe_pickle.py` is exemplary: an explicit, up-front "THREAT MODEL CAVEAT" docstring scoping exactly what the sidecar does and doesn't protect against, fail-closed-by-default missing-sidecar handling, a corrupted/mismatched sidecar that *always* raises regardless of the permissive env-var opt-in, atomic `os.replace` + `fsync` + Windows sharing-violation retry, and per-path locking to avoid a payload/sidecar interleave race.
- `web/web.py`'s `_ensure_http_scheme()` (lines 74-81) is a proactive, well-documented SSRF/`file://`-disclosure guard for the `urlopen`-based fetchers.
- `web/proxy/decodo.py` reads its API key strictly from an environment variable (`from_env()`), never hardcodes it, and its `_api_headers()` raises a clear error without ever including the key value in the exception message.
- `.env.example` is a clean placeholder template (every key genuinely empty); confirmed via `git log -p --follow` that its one and only commit never contained a real-looking secret, and the commit message documents a `detect-secrets` pre-commit hook + baseline.
- `build_upsert_query`/`db_command` correctly use parameterized `%s` placeholders for *values*; only *identifiers* (already validated) are string-spliced.

## Investigated, not an issue

- `verify_sidecar()`'s non-constant-time `==` — see the LOW finding above; not exploitable under this module's own documented (corruption-only) threat model since neither operand crosses a network boundary.
- `web/browser.py:359`'s XPath built via string concatenation (`"//div[text()='" + login.lower() + "']"`): technically unescaped, but `login` is the operator's own trusted credential (the value being used to log *into* the target site), not adversarial input — there's no attacker who benefits from breaking their own login XPath.
- Repo-wide `subprocess.run`/`Popen`/`check_call` usage (`system/system/sysinfo.py`, `system/system/probing.py`, `system/system/fsutils.py`, `llm/claude_code_provider.py:570`): all use fixed argv lists with `shell=False`, hardcoded trusted binaries (`wmic`, `lscpu`, `nvidia-smi`, `claude`, etc.) and no command text built by string concatenation from external input; each already carries a `# nosec B603 B607` with an accurate justification.
- No `eval(`, `exec(`, or `yaml.load(` anywhere in `src/` (grepped repo-wide).
- `pickle.load` in `text/tokenizers.py:load_tokens()` (line 286) only round-trips the same class's own `save_tokens_to_file()` output — a self-produced-data trust boundary, consistent with this module's own inline justification.
- `.env.example` git history: single commit, no real secret ever present (verified via `git log -p --follow -- .env.example`).

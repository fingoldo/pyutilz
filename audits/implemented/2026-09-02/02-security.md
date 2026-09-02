# Security Audit — pyutilz (2026-09-02)

## Summary

Read first: `CLAUDE.md` (community-health won't-fix respected — SECURITY.md/CODE_OF_CONDUCT.md/templates/CITATION.cff not raised), `audits/2026-07-21_full-audit/02-security.md` (all 12 findings), and `audits/2026-07-21_audit-round2/INDEX.md`.

**Round-1 regression check — all 12 prior security findings verified FIXED in the current tree, none still open:** `report_params()` now redacts (`web/web.py:583-589`); the proxy-auth Chrome extension uses `tempfile.mkstemp` + `os.chmod(0o600)` + atexit/finally cleanup (`web/browser.py:277-321`); `add_updated_at_timestamp` is now `validate_sql_identifier`'d (`database/db/sql_helpers.py:160`); `unserialize()` re-raises `PickleVerificationError` outside the blanket handler (`core/serialization.py:126-131`); the identifier regex uses `\Z` (`sql_helpers.py:37`); `u()`/`nu()` carry the dialect caveat; the delta lock name is a sha256 of the normcased abspath (`database/deltalakes.py:90`); `rexecute` carries the external-input warning (`database/redislib.py:66-70`); Chrome uses `--remote-debugging-port=0` + `--remote-debugging-address=127.0.0.1` (`web/browser.py:304-305`); `logged()`'s `special_vars` are genuinely redacted via `_redact_credential_shaped_value` (`dev/logginglib.py:306-384`); `ensure_installed` uses `sys.executable -m pip` (`core/pythonlib.py:69`); the `==` digest note stays non-actionable.

Also verified clean this round: zero `shell=True`/`os.system`/`os.popen`; zero `yaml.load`/`eval(`/`exec(`; zero `verify=False`/`CERT_NONE`; every `subprocess` call site is a fixed argv list with `shell=False` and an accurate `# nosec` justification; the only `zipfile` use is a write, no `extractall`/`tarfile` anywhere (no zip-slip surface); `random` uses are all non-cryptographic and annotated; all 7 GitHub Actions workflows pin third-party actions by SHA, set `persist-credentials: false` on every checkout, declare least-privilege `permissions:`, gate both `workflow_run` triggers on `conclusion == 'success'` + push-origin, use PyPI Trusted Publishing (no stored token), and interpolate no attacker-controllable `github.event.*` context into any `run:` block (only `github.repository` and a step output); `zizmor.yml`'s two ignores are each justified by a real job-level `if:` I confirmed in the workflow.

New findings: **2 High, 2 Medium, 3 Low** (7 total). All are OPEN and newly raised (none is a re-raise of a 2026-07-21 item).

## Findings

### F01. [High] SQL injection: `suggest_json_optimization`'s `path` argument is spliced into raw SQL with no validation — src/pyutilz/database/db/__init__.py:1088
- **Disposition**: COMPLETED - already fixed by the audit-fix wave that landed before this pass: `path` is now validated against `_JSON_PATH_RE` (src/pyutilz/database/db/__init__.py:600) before splicing, raising ValueError otherwise (src/pyutilz/database/db/__init__.py:1123). This pass only corrected the stale inline comment above the query, which still claimed only table/table_field/field were validated (src/pyutilz/database/db/__init__.py:1128).
- **Category**: sql-injection
- **Problem**: `suggest_json_optimization(table, table_field, path="", fields=None, ...)` validates `table` (line 1069), `table_field` (line 1070) and each `field` (line 1081) with `validate_sql_identifier`, but **never validates `path`**. Line 1088 builds `full_path = table_field + "->" + path`, and lines 1091-1092 splice `full_path` three times into an f-string executed via `safe_execute`:
  ```
  1088          full_path = table_field + "->" + path
  ...
  1091      vals = safe_execute(f"""
  1092                  select {full_path}->>'{field}' as val,count(*) as qty from {table} where {full_path} is not null group by {full_path}->>'{field}' order by qty desc
  1093          """)  # nosec B608
  ```
  The inline comment at line 1090 claims "table/table_field/field validated above" — it silently omits `path`, and the `# nosec B608` suppresses bandit's SQL-string warning on exactly this line. This is the identical bug class as the round-1 High on `bAddUpdatedAtTimestamp` (fixed at `sql_helpers.py:160`): one raw parameter left out of an otherwise-disciplined validation sweep, with no "accepted raw fragment by design" docstring warning (contrast `sql_helpers.py:182`, which does document its raw `clause` param).
- **Failure scenario**: A caller passes a JSON sub-path sourced from config, an admin UI, or a field-mapping table, e.g. `path="'a'; drop table users; --"`. psycopg2's `cursor.execute()` uses the simple query protocol, which permits semicolon-stacked statements, so the generated text `select data->'a'; drop table users; -->>'x' ...` executes the injected statement — full arbitrary SQL, including data exfiltration or DDL.
- **Suggested fix**: Validate `path` before line 1088. A JSON path is not a bare identifier, so either restrict it to a quoted-key/arrow grammar (e.g. `re.fullmatch(r"(?:'[A-Za-z0-9_]+'(?:->)?)+", path)`) and raise `ValueError` otherwise, or build the path with `psycopg2.sql.Literal`/parameter placeholders as is already done in `create_enum_from_table` (line 1050). If a raw fragment is genuinely intended, add the module's standard "WARNING: never build this from external input" docstring line, as `update_if_now` does.

### F02. [High] Claude Code subprocess/SDK run with permissions fully bypassed and no `--strict-mcp-config`, leaving the user's configured MCP tools reachable from untrusted prompt text — src/pyutilz/llm/claude_code_provider.py:559
- **Disposition**: COMPLETED - `--strict-mcp-config` added to the CLI argv (src/pyutilz/llm/claude_code_provider.py:571) and as `extra_args["strict-mcp-config"]=None` for the SDK (src/pyutilz/llm/claude_code_provider.py:521, the SDK emits a valueless flag for None), so no ambient MCP server from the invoking user's config is loaded; no --mcp-config is passed, so the set is empty. An escaped tool-use block is now a hard failure via the new `ClaudeCodeToolUseError` (RuntimeError, deliberately not OSError so the transient-retry arm cannot swallow it) instead of a "(blocked)" log line (src/pyutilz/llm/claude_code_provider.py:31, :540). Permission bypass itself was kept: with tools disabled and no MCP servers there is nothing left to auto-approve, and removing it risks a permission prompt hanging a headless --print run.
- **Category**: prompt-injection / privilege-bypass
- **Problem**: Both backends disable all permission checks:
  - CLI (`_generate_cli`, lines 552-562): argv contains `'--dangerously-skip-permissions'` (line 559) alongside `'--tools', ''` (line 561).
  - SDK (`_generate_sdk`, lines 474-481): `permission_mode="bypassPermissions"` (line 477) with `extra_args={"tools": ""}` (line 480).

  The only compensating control is `--tools ""`. Verified against the installed CLI's own help text: `--tools <tools...>` is scoped to **"the list of available tools from the built-in set"** ("Use \"\" to disable all tools, \"default\" to use all tools, or specify tool names (e.g. \"Bash,Edit,Read\")"). MCP-server-provided tools are not part of that built-in set; the CLI ships a separate `--strict-mcp-config` flag whose documented purpose is "Only use MCP servers from --mcp-config, ignoring all other MCP configurations". Neither backend passes `--strict-mcp-config` or `--mcp-config`, so any MCP server in the invoking user's global/project configuration is loaded into the session — and with permissions bypassed, its tools are auto-approved.

  The code's own handling confirms tool use is not actually prevented: the SDK message loop only *reports* it after the fact — `logger.warning("Model attempted tool use: %s (blocked)", tool_name)` (line 522) — while parsing the response, i.e. the "blocked" claim is a log string, not an enforced boundary. `env=sub_env` (line 574) filters only the five `_NESTED_BLOCK_VARS` (lines 452-458), so the rest of the parent environment is inherited by the subprocess.
- **Failure scenario**: A caller uses `ClaudeCodeProvider` to summarize or classify third-party text (a scraped page, an LLM-pipeline input — the provider's whole purpose). The text carries a prompt-injection instruction. On a developer box with, say, a filesystem or shell MCP server configured in `~/.claude.json`, that tool is available and every call is auto-approved by `--dangerously-skip-permissions`/`bypassPermissions` — the injected instruction obtains tool execution with the invoking user's privileges, and the only trace is a `logger.warning` line after the fact.
- **Suggested fix**: Add `--strict-mcp-config` to the CLI argv (lines 552-562) and the equivalent to the SDK `extra_args` (line 480), so no ambient MCP configuration is loaded. Prefer dropping `--dangerously-skip-permissions`/`bypassPermissions` entirely for a text-generation-only provider (with `--tools ""` there are no permission prompts left to skip), or replace it with a deny-all `--disallowedTools`. Also turn line 522's "(blocked)" into an actual hard failure (raise) so the claim matches the behaviour.

### F03. [Medium] Server-controlled `Retry-After` header drives an unbounded `time.sleep` — src/pyutilz/web/cached_client.py:123
- **Disposition**: COMPLETED - `Retry-After` is now clamped by the module-level `MAX_RETRY_AFTER_SECONDS = 120.0`; anything larger falls back to local exponential backoff and is logged at WARNING (src/pyutilz/web/cached_client.py:36, :147-160).
- **Category**: dos / untrusted-input
- **Problem**: In `_fetch_bytes`'s retry loop:
  ```
  121                  if exc.code in _RETRYABLE_HTTP_CODES and attempt < retries - 1:
  122                      retry_after = exc.headers.get("Retry-After") if exc.headers else None
  123                      time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt)
  ```
  `retry_after` comes straight from the remote server's response headers. `.isdigit()` only constrains it to digits — it imposes no upper bound, so any non-negative integer the server sends is slept verbatim, synchronously, with no cap and no cancellation. Nothing else bounds total wall time (`timeout` applies per-request only).
- **Failure scenario**: A remote host (compromised, hostile, or simply misconfigured) answers a throttled request with `HTTP 429` + `Retry-After: 999999999`. The calling thread blocks for ~31 years inside `time.sleep`. For a batch job iterating URLs this is an indefinite hang with no explanatory log line (the `logger.debug` at line 125 is only reached after the sleep). One hostile URL stalls the whole pipeline.
- **Suggested fix**: Clamp the parsed value, e.g. `delay = min(float(retry_after), MAX_RETRY_AFTER_S)` with a module-level cap (30-120s is typical), falling back to `2**attempt` when the header exceeds it; log when the header is clamped so the behaviour is visible.

### F04. [Medium] Path traversal: `tag` is used as an unvalidated directory component of the cache path — src/pyutilz/web/cached_client.py:104
- **Disposition**: COMPLETED - `_cache_path` now rejects any `tag` not matching `_SAFE_TAG_RE = [A-Za-z0-9_.-]+` (and `.`/`..`) with ValueError, before any mkdir/write (src/pyutilz/web/cached_client.py:43, :122-124).
- **Category**: path-traversal
- **Problem**: `_cache_path` hashes the URL (so the *filename* is safe) but joins the caller's `tag` verbatim as a directory segment:
  ```
  101      def _cache_path(self, url: str, tag: str) -> Path:
  103          digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
  104          return self.cache_dir / tag / f"{digest}.json"
  ```
  `tag` is a required positional parameter of the public `get_json` (line 143) and `get_text` (line 183) with no validation anywhere, and `_write_cache_entry` (line 108) does `path.parent.mkdir(parents=True, exist_ok=True)` — so a traversing `tag` both creates directories and writes files outside `cache_dir`. The sibling module `core/disk_cache.py:_key_path` (lines 272-287) has exactly this containment check, with a docstring explaining why; `cached_client` has no equivalent.
- **Failure scenario**: Any code deriving `tag` from data it did not author (an API's resource `type`/`kind` field, a config-driven per-source label, a CLI argument) can be given `tag="../../../../home/user/.config"`; the client then `mkdir -p`s and writes attacker-shaped JSON files anywhere the process can write, outside the intended cache root, and `get_json` reads them back on the next call.
- **Suggested fix**: Mirror `disk_cache._key_path`: resolve the candidate and refuse it when `self.cache_dir.resolve()` is not among `candidate.parents`; or reject any `tag` that is not `re.fullmatch(r"[A-Za-z0-9_.-]+", tag)`, and reject `.`/`..` explicitly.

### F05. [Low] Full FileMaker session-token response body logged at WARNING on the malformed-token path — src/pyutilz/core/filemaker.py:74
- **Disposition**: COMPLETED - the auth response body is no longer logged; only its top-level key list (or type name) is (src/pyutilz/core/filemaker.py:73-80).
- **Category**: credential-leak
- **Problem**: `get_session_token` parses the auth response and, when the extracted token is falsy or not a `str`, logs the **entire decoded JSON response object**:
  ```
  71                  def_token = get_attr(response_field if isinstance(response_field, dict) else {}, "token")
  72                  if not def_token or not isinstance(def_token, str):
  73                      logger.warning("Empty filemaker session token: %s", res)
  ```
  `res` at that point is the whole `res.json()` payload of the `/sessions` endpoint — the response whose entire purpose is to carry a session bearer token. The guard fires precisely when the token is *not* a string at `response.token`, which includes the case where the server returns it under a different key or a nested shape, i.e. exactly when `res` still contains a live token. The module also holds the Basic-auth credentials it just sent (lines 58-59), making this log stream credential-adjacent by construction.
- **Failure scenario**: A FileMaker Data API version/deployment returns the token at a slightly different JSON path (or wraps it in an object). The guard trips, and the live bearer token is written verbatim into the application's WARNING-level log — the same durable, aggregated, long-retained log stream the round-1 Critical on `report_params` was fixed to keep credentials out of.
- **Suggested fix**: Log only the response's shape, not its content — e.g. `logger.warning("Empty/invalid filemaker session token; response keys=%s", sorted(res) if isinstance(res, dict) else type(res).__name__)`, consistent with `web/web.py`'s `_redact_proxy_url` discipline.

### F06. [Low] `urlopen_checked` validates only the initial scheme; `urllib` still follows redirects to `ftp://` — src/pyutilz/web/url_guard.py:64
- **Disposition**: COMPLETED - `urlopen_checked` now opens through a dedicated opener carrying `_CheckedRedirectHandler`, which re-applies `require_http_url` to every redirect target, so an `ftp://` (or any non-allowed-scheme) hop raises UnsafeURLError (src/pyutilz/web/url_guard.py:48-64, :86-89). Host/IP-range filtering is documented as explicitly out of scope in the handler docstring.
- **Category**: ssrf
- **Problem**: `require_http_url` (lines 35-51) checks the scheme of the URL the caller supplies, then line 64 calls `urllib.request.urlopen(target, timeout=timeout)` with the default opener. `urllib.request.HTTPRedirectHandler.redirect_request` permits redirect targets whose scheme is `http`, `https` **or `ftp`** — the guard's `ALLOWED_SCHEMES = frozenset({"http", "https"})` (line 22) is never re-applied to the redirect chain. The module docstring frames itself as "a single checked entry point for outbound HTTP requests built from untrusted data", so a caller reasonably assumes the allow-list holds for the whole fetch, not just the first hop. Its one in-repo consumer, `web/cached_client.py:118`, passes `self.allowed_schemes` through and inherits the same gap.
- **Failure scenario**: An untrusted `https://` URL is fetched through `urlopen_checked`. The server answers `302 Location: ftp://internal-host/...`, and urllib transparently follows it, opening an FTP connection to an internal host that the caller's allow-list was written to forbid. (Redirects to internal `http://` addresses — localhost, 169.254.169.254, RFC1918 — are likewise unblocked, but that is arguably outside this module's stated scheme-only scope; the `ftp` case is a direct violation of the allow-list the module does claim to enforce.)
- **Suggested fix**: Install a custom opener whose `HTTPRedirectHandler.redirect_request` calls `require_http_url(newurl, allowed_schemes)` before allowing the hop, and use it in place of the module-level default opener. Document explicitly whether host/IP-range filtering is in scope.

### F07. [Low] Pickle payloads and cache entries are written with default (world-readable) permissions — src/pyutilz/core/serialization.py:211
- **Disposition**: COMPLETED - all three write paths now create their temp file 0o600 so `os.replace` carries the restrictive mode onto the target: `os.open(..., O_EXCL, 0o600)` (src/pyutilz/core/serialization.py:229), `os.fdopen(os.open(tmp, ..., 0o600), "wb")` (src/pyutilz/core/safe_pickle.py:268) and the same in `DiskCache.put` (src/pyutilz/core/disk_cache.py:373). `web/cached_client.py` inherits this through `atomic_write_bytes`.
- **Category**: file-permissions
- **Problem**: Every write path in the serialization/cache stack creates its temp file with default permissions and then `os.replace`s it onto the target, so the final file inherits them:
  - `core/serialization.py:211` — `fd = os.open(tmp_path, os.O_CREAT | os.O_WRONLY | os.O_EXCL)` with **no `mode` argument**, so the mode defaults to `0o777 & ~umask`. Verified on this box: the resulting file is `0o666` (POSIX with the usual `umask 022` gives `0o644`).
  - `core/safe_pickle.py:266` — `with open(tmp, "wb") as f:` (default `0o666 & ~umask`).
  - `core/disk_cache.py:357` — `with open(tmp_path, "wb") as f:` (same).

  `web/cached_client.py:110` routes through `atomic_write_bytes`, so cached HTTP response bodies inherit the same mode. This contrasts with the deliberate hardening added for the round-1 credential finding at `web/browser.py:279` (`os.chmod(pluginfile, 0o600)`), showing the project already treats restrictive modes as the right default where the content is sensitive.
- **Failure scenario**: On a shared/multi-user host, any local user can read another user's `DiskCache` entries, `safe_dump` payloads, or cached API responses — which for this library routinely hold API results, scraped pages, or model/feature state. The sha256 sidecar protects integrity, not confidentiality, and does nothing here.
- **Suggested fix**: Pass an explicit mode at creation: `os.open(tmp_path, os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o600)` in `serialization.py:211`, and for the two `open(..., "wb")` sites either use the same `os.open` + `os.fdopen` pattern or `os.chmod(tmp, 0o600)` immediately after creation and before `os.replace`. If world-readability is intentional for a shared cache directory, make it an explicit constructor argument rather than an accident of `umask`.

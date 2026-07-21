# Domain audit: web (HTTP, browser automation, GraphQL, proxy) + cloud (S3/GCS)

## Summary

Covered all 11 assigned files in full: `src/pyutilz/web/{__init__.py,browser.py,graphql.py,web.py}`,
`src/pyutilz/web/proxy/{__init__.py,base.py,decodo.py,ip_check.py,session.py}`, and
`src/pyutilz/cloud/{__init__.py,cloud.py}`. One shared helper outside the primary scope
(`pyutilz/text/strings/configfiles.py::read_config_file`) was also read to verify a scoping-bug
hypothesis. Several findings below were empirically verified with tiny isolated Python
reproductions (not just read) using the interpreter and packages actually installed in this
environment (selenium 4.39.0, httpx 0.28.1, curl_cffi 0.15.0, requests) — details noted inline.

Overall assessment: the newer `proxy/` package (base.py/decodo.py/session.py/ip_check.py) is
well-designed and mostly solid (thread-safe health tracking, proper context-manager session
cleanup, consistent timeouts on Decodo API calls) with a couple of real edge-case gaps. The
older, larger `web.py`/`browser.py` modules (long-lived, globals-based, clearly evolved
organically) carry the bulk of the real bugs: several unbounded-timeout hangs, a retry loop with
no inter-attempt backoff, a proxy-rotation crash on a supported (unauthenticated-proxy)
configuration, an unguarded infinite busy-loop, and a `.close()` vs `.quit()` driver-process
leak. `cloud/cloud.py` contains one severe, 100%-reproducible bug: `connect_to_s3()` never
actually applies the credentials it reads from the config file, due to a local/global variable
shadowing mistake — verified via isolated reproduction. `graphql.py`'s `text_to_graphql()` is a
verified complete no-op that defeats its own stated purpose, and is not caught by the test suite
because that test's own assertion is tautological.

Findings below, most severe first: 1 Critical, 13 High, 8 Medium, 4 Low.

## Findings

### [CRITICAL] `connect_to_s3()` never applies the credentials it reads — always authenticates with `None`/`None` — src/pyutilz/cloud/cloud.py:78-93
- **Category**: correctness / security
- **Problem**: `connect_to_s3()` declares `aws_access_key_id = None` and `aws_secret_access_key = None` as **local** variables (lines 88-89), then calls `read_config_file(file=file, object=globals(), section="S3", variables="aws_access_key_id,aws_secret_access_key")` (line 90). `read_config_file` (`src/pyutilz/text/strings/configfiles.py:60-64`) writes results via plain dict-item assignment `object[var] = val`. Passing `object=globals()` makes it write into the **module's global namespace dict**, not into the function's local variables — Python resolves `aws_access_key_id` inside `connect_to_s3()` as the local binding created at line 88 for the entire function body (compile-time scoping), which is never touched by the `globals()` mutation happening in a different dict object. The subsequent `boto3.Session(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)` (line 91) therefore always receives `None, None`, regardless of what is in `settings.ini`.
- **Failure scenario**: A user creates `settings.ini` with `[S3]\naws_access_key_id = AKIA...\naws_secret_access_key = ...` and calls `connect_to_s3()` expecting to authenticate with those credentials. Verified via isolated reproduction of the exact pattern (mock `read_config_file` that does `object[var]=val`, called with `object=globals()`): the local variables printed `None`/`None` after the call, while `globals()['aws_access_key_id']` correctly held the value. In an environment with no ambient AWS credentials (env vars / `~/.aws/credentials` / IAM role), every subsequent `s3.meta.client.head_object(...)` / `download_file(...)` call raises `NoCredentialsError`. Worse, in an environment where *some* ambient credentials exist (e.g. an EC2/ECS IAM role, or a developer's personal default AWS CLI profile — both common), the code silently authenticates as that **unrelated** identity instead of the one configured in `settings.ini`, with no error at all — `get_from_s3_or_cache()` may then silently read/write against the wrong AWS account/bucket.
- **Suggested fix**: Read into a local dict and pull the values back out, e.g. `creds: dict = {}; read_config_file(file=file, object=creds, section="S3", variables="aws_access_key_id,aws_secret_access_key"); aws_access_key_id = creds.get("aws_access_key_id"); aws_secret_access_key = creds.get("aws_secret_access_key")`, then pass those into `boto3.Session(...)`.

### [HIGH] `get_url()` retries any HTTP verb — including POST/PATCH — with no idempotency guard — src/pyutilz/web/web.py:387-569
- **Category**: correctness / security (data integrity)
- **Problem**: `get_url(url, verb="post", data=..., json=..., max_retries=10, retry_statuses=(...))` will, on any network exception (line 478) or any status in `retry_statuses` (line 547), loop back and re-issue the *exact same* request (line 473: `method(url, headers=..., params=..., data=..., json=..., proxies=..., timeout=...)`) up to `max_retries` times, with no idempotency-key mechanism and no distinction between safe verbs (GET/HEAD/PUT/DELETE) and unsafe ones (POST/PATCH).
- **Failure scenario**: Caller does `get_url(order_api_url, verb="post", json=order_payload, max_retries=10)` against an order/payment-creation endpoint. The server processes the POST and creates the order, but the response is lost in transit (a `ConnectionResetError`/`ChunkedEncodingError` — realistic given this library is built around proxy rotation, where mid-response connection drops are common). `get_url()` cannot distinguish "request never reached the server" from "response never reached the client" and automatically re-submits the same POST, creating a duplicate order/charge.
- **Suggested fix**: Document the non-idempotent-retry risk prominently, and/or let callers opt into "retry only if verb in a caller-declared idempotent set" or supply an idempotency key/header that's threaded through unchanged across retries.

### [HIGH] `download_in_parallel()` dereferences a possibly-`None` response before its own None-check, crashing the whole batch — src/pyutilz/web/web.py:173-191
- **Category**: correctness / efficiency
- **Problem**: `for resp, sub_url in zip(grequests.map(rs, size=nparallel_downloads), urls_to_process):` then immediately `if len(resp.history) > 0:` (line 175) — **before** the `if resp is not None:` guard at line 179. `grequests.map()` puts `None` into the results list for a request that raised (no `exception_handler` is passed here), which the function's own docstring explicitly anticipates ("Returns the list of URLs that errored (non-200 status, exception in `func`, or None response)" — line 158-159), and which the (misplaced) `if resp is not None:` check at line 179 was clearly written to guard against.
- **Failure scenario**: Any single URL among `urls_to_process` fails at the connection level (DNS failure, connection reset, TLS error — realistic in bulk crawling) → `grequests.map()` yields `None` for that entry → line 175's `resp.history` raises `AttributeError: 'NoneType' object has no attribute 'history'`, uncaught, aborting `download_in_parallel()` entirely (including all URLs not yet processed) instead of recording just that one URL in `errored_urls` as intended.
- **Suggested fix**: Move the `if resp is not None:` check before the `resp.history` access, e.g. restructure so the `None` branch is handled first and `resp.history`/`resp.status_code` are only touched afterward.

### [HIGH] `handle_blocking()` crashes with `IndexError` when the current proxy has no username/password — src/pyutilz/web/web.py:624-634
- **Category**: correctness / edge-case
- **Problem**: `logger.warning("IP %s blocked. ...", proxies["https"].split("@")[1], target)` (line 627) assumes the proxy URL contains `user:pass@host:port`. `make_proxies_dict()` (web.py:306-313) builds an `@`-free URL (`"%s:%s" % (proxy_server, proxy_port)`, no `@`) whenever `proxy_user`/`proxy_pass` are falsy — an explicitly supported configuration, since both `connect()` and `set_proxy()` accept `Optional[str] = None` for the proxy credentials (anonymous/IP-whitelisted proxies).
- **Failure scenario**: A caller configures an unauthenticated proxy (`set_proxy(m_proxy_server="1.2.3.4", m_proxy_user=None, m_proxy_pass=None, ...)`). The target site starts blocking (a status in `blocking_statuses`, or a `blocking_errors` string match in the response body) — exactly the moment `handle_blocking()` is meant to gracefully recover from. `proxies["https"]` is `"https://1.2.3.4:8080"` with no `@`; `.split("@")` returns a 1-element list; `[1]` raises `IndexError: list index out of range`, an unhandled crash of what should be a recoverable event, aborting the whole scraping job.
- **Suggested fix**: Guard the split, e.g. `proxies["https"].split("@")[-1] if "@" in proxies["https"] else proxies["https"]`, or use `urllib.parse.urlsplit(...).hostname`.

### [HIGH] `get_external_ip()` and `get_ipinfo(use_urllib=True)` call `urllib.request.urlopen()` with no timeout — can hang forever — src/pyutilz/web/web.py:108, 135
- **Category**: efficiency / production-outage risk
- **Problem**: Both `resp = urllib.request.urlopen(_ensure_http_scheme(source))` (line 108, inside `get_external_ip()`) and `resp = urllib.request.urlopen(_ensure_http_scheme(url))` (line 135, inside `get_ipinfo()`) omit the `timeout=` argument. `urlopen`'s default timeout is `socket._GLOBAL_DEFAULT_TIMEOUT`, which resolves to "block forever" unless something has called `socket.setdefaulttimeout()` — confirmed via `grep -r setdefaulttimeout src` returning no matches anywhere in the package.
- **Failure scenario**: `get_external_ip()` shuffles `IP_PROVIDERS` and tries each in a `for` loop expecting a fast failure to move to the next provider; if the first-tried provider accepts the TCP connection but never sends a response (a stalled/black-holed connection — common with `icanhazip.com`/other free IP-check services under load, or a proxy layer sitting in front of the socket), the call blocks the calling thread **indefinitely**, and the loop never reaches the other two providers.
- **Suggested fix**: Pass an explicit `timeout=` (e.g. reuse the module's `timeout` global, defaulting to 10) to both `urlopen()` calls.

### [HIGH] `download_in_parallel()`'s `grequests.get()` calls have no timeout at all — src/pyutilz/web/web.py:170
- **Category**: efficiency / production-outage risk
- **Problem**: `rs = (grequests.get(sub_url, verify=True, allow_redirects=True, headers=headers) for sub_url in urls_to_process)` — no `timeout=` kwarg anywhere in the generator, and no default is set elsewhere in the module for this code path.
- **Failure scenario**: `grequests.map(rs, size=nparallel_downloads)` does not return until every greenlet in the batch has completed or errored. If even one URL among a large `urls_to_process` batch stalls (server accepts the connection but never responds), that single greenlet blocks forever, and since `grequests.map()` waits for the whole batch, `download_in_parallel()` never returns — hanging the caller indefinitely regardless of how many of the *other* URLs in the batch succeeded.
- **Suggested fix**: Pass `timeout=` explicitly to `grequests.get(...)` (e.g. a new function parameter defaulting to a sane value like 30s).

### [HIGH] `close_browser()` (and every browser-teardown path) calls `.close()`, never `.quit()` — leaks the WebDriver/chromedriver process — src/pyutilz/web/browser.py:115-123 (also the restart path at 407-413)
- **Category**: correctness / resource leak
- **Problem**: `close_browser()` calls `browser.close()` (line 120), and the only other teardown site, the `restart_on_no_cookie` branch in `LoginAndGetCookies()`, also calls `browser.close()` (line 410). Neither calls `.quit()` anywhere in this file (grepped: zero occurrences of `.quit(` in `browser.py`). Selenium's `.close()` only closes the *current window/tab*; `.quit()` is what tells the driver server to end the session and terminate the underlying browser + driver-binary process tree. This is a well-documented Selenium pitfall, and is more acute here because `undetectable=True` mode explicitly supports `use_subprocess=True` (line 174), meaning each browser instance is backed by its own OS subprocess.
- **Failure scenario**: A long-running scraper calls `ensure_session_is_valid()` periodically (its whole purpose is periodic re-login), and/or hits the `restart_on_no_cookie=True` path in `LoginAndGetCookies()` repeatedly over the process lifetime. Each restart cycle calls `close()` then `start_selenium()` again, spawning a brand-new chromedriver/undetected_chromedriver process while the old one is left running (or at best zombied) — memory and process-table entries accumulate without bound over the life of a long-running job, eventually exhausting host resources.
- **Suggested fix**: Replace `browser.close()` with `browser.quit()` in both teardown sites; consider wrapping browser lifecycle in a context manager (`__enter__`/`__exit__` calling `.quit()`) for guaranteed cleanup on exceptions too.

### [HIGH] `LoginAndGetCookies()`'s session-refresh loop can spin forever with no delay on any unrecognized exception — src/pyutilz/web/browser.py:296-317
- **Category**: correctness / resource exhaustion
- **Problem**:
```python
while True:
    if browser is None:
        browser = start_selenium(); break
    else:
        try:
            browser.refresh()
            browser.execute_cdp_cmd(...)
        except Exception as e:
            logger.exception(e)
            if "window was already closed" in str(e) or "window already closed" in str(e) or "chrome not reachable" in str(e):
                logger.info("Restarting webdriver")
                browser = None
        else:
            break
```
  If `browser.refresh()`/`execute_cdp_cmd()` raises any exception whose message does **not** contain one of those three specific substrings, the `except` block does nothing (`browser` stays non-`None`), there is no `break`, no `sleep()`, and no re-raise — control falls straight back to the top of `while True:` and immediately retries the same failing call.
- **Failure scenario**: Any of the many other Selenium/CDP exceptions that don't match those three literal substrings (e.g. `InvalidSessionIdException`, a JavaScript exception from `execute_cdp_cmd`, a stale-element/timeout error, a translated non-English driver message) causes an unbounded, zero-delay busy loop — the function never returns, pinning a CPU core at 100% indefinitely, with `logger.exception(e)` re-logging the same traceback every iteration (log-flooding as a side effect).
- **Suggested fix**: Add an attempt cap and/or a `sleep()` before retrying; on the "else" (unmatched) branch, either `raise` or `return False` instead of silently looping.

### [HIGH] `text_to_graphql()` is a complete no-op — the "escaping" never happens — src/pyutilz/web/graphql.py:142-149
- **Category**: correctness
- **Problem**: `return text.replace(r"\n", "\\" + "n")`. The search pattern `r"\n"` is the 2-character string `\n` (backslash, n). The replacement `"\\" + "n"` is *also* the 2-character string `\n` (a normal, non-raw Python string literal `"\\"` is a single backslash character). Search and replacement are byte-for-byte identical, so `.replace()` changes nothing. Verified empirically: `text_to_graphql("line1\\nline2")` (a string containing the literal 2-char sequence `\n`) returns the exact same string unchanged (`unchanged: True` when checked programmatically). This directly contradicts the function's own docstring, which claims the 2-character sequence is "doubled into ``\\n`` (an escaped backslash followed by ``n``)" — i.e. it should become a 3-character sequence, but it never does.
- **Failure scenario**: A caller has text containing a literal `\n` (e.g. user-entered "use \n for line breaks") and calls `text_to_graphql(text)` before splicing it into a GraphQL query/variable string literal, expecting the backslash to be escaped so the downstream GraphQL/JSON parser treats it as literal text rather than an actual newline escape. Since the function is a no-op, the literal `\n` survives unescaped into the query text and gets interpreted by the GraphQL/JSON parser as a real newline escape, silently corrupting the text (100% reproducible on every call — this is not a probabilistic edge case). This is not caught by `tests/test_graphql.py:9` because that test's own expected value, `"line1\\nline2"` (non-raw), is textually identical to its raw-string input `r"line1\nline2"` — the assertion is tautologically true regardless of whether the function does anything.
- **Suggested fix**: `return text.replace("\\", "\\\\")` (or, to match the docstring's narrower "only double when followed by `n`" framing, a regex substitution) — and fix/strengthen the existing test so its expected value is actually distinct from the input.

### [HIGH] `get_ip()`'s exception tuple silently breaks the documented httpx-compatibility promise — src/pyutilz/web/proxy/ip_check.py:14, 72-83
- **Category**: correctness / edge-case
- **Problem**: The module docstring (line 3-4) advertises: "Shared helpers that work with any HTTP client (`requests`, `curl_cffi`, `httpx`, etc.)". `get_ip()` catches only `(OSError, ValueError, KeyError)` (line 81) when trying each of the 3 `IP_CHECK_URLS` in turn. Verified in this environment: `requests.exceptions.RequestException` **is** an `OSError` subclass (`issubclass(...) == True`), and `curl_cffi.requests.RequestsError` **is** an `OSError` subclass (`True`) — so those two named clients work correctly. But `httpx.HTTPError` (the base of all httpx exceptions, including `ConnectError`/`TimeoutException`) is **not** an `OSError`/`ValueError`/`KeyError` subclass at all — it subclasses plain `Exception` directly (verified: `issubclass(httpx.HTTPError, OSError) == False`, httpx 0.28.1).
- **Failure scenario**: A caller uses an `httpx.Client`-based session (exactly the third client the docstring names) and calls `get_ip(httpx_client_session)`. If the *first* `IP_CHECK_URLS` entry (`https://httpbin.org/ip`) times out or the connection fails, `httpx.TimeoutException`/`httpx.ConnectError` is raised and is **not** caught by the `except (OSError, ValueError, KeyError):` clause — it propagates straight out of `get_ip()` uncaught, instead of falling through to try `api.ipify.org` / `ifconfig.me` as the function's own docstring promises ("Try IP_CHECK_URLS in order, return first successful IP or '?'").
- **Suggested fix**: Broaden the catch to the client-agnostic base, e.g. `except Exception:` (the function already has a graceful `"?"` fallback for total failure, so a blanket catch here is safe), or explicitly import and include `httpx.HTTPError` when available.

### [HIGH] `find_element_by_xpath/by_name/by_tag_name()` fall back to Selenium APIs that no longer exist — src/pyutilz/web/browser.py:77-108
- **Category**: correctness / maintainability
- **Problem**: All three helpers do `try: browser.find_element(By.X, query) except Exception: res = browser.find_element_by_x(query)`. Verified empirically in this environment: with `selenium==4.39.0` installed (satisfying the project's own unbounded `selenium>=4.0` dependency pin in `pyproject.toml:75`), `WebDriver` has **no** `find_element_by_xpath`/`find_element_by_name` attributes at all (`hasattr(...) == False`) — these methods were removed from Selenium's Python bindings, not merely deprecated.
- **Failure scenario**: On any current `pip install pyutilz[web]` (which resolves to a recent Selenium 4.x release, since there's no upper bound), the intended "element genuinely not present" case (Selenium raises `NoSuchElementException` from `find_element(By.X, query)`) falls into the `except` branch and calls the now-nonexistent legacy method, raising `AttributeError: 'WebDriver' object has no attribute 'find_element_by_xpath'` instead. Any caller catching the expected `NoSuchElementException` specifically won't catch this `AttributeError`, and the real, actionable error ("element not found") is masked by a confusing, unrelated one.
- **Suggested fix**: Drop the legacy fallback entirely (Selenium 4.0+, which is the project's own floor, always has `find_element(By, value)`); it serves no purpose and actively obscures the real exception.

### [HIGH] `get_new_smartproxy()` can silently hang the calling thread for ~24h on a single transient failure, for the officially-documented "fixed sticky-session port" proxy pattern — src/pyutilz/web/web.py:316-379, 637-644
- **Category**: correctness / proxy statistical correctness
- **Problem**: `is_rotating_proxy()` (web.py:637-644) explicitly documents and recognizes a legitimate configuration pattern: a single fixed proxy port where the *server side* rotates the exit IP (e.g. Smartproxy/Decodo `gate.dc.smartproxy.com:20000` with `PROXY_MIN_PORT==PROXY_MAX_PORT==20000`). In that configuration, `get_new_smartproxy(proxy_port=<fixed>)` (line 350: `if proxy_port is None: ...` is skipped) always computes the *same* `proxies` dict / `proxy_key` on every call. If that single proxy ever hits one transient error (matched by `get_url()`'s except-block substring check at line 483), `handle_blocking()`/the except-branch marks it in `failed_dict` for `min_failed_idle_interval_minutes` (default `60 * 24` = 24 hours, e.g. web.py:497/534/615/273). Every subsequent call to `get_new_smartproxy()` then finds the *same* single candidate `proxy_key` "touched recently" and loops (`while True:` at line 346) sleeping `delay` seconds (default param value 5) every `warn_after_n_failures` (default 5) attempts — but **every call site in this module invokes it with the default `verbose=False`** (checked: none of `get_url()`'s two call sites, `get_new_session()`, or `set_proxy()` pass `verbose=True`), so the only log statement inside this retry loop (`if verbose: logger.info("Could not get an untouched proxy...")`, lines 376-377) never fires.
- **Failure scenario**: A single-proxy (or fixed-port, server-rotated) deployment — the exact pattern `is_rotating_proxy()` was written to recognize — has one momentary network blip. The calling thread then blocks inside `get_new_smartproxy()` for up to 24 hours, producing **zero log output** the entire time (just a `sleep(5)` roughly every 30 seconds), making the hang effectively undiagnosable from logs alone; the only symptom is "the job stopped making progress."
- **Suggested fix**: Add a `max_wait_attempts`/timeout parameter to `get_new_smartproxy()` that raises instead of blocking forever, and/or unconditionally log (not gated on `verbose`) when the wait exceeds some threshold (e.g. once every N seconds regardless of `verbose`).

### [HIGH] `get_from_s3_or_cache()` spins forever, with zero delay, if the cached `.zip` is corrupted — src/pyutilz/cloud/cloud.py:105-147
- **Category**: correctness / efficiency
- **Problem**: In the branch where a local `local_object_path + ".zip"` already exists (`bDownload = False`, lines 114-117/135-146), `shutil.unpack_archive(...)` (line 141) is wrapped in `try/except Exception as e: logger.error(...)` with **no `sleep()`** and the corrupted zip is **not removed** (only the success path at lines 144-146 calls `os.remove(...)`). Contrast with the sibling "not found in bucket" branch just above it (lines 132-134), which does `sleep(10)` on its failure path.
- **Failure scenario**: A previous download was interrupted or corrupted, leaving a truncated/invalid `local_object_path + ".zip"` on disk. `get_from_s3_or_cache()` is called: `not exists(local_object_path)` is `True` (never unpacked) → `bDownload=False` (zip exists) → `shutil.unpack_archive()` raises `zipfile.BadZipFile` (or similar) every time, since the file content never changes and is never deleted → the `while not exists(local_object_path):` loop (line 112) immediately re-enters with the exact same state, calling `unpack_archive()` again with no delay — an unbounded, zero-sleep busy loop pinning one CPU core at 100% forever, requiring an external process kill to recover.
- **Suggested fix**: On unpack failure, either `os.remove(local_object_path + ".zip")` so the next loop iteration re-downloads, or add a `sleep()`/attempt cap on this branch matching the "not found" branch's behavior.

### [MEDIUM] `get_url()`'s retry loop has no inter-attempt delay for generic network errors or generic retry-statuses — thundering-herd on a failing target — src/pyutilz/web/web.py:429-569
- **Category**: efficiency / retry-backoff correctness
- **Problem**: The only unconditional `sleep()` in `get_url()` is `if delay: sleep(delay * random())` at lines 564-565, which sits at the **same indentation level as `while n_retries < max_retries:`** (line 429) — i.e. it runs exactly **once**, *after* the retry loop exits (whether by `break` or by exhausting `max_retries`), not between each attempt. Inside the loop, the generic `except Exception` branch (line 478) only sleeps/rotates when the stringified exception matches one of `"proxy"/"timed out"/"bad handshake"/"connection broken"/"sslerror"` (line 483); any other exception (e.g. a bare `requests.exceptions.ConnectionError` from DNS failure, whose message contains none of those substrings) falls straight through with **no delay at all** back to the top of the `while` loop. Likewise, a status explicitly listed in `retry_statuses` (checked at line 547) loops back immediately with no delay.
- **Failure scenario**: `get_url(url, max_retries=10, retry_statuses=(500, 502, 503))` against a target that is briefly overloaded (returning 503s) or has a DNS/connection issue not matching the 5 substring checks: up to 10 requests fire back-to-back as fast as Python can loop, with no backoff and no jitter between them — the opposite of good retry hygiene, and likely to make an already-struggling target (or an anti-bot system watching for exactly this pattern) worse, not better.
- **Suggested fix**: Move a `sleep()` (ideally with exponential backoff + jitter) inside the loop, at the end of each failed iteration, rather than only once after the loop.

### [MEDIUM] `PortHealthTracker` can never ban a port when a pool has fewer than 2 "qualified" ports — "never blacklisted despite repeated failures" for small pools — src/pyutilz/web/proxy/base.py:99-106, 116, 139-162
- **Category**: proxy statistical correctness / design
- **Problem**: `_maybe_ban()` requires `len(qualified) >= 2` (ports with `>= min_requests` outcomes each) to use the peer-comparison ban path (lines 146-147). When fewer than 2 ports are "qualified" (the common case for a small proxy pool, or `ProxyConfig(port_range=1)`, or simply early in a run), the only remaining path is the `absolute_ban_rate` fallback (lines 153-161) — which **defaults to `0.0` (disabled)** per the class docstring ("Default is 0.0 (disabled) -- preserves the pre-wave-7 behaviour for existing callers"). With it disabled, `_maybe_ban()` just `return`s (line 162) and nothing is ever banned, no matter how many times that single/few port(s) fail.
- **Failure scenario**: A deployment configured with `ProxyConfig(port_range=1)` (a single dedicated proxy IP, a realistic Decodo/datacenter-proxy setup) has that one port failing 100% of the time (e.g. IP got hard-blocked by the target). Since there can never be 2 qualified ports in a 1-port pool, and `absolute_ban_rate` is `0.0` by default, `PortHealthTracker.is_banned()` returns `False` forever — the health-tracking/auto-ban feature this class exists for is a complete no-op for this deployment shape, silently.
- **Suggested fix**: This is a documented, deliberate default (not silently broken), but worth surfacing: consider defaulting `absolute_ban_rate` to a positive value for pools with `port_range` below some small threshold, or at minimum call this limitation out explicitly in `ProxyProvider`'s docs/README so operators of small pools know to opt in.

### [MEDIUM] `browser_get()` claims to retry but silently does not — the docstring and behavior diverge — src/pyutilz/web/browser.py:125-136
- **Category**: correctness / docs-code mismatch
- **Problem**: Docstring: "retrying once after a short sleep on transient loading-status errors." Actual code:
```python
try:
    ...
    browser.get(path)
except Exception as e:
    if ("cannot determine loading status" in str(e)) or ("unexpected command response" in str(e)):
        sleep(2)
    else:
        raise(e)
```
  On the matching branch, it only calls `sleep(2)` and then the function simply returns (no second call to `browser.get(path)`) — there is no retry.
- **Failure scenario**: `browser.get(home_page)` in `LoginAndGetCookies()` raises a "cannot determine loading status" transient error. `browser_get()` sleeps 2s and returns normally (no exception propagates), so the outer `while True: try: browser_get(home_page) ... else: break` in `LoginAndGetCookies()` (lines 320-333) treats this as success and proceeds to inspect `browser.title` / attempt login — but the browser may still be on the *previous* page (navigation to `home_page` never actually completed), so the login-detection logic (`anyof_elements_in_string(logout_signs, browser.title)`) can act on stale page content.
- **Suggested fix**: Actually retry: `try: browser.get(path) except ...: sleep(2); browser.get(path)` (or loop with a small retry counter).

### [MEDIUM] `ensure_session_is_valid(interval_minutes=None)` crashes with `TypeError` despite the type annotation allowing `None` — src/pyutilz/web/browser.py:274-286
- **Category**: correctness / mypy-discipline (CLAUDE.md)
- **Problem**: `def ensure_session_is_valid(interval_minutes: Optional[int] = 10) -> None:`. When a session already exists (`last_session_updated_at is not None`), the code does `(now_time - last_session_updated_at).total_seconds() / 60 >= interval_minutes` (line 282) with no `None` guard on `interval_minutes` itself.
- **Failure scenario**: A caller passes `ensure_session_is_valid(interval_minutes=None)` — a value the function's own `Optional[int]` annotation explicitly invites (e.g. to mean "always/never re-validate") — after the session has already been established once. `float >= None` raises `TypeError: '>=' not supported between instances of 'float' and 'NoneType'`, an unhandled crash.
- **Suggested fix**: Either guard explicitly (`if interval_minutes is None or elapsed >= interval_minutes:`) or change the annotation to a non-Optional `int` with a real default, per this project's own CLAUDE.md mypy-discipline rule ("never write `param: T = None`" / match behavior to the annotation).

### [MEDIUM] `s3_file_exists()`'s blanket `except Exception` conflates "object missing" with "client not configured"/credentials errors — src/pyutilz/cloud/cloud.py:96-103
- **Category**: correctness / silent error-swallowing
- **Problem**: `except Exception: return False` catches everything, including the module-level `s3` global being `None` (if `connect_to_s3()` was never called, or failed — `s3: _Any = None` at line 27), which raises `AttributeError: 'NoneType' object has no attribute 'meta'`. It's indistinguishable from a genuine "404, object not present" outcome.
- **Failure scenario**: `s3_file_exists()` is called before `connect_to_s3()` has succeeded (or after it silently failed, e.g. due to the `connect_to_s3()` bug reported above producing `NoCredentialsError` on the first real API call). `get_from_s3_or_cache()`'s caller sees a misleading `"Model %s not found in bucket %s"` log message (cloud.py:133) and enters its `sleep(10)`-polling loop *forever*, waiting for a file that will never appear "uploaded" no matter how long it waits, because the real root cause (missing/broken client) is masked.
- **Suggested fix**: Catch a narrower, S3-specific "not found" exception (e.g. `botocore.exceptions.ClientError` with a 404 code), and let other exceptions (auth/network/attribute errors) propagate or be logged distinctly.

### [MEDIUM] `PortHealthTracker.pick_port(port_range=0)` raises `ValueError` — src/pyutilz/web/proxy/base.py:217-224
- **Category**: edge-case / correctness
- **Problem**: `random.randint(1, port_range)` (line 224, and again at 226/238/239) requires `1 <= port_range`; `port_range=0` raises. `ProxyConfig.port_range` (base.py:31) has no validation preventing `0`, and `0` is a plausible way to express "single fixed proxy, no rotation range" — an interpretation directly analogous to the pattern `web.py`'s `is_rotating_proxy()` already recognizes for the *other* proxy subsystem in this same package (fixed min==max port).
- **Failure scenario**: `DecodoProvider(ProxyConfig(user=..., password=..., host=..., base_port=12345, port_range=0))`, then `.proxy_url()` → `pick_port()` → `self.health.pick_port(self.config.port_range)` → `random.randint(1, 0)` → `ValueError: empty range in randint(1, 0)` (verified empirically). Reproducible via `DecodoProvider.from_env()` too, if `PROXY_PORT_RANGE=0` is set.
- **Suggested fix**: Special-case `port_range <= 1` (or `<= 0`) to return a fixed offset (e.g. `0` or `1`) without calling `random.randint`.

### [MEDIUM] `get_url()` reads several `proxy_*` globals outside `_state_lock`, contradicting the module's own documented locking contract — src/pyutilz/web/web.py:34-38, 488-500, 525-537
- **Category**: correctness / concurrency
- **Problem**: The module-level comment (lines 34-38) explicitly states the lock guards "...`num_ip_queries`, `cur_max_ip_queries`, `was_blocked`, `proxy_*` fields..." when this module is driven from multiple threads. But the exception-handling proxy-rotation branch (lines 488-500) and the rate-limiting branch (lines 525-537) read `proxy_user, proxy_pass, proxy_server, proxy_min_port, proxy_max_port, proxy_port, proxy_type` directly as bare module globals with **no** `with _state_lock:` wrapper — unlike `sess`/`proxies`/`headers`, which are explicitly snapshotted under the lock a few lines earlier (lines 444-447).
- **Failure scenario**: Thread A is mid-way through `get_url()`'s except-block, reading `proxy_min_port`/`proxy_max_port`/`proxy_type` unprotected to build a replacement proxy. Concurrently, Thread B calls `set_proxy()` (which *does* take `_state_lock` for its writes at web.py:249-258) to reconfigure the proxy pool. Thread A can observe a torn mix of old and new values (e.g. old `proxy_min_port` paired with new `proxy_type`), producing an internally-inconsistent proxy configuration passed to `get_new_smartproxy()`.
- **Suggested fix**: Snapshot the `proxy_*` fields under `_state_lock` the same way `sess`/`proxies`/`headers` already are, at the top of each `except`/branch that reads them.

### [MEDIUM] `LoginAndGetCookies()`'s recursive retry silently drops caller overrides and has no depth limit — src/pyutilz/web/browser.py:396-414
- **Category**: correctness / robustness
- **Problem**: When a required cookie is missing and `restart_on_no_cookie=True`, the function recurses: `return LoginAndGetCookies(default_headers=default_headers)` (line 414) — passing **only** `default_headers`, dropping whatever `seconds_to_sleep_on_error` and `restart_on_no_cookie` the *original* caller supplied. Every subsequent recursive call therefore reverts to the function's defaults (`seconds_to_sleep_on_error=60`, `restart_on_no_cookie=False`), regardless of what was originally requested. There is also no recursion-depth cap.
- **Failure scenario**: Caller invokes `LoginAndGetCookies(seconds_to_sleep_on_error=5, restart_on_no_cookie=True)` expecting a fast 5s retry cadence with auto-restart on every failure. If the target site's cookie name changed (or the account is permanently locked out), the first recursive call already silently reverts to `seconds_to_sleep_on_error=60` and `restart_on_no_cookie=False` — so subsequent failures merely `sleep(60)` and recurse again *without* restarting the browser, indefinitely. Because this is real Python recursion (not a loop), a persistently-failing cookie check will eventually raise `RecursionError` (after ~1000 frames, i.e. many hours later given the 60s sleep each level) rather than failing gracefully.
- **Suggested fix**: Convert to an explicit `while` loop (preserving all original parameters across iterations) with a bounded retry count, instead of unbounded self-recursion.

### [LOW] `start_selenium()` indexes `proxy_server['PROXY_HOST']`/`['PROXY_PORT']` directly (no `.get()`) unlike the `PROXY_PASS` check just above — src/pyutilz/web/browser.py:257
- **Category**: edge-case / consistency
- **Problem**: Line 198 uses `.get("PROXY_PASS", "")` defensively, but line 257's else-branch does `proxy_server['PROXY_HOST']` / `proxy_server['PROXY_PORT']` with plain indexing.
- **Failure scenario**: `proxy_server = {"PROXY_USER": "u", "PROXY_PASS": ""}` (missing HOST/PORT keys, e.g. a partially-filled config) → `KeyError: 'PROXY_HOST'` instead of a clear configuration-validation error.
- **Suggested fix**: Use `.get(...)` with a clear `raise ValueError(...)` if the required keys are absent, for a more actionable error message.

### [LOW] `PortHealthTracker.pick_port()`'s fallback doesn't actually guarantee "an unbanned port," and only logs when >=90% banned even though the same unchecked fallback fires below that threshold too — src/pyutilz/web/proxy/base.py:217-239
- **Category**: docs-code mismatch / statistical correctness
- **Problem**: Docstring: "Falls back to any port if >90% are banned." Code: after 10 failed random attempts (lines 225-228), it expires stale bans (229-232) then **unconditionally** returns `random.randint(1, port_range)` (lines 238 and 239) without re-checking whether that specific port is banned — regardless of whether the actual ban ratio is above or below 90%. The `_log.warning(...)` (lines 233-237) is gated on `len(self._banned) >= port_range * 0.9`, but the unchecked-fallback *behavior* itself is not gated on that threshold — it's the same in both branches.
- **Failure scenario**: With, say, 50% of a 500-port pool banned, the probability all 10 random samples land on banned ports is low (~0.1%) but non-zero over the life of a long-running process making many `pick_port()` calls — when it happens, a caller silently receives a port known to be misbehaving, with **no** log line at all (since the 90% gate on the warning isn't met), unlike the >=90% case which at least logs.
- **Suggested fix**: Either scan the full `[1, port_range]` range for an unbanned candidate before giving up, or log a debug/info line on every "gave up after 10 tries" fallback regardless of the ban ratio.

### [LOW] `download_to_file()` has an unreachable status-code check in its exception handler — src/pyutilz/web/web.py:662-676
- **Category**: maintainability (dead code)
- **Problem**: `except Exception as e: if request is not None and request.status_code in exit_codes: return`. This is in the `except` branch of `request = requests.get(url, timeout=timeout, ...)` (line 666). If `requests.get()` itself raised, `request` was never reassigned by that statement (assignment target is only bound on success), so it always still holds its value from *before* this attempt — which, on every iteration of this specific loop, is `None` (the loop only exits via the `else: break` success path, so `request` can never hold a "failed" `Response` object here). The `request.status_code in exit_codes` check is therefore dead code that can never evaluate `request is not None` as `True`.
- **Failure scenario**: None (this doesn't cause incorrect behavior — it's just misleading dead code that could make a future maintainer believe status-code-based early exit is implemented here when it isn't).
- **Suggested fix**: Remove the dead check, or restructure so `exit_codes` is genuinely checked against a response object when one exists (e.g. by not using `requests.get`'s own exception path, or by checking `exit_codes` after a successful-but-bad-status response instead).

### [LOW] `DecodoProvider.get_subscriptions()` only catches `requests.HTTPError`, not other `RequestException` subtypes — src/pyutilz/web/proxy/decodo.py:262-280
- **Category**: edge-case
- **Problem**: `except requests.HTTPError: continue` (line 278) only fires from `r.raise_for_status()`. A `requests.exceptions.Timeout` or `ConnectionError` on the first endpoint (`/v2/subscriptions`) is a `RequestException` but **not** an `HTTPError`, so it propagates straight out of `get_subscriptions()` instead of falling through to try `/v2/sub-users` or surfacing the friendlier `RuntimeError("Could not fetch subscriptions from any Decodo API endpoint")`.
- **Failure scenario**: A transient network blip on the first endpoint attempt raises an uncaught `requests.exceptions.ConnectionError` from `get_subscriptions()`, even though the second endpoint might well have succeeded.
- **Suggested fix**: Broaden to `except requests.RequestException:` if the intent is "try the next endpoint on any failure," or keep `HTTPError`-only deliberately and document that network errors are meant to propagate.

## Things done well

- `proxy/base.py`'s `PortHealthTracker` is a genuinely well-designed statistical component: it requires both a minimum sample size (`min_requests=30`) *and* a minimum absolute error count (`min_errors=2`) before a port is even considered for banning, which correctly prevents a single fluke error from blacklisting a port (verified by reading `_maybe_ban()` end to end — confirmed no over-eager-banning bug).
- `proxy/session.py`'s `curl_session()`/`requests_session()` context managers correctly use `try: yield s finally: s.close()`, guaranteeing session cleanup even if the caller's code inside the `with` block raises.
- `web.py`'s `_ensure_http_scheme()` guard (applied to both `urlopen()` call sites) is a thoughtful defensive measure against `file://`/other-scheme SSRF/local-file-disclosure via a caller-supplied URL.
- `proxy/decodo.py` consistently passes explicit `timeout=` (15s/30s) on every one of its `requests` calls — a good contrast to the timeout gaps found elsewhere in this domain.
- `graphql.py`'s `execute()` passes `variables` through the client's own parameterized-variables mechanism (`client.graphql(query, variables=variables)`) rather than string-interpolating values into the query text — the correct, injection-safe pattern.
- The `web/__init__.py` / `cloud/__init__.py` explicit `from . import <submodule>` alongside `from .X import *` is a deliberate, well-commented pattern to keep `__all__` promises bound under static analysis / lazy-import edge cases.
- `ip_check.py`'s `_json_backend` orjson/json fallback includes a clear comment explaining a previously-fixed `UnboundLocalError` bug (referencing the exact failure mode), showing genuine iterative hardening rather than a first-pass implementation.
- Extensive, specific `# nosec` justification comments throughout `web.py`/`base.py` explain precisely *why* each `random()`/`randint()` use is non-cryptographic (jitter/load-spreading), which is good documentation discipline for a security-scanner suppression.

## Investigated, not an issue

- Verified the CHANGELOG-documented historical bug ("`pyutilz/web/browser.py` importing a nonexistent `pyutilz.web.pythonlib`") is **not** present or reintroduced: `browser.py:48` correctly imports `from pyutilz.core import pythonlib`, and a repo-wide grep for `web.pythonlib`/similar patterns found only this one correct import plus the equally-correct `system/distributed.py:23`.
- `PortHealthTracker._maybe_ban()` (base.py:139-162): initially suspected a possible "banned from one transient failure" bug, but confirmed the `min_requests=30` + `min_errors=2` gates make that impossible — a lone fluke error can never trigger a ban via either the peer-comparison or absolute-rate path.
- `graphql.py`'s `beautify_gql_query()` `while join_token_find in result: result = result.replace(...)` loop (lines 163-164): initially suspected this was redundant (since `str.replace()` already replaces all non-overlapping occurrences in one call), but traced a concrete example (`"}\n}\n}"`) showing a single `.replace()` call can leave a *new* match at the boundary between the replaced text and the remainder, so the `while` loop is genuinely necessary, not dead code.
- `proxy/decodo.py`: every `requests`/`session.get()`/`.post()` call site was checked for a missing timeout; all have an explicit one (15s or 30s) — no gap found here, unlike the general `web.py`/`ip_check.py` module.
- `proxy/session.py`: both context managers were checked for a bare-`yield`-without-`finally` resource leak; both correctly use `try/finally`.

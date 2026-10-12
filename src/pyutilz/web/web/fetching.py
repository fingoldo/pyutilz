"""The retrying, proxy-rotating, session-managing HTTP fetch loop (``get_url``) plus the session
rotation (``get_new_session``) and block-handling (``handle_blocking``) it drives."""

import http
import warnings
from dataclasses import dataclass as _dataclass
from typing import TYPE_CHECKING, Any, NamedTuple as _NamedTuple, Optional, Sequence

# Imported for the ``Optional[requests.Response]`` return annotation on get_url() only -- every
# actual call goes through ``_facade.requests`` so that patching ``pyutilz.web.web.requests``
# (which tests do, to swap in a mock Session factory) is honoured here.
if TYPE_CHECKING:  # runtime calls go through _facade.requests
    import requests

from ._common import logger

# See ipinfo.py for why the parent is reached through `import <parent> as _facade`.
import pyutilz.web.web as _facade


class _FetchSnapshot(_NamedTuple):
    """Module-state values captured under ``_state_lock`` for one attempt of ``get_url``."""

    sess: Any
    proxies: Optional[dict]
    headers: Optional[dict]
    proxy_user: Optional[str]
    proxy_pass: Optional[str]
    proxy_server: Optional[str]
    proxy_min_port: Optional[int]
    proxy_max_port: Optional[int]
    proxy_port: Optional[int]
    proxy_type: Optional[str]


@_dataclass(frozen=True)
class _RequestCtx:
    """Per-call configuration of ``get_url`` shared by its response and exception handlers."""

    url: str
    target: str
    params: Optional[dict]
    data: Optional[dict]
    json: Optional[dict]
    verbose: bool
    b_random_ua: bool
    b_use_proxy: bool
    quit_on_blocking: bool
    blocking_errors: Sequence
    blocking_statuses: Sequence
    retry_statuses: Sequence
    exit_statuses: Sequence
    ratelimiting_statuses: Sequence
    session_expired_statuses: Sequence
    ratelimited_sleep_interval: int
    ratelimited_proxy_sleep_interval: int


def _ensure_session(b_random_ua: bool, b_use_proxy: bool) -> None:
    """Create a new session when none exists or the per-session IP-query budget is spent."""
    # We are trying to fetch some url. Do we need to create new proxy session?
    with _facade._state_lock:
        need_new_session = _facade.sess is None or ((_facade.max_ip_queries or 0) > 0 and _facade.num_ip_queries > _facade.cur_max_ip_queries)
    if need_new_session:
        # If there is no session yet or we have downloaded too many items within current session already
        _facade.get_new_session(b_random_ua=b_random_ua, b_use_proxy=b_use_proxy)
        if (_facade.max_ip_queries or 0) > 0:
            with _facade._state_lock:
                _facade.cur_max_ip_queries = int(  # nosec B311 - randomizes the per-session IP-query budget to avoid a fixed rotation pattern; non-cryptographic jitter, not a security control
                    (_facade.max_ip_queries or 0) * (0.6 + 0.4 * _facade.random())
                )
            logger.info("cur_max_ip_queries set to %d", _facade.cur_max_ip_queries)


def _take_snapshot() -> _FetchSnapshot:
    """Read the session, proxy and header state under ``_state_lock`` as one consistent snapshot."""
    with _facade._state_lock:
        # Regression fix: the except/ratelimiting branches used to read these proxy_* globals directly,
        # unprotected by _state_lock, unlike sess/proxies/headers -- a concurrent set_proxy() call (which DOES take
        # the lock for its writes) could interleave, producing a torn mix of old and new proxy fields (e.g. old
        # proxy_min_port paired with new proxy_type) passed to get_new_smartproxy().
        return _FetchSnapshot(
            sess=_facade.sess,
            proxies=_facade.proxies,
            headers=_facade.headers,
            proxy_user=_facade.proxy_user,
            proxy_pass=_facade.proxy_pass,
            proxy_server=_facade.proxy_server,
            proxy_min_port=_facade.proxy_min_port,
            proxy_max_port=_facade.proxy_max_port,
            proxy_port=_facade.proxy_port,
            proxy_type=_facade.proxy_type,
        )


def _prepare_headers(
    custom_headers: Optional[dict],
    base_headers: Optional[dict],
    inject_headers: Optional[dict],
    sort_headers: bool,
    lowercase_headers: bool,
) -> Optional[dict]:
    """Pick the request headers (custom over session), apply injected headers, then sort and lowercase them."""
    headers_to_use = custom_headers if custom_headers else base_headers
    if inject_headers:
        if headers_to_use:
            headers_to_use = headers_to_use.copy()
        else:
            headers_to_use = {}
        for header, value in inject_headers.items():
            headers_to_use[header] = value

    if headers_to_use:
        if sort_headers:
            headers_to_use = dict(sorted(headers_to_use.items()))  # need this to avoid bot detection
        if lowercase_headers:
            headers_to_use = {key.lower(): value for key, value in headers_to_use.items()}
    return headers_to_use


def _rotate_proxy(snap: _FetchSnapshot, proxy_server: str) -> None:
    """Obtain a new smartproxy from the snapshotted proxy settings and store it as the active proxies."""
    new_proxies = _facade.get_new_smartproxy(
        snap.proxy_user,
        snap.proxy_pass,
        proxy_server,
        int(snap.proxy_min_port) if snap.proxy_min_port is not None else 20001,
        int(snap.proxy_max_port) if snap.proxy_max_port is not None else 37960,
        last_used_dict=_facade.last_used_dict,
        min_idle_interval_minutes=_facade.min_idle_interval_minutes if _facade.min_idle_interval_minutes is not None else 0,
        failed_dict=_facade.failed_dict,
        min_failed_idle_interval_minutes=(_facade.min_failed_idle_interval_minutes if _facade.min_failed_idle_interval_minutes is not None else 60 * 24),
        proxy_port=snap.proxy_port,
        proxy_type=snap.proxy_type if snap.proxy_type is not None else "http",
    )
    # Regression fix: this assignment used to write straight to the module global with no lock, unlike every OTHER
    # write to `proxies` in this file (see get_new_session/set_proxy) -- two threads hitting a bad proxy around the
    # same time could race here, and whichever assignment landed last silently discarded the other thread's rotation.
    with _facade._state_lock:
        _facade.proxies = new_proxies


def _handle_request_exception(e: Exception, ctx: _RequestCtx, snap: _FetchSnapshot, n_retries: int, max_retries: int) -> None:
    """Log a failed attempt, rotate the proxy on proxy-related errors, then sleep the jittered retry delay."""
    se = str(e)
    # Regression fix: this was the ONLY log statement for the exception, gated behind verbose=False (the default) --
    # every network exception across all max_retries attempts was completely silent, leaving only the generic
    # "Could not get url" warning at the very end with no information about *why* (DNS/TLS/proxy-auth/timeout).
    # verbose now controls only the extra report_params() request dump; the fetch error itself is always logged.
    logger.warning("get_url attempt %d/%d for %s failed: %s", n_retries, max_retries, ctx.url, e)
    se = se.lower()
    if "proxy" in se or "timed out" in se or "bad handshake" in se or "connection broken" in se or "sslerror" in se:
        if ctx.b_use_proxy and snap.proxy_server:
            if ctx.verbose:
                logger.warning("Seems to be a bad proxy. Receiving new proxy for %s", ctx.target)
            _rotate_proxy(snap, snap.proxy_server)
    # Regression fix: this loop's only unconditional sleep() previously sat OUTSIDE the while loop (fires once, after
    # the loop exits) -- any exception not already covered by a sleep above (e.g. a bare ConnectionError whose message
    # matches none of the "proxy"/"timed out"/etc substrings) looped back to the top of `while` immediately,
    # firing up to max_retries requests back-to-back with zero backoff/jitter.
    if _facade.delay:
        _facade.sleep(_facade.delay * _facade.random())  # nosec B311 - random jitter on the per-attempt retry backoff, not security-sensitive


def _handle_ratelimited(res: Any, ctx: _RequestCtx, snap: _FetchSnapshot) -> None:
    """Back off after a rate-limit status, rotating the proxy first when one is configured."""
    if ctx.verbose:
        logger.warning("Ratelimited [%s] while getting url %s: %s", res.status_code, ctx.url, res.text)
    if snap.proxy_server:
        if ctx.verbose:
            logger.warning("Seems to be a bad proxy. Receiving new proxy for %s", ctx.target)
        if ctx.ratelimited_proxy_sleep_interval:
            _facade.sleep(  # nosec B311 - random jitter on a rate-limit backoff sleep, not security-sensitive
                ctx.ratelimited_proxy_sleep_interval * _facade.random()
            )
        _rotate_proxy(snap, snap.proxy_server)
        # Rotating the proxy is not a backoff: without this the loop re-hits a 429-ing endpoint max_retries times
        # with no pause at all (the exception and generic-error branches already jitter the same way).
        retry_after = _facade._parse_retry_after(res)
        if retry_after is not None:
            _facade.sleep(retry_after)
        elif _facade.delay:
            _facade.sleep(_facade.delay * _facade.random())  # nosec B311 - random jitter on the rate-limit retry backoff, not security-sensitive
    else:
        retry_after = _facade._parse_retry_after(res)
        _facade.sleep(retry_after if retry_after is not None else ctx.ratelimited_sleep_interval)


def _handle_error_status(res: Any, ctx: _RequestCtx, snap: _FetchSnapshot, headers_to_use: Optional[dict]) -> bool:
    """Handle a non-OK status code; return True when the retry loop should stop."""
    if res.status_code in ctx.blocking_statuses:
        logger.info("Error %s while getting %s", res.status_code, ctx.url)
        # Regression fix: this used to read the module-global `proxies` directly, unprotected by _state_lock, in a
        # function that otherwise takes pains to snapshot everything under the lock -- another unprotected read
        # alongside the unprotected writes fixed elsewhere in this function.
        _facade.report_params(ctx.url, snap.proxies, ctx.params, ctx.data, ctx.json, headers_to_use, _facade.timeout)
        _facade.handle_blocking(ctx.target, b_random_ua=ctx.b_random_ua, b_use_proxy=ctx.b_use_proxy)
        _facade.was_blocked = True
        return ctx.quit_on_blocking
    if res.status_code in ctx.session_expired_statuses:
        logger.warning("Session expired while getting url=%s, code=%s, response=%s", ctx.url, res.status_code, res.text)
        return True
    if res.status_code in ctx.exit_statuses:
        if ctx.verbose:
            logger.info("status_code %s", res.status_code)
        return True
    if res.status_code in ctx.ratelimiting_statuses:
        _handle_ratelimited(res, ctx, snap)
        return False
    logger.warning("Error %s while getting url %s: %s", res.status_code, ctx.url, res.text)

    # if blocking or exit satuses are specified, we keep retrying on any error (after small pause)
    if len(ctx.blocking_statuses) == 0 and len(ctx.exit_statuses) == 0:
        # unless retry on this status is permitted explicitly
        if res.status_code not in ctx.retry_statuses:
            return True
    # A status explicitly listed in retry_statuses (or blocking_statuses/exit_statuses being non-empty, per the comment
    # above) previously looped back to retry immediately with no delay at all -- same thundering-herd gap as the
    # generic-exception branch above, just for the status-code path.
    if _facade.delay:
        _facade.sleep(_facade.delay * _facade.random())  # nosec B311 - random jitter on the per-attempt retry backoff, not security-sensitive
    return False


def _handle_response(res: Any, ctx: _RequestCtx, snap: _FetchSnapshot, headers_to_use: Optional[dict]) -> bool:
    """Interpret a received response; return True when the retry loop should stop."""
    if res.status_code not in (http.HTTPStatus.OK, http.HTTPStatus.PARTIAL_CONTENT):
        return _handle_error_status(res, ctx, snap, headers_to_use)
    if any(t in res.text for t in ctx.blocking_errors):
        _facade.handle_blocking(ctx.target, b_random_ua=ctx.b_random_ua, b_use_proxy=ctx.b_use_proxy)
        _facade.was_blocked = True
        return ctx.quit_on_blocking
    _facade.set_proxy_last_use_time(_facade.last_used_dict, _facade.proxies)
    _facade.was_blocked = False
    return True


def _send_request(
    url: str,
    verb: str,
    snap: _FetchSnapshot,
    headers_to_use: Optional[dict],
    params: Optional[dict],
    data: Optional[dict],
    json: Optional[dict],
    b_use_session: bool,
) -> Any:
    """Issue one request through the snapshotted session (or the module ``requests`` facade) with a timeout floor."""
    obj = snap.sess if b_use_session else _facade.requests
    method = getattr(obj, verb)

    # requests treats timeout=None as "wait forever", which would hang the retry loop; the
    # urlopen call sites in this module apply the same 10s floor for exactly this reason.
    return method(
        url,
        headers=headers_to_use,
        params=params,
        data=data,
        json=json,
        proxies=snap.proxies,
        timeout=_facade.timeout if _facade.timeout is not None else 10,
    )


def _count_ip_query() -> None:
    """Increment the per-session IP-query counter under the state lock."""
    with _facade._state_lock:
        _facade.num_ip_queries = _facade.num_ip_queries + 1


def _pause_between_requests() -> None:
    """Sleep the jittered inter-request delay when one is configured."""
    if _facade.delay:
        _facade.sleep(  # nosec B311 - random jitter on the inter-request delay to avoid a fixed request cadence, not security-sensitive
            _facade.delay * _facade.random()
        )


def get_url(
    url: str,
    target: str = "",
    params: Optional[dict] = None,
    data: Optional[dict] = None,
    json: Optional[dict] = None,
    max_retries: int = 10,
    exit_statuses: Sequence = (),
    blocking_statuses: Sequence = (),
    retry_statuses: Sequence = (),
    quit_on_blocking: bool = False,
    blocking_errors: Sequence = (),
    verb: str = "get",
    b_random_ua=True,
    b_use_proxy=True,
    b_use_session=True,
    verbose=False,
    custom_headers: Optional[dict] = None,
    inject_headers: Optional[dict] = None,
    sort_headers: bool = True,
    lowercase_headers: bool = True,
    ratelimited_sleep_interval: int = 30,
    ratelimited_proxy_sleep_interval: int = 0,
    ratelimiting_statuses: Sequence = (429,),
    session_expired_statuses: Sequence = (),
) -> Optional["requests.Response"]:
    """Fetch ``url`` (GET/POST/etc per ``verb``) with retries, proxy rotation and rate-limit/blocking handling.

    Manages the module-level session/proxy/header state (creating a new session when needed,
    merging/sorting/lowercasing headers), retries up to ``max_retries`` times on network errors
    or retryable statuses, rotates to a new proxy on proxy-related errors or when a status in
    ``blocking_statuses``/``ratelimiting_statuses`` is seen, and stops early on statuses in
    ``exit_statuses`` or ``session_expired_statuses``. Sleeps ``delay`` seconds (jittered)
    between attempts. Returns the final ``requests.Response`` (or None if nothing was fetched).

    WARNING -- non-idempotent verbs: on any network exception, or any status in
    ``retry_statuses``, this function re-issues the EXACT SAME request (same ``verb``/``data``/
    ``json``), with no idempotency-key mechanism and no distinction between safe verbs
    (GET/HEAD/PUT/DELETE) and unsafe ones (POST/PATCH). If the server actually processed a POST
    but the response was lost in transit (realistic with proxy rotation, where a mid-response
    connection drop is common), this cannot distinguish that from "request never reached the
    server" and will resubmit it, which can create duplicate side effects (e.g. a duplicate
    order/charge) against a non-idempotent endpoint. Pass a caller-generated idempotency-key
    header via ``inject_headers``/``custom_headers`` (if the target API supports one) when using
    ``verb="post"``/``"patch"`` against an endpoint with real side effects.
    """
    n_retries = 0
    res = None

    # Bound BEFORE the loop: the snapshot is assigned from _take_snapshot() AFTER the get_new_session() call. Any
    # exception out of get_new_session (proxy-gateway outage, bad credentials) lands in the `except` handler, whose
    # "proxy"/"timed out"/"sslerror" substring branch reads the proxy snapshot -- previously an UnboundLocalError on
    # attempt 1, escaping get_url() entirely and bypassing both the retry loop and the Optional[Response] return contract.
    snap = _FetchSnapshot(None, None, None, None, None, None, None, None, None, None)
    ctx = _RequestCtx(
        url=url,
        target=target,
        params=params,
        data=data,
        json=json,
        verbose=verbose,
        b_random_ua=b_random_ua,
        b_use_proxy=b_use_proxy,
        quit_on_blocking=quit_on_blocking,
        blocking_errors=blocking_errors,
        blocking_statuses=blocking_statuses,
        retry_statuses=retry_statuses,
        exit_statuses=exit_statuses,
        ratelimiting_statuses=ratelimiting_statuses,
        session_expired_statuses=session_expired_statuses,
        ratelimited_sleep_interval=ratelimited_sleep_interval,
        ratelimited_proxy_sleep_interval=ratelimited_proxy_sleep_interval,
    )

    while n_retries < max_retries:
        try:
            n_retries = n_retries + 1
            _ensure_session(b_random_ua, b_use_proxy)
            snap = _take_snapshot()
            headers_to_use = _prepare_headers(custom_headers, snap.headers, inject_headers, sort_headers, lowercase_headers)
            if verbose:
                _facade.report_params(url, snap.proxies, params, data, json, headers_to_use, _facade.timeout)

            res = _send_request(url, verb, snap, headers_to_use, params, data, json, b_use_session)
            _count_ip_query()

        except Exception as e:  # noqa: PERF203 -- per-attempt retry loop; the try/except IS the retry mechanism
            _handle_request_exception(e, ctx, snap, n_retries, max_retries)
        else:
            if _handle_response(res, ctx, snap, headers_to_use):
                break
    _pause_between_requests()

    if res is None:
        logger.warning("Could not get url %s", url)
    return res


def get_new_session(b_random_ua: bool = True, b_use_proxy: bool = True) -> None:
    """Create a fresh ``requests.Session``, reset the IP-query counter, optionally set a random user-agent header, and (if ``b_use_proxy``) obtain a new proxy -- storing all of it in module-level state."""
    new_sess = _facade.requests.Session()

    new_headers = _facade.template_headers
    if b_random_ua:
        if new_headers is None:
            new_headers = dict()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            # from fake_useragent import UserAgent
            # ua = UserAgent(verify_ssl=False)
            # headers['user-agent']=ua.random

            new_headers["user-agent"] = (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2762.73 Safari/537.36"
            )

    with _facade._state_lock:
        old_sess = _facade.sess
        _facade.sess = new_sess
        _facade.num_ip_queries = 0
        _facade.headers = new_headers
        local_proxy_user, local_proxy_pass, local_proxy_server = _facade.proxy_user, _facade.proxy_pass, _facade.proxy_server
        local_proxy_min_port, local_proxy_max_port, local_proxy_port, local_proxy_type = (
            _facade.proxy_min_port,
            _facade.proxy_max_port,
            _facade.proxy_port,
            _facade.proxy_type,
        )

    # Regression fix: the previous Session (owning its own urllib3 connection pool) was
    # dropped here with no .close() -- get_new_session() is called routinely, by design, every
    # time the per-session request budget is exhausted, so each rotation silently leaked open
    # sockets/connection-pool state with no bound over a long-running scraper.
    if old_sess is not None:
        try:
            old_sess.close()
        except Exception as e:
            logger.exception(e)

    logger.debug("Created new web session")

    if b_use_proxy:
        if local_proxy_server:
            new_proxies = _facade.get_new_smartproxy(
                local_proxy_user,
                local_proxy_pass,
                local_proxy_server,
                int(local_proxy_min_port) if local_proxy_min_port is not None else 20001,
                int(local_proxy_max_port) if local_proxy_max_port is not None else 37960,
                last_used_dict=_facade.last_used_dict,
                min_idle_interval_minutes=_facade.min_idle_interval_minutes if _facade.min_idle_interval_minutes is not None else 0,
                failed_dict=_facade.failed_dict,
                min_failed_idle_interval_minutes=(
                    _facade.min_failed_idle_interval_minutes if _facade.min_failed_idle_interval_minutes is not None else 60 * 24
                ),
                proxy_port=local_proxy_port,
                proxy_type=local_proxy_type if local_proxy_type is not None else "http",
            )
            with _facade._state_lock:
                _facade.proxies = new_proxies
            logger.info("proxy_server=%s", local_proxy_server)


def handle_blocking(target: str, b_random_ua: bool = True, b_use_proxy: bool = True) -> None:
    """Log that ``target`` got blocked, mark the current proxy as failed, sleep the configured jittered delay, then obtain a fresh session/proxy."""
    if _facade.proxies is not None:
        # Regression fix: proxies["https"].split("@")[1] raised IndexError for an
        # unauthenticated proxy (make_proxies_dict() builds an "@"-free URL when
        # proxy_user/proxy_pass are falsy -- an explicitly supported configuration).
        logger.warning("IP %s blocked. Receiving new proxy/session for %s", _facade._redact_proxy_url(_facade.proxies.get("https")), target)
    else:
        logger.warning("IP blocked.")

    if _facade.delay:
        _facade.sleep(  # nosec B311 - random jitter on the post-block backoff sleep before rotating session/proxy, not security-sensitive
            _facade.delay * _facade.random()
        )
    _facade.set_proxy_last_use_time(_facade.failed_dict, _facade.proxies)
    _facade.get_new_session(b_random_ua=b_random_ua, b_use_proxy=b_use_proxy)

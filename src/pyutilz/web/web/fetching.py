"""The retrying, proxy-rotating, session-managing HTTP fetch loop (``get_url``) plus the session
rotation (``get_new_session``) and block-handling (``handle_blocking``) it drives."""

import http
import warnings
from typing import Any, Optional, Sequence

# Imported for the ``Optional[requests.Response]`` return annotation on get_url() only -- every
# actual call goes through ``_facade.requests`` so that patching ``pyutilz.web.web.requests``
# (which tests do, to swap in a mock Session factory) is honoured here.
import requests

from ._common import logger

# See ipinfo.py for why the parent is reached through `import <parent> as _facade`.
import pyutilz.web.web as _facade


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
) -> Optional[requests.Response]:
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

    # Bound BEFORE the loop: these are assigned from the locked snapshot block below, which sits
    # AFTER the get_new_session() call. Any exception out of get_new_session (proxy-gateway
    # outage, bad credentials) lands in the `except` handler, whose "proxy"/"timed out"/"sslerror"
    # substring branch reads proxy_server_snapshot -- previously an UnboundLocalError on attempt 1,
    # escaping get_url() entirely and bypassing both the retry loop and the Optional[Response]
    # return contract.
    sess_snapshot: Any = None
    proxies_snapshot: Optional[dict] = None
    headers_snapshot: Optional[dict] = None
    proxy_user_snapshot: Optional[str] = None
    proxy_pass_snapshot: Optional[str] = None
    proxy_server_snapshot: Optional[str] = None
    proxy_min_port_snapshot: Optional[int] = None
    proxy_max_port_snapshot: Optional[int] = None
    proxy_port_snapshot: Optional[int] = None
    proxy_type_snapshot: Optional[str] = None

    while n_retries < max_retries:
        try:
            n_retries = n_retries + 1
            # print("Getting url %s,headers=%s,params=%s,proxies=%s,timeout=%s,cookies=%s" % (url,headers,params,proxies,timeout,sess.cookies.get_dict()))

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
            with _facade._state_lock:
                sess_snapshot = _facade.sess
                proxies_snapshot = _facade.proxies
                headers_snapshot = _facade.headers
                # Regression fix: the except/ratelimiting branches below used to read these
                # proxy_* globals directly, unprotected by _state_lock, unlike sess/proxies/
                # headers just above -- a concurrent set_proxy() call (which DOES take the lock
                # for its writes) could interleave, producing a torn mix of old and new proxy
                # fields (e.g. old proxy_min_port paired with new proxy_type) passed to
                # get_new_smartproxy() below.
                proxy_user_snapshot = _facade.proxy_user
                proxy_pass_snapshot = _facade.proxy_pass
                proxy_server_snapshot = _facade.proxy_server
                proxy_min_port_snapshot = _facade.proxy_min_port
                proxy_max_port_snapshot = _facade.proxy_max_port
                proxy_port_snapshot = _facade.proxy_port
                proxy_type_snapshot = _facade.proxy_type
            headers_to_use = custom_headers if custom_headers else headers_snapshot
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

            if verbose:
                _facade.report_params(url, proxies_snapshot, params, data, json, headers_to_use, _facade.timeout)

            if b_use_session:
                obj = sess_snapshot
            else:
                obj = _facade.requests

            method = getattr(obj, verb)

            # requests treats timeout=None as "wait forever", which would hang the retry loop; the
            # urlopen call sites in this module apply the same 10s floor for exactly this reason.
            res = method(
                url,
                headers=headers_to_use,
                params=params,
                data=data,
                json=json,
                proxies=proxies_snapshot,
                timeout=_facade.timeout if _facade.timeout is not None else 10,
            )

            with _facade._state_lock:
                _facade.num_ip_queries = _facade.num_ip_queries + 1

        except Exception as e:  # noqa: PERF203 -- per-attempt retry loop; the try/except IS the retry mechanism
            se = str(e)
            # Regression fix: this was the ONLY log statement for the exception, gated behind
            # verbose=False (the default) -- every network exception across all max_retries
            # attempts was completely silent, leaving only the generic "Could not get url"
            # warning at the very end with no information about *why* (DNS/TLS/proxy-auth/
            # timeout). verbose now controls only the extra report_params() request dump below;
            # the fetch error itself is always logged.
            logger.warning("get_url attempt %d/%d for %s failed: %s", n_retries, max_retries, url, e)
            se = se.lower()
            if "proxy" in se or "timed out" in se or "bad handshake" in se or "connection broken" in se or "sslerror" in se:
                if b_use_proxy:
                    if proxy_server_snapshot:
                        if verbose:
                            logger.warning("Seems to be a bad proxy. Receiving new proxy for %s", target)
                        new_proxies = _facade.get_new_smartproxy(
                            proxy_user_snapshot,
                            proxy_pass_snapshot,
                            proxy_server_snapshot,
                            int(proxy_min_port_snapshot) if proxy_min_port_snapshot is not None else 20001,
                            int(proxy_max_port_snapshot) if proxy_max_port_snapshot is not None else 37960,
                            last_used_dict=_facade.last_used_dict,
                            min_idle_interval_minutes=_facade.min_idle_interval_minutes if _facade.min_idle_interval_minutes is not None else 0,
                            failed_dict=_facade.failed_dict,
                            min_failed_idle_interval_minutes=(
                                _facade.min_failed_idle_interval_minutes if _facade.min_failed_idle_interval_minutes is not None else 60 * 24
                            ),
                            proxy_port=proxy_port_snapshot,
                            proxy_type=proxy_type_snapshot if proxy_type_snapshot is not None else "http",
                        )
                        # Regression fix: this assignment used to write straight to the module
                        # global with no lock, unlike every OTHER write to `proxies` in this file
                        # (see get_new_session/set_proxy) -- two threads hitting a bad proxy
                        # around the same time could race here, and whichever assignment landed
                        # last silently discarded the other thread's proxy rotation.
                        with _facade._state_lock:
                            _facade.proxies = new_proxies
            # Regression fix: this loop's only unconditional sleep() previously sat OUTSIDE the
            # while loop (fires once, after the loop exits) -- any exception not already covered
            # by a sleep above (e.g. a bare ConnectionError whose message matches none of the
            # "proxy"/"timed out"/etc substrings) looped back to the top of `while` immediately,
            # firing up to max_retries requests back-to-back with zero backoff/jitter.
            if _facade.delay:
                _facade.sleep(_facade.delay * _facade.random())  # nosec B311 - random jitter on the per-attempt retry backoff, not security-sensitive
        else:
            if res.status_code not in (http.HTTPStatus.OK, http.HTTPStatus.PARTIAL_CONTENT):
                if res.status_code in blocking_statuses:
                    logger.info("Error %s while getting %s", res.status_code, url)
                    # Regression fix: this used to read the module-global `proxies` directly,
                    # unprotected by _state_lock, in a function that otherwise takes pains to
                    # snapshot everything under the lock (see proxies_snapshot above) -- another
                    # unprotected read alongside the unprotected writes fixed elsewhere in this
                    # function.
                    _facade.report_params(url, proxies_snapshot, params, data, json, headers_to_use, _facade.timeout)
                    _facade.handle_blocking(target, b_random_ua=b_random_ua, b_use_proxy=b_use_proxy)
                    _facade.was_blocked = True
                    if quit_on_blocking:
                        break
                elif res.status_code in session_expired_statuses:
                    logger.warning("Session expired while getting url=%s, code=%s, response=%s", url, res.status_code, res.text)
                    break
                elif res.status_code in exit_statuses:
                    if verbose:
                        logger.info("status_code %s", res.status_code)
                    break
                elif res.status_code in ratelimiting_statuses:
                    if verbose:
                        logger.warning("Ratelimited [%s] while getting url %s: %s", res.status_code, url, res.text)
                    if proxy_server_snapshot:
                        if verbose:
                            logger.warning("Seems to be a bad proxy. Receiving new proxy for %s", target)
                        if ratelimited_proxy_sleep_interval:
                            _facade.sleep(  # nosec B311 - random jitter on a rate-limit backoff sleep, not security-sensitive
                                ratelimited_proxy_sleep_interval * _facade.random()
                            )
                        new_proxies = _facade.get_new_smartproxy(
                            proxy_user_snapshot,
                            proxy_pass_snapshot,
                            proxy_server_snapshot,
                            int(proxy_min_port_snapshot) if proxy_min_port_snapshot is not None else 20001,
                            int(proxy_max_port_snapshot) if proxy_max_port_snapshot is not None else 37960,
                            last_used_dict=_facade.last_used_dict,
                            min_idle_interval_minutes=_facade.min_idle_interval_minutes if _facade.min_idle_interval_minutes is not None else 0,
                            failed_dict=_facade.failed_dict,
                            min_failed_idle_interval_minutes=(
                                _facade.min_failed_idle_interval_minutes if _facade.min_failed_idle_interval_minutes is not None else 60 * 24
                            ),
                            proxy_port=proxy_port_snapshot,
                            proxy_type=proxy_type_snapshot if proxy_type_snapshot is not None else "http",
                        )
                        # See the identical fix/comment on the except-branch's proxy rotation above.
                        with _facade._state_lock:
                            _facade.proxies = new_proxies
                        # Rotating the proxy is not a backoff: without this the loop re-hits a
                        # 429-ing endpoint max_retries times with no pause at all (the exception
                        # and generic-error branches already jitter the same way).
                        retry_after = _facade._parse_retry_after(res)
                        if retry_after is not None:
                            _facade.sleep(retry_after)
                        elif _facade.delay:
                            _facade.sleep(  # nosec B311 - random jitter on the rate-limit retry backoff, not security-sensitive
                                _facade.delay * _facade.random()
                            )
                    else:
                        retry_after = _facade._parse_retry_after(res)
                        _facade.sleep(retry_after if retry_after is not None else ratelimited_sleep_interval)
                else:
                    logger.warning("Error %s while getting url %s: %s", res.status_code, url, res.text)

                    # if blocking or exit satuses are specified, we keep retrying on any error (after small pause)

                    if len(blocking_statuses) == 0 and len(exit_statuses) == 0:
                        # unless retry on this status is permitted explicitly
                        if res.status_code not in retry_statuses:
                            break
                    # A status explicitly listed in retry_statuses (or blocking_statuses/
                    # exit_statuses being non-empty, per the comment above) previously looped
                    # back to retry immediately with no delay at all -- same thundering-herd gap
                    # as the generic-exception branch above, just for the status-code path.
                    if _facade.delay:
                        _facade.sleep(_facade.delay * _facade.random())  # nosec B311 - random jitter on the per-attempt retry backoff, not security-sensitive
            else:
                err_found = False
                for t in blocking_errors:
                    if t in res.text:
                        err_found = True
                        break
                if err_found:
                    _facade.handle_blocking(target, b_random_ua=b_random_ua, b_use_proxy=b_use_proxy)
                    _facade.was_blocked = True
                    if quit_on_blocking:
                        break
                else:
                    _facade.set_proxy_last_use_time(_facade.last_used_dict, _facade.proxies)
                    _facade.was_blocked = False
                    break
    if _facade.delay:
        _facade.sleep(  # nosec B311 - random jitter on the inter-request delay to avoid a fixed request cadence, not security-sensitive
            _facade.delay * _facade.random()
        )

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

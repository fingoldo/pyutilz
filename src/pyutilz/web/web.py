"""Web scraping helpers: proxy rotation, session/header management, retrying URL fetches and parallel/streaming downloads."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import json
import logging
import os

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Any, Callable, List, Optional, Sequence

import requests
import urllib.request
import urllib.parse
from random import random, shuffle
from datetime import datetime
from joblib import hash as joblib_hash
from time import sleep
import warnings
import http
import ssl
import threading
from dataclasses import dataclass

# Guards read-then-mutate access to the module-level scraping state below (sess, proxies,
# headers, num_ip_queries, cur_max_ip_queries, was_blocked, proxy_* fields) when this module
# is driven from multiple threads. Only the shared-state read/mutate sections are locked --
# actual network I/O (the HTTP request itself) happens outside the lock.
_state_lock = threading.Lock()

delay: Optional[int] = 1
max_ip_queries: Optional[int] = 0
last_used_dict = None
min_idle_interval_minutes = None
failed_dict = None
min_failed_idle_interval_minutes = None

IP_PROVIDERS = ["https://api.ipify.org/", "https://ident.me/", "http://icanhazip.com/"]
cur_max_ip_queries = -1
proxy_server = None
was_blocked = False

# Bound for real by init_vars() / set_proxy_credentials(); declared here so module-level
# references resolve before the first call (and so static analysis can see these names exist).
sess: Optional[Any] = None
num_ip_queries: int = 0
# template_headers/proxy_min_port/proxy_max_port/proxy_port are genuinely inconsistently
# typed across call sites in this file today (some pass str, some pass int, with defensive
# int(...) casts at the point of use) -- Any reflects that honestly rather than picking a
# specific type that would be wrong for half the callers.
template_headers: Optional[Any] = None
headers: Optional[dict] = None
proxies: Optional[dict] = None
timeout: Optional[int] = 10
proxy_user: Optional[str] = None
proxy_pass: Optional[str] = None
proxy_min_port: Optional[Any] = None
proxy_max_port: Optional[Any] = None
proxy_port: Optional[Any] = None
proxy_type: Optional[str] = None

_ALLOWED_URL_SCHEMES = ("http", "https")


def _ensure_http_scheme(url: str) -> str:
    """Raise ValueError unless ``url`` uses http(s) -- guards ``urllib.request.urlopen``
    against a caller-supplied ``file:///etc/passwd``-style scheme (local file disclosure)
    or other unexpected custom scheme."""
    scheme = urllib.parse.urlsplit(url).scheme.lower()
    if scheme not in _ALLOWED_URL_SCHEMES:
        raise ValueError(f"Refusing to urlopen {url!r}: scheme {scheme!r} not in {_ALLOWED_URL_SCHEMES}")
    return url


_SENSITIVE_HEADER_NAMES = frozenset({"authorization", "cookie", "set-cookie", "proxy-authorization", "x-api-key"})


def _redact_proxy_url(proxy_url: Optional[str]) -> Optional[str]:
    """Strip embedded ``user:pass@`` credentials from a proxy URL for safe logging, keeping the
    host:port for diagnostic value. Returns the input unchanged if it isn't URL-shaped (e.g. no
    ``@`` at all -- an unauthenticated proxy, per make_proxies_dict()'s own credential-less branch)."""
    if not proxy_url or "@" not in proxy_url:
        return proxy_url
    scheme_sep = proxy_url.find("://")
    prefix = proxy_url[: scheme_sep + 3] if scheme_sep != -1 else ""
    rest = proxy_url[len(prefix) :]
    _, _, host_part = rest.rpartition("@")
    return f"{prefix}***@{host_part}"


def _redact_proxies_dict(proxies_dict: Optional[dict]) -> Optional[dict]:
    """Apply :func:`_redact_proxy_url` to every value of a ``requests``-style proxies dict."""
    if not proxies_dict:
        return proxies_dict
    return {scheme: _redact_proxy_url(url) for scheme, url in proxies_dict.items()}


def _redact_headers(headers_dict: Optional[dict]) -> Optional[dict]:
    """Mask ``Authorization``/``Cookie``/etc. header values for safe logging (case-insensitive)."""
    if not headers_dict:
        return headers_dict
    return {k: ("***" if k.lower() in _SENSITIVE_HEADER_NAMES else v) for k, v in headers_dict.items()}


def init_vars():
    """Reset the module-level session state (session, IP-query counter, headers, proxies, timeout) to defaults."""
    global sess, num_ip_queries, template_headers, headers, proxies, timeout
    # Regression fix: the old Session (a requests.Session owns a urllib3 connection pool --
    # potentially several open keep-alive sockets) was previously just dropped here with no
    # .close(), leaking its connection pool every time init_vars() runs (e.g. every
    # web.connect() call, including FileMaker's two-calls-per-authentication pattern).
    #
    # Regression fix (meta-test-driven finding, proactive lock-discipline audit): these globals
    # used to be reset here with no lock, unlike get_url()'s snapshot read of the same sess/
    # proxies/headers globals (which does take _state_lock) -- a concurrent get_url() call could
    # observe a torn mix of old and new state. The lock only needs to cover the state mutation,
    # not old_sess.close() (an I/O call with no reason to hold the lock during it).
    with _state_lock:
        old_sess = sess
        sess = None
        num_ip_queries = 0
        template_headers = None
        headers = {}
        proxies = None
        timeout = 10
    if old_sess is not None:
        try:
            old_sess.close()
        except Exception as e:
            logger.exception(e)
    logger.debug("Session cleared")


init_vars()


def get_external_ip(
    proxy_user: Optional[str] = None, proxy_pass: Optional[str] = None, proxy_server: Optional[str] = None, proxy_port: Optional[int] = None
) -> Optional[str]:
    """Return this machine's external IP by querying ``IP_PROVIDERS`` (shuffled) until one responds with a plausible address, or None if all fail."""
    global IP_PROVIDERS
    shuffle(IP_PROVIDERS)

    for source in IP_PROVIDERS:
        try:
            # timeout= is required: urlopen's default is socket._GLOBAL_DEFAULT_TIMEOUT, which
            # blocks forever unless something else in the process called
            # socket.setdefaulttimeout() (nothing in this package does) -- a single stalled
            # provider would otherwise hang this whole function indefinitely instead of moving
            # on to the next shuffled IP_PROVIDERS entry.
            resp = urllib.request.urlopen(_ensure_http_scheme(source), timeout=timeout if timeout is not None else 10)  # nosec B310 - scheme validated above -- `is not None`, not `or`: a caller-set timeout=0 must not be silently rewritten to 10
        except ssl.SSLCertVerificationError:  # noqa: PERF203 -- per-iteration fault isolation is intentional (skip this provider, try the next)
            pass
        except Exception as e:
            logger.exception(e)
        else:
            if resp.status == http.HTTPStatus.OK:
                res = resp.read().decode("utf8").strip()
                if "." in res or ":" in res:
                    return res  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime
                else:
                    logger.warning("Weird IP address received from provider %s: %s", source, res)
    return None


# Geolocation providers, tried in order until one answers with a country. All four are keyless and free
# for low volume; each states a DIFFERENT JSON shape, so the field names to read are carried alongside the
# URL rather than guessed. `{ip}` is substituted with the address to look up, or removed for a self-lookup
# (every one of these resolves the CALLER's own address when the path is empty).
#
# Deliberately NOT shuffled the way IP_PROVIDERS is: those three are interchangeable echo services, while
# these differ in accuracy and rate limit, so the order is a preference and shuffling would throw it away.
COUNTRY_PROVIDERS: List[dict] = [
    {"url": "https://ipapi.co/{ip}/json/", "code": "country_code", "name": "country_name", "continent": "continent_code"},
    # `fields=` is required, not decoration: ip-api's DEFAULT response omits `continent` entirely, so the
    # unqualified URL silently returned a blank continent for every lookup this provider answered.
    {
        "url": "http://ip-api.com/json/{ip}?fields=status,country,countryCode,continent,query",
        "code": "countryCode",
        "name": "country",
        "continent": "continent",
    },
    {"url": "https://ipwho.is/{ip}", "code": "country_code", "name": "country", "continent": "continent"},
    {"url": "https://freeipapi.com/api/json/{ip}", "code": "countryCode", "name": "countryName", "continent": "continent"},
]


@dataclass
class IpGeolocation:
    """What a geolocation lookup actually established, and from where.

    `provider` is not decoration: these services disagree, and a caller acting on a country (pricing,
    routing, a clinical prior) needs to be able to say which one said so. Country-level only - the finer
    fields these APIs return (city, coordinates, ISP) are deliberately not surfaced, because they are the
    fields with the worst accuracy and the highest privacy cost, and no caller in this codebase needs them.
    """

    country_code: str  # ISO 3166-1 alpha-2, uppercased
    country_name: str  # the provider's own English name; spellings differ BETWEEN providers
    continent: str = ""  # a code ("EU") from some providers and a name ("Europe") from others - see get_country_by_ip
    ip: str = ""
    provider: str = ""


def get_country_by_ip(ip: Optional[str] = None, providers: Optional[Sequence[dict]] = None) -> Optional[IpGeolocation]:
    """Resolve an IP address to a country, trying ``COUNTRY_PROVIDERS`` in order until one answers.

    ``ip=None`` looks up the CALLER's own address (every provider here does that for an empty path), so no
    separate `get_external_ip()` round trip is needed first.

    Returns ``None`` when every provider failed or none of them stated a country - never a guess and never
    a partial record. A geolocation miss is ordinary: private/loopback addresses, VPNs, corporate egress
    and rate limits all produce one, so callers must treat ``None`` as "unknown", not as an error.

    ACCURACY, stated plainly because callers keep assuming otherwise: IP geolocation is reliable at country
    level and unreliable below it, and it reports where the ADDRESS is, not where the person is - a VPN,
    a mobile carrier's national egress or a cloud host will all answer confidently and wrongly. Use it to
    PRE-FILL a field a human can correct, never as an established fact.

    ``continent`` is passed through unnormalised on purpose: ip-api and ipwho.is state a name ("Europe"),
    ipapi.co states a code ("EU"). Normalising here would mean this function owning a continent table;
    a caller that needs one shape should map the value it got, knowing which provider answered.
    """
    for provider in providers if providers is not None else COUNTRY_PROVIDERS:
        url = provider["url"].format(ip=ip or "")
        try:
            data = get_ipinfo(use_urllib=True, url=url)
        except Exception as e:
            # debug, not exception: a provider failing is the EXPECTED case this chain exists for (all four
            # rate-limit routinely), and logging a traceback per provider turns one successful lookup into
            # three stack traces in the caller's log.
            logger.debug("country lookup via %s failed: %s", url, e)
            continue
        if not isinstance(data, dict):
            # get_ipinfo swallows its own exceptions and returns None, so this -- not the
            # except branch above -- is where a failing provider actually shows up.
            logger.debug("country lookup via %s returned no usable payload", url)
            continue
        code = data.get(provider["code"])
        name = data.get(provider["name"])
        # Both, not either: a provider that answered with an error body ({"success": false, ...}) supplies
        # neither, and one that supplies only a code leaves the caller with nothing to show a human.
        if not isinstance(code, str) or not isinstance(name, str) or not code.strip() or not name.strip():
            continue
        continent = data.get(provider["continent"])
        return IpGeolocation(
            country_code=code.strip().upper(),
            country_name=name.strip(),
            continent=continent.strip() if isinstance(continent, str) else "",
            ip=str(data.get("ip") or data.get("query") or ip or ""),
            provider=url.split("/")[2],
        )
    return None


def get_ipinfo(use_urllib: bool = False, url: str = "https://api.ipify.org?format=json") -> Optional[Any]:
    """Fetch JSON IP info from ``url``, either directly via ``urllib`` (``use_urllib=True``) or via the module's ``get_url`` (session/proxy-aware) fetcher."""
    json_loads: Any
    try:
        import orjson

        json_loads = orjson.loads
    except ImportError:
        json_loads = json.loads

    if use_urllib:
        try:
            resp = urllib.request.urlopen(_ensure_http_scheme(url), timeout=timeout if timeout is not None else 10)  # nosec B310 - scheme validated above; timeout= avoids blocking forever, see get_external_ip's identical fix -- `is not None`, not `or`: a caller-set timeout=0 must not be silently rewritten to 10
        except Exception as e:
            logger.exception(e)
            return None
        else:
            if resp.status == http.HTTPStatus.OK:
                return json_loads(resp.read().decode("utf8").strip())
            # Regression fix (2026-07-21 audit round 2, LOW): a non-200 response used to return
            # `{}` here while an exception (above) returned None -- two different failure values
            # for the same "the request failed" outcome, unlike get_external_ip's single None
            # convention and this function's own use_urllib=False branch below.
            return None
    else:
        res = get_url(url, target="ipinfo", inject_headers={"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        try:
            res = res.json() if res is not None else None
        except Exception as e:
            logger.exception(e)
            res = None
        return res


def _error_log_throttle(n_errored: int, max_logged: int) -> bool:
    """True while per-URL error logging in ``download_in_parallel`` is still under its cap --
    a batch with a large fraction of failing URLs would otherwise log one ERROR line per failed
    URL with no limit, compounding into thousands of lines for a large batch."""
    return n_errored <= max_logged


def download_in_parallel(
    urls_to_process: Sequence,
    func: Callable,
    headers: Optional[dict] = None,
    nparallel_downloads: int = 3,
    report_each: int = 10,
    request_timeout: Optional[float] = 30,
) -> Optional[List]:
    """Fetch ``urls_to_process`` concurrently via ``grequests`` and call ``func(response, url)`` on each successful (HTTP 200) response.

    Returns the list of URLs that errored (non-200 status, exception in ``func``, or None
    response), or None if ``urls_to_process`` is None/empty.
    """
    import grequests

    if urls_to_process is None:
        return None
    if len(urls_to_process) == 0:
        return None

    # The aggregate "Processed N urls, n_errored=M" line (every report_each iterations, already
    # unconditional) and the final "Finished!" summary still report the true total either way.
    _MAX_LOGGED_ERRORS = 20
    n_processed = 0
    errored_urls = []
    # request_timeout=: grequests.get() previously had no timeout at all -- a single stalled URL
    # (server accepts the connection but never responds) blocked grequests.map() from ever
    # returning, hanging the whole batch regardless of how many OTHER urls had already succeeded.
    rs = (grequests.get(sub_url, verify=True, allow_redirects=True, headers=headers, timeout=request_timeout) for sub_url in urls_to_process)
    logger.info("Started crawling! nlinks=%d", len(urls_to_process))
    # pbar=tqdmu(urls_to_process)
    for resp, sub_url in zip(grequests.map(rs, size=nparallel_downloads), urls_to_process):
        n_processed = n_processed + 1
        # None-check MUST come before touching any attribute of resp: grequests.map() puts None
        # into the results for a request that raised (no exception_handler passed here), exactly
        # the case this function's own docstring says it handles ("None response"). Previously
        # `resp.history` was dereferenced first, raising an uncaught AttributeError that aborted
        # the ENTIRE batch (including URLs not yet processed) instead of recording just this one
        # URL as errored.
        if resp is None:
            errored_urls.append(sub_url)
            if _error_log_throttle(len(errored_urls), _MAX_LOGGED_ERRORS):
                logger.error("Response is None for url %s", sub_url)
        else:
            final_status_code = resp.history[-1].status_code if len(resp.history) > 0 else resp.status_code
            if resp.status_code == http.HTTPStatus.OK:
                try:
                    func(resp, sub_url)
                except Exception as e:
                    errored_urls.append(sub_url)
                    if _error_log_throttle(len(errored_urls), _MAX_LOGGED_ERRORS):
                        logger.error("Error processing url %s: %s", sub_url, e)
            else:
                errored_urls.append(sub_url)
                if _error_log_throttle(len(errored_urls), _MAX_LOGGED_ERRORS):
                    logger.error("Error fetching url %s: status_code=%s", sub_url, final_status_code)
        if (n_processed % report_each) == 0:
            # pbar.update(report_each)
            logger.info("Processed %d urls,n_errored=%d", n_processed, len(errored_urls))
    logger.info("Finished! n_processed=%d,n_errored=%d", n_processed, len(errored_urls))
    # pbar.close()
    return errored_urls


# ***************************************************************************************************************************
# Proxy downloads
# ***************************************************************************************************************************


def connect(
    m_proxy_user: Optional[str] = None,
    m_proxy_pass: Optional[str] = None,
    m_proxy_server: Optional[str] = None,
    m_proxy_min_port: Optional[str] = None,
    m_proxy_max_port: Optional[str] = None,
    m_template_headers: Optional[Any] = None,
    m_timeout: Optional[int] = 10,
    m_proxy_port: Optional[str] = None,
    m_proxy_type: str = "http",
    **kwargs,
) -> None:
    """Reset session state and set the module-level proxy/header/timeout config from the ``m_*`` arguments (used to configure proxy credentials before fetching)."""
    global proxy_user, proxy_pass, proxy_server, proxy_min_port, proxy_max_port, template_headers, timeout, proxy_port, proxy_type

    init_vars()
    # Regression fix (meta-test-driven finding, proactive lock-discipline audit): this used to
    # write proxy_user/proxy_pass/proxy_server/etc. with no lock, unlike set_proxy()'s write of
    # the identical field group (which does take _state_lock) and get_url()'s locked read of it.
    with _state_lock:
        proxy_user, proxy_pass, proxy_server, proxy_min_port, proxy_max_port, template_headers, timeout, proxy_port, proxy_type = (
            m_proxy_user,
            m_proxy_pass,
            m_proxy_server,
            m_proxy_min_port,
            m_proxy_max_port,
            m_template_headers,
            m_timeout,
            m_proxy_port,
            m_proxy_type,
        )


def set_proxy(
    m_proxy_user: Optional[str] = None,
    m_proxy_pass: Optional[str] = None,
    m_proxy_server: Optional[str] = None,
    m_proxy_min_port: Optional[int] = None,
    m_proxy_max_port: Optional[int] = None,
    m_proxy_port: Optional[int] = None,
    m_proxy_type: str = "http",
) -> None:
    """Set module-level proxy credentials/port range and immediately obtain a fresh proxy from ``get_new_smartproxy``, storing it in the module-level ``proxies`` dict.

    Raises ValueError if ``m_proxy_user``, ``m_proxy_pass`` or ``m_proxy_server`` is None.
    """
    global proxies
    global proxy_user, proxy_pass, proxy_server, proxy_min_port, proxy_max_port, proxy_port, proxy_type
    with _state_lock:
        proxy_user, proxy_pass, proxy_server, proxy_min_port, proxy_max_port, proxy_port, proxy_type = (
            m_proxy_user,
            m_proxy_pass,
            m_proxy_server,
            m_proxy_min_port,
            m_proxy_max_port,
            m_proxy_port,
            m_proxy_type,
        )
        if proxy_user is None or proxy_pass is None or proxy_server is None:
            raise ValueError("set_proxy: proxy_user, proxy_pass and proxy_server are required (got None)")
        local_proxy_user, local_proxy_pass, local_proxy_server = proxy_user, proxy_pass, proxy_server
        local_proxy_min_port, local_proxy_max_port, local_proxy_port, local_proxy_type = proxy_min_port, proxy_max_port, proxy_port, proxy_type

    new_proxies = get_new_smartproxy(
        local_proxy_user,
        local_proxy_pass,
        local_proxy_server,
        int(local_proxy_min_port) if local_proxy_min_port is not None else 20001,
        int(local_proxy_max_port) if local_proxy_max_port is not None else 37960,
        last_used_dict=last_used_dict,
        min_idle_interval_minutes=min_idle_interval_minutes if min_idle_interval_minutes is not None else 0,
        failed_dict=failed_dict,
        min_failed_idle_interval_minutes=min_failed_idle_interval_minutes if min_failed_idle_interval_minutes is not None else 60 * 24,
        proxy_port=local_proxy_port,
        proxy_type=local_proxy_type if local_proxy_type is not None else "http",
    )
    with _state_lock:
        proxies = new_proxies


def set_params(
    m_delay: Optional[int] = 0,
    m_max_ip_queries: Optional[int] = 0,
    m_last_used_dict: Optional[dict] = None,
    m_min_idle_interval_minutes: Optional[int] = None,
    m_failed_dict: Optional[dict] = None,
    m_min_failed_idle_interval_minutes: Optional[int] = None,
) -> None:
    """Set module-level rate-limiting/throttling parameters: inter-request delay, max IP queries per session, and the proxy-touched/proxy-failed tracking dicts and idle intervals."""
    global delay, max_ip_queries, last_used_dict, min_idle_interval_minutes, failed_dict, min_failed_idle_interval_minutes

    delay = m_delay
    max_ip_queries = m_max_ip_queries
    last_used_dict = m_last_used_dict
    min_idle_interval_minutes = m_min_idle_interval_minutes
    failed_dict = m_failed_dict
    min_failed_idle_interval_minutes = m_min_failed_idle_interval_minutes


def set_proxy_last_use_time(last_used_dict: Optional[dict], proxies: Optional[dict]) -> None:
    """Record the current UTC time as the last-use timestamp for ``proxies`` (keyed by its joblib hash) in ``last_used_dict``, if it's a dict."""
    if isinstance(last_used_dict, dict):
        last_used_dict[joblib_hash(proxies)] = datetime.utcnow()  # noqa: DTZ003 -- naive-UTC is this module's public dict-timestamp convention (see get_new_smartproxy/tests, which store/compare naive datetime.utcnow() values); switching to aware would break subtraction against caller-supplied entries


def make_proxies_dict(proxy_user: Optional[str], proxy_pass: Optional[str], proxy_server: str, proxy_port: int, proxy_type: str = "https") -> dict:
    """Build a ``requests``-style ``{"http": ..., "https": ...}`` proxy URL dict from the given credentials/server/port/scheme."""
    if proxy_user and proxy_pass:
        proxy_url = "%s:%s@%s:%s" % (proxy_user, proxy_pass, proxy_server, proxy_port)
    else:
        proxy_url = "%s:%s" % (proxy_server, proxy_port)
    # return {"http": f"http://{proxy_url}", "https": f"https://{proxy_url}"}
    return {"http": f"{proxy_type}://{proxy_url}", "https": f"{proxy_type}://{proxy_url}"}


def get_new_smartproxy(
    proxy_user: Optional[str],
    proxy_pass: Optional[str],
    proxy_server: str,
    proxy_min_port: int = 20001,
    proxy_max_port: int = 37960,
    job_desc: str = "",
    last_used_dict: Optional[dict] = None,
    min_idle_interval_minutes: float = 0,
    failed_dict: Optional[dict] = None,
    min_failed_idle_interval_minutes: float = 60 * 24,
    warn_after_n_failures: int = 5,
    delay: int = 5,
    proxy_port: Optional[int] = None,
    proxy_type: str = "http",
    verbose=False,
    max_wait_seconds: Optional[float] = None,
) -> dict:
    """Pick a proxy (random port within ``[proxy_min_port, proxy_max_port]`` unless ``proxy_port`` is fixed) that hasn't been touched recently per ``last_used_dict``/``failed_dict``.

    Keeps re-rolling a random port and sleeping ``delay`` seconds every ``warn_after_n_failures``
    attempts until an eligible (not recently used/failed within ``min_idle_interval_minutes``)
    proxy dict is found, then returns it. Blocks indefinitely if ``max_wait_seconds`` is None
    (the default, preserving historical behaviour) and no proxy in the range is ever eligible --
    a real risk for a fixed/single-proxy pool (e.g. ``proxy_port`` fixed, or
    ``proxy_min_port == proxy_max_port``, the exact "server-side rotation" pattern
    :func:`is_rotating_proxy` recognizes) after one transient failure marks the sole candidate
    "touched" for up to ``min_failed_idle_interval_minutes`` (default 24h). Every
    ``warn_after_n_failures`` attempts is now logged unconditionally (not gated on ``verbose``)
    specifically so a stuck wait is diagnosable from logs alone; pass ``max_wait_seconds`` to
    raise :class:`TimeoutError` instead of blocking forever once that budget is exhausted.
    """
    if failed_dict is None:
        failed_dict = {}
    if last_used_dict is None:
        last_used_dict = {}
    n = 0
    now_time = datetime.utcnow()  # noqa: DTZ003 -- must stay naive to subtract against caller-supplied last_used_dict/failed_dict entries, which follow this module's naive-UTC timestamp convention (see set_proxy_last_use_time)
    wait_started_at: Optional[datetime] = None
    # Captured once: rebinding the `proxy_port` PARAMETER inside the loop froze the first random
    # draw forever, so the "keeps re-rolling a random port" contract above never held.
    fixed_port = proxy_port
    while True:
        # ----------------------------------------------------------------------------------------------------------------------------
        # Get random port
        # ----------------------------------------------------------------------------------------------------------------------------
        if fixed_port is None:
            proxy_port = int(proxy_min_port) + int(random() * (int(proxy_max_port) - int(proxy_min_port)))  # nosec B311 - non-cryptographic random port pick within an allowed proxy port range, for load spreading, not security-sensitive
        else:
            proxy_port = fixed_port

        proxies = make_proxies_dict(proxy_user, proxy_pass, proxy_server, proxy_port, proxy_type)

        proxy_key = joblib_hash(proxies)
        # ----------------------------------------------------------------------------------------------------------------------------
        # Check if it's allowed for immediate use by the policies
        # ----------------------------------------------------------------------------------------------------------------------------
        b_time_to_check_now = True
        for dict_to_check, min_interval in ((failed_dict, min_failed_idle_interval_minutes), (last_used_dict, min_idle_interval_minutes)):
            if dict_to_check is not None:
                if proxy_key in dict_to_check:
                    # A failed proxy has its own (much longer) cooldown; comparing it against
                    # min_idle_interval_minutes (default 0) handed a just-blocked exit IP straight back.
                    if (now_time - dict_to_check[proxy_key]).total_seconds() / 60 < min_interval:
                        if verbose:
                            logger.info("Skipping proxy %s:%s, touched recently", proxy_server, proxy_port)
                        b_time_to_check_now = False
                        break

        if b_time_to_check_now:
            if verbose:
                logger.info("Got new proxy: %s:%s", proxy_server, proxy_port)
            return proxies
        else:
            n = n + 1
            if n > warn_after_n_failures:
                # Unconditional, not gated on verbose=True: every real call site in this module
                # invokes get_new_smartproxy() (directly or via set_proxy()) with the default
                # verbose=False, so this was previously the ONLY diagnostic signal available for
                # a stuck wait -- and it never fired, making an indefinite hang undiagnosable
                # from logs alone (the only other symptom being a silent sleep(delay) every ~5
                # attempts with no output at all).
                logger.info(
                    "Could not get an untouched proxy%s after %d attempts, sleeping %s sec.",
                    "" if job_desc == "" else " for " + job_desc,
                    n,
                    delay,
                )
                if max_wait_seconds is not None:
                    if wait_started_at is None:
                        wait_started_at = datetime.utcnow()  # noqa: DTZ003 -- naive-UTC, matches now_time's convention above
                    elif (datetime.utcnow() - wait_started_at).total_seconds() > max_wait_seconds:  # noqa: DTZ003
                        raise TimeoutError(
                            f"get_new_smartproxy: no eligible proxy found within {max_wait_seconds}s "
                            f"(pool [{proxy_min_port}, {proxy_max_port}] may be exhausted or too small "
                            f"for min_idle_interval_minutes={min_idle_interval_minutes})"
                        )
                sleep(delay)
                n = 0


def report_params(url, proxies, params, data, json, headers_to_use, timeout):
    """Log a request's url/proxies/params/data/json/headers/timeout at INFO level, for debugging a fetch.

    Proxy credentials and sensitive headers (Authorization/Cookie/etc.) are redacted before
    logging -- this function previously logged them in cleartext unconditionally on a common
    code path (get_url()'s blocking-status branch, not gated on verbose=True), and the plaintext
    proxy password embedded in the proxies URL string would land directly in the log stream.
    """
    logger.info(
        "url=%s, proxies=%s, params=%s, data=%s, json=%s, headers=%s, timeout=%s",
        url,
        str(_redact_proxies_dict(proxies)),
        params,
        data,
        json,
        _redact_headers(headers_to_use),
        timeout,
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
    global proxies
    global was_blocked
    global num_ip_queries, cur_max_ip_queries

    n_retries = 0
    res = None

    while n_retries < max_retries:
        try:
            n_retries = n_retries + 1
            # print("Getting url %s,headers=%s,params=%s,proxies=%s,timeout=%s,cookies=%s" % (url,headers,params,proxies,timeout,sess.cookies.get_dict()))

            # We are trying to fetch some url. Do we need to create new proxy session?
            with _state_lock:
                need_new_session = sess is None or ((max_ip_queries or 0) > 0 and num_ip_queries > cur_max_ip_queries)
            if need_new_session:
                # If there is no session yet or we have downloaded too many items within current session already
                get_new_session(b_random_ua=b_random_ua, b_use_proxy=b_use_proxy)
                if (max_ip_queries or 0) > 0:
                    with _state_lock:
                        cur_max_ip_queries = int((max_ip_queries or 0) * (0.6 + 0.4 * random()))  # nosec B311 - randomizes the per-session IP-query budget to avoid a fixed rotation pattern; non-cryptographic jitter, not a security control
                    logger.info("cur_max_ip_queries set to %d", cur_max_ip_queries)
            with _state_lock:
                sess_snapshot = sess
                proxies_snapshot = proxies
                headers_snapshot = headers
                # Regression fix: the except/ratelimiting branches below used to read these
                # proxy_* globals directly, unprotected by _state_lock, unlike sess/proxies/
                # headers just above -- a concurrent set_proxy() call (which DOES take the lock
                # for its writes) could interleave, producing a torn mix of old and new proxy
                # fields (e.g. old proxy_min_port paired with new proxy_type) passed to
                # get_new_smartproxy() below.
                proxy_user_snapshot = proxy_user
                proxy_pass_snapshot = proxy_pass
                proxy_server_snapshot = proxy_server
                proxy_min_port_snapshot = proxy_min_port
                proxy_max_port_snapshot = proxy_max_port
                proxy_port_snapshot = proxy_port
                proxy_type_snapshot = proxy_type
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
                report_params(url, proxies_snapshot, params, data, json, headers_to_use, timeout)

            if b_use_session:
                obj = sess_snapshot
            else:
                obj = requests

            method = getattr(obj, verb)

            # requests treats timeout=None as "wait forever", which would hang the retry loop; the
            # urlopen call sites in this module apply the same 10s floor for exactly this reason.
            res = method(
                url, headers=headers_to_use, params=params, data=data, json=json, proxies=proxies_snapshot, timeout=timeout if timeout is not None else 10
            )

            with _state_lock:
                num_ip_queries = num_ip_queries + 1

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
                        new_proxies = get_new_smartproxy(
                            proxy_user_snapshot,
                            proxy_pass_snapshot,
                            proxy_server_snapshot,
                            int(proxy_min_port_snapshot) if proxy_min_port_snapshot is not None else 20001,
                            int(proxy_max_port_snapshot) if proxy_max_port_snapshot is not None else 37960,
                            last_used_dict=last_used_dict,
                            min_idle_interval_minutes=min_idle_interval_minutes if min_idle_interval_minutes is not None else 0,
                            failed_dict=failed_dict,
                            min_failed_idle_interval_minutes=min_failed_idle_interval_minutes if min_failed_idle_interval_minutes is not None else 60 * 24,
                            proxy_port=proxy_port_snapshot,
                            proxy_type=proxy_type_snapshot if proxy_type_snapshot is not None else "http",
                        )
                        # Regression fix: this assignment used to write straight to the module
                        # global with no lock, unlike every OTHER write to `proxies` in this file
                        # (see get_new_session/set_proxy) -- two threads hitting a bad proxy
                        # around the same time could race here, and whichever assignment landed
                        # last silently discarded the other thread's proxy rotation.
                        with _state_lock:
                            proxies = new_proxies
            # Regression fix: this loop's only unconditional sleep() previously sat OUTSIDE the
            # while loop (fires once, after the loop exits) -- any exception not already covered
            # by a sleep above (e.g. a bare ConnectionError whose message matches none of the
            # "proxy"/"timed out"/etc substrings) looped back to the top of `while` immediately,
            # firing up to max_retries requests back-to-back with zero backoff/jitter.
            if delay:
                sleep(delay * random())  # nosec B311 - random jitter on the per-attempt retry backoff, not security-sensitive
        else:
            if res.status_code not in (http.HTTPStatus.OK, http.HTTPStatus.PARTIAL_CONTENT):
                if res.status_code in blocking_statuses:
                    logger.info("Error %s while getting %s", res.status_code, url)
                    # Regression fix: this used to read the module-global `proxies` directly,
                    # unprotected by _state_lock, in a function that otherwise takes pains to
                    # snapshot everything under the lock (see proxies_snapshot above) -- another
                    # unprotected read alongside the unprotected writes fixed elsewhere in this
                    # function.
                    report_params(url, proxies_snapshot, params, data, json, headers_to_use, timeout)
                    handle_blocking(target, b_random_ua=b_random_ua, b_use_proxy=b_use_proxy)
                    was_blocked = True
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
                            sleep(ratelimited_proxy_sleep_interval * random())  # nosec B311 - random jitter on a rate-limit backoff sleep, not security-sensitive
                        new_proxies = get_new_smartproxy(
                            proxy_user_snapshot,
                            proxy_pass_snapshot,
                            proxy_server_snapshot,
                            int(proxy_min_port_snapshot) if proxy_min_port_snapshot is not None else 20001,
                            int(proxy_max_port_snapshot) if proxy_max_port_snapshot is not None else 37960,
                            last_used_dict=last_used_dict,
                            min_idle_interval_minutes=min_idle_interval_minutes if min_idle_interval_minutes is not None else 0,
                            failed_dict=failed_dict,
                            min_failed_idle_interval_minutes=min_failed_idle_interval_minutes if min_failed_idle_interval_minutes is not None else 60 * 24,
                            proxy_port=proxy_port_snapshot,
                            proxy_type=proxy_type_snapshot if proxy_type_snapshot is not None else "http",
                        )
                        # See the identical fix/comment on the except-branch's proxy rotation above.
                        with _state_lock:
                            proxies = new_proxies
                        # Rotating the proxy is not a backoff: without this the loop re-hits a
                        # 429-ing endpoint max_retries times with no pause at all (the exception
                        # and generic-error branches already jitter the same way).
                        retry_after = _parse_retry_after(res)
                        if retry_after is not None:
                            sleep(retry_after)
                        elif delay:
                            sleep(delay * random())  # nosec B311 - random jitter on the rate-limit retry backoff, not security-sensitive
                    else:
                        retry_after = _parse_retry_after(res)
                        sleep(retry_after if retry_after is not None else ratelimited_sleep_interval)
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
                    if delay:
                        sleep(delay * random())  # nosec B311 - random jitter on the per-attempt retry backoff, not security-sensitive
            else:
                err_found = False
                for t in blocking_errors:
                    if t in res.text:
                        err_found = True
                        break
                if err_found:
                    handle_blocking(target, b_random_ua=b_random_ua, b_use_proxy=b_use_proxy)
                    was_blocked = True
                    if quit_on_blocking:
                        break
                else:
                    set_proxy_last_use_time(last_used_dict, proxies)
                    was_blocked = False
                    break
    if delay:
        sleep(delay * random())  # nosec B311 - random jitter on the inter-request delay to avoid a fixed request cadence, not security-sensitive

    if res is None:
        logger.warning("Could not get url %s", url)
    return res


def get_new_session(b_random_ua: bool = True, b_use_proxy: bool = True) -> None:
    """Create a fresh ``requests.Session``, reset the IP-query counter, optionally set a random user-agent header, and (if ``b_use_proxy``) obtain a new proxy -- storing all of it in module-level state."""
    global sess, proxies, headers
    global num_ip_queries

    new_sess = requests.Session()

    new_headers = template_headers
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

    with _state_lock:
        old_sess = sess
        sess = new_sess
        num_ip_queries = 0
        headers = new_headers
        local_proxy_user, local_proxy_pass, local_proxy_server = proxy_user, proxy_pass, proxy_server
        local_proxy_min_port, local_proxy_max_port, local_proxy_port, local_proxy_type = proxy_min_port, proxy_max_port, proxy_port, proxy_type

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
            new_proxies = get_new_smartproxy(
                local_proxy_user,
                local_proxy_pass,
                local_proxy_server,
                int(local_proxy_min_port) if local_proxy_min_port is not None else 20001,
                int(local_proxy_max_port) if local_proxy_max_port is not None else 37960,
                last_used_dict=last_used_dict,
                min_idle_interval_minutes=min_idle_interval_minutes if min_idle_interval_minutes is not None else 0,
                failed_dict=failed_dict,
                min_failed_idle_interval_minutes=min_failed_idle_interval_minutes if min_failed_idle_interval_minutes is not None else 60 * 24,
                proxy_port=local_proxy_port,
                proxy_type=local_proxy_type if local_proxy_type is not None else "http",
            )
            with _state_lock:
                proxies = new_proxies
            logger.info("proxy_server=%s", local_proxy_server)


def handle_blocking(target: str, b_random_ua: bool = True, b_use_proxy: bool = True) -> None:
    """Log that ``target`` got blocked, mark the current proxy as failed, sleep the configured jittered delay, then obtain a fresh session/proxy."""
    if proxies is not None:
        # Regression fix: proxies["https"].split("@")[1] raised IndexError for an
        # unauthenticated proxy (make_proxies_dict() builds an "@"-free URL when
        # proxy_user/proxy_pass are falsy -- an explicitly supported configuration).
        logger.warning("IP %s blocked. Receiving new proxy/session for %s", _redact_proxy_url(proxies.get("https")), target)
    else:
        logger.warning("IP blocked.")

    if delay:
        sleep(delay * random())  # nosec B311 - random jitter on the post-block backoff sleep before rotating session/proxy, not security-sensitive
    set_proxy_last_use_time(failed_dict, proxies)
    get_new_session(b_random_ua=b_random_ua, b_use_proxy=b_use_proxy)


def is_rotating_proxy(proxy_server: dict) -> Optional[bool]:
    """Return True if ``proxy_server`` config identifies Smartproxy's DC gateway with a fixed (non-range) port, i.e. a server-side auto-rotating proxy; otherwise None."""
    # {"PROXY_HOST": "gate.dc.smartproxy.com","PROXY_MIN_PORT": 20001,"PROXY_MAX_PORT": 37960}
    if proxy_server.get("PROXY_HOST", "").lower() == "gate.dc.smartproxy.com":
        if proxy_server.get("PROXY_MIN_PORT") == 20000:
            if proxy_server.get("PROXY_MAX_PORT") == 20000:
                return True
    return None


def _parse_retry_after(res: Any) -> Optional[float]:
    """Return the ``Retry-After`` delay in seconds, or None when the header is absent/unusable.

    Only the delta-seconds form is honoured; the HTTP-date form is rare in rate-limit responses
    and a mis-parsed date would produce a wildly wrong sleep. Absurd values are clamped so a
    hostile or buggy upstream cannot park the caller for hours.
    """
    try:
        raw = res.headers.get("Retry-After")
    except Exception as e:
        logger.debug("Response object exposes no readable headers mapping (%s); ignoring Retry-After.", e)
        return None
    if not raw:
        return None
    try:
        seconds = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if seconds < 0:
        return None
    return min(seconds, 300.0)


def download_to_file(
    url: str,
    filename: str,
    rewrite_existing: bool = True,
    timeout: int = 100,
    chunk_size: int = 1024,
    max_attempts: int = 5,
    headers: Optional[dict] = None,
    exit_codes: tuple = (),
):
    """Dropin replacement for urllib.request.urlretrieve(url, filename) that can hand for indefinitely long."""
    if headers is None:
        headers = {}
    # Make the actual request, set the timeout for no data to 10 seconds and enable streaming responses so we don't have to keep the large files in memory

    last_error: Optional[BaseException] = None
    for attempt in range(max_attempts):
        request: Optional[requests.Response] = None
        try:
            request = requests.get(url, timeout=timeout, headers=headers, stream=True)
            if request.status_code in exit_codes:
                # A caller-designated fatal status (e.g. 404): requests.get() does not raise on
                # 4xx/5xx, so without this the error page's body would be written as the file.
                return None
            # The GET must be re-issued on every attempt: a requests body is a single-use stream,
            # so retrying only the write loop iterates an already-consumed response, yields
            # nothing, and leaves the (already truncated by open(..., "wb")) file at 0 bytes while
            # the function returns the same value as a successful download.
            with open(filename, "wb") as fh:
                for chunk in request.iter_content(chunk_size * 1024):
                    fh.write(chunk)
        except Exception as e:
            last_error = e
            logger.exception(e)
            if attempt < max_attempts - 1:
                sleep(10 * random())  # nosec B311 - random jitter on the download-retry backoff sleep, not security-sensitive
                logger.info("Making another attempt")
        else:
            return None
        finally:
            if request is not None:
                request.close()

    # Every attempt failed. Remove whatever partial/truncated bytes are on disk so that
    # "no file" reliably signals failure instead of a plausible-looking empty artifact.
    try:
        os.remove(filename)
    except OSError:
        pass
    logger.error("download_to_file: all %d attempts failed for %s: %s", max_attempts, url, last_error)
    return None

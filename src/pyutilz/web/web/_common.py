"""Logging, URL-scheme guarding, credential redaction and small response-header parsing helpers
shared by the ``pyutilz.web.web`` submodules.

Nothing here reads or writes the package's mutable scraping state (session/proxy/header globals):
that state lives in the package ``__init__`` and is reached through the ``_facade`` indirection
documented there, so this module can be imported by every sibling without any cycle.
"""

import logging
import urllib.parse
from typing import Any, Optional

from joblib import hash as joblib_hash  # noqa: F401 -- re-exported: the package binds it for callers/tests

# ``__package__`` is "pyutilz.web.web" -- the name the historic flat module logged under. Spelling it
# this way rather than ``__name__`` (which would be "pyutilz.web.web._common") keeps every log record,
# and every caplog/logging-config lookup keyed on that name, behaving exactly as before the split.
logger = logging.getLogger(__package__)

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


def _error_log_throttle(n_errored: int, max_logged: int) -> bool:
    """True while per-URL error logging in ``download_in_parallel`` is still under its cap --
    a batch with a large fraction of failing URLs would otherwise log one ERROR line per failed
    URL with no limit, compounding into thousands of lines for a large batch."""
    return n_errored <= max_logged


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

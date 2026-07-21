"""IP verification utilities for proxy providers.

Shared helpers that work with any HTTP client (``requests``, ``curl_cffi``,
``httpx``, etc.) as long as it has a ``.get(url, **kwargs)`` returning an
object with a ``.text`` attribute.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, Optional

_log = logging.getLogger(__name__)

# orjson is faster than stdlib json; resolved once so the fallback branch never
# references the `orjson` name after a failed import (that raised UnboundLocalError
# here previously -- the name is local to the function once ANY `import orjson`
# appears in its body, so referencing `orjson.JSONDecodeError` in the except clause
# blew up when the import itself had failed).
_json_backend: Any
try:
    import orjson as _json_backend

    _json_loads = _json_backend.loads
    _JSONDecodeError = _json_backend.JSONDecodeError
except ImportError:
    import json as _json_backend  # type: ignore[no-redef]

    _json_loads = _json_backend.loads
    _JSONDecodeError = _json_backend.JSONDecodeError

__all__ = [
    "IP_CHECK_URLS",
    "parse_ip_response",
    "get_ip",
    "check_ip_matches_real",
    "verify_proxy_ip",
]

IP_CHECK_URLS = [
    "https://httpbin.org/ip",
    "https://api.ipify.org?format=json",
    "https://ifconfig.me/ip",
]


def parse_ip_response(text: str) -> str:
    """Extract IP from various IP-check service response formats.

    Handles JSON (``{"origin": "..."}`` or ``{"ip": "..."}``) and raw text.
    Comma-separated IPs (load-balanced httpbin) are split, first IP returned.

    >>> parse_ip_response('{"origin": "1.2.3.4"}')
    '1.2.3.4'
    >>> parse_ip_response('{"origin": "1.2.3.4, 5.6.7.8"}')
    '1.2.3.4'
    >>> parse_ip_response("9.10.11.12\\n")
    '9.10.11.12'
    """
    body = text.strip()
    if body.startswith("{"):
        try:
            data = _json_loads(body)
            raw = data.get("origin") or data.get("ip") or body
            return raw.split(",")[0].strip() if isinstance(raw, str) else raw
        except (_JSONDecodeError, ValueError):
            return body
    return body.split(",")[0].strip()


def get_ip(session_or_requests: Any, prx: Optional[Dict[str, str]] = None, *, timeout: int = 10) -> str:
    """Try :data:`IP_CHECK_URLS` in order, return first successful IP or ``'?'``."""
    kwargs: Dict[str, Any] = {"timeout": timeout}
    if prx is not None:
        kwargs["proxies"] = prx
    for url in IP_CHECK_URLS:
        try:
            r = session_or_requests.get(url, **kwargs)
            return parse_ip_response(r.text)
        except Exception:  # noqa: PERF203 -- per-iteration fault isolation is intentional (try the next URL)
            # Regression fix: this module's own docstring promises "any HTTP client (requests,
            # curl_cffi, httpx, etc.)" -- but (OSError, ValueError, KeyError) only actually
            # covers requests/curl_cffi (both raise OSError subclasses); httpx's exceptions
            # (HTTPError and its subclasses ConnectError/TimeoutException/etc.) subclass plain
            # Exception directly, verified against the installed httpx package. An httpx client
            # hitting a transient failure on the FIRST url would previously propagate uncaught
            # instead of falling through to try the next IP_CHECK_URLS entry as documented. The
            # function already degrades gracefully to "?" if every URL fails, so a blanket catch
            # here is safe.
            continue
    return "?"


def check_ip_matches_real(
    ip: str,
    real_ip: str,
    lib_name: str,
    *,
    exit_on_fail: bool = False,
) -> bool:
    """Return ``True`` if proxy works (IP differs from real).

    Returns ``False`` if the proxy leaks the real IP or all IP-check services
    are unreachable.  Calls ``sys.exit()`` when *exit_on_fail* is True.
    """
    if ip == "?":
        msg = f"[PROXY] {lib_name} via proxy FAILED: all IP check services unreachable"
        if exit_on_fail:
            sys.exit(msg)
        _log.error(msg)
        return False
    if ip == real_ip and real_ip != "?":
        msg = f"[PROXY] ERROR: {lib_name} returns real IP {real_ip} -- proxy not working!"
        if exit_on_fail:
            sys.exit(msg)
        _log.error(msg)
        return False
    _log.info("[PROXY] OK  %-10s: real=%s  proxy=%s", lib_name, real_ip, ip)
    return True


def verify_proxy_ip(
    proxy_session: Any,
    *,
    no_proxy_session: Any = None,
    lib_name: str = "proxy",
    exit_on_fail: bool = True,
    real_ip: Optional[str] = None,
) -> bool:
    """High-level proxy verification: compare proxy IP vs real IP.

    Parameters
    ----------
    proxy_session
        HTTP client configured to use the proxy.
    no_proxy_session
        HTTP client without proxy (to detect real IP).  If *None*,
        caller must provide *real_ip*.
    lib_name
        Label for log messages (e.g. ``"curl_cffi"``, ``"requests"``).
    exit_on_fail
        Call ``sys.exit()`` on failure.
    real_ip
        Pre-fetched real IP.  If *None*, fetched via *no_proxy_session*.
    """
    if real_ip is None:
        if no_proxy_session is None:
            raise ValueError("Provide either no_proxy_session or real_ip")
        real_ip = get_ip(no_proxy_session)
    proxy_ip = get_ip(proxy_session)
    return check_ip_matches_real(proxy_ip, real_ip, lib_name, exit_on_fail=exit_on_fail)

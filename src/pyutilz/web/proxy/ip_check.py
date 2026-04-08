"""IP verification utilities for proxy providers.

Shared helpers that work with any HTTP client (``requests``, ``curl_cffi``,
``httpx``, etc.) as long as it has a ``.get(url, **kwargs)`` returning an
object with a ``.text`` attribute.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict, Optional

_log = logging.getLogger(__name__)

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
            data = json.loads(body)
            raw = data.get("origin") or data.get("ip") or body
            return raw.split(",")[0].strip() if isinstance(raw, str) else raw
        except (json.JSONDecodeError, ValueError):
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
        except (OSError, ValueError, KeyError):
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
        msg = f"[PROXY] ERROR: {lib_name} returns real IP {real_ip} \u2014 proxy not working!"
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

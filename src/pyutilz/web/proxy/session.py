"""HTTP session helpers for proxy providers.

Provides context managers for ``curl_cffi`` and ``requests`` sessions
pre-configured with proxy settings from a :class:`ProxyProvider`.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .base import ProxyProvider

_log = logging.getLogger(__name__)

__all__ = ["curl_session", "requests_session"]


@contextmanager
def curl_session(
    provider: "ProxyProvider",
    impersonate: str = "chrome142",
    port_offset: Optional[int] = None,
):
    """Context manager yielding a ``curl_cffi.requests.Session`` with proxy.

    Parameters
    ----------
    provider
        Proxy provider instance.
    impersonate
        Browser TLS fingerprint (default ``"chrome142"``).
    port_offset
        Explicit port offset; if *None*, a healthy port is auto-selected.
    """
    from curl_cffi import requests as cr

    s = cr.Session(impersonate=impersonate, proxy=provider.proxy_url(port_offset))
    try:
        yield s
    finally:
        s.close()


@contextmanager
def requests_session(
    provider: "ProxyProvider",
    port_offset: Optional[int] = None,
):
    """Context manager yielding a ``requests.Session`` with proxy.

    Parameters
    ----------
    provider
        Proxy provider instance.
    port_offset
        Explicit port offset; if *None*, a healthy port is auto-selected.
    """
    import requests

    s = requests.Session()
    s.proxies = provider.proxies(port_offset)
    try:
        yield s
    finally:
        s.close()

"""HTTP session helpers for proxy providers.

Provides context managers for ``curl_cffi`` and ``requests`` sessions
pre-configured with proxy settings from a :class:`ProxyProvider`.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .base import ProxyProvider

_log = logging.getLogger(__name__)

__all__ = ["curl_session", "requests_session"]

# Neither helper had any request timeout, and ``requests.Session`` has no session-level timeout at
# all, so ``with requests_session(p) as s: s.get(url)`` blocked forever on a stalled exit IP.
DEFAULT_SESSION_TIMEOUT: float = 30.0


@contextmanager
def curl_session(
    provider: "ProxyProvider",
    impersonate: str = "chrome142",
    port_offset: Optional[int] = None,
    timeout: Optional[float] = DEFAULT_SESSION_TIMEOUT,
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
    timeout
        Default per-request timeout in seconds, applied by the session itself. ``None`` restores
        the historic "block forever" behaviour and must be chosen deliberately.

    Notes
    -----
    The resolved port offset is attached to the yielded session as ``proxy_port_offset`` (an
    extension, so ``with curl_session(p) as s`` keeps yielding the session itself). With
    ``port_offset=None`` the port used to be chosen inside ``provider.proxy_url()`` and never
    surfaced, so the caller could not attribute an outcome to it via
    ``provider.report_error()``/``report_success()`` and the health tracker these helpers exist to
    feed stayed permanently empty.
    """
    from curl_cffi import requests as cr

    offset = port_offset if port_offset is not None else provider.pick_port()
    s: Any = cr.Session(impersonate=impersonate, proxy=provider.proxy_url(offset), timeout=timeout)  # type: ignore[arg-type]  # impersonate is intentionally free-text str here; curl_cffi's stub pins it to a Literal enum that would need to be kept in sync with every curl_cffi release
    try:
        s.proxy_port_offset = offset
    except Exception as e:  # nosec B110 - a stubbed/slotted session object may reject attribute assignment; the timeout and proxy above are the load-bearing part
        _log.debug("Could not annotate session with proxy_port_offset: %s", e)
    try:
        yield s
    finally:
        s.close()


@contextmanager
def requests_session(
    provider: "ProxyProvider",
    port_offset: Optional[int] = None,
    timeout: Optional[float] = DEFAULT_SESSION_TIMEOUT,
):
    """Context manager yielding a ``requests.Session`` with proxy.

    Parameters
    ----------
    provider
        Proxy provider instance.
    port_offset
        Explicit port offset; if *None*, a healthy port is auto-selected.
    timeout
        Default per-request timeout in seconds. ``requests.Session`` has no timeout setting, so it
        is injected by wrapping ``Session.request``; an explicit ``timeout=`` passed by the caller
        on an individual call still wins. ``None`` restores the historic "block forever"
        behaviour and must be chosen deliberately.

    Notes
    -----
    The resolved port offset is attached to the yielded session as ``proxy_port_offset`` -- see
    :func:`curl_session` for why.
    """
    import requests

    offset = port_offset if port_offset is not None else provider.pick_port()
    s = requests.Session()
    s.proxies = provider.proxies(offset)
    s.proxy_port_offset = offset  # type: ignore[attr-defined]  # documented extension attribute, see the docstring above
    if timeout is not None:
        inner_request = s.request

        def _request_with_timeout(method: str, url: str, **kwargs: Any) -> Any:
            """Delegates to the session's own request, supplying the pool default timeout when the caller gave none."""
            kwargs.setdefault("timeout", timeout)
            return inner_request(method, url, **kwargs)

        s.request = _request_with_timeout  # type: ignore[method-assign]  # per-instance wrapper; subclassing Session would change the type callers receive
    try:
        yield s
    finally:
        s.close()

"""Rotating proxy support with provider abstraction and port health tracking.

Providers
---------
- :class:`DecodoProvider` — Decodo / Smartproxy (residential, mobile, datacenter)

Core
----
- :class:`ProxyProvider` — ABC for adding new providers
- :class:`ProxyConfig` — credential + endpoint dataclass
- :class:`PortHealthTracker` — thread-safe port banning

IP verification
---------------
- :func:`parse_ip_response`, :func:`get_ip`, :func:`verify_proxy_ip`

Session helpers
---------------
- :func:`curl_session`, :func:`requests_session`
"""

from .base import PortHealthTracker, ProxyConfig, ProxyProvider
from .decodo import (
    DecodoProvider,
    DecodoSubscription,
    DecodoTrafficReport,
    DecodoTrafficRow,
)
from .ip_check import (
    IP_CHECK_URLS,
    check_ip_matches_real,
    get_ip,
    parse_ip_response,
    verify_proxy_ip,
)
from .session import curl_session, requests_session

__all__ = [
    # Base
    "ProxyProvider",
    "ProxyConfig",
    "PortHealthTracker",
    # Decodo
    "DecodoProvider",
    "DecodoSubscription",
    "DecodoTrafficReport",
    "DecodoTrafficRow",
    # IP check
    "IP_CHECK_URLS",
    "parse_ip_response",
    "get_ip",
    "check_ip_matches_real",
    "verify_proxy_ip",
    # Sessions
    "curl_session",
    "requests_session",
]

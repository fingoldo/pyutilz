"""Decodo (formerly Smartproxy) proxy provider.

Supports residential, mobile, and datacenter proxies with sticky-session
port rotation and built-in API access for subscription & traffic stats.

API docs: https://help.decodo.com/reference/get-subscriptions

Usage::

    from pyutilz.web.proxy import DecodoProvider

    proxy = DecodoProvider.from_env()          # reads PROXY_* + DECODO_API_KEY from env
    url = proxy.proxy_url()                    # random healthy port
    proxy.report_error(42)                     # mark port 42 as problematic

    # Subscription & traffic
    subs = proxy.get_subscriptions()
    for s in subs:
        print(s.service_type, s.used_gb, "/", s.limit_gb)

    traffic = proxy.get_traffic(days=7, group_by="day")
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .base import PortHealthTracker, ProxyConfig, ProxyProvider

_log = logging.getLogger(__name__)

__all__ = [
    "DecodoProvider",
    "DecodoSubscription",
    "DecodoTrafficRow",
    "DecodoTrafficReport",
]

API_BASE = "https://api.decodo.com"

# Proxy types recognized by the Decodo API
PROXY_TYPES = (
    "residential_proxies",
    "mobile_proxies",
    "shared_dc_proxies",
    "rtc_universal_proxies",
    "rtc_universal_core_proxies",
)


# ── API data classes ────────────────────────────────────────────────────────

@dataclass
class DecodoSubscription:
    """Parsed subscription info from ``/v2/subscriptions``."""

    service_type: str
    traffic_limit_gb: float
    traffic_used_gb: float
    valid_from: str
    valid_until: str
    users_limit: int
    ip_address_limit: int
    raw: Dict[str, Any]

    @property
    def remaining_gb(self) -> float:
        return self.traffic_limit_gb - self.traffic_used_gb

    @property
    def usage_pct(self) -> float:
        return (self.traffic_used_gb / self.traffic_limit_gb * 100) if self.traffic_limit_gb > 0 else 0.0

    def summary(self) -> str:
        bar_len = 30
        filled = int(bar_len * self.usage_pct / 100)
        bar = "#" * filled + "-" * (bar_len - filled)
        return (
            f"{self.service_type}  "
            f"{self.valid_from} .. {self.valid_until}\n"
            f"  Limit: {_fmt_gb(self.traffic_limit_gb)}  "
            f"Used: {_fmt_gb(self.traffic_used_gb)}  "
            f"Remaining: {_fmt_gb(self.remaining_gb)}\n"
            f"  [{bar}] {self.usage_pct:.1f}%"
        )

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "DecodoSubscription":
        return cls(
            service_type=data.get("service_type", "unknown"),
            traffic_limit_gb=_safe_float(data.get("traffic_limit")),
            traffic_used_gb=_safe_float(data.get("traffic_per_period")),
            valid_from=data.get("valid_from", "?"),
            valid_until=data.get("valid_until", "?"),
            users_limit=int(data.get("users_limit", 0)),
            ip_address_limit=int(data.get("ip_address_limit", 0)),
            raw=data,
        )


@dataclass
class DecodoTrafficRow:
    """Single row from traffic statistics."""

    group_key: str
    requests: int
    traffic_bytes: float

    @property
    def traffic_gb(self) -> float:
        return self.traffic_bytes / (1024 ** 3)


@dataclass
class DecodoTrafficReport:
    """Parsed traffic statistics response."""

    rows: List[DecodoTrafficRow]
    total_requests: int
    total_bytes: float

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024 ** 3)

    def summary(self, group_by: str = "day") -> str:
        lines = [
            f"  {'Group':<25} {'Requests':>12} {'Traffic':>12}",
            f"  {'-' * 25} {'-' * 12} {'-' * 12}",
        ]
        for row in self.rows:
            traffic_str = _fmt_gb(row.traffic_gb) if row.traffic_bytes > 1_000_000 else f"{row.traffic_bytes:,.0f} B"
            lines.append(f"  {row.group_key:<25} {row.requests:>12,} {traffic_str:>12}")
        lines.append(f"  {'-' * 25} {'-' * 12} {'-' * 12}")
        total_str = _fmt_gb(self.total_gb) if self.total_bytes > 1_000_000 else f"{self.total_bytes:,.0f} B"
        lines.append(f"  {'TOTAL':<25} {self.total_requests:>12,} {total_str:>12}")
        return "\n".join(lines)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _safe_float(val: Any) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _fmt_gb(gb: float) -> str:
    if gb >= 1.0:
        return f"{gb:,.2f} GB"
    return f"{gb * 1024:,.1f} MB"


# ── Provider ────────────────────────────────────────────────────────────────

class DecodoProvider(ProxyProvider):
    """Decodo (Smartproxy) sticky-session proxy provider.

    Each port offset (added to ``base_port``) maps to a different exit IP
    that persists for the provider's sticky session TTL (~10 min for Decodo
    residential).  The health tracker auto-bans ports that produce repeated
    connection errors, forcing IP rotation.

    Parameters
    ----------
    config
        Proxy credentials and endpoint.
    api_key
        Decodo API key for subscription/traffic queries.  Optional —
        proxy routing works without it.
    health_tracker
        Shared :class:`PortHealthTracker` instance, or a new one is created.
    """

    def __init__(
        self,
        config: ProxyConfig,
        *,
        api_key: str = "",
        health_tracker: Optional[PortHealthTracker] = None,
    ) -> None:
        super().__init__(config, health_tracker=health_tracker)
        self.api_key = api_key

    @classmethod
    def from_env(
        cls,
        *,
        user_var: str = "PROXY_USER",
        pass_var: str = "PROXY_PASS",
        host_var: str = "PROXY_HOST",
        port_var: str = "PROXY_PORT",
        range_var: str = "PROXY_PORT_RANGE",
        api_key_var: str = "DECODO_API_KEY",
        default_range: int = 500,
        health_tracker: Optional[PortHealthTracker] = None,
    ) -> "DecodoProvider":
        """Create provider from environment variables.

        Required: ``PROXY_USER``, ``PROXY_PASS``, ``PROXY_HOST``, ``PROXY_PORT``.
        Optional: ``PROXY_PORT_RANGE`` (default 500), ``DECODO_API_KEY``.
        """
        missing = [v for v in [user_var, pass_var, host_var, port_var] if v not in os.environ]
        if missing:
            raise EnvironmentError(f"Missing required env vars: {', '.join(missing)}")

        try:
            base_port = int(os.environ[port_var])
        except ValueError:
            raise ValueError(f"{port_var} must be an integer, got '{os.environ[port_var]}'")

        port_range = int(os.environ.get(range_var, default_range))

        config = ProxyConfig(
            user=os.environ[user_var],
            password=os.environ[pass_var],
            host=os.environ[host_var],
            base_port=base_port,
            port_range=port_range,
        )
        return cls(config, api_key=os.environ.get(api_key_var, ""), health_tracker=health_tracker)

    def proxy_url(self, port_offset: Optional[int] = None) -> str:
        """Return proxy URL with sticky-session port.

        If *port_offset* is ``None``, picks a random healthy port.
        """
        offset = port_offset if port_offset is not None else self.pick_port()
        c = self.config
        return f"{c.protocol}://{c.user}:{c.password}@{c.host}:{c.base_port + offset}"

    # ── Decodo API ──────────────────────────────────────────────────────────

    def _api_headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise RuntimeError(
                "DECODO_API_KEY not set. "
                "Get it from: dashboard.decodo.com -> Settings -> API Keys"
            )
        return {"Authorization": self.api_key, "Content-Type": "application/json"}

    def get_subscriptions(self) -> List[DecodoSubscription]:
        """Fetch subscription info from Decodo API.

        Tries ``/v2/subscriptions`` first, falls back to ``/v2/sub-users``.
        Not all account types support all endpoints.
        """
        import requests

        headers = self._api_headers()
        for endpoint in ["/v2/subscriptions", "/v2/sub-users"]:
            try:
                r = requests.get(f"{API_BASE}{endpoint}", headers=headers, timeout=15)
                r.raise_for_status()
                data = r.json()
                items = data if isinstance(data, list) else [data]
                return [DecodoSubscription.from_api(item) for item in items]
            except requests.HTTPError:
                continue
        raise RuntimeError("Could not fetch subscriptions from any Decodo API endpoint")

    def get_endpoints(self) -> Dict[str, Any]:
        """Fetch available proxy endpoints from ``/v2/endpoints``.

        Returns dict with ``"random"`` and ``"sticky"`` keys, each mapping
        to a list of location entries (hostname, port_range, etc.).
        """
        import requests

        headers = self._api_headers()
        result: Dict[str, Any] = {}
        r = requests.get(f"{API_BASE}/v2/endpoints", headers=headers, timeout=15)
        r.raise_for_status()
        for item in r.json():
            ep_type = item.get("type", "unknown")
            sub_r = requests.get(f"{API_BASE}/v2/endpoints/{item.get('url', '').split('/')[-1]}",
                                 headers=headers, timeout=15)
            if sub_r.status_code == 200:
                result[ep_type] = sub_r.json()
        return result

    def get_traffic(
        self,
        *,
        proxy_type: str = "residential_proxies",
        days: int = 0,
        start: Optional[str] = None,
        end: Optional[str] = None,
        group_by: str = "day",
        limit: int = 500,
        sort_order: str = "desc",
    ) -> DecodoTrafficReport:
        """Fetch traffic statistics from ``/api/v2/statistics/traffic``.

        Parameters
        ----------
        proxy_type
            One of :data:`PROXY_TYPES`.
        days
            If > 0, compute *start*/*end* as ``[now - days, now]``.
        start, end
            Explicit date strings (``"YYYY-MM-DD HH:MM:SS"``).
        group_by
            ``"day"`` | ``"target"`` | ``"country"`` | ``"protocol"`` |
            ``"hour"`` | ``"week"`` | ``"month"``.
        """
        import requests

        now = datetime.now(timezone.utc)
        if days > 0:
            start = (now - timedelta(days=days)).strftime("%Y-%m-%d 00:00:00")
            end = now.strftime("%Y-%m-%d %H:%M:%S")
        elif start is None or end is None:
            raise ValueError("Provide either days>0 or explicit start/end")

        body = {
            "proxyType": proxy_type,
            "startDate": start,
            "endDate": end,
            "groupBy": group_by,
            "limit": limit,
            "page": 1,
            "sortBy": "grouping_key",
            "sortOrder": sort_order,
        }
        r = requests.post(
            f"{API_BASE}/api/v2/statistics/traffic",
            headers=self._api_headers(),
            json=body,
            timeout=30,
        )
        r.raise_for_status()
        return _parse_traffic_response(r.json(), group_by)

    def print_usage(
        self,
        *,
        days: int = 0,
        group_by: str = "day",
        proxy_type: str = "residential_proxies",
    ) -> None:
        """Print human-readable subscription + traffic summary to stdout."""
        try:
            subs = self.get_subscriptions()
            for sub in subs:
                print(f"\n{'=' * 55}")
                print(sub.summary())
                print(f"{'=' * 55}")
        except Exception as e:
            print(f"  Error fetching subscriptions: {e}")

        if days > 0:
            print(f"\nTraffic ({days}d, grouped by {group_by}):")
            try:
                report = self.get_traffic(proxy_type=proxy_type, days=days, group_by=group_by)
                print(report.summary(group_by))
            except Exception as e:
                print(f"  Error fetching traffic: {e}")


def _parse_traffic_response(data: Any, group_by: str) -> DecodoTrafficReport:
    """Parse the traffic API response into a :class:`DecodoTrafficReport`."""
    raw_rows = data if isinstance(data, list) else data.get("data", data.get("results", []))
    if not isinstance(raw_rows, list):
        raw_rows = []

    rows: List[DecodoTrafficRow] = []
    total_reqs = 0
    total_bytes = 0.0
    for row in raw_rows:
        key = row.get("grouping_key", row.get(group_by, "?"))
        reqs = int(row.get("requests", 0))
        traffic = _safe_float(row.get("totals", row.get("traffic", 0)))
        rows.append(DecodoTrafficRow(group_key=str(key), requests=reqs, traffic_bytes=traffic))
        total_reqs += reqs
        total_bytes += traffic

    return DecodoTrafficReport(rows=rows, total_requests=total_reqs, total_bytes=total_bytes)

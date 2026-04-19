"""Base proxy abstractions: ProxyProvider ABC and PortHealthTracker.

ProxyProvider defines the interface all proxy providers must implement.
PortHealthTracker provides thread-safe port banning for sticky-session proxies
(where each port maps to a different exit IP for some TTL).
"""

from __future__ import annotations

import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)

__all__ = ["ProxyProvider", "PortHealthTracker", "ProxyConfig"]


@dataclass
class ProxyConfig:
    """Credential + endpoint configuration for a proxy provider."""

    user: str
    password: str
    host: str
    base_port: int
    port_range: int = 500
    protocol: str = "http"


class _PortStats:
    """Per-port request outcomes within a sliding window."""

    __slots__ = ("outcomes",)

    def __init__(self) -> None:
        # List of (monotonic_time, is_error: bool)
        self.outcomes: List[tuple] = []

    def record(self, now: float, is_error: bool) -> None:
        self.outcomes.append((now, is_error))

    def trim(self, cutoff: float) -> None:
        self.outcomes = [o for o in self.outcomes if o[0] > cutoff]

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def errors(self) -> int:
        return sum(1 for _, e in self.outcomes if e)

    @property
    def error_rate(self) -> float:
        n = len(self.outcomes)
        return self.errors / n if n > 0 else 0.0


class PortHealthTracker:
    """Thread-safe tracker that bans ports whose error rate is statistically anomalous.

    Designed for sticky-session proxies (Decodo, Oxylabs, etc.) where each port
    offset maps to a different exit IP.

    **Ban logic**: all ports experience some baseline error rate.  A port is
    banned only when:

    1. It has at least *min_requests* outcomes in the sliding window, AND
    2. Its error rate is >= *ban_rate_multiplier* times the average error rate
       across all ports that also have >= *min_requests* outcomes.

    This avoids banning ports during normal background noise and only targets
    genuine outliers.

    Parameters
    ----------
    window : float
        Sliding window (seconds) for tracking outcomes.
    min_requests : int
        Minimum requests per port before it participates in comparison.
    ban_rate_multiplier : float
        A port is banned when ``port_error_rate >= avg_error_rate * multiplier``.
    ban_duration : float
        How long (seconds) a banned port stays excluded.
    min_errors : int
        Minimum number of errors a port must have before it can be banned.
        Prevents banning ports with a single fluke error even when the rate
        multiplier triggers.
    absolute_ban_rate : float
        Absolute error-rate threshold (0..1) that triggers a ban even when
        there are fewer than 2 qualified peers. The peer-comparison path
        cannot fire at startup or after a ban wipe — every port is fresh or
        unqualified — so a lone port with 100% errors would otherwise spin
        forever. Set to 0.0 to disable the absolute fallback.
    """

    def __init__(
        self,
        window: float = 600.0,
        min_requests: int = 30,
        ban_rate_multiplier: float = 2.0,
        ban_duration: float = 900.0,
        min_errors: int = 2,
        absolute_ban_rate: float = 0.5,
    ) -> None:
        self._lock = threading.Lock()
        self._ports: Dict[int, _PortStats] = {}
        self._banned: Dict[int, float] = {}  # port_offset -> ban expiry (monotonic)
        self.window = window
        self.min_requests = min_requests
        self.ban_rate_multiplier = ban_rate_multiplier
        self.ban_duration = ban_duration
        self.min_errors = min_errors
        self.absolute_ban_rate = absolute_ban_rate

    def _trim_all(self, now: float) -> None:
        """Remove stale outcomes outside the window (caller must hold _lock)."""
        cutoff = now - self.window
        empty = []
        for port, ps in self._ports.items():
            ps.trim(cutoff)
            if ps.total == 0:
                empty.append(port)
        for port in empty:
            del self._ports[port]

    def _maybe_ban(self, port_offset: int, now: float) -> None:
        """Check if *port_offset* should be banned relative to peers (caller holds _lock)."""
        ps = self._ports.get(port_offset)
        if ps is None or ps.total < self.min_requests or ps.errors < self.min_errors:
            return

        # Compute average error rate across all ports with enough data
        qualified: List[_PortStats] = [p for p in self._ports.values() if p.total >= self.min_requests]
        if len(qualified) < 2:
            # Peer-baseline path needs ≥2 qualified ports. Without a fallback,
            # a lone bad port at 100% error rate would never ban — common at
            # startup (only one port seen requests) or after ban churn (peers
            # wiped). Absolute-rate fallback: if this port is clearly broken
            # on its own merits, ban it.
            if self.absolute_ban_rate > 0 and ps.error_rate >= self.absolute_ban_rate:
                self._banned[port_offset] = now + self.ban_duration
                self._ports.pop(port_offset, None)
                _log.warning(
                    "[PROXY] Port %d banned for %.0fs: absolute error rate %.1f%% >= threshold %.1f%% "
                    "(no peer baseline — only %d qualified port(s))",
                    port_offset, self.ban_duration,
                    ps.error_rate * 100, self.absolute_ban_rate * 100, len(qualified),
                )
            return

        total_errors = sum(p.errors for p in qualified)
        total_reqs = sum(p.total for p in qualified)
        avg_rate = total_errors / total_reqs if total_reqs > 0 else 0.0

        if avg_rate <= 0:
            # No errors anywhere — nothing to ban
            return

        port_rate = ps.error_rate
        if port_rate >= avg_rate * self.ban_rate_multiplier:
            self._banned[port_offset] = now + self.ban_duration
            # Clear this port's history so it starts fresh after ban
            self._ports.pop(port_offset, None)
            _log.warning(
                "[PROXY] Port %d banned for %.0fs: error rate %.1f%% vs avg %.1f%% "
                "(%.1fx, threshold %.1fx, %d ports compared)",
                port_offset, self.ban_duration,
                port_rate * 100, avg_rate * 100,
                port_rate / avg_rate, self.ban_rate_multiplier,
                len(qualified),
            )

    def report_error(self, port_offset: int) -> None:
        """Record a connection error for *port_offset*."""
        now = time.monotonic()
        with self._lock:
            self._trim_all(now)
            ps = self._ports.setdefault(port_offset, _PortStats())
            ps.record(now, is_error=True)
            self._maybe_ban(port_offset, now)

    def report_success(self, port_offset: int) -> None:
        """Record a successful request for *port_offset*."""
        now = time.monotonic()
        with self._lock:
            ps = self._ports.setdefault(port_offset, _PortStats())
            ps.record(now, is_error=False)

    def is_banned(self, port_offset: int) -> bool:
        """Check if *port_offset* is currently banned."""
        with self._lock:
            return self._is_banned_unlocked(port_offset)

    def _is_banned_unlocked(self, port_offset: int) -> bool:
        """Check ban status (caller must hold ``_lock``)."""
        expiry = self._banned.get(port_offset)
        if expiry is None:
            return False
        if time.monotonic() >= expiry:
            del self._banned[port_offset]
            return False
        return True

    def pick_port(self, port_range: int) -> int:
        """Pick a random non-banned port offset in ``[1, port_range]``.

        Falls back to any port if >90% are banned.
        """
        with self._lock:
            if not self._banned:
                return random.randint(1, port_range)
            for _ in range(10):
                candidate = random.randint(1, port_range)
                if not self._is_banned_unlocked(candidate):
                    return candidate
            # Expire stale bans
            expired = [p for p, exp in self._banned.items() if time.monotonic() >= exp]
            for p in expired:
                del self._banned[p]
            if len(self._banned) >= port_range * 0.9:
                _log.warning(
                    "[PROXY] %d/%d ports banned — using random port anyway",
                    len(self._banned), port_range,
                )
                return random.randint(1, port_range)
            return random.randint(1, port_range)

    def stats(self) -> Dict[str, Any]:
        """Return current health stats (for debugging / metrics)."""
        now = time.monotonic()
        with self._lock:
            self._trim_all(now)
            qualified = {p: ps for p, ps in self._ports.items() if ps.total >= self.min_requests}
            total_errors = sum(ps.errors for ps in qualified.values())
            total_reqs = sum(ps.total for ps in qualified.values())
            avg_rate = total_errors / total_reqs if total_reqs > 0 else 0.0
            return {
                "banned_count": len(self._banned),
                "banned_ports": list(self._banned.keys())[:20],
                "tracked_ports": len(self._ports),
                "qualified_ports": len(qualified),
                "avg_error_rate": round(avg_rate, 4),
            }

    def reset(self) -> None:
        """Clear all tracking data."""
        with self._lock:
            self._ports.clear()
            self._banned.clear()


class ProxyProvider(ABC):
    """Abstract base class for proxy providers.

    Subclasses implement :meth:`proxy_url` and optionally override
    :meth:`verify` and provider-specific configuration.
    """

    def __init__(self, config: ProxyConfig, *, health_tracker: Optional[PortHealthTracker] = None) -> None:
        self.config = config
        self.health = health_tracker or PortHealthTracker()

    @abstractmethod
    def proxy_url(self, port_offset: Optional[int] = None) -> str:
        """Return a full proxy URL (``protocol://user:pass@host:port``)."""

    def proxies(self, port_offset: Optional[int] = None) -> Dict[str, str]:
        """Return ``{http, https}`` dict for the ``requests`` library."""
        url = self.proxy_url(port_offset)
        return {"http": url, "https": url}

    def pick_port(self) -> int:
        """Pick a healthy port offset via the health tracker."""
        return self.health.pick_port(self.config.port_range)

    def report_error(self, port_offset: int) -> None:
        self.health.report_error(port_offset)

    def report_success(self, port_offset: int) -> None:
        self.health.report_success(port_offset)

    @property
    def name(self) -> str:
        return self.__class__.__name__

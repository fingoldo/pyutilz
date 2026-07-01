"""Instrumentation hooks fired by ``get_or_tune`` / sweeps (pure, no HW/disk deps)."""
from __future__ import annotations

import logging

from typing import Optional, Protocol

logger = logging.getLogger(__name__)


class TuningHooks(Protocol):
    """Optional instrumentation fired by ``get_or_tune`` / sweeps. Supply a
    custom implementation for structured logging / dashboards; the default
    ``LoggerHooks`` logs at sensible levels."""

    def env_override(self, kernel: str, choice: str) -> None: ...
    def cache_hit(self, kernel: str, dims: dict, region: dict) -> None: ...
    def cache_miss(self, kernel: str, dims: dict) -> None: ...
    def sweep_start(self, kernel: str, axes: list) -> None: ...
    def sweep_end(self, kernel: str, n_regions: int) -> None: ...
    def winner_chosen(self, kernel: str, region: Optional[dict], reason: str) -> None: ...
    def persist(self, kernel: str, path: Optional[str], n_regions: int) -> None: ...
    def invalidation(self, kernel: str, reason: str) -> None: ...
    def concurrent_sweep_detected(self, kernel: str) -> None: ...


class LoggerHooks:
    """Default ``TuningHooks`` -- logs at the levels the cache used historically
    (info for sweep start/done + invalidation, debug for hit/miss/persist)."""

    def env_override(self, kernel, choice):
        logger.debug("kernel_tuning_cache: %s env override -> %r", kernel, choice)

    def cache_hit(self, kernel, dims, region):
        logger.debug("kernel_tuning_cache: %s cache hit %s", kernel, dims)

    def cache_miss(self, kernel, dims):
        logger.debug("kernel_tuning_cache: %s cache miss %s", kernel, dims)

    def sweep_start(self, kernel, axes):
        logger.info("kernel_tuning_cache: %s sweep starting (axes=%s)", kernel, axes)

    def sweep_end(self, kernel, n_regions):
        logger.info("kernel_tuning_cache: %s sweep done (%d regions)", kernel, n_regions)

    def winner_chosen(self, kernel, region, reason):
        logger.debug("kernel_tuning_cache: %s winner (%s): %s", kernel, reason, region)

    def persist(self, kernel, path, n_regions):
        logger.debug("kernel_tuning_cache: %s persisted %d regions -> %s", kernel, n_regions, path)

    def invalidation(self, kernel, reason):
        logger.info("kernel_tuning_cache: %s invalidated (%s); will re-tune", kernel, reason)

    def concurrent_sweep_detected(self, kernel):
        # v3 is lock-free (O_EXCL marker, no filelock, no timeout): this fires only
        # when a STALE marker is being self-healed (steal of a dead/over-budget
        # owner). That is a healthy recovery, not an error -- and under heavy
        # concurrent load (many fits, one killed mid-sweep) every peer that steals
        # would otherwise spam a scary per-call WARNING about a "lock timeout" that
        # no longer exists. Keep it at DEBUG with accurate wording; live-contention
        # losers don't reach here at all (they give up silently and fall back).
        logger.debug("kernel_tuning_cache: %s reclaiming stale sweep marker (self-heal)", kernel)


_DEFAULT_HOOKS = LoggerHooks()

"""Instrumentation hooks fired by ``get_or_tune`` / sweeps (pure, no HW/disk deps)."""
from __future__ import annotations

import logging

from typing import Optional, Protocol

logger = logging.getLogger(__name__)


class TuningHooks(Protocol):
    """Optional instrumentation fired by ``get_or_tune`` / sweeps. Supply a
    custom implementation for structured logging / dashboards; the default
    ``LoggerHooks`` logs at sensible levels."""

    def env_override(self, kernel: str, choice: str) -> None:
        """Called when an env var forces ``choice`` for ``kernel``, bypassing the cache/sweep."""
        ...

    def cache_hit(self, kernel: str, dims: dict, region: dict) -> None:
        """Called when a cached tuning ``region`` is found for ``kernel`` at the given ``dims``."""
        ...

    def cache_miss(self, kernel: str, dims: dict) -> None:
        """Called when no cached tuning region covers ``kernel`` at the given ``dims``."""
        ...

    def sweep_start(self, kernel: str, axes: list) -> None:
        """Called when a tuning sweep begins for ``kernel`` over the given ``axes``."""
        ...

    def sweep_end(self, kernel: str, n_regions: int) -> None:
        """Called when a tuning sweep for ``kernel`` finishes, having produced ``n_regions`` regions."""
        ...

    def winner_chosen(self, kernel: str, region: Optional[dict], reason: str) -> None:
        """Called once the winning ``region`` for ``kernel`` is selected, with the ``reason`` for the choice."""
        ...

    def persist(self, kernel: str, path: Optional[str], n_regions: int) -> None:
        """Called after ``n_regions`` tuning regions for ``kernel`` are persisted to ``path``."""
        ...

    def invalidation(self, kernel: str, reason: str) -> None:
        """Called when the cached tuning data for ``kernel`` is invalidated, with the ``reason`` why."""
        ...

    def concurrent_sweep_detected(self, kernel: str) -> None:
        """Called when a concurrent/stale sweep marker for ``kernel`` is detected and reclaimed."""
        ...


class LoggerHooks:
    """Default ``TuningHooks`` -- logs at the levels the cache used historically
    (info for sweep start/done + invalidation, debug for hit/miss/persist)."""

    def env_override(self, kernel, choice):
        """Log at DEBUG that an env var forced ``choice`` for ``kernel``."""
        logger.debug("kernel_tuning_cache: %s env override -> %r", kernel, choice)

    def cache_hit(self, kernel, dims, region):
        """Log at DEBUG that a cached ``region`` was found for ``kernel`` at ``dims``."""
        logger.debug("kernel_tuning_cache: %s cache hit %s", kernel, dims)

    def cache_miss(self, kernel, dims):
        """Log at DEBUG that no cached region covers ``kernel`` at ``dims``."""
        logger.debug("kernel_tuning_cache: %s cache miss %s", kernel, dims)

    def sweep_start(self, kernel, axes):
        """Log at INFO that a tuning sweep is starting for ``kernel`` over ``axes``."""
        logger.info("kernel_tuning_cache: %s sweep starting (axes=%s)", kernel, axes)

    def sweep_end(self, kernel, n_regions):
        """Log at INFO that the tuning sweep for ``kernel`` finished with ``n_regions`` regions."""
        logger.info("kernel_tuning_cache: %s sweep done (%d regions)", kernel, n_regions)

    def winner_chosen(self, kernel, region, reason):
        """Log at DEBUG the winning ``region`` chosen for ``kernel`` and the ``reason``."""
        logger.debug("kernel_tuning_cache: %s winner (%s): %s", kernel, reason, region)

    def persist(self, kernel, path, n_regions):
        """Log at DEBUG that ``n_regions`` tuning regions for ``kernel`` were persisted to ``path``."""
        logger.debug("kernel_tuning_cache: %s persisted %d regions -> %s", kernel, n_regions, path)

    def invalidation(self, kernel, reason):
        """Log at INFO that the cache for ``kernel`` was invalidated for ``reason`` and will be re-tuned."""
        logger.info("kernel_tuning_cache: %s invalidated (%s); will re-tune", kernel, reason)

    def concurrent_sweep_detected(self, kernel):
        """Log at DEBUG that a stale sweep marker for ``kernel`` was self-healed (reclaimed)."""
        # v3 is lock-free (O_EXCL marker, no filelock, no timeout): this fires only
        # when a STALE marker is being self-healed (steal of a dead/over-budget
        # owner). That is a healthy recovery, not an error -- and under heavy
        # concurrent load (many fits, one killed mid-sweep) every peer that steals
        # would otherwise spam a scary per-call WARNING about a "lock timeout" that
        # no longer exists. Keep it at DEBUG with accurate wording; live-contention
        # losers don't reach here at all (they give up silently and fall back).
        logger.debug("kernel_tuning_cache: %s reclaiming stale sweep marker (self-heal)", kernel)


_DEFAULT_HOOKS = LoggerHooks()

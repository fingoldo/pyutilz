"""Per-key async single-flight cache: dedupes concurrent fetches of the same key.

Wraps any ``MutableMapping`` (a plain dict, or a ``cachetools.TTLCache``/``LRUCache`` for bounded/
expiring storage) so that N coroutines racing to look up the SAME missing key trigger exactly ONE
call to the (presumably expensive -- a DB query, an API call) fetcher; every other waiter gets the
one fetch's result instead of independently re-fetching. Concurrent lookups for DIFFERENT keys
still run in parallel -- this is per-key deduplication, not a global lock.

Single-asyncio-event-loop model only (matches the common asyncio server/pipeline shape): every
:class:`SingleFlightCache` instance's internal state (the in-flight-fetch tracking dict and its
lock) is only safe when accessed from one event loop thread. Do not call ``get_or_fetch`` from
``asyncio.to_thread()`` or multiple OS threads sharing one instance.

Ported from a downstream project's own in-process caching module (originally hardcoded to one
global inflight dict per cache instance it needed, duplicated by hand for each of 6 module-level
caches); generalized into one small reusable class here so a new cache needs zero copy-pasted
dedup plumbing.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Awaitable, Callable, Dict, Generic, Hashable, MutableMapping, Optional, TypeVar

logger = logging.getLogger(__name__)

_K = TypeVar("_K", bound=Hashable)
_V = TypeVar("_V")


class SingleFlightCache(Generic[_K, _V]):
    """Owns the in-flight-fetch tracking state for one or more caches looked up via
    :meth:`get_or_fetch`. Typically one instance per logical cache (or shared across several
    caches when their key spaces are guaranteed disjoint -- e.g. by using tuple keys tagged with
    the cache name)."""

    def __init__(self) -> None:
        self._inflight: Dict[Hashable, "asyncio.Event"] = {}
        # Lazily created on first async use (see _get_inflight_lock) rather than here: on Python
        # 3.8/3.9, asyncio.Lock()'s constructor eagerly calls get_event_loop(), which raises
        # "There is no current event loop in thread 'MainThread'" when instantiated from sync
        # code with no event loop yet running/set in this thread (e.g. a plain, non-async test
        # or any other sync construction site) -- found 2026-08-03.
        self._inflight_lock: Optional[asyncio.Lock] = None
        self.hits = 0
        self.misses = 0

    def _get_inflight_lock(self) -> asyncio.Lock:
        """Return the shared in-flight lock, creating it on first (always async) use."""
        if self._inflight_lock is None:
            self._inflight_lock = asyncio.Lock()
        return self._inflight_lock

    def __getstate__(self) -> Dict[str, Any]:
        """Drop the unpicklable asyncio.Lock/Event state; __setstate__ rebuilds it fresh.

        asyncio.Lock/Event are event-loop-bound and unpicklable; the in-flight-fetch bookkeeping
        is also meaningless across a pickle round-trip (no waiter can be resumed after
        unpickling), so it's cleared too rather than carried over stale.
        """
        state = self.__dict__.copy()
        del state["_inflight_lock"]
        state["_inflight"] = {}
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state from __getstate__; the lock is rebuilt lazily on next async use
        (see __init__), not here -- unpickling itself is sync and must not eagerly construct one."""
        self.__dict__.update(state)
        self._inflight_lock = None

    def get_cache_metrics(self) -> Dict[str, float]:
        """Return current hit/miss counters and hit rate (0-100)."""
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": round(self.hits / total * 100, 1) if total else 0.0,
        }

    def reset_metrics(self) -> None:
        """Zero the hit/miss counters."""
        self.hits = 0
        self.misses = 0

    async def get_or_fetch(
        self,
        cache: MutableMapping[_K, _V],
        key: "_K",
        fetcher: Callable[[], Awaitable[_V]],
        default: Optional[_V] = None,
    ) -> Optional[_V]:
        """Return ``cache[key]``, populating it via ``await fetcher()`` on a miss.

        If two coroutines race on the same missing ``key``, only the first calls ``fetcher()``;
        the rest await its result instead of independently re-fetching. If ``fetcher`` raises, the
        exception is swallowed (logged at the raise site is the caller's responsibility -- this
        function only returns ``default``) and NOTHING is cached: a transient failure must not
        poison the cache for whatever TTL/lifetime it has. The next caller for this key retries
        the fetch from scratch. Callers waiting on a fetch that failed also get ``default``.
        """
        if __debug__ and threading.current_thread() is not threading.main_thread():
            raise RuntimeError(
                f"SingleFlightCache.get_or_fetch called from non-main thread " f"{threading.current_thread().name!r} -- this class is single-event-loop only."
            )
        if key in cache:
            self.hits += 1
            return cache[key]

        self.misses += 1
        is_fetcher = False
        evt: Optional[asyncio.Event] = None
        async with self._get_inflight_lock():
            # Re-check after acquiring the lock -- another coroutine may have finished the fetch
            # while we were waiting for the lock.
            if key in cache:
                return cache[key]
            if key not in self._inflight:
                self._inflight[key] = asyncio.Event()
                is_fetcher = True
            else:
                # Capture the Event ref while the lock is held -- it stays valid even after the
                # fetcher pops the key from self._inflight in its finally block.
                evt = self._inflight[key]

        if not is_fetcher:
            # Another coroutine is already fetching -- wait for it instead of re-fetching.
            if evt is not None:
                await evt.wait()
            if key in cache:
                return cache[key]
            # The fetcher failed -- it deliberately did not cache anything (see the except branch
            # below), so there is nothing to read back here.
            return default if default is not None else cache.get(key, default)

        try:
            value = await fetcher()
            cache[key] = value
            return value
        except Exception:
            # Do NOT cache the default -- a transient failure must not poison the cache for the
            # remainder of its lifetime/TTL.
            logger.debug("SingleFlightCache.get_or_fetch: fetcher for %r raised; returning default (not cached)", key, exc_info=True)
            return default
        finally:
            async with self._get_inflight_lock():
                evt = self._inflight.pop(key, None)
            if evt is not None:
                evt.set()

    def clear(self) -> None:
        """Reset in-flight tracking and hit/miss counters.

        Does NOT touch the caller's cache object(s) -- call this only when no concurrent
        ``get_or_fetch`` coroutines are in flight (e.g. between batch runs), since it discards
        in-flight-fetch bookkeeping without waking waiters.
        """
        assert threading.current_thread() is threading.main_thread(), "SingleFlightCache.clear() must run on the main thread"
        self._inflight.clear()
        self.reset_metrics()

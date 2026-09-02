"""Per-key async single-flight cache: dedupes concurrent fetches of the same key.

Wraps any ``MutableMapping`` (a plain dict, or a ``cachetools.TTLCache``/``LRUCache`` for bounded/
expiring storage) so that N coroutines racing to look up the SAME missing key trigger exactly ONE
call to the (presumably expensive -- a DB query, an API call) fetcher; every other waiter gets the
one fetch's result instead of independently re-fetching. Concurrent lookups for DIFFERENT keys
still run in parallel -- this is per-key deduplication, not a global lock.

Single-asyncio-event-loop model only (matches the common asyncio server/pipeline shape): every
:class:`SingleFlightCache` instance's internal state (the in-flight-fetch tracking dict and its
lock) is only safe when accessed from one event loop. The instance binds itself to the running
loop on first ``get_or_fetch`` and rejects any other loop afterwards; that loop may live on any
thread. Do not call ``get_or_fetch`` from ``asyncio.to_thread()`` or from a second event loop
sharing one instance.

Ported from a downstream project's own in-process caching module (originally hardcoded to one
global inflight dict per cache instance it needed, duplicated by hand for each of 6 module-level
caches); generalized into one small reusable class here so a new cache needs zero copy-pasted
dedup plumbing.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, Generic, Hashable, MutableMapping, Optional, TypeVar, cast

logger = logging.getLogger(__name__)

_K = TypeVar("_K", bound=Hashable)
_V = TypeVar("_V")

# Sentinel distinguishing "absent" from a legitimately cached None, in ONE mapping lookup.
_MISSING: Any = object()


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
        # The loop this instance was first used from. The real constraint is one EVENT LOOP, not
        # one OS thread -- a single loop running on a dedicated worker thread (the standard way
        # to embed an async subsystem in a sync application) satisfies the documented contract
        # and used to be rejected outright.
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.hits = 0
        self.misses = 0

    def _bind_loop(self, caller: str) -> None:
        """Bind this instance to the running event loop on first async use, then enforce identity.

        Replaces the old main-thread check: what the class actually requires is that all of its
        internal state is touched from ONE event loop, wherever that loop happens to run.
        """
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError(
                f"SingleFlightCache.{caller} called from event loop {loop!r}, but this instance is bound to {self._loop!r} -- this class is single-event-loop only."
            )

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
        self._loop = None

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
        # Unconditional (not `assert`/`__debug__`-gated): a safety property that disappears under
        # `python -O` is present exactly where it is least needed.
        self._bind_loop("get_or_fetch")
        # One lookup, not `in` followed by `[]`: on an expiring store (cachetools.TTLCache, the
        # docstring's own recommendation) a TTL boundary falling between the two raises KeyError
        # out of a cache-lookup helper.
        value = cache.get(key, _MISSING)
        if value is not _MISSING:
            self.hits += 1
            return cast("_V", value)

        self.misses += 1
        is_fetcher = False
        evt: Optional[asyncio.Event] = None
        async with self._get_inflight_lock():
            # Re-check after acquiring the lock -- another coroutine may have finished the fetch
            # while we were waiting for the lock.
            value = cache.get(key, _MISSING)
            if value is not _MISSING:
                return cast("_V", value)
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
            value = cache.get(key, _MISSING)
            if value is not _MISSING:
                return cast("_V", value)
            # The fetcher failed -- it deliberately did not cache anything (see the except branch
            # below), so there is nothing to read back here.
            return default

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
        self._inflight.clear()
        self.reset_metrics()

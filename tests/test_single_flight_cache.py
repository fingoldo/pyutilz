"""Tests for pyutilz.system.single_flight_cache."""

from __future__ import annotations

import asyncio
import pickle

import pytest

from pyutilz.system.single_flight_cache import SingleFlightCache


@pytest.mark.asyncio
async def test_cache_hit_returns_without_calling_fetcher():
    sfc: SingleFlightCache[str, int] = SingleFlightCache()
    cache = {"k": 42}
    calls = []

    async def fetcher():
        calls.append(1)
        return 99

    result = await sfc.get_or_fetch(cache, "k", fetcher)
    assert result == 42
    assert calls == []
    assert sfc.hits == 1
    assert sfc.misses == 0


@pytest.mark.asyncio
async def test_cache_miss_calls_fetcher_and_populates_cache():
    sfc: SingleFlightCache[str, int] = SingleFlightCache()
    cache: dict[str, int] = {}
    calls = []

    async def fetcher():
        calls.append(1)
        return 7

    result = await sfc.get_or_fetch(cache, "k", fetcher)
    assert result == 7
    assert cache["k"] == 7
    assert len(calls) == 1
    assert sfc.hits == 0
    assert sfc.misses == 1


@pytest.mark.asyncio
async def test_concurrent_fetches_for_same_key_dedup_to_one_call():
    sfc: SingleFlightCache[str, int] = SingleFlightCache()
    cache: dict[str, int] = {}
    call_count = 0

    async def fetcher():
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.05)
        return 123

    results = await asyncio.gather(*[sfc.get_or_fetch(cache, "shared", fetcher) for _ in range(10)])
    assert call_count == 1, "concurrent fetches for the same key must dedup to a single call"
    assert len(results) == 10, "every gathered waiter must have produced a value"
    assert all(r == 123 for r in results)


@pytest.mark.asyncio
async def test_concurrent_fetches_for_different_keys_run_independently():
    sfc: SingleFlightCache[str, int] = SingleFlightCache()
    cache: dict[str, int] = {}
    call_order = []

    async def fetcher(key):
        call_order.append(key)
        await asyncio.sleep(0.01)
        return len(key)

    results = await asyncio.gather(
        sfc.get_or_fetch(cache, "a", lambda: fetcher("a")),
        sfc.get_or_fetch(cache, "bb", lambda: fetcher("bb")),
        sfc.get_or_fetch(cache, "ccc", lambda: fetcher("ccc")),
    )
    assert sorted(call_order) == ["a", "bb", "ccc"]
    assert results == [1, 2, 3]


@pytest.mark.asyncio
async def test_fetcher_exception_returns_default_and_does_not_cache():
    sfc: SingleFlightCache[str, int] = SingleFlightCache()
    cache: dict[str, int] = {}

    async def failing_fetcher():
        raise RuntimeError("db down")

    result = await sfc.get_or_fetch(cache, "k", failing_fetcher, default=-1)
    assert result == -1
    assert "k" not in cache


@pytest.mark.asyncio
async def test_retry_after_failure_succeeds():
    sfc: SingleFlightCache[str, int] = SingleFlightCache()
    cache: dict[str, int] = {}
    attempt = 0

    async def flaky_fetcher():
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            raise RuntimeError("transient")
        return 55

    first = await sfc.get_or_fetch(cache, "k", flaky_fetcher, default=None)
    assert first is None
    second = await sfc.get_or_fetch(cache, "k", flaky_fetcher, default=None)
    assert second == 55
    assert cache["k"] == 55


@pytest.mark.asyncio
async def test_waiters_get_default_when_fetcher_fails():
    sfc: SingleFlightCache[str, int] = SingleFlightCache()
    cache: dict[str, int] = {}
    started = asyncio.Event()

    async def failing_fetcher():
        started.set()
        await asyncio.sleep(0.05)
        raise RuntimeError("boom")

    async def waiter():
        await started.wait()
        return await sfc.get_or_fetch(cache, "k", failing_fetcher, default=-1)

    fetch_task = asyncio.create_task(sfc.get_or_fetch(cache, "k", failing_fetcher, default=-1))
    wait_task = asyncio.create_task(waiter())
    results = await asyncio.gather(fetch_task, wait_task)
    assert results == [-1, -1]


def test_get_cache_metrics():
    sfc: SingleFlightCache[str, int] = SingleFlightCache()
    sfc.hits = 3
    sfc.misses = 1
    metrics = sfc.get_cache_metrics()
    assert metrics == {"hits": 3, "misses": 1, "total": 4, "hit_rate": 75.0}


def test_get_cache_metrics_empty():
    sfc: SingleFlightCache[str, int] = SingleFlightCache()
    assert sfc.get_cache_metrics() == {"hits": 0, "misses": 0, "total": 0, "hit_rate": 0.0}


def test_clear_resets_metrics():
    sfc: SingleFlightCache[str, int] = SingleFlightCache()
    sfc.hits = 5
    sfc.misses = 2
    sfc.clear()
    assert sfc.hits == 0
    assert sfc.misses == 0


def test_picklable():
    sfc: SingleFlightCache[str, int] = SingleFlightCache()
    sfc.hits = 3
    sfc.misses = 1
    restored = pickle.loads(pickle.dumps(sfc))
    assert restored.hits == 3
    assert restored.misses == 1
    assert restored.get_cache_metrics()["hit_rate"] == 75.0


@pytest.mark.asyncio
async def test_usable_after_pickle_roundtrip():
    sfc: SingleFlightCache[str, int] = SingleFlightCache()
    restored = pickle.loads(pickle.dumps(sfc))
    cache: dict[str, int] = {}

    async def fetcher():
        return 9

    result = await restored.get_or_fetch(cache, "k", fetcher)
    assert result == 9

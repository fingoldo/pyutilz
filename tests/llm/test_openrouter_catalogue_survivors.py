"""Behaviour, log and dispatch pins for the OpenRouter catalogue module (mutation survivors)."""

from __future__ import annotations

import asyncio
import logging

import pytest

import pyutilz.llm.openrouter_provider as pkg
from pyutilz.llm.openrouter_provider import _catalogue as cat


class _Resp:
    def __init__(self, payload, status: int = 200):
        self._payload = payload
        self._status = status

    def raise_for_status(self):
        if self._status >= 400:
            raise RuntimeError(f"HTTP {self._status}")

    def json(self):
        return self._payload


class _Http:
    def __init__(self, payload=None, status: int = 200, error: Exception | None = None):
        self.calls = []
        self._payload = payload if payload is not None else {"data": []}
        self._status = status
        self._error = error

    def get(self, url, timeout):
        self.calls.append((url, timeout))
        if self._error is not None:
            raise self._error
        return _Resp(self._payload, self._status)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setattr(pkg, "_MODELS_CATALOGUE", None)
    monkeypatch.setattr(pkg, "_MODELS_CATALOGUE_FETCHED_AT", float("inf"))
    cat._UNPRICED_MODELS_WARNED.clear()
    yield
    cat._UNPRICED_MODELS_WARNED.clear()


def _install(monkeypatch, http: _Http) -> _Http:
    monkeypatch.setattr(pkg, "httpx", http)
    return http


def _fixed_clock(monkeypatch, now: float) -> None:
    monkeypatch.setattr(cat.time, "monotonic", lambda: now)


def _entry(prompt=None, completion=None, extra=None):
    pricing = {}
    if prompt is not None:
        pricing["prompt"] = prompt
    if completion is not None:
        pricing["completion"] = completion
    if extra:
        pricing.update(extra)
    return {"pricing": pricing}


class TestFreshness:
    def test_no_cache_is_not_fresh(self):
        assert cat._catalogue_is_fresh() is False

    def test_a_cache_inside_the_ttl_is_fresh(self, monkeypatch):
        _fixed_clock(monkeypatch, 1000.0)
        monkeypatch.setattr(pkg, "_MODELS_CATALOGUE", {})
        monkeypatch.setattr(pkg, "_MODELS_CATALOGUE_FETCHED_AT", 950.0)
        monkeypatch.setattr(pkg, "_MODELS_CATALOGUE_TTL_SECONDS", 100.0)
        assert cat._catalogue_is_fresh() is True

    def test_a_cache_exactly_at_the_ttl_is_stale(self, monkeypatch):
        _fixed_clock(monkeypatch, 1000.0)
        monkeypatch.setattr(pkg, "_MODELS_CATALOGUE", {})
        monkeypatch.setattr(pkg, "_MODELS_CATALOGUE_FETCHED_AT", 900.0)
        monkeypatch.setattr(pkg, "_MODELS_CATALOGUE_TTL_SECONDS", 100.0)
        assert cat._catalogue_is_fresh() is False


class TestFetch:
    def test_the_default_timeout_is_ten_seconds_and_the_url_is_the_models_url(self, monkeypatch):
        http = _install(monkeypatch, _Http({"data": [{"id": "m"}]}))
        cat._fetch_models_catalogue()
        assert http.calls == [(pkg._MODELS_URL, 10.0)]

    def test_a_successful_fetch_is_cached_with_a_stamp_and_returned(self, monkeypatch):
        _fixed_clock(monkeypatch, 1234.0)
        _install(monkeypatch, _Http({"data": [{"id": "m", "pricing": {}}]}))
        out = cat._fetch_models_catalogue()
        assert list(out) == ["m"]
        assert pkg._MODELS_CATALOGUE is out
        assert pkg._MODELS_CATALOGUE_FETCHED_AT == 1234.0

    def test_the_models_key_is_used_when_data_is_absent(self, monkeypatch):
        _install(monkeypatch, _Http({"models": [{"id": "x"}]}))
        assert list(cat._fetch_models_catalogue()) == ["x"]

    def test_entries_without_an_id_or_that_are_not_dicts_are_dropped(self, monkeypatch):
        _install(monkeypatch, _Http({"data": [5, {"name": "no-id"}, {"id": "ok"}]}))
        assert list(cat._fetch_models_catalogue()) == ["ok"]

    def test_a_status_error_returns_empty_and_is_not_cached(self, monkeypatch):
        _install(monkeypatch, _Http({"data": [{"id": "m"}]}, status=500))
        assert cat._fetch_models_catalogue() == {}
        assert pkg._MODELS_CATALOGUE is None

    def test_a_transport_failure_is_logged_with_the_exact_message(self, monkeypatch, caplog):
        _install(monkeypatch, _Http(error=RuntimeError("boom")))
        caplog.set_level(logging.WARNING)
        assert cat._fetch_models_catalogue() == {}
        assert caplog.records[-1].getMessage() == (
            "OpenRouter /models catalogue fetch failed (boom); estimate_cost() will return 0 for this call. Next call retries."
        )

    def test_a_fresh_cache_is_served_without_a_request(self, monkeypatch):
        _fixed_clock(monkeypatch, 1000.0)
        cached = {"m": {"id": "m"}}
        monkeypatch.setattr(pkg, "_MODELS_CATALOGUE", cached)
        monkeypatch.setattr(pkg, "_MODELS_CATALOGUE_FETCHED_AT", 999.0)
        http = _install(monkeypatch, _Http())
        assert cat._fetch_models_catalogue() is cached
        assert http.calls == []

    def test_a_stale_cache_is_refetched_outside_an_event_loop(self, monkeypatch):
        _fixed_clock(monkeypatch, 10_000.0)
        monkeypatch.setattr(pkg, "_MODELS_CATALOGUE", {"old": {}})
        monkeypatch.setattr(pkg, "_MODELS_CATALOGUE_FETCHED_AT", 0.0)
        http = _install(monkeypatch, _Http({"data": [{"id": "new"}]}))
        assert list(cat._fetch_models_catalogue()) == ["new"]
        assert len(http.calls) == 1


class TestEventLoopThread:
    def test_an_uncached_call_inside_a_running_loop_returns_empty_without_a_request(self, monkeypatch, caplog):
        http = _install(monkeypatch, _Http())
        caplog.set_level(logging.DEBUG)

        async def inside():
            return cat._fetch_models_catalogue()

        assert asyncio.run(inside()) == {}
        assert http.calls == []
        assert "OpenRouter catalogue not yet cached and cannot be fetched from the event loop; returning empty" in caplog.text

    def test_a_fresh_cache_inside_a_running_loop_is_returned_without_the_stale_branch(self, monkeypatch, caplog):
        _fixed_clock(monkeypatch, 1000.0)
        cached = {"m": {"id": "m"}}
        monkeypatch.setattr(pkg, "_MODELS_CATALOGUE", cached)
        monkeypatch.setattr(pkg, "_MODELS_CATALOGUE_FETCHED_AT", 999.0)
        http = _install(monkeypatch, _Http())
        caplog.set_level(logging.DEBUG)

        async def inside():
            return cat._fetch_models_catalogue()

        assert asyncio.run(inside()) is cached
        assert http.calls == []
        assert "Serving a stale OpenRouter catalogue" not in caplog.text

    def test_a_stale_cache_inside_a_running_loop_is_served_and_logged(self, monkeypatch, caplog):
        _fixed_clock(monkeypatch, 10_000.0)
        cached = {"stale": {"id": "stale"}}
        monkeypatch.setattr(pkg, "_MODELS_CATALOGUE", cached)
        monkeypatch.setattr(pkg, "_MODELS_CATALOGUE_FETCHED_AT", 0.0)
        http = _install(monkeypatch, _Http())
        caplog.set_level(logging.DEBUG)

        async def inside():
            return cat._fetch_models_catalogue()

        assert asyncio.run(inside()) is cached
        assert http.calls == []
        assert "Serving a stale OpenRouter catalogue rather than blocking the event loop on a refetch" in caplog.text


class TestEnsureWarm:
    def test_a_fresh_cache_skips_the_fetch(self, monkeypatch):
        _fixed_clock(monkeypatch, 1000.0)
        monkeypatch.setattr(pkg, "_MODELS_CATALOGUE", {})
        monkeypatch.setattr(pkg, "_MODELS_CATALOGUE_FETCHED_AT", 999.0)
        seen = []
        monkeypatch.setattr(cat, "_fetch_models_catalogue", lambda timeout=10.0: seen.append(timeout) or {})
        asyncio.run(cat._ensure_catalogue_warm_async())
        assert seen == []

    def test_the_default_timeout_is_ten_seconds(self, monkeypatch):
        seen = []
        monkeypatch.setattr(cat, "_fetch_models_catalogue", lambda timeout=10.0: seen.append(timeout) or {})
        asyncio.run(cat._ensure_catalogue_warm_async())
        assert seen == [10.0]

    @pytest.mark.parametrize(
        "version, expected",
        [((3, 8), "executor"), ((3, 9), "thread"), ((3, 10), "thread"), ((4, 0), "thread")],
    )
    def test_the_dispatch_path_follows_the_python_version(self, monkeypatch, version, expected):
        monkeypatch.setattr(cat.sys, "version_info", version)
        used = []

        async def run():
            loop = asyncio.get_running_loop()
            real_to_thread = getattr(asyncio, "to_thread", None)  # absent on 3.8; the version parameter then picks the executor path
            real_exec = loop.run_in_executor

            in_thread = []

            async def spy_to_thread(func, *args):
                used.append("thread")
                in_thread.append(True)
                try:
                    if real_to_thread is None:  # 3.8: no asyncio.to_thread, so dispatch through the real executor
                        return await real_exec(None, func, *args)
                    return await real_to_thread(func, *args)
                finally:
                    in_thread.pop()

            def spy_exec(executor, func, *args):
                if not in_thread:
                    used.append("executor")
                return real_exec(executor, func, *args)

            # asyncio.to_thread is absent on 3.8: raising=False lets the spy stand in for the version the test fakes
            monkeypatch.setattr(cat.asyncio, "to_thread", spy_to_thread, raising=False)
            monkeypatch.setattr(loop, "run_in_executor", spy_exec)
            _install(monkeypatch, _Http({"data": [{"id": "m"}]}))
            await cat._ensure_catalogue_warm_async()

        asyncio.run(run())
        assert used == [expected]
        assert pkg._MODELS_CATALOGUE is not None

    def test_the_fetch_is_handed_to_the_thread_with_its_timeout(self, monkeypatch):
        monkeypatch.setattr(cat.sys, "version_info", (3, 9))
        seen = []

        async def fake_to_thread(func, *args):
            return func(*args)

        # asyncio.to_thread is absent on 3.8; the fake keeps the 3.9+ branch under test on that interpreter.
        monkeypatch.setattr(cat.asyncio, "to_thread", fake_to_thread, raising=False)
        monkeypatch.setattr(cat, "_fetch_models_catalogue", lambda timeout=10.0: seen.append(timeout) or {})
        asyncio.run(cat._ensure_catalogue_warm_async(timeout=3.0))
        assert seen == [3.0]

    def test_the_executor_receives_no_executor_and_the_fetch(self, monkeypatch):
        monkeypatch.setattr(cat.sys, "version_info", (3, 8))
        seen = []

        async def run():
            loop = asyncio.get_running_loop()

            async def fake_exec(executor, func, *args):
                seen.append((executor, func, args))
                return func(*args)

            monkeypatch.setattr(loop, "run_in_executor", fake_exec)
            monkeypatch.setattr(cat, "_fetch_models_catalogue", lambda timeout=10.0: {})
            await cat._ensure_catalogue_warm_async(timeout=2.0)

        asyncio.run(run())
        assert seen[0][0] is None
        assert seen[0][2] == (2.0,)


class TestPerTokenPair:
    def test_the_per_token_prices_are_scaled_to_per_million(self, monkeypatch):
        _install(monkeypatch, _Http({"data": [{"id": "m", "pricing": {"prompt": "0.000001", "completion": "0.000002"}}]}))
        assert cat._per_token_cost_pair_or_none("m") == pytest.approx((1.0, 2.0), abs=1e-9)

    def test_a_missing_model_is_none(self, monkeypatch):
        _install(monkeypatch, _Http({"data": []}))
        assert cat._per_token_cost_pair_or_none("nope") is None

    def test_an_empty_prompt_string_reads_as_zero(self, monkeypatch):
        _install(monkeypatch, _Http({"data": [{"id": "m", "pricing": {"prompt": "", "completion": "0"}}]}))
        assert cat._per_token_cost_pair_or_none("m") == (0.0, 0.0)

    def test_an_empty_completion_string_reads_as_zero(self, monkeypatch):
        _install(monkeypatch, _Http({"data": [{"id": "m", "pricing": {"prompt": "0", "completion": ""}}]}))
        assert cat._per_token_cost_pair_or_none("m") == (0.0, 0.0)

    def test_a_non_numeric_price_is_none_and_logged(self, monkeypatch, caplog):
        _install(monkeypatch, _Http({"data": [{"id": "m", "pricing": {"prompt": "abc", "completion": "1"}}]}))
        caplog.set_level(logging.DEBUG)
        assert cat._per_token_cost_pair_or_none("m") is None
        assert "_per_token_cost_pair_or_none: pricing is not numeric: ValueError: " in caplog.text

    def test_a_missing_price_key_is_zero_not_a_failure(self, monkeypatch):
        _install(monkeypatch, _Http({"data": [{"id": "m", "pricing": {"completion": "0.000001"}}]}))
        assert cat._per_token_cost_pair_or_none("m") == pytest.approx((0.0, 1.0), abs=1e-9)


class TestCacheReadCost:
    def test_the_cache_read_price_is_scaled_to_per_million(self, monkeypatch):
        _install(monkeypatch, _Http({"data": [{"id": "m", "pricing": {"input_cache_read": "0.000001"}}]}))
        assert cat._cache_read_cost_per_1m_or_none("m") == pytest.approx(1.0, abs=1e-9)

    def test_a_missing_model_is_none(self, monkeypatch):
        _install(monkeypatch, _Http({"data": []}))
        assert cat._cache_read_cost_per_1m_or_none("nope") is None

    def test_a_model_without_pricing_is_none_without_logging(self, monkeypatch, caplog):
        _install(monkeypatch, _Http({"data": [{"id": "m"}]}))
        caplog.set_level(logging.DEBUG)
        assert cat._cache_read_cost_per_1m_or_none("m") is None
        assert "pricing is not numeric" not in caplog.text

    def test_an_empty_string_price_is_none_without_logging(self, monkeypatch, caplog):
        _install(monkeypatch, _Http({"data": [{"id": "m", "pricing": {"input_cache_read": ""}}]}))
        caplog.set_level(logging.DEBUG)
        assert cat._cache_read_cost_per_1m_or_none("m") is None
        assert "pricing is not numeric" not in caplog.text

    def test_a_non_numeric_price_is_none_and_logged(self, monkeypatch, caplog):
        _install(monkeypatch, _Http({"data": [{"id": "m", "pricing": {"input_cache_read": "abc"}}]}))
        caplog.set_level(logging.DEBUG)
        assert cat._cache_read_cost_per_1m_or_none("m") is None
        assert "_cache_read_cost_per_1m_or_none: pricing is not numeric: ValueError: " in caplog.text

    def test_a_model_with_no_pricing_block_is_none(self, monkeypatch):
        _install(monkeypatch, _Http({"data": [{"id": "m"}]}))
        assert cat._cache_read_cost_per_1m_or_none("m") is None


class TestCacheWriteCost:
    def test_the_cache_write_price_is_scaled_to_per_million(self, monkeypatch):
        _install(monkeypatch, _Http({"data": [{"id": "m", "pricing": {"input_cache_write": "0.000002"}}]}))
        assert cat._cache_write_cost_per_1m_or_none("m") == pytest.approx(2.0, abs=1e-9)

    def test_a_missing_model_is_none_not_an_error(self, monkeypatch):
        _install(monkeypatch, _Http({"data": []}))
        assert cat._cache_write_cost_per_1m_or_none("nope") is None

    def test_a_model_without_pricing_block_is_none(self, monkeypatch):
        _install(monkeypatch, _Http({"data": [{"id": "m"}]}))
        assert cat._cache_write_cost_per_1m_or_none("m") is None

    def test_an_empty_string_price_is_none_without_logging(self, monkeypatch, caplog):
        _install(monkeypatch, _Http({"data": [{"id": "m", "pricing": {"input_cache_write": ""}}]}))
        caplog.set_level(logging.DEBUG)
        assert cat._cache_write_cost_per_1m_or_none("m") is None
        assert "pricing is not numeric" not in caplog.text

    def test_a_non_numeric_price_is_none_and_logged(self, monkeypatch, caplog):
        _install(monkeypatch, _Http({"data": [{"id": "m", "pricing": {"input_cache_write": "abc"}}]}))
        caplog.set_level(logging.DEBUG)
        assert cat._cache_write_cost_per_1m_or_none("m") is None
        assert "_cache_write_cost_per_1m_or_none: pricing is not numeric: ValueError: " in caplog.text


class TestUnpricedWarning:
    def test_an_unpriced_model_is_zero_and_warned_once_with_the_exact_message(self, monkeypatch, caplog):
        _install(monkeypatch, _Http({"data": []}))
        caplog.set_level(logging.WARNING)
        assert cat._per_token_cost_pair("x") == (0.0, 0.0)
        assert cat._per_token_cost_pair("x") == (0.0, 0.0)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "no catalogue pricing" in r.getMessage()]
        assert len(warnings) == 1
        assert warnings[0].getMessage() == (
            "OpenRouter has no catalogue pricing for 'x' (catalogue unavailable or model absent); "
            "per-token cost estimates for it are reported as 0. Reconcile with actual_cost_usd."
        )

    def test_a_priced_model_is_returned_and_not_warned(self, monkeypatch, caplog):
        _install(monkeypatch, _Http({"data": [{"id": "m", "pricing": {"prompt": "0.000001", "completion": "0"}}]}))
        caplog.set_level(logging.WARNING)
        assert cat._per_token_cost_pair("m")[0] == pytest.approx(1.0, abs=1e-9)
        assert "no catalogue pricing" not in caplog.text

    def test_each_unpriced_model_is_warned_separately(self, monkeypatch, caplog):
        _install(monkeypatch, _Http({"data": []}))
        caplog.set_level(logging.WARNING)
        cat._per_token_cost_pair("a")
        cat._per_token_cost_pair("b")
        assert cat._UNPRICED_MODELS_WARNED == {"a", "b"}

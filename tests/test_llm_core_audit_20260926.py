"""Regression tests for the 2026-09-26 shared-core audit (audits/2026-09-26/30_llm_core.md, CORE-*)."""

from __future__ import annotations

import asyncio
import base64
import gc
import json
import logging
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pyutilz.llm import _progress
from pyutilz.llm import base as base_mod
from pyutilz.llm._messages import images_on_disk
from pyutilz.llm.base import LLMProvider, PerCallAttr, _longest_prefix_pricing, longest_prefix_lookup
from pyutilz.llm.exceptions import JSONParsingError


class _Provider(LLMProvider):
    """Minimal concrete provider: records every generate() call's kwargs."""

    _PRICING = {"known-model": (1.0, 2.0), "fam-model": (3.0, 4.0)}

    def __init__(self, max_concurrent: int = 1, model: str = "known-model") -> None:
        self._max_concurrent = max_concurrent
        self.model_name = model
        self.calls: list[dict[str, Any]] = []

    async def generate(self, prompt: str, system: str | None = None, temperature: float | None = 0.7, max_tokens: int = 0, **kwargs: Any) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        async with self.semaphore:
            _progress.note_stream_progress(None, "x" * 10)
            await asyncio.sleep(0.01)
        return prompt

    async def count_tokens(self, text: str) -> int:
        return len(text)


def _png_uri(media_type: str = "image/png") -> str:
    return f"data:{media_type};base64," + base64.b64encode(b"\x89PNG-bytes").decode()


# ── CORE-1 ────────────────────────────────────────────────────────────────────────────────────────


class TestImagesOnDiskTraversal:
    def test_traversal_media_type_writes_nothing_outside_the_temp_dir(self, caplog):
        probe = Path(tempfile.gettempdir()) / "pyutilz_core1_trav_probe"
        probe.unlink(missing_ok=True)
        evil = "data:image/x/../../pyutilz_core1_trav_probe;base64," + base64.b64encode(b"hi").decode()
        with caplog.at_level(logging.WARNING), images_on_disk([evil]) as (fragment, written):
            assert fragment == ""
            assert written == []
        assert not probe.exists()
        assert "1 of 1 image(s) not sent" in caplog.text

    def test_allowlisted_types_get_fixed_extensions_and_are_removed_after(self):
        with images_on_disk([_png_uri("image/png"), _png_uri("image/jpeg"), _png_uri("image/svg+xml")]) as (fragment, written):
            assert [p.name for p in written] == ["attachment_1.png", "attachment_2.jpg", "attachment_3.svg"]
            directory = written[0].parent
            assert len(written) == 3 and all(p.parent == directory and p.read_bytes() == b"\x89PNG-bytes" for p in written)
            assert str(written[0]) in fragment
        assert not directory.exists()

    def test_unknown_image_subtype_is_dropped_not_written(self):
        with images_on_disk([_png_uri("image/x-evil"), _png_uri("image/png")]) as (_fragment, written):
            assert [p.name for p in written] == ["attachment_2.png"]

    def test_directory_removed_when_the_body_raises(self):
        seen: list[Path] = []
        with pytest.raises(RuntimeError), images_on_disk([_png_uri()]) as (_fragment, written):
            seen.extend(written)
            raise RuntimeError("boom")
        assert seen and not seen[0].parent.exists()


# ── CORE-2 ────────────────────────────────────────────────────────────────────────────────────────


class TestPerLoopResources:
    def test_semaphore_survives_a_second_asyncio_run_under_contention(self):
        provider = _Provider(max_concurrent=1)

        async def batch() -> list[str]:
            return list(await asyncio.gather(*(provider.generate(f"p{i}") for i in range(3))))

        assert asyncio.run(batch()) == ["p0", "p1", "p2"]
        assert asyncio.run(batch()) == ["p0", "p1", "p2"]

    def test_semaphore_is_distinct_per_loop_and_stable_within_one(self):
        provider = _Provider()

        async def grab() -> tuple[Any, Any]:
            return provider.semaphore, provider.semaphore

        a1, a2 = asyncio.run(grab())
        b1, _ = asyncio.run(grab())
        assert a1 is a2
        assert a1 is not b1

    def test_explicit_semaphore_is_kept_on_every_loop(self):
        provider = _Provider()
        sem = asyncio.Semaphore(5)
        provider.semaphore = sem

        async def grab() -> Any:
            return provider.semaphore

        assert asyncio.run(grab()) is sem
        assert asyncio.run(grab()) is sem

    def test_httpx_client_is_cloned_for_a_second_loop(self):
        httpx = pytest.importorskip("httpx")
        seen: list[str] = []

        def handler(request: Any) -> Any:
            seen.append(request.headers["Authorization"])
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(handler)
        provider = _Provider()
        provider._client = httpx.AsyncClient(base_url="https://example.test/v1", headers={"Authorization": "Bearer k"}, transport=transport)
        original = provider.__dict__["_client"]

        async def call() -> Any:
            client = provider._client
            resp = await client.get("/ping")
            assert resp.json() == {"ok": True}
            return client

        first = asyncio.run(call())
        second = asyncio.run(call())
        assert first is original
        assert second is not original
        assert str(second.base_url) == str(original.base_url)
        assert second._transport is transport  # a socket-free transport is shared, not rebuilt
        assert seen == ["Bearer k", "Bearer k"]
        assert base_mod.LLMProvider.__dict__["_client"].all_clients(provider)[0] is original

    def test_network_transport_is_rebuilt_not_shared(self):
        httpx = pytest.importorskip("httpx")
        provider = _Provider()
        provider._client = httpx.AsyncClient(base_url="https://example.test", timeout=httpx.Timeout(12.0, connect=3.0))

        async def grab() -> Any:
            return provider._client

        first = asyncio.run(grab())
        second = asyncio.run(grab())
        assert second is not first
        assert isinstance(second._transport, httpx.AsyncHTTPTransport)
        assert second._transport is not first._transport
        assert second.timeout == first.timeout

    def test_client_outside_a_loop_and_non_httpx_values_are_untouched(self):
        provider = _Provider()
        sentinel = object()
        provider._client = sentinel
        assert provider._client is sentinel

        async def grab() -> Any:
            return provider._client

        assert asyncio.run(grab()) is sentinel
        assert asyncio.run(grab()) is sentinel


# ── CORE-3 ────────────────────────────────────────────────────────────────────────────────────────


class _Holder:
    value = PerCallAttr(lambda: None)


class _Slotted:
    __slots__ = ()
    value = PerCallAttr(lambda: None)


class TestPerCallAttrPrunes:
    def test_dead_instances_are_dropped_from_the_store(self):
        def run() -> int:
            for i in range(2000):
                h = _Holder()
                h.value = i
                del h
            keep = _Holder()
            keep.value = "last"
            gc.collect()
            keep.value = "again"
            store = _Holder.__dict__["value"]._var.get({})
            assert keep.value == "again"
            return len(store)

        import contextvars

        assert contextvars.copy_context().run(run) == 1

    def test_type_without_weakrefs_is_refused(self):
        with pytest.raises(TypeError, match="__weakref__"):
            _Slotted().value = 1


# ── CORE-4 ────────────────────────────────────────────────────────────────────────────────────────


class ReadTimeout(Exception):  # noqa: N818 -- named like httpx.ReadTimeout on purpose
    """Named like httpx's, which is how the stop policy recognises it."""


def _state(exc: BaseException, elapsed: float, attempt: int) -> SimpleNamespace:
    return SimpleNamespace(outcome=SimpleNamespace(exception=lambda: exc), seconds_since_start=elapsed, attempt_number=attempt)


class TestCallDeadline:
    def test_deadline_stops_retries(self, monkeypatch):
        from pyutilz.llm import _retry

        monkeypatch.setattr(_retry, "MAX_CALL_SECONDS", 100.0)
        assert _retry._stop_policy(_state(ValueError("x"), 99.0, 2)) is False
        assert _retry._stop_policy(_state(ValueError("x"), 100.0, 2)) is True

    def test_zero_disables_the_deadline(self, monkeypatch):
        from pyutilz.llm import _retry

        monkeypatch.setattr(_retry, "MAX_CALL_SECONDS", 0.0)
        assert _retry._stop_policy(_state(ValueError("x"), 10_000_000.0, 2)) is False

    def test_default_deadline_is_bounded_and_fits_two_long_attempts(self):
        from pyutilz.llm import _retry
        from pyutilz.llm._timeouts import DerivedTimeoutMixin

        assert _retry._DEFAULT_MAX_CALL_SECONDS >= 2 * DerivedTimeoutMixin._max_derived_timeout_s
        assert _retry._DEFAULT_MAX_CALL_SECONDS <= 4 * 3600

    def test_consecutive_read_timeouts_stop_and_other_errors_reset_the_streak(self, monkeypatch):
        from pyutilz.llm import _retry

        monkeypatch.setattr(_retry, "MAX_CONSECUTIVE_TIMEOUTS", 3)
        monkeypatch.setattr(_retry, "MAX_CALL_SECONDS", 0.0)
        state = _state(ReadTimeout(), 1.0, 1)
        results = []
        for attempt, exc in enumerate([ReadTimeout(), ReadTimeout(), ValueError("429"), ReadTimeout(), ReadTimeout(), ReadTimeout()], 1):
            state.outcome = SimpleNamespace(exception=lambda e=exc: e)
            state.attempt_number = attempt
            results.append(_retry._stop_policy(state))
        assert results == [False, False, False, False, False, True]

    def test_connect_timeout_is_not_a_read_timeout(self):
        from pyutilz.llm._retry import _is_read_timeout

        class ConnectTimeout(Exception):  # noqa: N818 -- named like httpx.ConnectTimeout on purpose
            pass

        assert _is_read_timeout(ReadTimeout()) is True
        assert _is_read_timeout(TimeoutError()) is True
        assert _is_read_timeout(ConnectTimeout()) is False
        assert _is_read_timeout(None) is False

    def test_env_parsing(self, monkeypatch, caplog):
        from pyutilz.llm._retry import _env_number

        monkeypatch.setenv("PYUTILZ_TEST_NUM", "12.5")
        assert _env_number("PYUTILZ_TEST_NUM", 7.0) == 12.5
        monkeypatch.setenv("PYUTILZ_TEST_NUM", "abc")
        assert _env_number("PYUTILZ_TEST_NUM", 7.0) == 7.0
        monkeypatch.setenv("PYUTILZ_TEST_NUM", "-1")
        assert _env_number("PYUTILZ_TEST_NUM", 7.0) == 7.0
        monkeypatch.delenv("PYUTILZ_TEST_NUM")
        assert _env_number("PYUTILZ_TEST_NUM", 7.0) == 7.0

    def test_tenacity_actually_stops_on_the_deadline(self, monkeypatch):
        tenacity = pytest.importorskip("tenacity")
        from pyutilz.llm import _retry

        monkeypatch.setattr(_retry, "MAX_CALL_SECONDS", 0.05)
        attempts = []

        @tenacity.retry(wait=tenacity.wait_fixed(0.03), stop=_retry._STOP, reraise=True)
        def flaky() -> None:
            attempts.append(1)
            raise ValueError("transient")

        with pytest.raises(ValueError):
            flaky()
        assert 1 < len(attempts) < 10


# ── CORE-5 / CORE-11 ──────────────────────────────────────────────────────────────────────────────


class TestPricingFallback:
    def test_default_fallback_warns_once_per_model(self, caplog):
        base_mod._PRICING_WARNED.clear()
        with caplog.at_level(logging.WARNING, logger="pyutilz.llm.base"):
            for _ in range(3):
                assert _longest_prefix_pricing("unknown-x", {"a-1": (1.0, 2.0)}, (0.0, 0.0), "P") == (0.0, 0.0)
            _longest_prefix_pricing("unknown-y", {"a-1": (1.0, 2.0)}, (0.0, 0.0), "P")
        lines = [r.getMessage() for r in caplog.records if "not in the pricing table" in r.getMessage()]
        assert len(lines) == 2
        assert "'unknown-x'" in lines[0] and "'unknown-y'" in lines[1]

    def test_prefix_warning_is_once_not_per_estimate(self, caplog):
        base_mod._PRICING_WARNED.clear()
        provider = _Provider(model="fam-model-2027")
        with caplog.at_level(logging.WARNING, logger="pyutilz.llm.base"):
            for _ in range(5):
                assert provider.estimate_cost(1_000_000, 1_000_000) == pytest.approx(7.0)
        assert sum("longest-prefix match" in r.getMessage() for r in caplog.records) == 1

    def test_exact_match_is_silent(self, caplog):
        base_mod._PRICING_WARNED.clear()
        with caplog.at_level(logging.WARNING, logger="pyutilz.llm.base"):
            assert _longest_prefix_pricing("a-1", {"a-1": (1.0, 2.0)}, (0.0, 0.0)) == (1.0, 2.0)
        assert caplog.records == []

    @pytest.mark.parametrize("model", ["a-1", "a-1-2027", "claude-opus-4-6", "grok-5", "zzz", "a-flash-lite-x"])
    def test_both_lookups_share_one_algorithm(self, model):
        table = {"a-1": (1.0, 2.0), "a-flash": (0.3, 0.6), "a-flash-lite": (0.1, 0.4), "claude-opus-4-6-20250610": (5.0, 25.0), "grok-4-fast": (0.2, 0.5)}
        assert _longest_prefix_pricing(model, table, (9.0, 9.0)) == longest_prefix_lookup(model, table, (9.0, 9.0))


# ── CORE-6 / CORE-7 ───────────────────────────────────────────────────────────────────────────────


class TestExtractJson:
    @pytest.mark.parametrize("unit", ["[", '{"a":'], ids=["array", "object"])
    def test_deep_nesting_is_a_json_parsing_error(self, unit):
        text = unit * 100_000
        with pytest.raises(JSONParsingError, match="nesting too deep"):
            LLMProvider.extract_json(text)

    def test_runaway_scan_is_capped(self, monkeypatch):
        calls = []
        real = json.JSONDecoder

        class Counting(real):  # type: ignore[misc,valid-type]
            def raw_decode(self, s: str, idx: int = 0) -> Any:
                calls.append(idx)
                return super().raw_decode(s, idx)

        monkeypatch.setattr(base_mod.json, "JSONDecoder", Counting)
        monkeypatch.setattr(base_mod, "_JSON_SCAN_BUDGET_CHARS", 10_000)
        text = "prose " + '{"b": 1, ' * 5000
        with pytest.raises(JSONParsingError):
            LLMProvider.extract_json(text)
        # Error positions grow by 9 per candidate: the budget is spent after ~47 candidates, not 5000.
        assert 30 < len(calls) < 60

    def test_object_nested_in_an_unterminated_one_is_still_found(self):
        assert LLMProvider.extract_json('Result: {"a": {"b": 2}') == {"b": 2}

    def test_object_after_non_runaway_failures_is_found(self):
        text = "{bad} " * 200 + '{"ok": true}'
        assert LLMProvider.extract_json(text) == {"ok": True}

    def test_failure_log_is_truncated(self, caplog):
        text = "no json here at all " * 50_000
        with caplog.at_level(logging.ERROR, logger="pyutilz.llm.base"), pytest.raises(JSONParsingError):
            LLMProvider.extract_json(text)
        (record,) = [r for r in caplog.records if "Failed to parse JSON" in r.getMessage()]
        assert len(record.getMessage()) < 2500
        assert f"({len(text.strip())} chars)" in record.getMessage()


# ── CORE-8 ────────────────────────────────────────────────────────────────────────────────────────


class TestReasoningBudgetInTimeout:
    class _T:
        model_name = "m"
        max_output_tokens = 1000

        def _get_timeout(self, model: str) -> float:
            return 240.0

    @pytest.fixture
    def t(self) -> Any:
        from pyutilz.llm._timeouts import DerivedTimeoutMixin

        cls = type("T", (self._T, DerivedTimeoutMixin), {})
        return cls()

    def test_separate_reasoning_budget_raises_the_timeout(self, t):
        assert t._timeout_for({"max_tokens": 500}) == 240.0
        assert t._timeout_for({"max_tokens": 500, "reasoning": {"max_tokens": 12_000}}) == pytest.approx(12_500 / 30)
        assert t._timeout_for({"max_tokens": 500, "thinking": {"type": "enabled", "budget_tokens": 15_000}}) == pytest.approx(15_500 / 30)

    def test_reasoning_budget_without_max_tokens(self, t):
        assert t._timeout_for({"reasoning": {"max_tokens": 30_000}}) == pytest.approx(1000.0)

    def test_reasoning_is_still_capped(self, t):
        assert t._timeout_for({"max_tokens": 500, "reasoning": {"max_tokens": 10_000_000}}) == t._max_derived_timeout_s

    @pytest.mark.parametrize("block", [{"effort": "high"}, {"max_tokens": 0}, {"max_tokens": True}, "high", None])
    def test_non_budget_reasoning_is_ignored(self, t, block):
        assert t._timeout_for({"max_tokens": 500, "reasoning": block}) == 240.0


# ── CORE-9 / CORE-13 ──────────────────────────────────────────────────────────────────────────────


class TestGenerateBatch:
    def test_forwards_optional_options_only_when_present(self):
        provider = _Provider(max_concurrent=4)

        async def run() -> list[dict[str, Any]]:
            reqs = [
                {"id": "a", "prompt": "pa", "images": ["data:image/png;base64,AA=="], "thinking": "high"},
                {"id": "b", "prompt": "pb", "json_mode": True, "json_schema": {"name": "s"}},
                {"id": "c", "prompt": "pc"},
            ]
            return [item async for item in provider.generate_batch(reqs)]

        out = asyncio.run(run())
        assert sorted(o["id"] for o in out if "result" in o) == ["a", "b", "c"]
        by_prompt = {c["prompt"]: c for c in provider.calls}
        assert by_prompt["pa"] == {"prompt": "pa", "images": ["data:image/png;base64,AA=="], "thinking": "high"}
        assert by_prompt["pb"] == {"prompt": "pb", "json_mode": True, "json_schema": {"name": "s"}}
        assert by_prompt["pc"] == {"prompt": "pc"}

    def test_unknown_keys_are_reported(self, caplog):
        provider = _Provider(max_concurrent=4)

        async def run() -> None:
            async for _ in provider.generate_batch([{"id": "a", "prompt": "p", "tools": []}]):
                pass

        with caplog.at_level(logging.WARNING, logger="pyutilz.llm.base"):
            asyncio.run(run())
        assert "['tools']" in caplog.text

    def test_batch_requests_do_not_share_an_outer_stream_counter(self):
        provider = _Provider(max_concurrent=4)

        async def run() -> _progress.StreamProgress:
            with _progress.track_stream_progress() as progress:
                async for _ in provider.generate_batch([{"id": str(i), "prompt": "p"} for i in range(3)]):
                    pass
                _progress.note_stream_progress(None, "own")
            return progress

        progress = asyncio.run(run())
        assert progress.answer_chars == 3
        assert progress.answer_tail == "own"


# ── CORE-10 ───────────────────────────────────────────────────────────────────────────────────────


class TestRepetitionLoopMidUnit:
    @pytest.mark.parametrize("cut", ["", "H", "Hm", "Hmm", "Hmm."])
    def test_loop_found_when_the_stream_ends_mid_unit(self, cut):
        from pyutilz.llm.degeneracy import repetition_loop

        unit = repetition_loop("Hmm. " * 400 + cut)
        assert unit is not None
        assert sorted(unit.replace(" ", "")) == sorted("Hmm.")


# ── CORE-12 ───────────────────────────────────────────────────────────────────────────────────────


class TestShutdownClose:
    def _provider_with_client(self) -> tuple[_Provider, list[str]]:
        httpx = pytest.importorskip("httpx")
        closed: list[str] = []
        provider = _Provider()
        provider._client = httpx.AsyncClient(base_url="https://example.test", transport=httpx.MockTransport(lambda r: httpx.Response(200)))

        async def _close() -> None:
            closed.append("close")
            await provider._client.aclose()

        provider._close = _close  # type: ignore[method-assign]
        return provider, closed

    def test_unused_provider_closes_its_own_client_on_the_shutdown_loop(self, monkeypatch):
        from pyutilz.llm import factory

        provider, closed = self._provider_with_client()
        original = provider.__dict__["_client"]
        monkeypatch.setattr(factory, "_provider_cache", {("x",): provider})
        factory._close_cached_providers()
        assert closed == ["close"]
        assert original.is_closed  # the original, not a clone built for the shutdown loop

    def test_clones_are_closed_and_dead_home_loop_is_skipped(self, monkeypatch):
        from pyutilz.llm import factory

        provider, closed = self._provider_with_client()

        async def grab() -> Any:
            return provider._client

        asyncio.run(grab())  # home loop, closed after this
        loop = asyncio.new_event_loop()
        try:
            clone = loop.run_until_complete(grab())
            monkeypatch.setattr(factory, "_provider_cache", {("x",): provider})
            factory._close_cached_providers()
            assert closed == []  # its home loop is closed: its sockets cannot be closed from elsewhere
            assert clone.is_closed
        finally:
            loop.close()


# ── Follow-ups: handoffs from 10_openrouter / 20_providers, base.py split ──────────────────────────


def _compat_provider(**totals: int) -> Any:
    from pyutilz.llm._pricing import Pricing
    from pyutilz.llm.openai_compat import OpenAICompatibleProvider

    class _P(OpenAICompatibleProvider):
        _base_url = "https://example.test"
        _provider_name = "T"

        def _resolve_pricing(self, model: str) -> Pricing:
            return Pricing(2.0, 10.0, cache_hit=0.2, cache_write=2.5)

        def _input_cost_per_1m(self, model: str) -> float:
            return 2.0

        def _output_cost_per_1m(self, model: str) -> float:
            return 10.0

    p = _P(api_key="k", model="m")
    p.total_prompt_tokens = totals.get("prompt", 0)
    p.total_cache_hit_tokens = totals.get("hit", 0)
    p.total_completion_tokens = 0
    p.total_reasoning_tokens = 0
    if "write" in totals:
        p.total_cache_write_tokens = totals["write"]
    return p


class TestCacheWritePricing:
    def test_pricing_has_optional_cache_write(self):
        from pyutilz.llm._pricing import Pricing

        assert Pricing(1.0, 2.0).cache_write is None
        assert Pricing(1.0, 2.0, 0.1, 1.25).cache_write == 1.25

    def test_session_cost_bills_writes_at_the_write_rate(self):
        p = _compat_provider(prompt=3_000_000, hit=1_000_000, write=1_000_000)
        cost = p.get_session_cost()
        # 1M miss at 2.0 + 1M hit at 0.2 + 1M write at 2.5
        assert cost["input_cost_usd"] == pytest.approx(4.7)
        assert cost["cache_miss_tokens"] == 1_000_000

    def test_no_write_tracking_is_unchanged(self):
        p = _compat_provider(prompt=2_000_000, hit=1_000_000)
        assert p.get_session_cost()["input_cost_usd"] == pytest.approx(2.2)

    def test_write_rate_falls_back_to_input(self):
        from pyutilz.llm._pricing import Pricing

        p = _compat_provider()
        p._resolve_pricing = lambda model: Pricing(2.0, 10.0)  # type: ignore[method-assign]
        assert p._cache_write_cost_per_1m("m") == 2.0


class TestStreamInterruptedReexport:
    def test_one_class_under_every_import_path(self):
        import pyutilz.llm as llm
        from pyutilz.llm import _openai_compat_http, exceptions, openai_compat

        cls = exceptions.LLMStreamInterruptedError
        assert llm.LLMStreamInterruptedError is cls
        assert _openai_compat_http.LLMStreamInterruptedError is cls
        assert openai_compat.LLMStreamInterruptedError is cls
        assert issubclass(cls, exceptions.LLMProviderError)
        err = cls("boom", code="server_error", partial_text="ab", retryable=False)
        assert (err.code, err.partial_text, err.retryable) == ("server_error", "ab", False)


class TestCompatCloseClosesEveryLoopClient:
    def test_close_reaches_per_loop_clones(self):
        httpx = pytest.importorskip("httpx")
        p = _compat_provider()
        p._client = httpx.AsyncClient(base_url="https://example.test", transport=httpx.MockTransport(lambda r: httpx.Response(200)))

        async def grab() -> Any:
            return p._client

        original = asyncio.run(grab())
        loop = asyncio.new_event_loop()
        try:
            clone = loop.run_until_complete(grab())
            assert clone is not original
            loop.run_until_complete(p._close())
        finally:
            loop.close()
        assert clone.is_closed and original.is_closed


class TestBaseSplitKeepsImportPaths:
    def test_descriptors_reexported_from_base(self):
        from pyutilz.llm import _descriptors, base

        for name in ("PerCallAttr", "LazySemaphore", "PerLoopHTTPClient", "_NO_REBIND"):
            assert getattr(base, name) is getattr(_descriptors, name)

    def test_base_is_under_the_size_limit(self):
        from pyutilz.llm import base

        assert len(Path(base.__file__).read_text(encoding="utf-8").splitlines()) <= 1000


# ── Integration pass: generate_json(json_schema), route_fingerprint ────────────────────────────────


class TestBaseGenerateJsonSchema:
    def _provider(self) -> Any:
        class _J(_Provider):
            async def generate(self, prompt: str, system: str | None = None, temperature: float | None = 0.7, max_tokens: int = 0, **kwargs: Any) -> str:
                self.calls.append(kwargs)
                return '{"ok": 1}'

        return _J()

    def test_schema_forwarded_when_given(self):
        p = self._provider()
        schema = {"name": "s", "schema": {"type": "object"}}
        assert asyncio.run(p.generate_json("q", json_schema=schema)) == {"ok": 1}
        assert p.calls == [{"json_schema": schema}]

    def test_schema_absent_is_not_forwarded(self):
        p = self._provider()
        asyncio.run(p.generate_json("q"))
        assert p.calls == [{}]


def _or(**kwargs: Any) -> Any:
    pytest.importorskip("httpx")
    from pyutilz.llm.openrouter_provider import OpenRouterProvider

    return OpenRouterProvider(api_key="k", model=kwargs.pop("model", "openai/gpt-4o-mini"), **kwargs)


class TestRouteFingerprint:
    def test_same_config_same_key_and_is_sha256_hex(self):
        a, b = _or(provider_order=("a", "b")), _or(provider_order=("a", "b"))
        fp = a.route_fingerprint()
        assert fp == b.route_fingerprint()
        assert len(fp) == 64 and int(fp, 16) >= 0

    def test_tuple_and_list_digest_equal(self):
        assert _or(provider_quantizations=("fp8", "bf16")).route_fingerprint() == _or(provider_quantizations=["fp8", "bf16"]).route_fingerprint()

    def test_mapping_and_pair_tuple_digest_equal(self):
        assert _or(provider_max_price={"prompt": 1, "completion": 2}).route_fingerprint() == _or(provider_max_price=(("prompt", 1), ("completion", 2))).route_fingerprint()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"model": "anthropic/claude-x"},
            {"provider_quantizations": ("fp4",)},
            {"provider_order": ("b", "a")},
            {"provider_only": ("a",)},
            {"provider_ignore": ("a",)},
            {"provider_sort": "price"},
            {"provider_allow_fallbacks": False},
            {"provider_require_parameters": True},
            {"provider_data_collection": "deny"},
            {"provider_zdr": True},
            {"models_fallback": ("x/y",)},
            {"enable_web_search": True},
            {"transforms": ("middle-out",)},
            {"system_cache_control": True},
            {"anthropic_top_level_cache": True},
            {"extra_body": {"seed": 1}},
            {"pdf_engine": "native"},
        ],
    )
    def test_each_routing_knob_changes_the_key(self, kwargs):
        assert _or(**kwargs).route_fingerprint() != _or().route_fingerprint()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"api_key": "other"},  # pragma: allowlist secret -- a dummy key: the test is that credentials never reach the digest
            {"max_concurrent": 3},
            {"app_name": "x"},
            {"site_url": "https://x"},
            {"retry_routing_404": True},
        ],
    )
    def test_non_routing_kwargs_do_not_change_the_key(self, kwargs):
        from pyutilz.llm.openrouter_provider import OpenRouterProvider

        base_kwargs: dict[str, Any] = {"api_key": "k", "model": "openai/gpt-4o-mini"}
        base_kwargs.update(kwargs)
        assert OpenRouterProvider(**base_kwargs).route_fingerprint() == _or().route_fingerprint()

    def test_api_key_never_enters_the_payload(self):
        p = _or()
        dummy_key = "sk-secret"  # pragma: allowlist secret -- a dummy key, asserted absent from the route payload
        assert dummy_key not in json.dumps(_or(**{}).__class__(api_key=dummy_key)._route_payload(), default=str)
        assert "api_key" not in p._route_payload()

    def test_reassigned_model_is_honoured(self):
        p = _or()
        before = p.route_fingerprint()
        p.model_name = "anthropic/claude-x"
        assert p.route_fingerprint() != before

    def test_every_ctor_kwarg_is_classified(self):
        import inspect

        from pyutilz.llm.openrouter_provider import OpenRouterProvider
        from pyutilz.llm.openrouter_provider._route import _ROUTE_FINGERPRINT_ATTRS, _ROUTE_FINGERPRINT_EXCLUDED

        params = {n for n in inspect.signature(OpenRouterProvider.__init__).parameters if n != "self"}
        included, excluded = set(_ROUTE_FINGERPRINT_ATTRS), set(_ROUTE_FINGERPRINT_EXCLUDED)
        assert not included & excluded
        assert params == included | excluded, f"unclassified: {sorted(params - included - excluded)}; stale: {sorted((included | excluded) - params)}"

    def test_base_fingerprint_digests_provider_and_model(self):
        a, b = _Provider(model="m1"), _Provider(model="m2")
        assert a.route_fingerprint() == _Provider(model="m1").route_fingerprint()
        assert a.route_fingerprint() != b.route_fingerprint()

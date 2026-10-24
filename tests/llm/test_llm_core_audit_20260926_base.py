from __future__ import annotations
from pyutilz.llm.base import PerCallAttr
import gc
from pyutilz.llm.base import LLMProvider

from pyutilz.dev.logginglib import reset_log_throttles
import asyncio
import sys
import logging
from pathlib import Path
from typing import Any
import pytest
from pyutilz.llm import _progress
from pyutilz.llm import base as base_mod
from pyutilz.llm.base import _longest_prefix_pricing, longest_prefix_lookup
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

    @pytest.mark.skipif(sys.version_info < (3, 10), reason="before 3.10 an asyncio.Semaphore binds to a loop when created")
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


class TestPricingFallback:
    def test_default_fallback_warns_once_per_model(self, caplog):
        reset_log_throttles()
        with caplog.at_level(logging.WARNING, logger="pyutilz.llm.base"):
            for _ in range(3):
                assert _longest_prefix_pricing("unknown-x", {"a-1": (1.0, 2.0)}, (0.0, 0.0), "P") == (0.0, 0.0)
            _longest_prefix_pricing("unknown-y", {"a-1": (1.0, 2.0)}, (0.0, 0.0), "P")
        lines = [r.getMessage() for r in caplog.records if "not in the pricing table" in r.getMessage()]
        assert len(lines) == 2
        assert "'unknown-x'" in lines[0] and "'unknown-y'" in lines[1]

    def test_prefix_warning_is_once_not_per_estimate(self, caplog):
        reset_log_throttles()
        provider = _Provider(model="fam-model-2027")
        with caplog.at_level(logging.WARNING, logger="pyutilz.llm.base"):
            for _ in range(5):
                assert provider.estimate_cost(1_000_000, 1_000_000) == pytest.approx(7.0)
        assert sum("longest-prefix match" in r.getMessage() for r in caplog.records) == 1

    def test_exact_match_is_silent(self, caplog):
        reset_log_throttles()
        with caplog.at_level(logging.WARNING, logger="pyutilz.llm.base"):
            assert _longest_prefix_pricing("a-1", {"a-1": (1.0, 2.0)}, (0.0, 0.0)) == (1.0, 2.0)
        assert caplog.records == []

    @pytest.mark.parametrize("model", ["a-1", "a-1-2027", "claude-opus-4-6", "grok-5", "zzz", "a-flash-lite-x"])
    def test_both_lookups_share_one_algorithm(self, model):
        table = {"a-1": (1.0, 2.0), "a-flash": (0.3, 0.6), "a-flash-lite": (0.1, 0.4), "claude-opus-4-6-20250610": (5.0, 25.0), "grok-4-fast": (0.2, 0.5)}
        assert _longest_prefix_pricing(model, table, (9.0, 9.0)) == longest_prefix_lookup(model, table, (9.0, 9.0))


class TestExtractJson:
    @pytest.mark.parametrize("unit", ["[", '{"a":'], ids=["array", "object"])
    def test_deep_nesting_is_a_json_parsing_error(self, unit):
        text = unit * 100_000
        # 3.14's decoder no longer recurses, so it reports a plain decode error there; either way it must be
        # JSONParsingError, never a RecursionError escaping the documented except clause.
        with pytest.raises(JSONParsingError, match=r"nesting too deep|Invalid JSON"):
            LLMProvider.extract_json(text)

    def test_runaway_scan_is_capped(self, monkeypatch):
        calls: list[int] = []
        real = base_mod._object_span_end

        def counting(s: str, idx: int) -> int:
            calls.append(idx)
            return real(s, idx)

        monkeypatch.setattr(base_mod, "_object_span_end", counting)
        monkeypatch.setattr(base_mod, "_JSON_SCAN_BUDGET_CHARS", 10_000)
        text = "prose " + '{"b": 1, ' * 5000
        with pytest.raises(JSONParsingError):
            LLMProvider.extract_json(text)
        # Each unterminated candidate is charged the rest of the text, so the budget ends the scan after a few candidates, not 5000.
        assert 1 <= len(calls) < 60

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


class TestBaseSplitKeepsImportPaths:
    def test_descriptors_reexported_from_base(self):
        from pyutilz.llm import _descriptors, base

        for name in ("PerCallAttr", "LazySemaphore", "PerLoopHTTPClient", "_NO_REBIND"):
            assert getattr(base, name) is getattr(_descriptors, name)

    def test_base_is_under_the_size_limit(self):
        from pyutilz.llm import base

        assert len(Path(base.__file__).read_text(encoding="utf-8").splitlines()) <= 1000


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


class _Holder:
    value = PerCallAttr(lambda: None)


class _Slotted:
    __slots__ = ()
    value = PerCallAttr(lambda: None)


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

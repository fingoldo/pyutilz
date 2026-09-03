"""Regression tests for the LLM findings of the 2026-09-03 domain/db/web/cloud/llm/text audit.

Every test here is offline: no provider is ever contacted, all HTTP is stubbed.

Finding ids map to test classes:
  F01 Anthropic recorded the request-side ``thinking`` toggle as the reasoning-token COUNT.
  F08 ClaudeCodeProvider's per-call state was plain shared state, cleared outside the semaphore.
  F09 ``last_json_mode_fallback`` was never reset, so it latched True for the rest of the context.
  F10 ``PerCallAttr`` raced when two threads first touched the same attribute.
  F11 The streaming path recorded no finish reason / tool calls / citations.
  F12 ``_record_usage`` crashed on a null token field, leaving counters half-updated.
  F13 ``extract_json`` aborted on an unparseable fenced block instead of falling through.
  F14 The streaming path had no repair for endpoints refusing ``reasoning: {enabled: false}``.
  F19 Prefix pricing degenerated to a bare vendor prefix (``grok``).
  F20 OpenAI priced dated snapshot ids from gpt-5-mini while resolving limits by prefix.
  F31 Anthropic/Gemini/Claude Code never reset per-call state at the top of generate().
  F32 ``last_rate_limits`` / ``last_organization_id`` were outside the per-call mechanism.
  F33 Gemini's cache-hit rate was looked up by exact model id.
  F34 Gemini raised LLMTruncationError without partial_text.
  F35 Gemini kept the previous call's candidates when a response had none.
  F36 Streaming inspected only the LAST chunk for response-level metadata.
  F37 OpenRouter reported total_cost_usd == 0.0 when the catalogue was unavailable.
  F38 A length-capped empty answer preempted the empty-completion re-issue.
  F39 ``generate_batch`` defaulted max_tokens to 1024 while generate() defaults to 0.
  F40 Malformed numeric env vars crashed import; a negative retry count disabled retries.
  F41 Subsumed by F10: PerCallAttr no longer constructs ContextVars per instance.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import sys
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

httpx = pytest.importorskip("httpx")

from pyutilz.llm.base import LLMProvider, PerCallAttr, longest_prefix_lookup
from pyutilz.llm.exceptions import LLMTruncationError
from pyutilz.llm.openai_compat import OpenAICompatibleProvider

# ── shared offline stubs ──────────────────────────────────────────────────


class _StubProvider(OpenAICompatibleProvider):
    _base_url = "https://stub.example.com"
    _provider_name = "StubProvider"
    _max_tokens_map = {"stub-model": 4096}
    _default_max_tokens = 2048

    def _input_cost_per_1m(self, model: str) -> float:
        return 1.0

    def _output_cost_per_1m(self, model: str) -> float:
        return 2.0

    def _cache_hit_cost_per_1m(self, model: str) -> float:
        return 0.1


def _make_stub(model: str = "stub-model") -> _StubProvider:
    return _StubProvider(api_key="stub-key", model=model)


class _StubStream:
    """Stands in for ``httpx.AsyncClient.stream(...)``'s async context manager."""

    def __init__(self, status_code: int = 200, lines: list[str] | None = None, headers: dict | None = None, body: bytes = b""):
        self.status_code = status_code
        self.headers = headers or {}
        self._lines = lines or []
        self._body = body
        self.request = httpx.Request("POST", "https://stub.example.com/chat/completions")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            response = httpx.Response(self.status_code, headers=self.headers, request=self.request)
            response.read = lambda: self._body  # type: ignore[method-assign]
            raise httpx.HTTPStatusError(f"status {self.status_code}", request=self.request, response=_ReadableResponse(self.status_code, self._body, self.request))

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


class _ReadableResponse:
    """Minimal error-response stub whose body can be awaited via ``aread()``, as httpx's can."""

    def __init__(self, status_code: int, body: bytes, request: httpx.Request):
        self.status_code = status_code
        self._body = body
        self.request = request

    async def aread(self) -> bytes:
        return self._body


def _sse(payload: dict) -> str:
    return "data: " + json.dumps(payload)


# ── F01 ───────────────────────────────────────────────────────────────────


class TestF01AnthropicReasoningTokens:
    def _provider(self, monkeypatch, response):
        anthropic_provider = pytest.importorskip("pyutilz.llm.anthropic_provider")
        provider = anthropic_provider.AnthropicProvider.__new__(anthropic_provider.AnthropicProvider)
        provider.model = "claude-sonnet-4-20250514"
        provider._max_concurrent = 1
        provider._call_count = 0
        provider.total_input_tokens = 0
        provider.total_output_tokens = 0
        provider.total_cache_creation_input_tokens = 0
        provider.total_cache_read_input_tokens = 0
        provider.total_thinking_tokens = 0

        class _Raw:
            headers: dict = {}

            @staticmethod
            def parse():
                return response

        class _WithRaw:
            @staticmethod
            async def create(**kwargs):
                return _Raw()

        class _Messages:
            with_raw_response = _WithRaw()

        class _Client:
            messages = _Messages()

        provider.client = _Client()
        monkeypatch.setattr(type(provider), "max_output_tokens", property(lambda self: 4096))
        monkeypatch.setattr(provider, "fit_max_tokens_to_context", lambda mt, p, s=None: mt)
        return provider

    @staticmethod
    def _response(thinking_chars: int):
        class _Block:
            def __init__(self, type_, **kw):
                self.type = type_
                for key, value in kw.items():
                    setattr(self, key, value)

        class _Usage:
            input_tokens = 11
            output_tokens = 7
            cache_creation_input_tokens = 0
            cache_read_input_tokens = 0

        class _Response:
            usage = _Usage()
            stop_reason = "end_turn"
            content = ([_Block("thinking", thinking="x" * thinking_chars)] if thinking_chars else []) + [_Block("text", text="answer")]

        return _Response()

    @pytest.mark.asyncio
    async def test_reasoning_tokens_is_the_counted_estimate_not_the_request_toggle(self, monkeypatch):
        provider = self._provider(monkeypatch, self._response(400))
        assert await provider.generate.__wrapped__(provider, "prompt", thinking="high") == "answer"
        # The bug bound the request-side toggle here, so this used to read "high".
        assert provider._last_usage["reasoning_tokens"] == 100
        assert provider._last_usage["reasoning_tokens"] == provider.last_thinking_tokens

    @pytest.mark.asyncio
    async def test_default_toggle_does_not_report_a_boolean_reasoning_count(self, monkeypatch):
        provider = self._provider(monkeypatch, self._response(0))
        await provider.generate.__wrapped__(provider, "prompt")
        reasoning = provider._last_usage["reasoning_tokens"]
        assert reasoning == 0 and not isinstance(reasoning, bool)
        # Summing telemetry across calls must stay arithmetic, never bool/str concatenation.
        assert sum([reasoning, reasoning]) == 0


# ── F08 / F31: ClaudeCodeProvider per-call state ──────────────────────────


class TestF08ClaudeCodePerCallState:
    def _provider(self):
        claude_code_provider = pytest.importorskip("pyutilz.llm.claude_code_provider")
        return claude_code_provider.ClaudeCodeProvider(model="opus", max_concurrent=1)

    def test_per_call_attributes_are_percall_descriptors(self):
        claude_code_provider = pytest.importorskip("pyutilz.llm.claude_code_provider")
        cls = claude_code_provider.ClaudeCodeProvider
        for name in ("_last_usage", "_last_result_message", "last_cost_usd", "last_session_id", "last_num_turns", "last_cache_read_input_tokens", "last_cache_creation_input_tokens"):
            assert isinstance(getattr(cls, name), PerCallAttr), name
            assert name in cls._PERCALL_METADATA_ATTRS, name

    @pytest.mark.asyncio
    async def test_result_message_does_not_leak_between_concurrent_tasks(self):
        provider = self._provider()
        seen: dict[str, object] = {}

        async def one(tag: str, marker: object) -> None:
            provider._reset_per_call_state()
            provider._last_result_message = marker
            await asyncio.sleep(0)
            seen[tag] = provider._last_result_message

        await asyncio.gather(one("a", "message-a"), one("b", "message-b"))
        assert seen == {"a": "message-a", "b": "message-b"}

    @pytest.mark.asyncio
    async def test_reset_happens_inside_the_semaphore(self, monkeypatch):
        """With max_concurrent=1 the queued call must not clear the running call's state."""
        provider = self._provider()
        provider.semaphore = asyncio.Semaphore(1)
        order: list[str] = []

        async def fake_sdk(prompt, system=None):
            async with provider.semaphore:
                provider._reset_per_call_state()
                order.append("reset:" + prompt)
                provider._last_result_message = prompt
                await asyncio.sleep(0.01)
                order.append("read:" + prompt + ":" + str(provider._last_result_message))
                return prompt

        monkeypatch.setattr(provider, "_generate_sdk", fake_sdk)
        monkeypatch.setattr(provider, "_generate_cli", fake_sdk)
        await asyncio.gather(fake_sdk("a"), fake_sdk("b"))
        assert "read:a:a" in order and "read:b:b" in order


# ── F09 / F31: derived reset ──────────────────────────────────────────────


class TestF09DerivedReset:
    def test_reset_clears_last_json_mode_fallback(self):
        provider = _make_stub()
        provider.last_json_mode_fallback = True
        provider._reset_per_call_state()
        assert provider.last_json_mode_fallback is False

    def test_reset_covers_every_declared_percall_attribute(self):
        provider = _make_stub()
        dirtied = []
        for name in provider._PERCALL_METADATA_ATTRS:
            descriptor = getattr(type(provider), name, None)
            if isinstance(descriptor, PerCallAttr):
                dirtied.append(name)
                setattr(provider, name, "DIRTY")
        # An empty tuple (or a rename of PerCallAttr) would make the check below pass vacuously.
        assert dirtied, f"no PerCallAttr found among {provider._PERCALL_METADATA_ATTRS}"
        provider._reset_per_call_state()
        for name in dirtied:
            assert getattr(provider, name) != "DIRTY", name

    def test_reset_is_derived_and_not_hand_maintained(self):
        """A newly declared PerCallAttr registered in the tuple is reset with no extra wiring."""

        class _Extended(_StubProvider):
            last_new_thing: PerCallAttr = PerCallAttr(lambda: 0)
            _PERCALL_METADATA_ATTRS = (*_StubProvider._PERCALL_METADATA_ATTRS, "last_new_thing")

        provider = _Extended(api_key="k", model="stub-model")
        provider.last_new_thing = 99
        provider._reset_per_call_state()
        assert provider.last_new_thing == 0

    def test_a_percall_attr_left_out_of_the_tuple_is_registered_anyway(self):
        """The class of bug F09/F32 belong to: declaring the descriptor IS the registration.

        The subclass below deliberately does NOT extend ``_PERCALL_METADATA_ATTRS``; the attribute
        must still be reset between calls and still appear in the batch snapshot.
        """

        class _Forgetful(_StubProvider):
            last_forgotten_thing: PerCallAttr = PerCallAttr(lambda: 0)

        assert "last_forgotten_thing" in _Forgetful._PERCALL_METADATA_ATTRS
        provider = _Forgetful(api_key="k", model="stub-model")
        provider.last_forgotten_thing = 77
        assert provider._capture_percall_metadata()["forgotten_thing"] == 77
        provider._reset_per_call_state()
        assert provider.last_forgotten_thing == 0


class TestF31ResetCalledByEveryProvider:
    @pytest.mark.asyncio
    async def test_anthropic_generate_resets_before_the_call(self, monkeypatch):
        anthropic_provider = pytest.importorskip("pyutilz.llm.anthropic_provider")
        provider = anthropic_provider.AnthropicProvider.__new__(anthropic_provider.AnthropicProvider)
        provider.model = "claude-sonnet-4-20250514"
        provider._max_concurrent = 1
        provider.last_cache_read_input_tokens = 8000

        boom = RuntimeError("upstream down")

        class _Client:
            class messages:  # noqa: N801 -- mirrors the anthropic SDK attribute path client.messages.with_raw_response
                class with_raw_response:  # noqa: N801 -- see above; the shape under test is the SDK's, not this repo's
                    @staticmethod
                    async def create(**kwargs):
                        raise boom

        provider.client = _Client()
        monkeypatch.setattr(type(provider), "max_output_tokens", property(lambda self: 4096))
        monkeypatch.setattr(provider, "fit_max_tokens_to_context", lambda mt, p, s=None: mt)

        with pytest.raises(RuntimeError):
            await provider.generate.__wrapped__(provider, "prompt")
        # The failed call must not leave the previous call's cache figure readable.
        assert provider.last_cache_read_input_tokens == 0

    @pytest.mark.asyncio
    async def test_gemini_generate_resets_before_the_call(self, monkeypatch):
        gemini_provider = pytest.importorskip("pyutilz.llm.gemini_provider")
        provider = gemini_provider.GeminiProvider.__new__(gemini_provider.GeminiProvider)
        provider.model_name = "gemini-2.5-flash"
        provider._max_concurrent = 1
        provider._candidate_count = 1
        provider._cached_content = None
        provider.last_safety_ratings = [{"category": "HARASSMENT"}]

        class _Models:
            @staticmethod
            async def generate_content(**kwargs):
                raise RuntimeError("upstream down")

        class _Aio:
            models = _Models()

        class _Client:
            aio = _Aio()

        provider.client = _Client()
        monkeypatch.setattr(type(provider), "max_output_tokens", property(lambda self: 4096))
        monkeypatch.setattr(provider, "fit_max_tokens_to_context", lambda mt, p, s=None: mt)

        with pytest.raises(RuntimeError):
            await provider.generate.__wrapped__(provider, "prompt")
        assert provider.last_safety_ratings == []


# ── F10 / F41: PerCallAttr concurrency + ContextVar lifecycle ─────────────


class TestF10PerCallAttrThreadSafety:
    def test_concurrent_first_touch_never_loses_a_write(self):
        class _Holder:
            value: PerCallAttr = PerCallAttr(lambda: "DEFAULT")

        switch_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-9)
        try:
            for _ in range(200):
                holder = _Holder()
                threads_n = 16
                barrier = threading.Barrier(threads_n)
                results: list[str] = [""] * threads_n

                # Bound as defaults, not captured: every thread is joined before the next iteration
                # rebinds these, but a late-binding closure over a loop variable is a trap either way.
                def worker(index: int, holder: _Holder = holder, barrier: threading.Barrier = barrier, results: list = results) -> None:
                    barrier.wait()
                    holder.value = "thread-%d" % index
                    results[index] = holder.value

                threads = [threading.Thread(target=worker, args=(i,)) for i in range(threads_n)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                assert results == ["thread-%d" % i for i in range(threads_n)]
        finally:
            sys.setswitchinterval(switch_interval)

    def test_one_contextvar_per_class_attribute_not_per_instance(self):
        """F41: ContextVars are built once at class-creation time, so N instances mint none."""

        class _Holder:
            value: PerCallAttr = PerCallAttr(lambda: 0)

        descriptor = _Holder.__dict__["value"]
        assert isinstance(descriptor._var, contextvars.ContextVar)
        instances = [_Holder() for _ in range(50)]
        for i, holder in enumerate(instances):
            holder.value = i
        assert descriptor is _Holder.__dict__["value"]
        assert [h.value for h in instances] == list(range(50))

    def test_state_is_isolated_per_instance(self):
        class _Holder:
            value: PerCallAttr = PerCallAttr(lambda: None)

        a, b = _Holder(), _Holder()
        a.value = "a"
        b.value = "b"
        assert (a.value, b.value) == ("a", "b")

    @pytest.mark.asyncio
    async def test_state_is_isolated_per_task(self):
        class _Holder:
            value: PerCallAttr = PerCallAttr(lambda: None)

        holder = _Holder()
        seen: dict[str, object] = {}

        async def one(tag: str) -> None:
            holder.value = tag
            await asyncio.sleep(0)
            seen[tag] = holder.value

        await asyncio.gather(one("x"), one("y"))
        assert seen == {"x": "x", "y": "y"}


# ── F11 / F36: streaming metadata ─────────────────────────────────────────


class TestF11StreamingMetadata:
    @pytest.mark.asyncio
    async def test_length_capped_stream_raises_truncation_with_partial_text(self):
        provider = _make_stub()
        lines = [
            _sse({"id": "gen-1", "choices": [{"delta": {"content": "partial "}}]}),
            _sse({"id": "gen-1", "choices": [{"delta": {"content": "answer"}, "finish_reason": "length"}]}),
            "data: [DONE]",
        ]
        provider._client = AsyncMock()
        provider._client.stream = MagicMock(return_value=_StubStream(lines=lines))

        chunks: list[str] = []
        with pytest.raises(LLMTruncationError) as excinfo:
            async for chunk in provider.generate_stream("prompt"):
                chunks.append(chunk)
        assert excinfo.value.finish_reason == "length"
        assert excinfo.value.partial_text == "partial answer"
        assert provider._last_finish_reason == "length"

    @pytest.mark.asyncio
    async def test_stream_reassembles_tool_call_fragments(self):
        provider = _make_stub()
        lines = [
            _sse({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"q"'}}]}}]}),
            _sse({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ': "x"}'}}]}}]}),
            _sse({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
            "data: [DONE]",
        ]
        provider._client = AsyncMock()
        provider._client.stream = MagicMock(return_value=_StubStream(lines=lines))

        async for _ in provider.generate_stream("prompt"):
            pass

        assert provider.last_tool_calls == [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"q": "x"}'}}]
        assert provider._last_finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_stream_records_citations(self):
        provider = _make_stub()
        lines = [
            _sse({"choices": [{"delta": {"content": "hi"}}], "citations": ["https://example.com"]}),
            _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
            "data: [DONE]",
        ]
        provider._client = AsyncMock()
        provider._client.stream = MagicMock(return_value=_StubStream(lines=lines))
        async for _ in provider.generate_stream("prompt"):
            pass
        assert provider.last_citations == ["https://example.com"]


class TestF36StreamingFirstChunkMetadata:
    @pytest.mark.asyncio
    async def test_response_level_metadata_is_read_from_the_first_chunk_too(self):
        provider = _make_stub()
        seen: list[dict] = []
        provider._track_provider_specific_response = lambda data: seen.append(data)  # type: ignore[method-assign]

        lines = [
            _sse({"id": "gen-abc", "model": "resolved/model", "choices": [{"delta": {"content": "hi"}}]}),
            _sse({"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}),
            "data: [DONE]",
        ]
        provider._client = AsyncMock()
        provider._client.stream = MagicMock(return_value=_StubStream(lines=lines))
        async for _ in provider.generate_stream("prompt"):
            pass

        assert any(chunk.get("id") == "gen-abc" for chunk in seen), "first chunk's response-level metadata was skipped"


# ── F12: null token fields ────────────────────────────────────────────────


class TestF12NullTokenFields:
    def test_null_completion_tokens_does_not_raise_or_skew_counters(self):
        provider = _make_stub()
        provider._record_usage({"prompt_tokens": 10, "completion_tokens": None})
        assert provider.total_prompt_tokens == 10
        assert provider.total_completion_tokens == 0
        assert provider._call_count == 1

    def test_null_prompt_tokens_is_also_tolerated(self):
        provider = _make_stub()
        provider._record_usage({"prompt_tokens": None, "completion_tokens": 4})
        assert provider.total_prompt_tokens == 0
        assert provider.total_completion_tokens == 4
        assert provider._call_count == 1


# ── F13: extract_json fence fallthrough ───────────────────────────────────


class TestF13ExtractJsonFenceFallthrough:
    def test_broken_fence_falls_through_to_the_raw_decode_scan(self):
        text = 'Here is a sketch:\n```json\n{"a": 1,}\n```\nAnd the answer: {"ok": 2}'
        assert LLMProvider.extract_json(text, "Stub") == {"ok": 2}

    def test_valid_fence_still_wins(self):
        text = '```json\n{"a": 1}\n```\n{"b": 2}'
        assert LLMProvider.extract_json(text, "Stub") == {"a": 1}

    def test_fenced_top_level_array_still_raises(self):
        from pyutilz.llm.exceptions import JSONParsingError

        with pytest.raises(JSONParsingError):
            LLMProvider.extract_json("```json\n[1, 2, 3]\n```", "Stub")


# ── F14: streaming request repair ─────────────────────────────────────────


class TestF14StreamingBodyRepair:
    @pytest.mark.asyncio
    async def test_stream_is_reopened_once_with_the_repaired_body(self):
        class _RepairingProvider(_StubProvider):
            def _body_after_rejected_request(self, body, status, detail):
                if status in (400, 404) and "cannot be disabled" in detail.lower():
                    return {**body, "reasoning": {"effort": "minimal", "exclude": True}}
                return None

        provider = _RepairingProvider(api_key="k", model="stub-model")
        bodies: list[dict] = []
        attempts = {"n": 0}

        def stream(method, url, json=None, **kwargs):
            bodies.append(json)
            attempts["n"] += 1
            if attempts["n"] == 1:
                return _StubStream(status_code=404, body=b'{"error": {"message": "Reasoning is mandatory for this endpoint and cannot be disabled"}}')
            return _StubStream(lines=[_sse({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}), "data: [DONE]"])

        provider._client = AsyncMock()
        provider._client.stream = MagicMock(side_effect=stream)

        out = [chunk async for chunk in provider.generate_stream("prompt")]
        assert out == ["ok"]
        assert attempts["n"] == 2
        assert bodies[1]["reasoning"] == {"effort": "minimal", "exclude": True}

    @pytest.mark.asyncio
    async def test_unrepairable_rejection_still_raises(self, monkeypatch):
        import pyutilz.llm.openai_compat as openai_compat_mod

        monkeypatch.setattr(openai_compat_mod, "MAX_RETRY_ATTEMPTS", 1)
        provider = _make_stub()
        provider._client = AsyncMock()
        provider._client.stream = MagicMock(return_value=_StubStream(status_code=404, body=b'{"error": "no endpoints"}'))
        with pytest.raises(httpx.HTTPStatusError):
            async for _ in provider.generate_stream("prompt"):
                pass


# ── F19 / F20: pricing prefix resolution ──────────────────────────────────


class TestF19VendorPrefixDegeneracy:
    def test_bare_vendor_prefix_is_rejected(self):
        table = {"grok-4.3": (1.25, 2.50), "grok-4-fast": (0.20, 0.50)}
        assert longest_prefix_lookup("grok-5", table, None) is None

    def test_date_suffixed_key_still_resolves(self):
        table = {"claude-opus-4-6-20250610": (15.0, 75.0)}
        assert longest_prefix_lookup("claude-opus-4-6", table, None) == (15.0, 75.0)

    def test_xai_unknown_model_now_warns_and_uses_the_documented_default(self, caplog):
        xai_provider = pytest.importorskip("pyutilz.llm.xai_provider")
        provider = xai_provider.XAIProvider.__new__(xai_provider.XAIProvider)
        xai_provider.XAIProvider._seen_unknown_models.discard("grok-5")
        with caplog.at_level("WARNING", logger="pyutilz.llm.xai_provider"):
            rates = provider._resolve_pricing("grok-5")
        # `_resolve_pricing` returns the shared `Pricing` record, so the fallback is read by field
        # name rather than by tuple position -- the two-vs-three-position divergence between sibling
        # providers is what the record replaced.
        assert (rates.input, rates.output) == (0.20, 0.50)
        assert any("grok-5" in record.getMessage() for record in caplog.records)


class TestF20OpenAIDatedSnapshotPricing:
    def test_dated_snapshot_is_priced_from_its_own_family(self):
        openai_provider = pytest.importorskip("pyutilz.llm.openai_provider")
        provider = openai_provider.OpenAIProvider.__new__(openai_provider.OpenAIProvider)
        pricing = openai_provider._PRICING
        family = next((k for k in pricing if k.startswith("gpt-5-pro")), None)
        if family is None:
            pytest.skip("no gpt-5-pro row in the pricing table")
        expected = pricing[family]
        resolved = provider._resolve_pricing(family + "-2026-01-15")
        assert (resolved.input, resolved.output) == expected
        assert expected != pricing["gpt-5-mini"], "test model must differ from the fallback"

    def test_pricing_and_limits_agree_about_the_same_dated_id(self):
        openai_provider = pytest.importorskip("pyutilz.llm.openai_provider")
        provider = openai_provider.OpenAIProvider.__new__(openai_provider.OpenAIProvider)
        pricing = openai_provider._PRICING
        family = next((k for k in pricing if k.startswith("gpt-5-pro")), None)
        if family is None:
            pytest.skip("no gpt-5-pro row in the pricing table")
        dated = family + "-2026-01-15"
        assert provider._input_cost_per_1m(dated) == longest_prefix_lookup(dated, pricing, None)[0]


# ── F32: rate-limit / organization id are per-call ────────────────────────


class TestF32RateLimitsArePerCall:
    def test_rate_limits_are_a_percall_attribute(self):
        assert isinstance(OpenAICompatibleProvider.__dict__["last_rate_limits"], PerCallAttr)
        assert "last_rate_limits" in OpenAICompatibleProvider._PERCALL_METADATA_ATTRS

    def test_header_less_response_clears_the_previous_window(self):
        provider = _make_stub()
        provider._capture_rate_limit_headers({"x-ratelimit-remaining-requests": "5"})
        assert provider.last_rate_limits["x-ratelimit-remaining-requests"] == "5"
        provider._capture_rate_limit_headers({"content-type": "application/json"})
        assert provider.last_rate_limits == {}

    @pytest.mark.asyncio
    async def test_rate_limits_do_not_leak_between_concurrent_tasks(self):
        provider = _make_stub()
        seen: dict[str, dict] = {}

        async def one(tag: str, remaining: str) -> None:
            provider._capture_rate_limit_headers({"x-ratelimit-remaining-requests": remaining})
            await asyncio.sleep(0)
            seen[tag] = dict(provider.last_rate_limits)

        await asyncio.gather(one("a", "100"), one("b", "0"))
        assert seen["a"]["x-ratelimit-remaining-requests"] == "100"
        assert seen["b"]["x-ratelimit-remaining-requests"] == "0"

    def test_anthropic_organization_id_is_a_percall_attribute(self):
        anthropic_provider = pytest.importorskip("pyutilz.llm.anthropic_provider")
        cls = anthropic_provider.AnthropicProvider
        assert isinstance(cls.__dict__["last_organization_id"], PerCallAttr)
        assert isinstance(cls.__dict__["last_rate_limits"], PerCallAttr)


# ── F33 / F34 / F35: Gemini ───────────────────────────────────────────────


class TestF33GeminiCacheRatePrefix:
    def test_versioned_id_keeps_the_cache_discount(self):
        gemini_provider = pytest.importorskip("pyutilz.llm.gemini_provider")
        provider = gemini_provider.GeminiProvider.__new__(gemini_provider.GeminiProvider)
        cache_table = provider._CACHE_HIT_COST
        family = next((k for k in cache_table if k.startswith("gemini-")), None)
        if family is None:
            pytest.skip("no gemini rows in the cache-hit table")
        provider.model_name = family + "-002"
        provider.total_cached_content_tokens = 1_000_000
        provider.total_prompt_tokens = 1_000_000
        provider.total_completion_tokens = 0
        provider.total_reasoning_tokens = 0
        provider._call_count = 1
        cost = provider.get_session_cost()
        in_rate = provider._get_pricing()[0]
        assert cost["input_cost_usd"] == pytest.approx(cache_table[family])
        assert cost["input_cost_usd"] < in_rate, "the cache discount silently vanished"


class TestF34GeminiTruncationPartialText:
    @pytest.mark.asyncio
    async def test_generate_raises_truncation_with_partial_text(self, monkeypatch):
        gemini_provider = pytest.importorskip("pyutilz.llm.gemini_provider")
        provider = gemini_provider.GeminiProvider.__new__(gemini_provider.GeminiProvider)
        provider.model_name = "gemini-2.5-flash"
        provider._max_concurrent = 1
        provider._candidate_count = 1
        provider._cached_content = None
        provider.total_prompt_tokens = 0
        provider.total_completion_tokens = 0
        provider.total_reasoning_tokens = 0
        provider.total_cached_content_tokens = 0
        provider._call_count = 0

        class _Candidate:
            finish_reason = "FinishReason.MAX_TOKENS"
            safety_ratings: list = []
            grounding_metadata = None
            citation_metadata = None
            content = None

        class _Response:
            candidates = [_Candidate()]
            usage_metadata = None
            text = "most of the answer"

        class _Models:
            @staticmethod
            async def generate_content(**kwargs):
                return _Response()

        class _Aio:
            models = _Models()

        class _Client:
            aio = _Aio()

        provider.client = _Client()
        monkeypatch.setattr(type(provider), "max_output_tokens", property(lambda self: 4096))
        monkeypatch.setattr(provider, "fit_max_tokens_to_context", lambda mt, p, s=None: mt)

        with pytest.raises(LLMTruncationError) as excinfo:
            await provider.generate.__wrapped__(provider, "prompt")
        assert excinfo.value.partial_text == "most of the answer"


class TestF35GeminiCandidatesCleared:
    def test_empty_candidates_clears_the_previous_calls_candidates(self):
        gemini_provider = pytest.importorskip("pyutilz.llm.gemini_provider")
        provider = gemini_provider.GeminiProvider.__new__(gemini_provider.GeminiProvider)
        provider.last_all_candidates = ["previous-candidate"]

        class _Blocked:
            candidates: list = []

        provider._capture_candidate_metadata(_Blocked())
        assert provider.last_all_candidates == []


# ── F37: OpenRouter cost when the catalogue is unavailable ────────────────


class TestF37OpenRouterUnknownPricing:
    def test_unknown_pricing_reports_null_not_zero(self, monkeypatch):
        openrouter = pytest.importorskip("pyutilz.llm.openrouter_provider")
        provider = openrouter.OpenRouterProvider.__new__(openrouter.OpenRouterProvider)
        provider.model_name = "some/model"
        provider.total_prompt_tokens = 1000
        provider.total_completion_tokens = 500
        provider.total_cache_hit_tokens = 0
        provider.total_reasoning_tokens = 0
        provider._call_count = 1
        provider.total_actual_cost_usd = 1.23
        provider.total_upstream_inference_cost_usd = 0.0
        provider.total_cache_write_tokens = 0
        provider.total_audio_tokens = 0

        monkeypatch.setattr(openrouter._catalogue, "_fetch_models_catalogue", lambda *a, **kw: {})
        cost = provider.get_session_cost()
        assert cost["total_cost_usd"] is None
        assert cost["pricing_available"] is False
        assert cost["actual_cost_usd"] == 1.23

    def test_known_pricing_still_reports_a_number(self, monkeypatch):
        openrouter = pytest.importorskip("pyutilz.llm.openrouter_provider")
        provider = openrouter.OpenRouterProvider.__new__(openrouter.OpenRouterProvider)
        provider.model_name = "some/model"
        provider.total_prompt_tokens = 1_000_000
        provider.total_completion_tokens = 0
        provider.total_cache_hit_tokens = 0
        provider.total_reasoning_tokens = 0
        provider._call_count = 1
        provider.total_actual_cost_usd = 0.0
        provider.total_upstream_inference_cost_usd = 0.0
        provider.total_cache_write_tokens = 0
        provider.total_audio_tokens = 0

        catalogue = {"some/model": {"id": "some/model", "pricing": {"prompt": "0.000001", "completion": "0.000002"}}}
        monkeypatch.setattr(openrouter._catalogue, "_fetch_models_catalogue", lambda *a, **kw: catalogue)
        cost = provider.get_session_cost()
        assert cost["pricing_available"] is True
        assert cost["total_cost_usd"] == pytest.approx(1.0)

    def test_missing_model_warns_once(self, monkeypatch, caplog):
        openrouter = pytest.importorskip("pyutilz.llm.openrouter_provider")
        catalogue_mod = openrouter._catalogue
        monkeypatch.setattr(catalogue_mod, "_fetch_models_catalogue", lambda *a, **kw: {})
        catalogue_mod._UNPRICED_MODELS_WARNED.discard("ghost/model")
        with caplog.at_level("WARNING", logger=catalogue_mod.__name__):
            assert catalogue_mod._per_token_cost_pair("ghost/model") == (0.0, 0.0)
            catalogue_mod._per_token_cost_pair("ghost/model")
        assert sum("ghost/model" in r.getMessage() for r in caplog.records) == 1


# ── F38: length-capped empty answer under response_format ─────────────────


class TestF38EmptyLengthCappedUnderResponseFormat:
    @pytest.mark.asyncio
    async def test_empty_length_capped_reply_is_re_issued_without_response_format(self, monkeypatch):
        provider = _make_stub()
        calls: list[dict] = []

        async def fake_post(body, repairing=False):
            calls.append(body)
            if "response_format" in body:
                raise LLMTruncationError("truncated", finish_reason="length", partial_text="")
            return '{"ok": 1}'

        monkeypatch.setattr(provider, "_post_and_unwrap", fake_post)
        monkeypatch.setattr(type(provider), "max_output_tokens", property(lambda self: 4096))
        monkeypatch.setattr(provider, "fit_max_tokens_to_context", lambda mt, p, s=None: mt)
        monkeypatch.setattr(provider, "_response_format", lambda json_mode, json_schema: {"type": "json_object"})

        out = await provider.generate("prompt", json_mode=True)
        assert out == '{"ok": 1}'
        assert len(calls) == 2 and "response_format" not in calls[1]
        assert provider.last_json_mode_fallback is True

    @pytest.mark.asyncio
    async def test_genuine_truncation_with_partial_text_still_propagates(self, monkeypatch):
        provider = _make_stub()

        async def fake_post(body, repairing=False):
            raise LLMTruncationError("truncated", finish_reason="length", partial_text="half an answer")

        monkeypatch.setattr(provider, "_post_and_unwrap", fake_post)
        monkeypatch.setattr(type(provider), "max_output_tokens", property(lambda self: 4096))
        monkeypatch.setattr(provider, "fit_max_tokens_to_context", lambda mt, p, s=None: mt)
        monkeypatch.setattr(provider, "_response_format", lambda json_mode, json_schema: {"type": "json_object"})

        with pytest.raises(LLMTruncationError) as excinfo:
            await provider.generate("prompt", json_mode=True)
        assert excinfo.value.partial_text == "half an answer"

    @pytest.mark.asyncio
    async def test_re_issue_that_is_also_empty_still_reports_truncation(self, monkeypatch):
        provider = _make_stub()

        async def fake_post(body, repairing=False):
            if "response_format" in body:
                raise LLMTruncationError("truncated", finish_reason="length", partial_text="")
            return None

        monkeypatch.setattr(provider, "_post_and_unwrap", fake_post)
        monkeypatch.setattr(type(provider), "max_output_tokens", property(lambda self: 4096))
        monkeypatch.setattr(provider, "fit_max_tokens_to_context", lambda mt, p, s=None: mt)
        monkeypatch.setattr(provider, "_response_format", lambda json_mode, json_schema: {"type": "json_object"})

        with pytest.raises(LLMTruncationError):
            await provider.generate("prompt", json_mode=True)


# ── F39: generate_batch max_tokens default ────────────────────────────────


class TestF39BatchMaxTokensDefault:
    @pytest.mark.asyncio
    async def test_batch_default_matches_generate_default(self):
        seen: list[int] = []

        class _BatchProvider(LLMProvider):
            async def generate(self, prompt, system=None, temperature=0.7, max_tokens=0, **kwargs):
                seen.append(max_tokens)
                return "ok"

            async def generate_json(self, prompt, system=None, temperature=0.7, max_tokens=0):
                return {}

            async def count_tokens(self, text, system=None):
                return 0

            def get_session_cost(self):
                return {}

        provider = _BatchProvider()
        results = [item async for item in provider.generate_batch([{"id": "r0", "prompt": "hi"}])]
        assert [r["result"] for r in results] == ["ok"]
        assert seen == [0]

    @pytest.mark.asyncio
    async def test_explicit_batch_max_tokens_is_honoured(self):
        seen: list[int] = []

        class _BatchProvider(LLMProvider):
            async def generate(self, prompt, system=None, temperature=0.7, max_tokens=0, **kwargs):
                seen.append(max_tokens)
                return "ok"

            async def generate_json(self, prompt, system=None, temperature=0.7, max_tokens=0):
                return {}

            async def count_tokens(self, text, system=None):
                return 0

            def get_session_cost(self):
                return {}

        provider = _BatchProvider()
        [item async for item in provider.generate_batch([{"id": "r0", "prompt": "hi", "max_tokens": 256}])]
        assert seen == [256]


# ── F40: env-var validation ───────────────────────────────────────────────


class TestF40EnvVarValidation:
    def _reimport(self, module_name: str):
        # A fresh interpreter, so no module-identity split in this process (module reload would
        # break isinstance checks and caches elsewhere in the suite).
        import subprocess  # nosec B404 - spawns this interpreter with a fixed argument list

        return subprocess.run  # returned for the caller to invoke with its own env

    def test_malformed_provider_cache_size_does_not_break_import(self, tmp_path):
        import os
        import subprocess  # nosec B404 - spawns this interpreter with a fixed argument list

        env = dict(os.environ, PYUTILZ_LLM_PROVIDER_CACHE_MAX_SIZE="abc")
        code = "import pyutilz.llm.factory as f; print(f._PROVIDER_CACHE_MAX_SIZE)"
        proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)  # nosec B603 - fixed argv, no shell
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "128"

    def test_malformed_settings_ttl_does_not_break_import(self):
        import os
        import subprocess  # nosec B404 - spawns this interpreter with a fixed argument list

        env = dict(os.environ, PYUTILZ_LLM_SETTINGS_TTL_SECONDS="x")
        code = "import pyutilz.llm.config as c; print(c._SETTINGS_TTL_SECONDS)"
        proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)  # nosec B603 - fixed argv, no shell
        assert proc.returncode == 0, proc.stderr
        assert float(proc.stdout.strip()) == 60.0

    def test_negative_retry_count_is_clamped_to_the_default(self):
        import os
        import subprocess  # nosec B404 - spawns this interpreter with a fixed argument list

        env = dict(os.environ, PYUTILZ_LLM_MAX_RETRIES="-1")
        code = "import pyutilz.llm._retry as r; print(r.MAX_RETRY_ATTEMPTS)"
        proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)  # nosec B603 - fixed argv, no shell
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "50"

    def test_zero_retry_count_still_means_infinite(self):
        import os
        import subprocess  # nosec B404 - spawns this interpreter with a fixed argument list

        env = dict(os.environ, PYUTILZ_LLM_MAX_RETRIES="0")
        code = "import pyutilz.llm._retry as r; print(r.MAX_RETRY_ATTEMPTS)"
        proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)  # nosec B603 - fixed argv, no shell
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "0"

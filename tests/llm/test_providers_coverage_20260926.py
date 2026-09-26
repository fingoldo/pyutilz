"""Regression tests for audits/implemented/2026-09-26/20_providers_coverage.md (PROV-*): native provider feature coverage.

Every test drives the provider's own request building or response handling with the network stubbed out, and asserts
on the value that was wrong before the fix (the request field, the price, the raised type), so reverting a fix fails.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from pyutilz.llm import anthropic_provider as ap
from pyutilz.llm._claude_models import claude_model_spec, resolve_claude_model
from pyutilz.llm._thinking import gemini_thinking_config
from pyutilz.llm.anthropic_provider import AnthropicProvider, anthropic_thinking_request
from pyutilz.llm.exceptions import LLMProviderError, LLMRefusalError, LLMSafetyBlockError, LLMTruncationError

# ─── Anthropic ─────────────────────────────────────────────────────────────────────────────────────


def _message(*blocks: Any, stop: str = "end_turn", usage: Any = None, stop_details: Any = None) -> SimpleNamespace:
    usage = usage or SimpleNamespace(input_tokens=10, output_tokens=5, cache_creation_input_tokens=0, cache_read_input_tokens=0)
    return SimpleNamespace(content=list(blocks), stop_reason=stop, usage=usage, stop_details=stop_details)


def _text(t: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=t)


def _anthropic(model: str = "claude-opus-5-5", responses: list[Any] | None = None, **init: Any) -> tuple[AnthropicProvider, list[dict]]:
    p = AnthropicProvider(api_key="sk-fake-test-key-not-real", model=model, **init)
    sent: list[dict] = []
    queue = list(responses or [_message(_text("ok"))])

    async def _create(**kwargs: Any) -> Any:
        sent.append(json.loads(json.dumps(kwargs, default=str)))
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        raw = MagicMock()
        raw.parse.return_value = item
        raw.headers = {}
        return raw

    p.client = MagicMock()
    p.client.messages.with_raw_response.create = AsyncMock(side_effect=_create)
    p.fit_max_tokens_to_context = lambda mt, prompt, system=None: mt  # type: ignore[method-assign]
    return p, sent


class _Http400Error(Exception):
    status_code = 400


class TestProv1AdaptiveThinking:
    def test_current_model_gets_adaptive_plus_effort_not_a_budget(self) -> None:
        assert anthropic_thinking_request("high", 20_000, model="claude-opus-5-5") == {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
        }

    def test_minimal_maps_to_claudes_lowest_effort(self) -> None:
        assert anthropic_thinking_request("minimal", 20_000, model="claude-sonnet-5")["output_config"] == {"effort": "low"}

    def test_true_is_adaptive_at_the_model_default(self) -> None:
        assert anthropic_thinking_request(True, 20_000, model="claude-fable-5-1") == {"thinking": {"type": "adaptive"}}

    def test_budget_only_model_keeps_the_budget_and_gets_no_effort(self) -> None:
        assert anthropic_thinking_request("high", 20_000, model="claude-sonnet-4-5-20250929") == {
            "thinking": {"type": "enabled", "budget_tokens": 8192},
        }

    def test_opus_4_5_takes_budget_and_effort(self) -> None:
        assert anthropic_thinking_request("low", 20_000, model="claude-opus-4-5-20251101") == {
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "output_config": {"effort": "low"},
        }

    @pytest.mark.asyncio
    async def test_generate_sends_adaptive_on_a_current_model(self) -> None:
        p, sent = _anthropic("claude-opus-5-5")
        await p.generate("q", max_tokens=8000, thinking="medium")
        assert sent[0]["thinking"] == {"type": "adaptive"}
        assert sent[0]["output_config"] == {"effort": "medium"}

    @pytest.mark.asyncio
    async def test_documented_400_flips_the_mode_and_is_remembered(self) -> None:
        model = "claude-sonnet-4-5-20250929"
        err = _Http400Error('"thinking.type.enabled" is not supported for this model. Use "thinking.type.adaptive"')
        p, sent = _anthropic(model, [err, _message(_text("a")), _message(_text("b"))])
        try:
            assert await p.generate("q", max_tokens=8000, thinking="high") == "a"
            assert sent[0]["thinking"]["type"] == "enabled"
            assert sent[1]["thinking"] == {"type": "adaptive"}
            await p.generate("q", max_tokens=8000, thinking="high")
            assert sent[2]["thinking"] == {"type": "adaptive"}, "learned once, not paid for on every call"
        finally:
            ap._LEARNED_THINKING_MODE.pop(model, None)

    @pytest.mark.asyncio
    async def test_an_unrelated_400_is_not_swallowed(self) -> None:
        p, _sent = _anthropic("claude-opus-5-5", [_Http400Error("prompt is too long")])
        with pytest.raises(_Http400Error):
            await p.generate("q", max_tokens=8000, thinking="high")


class TestProv2Prov3ModelTable:
    @pytest.mark.parametrize(
        ("model", "price", "max_out", "ctx"),
        [
            ("claude-fable-5-1", (10.0, 50.0), 128_000, 1_000_000),
            ("claude-opus-5-5", (4.0, 20.0), 128_000, 1_000_000),
            ("claude-sonnet-5", (2.0, 10.0), 128_000, 1_000_000),
            ("claude-sonnet-4-6", (3.0, 15.0), 128_000, 1_000_000),
            ("claude-haiku-4-5-20251001", (1.0, 5.0), 64_000, 200_000),
            ("claude-opus-4-5-20251101", (5.0, 25.0), 64_000, 200_000),
            ("claude-3-haiku-20240307", (0.25, 1.25), 4_096, 200_000),
            ("claude-3-5-haiku-20241022", (0.80, 4.0), 8_192, 200_000),
        ],
    )
    def test_real_ids_resolve(self, model: str, price: tuple, max_out: int, ctx: int) -> None:
        p = AnthropicProvider.__new__(AnthropicProvider)
        p.model = model
        assert p._get_pricing() == price
        assert p.max_output_tokens == max_out
        assert p.context_window == ctx

    def test_a_future_model_is_not_priced_from_a_legacy_prefix(self, caplog: pytest.LogCaptureFixture) -> None:
        assert resolve_claude_model("claude-opus-4-9") is None, "claude-opus-4 is a prefix of it, but not the same model"
        with caplog.at_level(logging.WARNING):
            spec = claude_model_spec("claude-opus-4-9-20270101")
        assert (spec.input_per_1m, spec.output_per_1m) == (4.0, 20.0)
        assert any("not in the Claude model table" in r.getMessage() for r in caplog.records)

    def test_platform_ids_resolve(self) -> None:
        assert resolve_claude_model("anthropic.claude-opus-5") == resolve_claude_model("claude-opus-5")
        assert resolve_claude_model("claude-opus-4-5@20251101") == resolve_claude_model("claude-opus-4-5")
        assert resolve_claude_model("anthropic.claude-opus-5")[0] == "claude-opus-5"  # type: ignore[index]


class TestProv4AllTextBlocks:
    @pytest.mark.asyncio
    async def test_every_text_block_is_returned(self) -> None:
        thinking = SimpleNamespace(type="thinking", thinking="...")
        p, _ = _anthropic(responses=[_message(_text("first half, "), thinking, _text("second half"))])
        assert await p.generate("q", max_tokens=100) == "first half, second half"


class TestProv5StopReasons:
    @pytest.mark.asyncio
    async def test_refusal_raises_a_typed_refusal_with_its_details(self) -> None:
        details = SimpleNamespace(model_dump=lambda: {"type": "refusal_by_policy", "category": "cyber"})
        p, _ = _anthropic(responses=[_message(stop="refusal", stop_details=details)])
        with pytest.raises(LLMRefusalError) as exc:
            await p.generate("q", max_tokens=100)
        assert exc.value.details["stop_details"] == {"type": "refusal_by_policy", "category": "cyber"}
        assert p.last_stop_details["category"] == "cyber"

    @pytest.mark.asyncio
    async def test_context_window_exceeded_is_a_truncation(self) -> None:
        p, _ = _anthropic(responses=[_message(_text("partial"), stop="model_context_window_exceeded")])
        with pytest.raises(LLMTruncationError) as exc:
            await p.generate("q", max_tokens=100)
        assert exc.value.partial_text == "partial"


class TestProv6ReportedThinkingTokens:
    @pytest.mark.asyncio
    async def test_api_reported_count_wins_over_the_estimate(self) -> None:
        usage = SimpleNamespace(
            input_tokens=10, output_tokens=900, cache_creation_input_tokens=0, cache_read_input_tokens=0,
            output_tokens_details=SimpleNamespace(thinking_tokens=777),
        )
        p, _ = _anthropic(responses=[_message(SimpleNamespace(type="thinking", thinking="short summary"), _text("x"), usage=usage)])
        await p.generate("q", max_tokens=100)
        assert p.last_thinking_tokens == 777
        assert p.last_thinking_tokens_estimated is False

    @pytest.mark.asyncio
    async def test_estimate_only_without_the_field(self) -> None:
        p, _ = _anthropic(responses=[_message(SimpleNamespace(type="thinking", thinking="x" * 40), _text("x"))])
        await p.generate("q", max_tokens=100)
        assert p.last_thinking_tokens == 10
        assert p.last_thinking_tokens_estimated is True


class TestProv7CacheRates:
    def _cost(self, model: str, **totals: int) -> dict:
        p = AnthropicProvider.__new__(AnthropicProvider)
        p.model = model
        p._call_count = 1
        p.total_input_tokens = totals.get("inp", 0)
        p.total_output_tokens = 0
        p.total_thinking_tokens = 0
        p.total_cache_creation_input_tokens = totals.get("write", 0)
        p.total_cache_creation_1h_input_tokens = totals.get("write_1h", 0)
        p.total_cache_read_input_tokens = totals.get("read", 0)
        return p.get_session_cost()

    @pytest.mark.parametrize(("model", "expected"), [("claude-opus-5-5", 0.20), ("claude-fable-5-1", 0.25), ("claude-sonnet-5", 0.20)])
    def test_cache_read_uses_the_models_own_multiplier(self, model: str, expected: float) -> None:
        assert self._cost(model, read=1_000_000)["input_cost_usd"] == pytest.approx(expected)

    def test_one_hour_writes_bill_double(self) -> None:
        cost = self._cost("claude-sonnet-5", write=1_000_000, write_1h=1_000_000)
        assert cost["input_cost_usd"] == pytest.approx(4.0)
        assert self._cost("claude-sonnet-5", write=1_000_000)["input_cost_usd"] == pytest.approx(2.5)

    @pytest.mark.asyncio
    async def test_1h_ttl_is_requested_and_its_writes_are_tracked(self) -> None:
        usage = SimpleNamespace(
            input_tokens=1, output_tokens=1, cache_creation_input_tokens=500, cache_read_input_tokens=0,
            cache_creation=SimpleNamespace(ephemeral_1h_input_tokens=500, ephemeral_5m_input_tokens=0),
        )
        p, sent = _anthropic(responses=[_message(_text("x"), usage=usage)], cache_ttl="1h")
        await p.generate("q", system="sys", max_tokens=100)
        assert sent[0]["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        assert p.total_cache_creation_1h_input_tokens == 500


class TestProv8CountTokens:
    @pytest.mark.asyncio
    async def test_permanent_4xx_is_raised_not_counted(self) -> None:
        p = AnthropicProvider(api_key="sk-fake")
        err = Exception("model not found")
        err.status_code = 404  # type: ignore[attr-defined]
        p.client = MagicMock()
        p.client.messages.count_tokens = AsyncMock(side_effect=err)
        with pytest.raises(Exception, match="model not found"):
            await p.count_tokens("hello")

    @pytest.mark.asyncio
    async def test_transient_failure_falls_back_with_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        p = AnthropicProvider(api_key="sk-fake")
        AnthropicProvider._count_tokens_fallback_warned = False
        err = Exception("overloaded")
        err.status_code = 529  # type: ignore[attr-defined]
        p.client = MagicMock()
        p.client.messages.count_tokens = AsyncMock(side_effect=err)
        with caplog.at_level(logging.WARNING):
            assert await p.count_tokens("hello world") > 0
        assert any("falling back" in r.getMessage() for r in caplog.records)


class TestProv9DefaultModel:
    def test_default_is_a_current_model(self) -> None:
        assert AnthropicProvider(api_key="sk-fake").model == "claude-sonnet-5"  # pragma: allowlist secret -- dummy key
        assert resolve_claude_model("claude-sonnet-5") is not None


class TestAnthropicNotUsedFeatures:
    @pytest.mark.asyncio
    async def test_structured_output_sends_the_bare_schema(self) -> None:
        p, sent = _anthropic("claude-opus-5-5", [_message(_text('{"a": 1}'))])
        schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": False}
        out = await p.generate_json("q", json_schema={"name": "x", "strict": True, "schema": schema})
        assert out == {"a": 1}
        assert sent[0]["output_config"]["format"] == {"type": "json_schema", "schema": schema}

    def test_structured_output_support_follows_the_table(self) -> None:
        p = AnthropicProvider.__new__(AnthropicProvider)
        p.model = "claude-sonnet-4-20250514"
        assert p.supports_json_schema() is False
        p.model = "claude-haiku-4-5-20251001"
        assert p.supports_json_schema() is True

    @pytest.mark.asyncio
    async def test_large_max_tokens_is_streamed(self) -> None:
        p, sent = _anthropic("claude-opus-5-5")
        final = _message(_text("long answer"))
        stream = MagicMock()
        stream.get_final_message = AsyncMock(return_value=final)
        stream.response = SimpleNamespace(headers={"anthropic-ratelimit-requests-limit": "50"})
        manager = MagicMock()
        manager.__aenter__ = AsyncMock(return_value=stream)
        manager.__aexit__ = AsyncMock(return_value=False)
        p.client.messages.stream = MagicMock(return_value=manager)
        assert await p.generate("q", max_tokens=100_000) == "long answer"
        assert p.client.messages.stream.call_args.kwargs["max_tokens"] == 100_000
        assert sent == [], "a 100K request must not go through the buffered create()"
        assert p.last_rate_limits == {"anthropic-ratelimit-requests-limit": "50"}

    @pytest.mark.asyncio
    async def test_message_batch_returns_in_order_at_half_price(self) -> None:
        p = AnthropicProvider(api_key="sk-fake", model="claude-sonnet-5")
        usage = SimpleNamespace(input_tokens=1_000_000, output_tokens=0, cache_creation_input_tokens=0, cache_read_input_tokens=0)
        ok = SimpleNamespace(custom_id="b", result=SimpleNamespace(type="succeeded", message=_message(_text("B"), usage=usage)))
        bad = SimpleNamespace(custom_id="a", result=SimpleNamespace(type="errored", error="invalid_request"))

        async def _results(_bid: str) -> Any:
            async def gen() -> Any:
                for e in (ok, bad):
                    yield e
            return gen()

        p.client = MagicMock()
        p.client.messages.batches.create = AsyncMock(return_value=SimpleNamespace(id="batch_1", processing_status="in_progress"))
        p.client.messages.batches.retrieve = AsyncMock(return_value=SimpleNamespace(id="batch_1", processing_status="ended"))
        p.client.messages.batches.results = _results
        with patch("asyncio.sleep", new=AsyncMock()):
            out = await p.generate_message_batch([{"id": "a", "prompt": "x"}, {"id": "b", "prompt": "y"}], poll_interval=1)
        assert [o["id"] for o in out] == ["a", "b"]
        assert "error" in out[0] and out[1]["result"] == "B"
        assert p.total_batch_cost_usd == pytest.approx(1.0), "1M Sonnet 5 input tokens at $2, halved"
        params = p.client.messages.batches.create.call_args.kwargs["requests"][0]
        assert params["custom_id"] == "a" and params["params"]["model"] == "claude-sonnet-5"


# ─── OpenAI ────────────────────────────────────────────────────────────────────────────────────────


def _openai(model: str, handler: Any) -> Any:
    from pyutilz.llm.openai_provider import OpenAIProvider

    p = OpenAIProvider(api_key="sk-fake", model=model)
    p._client = httpx.AsyncClient(base_url="https://api.openai.com/v1", transport=httpx.MockTransport(handler))
    p.fit_max_tokens_to_context = lambda mt, prompt, system=None: mt
    return p


def _chat_ok(body: dict) -> httpx.Response:
    return httpx.Response(200, json={
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    })


class TestProv10Prov12OpenAIBody:
    @pytest.mark.asyncio
    async def test_reasoning_model_gets_max_completion_tokens_and_no_temperature(self) -> None:
        seen: list[dict] = []

        def handler(req: httpx.Request) -> httpx.Response:
            seen.append(json.loads(req.content))
            return _chat_ok(seen[-1])

        p = _openai("o3", handler)
        assert await p.generate("q", max_tokens=500, temperature=0.2) == "ok"
        assert seen[0]["max_completion_tokens"] == 500
        assert "max_tokens" not in seen[0]
        assert "temperature" not in seen[0]

    @pytest.mark.asyncio
    async def test_non_reasoning_model_keeps_its_temperature(self) -> None:
        seen: list[dict] = []

        def handler(req: httpx.Request) -> httpx.Response:
            seen.append(json.loads(req.content))
            return _chat_ok(seen[-1])

        p = _openai("gpt-4o", handler)
        await p.generate("q", max_tokens=50, temperature=0.2)
        assert seen[0]["temperature"] == 0.2 and seen[0]["max_completion_tokens"] == 50

    @pytest.mark.asyncio
    async def test_a_temperature_400_is_repaired_and_learned(self) -> None:
        from pyutilz.llm import openai_provider as op

        seen: list[dict] = []

        def handler(req: httpx.Request) -> httpx.Response:
            body = json.loads(req.content)
            seen.append(body)
            if "temperature" in body:
                return httpx.Response(400, json={"error": {"message": "Unsupported value: 'temperature' does not support 0.2"}})
            return _chat_ok(body)

        p = _openai("gpt-4.1-mini", handler)
        try:
            assert await p.generate("q", max_tokens=50, temperature=0.2) == "ok"
            assert "temperature" in seen[0] and "temperature" not in seen[1]
            assert "gpt-4.1-mini" in op._MODELS_REJECTING_TEMPERATURE
        finally:
            op._MODELS_REJECTING_TEMPERATURE.discard("gpt-4.1-mini")


class TestProv11OpenAIReasoningEffort:
    @pytest.mark.parametrize(
        ("model", "thinking", "expected"),
        [
            ("gpt-5-mini", True, {"reasoning_effort": "medium"}),
            ("gpt-5-mini", "high", {"reasoning_effort": "high"}),
            ("gpt-5-mini", False, {"reasoning_effort": "minimal"}),
            ("gpt-5.1", False, {"reasoning_effort": "none"}),
            ("o3", False, {"reasoning_effort": "low"}),
            ("gpt-6-astra", False, {"reasoning_effort": "low"}),
            ("gpt-4o", "high", None),
            ("gpt-5-chat-latest", "high", None),
        ],
    )
    def test_mapping(self, model: str, thinking: Any, expected: Any) -> None:
        from pyutilz.llm.openai_provider import OpenAIProvider

        p = OpenAIProvider.__new__(OpenAIProvider)
        p.model_name = model
        assert p._thinking_request_field(thinking) == expected


class TestProv13Prov14OpenAIAccounting:
    @pytest.mark.asyncio
    async def test_check_account_limits_returns_the_captured_headers(self) -> None:
        from pyutilz.llm.openai_provider import OpenAIProvider

        p = OpenAIProvider(api_key="sk-fake")
        p.last_rate_limits = {"x-ratelimit-remaining-requests": "99"}
        out = await p.check_account_limits()
        assert out["remaining_requests"] == "99"

    def test_dated_snapshot_prices_by_prefix_without_the_unknown_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        from pyutilz.llm.openai_provider import OpenAIProvider

        p = OpenAIProvider.__new__(OpenAIProvider)
        OpenAIProvider._seen_unknown_models.discard("gpt-5-pro-2026-01-15")
        with caplog.at_level(logging.WARNING):
            assert p._input_cost_per_1m("gpt-5-pro-2026-01-15") == 15.0
            assert p._cache_hit_cost_per_1m("gpt-5-pro-2026-01-15") == 15.0
        assert not any("unknown" in r.getMessage() for r in caplog.records)

    def test_genuine_miss_still_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        from pyutilz.llm.openai_provider import OpenAIProvider

        p = OpenAIProvider.__new__(OpenAIProvider)
        OpenAIProvider._seen_unknown_models.discard("zzz-unknown")
        with caplog.at_level(logging.WARNING):
            p._input_cost_per_1m("zzz-unknown")
        assert any("unknown" in r.getMessage() for r in caplog.records)


# ─── DeepSeek ──────────────────────────────────────────────────────────────────────────────────────


def _deepseek(model: str = "deepseek-flash", **kw: Any) -> Any:
    from pyutilz.llm.deepseek_provider import DeepSeekProvider

    return DeepSeekProvider(api_key="sk-fake", model=model, **kw)  # pragma: allowlist secret -- dummy key


class TestProv16Prov18DeepSeekFlash:
    def test_flash_limits(self) -> None:
        p = _deepseek()
        assert (p.max_output_tokens, p.context_window) == (384_000, 1_000_000)
        assert p._resolve_pricing("deepseek-flash").input == 0.30

    def test_unknown_model_defaults_to_the_current_generation(self) -> None:
        p = _deepseek("deepseek-v5-something")
        assert (p.max_output_tokens, p.context_window) == (384_000, 1_000_000)

    def test_flash_takes_the_thinking_toggle(self) -> None:
        assert _deepseek()._thinking_request_field(True) == {"thinking": {"type": "enabled"}}
        assert _deepseek("deepseek-chat")._thinking_request_field(True) is None


class TestProv17DeepSeekOffPeak:
    @pytest.mark.parametrize(
        ("when", "mult"),
        [
            (datetime(2026, 9, 28, 2, 30, tzinfo=timezone.utc), 1.0),  # Monday, inside 01-04
            (datetime(2026, 9, 28, 5, 0, tzinfo=timezone.utc), 0.5),  # Monday, the 04-06 gap
            (datetime(2026, 9, 28, 9, 59, tzinfo=timezone.utc), 1.0),  # Monday, inside 06-10
            (datetime(2026, 9, 28, 10, 0, tzinfo=timezone.utc), 0.5),
            (datetime(2026, 9, 26, 2, 30, tzinfo=timezone.utc), 0.5),  # Saturday
        ],
    )
    def test_multiplier(self, when: datetime, mult: float) -> None:
        from pyutilz.llm.deepseek_provider import deepseek_price_multiplier

        assert deepseek_price_multiplier(when) == mult

    def test_off_peak_call_is_billed_at_half(self) -> None:
        p = _deepseek()
        usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000, "prompt_cache_hit_tokens": 0}
        with patch("pyutilz.llm.deepseek_provider.deepseek_price_multiplier", return_value=0.5):
            p._record_usage(usage)
        cost = p.get_session_cost()
        assert cost["total_cost_usd"] == pytest.approx((0.30 + 1.20) / 2)
        assert cost["offpeak_discount_usd"] == pytest.approx((0.30 + 1.20) / 2)

    def test_peak_call_is_billed_in_full(self) -> None:
        p = _deepseek()
        with patch("pyutilz.llm.deepseek_provider.deepseek_price_multiplier", return_value=1.0):
            p._record_usage({"prompt_tokens": 1_000_000, "completion_tokens": 0})
        assert p.get_session_cost()["total_cost_usd"] == pytest.approx(0.30)


class TestProv19DeepSeek402:
    def test_fails_fast_by_default(self) -> None:
        resp = httpx.Response(402, request=httpx.Request("POST", "https://api.deepseek.com/chat/completions"))
        with pytest.raises(LLMProviderError, match="insufficient balance"):
            _deepseek()._handle_special_status(resp)

    def test_opt_in_keeps_waiting(self, caplog: pytest.LogCaptureFixture) -> None:
        resp = httpx.Response(402, request=httpx.Request("POST", "https://api.deepseek.com/chat/completions"))
        with caplog.at_level(logging.WARNING):
            _deepseek(wait_on_insufficient_balance=True)._handle_special_status(resp)
        assert any("retrying indefinitely" in r.getMessage() for r in caplog.records)


# ─── xAI ───────────────────────────────────────────────────────────────────────────────────────────


def _xai(model: str = "grok-4.7", handler: Any = None, **kw: Any) -> Any:
    from pyutilz.llm.xai_provider import XAIProvider

    p = XAIProvider(api_key="xai-fake", model=model, **kw)
    if handler is not None:
        p._client = httpx.AsyncClient(base_url="https://api.x.ai/v1", transport=httpx.MockTransport(handler))
    p.fit_max_tokens_to_context = lambda mt, prompt, system=None: mt
    return p


class TestProv20XaiSearchTools:
    @pytest.mark.asyncio
    async def test_live_search_uses_the_responses_tools_not_search_parameters(self) -> None:
        seen: list[tuple[str, dict]] = []

        def handler(req: httpx.Request) -> httpx.Response:
            seen.append((req.url.path, json.loads(req.content)))
            return httpx.Response(200, json={
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "answer", "annotations": []}]}],
                "citations": ["https://example.com"],
                "usage": {"input_tokens": 10, "output_tokens": 4},
            })

        p = _xai(handler=handler, live_search=True)
        assert await p.generate("news?", max_tokens=100) == "answer"
        path, body = seen[0]
        assert path.endswith("/responses")
        assert {"type": "web_search"} in body["tools"] and "search_parameters" not in body
        assert p.last_citations == ["https://example.com"]
        assert p.total_prompt_tokens == 10

    @pytest.mark.asyncio
    async def test_without_live_search_it_stays_on_chat_completions(self) -> None:
        seen: list[str] = []

        def handler(req: httpx.Request) -> httpx.Response:
            seen.append(req.url.path)
            return _chat_ok({})

        await _xai(handler=handler).generate("q", max_tokens=10)
        assert seen[0].endswith("/chat/completions")


class TestProv21XaiEffortAndTier:
    @pytest.mark.parametrize(
        ("model", "thinking", "expected"),
        [
            ("grok-4.7", "high", {"reasoning_effort": "high"}),
            ("grok-4.7", "max", {"reasoning_effort": "xhigh"}),
            ("grok-4.5", "xhigh", {"reasoning_effort": "high"}),
            ("grok-4.6", False, {"reasoning_effort": "low"}),
            ("grok-4.6", "minimal", {"reasoning_effort": "low"}),
            ("grok-4-1-fast-reasoning", "high", None),
        ],
    )
    def test_effort(self, model: str, thinking: Any, expected: Any) -> None:
        assert _xai(model)._thinking_request_field(thinking) == expected

    def test_new_models_are_priced(self) -> None:
        p = _xai("grok-4.7")
        assert (p._input_cost_per_1m("grok-4.7"), p._output_cost_per_1m("grok-4.7")) == (2.0, 6.0)
        assert p.context_window == 500_000

    def test_long_context_request_bills_the_doubled_tier(self) -> None:
        p = _xai("grok-4.3")
        p._record_usage({"prompt_tokens": 250_000, "completion_tokens": 0})
        assert p.get_session_cost()["total_cost_usd"] == pytest.approx(2 * 250_000 * 1.25 / 1_000_000)
        q = _xai("grok-4.3")
        q._record_usage({"prompt_tokens": 150_000, "completion_tokens": 0})
        assert q.get_session_cost()["total_cost_usd"] == pytest.approx(150_000 * 1.25 / 1_000_000)


# ─── Gemini ────────────────────────────────────────────────────────────────────────────────────────


class TestProv22GeminiThinkingLevel:
    @pytest.mark.parametrize(
        ("thinking", "model", "expected"),
        [
            ("high", "gemini-3-flash-preview", {"thinking_level": "high"}),
            (False, "gemini-3-flash-preview", {"thinking_level": "minimal"}),
            (False, "gemini-3.1-pro-preview", {"thinking_level": "low"}),
            ("minimal", "gemini-3.1-pro-preview", {"thinking_level": "low"}),
            ("max", "gemini-3.8-flash", {"thinking_level": "high"}),
            (True, "gemini-3.5-flash", {"thinking_level": "medium"}),
            ("low", "gemini-2.5-flash", {"thinking_budget": 2048}),
            (False, "gemini-2.5-flash", {"thinking_budget": 0}),
            (False, "gemini-2.5-pro", None),
            (None, "gemini-3-flash-preview", None),
        ],
    )
    def test_config(self, thinking: Any, model: str, expected: Any) -> None:
        assert gemini_thinking_config(thinking, model) == expected


def _gemini(model: str, response: Any) -> tuple[Any, dict]:
    gp = pytest.importorskip("pyutilz.llm.gemini_provider")
    pytest.importorskip("google.genai")
    p = gp.GeminiProvider(api_key="g-fake", model=model)
    captured: dict = {}

    async def generate_content(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return response

    p.client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))
    p.fit_max_tokens_to_context = lambda mt, prompt, system=None: mt
    return p, captured


def _gem_response(text: str = "ok", finish: str = "STOP", candidates: bool = True, feedback: Any = None, prompt_tokens: int = 5) -> Any:
    cand = SimpleNamespace(finish_reason=finish, safety_ratings=[], grounding_metadata=None, citation_metadata=None, content=None)
    um = SimpleNamespace(prompt_token_count=prompt_tokens, candidates_token_count=3, thoughts_token_count=0, cached_content_token_count=0)
    return SimpleNamespace(candidates=[cand] if candidates else [], text=text, usage_metadata=um, prompt_feedback=feedback)


class TestProv22Prov24GeminiRequest:
    @pytest.mark.asyncio
    async def test_gemini3_request_carries_thinking_level(self) -> None:
        p, captured = _gemini("gemini-3-flash-preview", _gem_response())
        await p.generate.__wrapped__(p, "q", max_tokens=100, thinking="low")
        tc = captured["config"].thinking_config
        assert str(tc.thinking_level).upper().endswith("LOW") and tc.thinking_budget is None

    @pytest.mark.asyncio
    async def test_json_schema_is_sent_as_response_json_schema(self) -> None:
        p, captured = _gemini("gemini-2.5-flash", _gem_response('{"a": 1}'))
        schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
        assert await p.generate_json("q", json_schema={"name": "n", "strict": True, "schema": schema}) == {"a": 1}
        cfg = captured["config"]
        assert cfg.response_json_schema == schema and cfg.response_mime_type == "application/json"
        assert p.supports_json_schema() is True


class TestProv23GeminiTransport:
    def test_httpx_transport_errors_are_retryable(self) -> None:
        gp = pytest.importorskip("pyutilz.llm.gemini_provider")
        assert httpx.ConnectError in gp._TRANSIENT_TRANSPORT_ERRORS or any(issubclass(httpx.ConnectError, t) for t in gp._TRANSIENT_TRANSPORT_ERRORS)
        assert not issubclass(httpx.ConnectError, OSError), "the reason OSError alone did not cover it"

    def test_client_has_an_explicit_timeout(self) -> None:
        gp = pytest.importorskip("pyutilz.llm.gemini_provider")
        pytest.importorskip("google.genai")
        with patch.object(gp.genai, "Client") as client_cls:
            gp.GeminiProvider(api_key="g-fake")  # pragma: allowlist secret -- dummy key
        options = client_cls.call_args.kwargs["http_options"]
        assert options.timeout == 600_000


class TestProv25GeminiBlockReasons:
    @pytest.mark.asyncio
    async def test_prompt_block_reason_is_reported(self) -> None:
        feedback = SimpleNamespace(block_reason="PROHIBITED_CONTENT", block_reason_message="nope")
        p, _ = _gemini("gemini-2.5-flash", _gem_response(candidates=False, feedback=feedback))
        with pytest.raises(LLMSafetyBlockError) as exc:
            await p.generate.__wrapped__(p, "q", max_tokens=100)
        assert exc.value.details["block_reason"] == "PROHIBITED_CONTENT"

    @pytest.mark.asyncio
    async def test_recitation_finish_is_a_safety_block_with_details(self) -> None:
        p, _ = _gemini("gemini-2.5-flash", _gem_response(finish="FinishReason.RECITATION"))
        with pytest.raises(LLMSafetyBlockError, match="safety filter"):
            await p.generate.__wrapped__(p, "q", max_tokens=100)

    @pytest.mark.asyncio
    async def test_pro_long_context_call_bills_the_upper_tier(self) -> None:
        p, _ = _gemini("gemini-2.5-pro", _gem_response(prompt_tokens=300_000))
        await p.generate.__wrapped__(p, "q", max_tokens=100)
        cost = p.get_session_cost()
        # 300K input at $2.50 (not $1.25) + 3 output tokens at $15 (not $10).
        assert cost["total_cost_usd"] == pytest.approx((300_000 * 2.50 + 3 * 15.0) / 1_000_000)


# ─── Claude Code ───────────────────────────────────────────────────────────────────────────────────


class TestProv26Prov27ClaudeCodeResult:
    @pytest.mark.asyncio
    async def test_dict_usage_from_the_sdk_is_read(self) -> None:
        from pyutilz.llm.claude_code_provider import ClaudeCodeProvider

        p = ClaudeCodeProvider()

        async def fake_sdk(prompt: str, system: Any = None, **_: Any) -> str:
            p._last_result_message = SimpleNamespace(
                usage={"input_tokens": 11, "output_tokens": 7, "cache_creation_input_tokens": 3, "cache_read_input_tokens": 5,
                       "output_tokens_details": {"thinking_tokens": 4}},
                total_cost_usd=0.5, session_id="s", num_turns=1,
            )
            return "ok"

        with patch.object(p, "_generate_sdk", side_effect=fake_sdk), patch("pyutilz.llm.claude_code_provider._HAS_SDK", True):
            assert await p.generate("hi") == "ok"
        assert p._last_usage["input_tokens"] == 11 and p._last_usage["output_tokens"] == 7
        assert p._last_usage["cache_read_input_tokens"] == 5 and p._last_usage["reasoning_tokens"] == 4

    def test_cli_is_error_result_is_an_error_not_an_answer(self) -> None:
        import queue

        from pyutilz.llm.claude_code_cli import _consume_cli_stream

        q: queue.Queue = queue.Queue()
        q.put(json.dumps({"type": "result", "subtype": "success", "is_error": True, "result": "API Error: 529 overloaded"}))
        q.put(None)
        text, error, timed_out, _event = _consume_cli_stream(q, 5.0)
        assert text is None and error == "API Error: 529 overloaded" and timed_out is False

    @pytest.mark.asyncio
    async def test_sdk_is_error_result_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pyutilz.llm import claude_code_provider as ccp

        class FakeResult:
            def __init__(self) -> None:
                self.is_error = True
                self.result = "You've hit your limit"
                self.subtype = "success"
                self.usage = None

        seen: dict = {}

        class FakeOptions:
            def __init__(self, **kw: Any) -> None:
                self.extra_args = kw.get("extra_args")
                seen.update(kw)

        async def fake_query(prompt: str, options: Any) -> Any:
            yield FakeResult()

        monkeypatch.setattr(ccp, "ResultMessage", FakeResult, raising=False)
        monkeypatch.setattr(ccp, "TextBlock", type("TB", (), {}), raising=False)
        monkeypatch.setattr(ccp, "ClaudeCodeOptions", FakeOptions, raising=False)
        monkeypatch.setattr(ccp, "cc_query", fake_query, raising=False)
        p = ccp.ClaudeCodeProvider()
        with pytest.raises(RuntimeError, match="hit your limit") as exc:
            await p._generate_sdk("hi", effort="high")
        assert ccp._is_rate_limit_error(exc.value), "the error text must still reach the rate-limit wait"
        assert seen["extra_args"]["effort"] == "high"


class TestProv28Prov29ClaudeCodeThinkingAndLimits:
    def test_adaptive_alias_sends_effort_not_a_budget(self) -> None:
        from pyutilz.llm.claude_code_provider import ClaudeCodeProvider

        p = ClaudeCodeProvider(model="opus")
        assert p._thinking_transport_kwargs("minimal") == {"effort": "low"}
        assert p._thinking_transport_kwargs(False) == {"thinking_tokens": 0}
        assert p._thinking_transport_kwargs(None) == {}

    def test_budget_model_keeps_the_budget(self) -> None:
        from pyutilz.llm.claude_code_provider import ClaudeCodeProvider

        assert ClaudeCodeProvider(model="haiku")._thinking_transport_kwargs("high") == {"thinking_tokens": 8192}

    @pytest.mark.asyncio
    async def test_cli_argv_carries_effort(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pyutilz.llm import claude_code_provider as ccp

        seen: dict = {}

        def fake_run_cli(cmd: list, *args: Any) -> tuple:
            seen["cmd"] = cmd
            return 0, "ok", "", None

        monkeypatch.setattr(ccp, "run_cli", fake_run_cli)
        monkeypatch.setattr(ccp, "_find_claude_executable", lambda: "claude")
        p = ccp.ClaudeCodeProvider()
        assert await p._generate_cli("hi", effort="xhigh") == "ok"
        cmd = seen["cmd"]
        assert cmd[cmd.index("--effort") + 1] == "xhigh"
        assert cmd[-3:] == ["--tools", "", "-"], "--tools stays last before the stdin marker"

    @pytest.mark.parametrize(("model", "out", "ctx"), [("opus", 128_000, 1_000_000), ("haiku", 64_000, 200_000), ("claude-opus-5-5", 128_000, 1_000_000)])
    def test_limits_follow_the_model(self, model: str, out: int, ctx: int) -> None:
        from pyutilz.llm.claude_code_provider import ClaudeCodeProvider

        p = ClaudeCodeProvider(model=model)
        assert (p.max_output_tokens, p.context_window) == (out, ctx)


class TestProv30Prov31ClaudeCodeMisc:
    @pytest.mark.asyncio
    async def test_batch_does_not_invent_max_tokens(self) -> None:
        from pyutilz.llm.claude_code_provider import ClaudeCodeProvider

        p = ClaudeCodeProvider()
        calls: list[dict] = []

        async def fake_generate(**kw: Any) -> str:
            calls.append(kw)
            return "r"

        with patch.object(p, "generate", side_effect=fake_generate):
            out = [r async for r in p.generate_batch([{"id": 1, "prompt": "a"}, {"id": 2, "prompt": "b", "max_tokens": 9}])]
        assert [o["result"] for o in out] == ["r", "r"]
        assert "max_tokens" not in calls[0] and "temperature" not in calls[0]
        assert calls[1]["max_tokens"] == 9

    @pytest.mark.asyncio
    async def test_count_tokens_closes_its_client_and_resolves_the_alias(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import anthropic

        from pyutilz.llm.claude_code_provider import ClaudeCodeProvider

        state: dict = {}

        class FakeClient:
            def __init__(self, **_: Any) -> None:
                self.messages = SimpleNamespace(count_tokens=self._count)

            async def _count(self, **kw: Any) -> Any:
                state["model"] = kw["model"]
                return SimpleNamespace(input_tokens=42)

            async def __aenter__(self) -> FakeClient:
                return self

            async def __aexit__(self, *exc: Any) -> None:
                state["closed"] = True

        monkeypatch.setattr(anthropic, "AsyncAnthropic", FakeClient)
        assert await ClaudeCodeProvider(model="opus").count_tokens("hi") == 42
        assert state == {"model": "claude-opus-5", "closed": True}

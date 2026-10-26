from __future__ import annotations
from pyutilz.core import jsonlib
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from pyutilz.llm import anthropic_provider as ap
from pyutilz.llm.anthropic_provider import AnthropicProvider, anthropic_thinking_request


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
        sent.append(jsonlib.loads(jsonlib.dumps(kwargs, default=str)))
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


class TestAnthropicNotUsedFeatures:
    @pytest.mark.asyncio
    async def test_structured_output_sends_the_bare_schema(self) -> None:
        p, sent = _anthropic("claude-opus-5-5", [_message(_text('{"a": 1}'))])
        schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": False}
        out = await p.generate_json("q", json_schema={"name": "x", "strict": True, "schema": schema}, max_tokens=4096)
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


class TestProv4AllTextBlocks:
    @pytest.mark.asyncio
    async def test_every_text_block_is_returned(self) -> None:
        thinking = SimpleNamespace(type="thinking", thinking="...")
        p, _ = _anthropic(responses=[_message(_text("first half, "), thinking, _text("second half"))])
        assert await p.generate("q", max_tokens=100) == "first half, second half"

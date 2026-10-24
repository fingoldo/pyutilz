from __future__ import annotations
from pyutilz.core import jsonlib
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock
import httpx
import pytest
from pyutilz.llm.anthropic_provider import AnthropicProvider
from pyutilz.llm.exceptions import LLMProviderError, LLMRefusalError, LLMSafetyBlockError, LLMTruncationError


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


def _deepseek(model: str = "deepseek-flash", **kw: Any) -> Any:
    from pyutilz.llm.deepseek_provider import DeepSeekProvider

    return DeepSeekProvider(api_key="sk-fake", model=model, **kw)  # pragma: allowlist secret -- dummy key


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

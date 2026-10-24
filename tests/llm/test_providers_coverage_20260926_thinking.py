from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock
from pyutilz.core import jsonlib
from pyutilz.llm.anthropic_provider import AnthropicProvider
from types import SimpleNamespace
from typing import Any
import pytest
from pyutilz.llm._thinking import gemini_thinking_config


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

"""A caller's ``thinking=`` must reach the upstream on every provider, not be accepted and dropped.

Gemini and Claude Code declared no ``thinking`` on ``generate`` (and logged-and-ignored it on ``generate_json``), so a
validation run asking for medium effort ran at whatever the upstream defaulted to, and a comparison across providers
measured the settings rather than the models. Gemini now sends ``thinking_config.thinking_budget``; the Claude Code CLI
reads MAX_THINKING_TOKENS from its environment. Both use the Anthropic API provider's effort budgets.
"""

from __future__ import annotations

import asyncio
import inspect
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import pyutilz.llm.claude_code_provider as ccp
from pyutilz.llm._thinking import THINKING_BUDGETS
from pyutilz.llm.claude_code_provider import ClaudeCodeProvider, claude_code_thinking_tokens
from pyutilz.llm.gemini_provider import GeminiProvider, gemini_thinking_budget


def _provider_classes() -> list[type]:
    from pyutilz.llm.anthropic_provider import AnthropicProvider
    from pyutilz.llm.deepseek_provider import DeepSeekProvider
    from pyutilz.llm.openai_provider import OpenAIProvider
    from pyutilz.llm.openrouter_provider import OpenRouterProvider
    from pyutilz.llm.xai_provider import XAIProvider

    return [AnthropicProvider, ClaudeCodeProvider, DeepSeekProvider, GeminiProvider, OpenAIProvider, OpenRouterProvider, XAIProvider]


def _declared(cls: type) -> set[str]:
    names: set[str] = set()
    for owner in cls.__mro__:
        target = owner.__dict__.get("generate")
        if target is None:
            continue
        params = inspect.signature(target).parameters
        names |= set(params)
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            break
    return names


@pytest.mark.parametrize("cls", _provider_classes(), ids=lambda c: c.__name__)
def test_every_provider_generate_takes_temperature_and_thinking(cls: type) -> None:
    """glossum's LLM client refuses a request option the provider would drop; that is only safe while every one takes both."""
    assert {"temperature", "thinking"} <= _declared(cls)


class TestClaudeCode:
    def test_effort_names_map_to_the_shared_budgets(self) -> None:
        assert claude_code_thinking_tokens(None) is None
        assert claude_code_thinking_tokens(False) == 0
        assert claude_code_thinking_tokens("off") == 0
        assert claude_code_thinking_tokens(True) == THINKING_BUDGETS["medium"]
        assert claude_code_thinking_tokens("high") == THINKING_BUDGETS["high"]

    def test_the_budget_reaches_the_cli_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict = {}

        class _Options:
            def __init__(self, **kwargs):
                seen.update(kwargs)
                self.extra_args = kwargs.get("extra_args")

        class _Result:
            result = "pong"
            usage = types.SimpleNamespace(input_tokens=1, output_tokens=1, cache_creation_input_tokens=0, cache_read_input_tokens=0)
            total_cost_usd = 0.0
            session_id = "s"
            num_turns = 1

        async def _query(prompt=None, options=None, **_kw):
            yield _Result()

        monkeypatch.setattr(ccp, "ClaudeCodeOptions", _Options, raising=False)
        monkeypatch.setattr(ccp, "cc_query", _query, raising=False)
        monkeypatch.setattr(ccp, "_HAS_SDK", True, raising=False)
        monkeypatch.setattr(ccp, "ResultMessage", _Result, raising=False)

        # haiku thinks on a budget, so the budget goes to the environment. The adaptive aliases take
        # `--effort` instead (asserted below).
        asyncio.run(ClaudeCodeProvider(model="haiku").generate("say pong", thinking="low"))
        assert seen["env"]["MAX_THINKING_TOKENS"] == str(THINKING_BUDGETS["low"])

        seen.clear()
        asyncio.run(ClaudeCodeProvider(model="sonnet").generate("say pong", thinking="low"))
        assert seen["extra_args"]["effort"] == "low"
        assert "MAX_THINKING_TOKENS" not in seen["env"]

    def test_no_thinking_request_leaves_the_cli_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list = []
        provider = ClaudeCodeProvider(model="sonnet")

        async def _sdk(prompt, system=None, **kwargs):
            seen.append(kwargs.get("thinking_tokens"))
            return "ok"

        monkeypatch.setattr(ccp, "_HAS_SDK", True, raising=False)
        monkeypatch.setattr(provider, "_generate_sdk", _sdk)
        asyncio.run(provider.generate("p"))
        assert seen == [None]


class TestGemini:
    def test_effort_names_map_to_the_shared_budgets(self) -> None:
        assert gemini_thinking_budget(None, "gemini-2.5-flash") is None
        assert gemini_thinking_budget(False, "gemini-2.5-flash") == 0
        assert gemini_thinking_budget("low", "gemini-2.5-flash") == THINKING_BUDGETS["low"]

    def test_a_pro_model_that_cannot_stop_thinking_is_sent_nothing(self) -> None:
        assert gemini_thinking_budget(False, "gemini-2.5-pro") is None

    def test_the_budget_reaches_the_request(self) -> None:
        mock_settings = MagicMock()
        mock_settings.gemini_api_key = None
        with patch("pyutilz.llm.gemini_provider.get_llm_settings", return_value=mock_settings), patch(
            "pyutilz.llm.gemini_provider.GENAI_AVAILABLE", True
        ), patch("pyutilz.llm.gemini_provider.genai") as mock_genai:
            mock_genai.Client.return_value = MagicMock()
            provider = GeminiProvider(api_key="test-key", model="gemini-2.5-flash")  # pragma: allowlist secret -- test placeholder
        response = MagicMock()
        response.text = "ok"
        response.candidates = [MagicMock(finish_reason="STOP")]
        response.usage_metadata = types.SimpleNamespace(
            prompt_token_count=3, candidates_token_count=2, thoughts_token_count=1, cached_content_token_count=0, total_token_count=6
        )
        provider.client.aio.models.generate_content = AsyncMock(return_value=response)

        assert asyncio.run(provider.generate("p", thinking="high")) == "ok"

        config = provider.client.aio.models.generate_content.call_args.kwargs["config"]
        assert config.thinking_config.thinking_budget == THINKING_BUDGETS["high"]

"""Anthropic extended thinking: the request must actually carry it.

Until ``AnthropicProvider.generate`` grew a ``thinking`` parameter, the
caller-side flag was silently DROPPED for Anthropic: ``llm_client`` only
forwards ``thinking=`` to providers whose signature declares it, so every
Anthropic call ran without reasoning no matter what the caller asked for --
including call sites whose comments explained why they were "disabling" it.
The provider already parsed ``thinking`` blocks out of the response, so it was
prepared to receive reasoning it had no way to request.

These tests assert on the request body the provider builds, which is where the
defect lived; they make no network call.
"""

from __future__ import annotations

import pytest

from pyutilz.llm.anthropic_provider import AnthropicProvider


def _provider(model: str = "claude-sonnet-4-20250514") -> AnthropicProvider:
    p = AnthropicProvider.__new__(AnthropicProvider)
    p.model = model
    return p


class TestThinkingRequestField:
    def test_off_by_default(self) -> None:
        assert _provider()._thinking_request_field(False, 8000) is None

    def test_empty_string_is_off(self) -> None:
        """Matches the shared normalize_thinking contract."""
        assert _provider()._thinking_request_field("", 8000) is None

    def test_true_uses_the_medium_budget(self) -> None:
        field = _provider()._thinking_request_field(True, 20_000)
        assert field == {"type": "enabled", "budget_tokens": 4096}

    @pytest.mark.parametrize(
        ("effort", "budget"),
        [("minimal", 1024), ("low", 2048), ("medium", 4096), ("high", 8192)],
    )
    def test_effort_selects_its_budget(self, effort: str, budget: int) -> None:
        field = _provider()._thinking_request_field(effort, 20_000)
        assert field == {"type": "enabled", "budget_tokens": budget}

    def test_effort_is_case_insensitive(self) -> None:
        assert _provider()._thinking_request_field("HIGH", 20_000) == {
            "type": "enabled", "budget_tokens": 8192,
        }

    def test_unknown_effort_leaves_thinking_off(self) -> None:
        """Substituting a different budget than the caller asked for would bill
        them for reasoning they did not request, while they believe their
        setting took effect."""
        assert _provider()._thinking_request_field("extreme", 20_000) is None

    def test_budget_is_capped_to_leave_room_for_the_answer(self) -> None:
        """Anthropic carves the budget OUT of max_tokens, so a budget at or above
        it leaves nothing for the response and the API rejects the request."""
        field = _provider()._thinking_request_field("high", 6000)
        assert field is not None
        assert field["budget_tokens"] == 6000 - 1024
        assert field["budget_tokens"] < 6000

    def test_too_small_max_tokens_leaves_thinking_off(self) -> None:
        assert _provider()._thinking_request_field("high", 1500) is None


class TestGenerateRequestBody:
    """The parameter has to reach the request, not just exist on the signature."""

    @staticmethod
    async def _captured_kwargs(**generate_kwargs):
        from unittest.mock import AsyncMock, MagicMock

        # Constructed properly (with a fake key) rather than via __new__: generate()
        # touches a dozen counters __init__ seeds, and stubbing them one by one is
        # how a test ends up asserting against a half-built object.
        p = AnthropicProvider(api_key="sk-fake-test-key-not-real")

        captured: dict = {}
        block = MagicMock()
        block.type = "text"
        block.text = "ok"
        response = MagicMock()
        response.content = [block]
        response.stop_reason = "end_turn"
        response.usage = MagicMock(
            input_tokens=10, output_tokens=5,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )
        raw = MagicMock()
        raw.parse.return_value = response
        raw.headers = {}

        async def _create(**kwargs):
            captured.update(kwargs)
            return raw

        p.client = MagicMock()
        p.client.messages.with_raw_response.create = AsyncMock(side_effect=_create)
        p._capture_response_headers = lambda headers: None
        p.fit_max_tokens_to_context = lambda mt, prompt, system: mt

        await p.generate("prompt", max_tokens=20_000, **generate_kwargs)
        return captured

    @pytest.mark.asyncio
    async def test_thinking_absent_from_the_request_when_off(self) -> None:
        kwargs = await self._captured_kwargs(thinking=False, temperature=0.1)
        assert "thinking" not in kwargs
        assert kwargs["temperature"] == 0.1, "an off request must keep the caller's temperature"

    @pytest.mark.asyncio
    async def test_thinking_reaches_the_request_when_on(self) -> None:
        kwargs = await self._captured_kwargs(thinking="high", temperature=0.1)
        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 8192}

    @pytest.mark.asyncio
    async def test_temperature_is_forced_to_one_with_thinking(self) -> None:
        """Anthropic rejects any temperature but 1 while extended thinking is on.
        Callers pass a low temperature for determinism, so honouring both is
        impossible -- overriding beats a 400 the caller could not have predicted.
        """
        kwargs = await self._captured_kwargs(thinking=True, temperature=0.1)
        assert kwargs["temperature"] == 1

    @pytest.mark.asyncio
    async def test_default_call_is_unchanged(self) -> None:
        """Non-vacuousness: adding the parameter must not alter existing calls."""
        kwargs = await self._captured_kwargs(temperature=0.7)
        assert "thinking" not in kwargs
        assert kwargs["temperature"] == 0.7

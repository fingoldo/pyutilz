"""Haiku 5.5 through the Claude Code CLI: the reasoning level is passed as --effort, never as a token budget of ours."""

from __future__ import annotations

from pyutilz.llm.claude_code_provider import ClaudeCodeProvider


def _provider(model: str) -> ClaudeCodeProvider:
    p = ClaudeCodeProvider.__new__(ClaudeCodeProvider)
    p.model = model
    return p


def test_haiku_55_low_sends_effort_and_no_budget() -> None:
    assert _provider("claude-haiku-5-5")._thinking_transport_kwargs("low") == {"effort": "low"}


def test_haiku_55_low_is_not_a_token_budget() -> None:
    kwargs = _provider("claude-haiku-5-5")._thinking_transport_kwargs("low")
    assert "thinking_tokens" not in kwargs


def test_haiku_55_off_is_still_the_zero_budget() -> None:
    assert _provider("claude-haiku-5-5")._thinking_transport_kwargs(False) == {"thinking_tokens": 0}

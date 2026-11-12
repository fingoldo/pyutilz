"""The SDK backend asks for ``--thinking-display`` and serves the ThinkingBlock text as ``last_reasoning_text``.

The CLI backend does this since 42acad9; the Track B sweep reaches the provider through the SDK, where the generic ``.text`` branch never saw a ``ThinkingBlock`` (its text is in
``thinking``) and the archive said ``reasoning_text_absent`` for calls billed hundreds of reasoning tokens. Driven with a stubbed SDK, no live call.
"""

from __future__ import annotations

import asyncio
import types

import pytest

import pyutilz.llm.claude_code_provider as ccp


class _Options:
    """Stands in for ClaudeCodeOptions: records what the provider asked for."""

    instances: "list[_Options]" = []

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        _Options.instances.append(self)


class ThinkingBlock:
    """Same class name as the SDK's block, whose text sits in ``thinking``."""

    def __init__(self, thinking: str):
        self.thinking = thinking
        self.signature = "sig"


class _Text:
    def __init__(self, text: str):
        self.text = text


class AssistantMessage:
    """Same class name as the SDK's message."""

    def __init__(self, *blocks):
        self.content = list(blocks)


class _ResultMessage:
    def __init__(self, text):
        self.result = text
        self.is_error = False
        self.subtype = "success"
        self.usage = {"input_tokens": 1, "output_tokens": 1}
        self.total_cost_usd = 0.0
        self.session_id = "s"
        self.num_turns = 1


def _install(monkeypatch, messages_by_call):
    """Stub the SDK; ``messages_by_call`` is a list of message lists (or exceptions) answered in order."""
    calls: "list[dict]" = []

    async def _query(prompt=None, options=None, **_kw):
        calls.append(dict(options.extra_args))
        outcome = messages_by_call[min(len(calls), len(messages_by_call)) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        for message in outcome:
            yield message

    monkeypatch.setattr(ccp, "ClaudeCodeOptions", _Options, raising=False)
    monkeypatch.setattr(ccp, "cc_query", _query, raising=False)
    monkeypatch.setattr(ccp, "_HAS_SDK", True, raising=False)
    monkeypatch.setattr(ccp, "ResultMessage", _ResultMessage, raising=False)
    monkeypatch.setattr(ccp, "TextBlock", _Text, raising=False)
    monkeypatch.setattr(ccp.ClaudeCodeProvider, "_thinking_display_unsupported", False)
    _Options.instances = []
    return calls


def _run(provider):
    async def go():
        answer = await provider._generate_sdk("q", effort="low")
        return answer, provider.last_reasoning_text

    return asyncio.run(go())


def test_the_flag_is_sent_and_the_thinking_block_text_is_served(monkeypatch):
    calls = _install(monkeypatch, [[AssistantMessage(ThinkingBlock("weighing 17 times 23"), _Text("yes")), _ResultMessage("yes")]])
    answer, reasoning = _run(ccp.ClaudeCodeProvider(model="sonnet"))
    assert answer == "yes" and reasoning == "weighing 17 times 23"
    assert calls[0]["thinking-display"] == "summarized"


def test_blocks_of_several_turns_are_separated_by_a_blank_line(monkeypatch):
    _install(monkeypatch, [[AssistantMessage(ThinkingBlock("first")), AssistantMessage(ThinkingBlock("second"), _Text("a")), _ResultMessage("a")]])
    assert _run(ccp.ClaudeCodeProvider(model="sonnet"))[1] == "first\n\nsecond"


def test_an_empty_thinking_block_leaves_no_reasoning_and_the_answer_unchanged(monkeypatch):
    _install(monkeypatch, [[AssistantMessage(ThinkingBlock(""), _Text("ok")), _ResultMessage("ok")]])
    assert _run(ccp.ClaudeCodeProvider(model="sonnet")) == ("ok", None)


def test_a_call_without_thinking_blocks_has_none(monkeypatch):
    _install(monkeypatch, [[AssistantMessage(_Text("ok")), _ResultMessage("ok")]])
    assert _run(ccp.ClaudeCodeProvider(model="sonnet")) == ("ok", None)


def test_the_flag_is_not_sent_when_the_knob_is_none(monkeypatch):
    calls = _install(monkeypatch, [[_ResultMessage("ok")]])
    _run(ccp.ClaudeCodeProvider(model="sonnet", thinking_display=None))
    assert "thinking-display" not in calls[0]


def test_a_cli_that_rejects_the_flag_is_asked_again_without_it_once_per_process(monkeypatch):
    rejected = RuntimeError("Command failed: error: unknown option '--thinking-display'")
    calls = _install(monkeypatch, [rejected, [_ResultMessage("ok")], [_ResultMessage("ok")]])
    provider = ccp.ClaudeCodeProvider(model="sonnet")
    assert _run(provider)[0] == "ok"
    assert "thinking-display" in calls[0] and "thinking-display" not in calls[1]
    assert _run(provider)[0] == "ok"
    assert "thinking-display" not in calls[2]


def test_any_other_error_is_raised_and_the_flag_stays(monkeypatch):
    _install(monkeypatch, [RuntimeError("boom")])
    with pytest.raises(RuntimeError, match="boom"):
        _run(ccp.ClaudeCodeProvider(model="sonnet"))
    assert ccp.ClaudeCodeProvider._thinking_display_unsupported is False


def test_a_rejection_is_read_from_the_stderr_attribute_too(monkeypatch):
    exc = RuntimeError("Command failed with exit code 1")
    exc.stderr = "error: unknown option '--thinking-display'"  # type: ignore[attr-defined]
    calls = _install(monkeypatch, [exc, [_ResultMessage("ok")]])
    assert _run(ccp.ClaudeCodeProvider(model="sonnet"))[0] == "ok"
    assert len(calls) == 2


def test_a_tool_use_block_still_fails_hard(monkeypatch):
    class ToolUseBlock:
        name = "Bash"

    _install(monkeypatch, [[AssistantMessage(ThinkingBlock("x"), ToolUseBlock())]])
    with pytest.raises(ccp.ClaudeCodeToolUseError):
        _run(ccp.ClaudeCodeProvider(model="sonnet"))


def test_the_helper_reads_only_thinking_blocks():
    from pyutilz.llm.claude_code_cli import sdk_thinking_text

    assert sdk_thinking_text([ThinkingBlock("a"), _Text("b"), ThinkingBlock("c")]) == "a\n\nc"
    assert sdk_thinking_text(None) == "" and sdk_thinking_text([]) == ""
    assert isinstance(types.SimpleNamespace(), object)

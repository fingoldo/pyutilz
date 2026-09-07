"""The SDK path must leave the system prompt where the cache can see it.

``_generate_sdk`` folded the system prompt into the user message whenever the two together
exceeded 6000 characters -- which is every real call, the glossum system prompt being 48-58 KB --
and set ``system_prompt`` to "". The system prompt is the cacheable prefix: two identical CLI
invocations with a 48,051-character system prompt measured 14813 cache-creation tokens and then
14813 cache-read, $0.14824 then $0.00752. Folding it into the per-call user message gives all of
that up.

Latent, because ``claude-code-sdk`` is not installed here -- it fires on the first install, which
is exactly when nobody would be looking. The test drives the branch with a stubbed SDK.
"""

from __future__ import annotations

import asyncio
import types

import pytest

import pyutilz.llm.claude_code_provider as ccp

_BIG_SYSTEM = "RULES\n" + ("x" * 50000)


class _Options:
    """Stands in for ClaudeCodeOptions: records what the provider asked for."""

    last: "_Options | None" = None

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        _Options.last = self


class _ResultMessage:
    def __init__(self, text):
        self.result = text
        self.usage = types.SimpleNamespace(input_tokens=1, output_tokens=1, cache_creation_input_tokens=0, cache_read_input_tokens=0)
        self.total_cost_usd = 0.0
        self.session_id = "s"
        self.num_turns = 1


@pytest.fixture
def stub_sdk(monkeypatch):
    """Make the SDK branch reachable without installing claude-code-sdk."""
    captured: "list[str]" = []

    async def _query(prompt=None, options=None, **_kw):
        captured.append(prompt)
        yield _ResultMessage("pong")

    monkeypatch.setattr(ccp, "ClaudeCodeOptions", _Options, raising=False)
    monkeypatch.setattr(ccp, "cc_query", _query, raising=False)
    monkeypatch.setattr(ccp, "_HAS_SDK", True, raising=False)
    monkeypatch.setattr(ccp, "ResultMessage", _ResultMessage, raising=False)
    _Options.last = None
    return captured


def _run(provider, **kwargs):
    return asyncio.run(provider._generate_sdk(**kwargs))


class TestTheSystemPromptStaysASystemPrompt:
    def test_a_large_system_prompt_is_not_folded_into_the_user_message(self, stub_sdk):
        provider = ccp.ClaudeCodeProvider(model="sonnet")

        _run(provider, prompt="say pong", system=_BIG_SYSTEM)

        assert _Options.last is not None, "the SDK branch was not reached"
        assert _Options.last.system_prompt == _BIG_SYSTEM, "the cacheable prefix was moved out of system_prompt"
        assert "[System instructions]" not in stub_sdk[0], "the system prompt was folded into the per-call user message"
        assert stub_sdk[0] == "say pong"

    def test_a_small_one_is_left_alone_too(self, stub_sdk):
        provider = ccp.ClaudeCodeProvider(model="sonnet")

        _run(provider, prompt="say pong", system="terse")

        assert _Options.last.system_prompt == "terse"
        assert stub_sdk[0] == "say pong"

    def test_no_system_prompt_is_an_empty_string_not_a_crash(self, stub_sdk):
        provider = ccp.ClaudeCodeProvider(model="sonnet")

        _run(provider, prompt="say pong")

        assert _Options.last.system_prompt == ""

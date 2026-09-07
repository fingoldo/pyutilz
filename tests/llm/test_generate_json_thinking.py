"""`generate_json` can ask for reasoning effort, the way `generate` already could.

A structured task is often the one that most deserves a reasoning budget -- the caller that prompted
this asks a model to answer eleven technical screening questions and return them as JSON -- and
until now the only way to request it was to give up JSON mode.

The Liskov half matters as much as the feature: the base class offers the argument, so every
override has to ACCEPT it, including the two providers that cannot honour it. A provider that
omitted it would raise TypeError for a caller who followed the base contract, and only for whichever
provider happened to be configured.
"""

from __future__ import annotations

import inspect

import pytest

from pyutilz.llm.base import LLMProvider


class _Recorder(LLMProvider):
    """The smallest provider that records what `generate` was called with."""

    def __init__(self):
        self.calls: list[dict] = []

    async def generate(self, prompt, system=None, temperature=0.7, max_tokens=0, **kwargs):
        self.calls.append({"prompt": prompt, "system": system, **kwargs})
        return '{"ok": true}'

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class TestTheEffortReachesTheProvider:
    @pytest.mark.asyncio
    async def test_an_effort_string_is_forwarded(self):
        provider = _Recorder()
        await provider.generate_json("q", thinking="high")
        assert provider.calls[0]["thinking"] == "high"

    @pytest.mark.asyncio
    async def test_a_plain_true_is_forwarded(self):
        provider = _Recorder()
        await provider.generate_json("q", thinking=True)
        assert provider.calls[0]["thinking"] is True

    @pytest.mark.asyncio
    async def test_nothing_is_sent_when_it_was_not_asked_for(self):
        """A provider whose `generate` has no such parameter must be unaffected -- passing
        `thinking=None` down would be a TypeError on every ordinary call."""
        provider = _Recorder()
        await provider.generate_json("q")
        assert "thinking" not in provider.calls[0]

    @pytest.mark.asyncio
    async def test_thinking_false_is_still_forwarded(self):
        """False is a decision ("no reasoning"), not an absence; only None means "unspecified"."""
        provider = _Recorder()
        await provider.generate_json("q", thinking=False)
        assert provider.calls[0]["thinking"] is False

    @pytest.mark.asyncio
    async def test_the_json_is_still_parsed(self):
        provider = _Recorder()
        assert await provider.generate_json("q", thinking="high") == {"ok": True}


class TestEveryOverrideAcceptsIt:
    """A signature check, deliberately: this is a Liskov contract, and the failure it prevents is a
    TypeError raised only for the provider a deployment happens to have configured."""

    def _overrides(self):
        from pyutilz.llm import claude_code_provider, gemini_provider, openai_compat

        return [
            claude_code_provider.ClaudeCodeProvider,
            gemini_provider.GeminiProvider,
            openai_compat.OpenAICompatibleProvider,
        ]

    def test_each_one_takes_thinking(self):
        for cls in self._overrides():
            parameters = inspect.signature(cls.generate_json).parameters
            assert "thinking" in parameters, f"{cls.__name__}.generate_json refuses the base contract"

    def test_it_is_keyword_reachable_and_optional_everywhere(self):
        for cls in self._overrides():
            parameter = inspect.signature(cls.generate_json).parameters["thinking"]
            assert parameter.default is None, f"{cls.__name__} makes it mandatory"

    def test_images_still_precedes_it_so_positional_callers_are_unbroken(self):
        """`images` was the fifth parameter before this change and has to stay there."""
        for cls in self._overrides():
            names = list(inspect.signature(cls.generate_json).parameters)
            assert names.index("images") < names.index("thinking"), cls.__name__

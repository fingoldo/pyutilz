"""Behavioral coverage for ClaudeCodeProvider.generate()'s retry/rate-limit orchestration and
count_tokens/supports_json_mode -- previously only the parsing helpers, generate_json, and the
zero-cost accessors were exercised (test_llm_providers.py), leaving generate()'s own control flow
(transient-error retry, rate-limit wait, max-attempts, usage bookkeeping) untested."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from pyutilz.llm.claude_code_provider import ClaudeCodeProvider


def _provider(**overrides):
    p = ClaudeCodeProvider()
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


class TestSupportsJsonMode:
    def test_returns_false(self):
        assert _provider().supports_json_mode() is False


class TestGenerateRetryOrchestration:
    @pytest.mark.asyncio
    async def test_transient_error_retries_then_succeeds(self):
        p = _provider()
        calls = {"n": 0}

        async def flaky_sdk(prompt, system=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("blip")
            return "ok"

        with patch.object(p, "_generate_sdk", side_effect=flaky_sdk), patch("pyutilz.llm.claude_code_provider._HAS_SDK", True), patch(
            "asyncio.sleep", new=AsyncMock()
        ):
            result = await p.generate("hi")
        assert result == "ok"
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_subprocess_timeout_expired_is_treated_as_transient(self):
        """subprocess.TimeoutExpired is a SubprocessError, NOT a TimeoutError subclass -- must
        still be retried by the same except clause as ConnectionError/TimeoutError/OSError."""
        p = _provider()
        calls = {"n": 0}

        async def flaky(prompt, system=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise subprocess.TimeoutExpired(cmd=["claude"], timeout=5)
            return "recovered"

        with patch.object(p, "_generate_sdk", side_effect=flaky), patch("pyutilz.llm.claude_code_provider._HAS_SDK", True), patch(
            "asyncio.sleep", new=AsyncMock()
        ):
            result = await p.generate("hi")
        assert result == "recovered"
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_exceeding_max_attempts_raises(self):
        p = _provider()

        async def always_fails(prompt, system=None):
            raise ConnectionError("still down")

        with patch.object(p, "_generate_sdk", side_effect=always_fails), patch("pyutilz.llm.claude_code_provider._HAS_SDK", True), patch(
            "pyutilz.llm.claude_code_provider.MAX_RETRY_ATTEMPTS", 2
        ), patch("asyncio.sleep", new=AsyncMock()):
            with pytest.raises(RuntimeError, match="exceeded 2 retry attempts"):
                await p.generate("hi")

    @pytest.mark.asyncio
    async def test_non_rate_limit_exception_reraises_immediately(self):
        p = _provider()

        async def boom(prompt, system=None):
            raise ValueError("totally unrelated failure")

        with patch.object(p, "_generate_sdk", side_effect=boom), patch("pyutilz.llm.claude_code_provider._HAS_SDK", True):
            with pytest.raises(ValueError, match="totally unrelated failure"):
                await p.generate("hi")

    @pytest.mark.asyncio
    async def test_rate_limit_error_waits_parsed_seconds_then_succeeds(self):
        p = _provider()
        calls = {"n": 0}

        async def rate_limited_once(prompt, system=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("You've hit your limit, resets 5 pm")
            return "ok"

        with patch.object(p, "_generate_sdk", side_effect=rate_limited_once), patch("pyutilz.llm.claude_code_provider._HAS_SDK", True), patch(
            "asyncio.sleep", new=AsyncMock()
        ) as mock_sleep:
            result = await p.generate("hi")
        assert result == "ok"
        assert calls["n"] == 2
        assert mock_sleep.await_count == 1
        # A parsed reset time was used, not the 3600s fallback.
        assert mock_sleep.await_args.args[0] != 3600

    @pytest.mark.asyncio
    async def test_rate_limit_error_unparseable_falls_back_to_default_wait(self):
        p = _provider()
        calls = {"n": 0}

        async def rate_limited_once(prompt, system=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("quota exceeded, no parseable time here")
            return "ok"

        with patch.object(p, "_generate_sdk", side_effect=rate_limited_once), patch("pyutilz.llm.claude_code_provider._HAS_SDK", True), patch(
            "asyncio.sleep", new=AsyncMock()
        ) as mock_sleep:
            result = await p.generate("hi")
        assert result == "ok"
        assert mock_sleep.await_args.args[0] == 3600

    @pytest.mark.asyncio
    async def test_json_mode_strips_markdown_fence(self):
        p = _provider()

        async def fenced(prompt, system=None):
            return '```json\n{"a": 1}\n```'

        with patch.object(p, "_generate_sdk", side_effect=fenced), patch("pyutilz.llm.claude_code_provider._HAS_SDK", True):
            result = await p.generate("hi", json_mode=True)
        assert result == '{"a": 1}'

    @pytest.mark.asyncio
    async def test_usage_from_result_message_preferred_over_tiktoken(self):
        p = _provider()

        async def with_result_message(prompt, system=None):
            p._last_result_message = SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=11, output_tokens=22,
                    cache_creation_input_tokens=3, cache_read_input_tokens=4,
                ),
                total_cost_usd=0.05,
                session_id="sess-1",
                num_turns=2,
            )
            return "text"

        with patch.object(p, "_generate_sdk", side_effect=with_result_message), patch("pyutilz.llm.claude_code_provider._HAS_SDK", True):
            await p.generate("hi")
        assert p.total_prompt_tokens == 11
        assert p.total_completion_tokens == 22
        assert p.total_cache_creation_input_tokens == 3
        assert p.total_cache_read_input_tokens == 4
        assert p.total_cost_usd == 0.05
        assert p.last_session_id == "sess-1"
        assert p.last_num_turns == 2

    @pytest.mark.asyncio
    async def test_cli_backend_used_when_sdk_unavailable(self):
        p = _provider()

        async def cli_gen(prompt, system=None, temperature=0.7, max_tokens=0):
            return "cli-result"

        with patch.object(p, "_generate_cli", side_effect=cli_gen), patch("pyutilz.llm.claude_code_provider._HAS_SDK", False):
            result = await p.generate("hi")
        assert result == "cli-result"


class TestCountTokens:
    @pytest.mark.asyncio
    async def test_uses_anthropic_api_when_available(self):
        p = _provider()
        fake_result = SimpleNamespace(input_tokens=42)
        fake_client = SimpleNamespace(messages=SimpleNamespace(count_tokens=AsyncMock(return_value=fake_result)))
        fake_anthropic_module = SimpleNamespace(AsyncAnthropic=lambda: fake_client)

        with patch.dict("sys.modules", {"anthropic": fake_anthropic_module}):
            out = await p.count_tokens("hello world")
        assert out == 42

    @pytest.mark.asyncio
    async def test_falls_back_to_tiktoken_when_anthropic_unavailable(self):
        p = _provider()

        def _raise_import(*a, **k):
            raise ImportError("no anthropic package")

        fake_anthropic_module = SimpleNamespace(AsyncAnthropic=_raise_import)
        with patch.dict("sys.modules", {"anthropic": fake_anthropic_module}):
            out = await p.count_tokens("hello")
        assert isinstance(out, int)
        assert out > 0

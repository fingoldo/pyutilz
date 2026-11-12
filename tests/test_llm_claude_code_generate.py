"""Behavioral coverage for ClaudeCodeProvider.generate()'s retry/rate-limit orchestration and
count_tokens/supports_json_mode -- previously only the parsing helpers, generate_json, and the
zero-cost accessors were exercised (test_llm_providers.py), leaving generate()'s own control flow
(transient-error retry, rate-limit wait, max-attempts, usage bookkeeping) untested."""

from __future__ import annotations

import subprocess
from pathlib import Path
import io
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

        class _Ctx:
            """The client is used as an async context manager now, so it is closed after the count (PROV-31)."""

            async def __aenter__(self):
                return SimpleNamespace(messages=SimpleNamespace(count_tokens=AsyncMock(return_value=fake_result)))

            async def __aexit__(self, *exc):
                return None

        fake_anthropic_module = SimpleNamespace(AsyncAnthropic=lambda: _Ctx())

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


class TestSystemPromptGoesInAFile:
    """2026-09-06: a real system prompt does not fit in a Windows command line.

    ``_generate_cli`` used to pass ``--system-prompt <text>`` as an argv element. Windows
    caps a whole command line at 32767 characters (CreateProcess), and the measured case --
    glossum's English enrichment prompt -- is 48378 characters on its own. Every attempt
    failed with ``WinError 206, the filename or extension is too long``; the retry loop
    treated that permanent error as transient and spent thirteen attempts and forty minutes
    on it before the caller's budget expired.

    The CLI accepts ``--system-prompt-file``, so the path is what argv carries, whatever the
    prompt's length. No size threshold to pick, and none to get wrong.
    """

    def _captured_cmd(self, system: str):
        """Run _generate_cli far enough to capture argv, with the subprocess stubbed out."""
        import asyncio

        p = _provider()
        p._claude_path = "claude"
        captured = {}

        class _FakeProc:
            def poll(self):
                """The process has already produced its whole transcript, so it is never still running."""
                return self.returncode

            def __enter__(self):
                """`run_cli` uses `with Popen(...)`; a fake that is not a context manager fails there and nowhere else."""
                return self

            def __exit__(self, *_exc):
                """Nothing to release - the streams are in-memory."""
                return False

            returncode = 0
            stdin = SimpleNamespace(write=lambda _s: None, close=lambda: None)
            stdout = io.StringIO("")  # iterable and readline-capable, like Popen.stdout
            stderr = io.StringIO("")  # iterable like Popen.stderr, so the drain thread finishes cleanly

            def kill(self):
                pass

            def wait(self, timeout=None):  # unused-ok: fake keeps the real API signature (timeout accepted, nothing to wait on in-memory)
                return 0

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = list(cmd)
            captured["system_file"] = None
            if "--system-prompt-file" in cmd:
                path = cmd[cmd.index("--system-prompt-file") + 1]
                captured["system_file"] = path
                captured["file_contents"] = Path(path).read_text(encoding="utf-8")
            return _FakeProc()

        with patch.object(subprocess, "Popen", fake_popen):
            try:
                asyncio.run(p._generate_cli(prompt="hi", system=system))
            except Exception:
                pass  # the fake process yields no result event; argv is what this asserts on
        return captured

    def test_the_system_prompt_is_not_an_argv_element(self):
        system = "You are terse.\n" + ("filler line to blow past the OS limit\n" * 1200)
        assert len(system) > 32767, "the fixture must exceed the Windows command-line cap"

        captured = self._captured_cmd(system)

        assert "--system-prompt" not in captured["cmd"], "the prompt is back on the command line"
        assert "--system-prompt-file" in captured["cmd"]
        assert sum(len(a) for a in captured["cmd"]) < 32767, "the command line still exceeds the OS cap"

    def test_the_file_holds_exactly_the_prompt(self):
        system = "You are a terse assistant. Reply with one word."

        captured = self._captured_cmd(system)

        assert captured["file_contents"] == system

    def test_the_temporary_file_is_removed(self):
        captured = self._captured_cmd("You are terse.")

        assert not Path(captured["system_file"]).exists(), "the temporary system-prompt file leaked"

    def test_no_system_prompt_means_no_flag(self):
        """Negative control: a call without a system prompt must not grow a file or a flag."""
        captured = self._captured_cmd("")

        assert "--system-prompt-file" not in captured["cmd"]
        assert captured["system_file"] is None


_THINKING_FIXTURE = Path(__file__).parent / "data" / "claude_cli_thinking_summarized.jsonl"


class _ScriptedProc:
    """A finished CLI process with a scripted exit code, stdout transcript and stderr; just enough of Popen for ``run_cli``."""

    stdin = SimpleNamespace(write=lambda _s: None, close=lambda: None)

    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)

    def poll(self):
        """The transcript is complete, so the process is never still running."""
        return self.returncode

    def __enter__(self):
        """`run_cli` uses `with Popen(...)`."""
        return self

    def __exit__(self, *_exc):
        """Nothing to release; the streams are in-memory."""
        return False

    def kill(self):
        """Never running, nothing to kill."""

    def wait(self, timeout=None):  # unused-ok: keeps the real API signature
        """Already finished."""
        return self.returncode


def _withheld_transcript() -> str:
    """The same call as the fixture but as the CLI answers WITHOUT ``--thinking-display``: the thinking block is there and its text is empty."""
    import json

    lines = []
    for line in _THINKING_FIXTURE.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event["type"] == "assistant":
            for block in event["message"]["content"]:
                if block["type"] == "thinking":
                    block["thinking"] = ""
        lines.append(json.dumps(event))
    return "\n".join(lines) + "\n"


def _drive(provider, scripts):
    """Run ``_generate_cli`` against scripted processes (one per spawn, in order); returns (answer, every argv spawned, the reasoning text).

    The reasoning is read inside the same task that made the call: ``last_reasoning_text`` is context-local (``PerCallAttr``), so it is gone once ``asyncio.run`` returns.
    """
    import asyncio

    provider._claude_path = "claude"
    argvs: list[list[str]] = []
    queue_of_scripts = list(scripts)

    def fake_popen(cmd, **kwargs):
        argvs.append(list(cmd))
        return queue_of_scripts.pop(0)

    async def call():
        answer = await provider._generate_cli(prompt="hi")
        return answer, provider.last_reasoning_text

    with patch.object(subprocess, "Popen", fake_popen):
        answer, reasoning = asyncio.run(call())
    return answer, argvs, reasoning


class TestTheReasoningSummaryReachesThePublicApi:
    """``--thinking-display summarized`` is sent by default and the summary comes back as ``last_reasoning_text``, like any other provider's reasoning."""

    def setup_method(self):
        ClaudeCodeProvider._thinking_display_unsupported = False

    def teardown_method(self):
        ClaudeCodeProvider._thinking_display_unsupported = False

    def test_the_flag_is_sent_by_default_and_the_summary_is_served(self):
        transcript = _THINKING_FIXTURE.read_text(encoding="utf-8")
        p = _provider()

        answer, argvs, reasoning = _drive(p, [_ScriptedProc(0, transcript, "")])

        assert argvs[0][argvs[0].index("--thinking-display") + 1] == "summarized"
        assert answer
        assert reasoning is not None and len(reasoning) == 472

    def test_none_sends_no_flag_and_serves_no_reasoning(self):
        transcript = _withheld_transcript()
        p = ClaudeCodeProvider(thinking_display=None)

        _answer, argvs, reasoning = _drive(p, [_ScriptedProc(0, transcript, "")])

        assert "--thinking-display" not in argvs[0]
        assert reasoning is None

    def test_a_value_the_cli_does_not_offer_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="thinking_display"):
            ClaudeCodeProvider(thinking_display="full")

    def test_a_cli_that_rejects_the_flag_is_asked_again_without_it_and_remembered(self):
        p = _provider()
        rejected = _ScriptedProc(1, "", "error: unknown option '--thinking-display'\n")

        answer, argvs, reasoning = _drive(p, [rejected, _ScriptedProc(0, _withheld_transcript(), "")])

        assert "--thinking-display" in argvs[0] and "--thinking-display" not in argvs[1]
        assert answer
        assert reasoning is None, "the repeated call carried no flag, so there is no summary"
        assert ClaudeCodeProvider._thinking_display_unsupported is True
        _answer, later, _reasoning = _drive(p, [_ScriptedProc(0, _withheld_transcript(), "")])
        assert "--thinking-display" not in later[0], "the rejection was not remembered"

    def test_another_failure_is_not_taken_for_a_rejected_flag(self):
        p = _provider()

        with pytest.raises(RuntimeError, match="authentication failed"):
            _drive(p, [_ScriptedProc(1, "", "authentication failed")])

        assert ClaudeCodeProvider._thinking_display_unsupported is False

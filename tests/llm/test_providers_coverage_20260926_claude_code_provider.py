from __future__ import annotations
from pyutilz.core import jsonlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
import pytest


class TestProv26Prov27ClaudeCodeResult:
    @pytest.mark.asyncio
    async def test_dict_usage_from_the_sdk_is_read(self) -> None:
        from pyutilz.llm.claude_code_provider import ClaudeCodeProvider

        p = ClaudeCodeProvider()

        async def fake_sdk(prompt: str, system: Any = None, **_: Any) -> str:
            p._last_result_message = SimpleNamespace(
                usage={"input_tokens": 11, "output_tokens": 7, "cache_creation_input_tokens": 3, "cache_read_input_tokens": 5,
                       "output_tokens_details": {"thinking_tokens": 4}},
                total_cost_usd=0.5, session_id="s", num_turns=1,
            )
            return "ok"

        with patch.object(p, "_generate_sdk", side_effect=fake_sdk), patch("pyutilz.llm.claude_code_provider._HAS_SDK", True):
            assert await p.generate("hi") == "ok"
        assert p._last_usage["input_tokens"] == 11 and p._last_usage["output_tokens"] == 7
        assert p._last_usage["cache_read_input_tokens"] == 5 and p._last_usage["reasoning_tokens"] == 4

    def test_cli_is_error_result_is_an_error_not_an_answer(self) -> None:
        import queue

        from pyutilz.llm.claude_code_cli import _consume_cli_stream

        q: queue.Queue = queue.Queue()
        q.put(jsonlib.dumps({"type": "result", "subtype": "success", "is_error": True, "result": "API Error: 529 overloaded"}))
        q.put(None)
        text, error, timed_out, _event = _consume_cli_stream(q, 5.0)
        assert text is None and error == "API Error: 529 overloaded" and timed_out is False

    @pytest.mark.asyncio
    async def test_sdk_is_error_result_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pyutilz.llm import claude_code_provider as ccp

        class FakeResult:
            def __init__(self) -> None:
                self.is_error = True
                self.result = "You've hit your limit"
                self.subtype = "success"
                self.usage = None

        seen: dict = {}

        class FakeOptions:
            def __init__(self, **kw: Any) -> None:
                self.extra_args = kw.get("extra_args")
                seen.update(kw)

        async def fake_query(prompt: str, options: Any) -> Any:
            yield FakeResult()

        monkeypatch.setattr(ccp, "ResultMessage", FakeResult, raising=False)
        monkeypatch.setattr(ccp, "TextBlock", type("TB", (), {}), raising=False)
        monkeypatch.setattr(ccp, "ClaudeCodeOptions", FakeOptions, raising=False)
        monkeypatch.setattr(ccp, "cc_query", fake_query, raising=False)
        p = ccp.ClaudeCodeProvider()
        with pytest.raises(RuntimeError, match="hit your limit") as exc:
            await p._generate_sdk("hi", effort="high")
        assert ccp._is_rate_limit_error(exc.value), "the error text must still reach the rate-limit wait"
        assert seen["extra_args"]["effort"] == "high"


class TestProv28Prov29ClaudeCodeThinkingAndLimits:
    def test_adaptive_alias_sends_effort_not_a_budget(self) -> None:
        from pyutilz.llm.claude_code_provider import ClaudeCodeProvider

        p = ClaudeCodeProvider(model="opus")
        assert p._thinking_transport_kwargs("minimal") == {"effort": "low"}
        assert p._thinking_transport_kwargs(False) == {"thinking_tokens": 0}
        assert p._thinking_transport_kwargs(None) == {}

    def test_budget_model_keeps_the_budget(self) -> None:
        from pyutilz.llm.claude_code_provider import ClaudeCodeProvider

        assert ClaudeCodeProvider(model="haiku")._thinking_transport_kwargs("high") == {"thinking_tokens": 8192}

    @pytest.mark.asyncio
    async def test_cli_argv_carries_effort(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pyutilz.llm import claude_code_provider as ccp

        seen: dict = {}

        def fake_run_cli(cmd: list, *args: Any) -> tuple:
            seen["cmd"] = cmd
            return 0, "ok", "", None

        monkeypatch.setattr(ccp, "run_cli", fake_run_cli)
        monkeypatch.setattr(ccp, "_find_claude_executable", lambda: "claude")
        p = ccp.ClaudeCodeProvider()
        assert await p._generate_cli("hi", effort="xhigh") == "ok"
        cmd = seen["cmd"]
        assert cmd[cmd.index("--effort") + 1] == "xhigh"
        assert cmd[-3:] == ["--tools", "", "-"], "--tools stays last before the stdin marker"

    @pytest.mark.parametrize(("model", "out", "ctx"), [("opus", 128_000, 1_000_000), ("haiku", 64_000, 200_000), ("claude-opus-5-5", 128_000, 1_000_000)])
    def test_limits_follow_the_model(self, model: str, out: int, ctx: int) -> None:
        from pyutilz.llm.claude_code_provider import ClaudeCodeProvider

        p = ClaudeCodeProvider(model=model)
        assert (p.max_output_tokens, p.context_window) == (out, ctx)


class TestProv30Prov31ClaudeCodeMisc:
    @pytest.mark.asyncio
    async def test_batch_does_not_invent_max_tokens(self) -> None:
        from pyutilz.llm.claude_code_provider import ClaudeCodeProvider

        p = ClaudeCodeProvider()
        calls: list[dict] = []

        async def fake_generate(**kw: Any) -> str:
            calls.append(kw)
            return "r"

        with patch.object(p, "generate", side_effect=fake_generate):
            out = [r async for r in p.generate_batch([{"id": 1, "prompt": "a"}, {"id": 2, "prompt": "b", "max_tokens": 9}])]
        assert [o["result"] for o in out] == ["r", "r"]
        assert "max_tokens" not in calls[0] and "temperature" not in calls[0]
        assert calls[1]["max_tokens"] == 9

    @pytest.mark.asyncio
    async def test_count_tokens_closes_its_client_and_resolves_the_alias(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import anthropic

        from pyutilz.llm.claude_code_provider import ClaudeCodeProvider

        state: dict = {}

        class FakeClient:
            def __init__(self, **_: Any) -> None:
                self.messages = SimpleNamespace(count_tokens=self._count)

            async def _count(self, **kw: Any) -> Any:
                state["model"] = kw["model"]
                return SimpleNamespace(input_tokens=42)

            async def __aenter__(self) -> FakeClient:
                return self

            async def __aexit__(self, *exc: Any) -> None:
                state["closed"] = True

        monkeypatch.setattr(anthropic, "AsyncAnthropic", FakeClient)
        assert await ClaudeCodeProvider(model="opus").count_tokens("hi") == 42
        assert state == {"model": "claude-opus-5", "closed": True}

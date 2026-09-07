"""``_generate_cli`` must actually record the usage it now parses.

The first version of this fix put the recording line after the ``raise`` in the newly added
``except asyncio.CancelledError`` arm, where it was unreachable: the parsing was correct, the
provider still reported tiktoken estimates and a zero cost, and only a live call showed it. A test
that exercises ``_generate_cli`` against a fake CLI catches that; a test of the parser alone does
not.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import pyutilz.llm.claude_code_provider as ccp

_RESULT = {
    "type": "result",
    "subtype": "success",
    "result": "pong",
    "total_cost_usd": 0.0413,
    "session_id": "sess-7",
    "num_turns": 1,
    "usage": {
        "input_tokens": 191,
        "output_tokens": 1608,
        "cache_creation_input_tokens": 13204,
        "cache_read_input_tokens": 41077,
    },
}


class _FakePipe:
    """Just enough of a text pipe: iterable, closeable, writable."""

    def __init__(self, lines: "list[str]" = None):
        self._lines = list(lines or [])
        self.closed = False
        self.written: "list[str]" = []

    def __iter__(self):
        return iter(self._lines)

    def write(self, s):
        self.written.append(s)

    def close(self):
        self.closed = True


class _FakeProc:
    def __init__(self, stdout_lines):
        self.args = ["claude"]
        self.stdin = _FakePipe()
        self.stdout = _FakePipe(stdout_lines)
        self.stderr = _FakePipe([])
        self.killed = False

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0


@pytest.fixture
def fake_cli(monkeypatch):
    """Replace the spawn with a canned stream-json transcript."""
    made: "list[_FakeProc]" = []

    def _popen(cmd, **kwargs):
        proc = _FakeProc([json.dumps({"type": "system", "subtype": "init"}) + "\n", json.dumps(_RESULT) + "\n"])
        proc.cmd = cmd
        proc.kwargs = kwargs
        made.append(proc)
        return proc

    monkeypatch.setattr(ccp.subprocess, "Popen", _popen)
    return made


def _run(provider, **kwargs):
    return asyncio.run(provider._generate_cli(prompt="say pong", **kwargs))


def _run_and_read(coro_factory, provider, *attrs):
    """Call, then read the per-call state INSIDE the same task.

    ``_last_usage`` and ``_last_result_message`` are PerCallAttr descriptors backed by
    contextvars, so a read after ``asyncio.run`` returns has left the task that wrote them and
    sees the default -- deliberately, that is how a concurrent ``generate_batch`` is kept from
    reading another request's numbers. The caller that consumes them (``generate`` itself, and
    glossum's llm_client through it) is inside the task; so is this.
    """

    async def _main():
        out = await coro_factory()
        return out, [getattr(provider, a) for a in attrs]

    return asyncio.run(_main())


class TestTheCliPathRecordsWhatItParsed:
    def test_the_result_message_reaches_the_provider(self, fake_cli):
        provider = ccp.ClaudeCodeProvider(model="sonnet")
        provider._claude_path = "claude"

        out, (rm,) = _run_and_read(lambda: provider._generate_cli(prompt="say pong"), provider, "_last_result_message")

        assert out == "pong"
        assert rm is not None, "parsed and then dropped: the recording line is unreachable again"
        assert rm.usage.cache_read_input_tokens == 41077
        assert rm.total_cost_usd == pytest.approx(0.0413)

    def test_generate_reports_the_real_numbers_not_an_estimate(self, fake_cli):
        provider = ccp.ClaudeCodeProvider(model="sonnet")
        provider._claude_path = "claude"

        out, (usage,) = _run_and_read(lambda: provider.generate(prompt="say pong"), provider, "_last_usage")

        assert out == "pong"
        assert usage["input_tokens"] == 191
        assert usage["output_tokens"] == 1608
        assert usage["cache_read_input_tokens"] == 41077
        assert provider.total_cost_usd == pytest.approx(0.0413)

    def test_the_session_summary_stops_claiming_a_zero_hit_rate(self, fake_cli):
        provider = ccp.ClaudeCodeProvider(model="sonnet")
        provider._claude_path = "claude"
        asyncio.run(provider.generate(prompt="say pong"))

        summary = provider.get_session_cost()

        assert summary["cache_hit_tokens"] == 41077
        assert summary["cache_creation_input_tokens"] == 13204
        assert summary["total_cost_usd"] == pytest.approx(0.0413)


class TestTheSpawnIsConfinedAndPermissionless:
    def test_permissions_are_not_bypassed(self, fake_cli):
        provider = ccp.ClaudeCodeProvider(model="sonnet")
        provider._claude_path = "claude"
        _run(provider)

        cmd = fake_cli[0].cmd
        assert "--dangerously-skip-permissions" not in cmd
        assert "--restricted" in cmd
        assert cmd[cmd.index("--permission-prompts") + 1] == "none"

    def test_the_child_does_not_inherit_the_callers_directory(self, fake_cli):
        provider = ccp.ClaudeCodeProvider(model="sonnet")
        provider._claude_path = "claude"
        _run(provider)

        assert fake_cli[0].kwargs.get("cwd"), "no cwd: --restricted would confine file tools to the caller's repo"

    def test_a_long_system_prompt_goes_to_a_file_not_argv(self, fake_cli):
        provider = ccp.ClaudeCodeProvider(model="sonnet")
        provider._claude_path = "claude"
        _run(provider, system="x" * 50000)

        cmd = fake_cli[0].cmd
        assert "--system-prompt-file" in cmd
        assert max(len(a) for a in cmd) < 32767

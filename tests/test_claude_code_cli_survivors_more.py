"""Stream, subprocess and error-classification pins for claude_code_cli (mutation survivors, round 2)."""

from __future__ import annotations

import json
import logging
import queue
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from pyutilz.llm import claude_code_cli as cli

_EMPTY = object()


class _Q:
    """Minimal stand-in for the stdout line queue: ``_EMPTY`` makes ``get`` time out once."""

    def __init__(self, items):
        self._items = list(items)

    def get(self, timeout=None):
        item = self._items.pop(0)
        if item is _EMPTY:
            raise queue.Empty
        return item


def _line(event):
    return json.dumps(event)


RESULT_OK = _line({"type": "result", "subtype": "success", "result": "hi"})


class TestWinErrorPermanence:
    @pytest.mark.parametrize("code", [2, 3, 5, 8, 87, 206, 267])
    def test_each_listed_windows_error_is_permanent(self, code):
        err = OSError("boom")
        err.winerror = code
        assert cli._is_transient_subprocess_error(err) is False

    def test_an_unlisted_windows_error_is_transient(self):
        err = OSError("boom")
        err.winerror = 1
        assert cli._is_transient_subprocess_error(err) is True


class TestFindClaudeExecutable:
    def test_looks_up_the_bare_claude_name(self, monkeypatch):
        seen = []

        def fake_which(name):
            seen.append(name)
            return "/usr/bin/claude" if name == "claude" else None

        monkeypatch.setattr(cli.shutil, "which", fake_which)
        assert cli._find_claude_executable() == "/usr/bin/claude"
        assert seen == ["claude"]


class TestResultOutcomeMessages:
    def test_error_text_is_taken_from_the_error_key_when_result_is_empty(self):
        event = {"type": "result", "is_error": True, "result": "", "error": "rate boom"}
        assert cli._result_outcome(event) == (None, "rate boom")

    def test_empty_success_message_names_the_subtype_in_full(self):
        event = {"type": "result", "subtype": "success", "result": "   "}
        _, err = cli._result_outcome(event)
        assert err == "the CLI reported success with an empty result (subtype='success')"

    def test_error_without_any_message_reports_the_subtype(self):
        event = {"type": "result", "is_error": True, "subtype": "x"}
        _, err = cli._result_outcome(event)
        assert err == "the CLI reported an error (subtype='x')"


class TestObserveStreamEventLogging:
    def test_init_event_logs_the_initialised_line(self, caplog):
        caplog.set_level(logging.DEBUG)
        cli._observe_stream_event({"type": "system", "subtype": "init"})
        assert "Claude CLI initialized" in caplog.text

    def test_other_system_events_do_not_claim_initialisation(self, caplog):
        caplog.set_level(logging.DEBUG)
        cli._observe_stream_event({"type": "system", "subtype": "other"})
        assert "Claude CLI initialized" not in caplog.text

    def test_rate_limit_event_is_logged_as_continuing(self, caplog):
        caplog.set_level(logging.DEBUG)
        assert cli._observe_stream_event({"type": "rate_limit_event"}) == 0
        assert "Claude CLI rate_limit_event (continuing)" in caplog.text


class TestConsumeCliStream:
    def test_a_quiet_poll_interval_does_not_end_the_wait(self):
        # The first poll finds nothing; the loop must keep waiting, not give up with no result.
        text, err, timed_out, _ = cli._consume_cli_stream(_Q([_EMPTY, RESULT_OK, None]), timeout=10)
        assert (text, err, timed_out) == ("hi", None, False)

    def test_blank_lines_are_skipped_not_treated_as_end_of_stream(self):
        text, _, _, _ = cli._consume_cli_stream(_Q(["", "   \n", RESULT_OK]), timeout=10)
        assert text == "hi"

    def test_a_non_json_line_is_logged_with_its_length_and_skipped(self, caplog):
        caplog.set_level(logging.DEBUG)
        text, _, _, _ = cli._consume_cli_stream(_Q(["garbage", RESULT_OK]), timeout=10)
        assert text == "hi"
        assert "Ignoring a stream line that is not JSON (7 chars): garbage" in caplog.text

    def test_thinking_blocks_across_several_assistant_events_are_summed(self):
        thinking = {"type": "assistant", "message": {"content": [{"type": "thinking"}]}}
        _, _, _, result_event = cli._consume_cli_stream(_Q([_line(thinking), _line(thinking), RESULT_OK]), timeout=10)
        assert result_event["thinking_blocks"] == 2


class TestUsageThinkingTokens:
    def test_dict_usage_reports_the_thinking_count(self):
        assert cli.usage_thinking_tokens({"output_tokens_details": {"thinking_tokens": 7}}) == 7

    def test_attribute_usage_reports_the_thinking_count(self):
        usage = SimpleNamespace(output_tokens_details=SimpleNamespace(thinking_tokens=7))
        assert cli.usage_thinking_tokens(usage) == 7

    def test_missing_details_gives_none(self):
        assert cli.usage_thinking_tokens({}) is None

    @pytest.mark.parametrize("value", [True, "5", None])
    def test_a_non_integer_or_boolean_count_gives_none(self, value):
        assert cli.usage_thinking_tokens({"output_tokens_details": {"thinking_tokens": value}}) is None


class _FakeProc:
    def __init__(self, returncode=None, args=("claude",)):
        self.pid = 4321
        self.args = list(args)
        self._rc = returncode
        self.waits = []
        self.kills = 0

    def poll(self):
        return self._rc

    def wait(self, timeout=None):
        self.waits.append(timeout)

    def kill(self):
        self.kills += 1


class _FakeThread:
    def __init__(self):
        self.joins = []

    def join(self, timeout=None):
        self.joins.append(timeout)


class TestKillProcessTree:
    def test_an_already_exited_process_is_left_alone(self, monkeypatch):
        monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: pytest.fail("no kill expected"))
        proc = _FakeProc(returncode=0)
        cli._kill_process_tree(proc)
        assert proc.kills == 0

    def test_windows_kill_walks_the_tree_by_pid_with_expected_flags(self, monkeypatch):
        calls = []
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(cli.subprocess, "run", lambda argv, **kw: calls.append((argv, kw)))
        proc = _FakeProc()
        cli._kill_process_tree(proc)
        assert calls == [(
            ["taskkill", "/T", "/F", "/PID", "4321"],
            {"capture_output": True, "check": False, "timeout": 10},
        )]
        assert proc.kills == 1


class TestRunCliPopenArguments:
    def test_the_child_is_spawned_with_text_utf8_and_no_shell(self, monkeypatch):
        captured = {}

        class _Ctx:
            def __enter__(self):
                return _FakeProc()

            def __exit__(self, *exc):
                return False

        def fake_popen(argv, **kw):
            captured.update(kw)
            return _Ctx()

        monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(cli, "_stream_one_call", lambda *a, **k: (0, "ok", "", None))
        monkeypatch.setattr(sys, "platform", "win32")
        cli.run_cli(["claude"], "p", {}, ".", 5.0, [], threading.Event())
        assert captured["text"] is True
        assert captured["encoding"] == "utf-8"
        assert captured["errors"] == "replace"
        assert captured["shell"] is False
        assert captured["start_new_session"] is False


class TestReapProcess:
    def test_reaping_kills_the_tree_waits_and_joins_with_five_seconds(self, monkeypatch):
        killed = []
        monkeypatch.setattr(cli, "_kill_process_tree", lambda p: killed.append(p))
        proc, rt, st = _FakeProc(), _FakeThread(), _FakeThread()
        cli._reap_process(proc, rt, st)
        assert killed == [proc]
        assert proc.waits == [5]
        assert rt.joins == [5]
        assert st.joins == [5]


class TestStreamOneCall:
    def _patch(self, monkeypatch, *, timed_out=False, stderr_chunks=None, reap_calls=None):
        r_thread, s_thread = _FakeThread(), _FakeThread()
        chunks = stderr_chunks if stderr_chunks is not None else []
        monkeypatch.setattr(cli, "_write_prompt", lambda proc, prompt: None)
        monkeypatch.setattr(cli, "_start_stdout_reader", lambda proc: (_Q([]), r_thread))
        monkeypatch.setattr(cli, "_start_stderr_reader", lambda proc: (chunks, s_thread))
        monkeypatch.setattr(
            cli, "_consume_cli_stream",
            lambda q, t, c: (None, None, True, None) if timed_out else ("ok", None, False, {"x": 1}),
        )
        monkeypatch.setattr(
            cli, "_reap_process",
            lambda proc, rt, st: (reap_calls.append((proc, rt, st)) if reap_calls is not None else None),
        )
        return r_thread, s_thread

    def test_reaping_receives_the_stdout_and_stderr_threads_in_order(self, monkeypatch):
        reaps = []
        r_thread, s_thread = self._patch(monkeypatch, reap_calls=reaps)
        proc = _FakeProc()
        cli._stream_one_call(proc, "p", 5.0, threading.Event())
        assert reaps == [(proc, r_thread, s_thread)]

    def test_stderr_is_capped_to_the_last_two_thousand_characters(self, monkeypatch):
        self._patch(monkeypatch, stderr_chunks=["X" * 500, "Y" * 2000])
        _, _, stderr, _ = cli._stream_one_call(_FakeProc(), "p", 5.0, threading.Event())
        assert stderr == "Y" * 2000

    def test_a_timeout_names_the_process_args_and_the_limit(self, monkeypatch):
        self._patch(monkeypatch, timed_out=True)
        with pytest.raises(subprocess.TimeoutExpired) as exc:
            cli._stream_one_call(_FakeProc(args=("claude", "-p")), "p", 1.5, threading.Event())
        assert exc.value.cmd == ["claude", "-p"]
        assert exc.value.timeout == 1.5


class TestTimezonePatternRequiresLetters:
    def test_a_slash_with_no_region_name_is_not_a_zone(self):
        assert cli._TIMEZONE_PATTERN.search("resets 3pm (/Zone)") is None

    def test_a_proper_region_zone_is_captured(self):
        assert cli._TIMEZONE_PATTERN.search("resets 3pm (America/New_York)").group(1) == "America/New_York"

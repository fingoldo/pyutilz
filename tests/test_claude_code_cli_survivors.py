"""Reset-time parsing and rate-limit detection pins for claude_code_cli (mutation survivors)."""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from pyutilz.llm import claude_code_cli as cli

BASE = datetime(2026, 1, 15, 10, 0, 0)
BASE_UTC = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return BASE
        return BASE_UTC.astimezone(tz)


@pytest.fixture
def frozen_clock(monkeypatch):
    monkeypatch.setattr(cli, "datetime", _FrozenDatetime)


class TestHour24:
    @pytest.mark.parametrize(
        ("hour", "ampm", "expected"),
        [(3, "pm", 15), (12, "am", 0), (12, "pm", 12), (5, "am", 5), (9, None, 9), (7, "AM", 7), (1, "PM", 13)],
    )
    def test_twelve_hour_clock_becomes_twenty_four_hour(self, hour, ampm, expected):
        assert cli._hour_24(hour, ampm) == expected


class TestResetTimezone:
    def test_an_unknown_zone_falls_back_to_none_and_logs_the_name(self, caplog):
        caplog.set_level("DEBUG")
        assert cli._reset_timezone("resets 3pm (Nowhere/NoCity)") is None
        assert "Could not load timezone Nowhere/NoCity, using local time" in caplog.text

    def test_no_zone_in_the_text_gives_none(self):
        assert cli._reset_timezone("resets 3pm") is None


class TestAbsoluteResetTime:
    def test_a_later_date_in_the_same_year_is_kept(self):
        assert cli._absolute_reset_time(BASE, 9, 0, "Feb", "3") == datetime(2026, 2, 3, 9, 0)

    def test_a_date_already_passed_rolls_to_next_year(self):
        assert cli._absolute_reset_time(BASE, 9, 0, "Jan", "10") == datetime(2027, 1, 10, 9, 0)

    def test_a_reset_exactly_at_now_counts_as_passed_and_rolls_to_next_year(self):
        assert cli._absolute_reset_time(BASE, 10, 0, "Jan", "15") == datetime(2027, 1, 15, 10, 0)

    def test_an_impossible_date_gives_none(self):
        assert cli._absolute_reset_time(BASE, 9, 0, "Feb", "30") is None

    def test_an_impossible_date_is_warned_with_the_default_wait_message(self, caplog):
        caplog.set_level("WARNING")
        cli._absolute_reset_time(BASE, 9, 0, "Feb", "30")
        assert caplog.records[-1].getMessage() == "Could not interpret rate-limit reset date 'Feb'-'30'; using the default wait"

    @pytest.mark.parametrize(
        ("month", "number"),
        [("jan", 1), ("feb", 2), ("mar", 3), ("apr", 4), ("may", 5), ("jun", 6), ("jul", 7), ("aug", 8), ("sep", 9), ("oct", 10), ("nov", 11), ("dec", 12)],
    )
    def test_every_month_abbreviation_maps_to_its_number(self, month, number):
        # Base year-end guard: a date later in the year is kept in the current year, so the month alone decides the result.
        reset = cli._absolute_reset_time(datetime(2026, 1, 1, 0, 0, 0), 9, 0, month, "5")
        assert reset.month == number


class TestResetWaitSeconds:
    def test_no_reset_phrase_gives_none(self, frozen_clock):
        assert cli._parse_reset_wait_seconds("something else entirely") is None

    def test_an_afternoon_reset_waits_until_then_plus_one_minute(self, frozen_clock):
        assert cli._parse_reset_wait_seconds("Your limit resets 3pm") == 5 * 3600 + 60

    def test_a_clock_time_with_minutes_waits_until_that_minute(self, frozen_clock):
        assert cli._parse_reset_wait_seconds("Your limit resets 3:30pm") == 5 * 3600 + 30 * 60 + 60

    def test_a_reset_time_equal_to_now_counts_as_passed_and_waits_a_day(self, frozen_clock):
        assert cli._parse_reset_wait_seconds("resets 10am") == 24 * 3600 + 60

    def test_the_wait_is_truncated_to_whole_seconds(self, monkeypatch):
        class _Clock(_FrozenDatetime):
            @classmethod
            def now(cls, tz=None):
                return BASE.replace(microsecond=1)

        monkeypatch.setattr(cli, "datetime", _Clock)
        assert cli._parse_reset_wait_seconds("resets 3pm") == 5 * 3600 - 1 + 60

    def test_a_zone_in_the_text_measures_the_wait_in_that_zone(self, frozen_clock):
        # 10:00 UTC is 19:00 in Tokyo, so a 3pm Tokyo reset has passed and waits until tomorrow.
        assert cli._parse_reset_wait_seconds("resets 3pm (Asia/Tokyo)") == 20 * 3600 + 60

    def test_an_earlier_clock_time_waits_until_tomorrow(self, frozen_clock):
        assert cli._parse_reset_wait_seconds("resets 9am") == 23 * 3600 + 60

    def test_an_explicit_date_waits_until_that_day(self, frozen_clock):
        assert cli._parse_reset_wait_seconds("resets Feb 3, 9am") == 19 * 86400 - 3600 + 60

    def test_an_impossible_explicit_date_gives_none(self, frozen_clock):
        assert cli._parse_reset_wait_seconds("resets Feb 30, 9am") is None


class TestFindClaudeExecutable:
    def test_a_claude_on_the_path_is_used_first(self, monkeypatch):
        monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/claude")
        assert cli._find_claude_executable() == "/usr/bin/claude"

    def test_an_npm_install_under_appdata_is_found_when_not_on_the_path(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli.shutil, "which", lambda name: None)
        monkeypatch.setattr(cli.sys, "platform", "win32")
        monkeypatch.setenv("APPDATA", str(tmp_path))
        npm = tmp_path / "npm"
        npm.mkdir()
        (npm / "claude.cmd").write_text("@echo off")
        assert cli._find_claude_executable() == str(npm / "claude.cmd")

    def test_a_bare_claude_file_under_npm_is_found_when_there_is_no_cmd(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli.shutil, "which", lambda name: None)
        monkeypatch.setattr(cli.sys, "platform", "win32")
        monkeypatch.setenv("APPDATA", str(tmp_path))
        npm = tmp_path / "npm"
        npm.mkdir()
        (npm / "claude").write_text("#!/bin/sh")
        assert cli._find_claude_executable() == str(npm / "claude")

    def test_a_user_profile_install_is_found_when_appdata_has_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli.shutil, "which", lambda name: None)
        monkeypatch.setattr(cli.sys, "platform", "win32")
        monkeypatch.setenv("APPDATA", str(tmp_path / "missing"))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))  # os.path.expanduser reads HOME off Windows
        roaming_npm = tmp_path / "AppData" / "Roaming" / "npm"
        roaming_npm.mkdir(parents=True)
        (roaming_npm / "claude.cmd").write_text("@echo off")
        assert cli._find_claude_executable() == str(roaming_npm / "claude.cmd")

    def test_a_missing_binary_raises_naming_the_install_command(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli.shutil, "which", lambda name: None)
        monkeypatch.setattr(cli.sys, "platform", "win32")
        monkeypatch.setenv("APPDATA", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        with pytest.raises(FileNotFoundError, match=re.escape("Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code")):
            cli._find_claude_executable()


class TestRetryConstants:
    def test_three_timeout_retries_is_the_budget(self):
        assert cli.MAX_TIMEOUT_RETRIES == 3

    def test_a_too_long_command_line_on_windows_is_not_retried(self):
        error = OSError("filename too long")
        error.winerror = 206
        assert cli._is_transient_subprocess_error(error) is False


class TestPermanentErrorCodes:
    @pytest.mark.parametrize("winerror", sorted(cli._PERMANENT_WINERRORS))
    def test_every_permanent_windows_code_is_not_retried(self, winerror):
        error = OSError("permanent")
        error.winerror = winerror
        assert cli._is_transient_subprocess_error(error) is False

    @pytest.mark.parametrize("errno_value", sorted(cli._PERMANENT_ERRNOS))
    def test_every_permanent_posix_errno_is_not_retried(self, errno_value):
        assert cli._is_transient_subprocess_error(OSError(errno_value, "permanent")) is False


class TestTransientSubprocessErrors:
    def test_a_timeout_is_worth_retrying(self):
        assert cli._is_transient_subprocess_error(cli.subprocess.TimeoutExpired("claude", 5)) is True

    def test_a_network_blip_is_worth_retrying(self):
        assert cli._is_transient_subprocess_error(ConnectionResetError("reset")) is True

    def test_a_missing_binary_is_not_worth_retrying(self):
        assert cli._is_transient_subprocess_error(FileNotFoundError("no claude")) is False

    def test_a_permanent_errno_is_not_worth_retrying(self):
        assert cli._is_transient_subprocess_error(OSError(cli.errno.EACCES, "denied")) is False

    def test_a_transient_errno_is_worth_retrying(self):
        assert cli._is_transient_subprocess_error(OSError(cli.errno.EAGAIN, "again")) is True

    def test_a_non_os_error_is_not_retried(self):
        assert cli._is_transient_subprocess_error(ValueError("bad")) is False


class TestToolUseTripwire:
    def test_a_tool_use_block_without_a_name_is_reported_with_a_placeholder(self):
        event = {"message": {"content": [{"type": "tool_use"}]}}
        with pytest.raises(cli.ClaudeCodeToolUseError) as info:
            cli._raise_on_cli_tool_use(event)
        assert str(info.value) == "Claude Code returned a tool-use block ('?'); this provider is text-generation-only"

    def test_a_non_list_content_or_a_non_dict_block_raises_nothing(self):
        assert cli._raise_on_cli_tool_use({"message": {"content": "text"}}) is None
        assert cli._raise_on_cli_tool_use({"message": {"content": ["plain string"]}}) is None
        assert cli._raise_on_cli_tool_use({"message": {"content": [{"type": "text"}]}}) is None

    def test_a_thinking_count_ignores_non_dict_blocks_and_a_non_list_content(self):
        assert cli._count_thinking_blocks({"message": {"content": ["x", {"type": "thinking"}]}}) == 1
        assert cli._count_thinking_blocks({"message": {"content": "x"}}) == 0
        assert cli._count_thinking_blocks({}) == 0


class TestStreamEvents:
    def test_thinking_blocks_are_counted_on_an_assistant_event(self):
        event = {"type": "assistant", "message": {"content": [{"type": "thinking"}, {"type": "text"}, {"type": "thinking"}]}}
        assert cli._observe_stream_event(event) == 2

    def test_a_non_list_content_counts_no_thinking(self):
        assert cli._count_thinking_blocks({"message": {"content": "text"}}) == 0

    def test_a_tool_use_block_raises_the_tripwire(self):
        event = {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash"}]}}
        with pytest.raises(cli.ClaudeCodeToolUseError, match="Bash"):
            cli._observe_stream_event(event)

    def test_a_rate_limit_event_is_only_logged_and_counts_nothing(self, caplog):
        caplog.set_level("DEBUG")
        assert cli._observe_stream_event({"type": "rate_limit_event"}) == 0
        assert "rate_limit_event (continuing)" in caplog.text


class TestResultOutcome:
    def test_a_success_returns_its_text(self):
        assert cli._result_outcome({"subtype": "success", "result": "hello"}) == ("hello", None)

    def test_an_empty_success_is_an_error_not_an_answer(self):
        text, error = cli._result_outcome({"subtype": "success", "result": "   "})
        assert text is None
        assert "empty result" in error

    def test_an_error_flag_outranks_a_success_subtype(self):
        assert cli._result_outcome({"subtype": "success", "is_error": True, "result": "limit hit"}) == (None, "limit hit")

    def test_an_error_without_text_names_the_subtype(self):
        assert cli._result_outcome({"is_error": True, "subtype": "boom"}) == (None, "the CLI reported an error (subtype='boom')")

    def test_a_failure_falls_back_to_the_error_then_the_subtype(self):
        assert cli._result_outcome({"subtype": "x", "error": "bad"}) == (None, "bad")
        assert cli._result_outcome({"subtype": "x"}) == (None, "x")


class TestConsumeStream:
    def test_the_result_event_is_returned_with_its_thinking_count(self):
        import queue

        q = queue.Queue()
        q.put('{"type": "assistant", "message": {"content": [{"type": "thinking"}]}}')
        q.put('{"type": "result", "subtype": "success", "result": "answer"}')
        text, error, timed_out, event = cli._consume_cli_stream(q, timeout=5.0)
        assert (text, error, timed_out) == ("answer", None, False)
        assert event["thinking_blocks"] == 1

    def test_a_non_json_line_is_skipped(self):
        import queue

        q = queue.Queue()
        q.put("not json at all")
        q.put('{"type": "result", "subtype": "success", "result": "ok"}')
        assert cli._consume_cli_stream(q, timeout=5.0)[0] == "ok"

    def test_an_expired_deadline_reports_a_timeout(self):
        import queue

        assert cli._consume_cli_stream(queue.Queue(), timeout=0.0) == (None, None, True, None)

    def test_a_cancelled_call_returns_nothing(self):
        import queue
        import threading

        evt = threading.Event()
        evt.set()
        assert cli._consume_cli_stream(queue.Queue(), timeout=5.0, cancel_evt=evt) == (None, None, False, None)


class TestUsageFields:
    def test_a_dict_usage_field_is_read_as_an_integer(self):
        assert cli.usage_int({"input_tokens": 7.9}, "input_tokens") == 7

    def test_a_missing_or_boolean_usage_field_is_zero(self):
        assert cli.usage_int({}, "input_tokens") == 0
        assert cli.usage_int({"input_tokens": True}, "input_tokens") == 0

    def test_an_attribute_usage_object_is_read_too(self):
        class Usage:
            input_tokens = 4

        assert cli.usage_int(Usage(), "input_tokens") == 4

    def test_the_thinking_token_count_is_read_from_its_details(self):
        assert cli.usage_thinking_tokens({"output_tokens_details": {"thinking_tokens": 12}}) == 12
        assert cli.usage_thinking_tokens({}) is None


class TestCliResultMessage:
    def test_the_usage_counts_and_cost_come_off_the_result_event(self):
        event = {
            "usage": {
                "input_tokens": 10,
                "output_tokens": 4,
                "cache_creation_input_tokens": 3,
                "cache_read_input_tokens": 2,
                "output_tokens_details": {"thinking_tokens": 1},
            },
            "total_cost_usd": 0.5,
            "session_id": "s1",
            "num_turns": 2,
            "duration_ms": 900,
        }
        msg = cli._CliResultMessage(event)
        assert (msg.usage.input_tokens, msg.usage.output_tokens) == (10, 4)
        assert (msg.usage.cache_creation_input_tokens, msg.usage.cache_read_input_tokens) == (3, 2)
        assert msg.usage.output_tokens_details == {"thinking_tokens": 1}
        assert (msg.total_cost_usd, msg.session_id, msg.num_turns, msg.duration_ms) == (0.5, "s1", 2, 900)

    def test_a_missing_usage_block_reads_as_zeros_and_no_cost(self):
        msg = cli._CliResultMessage({})
        assert (msg.usage.input_tokens, msg.usage.output_tokens, msg.total_cost_usd) == (0, 0, 0.0)


class _FakeProc:
    pid = 4242

    def __init__(self, running: bool) -> None:
        self._running = running
        self.killed = False

    def poll(self):
        return None if self._running else 0

    def kill(self) -> None:
        self.killed = True


class TestKillProcessTree:
    def test_an_already_exited_process_is_left_alone(self, monkeypatch):
        calls = []
        monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: calls.append(a))
        proc = _FakeProc(running=False)
        cli._kill_process_tree(proc)
        assert calls == []
        assert proc.killed is False

    def test_a_running_process_is_killed_as_a_tree_and_then_directly(self, monkeypatch):
        calls = []
        monkeypatch.setattr(cli.subprocess, "run", lambda argv, **k: calls.append(argv))
        proc = _FakeProc(running=True)
        cli._kill_process_tree(proc)
        assert proc.killed is True
        if cli.sys.platform == "win32":
            assert calls == [["taskkill", "/T", "/F", "/PID", "4242"]]


class _Stdin:
    def __init__(self) -> None:
        self.written = ""
        self.closed = False

    def write(self, text: str) -> None:
        self.written += text

    def close(self) -> None:
        self.closed = True


class _FakeChild:
    pid = 1
    args = ["claude"]

    def __init__(self, stdout_text: str, stderr_text: str = "") -> None:
        import io

        self.stdin = _Stdin()
        self.stdout = io.StringIO(stdout_text)
        self.stderr = io.StringIO(stderr_text)

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0

    def kill(self) -> None:
        pass


class TestStreamOneCall:
    def test_a_success_event_gives_the_answer_and_the_stderr_tail(self):
        import threading

        child = _FakeChild('{"type": "result", "subtype": "success", "result": "answer"}\n', stderr_text="warn")
        code, text, stderr, event = cli._stream_one_call(child, "prompt", 5.0, threading.Event())
        assert (code, text, stderr) == (0, "answer", "warn")
        assert event["result"] == "answer"

    def test_the_prompt_is_written_to_stdin_and_stdin_is_closed(self):
        import threading

        child = _FakeChild('{"type": "result", "subtype": "success", "result": "ok"}\n')
        cli._stream_one_call(child, "hello prompt", 5.0, threading.Event())
        assert child.stdin.written == "hello prompt"
        assert child.stdin.closed is True

    def test_an_error_result_raises_with_the_error_text(self):
        import threading

        child = _FakeChild('{"type": "result", "is_error": true, "result": "limit hit"}\n')
        with pytest.raises(RuntimeError, match="Claude CLI error: limit hit"):
            cli._stream_one_call(child, "p", 5.0, threading.Event())

    def test_a_stream_with_no_result_raises_naming_the_stderr(self):
        import threading

        child = _FakeChild("", stderr_text="boom on stderr")
        with pytest.raises(RuntimeError, match=re.escape("produced no result. stderr: boom on stderr")):
            cli._stream_one_call(child, "p", 5.0, threading.Event())

    def test_an_expired_deadline_raises_a_timeout(self):
        import threading

        child = _FakeChild('{"type": "result", "subtype": "success", "result": "late"}\n')
        with pytest.raises(cli.subprocess.TimeoutExpired):
            cli._stream_one_call(child, "p", 0.0, threading.Event())

    def test_a_cancelled_call_raises_cancelled(self):
        import asyncio
        import threading

        evt = threading.Event()
        evt.set()
        child = _FakeChild('{"type": "result", "subtype": "success", "result": "x"}\n')
        with pytest.raises(asyncio.CancelledError):
            cli._stream_one_call(child, "p", 5.0, evt)


class TestPipeReaders:
    def test_stdout_lines_are_queued_and_end_with_the_sentinel(self):
        child = _FakeChild('{"a": 1}\n{"b": 2}\n')
        line_q, thread = cli._start_stdout_reader(child)
        thread.join(timeout=5)
        items = []
        while not line_q.empty():
            items.append(line_q.get())
        assert items == ['{"a": 1}\n', '{"b": 2}\n', None]

    def test_stderr_is_collected_in_full(self):
        child = _FakeChild("", stderr_text="warn one\nwarn two\n")
        chunks, thread = cli._start_stderr_reader(child)
        thread.join(timeout=5)
        assert "".join(chunks) == "warn one\nwarn two\n"


class _Ctx(_FakeChild):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestRunCli:
    def test_run_cli_spawns_the_child_and_returns_its_result(self, monkeypatch):
        child = _Ctx('{"type": "result", "subtype": "success", "result": "via run"}\n', stderr_text="note")
        monkeypatch.setattr(cli.subprocess, "Popen", lambda *a, **k: child)
        import threading

        holder: list = []
        code, text, stderr, event = cli.run_cli(["claude"], "p", {}, ".", 5.0, holder, threading.Event())
        assert (code, text, stderr) == (0, "via run", "note")
        assert holder == [child]
        assert event["result"] == "via run"


class TestRateLimitDetection:
    @pytest.mark.parametrize("message", ["You hit your limit", "rate limit reached", "quota exceeded", "usage limit"])
    def test_the_known_limit_wordings_are_detected(self, message):
        assert cli._is_rate_limit_error(RuntimeError(message)) is True

    def test_an_unrelated_error_is_not_a_rate_limit(self):
        assert cli._is_rate_limit_error(RuntimeError("all good")) is False

    def test_the_limit_wording_on_stderr_is_detected(self):
        error = RuntimeError("process failed")
        error.stderr = "quota exceeded for this key"
        assert cli._is_rate_limit_error(error) is True

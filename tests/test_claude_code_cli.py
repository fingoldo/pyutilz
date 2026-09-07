"""``claude_code_cli``: the stream consumer, the result adapter, and the error predicate.

Named for the module it covers (the test-source-parity gate wants one file per production
module); the subprocess-level behaviour of ``run_cli`` is covered by
``test_claude_cli_usage_end_to_end.py``, which drives it through the provider against a fake
spawn.

The CLI ``result`` event carries real usage and cost; none of it was being kept.

``_consume_cli_stream`` read the event's ``result`` field and returned the text alone, so on the
CLI path -- the path actually in use, the SDK being optional -- ``generate()`` found no
``_last_result_message``, fell back to a tiktoken estimate for both token counts, recorded zero for
both cache-token figures and added nothing to ``total_cost_usd``. The SDK path had captured all of
it since it was written. These fail against the pre-fix consumer.
"""

from __future__ import annotations

import queue

import pytest

from pyutilz.llm.claude_code_cli import _CliResultMessage, _consume_cli_stream, _is_transient_subprocess_error

_EVENT = {
    "type": "result",
    "subtype": "success",
    "result": "hello",
    "total_cost_usd": 0.0413,
    "duration_ms": 4412,
    "num_turns": 1,
    "session_id": "sess-7",
    "usage": {
        "input_tokens": 191,
        "output_tokens": 1608,
        "cache_creation_input_tokens": 13204,
        "cache_read_input_tokens": 41077,
    },
}


def _consume(event: dict):
    import json

    q: "queue.Queue" = queue.Queue()
    q.put(json.dumps(event))
    return _consume_cli_stream(q, timeout=5.0)


class TestTheResultEventSurvivesTheConsumer:
    def test_the_event_is_handed_back_alongside_the_text(self):
        text, error, timed_out, event = _consume(_EVENT)

        assert (text, error, timed_out) == ("hello", None, False)
        assert event is not None, "the consumer dropped the event, so usage and cost are unrecoverable"
        assert event["usage"]["cache_read_input_tokens"] == 41077

    def test_an_error_result_is_kept_too(self):
        """An errored call still bills for the tokens it burned."""
        text, error, timed_out, event = _consume({"type": "result", "subtype": "error_max_turns", "result": "nope", "usage": {"input_tokens": 12}})

        assert (text, error) == (None, "nope")
        assert event is not None and event["usage"]["input_tokens"] == 12

    def test_a_timeout_reports_no_event(self):
        q: "queue.Queue" = queue.Queue()
        q.put(None)

        assert _consume_cli_stream(q, timeout=5.0) == (None, None, False, None)


class TestTheAdapterReadsLikeAResultMessage:
    """``generate()`` has one usage-extraction branch, written against the SDK's ResultMessage."""

    def test_every_field_generate_reads_is_present(self):
        rm = _CliResultMessage(_EVENT)

        assert rm.usage.input_tokens == 191
        assert rm.usage.output_tokens == 1608
        assert rm.usage.cache_creation_input_tokens == 13204
        assert rm.usage.cache_read_input_tokens == 41077
        assert rm.total_cost_usd == pytest.approx(0.0413)

    def test_the_session_metadata_comes_along(self):
        rm = _CliResultMessage(_EVENT)

        assert (rm.session_id, rm.num_turns, rm.duration_ms) == ("sess-7", 1, 4412)

    def test_a_missing_usage_block_is_zeros_not_a_crash(self):
        """Older CLI builds, and the error subtypes, may omit it entirely."""
        rm = _CliResultMessage({"type": "result", "subtype": "success", "result": ""})

        assert rm.usage.input_tokens == 0
        assert rm.usage.cache_read_input_tokens == 0
        assert rm.total_cost_usd == 0.0

    def test_explicit_nulls_are_tolerated(self):
        rm = _CliResultMessage({"usage": {"input_tokens": None}, "total_cost_usd": None})

        assert rm.usage.input_tokens == 0
        assert rm.total_cost_usd == 0.0


class TestOnlyARetryableFailureIsRetried:
    """WinError 206 was retried thirteen times and burned 2457 seconds on a permanent failure."""

    def test_a_missing_binary_is_permanent(self):
        assert _is_transient_subprocess_error(FileNotFoundError("Claude CLI not found. Install with: npm install -g ...")) is False

    def test_a_command_line_over_the_windows_limit_is_permanent(self):
        exc = OSError("The filename or extension is too long")
        exc.winerror = 206

        assert _is_transient_subprocess_error(exc) is False

    def test_access_denied_is_permanent(self):
        exc = OSError("Access is denied")
        exc.winerror = 5

        assert _is_transient_subprocess_error(exc) is False

    def test_a_timeout_is_transient(self):
        import subprocess

        assert _is_transient_subprocess_error(subprocess.TimeoutExpired("claude", 2400)) is True

    def test_a_connection_error_is_transient_whatever_its_errno(self):
        assert _is_transient_subprocess_error(ConnectionResetError(104, "Connection reset by peer")) is True

    def test_an_unrecognised_oserror_is_transient(self):
        """Unknown means "might work next time"; the permanent list is the closed set."""
        exc = OSError(99, "something new")

        assert _is_transient_subprocess_error(exc) is True

    def test_a_non_os_error_is_not_this_predicate_business(self):
        assert _is_transient_subprocess_error(ValueError("nonsense")) is False

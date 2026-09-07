"""The CLI ``result`` event carries real usage and cost; none of it was being kept.

``_consume_cli_stream`` read the event's ``result`` field and returned the text alone, so on the
CLI path -- the path actually in use, the SDK being optional -- ``generate()`` found no
``_last_result_message``, fell back to a tiktoken estimate for both token counts, recorded zero for
both cache-token figures and added nothing to ``total_cost_usd``. The SDK path had captured all of
it since it was written. These fail against the pre-fix consumer.
"""

from __future__ import annotations

import queue

import pytest

from pyutilz.llm.claude_code_provider import _CliResultMessage, _consume_cli_stream

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

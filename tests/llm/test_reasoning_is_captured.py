"""Reasoning text is billed output and must reach the caller, not be dropped on the floor.

Both extraction points read the answer and walked past the thinking beside it: the buffered unwrap
took ``message.content`` and ignored ``message.reasoning``, and the streaming chunk handler yielded
``delta.content`` and ignored ``delta.reasoning``. Upstreams bill reasoning as output tokens -- and
OpenRouter counts them INSIDE ``completion_tokens`` -- so a caller holding only that sum cannot say
what it paid for thinking rather than for an answer.

Measured 2026-09-13 on one ``z-ai/glm-5.3-flash`` enrichment call: 39,641 of 65,577 output tokens
were reasoning, 118,004 characters of it, and every character was discarded. The same call also
spent its first 393 seconds thinking before emitting any content, which three separate runs misread
as a hung request.
"""

from __future__ import annotations

import pytest

from pyutilz.llm.openai_compat import OpenAICompatibleProvider


class _Concrete(OpenAICompatibleProvider):
    """The abstract pricing hooks filled in; nothing here reads them."""

    def _input_cost_per_1m(self, model: str) -> float:
        return 0.0

    def _output_cost_per_1m(self, model: str) -> float:
        return 0.0


def _bare_provider() -> OpenAICompatibleProvider:
    """An instance without __init__: no API key or HTTP client is needed to fold one SSE chunk."""
    return object.__new__(_Concrete)


def _chunk(**delta: object) -> dict:
    return {"choices": [{"delta": delta}]}


class TestTheStreamingPath:
    def test_a_reasoning_delta_is_accumulated(self) -> None:
        provider = _bare_provider()

        provider._apply_stream_chunk(_chunk(reasoning="first "), {})
        provider._apply_stream_chunk(_chunk(reasoning="second"), {})

        assert provider.last_reasoning_text == "first second"

    def test_reasoning_is_not_yielded_as_content(self) -> None:
        """`generate_stream` is a stream of the ANSWER; reasoning must not be spliced into it."""
        provider = _bare_provider()

        assert provider._apply_stream_chunk(_chunk(reasoning="thinking"), {}) is None

    def test_content_still_comes_back(self) -> None:
        provider = _bare_provider()

        assert provider._apply_stream_chunk(_chunk(content="Paris"), {}) == "Paris"

    def test_a_chunk_carrying_both_keeps_them_apart(self) -> None:
        provider = _bare_provider()

        content = provider._apply_stream_chunk(_chunk(reasoning="because France", content="Paris"), {})

        assert content == "Paris"
        assert provider.last_reasoning_text == "because France"

    def test_a_model_that_never_reasons_records_nothing(self) -> None:
        provider = _bare_provider()

        provider._apply_stream_chunk(_chunk(content="Paris"), {})

        assert provider.last_reasoning_text is None

    @pytest.mark.parametrize("value", [None, "", 0, [], {}])
    def test_a_non_string_or_empty_reasoning_is_ignored(self, value: object) -> None:
        """An upstream that sends the key with nothing in it must not create an empty record."""
        provider = _bare_provider()

        provider._apply_stream_chunk(_chunk(reasoning=value), {})

        assert provider.last_reasoning_text is None

    def test_nothing_is_truncated(self) -> None:
        """The measured call arrived in tens of thousands of deltas; a cap would make it a sample."""
        provider = _bare_provider()

        for _ in range(5000):
            provider._apply_stream_chunk(_chunk(reasoning="step. "), {})

        assert len(provider.last_reasoning_text) == 5000 * len("step. ")

    def test_fragments_are_kept_apart_rather_than_concatenated(self) -> None:
        """The observable difference between the linear implementation and the quadratic one.

        Building the text by `text = text + fragment` copies the whole accumulation on every delta,
        and a reasoning model sends tens of thousands of them: measured on the live pipeline
        2026-09-13, that dropped throughput from ~300 characters per second to ~60 and turned a call
        that answers in 538 s into one still unfinished when a 2,999 s budget cancelled it.

        Asserted as a shape rather than a duration: N deltas leave N fragments, which a concatenating
        implementation cannot satisfy however fast the machine running this happens to be.
        """
        provider = _bare_provider()
        chunk = _chunk(reasoning="x" * 200)

        for _ in range(1_000):
            provider._apply_stream_chunk(chunk, {})

        assert len(provider._reasoning_fragments) == 1_000, "the deltas were merged instead of kept"
        assert provider.last_reasoning_text == "x" * 200 * 1_000


class TestEveryFieldAnUpstreamMayUse:
    """One spelling is not enough: a route using another is captured as nothing, with no error.

    OpenRouter normalises most routes to `reasoning`; `reasoning_content` is the DeepSeek-style
    spelling, and some routes send only the structured `reasoning_details`. Reading one field would
    reproduce, on the next model we try, exactly the silence this capture was written to end.
    """

    def test_the_deepseek_spelling_is_read(self) -> None:
        provider = _bare_provider()

        provider._apply_stream_chunk(_chunk(reasoning_content="thinking in another field"), {})

        assert provider.last_reasoning_text == "thinking in another field"

    def test_structured_details_are_read(self) -> None:
        provider = _bare_provider()

        provider._apply_stream_chunk(
            _chunk(reasoning_details=[{"type": "reasoning.text", "text": "first "}, {"type": "reasoning.text", "text": "second"}]),
            {},
        )

        assert provider.last_reasoning_text == "first second"

    def test_a_route_sending_both_records_the_text_once(self) -> None:
        """`reasoning` and `reasoning_details` are the same thought twice; adding them doubles it."""
        provider = _bare_provider()

        provider._apply_stream_chunk(
            _chunk(reasoning="the thought", reasoning_details=[{"type": "reasoning.text", "text": "the thought"}]),
            {},
        )

        assert provider.last_reasoning_text == "the thought"

    def test_a_detail_entry_without_text_contributes_nothing(self) -> None:
        """An encrypted or redacted block must not become the string 'None' in the record."""
        provider = _bare_provider()

        provider._apply_stream_chunk(
            _chunk(reasoning_details=[{"type": "reasoning.encrypted", "data": "…"}, {"type": "reasoning.text", "text": "kept"}]),
            {},
        )

        assert provider.last_reasoning_text == "kept"

    def test_the_buffered_path_reads_the_same_fields(self) -> None:
        from pyutilz.llm import _reasoning

        assert _reasoning.from_message({"reasoning_content": "buffered"}) == "buffered"
        assert _reasoning.from_message({"reasoning_details": [{"text": "structured"}]}) == "structured"
        assert _reasoning.from_message({"content": "answer only"}) is None

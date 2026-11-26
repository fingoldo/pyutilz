"""Mutation-survivor pins for openai_compat: the streamed path -- chunks, retries, repairs, close and error messages."""

from pyutilz.core import jsonlib
import logging
from unittest.mock import AsyncMock, MagicMock
import pytest
from pyutilz.llm.exceptions import LLMProviderError
from pyutilz.llm.openai_compat import OpenAICompatibleProvider
from tests.openai_compat_support import _MockStreamResponse, _TestProvider, _make_provider, _mock_response
httpx = pytest.importorskip("httpx")


httpx = pytest.importorskip("httpx")


class TestStreamDefaultsSurvivors:
    """A plain generate_stream call sends temperature 0.7, no max_tokens and no response_format (openai_compat.py 428-430)."""

    @pytest.mark.asyncio
    async def test_a_plain_stream_call_sends_the_documented_defaults(self):
        p = _make_provider()
        p._client = AsyncMock()
        p._client.stream = MagicMock(return_value=_MockStreamResponse(lines=["data: [DONE]"]))
        async for _ in p.generate_stream("q"):
            pass
        body = p._client.stream.call_args.kwargs["json"]
        assert body["temperature"] == 0.7
        assert "max_tokens" not in body
        assert "response_format" not in body

class TestStreamPreparationSurvivors:
    """A streamed call awaits the provider's async preparation and starts from a clean per-call state
    (openai_compat.py 457, 461)."""

    @pytest.mark.asyncio
    async def test_a_stream_call_awaits_the_async_preparation(self):
        p = _make_provider()
        p._async_prepare = AsyncMock()
        p._client = AsyncMock()
        p._client.stream = MagicMock(return_value=_MockStreamResponse(lines=["data: [DONE]"]))
        async for _ in p.generate_stream("q"):
            pass
        p._async_prepare.assert_awaited_once()

    def test_a_stream_call_starts_from_a_clean_per_call_state(self):
        import asyncio

        p = _make_provider()
        p._client = AsyncMock()
        p._client.stream = MagicMock(return_value=_MockStreamResponse(lines=["data: [DONE]"]))

        async def _go():
            # PerCallAttr state is context-local: the flag is set and read inside the same coroutine.
            p.last_json_mode_fallback = True
            async for _ in p.generate_stream("q"):
                pass
            return p.last_json_mode_fallback

        assert asyncio.run(_go()) is False

class TestStreamRetryAndBodyArgumentsSurvivors:
    """generate_stream hands its options to the body builder in order, and a stream that fails to open is retried up to
    MAX_RETRY_ATTEMPTS times, succeeding if a later open works (openai_compat.py 463-482)."""

    @pytest.mark.asyncio
    async def test_the_body_builder_receives_the_options_in_order(self):
        p = _make_provider()
        spy = MagicMock(return_value={"model": "m", "messages": []})
        p._build_stream_body = spy
        p._client = AsyncMock()
        p._client.stream = MagicMock(return_value=_MockStreamResponse(lines=["data: [DONE]"]))
        async for _ in p.generate_stream("q", "sys", 0.2, 50, True, False, {"name": "s"}, ["img"]):
            pass
        args = spy.call_args.args
        assert args[:8] == ("q", "sys", 0.2, 50, True, False, {"name": "s"}, ["img"])

    @pytest.mark.asyncio
    async def test_a_stream_that_never_opens_is_retried_exactly_the_configured_number_of_times(self, monkeypatch):
        import pyutilz.llm.openai_compat as openai_compat_mod

        monkeypatch.setattr(openai_compat_mod, "MAX_RETRY_ATTEMPTS", 3)
        monkeypatch.setattr(openai_compat_mod.asyncio, "sleep", AsyncMock())
        p = _make_provider()
        p._client = AsyncMock()
        p._client.stream = MagicMock(side_effect=lambda *a, **k: _MockStreamResponse(status_code=429))
        with pytest.raises(httpx.HTTPStatusError):
            async for _ in p.generate_stream("q"):
                pass
        assert p._client.stream.call_count == 3

    @pytest.mark.asyncio
    async def test_a_stream_that_opens_on_a_later_attempt_still_answers(self, monkeypatch):
        import pyutilz.llm.openai_compat as openai_compat_mod

        monkeypatch.setattr(openai_compat_mod, "MAX_RETRY_ATTEMPTS", 3)
        monkeypatch.setattr(openai_compat_mod.asyncio, "sleep", AsyncMock())
        answer = f"data: {jsonlib.dumps({'choices': [{'delta': {'content': 'ok'}, 'finish_reason': 'stop'}]})}"
        p = _make_provider()
        p._client = AsyncMock()
        p._client.stream = MagicMock(side_effect=[_MockStreamResponse(status_code=429), _MockStreamResponse(status_code=429), _MockStreamResponse(lines=[answer, "data: [DONE]"])])
        out = [chunk async for chunk in p.generate_stream("q")]
        assert out == ["ok"]
        assert p._client.stream.call_count == 3
        # The two refused opens were abandoned before any response started.
        assert [r["response_started"] for r in p.last_aborted_stream_attempts] == [False, False]

class TestStreamRepairSurvivors:
    """A stream refused over a request parameter is re-opened once with the provider's repaired body, and the refused
    attempt is recorded (openai_compat.py 504-518)."""

    @pytest.mark.asyncio
    async def test_a_refused_stream_is_reopened_once_with_the_repaired_body(self, caplog):
        answer = f"data: {jsonlib.dumps({'choices': [{'delta': {'content': 'ok'}, 'finish_reason': 'stop'}]})}"
        repaired_body = {"model": "m", "messages": [], "reasoning": {"enabled": False}}

        class _Repairing(_TestProvider):
            def _body_after_rejected_request(self, body, status, detail):
                return repaired_body if status == 400 else None

        p = _Repairing(api_key="test-key", model="test-model")  # pragma: allowlist secret
        p._client = AsyncMock()
        p._client.stream = MagicMock(side_effect=[_MockStreamResponse(status_code=400), _MockStreamResponse(lines=[answer, "data: [DONE]"])])
        with caplog.at_level(logging.WARNING):
            out = [chunk async for chunk in p.generate_stream("q")]
        assert out == ["ok"]
        assert p._client.stream.call_count == 2
        assert p._client.stream.call_args.kwargs["json"] == repaired_body
        assert "TestProvider/test-model rejected a streaming request parameter" in caplog.text
        assert [r["response_started"] for r in p.last_aborted_stream_attempts] == [False]

class TestStreamRepairConditionsSurvivors:
    """The repaired re-issue happens only before any content and only once; a retried open is logged
    (openai_compat.py 509-524)."""

    @staticmethod
    def _repairing_provider(repaired_body):
        class _Repairing(_TestProvider):
            def _body_after_rejected_request(self, body, status, detail):
                return repaired_body if status == 400 else None

        return _Repairing(api_key="test-key", model="test-model")  # pragma: allowlist secret

    @pytest.mark.asyncio
    async def test_a_second_refusal_is_not_repaired_again(self):
        p = self._repairing_provider({"model": "m", "messages": []})
        p._client = AsyncMock()
        p._client.stream = MagicMock(side_effect=[_MockStreamResponse(status_code=400), _MockStreamResponse(status_code=400)])
        with pytest.raises(httpx.HTTPStatusError):
            async for _ in p.generate_stream("q"):
                pass
        assert p._client.stream.call_count == 2

    @pytest.mark.asyncio
    async def test_an_error_after_content_is_not_repaired(self):
        content = f"data: {jsonlib.dumps({'choices': [{'delta': {'content': 'partial'}}]})}"
        refusal = f"data: {jsonlib.dumps({'error': {'code': 400, 'message': 'bad parameter'}})}"
        p = self._repairing_provider({"model": "m", "messages": []})
        p._client = AsyncMock()
        p._client.stream = MagicMock(return_value=_MockStreamResponse(lines=[content, refusal, "data: [DONE]"]))
        with pytest.raises(httpx.HTTPStatusError):
            async for _ in p.generate_stream("q"):
                pass
        assert p._client.stream.call_count == 1

    @pytest.mark.asyncio
    async def test_a_retried_open_is_logged_with_its_attempt_number(self, monkeypatch, caplog):
        import logging

        import pyutilz.llm.openai_compat as openai_compat_mod

        monkeypatch.setattr(openai_compat_mod, "MAX_RETRY_ATTEMPTS", 3)
        monkeypatch.setattr(openai_compat_mod.asyncio, "sleep", AsyncMock())
        answer = f"data: {jsonlib.dumps({'choices': [{'delta': {'content': 'ok'}, 'finish_reason': 'stop'}]})}"
        p = _make_provider()
        p._client = AsyncMock()
        p._client.stream = MagicMock(side_effect=[_MockStreamResponse(status_code=429), _MockStreamResponse(lines=[answer, "data: [DONE]"])])
        with caplog.at_level(logging.WARNING):
            [chunk async for chunk in p.generate_stream("q")]
        assert "Streaming attempt 1 failed" in caplog.text

class TestStreamWaitAndFinishSurvivors:
    """A retried open waits the backoff before reopening and logs the attempt with its wait; a stream that closes with
    finish_reason 'error' or 'length' raises rather than returning a fragment (openai_compat.py 524-536)."""

    @pytest.mark.asyncio
    async def test_a_retried_open_waits_the_backoff_then_reopens_and_logs_it(self, monkeypatch, caplog):
        import logging

        import pyutilz.llm.openai_compat as openai_compat_mod

        monkeypatch.setattr(openai_compat_mod, "MAX_RETRY_ATTEMPTS", 3)
        sleep = AsyncMock()
        monkeypatch.setattr(openai_compat_mod.asyncio, "sleep", sleep)
        p = _make_provider()
        monkeypatch.setattr(p, "_stream_retry_backoff", lambda exc, attempt: 7.0)
        answer = f"data: {jsonlib.dumps({'choices': [{'delta': {'content': 'ok'}, 'finish_reason': 'stop'}]})}"
        p._client = AsyncMock()
        p._client.stream = MagicMock(side_effect=[_MockStreamResponse(status_code=429), _MockStreamResponse(lines=[answer, "data: [DONE]"])])
        with caplog.at_level(logging.WARNING):
            [chunk async for chunk in p.generate_stream("q")]
        sleep.assert_awaited_once_with(7.0)
        assert "Streaming attempt 1 failed (HTTPStatusError: " in caplog.text
        assert "retrying in 7s" in caplog.text

    @pytest.mark.asyncio
    async def test_a_stream_closing_with_finish_reason_error_raises_with_the_partial_text(self):
        from pyutilz.llm.exceptions import LLMStreamInterruptedError

        partial = f"data: {jsonlib.dumps({'choices': [{'delta': {'content': 'partial'}}]})}"
        failed = f"data: {jsonlib.dumps({'choices': [{'delta': {}, 'finish_reason': 'error'}]})}"
        p = _make_provider()
        p._client = AsyncMock()
        p._client.stream = MagicMock(return_value=_MockStreamResponse(lines=[partial, failed, "data: [DONE]"]))
        with pytest.raises(LLMStreamInterruptedError) as exc:
            async for _ in p.generate_stream("q"):
                pass
        assert exc.value.partial_text == "partial"

    @pytest.mark.asyncio
    async def test_a_stream_closing_with_finish_reason_length_raises_a_truncation(self):
        from pyutilz.llm.exceptions import LLMTruncationError

        partial = f"data: {jsonlib.dumps({'choices': [{'delta': {'content': 'cut'}}]})}"
        cut = f"data: {jsonlib.dumps({'choices': [{'delta': {}, 'finish_reason': 'length'}]})}"
        p = _make_provider()
        p._client = AsyncMock()
        p._client.stream = MagicMock(return_value=_MockStreamResponse(lines=[partial, cut, "data: [DONE]"]))
        with pytest.raises(LLMTruncationError) as exc:
            async for _ in p.generate_stream("q"):
                pass
        assert exc.value.finish_reason == "length"
        assert exc.value.partial_text == "cut"

class TestStreamChunkFoldingSurvivors:
    """A streamed call records its finish reason, the citations it carried, the merged tool-call fragments and the
    reasoning text (openai_compat.py 554-564). Read inside the coroutine: per-call state is context-local."""

    @staticmethod
    def _line(choice=None, **top):
        return f"data: {jsonlib.dumps({'choices': [choice] if choice is not None else [], **top})}"

    @pytest.mark.asyncio
    async def test_the_finish_reason_and_citations_of_the_stream_are_kept(self):
        p = _make_provider()
        p._client = AsyncMock()
        lines = [
            self._line({"delta": {"content": "a"}}, citations=["https://src.example/1"]),
            self._line({"delta": {}, "finish_reason": "stop"}),
            "data: [DONE]",
        ]
        p._client.stream = MagicMock(return_value=_MockStreamResponse(lines=lines))
        async for _ in p.generate_stream("q"):
            pass
        assert p._last_finish_reason == "stop"
        assert p.last_citations == ["https://src.example/1"]

    @pytest.mark.asyncio
    async def test_tool_call_fragments_are_merged_by_index(self):
        p = _make_provider()
        p._client = AsyncMock()
        lines = [
            self._line({"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "lookup", "arguments": '{"a":'}}]}}),
            self._line({"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}, "finish_reason": "tool_calls"}),
            "data: [DONE]",
        ]
        p._client.stream = MagicMock(return_value=_MockStreamResponse(lines=lines))
        async for _ in p.generate_stream("q"):
            pass
        assert len(p.last_tool_calls) == 1
        assert p.last_tool_calls[0]["function"]["name"] == "lookup"
        assert p.last_tool_calls[0]["function"]["arguments"] == '{"a":1}'

    @pytest.mark.asyncio
    async def test_reasoning_text_is_collected_separately_from_the_answer(self):
        p = _make_provider()
        p._client = AsyncMock()
        lines = [
            self._line({"delta": {"reasoning": "think first"}}),
            self._line({"delta": {"content": "answer"}}),
            "data: [DONE]",
        ]
        p._client.stream = MagicMock(return_value=_MockStreamResponse(lines=lines))
        out = [chunk async for chunk in p.generate_stream("q")]
        assert out == ["answer"]
        assert p.last_reasoning_text == "think first"

class TestEmptyCompletionFallbackSurvivors:
    """An empty buffered answer under response_format is re-issued once without it and flagged; a second empty answer
    raises, naming whether the format was sent, dropped or never requested (openai_compat.py 727-752)."""

    @pytest.mark.asyncio
    async def test_an_empty_json_answer_is_reissued_once_without_response_format(self, caplog):
        import logging

        p = _make_provider()
        p._post_and_unwrap = AsyncMock(side_effect=[None, "fallback answer"])
        with caplog.at_level(logging.WARNING):
            out = await p.generate("q", json_mode=True)
        assert out == "fallback answer"
        assert p._post_and_unwrap.await_count == 2
        second_body = p._post_and_unwrap.call_args_list[1].args[0]
        assert "response_format" not in second_body
        assert "re-issuing once without it" in caplog.text
        assert "TestProvider/test-model returned an empty completion under response_format" in caplog.text
        assert p.last_json_mode_fallback is True

    @pytest.mark.asyncio
    async def test_an_empty_truncated_json_answer_is_reissued_once_without_response_format(self, caplog):
        import logging

        from pyutilz.llm.exceptions import LLMTruncationError

        p = _make_provider()
        p._post_and_unwrap = AsyncMock(side_effect=[LLMTruncationError("cut", finish_reason="length", partial_text=""), "answer"])
        with caplog.at_level(logging.WARNING):
            out = await p.generate("q", json_mode=True)
        assert out == "answer"
        assert "response_format" not in p._post_and_unwrap.call_args_list[1].args[0]
        assert "EMPTY completion cut off by max_tokens under response_format" in caplog.text
        assert p.last_json_mode_fallback is True

    @pytest.mark.asyncio
    async def test_a_second_empty_answer_reports_the_format_as_dropped(self):
        p = _make_provider()
        p._post_and_unwrap = AsyncMock(side_effect=[None, None])
        with pytest.raises(LLMProviderError, match=r"response_format=dropped\)"):
            await p.generate("q", json_mode=True)

    @pytest.mark.asyncio
    async def test_an_empty_answer_without_a_json_request_reports_the_format_as_not_sent(self):
        p = _make_provider()
        p._post_and_unwrap = AsyncMock(return_value=None)
        with pytest.raises(LLMProviderError, match=r"response_format=not sent\)"):
            await p.generate("q")
        assert p._post_and_unwrap.await_count == 1

class TestBufferedPostAndEmptyMessageSurvivors:
    """A buffered call posts to /chat/completions, hands the response to the provider's response hook, and an empty
    answer names its finish reason, reasoning tokens and format state in one fixed message (openai_compat.py 754-792)."""

    @pytest.mark.asyncio
    async def test_a_buffered_call_posts_to_chat_completions_and_tracks_the_response(self):
        p = _make_provider()
        p._track_provider_specific_response = MagicMock()
        data = {"id": "gen-9", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=_mock_response(body=data))
        assert await p.generate("q") == "ok"
        assert p._client.post.call_args.args[0] == "/chat/completions"
        assert any(c.args[0] == data for c in p._track_provider_specific_response.call_args_list)

    @pytest.mark.asyncio
    async def test_an_empty_answer_without_a_json_request_has_one_fixed_message(self):
        p = _make_provider()
        p._post_and_unwrap = AsyncMock(return_value=None)
        with pytest.raises(LLMProviderError) as exc:
            await p.generate("q")
        assert str(exc.value) == "TestProvider/test-model returned an empty completion (finish_reason=None, reasoning_tokens=0, response_format=not sent)"

class TestBufferedRepairAndMessageExtrasSurvivors:
    """A buffered call re-issues once with a provider-repaired body when the upstream rejects a parameter; any other
    rejection raises the API error with its detail; a completion message's tool calls, citations and reasoning are kept
    (openai_compat.py 798-831)."""

    @pytest.mark.asyncio
    async def test_a_non_retryable_rejection_raises_the_api_error_with_its_detail(self):
        p = _make_provider()
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=_mock_response(status_code=400, body={"error": {"message": "bad param"}}))
        with pytest.raises(LLMProviderError) as exc:
            await p.generate("q")
        assert str(exc.value) == "TestProvider API error 400: bad param"

    @pytest.mark.asyncio
    async def test_a_rejected_parameter_is_repaired_once_and_logged(self, caplog):
        import logging

        class _Repairing(_TestProvider):
            def _body_after_rejected_request(self, body, status, detail):
                return {"model": "m", "messages": [], "repaired": True} if status == 400 else None

        p = _Repairing(api_key="test-key", model="test-model")  # pragma: allowlist secret
        p._client = AsyncMock()
        ok = _mock_response(body={"choices": [{"message": {"content": "fixed"}, "finish_reason": "stop"}]})
        p._client.post = AsyncMock(side_effect=[_mock_response(status_code=400, body={"error": {"message": "bad param"}}), ok])
        with caplog.at_level(logging.WARNING):
            assert await p.generate("q") == "fixed"
        assert p._client.post.call_args.kwargs["json"] == {"model": "m", "messages": [], "repaired": True}
        assert "TestProvider/test-model rejected a request parameter (HTTP 400) - re-issuing once with it adjusted: bad param" in caplog.text

    def test_the_message_extras_keep_tool_calls_citations_and_reasoning(self):
        p = _make_provider()
        p._capture_message_extras({"tool_calls": [{"id": "t1"}], "citations": ["https://a.example"], "reasoning": "why"})
        assert p.last_tool_calls == [{"id": "t1"}]
        assert p.last_citations == ["https://a.example"]
        assert p.last_reasoning_text == "why"

    def test_the_message_extras_default_to_empty_when_absent(self):
        p = _make_provider()
        p._capture_message_extras({"tool_calls": "not-a-list", "citations": None})
        assert p.last_tool_calls == []
        assert p.last_citations == []

class TestExactMessageTextSurvivors:
    """The error and truncation messages carry their provider name and state exactly (openai_compat.py 516, 535, 538, 730,
    731, 871)."""

    @pytest.mark.asyncio
    async def test_a_stream_ending_in_error_names_the_provider_exactly(self):
        from pyutilz.llm.exceptions import LLMStreamInterruptedError

        failed = f"data: {jsonlib.dumps({'choices': [{'delta': {}, 'finish_reason': 'error'}]})}"
        partial = f"data: {jsonlib.dumps({'choices': [{'delta': {'content': 'x'}}]})}"
        p = _make_provider()
        p._client = AsyncMock()
        p._client.stream = MagicMock(return_value=_MockStreamResponse(lines=[partial, failed, "data: [DONE]"]))
        with pytest.raises(LLMStreamInterruptedError) as exc:
            async for _ in p.generate_stream("q"):
                pass
        assert str(exc.value) == "TestProvider stream ended with finish_reason='error'"

    @pytest.mark.asyncio
    async def test_a_stream_truncated_by_max_tokens_names_the_cut_exactly(self):
        from pyutilz.llm.exceptions import LLMTruncationError

        partial = f"data: {jsonlib.dumps({'choices': [{'delta': {'content': 'cut'}}]})}"
        cut = f"data: {jsonlib.dumps({'choices': [{'delta': {}, 'finish_reason': 'length'}]})}"
        p = _make_provider()
        p._client = AsyncMock()
        p._client.stream = MagicMock(return_value=_MockStreamResponse(lines=[partial, cut, "data: [DONE]"]))
        with pytest.raises(LLMTruncationError) as exc:
            async for _ in p.generate_stream("q"):
                pass
        assert str(exc.value) == "TestProvider streamed response truncated by max_tokens (finish_reason='length')"

    @pytest.mark.asyncio
    async def test_a_buffered_answer_truncated_by_max_tokens_names_the_cut_exactly(self):
        from pyutilz.llm.exceptions import LLMTruncationError

        p = _make_provider()
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=_mock_response(body={"choices": [{"message": {"content": "cut"}, "finish_reason": "length"}]}))
        with pytest.raises(LLMTruncationError) as exc:
            await p.generate("q")
        assert str(exc.value) == "TestProvider response truncated by max_tokens (finish_reason='length')"

    @pytest.mark.asyncio
    async def test_the_empty_truncation_reissue_log_names_the_provider_and_the_action(self, caplog):
        import logging

        from pyutilz.llm.exceptions import LLMTruncationError

        p = _make_provider()
        p._post_and_unwrap = AsyncMock(side_effect=[LLMTruncationError("cut", finish_reason="length", partial_text=""), "answer"])
        with caplog.at_level(logging.WARNING):
            await p.generate("q", json_mode=True)
        assert "TestProvider/test-model returned an EMPTY completion cut off by max_tokens under response_format - re-issuing once without it before reporting truncation" in caplog.text

class TestStreamMetadataHookSurvivors:
    """The response hook sees the first chunk as it arrives, then the first and last chunks in that order once the
    stream ends; a stream with no chunks at all gives it nothing (openai_compat.py 400-406)."""

    @pytest.mark.asyncio
    async def test_the_hook_sees_the_first_chunk_then_the_first_and_last_in_order(self):
        p = _make_provider()
        p._track_provider_specific_response = MagicMock()
        first = {"id": "gen-1", "choices": [{"delta": {"content": "a"}}]}
        last = {"id": "gen-2", "choices": [{"delta": {"content": "b"}, "finish_reason": "stop"}]}
        p._client = AsyncMock()
        p._client.stream = MagicMock(return_value=_MockStreamResponse(lines=[f"data: {jsonlib.dumps(first)}", f"data: {jsonlib.dumps(last)}", "data: [DONE]"]))
        async for _ in p.generate_stream("q"):
            pass
        ids = [c.args[0]["id"] for c in p._track_provider_specific_response.call_args_list]
        assert ids == ["gen-1", "gen-1", "gen-2"]

    @pytest.mark.asyncio
    async def test_a_stream_without_chunks_gives_the_hook_nothing(self):
        p = _make_provider()
        p._track_provider_specific_response = MagicMock()
        p._client = AsyncMock()
        p._client.stream = MagicMock(return_value=_MockStreamResponse(lines=["data: [DONE]"]))
        async for _ in p.generate_stream("q"):
            pass
        p._track_provider_specific_response.assert_not_called()

class TestPerLoopClientCloseSurvivors:
    """_close also closes the copies made for other event loops: open ones are closed, closed ones skipped without
    stopping the loop, and a copy that fails to close is logged at debug and does not raise (openai_compat.py 325-335)."""

    @staticmethod
    def _provider_with_loop_copies(monkeypatch, current, others):
        from pyutilz.llm.base import PerLoopHTTPClient

        monkeypatch.setattr(PerLoopHTTPClient, "all_clients", lambda descriptor, owner: [current, *others])
        p = _make_provider()
        p._client = current
        return p

    @pytest.mark.asyncio
    async def test_open_copies_for_other_loops_are_closed(self, monkeypatch):
        current = AsyncMock(is_closed=False)
        other = AsyncMock(is_closed=False)
        p = self._provider_with_loop_copies(monkeypatch, current, [other])
        await p._close()
        other.aclose.assert_awaited_once()
        current.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_closed_copy_is_skipped_and_the_loop_goes_on(self, monkeypatch):
        current = AsyncMock(is_closed=True)
        closed_copy = AsyncMock(is_closed=True)
        open_copy = AsyncMock(is_closed=False)
        p = self._provider_with_loop_copies(monkeypatch, current, [closed_copy, open_copy])
        await p._close()
        closed_copy.aclose.assert_not_awaited()
        open_copy.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_copy_that_fails_to_close_is_logged_and_not_raised(self, monkeypatch, caplog):
        import logging

        current = AsyncMock(is_closed=True)
        failing = AsyncMock(is_closed=False)
        failing.aclose.side_effect = RuntimeError("loop is gone")
        p = self._provider_with_loop_copies(monkeypatch, current, [failing])
        with caplog.at_level(logging.DEBUG, logger="pyutilz.llm.openai_compat"):
            await p._close()
        assert "Closing a per-loop HTTP client failed: loop is gone" in caplog.text

class TestLoggedDetailLengthSurvivors:
    """Error detail in the retry log is cut to 200 characters and in the repair log to 160 (openai_compat.py 526, 812)."""

    @pytest.mark.asyncio
    async def test_the_retry_log_keeps_exactly_200_characters_of_the_error(self, monkeypatch, caplog):
        import logging

        import pyutilz.llm.openai_compat as openai_compat_mod

        monkeypatch.setattr(openai_compat_mod, "MAX_RETRY_ATTEMPTS", 3)
        monkeypatch.setattr(openai_compat_mod.asyncio, "sleep", AsyncMock())
        answer = f"data: {jsonlib.dumps({'choices': [{'delta': {'content': 'ok'}, 'finish_reason': 'stop'}]})}"
        long_text = "x" * 300
        p = _make_provider()
        p._client = AsyncMock()
        p._client.stream = MagicMock(side_effect=[httpx.ReadError(long_text), _MockStreamResponse(lines=[answer, "data: [DONE]"])])
        with caplog.at_level(logging.WARNING):
            [chunk async for chunk in p.generate_stream("q")]
        assert "x" * 200 + ")," in caplog.text
        assert "x" * 201 not in caplog.text

    @pytest.mark.asyncio
    async def test_the_repair_log_keeps_exactly_160_characters_of_the_detail(self, caplog):
        import logging

        class _Repairing(_TestProvider):
            def _body_after_rejected_request(self, body, status, detail):
                return {"model": "m", "messages": []} if status == 400 else None

        long_text = "y" * 300
        p = _Repairing(api_key="test-key", model="test-model")  # pragma: allowlist secret
        p._client = AsyncMock()
        ok = _mock_response(body={"choices": [{"message": {"content": "fixed"}, "finish_reason": "stop"}]})
        p._client.post = AsyncMock(side_effect=[_mock_response(status_code=400, body={"error": {"message": long_text}}), ok])
        with caplog.at_level(logging.WARNING):
            assert await p.generate("q") == "fixed"
        assert "y" * 160 in caplog.text
        assert "y" * 161 not in caplog.text

class TestContextWindowDefaultAndStreamIdentitySurvivors:
    """A model with no window in the map gets the 128,000-token default; a streamed attempt keeps the generation id of its
    first chunk, not the last (openai_compat.py 305, 383)."""

    def test_an_unlisted_model_gets_the_128000_token_window(self):
        assert _make_provider().context_window == 128_000

    @pytest.mark.asyncio
    async def test_the_generation_id_comes_from_the_first_chunk(self):
        first = {"id": "gen-1", "choices": [{"delta": {"content": "a"}}]}
        second = {"id": "gen-2", "choices": [{"delta": {"content": "b"}}]}
        error = f"data: {jsonlib.dumps({'error': {'code': 502, 'message': 'upstream connection lost'}})}"
        p = _make_provider()
        p._client = AsyncMock()
        p._client.stream = MagicMock(return_value=_MockStreamResponse(lines=[f"data: {jsonlib.dumps(first)}", f"data: {jsonlib.dumps(second)}", error]))
        with pytest.raises(httpx.HTTPStatusError):
            async for _ in p.generate_stream("q"):
                pass
        assert p.last_aborted_stream_attempts[0]["generation_id"] == "gen-1"

class TestStreamRetryGuardSurvivors:
    """A truncated stream raises at once and is never reopened, and a stream's usage is recorded once (openai_compat.py
    493-505)."""

    @pytest.mark.asyncio
    async def test_a_truncated_stream_is_raised_and_never_reopened(self, monkeypatch):
        import pyutilz.llm.openai_compat as openai_compat_mod
        from pyutilz.llm.exceptions import LLMTruncationError

        monkeypatch.setattr(openai_compat_mod, "MAX_RETRY_ATTEMPTS", 3)
        monkeypatch.setattr(openai_compat_mod.asyncio, "sleep", AsyncMock())
        partial = f"data: {jsonlib.dumps({'choices': [{'delta': {'content': 'cut'}}]})}"
        cut = f"data: {jsonlib.dumps({'choices': [{'delta': {}, 'finish_reason': 'length'}]})}"
        p = _make_provider()
        p._client = AsyncMock()
        p._client.stream = MagicMock(return_value=_MockStreamResponse(lines=[partial, cut, "data: [DONE]"]))
        with pytest.raises(LLMTruncationError):
            async for _ in p.generate_stream("q"):
                pass
        assert p._client.stream.call_count == 1

    @pytest.mark.asyncio
    async def test_an_empty_truncated_stream_is_not_offered_a_repair(self):
        from pyutilz.llm.exceptions import LLMTruncationError

        class _AlwaysRepairs(_TestProvider):
            def _body_after_rejected_request(self, body, status, detail):
                return {"model": "m", "messages": [], "repaired": True}

        cut_empty = f"data: {jsonlib.dumps({'choices': [{'delta': {}, 'finish_reason': 'length'}]})}"
        p = _AlwaysRepairs(api_key="test-key", model="test-model")  # pragma: allowlist secret
        p._client = AsyncMock()
        p._client.stream = MagicMock(return_value=_MockStreamResponse(lines=[cut_empty, "data: [DONE]"]))
        with pytest.raises(LLMTruncationError):
            async for _ in p.generate_stream("q"):
                pass
        assert p._client.stream.call_count == 1

    @pytest.mark.asyncio
    async def test_a_streams_usage_is_recorded_once(self, monkeypatch):
        usage = {"prompt_tokens": 3, "completion_tokens": 1}
        p = _make_provider()
        recorded = MagicMock()
        monkeypatch.setattr(p, "_track_streaming_usage", recorded)
        answer = f"data: {jsonlib.dumps({'choices': [{'delta': {'content': 'ok'}, 'finish_reason': 'stop'}]})}"
        trailer = f"data: {jsonlib.dumps({'choices': [], 'usage': usage})}"
        p._client = AsyncMock()
        p._client.stream = MagicMock(return_value=_MockStreamResponse(lines=[answer, trailer, "data: [DONE]"]))
        async for _ in p.generate_stream("q"):
            pass
        recorded.assert_called_once_with(usage)

class TestProgressAndReasoningOrderSurvivors:
    """Each streamed chunk reports its reasoning piece and content piece to the stream-progress counter in that order, and
    reasoning from successive messages is kept in arrival order (openai_compat.py 568, 831)."""

    def test_the_progress_counter_gets_the_reasoning_piece_then_the_content(self, monkeypatch):
        import pyutilz.llm.openai_compat as openai_compat_mod

        spy = MagicMock()
        monkeypatch.setattr(openai_compat_mod, "note_stream_progress", spy)
        p = _make_provider()
        p._apply_stream_chunk({"choices": [{"delta": {"reasoning": "think", "content": "hi"}}]}, {})
        spy.assert_called_once()
        reasoning_piece, content_piece = spy.call_args.args
        assert content_piece == "hi"
        assert reasoning_piece is not None and "think" in str(reasoning_piece)

    def test_reasoning_from_successive_messages_keeps_arrival_order(self):
        p = _make_provider()
        p._capture_message_extras({"reasoning": "first"})
        p._capture_message_extras({"reasoning": "second"})
        text = p.last_reasoning_text
        assert "first" in text and "second" in text
        assert text.index("first") < text.index("second")

class TestStreamCloseSurvivors:
    """A consumer that stops reading a stream closes the HTTP response at once (openai_compat.py 490)."""

    @pytest.mark.asyncio
    async def test_stopping_early_closes_the_response(self):
        class _Recording(_MockStreamResponse):
            closed = False

            async def __aexit__(self, *exc_info):
                type(self).closed = True
                return False

        resp = _Recording(lines=[f"data: {jsonlib.dumps({'choices': [{'delta': {'content': 'a'}}]})}", f"data: {jsonlib.dumps({'choices': [{'delta': {'content': 'b'}}]})}", "data: [DONE]"])
        p = _make_provider()
        p._client = AsyncMock()
        p._client.stream = MagicMock(return_value=resp)
        gen = p.generate_stream("q")
        assert await gen.__anext__() == "a"
        await gen.aclose()
        assert _Recording.closed is True

class TestStreamBackoffSurvivors:
    """The stream retry wait doubles from 5 s, is capped at 300 s, adds jitter in [0, 5), and is never shorter than a
    Retry-After the server sent (openai_compat.py 411-422)."""

    @staticmethod
    def _wait(monkeypatch, attempt, jitter=0.0, exc=None):
        import pyutilz.llm.openai_compat as openai_compat_mod

        seen = []

        def _uniform(a, b):
            seen.append((a, b))
            return jitter

        monkeypatch.setattr(openai_compat_mod.random, "uniform", _uniform)
        wait = OpenAICompatibleProvider._stream_retry_backoff(exc if exc is not None else RuntimeError("x"), attempt)
        return wait, seen

    def test_the_wait_doubles_from_five_seconds_and_is_capped_at_300(self, monkeypatch):
        assert self._wait(monkeypatch, 1)[0] == 5.0
        assert self._wait(monkeypatch, 3)[0] == 8.0
        assert self._wait(monkeypatch, 8)[0] == 256.0
        assert self._wait(monkeypatch, 9)[0] == 300.0

    def test_the_jitter_is_drawn_from_zero_to_five_and_added(self, monkeypatch):
        wait, seen = self._wait(monkeypatch, 3, jitter=2.5)
        assert seen == [(0, 5)]
        assert wait == 10.5

    def test_a_retry_after_longer_than_the_backoff_is_honoured(self, monkeypatch):
        response = httpx.Response(429, headers={"retry-after": "60"})
        exc = httpx.HTTPStatusError("rate limited", request=httpx.Request("POST", "https://x"), response=response)
        assert self._wait(monkeypatch, 1, exc=exc)[0] == 60.0

    def test_a_retry_after_shorter_than_the_backoff_does_not_shorten_it(self, monkeypatch):
        response = httpx.Response(429, headers={"retry-after": "1"})
        exc = httpx.HTTPStatusError("rate limited", request=httpx.Request("POST", "https://x"), response=response)
        assert self._wait(monkeypatch, 9, exc=exc)[0] == 300.0

class TestCloseSurvivors:
    """_close closes the current HTTP client only when it is still open (openai_compat.py 325-326)."""

    @pytest.mark.asyncio
    async def test_an_open_client_is_closed(self):
        p = _make_provider()
        p._client = AsyncMock()
        p._client.is_closed = False
        await p._close()
        p._client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_client_without_an_is_closed_flag_is_treated_as_open(self):
        p = _make_provider()
        from types import SimpleNamespace

        client = SimpleNamespace(aclose=AsyncMock())
        p._client = client
        await p._close()
        client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_an_already_closed_client_is_not_closed_again(self):
        p = _make_provider()
        p._client = AsyncMock()
        p._client.is_closed = True
        await p._close()
        p._client.aclose.assert_not_awaited()

class TestStreamAttemptSurvivors:
    """One streamed attempt posts the body to /chat/completions, reads the body of a 4xx error before classifying it,
    marks the response as started once it is accepted, and skips lines that are not ``data:`` events
    (openai_compat.py 337-366)."""

    @staticmethod
    async def _drain(p, body=None, state=None, lines=None, status=200):
        box = {"streamed_text": []}
        state = {} if state is None else state
        resp = _MockStreamResponse(status_code=status, lines=lines or [])
        p._client = AsyncMock()
        p._client.stream = MagicMock(return_value=resp)
        async for _ in p._stream_attempt(body if body is not None else {"model": "m"}, state, box):
            pass
        return p._client.stream, box, state, resp

    @pytest.mark.asyncio
    async def test_the_attempt_posts_the_body_to_chat_completions(self):
        body = {"model": "m", "messages": []}
        stream, _box, _state, _resp = await self._drain(_make_provider(), body=body, lines=["data: [DONE]"])
        args, kwargs = stream.call_args
        assert args[:2] == ("POST", "/chat/completions")
        assert kwargs["json"] == body

    @pytest.mark.asyncio
    async def test_an_error_body_is_read_before_the_status_is_classified(self):
        p = _make_provider()
        p._client = AsyncMock()
        resp = _MockStreamResponse(status_code=400)
        resp.aread = AsyncMock(return_value=b"{}")
        p._client.stream = MagicMock(return_value=resp)
        with pytest.raises((httpx.HTTPStatusError, LLMProviderError)):
            async for _ in p._stream_attempt({"model": "m"}, {}, {"streamed_text": []}):
                pass
        resp.aread.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_providers_special_status_handler_runs_before_the_generic_raise(self):
        from tests.openai_compat_support import _TestProvider

        class _Special(_TestProvider):
            def _handle_special_status(self, resp):
                if resp.status_code == 402:
                    raise LLMProviderError("credits exhausted")

        p = _Special(api_key="test-key", model="test-model")  # pragma: allowlist secret
        p._client = AsyncMock()
        p._client.stream = MagicMock(return_value=_MockStreamResponse(status_code=402))
        with pytest.raises(LLMProviderError, match="credits exhausted"):
            async for _ in p._stream_attempt({"model": "m"}, {}, {"streamed_text": []}):
                pass

    @pytest.mark.asyncio
    async def test_an_accepted_stream_marks_the_response_as_started(self):
        line = f"data: {jsonlib.dumps({'choices': [{'delta': {'content': 'hi'}, 'finish_reason': 'stop'}]})}"
        _stream, _box, state, _resp = await self._drain(_make_provider(), lines=[line, "data: [DONE]"])
        assert state["response_started"] is True

    @staticmethod
    def _chunk(text: str) -> str:
        return f"data: {jsonlib.dumps({'choices': [{'delta': {'content': text}}]})}"

    @pytest.mark.asyncio
    async def test_a_malformed_json_line_is_skipped_and_the_stream_goes_on(self):
        _stream, box, _state, _resp = await self._drain(_make_provider(), lines=["data: {not json", self._chunk("hi"), "data: [DONE]"])
        assert box["streamed_text"] == ["hi"]

    @pytest.mark.asyncio
    async def test_nothing_after_done_is_read(self):
        _stream, box, _state, _resp = await self._drain(_make_provider(), lines=[self._chunk("hi"), "data: [DONE]", self._chunk("late")])
        assert box["streamed_text"] == ["hi"]

    @pytest.mark.asyncio
    async def test_an_empty_line_does_not_end_the_stream(self):
        _stream, box, _state, _resp = await self._drain(_make_provider(), lines=["", self._chunk("hi"), "", "data: [DONE]"])
        assert box["streamed_text"] == ["hi"]

    @pytest.mark.asyncio
    async def test_the_usage_block_is_kept_from_whichever_chunk_carries_it(self):
        usage = {"prompt_tokens": 3, "completion_tokens": 1}
        trailer = f"data: {jsonlib.dumps({'choices': [], 'usage': usage})}"
        _stream, box, _state, _resp = await self._drain(_make_provider(), lines=[self._chunk("hi"), trailer, "data: [DONE]"])
        assert box["latest_usage"] == usage

    @pytest.mark.asyncio
    async def test_an_event_line_without_a_space_after_the_prefix_is_read(self):
        bare = f"data:{jsonlib.dumps({'choices': [{'delta': {'content': 'hi'}}]})}"
        _stream, box, _state, _resp = await self._drain(_make_provider(), lines=[bare, "data: [DONE]"])
        assert box["streamed_text"] == ["hi"]

    @pytest.mark.asyncio
    async def test_a_stream_dying_mid_answer_still_has_its_first_chunk_tracked(self):
        p = _make_provider()
        p._track_provider_specific_response = MagicMock()
        first = {"id": "gen-7", "choices": [{"delta": {"content": "partial"}}]}
        error = f"data: {jsonlib.dumps({'error': {'code': 502, 'message': 'upstream connection lost'}})}"
        with pytest.raises(httpx.HTTPStatusError):
            await self._drain(p, lines=[f"data: {jsonlib.dumps(first)}", error, "data: [DONE]"])
        assert any(c.args[0] == first for c in p._track_provider_specific_response.call_args_list)

    @pytest.mark.asyncio
    async def test_the_first_chunk_is_handed_to_the_provider_hook_as_it_arrives(self):
        p = _make_provider()
        p._track_provider_specific_response = MagicMock()
        first = {"choices": [{"delta": {"content": "hi"}}]}
        await self._drain(p, lines=[f"data: {jsonlib.dumps(first)}", "data: [DONE]"])
        assert any(c.args[0] == first for c in p._track_provider_specific_response.call_args_list)

    @pytest.mark.asyncio
    async def test_lines_without_the_data_prefix_are_skipped(self):
        line = f"data: {jsonlib.dumps({'choices': [{'delta': {'content': 'hi'}, 'finish_reason': 'stop'}]})}"
        # The skipped line carries valid JSON after its first five characters: if the prefix test were
        # ever matched by the empty string, that payload would be parsed and its content would leak in.
        smuggled = "event" + jsonlib.dumps({"choices": [{"delta": {"content": "LEAK"}}]})
        _stream, box, _state, _resp = await self._drain(_make_provider(), lines=[smuggled, "not an event line", "", line, "data: [DONE]"])
        assert box["streamed_text"] == ["hi"]

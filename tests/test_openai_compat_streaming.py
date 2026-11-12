"""OpenAI-compatible provider tests: streaming: opening, lines, retries and teardown."""

from pyutilz.core import jsonlib
import logging
import pytest
from pyutilz.llm.openai_compat import OpenAICompatibleProvider
from tests.openai_compat_support import (
    _FakeHttpClient,
    _FakeStreamResponse,
    _FakeStreamingHttpClient,
    _PreparationOrderError,
    _SequenceStreamClient,
    _StatusError,
    _StatusStreamResponse,
    _WarnLog,
    _WarnRecorder,
    _collect_stream,
    _content_line,
    _drain_stream,
    _drain_with_box,
    _make_provider,
    _no_sleep,
)

httpx = pytest.importorskip("httpx")


class TestRateLimitCaptureAndAccountLimits:
    """Header capture, the no-snapshot refusal and the summary of a snapshot (openai_compat 216-242)."""

    def test_unparseable_headers_are_warned_and_no_snapshot_is_made(self, monkeypatch):
        import pyutilz.llm.openai_compat as oc_module

        log = _WarnLog()
        monkeypatch.setattr(oc_module, "logger", log)
        provider = _make_provider()
        provider._capture_rate_limit_headers(5)
        assert len(log.calls) == 1
        msg, args = log.calls[0]
        assert msg == "Could not parse response headers for rate-limit capture: %s"
        assert isinstance(args[0], TypeError)
        assert provider.last_rate_limits == {}

    def test_only_rate_limit_headers_are_kept_and_lowercased(self):
        provider = _make_provider()
        provider._capture_rate_limit_headers({"X-RateLimit-Limit-Requests": "5", "RateLimit-Remaining": "3", "Content-Type": "json"})
        assert provider.last_rate_limits == {"x-ratelimit-limit-requests": "5", "ratelimit-remaining": "3"}

    def test_a_response_without_rate_limit_headers_clears_the_snapshot(self):
        provider = _make_provider()
        provider._capture_rate_limit_headers({"X-RateLimit-Remaining": "3"})
        provider._capture_rate_limit_headers({"Content-Type": "json"})
        assert provider.last_rate_limits == {}

    def test_no_snapshot_refuses_with_the_exact_message(self):
        import asyncio

        provider = _make_provider()
        with pytest.raises(NotImplementedError) as info:
            asyncio.run(provider.check_account_limits())
        assert str(info.value) == (
            "TestProvider: no rate-limit snapshot captured yet -- issue at least one generate() call first; the headers are "
            "captured automatically. Standalone introspection endpoints are not exposed for regular keys on this provider."
        )

    def test_a_snapshot_is_summarised_under_short_keys_with_the_raw_copy(self):
        import asyncio

        provider = _make_provider()
        provider.last_rate_limits = {"x-ratelimit-limit-requests": "5", "ratelimit-remaining": "3"}
        out = asyncio.run(provider.check_account_limits())
        assert out == {"raw": {"x-ratelimit-limit-requests": "5", "ratelimit-remaining": "3"}, "limit_requests": "5", "remaining": "3"}


class TestPricingResolution:
    """The base pricing record and the cached-input and cache-write fallbacks (openai_compat 256-273)."""

    def test_the_default_pricing_record_carries_input_then_output(self):
        from pyutilz.llm._pricing import Pricing

        priced = _make_provider()._resolve_pricing("test-model")
        assert priced == Pricing(input=1.0, output=2.0)

    def test_an_absent_cache_hit_rate_falls_back_to_the_input_rate(self):
        provider = _make_provider()
        assert OpenAICompatibleProvider._cache_hit_cost_per_1m(provider, "test-model") == 1.0

    def test_a_published_cache_hit_rate_is_used(self, monkeypatch):
        from pyutilz.llm._pricing import Pricing

        provider = _make_provider()
        monkeypatch.setattr(provider, "_resolve_pricing", lambda model: Pricing(input=1.0, output=2.0, cache_hit=0.25))
        assert OpenAICompatibleProvider._cache_hit_cost_per_1m(provider, "test-model") == 0.25

    def test_an_absent_cache_write_rate_falls_back_to_the_input_rate(self):
        provider = _make_provider()
        assert OpenAICompatibleProvider._cache_write_cost_per_1m(provider, "test-model") == 1.0

    def test_a_published_cache_write_rate_is_used(self, monkeypatch):
        from pyutilz.llm._pricing import Pricing

        provider = _make_provider()
        monkeypatch.setattr(provider, "_resolve_pricing", lambda model: Pricing(input=1.0, output=2.0, cache_write=1.25))
        assert OpenAICompatibleProvider._cache_write_cost_per_1m(provider, "test-model") == 1.25


class TestStreamRetryBackoff:
    """The exponential floor, the jitter draw and the Retry-After floor of a stream re-open (openai_compat 415)."""

    @pytest.fixture
    def no_jitter(self, monkeypatch):
        import pyutilz.llm.openai_compat as oc_module

        draws: list = []

        def fake_uniform(low, high):
            draws.append((low, high))
            return 0.0

        monkeypatch.setattr(oc_module.random, "uniform", fake_uniform)
        return draws

    def _wait(self, attempt, exc=None):
        return OpenAICompatibleProvider._stream_retry_backoff(exc if exc is not None else _StatusError(), attempt)

    def test_the_first_attempt_waits_the_five_second_floor(self, no_jitter):
        assert self._wait(1) == 5.0

    def test_the_wait_doubles_per_attempt_once_above_the_floor(self, no_jitter):
        assert self._wait(4) == 16.0
        assert self._wait(8) == 256.0

    def test_the_wait_is_capped_at_three_hundred_seconds(self, no_jitter):
        assert self._wait(9) == 300.0
        assert self._wait(20) == 300.0

    def test_the_jitter_is_drawn_from_zero_to_five_and_added(self, no_jitter, monkeypatch):
        import pyutilz.llm.openai_compat as oc_module

        monkeypatch.setattr(oc_module.random, "uniform", lambda low, high: 2.5)
        assert self._wait(4) == 18.5
        assert no_jitter == []

    def test_the_jitter_range_is_zero_to_five(self, no_jitter):
        self._wait(1)
        assert no_jitter == [(0, 5)]

    def test_a_longer_retry_after_hint_lengthens_the_wait(self, no_jitter):
        import httpx

        response = httpx.Response(429, headers={"retry-after": "120"})
        assert self._wait(1, _StatusError(response)) == 120.0

    def test_a_shorter_retry_after_hint_never_shortens_the_wait(self, no_jitter):
        import httpx

        response = httpx.Response(429, headers={"retry-after": "2"})
        assert self._wait(4, _StatusError(response)) == 16.0


class TestCloseClosesEveryPerLoopClient:
    """_close closes the current client and each live per-loop copy, skipping closed ones, and logs a failed copy (openai_compat 325-335)."""

    @pytest.fixture
    def recorder(self, monkeypatch):
        import pyutilz.llm.openai_compat as oc_module

        class _DebugLog:
            def __init__(self):
                self.calls: list = []

            def debug(self, msg, *args, **kwargs):
                self.calls.append((msg, args))

        log = _DebugLog()
        monkeypatch.setattr(oc_module, "logger", log)
        return log

    def _with_clones(self, provider, *clones):
        import asyncio
        import weakref

        loops = [asyncio.new_event_loop() for _ in clones]
        per_loop = weakref.WeakKeyDictionary()
        for loop, client in zip(loops, clones):
            per_loop[loop] = client
        provider.__dict__["__client_per_loop"] = per_loop
        return loops

    def test_an_open_current_client_is_closed(self):
        import asyncio

        provider = _make_provider()
        current = _FakeHttpClient()
        provider._client = current
        asyncio.run(provider._close())
        assert current.close_calls == 1

    def test_a_closed_current_client_is_not_closed_again(self):
        import asyncio

        provider = _make_provider()
        current = _FakeHttpClient(closed=True)
        provider._client = current
        asyncio.run(provider._close())
        assert current.close_calls == 0

    def test_an_open_per_loop_copy_is_closed_too(self):
        import asyncio

        provider = _make_provider()
        provider._client = _FakeHttpClient()
        copy = _FakeHttpClient()
        loops = self._with_clones(provider, copy)
        asyncio.run(provider._close())
        assert copy.close_calls == 1
        for loop in loops:
            loop.close()

    def test_a_closed_per_loop_copy_is_skipped(self):
        import asyncio

        provider = _make_provider()
        provider._client = _FakeHttpClient()
        copy = _FakeHttpClient(closed=True)
        loops = self._with_clones(provider, copy)
        asyncio.run(provider._close())
        assert copy.close_calls == 0
        for loop in loops:
            loop.close()

    def test_a_failing_copy_is_logged_and_the_rest_are_still_closed(self, recorder):
        import asyncio

        provider = _make_provider()
        provider._client = _FakeHttpClient()
        broken = _FakeHttpClient(fail=True)
        healthy = _FakeHttpClient()
        loops = self._with_clones(provider, broken, healthy)
        asyncio.run(provider._close())
        assert healthy.close_calls == 1
        assert len(recorder.calls) == 1
        msg, args = recorder.calls[0]
        assert msg == "Closing a per-loop HTTP client failed: %s"
        assert isinstance(args[0], RuntimeError)
        for loop in loops:
            loop.close()


class TestDefaultContextWindow:
    """A model missing from the per-model map gets the 128 000-token default (openai_compat 305)."""

    def test_an_unlisted_model_gets_the_128k_default(self):
        assert _make_provider("unlisted-model").context_window == 128_000

    def test_the_default_constant_is_128k(self):
        assert OpenAICompatibleProvider._default_context_window == 128_000


class TestStreamAttemptOpensAndChecks:
    """The request posted, the error-body read on a failed status, the special-status hook, and the response-started flag (openai_compat 344-354)."""

    def test_the_stream_posts_to_the_chat_completions_path(self):
        import asyncio

        response = _FakeStreamResponse(lines=())
        provider = _make_provider()
        client = _FakeStreamingHttpClient(response)
        provider._client = client
        asyncio.run(_drain_stream(provider, body={"model": "test-model", "stream": True}))
        method, url, json_body, _timeout = client.calls[0]
        assert (method, url) == ("POST", "/chat/completions")
        assert json_body == {"model": "test-model", "stream": True}

    def test_an_error_status_body_is_read_before_it_is_judged(self, monkeypatch):
        import asyncio

        response = _FakeStreamResponse(status=400, lines=())
        provider = _make_provider()
        provider._client = _FakeStreamingHttpClient(response)
        monkeypatch.setattr(provider, "_handle_special_status", lambda resp: None)
        with pytest.raises(RuntimeError):
            asyncio.run(_drain_stream(provider))
        assert response.aread_calls == 1

    def test_a_status_just_below_four_hundred_is_not_read_as_an_error(self, monkeypatch):
        import asyncio

        response = _FakeStreamResponse(status=399, lines=())
        provider = _make_provider()
        provider._client = _FakeStreamingHttpClient(response)
        monkeypatch.setattr(provider, "_handle_special_status", lambda resp: None)
        asyncio.run(_drain_stream(provider))
        assert response.aread_calls == 0

    def test_the_special_status_hook_is_consulted_on_every_response(self, monkeypatch):
        import asyncio

        response = _FakeStreamResponse(status=200, lines=())
        provider = _make_provider()
        provider._client = _FakeStreamingHttpClient(response)
        seen: list = []
        monkeypatch.setattr(provider, "_handle_special_status", lambda resp: seen.append(resp.status_code))
        asyncio.run(_drain_stream(provider))
        assert seen == [200]

    def test_a_started_response_sets_the_response_started_flag(self):
        import asyncio

        response = _FakeStreamResponse(status=200, lines=())
        provider = _make_provider()
        provider._client = _FakeStreamingHttpClient(response)
        state = asyncio.run(_drain_stream(provider))
        assert state["response_started"] is True


class TestStreamLineHandling:
    """Which SSE lines are read, where the stream stops, what is skipped, and what metadata is kept (openai_compat 366-407)."""

    def test_only_data_lines_are_read(self):
        import asyncio

        lines = ('hello{"choices":[{"delta":{"content":"BAD"}}]}', "", ": keepalive", _content_line("good"), "data: [DONE]")
        yielded, _box, _state = asyncio.run(_drain_with_box(_make_provider(), lines))
        assert yielded == ["good"]

    def test_the_done_marker_ends_the_stream(self):
        import asyncio

        lines = (_content_line("one"), "data: [DONE]", _content_line("two"))
        yielded, _box, _state = asyncio.run(_drain_with_box(_make_provider(), lines))
        assert yielded == ["one"]

    def test_an_unparseable_data_line_is_skipped_and_reading_goes_on(self):
        import asyncio

        lines = ("data: {not json", _content_line("after"), "data: [DONE]")
        yielded, _box, _state = asyncio.run(_drain_with_box(_make_provider(), lines))
        assert yielded == ["after"]

    def test_each_yielded_piece_is_also_kept_in_the_streamed_text(self):
        import asyncio

        lines = (_content_line("a"), _content_line("b"), "data: [DONE]")
        _yielded, box, _state = asyncio.run(_drain_with_box(_make_provider(), lines))
        assert box["streamed_text"] == ["a", "b"]

    def test_the_stream_started_flag_is_set_for_a_stream_with_chunks(self):
        import asyncio

        _yielded, _box, state = asyncio.run(_drain_with_box(_make_provider(), (_content_line("a"), "data: [DONE]")))
        assert state["response_started"] is True

    def test_the_first_chunk_is_tracked_as_it_arrives(self, monkeypatch):
        import asyncio

        provider = _make_provider()
        tracked: list = []
        monkeypatch.setattr(provider, "_track_provider_specific_response", lambda chunk: tracked.append(chunk))
        lines = (_content_line("a"), _content_line("b"), "data: [DONE]")
        asyncio.run(_drain_with_box(provider, lines))
        assert tracked[0] == {"choices": [{"delta": {"content": "a"}}]}

    def test_the_first_flag_is_true_only_for_the_first_chunk(self, monkeypatch):
        import asyncio

        provider = _make_provider()
        firsts: list = []
        monkeypatch.setattr(provider, "_note_stream_chunk", lambda state, chunk, first: firsts.append(first))
        asyncio.run(_drain_with_box(provider, (_content_line("a"), _content_line("b"), "data: [DONE]")))
        assert firsts == [True, False]

    def test_the_metadata_finish_receives_the_first_and_last_chunks(self, monkeypatch):
        import asyncio

        provider = _make_provider()
        seen: list = []
        monkeypatch.setattr(provider, "_finish_stream_metadata", lambda first, last, fragments: seen.append((first, last, fragments)))
        asyncio.run(_drain_with_box(provider, (_content_line("a"), _content_line("b"), "data: [DONE]")))
        first, last, fragments = seen[0]
        assert first == {"choices": [{"delta": {"content": "a"}}]}
        assert last == {"choices": [{"delta": {"content": "b"}}]}
        assert fragments == {}

    def test_every_chunk_seen_is_tracked_at_the_end_when_no_metadata_arrives(self, monkeypatch):
        provider = _make_provider()
        tracked: list = []
        monkeypatch.setattr(provider, "_track_provider_specific_response", lambda chunk: tracked.append(chunk))
        provider._finish_stream_metadata({"id": "x"}, {"id": "y"}, {})
        assert tracked == [{"id": "x"}, {"id": "y"}]

    def test_an_empty_stream_tracks_no_metadata(self, monkeypatch):
        provider = _make_provider()
        tracked: list = []
        monkeypatch.setattr(provider, "_track_provider_specific_response", lambda chunk: tracked.append(chunk))
        provider._finish_stream_metadata(None, None, {})
        assert tracked == []

    def test_assembled_tool_calls_are_kept_in_index_order(self):
        provider = _make_provider()
        provider._finish_stream_metadata(None, None, {1: {"name": "second"}, 0: {"name": "first"}})
        assert provider.last_tool_calls == [{"name": "first"}, {"name": "second"}]

    def test_no_tool_calls_leaves_the_last_tool_calls_alone(self):
        provider = _make_provider()
        provider.last_tool_calls = [{"name": "kept"}]
        provider._finish_stream_metadata(None, None, {})
        assert provider.last_tool_calls == [{"name": "kept"}]


class TestGenerateStreamRetryAndFinish:
    """Retry of a failed stream open, the once-only body repair, the attempt cap, the finish-reason refusals and the request defaults (openai_compat 428-536)."""

    def test_a_retryable_failure_before_any_token_is_retried_after_the_backoff(self, monkeypatch):
        import asyncio

        provider = _make_provider()
        waits = _no_sleep(monkeypatch, provider, wait=7.0)
        ok = _FakeStreamResponse(status=200, lines=(_content_line("hello"), "data: [DONE]"))
        provider._client = _SequenceStreamClient([_StatusStreamResponse(status=503), ok])
        assert asyncio.run(_collect_stream(provider)) == ["hello"]
        assert waits == [7.0]

    def test_a_non_retryable_failure_is_raised_without_a_retry(self, monkeypatch):
        import asyncio

        provider = _make_provider()
        waits = _no_sleep(monkeypatch, provider)
        provider._client = _SequenceStreamClient([_StatusStreamResponse(status=401)])
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(_collect_stream(provider))
        assert waits == []

    def test_the_attempt_cap_stops_retrying(self, monkeypatch):
        import asyncio

        import pyutilz.llm.openai_compat as oc_module

        provider = _make_provider()
        waits = _no_sleep(monkeypatch, provider)
        monkeypatch.setattr(oc_module, "MAX_RETRY_ATTEMPTS", 2)
        client = _SequenceStreamClient([_StatusStreamResponse(status=503)])
        provider._client = client
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(_collect_stream(provider))
        assert len(client.calls) == 2
        assert waits == [7.0]

    def test_a_zero_cap_retries_without_limit(self, monkeypatch):
        import asyncio

        import pyutilz.llm.openai_compat as oc_module

        provider = _make_provider()
        _no_sleep(monkeypatch, provider)
        monkeypatch.setattr(oc_module, "MAX_RETRY_ATTEMPTS", 0)
        ok = _FakeStreamResponse(status=200, lines=(_content_line("late"), "data: [DONE]"))
        provider._client = _SequenceStreamClient([_StatusStreamResponse(status=503), _StatusStreamResponse(status=503), ok])
        assert asyncio.run(_collect_stream(provider)) == ["late"]

    def test_an_aborted_attempt_is_noted_once_per_failure(self, monkeypatch):
        import asyncio

        provider = _make_provider()
        _no_sleep(monkeypatch, provider)
        noted: list = []
        monkeypatch.setattr(provider, "_note_aborted_stream_attempt", lambda attempt, state, exc: noted.append(attempt))
        ok = _FakeStreamResponse(status=200, lines=(_content_line("x"), "data: [DONE]"))
        provider._client = _SequenceStreamClient([_StatusStreamResponse(status=503), ok])
        asyncio.run(_collect_stream(provider))
        assert noted == [1]

    def test_a_request_is_repaired_once_and_the_repair_is_logged(self, monkeypatch, caplog):
        import asyncio

        provider = _make_provider()
        _no_sleep(monkeypatch, provider)

        async def fake_repair(exc, body):
            return {**body, "reasoning": {"enabled": True}}

        monkeypatch.setattr(provider, "_repaired_stream_body", fake_repair)
        ok = _FakeStreamResponse(status=200, lines=(_content_line("fixed"), "data: [DONE]"))
        client = _SequenceStreamClient([_StatusStreamResponse(status=400), ok])
        provider._client = client
        with caplog.at_level(logging.WARNING, logger="pyutilz.llm.openai_compat"):
            assert asyncio.run(_collect_stream(provider)) == ["fixed"]
        assert client.calls[1]["reasoning"] == {"enabled": True}
        assert any("rejected a streaming request parameter - re-opening the stream once with it adjusted." in r.getMessage() for r in caplog.records)

    def test_a_second_refusal_after_the_repair_is_raised(self, monkeypatch):
        import asyncio

        provider = _make_provider()
        _no_sleep(monkeypatch, provider)
        repairs: list = []

        async def fake_repair(exc, body):
            repairs.append(1)
            return {**body, "reasoning": {"enabled": True}}

        monkeypatch.setattr(provider, "_repaired_stream_body", fake_repair)
        provider._client = _SequenceStreamClient([_StatusStreamResponse(status=400)])
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(_collect_stream(provider))
        assert repairs == [1]

    def test_a_length_finish_is_refused_with_the_truncation_message(self):
        import asyncio

        from pyutilz.llm.exceptions import LLMTruncationError

        provider = _make_provider()
        line = "data: " + jsonlib.dumps({"choices": [{"delta": {"content": "ab"}, "finish_reason": "length"}]})
        provider._client = _SequenceStreamClient([_FakeStreamResponse(status=200, lines=(line, "data: [DONE]"))])
        with pytest.raises(LLMTruncationError) as info:
            asyncio.run(_collect_stream(provider))
        assert str(info.value) == "TestProvider streamed response truncated by max_tokens (finish_reason='length')"
        assert info.value.partial_text == "ab"

    def test_an_error_finish_is_refused_with_the_interrupted_message(self):
        import asyncio

        from pyutilz.llm.exceptions import LLMStreamInterruptedError

        provider = _make_provider()
        line = "data: " + jsonlib.dumps({"choices": [{"delta": {"content": "cut"}, "finish_reason": "error"}]})
        provider._client = _SequenceStreamClient([_FakeStreamResponse(status=200, lines=(line, "data: [DONE]"))])
        with pytest.raises(LLMStreamInterruptedError) as info:
            asyncio.run(_collect_stream(provider))
        assert str(info.value) == "TestProvider stream ended with finish_reason='error'"
        assert info.value.partial_text == "cut"

    def test_the_stream_request_defaults_are_pinned(self, monkeypatch):
        import asyncio

        provider = _make_provider()
        seen: list = []
        original = provider._build_stream_body

        def spy(*args, **kwargs):
            seen.append(args)
            return original(*args, **kwargs)

        monkeypatch.setattr(provider, "_build_stream_body", spy)
        ok = _FakeStreamResponse(status=200, lines=("data: [DONE]",))
        provider._client = _SequenceStreamClient([ok])
        asyncio.run(_collect_stream(provider))
        _prompt, _system, temperature, max_tokens, json_mode, thinking, json_schema, images = seen[0]
        assert temperature == 0.7
        assert max_tokens == 0
        assert (json_mode, thinking, json_schema, images) == (False, None, None, None)


class TestApplyStreamChunkAndUsage:
    """Per-chunk finish reason, citations, tool-call fragments, reasoning and progress, and the usage fields (openai_compat 537-600)."""

    def test_a_finish_reason_is_recorded_when_present(self):
        provider = _make_provider()
        provider._apply_stream_chunk({"choices": [{"delta": {}, "finish_reason": "stop"}]}, {})
        assert provider._last_finish_reason == "stop"

    def test_a_chunk_without_a_finish_reason_leaves_it_alone(self):
        provider = _make_provider()
        provider._apply_stream_chunk({"choices": [{"delta": {"content": "x"}}]}, {})
        assert provider._last_finish_reason is None

    def test_citations_on_the_chunk_are_kept_as_a_list(self):
        provider = _make_provider()
        provider._apply_stream_chunk({"choices": [{"delta": {}}], "citations": ["https://a.example"]}, {})
        assert provider.last_citations == ["https://a.example"]

    def test_tool_call_deltas_are_accumulated_into_the_fragments(self):
        import pyutilz.llm.openai_compat as oc_module

        delta = {"tool_calls": [{"index": 0, "id": "c1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]}
        expected: dict = {}
        oc_module._accumulate_stream_tool_calls(expected, delta["tool_calls"])
        provider = _make_provider()
        fragments: dict = {}
        provider._apply_stream_chunk({"choices": [{"delta": delta}]}, fragments)
        assert fragments == expected
        assert fragments != {}

    def test_reasoning_is_folded_into_the_collected_fragments(self):
        import pyutilz.llm.openai_compat as oc_module

        delta = {"reasoning_content": "think"}
        piece = oc_module._reasoning.from_payload(delta)
        provider = _make_provider()
        before = provider._reasoning_fragments
        provider._apply_stream_chunk({"choices": [{"delta": delta}]}, {})
        assert provider._reasoning_fragments == oc_module._reasoning.collect(before, piece)

    def test_progress_is_noted_with_the_reasoning_piece_then_the_content(self, monkeypatch):
        import pyutilz.llm.openai_compat as oc_module

        noted: list = []
        monkeypatch.setattr(oc_module, "note_stream_progress", lambda piece, content: noted.append((piece, content)))
        provider = _make_provider()
        delta = {"reasoning_content": "think", "content": "answer"}
        piece = oc_module._reasoning.from_payload(delta)
        provider._apply_stream_chunk({"choices": [{"delta": delta}]}, {})
        assert noted == [(piece, "answer")]

    def test_only_a_string_content_is_returned(self):
        provider = _make_provider()
        assert provider._apply_stream_chunk({"choices": [{"delta": {"content": "hi"}}]}, {}) == "hi"
        assert provider._apply_stream_chunk({"choices": [{"delta": {"content": 5}}]}, {}) is None

    def test_an_empty_usage_block_adds_nothing(self):
        provider = _make_provider()
        provider._record_usage({})
        assert (provider.total_prompt_tokens, provider.total_completion_tokens) == (0, 0)
        assert (provider.total_cache_hit_tokens, provider.total_reasoning_tokens, provider.total_cache_write_tokens) == (0, 0, 0)

    def test_the_prompt_and_completion_counts_are_summed(self):
        provider = _make_provider()
        provider._record_usage({"prompt_tokens": 11, "completion_tokens": 4})
        assert (provider.total_prompt_tokens, provider.total_completion_tokens) == (11, 4)

    def test_a_null_count_is_taken_as_zero(self):
        provider = _make_provider()
        provider._record_usage({"prompt_tokens": None, "completion_tokens": None})
        assert (provider.total_prompt_tokens, provider.total_completion_tokens) == (0, 0)

    def test_the_legacy_cache_hit_key_is_read(self):
        provider = _make_provider()
        provider._record_usage({"prompt_cache_hit_tokens": 5})
        assert provider.total_cache_hit_tokens == 5

    def test_the_nested_cached_tokens_key_is_read_when_the_legacy_one_is_absent(self):
        provider = _make_provider()
        provider._record_usage({"prompt_tokens_details": {"cached_tokens": 7}})
        assert provider.total_cache_hit_tokens == 7

    def test_reasoning_tokens_are_read_from_the_completion_details(self):
        provider = _make_provider()
        provider._record_usage({"completion_tokens_details": {"reasoning_tokens": 3}})
        assert provider.total_reasoning_tokens == 3

    def test_cache_writes_are_read_from_the_chat_details(self):
        provider = _make_provider()
        provider._record_usage({"prompt_tokens_details": {"cache_write_tokens": 4}})
        assert provider.total_cache_write_tokens == 4

    def test_cache_writes_are_read_from_the_responses_details_when_the_chat_one_is_absent(self):
        provider = _make_provider()
        provider._record_usage({"input_tokens_details": {"cache_write_tokens": 6}})
        assert provider.total_cache_write_tokens == 6


class TestUsageAccumulationAndLog:
    """Per-call sums, the cache-write total, the surcharge arguments, the cumulative log line (openai_compat 604-635)."""

    def test_the_per_call_usage_sums_every_post_within_one_call(self):
        provider = _make_provider()
        provider._record_usage({"prompt_tokens": 10, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 3}})
        provider._record_usage({"prompt_tokens": 5, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 2}})
        assert provider._last_usage["input_tokens"] == 15
        assert provider._last_usage["cached_tokens"] == 5

    def test_the_per_call_usage_has_the_four_documented_keys(self):
        provider = _make_provider()
        provider._record_usage({"prompt_tokens": 7, "completion_tokens": 4, "completion_tokens_details": {"reasoning_tokens": 3}, "prompt_cache_hit_tokens": 2})
        assert set(provider._last_usage) == {"input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens"}
        assert provider._last_usage["input_tokens"] == 7
        assert provider._last_usage["reasoning_tokens"] == 3
        assert provider._last_usage["cached_tokens"] == 2
        assert provider._last_usage["output_tokens"] == provider._compute_billed_output(4, 3)

    def test_the_last_post_reasoning_count_is_that_of_the_latest_post_alone(self):
        provider = _make_provider()
        provider._record_usage({"completion_tokens_details": {"reasoning_tokens": 9}})
        provider._record_usage({"completion_tokens_details": {"reasoning_tokens": 2}})
        assert provider._last_post_reasoning_tokens == 2

    def test_the_cache_write_total_accumulates_across_calls(self):
        provider = _make_provider()
        provider._record_usage({"prompt_tokens_details": {"cache_write_tokens": 4}})
        provider._record_usage({"prompt_tokens_details": {"cache_write_tokens": 6}})
        assert provider.total_cache_write_tokens == 10

    def test_the_call_count_advances_once_per_usage_record(self):
        provider = _make_provider()
        provider._record_usage({})
        provider._record_usage({})
        assert provider._call_count == 2

    def test_the_provider_specific_usage_hook_sees_the_usage_block(self, monkeypatch):
        provider = _make_provider()
        seen: list = []
        monkeypatch.setattr(provider, "_track_provider_specific_usage", lambda usage: seen.append(usage))
        provider._record_usage({"prompt_tokens": 1})
        assert seen == [{"prompt_tokens": 1}]

    def test_the_surcharge_receives_its_four_values_in_order(self, monkeypatch):
        provider = _make_provider()
        calls: list = []
        monkeypatch.setattr(provider, "_add_long_context_surcharge", lambda *args: calls.append(args))
        provider._record_usage(
            {
                "prompt_tokens": 11,
                "completion_tokens": 4,
                "completion_tokens_details": {"reasoning_tokens": 3},
                "prompt_cache_hit_tokens": 5,
                "prompt_tokens_details": {"cache_write_tokens": 2},
            }
        )
        assert calls == [(11, provider._compute_billed_output(4, 3), 5, 2)]

    def test_the_log_line_names_the_reasoning_tokens_when_there_are_some(self, monkeypatch):
        import pyutilz.llm.openai_compat as oc_module

        logged: list = []

        class _InfoLog:
            def info(self, fmt, *args):
                logged.append((fmt, args))

        monkeypatch.setattr(oc_module, "logger", _InfoLog())
        provider = _make_provider()
        provider._record_usage({"prompt_tokens": 11, "completion_tokens": 4, "completion_tokens_details": {"reasoning_tokens": 3}, "prompt_cache_hit_tokens": 5})
        fmt, args = logged[0]
        assert fmt == "%s [call #%d] %d prompt (%d cached) + %d completion%s | cumulative: %d in, %d out"
        assert args == ("TestProvider", 1, 11, 5, 4, " (3 reasoning)", 11, 4)

    def test_the_log_line_has_no_reasoning_suffix_without_reasoning(self, monkeypatch):
        import pyutilz.llm.openai_compat as oc_module

        logged: list = []

        class _InfoLog:
            def info(self, fmt, *args):
                logged.append((fmt, args))

        monkeypatch.setattr(oc_module, "logger", _InfoLog())
        provider = _make_provider()
        provider._record_usage({"prompt_tokens": 2, "completion_tokens": 1})
        assert logged[0][1][5] == ""

    def test_the_default_generation_temperature_is_point_seven(self):
        import inspect

        assert inspect.signature(OpenAICompatibleProvider.generate).parameters["temperature"].default == 0.7


class TestCumulativeCacheAndReasoningTotals:
    """The session totals for cache hits and reasoning tokens add up across calls (openai_compat 604-605)."""

    def test_the_cache_hit_total_adds_up_across_calls(self):
        provider = _make_provider()
        provider._record_usage({"prompt_cache_hit_tokens": 3})
        provider._record_usage({"prompt_cache_hit_tokens": 2})
        assert provider.total_cache_hit_tokens == 5

    def test_the_reasoning_total_adds_up_across_calls(self):
        provider = _make_provider()
        provider._record_usage({"completion_tokens_details": {"reasoning_tokens": 4}})
        provider._record_usage({"completion_tokens_details": {"reasoning_tokens": 6}})
        assert provider.total_reasoning_tokens == 10


class TestGenerateStartsWithAFreshCallStateAndPrepares:
    """generate resets the per-call state, then awaits the provider preparation before it builds a request (openai_compat 692-695)."""

    def test_the_per_call_state_is_reset_and_then_the_provider_is_prepared(self, monkeypatch):
        import asyncio

        provider = _make_provider()
        order: list = []
        monkeypatch.setattr(provider, "_reset_per_call_state", lambda: order.append("reset"))

        async def prepare():
            order.append("prepare")
            raise _PreparationOrderError()

        monkeypatch.setattr(provider, "_async_prepare", prepare)

        class _TouchedClient:
            def __getattr__(self, name):
                raise _PreparationOrderError("the request was built before the provider was prepared")

        provider._client = _TouchedClient()
        with pytest.raises(_PreparationOrderError):
            asyncio.run(provider.generate("hi"))
        assert order == ["reset", "prepare"]


class TestGenerateBudgetArgumentsAndReissue:
    """The token budget, the order of the request arguments, and the one re-issue without response_format (openai_compat 696-744)."""

    def _spy_budget(self, monkeypatch, provider):
        seen: list = []
        monkeypatch.setattr(provider, "fit_max_tokens_to_context", lambda budget, prompt, system: seen.append(budget) or budget)

        async def post(body, repairing=False):
            return "ok"

        monkeypatch.setattr(provider, "_post_and_unwrap", post)
        return seen

    def test_no_max_tokens_means_no_budget_ceiling(self, monkeypatch):
        import asyncio

        provider = _make_provider()
        seen = self._spy_budget(monkeypatch, provider)
        asyncio.run(provider.generate("hi", max_tokens=0))
        assert seen == [None]

    def test_a_positive_max_tokens_is_the_budget(self, monkeypatch):
        import asyncio

        provider = _make_provider()
        seen = self._spy_budget(monkeypatch, provider)
        asyncio.run(provider.generate("hi", max_tokens=1))
        assert seen == [1]

    def test_the_request_arguments_are_passed_in_their_declared_order(self, monkeypatch):
        import asyncio

        provider = _make_provider()
        monkeypatch.setattr(provider, "fit_max_tokens_to_context", lambda budget, prompt, system: budget)
        seen: list = []

        def spy_body(*args):
            seen.append(args)
            return {"model": "test-model"}

        monkeypatch.setattr(provider, "_request_body", spy_body)

        async def post(body, repairing=False):
            return "ok"

        monkeypatch.setattr(provider, "_post_and_unwrap", post)
        asyncio.run(
            provider.generate(
                "P", system="S", temperature=0.3, max_tokens=5, thinking="low", images=["img"], tools=[{"t": 1}], tool_choice="auto", extra_body={"seed": 1}
            )
        )
        prompt, system, temperature, budget, rf, thinking, images, tools, tool_choice, extra_body = seen[0]
        assert (prompt, system, temperature, budget) == ("P", "S", 0.3, 5)
        assert (thinking, images, tools, tool_choice, extra_body) == ("low", ["img"], [{"t": 1}], "auto", {"seed": 1})

    def test_an_empty_truncated_answer_under_json_mode_is_re_issued_once_without_the_format(self, monkeypatch):
        import asyncio

        from pyutilz.llm.exceptions import LLMTruncationError

        provider = _make_provider()
        monkeypatch.setattr(provider, "fit_max_tokens_to_context", lambda budget, prompt, system: budget)
        recorder = _WarnRecorder()
        import pyutilz.llm.openai_compat as oc_module

        monkeypatch.setattr(oc_module, "logger", recorder)
        bodies: list = []

        async def post(body, repairing=False):
            bodies.append(body)
            if len(bodies) == 1:
                raise LLMTruncationError("cut", finish_reason="length", partial_text="")
            return "fallback answer"

        monkeypatch.setattr(provider, "_post_and_unwrap", post)
        async def run_and_read_flag():
            answer = await provider.generate("hi", json_mode=True)
            return answer, provider.last_json_mode_fallback

        answer, flag = asyncio.run(run_and_read_flag())
        assert answer == "fallback answer"
        assert flag is True
        assert "response_format" in bodies[0]
        assert "response_format" not in bodies[1]
        fmt, args = recorder.warnings[0]
        assert fmt == (
            "%s/%s returned an EMPTY completion cut off by max_tokens under response_format " "- re-issuing once without it before reporting truncation"
        )
        assert args == ("TestProvider", "test-model")

    def test_a_truncation_with_partial_text_is_not_re_issued(self, monkeypatch):
        import asyncio

        from pyutilz.llm.exceptions import LLMTruncationError

        provider = _make_provider()
        monkeypatch.setattr(provider, "fit_max_tokens_to_context", lambda budget, prompt, system: budget)
        calls: list = []

        async def post(body, repairing=False):
            calls.append(body)
            raise LLMTruncationError("cut", finish_reason="length", partial_text="half")

        monkeypatch.setattr(provider, "_post_and_unwrap", post)
        with pytest.raises(LLMTruncationError):
            asyncio.run(provider.generate("hi", json_mode=True))
        assert len(calls) == 1

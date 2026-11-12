"""OpenAI-compatible provider tests: usage, cost, rate limits and error labels."""

from pyutilz.core import jsonlib
import pytest
from pyutilz.llm.exceptions import LLMProviderError
from pyutilz.llm.openai_compat import OpenAICompatibleProvider
from tests.openai_compat_support import (
    _BareCloseClient,
    _DyingStreamResponse,
    _ExitRecordingClient,
    _FakeStreamResponse,
    _JsonPostResponse,
    _MidStreamFailure,
    _PostClient,
    _RequestedBodyResponse,
    _RequestlessBodyResponse,
    _SequenceStreamClient,
    _StatusBodyResponse,
    _StatusStreamResponse,
    _Tier,
    _WarnRecorder,
    _collect_stream,
    _content_line,
    _drain_with_box,
    _make_provider,
    _no_sleep,
)

httpx = pytest.importorskip("httpx")


class TestGenerateEmptyCompletion:
    """An empty completion under response_format is re-issued once, and an empty answer that stays empty is refused with its diagnosis (openai_compat 742-753)."""

    def _posts(self, monkeypatch, provider, answers):
        bodies: list = []
        queue = list(answers)

        async def post(body, repairing=False):
            bodies.append(body)
            return queue.pop(0)

        monkeypatch.setattr(provider, "fit_max_tokens_to_context", lambda budget, prompt, system: budget)
        monkeypatch.setattr(provider, "_post_and_unwrap", post)
        return bodies

    def test_an_empty_answer_under_response_format_is_re_issued_and_the_second_answer_is_returned(self, monkeypatch):
        import asyncio

        import pyutilz.llm.openai_compat as oc_module

        provider = _make_provider()
        recorder = _WarnRecorder()
        monkeypatch.setattr(oc_module, "logger", recorder)
        bodies = self._posts(monkeypatch, provider, [None, "second"])

        async def run_and_read_flag():
            answer = await provider.generate("hi", json_mode=True)
            return answer, provider.last_json_mode_fallback

        answer, flag = asyncio.run(run_and_read_flag())
        assert answer == "second"
        assert flag is True
        assert "response_format" in bodies[0] and "response_format" not in bodies[1]
        fmt, args = recorder.warnings[0]
        assert fmt == "%s/%s returned an empty completion under response_format - re-issuing once without it"
        assert args == ("TestProvider", "test-model")

    def test_an_answer_that_stays_empty_is_refused_with_its_diagnosis(self, monkeypatch):
        import asyncio

        provider = _make_provider()
        self._posts(monkeypatch, provider, [None, None])
        with pytest.raises(LLMProviderError) as info:
            asyncio.run(provider.generate("hi", json_mode=True))
        assert str(info.value) == "TestProvider/test-model returned an empty completion (finish_reason=None, reasoning_tokens=0, response_format=dropped)"

    def test_an_empty_answer_without_a_response_format_is_refused_as_not_sent(self, monkeypatch):
        import asyncio

        provider = _make_provider()
        self._posts(monkeypatch, provider, [None])
        with pytest.raises(LLMProviderError) as info:
            asyncio.run(provider.generate("hi", json_mode=False))
        assert str(info.value) == "TestProvider/test-model returned an empty completion (finish_reason=None, reasoning_tokens=0, response_format=not sent)"

    def test_a_refusal_names_the_finish_reason_the_stream_reported(self, monkeypatch):
        import asyncio

        provider = _make_provider()
        provider._last_finish_reason = "stop"

        async def post(body, repairing=False):
            provider._last_finish_reason = "stop"
            return None

        monkeypatch.setattr(provider, "fit_max_tokens_to_context", lambda budget, prompt, system: budget)
        monkeypatch.setattr(provider, "_post_and_unwrap", post)
        with pytest.raises(LLMProviderError) as info:
            asyncio.run(provider.generate("hi", json_mode=False))
        assert "(finish_reason='stop', " in str(info.value)


class TestPostAndUnwrap:
    """One POST to the chat path, the special-status hook, the response-metadata hook, and the unwrapped completion (openai_compat 773-790)."""

    _PAYLOAD = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    def test_the_post_goes_to_the_chat_completions_path_with_the_body(self):
        import asyncio

        provider = _make_provider()
        client = _PostClient(_JsonPostResponse(self._PAYLOAD))
        provider._client = client
        assert asyncio.run(provider._post_and_unwrap({"model": "test-model"})) == "ok"
        assert client.calls[0][0] == "/chat/completions"
        assert client.calls[0][1] == {"model": "test-model"}

    def test_the_special_status_hook_sees_every_response(self, monkeypatch):
        import asyncio

        provider = _make_provider()
        provider._client = _PostClient(_JsonPostResponse(self._PAYLOAD))
        seen: list = []
        monkeypatch.setattr(provider, "_handle_special_status", lambda resp: seen.append(resp.status_code))
        asyncio.run(provider._post_and_unwrap({"model": "test-model"}))
        assert seen == [200]

    def test_the_response_metadata_hook_gets_the_envelope(self, monkeypatch):
        import asyncio

        provider = _make_provider()
        provider._client = _PostClient(_JsonPostResponse(self._PAYLOAD))
        tracked: list = []
        monkeypatch.setattr(provider, "_track_provider_specific_response", lambda data: tracked.append(data))
        asyncio.run(provider._post_and_unwrap({"model": "test-model"}))
        assert tracked == [self._PAYLOAD]


class TestParameterRejectionAndUnwrap:
    """The rejection detail, the one repair with its warning, the final error text, the message extras and the completion's finish handling (openai_compat 803-869)."""

    def _recording_repair(self, monkeypatch, provider, repaired=None):
        seen: list = []

        def hook(body, status, detail):
            seen.append((body, status, detail))
            return repaired

        monkeypatch.setattr(provider, "_body_after_rejected_request", hook)
        return seen

    def test_the_rejection_detail_is_the_error_message(self, monkeypatch):
        provider = _make_provider()
        seen = self._recording_repair(monkeypatch, provider)
        with pytest.raises(LLMProviderError):
            provider._repaired_body_or_raise(_StatusBodyResponse(400, {"error": {"message": "bad param"}}), {"m": 1}, True)
        assert seen == [({"m": 1}, 400, "bad param")]

    def test_without_an_error_message_the_raw_text_is_the_detail(self, monkeypatch):
        provider = _make_provider()
        seen = self._recording_repair(monkeypatch, provider)
        with pytest.raises(LLMProviderError):
            provider._repaired_body_or_raise(_StatusBodyResponse(400, {"other": 1}, text="raw body"), {}, True)
        assert seen[0][2] == "raw body"

    def test_a_non_json_body_gives_the_raw_text_as_detail(self, monkeypatch):
        provider = _make_provider()
        seen = self._recording_repair(monkeypatch, provider)
        with pytest.raises(LLMProviderError):
            provider._repaired_body_or_raise(_StatusBodyResponse(400, None, text="<html>"), {}, True)
        assert seen[0][2] == "<html>"

    def test_a_repair_is_returned_and_warned_once(self, monkeypatch, caplog):
        import pyutilz.llm.openai_compat as oc_module

        provider = _make_provider()
        logged: list = []

        class _WarnLog:
            def warning(self, fmt, *args, **kwargs):
                logged.append((fmt, args))

        monkeypatch.setattr(oc_module, "logger", _WarnLog())
        self._recording_repair(monkeypatch, provider, repaired={"fixed": True})
        out = provider._repaired_body_or_raise(_StatusBodyResponse(400, {"error": {"message": "bad param"}}), {}, False)
        assert out == {"fixed": True}
        fmt, args = logged[0]
        assert fmt == "%s/%s rejected a request parameter (HTTP %s) - re-issuing once with it adjusted: %s"
        assert args == ("TestProvider", "test-model", 400, "bad param")

    def test_a_long_detail_is_cut_to_one_hundred_and_sixty_characters_in_the_warning(self, monkeypatch):
        import pyutilz.llm.openai_compat as oc_module

        provider = _make_provider()
        logged: list = []

        class _WarnLog:
            def warning(self, fmt, *args, **kwargs):
                logged.append(args)

        monkeypatch.setattr(oc_module, "logger", _WarnLog())
        self._recording_repair(monkeypatch, provider, repaired={"fixed": True})
        provider._repaired_body_or_raise(_StatusBodyResponse(400, {"error": {"message": "x" * 300}}), {}, False)
        assert len(logged[0][-1]) == 160

    def test_a_repaired_body_that_is_refused_again_raises_with_the_exact_message(self, monkeypatch):
        provider = _make_provider()
        self._recording_repair(monkeypatch, provider, repaired={"fixed": True})
        with pytest.raises(LLMProviderError) as info:
            provider._repaired_body_or_raise(_StatusBodyResponse(400, {"error": {"message": "bad param"}}), {}, True)
        assert str(info.value) == "TestProvider API error 400: bad param"

    def test_message_extras_keep_tool_calls_and_citations_only_when_they_are_lists(self):
        provider = _make_provider()
        provider._capture_message_extras({"tool_calls": [{"id": "t"}], "citations": ["u"]})
        assert provider.last_tool_calls == [{"id": "t"}]
        assert provider.last_citations == ["u"]
        provider._capture_message_extras({"tool_calls": "no", "citations": "no"})
        assert provider.last_tool_calls == []
        assert provider.last_citations == []

    def test_message_reasoning_is_folded_into_the_collected_fragments(self):
        import pyutilz.llm.openai_compat as oc_module

        message = {"reasoning_content": "think"}
        provider = _make_provider()
        before = provider._reasoning_fragments
        provider._capture_message_extras(message)
        assert provider._reasoning_fragments == oc_module._reasoning.collect(before, oc_module._reasoning.from_message(message))

    def test_a_completion_without_choices_is_refused(self):
        provider = _make_provider()
        with pytest.raises(LLMProviderError) as info:
            provider._unwrap_completion({"choices": []}, _StatusBodyResponse(200))
        assert str(info.value) == "TestProvider returned no choices"

    def test_a_missing_finish_reason_is_unknown(self):
        provider = _make_provider()
        provider._unwrap_completion({"choices": [{"message": {"content": "ok"}}]}, _StatusBodyResponse(200))
        assert provider._last_finish_reason == "unknown"

    def test_an_error_finish_carries_the_partial_text_and_the_exact_message(self):
        from pyutilz.llm.exceptions import LLMStreamInterruptedError

        provider = _make_provider()
        data = {"choices": [{"message": {"content": "cut"}, "finish_reason": "error"}]}
        with pytest.raises(LLMStreamInterruptedError) as info:
            provider._unwrap_completion(data, _StatusBodyResponse(200))
        assert str(info.value) == "TestProvider response ended with finish_reason='error'"
        assert info.value.partial_text == "cut"

    def test_a_length_finish_is_refused_with_the_exact_truncation_message(self):
        from pyutilz.llm.exceptions import LLMTruncationError

        provider = _make_provider()
        data = {"choices": [{"message": {"content": "part"}, "finish_reason": "length"}]}
        with pytest.raises(LLMTruncationError) as info:
            provider._unwrap_completion(data, _StatusBodyResponse(200))
        assert str(info.value) == "TestProvider response truncated by max_tokens (finish_reason='length')"

    def test_a_completion_returns_its_content(self):
        provider = _make_provider()
        assert provider._unwrap_completion({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}, _StatusBodyResponse(200)) == "ok"

    def test_a_tool_call_only_reply_returns_the_empty_string(self):
        provider = _make_provider()
        data = {"choices": [{"message": {"tool_calls": [{"id": "t"}]}, "finish_reason": "tool_calls"}]}
        assert provider._unwrap_completion(data, _StatusBodyResponse(200)) == ""
        assert provider.last_tool_calls == [{"id": "t"}]


class TestCostAccountingAndGenerateJsonDefaults:
    """Per-call cost, the session cost summary, the long-context surcharge tally, and generate_json's defaults (openai_compat 871-961)."""

    def test_the_estimate_is_the_input_rate_plus_the_output_rate_per_million_tokens(self):
        provider = _make_provider()
        assert provider.estimate_cost(1_000_000, 1_000_000) == pytest.approx(3.0, abs=1e-12)
        assert provider.estimate_cost(500_000, 250_000) == pytest.approx(0.5 + 0.5, abs=1e-12)

    def test_the_session_summary_separates_cache_misses_and_prices_each_part(self):
        provider = _make_provider()
        provider.total_prompt_tokens = 1_000_000
        provider.total_cache_hit_tokens = 100_000
        provider.total_completion_tokens = 500_000
        summary = provider.get_session_cost()
        assert summary["cache_miss_tokens"] == 900_000
        assert summary["input_cost_usd"] == pytest.approx(0.9 + 0.01, abs=1e-12)
        assert summary["output_cost_usd"] == pytest.approx(1.0, abs=1e-12)
        assert summary["total_cost_usd"] == pytest.approx(0.9 + 0.01 + 1.0, abs=1e-12)

    def test_the_session_summary_has_the_documented_keys(self):
        summary = _make_provider().get_session_cost()
        assert set(summary) == {
            "calls",
            "prompt_tokens",
            "cache_hit_tokens",
            "cache_miss_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "input_cost_usd",
            "output_cost_usd",
            "long_context_surcharge_usd",
            "long_context_calls",
            "total_cost_usd",
        }

    def test_cache_writes_are_taken_out_of_the_miss_count_and_priced_at_their_own_rate(self, monkeypatch):
        provider = _make_provider()
        provider.total_prompt_tokens = 1_000_000
        provider.total_cache_hit_tokens = 100_000
        provider.total_cache_write_tokens = 200_000
        monkeypatch.setattr(provider, "_cache_write_cost_per_1m", lambda model: 1.25)
        summary = provider.get_session_cost()
        assert summary["cache_miss_tokens"] == 700_000
        assert summary["input_cost_usd"] == pytest.approx(0.7 + 0.01 + 0.25, abs=1e-12)

    def test_a_surcharge_is_tallied_for_each_call_that_crosses_the_tier(self, monkeypatch):
        import pyutilz.llm.openai_compat as oc_module

        provider = _make_provider()
        monkeypatch.setattr(provider, "_long_context_tier", lambda model: _Tier(True))
        seen: list = []

        def fake_surcharge(pricing, prompt, billed, cache_hit, cache_write):
            seen.append((prompt, billed, cache_hit, cache_write))
            return (1.0, 2.0)

        monkeypatch.setattr(oc_module, "long_context_surcharge", fake_surcharge)
        provider._add_long_context_surcharge(300_000, 40, 5, 6)
        provider._add_long_context_surcharge(300_000, 40, 5, 6)
        assert seen == [(300_000, 40, 5, 6), (300_000, 40, 5, 6)]
        assert provider._long_context_surcharge_usd == pytest.approx(6.0)
        assert provider._long_context_calls == 2

    def test_a_call_below_the_tier_adds_no_surcharge(self, monkeypatch):
        provider = _make_provider()
        monkeypatch.setattr(provider, "_long_context_tier", lambda model: _Tier(False))
        provider._add_long_context_surcharge(10, 1, 0, 0)
        assert provider._long_context_calls == 0
        assert provider._long_context_surcharge_usd == 0.0

    def test_a_model_without_a_tier_adds_no_surcharge(self):
        provider = _make_provider()
        provider._add_long_context_surcharge(10**9, 1, 0, 0)
        assert provider._long_context_calls == 0

    def test_generate_json_defaults_are_point_three_and_no_token_ceiling(self):
        import inspect

        params = inspect.signature(OpenAICompatibleProvider.generate_json).parameters
        assert params["temperature"].default == 0.3
        assert params["max_tokens"].default == 0
        assert params["force_json_mode"].default is True

    def test_generate_json_without_thinking_omits_the_field(self, monkeypatch):
        import asyncio

        provider = _make_provider()
        calls: list = []

        async def spy(prompt, system, temperature, max_tokens, **kwargs):
            calls.append((prompt, system, temperature, max_tokens, kwargs))
            return {}

        monkeypatch.setattr(provider, "_generate_json_via", spy)
        asyncio.run(provider.generate_json("p"))
        prompt, system, temperature, max_tokens, kwargs = calls[0]
        assert (prompt, system, temperature, max_tokens) == ("p", None, 0.3, 0)
        assert kwargs["json_mode"] is True
        assert "thinking" not in kwargs

    def test_generate_json_with_thinking_passes_it_through(self, monkeypatch):
        import asyncio

        provider = _make_provider()
        calls: list = []

        async def spy(prompt, system, temperature, max_tokens, **kwargs):
            calls.append(kwargs)
            return {}

        monkeypatch.setattr(provider, "_generate_json_via", spy)
        asyncio.run(provider.generate_json("p", thinking="low", force_json_mode=False))
        assert calls[0]["thinking"] == "low"
        assert calls[0]["json_mode"] is False


class TestInputRateIsMultipliedNotDivided:
    """The input rate scales the token count: the test provider's rate of 1.0 cannot tell multiplication from division (openai_compat 931)."""

    def test_the_input_cost_is_the_rate_times_the_millions_of_tokens(self, monkeypatch):
        provider = _make_provider()
        monkeypatch.setattr(provider, "_input_cost_per_1m", lambda model: 2.5)
        assert provider.estimate_cost(2_000_000, 0) == pytest.approx(5.0, abs=1e-12)


class TestClientSetupAndAccumulatedUsage:
    """The client headers and timeout, the per-call metadata names, the per-call sums across calls, and a data line with no space (openai_compat 103-122, 618-619, 368)."""

    def test_the_metadata_names_include_the_json_schema_and_fallback_flags(self):
        names = _make_provider()._PERCALL_METADATA_ATTRS
        assert "_last_json_schema_applied" in names
        assert "last_json_mode_fallback" in names

    def test_the_client_sends_the_bearer_key_and_the_json_content_type(self):
        provider = _make_provider()
        assert provider._client.headers["Authorization"] == "Bearer test-key"
        assert provider._client.headers["Content-Type"] == "application/json"

    def test_the_connect_timeout_is_ten_seconds(self):
        assert _make_provider()._client.timeout.connect == 10.0

    def test_the_per_call_output_and_reasoning_sums_add_up_across_posts(self):
        provider = _make_provider()
        provider._record_usage({"completion_tokens": 4, "completion_tokens_details": {"reasoning_tokens": 3}})
        provider._record_usage({"completion_tokens": 6, "completion_tokens_details": {"reasoning_tokens": 5}})
        assert provider._last_usage["reasoning_tokens"] == 8
        assert provider._last_usage["output_tokens"] == provider._compute_billed_output(4, 3) + provider._compute_billed_output(6, 5)

    def test_a_data_line_with_no_space_after_the_colon_is_read(self):
        import asyncio

        lines = (_content_line("tight").replace("data: ", "data:"), "data: [DONE]")
        yielded, _box, _state = asyncio.run(_drain_with_box(_make_provider(), lines))
        assert yielded == ["tight"]


class TestCloseWithoutAnOpenFlag:
    """A client that does not report is_closed counts as open for both the current client and the per-loop copies (openai_compat 325, 330)."""

    def test_a_current_client_without_the_flag_is_closed(self):
        import asyncio

        provider = _make_provider()
        current = _BareCloseClient()
        provider._client = current
        asyncio.run(provider._close())
        assert current.close_calls == 1

    def test_a_per_loop_copy_without_the_flag_is_closed(self):
        import asyncio
        import weakref

        provider = _make_provider()
        provider._client = _BareCloseClient()
        copy = _BareCloseClient()
        loop = asyncio.new_event_loop()
        per_loop = weakref.WeakKeyDictionary()
        per_loop[loop] = copy
        provider.__dict__["__client_per_loop"] = per_loop
        asyncio.run(provider._close())
        loop.close()
        assert copy.close_calls == 1


class TestFirstChunkTrackedBeforeAnInterruption:
    """The first chunk's response metadata is recorded as it arrives, so a stream that dies mid-answer keeps its id (openai_compat 378)."""

    def test_a_stream_that_dies_after_its_first_chunk_still_tracks_that_chunk(self, monkeypatch):
        import asyncio

        provider = _make_provider()
        tracked: list = []
        monkeypatch.setattr(provider, "_track_provider_specific_response", lambda chunk: tracked.append(chunk))
        provider._client = _SequenceStreamClient([_DyingStreamResponse(status=200, lines=(_content_line("half"),))])
        with pytest.raises(RuntimeError):
            asyncio.run(_collect_stream(provider))
        assert tracked == [{"choices": [{"delta": {"content": "half"}}]}]


class TestGenerateStreamArgumentOrderAndTeardown:
    """The stream body arguments in their order, the attempt state at the start, closing the stream on an early stop, and no retry after content has been yielded (openai_compat 463, 482, 490, 503)."""

    def test_the_stream_body_gets_each_optional_argument_in_its_place(self, monkeypatch):
        import asyncio

        provider = _make_provider()
        seen: list = []
        original = provider._build_stream_body

        def spy(*args, **kwargs):
            seen.append(args)
            return original(*args, **kwargs)

        monkeypatch.setattr(provider, "_build_stream_body", spy)
        provider._client = _SequenceStreamClient([_FakeStreamResponse(status=200, lines=("data: [DONE]",))])
        asyncio.run(_collect_stream(provider, json_mode=True, thinking="low", json_schema={"name": "x"}, images=["img"]))
        _prompt, _system, _temp, _max, json_mode, thinking, json_schema, images = seen[0]
        assert (json_mode, thinking, json_schema, images) == (True, "low", {"name": "x"}, ["img"])

    def test_the_attempt_state_starts_with_no_response_and_no_generation_id(self, monkeypatch):
        import asyncio

        provider = _make_provider()
        _no_sleep(monkeypatch, provider)
        states: list = []
        monkeypatch.setattr(provider, "_note_aborted_stream_attempt", lambda attempt, state, exc: states.append(dict(state)))
        ok = _FakeStreamResponse(status=200, lines=(_content_line("x"), "data: [DONE]"))
        provider._client = _SequenceStreamClient([_StatusStreamResponse(status=503), ok])
        asyncio.run(_collect_stream(provider))
        assert states[0] == {"generation_id": None, "response_started": False, "generated_chunks": 0}

    def test_closing_the_stream_early_closes_the_upstream_response(self):
        import asyncio

        provider = _make_provider()
        client = _ExitRecordingClient([_FakeStreamResponse(status=200, lines=(_content_line("a"), _content_line("b"), "data: [DONE]"))])
        provider._client = client

        async def take_one_then_close():
            gen = provider.generate_stream("hi")
            first = await gen.__anext__()
            await gen.aclose()
            return first, client.contexts[0].exited

        first, exited_at_once = asyncio.run(take_one_then_close())
        assert first == "a"
        assert exited_at_once is True

    def test_a_failure_after_content_is_raised_and_not_retried(self, monkeypatch):
        import asyncio

        provider = _make_provider()
        waits = _no_sleep(monkeypatch, provider)
        client = _SequenceStreamClient([_MidStreamFailure(status=200, lines=(_content_line("partial"),)), _FakeStreamResponse(status=200, lines=(_content_line("again"), "data: [DONE]"))])
        provider._client = client
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(_collect_stream(provider))
        assert waits == []
        assert len(client.calls) == 1


class TestStreamRetryWarningAndTruncationFinish:
    """The retry warning's arguments and its cut-off reason, and the finish reason a truncation carries (openai_compat 522-524, 537)."""

    def test_a_retried_stream_open_is_warned_with_attempt_error_and_wait(self, monkeypatch):
        import asyncio

        import pyutilz.llm.openai_compat as oc_module

        recorder = _WarnRecorder()
        monkeypatch.setattr(oc_module, "logger", recorder)
        provider = _make_provider()
        _no_sleep(monkeypatch, provider, wait=7.0)
        ok = _FakeStreamResponse(status=200, lines=(_content_line("hi"), "data: [DONE]"))
        provider._client = _SequenceStreamClient([_StatusStreamResponse(status=503), ok])
        asyncio.run(_collect_stream(provider))
        assert len(recorder.warnings) == 1
        fmt, args = recorder.warnings[0]
        assert fmt == "Streaming attempt %d failed (%s: %s), retrying in %.0fs..."
        assert args[0] == 1
        assert args[1] == "HTTPStatusError"
        assert args[3] == 7.0

    def test_the_reason_in_the_retry_warning_is_cut_to_two_hundred_characters(self, monkeypatch):
        import asyncio

        import pyutilz.llm.openai_compat as oc_module

        recorder = _WarnRecorder()
        monkeypatch.setattr(oc_module, "logger", recorder)
        provider = _make_provider()
        _no_sleep(monkeypatch, provider)
        long_failure = _StatusStreamResponse(status=503)

        def loud():
            request = httpx.Request("POST", "https://test.example.com/chat/completions")
            raise httpx.HTTPStatusError("y" * 250, request=request, response=httpx.Response(503, request=request))

        long_failure.raise_for_status = loud
        ok = _FakeStreamResponse(status=200, lines=(_content_line("hi"), "data: [DONE]"))
        provider._client = _SequenceStreamClient([long_failure, ok])
        asyncio.run(_collect_stream(provider))
        assert len(recorder.warnings[0][1][2]) == 200

    def test_a_length_truncation_carries_its_finish_reason(self):
        import asyncio

        from pyutilz.llm.exceptions import LLMTruncationError

        provider = _make_provider()
        line = "data: " + jsonlib.dumps({"choices": [{"delta": {"content": "ab"}, "finish_reason": "length"}]})
        provider._client = _SequenceStreamClient([_FakeStreamResponse(status=200, lines=(line, "data: [DONE]"))])
        with pytest.raises(LLMTruncationError) as info:
            asyncio.run(_collect_stream(provider))
        assert info.value.finish_reason == "length"


class TestErrorLabelsTheRequest:
    """An error carried by a 200 body is labelled with the request that produced it, or with the chat path when there is none (openai_compat 837-839)."""

    def test_the_error_carries_the_response_request_when_there_is_one(self):
        provider = _make_provider()
        request = httpx.Request("POST", "https://custom.example/v1/chat")
        with pytest.raises(httpx.HTTPStatusError) as info:
            provider._unwrap_completion({"error": {"code": 429, "message": "rate limited"}}, _RequestedBodyResponse(request))
        assert str(info.value.request.url) == "https://custom.example/v1/chat"

    def test_without_a_response_request_the_chat_path_labels_the_error(self):
        provider = _make_provider()
        with pytest.raises(httpx.HTTPStatusError) as info:
            provider._unwrap_completion({"error": {"code": 429, "message": "rate limited"}}, _RequestlessBodyResponse())
        assert info.value.request.method == "POST"
        assert info.value.request.url.path == "/chat/completions"


class TestSessionSummaryWithoutTheSurchargeTally:
    """A provider whose surcharge tally attributes are absent reports zero surcharge and zero tiered calls (openai_compat 923-924)."""

    def test_a_missing_surcharge_tally_reads_as_zero(self):
        provider = _make_provider()
        del provider._long_context_surcharge_usd
        del provider._long_context_calls
        summary = provider.get_session_cost()
        assert summary["long_context_surcharge_usd"] == 0.0
        assert summary["long_context_calls"] == 0
        assert summary["total_cost_usd"] == summary["input_cost_usd"] + summary["output_cost_usd"]


class TestSummaryAndTieredEstimateArithmetic:
    """The summary's tally and total with non-unit rates, and the tiered estimate's pricing call (openai_compat 930-961)."""

    def test_the_summary_adds_a_nonzero_tally_and_prices_with_a_non_unit_rate(self, monkeypatch):
        provider = _make_provider()
        monkeypatch.setattr(provider, "_input_cost_per_1m", lambda model: 2.5)
        provider.total_prompt_tokens = 2_000_000
        provider.total_cache_hit_tokens = 0
        provider._long_context_surcharge_usd = 2.0
        provider._long_context_calls = 3
        summary = provider.get_session_cost()
        assert summary["input_cost_usd"] == pytest.approx(5.0, abs=1e-12)
        assert summary["long_context_surcharge_usd"] == 2.0
        assert summary["long_context_calls"] == 3
        assert summary["total_cost_usd"] == pytest.approx(5.0 + summary["output_cost_usd"] + 2.0, abs=1e-12)

    def test_a_tiered_estimate_prices_the_input_and_output_in_order(self, monkeypatch):
        import pyutilz.llm.openai_compat as oc_module

        provider = _make_provider()
        monkeypatch.setattr(provider, "_long_context_tier", lambda model: _Tier(True))
        calls: list = []

        def fake_price_call(pricing, input_tokens, output_tokens):
            calls.append((pricing, input_tokens, output_tokens))
            return (1.5, 0.25)

        monkeypatch.setattr(oc_module, "price_call", fake_price_call)
        assert provider.estimate_cost(300_000, 40) == pytest.approx(1.75)
        pricing, input_tokens, output_tokens = calls[0]
        assert (input_tokens, output_tokens) == (300_000, 40)
        assert pricing == provider._resolve_pricing("test-model")


class TestSurchargeTallyWithoutItsAttributes:
    """Adding a surcharge to a provider without the tally attributes starts both tallies from zero (openai_compat 924)."""

    def test_the_first_surcharge_on_a_provider_without_the_tally_counts_one_call(self, monkeypatch):
        import pyutilz.llm.openai_compat as oc_module

        provider = _make_provider()
        monkeypatch.setattr(provider, "_long_context_tier", lambda model: _Tier(True))
        monkeypatch.setattr(oc_module, "long_context_surcharge", lambda pricing, prompt, billed, cache_hit, cache_write: (1.0, 0.5))
        del provider._long_context_surcharge_usd
        del provider._long_context_calls
        provider._add_long_context_surcharge(300_000, 10, 0, 0)
        assert provider._long_context_calls == 1
        assert provider._long_context_surcharge_usd == pytest.approx(1.5)

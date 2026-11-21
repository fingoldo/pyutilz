"""OpenAI-compatible provider tests: requests, timeouts and buffered generation."""

from pyutilz.core import jsonlib
import logging
from unittest.mock import AsyncMock, MagicMock
import pytest
from tenacity import stop_after_attempt, wait_none
from pyutilz.llm.exceptions import LLMProviderError
from pyutilz.llm.openai_compat import OpenAICompatibleProvider
from tests.openai_compat_support import (
    _FakeResponse,
    _MockStreamResponse,
    _TestProvider,
    _make_provider,
    _mock_response,
)

httpx = pytest.importorskip("httpx")


class TestProperties:
    def test_max_output_tokens_known_model(self):
        p = _make_provider("test-model")
        assert p.max_output_tokens == 4096

    def test_max_output_tokens_fallback(self):
        p = _make_provider("unknown-model")
        assert p.max_output_tokens == 2048

    def test_build_messages_with_system(self):
        p = _make_provider()
        msgs = p._build_messages("hello", system="You are a bot")
        assert msgs == [
            {"role": "system", "content": "You are a bot"},
            {"role": "user", "content": "hello"},
        ]

    def test_build_messages_no_system(self):
        p = _make_provider()
        msgs = p._build_messages("hello")
        assert msgs == [{"role": "user", "content": "hello"}]


class TestTemperature:
    """`None` omits the field; `0.0` is a real temperature and must survive every guard on the way out.

    Both halves matter. A hardcoded temperature silently overrides whatever the caller configured, and a
    measurement in a sibling project found 0.3 doubling the reasoning a model emits for the same answer - big
    enough on a long prompt to push the first content byte past any sane timeout, which reads as a dead route.
    And `if temperature:` would drop exactly 0.0, the most deterministic setting there is, while looking right.
    """

    def test_none_omits_the_field_from_a_streamed_body(self):
        body = _make_provider()._build_stream_body("p", None, None, 100, False, None, None, None)

        assert "temperature" not in body

    def test_zero_is_sent_on_a_streamed_body(self):
        body = _make_provider()._build_stream_body("p", None, 0.0, 100, False, None, None, None)

        assert body["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_none_omits_the_field_from_a_buffered_call(self):
        p = _make_provider()
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=_mock_response())

        await p.generate("test prompt", temperature=None)

        assert "temperature" not in p._client.post.call_args.kwargs["json"]

    @pytest.mark.asyncio
    async def test_zero_is_sent_on_a_buffered_call(self):
        p = _make_provider()
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=_mock_response())

        await p.generate("test prompt", temperature=0.0)

        assert p._client.post.call_args.kwargs["json"]["temperature"] == 0.0


class TestGenerate:
    @pytest.mark.asyncio
    async def test_successful_generation(self):
        p = _make_provider()
        mock_resp = _mock_response()
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=mock_resp)

        result = await p.generate("test prompt")
        assert result == "hello"
        assert p._call_count == 1
        assert p.total_prompt_tokens == 10
        assert p.total_completion_tokens == 5
        assert p.total_cache_hit_tokens == 2
        assert p.total_reasoning_tokens == 1

    @pytest.mark.asyncio
    async def test_last_usage_set(self):
        p = _make_provider()
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=_mock_response())

        await p.generate("test")
        assert p._last_usage["input_tokens"] == 10
        assert p._last_usage["output_tokens"] == 5
        assert p._last_usage["reasoning_tokens"] == 1

    @pytest.mark.asyncio
    async def test_a_length_capped_empty_answer_under_response_format_is_re_issued_before_being_called_truncation(self):
        """Live 2026-09-02: z-ai/glm-4.7-flash spent all 15,775 output tokens on reasoning and returned no
        text - and the SAME prompt without ``response_format`` answered. Reframed 2026-09-03 (audit F38):
        an EMPTY length-capped answer under response_format is the empty-completion failure wearing a
        different finish_reason, so the one no-response_format re-issue must be taken first; a caller whose
        truncation handler only doubles max_tokens otherwise re-buys the identical broken shape. Truncation
        is still what the caller sees when that re-issue also comes back empty, and a truncation carrying
        partial text (a genuine budget cutoff) is never re-issued at all - both asserted below."""
        from pyutilz.llm.exceptions import LLMTruncationError

        empty_length_capped = _mock_response(body={"choices": [{"message": {"content": None}, "finish_reason": "length"}], "usage": {}})
        p = _make_provider()
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=empty_length_capped)

        with pytest.raises(LLMTruncationError) as caught:
            await p.generate("q", json_mode=True)
        assert caught.value.partial_text == ""
        assert p._client.post.call_count == 2, "the one response_format-free re-issue must be attempted"
        assert "response_format" not in p._client.post.call_args_list[1].kwargs["json"]

    @pytest.mark.asyncio
    async def test_a_length_capped_answer_with_partial_text_is_never_re_issued(self):
        """A genuine budget cutoff that produced text is a truncation, full stop - re-issuing it would pay
        twice for output the caller can already salvage from ``partial_text``."""
        from pyutilz.llm.exceptions import LLMTruncationError

        p = _make_provider()
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=_mock_response(body={"choices": [{"message": {"content": "half"}, "finish_reason": "length"}], "usage": {}}))

        with pytest.raises(LLMTruncationError) as caught:
            await p.generate("q", json_mode=True)
        assert caught.value.partial_text == "half"
        assert p._client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_a_rejected_parameter_is_repaired_once_and_only_once(self):
        p = _make_provider()
        rejected = _mock_response(status_code=400, body={"error": {"message": "Reasoning is mandatory for this endpoint and cannot be disabled"}})
        answered = _mock_response()
        p._client = AsyncMock()
        p._client.post = AsyncMock(side_effect=[rejected, answered])
        p._body_after_rejected_request = lambda body, status, detail: {**body, "repaired": True}

        assert await p.generate("q") == "hello"
        first, second = (c.kwargs["json"] for c in p._client.post.call_args_list)
        assert "repaired" not in first and second["repaired"] is True

    @pytest.mark.asyncio
    async def test_a_repair_that_is_itself_refused_raises_rather_than_looping(self):
        p = _make_provider()
        rejected = _mock_response(status_code=400, body={"error": {"message": "Reasoning is mandatory for this endpoint and cannot be disabled"}})
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=rejected)
        p._body_after_rejected_request = lambda body, status, detail: {**body, "repaired": True}

        with pytest.raises(LLMProviderError, match="API error 400"):
            await p.generate("q")
        assert p._client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_empty_json_mode_completion_is_reissued_without_response_format(self):
        """Live shape (OpenRouter z-ai/glm-4.7-flash, 2026-09-02): finish_reason=stop, content=None, no tool
        calls, under response_format - and a real answer once response_format is dropped."""
        p = _make_provider()
        empty = _mock_response(body={"choices": [{"message": {"content": None}, "finish_reason": "stop"}], "usage": {}})
        answered = _mock_response(body={"choices": [{"message": {"content": '{"pick": 2}'}, "finish_reason": "stop"}], "usage": {}})
        p._client = AsyncMock()
        p._client.post = AsyncMock(side_effect=[empty, answered])

        assert await p.generate("q", json_mode=True) == '{"pick": 2}'
        assert p.last_json_mode_fallback is True
        first, second = (c.kwargs["json"] for c in p._client.post.call_args_list)
        assert "response_format" in first
        assert "response_format" not in second

    @pytest.mark.asyncio
    async def test_empty_completion_without_tool_calls_raises_after_the_fallback(self):
        """Returning "" here used to surface downstream as "unparsable JSON" 58 times in one benchmark;
        the caller must see an empty completion for what it is."""
        p = _make_provider()
        empty = _mock_response(body={"choices": [{"message": {"content": None}, "finish_reason": "stop"}], "usage": {"completion_tokens_details": {"reasoning_tokens": 52}}})
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=empty)

        with pytest.raises(LLMProviderError, match=r"empty completion.*reasoning_tokens=52.*response_format=dropped"):
            await p.generate("q", json_mode=True)
        assert p._client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_empty_completion_without_json_mode_raises_without_a_retry(self):
        p = _make_provider()
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=_mock_response(body={"choices": [{"message": {"content": ""}, "finish_reason": "stop"}], "usage": {}}))

        with pytest.raises(LLMProviderError, match="response_format=not sent"):
            await p.generate("q")
        assert p._client.post.call_count == 1
        assert p.last_json_mode_fallback is False

    @pytest.mark.asyncio
    async def test_tool_call_only_response_still_returns_empty_text(self):
        p = _make_provider()
        body = {"choices": [{"message": {"content": None, "tool_calls": [{"id": "t1"}]}, "finish_reason": "tool_calls"}], "usage": {}}
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=_mock_response(body=body))

        assert await p.generate("q", json_mode=True) == ""
        assert p.last_tool_calls == [{"id": "t1"}]
        assert p._client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_empty_choices_raises(self):
        p = _make_provider()
        resp = _mock_response(body={"choices": [], "usage": {}})
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=resp)

        with pytest.raises(LLMProviderError, match="returned no choices"):
            await p.generate("test")

    @pytest.mark.asyncio
    async def test_400_error_raises(self):
        p = _make_provider()
        resp = _mock_response(
            status_code=400,
            body={"error": {"message": "bad request"}},
        )
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=resp)

        with pytest.raises(LLMProviderError, match="API error 400"):
            await p.generate("test")

    @pytest.mark.asyncio
    async def test_json_mode_sets_response_format(self):
        p = _make_provider()
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=_mock_response())

        await p.generate("test", json_mode=True)
        call_args = p._client.post.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        assert body["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_thinking_none_omits_field(self):
        # Default (thinking=None) must not inject any thinking field —
        # provider/model defaults apply.
        p = _make_provider()
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=_mock_response())

        await p.generate("test")
        body = p._client.post.call_args.kwargs.get("json") or p._client.post.call_args[1].get("json")
        assert "thinking" not in body

    @pytest.mark.asyncio
    async def test_thinking_ignored_for_unsupported_provider(self):
        # Base provider has no thinking support; flag must be silently ignored.
        p = _make_provider()
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=_mock_response())

        await p.generate("test", thinking=False)
        body = p._client.post.call_args.kwargs.get("json") or p._client.post.call_args[1].get("json")
        assert "thinking" not in body

    @pytest.mark.asyncio
    async def test_thinking_disabled_passed_to_supporting_provider(self):
        # Subclass that emits thinking field for any model
        class _ThinkingProvider(_TestProvider):
            def _thinking_request_field(self, enabled):
                return {"thinking": {"type": "enabled" if enabled else "disabled"}}

        p = _ThinkingProvider(api_key="k", model="test-model")
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=_mock_response())

        await p.generate("test", thinking=False)
        body = p._client.post.call_args.kwargs.get("json") or p._client.post.call_args[1].get("json")
        assert body["thinking"] == {"type": "disabled"}

        await p.generate("test", thinking=True)
        body = p._client.post.call_args.kwargs.get("json") or p._client.post.call_args[1].get("json")
        assert body["thinking"] == {"type": "enabled"}

    @pytest.mark.asyncio
    async def test_cumulative_token_tracking(self):
        p = _make_provider()
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=_mock_response())

        await p.generate("test1")
        await p.generate("test2")
        assert p._call_count == 2
        assert p.total_prompt_tokens == 20
        assert p.total_completion_tokens == 10


class TestCosts:
    def test_estimate_cost(self):
        p = _make_provider()
        cost = p.estimate_cost(1_000_000, 1_000_000)
        assert cost == pytest.approx(3.0)

    def test_session_cost_with_cache(self):
        p = _make_provider()
        p.total_prompt_tokens = 1_000_000
        p.total_cache_hit_tokens = 500_000
        p.total_completion_tokens = 1_000_000
        p.total_reasoning_tokens = 0
        p._call_count = 5

        cost = p.get_session_cost()
        assert cost["input_cost_usd"] == pytest.approx(0.55)
        assert cost["output_cost_usd"] == pytest.approx(2.0)
        assert cost["total_cost_usd"] == pytest.approx(2.55)
        assert cost["calls"] == 5

    @pytest.mark.asyncio
    async def test_count_tokens_uses_tiktoken(self):
        p = _make_provider()
        result = await p.count_tokens("a" * 100)
        from pyutilz.llm.token_counter import _HAS_TIKTOKEN

        # 13 = the real BPE count for "a"*100; 25 = the len//4 heuristic used without tiktoken.
        assert result == (13 if _HAS_TIKTOKEN else 25)


class TestGenerateJson:
    @pytest.mark.asyncio
    async def test_generate_json_success(self):
        p = _make_provider()
        json_resp = _mock_response(
            body={
                "choices": [{"message": {"content": '{"key": "value"}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        )
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=json_resp)

        result = await p.generate_json("give me json")
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_generate_json_appends_system(self):
        p = _make_provider()
        p._client = AsyncMock()
        json_resp = _mock_response(
            body={
                "choices": [{"message": {"content": '{"k": "v"}'}, "finish_reason": "stop"}],
                "usage": {},
            }
        )
        p._client.post = AsyncMock(return_value=json_resp)

        await p.generate_json("test", system="Be helpful")
        call_args = p._client.post.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        system_msg = body["messages"][0]["content"]
        assert "Respond with valid JSON only" in system_msg
        assert "Be helpful" in system_msg


class TestHooks:
    def test_default_compute_billed_output(self):
        p = _make_provider()
        assert p._compute_billed_output(100, 50) == 100

    def test_default_timeout(self):
        p = _make_provider()
        assert p._get_timeout("any-model") == 120.0

    def test_handle_special_status_noop(self, caplog):
        """The base hook is a documented no-op: it must neither raise nor emit anything,
        so a subclass override that starts logging on every status is visible here."""
        p = _make_provider()
        resp = _mock_response(status_code=200)
        with caplog.at_level(logging.DEBUG, logger="pyutilz.llm.openai_compat"):
            assert p._handle_special_status(resp) is None
        assert caplog.records == []


class TestPricingFallbackSurvivors:
    """The default pricing record keeps input and output in their own fields, and a missing cached-input or cache-write
    rate falls back to the uncached input rate, while a published one is used (openai_compat.py 263-277)."""

    def test_the_default_record_keeps_input_and_output_apart(self):
        from pyutilz.llm._pricing import Pricing

        record = _make_provider()._resolve_pricing("m")
        assert record == Pricing(input=1.0, output=2.0, cache_hit=None)

    def test_a_published_cache_hit_rate_is_used(self):
        from pyutilz.llm._pricing import Pricing

        p = _make_provider()
        p._resolve_pricing = lambda model: Pricing(input=1.0, output=2.0, cache_hit=0.25)
        assert OpenAICompatibleProvider._cache_hit_cost_per_1m(p, "m") == 0.25

    def test_a_missing_cache_hit_rate_falls_back_to_the_input_rate(self):
        from pyutilz.llm._pricing import Pricing

        p = _make_provider()
        p._resolve_pricing = lambda model: Pricing(input=1.0, output=2.0, cache_hit=None)
        assert OpenAICompatibleProvider._cache_hit_cost_per_1m(p, "m") == 1.0

    def test_a_published_cache_write_rate_is_used_and_a_missing_one_falls_back(self):
        from pyutilz.llm._pricing import Pricing

        p = _make_provider()
        p._resolve_pricing = lambda model: Pricing(input=1.0, output=2.0, cache_write=3.0)
        assert p._cache_write_cost_per_1m("m") == 3.0
        p._resolve_pricing = lambda model: Pricing(input=1.0, output=2.0, cache_write=None)
        assert p._cache_write_cost_per_1m("m") == 1.0


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
        assert "rejected a streaming request parameter" in caplog.text
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


class TestUsageAccountingSurvivors:
    """Each usage block adds its prompt, completion, cache-hit, reasoning and cache-write tokens to the totals, read from
    the DeepSeek key or the nested OpenAI/xAI key, and a null field counts as zero (openai_compat.py 587-599)."""

    def test_plain_counts_add_to_the_totals(self):
        p = _make_provider()
        p._record_usage({"prompt_tokens": 10, "completion_tokens": 4})
        assert (p.total_prompt_tokens, p.total_completion_tokens, p.total_cache_hit_tokens, p.total_reasoning_tokens) == (10, 4, 0, 0)

    def test_the_per_call_usage_dict_sums_its_blocks_under_each_key(self):
        p = _make_provider()
        p._record_usage({"prompt_tokens": 10, "completion_tokens": 4, "completion_tokens_details": {"reasoning_tokens": 3}, "prompt_cache_hit_tokens": 6})
        p._record_usage({"prompt_tokens": 5, "completion_tokens": 2})
        assert p._last_usage == {"input_tokens": 15, "output_tokens": 6, "reasoning_tokens": 3, "cached_tokens": 6}

    def test_the_usage_log_line_states_prompt_cache_completion_and_reasoning_in_order(self, caplog):
        import logging

        p = _make_provider()
        with caplog.at_level(logging.INFO):
            p._record_usage({"prompt_tokens": 10, "completion_tokens": 4, "prompt_cache_hit_tokens": 6, "completion_tokens_details": {"reasoning_tokens": 3}})
        assert "10 prompt (6 cached) + 4 completion (3 reasoning) | cumulative: 10 in, 4 out" in caplog.text

    def test_totals_accumulate_across_calls(self):
        p = _make_provider()
        p._record_usage({"prompt_tokens": 10, "completion_tokens": 4, "prompt_tokens_details": {"cache_write_tokens": 3}})
        p._record_usage({"prompt_tokens": 5, "completion_tokens": 2, "prompt_tokens_details": {"cache_write_tokens": 1}})
        assert (p.total_prompt_tokens, p.total_completion_tokens, p.total_cache_write_tokens) == (15, 6, 4)

    def test_the_deepseek_cache_key_is_read(self):
        p = _make_provider()
        p._record_usage({"prompt_tokens": 10, "prompt_cache_hit_tokens": 6})
        assert p.total_cache_hit_tokens == 6

    def test_the_nested_openai_cache_key_is_read(self):
        p = _make_provider()
        p._record_usage({"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 5}})
        assert p.total_cache_hit_tokens == 5

    def test_reasoning_tokens_are_read_from_the_completion_details(self):
        p = _make_provider()
        p._record_usage({"completion_tokens": 9, "completion_tokens_details": {"reasoning_tokens": 7}})
        assert p.total_reasoning_tokens == 7

    def test_cache_writes_are_read_from_either_nesting(self):
        p = _make_provider()
        p._record_usage({"prompt_tokens_details": {"cache_write_tokens": 3}})
        p._record_usage({"input_tokens_details": {"cache_write_tokens": 2}})
        assert p.total_cache_write_tokens == 5

    def test_null_fields_count_as_zero(self):
        p = _make_provider()
        p._record_usage({"prompt_tokens": None, "completion_tokens": None, "prompt_cache_hit_tokens": None, "completion_tokens_details": {"reasoning_tokens": None}})
        assert (p.total_prompt_tokens, p.total_completion_tokens, p.total_cache_hit_tokens, p.total_reasoning_tokens) == (0, 0, 0, 0)


class TestGenerateDefaultsAndOrderSurvivors:
    """A buffered generate call: its defaults reach the body as documented, the options reach the body builder in order,
    the per-call state is reset and the async preparation awaited (openai_compat.py 656, 694-714)."""

    @pytest.mark.asyncio
    async def test_a_plain_generate_call_sends_the_documented_defaults(self):
        p = _make_provider()
        p._post_and_unwrap = AsyncMock(return_value="ok")
        assert await p.generate("q") == "ok"
        body = p._post_and_unwrap.call_args.args[0]
        assert body["temperature"] == 0.7
        assert "max_tokens" not in body
        assert "response_format" not in body

    @pytest.mark.asyncio
    async def test_the_options_reach_the_body_builder_in_order(self):
        p = _make_provider()
        spy = MagicMock(return_value={"model": "m", "messages": []})
        p._request_body = spy
        p._post_and_unwrap = AsyncMock(return_value="ok")
        await p.generate("q", "sys", 0.2, 50, False, None, None, ["img"], [{"t": 1}], "auto", {"x": 1})
        args = spy.call_args.args
        assert args[0] == "q" and args[1] == "sys" and args[2] == 0.2
        assert args[6] == ["img"]
        assert args[7] == [{"t": 1}]
        assert args[8] == "auto"
        assert args[9] == {"x": 1}

    @pytest.mark.asyncio
    async def test_a_generate_call_awaits_the_preparation_and_starts_clean(self):
        p = _make_provider()
        p._async_prepare = AsyncMock()
        p._post_and_unwrap = AsyncMock(return_value="ok")
        p.last_json_mode_fallback = True
        await p.generate("q")
        p._async_prepare.assert_awaited_once()
        assert p.last_json_mode_fallback is False


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


class TestBufferedFinishReasonSurvivors:
    """A buffered completion ending with finish_reason 'error' raises with its partial text; a tool-call-only completion
    returns empty text with the tool calls kept; a missing finish reason reads as 'unknown' (openai_compat.py 837-851)."""

    @pytest.mark.asyncio
    async def test_an_error_finish_raises_with_the_partial_text(self):
        from pyutilz.llm.exceptions import LLMStreamInterruptedError

        p = _make_provider()
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=_mock_response(body={"choices": [{"message": {"content": "partial"}, "finish_reason": "error"}]}))
        # Retryable, so the decorator would back off for real: one attempt, no wait.
        original_stop, original_wait = p.generate.retry.stop, p.generate.retry.wait
        p.generate.retry.stop = stop_after_attempt(1)
        p.generate.retry.wait = wait_none()
        try:
            with pytest.raises(LLMStreamInterruptedError, match=r"TestProvider response ended with finish_reason='error'") as exc:
                await p.generate("q")
        finally:
            p.generate.retry.stop = original_stop
            p.generate.retry.wait = original_wait
        assert exc.value.partial_text == "partial"

    @pytest.mark.asyncio
    async def test_a_tool_call_only_completion_returns_empty_text_and_keeps_the_calls(self):
        p = _make_provider()
        p._client = AsyncMock()
        call = {"id": "t1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
        p._client.post = AsyncMock(return_value=_mock_response(body={"choices": [{"message": {"content": None, "tool_calls": [call]}, "finish_reason": "tool_calls"}]}))
        assert await p.generate("q") == ""
        assert p.last_tool_calls == [call]

    @pytest.mark.asyncio
    async def test_a_completion_without_a_finish_reason_reads_as_unknown(self):
        p = _make_provider()
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=_mock_response(body={"choices": [{"message": {"content": "ok"}}]}))
        assert await p.generate("q") == "ok"
        assert p._last_finish_reason == "unknown"


class TestLongContextSurchargeSurvivors:
    """A call whose prompt reaches the long-context tier adds that tier's surcharge to a running total and counts it;
    a short prompt adds nothing (openai_compat.py 907-925)."""

    def test_a_long_prompt_adds_the_surcharge_and_counts_the_call(self, monkeypatch):
        import pyutilz.llm.openai_compat as openai_compat_mod

        class _Tier:
            def applies(self, n):
                return n >= 10

        class _LongTier(_TestProvider):
            def _long_context_tier(self, model):
                return _Tier()

        seen = []

        def _surcharge(pricing, prompt, billed, cache_read, cache_write):
            seen.append((prompt, billed, cache_read, cache_write))
            return (1.5, 2.5)

        monkeypatch.setattr(openai_compat_mod, "long_context_surcharge", _surcharge)
        p = _LongTier(api_key="test-key", model="test-model")  # pragma: allowlist secret
        p._add_long_context_surcharge(20, 5, 3, 2)
        p._add_long_context_surcharge(20, 5, 3, 2)
        assert seen == [(20, 5, 3, 2), (20, 5, 3, 2)]
        assert p._long_context_surcharge_usd == pytest.approx(8.0)
        assert p._long_context_calls == 2

    def test_a_short_prompt_adds_no_surcharge(self):
        p = _make_provider()
        p._add_long_context_surcharge(20, 5, 3, 2)
        assert p._long_context_calls == 0


class TestSessionCostArithmeticSurvivors:
    """The session cost splits prompt tokens into cache-hit, cache-write and miss, prices each at its own rate, prices
    the billed output, and adds any long-context surcharge (openai_compat.py 936-968). Test rates: input 1.0,
    cache-hit 0.1 (from the test provider), output 2.0; cache-write falls back to input."""

    def test_the_session_cost_prices_each_token_class_at_its_own_rate(self):
        p = _make_provider()
        p.total_prompt_tokens = 1_000_000
        p.total_cache_hit_tokens = 400_000
        p.total_cache_write_tokens = 100_000
        p.total_completion_tokens = 500_000
        p.total_reasoning_tokens = 0
        cost = p.get_session_cost()
        assert cost["cache_miss_tokens"] == 500_000
        assert cost["prompt_tokens"] == 1_000_000
        assert cost["cache_hit_tokens"] == 400_000
        assert cost["completion_tokens"] == 500_000
        assert cost["reasoning_tokens"] == 0
        assert cost["calls"] == 0
        assert cost["input_cost_usd"] == pytest.approx(0.5 * 1.0 + 0.4 * 0.1 + 0.1 * 1.0, rel=1e-9)
        assert cost["output_cost_usd"] == pytest.approx(0.5 * 2.0, rel=1e-9)
        assert cost["total_cost_usd"] == pytest.approx(0.64 + 1.0, rel=1e-9)

    def test_the_surcharge_is_added_to_the_total_and_reported(self):
        p = _make_provider()
        p.total_prompt_tokens = 1_000
        p._long_context_surcharge_usd = 0.25
        p._long_context_calls = 3
        cost = p.get_session_cost()
        assert cost["long_context_surcharge_usd"] == 0.25
        assert cost["long_context_calls"] == 3
        assert cost["total_cost_usd"] == pytest.approx(cost["input_cost_usd"] + cost["output_cost_usd"] + 0.25, rel=1e-9)


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
    async def test_an_already_closed_client_is_not_closed_again(self):
        p = _make_provider()
        p._client = AsyncMock()
        p._client.is_closed = True
        await p._close()
        p._client.aclose.assert_not_awaited()


class TestClientConfigurationSurvivors:
    """The shared client carries the bearer token and JSON content type, a 10 s connect timeout, and the per-call
    flags are reset at the start of each call (openai_compat.py 94-122, 103, 115)."""

    def test_the_client_sends_the_bearer_token_and_a_json_content_type(self):
        p = _make_provider()
        assert p._client.headers["Authorization"] == "Bearer test-key"
        assert p._client.headers["Content-Type"] == "application/json"

    def test_the_connect_timeout_is_ten_seconds(self):
        assert _make_provider()._client.timeout.connect == 10.0

    def test_a_reset_clears_the_schema_applied_flag_and_the_json_mode_fallback(self):
        p = _make_provider()
        p._last_json_schema_applied = True
        p.last_json_mode_fallback = True
        p._reset_per_call_state()
        assert p._last_json_schema_applied is False
        assert p.last_json_mode_fallback is False


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


class TestRateLimitSnapshotSurvivors:
    """The captured snapshot keeps only the x-ratelimit-/ratelimit- headers, lower-cased, replaces the previous one
    on every response, and check_account_limits reports it with short keys (openai_compat.py 203-242)."""

    def test_only_the_ratelimit_headers_are_kept_and_lower_cased(self):
        p = _make_provider()
        p._capture_rate_limit_headers({"X-RateLimit-Remaining-Requests": "7", "ratelimit-reset": "9", "x-request-id": "abc", "Content-Type": "json"})
        assert p.last_rate_limits == {"x-ratelimit-remaining-requests": "7", "ratelimit-reset": "9"}

    def test_a_response_without_rate_limit_headers_clears_the_previous_snapshot(self):
        p = _make_provider()
        p._capture_rate_limit_headers({"x-ratelimit-remaining-requests": "7"})
        p._capture_rate_limit_headers({"x-request-id": "abc"})
        assert p.last_rate_limits == {}

    def test_account_limits_report_the_raw_headers_and_short_keys(self):
        import asyncio

        p = _make_provider()
        p._capture_rate_limit_headers({"x-ratelimit-remaining-requests": "0", "ratelimit-reset-tokens": "30"})
        out = asyncio.run(p.check_account_limits())
        assert out == {
            "raw": {"x-ratelimit-remaining-requests": "0", "ratelimit-reset-tokens": "30"},
            "remaining_requests": "0",
            "reset_tokens": "30",
        }

    def test_account_limits_without_a_snapshot_name_the_provider_and_the_fix(self):
        import asyncio

        p = _make_provider()
        with pytest.raises(NotImplementedError) as exc:
            asyncio.run(p.check_account_limits())
        assert str(exc.value) == (
            "TestProvider: no rate-limit snapshot captured yet -- issue at least one generate() call first; the headers are "
            "captured automatically. Standalone introspection endpoints are not exposed for regular keys on this provider."
        )


class TestRateLimitHeaderCapture:
    @pytest.mark.asyncio
    async def test_generate_captures_headers_on_429_before_raising(self):
        # Regression: headers must be captured even when raise_for_status()
        # subsequently raises -- otherwise a 429's rate-limit headers (the
        # ones that matter most for backoff/quota decisions) are lost.
        p = _make_provider()
        resp = _mock_response(
            status_code=429,
            body={"error": {"message": "rate limited"}},
        )
        resp.headers["x-ratelimit-remaining-requests"] = "0"
        resp.headers["x-ratelimit-reset-requests"] = "30"
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=resp)

        # Force a single attempt (no real backoff sleep) so the test is fast;
        # 429 is retryable so the tenacity-decorated generate() would
        # otherwise retry for real.
        original_stop, original_wait = p.generate.retry.stop, p.generate.retry.wait
        p.generate.retry.stop = stop_after_attempt(1)
        p.generate.retry.wait = wait_none()
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await p.generate("test")
        finally:
            p.generate.retry.stop = original_stop
            p.generate.retry.wait = original_wait

        assert p.last_rate_limits.get("x-ratelimit-remaining-requests") == "0"
        assert p.last_rate_limits.get("x-ratelimit-reset-requests") == "30"

    @pytest.mark.asyncio
    async def test_generate_stream_captures_headers_on_429_before_raising(self, monkeypatch):
        import pyutilz.llm.openai_compat as openai_compat_mod

        monkeypatch.setattr(openai_compat_mod, "MAX_RETRY_ATTEMPTS", 1)

        p = _make_provider()
        mock_resp = _MockStreamResponse(
            status_code=429,
            headers={"x-ratelimit-remaining-requests": "0"},
        )
        p._client = AsyncMock()
        p._client.stream = MagicMock(return_value=mock_resp)

        with pytest.raises(httpx.HTTPStatusError):
            async for _ in p.generate_stream("test"):
                pass

        assert p.last_rate_limits.get("x-ratelimit-remaining-requests") == "0"


class TestCumulativeUsageLogging:
    @pytest.mark.asyncio
    async def test_generate_and_generate_stream_log_same_message_format(self, caplog):
        import logging as _logging

        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "prompt_cache_hit_tokens": 2,
            "completion_tokens_details": {"reasoning_tokens": 1},
        }

        # -- generate() --
        p1 = _make_provider()
        p1._client = AsyncMock()
        p1._client.post = AsyncMock(return_value=_mock_response(body={
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": usage,
        }))
        with caplog.at_level(_logging.INFO, logger="pyutilz.llm.openai_compat"):
            await p1.generate("test")
        generate_messages = [r.getMessage() for r in caplog.records]
        caplog.clear()

        # -- generate_stream() --
        p2 = _make_provider()
        lines = [
            f"data: {jsonlib.dumps({'choices': [], 'usage': usage})}",
            "data: [DONE]",
        ]
        p2._client = AsyncMock()
        p2._client.stream = MagicMock(return_value=_MockStreamResponse(status_code=200, lines=lines))
        with caplog.at_level(_logging.INFO, logger="pyutilz.llm.openai_compat"):
            async for _ in p2.generate_stream("test"):
                pass
        stream_messages = [r.getMessage() for r in caplog.records]

        assert generate_messages, "generate() should emit a cumulative-usage log message"
        assert stream_messages, "generate_stream() should emit a cumulative-usage log message"
        assert generate_messages == stream_messages


class TestStrictJsonSchema:
    """A strict json_schema CONSTRAINS generation (closed enums cannot be violated); plain json_mode
    only promises syntactically valid JSON. Callers must be able to tell which one actually applied."""

    SCHEMA = {
        "name": "t",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"relation": {"type": "string", "enum": ["CAUSES", "INDICATES"]}},
            "required": ["relation"],
        },
    }

    class _NoSchemaProvider(_TestProvider):
        def supports_json_schema(self) -> bool:
            return False

    @pytest.mark.asyncio
    async def test_schema_is_sent_when_supported(self):
        p = _make_provider()
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=_mock_response())
        await p.generate("x", json_mode=True, json_schema=self.SCHEMA)
        body = p._client.post.call_args.kwargs.get("json") or p._client.post.call_args[1].get("json")
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["strict"] is True
        assert p.last_json_schema_applied is True

    @pytest.mark.asyncio
    async def test_degrades_to_json_object_when_unsupported(self):
        p = self._NoSchemaProvider(api_key="k", model="test-model")
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=_mock_response())
        await p.generate("x", json_mode=True, json_schema=self.SCHEMA)
        body = p._client.post.call_args.kwargs.get("json") or p._client.post.call_args[1].get("json")
        # The call still works, but the guarantee is explicitly NOT claimed.
        assert body["response_format"] == {"type": "json_object"}
        assert p.last_json_schema_applied is False

    @pytest.mark.asyncio
    async def test_no_response_format_without_json_mode_or_schema(self):
        p = _make_provider()
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=_mock_response())
        await p.generate("x")
        body = p._client.post.call_args.kwargs.get("json") or p._client.post.call_args[1].get("json")
        assert "response_format" not in body

    @pytest.mark.asyncio
    async def test_generate_json_forwards_schema(self):
        p = _make_provider()
        p._client = AsyncMock()
        p._client.post = AsyncMock(
            return_value=_mock_response(
                body={
                    "choices": [{"message": {"content": '{"relation": "CAUSES"}'}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            )
        )
        out = await p.generate_json("x", json_schema=self.SCHEMA)
        body = p._client.post.call_args.kwargs.get("json") or p._client.post.call_args[1].get("json")
        assert body["response_format"]["type"] == "json_schema"
        assert out == {"relation": "CAUSES"}


class TestContextClamp:
    """An explicit output budget must fit the context window, or the upstream rejects the whole request with
    HTTP 400 before generating anything (llama-3.3-70b: 128k output cap, 131k window). An unset budget is not
    sent at all (see test_llm_unset_max_tokens_is_omitted.py): the route chooses the limit."""

    class _WideOutputProvider(_TestProvider):
        _max_tokens_map = {"wide-model": 128_000}
        _context_window_map = {"wide-model": 131_072}

    @pytest.mark.asyncio
    async def test_unset_budget_is_omitted_not_clamped(self):
        p = self._WideOutputProvider(api_key="k", model="wide-model")
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=_mock_response())
        await p.generate("x " * 8000)
        body = p._client.post.call_args.kwargs.get("json") or p._client.post.call_args[1].get("json")
        assert "max_tokens" not in body

    @pytest.mark.asyncio
    async def test_explicit_impossible_budget_is_also_clamped(self):
        p = self._WideOutputProvider(api_key="k", model="wide-model")
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=_mock_response())
        await p.generate("x " * 8000, max_tokens=128_000)
        body = p._client.post.call_args.kwargs.get("json") or p._client.post.call_args[1].get("json")
        assert body["max_tokens"] < 128_000

    @pytest.mark.asyncio
    async def test_fitting_budget_is_forwarded_verbatim(self):
        p = self._WideOutputProvider(api_key="k", model="wide-model")
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=_mock_response())
        await p.generate("short", max_tokens=1000)
        body = p._client.post.call_args.kwargs.get("json") or p._client.post.call_args[1].get("json")
        assert body["max_tokens"] == 1000


def test_an_empty_body_is_raised_as_a_retryable_unparsable_response():
    """OpenRouter intermittently returns an empty body on a 200. `resp.json()` reports that as
    `jsonlib.JSONDecodeError` - a `ValueError` - which matches neither the HTTP-status nor the transport branch
    of `_is_retryable_http_error`, so the call used to fail outright instead of being retried. MEASURED
    2026-08-06 in a downstream repo: across three runs of one live-LLM tier a DIFFERENT test failed each
    time and one run was green, which reads as flaky tests rather than one retryable fault.
    """
    from pyutilz.llm.exceptions import LLMUnparseableResponseError
    from pyutilz.llm.openai_compat import _is_retryable_http_error, parse_response_envelope

    with pytest.raises(LLMUnparseableResponseError) as caught:
        parse_response_envelope(_FakeResponse(""), "OpenRouter")
    assert caught.value.status_code == 200
    # The point of the change: the raised type must be one the retry predicate accepts.
    assert _is_retryable_http_error(caught.value) is True


def test_a_non_json_body_keeps_an_excerpt_so_the_log_can_tell_a_gateway_page_from_a_truncation():
    from pyutilz.llm.exceptions import LLMUnparseableResponseError
    from pyutilz.llm.openai_compat import _is_retryable_http_error, parse_response_envelope

    page = "<html><head><title>504 Gateway Time-out</title></head><body>nginx</body></html>"
    with pytest.raises(LLMUnparseableResponseError) as caught:
        parse_response_envelope(_FakeResponse(page, status_code=200, raises=True), "OpenRouter")
    assert "504 Gateway Time-out" in caught.value.body_excerpt
    assert _is_retryable_http_error(caught.value) is True


def test_valid_json_that_is_not_an_object_is_refused_rather_than_crashing_downstream():
    """A bare list or string parses fine and then explodes on the first `.get` far from here."""
    from pyutilz.llm.exceptions import LLMUnparseableResponseError
    from pyutilz.llm.openai_compat import parse_response_envelope

    with pytest.raises(LLMUnparseableResponseError, match="not the expected object"):
        parse_response_envelope(_FakeResponse("[]", payload=[]), "OpenRouter")


def test_a_well_formed_envelope_is_returned_unchanged():
    """The guard must not touch the happy path."""
    from pyutilz.llm.openai_compat import parse_response_envelope

    envelope = {"choices": [{"message": {"content": "hi"}}], "usage": {"total_tokens": 3}}
    assert parse_response_envelope(_FakeResponse('{"choices": []}', payload=envelope), "OpenRouter") == envelope


def test_a_malformed_model_answer_inside_a_good_envelope_is_still_not_retryable():
    """The distinction the new exception exists to preserve: JSONParsingError is the MODEL's output being
    malformed inside a well-formed envelope - re-issuing usually returns the same thing, so it must NOT
    become retryable as collateral of this change."""
    from pyutilz.llm.exceptions import JSONParsingError
    from pyutilz.llm.openai_compat import _is_retryable_http_error

    assert _is_retryable_http_error(JSONParsingError("model emitted broken JSON")) is False


def test_the_request_timeout_scales_with_the_output_the_body_asks_for():
    """`_get_timeout` classifies by model NAME and so cannot see how much output a call requested.

    Measured 2026-09-03 on an autopsia arena run: `z-ai/glm-5.3-flash` asked for 54,853 output tokens,
    matched none of the "slow tier" name substrings, and got the 240 s default. All nine of its captures
    died in a ReadTimeout storm - attempts 5, 6 and 7 with 34 s, 67 s and 130 s backoffs - while the model
    was still generating; the one call that did finish emitted 35,185 tokens, which no 240 s budget covers.
    """
    from pyutilz.llm.openai_compat import OpenAICompatibleProvider

    class _Probe(OpenAICompatibleProvider):
        _base_url = "https://example.invalid"
        _provider_name = "probe"
        _input_cost_per_1m = 0.0
        _output_cost_per_1m = 0.0

        def _get_timeout(self, model: str) -> float:
            return 240.0

    probe = _Probe.__new__(_Probe)
    probe.model_name = "z-ai/glm-5.3-flash"

    # 54,853 tokens at the pessimistic 30 tok/s floor is ~1,828 s, which must win over the 240 s default -
    # and is then held to `_max_derived_timeout_s`, see the cap's own assertions below.
    assert probe._timeout_for({"max_tokens": 54_853}) == pytest.approx(54_853 / 30.0)
    # The name heuristic stays a FLOOR: a small request on a slow-tier model keeps its long allowance,
    # because the reason that model is slow has nothing to do with how much it was asked to write.
    assert probe._timeout_for({"max_tokens": 500}) == 240.0
    # `max_completion_tokens` is the same request under the newer field name.
    assert probe._timeout_for({"max_completion_tokens": 54_853}) == pytest.approx(54_853 / 30.0)
    # A ceiling is NOT a target, so the derived half is capped - but the cap has to clear the largest
    # answer that actually ARRIVES. Measured 2026-09-08: a 128,000-token request derives 71 minutes, while
    # the largest clean emission across a 19-model fleet is 85,694 tokens and needs 48. A first attempt at
    # 20 minutes killed every capture from the one model that emits 70,783.
    assert probe._timeout_for({"max_tokens": 128_000}) == 3000.0
    assert probe._timeout_for({"max_tokens": 85_694}) == pytest.approx(85_694 / 30.0), "the largest measured answer is not clamped"
    assert probe._timeout_for({"max_tokens": 30_000}) == pytest.approx(1000.0), "below the cap, nothing is clamped"
    # Only the DERIVED half is clamped: a slow-tier model keeps whatever the name heuristic grants it.
    class _SlowTier(type(probe)):  # type: ignore[misc]
        def _get_timeout(self, model: str) -> float:
            return 1800.0

    slow = _SlowTier.__new__(_SlowTier)
    slow.model_name = "openai/o3-pro"
    assert slow._timeout_for({"max_tokens": 128_000}) == 3000.0, "the derived half still wins when it is larger than the name heuristic"

    # A body that names no budget at all falls back to the heuristic rather than to zero.
    assert probe._timeout_for({}) == 240.0
    assert probe._timeout_for({"max_tokens": None}) == 240.0
    assert probe._timeout_for({"max_tokens": "not a number"}) == 240.0


def test_the_zero_max_tokens_sentinel_sizes_the_timeout_for_the_models_real_ceiling():
    """`max_tokens=0` means "the model's real ceiling" (generate() resolves it to max_output_tokens), not "no budget".

    Read with `or`, the 0 was treated as absent, the derived half evaluated to 0 and the timeout collapsed to the 240 s
    name default: the recommended way to ask for the full budget was also the way that disabled the sizing meant for it.
    """
    from pyutilz.llm.openai_compat import OpenAICompatibleProvider

    class _Probe(OpenAICompatibleProvider):
        _base_url = "https://example.invalid"
        _provider_name = "probe"
        _input_cost_per_1m = 0.0
        _output_cost_per_1m = 0.0
        _max_tokens_map = {"z-ai/glm-5.3-flash": 60_000}

        def _get_timeout(self, model: str) -> float:
            return 240.0

    probe = _Probe.__new__(_Probe)
    probe.model_name = "z-ai/glm-5.3-flash"

    assert probe._timeout_for({"max_tokens": 0}) == pytest.approx(60_000 / 30.0)
    assert probe._timeout_for({"max_tokens": 0}) == probe._timeout_for({"max_tokens": probe.max_output_tokens}), "the sentinel and its resolution differ"
    assert probe._timeout_for({"max_completion_tokens": 0}) == pytest.approx(60_000 / 30.0)
    # A stated 0 is not an absent field: it must not fall through to a max_completion_tokens left over in the body.
    assert probe._timeout_for({"max_tokens": 0, "max_completion_tokens": 900}) == pytest.approx(60_000 / 30.0)


def test_the_post_passes_that_timeout_rather_than_the_clients_construction_time_one():
    """A timeout computed and then not sent is the same bug with a passing unit test behind it."""
    import asyncio

    from pyutilz.llm.openai_compat import OpenAICompatibleProvider

    class _Probe(OpenAICompatibleProvider):
        _base_url = "https://example.invalid"
        _provider_name = "probe"
        _input_cost_per_1m = 0.0
        _output_cost_per_1m = 0.0

    probe = _Probe.__new__(_Probe)
    probe.model_name = "some/model"
    seen: dict[str, object] = {}

    class _Client:
        async def post(self, path, json, timeout):  # `json` mirrors httpx's own parameter name
            seen["timeout"] = timeout
            raise RuntimeError("stop here - the timeout is what this test is about")

    probe._client = _Client()

    with pytest.raises(RuntimeError):
        asyncio.run(probe._post_and_unwrap({"max_tokens": 30_000}))
    assert seen["timeout"] == pytest.approx(1000.0)


@pytest.mark.asyncio
async def test_a_streamed_call_gets_the_same_derived_timeout_as_a_buffered_one():
    """The stream used to inherit the client's constructor timeout, and died mid-thought because of it.

    `_timeout_for` was written for exactly this model after a ReadTimeout storm, and only the buffered
    path passed it. A name heuristic cannot see TIME TO FIRST TOKEN any more than it can see output
    size: z-ai/glm-5.3-flash thinks for 393 s before emitting content (measured 2026-09-13), so a
    streamed enrichment call on the 240 s default was cut off while the model was working normally.
    """
    from pyutilz.llm.openai_compat import OpenAICompatibleProvider

    class _Probe(OpenAICompatibleProvider):
        _base_url = "https://example.invalid"
        _provider_name = "probe"
        _input_cost_per_1m = 0.0
        _output_cost_per_1m = 0.0

        def _get_timeout(self, model: str) -> float:
            return 240.0

    probe = _Probe.__new__(_Probe)
    probe.model_name = "z-ai/glm-5.3-flash"
    probe._max_concurrent = 1

    seen: dict = {}

    class _Client:
        def stream(self, method, url, json=None, timeout=None):
            seen["timeout"] = timeout
            raise RuntimeError("stop here: the timeout is all this test needs")

    probe._client = _Client()
    body = {"max_tokens": 54_853}

    with pytest.raises(RuntimeError):
        async for _ in probe.generate_stream("p", max_tokens=54_853):
            pass

    assert seen["timeout"] == probe._timeout_for(body), "the stream must not fall back to the client default"
    assert seen["timeout"] > 240.0


class TestAnErrorInsideA200EnvelopeIsTheStatusItNames:
    """OpenRouter delivers some upstream failures as HTTP 200 with `{"error": {...}}` and no choices. Measured 2026-09-15:
    a 429 "temporarily rate-limited upstream" for openai/gpt-5.6-luna arrived that way and was reported as "returned no
    choices" - not retried, and indistinguishable from a dead route."""

    @staticmethod
    def _fast_retries(p, attempts):
        original = (p.generate.retry.stop, p.generate.retry.wait)
        p.generate.retry.stop = stop_after_attempt(attempts)
        p.generate.retry.wait = wait_none()
        return original

    @pytest.mark.asyncio
    async def test_a_rate_limit_in_the_body_is_retried_and_the_next_answer_used(self):
        p = _make_provider()
        rate_limited = _mock_response(body={"id": "gen-1", "error": {"code": 429, "message": "temporarily rate-limited upstream"}})
        p._client = AsyncMock()
        p._client.post = AsyncMock(side_effect=[rate_limited, _mock_response()])
        original = self._fast_retries(p, 3)
        try:
            assert await p.generate("q") == "hello"
        finally:
            p.generate.retry.stop, p.generate.retry.wait = original
        assert p._client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_a_client_error_in_the_body_is_raised_once_with_its_message(self):
        p = _make_provider()
        rejected = _mock_response(body={"error": {"code": 400, "message": "Invalid schema for response_format"}})
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=rejected)
        original = self._fast_retries(p, 3)
        try:
            with pytest.raises(httpx.HTTPStatusError, match="Invalid schema"):
                await p.generate("q")
        finally:
            p.generate.retry.stop, p.generate.retry.wait = original
        assert p._client.post.call_count == 1, "a 400 must not be retried"

    @pytest.mark.asyncio
    async def test_a_body_with_neither_choices_nor_error_still_says_no_choices(self):
        """Negative control: the old message stays for the shape it actually describes."""
        p = _make_provider()
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=_mock_response(body={"id": "gen-2"}))
        original = self._fast_retries(p, 3)
        try:
            with pytest.raises(LLMProviderError, match="no choices"):
                await p.generate("q")
        finally:
            p.generate.retry.stop, p.generate.retry.wait = original


class TestAnErrorChunkInAStreamIsNotSkipped:
    """The streamed form of an error inside a 200 envelope. `_apply_stream_chunk` returns None for a chunk without
    choices, so an `{"error": ...}` chunk was skipped and the stream ended empty or cut short with no exception -
    which a caller that caches before parsing stores as the model's answer."""

    @pytest.mark.asyncio
    async def test_an_error_before_any_content_reopens_the_stream(self, monkeypatch):
        import pyutilz.llm.openai_compat as openai_compat_mod

        async def _no_sleep(_seconds):
            return None

        monkeypatch.setattr(openai_compat_mod, "MAX_RETRY_ATTEMPTS", 3)
        monkeypatch.setattr(openai_compat_mod.asyncio, "sleep", _no_sleep)
        error_line = f"data: {jsonlib.dumps({'id': 'gen-1', 'error': {'code': 429, 'message': 'temporarily rate-limited upstream'}})}"
        answer_lines = [f"data: {jsonlib.dumps({'choices': [{'delta': {'content': 'ok'}, 'finish_reason': 'stop'}]})}", "data: [DONE]"]
        p = _make_provider()
        p._client = AsyncMock()
        p._client.stream = MagicMock(side_effect=[_MockStreamResponse(lines=[error_line]), _MockStreamResponse(lines=answer_lines)])

        out = [chunk async for chunk in p.generate_stream("q")]

        assert out == ["ok"]
        assert p._client.stream.call_count == 2

    @pytest.mark.asyncio
    async def test_an_error_after_content_raises_instead_of_returning_a_cut_answer(self):
        lines = [
            f"data: {jsonlib.dumps({'choices': [{'delta': {'content': 'partial answer text'}}]})}",
            f"data: {jsonlib.dumps({'error': {'code': 502, 'message': 'upstream connection lost'}})}",
            "data: [DONE]",
        ]
        p = _make_provider()
        p._client = AsyncMock()
        p._client.stream = MagicMock(return_value=_MockStreamResponse(lines=lines))
        received = []

        with pytest.raises(httpx.HTTPStatusError, match="upstream connection lost"):
            async for chunk in p.generate_stream("q"):
                received.append(chunk)

        assert received == ["partial answer text"]


class TestFreshProviderDefaults:
    """Per-call and cumulative state start at their documented zero values (openai_compat 84-135)."""

    def test_the_per_call_usage_starts_with_zero_token_counts(self):
        provider = _make_provider()
        assert provider._last_usage == {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}

    def test_the_per_call_post_reasoning_count_starts_at_zero(self):
        assert _make_provider()._last_post_reasoning_tokens == 0

    def test_the_per_call_finish_reason_starts_as_none(self):
        assert _make_provider()._last_finish_reason is None

    def test_the_per_call_collections_start_empty(self):
        provider = _make_provider()
        assert provider.last_tool_calls == []
        assert provider.last_citations == []
        assert provider.last_rate_limits == {}

    def test_the_per_call_flags_start_false(self):
        provider = _make_provider()
        assert provider._last_json_schema_applied is False
        assert provider.last_json_mode_fallback is False

    def test_the_cumulative_counters_start_at_zero(self):
        provider = _make_provider()
        assert (provider.total_prompt_tokens, provider.total_completion_tokens) == (0, 0)
        assert (provider.total_cache_hit_tokens, provider.total_reasoning_tokens, provider.total_cache_write_tokens) == (0, 0, 0)
        assert provider._call_count == 0

    def test_the_long_context_tally_starts_at_zero(self):
        provider = _make_provider()
        assert provider._long_context_surcharge_usd == 0.0
        assert provider._long_context_calls == 0

    def test_the_concurrency_limit_defaults_to_ten(self):
        import inspect

        assert inspect.signature(OpenAICompatibleProvider.__init__).parameters["max_concurrent"].default == 10
        assert _make_provider()._max_concurrent == 10

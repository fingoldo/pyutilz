"""Tests for OpenAICompatibleProvider base class."""

from __future__ import annotations

from pyutilz.core import jsonlib
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from tenacity import stop_after_attempt, wait_none

httpx = pytest.importorskip("httpx")

from pyutilz.llm.exceptions import LLMProviderError
from pyutilz.llm.openai_compat import OpenAICompatibleProvider


class _TestProvider(OpenAICompatibleProvider):
    _base_url = "https://test.example.com"
    _provider_name = "TestProvider"
    _max_tokens_map = {"test-model": 4096}
    _default_max_tokens = 2048

    def _input_cost_per_1m(self, model: str) -> float:
        return 1.0

    def _output_cost_per_1m(self, model: str) -> float:
        return 2.0

    def _cache_hit_cost_per_1m(self, model: str) -> float:
        return 0.1


def _make_provider(model: str = "test-model") -> _TestProvider:
    return _TestProvider(api_key="test-key", model=model)  # pragma: allowlist secret


def _mock_response(status_code: int = 200, body: dict | None = None, text: str = "") -> httpx.Response:
    if body is None:
        body = {
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "prompt_cache_hit_tokens": 2,
                "completion_tokens_details": {"reasoning_tokens": 1},
            },
        }
    resp = httpx.Response(
        status_code=status_code,
        json=body,
        request=httpx.Request("POST", "https://test.example.com/chat/completions"),
    )
    return resp


class _MockStreamResponse:
    """Minimal async-context-manager mock standing in for
    ``httpx.AsyncClient.stream(...)``'s ``async with ... as resp:`` result.
    """

    def __init__(self, status_code: int = 200, headers: dict | None = None, lines: list[str] | None = None):
        self.status_code = status_code
        self.headers = headers or {}
        self._lines = lines or []
        self.request = httpx.Request("POST", "https://test.example.com/chat/completions")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"status {self.status_code}",
                request=self.request,
                response=httpx.Response(self.status_code, headers=self.headers, request=self.request),
            )

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


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


class _FakeResponse:
    """The three response shapes an intermediary can return in place of the API's JSON envelope."""

    def __init__(self, text: str, status_code: int = 200, payload: object | None = None, raises: bool = False):
        self.text = text
        self.status_code = status_code
        self._payload = payload
        self._raises = raises

    def json(self) -> object:
        if self._raises:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._payload


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


class _WarnLog:
    def __init__(self):
        self.calls: list = []

    def warning(self, msg, *args, **kwargs):
        self.calls.append((msg, args))


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


class _StatusError(Exception):
    def __init__(self, response=None):
        super().__init__("status")
        self.response = response


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


class _FakeHttpClient:
    def __init__(self, closed: bool = False, fail: bool = False):
        self.is_closed = closed
        self.fail = fail
        self.close_calls = 0

    async def aclose(self):
        if self.fail:
            raise RuntimeError("socket already gone")
        self.close_calls += 1
        self.is_closed = True


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


class _FakeStreamResponse:
    def __init__(self, status: int = 200, lines: tuple = (), headers: dict | None = None):
        self.status_code = status
        self.headers = headers or {}
        self._lines = lines
        self.aread_calls = 0
        self.request = httpx.Request("POST", "https://test.example.com/chat/completions")

    async def aread(self):
        self.aread_calls += 1
        return b""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeStreamContext:
    def __init__(self, owner, response):
        self._owner = owner
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


class _FakeStreamingHttpClient:
    def __init__(self, response):
        self._response = response
        self.calls: list = []

    def stream(self, method, url, json=None, timeout=None):
        self.calls.append((method, url, json, timeout))
        return _FakeStreamContext(self, self._response)


async def _drain_stream(provider, body=None, state=None):
    box = {"streamed_text": []}
    attempt_state = state if state is not None else {}
    async for _ in provider._stream_attempt(body or {"model": "test-model"}, attempt_state, box):
        pass
    return attempt_state


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


def _content_line(text: str) -> str:
    return "data: " + jsonlib.dumps({"choices": [{"delta": {"content": text}}]})


async def _drain_with_box(provider, lines: tuple):
    provider._client = _FakeStreamingHttpClient(_FakeStreamResponse(status=200, lines=lines))
    box: dict = {"streamed_text": []}
    attempt_state: dict = {}
    yielded = [piece async for piece in provider._stream_attempt({"model": "test-model"}, attempt_state, box)]
    return yielded, box, attempt_state


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


class _StatusStreamResponse(_FakeStreamResponse):
    """A failed stream open: raise_for_status raises the httpx error the retry predicate classifies."""

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://test.example.com/chat/completions")
            raise httpx.HTTPStatusError("boom", request=request, response=httpx.Response(self.status_code, request=request))


class _SequenceStreamClient:
    """Hands out one scripted response per stream() call; the last one repeats."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list = []

    def stream(self, method, url, json=None, timeout=None):
        self.calls.append(json)
        response = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        return _FakeStreamContext(self, response)


def _no_sleep(monkeypatch, provider, wait: float = 7.0):
    import asyncio

    waits: list = []

    async def fake_sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(provider, "_stream_retry_backoff", lambda exc, attempt: wait)
    return waits


async def _collect_stream(provider, **kwargs):
    return [piece async for piece in provider.generate_stream("hi", **kwargs)]


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


class _PreparationOrderError(Exception):
    pass


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


class _WarnRecorder:
    def __init__(self):
        self.warnings: list = []

    def warning(self, fmt, *args, **kwargs):
        self.warnings.append((fmt, args))


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


class _JsonPostResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status_code = status
        self.headers: dict = {}
        self.text = jsonlib.dumps(payload)
        self.content = self.text.encode()

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


class _PostClient:
    def __init__(self, response):
        self._response = response
        self.calls: list = []

    async def post(self, url, json=None, timeout=None):
        self.calls.append((url, json))
        return self._response


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


class _StatusBodyResponse:
    def __init__(self, status: int, payload=None, text: str = ""):
        self.status_code = status
        self._payload = payload
        self.text = text
        self.headers: dict = {}

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


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


class _Tier:
    def __init__(self, applies: bool):
        self._applies = applies

    def applies(self, tokens):
        return self._applies


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

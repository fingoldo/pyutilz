"""Tests for OpenAICompatibleProvider base class."""

from __future__ import annotations

import json
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
    return _TestProvider(api_key="test-key", model=model)


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
        if _HAS_TIKTOKEN:
            assert result == 13
        else:
            assert result == 25


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

    def test_handle_special_status_noop(self):
        p = _make_provider()
        resp = _mock_response(status_code=200)
        p._handle_special_status(resp)


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
            f"data: {json.dumps({'choices': [], 'usage': usage})}",
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
    """The auto-selected output budget must fit the context window, or the upstream rejects the whole
    request with HTTP 400 before generating anything (llama-3.3-70b: 128k output cap, 131k window)."""

    class _WideOutputProvider(_TestProvider):
        _max_tokens_map = {"wide-model": 128_000}
        _context_window_map = {"wide-model": 131_072}

    @pytest.mark.asyncio
    async def test_auto_budget_is_clamped_into_the_window(self):
        p = self._WideOutputProvider(api_key="k", model="wide-model")
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=_mock_response())
        await p.generate("x " * 8000)  # max_tokens omitted => provider max (128k) would overflow
        body = p._client.post.call_args.kwargs.get("json") or p._client.post.call_args[1].get("json")
        assert body["max_tokens"] < 128_000
        assert body["max_tokens"] > 0

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
    `json.JSONDecodeError` - a `ValueError` - which matches neither the HTTP-status nor the transport branch
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

    # 54,853 tokens at the pessimistic 30 tok/s floor is ~1,828 s, which must win over the 240 s default.
    assert probe._timeout_for({"max_tokens": 54_853}) == pytest.approx(54_853 / 30.0)
    # The name heuristic stays a FLOOR: a small request on a slow-tier model keeps its long allowance,
    # because the reason that model is slow has nothing to do with how much it was asked to write.
    assert probe._timeout_for({"max_tokens": 500}) == 240.0
    # `max_completion_tokens` is the same request under the newer field name.
    assert probe._timeout_for({"max_completion_tokens": 54_853}) == pytest.approx(54_853 / 30.0)
    # A body that names no budget at all falls back to the heuristic rather than to zero.
    assert probe._timeout_for({}) == 240.0
    assert probe._timeout_for({"max_tokens": None}) == 240.0
    assert probe._timeout_for({"max_tokens": "not a number"}) == 240.0


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
        asyncio.run(probe._post_and_unwrap({"max_tokens": 54_853}))
    assert seen["timeout"] == pytest.approx(54_853 / 30.0)

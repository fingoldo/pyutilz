"""Tests for OpenAICompatibleProvider base class."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch, MagicMock

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

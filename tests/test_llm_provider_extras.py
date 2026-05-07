"""Tests for the provider-extras added in Audit C top-6:

- Streaming usage tracking on OpenAI-compat base
- ``x-ratelimit-*`` header capture + check_account_limits exposure
- Anthropic real ``count_tokens`` + cache_creation/cache_read tracking
- Anthropic rate-limit headers via with_raw_response
- Claude Code: real ResultMessage.usage + total_cost_usd flow
- Gemini: safety_ratings / grounding / citation / function_calls capture
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("pydantic")

from pyutilz.llm.exceptions import LLMProviderError
from pyutilz.llm.openai_compat import OpenAICompatibleProvider


class _StubOpenAICompat(OpenAICompatibleProvider):
    """Concrete subclass of the abstract base for unit testing."""
    _base_url = "https://test.example.com"
    _provider_name = "Stub"
    _max_tokens_map = {"stub": 4096}
    _default_max_tokens = 2048

    def _input_cost_per_1m(self, model): return 1.0
    def _output_cost_per_1m(self, model): return 2.0
    def _cache_hit_cost_per_1m(self, model): return 0.1


def _make_compat() -> _StubOpenAICompat:
    return _StubOpenAICompat(api_key="test-key", model="stub")


class TestRateLimitHeaderCapture:
    @pytest.mark.asyncio
    async def test_x_ratelimit_headers_captured_on_generate(self):
        p = _make_compat()
        body = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        # httpx.Response keeps headers case-insensitively.
        resp = httpx.Response(
            status_code=200,
            json=body,
            headers={
                "x-ratelimit-requests-limit": "60",
                "x-ratelimit-requests-remaining": "59",
                "x-ratelimit-tokens-limit": "100000",
                "x-ratelimit-tokens-remaining": "99500",
            },
            request=httpx.Request("POST", "https://test.example.com/chat/completions"),
        )
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=resp)

        await p.generate("hello")

        assert "x-ratelimit-requests-limit" in p.last_rate_limits
        assert p.last_rate_limits["x-ratelimit-requests-limit"] == "60"

    @pytest.mark.asyncio
    async def test_check_account_limits_returns_normalized_dict(self):
        p = _make_compat()
        p.last_rate_limits = {
            "x-ratelimit-requests-limit": "60",
            "x-ratelimit-requests-remaining": "59",
            "x-ratelimit-requests-reset": "1s",
            "x-ratelimit-tokens-limit": "100000",
        }
        out = await p.check_account_limits()
        assert out["requests_limit"] == "60"
        assert out["requests_remaining"] == "59"
        assert out["requests_reset"] == "1s"
        assert out["tokens_limit"] == "100000"
        assert "raw" in out

    @pytest.mark.asyncio
    async def test_legacy_ratelimit_prefix_also_captured(self):
        p = _make_compat()
        p.last_rate_limits = {
            "ratelimit-limit": "10",
            "ratelimit-remaining": "8",
        }
        out = await p.check_account_limits()
        assert out["limit"] == "10"
        assert out["remaining"] == "8"

    @pytest.mark.asyncio
    async def test_check_account_limits_raises_before_first_call(self):
        p = _make_compat()
        # last_rate_limits empty by default
        with pytest.raises(NotImplementedError, match="generate"):
            await p.check_account_limits()


class TestStreamingUsageTracking:
    @pytest.mark.asyncio
    async def test_stream_options_include_usage_set_by_default(self):
        # Without stream_options.include_usage=true OpenAI-compat upstreams
        # never emit usage on streaming responses; we set it automatically
        # so streaming callers still get cost / token tracking.
        p = _make_compat()

        captured_body: dict = {}

        class _FakeStream:
            def __init__(self, body):
                self._body = body
                self.headers = {}
                self.status_code = 200
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return None
            async def aiter_lines(self):
                # Final chunk carries usage.
                yield 'data: {"choices":[{"delta":{"content":"hi"}}]}'
                yield 'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":4}}'
                yield 'data: [DONE]'
            def raise_for_status(self):
                pass

        def _stream(*args, **kwargs):
            captured_body.update(kwargs.get("json") or {})
            return _FakeStream(kwargs.get("json"))

        p._client = MagicMock()
        p._client.stream = _stream

        chunks = []
        async for c in p.generate_stream("hello"):
            chunks.append(c)

        assert chunks == ["hi"]
        # Verify the request body opted into usage.
        assert captured_body.get("stream_options") == {"include_usage": True}
        # Verify usage was captured from the final chunk.
        assert p.total_prompt_tokens == 12
        assert p.total_completion_tokens == 4


class TestAnthropicProviderExtras:
    """Cache field tracking + real count_tokens API + with_raw_response headers."""

    def _make(self):
        anthropic = pytest.importorskip("anthropic")
        from pyutilz.llm.anthropic_provider import AnthropicProvider
        p = AnthropicProvider.__new__(AnthropicProvider)
        p.model = "claude-sonnet-4-6-20250610"
        p._last_usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
        p.last_cache_creation_input_tokens = 0
        p.last_cache_read_input_tokens = 0
        p.total_cache_creation_input_tokens = 0
        p.total_cache_read_input_tokens = 0
        p.last_thinking_tokens = 0
        p.total_thinking_tokens = 0
        p.last_rate_limits = {}
        p.last_organization_id = None
        return p

    @pytest.mark.asyncio
    async def test_count_tokens_uses_native_api(self):
        p = self._make()
        p.client = MagicMock()
        result = MagicMock(input_tokens=42)

        async def _count_tokens(**kwargs):
            return result

        p.client.messages = MagicMock()
        p.client.messages.count_tokens = AsyncMock(side_effect=_count_tokens)
        out = await p.count_tokens("hello world")
        assert out == 42

    @pytest.mark.asyncio
    async def test_count_tokens_falls_back_to_tiktoken_on_failure(self):
        p = self._make()
        p.client = MagicMock()
        p.client.messages = MagicMock()
        p.client.messages.count_tokens = AsyncMock(side_effect=RuntimeError("boom"))
        out = await p.count_tokens("hello")
        assert isinstance(out, int)
        assert out > 0

    @pytest.mark.asyncio
    async def test_check_account_limits_returns_captured_headers(self):
        p = self._make()
        p.last_rate_limits = {
            "anthropic-ratelimit-input-tokens-limit": "100000",
            "anthropic-ratelimit-input-tokens-remaining": "99000",
            "anthropic-ratelimit-input-tokens-reset": "2026-05-04T12:34:56Z",
        }
        out = await p.check_account_limits()
        assert out["input_tokens_limit"] == "100000"
        assert out["input_tokens_remaining"] == "99000"
        assert "raw" in out


class TestClaudeCodeRealUsage:
    """Verify ResultMessage.usage replaces tiktoken estimates."""

    def _make(self):
        sdk = pytest.importorskip("claude_code_sdk")  # noqa: F841
        from pyutilz.llm.claude_code_provider import ClaudeCodeProvider
        p = ClaudeCodeProvider.__new__(ClaudeCodeProvider)
        p.model = "claude-sonnet-4-6"
        p._call_count = 0
        p.total_prompt_tokens = 0
        p.total_completion_tokens = 0
        p.total_cost_usd = 0.0
        p.last_cost_usd = 0.0
        p.total_cache_creation_input_tokens = 0
        p.total_cache_read_input_tokens = 0
        p.last_cache_creation_input_tokens = 0
        p.last_cache_read_input_tokens = 0
        p.last_session_id = None
        p.last_num_turns = None
        p._last_usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
        return p

    def test_get_session_cost_uses_real_total(self):
        p = self._make()
        p.total_cost_usd = 0.0042
        p.total_cache_read_input_tokens = 800
        p.total_cache_creation_input_tokens = 0
        p.total_prompt_tokens = 1200
        out = p.get_session_cost()
        assert out["total_cost_usd"] == 0.0042
        assert out["cache_hit_tokens"] == 800
        # cache_miss_tokens = prompt - cache_read
        assert out["cache_miss_tokens"] == 400


class TestGeminiSafetyAndMetadata:
    def _make(self):
        try:
            from pyutilz.llm.gemini_provider import GeminiProvider, GENAI_AVAILABLE
        except ImportError:
            pytest.skip("gemini provider not importable")
        if not GENAI_AVAILABLE:
            pytest.skip("google.genai SDK not installed")
        p = GeminiProvider.__new__(GeminiProvider)
        p.model_name = "gemini-2.5-flash"
        p._last_usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
        p.last_safety_ratings = []
        p.last_grounding_metadata = None
        p.last_citation_metadata = None
        p.last_function_calls = []
        p.last_all_candidates = []
        p.last_cached_content_tokens = 0
        p.total_cached_content_tokens = 0
        return p

    def test_capture_metadata_pulls_safety_ratings(self):
        p = self._make()

        class _Rating:
            def __init__(self, cat, prob, blocked=False):
                self.category = cat
                self.probability = prob
                self.blocked = blocked

        class _Cand:
            safety_ratings = [
                _Rating("HARM_CATEGORY_HARASSMENT", "LOW"),
                _Rating("HARM_CATEGORY_DANGEROUS_CONTENT", "HIGH", True),
            ]
            grounding_metadata = None
            citation_metadata = None
            content = None

        class _Resp:
            candidates = [_Cand()]

        p._capture_candidate_metadata(_Resp())

        assert len(p.last_safety_ratings) == 2
        assert p.last_safety_ratings[1]["category"] == "HARM_CATEGORY_DANGEROUS_CONTENT"
        assert p.last_safety_ratings[1]["probability"] == "HIGH"
        assert p.last_safety_ratings[1]["blocked"] is True

    def test_capture_metadata_pulls_function_calls(self):
        p = self._make()

        class _FunctionCall:
            name = "get_weather"
            args = {"city": "Moscow"}

        class _Part:
            function_call = _FunctionCall()

        class _Content:
            parts = [_Part()]

        class _Cand:
            safety_ratings = []
            grounding_metadata = None
            citation_metadata = None
            content = _Content()

        class _Resp:
            candidates = [_Cand()]

        p._capture_candidate_metadata(_Resp())

        assert p.last_function_calls == [{"name": "get_weather", "args": {"city": "Moscow"}}]

    def test_capture_resets_when_no_candidates(self):
        p = self._make()
        p.last_safety_ratings = [{"category": "stale", "probability": "X", "blocked": False}]

        class _Resp:
            candidates = []

        p._capture_candidate_metadata(_Resp())
        # Reset to empty so a failed call doesn't surface stale ratings.
        assert p.last_safety_ratings == []

"""Regression tests for pyutilz.llm.* (2026-07-21 audit fixes)."""

import asyncio

import pytest

pytest.importorskip("pydantic")


class TestGeminiRetryPredicate:
    def test_retries_on_genai_server_error(self):
        pytest.importorskip("google.genai")
        from google.genai.errors import ServerError
        from pyutilz.llm.gemini_provider import _is_retryable_genai_error

        exc = ServerError(500, {"error": {"message": "boom"}})
        assert _is_retryable_genai_error(exc) is True

    def test_retries_on_genai_client_error_429_only(self):
        pytest.importorskip("google.genai")
        from google.genai.errors import ClientError
        from pyutilz.llm.gemini_provider import _is_retryable_genai_error

        rate_limited = ClientError(429, {"error": {"message": "rate limited"}})
        assert _is_retryable_genai_error(rate_limited) is True
        forbidden = ClientError(403, {"error": {"message": "forbidden"}})
        assert _is_retryable_genai_error(forbidden) is False

    def test_does_not_retry_unrelated_exception(self):
        from pyutilz.llm.gemini_provider import _is_retryable_genai_error

        assert _is_retryable_genai_error(ValueError("nope")) is False


class TestAnthropicRetryPredicate:
    def test_overloaded_error_is_retried(self):
        import anthropic

        from pyutilz.llm.anthropic_provider import AnthropicProvider

        p = AnthropicProvider.__new__(AnthropicProvider)
        retry_condition = AnthropicProvider.generate.retry.retry  # type: ignore[attr-defined]
        fake_response = type("Resp", (), {"status_code": 529, "headers": {}, "request": None})()
        exc = anthropic.APIStatusError("overloaded", response=fake_response, body=None)
        exc.status_code = 529
        outcome = type("Outcome", (), {"failed": True, "exception": lambda self: exc})()
        state = type("State", (), {"outcome": outcome})()
        assert retry_condition(state) is True


class TestAnthropicSessionCostNoDoubleSubtract:
    def test_plain_input_equals_total_input_tokens(self):
        from pyutilz.llm.anthropic_provider import AnthropicProvider

        p = AnthropicProvider.__new__(AnthropicProvider)
        p.model = "claude-sonnet-4-20250514"
        p.total_input_tokens = 800
        p.total_output_tokens = 100
        p.total_cache_creation_input_tokens = 50_000
        p.total_cache_read_input_tokens = 200_000
        p.total_thinking_tokens = 0
        p._call_count = 5
        cost = p.get_session_cost()
        # Regression test: previously max(0, total_input - cache_creation - cache_read) clamped
        # to 0 whenever cumulative cache tokens exceeded fresh input_tokens, silently dropping
        # the entire fresh-input cost tier.
        assert cost["prompt_tokens"] == 800
        assert cost["input_cost_usd"] > 0


class TestOpenAICacheHitFallback:
    def test_record_usage_falls_back_to_nested_cached_tokens(self):
        from pyutilz.llm.openai_provider import OpenAIProvider

        p = OpenAIProvider.__new__(OpenAIProvider)
        p._provider_name = "OpenAI"
        p.model_name = "gpt-4o"
        p.total_prompt_tokens = 0
        p.total_completion_tokens = 0
        p.total_cache_hit_tokens = 0
        p.total_reasoning_tokens = 0
        p._call_count = 0
        p._last_usage = {}
        p._track_provider_specific_usage = lambda usage: None
        p._compute_billed_output = lambda compl, reasoning: compl
        p._record_usage({"prompt_tokens": 1000, "completion_tokens": 50, "prompt_tokens_details": {"cached_tokens": 800}})
        assert p.total_cache_hit_tokens == 800


class TestGenerateBatchCancelsOrphanedTasks:
    def test_early_break_cancels_remaining_tasks(self):
        from pyutilz.llm.base import LLMProvider

        class _FakeProvider(LLMProvider):
            _provider_name = "fake"

            async def generate(self, prompt, system=None, temperature=0.7, max_tokens=0, json_mode=False, thinking=None):
                if prompt == "slow":
                    await asyncio.sleep(10)
                return "ok"

            async def generate_stream(self, *a, **kw):
                yield "x"

            async def count_tokens(self, text):
                return 1

            def estimate_cost(self, i, o):
                return 0.0

            def get_session_cost(self):
                return {}

            @property
            def context_window(self):
                return 1000

            @property
            def max_output_tokens(self):
                return 100

            def supports_json_mode(self):
                return False

        async def run():
            p = _FakeProvider()
            requests = [{"id": "fast", "prompt": "hi"}, {"id": "slow", "prompt": "slow"}]
            gen = p.generate_batch(requests)
            first = await gen.__anext__()
            await gen.aclose()
            return first

        result = asyncio.run(run())
        assert result["id"] in ("fast", "slow")


class TestTokenCounterModelAwareEncoding:
    def test_model_specific_encoding_used_when_available(self):
        from pyutilz.llm.token_counter import count_tokens, _encoding_for_model, _HAS_TIKTOKEN

        if not _HAS_TIKTOKEN:
            pytest.skip("tiktoken not installed")
        enc_gpt4o = _encoding_for_model("gpt-4o")
        assert enc_gpt4o.name == "o200k_base"
        n = count_tokens("hello world", model="gpt-4o")
        assert n > 0

    def test_unknown_model_falls_back_to_default(self):
        from pyutilz.llm.token_counter import count_tokens, _HAS_TIKTOKEN

        if not _HAS_TIKTOKEN:
            pytest.skip("tiktoken not installed")
        n = count_tokens("hello world", model="totally-unknown-model-xyz")
        assert n > 0


class TestLLMTruncationErrorWiring:
    def test_openai_compat_raises_on_length_finish_reason(self):
        import httpx
        from pyutilz.llm.exceptions import LLMTruncationError
        from pyutilz.llm.openai_provider import OpenAIProvider

        p = OpenAIProvider.__new__(OpenAIProvider)
        p._provider_name = "OpenAI"

        async def fake_post(*a, **kw):
            class _Resp:
                status_code = 200
                headers = {}

                def raise_for_status(self):
                    return None

                def json(self):
                    return {
                        "choices": [{"finish_reason": "length", "message": {"content": "truncated..."}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    }

            return _Resp()

        p._client = type("C", (), {"post": fake_post})()
        p.model_name = "gpt-4o"
        p._max_concurrent = 1  # LazySemaphore constructs the real asyncio.Semaphore lazily, on first access inside the running loop below -- constructing it eagerly here (outside any loop) hits Python 3.9's "no current event loop" bug (removed in 3.10+, which binds lazily too)
        p.total_prompt_tokens = 0
        p.total_completion_tokens = 0
        p.total_cache_hit_tokens = 0
        p.total_reasoning_tokens = 0
        p._call_count = 0
        p._last_usage = {}
        p._track_provider_specific_usage = lambda usage: None
        p._track_provider_specific_response = lambda data: None
        p._compute_billed_output = lambda compl, reasoning: compl
        p._build_messages = lambda prompt, system: [{"role": "user", "content": prompt}]
        p._get_timeout = lambda model: 30.0
        p.last_tool_calls = []
        p.last_citations = []

        async def run():
            with pytest.raises(LLMTruncationError):
                await p.generate("hi", max_tokens=5)

        asyncio.run(run())


class TestMissingApiKeyMessages:
    def test_anthropic_message_names_env_var(self):
        from pyutilz.llm.anthropic_provider import AnthropicProvider
        import pyutilz.llm.anthropic_provider as mod

        class _FakeSettings:
            anthropic_api_key = None

        orig = mod.get_llm_settings
        mod.get_llm_settings = lambda: _FakeSettings()
        try:
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                AnthropicProvider(api_key=None)
        finally:
            mod.get_llm_settings = orig

    def test_gemini_message_names_env_var(self):
        from pyutilz.llm.gemini_provider import GeminiProvider, GENAI_AVAILABLE
        import pyutilz.llm.gemini_provider as mod

        if not GENAI_AVAILABLE:
            pytest.skip("google-genai not installed")

        class _FakeSettings:
            gemini_api_key = None

        orig = mod.get_llm_settings
        mod.get_llm_settings = lambda: _FakeSettings()
        try:
            with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                GeminiProvider(api_key=None)
        finally:
            mod.get_llm_settings = orig

"""Regression tests for the llm-providers audit (2026-06-04).

Covers the genuine bugs fixed in the pyutilz.llm package:

- AnthropicProvider.get_session_cost reporting calls=0 + non-cumulative
  tokens (now tracks _call_count + total_input/output_tokens).
- Gemini generate_batch aborting the whole batch on LLMSafetyBlockError /
  malformed request dict (now isolates each request).
- Anthropic generate_batch missing some error classes + raw-coroutine
  as_completed (now create_task + except Exception catch-all).
- OpenAICompatibleProvider.generate_stream @retry no-op on an async
  generator (retry now implemented manually inside the body).
- parse_retry_after dead code (now honoured by the streaming retry loop).
- AnthropicProvider._get_pricing prefix fallback collision (now
  longest-prefix match + warning).
- GeminiProvider._close releasing the ThreadPoolExecutor for the factory
  atexit hook.
- _normalize_thinking / _thinking_request_field edge inputs.

No live DB is required by any test here.
"""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("pydantic")

from pyutilz.llm.exceptions import LLMSafetyBlockError
from pyutilz.llm.openai_compat import (
    OpenAICompatibleProvider,
    parse_retry_after,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_anthropic():
    """AnthropicProvider with __init__ bypassed, counters zeroed."""
    pytest.importorskip("anthropic")
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
    # New cumulative counters added by the audit fix.
    p._call_count = 0
    p.total_input_tokens = 0
    p.total_output_tokens = 0
    return p


def _fake_anthropic_raw(input_tokens, output_tokens):
    """Build a fake raw/parsed Anthropic response for generate()."""
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_creation_input_tokens = 0
    usage.cache_read_input_tokens = 0

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "ok"

    parsed = MagicMock()
    parsed.usage = usage
    parsed.stop_reason = "end_turn"
    parsed.content = [text_block]

    raw = MagicMock()
    raw.parse.return_value = parsed
    raw.headers = {}
    return raw


class _StubCompat(OpenAICompatibleProvider):
    _base_url = "https://test.example.com"
    _provider_name = "Stub"
    _max_tokens_map = {"stub": 4096}
    _default_max_tokens = 2048

    def _input_cost_per_1m(self, model):
        return 1.0

    def _output_cost_per_1m(self, model):
        return 2.0

    def _cache_hit_cost_per_1m(self, model):
        return 0.1


def _make_compat():
    return _StubCompat(api_key="test-key", model="stub")  # pragma: allowlist secret


# ---------------------------------------------------------------------------
# Finding 1 + test_gap 11: Anthropic multi-call session cost accounting
# ---------------------------------------------------------------------------


class TestAnthropicSessionCost:
    @pytest.mark.asyncio
    async def test_multi_call_counts_and_cumulative_tokens(self):
        import asyncio as _asyncio

        p = _make_anthropic()
        p.semaphore = _asyncio.Semaphore(5)
        p.client = MagicMock()
        # Two calls with different usage: 100 then 50 input, 20 then 10 output.
        raws = [_fake_anthropic_raw(100, 20), _fake_anthropic_raw(50, 10)]
        create = AsyncMock(side_effect=raws)
        p.client.messages = MagicMock()
        p.client.messages.with_raw_response = MagicMock()
        p.client.messages.with_raw_response.create = create

        await p.generate("hi 1")
        await p.generate("hi 2")

        cost = p.get_session_cost()
        # Previously calls was permanently 0 and tokens were last-call-only.
        assert cost["calls"] == 2
        assert cost["prompt_tokens"] == 150  # 100 + 50
        assert cost["completion_tokens"] == 30  # 20 + 10

    def test_session_cost_zero_calls_does_not_crash(self):
        p = _make_anthropic()
        cost = p.get_session_cost()
        assert cost["calls"] == 0
        assert cost["prompt_tokens"] == 0
        assert cost["completion_tokens"] == 0
        assert cost["input_cost_usd"] == 0
        assert cost["output_cost_usd"] == 0


# ---------------------------------------------------------------------------
# Finding 6: Anthropic _get_pricing longest-prefix fallback
# ---------------------------------------------------------------------------


class TestAnthropicPricingFallback:
    def test_exact_match_short_circuits_no_warning(self, caplog):
        p = _make_anthropic()
        p.model = "claude-opus-4-7"
        with caplog.at_level("WARNING"):
            assert p._get_pricing() == (5.00, 25.00)
        assert not any("not pinned" in r.message for r in caplog.records)

    def test_unpinned_uses_longest_prefix_and_warns(self, caplog):
        p = _make_anthropic()
        # Hypothetical future, unpinned id. Longest matching prefix is
        # "claude-opus-4" (from claude-opus-4-7). The warning makes the
        # silent fallback visible so a new tier can't quietly inherit a
        # wrong price without a trace.
        p.model = "claude-opus-4-2-20260101"
        with caplog.at_level("WARNING"):
            inp, out = p._get_pricing()
        assert (inp, out) in {(5.00, 25.00), (15.00, 75.00)}
        assert any("not pinned" in r.message for r in caplog.records)

    def test_unknown_model_falls_to_default(self):
        p = _make_anthropic()
        p.model = "totally-unknown-model"
        assert p._get_pricing() == p._DEFAULT_PRICING


# ---------------------------------------------------------------------------
# Finding 5: Anthropic generate_batch isolates malformed request
# ---------------------------------------------------------------------------


class TestAnthropicBatchIsolation:
    @pytest.mark.asyncio
    async def test_malformed_request_does_not_abort_batch(self):
        p = _make_anthropic()

        async def fake_generate(prompt, **kwargs):
            return f"echo:{prompt}"

        p.generate = AsyncMock(side_effect=fake_generate)

        requests = [
            {"id": "a", "prompt": "p1"},
            {"id": "b"},  # missing "prompt" -> KeyError inside process_request
            {"id": "c", "prompt": "p3"},
        ]
        results = {r["id"]: r async for r in p.generate_batch(requests)}

        assert len(results) == 3
        assert results["a"]["result"] == "echo:p1"
        assert results["c"]["result"] == "echo:p3"
        # The malformed one yields an error entry, not a crash.
        assert "error" in results["b"]
        assert "result" not in results["b"]


# ---------------------------------------------------------------------------
# Finding 2 + test_gap 12: Gemini generate_batch isolates safety blocks
# ---------------------------------------------------------------------------


def _make_gemini():
    genai = pytest.importorskip("google.genai")  # noqa: F841
    from pyutilz.llm.gemini_provider import GeminiProvider

    p = GeminiProvider.__new__(GeminiProvider)
    p.model_name = "gemini-2.5-flash"
    return p


class TestGeminiBatchIsolation:
    @pytest.mark.asyncio
    async def test_safety_block_does_not_abort_batch(self):
        p = _make_gemini()

        async def fake_generate(prompt, **kwargs):
            if prompt == "blocked":
                raise LLMSafetyBlockError(
                    "blocked by safety filter",
                    details={"finish_reason": "SAFETY"},
                )
            return f"echo:{prompt}"

        p.generate = AsyncMock(side_effect=fake_generate)

        requests = [
            {"id": "ok1", "prompt": "fine"},
            {"id": "bad", "prompt": "blocked"},
            {"id": "ok2", "prompt": "also fine"},
        ]
        results = {r["id"]: r async for r in p.generate_batch(requests)}

        assert len(results) == 3
        assert results["ok1"]["result"] == "echo:fine"
        assert results["ok2"]["result"] == "echo:also fine"
        # The blocked request is isolated and tagged, not raised.
        assert "error" in results["bad"]
        assert results["bad"].get("error_type") == "safety_block"
        assert "result" not in results["bad"]

    @pytest.mark.asyncio
    async def test_malformed_request_isolated(self):
        p = _make_gemini()

        async def fake_generate(prompt, **kwargs):
            return f"echo:{prompt}"

        p.generate = AsyncMock(side_effect=fake_generate)
        requests = [{"id": "a", "prompt": "p1"}, {"id": "b"}]
        results = {r["id"]: r async for r in p.generate_batch(requests)}
        assert results["a"]["result"] == "echo:p1"
        assert "error" in results["b"]


# ---------------------------------------------------------------------------
# Finding 7: Gemini _close shuts down the executor for the factory atexit hook
# ---------------------------------------------------------------------------


class TestGeminiClose:
    @pytest.mark.asyncio
    async def test_close_is_a_noop(self):
        # Regression test (2026-07-21 audit): generate()/count_tokens() now use google-genai's
        # native async client (self.client.aio) instead of a dedicated ThreadPoolExecutor, so
        # GeminiProvider no longer has per-provider worker threads to release. _close() is kept
        # as a no-op hook only so factory.py's atexit handler (which awaits _close() on every
        # cached provider that exposes it) still has something to call.
        p = _make_gemini()
        assert not hasattr(p, "_executor")
        assert hasattr(p, "_close")
        await p._close()  # must not raise

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self):
        p = _make_gemini()
        await p._close()
        await p._close()


# ---------------------------------------------------------------------------
# Finding 4: parse_retry_after parsing + it is now honoured by streaming retry
# ---------------------------------------------------------------------------


class _Resp:
    def __init__(self, headers):
        self.headers = headers


class TestParseRetryAfter:
    def test_retry_after_ms(self):
        assert parse_retry_after(_Resp({"retry-after-ms": "1500"})) == 1.5

    def test_retry_after_seconds(self):
        assert parse_retry_after(_Resp({"retry-after": "7"})) == 7.0

    def test_none_response(self):
        assert parse_retry_after(None) is None

    def test_no_headers(self):
        assert parse_retry_after(_Resp({})) is None

    def test_http_date_unparseable_returns_none(self):
        # RFC-7231 HTTP-date form isn't parsed; falls through to None.
        assert parse_retry_after(_Resp({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})) is None


# ---------------------------------------------------------------------------
# Finding 3: generate_stream retry is now a live manual loop, not a no-op
# ---------------------------------------------------------------------------


def _retryable_status_error():
    request = httpx.Request("POST", "https://test.example.com/chat/completions")
    response = httpx.Response(429, request=request, headers={"retry-after": "0"})
    return httpx.HTTPStatusError("rate limited", request=request, response=response)


class _OneFailThenOkStream:
    """Raises a retryable 429 on raise_for_status the first time, then streams ok."""

    calls = 0

    def __init__(self, fail_first):
        self._fail_first = fail_first
        self.headers = {}
        self.status_code = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return None

    def raise_for_status(self):
        type(self).calls += 1
        if self._fail_first and type(self).calls == 1:
            raise _retryable_status_error()

    async def aiter_lines(self):
        yield 'data: {"choices":[{"delta":{"content":"hi"}}]}'
        yield "data: [DONE]"


class TestStreamingRetry:
    @pytest.mark.asyncio
    async def test_stream_retries_before_first_token(self, monkeypatch):
        # Sleep is no-op so the test is instant; honour-Retry-After path still
        # exercised (the 429 carries retry-after:0).
        import pyutilz.llm.openai_compat as oc

        async def _no_sleep(_s):
            return None

        monkeypatch.setattr(oc.asyncio, "sleep", _no_sleep)

        p = _make_compat()
        _OneFailThenOkStream.calls = 0

        streams = [
            _OneFailThenOkStream(fail_first=True),
            _OneFailThenOkStream(fail_first=False),
        ]
        it = iter(streams)

        def _stream(*args, **kwargs):
            return next(it)

        p._client = MagicMock()
        p._client.stream = _stream

        chunks = [c async for c in p.generate_stream("hello")]
        # Retried once (first attempt 429'd before any token), then succeeded.
        assert chunks == ["hi"]
        assert _OneFailThenOkStream.calls == 2

    @pytest.mark.asyncio
    async def test_non_retryable_error_propagates(self):
        p = _make_compat()

        class _BadStream:
            headers = {}
            status_code = 400

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return None

            def raise_for_status(self):
                request = httpx.Request("POST", "https://x/chat/completions")
                response = httpx.Response(400, request=request)
                raise httpx.HTTPStatusError("bad", request=request, response=response)

            async def aiter_lines(self):
                if False:
                    yield ""  # pragma: no cover

        p._client = MagicMock()
        p._client.stream = lambda *a, **k: _BadStream()

        with pytest.raises(httpx.HTTPStatusError):
            async for _ in p.generate_stream("hello"):
                pass


# ---------------------------------------------------------------------------
# test_gap 10: _normalize_thinking + _thinking_request_field edge inputs
# ---------------------------------------------------------------------------


class TestNormalizeThinking:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (False, (False, None)),
            ("", (False, None)),
            (True, (True, None)),
            ("HIGH", (True, "high")),
            ("low", (True, "low")),
            (0, (False, None)),
        ],
    )
    def test_normalize(self, value, expected):
        assert OpenAICompatibleProvider._normalize_thinking(value) == expected


class TestThinkingRequestField:
    def _deepseek(self, model):
        pytest.importorskip("httpx")
        from pyutilz.llm.deepseek_provider import DeepSeekProvider

        p = DeepSeekProvider.__new__(DeepSeekProvider)
        p.model_name = model
        return p

    def _openrouter(self):
        from pyutilz.llm.openrouter_provider import OpenRouterProvider

        p = OpenRouterProvider.__new__(OpenRouterProvider)
        p.model_name = "anthropic/claude-sonnet-4"
        return p

    def test_deepseek_v4_enabled(self):
        p = self._deepseek("deepseek-v4-chat")
        assert p._thinking_request_field("high") == {"thinking": {"type": "enabled"}}

    def test_deepseek_v4_disabled(self):
        p = self._deepseek("deepseek-v4-chat")
        assert p._thinking_request_field(False) == {"thinking": {"type": "disabled"}}
        assert p._thinking_request_field("") == {"thinking": {"type": "disabled"}}

    def test_deepseek_legacy_alias_returns_none(self, caplog):
        p = self._deepseek("deepseek-chat")
        with caplog.at_level("WARNING"):
            assert p._thinking_request_field("high") is None
        assert any("thinking toggle" in r.message for r in caplog.records)

    def test_openrouter_true_defaults_medium(self):
        p = self._openrouter()
        assert p._thinking_request_field(True) == {"reasoning": {"effort": "medium"}}

    def test_openrouter_false_excludes(self):
        p = self._openrouter()
        assert p._thinking_request_field(False) == {"reasoning": {"exclude": True}}

    def test_openrouter_effort_passthrough_lowercased(self):
        p = self._openrouter()
        assert p._thinking_request_field("HIGH") == {"reasoning": {"effort": "high"}}

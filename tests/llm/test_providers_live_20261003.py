"""Regression tests for the 2026-10-03 LIVE verification of the 2026-09-26 provider fixes (PROV-*).

The 2026-09-26 round was written from vendor docs alone. These tests pin what the real APIs answered on 2026-10-03,
replayed from SANITISED recorded response shapes (ids and keys removed, token counts and billed costs kept exactly), see
``audits/implemented/2026-09-26/provider_live_checks.json``. Each one asserts the value that was wrong before the fix:
a price, a request field, or whether an error is retried.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx
import pytest

from pyutilz.llm._claude_models import claude_model_spec, resolve_claude_model
from pyutilz.llm.anthropic_provider import anthropic_thinking_request
from pyutilz.llm.exceptions import LLMProviderError

# ─── Anthropic: GET /v1/models, 2026-10-03 ─────────────────────────────────────────────────────────


class TestClaudeTableMatchesTheModelsEndpoint:
    def test_sonnet_5_5_is_known_and_priced_at_its_list_rate(self, caplog: pytest.LogCaptureFixture) -> None:
        # Listed by /v1/models (max_input_tokens 1,000,000, max_tokens 128,000), priced $2 / $10 on the pricing page.
        # Before: no row, so it took the UNKNOWN fallback, $4 / $20 with a 64K cap and 200K context.
        with caplog.at_level(logging.WARNING):
            spec = claude_model_spec("claude-sonnet-5-5")
        assert (spec.input_per_1m, spec.output_per_1m, spec.max_output, spec.context_window) == (2.0, 10.0, 128_000, 1_000_000)
        assert spec.cache_read_multiplier == 0.10
        assert not caplog.records
        assert resolve_claude_model("claude-sonnet-5-5") is not None

    def test_sonnet_4_5_context_is_the_listed_1m(self) -> None:
        assert claude_model_spec("claude-sonnet-4-5-20250929").context_window == 1_000_000

    @pytest.mark.parametrize(
        ("model", "asked", "sent"),
        [
            # capabilities.effort.xhigh.supported is false on the 4.6 models; xhigh AND max are false on Opus 4.5.
            ("claude-sonnet-4-6", "xhigh", "high"),
            ("claude-opus-4-6", "xhigh", "high"),
            ("claude-sonnet-4-6", "max", "max"),
            ("claude-opus-4-5-20251101", "max", "high"),
            ("claude-opus-4-5-20251101", "xhigh", "high"),
            ("claude-opus-4-5-20251101", "medium", "medium"),
            ("claude-sonnet-5", "xhigh", "xhigh"),
            ("claude-sonnet-5-5", "max", "max"),
        ],
    )
    def test_effort_is_clamped_to_the_levels_the_model_takes(self, model: str, asked: str, sent: str) -> None:
        assert anthropic_thinking_request(asked, 20_000, model=model)["output_config"] == {"effort": sent}


# ─── OpenAI ────────────────────────────────────────────────────────────────────────────────────────

# Recorded 2026-10-03 from POST /v1/chat/completions on an account with no credit left.
_OPENAI_NO_CREDIT = {
    "error": {
        "message": "You have no credits remaining. Add credits to continue using the API at https://platform.openai.com/settings/organization/billing/.",
        "type": "insufficient_quota",
        "param": None,
        "code": "credit_balance_exhausted",
    }
}


def _openai(model: str, handler: Any) -> Any:
    from pyutilz.llm.openai_provider import OpenAIProvider

    p = OpenAIProvider(api_key="sk-fake", model=model)  # pragma: allowlist secret -- dummy key
    p._client = httpx.AsyncClient(base_url="https://api.openai.com/v1", transport=httpx.MockTransport(handler))
    p.fit_max_tokens_to_context = lambda mt, prompt, system=None: mt
    return p


class TestOpenAIExhaustedCreditIsNotRetried:
    @pytest.mark.asyncio
    async def test_generate_fails_at_once(self) -> None:
        posts: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            posts.append(1)
            return httpx.Response(429, json=_OPENAI_NO_CREDIT)

        p = _openai("gpt-5-nano", handler)
        # Before the fix the shared predicate retried every 429, so this call never returned (observed live: it slept
        # and re-sent until killed). wait_for turns a regression into a failure instead of a hang.
        with pytest.raises(LLMProviderError, match="no credit left") as info:
            await asyncio.wait_for(p.generate("Reply OK.", max_tokens=50, thinking=False), timeout=20)
        assert posts == [1]
        assert info.value.details["code"] == "credit_balance_exhausted"

    @pytest.mark.asyncio
    async def test_stream_fails_at_once(self) -> None:
        posts: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            posts.append(1)
            return httpx.Response(429, json=_OPENAI_NO_CREDIT)

        p = _openai("gpt-5-nano", handler)

        async def consume() -> None:
            async for _ in p.generate_stream("Reply OK.", max_tokens=50):
                pass

        with pytest.raises(LLMProviderError, match="no credit left"):
            await asyncio.wait_for(consume(), timeout=20)
        assert posts == [1]

    def test_an_ordinary_rate_limit_is_still_left_to_the_retry_policy(self) -> None:
        from pyutilz.llm.openai_provider import OpenAIProvider

        p = OpenAIProvider(api_key="sk-fake", model="gpt-5-nano")  # pragma: allowlist secret -- dummy key
        busy = {"error": {"message": "Rate limit reached for requests", "type": "requests", "code": "rate_limit_exceeded"}}
        resp = httpx.Response(429, json=busy, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
        assert p._handle_special_status(resp) is None


class TestOpenAIGpt6Table:
    @pytest.mark.parametrize(
        ("model", "effort"),
        [("gpt-6-sol", "low"), ("gpt-6.1-sol", "low"), ("gpt-6-astra", "low"), ("gpt-6-luna", "none"), ("gpt-5.6-sol", "none")],
    )
    def test_off_is_the_lowest_effort_the_model_accepts(self, model: str, effort: str) -> None:
        # Only Luna takes "none" in the GPT-6 family; "none" to Sol drew a 400 whose repair dropped reasoning_effort,
        # so "off" ran at the model's default effort.
        from pyutilz.llm.openai_provider import OpenAIProvider

        p = OpenAIProvider(api_key="sk-fake", model=model)  # pragma: allowlist secret -- dummy key
        assert p._thinking_request_field(False) == {"reasoning_effort": effort}

    def test_gpt_6_1_sol_has_its_own_row(self, caplog: pytest.LogCaptureFixture) -> None:
        from pyutilz.llm.openai_provider import OpenAIProvider

        p = OpenAIProvider(api_key="sk-fake", model="gpt-6.1-sol")  # pragma: allowlist secret -- dummy key
        with caplog.at_level(logging.WARNING):
            assert (p._input_cost_per_1m("gpt-6.1-sol"), p._output_cost_per_1m("gpt-6.1-sol")) == (2.0, 10.0)
            assert p._cache_hit_cost_per_1m("gpt-6.1-sol") == 0.10
        assert not caplog.records
        assert (p.max_output_tokens, p.context_window) == (128_000, 1_050_000)


# ─── xAI ───────────────────────────────────────────────────────────────────────────────────────────

_RL_HEADERS = {
    "x-ratelimit-limit-requests": "480",
    "x-ratelimit-remaining-requests": "479",
    "x-ratelimit-limit-tokens": "2000000",
    "x-ratelimit-remaining-tokens": "1999000",
}


def _xai(model: str, handler: Any = None, **kw: Any) -> Any:
    from pyutilz.llm.xai_provider import XAIProvider

    p = XAIProvider(api_key="xai-fake", model=model, **kw)  # pragma: allowlist secret -- dummy key
    if handler is not None:
        p._client = httpx.AsyncClient(base_url="https://api.x.ai/v1", transport=httpx.MockTransport(handler))
    p.fit_max_tokens_to_context = lambda mt, prompt, system=None: mt
    return p


def _chat(served: str, usage: dict[str, Any], text: str = "OK.") -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers=_RL_HEADERS,
            json={"model": served, "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}], "usage": usage},
        )

    return handler


def _details(cached: int, reasoning: int) -> dict[str, Any]:
    return {
        "prompt_tokens_details": {"text_tokens": 0, "audio_tokens": 0, "image_tokens": 0, "cached_tokens": cached},
        "completion_tokens_details": {"reasoning_tokens": reasoning, "audio_tokens": 0, "accepted_prediction_tokens": 0, "rejected_prediction_tokens": 0},
        "num_sources_used": 0,
    }


class TestXaiBilledOutputMatchesTheMoney:
    """PROV-21: ``completion_tokens`` EXCLUDES reasoning on chat completions; completion + reasoning is what is billed."""

    @pytest.mark.asyncio
    async def test_reasoning_call_costs_exactly_the_reported_ticks(self) -> None:
        # grok-4.3, thinking="medium", recorded: 219 prompt (128 cached), 3 completion, 210 reasoning, total 432.
        usage = {"prompt_tokens": 219, "completion_tokens": 3, "total_tokens": 432, **_details(128, 210), "cost_in_usd_ticks": 6718500}
        p = _xai("grok-4.3", _chat("grok-4.3", usage, "0.05"))
        await p.generate("q", max_tokens=300, thinking="medium", temperature=None)
        cost = p.get_session_cost()
        assert cost["total_cost_usd"] == pytest.approx(0.00067185, abs=1e-12)
        assert cost["reported_cost_usd"] == pytest.approx(0.00067185, abs=1e-12)
        # completion-only output would have been 0.00013935 + 3 * 2.5e-6: the reasoning tokens ARE billed.
        assert cost["output_cost_usd"] == pytest.approx(213 * 2.50 / 1e6, abs=1e-12)


class TestXaiPricesTheServingModel:
    @pytest.mark.asyncio
    async def test_retired_id_answered_by_grok_4_3_is_priced_as_grok_4_3(self, caplog: pytest.LogCaptureFixture) -> None:
        # Recorded: request "grok-4-1-fast-reasoning", response "model": "grok-4.3", 3,868,500 ticks.
        usage = {"prompt_tokens": 195, "completion_tokens": 2, "total_tokens": 306, **_details(128, 109), "cost_in_usd_ticks": 3868500}
        p = _xai("grok-4-1-fast-reasoning", _chat("grok-4.3", usage))
        with caplog.at_level(logging.WARNING, logger="pyutilz.llm.xai_provider"):
            await p.generate("Reply OK.", max_tokens=50, temperature=None)
        # Before: priced at the retired $0.20 / $0.50 row, $0.0000753 against $0.00038685 billed.
        assert p.get_session_cost()["total_cost_usd"] == pytest.approx(0.00038685, abs=1e-12)
        assert any("grok-4.3" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_grok_code_fast_1_is_billed_as_grok_build(self) -> None:
        # Recorded: request "grok-code-fast-1", response "model": "grok-build-0.1", 3,046,000 ticks.
        usage = {"prompt_tokens": 189, "completion_tokens": 2, "total_tokens": 298, **_details(128, 107), "cost_in_usd_ticks": 3046000}
        p = _xai("grok-code-fast-1", _chat("grok-build-0.1", usage))
        await p.generate("Reply OK.", max_tokens=50, temperature=None)
        assert p.get_session_cost()["total_cost_usd"] == pytest.approx(0.0003046, abs=1e-12)

    def test_default_model_is_a_current_one(self) -> None:
        from pyutilz.llm.xai_provider import XAIProvider

        assert XAIProvider(api_key="xai-fake").model_name == "grok-4.3"  # pragma: allowlist secret -- dummy key


class TestXaiEffortAndLimits:
    @pytest.mark.parametrize(
        ("model", "thinking", "expected"),
        [
            ("grok-4.3", False, {"reasoning_effort": "none"}),
            ("grok-4.3", "low", {"reasoning_effort": "low"}),
            ("grok-4.3", "max", {"reasoning_effort": "xhigh"}),
            ("grok-4.5", "xhigh", {"reasoning_effort": "xhigh"}),
            ("grok-4.5", False, {"reasoning_effort": "low"}),
        ],
    )
    def test_effort_follows_the_listed_capabilities(self, model: str, thinking: Any, expected: Any) -> None:
        assert _xai(model)._thinking_request_field(thinking) == expected

    @pytest.mark.asyncio
    async def test_check_account_limits_returns_the_headers_xai_sends(self) -> None:
        usage = {"prompt_tokens": 187, "completion_tokens": 2, "total_tokens": 189, **_details(128, 0), "cost_in_usd_ticks": 1043500}
        p = _xai("grok-4.3", _chat("grok-4.3", usage))
        await p.generate("Reply OK.", max_tokens=20, thinking=False, temperature=None)
        limits = await p.check_account_limits()
        assert limits["remaining_requests"] == "479"
        assert limits["limit_tokens"] == "2000000"


# Recorded 2026-10-03 from POST /v1/responses with the web_search + x_search tools on grok-4.3 (output text trimmed).
_RESPONSES_USAGE = {
    "input_tokens": 38134,
    "input_tokens_details": {"cached_tokens": 16576},
    "output_tokens": 1051,
    "output_tokens_details": {"reasoning_tokens": 814},
    "total_tokens": 39185,
    "num_sources_used": 0,
    "num_server_side_tools_used": 5,
    "cost_in_usd_ticks": 578902000,
    "server_side_tool_usage_details": {
        "web_search_calls": 5, "x_search_calls": 0, "x_posts_fetched": 0, "x_users_fetched": 0, "code_interpreter_calls": 0,
        "file_search_calls": 0, "mcp_calls": 0, "document_search_calls": 0, "image_generation_calls": 0,
    },
}


class TestXaiLiveSearchCost:
    @pytest.mark.asyncio
    async def test_responses_call_costs_exactly_the_reported_ticks(self) -> None:
        sent: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(json.loads(request.content))
            return httpx.Response(200, headers=_RL_HEADERS, json={
                "model": "grok-4.3",
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "A headline.", "annotations": [{"type": "url_citation", "url": "https://example.org/a"}]}]}],
                "usage": _RESPONSES_USAGE,
            })

        p = _xai("grok-4.3", handler, live_search=True)
        assert await p.generate("q", max_tokens=300, thinking=False, temperature=None) == "A headline."
        assert sent[0]["tools"] == [{"type": "web_search"}, {"type": "x_search"}]
        cost = p.get_session_cost()
        # Responses output_tokens INCLUDES reasoning: 237 visible + 814 reasoning, not 1051 + 814.
        assert (cost["completion_tokens"], cost["reasoning_tokens"]) == (237, 814)
        assert cost["tool_cost_usd"] == pytest.approx(5 * 0.005, abs=1e-12)
        # Before: $0.0349252 (814 reasoning tokens billed twice, the 5 searches not at all) against $0.0578902 billed.
        assert cost["total_cost_usd"] == pytest.approx(0.0578902, abs=1e-12)
        assert cost["reported_cost_usd"] == pytest.approx(0.0578902, abs=1e-12)
        assert p.last_citations == [{"type": "url_citation", "url": "https://example.org/a"}]


# ─── DeepSeek ──────────────────────────────────────────────────────────────────────────────────────


class TestDeepSeekLegacyAliasIsFlash:
    @pytest.mark.asyncio
    async def test_deepseek_chat_is_priced_and_sized_as_flash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pyutilz.llm.deepseek_provider as dsp

        monkeypatch.setattr(dsp, "deepseek_price_multiplier", lambda when=None: 1.0)  # peak, so the list price applies
        sent: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(json.loads(request.content))
            # Recorded: request "deepseek-chat", response "model": "deepseek-flash".
            return httpx.Response(200, json={
                "model": "deepseek-flash",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK."}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9, "prompt_tokens_details": {"cached_tokens": 0},
                          "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 7},
            })

        p = dsp.DeepSeekProvider(api_key="sk-fake", model="deepseek-chat")  # pragma: allowlist secret -- dummy key
        p._client = httpx.AsyncClient(base_url="https://api.deepseek.com", transport=httpx.MockTransport(handler))
        await p.generate("Reply OK.", max_tokens=50, thinking=False)
        assert sent[0]["thinking"] == {"type": "disabled"}
        # Flash peak rates: 7 * $0.30 + 2 * $1.20 per 1M. The old V3.2 row ($0.28 / $0.42) gave 0.0000028.
        assert p.get_session_cost()["total_cost_usd"] == pytest.approx(0.0000045, abs=1e-12)
        assert (p.max_output_tokens, p.context_window) == (393_216, 1_048_576)

    def test_effort_reaches_the_request(self) -> None:
        from pyutilz.llm.deepseek_provider import DeepSeekProvider

        p = DeepSeekProvider(api_key="sk-fake", model="deepseek-flash")  # pragma: allowlist secret -- dummy key
        assert p._thinking_request_field("low") == {"thinking": {"type": "enabled"}, "reasoning_effort": "low"}
        assert p._thinking_request_field("max") == {"thinking": {"type": "enabled"}, "reasoning_effort": "max"}


# ─── Gemini ────────────────────────────────────────────────────────────────────────────────────────

# Recorded 2026-10-03 (abridged): a free-tier key calling gemini-3.1-pro-preview.
_ZERO_QUOTA_BODY = {
    "error": {
        "code": 429,
        "message": "You exceeded your current quota, please check your plan and billing details. \n* Quota exceeded for metric: "
        "generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-3.1-pro\n"
        "Please retry in 4h59m49.874225726s.",
        "status": "RESOURCE_EXHAUSTED",
    }
}


class TestGeminiLive:
    def test_zero_quota_429_is_not_retried(self) -> None:
        ClientError = pytest.importorskip("google.genai.errors").ClientError

        from pyutilz.llm.gemini_provider import _is_retryable_genai_error

        assert _is_retryable_genai_error(ClientError(429, _ZERO_QUOTA_BODY)) is False

    def test_an_ordinary_429_still_is(self) -> None:
        ClientError = pytest.importorskip("google.genai.errors").ClientError

        from pyutilz.llm.gemini_provider import _is_retryable_genai_error

        busy = {"error": {"code": 429, "message": "Quota exceeded for metric: generate_content_requests, limit: 10, model: gemini-2.5-flash", "status": "RESOURCE_EXHAUSTED"}}
        assert _is_retryable_genai_error(ClientError(429, busy)) is True

    @pytest.mark.parametrize(
        ("model", "rates"),
        [
            ("gemini-3.6-flash", (0.75, 3.75, 0.075)),
            ("gemini-3.5-flash-lite", (0.30, 2.50, 0.03)),
            ("gemini-3.1-flash-lite", (0.25, 1.50, 0.025)),
        ],
    )
    def test_served_models_have_their_own_rows(self, model: str, rates: tuple[float, float, float], caplog: pytest.LogCaptureFixture) -> None:
        pytest.importorskip("google.genai")
        from pyutilz.llm.gemini_provider import GeminiProvider

        p = GeminiProvider.__new__(GeminiProvider)
        p.model_name = model
        with caplog.at_level(logging.WARNING):
            assert p._get_pricing() == rates[:2]
        assert not caplog.records
        assert GeminiProvider._CACHE_HIT_COST[model] == rates[2]

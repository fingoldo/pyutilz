"""Live LLM-provider tests -- cost a few cents per run; opt-in only.

Run with:
    pytest --run-live tests/test_llm_live.py
    pytest --run-live -m live

By DEFAULT every live test is skipped (even when the key is present)
via the ``pytest_collection_modifyitems`` hook in conftest.py. You
must pass ``--run-live`` to execute them. Each test ALSO skips
individually when its per-provider API key is missing, so contributors
running the live suite without every account still get partial coverage.

Each live test asserts ``assert_under_budget`` so an accidental huge
prompt fails the test rather than blowing the spend cap.
"""

import pytest

pytest.importorskip("httpx")
pytest.importorskip("pydantic")


@pytest.mark.live
@pytest.mark.asyncio
async def test_openrouter_minimal_call(openrouter_key, assert_under_budget):
    """Smoke test: cheapest stable model, tiny prompt, max_tokens=10."""
    from pyutilz.llm.openrouter_provider import OpenRouterProvider

    p = OpenRouterProvider(
        api_key=openrouter_key,
        model="openai/gpt-4o-mini",
    )
    out = await p.generate(
        "Reply with exactly the word OK.",
        max_tokens=10,
        temperature=0.0,
    )
    assert "OK" in out.upper()
    assert p.last_actual_cost_usd >= 0.0
    assert_under_budget(p.last_actual_cost_usd)
    # Confirm last_call_summary fully populated (Phase-1 reset hook works).
    s = p.last_call_summary()
    assert s["generation_id"] is not None
    assert s["upstream_provider"] is not None


@pytest.mark.live
@pytest.mark.asyncio
async def test_openrouter_health_check_free(openrouter_key):
    """Pre-flight health check costs zero credits."""
    from pyutilz.llm.openrouter_provider import OpenRouterProvider

    p = OpenRouterProvider(
        api_key=openrouter_key,
        model="openai/gpt-4o-mini",
    )
    h = await p.check_model_health()
    assert h["model"] == "openai/gpt-4o-mini"
    assert isinstance(h["endpoints"], list)
    # check_model_health does NOT bill against credits.
    assert p.last_actual_cost_usd == 0.0


@pytest.mark.live
@pytest.mark.asyncio
async def test_openrouter_account_credits(openrouter_key):
    """Verify the free /credits + /key introspection works on a real account."""
    from pyutilz.llm.openrouter_provider import OpenRouterProvider

    p = OpenRouterProvider(
        api_key=openrouter_key,
        model="openai/gpt-4o-mini",
    )
    creds = await p.get_account_credits()
    assert "balance_usd" in creds
    assert creds["currency"] == "USD"
    assert creds["is_available"] in (True, False)


@pytest.mark.live
@pytest.mark.asyncio
async def test_anthropic_minimal_call(anthropic_key, assert_under_budget):
    """Smoke test for the Anthropic provider with cache + ratelimit capture."""
    from pyutilz.llm.anthropic_provider import AnthropicProvider

    p = AnthropicProvider(
        api_key=anthropic_key,
        model="claude-haiku-4-5-20251001",
    )
    out = await p.generate(
        "Reply with exactly the word OK.",
        max_tokens=10,
        temperature=0.0,
    )
    assert "OK" in out.upper()
    # Phase-2 cache + header capture
    assert p.last_cache_creation_input_tokens >= 0
    assert p.last_cache_read_input_tokens >= 0
    assert isinstance(p.last_rate_limits, dict)
    # After a real call, headers must be populated.
    assert p.last_rate_limits, "anthropic-ratelimit-* headers missing"
    cost_estimate = p.estimate_cost(
        p._last_usage["input_tokens"], p._last_usage["output_tokens"]
    )
    assert_under_budget(cost_estimate)


@pytest.mark.live
@pytest.mark.asyncio
async def test_anthropic_native_count_tokens(anthropic_key):
    """Phase-2: real count_tokens API replaces tiktoken."""
    from pyutilz.llm.anthropic_provider import AnthropicProvider

    p = AnthropicProvider(
        api_key=anthropic_key,
        model="claude-haiku-4-5-20251001",
    )
    n = await p.count_tokens("Hello world")
    # Claude tokenizer rounds short inputs to 8-15 tokens; just sanity-check.
    assert 1 <= n <= 30


@pytest.mark.live
@pytest.mark.asyncio
async def test_deepseek_balance(deepseek_key):
    """Phase-1: /user/balance is the only LLM provider with public balance."""
    from pyutilz.llm.deepseek_provider import DeepSeekProvider

    p = DeepSeekProvider(api_key=deepseek_key, model="deepseek-v4-flash")
    info = await p.get_account_credits()
    assert "balance_usd" in info
    assert info["raw"] is not None


@pytest.mark.live
@pytest.mark.asyncio
async def test_deepseek_minimal_call(deepseek_key, assert_under_budget):
    from pyutilz.llm.deepseek_provider import DeepSeekProvider

    p = DeepSeekProvider(api_key=deepseek_key, model="deepseek-v4-flash")
    out = await p.generate(
        "Reply with exactly the word OK.",
        max_tokens=10,
        temperature=0.0,
        thinking=False,  # save tokens by skipping V4 thinking-by-default
    )
    assert "OK" in out.upper()
    cost = p.estimate_cost(
        p._last_usage["input_tokens"], p._last_usage["output_tokens"]
    )
    assert_under_budget(cost)


@pytest.mark.live
@pytest.mark.asyncio
async def test_xai_minimal_call(xai_key, assert_under_budget):
    from pyutilz.llm.xai_provider import XAIProvider

    p = XAIProvider(api_key=xai_key, model="grok-4-1-fast-non-reasoning")
    out = await p.generate(
        "Reply with exactly the word OK.",
        max_tokens=10,
        temperature=0.0,
    )
    assert "OK" in out.upper()
    cost = p.estimate_cost(
        p._last_usage["input_tokens"], p._last_usage["output_tokens"]
    )
    assert_under_budget(cost)


@pytest.mark.live
@pytest.mark.asyncio
async def test_openai_minimal_call(openai_key, assert_under_budget):
    import asyncio
    from pyutilz.llm.openai_provider import OpenAIProvider

    # gpt-4o-mini is the broadly-available cheap model. gpt-5-nano costs
    # less per token but isn't on every account / region; preferring the
    # available one over the cheaper one for a smoke test.
    #
    # The provider's tenacity retry loop is INFINITE for retryable
    # errors (429 rate-limit specifically). On an account that's been
    # heavily rate-limited the call would hang for many minutes.
    # ``asyncio.wait_for`` short-circuits to a clean test SKIP so a
    # rate-limited key doesn't block CI / dev iteration.
    p = OpenAIProvider(api_key=openai_key, model="gpt-4o-mini")
    try:
        out = await asyncio.wait_for(
            p.generate(
                "Reply with exactly the word OK.",
                max_tokens=10,
                temperature=0.0,
            ),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        pytest.skip(
            "OpenAI rate-limited or unreachable within 30s -- "
            "account-level issue, not a provider-code bug"
        )
    assert "OK" in out.upper()
    cost = p.estimate_cost(
        p._last_usage["input_tokens"], p._last_usage["output_tokens"]
    )
    assert_under_budget(cost)


@pytest.mark.live
@pytest.mark.asyncio
async def test_gemini_minimal_call(gemini_key, assert_under_budget):
    pytest.importorskip("google.genai")
    from pyutilz.llm.gemini_provider import GeminiProvider

    p = GeminiProvider(api_key=gemini_key, model="gemini-2.5-flash")
    try:
        out = await p.generate(
            "Reply with exactly the word OK.",
            max_tokens=10,
            temperature=0.0,
        )
    except Exception as exc:
        # Gemini occasionally returns 503 UNAVAILABLE during demand
        # spikes; that's an upstream concern, not pyutilz's. Don't
        # let it fail CI; skip cleanly so the rest of the live suite
        # still has signal.
        msg = str(exc)
        if "503" in msg or "UNAVAILABLE" in msg or "RESOURCE_EXHAUSTED" in msg:
            pytest.skip(f"Gemini upstream unavailable / quota: {msg[:100]}")
        raise
    assert "OK" in out.upper()
    # Phase-2: safety_ratings populated after every call (often empty list when nothing flagged)
    assert isinstance(p.last_safety_ratings, list)
    cost = p.estimate_cost(
        p._last_usage["input_tokens"], p._last_usage["output_tokens"]
    )
    assert_under_budget(cost)

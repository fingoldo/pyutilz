"""The shared pricing mechanism: cache-write rates and long-context tiers (``pyutilz.llm._pricing``).

Every expected USD figure is worked out by hand in the comment next to it from the rates in the provider tables
(audits/implemented/2026-09-26/provider_pricing_tiers.md). The xAI and Gemini equivalence tables run the pre-change
surcharge functions, copied verbatim below, against the migrated providers.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import pytest

from pyutilz.llm._pricing import LongContextTier, Pricing, long_context_surcharge, price_call

TIER = LongContextTier(100, 2.0, 1.5, 3.0, 4.0)


# ─── price_call ────────────────────────────────────────────────────────────────────────────────────


class TestPriceCall:
    def test_no_tier_is_the_plain_linear_price(self) -> None:
        p = Pricing(1.0, 10.0, cache_hit=0.1, cache_write=1.25)
        # miss 600 * 1.0 + read 100 * 0.1 + write 300 * 1.25 = 600 + 10 + 375 = 985; output (50 + 20) * 10 = 700.
        assert price_call(p, 1000, 50, cache_read=100, cache_write=300, reasoning=20) == pytest.approx((985e-6, 700e-6), rel=1e-12)

    def test_none_tier_changes_nothing(self) -> None:
        assert price_call(Pricing(1.0, 10.0), 10**6, 10**6) == price_call(Pricing(1.0, 10.0, None, None, None), 10**6, 10**6)
        assert long_context_surcharge(Pricing(1.0, 10.0), 10**9, 10**9) == (0.0, 0.0)

    def test_exclusive_threshold_boundary(self) -> None:
        p = Pricing(1.0, 10.0, long_context=TIER)
        # 100 prompt tokens do not EXCEED 100: 100 * 1.0 + output 10 * 10.
        assert price_call(p, 100, 10) == pytest.approx((100e-6, 100e-6), rel=1e-12)
        # 101 do: 101 * 1.0 * 2 = 202; output 10 * 10 * 1.5 = 150.
        assert price_call(p, 101, 10) == pytest.approx((202e-6, 150e-6), rel=1e-12)

    def test_inclusive_threshold_boundary(self) -> None:
        p = Pricing(1.0, 10.0, long_context=TIER._replace(inclusive=True))
        assert price_call(p, 99, 0)[0] == pytest.approx(99e-6, rel=1e-12)
        assert price_call(p, 100, 0)[0] == pytest.approx(200e-6, rel=1e-12)

    def test_whole_request_not_only_the_excess(self) -> None:
        p = Pricing(1.0, 10.0, cache_hit=0.5, cache_write=2.0, long_context=TIER)
        # 1000 prompt = 700 miss + 200 read + 100 write, all at the tier: 700*1*2 + 200*0.5*3 + 100*2*4 = 1400+300+800.
        # Excess-only pricing would give a different figure; the whole request is multiplied.
        assert price_call(p, 1000, 0, cache_read=200, cache_write=100)[0] == pytest.approx(2500e-6, rel=1e-12)

    def test_cache_reads_and_writes_are_clamped_to_the_prompt(self) -> None:
        p = Pricing(1.0, 10.0, cache_hit=0.5, cache_write=2.0)
        # read 10 (clamped from 50) * 0.5 = 5, nothing left to write or miss.
        assert price_call(p, 10, 0, cache_read=50, cache_write=50)[0] == pytest.approx(5e-6, rel=1e-12)

    def test_positional_pricing_construction_still_works(self) -> None:
        p = Pricing(1.0, 2.0, 0.1, 1.25)
        assert (p.input, p.output, p.cache_hit, p.cache_write, p.long_context) == (1.0, 2.0, 0.1, 1.25, None)
        assert Pricing(3.0, 9.0, None) == Pricing(3.0, 9.0)


# ─── xAI ───────────────────────────────────────────────────────────────────────────────────────────


def _xai(model: str = "grok-4.3") -> Any:
    from pyutilz.llm.xai_provider import XAIProvider

    return XAIProvider(api_key="xai-fake", model=model)  # pragma: allowlist secret -- dummy key


def _xai_usage(prompt: int, hit: int = 0, completion: int = 0, reasoning: int = 0) -> dict[str, Any]:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "prompt_tokens_details": {"cached_tokens": hit},
        "completion_tokens_details": {"reasoning_tokens": reasoning},
    }


class TestXAITier:
    # grok-4.3: input 1.25, cached 0.20, output 2.50; output billed = completion + reasoning.

    def test_just_below_the_threshold(self) -> None:
        p = _xai()
        p._record_usage(_xai_usage(199_999, completion=1000))
        cost = p.get_session_cost()
        # 199,999 * 1.25 = 249,998.75 + 1000 * 2.5 = 2,500 -> 0.25249875
        assert cost["total_cost_usd"] == pytest.approx(0.25249875, rel=1e-12)
        assert cost["long_context_calls"] == 0 and cost["long_context_surcharge_usd"] == 0.0

    def test_at_the_threshold_the_whole_request_doubles(self) -> None:
        p = _xai()
        p._record_usage(_xai_usage(200_000, hit=50_000, completion=1000, reasoning=500))
        cost = p.get_session_cost()
        # short: 150,000 * 1.25 + 50,000 * 0.20 + 1,500 * 2.5 = 187,500 + 10,000 + 3,750 = 201,250 -> doubled 0.4025
        assert cost["total_cost_usd"] == pytest.approx(0.4025, rel=1e-12)
        assert cost["long_context_surcharge_usd"] == pytest.approx(0.20125, rel=1e-12)
        assert cost["long_context_calls"] == 1

    def test_session_is_the_sum_of_per_call_costs_not_the_pooled_totals(self) -> None:
        p = _xai()
        p._record_usage(_xai_usage(199_999, completion=1000))
        p._record_usage(_xai_usage(200_000, hit=50_000, completion=1000, reasoning=500))
        total = p.get_session_cost()["total_cost_usd"]
        assert total == pytest.approx(0.25249875 + 0.4025, rel=1e-12)
        # Pooled totals (399,999 prompt) priced as one long request would double both calls: 2 * 0.45374875.
        assert total != pytest.approx(0.9074975, rel=1e-6)

    def test_estimate_cost_applies_the_tier(self) -> None:
        p = _xai()
        assert p.estimate_cost(199_999, 0) == pytest.approx(0.24999875, rel=1e-12)
        # (200,000 * 1.25 + 1,000 * 2.5) * 2 = 0.505
        assert p.estimate_cost(200_000, 1000) == pytest.approx(0.505, rel=1e-12)


def _old_xai_surcharge(pricing: Pricing, usage: dict[str, Any]) -> float:
    """``XAIProvider._track_provider_specific_usage``'s tier arithmetic before the migration (423fdc4), verbatim."""
    prompt = int(usage.get("prompt_tokens") or 0)
    if prompt < 200_000:
        return 0.0
    hit = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    details = usage.get("completion_tokens_details") or {}
    output = int(usage.get("completion_tokens") or 0) + int(details.get("reasoning_tokens") or 0)
    cache_rate = pricing.cache_hit if pricing.cache_hit is not None else pricing.input
    return ((prompt - hit) * pricing.input + hit * cache_rate + output * pricing.output) / 1_000_000


@pytest.mark.parametrize("model", ["grok-4.3", "grok-4.5", "grok-4.7", "grok-4.20-0309-reasoning", "grok-4-1-fast-reasoning"])
@pytest.mark.parametrize(
    ("prompt", "hit", "completion", "reasoning"),
    [(0, 0, 0, 0), (199_999, 10_000, 500, 0), (200_000, 0, 0, 0), (200_000, 150_000, 3, 900), (731_337, 1, 12_345, 6_789), (1_000_000, 999_999, 1, 1)],
)
def test_xai_equivalence_with_the_pre_change_surcharge(model: str, prompt: int, hit: int, completion: int, reasoning: int) -> None:
    p = _xai(model)
    usage = _xai_usage(prompt, hit, completion, reasoning)
    expected = _old_xai_surcharge(p._resolve_pricing(model), usage)
    p._record_usage(usage)
    assert p.get_session_cost()["long_context_surcharge_usd"] == pytest.approx(expected, rel=1e-12, abs=0.0)


# ─── Gemini ────────────────────────────────────────────────────────────────────────────────────────


def _gemini(model: str, *responses: Any) -> Any:
    gp = pytest.importorskip("pyutilz.llm.gemini_provider")
    pytest.importorskip("google.genai")
    p = gp.GeminiProvider(api_key="g-fake", model=model)  # pragma: allowlist secret -- dummy key
    queue = list(responses)

    async def generate_content(**kwargs: Any) -> Any:
        return queue.pop(0)

    p.client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))
    p.fit_max_tokens_to_context = lambda mt, prompt, system=None: mt
    return p


def _gem(prompt: int, output: int = 100, cached: int = 0, thoughts: int = 0) -> Any:
    cand = SimpleNamespace(finish_reason="STOP", safety_ratings=[], grounding_metadata=None, citation_metadata=None, content=None)
    um = SimpleNamespace(prompt_token_count=prompt, candidates_token_count=output, thoughts_token_count=thoughts, cached_content_token_count=cached)
    return SimpleNamespace(candidates=[cand], text="ok", usage_metadata=um, prompt_feedback=None)


async def _run(p: Any, n: int) -> None:
    for _ in range(n):
        await p.generate.__wrapped__(p, "q", max_tokens=100)


class TestGeminiTier:
    # gemini-2.5-pro: input 1.25, cached 0.125, output 10; tier >200K: x2 / x2 / x1.5.

    @pytest.mark.asyncio
    async def test_at_200k_exactly_is_still_the_short_tier(self) -> None:
        p = _gemini("gemini-2.5-pro", _gem(200_000))
        await _run(p, 1)
        # 200,000 * 1.25 = 250,000 + 100 * 10 = 1,000 -> 0.251
        assert p.get_session_cost()["total_cost_usd"] == pytest.approx(0.251, rel=1e-12)
        assert p.get_session_cost()["long_context_calls"] == 0

    @pytest.mark.asyncio
    async def test_just_above_200k_the_whole_request_is_long(self) -> None:
        p = _gemini("gemini-2.5-pro", _gem(200_001, cached=1))
        await _run(p, 1)
        # 200,000 * 2.5 = 500,000 + 1 * 0.25 + 100 * 15 = 1,500 -> 0.50150025
        cost = p.get_session_cost()
        assert cost["total_cost_usd"] == pytest.approx(0.50150025, rel=1e-12)
        assert cost["long_context_calls"] == 1

    @pytest.mark.asyncio
    async def test_session_is_the_sum_of_per_call_costs(self) -> None:
        p = _gemini("gemini-2.5-pro", _gem(200_000), _gem(200_001, cached=1))
        await _run(p, 2)
        cost = p.get_session_cost()
        assert cost["total_cost_usd"] == pytest.approx(0.251 + 0.50150025, rel=1e-12)
        assert cost["long_context_calls"] == 1

    @pytest.mark.asyncio
    async def test_flash_has_no_tier(self) -> None:
        # The old prefix lookup trimmed gemini-2.5-pro to gemini-2.5 and surcharged 2.5 Flash too.
        p = _gemini("gemini-2.5-flash", _gem(300_000, output=3))
        await _run(p, 1)
        # 300,000 * 0.30 = 90,000 + 3 * 2.50 = 7.5 -> 0.0900075
        assert p.get_session_cost()["total_cost_usd"] == pytest.approx(0.0900075, rel=1e-12)

    def test_estimate_cost_applies_the_tier(self) -> None:
        gp = pytest.importorskip("pyutilz.llm.gemini_provider")
        p = gp.GeminiProvider.__new__(gp.GeminiProvider)
        p.model_name = "gemini-2.5-pro"
        assert p.estimate_cost(200_000, 100) == pytest.approx(0.251, rel=1e-12)
        # 200,001 * 2.5 = 500,002.5 + 100 * 15 = 1,500 -> 0.5015025
        assert p.estimate_cost(200_001, 100) == pytest.approx(0.5015025, rel=1e-12)
        p.model_name = "gemini-3.1-pro-preview-06-05"
        # A versioned 3.1 Pro id keeps its tier: 300,000 * 4 = 1.2
        assert p.estimate_cost(300_000, 0) == pytest.approx(1.2, rel=1e-12)


def _old_gemini_surcharge(model: str, prompt_tokens: int, cached: int, output_tokens: int) -> float:
    """``GeminiProvider._add_long_context_surcharge`` before the migration (423fdc4), on its two Pro rows."""
    gp = pytest.importorskip("pyutilz.llm.gemini_provider")
    mult = {"gemini-2.5-pro": (2.0, 1.5, 2.0), "gemini-3.1-pro-preview": (2.0, 1.5, 2.0)}[model]
    if prompt_tokens <= 200_000:
        return 0.0
    in_rate, out_rate = gp.GeminiProvider._PRICING[model]
    cache_rate = gp.GeminiProvider._CACHE_HIT_COST[model]
    cached = min(cached, prompt_tokens)
    return ((prompt_tokens - cached) * in_rate * (mult[0] - 1) + cached * cache_rate * (mult[2] - 1) + output_tokens * out_rate * (mult[1] - 1)) / 1_000_000


@pytest.mark.parametrize("model", ["gemini-2.5-pro", "gemini-3.1-pro-preview"])
@pytest.mark.parametrize(
    ("prompt", "cached", "output"),
    [(0, 0, 0), (200_000, 5, 77), (200_001, 0, 0), (250_000, 100_000, 4_321), (987_654, 987_654, 1), (1_048_576, 2_000_000, 65_536)],
)
def test_gemini_equivalence_with_the_pre_change_surcharge(model: str, prompt: int, cached: int, output: int) -> None:
    gp = pytest.importorskip("pyutilz.llm.gemini_provider")
    p = gp.GeminiProvider.__new__(gp.GeminiProvider)
    p.model_name = model
    p._long_context_surcharge_usd = 0.0
    p._long_context_calls = 0
    p._add_long_context_surcharge(prompt, cached, output)
    assert p._long_context_surcharge_usd == pytest.approx(_old_gemini_surcharge(model, prompt, cached, output), rel=1e-12, abs=0.0)


# ─── OpenAI ────────────────────────────────────────────────────────────────────────────────────────


def _openai(model: str) -> Any:
    from pyutilz.llm.openai_provider import OpenAIProvider

    return OpenAIProvider(api_key="sk-fake", model=model)  # pragma: allowlist secret -- dummy key


def _oa_usage(prompt: int, completion: int = 0, cached: int = 0, write: Optional[int] = None, responses_write: Optional[int] = None) -> dict[str, Any]:
    details: dict[str, Any] = {"cached_tokens": cached}
    if write is not None:
        details["cache_write_tokens"] = write
    usage: dict[str, Any] = {"prompt_tokens": prompt, "completion_tokens": completion, "prompt_tokens_details": details}
    if responses_write is not None:
        usage["input_tokens_details"] = {"cache_write_tokens": responses_write}
    return usage


class TestOpenAITier:
    # gpt-5.4: input 2.50, cached 0.25, output 15; tier >272K: input/cached x2, output x1.5.

    def test_at_272k_exactly_is_the_short_tier(self) -> None:
        p = _openai("gpt-5.4")
        p._record_usage(_oa_usage(272_000, completion=1000))
        # 272,000 * 2.5 = 680,000 + 1,000 * 15 = 15,000 -> 0.695
        assert p.get_session_cost()["total_cost_usd"] == pytest.approx(0.695, rel=1e-12)

    def test_just_above_272k_the_whole_request_is_long(self) -> None:
        p = _openai("gpt-5.4")
        p._record_usage(_oa_usage(272_001, completion=1000, cached=72_001))
        # 200,000 * 5 = 1,000,000 + 72,001 * 0.5 = 36,000.5 + 1,000 * 22.5 = 22,500 -> 1.0585005
        cost = p.get_session_cost()
        assert cost["total_cost_usd"] == pytest.approx(1.0585005, rel=1e-12)
        assert cost["long_context_calls"] == 1

    def test_session_is_the_sum_of_per_call_costs(self) -> None:
        p = _openai("gpt-5.4")
        p._record_usage(_oa_usage(272_000, completion=1000))
        p._record_usage(_oa_usage(272_001, completion=1000, cached=72_001))
        assert p.get_session_cost()["total_cost_usd"] == pytest.approx(0.695 + 1.0585005, rel=1e-12)

    @pytest.mark.parametrize("model", ["gpt-5.4-mini", "gpt-5.5-pro", "gpt-5.2", "gpt-4.1"])
    def test_models_without_a_long_column_get_no_tier(self, model: str) -> None:
        p = _openai(model)
        p._record_usage(_oa_usage(300_000))
        cost = p.get_session_cost()
        assert cost["long_context_calls"] == 0 and cost["long_context_surcharge_usd"] == 0.0

    def test_dated_snapshot_keeps_the_tier(self) -> None:
        p = _openai("gpt-5.5-2026-04-01")
        # 300,000 * 5 * 2 = 3.0
        assert p.estimate_cost(300_000, 0) == pytest.approx(3.0, rel=1e-12)

    def test_estimate_cost_applies_the_tier(self) -> None:
        p = _openai("gpt-5.4")
        assert p.estimate_cost(272_000, 1000) == pytest.approx(0.695, rel=1e-12)
        # 300,000 * 5 = 1.5 + 1,000 * 22.5 = 0.0225 -> 1.5225
        assert p.estimate_cost(300_000, 1000) == pytest.approx(1.5225, rel=1e-12)


class TestOpenAICacheWrites:
    # gpt-6-sol: input 2.00, cached 0.20, cache write 2.50 (1.25x), output 10.

    def test_chat_completions_write_field_is_billed_at_the_write_rate(self) -> None:
        p = _openai("gpt-6-sol")
        p._record_usage(_oa_usage(100_000, cached=10_000, write=40_000))
        cost = p.get_session_cost()
        # miss 50,000 * 2 = 100,000 + read 10,000 * 0.2 = 2,000 + write 40,000 * 2.5 = 100,000 -> 0.202
        assert cost["input_cost_usd"] == pytest.approx(0.202, rel=1e-12)
        assert cost["cache_miss_tokens"] == 50_000

    def test_responses_write_field_is_read_too(self) -> None:
        p = _openai("gpt-6-sol")
        p._record_usage(_oa_usage(100_000, cached=10_000, responses_write=40_000))
        assert p.get_session_cost()["input_cost_usd"] == pytest.approx(0.202, rel=1e-12)

    def test_writes_inside_a_long_request_take_the_tier_too(self) -> None:
        p = _openai("gpt-6-sol")
        p._record_usage(_oa_usage(300_000, completion=2000, write=100_000))
        # miss 200,000 * 4 = 0.8 + write 100,000 * 5 = 0.5 + output 2,000 * 15 = 0.03 -> 1.33
        assert p.get_session_cost()["total_cost_usd"] == pytest.approx(1.33, rel=1e-12)

    def test_pre_5_6_models_bill_writes_at_the_input_rate(self) -> None:
        p = _openai("gpt-5.5")
        assert p._resolve_pricing("gpt-5.5").cache_write is None
        p._record_usage(_oa_usage(1_000_000, write=1_000_000))
        # 1,000,000 written tokens at the plain $5 input rate.
        assert p.get_session_cost()["input_cost_usd"] == pytest.approx(5.0, rel=1e-12)

    @pytest.mark.parametrize(
        ("model", "write"),
        [("gpt-6-astra", 12.50), ("gpt-6.1-sol", 2.50), ("gpt-6-luna", 0.125), ("gpt-5.6-sol", 5.00), ("gpt-5.6-terra", 2.50), ("gpt-5.6-luna", 0.25)],
    )
    def test_write_rates_are_1_25x_input(self, model: str, write: float) -> None:
        pricing = _openai(model)._resolve_pricing(model)
        assert pricing.cache_write == write == pytest.approx(1.25 * pricing.input, rel=1e-12)


# ─── Anthropic, DeepSeek, OpenRouter ───────────────────────────────────────────────────────────────


def _old_anthropic_cost(spec: Any, input_tokens: int, output_tokens: int, w5: int, w1h: int, read: int) -> tuple[float, float]:
    """``AnthropicProvider._cost_usd`` before the migration (423fdc4), verbatim."""
    in_rate, out_rate = spec.input_per_1m, spec.output_per_1m
    input_cost = (input_tokens * in_rate + w5 * in_rate * 1.25 + w1h * in_rate * 2.0 + read * in_rate * spec.cache_read_multiplier) / 1_000_000
    return input_cost, output_tokens * out_rate / 1_000_000


class TestAnthropicCacheWrites:
    def _p(self, model: str) -> Any:
        from pyutilz.llm.anthropic_provider import AnthropicProvider

        p = AnthropicProvider.__new__(AnthropicProvider)
        p.model = model
        return p

    def test_hand_computed_opus_5_5(self) -> None:
        # input 4, output 20, reads 0.05x: 1000*4 + 2000*5 + 3000*8 + 4000*0.2 = 38,800; output 500 * 20 = 10,000.
        assert self._p("claude-opus-5-5")._cost_usd(1000, 500, 2000, 3000, 4000) == pytest.approx((0.0388, 0.01), rel=1e-12)

    @pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5", "claude-fable-5-1"])
    @pytest.mark.parametrize("counts", [(0, 0, 0, 0, 0), (1, 1, 1, 1, 1), (123_456, 7_890, 50_000, 0, 900_000), (10, 0, 0, 2_000_000, 3)])
    def test_equivalence_with_the_pre_change_cost(self, model: str, counts: tuple[int, int, int, int, int]) -> None:
        p = self._p(model)
        assert p._cost_usd(*counts) == pytest.approx(_old_anthropic_cost(p._spec, *counts), rel=1e-12, abs=0.0)


def test_deepseek_has_no_tier() -> None:
    from pyutilz.llm.deepseek_provider import DeepSeekProvider

    p = DeepSeekProvider(api_key="sk-fake", model="deepseek-v4-flash")  # pragma: allowlist secret -- dummy key
    assert p._resolve_pricing("deepseek-v4-flash").long_context is None
    p._record_usage(_oa_usage(900_000))
    cost = p.get_session_cost()
    assert cost["long_context_calls"] == 0 and cost["long_context_surcharge_usd"] == 0.0


def test_openrouter_counts_cache_writes_once_and_never_resolves_pricing_per_call() -> None:
    from unittest.mock import MagicMock, patch

    from pyutilz.llm.openrouter_provider import OpenRouterProvider

    settings = MagicMock()
    settings.openrouter_api_key = None
    with patch("pyutilz.llm.openrouter_provider.get_llm_settings", return_value=settings):
        p = OpenRouterProvider(api_key="k", model="anthropic/claude-x")
    with patch.object(OpenRouterProvider, "_resolve_pricing", side_effect=AssertionError("catalogue lookup per call")):
        p._record_usage({"prompt_tokens": 500_000, "completion_tokens": 1, "prompt_tokens_details": {"cache_write_tokens": 40}})
        p._record_usage({"prompt_tokens": 10, "completion_tokens": 1, "prompt_tokens_details": {"cache_write_tokens": 2}})
    assert p.total_cache_write_tokens == 42
    assert p._long_context_calls == 0

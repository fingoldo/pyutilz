"""OpenAI LLM provider via the official OpenAI-compatible chat/completions API."""

import logging

import httpx

from pyutilz.llm.config import get_llm_settings
from pyutilz.llm.openai_compat import OpenAICompatibleProvider

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (USD): (input, output).
# Source: https://platform.openai.com/docs/pricing + OpenRouter cross-check.
# Verified 2026-05-01.
_PRICING: dict[str, tuple[float, float]] = {
    # GPT-5 family — flagship 2026 lineup.
    "gpt-5":          (1.25, 10.00),
    "gpt-5-pro":      (15.00, 120.00),
    "gpt-5-chat":     (1.25, 10.00),
    "gpt-5-mini":     (0.25, 2.00),
    "gpt-5-nano":     (0.05, 0.40),
    # GPT-5.5 family (premium tier, higher cost than GPT-5).
    "gpt-5.5":        (5.00, 30.00),
    "gpt-5.5-pro":    (30.00, 180.00),
    # GPT-4.1 family (intermediate; cheaper alternatives to 4o).
    "gpt-4.1":        (2.00, 8.00),
    "gpt-4.1-mini":   (0.40, 1.60),
    "gpt-4.1-nano":   (0.10, 0.40),
    # GPT-4o family.
    "gpt-4o":         (2.50, 10.00),
    "gpt-4o-mini":    (0.15, 0.60),
    # o-series reasoning models. o1 deprecated for new traffic but
    # included for cost calculation on legacy logs.
    "o1":             (15.00, 60.00),
    "o1-pro":         (150.00, 600.00),
    "o3":             (2.00, 8.00),
    "o3-mini":        (1.10, 4.40),
    "o4-mini":        (1.10, 4.40),
    # Specialised: agentic coding (Codex-class).
    "gpt-5-codex":    (1.25, 10.00),
    "gpt-5.1-codex":  (1.25, 10.00),
}

# Cached input prices per 1M tokens (~50% off input typically; verify per
# model in the official pricing page since OpenAI's discount varies by
# model family). NOT 90% like Anthropic — OpenAI's cache is shallower.
_CACHE_HIT_COST: dict[str, float] = {
    "gpt-5":         0.125,
    "gpt-5-pro":     1.50,
    "gpt-5-chat":    0.125,
    "gpt-5-mini":    0.025,
    "gpt-5-nano":    0.005,
    "gpt-5.5":       0.50,
    "gpt-5.5-pro":   3.00,
    "gpt-4.1":       0.50,
    "gpt-4.1-mini":  0.10,
    "gpt-4.1-nano":  0.025,
    "gpt-4o":        1.25,
    "gpt-4o-mini":   0.075,
    "o1":            7.50,
    "o3":            0.50,
    "o3-mini":       0.55,
    "o4-mini":       0.275,
}

_MAX_TOKENS: dict[str, int] = {
    "gpt-5":         128_000,
    "gpt-5-pro":     128_000,
    "gpt-5-chat":    128_000,
    "gpt-5-mini":    128_000,
    "gpt-5-nano":    128_000,
    "gpt-5.5":       128_000,
    "gpt-5.5-pro":   128_000,
    "gpt-4.1":       32_000,
    "gpt-4.1-mini":  32_000,
    "gpt-4.1-nano":  32_000,
    "gpt-4o":        16_384,
    "gpt-4o-mini":   16_384,
    "o1":            100_000,
    "o1-pro":        100_000,
    "o3":            100_000,
    "o3-mini":       100_000,
    "o4-mini":       100_000,
}

_CONTEXT_WINDOW: dict[str, int] = {
    "gpt-5":         400_000,
    "gpt-5-pro":     400_000,
    "gpt-5-chat":    400_000,
    "gpt-5-mini":    400_000,
    "gpt-5-nano":    400_000,
    "gpt-5.5":       400_000,
    "gpt-5.5-pro":   400_000,
    "gpt-4.1":       1_000_000,
    "gpt-4.1-mini":  1_000_000,
    "gpt-4.1-nano":  1_000_000,
    "gpt-4o":        128_000,
    "gpt-4o-mini":   128_000,
    "o1":            200_000,
    "o1-pro":        200_000,
    "o3":            200_000,
    "o3-mini":       200_000,
    "o4-mini":       200_000,
}


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI provider via the native OpenAI chat/completions API.

    Inherits the OpenAI-compatible base implementation since the public
    OpenAI HTTP API IS the canonical OpenAI-compatible API. Subclassing
    is mostly for default base_url and pricing tables; request-level
    semantics are identical to the base.
    """

    _base_url = "https://api.openai.com/v1"
    _provider_name = "OpenAI"
    _max_tokens_map = _MAX_TOKENS
    _default_max_tokens = 16_384
    _context_window_map = _CONTEXT_WINDOW
    _default_context_window = 128_000

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5-mini",
        max_concurrent: int = 10,
    ):
        settings = get_llm_settings()
        resolved_key = api_key or (
            settings.openai_api_key.get_secret_value()
            if settings.openai_api_key
            else None
        )
        if not resolved_key:
            raise ValueError(
                "OpenAI API key not provided. "
                "Set OPENAI_API_KEY in .env or pass api_key="
            )
        super().__init__(api_key=resolved_key, model=model, max_concurrent=max_concurrent)

    def _get_timeout(self, model: str) -> float:
        # Reasoning models (o-series, gpt-5-pro / 5.5-pro) need long timeouts.
        if model.startswith(("o1", "o3", "o4")):
            return 1200.0
        if "pro" in model:
            return 600.0
        return 240.0

    def _handle_special_status(self, resp: httpx.Response) -> None:
        if resp.status_code == 429:
            logger.warning(
                "OpenAI rate limit hit (HTTP 429). Retrying via tenacity loop."
            )

    def _compute_billed_output(
        self, completion_tokens: int, reasoning_tokens: int
    ) -> int:
        # OpenAI bills reasoning tokens as part of output for o-series
        # (verified in API response payload's ``usage.completion_tokens``
        # which already includes reasoning_tokens). Return completion
        # tokens as-is to avoid double-counting.
        return completion_tokens

    async def get_account_credits(self) -> dict:
        # OpenAI dropped the only "remaining balance" endpoint
        # (/v1/dashboard/billing/credit_grants) for regular user keys. The
        # modern Usage API reports SPEND under an admin/org-key, not the
        # remaining credit. Long-standing community feature request without
        # an ETA.
        raise NotImplementedError(
            "OpenAI has no public API to fetch remaining credit. "
            "Check platform.openai.com/usage or platform.openai.com/account/billing/overview."
        )

    async def check_account_limits(self) -> dict:
        # Per-key rate limits are returned in ``x-ratelimit-*`` response
        # headers on real calls; no standalone endpoint exists.
        raise NotImplementedError(
            "OpenAI does not expose per-key rate limits via API. "
            "Inspect ``x-ratelimit-*`` headers on any real response, "
            "or check platform.openai.com/account/limits."
        )

    def _input_cost_per_1m(self, model: str) -> float:
        return _PRICING.get(model, (1.25, 10.00))[0]

    def _output_cost_per_1m(self, model: str) -> float:
        return _PRICING.get(model, (1.25, 10.00))[1]

    def _cache_hit_cost_per_1m(self, model: str) -> float:
        return _CACHE_HIT_COST.get(model, 0.125)

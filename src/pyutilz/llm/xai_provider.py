"""xAI (Grok) LLM provider using OpenAI-compatible chat/completions API."""

import logging

from pyutilz.llm.config import get_llm_settings
from pyutilz.llm.openai_compat import OpenAICompatibleProvider

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (USD): (input, output)
# Source: https://docs.x.ai/developers/models, OpenRouter, official xAI announcements
# Updated 2026-04-28:
#   - Fixed prior 1000x error on grok-4.20 family ($2000/$6000 → $2/$6)
#   - Added grok-4.20-beta (general flagship alias)
#   - Added grok-4 (current alias for grok-4-0709)
#   - Confirmed grok-4.20 cache-hit = $0.20/M (90% discount on input miss)
_MAX_TOKENS: dict[str, int] = {
    "grok-4.20-beta": 30000,
    "grok-4.20-multi-agent-beta-0309": 30000,
    "grok-4.20-beta-0309-reasoning": 30000,
    "grok-4.20-beta-0309-non-reasoning": 30000,
    "grok-4-1-fast-reasoning": 30000,
    "grok-4-1-fast-non-reasoning": 30000,
    "grok-4-fast-reasoning": 30000,
    "grok-4-fast-non-reasoning": 30000,
    "grok-4": 30000,
    "grok-4-0709": 30000,
    "grok-code-fast-1": 30000,
}

_PRICING: dict[str, tuple[float, float]] = {
    # grok-4.20 family — flagship, $2/$6 per 1M tokens, 2M context
    "grok-4.20-beta": (2.00, 6.00),
    "grok-4.20-multi-agent-beta-0309": (2.00, 6.00),
    "grok-4.20-beta-0309-reasoning": (2.00, 6.00),
    "grok-4.20-beta-0309-non-reasoning": (2.00, 6.00),
    # grok-4 fast family — cheapest, 2M context
    "grok-4-1-fast-reasoning": (0.20, 0.50),
    "grok-4-1-fast-non-reasoning": (0.20, 0.50),
    "grok-4-fast-reasoning": (0.20, 0.50),
    "grok-4-fast-non-reasoning": (0.20, 0.50),
    # grok-4 premium reasoning, 256K context
    "grok-4": (3.00, 15.00),
    "grok-4-0709": (3.00, 15.00),
    # Legacy Grok 3
    "grok-3": (3.00, 15.00),
    "grok-3-mini": (0.30, 0.50),
    # Coding-specialized
    "grok-code-fast-1": (0.20, 1.50),
}

_CACHE_HIT_COST: dict[str, float] = {
    "grok-4.20-beta": 0.20,
    "grok-4.20-multi-agent-beta-0309": 0.20,
    "grok-4.20-beta-0309-reasoning": 0.20,
    "grok-4.20-beta-0309-non-reasoning": 0.20,
    "grok-4-1-fast-reasoning": 0.05,
    "grok-4-1-fast-non-reasoning": 0.05,
    "grok-4-fast-reasoning": 0.05,
    "grok-4-fast-non-reasoning": 0.05,
    "grok-4": 0.75,
    "grok-4-0709": 0.75,
    "grok-3": 0.75,
    "grok-3-mini": 0.07,
    "grok-code-fast-1": 0.02,
}


class XAIProvider(OpenAICompatibleProvider):
    """xAI Grok provider via OpenAI-compatible chat/completions API."""

    _base_url = "https://api.x.ai/v1"
    _provider_name = "xAI"
    _max_tokens_map = _MAX_TOKENS
    _default_max_tokens = 30000
    _context_window_map = {
        "grok-4": 256_000,
        "grok-4-0709": 256_000,
        "grok-3": 131_000,
        "grok-3-mini": 131_000,
    }
    _default_context_window = 2_000_000

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "grok-4-1-fast-reasoning",
        max_concurrent: int = 10,
    ):
        settings = get_llm_settings()
        resolved_key = api_key or (
            settings.xai_api_key.get_secret_value()
            if settings.xai_api_key
            else None
        )
        if not resolved_key:
            raise ValueError(
                "xAI API key not provided. "
                "Set XAI_API_KEY in .env or pass api_key="
            )
        super().__init__(api_key=resolved_key, model=model, max_concurrent=max_concurrent)

    def _get_timeout(self, model: str) -> float:
        # Reasoning-mode variants need long timeout (chain-of-thought can be slow).
        # Note: substring check would mistakenly match "non-reasoning" to reasoning,
        # so explicitly exclude that suffix first.
        if model.endswith("-non-reasoning"):
            return 240.0
        if "reasoning" in model or model in ("grok-4", "grok-4-0709",
                                              "grok-4.20-beta"):
            return 1200.0
        return 240.0

    def _compute_billed_output(
        self, completion_tokens: int, reasoning_tokens: int
    ) -> int:
        return completion_tokens + reasoning_tokens

    def _input_cost_per_1m(self, model: str) -> float:
        return _PRICING.get(model, (0.20, 0.50))[0]

    def _output_cost_per_1m(self, model: str) -> float:
        return _PRICING.get(model, (0.20, 0.50))[1]

    def _cache_hit_cost_per_1m(self, model: str) -> float:
        return _CACHE_HIT_COST.get(model, 0.05)

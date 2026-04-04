"""xAI (Grok) LLM provider using OpenAI-compatible chat/completions API."""

import logging

from pyutilz.llm.config import get_llm_settings
from pyutilz.llm.openai_compat import OpenAICompatibleProvider

logger = logging.getLogger(__name__)

_MAX_TOKENS: dict[str, int] = {
    "grok-4-1-fast-reasoning": 30000,
    "grok-4-1-fast-non-reasoning": 30000,
    "grok-4-fast-reasoning": 30000,
    "grok-4-fast-non-reasoning": 30000,
    "grok-4-0709": 30000,
    "grok-code-fast-1": 30000,
}

_PRICING: dict[str, tuple[float, float]] = {
    "grok-4.20-multi-agent-beta-0309": (2000.00, 6000.00),
    "grok-4.20-beta-0309-reasoning": (2000.00, 6000.00),
    "grok-4.20-beta-0309-non-reasoning": (2000.00, 6000.00),
    "grok-4-1-fast-reasoning": (0.20, 0.50),
    "grok-4-1-fast-non-reasoning": (0.20, 0.50),
    "grok-4-fast-reasoning": (0.20, 0.50),
    "grok-4-fast-non-reasoning": (0.20, 0.50),
    "grok-4-0709": (3.00, 15.00),
    "grok-3": (3.00, 15.00),
    "grok-3-mini": (0.30, 0.50),
    "grok-code-fast-1": (0.20, 1.50),
}

_CACHE_HIT_COST: dict[str, float] = {
    "grok-4.20-multi-agent-beta-0309": 200.00,
    "grok-4.20-beta-0309-reasoning": 200.00,
    "grok-4.20-beta-0309-non-reasoning": 200.00,
    "grok-4-1-fast-reasoning": 0.05,
    "grok-4-1-fast-non-reasoning": 0.05,
    "grok-4-fast-reasoning": 0.05,
    "grok-4-fast-non-reasoning": 0.05,
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
        "grok-4-0709": 256_000,
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
        return 1200.0 if "reasoning" in model else 240.0

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

"""DeepSeek LLM provider using the OpenAI-compatible API."""

import logging

import httpx

from pyutilz.llm.config import get_llm_settings
from pyutilz.llm.openai_compat import OpenAICompatibleProvider

logger = logging.getLogger(__name__)

# Pricing per 1M tokens: (input_cache_miss, input_cache_hit, output)
_PRICING = {
    "deepseek-chat": (0.28, 0.028, 0.42),
    "deepseek-reasoner": (0.28, 0.028, 0.42),
}

_MAX_TOKENS = {
    "deepseek-chat": 8192,
    "deepseek-reasoner": 65536,
}


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek provider via OpenAI-compatible chat/completions API."""

    _base_url = "https://api.deepseek.com"
    _provider_name = "DeepSeek"
    _max_tokens_map = _MAX_TOKENS
    _default_max_tokens = 8192
    _context_window_map = {
        "deepseek-reasoner": 128_000,
    }
    _default_context_window = 64_000

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-reasoner",
        max_concurrent: int = 10,
    ):
        settings = get_llm_settings()
        resolved_key = api_key or (
            settings.deepseek_api_key.get_secret_value()
            if settings.deepseek_api_key
            else None
        )
        if not resolved_key:
            raise ValueError(
                "DeepSeek API key not provided. "
                "Set DEEPSEEK_API_KEY in .env or pass api_key="
            )
        super().__init__(api_key=resolved_key, model=model, max_concurrent=max_concurrent)

    def _get_timeout(self, model: str) -> float:
        return 300.0 if "reasoner" in model else 120.0

    def _handle_special_status(self, resp: httpx.Response) -> None:
        if resp.status_code == 402:
            logger.warning(
                "DeepSeek account has insufficient balance (HTTP 402). "
                "Top up at https://platform.deepseek.com/top_up — "
                "retrying indefinitely until balance is restored..."
            )

    def _compute_billed_output(
        self, completion_tokens: int, reasoning_tokens: int
    ) -> int:
        return completion_tokens

    def _input_cost_per_1m(self, model: str) -> float:
        return _PRICING.get(model, _PRICING["deepseek-chat"])[0]

    def _output_cost_per_1m(self, model: str) -> float:
        return _PRICING.get(model, _PRICING["deepseek-chat"])[2]

    def _cache_hit_cost_per_1m(self, model: str) -> float:
        return _PRICING.get(model, _PRICING["deepseek-chat"])[1]

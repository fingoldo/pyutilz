"""DeepSeek LLM provider using the OpenAI-compatible API."""

import logging

import httpx

from pyutilz.llm.config import get_llm_settings
from pyutilz.llm.openai_compat import OpenAICompatibleProvider

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (USD): (input_cache_miss, input_cache_hit, output)
# Source: https://api-docs.deepseek.com/quick_start/pricing
# Updated 2026-04-28:
#   - V4 models (deepseek-v4-flash, deepseek-v4-pro) launched
#   - Cache-hit rates reduced to 1/10 of launch price (effective 2026-04-26)
#   - Legacy aliases deepseek-chat / deepseek-reasoner deprecated 2026-07-24
_PRICING = {
    # New V4 family -- recommended
    "deepseek-v4-flash": (0.14, 0.0028, 0.28),
    "deepseek-v4-pro": (1.74, 0.0145, 3.48),
    # Legacy aliases (deprecated 2026-07-24, still functional, V3.2-backed)
    "deepseek-chat": (0.28, 0.028, 0.42),
    "deepseek-reasoner": (0.28, 0.028, 0.42),
}

_MAX_TOKENS = {
    "deepseek-v4-flash": 384_000,
    "deepseek-v4-pro": 384_000,
    "deepseek-chat": 8192,
    "deepseek-reasoner": 65536,
}

_CONTEXT_WINDOW = {
    "deepseek-v4-flash": 1_000_000,
    "deepseek-v4-pro": 1_000_000,
    "deepseek-reasoner": 128_000,
    "deepseek-chat": 64_000,
}


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek provider via OpenAI-compatible chat/completions API."""

    _base_url = "https://api.deepseek.com"
    _provider_name = "DeepSeek"
    _max_tokens_map = _MAX_TOKENS
    _default_max_tokens = 8192
    _context_window_map = _CONTEXT_WINDOW
    _default_context_window = 64_000

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-v4-flash",
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
        # V4 models support thinking via parameter; reasoner is the legacy thinking alias
        return 300.0 if "reasoner" in model or "pro" in model else 120.0

    def _handle_special_status(self, resp: httpx.Response) -> None:
        if resp.status_code == 402:
            logger.warning(
                "DeepSeek account has insufficient balance (HTTP 402). "
                "Top up at https://platform.deepseek.com/top_up -- "
                "retrying indefinitely until balance is restored..."
            )

    # _compute_billed_output not overridden — base default returns
    # completion_tokens which matches DeepSeek's billing semantics
    # (reasoning tokens are already included in completion_tokens).

    # NOTE: thinking mode is intentionally LEFT ENABLED by default for V4.
    # Callers who need non-thinking mode (e.g. tight max_tokens budget on
    # structured JSON output) should pass ``thinking=False`` to ``generate()``
    # or use the legacy ``deepseek-chat`` alias which routes to non-thinking
    # server-side. See DeepSeek docs:
    # https://api-docs.deepseek.com/api/create-chat-completion

    def _thinking_request_field(self, thinking: bool | str) -> dict | None:
        # Only V4 models support this toggle; legacy aliases (chat/reasoner)
        # are fixed-mode server-side and reject the field. Log a warning
        # so a caller passing thinking= to a legacy alias notices the
        # request goes through unchanged rather than silently ignored.
        if not self.model_name.startswith("deepseek-v4"):
            if thinking:
                logger.warning(
                    "DeepSeek %r does not support the thinking toggle "
                    "(only deepseek-v4-* models do); thinking=%r ignored.",
                    self.model_name, thinking,
                )
            return None
        # DeepSeek V4 expects a hard on/off, not an effort string.
        # Coerce: ``True`` or any non-empty effort string -> enabled;
        # ``False`` / empty string -> disabled.
        enabled, _effort = self._normalize_thinking(thinking)
        return {"thinking": {"type": "enabled" if enabled else "disabled"}}

    async def get_account_credits(self) -> dict:
        """Query DeepSeek's ``/user/balance`` endpoint.

        DeepSeek splits balance into ``granted_balance`` (free credits with
        expiry — spent first) and ``topped_up_balance`` (paid credits).
        ``total_balance`` is their sum and is what's actually available.

        Returns a dict with normalized keys plus the raw payload:
            ``balance_usd``     — total available, in USD (or native currency)
            ``total_granted``   — granted (promo / free) portion
            ``total_topped_up`` — paid portion
            ``currency``        — e.g. ``"USD"`` or ``"CNY"``
            ``is_available``    — boolean flag from the API
            ``raw``             — full response (multi-currency-aware)
        """
        resp = await self._client.get("/user/balance")
        resp.raise_for_status()
        data = resp.json()
        infos = data.get("balance_infos") or []
        # Pick the USD entry first; else first non-empty entry; else empty.
        primary = next(
            (i for i in infos if str(i.get("currency", "")).upper() == "USD"),
            infos[0] if infos else {},
        )

        def _to_float(v) -> float | None:
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        return {
            "balance_usd": _to_float(primary.get("total_balance")),
            "total_granted": _to_float(primary.get("granted_balance")),
            "total_topped_up": _to_float(primary.get("topped_up_balance")),
            "currency": primary.get("currency"),
            "is_available": data.get("is_available"),
            "raw": data,
        }

    def _resolve_pricing(self, model: str) -> tuple[float, float, float]:
        """Look up (input, cache_hit, output) per-1M USD rates for ``model``.

        Falls back to ``deepseek-v4-flash`` pricing on miss, with a single
        warning per unknown model name (logged once via the cache itself
        as a side-effect) so callers don't get silently mis-priced. A
        typo like ``"deepseekv4"`` would otherwise estimate cost using
        flash rates without any signal.
        """
        if model not in _PRICING:
            self._warn_unknown_model_once(model)
            return _PRICING["deepseek-v4-flash"]
        return _PRICING[model]

    _seen_unknown_models: set[str] = set()

    def _warn_unknown_model_once(self, model: str) -> None:
        if model in DeepSeekProvider._seen_unknown_models:
            return
        DeepSeekProvider._seen_unknown_models.add(model)
        logger.warning(
            "DeepSeek pricing for %r is unknown; falling back to "
            "deepseek-v4-flash rates. Cost estimates may be off.",
            model,
        )

    def _input_cost_per_1m(self, model: str) -> float:
        return self._resolve_pricing(model)[0]

    def _output_cost_per_1m(self, model: str) -> float:
        return self._resolve_pricing(model)[2]

    def _cache_hit_cost_per_1m(self, model: str) -> float:
        return self._resolve_pricing(model)[1]

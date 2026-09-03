"""OpenAI LLM provider via the official OpenAI-compatible chat/completions API."""

from __future__ import annotations

import logging

import httpx

from pyutilz.llm.base import longest_prefix_lookup
from pyutilz.llm.config import get_llm_settings
from pyutilz.llm.openai_compat import OpenAICompatibleProvider

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (USD): (input, output).
# Source: https://platform.openai.com/docs/pricing + OpenRouter cross-check.
# Verified 2026-05-01.
_PRICING: dict[str, tuple[float, float]] = {
    # GPT-5 family — flagship 2026 lineup.
    "gpt-5": (1.25, 10.00),
    "gpt-5-pro": (15.00, 120.00),
    "gpt-5-chat": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    # GPT-5.5 family (premium tier, higher cost than GPT-5).
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.5-pro": (30.00, 180.00),
    # GPT-4.1 family (intermediate; cheaper alternatives to 4o).
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    # GPT-4o family.
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    # o-series reasoning models. o1 deprecated for new traffic but
    # included for cost calculation on legacy logs.
    "o1": (15.00, 60.00),
    "o1-pro": (150.00, 600.00),
    "o3": (2.00, 8.00),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
    # Specialised: agentic coding (Codex-class).
    "gpt-5-codex": (1.25, 10.00),
    "gpt-5.1-codex": (1.25, 10.00),
}

# Cached input prices per 1M tokens (~50% off input typically; verify per
# model in the official pricing page since OpenAI's discount varies by
# model family). NOT 90% like Anthropic — OpenAI's cache is shallower.
_CACHE_HIT_COST: dict[str, float] = {
    "gpt-5": 0.125,
    "gpt-5-pro": 1.50,
    "gpt-5-chat": 0.125,
    "gpt-5-mini": 0.025,
    "gpt-5-nano": 0.005,
    "gpt-5.5": 0.50,
    "gpt-5.5-pro": 3.00,
    "gpt-4.1": 0.50,
    "gpt-4.1-mini": 0.10,
    "gpt-4.1-nano": 0.025,
    "gpt-4o": 1.25,
    "gpt-4o-mini": 0.075,
    "o1": 7.50,
    "o3": 0.50,
    "o3-mini": 0.55,
    "o4-mini": 0.275,
    "o1-pro": 75.00,
    "gpt-5-codex": 0.125,
    "gpt-5.1-codex": 0.125,
}

_MAX_TOKENS: dict[str, int] = {
    "gpt-5": 128_000,
    "gpt-5-pro": 128_000,
    "gpt-5-chat": 128_000,
    "gpt-5-mini": 128_000,
    "gpt-5-nano": 128_000,
    "gpt-5.5": 128_000,
    "gpt-5.5-pro": 128_000,
    "gpt-4.1": 32_000,
    "gpt-4.1-mini": 32_000,
    "gpt-4.1-nano": 32_000,
    "gpt-4o": 16_384,
    "gpt-4o-mini": 16_384,
    "o1": 100_000,
    "o1-pro": 100_000,
    "o3": 100_000,
    "o3-mini": 100_000,
    "o4-mini": 100_000,
    "gpt-5-codex": 128_000,
    "gpt-5.1-codex": 128_000,
}

_CONTEXT_WINDOW: dict[str, int] = {
    "gpt-5": 400_000,
    "gpt-5-pro": 400_000,
    "gpt-5-chat": 400_000,
    "gpt-5-mini": 400_000,
    "gpt-5-nano": 400_000,
    "gpt-5.5": 400_000,
    "gpt-5.5-pro": 400_000,
    "gpt-4.1": 1_000_000,
    "gpt-4.1-mini": 1_000_000,
    "gpt-4.1-nano": 1_000_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "o1": 200_000,
    "o1-pro": 200_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    "gpt-5-codex": 400_000,
    "gpt-5.1-codex": 400_000,
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
        resolved_key = api_key or (settings.openai_api_key.get_secret_value() if settings.openai_api_key else None)
        if not resolved_key:
            raise ValueError("OpenAI API key not provided. " "Set OPENAI_API_KEY in .env or pass api_key=")
        super().__init__(api_key=resolved_key, model=model, max_concurrent=max_concurrent)

    def _get_timeout(self, model: str) -> float:
        """Return the request timeout (seconds) for a given model, scaled up for slow reasoning models."""
        # Reasoning models (o-series, gpt-5-pro / 5.5-pro) need long timeouts.
        if model.startswith(("o1", "o3", "o4")):
            return 1200.0
        if "pro" in model:
            return 600.0
        return 240.0

    def _handle_special_status(self, resp: httpx.Response) -> None:
        """Log a warning when the response signals an OpenAI rate limit (HTTP 429)."""
        if resp.status_code == 429:
            logger.warning("OpenAI rate limit hit (HTTP 429). Retrying via tenacity loop.")

    def _compute_billed_output(self, completion_tokens: int, reasoning_tokens: int) -> int:
        """Return the output token count OpenAI actually bills for (completion_tokens already includes reasoning_tokens)."""
        # OpenAI bills reasoning tokens as part of output for o-series
        # (verified in API response payload's ``usage.completion_tokens``
        # which already includes reasoning_tokens). Return completion
        # tokens as-is to avoid double-counting.
        return completion_tokens

    async def get_account_credits(self) -> dict:
        """Always raise: OpenAI has no public API endpoint to fetch remaining account credit."""
        # OpenAI dropped the only "remaining balance" endpoint
        # (/v1/dashboard/billing/credit_grants) for regular user keys. The
        # modern Usage API reports SPEND under an admin/org-key, not the
        # remaining credit. Long-standing community feature request without
        # an ETA.
        raise NotImplementedError(
            "OpenAI has no public API to fetch remaining credit. " "Check platform.openai.com/usage or platform.openai.com/account/billing/overview."
        )

    async def check_account_limits(self) -> dict:
        """Always raise: OpenAI does not expose per-key rate limits via a standalone API endpoint."""
        # Per-key rate limits are returned in ``x-ratelimit-*`` response
        # headers on real calls; no standalone endpoint exists.
        raise NotImplementedError(
            "OpenAI does not expose per-key rate limits via API. "
            "Inspect ``x-ratelimit-*`` headers on any real response, "
            "or check platform.openai.com/account/limits."
        )

    _seen_unknown_models: set[str] = set()  # noqa: RUF012 -- intentional shared class-level dedupe set (warn once per model name, across all instances), not a per-instance mutable-default bug

    def _warn_unknown_model_once(self, model: str) -> None:
        """Log a one-time warning that pricing for `model` is unknown and gpt-5-mini rates are used as a fallback."""
        if model in OpenAIProvider._seen_unknown_models:
            return
        OpenAIProvider._seen_unknown_models.add(model)
        logger.warning(
            "OpenAI pricing for %r is unknown; falling back to " "gpt-5-mini rates. Cost estimates may be off.",
            model,
        )

    def _resolve_pricing(self, model: str) -> tuple[float, float]:
        """Return ``(input, output)`` USD per 1M for ``model``, longest-prefix resolved.

        Prefix-matched for the same reason ``max_output_tokens``/``context_window`` are: a dated
        snapshot id such as ``gpt-5-pro-2026-01-15`` used to miss the exact ``dict.get`` here and
        be priced from ``gpt-5-mini`` while resolving its LIMITS correctly by prefix -- the two
        lookups disagreed about the same id, understating that model's spend ~60x behind a single
        WARNING line (2026-09-03 audit F20). The warning still fires for a genuine miss.
        """
        exact = _PRICING.get(model)
        if exact is not None:
            return exact
        resolved = longest_prefix_lookup(model, _PRICING, None)
        self._warn_unknown_model_once(model)
        if resolved is None:
            return _PRICING["gpt-5-mini"]
        return resolved  # type: ignore[no-any-return]  # longest_prefix_lookup is typed Any; the table's value type is the tuple returned here

    def _input_cost_per_1m(self, model: str) -> float:
        """Return USD cost per 1M input tokens for `model`, warning and falling back to gpt-5-mini rates if unknown."""
        return self._resolve_pricing(model)[0]

    def _output_cost_per_1m(self, model: str) -> float:
        """Return USD cost per 1M output tokens for `model`, warning and falling back to gpt-5-mini rates if unknown."""
        return self._resolve_pricing(model)[1]

    @property
    def max_output_tokens(self) -> int:
        """Max output tokens, resolved by longest matching prefix over ``_MAX_TOKENS``.

        A dated snapshot id (``gpt-5-mini-2026-01-01``) used to miss the exact-match lookup and
        fall back to the 16,384 class default -- an 8x-too-small budget that made
        fit_max_tokens_to_context truncate requests that would have fit. Safe for this table
        specifically: every family prefix here shares one budget across its variants.
        """
        return int(longest_prefix_lookup(self.model_name, self._max_tokens_map, self._default_max_tokens))

    @property
    def context_window(self) -> int:
        """Context window, longest-prefix resolved for the same dated-snapshot reason as :attr:`max_output_tokens`."""
        return int(longest_prefix_lookup(self.model_name, self._context_window_map, self._default_context_window))

    def _cache_hit_cost_per_1m(self, model: str) -> float:
        """Return USD cost per 1M cache-hit input tokens for `model`, warning and falling back to gpt-5-mini rates if unknown."""
        exact = _CACHE_HIT_COST.get(model)
        if exact is not None:
            return exact
        self._warn_unknown_model_once(model)
        # Prefix-resolved for the same reason as the base rates above (audit F20).
        return float(longest_prefix_lookup(model, _CACHE_HIT_COST, _CACHE_HIT_COST["gpt-5-mini"]))

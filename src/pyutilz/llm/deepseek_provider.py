"""DeepSeek LLM provider using the OpenAI-compatible API."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from pyutilz.llm.config import get_llm_settings
from pyutilz.llm.exceptions import LLMProviderError
from pyutilz.llm.openai_compat import OpenAICompatibleProvider, Pricing

logger = logging.getLogger(__name__)

# PEAK pricing per 1M tokens (USD): (input_cache_miss, input_cache_hit, output).
# Source: https://api-docs.deepseek.com/quick_start/pricing (fetched 2026-09-26).
# Off-peak is half of every figure; see `deepseek_price_multiplier`.
_PRICING = {
    # DeepSeek-V4.1-Flash, the current Flash model.
    "deepseek-flash": (0.30, 0.006, 1.20),
    # Legacy name, "still accepted ... served by the DeepSeek-V4.1-Flash model and billed at the Flash price".
    "deepseek-v4-flash": (0.30, 0.006, 1.20),
    "deepseek-v4-pro": (1.32, 0.044, 3.96),
    # Legacy aliases (deprecated 2026-07-24, V3.2-backed); no longer on the pricing page, last published rates.
    "deepseek-chat": (0.28, 0.028, 0.42),
    "deepseek-reasoner": (0.28, 0.028, 0.42),
}

_MAX_TOKENS = {
    "deepseek-flash": 384_000,
    "deepseek-v4-flash": 384_000,
    "deepseek-v4-pro": 384_000,
    "deepseek-chat": 8192,
    "deepseek-reasoner": 65536,
}

_CONTEXT_WINDOW = {
    "deepseek-flash": 1_000_000,
    "deepseek-v4-flash": 1_000_000,
    "deepseek-v4-pro": 1_000_000,
    "deepseek-reasoner": 128_000,
    "deepseek-chat": 64_000,
}

# The fixed-mode legacy aliases: every other model (the V4 family, deepseek-flash) takes the `thinking` toggle.
_LEGACY_FIXED_MODE_MODELS = frozenset({"deepseek-chat", "deepseek-reasoner"})

# "Peak hours are 01:00 - 04:00 and 06:00 - 10:00 UTC, Monday through Friday, excluding Chinese public holidays."
# Holidays are not modelled: a call on one is costed at the peak rate, an overestimate, never an underestimate.
_PEAK_HOURS_UTC = frozenset({1, 2, 3, 6, 7, 8, 9})


def deepseek_price_multiplier(when: datetime | None = None) -> float:
    """1.0 during DeepSeek's peak window, 0.5 off-peak, for a call completing at ``when`` (default now, UTC)."""
    moment = (datetime.now(timezone.utc) if when is None else when).astimezone(timezone.utc)
    peak = moment.weekday() < 5 and moment.hour in _PEAK_HOURS_UTC
    return 1.0 if peak else 0.5


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek provider via OpenAI-compatible chat/completions API."""

    _base_url = "https://api.deepseek.com"
    _provider_name = "DeepSeek"
    _max_tokens_map = _MAX_TOKENS
    # An unknown model gets the CURRENT generation's limits: the old 8192 / 64K defaults were the retired
    # deepseek-chat's, and silently capped the current deepseek-flash (384K output, 1M context) to them.
    _default_max_tokens = 384_000
    _context_window_map = _CONTEXT_WINDOW
    _default_context_window = 1_000_000

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-flash",
        max_concurrent: int = 10,
        wait_on_insufficient_balance: bool = False,
    ):
        """``wait_on_insufficient_balance``: on HTTP 402 keep retrying until the account is topped up (the old
        behaviour) instead of failing the call at once. Off by default, because an unattended batch job otherwise
        hangs with nothing but a repeating warning to show for it."""
        settings = get_llm_settings()
        resolved_key = api_key or (settings.deepseek_api_key.get_secret_value() if settings.deepseek_api_key else None)
        if not resolved_key:
            raise ValueError("DeepSeek API key not provided. " "Set DEEPSEEK_API_KEY in .env or pass api_key=")
        super().__init__(api_key=resolved_key, model=model, max_concurrent=max_concurrent)
        self.wait_on_insufficient_balance = wait_on_insufficient_balance
        # USD by which the off-peak discount lowered this session's spend below the peak-rate figure the base
        # `get_session_cost` computes from token totals. Accumulated per call, at the call's own completion time.
        self._offpeak_input_discount_usd = 0.0
        self._offpeak_output_discount_usd = 0.0

    def _get_timeout(self, model: str) -> float:
        """Return the request timeout in seconds for ``model`` (longer for reasoning/"pro" models)."""
        # V4 models support thinking via parameter; reasoner is the legacy thinking alias
        return 300.0 if "reasoner" in model or "pro" in model else 120.0

    def _handle_special_status(self, resp: httpx.Response) -> None:
        """On HTTP 402 (insufficient balance): fail the call, or with ``wait_on_insufficient_balance`` warn and let the
        shared retry policy wait for a top-up.

        The shared predicate retries 402 forever, which is right for a supervised process and wrong for a batch job
        nobody is watching: it hung silently until someone topped the account up. Raising ``LLMProviderError`` here
        (not an ``httpx`` error, so no retry predicate matches it) ends the call with the reason.
        """
        if resp.status_code != 402:
            return
        if not getattr(self, "wait_on_insufficient_balance", False):
            raise LLMProviderError(
                "DeepSeek account has insufficient balance (HTTP 402). Top up at https://platform.deepseek.com/top_up, "
                "or construct DeepSeekProvider(wait_on_insufficient_balance=True) to wait for the top-up instead.",
                details={"status_code": 402},
            )
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
        """Build the request-body ``thinking`` field for V4 models, or None for legacy aliases that don't support it."""
        # Only V4 models support this toggle; legacy aliases (chat/reasoner)
        # are fixed-mode server-side and reject the field. Log a warning
        # so a caller passing thinking= to a legacy alias notices the
        # request goes through unchanged rather than silently ignored.
        if self.model_name in _LEGACY_FIXED_MODE_MODELS:
            if thinking:
                logger.warning(
                    "DeepSeek %r does not support the thinking toggle (only the legacy aliases lack it); thinking=%r ignored.",
                    self.model_name,
                    thinking,
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
            """Coerce ``v`` to float, returning None if it's missing or not coercible."""
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError) as exc:
                logger.debug("DeepSeek balance field %r not coercible to float: %s", v, exc)
                return None

        return {
            "balance_usd": _to_float(primary.get("total_balance")),
            "total_granted": _to_float(primary.get("granted_balance")),
            "total_topped_up": _to_float(primary.get("topped_up_balance")),
            "currency": primary.get("currency"),
            "is_available": data.get("is_available"),
            "raw": data,
        }

    def _resolve_pricing(self, model: str) -> Pricing:
        """Look up the :class:`Pricing` record for ``model``.

        The module's ``_PRICING`` table is stored in DeepSeek's own upstream order
        ``(input, cache_hit, output)``; it is reordered into ``Pricing``'s named fields here so no
        caller ever has to remember which position means what.

        Falls back to ``deepseek-flash`` pricing on miss, with a single
        warning per unknown model name (logged once via the cache itself
        as a side-effect) so callers don't get silently mis-priced. A
        typo like ``"deepseekv4"`` would otherwise estimate cost using
        flash rates without any signal.
        """
        row = _PRICING.get(model)
        if row is None:
            self._warn_unknown_model_once(model)
            row = _PRICING["deepseek-flash"]
        in_cost, cache_hit, out_cost = row
        return Pricing(float(in_cost), float(out_cost), float(cache_hit))

    _seen_unknown_models: set[str] = set()  # noqa: RUF012 -- intentional shared class-level dedupe set (warn once per model name, across all instances), not a per-instance mutable-default bug

    def _warn_unknown_model_once(self, model: str) -> None:
        """Log a one-time warning (per model name, across instances) that pricing is unknown and flash rates are used."""
        if model in DeepSeekProvider._seen_unknown_models:
            return
        DeepSeekProvider._seen_unknown_models.add(model)
        logger.warning(
            "DeepSeek pricing for %r is unknown; falling back to deepseek-flash rates. Cost estimates may be off.",
            model,
        )

    def _input_cost_per_1m(self, model: str) -> float:
        """Return the USD cost per 1M input tokens (cache miss) for ``model``."""
        return self._resolve_pricing(model).input

    def _output_cost_per_1m(self, model: str) -> float:
        """Return the USD cost per 1M output tokens for ``model``."""
        return self._resolve_pricing(model).output

    def _track_provider_specific_usage(self, usage: dict[str, Any]) -> None:
        """Record how much this call's off-peak discount takes off the peak-rate cost ``get_session_cost`` computes.

        DeepSeek halves every rate outside its peak window, so pricing a session from its token totals alone
        overstated off-peak spend twofold. The rate is a property of each call's time, so it is applied per call.
        """
        super()._track_provider_specific_usage(usage)
        discount = 1.0 - deepseek_price_multiplier()
        if discount <= 0:
            return
        pricing = self._resolve_pricing(self.model_name)
        prompt = int(usage.get("prompt_tokens") or 0)
        details = usage.get("prompt_tokens_details")
        raw_hit = usage.get("prompt_cache_hit_tokens")
        if raw_hit is None:
            raw_hit = (details or {}).get("cached_tokens")
        hit = int(raw_hit) if isinstance(raw_hit, (int, float)) else 0
        completion = int(usage.get("completion_tokens") or 0)
        cache_rate = pricing.cache_hit if pricing.cache_hit is not None else pricing.input
        self._offpeak_input_discount_usd = (
            getattr(self, "_offpeak_input_discount_usd", 0.0) + discount * ((prompt - hit) * pricing.input + hit * cache_rate) / 1_000_000
        )
        self._offpeak_output_discount_usd = getattr(self, "_offpeak_output_discount_usd", 0.0) + discount * completion * pricing.output / 1_000_000

    def get_session_cost(self) -> dict[str, Any]:
        """Session cost at the rate each call was actually billed at (peak or off-peak), see ``_track_provider_specific_usage``."""
        cost = super().get_session_cost()
        in_discount = getattr(self, "_offpeak_input_discount_usd", 0.0)
        out_discount = getattr(self, "_offpeak_output_discount_usd", 0.0)
        cost["input_cost_usd"] -= in_discount
        cost["output_cost_usd"] -= out_discount
        cost["total_cost_usd"] = cost["input_cost_usd"] + cost["output_cost_usd"]
        cost["offpeak_discount_usd"] = in_discount + out_discount
        return cost

"""OpenAI LLM provider via the official OpenAI-compatible chat/completions API."""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from pyutilz.llm.base import longest_prefix_lookup, normalize_thinking
from pyutilz.llm.config import get_llm_settings
from pyutilz.llm.openai_compat import OpenAICompatibleProvider, Pricing

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (USD): (input, output), standard tier.
# Source: https://developers.openai.com/api/docs/pricing (fetched 2026-09-26).
_PRICING: dict[str, tuple[float, float]] = {
    # GPT-6 family (reasoning; 1.05M context, 128K output per https://developers.openai.com/api/docs/models).
    "gpt-6-astra": (10.00, 50.00),
    "gpt-6-sol": (2.00, 10.00),
    "gpt-6-luna": (0.10, 0.50),
    # GPT-5.6 family.
    "gpt-5.6-sol": (4.00, 20.00),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-luna": (0.20, 1.20),
    # GPT-5.5 family (premium tier, higher cost than GPT-5).
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.5-pro": (30.00, 180.00),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.4-pro": (30.00, 180.00),
    "gpt-5.2": (1.75, 14.00),
    "gpt-5.2-pro": (21.00, 168.00),
    "gpt-5.1": (1.25, 10.00),
    # GPT-5 family.
    "gpt-5": (1.25, 10.00),
    "gpt-5-pro": (15.00, 120.00),
    "gpt-5-chat": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
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
    "o3-pro": (20.00, 80.00),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
    # Specialised: agentic coding (Codex-class).
    "gpt-5-codex": (1.25, 10.00),
    "gpt-5.1-codex": (1.25, 10.00),
}

# Cached input prices per 1M tokens, from the same pricing page. The -pro models list no cache discount, so their
# cached rate is recorded as the full input price: a cached token is never priced below what it is billed at.
_CACHE_HIT_COST: dict[str, float] = {
    "gpt-6-astra": 1.00,
    "gpt-6-sol": 0.20,
    "gpt-6-luna": 0.01,
    "gpt-5.6-sol": 0.40,
    "gpt-5.6-terra": 0.20,
    "gpt-5.6-luna": 0.02,
    "gpt-5.5": 0.50,
    "gpt-5.5-pro": 30.00,
    "gpt-5.4": 0.25,
    "gpt-5.4-mini": 0.075,
    "gpt-5.4-nano": 0.02,
    "gpt-5.4-pro": 30.00,
    "gpt-5.2": 0.175,
    "gpt-5.2-pro": 21.00,
    "gpt-5.1": 0.125,
    "gpt-5": 0.125,
    "gpt-5-pro": 15.00,
    "gpt-5-chat": 0.125,
    "gpt-5-mini": 0.025,
    "gpt-5-nano": 0.005,
    "gpt-4.1": 0.50,
    "gpt-4.1-mini": 0.10,
    "gpt-4.1-nano": 0.025,
    "gpt-4o": 1.25,
    "gpt-4o-mini": 0.075,
    "o1": 7.50,
    "o3": 0.50,
    "o3-pro": 20.00,
    "o3-mini": 0.55,
    "o4-mini": 0.275,
    "o1-pro": 150.00,
    "gpt-5-codex": 0.125,
    "gpt-5.1-codex": 0.125,
}

# Output limits. GPT-6: 128K (models page). The GPT-5.x rows keep the GPT-5 family's 128K, which the models page
# states for GPT-6 only; a request above a model's real cap is answered with a 400 naming the cap.
_MAX_TOKENS: dict[str, int] = {
    "gpt-6-astra": 128_000,
    "gpt-6-sol": 128_000,
    "gpt-6-luna": 128_000,
    "gpt-5.6-sol": 128_000,
    "gpt-5.6-terra": 128_000,
    "gpt-5.6-luna": 128_000,
    "gpt-5.5": 128_000,
    "gpt-5.5-pro": 128_000,
    "gpt-5.4": 128_000,
    "gpt-5.4-mini": 128_000,
    "gpt-5.4-nano": 128_000,
    "gpt-5.4-pro": 128_000,
    "gpt-5.2": 128_000,
    "gpt-5.2-pro": 128_000,
    "gpt-5.1": 128_000,
    "gpt-5": 128_000,
    "gpt-5-pro": 128_000,
    "gpt-5-chat": 128_000,
    "gpt-5-mini": 128_000,
    "gpt-5-nano": 128_000,
    "gpt-4.1": 32_000,
    "gpt-4.1-mini": 32_000,
    "gpt-4.1-nano": 32_000,
    "gpt-4o": 16_384,
    "gpt-4o-mini": 16_384,
    "o1": 100_000,
    "o1-pro": 100_000,
    "o3": 100_000,
    "o3-pro": 100_000,
    "o3-mini": 100_000,
    "o4-mini": 100_000,
    "gpt-5-codex": 128_000,
    "gpt-5.1-codex": 128_000,
}

_CONTEXT_WINDOW: dict[str, int] = {
    "gpt-6-astra": 1_050_000,
    "gpt-6-sol": 1_050_000,
    "gpt-6-luna": 1_050_000,
    "gpt-5.6-sol": 400_000,
    "gpt-5.6-terra": 400_000,
    "gpt-5.6-luna": 400_000,
    "gpt-5.5": 400_000,
    "gpt-5.5-pro": 400_000,
    "gpt-5.4": 400_000,
    "gpt-5.4-mini": 400_000,
    "gpt-5.4-nano": 400_000,
    "gpt-5.4-pro": 400_000,
    "gpt-5.2": 400_000,
    "gpt-5.2-pro": 400_000,
    "gpt-5.1": 400_000,
    "gpt-5": 400_000,
    "gpt-5-pro": 400_000,
    "gpt-5-chat": 400_000,
    "gpt-5-mini": 400_000,
    "gpt-5-nano": 400_000,
    "gpt-4.1": 1_000_000,
    "gpt-4.1-mini": 1_000_000,
    "gpt-4.1-nano": 1_000_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "o1": 200_000,
    "o1-pro": 200_000,
    "o3": 200_000,
    "o3-pro": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    "gpt-5-codex": 400_000,
    "gpt-5.1-codex": 400_000,
}

# Reasoning families: they take `reasoning_effort` and `max_completion_tokens`, and reject a non-default
# `temperature` ("Only the default (1) value is supported"). `gpt-5-chat` is the non-reasoning chat snapshot.
_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5", "gpt-6")
_NON_REASONING_PREFIXES = ("gpt-5-chat",)

# The lowest effort each family accepts, which is what `thinking=False` becomes: reasoning cannot be switched off on
# the o-series or GPT-6 Astra (`none` is a 400 there), GPT-5 offers `minimal`, and GPT-5.1 onward `none`
# (https://developers.openai.com/api/docs/guides/reasoning). Longest prefix wins.
_LOWEST_EFFORT: dict[str, str] = {
    "o1": "low",
    "o3": "low",
    "o4": "low",
    "gpt-5": "minimal",
    "gpt-5.1": "none",
    "gpt-5.2": "none",
    "gpt-5.4": "none",
    "gpt-5.5": "none",
    "gpt-5.6": "none",
    "gpt-6": "none",
    "gpt-6-astra": "low",
}

# A dated snapshot (`-2026-01-15`, `-20260115`) or `-latest` of a known id: the same model, priced as its base row.
_SNAPSHOT_SUFFIX = re.compile(r"-(?:\d{4}-\d{2}-\d{2}|\d{8}|latest)$")

# Models whose API answered a `temperature` with a 400, learned at runtime (see `_body_after_rejected_request`).
_MODELS_REJECTING_TEMPERATURE: set[str] = set()


def is_reasoning_model(model: str) -> bool:
    """Is ``model`` one of OpenAI's reasoning families (o-series, GPT-5 and later, except the chat snapshot)?"""
    return model.startswith(_REASONING_PREFIXES) and not model.startswith(_NON_REASONING_PREFIXES)


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

    def _openai_body(self, body: dict[str, Any]) -> dict[str, Any]:
        """The chat/completions body in the shape OpenAI's current models take.

        ``max_tokens`` is deprecated for ``max_completion_tokens`` and "not compatible with o-series models", which
        answered it with a 400 (a non-retryable ``LLMProviderError``): the shared body builder writes ``max_tokens``
        for every OpenAI-compatible upstream, so the rename happens here. Reasoning models also reject a non-default
        ``temperature``, which this library defaults to 0.7, so it is not sent to them.
        """
        out = dict(body)
        if "max_tokens" in out:
            out["max_completion_tokens"] = out.pop("max_tokens")
        if "temperature" in out and (is_reasoning_model(self.model_name) or self.model_name in _MODELS_REJECTING_TEMPERATURE):
            out.pop("temperature")
        return out

    async def _post_and_unwrap(self, body: dict[str, Any], repairing: bool = False) -> str | None:
        """The shared POST, with the body reshaped for OpenAI first (see ``_openai_body``)."""
        return await super()._post_and_unwrap(self._openai_body(body), repairing=repairing)

    def _build_stream_body(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """The shared streaming body, reshaped for OpenAI (see ``_openai_body``)."""
        return self._openai_body(super()._build_stream_body(*args, **kwargs))

    def _body_after_rejected_request(self, body: dict[str, Any], status: int, detail: str) -> dict[str, Any] | None:
        """Repair a 400 over ``temperature`` (learned per model) or over a ``reasoning_effort`` value the model lacks."""
        if status != 400:
            return None
        text = detail.lower()
        if "temperature" in body and "temperature" in text:
            _MODELS_REJECTING_TEMPERATURE.add(self.model_name)
            return {k: v for k, v in body.items() if k != "temperature"}
        if "reasoning_effort" in body and "reasoning_effort" in text:
            return {k: v for k, v in body.items() if k != "reasoning_effort"}
        return None

    def _thinking_request_field(self, thinking: bool | str) -> dict[str, Any] | None:
        """``{"reasoning_effort": ...}`` for a reasoning model, None otherwise.

        The base returned None for OpenAI, so ``thinking=`` was silently ignored here although the base docstring
        promised ``reasoning_effort``. ``True`` asks for ``medium``; off asks for the family's lowest effort
        (``_LOWEST_EFFORT``), since reasoning cannot be disabled on every family. A non-reasoning model has no such
        control: a request for reasoning there is warned about and not sent.
        """
        enabled, effort = normalize_thinking(thinking)
        if not is_reasoning_model(self.model_name):
            if enabled:
                logger.warning("OpenAI %r is not a reasoning model; thinking=%r is not sent.", self.model_name, thinking)
            return None
        if not enabled:
            return {"reasoning_effort": str(longest_prefix_lookup(self.model_name, _LOWEST_EFFORT, "low"))}
        return {"reasoning_effort": "medium" if effort is None else effort}

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

    _seen_unknown_models: set[str] = set()  # noqa: RUF012 -- intentional shared class-level dedupe set (warn once per model name, across all instances), not a per-instance mutable-default bug

    def _warn_unknown_model_once(self, model: str, priced_as: str = "gpt-5-mini") -> None:
        """Log a one-time warning that pricing for `model` is unknown and which row's rates it is priced at instead."""
        if model in OpenAIProvider._seen_unknown_models:
            return
        OpenAIProvider._seen_unknown_models.add(model)
        logger.warning(
            "OpenAI pricing for %r is unknown; pricing it at %s rates. Cost estimates may be off.",
            model,
            priced_as,
        )

    @staticmethod
    def _known_row(model: str, table: dict[str, Any]) -> str | None:
        """The table key ``model`` IS, exactly or as a dated/``-latest`` snapshot of it; None for anything else.

        The distinction the unknown-model warning needs: ``gpt-5-pro-2026-01-15`` is gpt-5-pro and prices correctly
        without comment, while ``gpt-5-typo`` only shares a family prefix and deserves the warning.
        """
        if model in table:
            return model
        match = _SNAPSHOT_SUFFIX.search(model)
        base = model[: match.start()] if match else None
        return base if base in table else None

    def _resolve_pricing(self, model: str) -> Pricing:
        """Return the :class:`Pricing` record (input, output) USD per 1M for ``model``, longest-prefix resolved.

        Prefix-matched for the same reason ``max_output_tokens``/``context_window`` are: a dated
        snapshot id such as ``gpt-5-pro-2026-01-15`` used to miss the exact ``dict.get`` here and
        be priced from ``gpt-5-mini`` while resolving its LIMITS correctly by prefix -- the two
        lookups disagreed about the same id, understating that model's spend ~60x behind a single
        WARNING line (2026-09-03 audit F20). A snapshot of a known row prices silently; any other
        prefix match, or a genuine miss, warns once and names the row actually used.
        """
        row = self._known_row(model, _PRICING)
        if row is not None:
            pair = _PRICING[row]
        else:
            pair = longest_prefix_lookup(model, _PRICING, None)
            self._warn_unknown_model_once(model, "its family's" if pair is not None else "gpt-5-mini")
            if pair is None:
                pair = _PRICING["gpt-5-mini"]
        return Pricing(float(pair[0]), float(pair[1]))

    def _input_cost_per_1m(self, model: str) -> float:
        """Return USD cost per 1M input tokens for `model`, warning and falling back to gpt-5-mini rates if unknown."""
        return self._resolve_pricing(model).input

    def _output_cost_per_1m(self, model: str) -> float:
        """Return USD cost per 1M output tokens for `model`, warning and falling back to gpt-5-mini rates if unknown."""
        return self._resolve_pricing(model).output

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
        """Return USD cost per 1M cache-hit input tokens for `model`, resolved (and warned about) like ``_resolve_pricing``."""
        row = self._known_row(model, _CACHE_HIT_COST)
        if row is not None:
            return _CACHE_HIT_COST[row]
        resolved = longest_prefix_lookup(model, _CACHE_HIT_COST, None)
        self._warn_unknown_model_once(model, "its family's" if resolved is not None else "gpt-5-mini")
        return float(resolved) if resolved is not None else _CACHE_HIT_COST["gpt-5-mini"]

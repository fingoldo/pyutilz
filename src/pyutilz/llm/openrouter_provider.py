"""OpenRouter LLM provider — meta-provider exposing 200+ models behind one API.

OpenRouter is OpenAI-compatible at the wire level (same /chat/completions
schema), so almost everything inherits from ``OpenAICompatibleProvider``.
What's distinctive about a meta-provider, and how this class handles it:

1. **Authoritative per-call cost via ``usage.cost``** — OpenRouter returns
   the actual USD billed by the upstream provider in every response. We
   track it as ground truth (``total_actual_cost_usd``) instead of relying
   on a hand-maintained per-model pricing table that would be wrong the
   moment OR reroutes to a different backend. The ``usage:{include:true}``
   request flag is a no-op (deprecated, auto-included now).

2. **Lazy ``/api/v1/models`` lookup for ``estimate_cost()``** — predictions
   before the call still need a per-token rate. We fetch the public model
   catalogue once per process, cache it, and look up the active model.
   On any failure (network, schema drift) we degrade silently to zeros.

3. **Routing knobs as hashable kwargs** — ``provider_order``,
   ``provider_sort``, ``provider_allow_fallbacks``, ``provider_ignore``,
   ``models_fallback`` are forwarded to the request body as ``provider``
   and ``models`` fields. Kept as tuples (not lists/dicts) so the factory's
   ``tuple(sorted(kwargs.items()))`` cache key keeps working.

4. **App-attribution headers** — ``app_name`` and ``site_url`` set
   ``X-Title`` and ``HTTP-Referer`` so calls show up in the openrouter.ai
   public rankings dashboard (optional).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import httpx

from pyutilz.llm.config import get_llm_settings
from pyutilz.llm.openai_compat import OpenAICompatibleProvider

logger = logging.getLogger(__name__)

# Process-wide cache of /api/v1/models — pricing is published per token,
# not per request, so a single fetch covers every model on the catalogue.
# Refresh would require a process restart; OR pricing changes are rare
# enough that a long-lived cache is the right tradeoff.
_MODELS_CATALOGUE: dict[str, dict[str, Any]] | None = None
_MODELS_LOCK = threading.Lock()
_MODELS_URL = "https://openrouter.ai/api/v1/models"


def _fetch_models_catalogue(timeout: float = 10.0) -> dict[str, dict[str, Any]]:
    """Fetch and parse the public /api/v1/models catalogue.

    Returns a dict mapping ``model_id`` to its raw entry. Empty dict on any
    failure — callers must treat missing entries as "unknown pricing" and
    fall back to zero cost rather than crashing the request flow.
    """
    global _MODELS_CATALOGUE
    if _MODELS_CATALOGUE is not None:
        return _MODELS_CATALOGUE
    with _MODELS_LOCK:
        if _MODELS_CATALOGUE is not None:
            return _MODELS_CATALOGUE
        try:
            resp = httpx.get(_MODELS_URL, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
            entries = payload.get("data") or payload.get("models") or []
            _MODELS_CATALOGUE = {
                e["id"]: e for e in entries if isinstance(e, dict) and "id" in e
            }
        except Exception as exc:
            logger.warning(
                "OpenRouter /models catalogue fetch failed (%s); "
                "estimate_cost() will return 0 until restart.", exc,
            )
            _MODELS_CATALOGUE = {}
        return _MODELS_CATALOGUE


def list_openrouter_models(
    refresh: bool = False,
    *,
    sort_by: str | None = None,
    name_contains: str | None = None,
    max_input_per_1m: float | None = None,
) -> list[dict[str, Any]]:
    """Return OpenRouter's full model catalogue with pricing and limits.

    Each entry contains (at minimum, depending on what OR publishes):
        ``id``, ``name``, ``description``, ``context_length``,
        ``pricing.prompt`` / ``pricing.completion`` (USD per token, as str),
        ``top_provider.context_length`` / ``max_completion_tokens``,
        ``architecture.modality`` / ``input_modalities`` / ``tokenizer``,
        ``supported_parameters`` (e.g. ``tools``, ``response_format``).

    Args:
        refresh: Drop the in-process cache and re-fetch from /api/v1/models.
        sort_by: Optional sort key — ``"input_price"``, ``"output_price"``,
            ``"context"``, or ``"name"``.
        name_contains: Substring filter on ``id`` (case-insensitive), e.g.
            ``"claude"`` to list only Anthropic-routed models.
        max_input_per_1m: Filter to models with input price ≤ this (USD per
            1M tokens). Useful for budget shopping.

    Returns:
        List of catalogue entries (list of dicts). Empty list on fetch failure.

    Example:
        >>> from pyutilz.llm import list_openrouter_models
        >>> models = list_openrouter_models(name_contains="claude", sort_by="input_price")
        >>> for m in models[:3]:
        ...     pricing = m.get("pricing", {})
        ...     print(m["id"], pricing.get("prompt"), pricing.get("completion"))
    """
    if refresh:
        global _MODELS_CATALOGUE
        with _MODELS_LOCK:
            _MODELS_CATALOGUE = None
    catalogue = _fetch_models_catalogue()
    rows = list(catalogue.values())

    if name_contains:
        needle = name_contains.lower()
        rows = [r for r in rows if needle in str(r.get("id", "")).lower()]

    if max_input_per_1m is not None:
        def _input_per_1m(r: dict) -> float:
            try:
                return float((r.get("pricing") or {}).get("prompt", "0") or "0") * 1_000_000
            except (TypeError, ValueError):
                return float("inf")
        rows = [r for r in rows if _input_per_1m(r) <= max_input_per_1m]

    if sort_by:
        def _key(r: dict):
            pricing = r.get("pricing") or {}
            try:
                in_price = float(pricing.get("prompt", "0") or "0")
                out_price = float(pricing.get("completion", "0") or "0")
            except (TypeError, ValueError):
                in_price = out_price = float("inf")
            ctx = r.get("context_length") or 0
            return {
                "input_price": in_price,
                "output_price": out_price,
                "context": -int(ctx),  # descending — biggest context first
                "name": str(r.get("id", "")),
            }.get(sort_by, str(r.get("id", "")))
        rows.sort(key=_key)
    return rows


def _per_token_cost_pair(model: str) -> tuple[float, float]:
    """Return (input_cost_per_1m, output_cost_per_1m) for ``model``.

    OpenRouter publishes pricing as USD-per-token strings under
    ``pricing.prompt`` and ``pricing.completion``. Multiply by 1e6 to match
    the per-1M-tokens convention used everywhere else in pyutilz.
    """
    catalogue = _fetch_models_catalogue()
    entry = catalogue.get(model)
    if not entry:
        return (0.0, 0.0)
    pricing = entry.get("pricing") or {}
    try:
        in_per_1m = float(pricing.get("prompt", "0") or "0") * 1_000_000
        out_per_1m = float(pricing.get("completion", "0") or "0") * 1_000_000
    except (TypeError, ValueError):
        return (0.0, 0.0)
    return (in_per_1m, out_per_1m)


def _resolve_model_limits(model: str) -> tuple[int | None, int | None]:
    """Return (context_length, max_completion_tokens) for ``model``.

    Prefers ``top_provider.context_length`` over the model-level
    ``context_length`` — the top-level field is the model's THEORETICAL
    max, but OR routes through one specific upstream which may cap shorter
    (e.g. a 1M-context model served by a provider that only exposes 200K).
    The upstream cap is what actually triggers ``HTTP 400: prompt too long``.

    Returns ``(None, None)`` for unknown models — callers fall back to
    the class-level defaults.
    """
    catalogue = _fetch_models_catalogue()
    entry = catalogue.get(model)
    if not entry:
        return (None, None)
    top = entry.get("top_provider") or {}
    ctx = top.get("context_length") or entry.get("context_length")
    max_out = top.get("max_completion_tokens")
    try:
        ctx_int = int(ctx) if ctx else None
    except (TypeError, ValueError):
        ctx_int = None
    try:
        max_out_int = int(max_out) if max_out else None
    except (TypeError, ValueError):
        max_out_int = None
    return (ctx_int, max_out_int)


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter meta-provider via OpenAI-compatible chat/completions API.

    Common usage:

    >>> p = OpenRouterProvider(model="anthropic/claude-sonnet-4.6")
    >>> txt = await p.generate("hello")
    >>> p.get_session_cost()["actual_cost_usd"]   # authoritative billed cost

    Routing example — prefer Anthropic, fall back to OpenAI; if both fail,
    fall back across models entirely:

    >>> p = OpenRouterProvider(
    ...     model="anthropic/claude-sonnet-4.6",
    ...     provider_order=("anthropic", "openai"),
    ...     provider_sort="throughput",
    ...     models_fallback=("openai/gpt-4o", "google/gemini-2.0-flash"),
    ... )
    """

    _base_url = "https://openrouter.ai/api/v1"
    _provider_name = "OpenRouter"
    _default_max_tokens = 8192
    _default_context_window = 128_000

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "openai/gpt-4o-mini",
        max_concurrent: int = 10,
        app_name: str | None = None,
        site_url: str | None = None,
        provider_order: tuple[str, ...] | None = None,
        provider_ignore: tuple[str, ...] | None = None,
        provider_sort: str | None = None,
        provider_allow_fallbacks: bool = True,
        models_fallback: tuple[str, ...] | None = None,
    ):
        settings = get_llm_settings()
        resolved_key = api_key or (
            settings.openrouter_api_key.get_secret_value()
            if settings.openrouter_api_key
            else None
        )
        if not resolved_key:
            raise ValueError(
                "OpenRouter API key not provided. "
                "Set OPENROUTER_API_KEY in .env or pass api_key="
            )
        super().__init__(api_key=resolved_key, model=model, max_concurrent=max_concurrent)

        if app_name:
            self._client.headers["X-Title"] = app_name
        if site_url:
            self._client.headers["HTTP-Referer"] = site_url

        self._provider_order = provider_order
        self._provider_ignore = provider_ignore
        self._provider_sort = provider_sort
        self._provider_allow_fallbacks = provider_allow_fallbacks
        self._models_fallback = models_fallback

        self.total_actual_cost_usd = 0.0
        self.last_actual_cost_usd = 0.0
        self.total_cache_write_tokens = 0

    @property
    def context_window(self) -> int:
        """Return the upstream-enforced context window for the active model.

        Prefers ``top_provider.context_length`` from the OR catalogue (what
        the routed backend actually accepts) over the model-level theoretical
        max. Falls back to ``_default_context_window`` if the catalogue is
        unreachable / model isn't listed.
        """
        ctx, _ = _resolve_model_limits(self.model_name)
        return ctx if ctx is not None else self._default_context_window

    @property
    def max_output_tokens(self) -> int:
        """Return the upstream-enforced output cap for the active model.

        Pulls ``top_provider.max_completion_tokens`` from the catalogue when
        available — ignored by some upstreams but a hard cap on others.
        Falls back to ``_default_max_tokens`` if absent.
        """
        _, max_out = _resolve_model_limits(self.model_name)
        return max_out if max_out is not None else self._default_max_tokens

    def supports_json_mode(self) -> bool:
        """Per-model JSON-mode support: consult the OR catalogue's
        ``supported_parameters`` field. Many free-tier models route to
        upstreams that DON'T accept ``response_format`` even though OR
        as a router does — sending the param silently no-ops, the
        model emits prose+JSON, and parsers fail.

        Returns True only when the catalogue explicitly lists
        ``response_format`` (or its alias ``structured_outputs``) for
        the active model. Falls back to True on catalogue-fetch failure
        — it's safer to TRY the kwarg (most upstreams accept it) than
        to silently skip on transient catalogue outages.
        """
        try:
            catalogue = _fetch_models_catalogue()
        except Exception:
            return True
        entry = catalogue.get(self.model_name)
        if not entry:
            # Unknown model — best-effort: assume support, rely on the
            # upstream to reject if it can't handle it.
            return True
        params = entry.get("supported_parameters") or []
        if isinstance(params, list):
            params_set = {str(p).lower() for p in params}
            return "response_format" in params_set or "structured_outputs" in params_set
        return True

    def _get_timeout(self, model: str) -> float:
        # Reasoning-class upstream models may need long timeouts; same heuristic
        # as the per-provider classes — if we route through OR to a known slow
        # tier (o-series, gpt-5-pro, claude-opus, deepseek-reasoner), bump it.
        slow_substrings = (
            "/o1", "/o3", "/o4", "-pro", "claude-opus",
            "deepseek-reasoner", "deepseek-v4-pro", "reasoning",
        )
        if any(s in model for s in slow_substrings):
            return 1200.0
        return 240.0

    def _extra_request_body(self, model: str) -> dict[str, Any]:
        body: dict[str, Any] = {}
        provider_field: dict[str, Any] = {}
        if self._provider_order:
            provider_field["order"] = list(self._provider_order)
        if self._provider_ignore:
            provider_field["ignore"] = list(self._provider_ignore)
        if self._provider_sort:
            provider_field["sort"] = self._provider_sort
        # Only emit allow_fallbacks when explicitly disabled; default at OR
        # is true, so omitting keeps requests minimal.
        if not self._provider_allow_fallbacks:
            provider_field["allow_fallbacks"] = False
        if provider_field:
            body["provider"] = provider_field
        if self._models_fallback:
            body["models"] = list(self._models_fallback)
        return body

    def _handle_special_status(self, resp: httpx.Response) -> None:
        if resp.status_code == 402:
            logger.warning(
                "OpenRouter account out of credits (HTTP 402). "
                "Top up at https://openrouter.ai/credits — "
                "retrying indefinitely until balance is restored..."
            )

    def _track_provider_specific_usage(self, usage: dict[str, Any]) -> None:
        """Capture OpenRouter's ``usage.cost`` (authoritative per-call USD)."""
        cost = usage.get("cost")
        if isinstance(cost, (int, float)):
            self.last_actual_cost_usd = float(cost)
            self.total_actual_cost_usd += float(cost)
        prompt_details = usage.get("prompt_tokens_details") or {}
        cache_write = prompt_details.get("cache_write_tokens", 0) or 0
        if cache_write:
            self.total_cache_write_tokens += int(cache_write)
        cached = prompt_details.get("cached_tokens", 0) or 0
        if cached and not usage.get("prompt_cache_hit_tokens"):
            self.total_cache_hit_tokens += int(cached)

    def _input_cost_per_1m(self, model: str) -> float:
        return _per_token_cost_pair(model)[0]

    def _output_cost_per_1m(self, model: str) -> float:
        return _per_token_cost_pair(model)[1]

    async def check_account_limits(self) -> dict[str, Any]:
        """Query ``/api/v1/key`` for live quota / usage state on the active key.

        Surfaces the raw fields OR returns under ``data`` (label, limit,
        limit_remaining, limit_reset, usage, usage_daily/weekly/monthly,
        byok_usage*, is_free_tier, rate_limit) plus the full payload under
        ``raw`` for forward compatibility.
        """
        resp = await self._client.get("/key")
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data", payload) if isinstance(payload, dict) else {}
        out = dict(data) if isinstance(data, dict) else {}
        out["raw"] = data
        return out

    async def get_account_credits(self) -> dict[str, Any]:
        """Query ``/api/v1/credits`` and normalize to the base schema.

        Returns:
            ``balance_usd``  — remaining credits (total_credits - total_usage)
            ``total_granted`` — total credits ever loaded (USD)
            ``total_used``    — lifetime spend (USD)
            ``currency``      — always "USD" for OpenRouter
            ``is_available``  — True iff balance > 0
            ``raw``           — provider's full response under ``data``
        """
        resp = await self._client.get("/credits")
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data", payload) if isinstance(payload, dict) else {}

        def _to_float(v) -> float | None:
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        granted = _to_float(data.get("total_credits"))
        used = _to_float(data.get("total_usage"))
        balance = (granted - used) if (granted is not None and used is not None) else None

        return {
            "balance_usd": balance,
            "total_granted": granted,
            "total_used": used,
            "currency": "USD",
            "is_available": (balance is not None and balance > 0),
            "raw": data,
        }

    def get_session_cost(self) -> dict[str, Any]:
        """Return cumulative usage. Adds ``actual_cost_usd`` (OR-billed truth).

        ``total_cost_usd`` is the per-token *estimate* (from /models pricing),
        kept for parity with other providers. ``actual_cost_usd`` is what
        OpenRouter actually billed — preferred when reconciling spend.
        """
        base = super().get_session_cost()
        base["actual_cost_usd"] = self.total_actual_cost_usd
        base["last_actual_cost_usd"] = self.last_actual_cost_usd
        base["cache_write_tokens"] = self.total_cache_write_tokens
        return base

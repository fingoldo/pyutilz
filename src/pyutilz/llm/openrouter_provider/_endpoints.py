"""OpenRouter's account and model-metadata endpoints, mixed into ``OpenRouterProvider``.

Split out of ``_provider.py`` for the 1,000-line module limit; the method bodies moved verbatim. Each one is a
plain GET against the provider's ``_client`` (``/parameters``, ``/models/{id}/endpoints``, ``/generation``,
``/key``, ``/credits``) and needs nothing from the provider but ``_client``, ``model_name`` and
``last_generation_id``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pyutilz.llm.openrouter_provider._health import _summarize_endpoints

# The logger name the methods used before the split, so existing log filters and handlers keep matching.
logger = logging.getLogger("pyutilz.llm.openrouter_provider._provider")


class OpenRouterEndpointsMixin:
    """Metadata, health and billing lookups against OpenRouter's REST endpoints."""

    model_name: str

    if TYPE_CHECKING:  # supplied by OpenRouterProvider / OpenAICompatibleProvider
        import httpx

        _client: httpx.AsyncClient
        last_generation_id: Any

    async def fetch_model_parameters(
        self,
        model: str | None = None,
    ) -> dict[str, Any]:
        """``GET /api/v1/parameters/{author}/{slug}`` — supported parameters
        and their default values for a model.

        Returns a dict with keys like ``temperature``, ``top_p``, ``top_k``,
        ``max_tokens``, ``frequency_penalty``, ``presence_penalty``, plus
        whatever the upstream supports. Useful before sending a request so
        you can pre-populate sensible defaults and warn early on unsupported
        kwargs.

        Args:
            model: Defaults to ``self.model_name``.
        """
        target = model or self.model_name
        resp = await self._client.get(f"/parameters/{target}")
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        return data if isinstance(data, dict) else {}

    async def check_model_health(
        self,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Pre-flight check: per-upstream uptime / latency / throughput.

        Calls ``GET /api/v1/models/{model}/endpoints`` — this endpoint
        requires the API key but is NOT charged against credits. Use as
        a "free ping" before kicking off a long batch:

        >>> health = await p.check_model_health()
        >>> print(health["best_uptime_30m"], "uptime,",
        ...       len(health["endpoints"]), "upstreams")

        Args:
            model: Defaults to ``self.model_name``. Pass explicitly to
                check a different model without rebuilding the provider.

        Returns:
            ``{"model", "name", "endpoints": [...], "best_uptime_30m",
            "best_latency_p50_ms", "best_throughput_p50_tps"}`` —
            see ``_summarize_endpoints`` for the per-endpoint shape.
        """
        target = model or self.model_name
        resp = await self._client.get(f"/models/{target}/endpoints")
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data") or {}
        return {
            "model": target,
            "name": data.get("name"),
            **_summarize_endpoints(data.get("endpoints") or []),
        }

    async def is_model_healthy(
        self,
        model: str | None = None,
        min_uptime: float = 0.99,
    ) -> bool:
        """One-shot bool guard: any upstream meeting ``min_uptime`` over 30m?

        Defaults to a strict 0.99 threshold — production batches.
        Lower (0.95 / 0.90) for tolerant pre-flights. Network errors
        return ``False`` rather than propagate, since a guard that
        crashes is worse than one that says "no, hold off".
        """
        try:
            h = await self.check_model_health(model)
        except Exception as exc:
            logger.warning("is_model_healthy: health check failed (%s)", exc)
            return False
        uptime = h.get("best_uptime_30m")
        return uptime is not None and uptime >= min_uptime

    async def fetch_generation_stats(
        self,
        generation_id: str | None = None,
    ) -> dict[str, Any]:
        """Look up post-hoc stats for a single generation by ID.

        Calls ``GET /api/v1/generation?id=<id>``. Useful when:
          * you streamed a response and want authoritative usage / cost
            (the streaming usage chunk can lag or be missing on early errors)
          * you need fields not in the inline ``usage`` block —
            ``latency``, ``generation_time``, ``moderation_latency``,
            ``provider_responses`` (the per-attempt log if a fallback chain
            was traversed), ``cache_discount``, ``response_cache_source_id``
            (was this served from a CDN-level response cache?), ``is_byok``
          * you're auditing spend after the fact

        Args:
            generation_id: Defaults to ``self.last_generation_id``. Pass
                explicitly when reconciling historical IDs.

        Returns:
            The raw ``data`` payload (see OR docs for the full ~30 field schema).
            See ``OpenRouterProvider.fetch_generation_stats.__doc__`` and
            https://openrouter.ai/docs/api/api-reference/generations/get-generation
            for fields.
        """
        gid = generation_id or self.last_generation_id
        if not gid:
            raise ValueError("No generation_id passed and self.last_generation_id is unset — " "call generate() first or pass a known ID.")
        resp = await self._client.get("/generation", params={"id": gid})
        resp.raise_for_status()
        payload = resp.json()
        # Narrow the EXTRACTED value, not just the container: OpenRouter's `data` field is not
        # contractually an object, so a list there would otherwise be returned under a
        # `dict[str, Any]` annotation and blow up at the caller's first subscript.
        data = payload.get("data", payload) if isinstance(payload, dict) else {}
        return data if isinstance(data, dict) else {}

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

        Cross-checks ``/api/v1/key`` for ``is_free_tier`` so a free-tier
        user (``balance=None`` because they never purchased credits) is
        marked ``is_available=True`` — they can still issue calls
        against the free-models quota. Without this cross-check, a
        free-tier user looks "unavailable" purely because the credits
        endpoint can't compute a balance.

        Returns:
            ``balance_usd``   — remaining credits (total_credits - total_usage)
                                or ``None`` for free-tier users
            ``total_granted`` — total credits ever loaded (USD)
            ``total_used``    — lifetime spend (USD)
            ``currency``      — always "USD" for OpenRouter
            ``is_available``  — True if balance > 0 OR free-tier user
            ``is_free_tier``  — never-purchased-credits flag (or None on lookup error)
            ``raw``           — provider's full response under ``data``
        """
        resp = await self._client.get("/credits")
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data", payload) if isinstance(payload, dict) else {}

        def _to_float(v: Any) -> float | None:
            """Coerce ``v`` to float, returning None for None or unconvertible values."""
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        granted = _to_float(data.get("total_credits"))
        used = _to_float(data.get("total_usage"))
        balance = (granted - used) if (granted is not None and used is not None) else None

        # Free-tier check: secondary GET to /key. Best-effort — if that
        # call fails, ``is_free_tier=None`` and ``is_available`` falls
        # back to the strict balance > 0 check.
        is_free_tier: bool | None = None
        try:
            key_info = await self.check_account_limits()
            raw_key = key_info.get("raw") or key_info
            if isinstance(raw_key, dict) and "is_free_tier" in raw_key:
                is_free_tier = bool(raw_key["is_free_tier"])
        except Exception as exc:
            logger.debug("OR /key lookup for is_free_tier failed: %s", exc)

        is_available = (balance is not None and balance > 0) or is_free_tier is True

        return {
            "balance_usd": balance,
            "total_granted": granted,
            "total_used": used,
            "currency": "USD",
            "is_available": is_available,
            "is_free_tier": is_free_tier,
            "raw": data,
        }

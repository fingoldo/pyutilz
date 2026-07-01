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

import asyncio
import logging
import sys
from typing import Any

import httpx

from pyutilz.llm.exceptions import LLMProviderError
from pyutilz.llm.openai_compat import OpenAICompatibleProvider
from pyutilz.llm.openrouter_provider._catalogue import (
    _per_token_cost_pair,
    _resolve_model_limits,
    _fetch_models_catalogue,
)
from pyutilz.llm.openrouter_provider._health import _summarize_endpoints

logger = logging.getLogger(__name__)


def _pkg():
    """Return the package facade (owns mutable caches + monkeypatch targets).

    Resolved via ``sys.modules`` (not a top-level import) so the static
    import graph stays acyclic; the facade is fully initialised by the
    time any of these functions run.
    """
    return sys.modules["pyutilz.llm.openrouter_provider"]

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
        enable_web_search: bool = False,
        anthropic_top_level_cache: bool = False,
        retry_routing_404: bool = False,
        routing_404_max_attempts: int = 3,
        routing_404_pause_sec: float = 60.0,
    ):
        settings = _pkg().get_llm_settings()
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
        # Phase-4 request-shape knobs.
        # ``enable_web_search``: attaches OR's web-search plugin so the
        #   model can fetch live web results (citations come back in
        #   choices[0].message.annotations and are auto-captured into
        #   ``last_web_search_citations``). Adds ~$0.005/call surcharge.
        # ``anthropic_top_level_cache``: emits ``cache_control:
        #   {"type":"ephemeral"}`` at the request top level (Anthropic
        #   only; routes that go through Bedrock/Vertex will have it
        #   ignored). Per-message breakpoints work everywhere; this is
        #   the cheap "just cache the system prompt" lever.
        self._enable_web_search = enable_web_search
        self._anthropic_top_level_cache = anthropic_top_level_cache

        # Per-call usage breakdown — set after every generate(). All
        # cumulative counters mirror their last_* counterpart.
        self.total_actual_cost_usd = 0.0
        self.last_actual_cost_usd = 0.0
        self.total_cache_write_tokens = 0
        self.last_cache_write_tokens = 0
        self.last_cache_hit_tokens = 0
        self.total_audio_tokens = 0
        self.last_audio_tokens = 0
        # cost_details.upstream_inference_cost — populated only on BYOK calls.
        # Lets you see the bare upstream price separately from any OR markup.
        self.total_upstream_inference_cost_usd = 0.0
        self.last_upstream_inference_cost_usd: float | None = None
        # Response-level metadata (set by _track_provider_specific_response).
        self.last_generation_id: str | None = None
        self.last_upstream_provider: str | None = None
        self.last_upstream_model: str | None = None
        self.last_native_finish_reason: str | None = None
        # Phase-4 OR-extra fields. cache_discount: usage-block field --
        # negative on cache writes (extra cost) or positive on cache reads
        # (savings). is_byok: whether the call was billed via the user's
        # own upstream key (5% surcharge after monthly free tier).
        # response_cache_source_id: present when the response was served
        # from OR's CDN-level response cache rather than a fresh upstream
        # call -- distinct from the per-prompt input cache.
        self.last_cache_discount_usd: float | None = None
        self.total_cache_discount_usd = 0.0
        self.last_is_byok: bool | None = None
        self.last_response_cache_source_id: str | None = None
        # Phase-4 web-search citations from OR's web plugin (when enabled).
        self.last_web_search_citations: list[dict[str, Any]] = []

        # Bench-only opt-in: bounded retry on routing 404/405. Disabled by
        # default because the parent's _NON_RETRYABLE_STATUSES rule (instant
        # fail on 404) protects production from spinning 30+ min on a model
        # that's permanently gone. But OR routing 404s are also commonly
        # transient (provider redeploy, cold start, regional failover) -
        # benchmarks lose 18% of rows to this. With ``retry_routing_404=True``
        # the provider retries the call up to ``routing_404_max_attempts``
        # times with ``routing_404_pause_sec`` between attempts. Total
        # wall-clock cap = max_attempts * pause - bounded so concurrency
        # pool never spins forever even if the model is genuinely dead.
        self._retry_routing_404 = retry_routing_404
        self._routing_404_max_attempts = max(1, int(routing_404_max_attempts))
        self._routing_404_pause_sec = max(0.0, float(routing_404_pause_sec))

    def _reset_per_call_state(self) -> None:
        """Clear per-call ``last_*`` state at the start of every generate().

        Without this, a failed or partial call would leave stale metadata
        from the previous successful call — ``last_call_summary()`` would
        report fields from the wrong call, silently misleading callers.
        Cumulative ``total_*`` counters are NOT reset.
        """
        self.last_actual_cost_usd = 0.0
        self.last_cache_write_tokens = 0
        self.last_cache_hit_tokens = 0
        self.last_audio_tokens = 0
        self.last_upstream_inference_cost_usd = None
        self.last_generation_id = None
        self.last_upstream_provider = None
        self.last_upstream_model = None
        self.last_native_finish_reason = None
        self._last_finish_reason = None
        self._last_usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
        self.last_cache_discount_usd = None
        self.last_is_byok = None
        self.last_response_cache_source_id = None
        self.last_web_search_citations = []

    async def generate(self, *args: Any, **kwargs: Any) -> str:
        """OR-specific bounded retry on routing 404/405.

        When ``retry_routing_404=True`` was passed at construction, an
        ``LLMProviderError`` carrying "API error 404"/"405" with the
        OR-specific routing wording ("No endpoints found" / "Method not
        allowed") triggers up to ``routing_404_max_attempts`` retries with
        ``routing_404_pause_sec`` between each. All other errors propagate
        immediately so the parent's tenacity retry machinery handles them.

        Default ``retry_routing_404=False`` keeps production behaviour:
        instant fail on 404, no retries (the original design intent of
        ``_NON_RETRYABLE_STATUSES`` - protects concurrency pool from
        spinning on permanently-gone models).
        """
        if not self._retry_routing_404:
            return await super().generate(*args, **kwargs)
        last_exc: LLMProviderError | None = None
        for attempt in range(1, self._routing_404_max_attempts + 1):
            try:
                return await super().generate(*args, **kwargs)
            except LLMProviderError as exc:
                msg = str(exc).lower()
                is_routing = (
                    ("api error 404" in msg or "api error 405" in msg)
                    and (
                        "no endpoints found" in msg
                        or "method not allowed" in msg
                        or " not found" in msg
                    )
                )
                if not is_routing:
                    raise
                last_exc = exc
                if attempt < self._routing_404_max_attempts:
                    logger.warning(
                        "[OR routing %s] %s attempt %d/%d - sleeping %ss "
                        "before retry: %s",
                        "404/405", self.model_name, attempt,
                        self._routing_404_max_attempts,
                        self._routing_404_pause_sec, str(exc)[:120],
                    )
                    await asyncio.sleep(self._routing_404_pause_sec)
        assert last_exc is not None
        raise last_exc

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
        if self._enable_web_search:
            # The "web" plugin id is the canonical OR shape; default
            # engine = exa unless OR overrides per-account.
            body["plugins"] = [{"id": "web"}]
        if self._anthropic_top_level_cache:
            body["cache_control"] = {"type": "ephemeral"}
        return body

    def _thinking_request_field(
        self, thinking: bool | str
    ) -> dict[str, Any] | None:
        """OpenRouter's unified ``reasoning`` field.

        OR auto-routes the body fragment to the correct upstream-specific
        shape (Anthropic ``thinking``, OpenAI ``reasoning_effort``,
        DeepSeek V4 thinking-toggle, etc.) based on the resolved model
        ID. See https://openrouter.ai/docs/use-cases/reasoning-tokens.

        Mapping:
          * ``False`` / empty string -> ``{"reasoning": {"exclude": True}}``
            (explicit no-think; useful for hybrid models that default to
            thinking when a stage doesn't need it).
          * ``True`` -> ``{"reasoning": {"effort": "medium"}}``
            (provider's middle-ground default).
          * ``"low" | "medium" | "high" | "minimal"`` (or any other str)
            -> ``{"reasoning": {"effort": <str>}}`` (passed through as-is;
            unknown strings are accepted by OR and forwarded to upstream
            which may reject them with a 400).

        Models that don't support reasoning ignore the field server-side,
        so emitting it on a non-reasoning model is a no-op rather than
        an error.
        """
        enabled, effort = self._normalize_thinking(thinking)
        if not enabled:
            return {"reasoning": {"exclude": True}}
        return {"reasoning": {"effort": effort or "medium"}}

    def _handle_special_status(self, resp: httpx.Response) -> None:
        if resp.status_code == 402:
            logger.warning(
                "OpenRouter account out of credits (HTTP 402). "
                "Top up at https://openrouter.ai/credits - "
                "retrying indefinitely until balance is restored..."
            )

    def _track_provider_specific_usage(self, usage: dict[str, Any]) -> None:
        """Capture every OR-specific field in the ``usage`` block.

        Records:
          * ``usage.cost``                                     → actual_cost_usd
          * ``usage.cost_details.upstream_inference_cost``     → upstream cost (BYOK)
          * ``usage.prompt_tokens_details.cache_write_tokens`` → cache_write_tokens
          * ``usage.prompt_tokens_details.cached_tokens``      → cache_hit_tokens
                (fallback only — base already records ``prompt_cache_hit_tokens``
                if upstream uses the legacy field)
          * ``usage.prompt_tokens_details.audio_tokens``       → audio_tokens
        """
        cost = usage.get("cost")
        if isinstance(cost, (int, float)):
            self.last_actual_cost_usd = float(cost)
            self.total_actual_cost_usd += float(cost)
        else:
            self.last_actual_cost_usd = 0.0

        cost_details = usage.get("cost_details") or {}
        upstream_cost = cost_details.get("upstream_inference_cost")
        if isinstance(upstream_cost, (int, float)):
            self.last_upstream_inference_cost_usd = float(upstream_cost)
            self.total_upstream_inference_cost_usd += float(upstream_cost)
        else:
            self.last_upstream_inference_cost_usd = None

        prompt_details = usage.get("prompt_tokens_details") or {}
        cache_write = int(prompt_details.get("cache_write_tokens", 0) or 0)
        self.last_cache_write_tokens = cache_write
        if cache_write:
            self.total_cache_write_tokens += cache_write

        cached = int(prompt_details.get("cached_tokens", 0) or 0)
        # Fall back to "cached_tokens" only when upstream didn't use the legacy
        # "prompt_cache_hit_tokens" field (which the base class already tracks).
        if cached and not usage.get("prompt_cache_hit_tokens"):
            self.total_cache_hit_tokens += cached
        self.last_cache_hit_tokens = cached or int(usage.get("prompt_cache_hit_tokens") or 0)

        audio = int(prompt_details.get("audio_tokens", 0) or 0)
        self.last_audio_tokens = audio
        if audio:
            self.total_audio_tokens += audio

        # Phase-4: cache_discount (OR's per-call line item showing how
        # much the cache hit saved -- positive = savings vs cold call).
        cache_discount = usage.get("cache_discount")
        if isinstance(cache_discount, (int, float)):
            self.last_cache_discount_usd = float(cache_discount)
            self.total_cache_discount_usd += float(cache_discount)
        else:
            self.last_cache_discount_usd = None

    def _track_provider_specific_response(self, data: dict[str, Any]) -> None:
        """Capture response-level metadata outside the ``usage`` block.

        Records:
          * ``id``       → generation ID, usable with /api/v1/generation
          * ``provider`` → upstream that actually served the request
                           (e.g. "Anthropic", "DeepInfra"); critical when
                           debugging routing or auditing where requests went
          * ``model``    → resolved model — differs from ``self.model_name``
                           when ``models_fallback`` kicked in
          * ``choices[0].native_finish_reason`` → upstream's native code
                           (e.g. "tool_calls", "max_tokens", "content_filter")
        """
        gen_id = data.get("id")
        if isinstance(gen_id, str):
            self.last_generation_id = gen_id

        provider = data.get("provider")
        if isinstance(provider, str):
            self.last_upstream_provider = provider

        resolved = data.get("model")
        if isinstance(resolved, str):
            self.last_upstream_model = resolved

        choices = data.get("choices") or []
        if choices:
            native = choices[0].get("native_finish_reason")
            if isinstance(native, str):
                self.last_native_finish_reason = native
            # OR's web-search plugin attaches citations on the message.
            msg = choices[0].get("message") or {}
            annotations = msg.get("annotations") or []
            citations: list[dict[str, Any]] = []
            for ann in annotations:
                if not isinstance(ann, dict):
                    continue
                if ann.get("type") == "url_citation":
                    citations.append(ann.get("url_citation") or {})
            if citations:
                self.last_web_search_citations = citations

        # Phase-4 OR-extra response-level fields.
        is_byok = data.get("is_byok")
        if isinstance(is_byok, bool):
            self.last_is_byok = is_byok
        cache_src = data.get("response_cache_source_id")
        if isinstance(cache_src, str):
            self.last_response_cache_source_id = cache_src

    def _input_cost_per_1m(self, model: str) -> float:
        return _per_token_cost_pair(model)[0]

    def _output_cost_per_1m(self, model: str) -> float:
        return _per_token_cost_pair(model)[1]

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
            raise ValueError(
                "No generation_id passed and self.last_generation_id is unset — "
                "call generate() first or pass a known ID."
            )
        resp = await self._client.get("/generation", params={"id": gid})
        resp.raise_for_status()
        payload = resp.json()
        return payload.get("data", payload) if isinstance(payload, dict) else {}

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

        # Free-tier check: secondary GET to /key. Best-effort — if that
        # call fails, ``is_free_tier=None`` and ``is_available`` falls
        # back to the strict balance > 0 check.
        is_free_tier: bool | None = None
        try:
            key_info = await self.check_account_limits()
            raw_key = key_info.get("raw") or key_info
            if isinstance(raw_key, dict) and "is_free_tier" in raw_key:
                is_free_tier = bool(raw_key["is_free_tier"])
        except Exception as exc:  # noqa: BLE001
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

    def get_session_cost(self) -> dict[str, Any]:
        """Return cumulative usage. Adds OR-specific fields on top of base.

        ``total_cost_usd`` (from base) is the per-token *estimate* via
        /models pricing. ``actual_cost_usd`` is what OR actually billed —
        preferred when reconciling spend.
        ``upstream_inference_cost_usd`` populates only on BYOK calls and
        gives you the bare upstream price, separate from any OR markup.
        """
        base = super().get_session_cost()
        base["actual_cost_usd"] = self.total_actual_cost_usd
        base["last_actual_cost_usd"] = self.last_actual_cost_usd
        base["upstream_inference_cost_usd"] = self.total_upstream_inference_cost_usd
        base["last_upstream_inference_cost_usd"] = self.last_upstream_inference_cost_usd
        base["cache_write_tokens"] = self.total_cache_write_tokens
        base["audio_tokens"] = self.total_audio_tokens
        return base

    def last_call_summary(self) -> dict[str, Any]:
        """Snapshot of every metric captured for the most recent ``generate()``.

        Convenience for ad-hoc inspection / logging — pulls every ``last_*``
        attribute into one dict so you don't fish for them individually:

            >>> await p.generate("hello")
            >>> p.last_call_summary()
            {
                'generation_id': 'gen-abc123',
                'upstream_provider': 'Anthropic',
                'upstream_model': 'anthropic/claude-sonnet-4.6',
                'requested_model': 'anthropic/claude-sonnet-4.6',
                'finish_reason': 'stop',
                'native_finish_reason': 'end_turn',
                'cost_usd': 0.0042,
                'upstream_inference_cost_usd': None,
                'input_tokens': 1200,
                'output_tokens': 56,
                'reasoning_tokens': 0,
                'cache_hit_tokens': 800,
                'cache_write_tokens': 0,
                'audio_tokens': 0,
            }
        """
        return {
            "generation_id": self.last_generation_id,
            "upstream_provider": self.last_upstream_provider,
            "upstream_model": self.last_upstream_model,
            "requested_model": self.model_name,
            "finish_reason": getattr(self, "_last_finish_reason", None),
            "native_finish_reason": self.last_native_finish_reason,
            "cost_usd": self.last_actual_cost_usd,
            "upstream_inference_cost_usd": self.last_upstream_inference_cost_usd,
            "cache_discount_usd": self.last_cache_discount_usd,
            "is_byok": self.last_is_byok,
            "response_cache_source_id": self.last_response_cache_source_id,
            "input_tokens": self._last_usage.get("input_tokens", 0),
            "output_tokens": self._last_usage.get("output_tokens", 0),
            "reasoning_tokens": self._last_usage.get("reasoning_tokens", 0),
            "cache_hit_tokens": self.last_cache_hit_tokens,
            "cache_write_tokens": self.last_cache_write_tokens,
            "audio_tokens": self.last_audio_tokens,
            "web_search_citations": self.last_web_search_citations,
        }

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

3. **Routing knobs as hashable kwargs** — ``provider_order``, ``provider_only``,
   ``provider_ignore``, ``provider_sort``, ``provider_allow_fallbacks``,
   ``provider_require_parameters``, ``provider_data_collection``, ``provider_zdr``,
   ``provider_quantizations``, ``provider_max_price``, ``provider_preferred_min_throughput``,
   ``provider_preferred_max_latency`` and ``models_fallback`` are forwarded to the request body
   as the ``provider`` and ``models`` fields (see ``_request.py``). Sequences are tuples and
   mappings may be given as tuples of pairs, so the factory's ``tuple(sorted(kwargs.items()))``
   cache key keeps working.

4. **App-attribution headers** — ``app_name`` and ``site_url`` set
   ``X-Title`` and ``HTTP-Referer`` so calls show up in the openrouter.ai
   public rankings dashboard (optional).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, ClassVar

import httpx

from pyutilz.llm.exceptions import LLMProviderError
from pyutilz.llm.base import PerCallAttr
from pyutilz.llm.openai_compat import OpenAICompatibleProvider, Pricing
from pyutilz.llm.openrouter_provider._accounting import OpenRouterAccountingMixin
from pyutilz.llm.openrouter_provider._endpoints import OpenRouterEndpointsMixin
from pyutilz.llm.openrouter_provider._route import _ROUTE_FINGERPRINT_ATTRS, OpenRouterRouteMixin
from pyutilz.llm.openrouter_provider._request import (
    OpenRouterRequestMixin,
    as_plain,
    url_citations,
    build_plugins,
    build_provider_field,
    catalogue_id,
)
from pyutilz.llm.openrouter_provider._catalogue import (
    _cache_read_cost_per_1m_or_none,
    _cache_write_cost_per_1m_or_none,
    _per_token_cost_pair,
    _catalogue_is_loaded,
    _resolve_model_limits,
    _fetch_models_catalogue,
    _ensure_catalogue_warm_async,
    _pkg,
)

logger = logging.getLogger(__name__)


# Model ids whose resolved endpoint answered "Reasoning is mandatory for this endpoint and cannot be
# disabled" to `reasoning: {enabled: false}` - a per-process note so that refusal is paid once, not per call.
_REASONING_CANNOT_BE_DISABLED: set[str] = set()


class OpenRouterProvider(OpenRouterAccountingMixin, OpenRouterEndpointsMixin, OpenRouterRouteMixin, OpenRouterRequestMixin, OpenAICompatibleProvider):
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
    # Models whose ceiling has already been reported as falling back, so the warning in
    # max_output_tokens fires once per model rather than on every read.
    _warned_default_ceiling: ClassVar[set[str]] = set()
    _default_context_window = 128_000

    # Per-call OpenRouter metadata -- contextvar-backed for the same reason as the parent's
    # _last_usage/last_tool_calls (see PerCallAttr in base.py): generate_batch() fires N
    # concurrent generate() calls on ONE shared instance, and as plain instance attributes each
    # request zeroed the previous one's in-flight cost/generation-id, so a cost-accounting loop
    # attributed the wrong billed USD and the wrong upstream provider to each request id.
    last_actual_cost_usd: PerCallAttr = PerCallAttr(lambda: 0.0)
    # Whether ANY POST of the last call carried `usage.cost`. `last_actual_cost_usd` is 0.0 both for a genuinely free
    # call and for a response with no cost field, and a caller cannot tell a $0 bill from a missing one without this
    # (realtime_applications re-audit 2026-09-11, RA2-C11-8).
    last_actual_cost_reported: PerCallAttr = PerCallAttr(lambda: False)
    last_cache_write_tokens: PerCallAttr = PerCallAttr(lambda: 0)
    last_cache_hit_tokens: PerCallAttr = PerCallAttr(lambda: 0)
    last_audio_tokens: PerCallAttr = PerCallAttr(lambda: 0)
    last_upstream_inference_cost_usd: PerCallAttr = PerCallAttr(lambda: None)
    last_generation_id: PerCallAttr = PerCallAttr(lambda: None)
    last_upstream_provider: PerCallAttr = PerCallAttr(lambda: None)
    last_upstream_model: PerCallAttr = PerCallAttr(lambda: None)
    last_native_finish_reason: PerCallAttr = PerCallAttr(lambda: None)
    last_cache_discount_usd: PerCallAttr = PerCallAttr(lambda: None)
    last_is_byok: PerCallAttr = PerCallAttr(lambda: None)
    last_response_cache_source_id: PerCallAttr = PerCallAttr(lambda: None)
    last_web_search_citations: PerCallAttr = PerCallAttr(list)

    # So generate_batch()'s yielded per-request dict carries them too -- the only race-free way
    # for the caller to read them after a batch.
    _PERCALL_METADATA_ATTRS: tuple[str, ...] = (
        *OpenAICompatibleProvider._PERCALL_METADATA_ATTRS,
        "last_actual_cost_usd",
        "last_actual_cost_reported",
        "last_cache_write_tokens",
        "last_cache_hit_tokens",
        "last_audio_tokens",
        "last_upstream_inference_cost_usd",
        "last_generation_id",
        "last_upstream_provider",
        "last_upstream_model",
        "last_native_finish_reason",
        "last_cache_discount_usd",
        "last_is_byok",
        "last_response_cache_source_id",
        "last_web_search_citations",
    )

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
        provider_only: tuple[str, ...] | None = None,
        provider_require_parameters: bool | None = None,
        provider_data_collection: str | None = None,
        provider_zdr: bool | None = None,
        provider_quantizations: tuple[str, ...] | None = None,
        provider_max_price: Any = None,
        provider_preferred_min_throughput: Any = None,
        provider_preferred_max_latency: Any = None,
        web_search_engine: str | None = None,
        web_search_max_results: int | None = None,
        pdf_engine: str | None = None,
        transforms: tuple[str, ...] | None = None,
        system_cache_control: bool = False,
        extra_body: Any = None,
    ):
        """See the class and module docstrings; the routing/request options are documented in the README's OpenRouter section.

        ``provider_require_parameters``: None (default) = automatic, True/False = always / never. Automatic sends
        ``require_parameters: true`` when the request carries ``response_format`` or asks the model to reason AND the
        catalogue lists every parameter in the body for the model, so the answer never silently loses the constraint
        (``_request.auto_require_parameters``). ``provider_max_price`` / ``provider_preferred_*`` / ``extra_body`` take a
        mapping or a tuple of pairs (hashable, for the factory cache). ``pdf_engine`` (``"mistral-ocr"``,
        ``"cloudflare-ai"``, ``"native"``) attaches the file-parser plugin. ``transforms=("middle-out",)`` lets OpenRouter
        compress an over-long prompt, which DROPS content. ``system_cache_control`` sends the system prompt as a
        ``cache_control`` breakpoint. ``extra_body`` is merged into every request (``seed``, ``user``, ``stop``, ...).
        """
        # Captured first, before any other local exists: the inputs route_fingerprint() digests (see _route.py).
        self._route_kwargs = {name: value for name, value in locals().items() if name in _ROUTE_FINGERPRINT_ATTRS}
        settings = _pkg().get_llm_settings()
        resolved_key = api_key or (settings.openrouter_api_key.get_secret_value() if settings.openrouter_api_key else None)
        if not resolved_key:
            raise ValueError("OpenRouter API key not provided. " "Set OPENROUTER_API_KEY in .env or pass api_key=")
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
        self._provider_only = provider_only
        self._provider_require_parameters = provider_require_parameters
        # Built once: the routing block is the same for every request of this instance (validated here, so a typo such
        # as data_collection="denied" fails at construction rather than as a 400 on the first paid call).
        self._provider_field = build_provider_field(
            order=provider_order,
            ignore=provider_ignore,
            only=provider_only,
            sort=provider_sort,
            allow_fallbacks=provider_allow_fallbacks,
            data_collection=provider_data_collection,
            zdr=provider_zdr,
            quantizations=provider_quantizations,
            max_price=provider_max_price,
            preferred_min_throughput=provider_preferred_min_throughput,
            preferred_max_latency=provider_preferred_max_latency,
            require_parameters=provider_require_parameters,
        )
        self._plugins = build_plugins(enable_web_search, web_search_engine, web_search_max_results, pdf_engine)
        self._transforms = transforms
        self._system_cache_control = system_cache_control
        self._default_extra_body = as_plain(extra_body) if extra_body else None
        # Session token tallies per model that SERVED the call (`models` fallback can differ from `model_name`):
        # [prompt, cache_hit, completion, reasoning, cache_write]. The estimate prices each at its own model's rates.
        self._usage_by_model: dict[str, list[int]] = {}

        # Per-call usage breakdown — set after every generate(). All
        # cumulative counters mirror their last_* counterpart.
        # The last_* counterparts are class-level PerCallAttr descriptors (declared above), so
        # they need no per-instance initialization.
        self.total_actual_cost_usd = 0.0
        self.total_cache_write_tokens = 0
        self.total_audio_tokens = 0
        # cost_details.upstream_inference_cost — populated only on BYOK calls.
        # Lets you see the bare upstream price separately from any OR markup.
        self.total_upstream_inference_cost_usd = 0.0
        # Phase-4 OR-extra fields. cache_discount: usage-block field --
        # negative on cache writes (extra cost) or positive on cache reads
        # (savings). is_byok: whether the call was billed via the user's
        # own upstream key (5% surcharge after monthly free tier).
        # response_cache_source_id: present when the response was served
        # from OR's CDN-level response cache rather than a fresh upstream
        # call -- distinct from the per-prompt input cache.
        self.total_cache_discount_usd = 0.0

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

    # No ``_reset_per_call_state`` override: LLMProvider's derives the reset from
    # ``_PERCALL_METADATA_ATTRS`` (extended above with every OR-specific attribute), so the
    # hand-maintained copy that used to live here can no longer drift out of sync with it.

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
            return await super().generate(*args, **kwargs)  # type: ignore[no-any-return]  # the base generate() is declared in an optional-dependency mixin with no stubs
        last_exc: LLMProviderError | None = None
        for attempt in range(1, self._routing_404_max_attempts + 1):
            try:
                return await super().generate(*args, **kwargs)  # type: ignore[no-any-return]  # same base call, inside the 404 retry loop
            except LLMProviderError as exc:  # noqa: PERF203 -- per-attempt retry loop; the try/except IS the retry mechanism
                msg = str(exc).lower()
                is_routing = ("api error 404" in msg or "api error 405" in msg) and (
                    "no endpoints found" in msg or "method not allowed" in msg or " not found" in msg
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
        assert last_exc is not None  # nosec B101 - internal invariant (loop always sets last_exc before falling through); not a security check, and python -O stripping it just means last_exc could be None here, causing a TypeError instead of this assert
        raise last_exc

    async def _async_prepare(self) -> None:
        """Pre-warm the models catalogue cache off the event-loop thread before
        ``context_window``/``max_output_tokens`` (sync properties) are read on the hot request
        path -- see ``_ensure_catalogue_warm_async``'s docstring for the blocking-call bug this
        closes."""
        await _ensure_catalogue_warm_async()

    @property
    def context_window(self) -> int:
        """Return the upstream-enforced context window for the active model.

        Prefers ``top_provider.context_length`` from the OR catalogue (what
        the routed backend actually accepts) over the model-level theoretical
        max. Falls back to ``_default_context_window`` if the catalogue is
        unreachable / model isn't listed.
        """
        ctx = self._smallest_known_limit(0)
        return ctx if ctx is not None else self._default_context_window

    def _served_models(self) -> list[str]:
        """Catalogue ids of every model a request may be served by: the requested one, then the ``models`` fallbacks."""
        return [catalogue_id(m) for m in (self.model_name, *(getattr(self, "_models_fallback", None) or ()))]

    def _smallest_known_limit(self, index: int) -> int | None:
        """The smallest known ``_resolve_model_limits`` value (0 = context, 1 = output cap) across the served models.

        With a ``models`` fallback list a request sized to the primary's cap is refused (HTTP 400) by a fallback with a
        smaller one, so the limit that holds for the whole list is the minimum. Models the catalogue does not list are
        skipped rather than turned into a guess.
        """
        known = [lim for lim in (_resolve_model_limits(m)[index] for m in self._served_models()) if lim is not None]
        return min(known) if known else None

    def _catalogue_entry(self) -> dict[str, Any] | None:
        """This model's ``/models`` catalogue entry (routing suffix stripped), or None."""
        entry = _fetch_models_catalogue().get(catalogue_id(self.model_name))
        return entry if isinstance(entry, dict) else None

    @property
    def max_output_tokens(self) -> int:
        """Return the upstream-enforced output cap for the active model.

        Pulls ``top_provider.max_completion_tokens`` from the catalogue when
        available — ignored by some upstreams but a hard cap on others.
        Falls back to ``_default_max_tokens`` if absent.

        2026-09-06: the fallback is now logged, once per model. A transient failure of the
        single catalogue request silently turned a 131072-token ceiling into 8192 for
        several current models — a 16x cut, mid-run, with nothing in the log to say so.
        Downstream that reads as a truncated response and a failed pipeline whose cause is
        invisible; it cost two full pipeline runs to find. The fallback itself is correct
        behaviour, so this warns rather than raising.

        2026-09-14, a third run lost to the same 8192: the cause is not a flaky network but the
        ORDER a caller reads these in. A caller computes its own ``max_tokens`` from this property
        BEFORE calling ``generate``, so ``_async_prepare``'s warm has not run yet; asked from the
        event-loop thread with a cold cache, ``_fetch_models_catalogue`` deliberately returns ``{}``
        rather than blocking the loop, and this property then reported a ceiling of 8192 for a model
        that serves 65,536. The first article of a corpus wave came back cut mid-JSON, billed in full.

        **So neither fallback is a small round number any more.** The catalogue not being readable
        yet, and the catalogue listing no ``max_completion_tokens``, are both statements that the real
        cap is UNKNOWN — the second one positively so, since a model with no entry has no cap enforced
        at the model level. An unknown ceiling becomes the context window: a real, generous number that
        the caller's own fitting then clamps to what is left after the prompt. 8192 was neither
        measured nor generous, it was small enough to truncate a long answer and large enough to look
        deliberate. A caller that needs the exact ceiling should pin a route and read the cap off the
        endpoint it pinned.
        """
        max_out = self._smallest_known_limit(1)
        if max_out is not None:
            return max_out
        cold = not _catalogue_is_loaded()
        ceiling = self.context_window
        if self.model_name not in self._warned_default_ceiling:
            self._warned_default_ceiling.add(self.model_name)
            reason = (
                "has not been read yet (asked from an event loop before the warm-up ran)"
                if cold
                else "lists no max_completion_tokens for this model, which means no cap is enforced at the model level"
            )
            logger.warning(
                "OpenRouter catalogue %s, so the real output cap for %r is UNKNOWN rather than small: "
                "using the context window, %d. Await the provider's own prepare step before sizing a budget "
                "if the exact number matters, or pin a route and take its advertised cap.",
                reason, self.model_name, ceiling,
            )
        return ceiling

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
        except Exception as e:
            # _fetch_models_catalogue() already catches and logs its own network/parse failures
            # (returning {} rather than raising), so reaching this handler means something more
            # unexpected went wrong in the surrounding plumbing -- worth a trail even though the
            # fail-open behavior itself is deliberate (see docstring above).
            logger.debug("supports_json_mode: unexpected catalogue-fetch error (%s), assuming supported", e)
            return True
        entry = catalogue.get(catalogue_id(self.model_name))
        if not entry:
            # Unknown model — best-effort: assume support, rely on the
            # upstream to reject if it can't handle it.
            return True
        params = entry.get("supported_parameters") or []
        if isinstance(params, list):
            params_set = {str(p).lower() for p in params}
            return "response_format" in params_set or "structured_outputs" in params_set
        return True

    def supports_json_schema(self) -> bool:
        """Per-model STRICT-schema support from the OR catalogue's ``supported_parameters``.

        Stricter than ``supports_json_mode``: plain ``response_format`` only buys valid JSON, whereas a
        closed enum needs constrained generation, which OR advertises as ``structured_outputs``. Unlike
        the json-mode check this returns False on catalogue failure or an unknown model — claiming a
        guarantee we cannot verify is worse than degrading to json_object, because the caller would
        stop validating the enum it believes was enforced.
        """
        try:
            catalogue = _fetch_models_catalogue()
        except Exception as exc:
            logger.warning("OR catalogue unavailable (%s); treating %s as NOT supporting strict json_schema", exc, self.model_name)
            return False
        entry = catalogue.get(catalogue_id(self.model_name))
        if not entry:
            logger.warning("%s absent from the OR catalogue; treating it as NOT supporting strict json_schema", self.model_name)
            return False
        params = entry.get("supported_parameters") or []
        if isinstance(params, list):
            return "structured_outputs" in {str(p).lower() for p in params}
        return False

    def _get_timeout(self, model: str) -> float:
        """Return the request timeout in seconds for ``model``: 1200.0 for known slow reasoning-class models, else 240.0."""
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

    def _thinking_request_field(self, thinking: bool | str | int) -> dict[str, Any] | None:
        """OpenRouter's unified ``reasoning`` field.

        OR auto-routes the body fragment to the correct upstream-specific
        shape (Anthropic ``thinking``, OpenAI ``reasoning_effort``,
        DeepSeek V4 thinking-toggle, etc.) based on the resolved model
        ID. See https://openrouter.ai/docs/use-cases/reasoning-tokens.

        Mapping:
          * ``False`` / empty string -> ``{"reasoning": {"effort": "minimal", "exclude": True}}``
            (smallest available reasoning budget, hidden from the response too).
          * ``True`` -> ``{"reasoning": {"effort": "medium"}}``
            (provider's middle-ground default).
          * a positive ``int`` -> ``{"reasoning": {"max_tokens": n}}`` (a reasoning budget in tokens).
          * ``"low" | "medium" | "high" | "minimal" | "xhigh" | "max"`` (or any other str; ``"none"``
            normalises to disabled, i.e. the ``enabled: False`` form above)
            -> ``{"reasoning": {"effort": <str>}}`` (passed through as-is;
            unknown strings are accepted by OR and forwarded to upstream
            which may reject them with a 400).

        Models that don't support reasoning ignore the field server-side,
        so emitting it on a non-reasoning model is a no-op rather than
        an error.

        ``exclude`` ALONE is not "disable" - MEASURED 2026-08-07 (autopsia,
        `bench/lay_synonym_full_run.py` against deepseek/deepseek-v4-flash,
        whose OWN catalogue entry defaults ``reasoning.default_effort`` to
        ``"high"``): calls sent with the previous mapping (``exclude`` only,
        no ``effort``) kept intermittently burning 900-3000 reasoning tokens
        - repeatedly hitting ``finish_reason='length'`` before any JSON was
        written - because ``exclude`` only suppresses the reasoning text
        from the RESPONSE; per OpenRouter's own docs the model still REASONS
        and is still BILLED for it at whatever effort the resolved model
        defaults to, unconstrained, unless ``effort`` is also set. Passing
        both fields is the actual "spend as little as possible on thinking"
        request - confirmed live afterward: 0 truncations across a follow-up
        batch that previously truncated intermittently.

        AND "as little as possible" is still not "none". Re-measured 2026-09-02 on one identical question
        across eight models, comparing that mapping against ``{"enabled": False}`` - billed reasoning tokens,
        then what the same call returned:

        z-ai/glm-4.7-flash 348 tok and an EMPTY answer cut off by `length`, against 0 tok and an answer;
        deepseek/deepseek-v3.2 259 -> 0; xiaomi/mimo-v2.5 202 -> 0; nvidia/nemotron-3.5-lightning 162 -> 0;
        deepseek/deepseek-v4-flash 13 -> 0; qwen/qwen3.8-flash rate-limited -> 0. So ``enabled: False`` is
        the real switch, worth 150-350 billed tokens a call on a reasoning model and, on glm, the difference
        between an answer and nothing.

        The other two of the eight - google/gemini-3.7-flash and openai/gpt-oss-120b - answer "Reasoning is
        mandatory for this endpoint and cannot be disabled" and fail the WHOLE call. Neither the catalogue's
        ``supported_parameters`` nor its reasoning block distinguishes them from the six, so the fallback is
        keyed on that refusal (`_body_after_rejected_request`) and remembered per model id for the process.

        Re-measured 2026-09-25 on pinned upstreams, one short JSON question, ``max_tokens`` 2000, billed reasoning
        tokens per setting (upstream default / ``enabled: False`` / minimal / low / medium / high):

        z-ai/glm-5.3-flash @morph: 1607 / HTTP 400 "Reasoning is mandatory" / 0 / 93 / 87 / 30. Its off switch
        is therefore the minimal-effort fallback above, which here does reach 0 tokens.
        deepseek/deepseek-v4.1-flash @morph: 1046 / 0 / 789 / 1273 / 2000 (all of max_tokens, no answer) / 774.
        ``enabled: False`` is the real off; the efforts are not monotone, and medium ran away to the cap on a
        one-word question, so a caller that enables reasoning must bound ``max_tokens`` for the answer it needs.
        google/gemini-3-flash-preview @google-ai-studio: 0 / 0 / 0 / 220 / 151 / 181. Off works; its default is
        already no reasoning.

        By then the catalogue did expose the difference: glm's entry carries ``reasoning.mandatory: true`` with
        efforts max/high/low only. The refusal-keyed fallback stays, since older entries lack the field.
        """
        if isinstance(thinking, int) and not isinstance(thinking, bool) and thinking > 0:
            # A positive int is a reasoning BUDGET in tokens: the reasoning doc's `max_tokens`, the alternative to
            # `effort` (the two are exclusive). It is the cap the 2026-09-25 deepseek-v4.1-flash runaway above needed.
            return {"reasoning": {"max_tokens": thinking}}
        enabled, effort = self._normalize_thinking(thinking)
        if not enabled:
            if self.model_name in _REASONING_CANNOT_BE_DISABLED or self._catalogue_says_reasoning_mandatory():
                return {"reasoning": {"effort": "minimal", "exclude": True}}
            return {"reasoning": {"enabled": False}}
        return {"reasoning": {"effort": effort or "medium"}}

    def _catalogue_says_reasoning_mandatory(self) -> bool:
        """True when the catalogue marks this model's reasoning ``mandatory`` (then the off switch is refused with a 400).

        Measured live 2026-09-26 (``audits/implemented/2026-09-26/or9_live_results.json``): on all five catalogue-mandatory models
        probed (gpt-oss-20b, gpt-5-nano, glm-5.3-flash, step-3.5-flash, gemini-3.7-flash) BOTH ``enabled: false`` and
        ``effort: "none"`` were refused with HTTP 400 "Reasoning is mandatory", while ``effort: "minimal"`` succeeded on
        every one (0-121 reasoning tokens). Reading the flag up front saves the refused round trip the per-process set
        otherwise pays once per model.
        """
        entry = self._catalogue_entry()
        reasoning = entry.get("reasoning") if entry else None
        return isinstance(reasoning, dict) and reasoning.get("mandatory") is True

    def _body_after_rejected_request(self, body: dict[str, Any], status: int, detail: str) -> dict[str, Any] | None:
        """Re-issue a ``reasoning: {enabled: false}`` call this endpoint refuses, with the budget form instead.

        The model id is remembered, so an endpoint that mandates reasoning costs one refused call per process
        rather than one per request.
        """
        if status not in (400, 404, 422):
            # A refusal of the PARAMETER, not a server fault: repairing a 500 or a 429 would re-issue a call
            # the upstream never rejected on its content, and hide a retryable failure as a permanent one.
            return None
        reasoning = body.get("reasoning")
        # `effort: "none"` (sent via extra_body, say) is refused exactly like `enabled: false` on a mandatory endpoint
        # (measured 2026-09-26, or9_live_results.json), so it gets the same repair.
        if not isinstance(reasoning, dict) or not (reasoning.get("enabled") is False or reasoning.get("effort") == "none"):
            return None
        said = detail.lower()
        if "reasoning" not in said or not ("mandatory" in said or "cannot be disabled" in said):
            return None
        _REASONING_CANNOT_BE_DISABLED.add(self.model_name)
        return {**body, "reasoning": {"effort": "minimal", "exclude": True}}

    def _handle_special_status(self, resp: httpx.Response) -> None:
        """Log a warning when ``resp`` signals HTTP 402 (OpenRouter account out of credits)."""
        if resp.status_code == 402:
            logger.warning(
                "OpenRouter account out of credits (HTTP 402). "
                "Top up at https://openrouter.ai/credits - "
                "retrying for a short grace window in case the balance is being topped up, "
                "then giving up (PYUTILZ_LLM_BILLING_GRACE_SECONDS)..."
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
        # ACCUMULATED within one call, not assigned (realtime_applications audit 2026-09-10, LLM-7).
        # `generate()` can make a second billed POST -- the re-issue without `response_format`, or
        # with a repaired parameter -- and each POST reaches this hook. Assigning kept only the last
        # POST's cost, so a call billed twice was recorded once. `last_actual_cost_usd` is a
        # `PerCallAttr` reset to 0.0 at the start of every `generate()`, so summing here is exact
        # per call; a POST that reports no cost adds nothing rather than wiping an earlier one.
        cost = usage.get("cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            self.last_actual_cost_usd = float(self.last_actual_cost_usd or 0.0) + float(cost)
            self.last_actual_cost_reported = True
            self.total_actual_cost_usd += float(cost)

        # Every other per-call field is summed the same way (2026-09-26 OR-5): they were ASSIGNED, so after a re-issue
        # the breakdown described only the second POST while `last_actual_cost_usd` and `_last_usage` covered both, and a
        # second POST lacking a field reset it to None or 0. A POST without a field adds nothing and never erases.
        cost_details = usage.get("cost_details") or {}
        upstream_cost = cost_details.get("upstream_inference_cost")
        if isinstance(upstream_cost, (int, float)) and not isinstance(upstream_cost, bool):
            self.last_upstream_inference_cost_usd = float(self.last_upstream_inference_cost_usd or 0.0) + float(upstream_cost)
            self.total_upstream_inference_cost_usd += float(upstream_cost)

        prompt_details = usage.get("prompt_tokens_details") or {}
        cache_write = int(prompt_details.get("cache_write_tokens", 0) or 0)
        self.last_cache_write_tokens = int(self.last_cache_write_tokens or 0) + cache_write
        if cache_write:
            self.total_cache_write_tokens += cache_write

        cached = int(prompt_details.get("cached_tokens", 0) or 0)
        # NOTE: total_cache_hit_tokens itself is now accumulated in the shared base class's
        # _record_usage (which, as of the 2026-07-21 audit fix, also falls back to
        # prompt_tokens_details.cached_tokens) -- do NOT add it again here, that would double-count.
        # The OpenAI-style field wins when it reports hits; the DeepSeek-style top-level field is the fallback.
        cache_hit = cached if cached > 0 else int(usage.get("prompt_cache_hit_tokens") or 0)
        self.last_cache_hit_tokens = int(self.last_cache_hit_tokens or 0) + cache_hit

        audio = int(prompt_details.get("audio_tokens", 0) or 0)
        self.last_audio_tokens = int(self.last_audio_tokens or 0) + audio
        if audio:
            self.total_audio_tokens += audio

        # Phase-4: cache_discount (OR's per-call line item showing how
        # much the cache hit saved -- positive = savings vs cold call).
        cache_discount = usage.get("cache_discount")
        if isinstance(cache_discount, (int, float)) and not isinstance(cache_discount, bool):
            self.last_cache_discount_usd = float(self.last_cache_discount_usd or 0.0) + float(cache_discount)
            self.total_cache_discount_usd += float(cache_discount)

        # Tally by the model that served this POST (set from the envelope before usage is recorded), for the estimate.
        served = catalogue_id(self.last_upstream_model or self.model_name)
        tally = self.__dict__.setdefault("_usage_by_model", {}).setdefault(self._pricing_model(served), [0, 0, 0, 0, 0])
        completion_details = usage.get("completion_tokens_details") or {}
        tally[0] += int(usage.get("prompt_tokens") or 0)
        tally[1] += cache_hit
        tally[2] += int(usage.get("completion_tokens") or 0)
        tally[3] += int(completion_details.get("reasoning_tokens") or 0)
        tally[4] += cache_write

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
            citations = url_citations(msg.get("annotations"))
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
        """Return the USD cost per 1M input tokens for ``model``."""
        return _per_token_cost_pair(catalogue_id(model))[0]

    def _output_cost_per_1m(self, model: str) -> float:
        """Return the USD cost per 1M output tokens for ``model``."""
        return _per_token_cost_pair(catalogue_id(model))[1]

    def _resolve_pricing(self, model: str) -> Pricing:
        """Return the catalogue-derived :class:`Pricing` for ``model``, cached-input rate included.

        This provider tracks cache-hit tokens (``last_cache_hit_tokens``) but used to inherit the
        base ``_cache_hit_cost_per_1m``, which prices every cached prompt token at the full input
        rate -- a long-context conversation with a high cache-hit ratio reported several times its
        true cost. ``pricing.input_cache_read`` is the catalogue's own cached-input rate; when the
        catalogue does not publish one, ``cache_hit`` stays None and the base fallback (the
        uncached input rate, a known over-estimate) applies as before.
        ``cache_write`` comes from ``pricing.input_cache_write`` the same way, so every caller of this record prices
        cache writes at the write rate, not only the per-model session estimate.
        """
        model = catalogue_id(model)
        in_cost, out_cost = _per_token_cost_pair(model)
        return Pricing(in_cost, out_cost, _cache_read_cost_per_1m_or_none(model), _cache_write_cost_per_1m_or_none(model))

    def get_session_cost(self) -> dict[str, Any]:
        """Return cumulative usage. Adds OR-specific fields on top of base.

        ``total_cost_usd`` (from base) is the per-token *estimate* via
        /models pricing. ``actual_cost_usd`` is what OR actually billed —
        preferred when reconciling spend.
        ``upstream_inference_cost_usd`` populates only on BYOK calls and
        gives you the bare upstream price, separate from any OR markup.
        """
        base = super().get_session_cost()
        estimate = self._estimate_by_served_model()
        if estimate is None:
            # Null, not a confident 0: the per-token estimate is unavailable (catalogue outage, or
            # a model absent from /models), and a dashboard summing total_cost_usd across
            # providers used to show this whole process as free (2026-09-03 audit F37).
            # ``actual_cost_usd`` below stays truthful either way.
            base["input_cost_usd"] = None
            base["output_cost_usd"] = None
            base["total_cost_usd"] = None
            base["pricing_available"] = False
        else:
            base["input_cost_usd"], base["output_cost_usd"] = estimate
            base["total_cost_usd"] = estimate[0] + estimate[1]
            base["pricing_available"] = True
        base["actual_cost_usd"] = self.total_actual_cost_usd
        base["last_actual_cost_usd"] = self.last_actual_cost_usd
        base["upstream_inference_cost_usd"] = self.total_upstream_inference_cost_usd
        base["last_upstream_inference_cost_usd"] = self.last_upstream_inference_cost_usd
        base["cache_write_tokens"] = self.total_cache_write_tokens
        base["audio_tokens"] = self.total_audio_tokens
        return base

    def last_call_summary(self) -> dict[str, Any]:
        """Snapshot of every metric captured for the most recent ``generate()``.

        Every field is backed by a ``PerCallAttr``, so this is the calling context's own last
        call. After ``generate_batch()`` the outer caller's context never ran a call and the whole
        dict reads as defaults, rather than mixing one request's real cost with another's tokens
        -- read the per-request values off the yielded batch dict instead.

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

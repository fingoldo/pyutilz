"""xAI (Grok) LLM provider using OpenAI-compatible chat/completions API."""

from __future__ import annotations

import logging
from typing import Any

from tenacity import retry, retry_if_exception

from pyutilz.llm._openai_compat_http import _is_retryable_http_error, parse_response_envelope, wait_honoring_retry_after
from pyutilz.llm._retry import INFINITE_RETRY_KWARGS
from pyutilz.llm.base import longest_prefix_lookup, normalize_thinking
from pyutilz.llm.config import get_llm_settings
from pyutilz.dev.logginglib import ONCE, log_throttle
from pyutilz.llm.exceptions import LLMProviderError, LLMTruncationError
from pyutilz.llm._pricing import LongContextTier
from pyutilz.llm.openai_compat import OpenAICompatibleProvider, Pricing

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (USD): (input, output), the <200K-token tier.
# Source: https://docs.x.ai/docs/models (fetched 2026-09-26). Every model there bills a request of 200K prompt
# tokens or more at DOUBLE every rate (input, cached input and output); see `_LONG_CONTEXT_THRESHOLD`.
# The grok-4.20 "-beta" and grok-4 / grok-3 rows are no longer on that page; last published rates kept for old logs.
_MAX_TOKENS: dict[str, int] = {
    "grok-4.7": 30000,
    "grok-4.6": 30000,
    "grok-4.5": 30000,
    "grok-4.3": 30000,
    "grok-4.20-0309-reasoning": 30000,
    "grok-4.20-0309-non-reasoning": 30000,
    "grok-4.20-multi-agent-0309": 30000,
    "grok-build-0.1": 30000,
    "grok-4.20-beta": 30000,
    "grok-4.20-multi-agent-beta-0309": 30000,
    "grok-4.20-beta-0309-reasoning": 30000,
    "grok-4.20-beta-0309-non-reasoning": 30000,
    "grok-4-1-fast-reasoning": 30000,
    "grok-4-1-fast-non-reasoning": 30000,
    "grok-4-fast-reasoning": 30000,
    "grok-4-fast-non-reasoning": 30000,
    "grok-4": 30000,
    "grok-4-0709": 30000,
    "grok-code-fast-1": 30000,
    "grok-3": 30000,
    "grok-3-mini": 30000,
}

_PRICING: dict[str, tuple[float, float]] = {
    # Current lineup (docs.x.ai/docs/models, 2026-09-26).
    "grok-4.7": (2.00, 6.00),
    "grok-4.6": (2.00, 6.00),
    "grok-4.5": (2.00, 6.00),
    "grok-4.3": (1.25, 2.50),
    "grok-4.20-0309-reasoning": (1.25, 2.50),
    "grok-4.20-0309-non-reasoning": (1.25, 2.50),
    "grok-4.20-multi-agent-0309": (1.25, 2.50),
    "grok-build-0.1": (1.00, 2.00),
    # grok-4.20 beta ids: ALIASES of the 0309 releases in GET /v1/language-models (2026-10-03), billed at their rates.
    # They used to be priced at the old $2/$6 beta tariff.
    "grok-4.20-beta": (1.25, 2.50),
    "grok-4.20-multi-agent-beta-0309": (1.25, 2.50),
    "grok-4.20-beta-0309-reasoning": (1.25, 2.50),
    "grok-4.20-beta-0309-non-reasoning": (1.25, 2.50),
    # grok-4 fast family — cheapest, 2M context
    "grok-4-1-fast-reasoning": (0.20, 0.50),
    "grok-4-1-fast-non-reasoning": (0.20, 0.50),
    "grok-4-fast-reasoning": (0.20, 0.50),
    "grok-4-fast-non-reasoning": (0.20, 0.50),
    # grok-4 premium reasoning, 256K context (pricing increases >128K)
    "grok-4": (3.00, 15.00),
    "grok-4-0709": (3.00, 15.00),
    # Legacy Grok 3
    "grok-3": (3.00, 15.00),
    "grok-3-mini": (0.30, 0.50),
    # An alias of grok-build-0.1 in GET /v1/language-models (2026-10-03): served AND billed as it (a live call
    # returned "model": "grok-build-0.1" and cost_in_usd_ticks at the build rates, 1.7x what the old row said).
    "grok-code-fast-1": (1.00, 2.00),
}

_CACHE_HIT_COST: dict[str, float] = {
    "grok-4.7": 0.50,
    "grok-4.6": 0.50,
    "grok-4.5": 0.30,
    "grok-4.3": 0.20,
    "grok-4.20-0309-reasoning": 0.20,
    "grok-4.20-0309-non-reasoning": 0.20,
    "grok-4.20-multi-agent-0309": 0.20,
    "grok-build-0.1": 0.20,
    "grok-4.20-beta": 0.20,
    "grok-4.20-multi-agent-beta-0309": 0.20,
    "grok-4.20-beta-0309-reasoning": 0.20,
    "grok-4.20-beta-0309-non-reasoning": 0.20,
    "grok-4-1-fast-reasoning": 0.05,
    "grok-4-1-fast-non-reasoning": 0.05,
    "grok-4-fast-reasoning": 0.05,
    "grok-4-fast-non-reasoning": 0.05,
    "grok-4": 0.75,
    "grok-4-0709": 0.75,
    "grok-3": 0.75,
    "grok-3-mini": 0.07,
    "grok-code-fast-1": 0.20,
}

# A request with this many prompt tokens or more is billed at the long-context tier: every rate doubled, for all tokens
# of the request (https://docs.x.ai/docs/models; the threshold is GET /v1/language-models' ``long_context_threshold``).
_LONG_CONTEXT_THRESHOLD = 200_000
_LONG_CONTEXT_TIER = LongContextTier(_LONG_CONTEXT_THRESHOLD, 2.0, 2.0, 2.0, 2.0, inclusive=True)

# `reasoning_effort` levels as GET /v1/language-models reports them (``capabilities.reasoning_effort``, 2026-10-03):
# grok-4.5 / 4.6 / 4.7 take low|medium|high|xhigh (reasoning is mandatory, so "off" becomes ``low``); grok-4.3 also
# takes ``none``, which is what "off" becomes there. grok-4.3 had no row, so ``thinking=`` was never sent to it.
# grok-4.20-multi-agent also accepts the field but it sets the AGENT COUNT there, so it is never sent to it.
_EFFORT_LEVELS: dict[str, tuple[str, ...]] = {
    "grok-4.7": ("low", "medium", "high", "xhigh"),
    "grok-4.6": ("low", "medium", "high", "xhigh"),
    "grok-4.5": ("low", "medium", "high", "xhigh"),
    "grok-4.3": ("none", "low", "medium", "high", "xhigh"),
}

# Server-side tool prices, USD per invocation or per fetched item (https://docs.x.ai/docs/pricing, 2026-10-03:
# web search $5 / 1k calls, code execution $5 / 1k, collections/file search $2.50 / 1k, X search $5 / 1k posts and
# $10 / 1k profiles). Keyed by the ``usage.server_side_tool_usage_details`` field that counts them.
_TOOL_PRICES_USD: dict[str, float] = {
    "web_search_calls": 0.005,
    "code_interpreter_calls": 0.005,
    "file_search_calls": 0.0025,
    "document_search_calls": 0.0025,
    "x_posts_fetched": 0.005,
    "x_users_fetched": 0.010,
}

# ``usage.cost_in_usd_ticks``: the amount xAI billed for the request, in units of 1e-10 USD (a live grok-4.3 call of
# 91 uncached + 128 cached prompt and 213 output tokens reported 6,718,500 ticks = $0.00067185, its list price).
_USD_PER_TICK = 1e-10


class XAIProvider(OpenAICompatibleProvider):
    """xAI Grok provider via OpenAI-compatible chat/completions API."""

    _base_url = "https://api.x.ai/v1"
    _provider_name = "xAI"
    _max_tokens_map = _MAX_TOKENS
    _default_max_tokens = 30000
    _context_window_map = {  # noqa: RUF012 -- intentional shared class-level lookup table (overrides base class default), not a per-instance mutable-default bug
        "grok-4.7": 500_000,
        "grok-4.6": 500_000,
        "grok-4.5": 500_000,
        "grok-4.3": 1_000_000,
        "grok-4.20-0309-reasoning": 1_000_000,
        "grok-4.20-0309-non-reasoning": 1_000_000,
        "grok-4.20-multi-agent-0309": 1_000_000,
        "grok-build-0.1": 256_000,
        "grok-4": 256_000,
        "grok-4-0709": 256_000,
        "grok-3": 131_000,
        "grok-3-mini": 131_000,
    }
    _default_context_window = 2_000_000

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "grok-4.3",
        max_concurrent: int = 10,
        live_search: bool | str = False,
        live_search_max_sources: int | None = None,
        return_citations: bool = True,
    ):
        """``live_search``: let the model search the web and X while answering. ``True``, ``"on"`` or ``"auto"``
        enable it, ``False`` / ``"off"`` do not. It runs through the Responses API's server-side ``web_search`` and
        ``x_search`` tools: the chat-completions ``search_parameters`` block this used to send was retired on
        2026-01-12 and now answers ``410 Gone``, which failed every call made with the option on. The model decides
        when to search, so ``"on"`` can no longer force a search. ``live_search_max_sources`` has no counterpart in
        the tools API and is ignored with a warning; citations are always returned, into ``last_citations``.
        """
        settings = get_llm_settings()
        resolved_key = api_key or (settings.xai_api_key.get_secret_value() if settings.xai_api_key else None)
        if not resolved_key:
            raise ValueError("xAI API key not provided. " "Set XAI_API_KEY in .env or pass api_key=")
        super().__init__(api_key=resolved_key, model=model, max_concurrent=max_concurrent)
        self._live_search = live_search not in (False, None, "", "off")
        if self._live_search and live_search_max_sources is not None:
            logger.warning("xAI search tools take no source cap; live_search_max_sources=%r is ignored", live_search_max_sources)
        self._live_search_max_sources = live_search_max_sources
        self._return_citations = return_citations
        # Server-side tool spend (live search), and the total xAI itself reported (``cost_in_usd_ticks``).
        self._tool_cost_usd = 0.0
        self._reported_cost_usd = 0.0
        # The model the API says served the requests, when it differs from the requested id (see `_resolve_pricing`).
        self._served_model: str | None = None

    def _get_timeout(self, model: str) -> float:
        """Return the request timeout in seconds, using a longer timeout for reasoning-mode models whose chain-of-thought generation can be slow."""
        # Reasoning-mode variants need long timeout (chain-of-thought can be slow).
        # Note: substring check would mistakenly match "non-reasoning" to reasoning,
        # so explicitly exclude that suffix first.
        if model.endswith("-non-reasoning"):
            return 240.0
        if "reasoning" in model or model in ("grok-4", "grok-4-0709", "grok-4.20-beta") or model in _EFFORT_LEVELS:
            return 1200.0
        return 240.0

    def _compute_billed_output(self, completion_tokens: int, reasoning_tokens: int) -> int:
        """Return the total output tokens billed by xAI: completion plus reasoning.

        Measured live 2026-10-03: on chat completions ``completion_tokens`` EXCLUDES the reasoning tokens
        (grok-4.3: prompt 219 + completion 3 + reasoning 210 = total_tokens 432), and the sum priced at the output
        rate reproduces the billed ``cost_in_usd_ticks`` exactly. The Responses API counts differently (its
        ``output_tokens`` includes reasoning) and is mapped onto this convention in `_unwrap_responses_output`.
        """
        return completion_tokens + reasoning_tokens

    def _thinking_request_field(self, thinking: bool | str) -> dict[str, Any] | None:
        """``{"reasoning_effort": ...}`` on the models that take it; None elsewhere (with a warning when reasoning was asked for).

        ``thinking=`` used to be dropped for xAI entirely. Off becomes ``low``, the lowest level: reasoning cannot be
        disabled on these models. An effort above what the model takes clamps to its highest level.
        """
        enabled, effort = normalize_thinking(thinking)
        levels = longest_prefix_lookup(self.model_name, _EFFORT_LEVELS, None)
        if levels is None:
            if enabled:
                logger.warning("xAI %r takes no reasoning_effort; thinking=%r is not sent.", self.model_name, thinking)
            return None
        if not enabled:
            return {"reasoning_effort": levels[0]}
        requested = "high" if effort is None else effort
        wanted = {"minimal": "low", "max": "xhigh"}.get(requested, requested)
        if wanted not in levels:
            wanted = levels[-1] if wanted == "xhigh" else "high"
        return {"reasoning_effort": wanted}

    def _track_provider_specific_response(self, data: dict[str, Any]) -> None:
        """Note the model that actually served the request when it is not the one asked for.

        xAI silently redirects retired and alias ids: ``grok-4-1-fast-reasoning`` is answered by ``grok-4.3`` and
        ``grok-code-fast-1`` by ``grok-build-0.1`` (live, 2026-10-03), and bills at the SERVING model's rates, so a
        session priced by the requested id understated a redirected grok-4-1-fast call fivefold.
        """
        super()._track_provider_specific_response(data)
        served = data.get("model")
        if not isinstance(served, str) or not served or served == self.model_name:
            return
        if getattr(self, "_served_model", None) != served:
            logger.warning("xAI served %r with %r; pricing this session at %r rates", self.model_name, served, served)
        self._served_model = served

    def _track_provider_specific_usage(self, usage: dict[str, Any]) -> None:
        """Record the billed total xAI reports (the long-context tier is charged by the shared base, `_pricing`)."""
        super()._track_provider_specific_usage(usage)
        ticks = usage.get("cost_in_usd_ticks")
        if isinstance(ticks, (int, float)) and not isinstance(ticks, bool):
            self._reported_cost_usd = getattr(self, "_reported_cost_usd", 0.0) + ticks * _USD_PER_TICK

    def _long_context_tier(self, model: str) -> LongContextTier:
        """Every xAI model doubles every rate for a request whose prompt reaches 200K tokens."""
        return _LONG_CONTEXT_TIER

    def get_session_cost(self) -> dict[str, Any]:
        """Session cost including the long-context tier (summed per call by the base) and server-side tool calls.

        ``reported_cost_usd`` is the sum of what xAI itself said it billed (``usage.cost_in_usd_ticks``), for
        reconciling against ``total_cost_usd``, which is computed from the price tables.
        """
        cost = super().get_session_cost()
        tools = getattr(self, "_tool_cost_usd", 0.0)
        cost["tool_cost_usd"] = tools
        cost["total_cost_usd"] += tools
        cost["reported_cost_usd"] = getattr(self, "_reported_cost_usd", 0.0)
        return cost

    async def generate(  # type: ignore[override]  # same parameters as the base; overridden to route live search
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = 0.7,
        max_tokens: int = 0,
        json_mode: bool = False,
        thinking: bool | str | None = None,
        json_schema: dict[str, Any] | None = None,
        images: list[str] | None = None,
    ) -> str:
        """Chat completions, or with live search on, the Responses API with the ``web_search`` + ``x_search`` tools."""
        if not self._live_search:
            text: str = await super().generate(prompt, system, temperature, max_tokens, json_mode, thinking, json_schema, images)
            return text
        if json_mode or json_schema is not None or images:
            logger.warning("xAI live search goes through the Responses API; json_mode / json_schema / images are not sent with it")
        answer: str = await self._generate_with_search(prompt, system, temperature, max_tokens, thinking)
        return answer

    @retry(  # type: ignore[call-overload]  # tenacity's retry() overloads can't be resolved through a **dict unpack; correct at runtime
        retry=retry_if_exception(_is_retryable_http_error),
        **{**INFINITE_RETRY_KWARGS, "wait": wait_honoring_retry_after(INFINITE_RETRY_KWARGS["wait"])},
    )
    async def _generate_with_search(self, prompt: str, system: str | None, temperature: float | None, max_tokens: int, thinking: bool | str | None) -> str:
        """One ``POST /responses`` with the search tools (https://docs.x.ai/docs/guides/tools/search-tools)."""
        self._reset_per_call_state()
        await self._async_prepare()
        if max_tokens <= 0:
            max_tokens = self.max_output_tokens
        max_tokens = self.fit_max_tokens_to_context(max_tokens, prompt, system)
        messages: list[dict[str, Any]] = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body: dict[str, Any] = {
            "model": self.model_name,
            "input": messages,
            "tools": [{"type": "web_search"}, {"type": "x_search"}],
            "max_output_tokens": max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        effort = self._thinking_request_field(thinking) if thinking is not None else None
        if effort is not None:
            body["reasoning"] = {"effort": effort["reasoning_effort"]}
        async with self.semaphore:
            resp = await self._client.post("/responses", json=body, timeout=self._timeout_for({"max_tokens": max_tokens, "model": self.model_name}))
            self._capture_rate_limit_headers(resp.headers)
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                raise LLMProviderError(f"xAI API error {resp.status_code}: {resp.text[:500]}")
            resp.raise_for_status()
            data = parse_response_envelope(resp, self._provider_name)
        return self._unwrap_responses_output(data)

    def _unwrap_responses_output(self, data: dict[str, Any]) -> str:
        """Answer text, usage and citations out of a Responses API body."""
        self._track_provider_specific_response(data)
        usage = data.get("usage") or {}
        if usage:
            # The Responses usage names mapped onto the chat-completions ones `_record_usage` reads. Unlike chat
            # completions, Responses ``output_tokens`` INCLUDES the reasoning tokens (live 2026-10-03: input 38,134 +
            # output 1,051 = total 39,185, of which 814 reasoning), so the reasoning count is taken out of the
            # completion count here; passing both through billed those 814 tokens twice.
            output = int(usage.get("output_tokens") or 0)
            reasoning = min(output, int((usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0))
            mapped: dict[str, Any] = {
                "prompt_tokens": usage.get("input_tokens") or 0,
                "completion_tokens": output - reasoning,
                "prompt_tokens_details": {"cached_tokens": (usage.get("input_tokens_details") or {}).get("cached_tokens") or 0},
                "completion_tokens_details": {"reasoning_tokens": reasoning},
            }
            if "cost_in_usd_ticks" in usage:
                mapped["cost_in_usd_ticks"] = usage["cost_in_usd_ticks"]
            self._record_usage(mapped)
            # Server-side tool calls are billed on top of tokens: that same call cost $0.0250 more than its tokens,
            # which is its 5 web_search calls at $5 / 1k.
            tools = usage.get("server_side_tool_usage_details")
            if isinstance(tools, dict):
                self._tool_cost_usd = getattr(self, "_tool_cost_usd", 0.0) + sum(
                    float(tools.get(field) or 0) * price for field, price in _TOOL_PRICES_USD.items()
                )
        texts: list[str] = []
        annotations: list[Any] = []
        for item in data.get("output") or []:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
                    texts.append(part["text"])
                    annotations.extend(part.get("annotations") or [])
        citations = data.get("citations")
        self.last_citations = citations if isinstance(citations, list) else annotations
        self._last_finish_reason = data.get("status")
        text = "".join(texts)
        if data.get("status") == "incomplete":
            reason = (data.get("incomplete_details") or {}).get("reason")
            raise LLMTruncationError(f"xAI response incomplete ({reason})", finish_reason=str(reason), partial_text=text)
        if not text:
            raise LLMProviderError(f"xAI/{self.model_name} returned no output text (status={data.get('status')!r})")
        return text

    async def get_account_credits(self) -> dict:
        """Not supported for xAI: always raises NotImplementedError since credit balance is only available via the console.x.ai dashboard, not a public API."""
        # As of May 2026, the public xAI REST docs (api.x.ai + management-api.x.ai)
        # don't list any endpoint for remaining credit; the management API
        # only handles key/ACL CRUD. Balance is dashboard-only.
        raise NotImplementedError("xAI has no public API to fetch remaining credit. " "Check console.x.ai for credit balance and usage.")

    # check_account_limits is inherited: every xAI response carries x-ratelimit-{limit,remaining}-{requests,tokens}
    # (live, 2026-10-03), which the base captures. The override that always raised NotImplementedError hid them.

    def _warn_unknown_model_once(self, model: str) -> None:
        """Log a one-time warning that pricing for ``model`` is unknown and the fast-tier fallback is used."""
        if not log_throttle(f"xai.unknown_model:{model}", ONCE):
            return
        logger.warning(
            "xAI pricing for %r is unknown; falling back to the grok-4-fast tariff. Cost estimates may be off.",
            model,
        )

    def _resolve_pricing(self, model: str) -> Pricing:
        """Return the :class:`Pricing` record (input, output, cache_hit) USD per 1M for ``model``.

        Longest-prefix resolution, then the fast-tier default WITH a one-time warning: an
        unrecognized or dated snapshot id used to silently take the cheapest tariff in the table,
        so session cost under-reported by several times with nothing in the log.
        """
        model = self._billed_model(model)
        pair = _PRICING.get(model)
        if pair is None:
            pair = longest_prefix_lookup(model, _PRICING, None)
        if pair is None:
            self._warn_unknown_model_once(model)
            pair = (0.20, 0.50)
        return Pricing(float(pair[0]), float(pair[1]), self._resolve_cache_hit(model), long_context=_LONG_CONTEXT_TIER)

    def _billed_model(self, model: str) -> str:
        """The id to price ``model`` by: the model xAI reported serving this session's requests when ``model`` is the
        requested one and the tables know the served id, else ``model`` itself."""
        served = getattr(self, "_served_model", None)
        if served and model == getattr(self, "model_name", None) and longest_prefix_lookup(served, _PRICING, None) is not None:
            return str(served)
        return model

    def _resolve_cache_hit(self, model: str) -> float:
        """Return the cached-input price per 1M for ``model`` from xAI's separate cache-hit table."""
        model = self._billed_model(model)
        resolved = _CACHE_HIT_COST.get(model)
        if resolved is None:
            resolved = longest_prefix_lookup(model, _CACHE_HIT_COST, None)
        if resolved is None:
            self._warn_unknown_model_once(model)
            return 0.05
        return float(resolved)

    def _input_cost_per_1m(self, model: str) -> float:
        """Return the input token price per 1M tokens for the given model."""
        return self._resolve_pricing(model).input

    def _output_cost_per_1m(self, model: str) -> float:
        """Return the output token price per 1M tokens for the given model."""
        return self._resolve_pricing(model).output

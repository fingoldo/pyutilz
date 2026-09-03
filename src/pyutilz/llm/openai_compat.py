"""Base class for OpenAI-compatible LLM providers (DeepSeek, xAI, etc.).

Consolidates the shared HTTP + retry + token-tracking logic that was
duplicated across deepseek_provider.py and xai_provider.py.
Subclasses override a small set of hooks for provider-specific behaviour.
"""

from __future__ import annotations

import asyncio
import logging
import random
from abc import abstractmethod
from typing import Any, NamedTuple, Optional

import httpx
from tenacity import retry, retry_if_exception

from ._messages import build_chat_messages
from pyutilz.llm.exceptions import LLMProviderError, LLMTruncationError
from pyutilz.llm._retry import INFINITE_RETRY_KWARGS, MAX_RETRY_ATTEMPTS
from pyutilz.llm.base import LLMProvider, PerCallAttr, normalize_thinking
from pyutilz.llm._openai_compat_http import (  # noqa: F401  -- re-exported: this module stays the public facade for these helpers
    _NON_RETRYABLE_STATUSES,
    _JSONDecodeError,
    _accumulate_stream_tool_calls,
    _is_retryable_http_error,
    _json_backend,
    _json_loads,
    parse_response_envelope,
    parse_retry_after,
)

logger = logging.getLogger(__name__)


class Pricing(NamedTuple):
    """One provider-independent pricing record, USD per 1M tokens.

    The ONE tuple contract every provider's ``_resolve_pricing`` returns. It exists because the
    same private method name used to carry two different shapes in sibling providers -- xAI's
    ``(input, output)`` and DeepSeek's ``(input, cache_hit, output)`` -- so the accessors indexed
    ``[1]`` and ``[2]`` for the same quantity. Nothing raises when those shapes get copied across:
    both positions hold a float, and the only symptom is a silently wrong USD figure. Named fields
    make the mix-up unrepresentable.

    ``cache_hit`` is None when the provider publishes no cached-input rate; the base
    ``_cache_hit_cost_per_1m`` then falls back to the uncached input rate.
    """

    input: float
    output: float
    cache_hit: Optional[float] = None


class OpenAICompatibleProvider(LLMProvider):
    """Base for providers exposing an OpenAI-compatible chat/completions API.

    Subclasses MUST define:
        _base_url          — e.g. "https://api.deepseek.com"
        _provider_name     — e.g. "DeepSeek"
        _max_tokens_map    — dict[model_name, max_output_tokens]
        _default_max_tokens — fallback when model not in map
        _input_cost_per_1m(model)  — input pricing
        _output_cost_per_1m(model) — output pricing
        _cache_hit_cost_per_1m(model) — cache-hit input pricing

    Subclasses MAY override:
        _handle_special_status(resp) — raise on provider-specific HTTP codes
        _compute_billed_output(compl, reasoning) — billing semantics
        _get_timeout(model) — request timeout in seconds
    """

    # ── subclass configuration (override in subclass) ────────────────
    _base_url: str
    _provider_name: str
    _max_tokens_map: dict[str, int] = {}  # noqa: RUF012 -- intentional shared class-level lookup table (subclasses override with their own), not a per-instance mutable-default bug
    _default_max_tokens: int = 8192

    # Per-call "last successful call" state -- backed by contextvars via PerCallAttr, NOT plain
    # instance attributes. Regression fix (2026-07-21 audit round 2, HIGH): generate_batch() fires
    # N concurrent self.generate() calls on one shared/cached provider instance
    # (llm.factory.get_llm_provider's cache); a plain attribute write from one in-flight request
    # was visible to every other concurrently-running request reading the same attribute (see
    # PerCallAttr's docstring in base.py for the confirmed repro). A direct, non-batched
    # ``await provider.generate(...)`` is unaffected -- no task boundary is crossed between the
    # write and the caller's immediately-following read.
    _last_usage: PerCallAttr = PerCallAttr(lambda: {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0})
    _last_finish_reason: PerCallAttr = PerCallAttr(lambda: None)
    last_tool_calls: PerCallAttr = PerCallAttr(list)
    last_citations: PerCallAttr = PerCallAttr(list)
    # Same treatment: as a plain attribute this flag reported another concurrent call's
    # strict-schema outcome, so a caller skipping enum validation on True could accept
    # unvalidated output.
    _last_json_schema_applied: PerCallAttr = PerCallAttr(lambda: False)

    # True when a json_mode call came back with NO content and was re-issued without response_format.
    last_json_mode_fallback: PerCallAttr = PerCallAttr(bool)

    # Rate-limit headers of the response THIS call received (audit F32): as a plain attribute on a
    # shared instance, a concurrent request's 429 window was readable as this one's.
    last_rate_limits: PerCallAttr = PerCallAttr(dict)

    _PERCALL_METADATA_ATTRS: tuple[str, ...] = (*LLMProvider._PERCALL_METADATA_ATTRS, "_last_json_schema_applied", "last_json_mode_fallback")

    def __init__(
        self,
        api_key: str,
        model: str,
        max_concurrent: int = 10,
    ):
        self.api_key = api_key
        self.model_name = model
        self._max_concurrent = max_concurrent

        timeout_s = self._get_timeout(model)
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout_s, connect=10.0),
        )

        # Cumulative token counters
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cache_hit_tokens = 0
        self.total_reasoning_tokens = 0
        self._call_count = 0
        # Per-call usage/tool_calls/citations/finish_reason: PerCallAttr class-level descriptors
        # (declared above __init__) provide the defaults; nothing to initialize here.
        # ``last_rate_limits`` is a PerCallAttr declared above -- captured automatically from
        # x-ratelimit-* (OpenAI-family) and the legacy ratelimit-* form some providers use, and
        # read from ``check_account_limits()``. Nothing to initialize here.

    # ── hooks for subclasses ─────────────────────────────────────────

    def _get_timeout(self, model: str) -> float:
        """Return request timeout in seconds. Override for model-specific."""
        return 120.0

    #: Seconds of headroom to allow per requested output token. A 30 tok/s floor is deliberately
    #: pessimistic: the cost of guessing high is a request that hangs a little longer before it is
    #: retried, and the cost of guessing low is every long generation dying at the same wall.
    _seconds_per_output_token: float = 1.0 / 30.0

    def _timeout_for(self, body: dict[str, Any]) -> float:
        """Request timeout in seconds, taking the LARGER of the model heuristic and what this body asks for.

        `_get_timeout` classifies by model NAME, which cannot see how much output was requested: a
        `z-ai/glm-5.3-flash` asked for 54,853 tokens matches no "slow tier" substring and got the 240 s
        default, so every one of nine captures died in a ReadTimeout storm while the model was still
        generating - 35,185 tokens on the one that finished, which no 240 s budget could ever cover. The
        floor stays: a small request on a slow-tier model keeps its long allowance.
        """
        base = self._get_timeout(self.model_name)
        requested = body.get("max_tokens") or body.get("max_completion_tokens") or 0
        try:
            needed = float(requested) * self._seconds_per_output_token
        except (TypeError, ValueError):
            return base
        return max(base, needed)

    def _handle_special_status(self, resp: httpx.Response) -> None:
        """Raise on provider-specific non-retryable HTTP status codes.

        Called *before* the generic raise_for_status().
        """
        pass

    def _compute_billed_output(self, completion_tokens: int, reasoning_tokens: int) -> int:
        """Return the number of output tokens that count toward billing.

        DeepSeek: reasoning ⊂ completion → return completion_tokens
        xAI:      reasoning separate     → return completion + reasoning
        """
        return completion_tokens

    def _track_provider_specific_usage(self, usage: dict[str, Any]) -> None:
        """Hook for providers exposing extra usage fields beyond OpenAI's standard.

        Called after standard token tracking, with the raw ``usage`` dict from
        the response. OpenRouter uses this to capture ``usage.cost`` (USD
        billed by upstream — authoritative for meta-provider routing where
        per-token tables don't apply). Default: no-op.
        """
        return None

    def _track_provider_specific_response(self, data: dict[str, Any]) -> None:
        """Hook for response-level metadata outside the ``usage`` block.

        Called once per response with the full decoded JSON. OpenRouter uses
        this to record the generation ``id`` (for async ``/generation``
        lookup), the actual upstream ``provider`` that served the request,
        and the resolved ``model`` (which may differ from the requested one
        when ``models_fallback`` kicked in). Default: no-op.
        """
        return None

    # No ``_reset_per_call_state`` override: LLMProvider's derives the reset from
    # ``_PERCALL_METADATA_ATTRS``. The hand-written version that used to live here reset only
    # five of the seven attributes it declared -- ``last_json_mode_fallback`` latched True for the
    # rest of the context (2026-09-03 audit F09), which is exactly the drift the derived
    # implementation makes impossible.

    async def _async_prepare(self) -> None:
        """Async hook called (only when ``max_tokens<=0``, i.e. before ``self.max_output_tokens``
        is read) at the START of every ``generate()`` / ``generate_stream()``, BEFORE the sync
        ``self.max_output_tokens`` property is accessed.

        Default no-op. A subclass whose ``max_output_tokens``/``context_window`` properties do
        network I/O on a cache miss (e.g. OpenRouterProvider's catalogue fetch) should override
        this to pre-warm that cache via a genuinely async path (``asyncio.to_thread`` or an async
        HTTP client) -- properties can't themselves be async, so without this hook a sync network
        call reachable from an async property blocks the WHOLE event loop with no ``await`` point.
        """
        return None

    def _capture_rate_limit_headers(self, headers: Any) -> None:
        """Snapshot rate-limit headers from the most recent response.

        OpenAI-family providers send ``x-ratelimit-{requests,tokens,
        input-tokens,output-tokens}-{limit,remaining,reset}``; some send
        the legacy ``ratelimit-*`` (no x prefix). We capture both forms
        case-insensitively. Read via ``check_account_limits()``.
        """
        if headers is None:
            return
        try:
            mapping = {k.lower(): v for k, v in dict(headers).items()}
        except Exception as e:
            logger.debug("Could not parse response headers for rate-limit capture: %s", e)
            return
        # Assigned unconditionally: the snapshot describes THIS response, so a response carrying
        # no rate-limit headers must read as "none", not silently keep the previous call's window.
        self.last_rate_limits = {k: v for k, v in mapping.items() if k.startswith("x-ratelimit-") or k.startswith("ratelimit-")}

    async def check_account_limits(self) -> dict[str, Any]:
        """Return rate-limit info from the most recent response headers.

        Most OpenAI-compatible upstreams (OpenAI, xAI, DeepSeek) lack a
        standalone introspection endpoint, so per-call headers are the
        only source. After at least one ``generate()`` call has
        succeeded, this returns the captured snapshot. Subclasses with
        a real introspection endpoint (OpenRouter) override.
        """
        # ``getattr`` not direct attr — covers tests that bypass __init__.
        rl = getattr(self, "last_rate_limits", {}) or {}
        if not rl:
            raise NotImplementedError(
                f"{self._provider_name}: no rate-limit snapshot captured yet -- "
                "issue at least one generate() call first; the headers are "
                "captured automatically. Standalone introspection endpoints "
                "are not exposed for regular keys on this provider."
            )
        out: dict[str, Any] = {"raw": dict(rl)}
        for key, value in rl.items():
            short = key.replace("x-ratelimit-", "").replace("ratelimit-", "").replace("-", "_")
            out[short] = value
        return out

    @abstractmethod
    def _input_cost_per_1m(self, model: str) -> float:
        """Return USD cost per 1M input tokens for ``model``. Implemented by subclasses per their pricing table."""
        ...

    @abstractmethod
    def _output_cost_per_1m(self, model: str) -> float:
        """Return USD cost per 1M output tokens for ``model``. Implemented by subclasses per their pricing table."""
        ...

    def _resolve_pricing(self, model: str) -> Pricing:
        """Return the :class:`Pricing` record for ``model``.

        Providers with a pricing table override this and implement the three accessors in terms of
        it. The default derives the record from the accessors, so a provider that only implements
        those keeps working unchanged.
        """
        return Pricing(self._input_cost_per_1m(model), self._output_cost_per_1m(model))

    def _cache_hit_cost_per_1m(self, model: str) -> float:
        """Return the cached-input price per 1M tokens, falling back to the uncached input rate
        when the provider publishes no cached-input rate (``Pricing.cache_hit`` is None)."""
        cache_hit = self._resolve_pricing(model).cache_hit
        if cache_hit is None:
            return self._input_cost_per_1m(model)
        return float(cache_hit)

    # ── LLMProvider interface ────────────────────────────────────────

    @property
    def max_output_tokens(self) -> int:
        """Max output tokens for the current model, from ``_max_tokens_map`` or ``_default_max_tokens``.

        Exact lookup: a subclass whose default is deliberately LARGER than its listed entries
        (xAI, where the unlisted fast models carry the 2M window) must not have a shorter family
        prefix silently override it. Subclasses whose table is prefix-safe override this.
        """
        return self._max_tokens_map.get(self.model_name, self._default_max_tokens)

    # Subclasses override for per-model context windows
    _context_window_map: dict[str, int] = {}  # noqa: RUF012 -- intentional shared class-level lookup table (subclasses override with their own), not a per-instance mutable-default bug
    _default_context_window: int = 128_000

    @property
    def context_window(self) -> int:
        """Context window size for the current model, from ``_context_window_map`` or ``_default_context_window``.

        Exact lookup, for the same reason as :attr:`max_output_tokens`.
        """
        return self._context_window_map.get(self.model_name, self._default_context_window)

    def supports_json_mode(self) -> bool:
        """All OpenAI-compatible Chat Completions endpoints accept
        ``response_format={"type": "json_object"}`` since 2023-11.
        Subclasses with model-specific gating (notably OpenRouter, where
        per-model support varies) override this with a catalogue check.
        """
        return True

    def supports_json_schema(self) -> bool:
        """OpenAI-compatible endpoints have accepted strict ``json_schema`` response formats since
        2024-08. Support is per-model in practice, so routers (OpenRouter) override with a catalogue
        check; a direct single-vendor endpoint can assume its own models.
        """
        return True

    def _response_format(self, json_mode: bool, json_schema: dict[str, Any] | None) -> dict[str, Any] | None:
        """Choose the strictest response_format the model actually supports, and record which applied.

        A caller passing ``json_schema`` to a model without strict-schema support degrades to plain JSON
        mode rather than failing the call — a mixed-model sweep stays runnable — but the degradation is
        logged and exposed via ``last_json_schema_applied`` so the caller can tell a guaranteed-shape
        response from a merely-hopeful one instead of assuming the guarantee held.
        """
        self._last_json_schema_applied = False
        if json_schema is not None:
            if self.supports_json_schema():
                self._last_json_schema_applied = True
                return {"type": "json_schema", "json_schema": json_schema}
            logger.warning(
                "%s/%s does not support strict json_schema; falling back to json_object (enums NOT enforced)",
                self._provider_name,
                self.model_name,
            )
        if json_mode:
            return {"type": "json_object"}
        return None

    @property
    def last_json_schema_applied(self) -> bool:
        """Whether the most recent call actually constrained generation to the caller's JSON schema."""
        return getattr(self, "_last_json_schema_applied", False)

    async def _close(self):
        """Close the underlying httpx client."""
        await self._client.aclose()

    def _messages_for(self, prompt: str, system: str | None, images: "list[str] | None") -> list[dict[str, Any]]:
        """``_build_messages``, called with the arity the request needs.

        The third argument goes only when there ARE images: ``_build_messages`` is an override point
        that a subclass or test double may still define as ``(self, prompt, system)``, so a text-only
        call must reach it with two arguments and an unchanged body.
        """
        return self._build_messages(prompt, system, images) if images else self._build_messages(prompt, system)

    def _build_messages(
        self,
        prompt: str,
        system: str | None = None,
        images: "list[str] | None" = None,
    ) -> list[dict[str, Any]]:
        """The chat ``messages`` list -- see :func:`pyutilz.llm._messages.build_chat_messages`.

        Kept as a delegating METHOD because it is a documented override point.
        """
        return build_chat_messages(prompt, system, images)

    def _extra_request_body(self, model: str) -> dict[str, Any]:
        """Return provider-specific extra fields to merge into the request body.

        Subclasses override for things like vendor-specific defaults.
        Defaults to empty so callers see vanilla OpenAI-compatible behavior.
        """
        return {}

    def _thinking_request_field(self, thinking: bool | str) -> dict[str, Any] | None:
        """Return the request-body fragment that toggles thinking mode.

        ``thinking`` accepts BOTH a plain bool (legacy) AND an effort
        string (``"low"`` / ``"medium"`` / ``"high"`` / ``"minimal"``).
        Subclasses normalise to the upstream's actual schema:

          * Effort-string upstreams (OpenRouter's unified ``reasoning``
            field, OpenAI ``reasoning_effort``) consume the literal
            string; ``True`` is mapped to a sensible default
            (``"medium"``).
          * Boolean-flag upstreams (DeepSeek V4 ``thinking.type``)
            coerce a non-empty effort string to ``True`` so
            ``thinking="high"`` still enables on those models.

        Provider-specific. Default returns ``None`` so callers see
        vanilla OpenAI-compatible behavior (no thinking control).
        DeepSeek V4 overrides to return ``{"thinking": {"type": ...}}``;
        OpenRouter overrides to return
        ``{"reasoning": {"effort": ...}}``.
        """
        return None

    @staticmethod
    def _normalize_thinking(thinking: bool | str | int) -> tuple[bool, str | None]:
        """Delegates to :func:`pyutilz.llm.base.normalize_thinking` -- kept as a
        method so existing subclass overrides and call sites keep working."""
        return normalize_thinking(thinking)

    async def generate_stream(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 0,
        json_mode: bool = False,
        thinking: bool | str | None = None,
        json_schema: dict[str, Any] | None = None,
        images: list[str] | None = None,
    ):
        """Stream the model's response token-by-token via SSE.

        Yields each content delta as a string; the caller concatenates.
        Token-usage accounting is updated only after the stream completes
        (the final ``[DONE]`` chunk carries it for OpenAI-compat APIs).

        Retry semantics: a tenacity ``@retry`` decorator is a NO-OP on an
        async-generator function (tenacity dispatches on
        ``iscoroutinefunction``, which is False for async generators), so
        retry is implemented manually below. We retry the stream-open phase
        on transient HTTP errors (``_is_retryable_http_error``) using the
        shared exponential+jitter wait. Once the FIRST content delta has
        been yielded we stop retrying and let mid-stream failures propagate
        — restarting a partially consumed stream would duplicate already-
        emitted tokens.
        """
        self._reset_per_call_state()

        # Awaited unconditionally: BOTH the auto-budget (``max_output_tokens``) and the context clamp
        # (``context_window``) are sync properties that may hit the network on a catalogue miss.
        await self._async_prepare()
        body = self._build_stream_body(prompt, system, temperature, max_tokens, json_mode, thinking, json_schema, images)

        attempt = 0
        emitted_any = False
        # One repair re-issue, mirroring _post_and_unwrap's: an endpoint that refuses
        # ``reasoning: {enabled: false}`` used to fail 100% of STREAMING calls made with
        # thinking=False (a 400/404 is non-retryable) while the identical non-streaming call
        # succeeded after one repaired re-issue (2026-09-03 audit F14).
        body_repaired = False
        # Usage is RECORDED ONCE, after the stream closes, from the last usage block seen.
        # Recording per chunk double-counted spend and call count on upstreams that emit
        # cumulative usage on more than the final chunk, and again whenever a stream that had
        # already carried a usage block was retried by the loop below.
        usage_recorded = False
        while True:
            attempt += 1
            latest_usage: dict[str, Any] | None = None
            try:
                async with self.semaphore:
                    async with self._client.stream(
                        "POST", "/chat/completions", json=body,
                    ) as resp:
                        self._capture_rate_limit_headers(resp.headers)
                        self._handle_special_status(resp)
                        resp.raise_for_status()
                        first_chunk: dict[str, Any] | None = None
                        last_chunk: dict[str, Any] | None = None
                        # Truncation / tool-call / citation state, previously never written on the
                        # streaming path at all: _reset_per_call_state() zeroed them at the top and
                        # nothing here filled them in, so a stream cut off by max_tokens returned
                        # normally with _last_finish_reason None and the documented "double
                        # max_tokens and re-issue" contract never engaged, while a tool-call-only
                        # reply reported last_tool_calls == [] as fact (2026-09-03 audit F11).
                        streamed_text: list[str] = []
                        tool_call_fragments: dict[int, dict[str, Any]] = {}
                        async for line in resp.aiter_lines():
                            if not line or not line.startswith("data:"):
                                continue
                            data_part = line[5:].strip()
                            if data_part == "[DONE]":
                                break
                            try:
                                chunk = _json_loads(data_part)
                            except _JSONDecodeError:
                                continue
                            if first_chunk is None:
                                first_chunk = chunk
                            last_chunk = chunk
                            # Usage block tends to arrive on a chunk with empty
                            # choices AFTER the last content delta; track it
                            # whenever it's seen.
                            usage = chunk.get("usage")
                            if usage:
                                latest_usage = usage
                            content = self._apply_stream_chunk(chunk, tool_call_fragments)
                            if content:
                                emitted_any = True
                                streamed_text.append(content)
                                yield content
                        # Response-level metadata (id, model, provider) usually rides on the FIRST
                        # chunk; some upstreams send it on the last. BOTH are consulted now --
                        # passing only the last chunk (typically the usage-only trailer, whose
                        # choices are empty) left last_generation_id None after every streamed
                        # call, so the fetch_generation_stats() reconciliation OpenRouter's own
                        # docstring recommends for streams raised ValueError (audit F36).
                        for meta_chunk in (first_chunk, last_chunk):
                            if meta_chunk is not None:
                                self._track_provider_specific_response(meta_chunk)
                        if tool_call_fragments:
                            self.last_tool_calls = [tool_call_fragments[i] for i in sorted(tool_call_fragments)]
                if latest_usage is not None and not usage_recorded:
                    usage_recorded = True
                    self._track_streaming_usage(latest_usage)
                if self._last_finish_reason == "length":
                    raise LLMTruncationError(
                        f"{self._provider_name} streamed response truncated by max_tokens (finish_reason='length')",
                        finish_reason="length",
                        partial_text="".join(streamed_text),
                    )
                return
            except Exception as exc:
                # Only the stream-open / pre-first-token phase is safely
                # retryable. After we've yielded content, re-raise so the
                # caller doesn't receive duplicated tokens.
                if isinstance(exc, LLMTruncationError):
                    raise
                if not emitted_any and not body_repaired:
                    repaired = await self._repaired_stream_body(exc, body)
                    if repaired is not None:
                        body_repaired = True
                        body = repaired
                        logger.warning(
                            "%s/%s rejected a streaming request parameter - re-opening the stream once with it adjusted.",
                            self._provider_name, self.model_name,
                        )
                        continue
                if emitted_any or not _is_retryable_http_error(exc):
                    raise
                if MAX_RETRY_ATTEMPTS != 0 and attempt >= MAX_RETRY_ATTEMPTS:
                    raise
                # Mirror _retry.RETRY_WAIT: exponential 5->10->20->...->300s
                # (multiplier=2, min=5, max=300) plus random jitter [0,5).
                backoff = min(300.0, max(5.0, 2.0 * (2 ** (attempt - 1)))) + random.uniform(0, 5)  # nosec B311 - retry-jitter timing only, not security/cryptographic use
                # Honour a server-supplied Retry-After when present on a 429,
                # taking the larger of (server hint, exponential floor) so we
                # never retry sooner than the server asked and re-trigger the
                # rate limit. This is the one live caller of parse_retry_after.
                resp_obj = getattr(exc, "response", None)
                server_hint = parse_retry_after(resp_obj)
                wait_s = max(backoff, server_hint) if server_hint is not None else backoff
                logger.warning(
                    "Streaming attempt %d failed (%s: %s), retrying in %.0fs...",
                    attempt, type(exc).__name__, str(exc)[:200], wait_s,
                )
                await asyncio.sleep(wait_s)

    def _build_stream_body(
        self,
        prompt: str,
        system: str | None,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        thinking: bool | str | None,
        json_schema: dict[str, Any] | None,
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        """Assemble the ``/chat/completions`` request body for a STREAMING call.

        Split out of ``generate_stream`` so the streaming state machine stays within the project's
        C901 budget; the caller must already have awaited ``_async_prepare()``, because the
        ``max_output_tokens`` / ``context_window`` properties read here can hit the catalogue.
        """
        if max_tokens <= 0:
            max_tokens = self.max_output_tokens
        # `fit_max_tokens_to_context`'s reserve now scales with input size (see `base.py`'s
        # `_context_reserve_tokens`), not a flat 1024-token constant - fixes a measured incident
        # (2026-08-08, autopsia pilot): a large prompt's real token count exceeded `count_tokens`'s
        # estimate enough to overflow the context window even after clamping. Omitting `max_tokens`
        # entirely instead of clamping was tried and reverted: OpenRouter's own docs confirm there is no
        # universal fixed default across models when the field is absent, so widening the reserve keeps
        # the guaranteed, provider-agnostic behavior instead of an unproven cross-model assumption.
        max_tokens = self.fit_max_tokens_to_context(max_tokens, prompt, system)

        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": self._messages_for(prompt, system, images),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        rf = self._response_format(json_mode, json_schema)
        if rf is not None:
            body["response_format"] = rf
        body.update(self._extra_request_body(self.model_name))
        if thinking is not None:
            tf = self._thinking_request_field(thinking)
            if tf is not None:
                body.update(tf)

        # OR + many OpenAI-compat upstreams emit usage on the FINAL
        # SSE chunk only when ``stream_options: {"include_usage": true}``
        # is set. Without it the stream never publishes usage at all,
        # leaving streaming callers with zero cost / token tracking.
        body.setdefault("stream_options", {"include_usage": True})
        return body

    def _apply_stream_chunk(self, chunk: dict[str, Any], tool_call_fragments: dict[int, dict[str, Any]]) -> str | None:
        """Fold one SSE chunk's choice into per-call metadata, returning its content delta if any.

        Finish reason, citations and tool-call fragments are recorded here; the caller keeps the
        usage block and the first/last-chunk bookkeeping, which are read off the envelope rather
        than the choice.
        """
        choices = chunk.get("choices") or []
        if not choices:
            return None
        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        if finish_reason:
            self._last_finish_reason = finish_reason
        chunk_citations = chunk.get("citations")
        if chunk_citations:
            self.last_citations = list(chunk_citations)
        delta = choice.get("delta") or {}
        _accumulate_stream_tool_calls(tool_call_fragments, delta.get("tool_calls"))
        content = delta.get("content")
        return content if isinstance(content, str) else None

    async def _repaired_stream_body(self, exc: Exception, body: dict[str, Any]) -> dict[str, Any] | None:
        """Return a repaired request body for a STREAM the upstream refused over a parameter, else ``None``.

        Streaming counterpart of the ``_body_after_rejected_request`` consultation in
        ``_post_and_unwrap``: a streamed error response's body has not been read yet, so it is
        pulled here before the detail is handed to the same provider hook. No delta can have been
        yielded on a 4xx, so re-opening the stream cannot duplicate already-emitted tokens.
        """
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None)
        if resp is None or status is None:
            return None
        detail = ""
        try:
            raw = await resp.aread()
            detail = raw.decode("utf-8", errors="replace")
        except Exception as read_exc:
            logger.debug("Could not read the refused stream's error body: %s", read_exc)
        return self._body_after_rejected_request(body, int(status), detail)

    def _record_usage(self, usage: dict[str, Any]) -> None:
        """Shared token-usage accounting for both ``generate()`` and
        ``generate_stream()``.

        Updates cumulative totals, ``_last_usage``, invokes the
        provider-specific usage hook, and logs the same cumulative-usage
        message from both call paths — this used to be duplicated
        (``generate()`` inline vs. ``_track_streaming_usage()``) and had
        drifted (only the non-streaming path logged; the cache_hit/
        reasoning-token default computation differed in style).
        """
        # ``or 0``, matching the cache_hit/reasoning lines below: an upstream or proxy that emits
        # an explicit JSON ``null`` for either field used to yield None here and blow up on the
        # ``+=`` a few lines down with total_prompt_tokens already advanced and _call_count not --
        # a TypeError no retry predicate matches, failing a call the model already billed
        # (2026-09-03 audit F12).
        prompt_tok = usage.get("prompt_tokens") or 0
        compl_tok = usage.get("completion_tokens") or 0
        # DeepSeek reports cache hits under the legacy "prompt_cache_hit_tokens" key. OpenAI's
        # actual Chat Completions response (and xAI, whose API is explicitly OpenAI-compatible)
        # instead nests it under "prompt_tokens_details.cached_tokens" -- a DIFFERENT key one
        # level deeper. Without this fallback, OpenAIProvider/XAIProvider never see any cache-hit
        # tokens at all (total_cache_hit_tokens stays 0 forever), making their entire
        # _CACHE_HIT_COST pricing table unreachable dead code and silently over-reporting cost by
        # treating every cache-hit token as a full-price cache-miss.
        prompt_details = usage.get("prompt_tokens_details") or {}
        cache_hit = usage.get("prompt_cache_hit_tokens") or prompt_details.get("cached_tokens", 0) or 0
        details = usage.get("completion_tokens_details", {}) or {}
        reasoning_tok = details.get("reasoning_tokens", 0) or 0

        self.total_prompt_tokens += prompt_tok
        self.total_completion_tokens += compl_tok
        self.total_cache_hit_tokens += cache_hit
        self.total_reasoning_tokens += reasoning_tok
        self._call_count += 1

        self._last_usage = {
            "input_tokens": prompt_tok,
            "output_tokens": self._compute_billed_output(compl_tok, reasoning_tok),
            "reasoning_tokens": reasoning_tok,
        }

        self._track_provider_specific_usage(usage)

        logger.info(
            "%s [call #%d] %d prompt (%d cached) + %d completion" "%s | cumulative: %d in, %d out",
            self._provider_name,
            self._call_count,
            prompt_tok,
            cache_hit,
            compl_tok,
            f" ({reasoning_tok} reasoning)" if reasoning_tok else "",
            self.total_prompt_tokens,
            self.total_completion_tokens,
        )

    def _track_streaming_usage(self, usage: dict[str, Any]) -> None:
        """Mirror of the non-streaming usage path, for streaming responses.

        Called when an SSE chunk's ``usage`` field is non-empty (OpenAI-
        compat upstreams send it on the final chunk when
        ``stream_options.include_usage=true``). Delegates to
        ``_record_usage`` so both paths stay in sync.
        """
        self._record_usage(usage)

    @retry(  # type: ignore[call-overload]  # tenacity's retry() overloads can't be resolved through a **dict unpack; correct at runtime
        retry=retry_if_exception(_is_retryable_http_error),
        **INFINITE_RETRY_KWARGS,
    )
    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 0,
        json_mode: bool = False,
        thinking: bool | str | None = None,
        json_schema: dict[str, Any] | None = None,
        images: list[str] | None = None,
    ) -> str:
        """Generate text using OpenAI-compatible chat/completions API.

        ``json_schema``: a strict OpenAI-style schema dict (``{"name":..., "strict": True, "schema":
        {...}}``). When the model supports it, generation is CONSTRAINED to that schema, so closed enums
        cannot be violated; otherwise the call degrades to plain JSON mode with a warning and
        ``last_json_schema_applied`` stays False.

        ``thinking``: provider-specific chain-of-thought toggle. Accepts
        bool OR effort string for finer control (provider-normalised).
        - ``None`` (default): use the provider's/model's default behavior.
        - ``True``: explicitly request thinking mode (effort-string
          upstreams default to ``"medium"``; bool upstreams enable
          their flag).
        - ``False`` / ``""``: explicitly disable thinking. Useful when a
          tight ``max_tokens`` budget would otherwise be consumed entirely
          by reasoning (DeepSeek V4 returns ``finish_reason='length'``
          with an empty completion in that case).
        - ``"low" | "medium" | "high" | "minimal"``: effort string for
          providers that route to OpenRouter's unified ``reasoning.effort``
          or OpenAI's ``reasoning_effort``. Bool-flag upstreams (DeepSeek
          V4) coerce a non-empty string to ``True``.
        Providers that don't support a thinking toggle ignore this flag.
        """
        self._reset_per_call_state()
        # Awaited unconditionally: BOTH the auto-budget (``max_output_tokens``) and the context clamp
        # (``context_window``) are sync properties that may hit the network on a catalogue miss.
        await self._async_prepare()
        if max_tokens <= 0:
            max_tokens = self.max_output_tokens
        # `fit_max_tokens_to_context`'s reserve now scales with input size (see `base.py`'s
        # `_context_reserve_tokens`), not a flat 1024-token constant - fixes a measured incident
        # (2026-08-08, autopsia pilot): a real ~23,617-token prompt was undercounted by `count_tokens` at
        # ~20,864 (an ~11.8% gap), so the OLD flat reserve produced a clamped budget that, added to the
        # REAL input, still overflowed the context window by ~1,729 tokens - the upstream rejected the
        # whole call with HTTP 400 before generating a single token. A proportional reserve absorbs a
        # tokeniser-estimation error that grows with the prompt instead of staying pinned to a constant
        # that was only ever sized for small prompts. (Omitting `max_tokens` entirely instead of clamping
        # was tried and reverted: OpenRouter's own docs confirm there is no universal fixed default across
        # models when the field is absent - "the provider applies its own default," unspecified in detail
        # - and only one model/route was verified live before this fix landed. Widening the reserve keeps
        # the guaranteed, provider-agnostic behavior instead of an unproven cross-model assumption.)
        max_tokens = self.fit_max_tokens_to_context(max_tokens, prompt, system)
        rf = self._response_format(json_mode, json_schema)
        async with self.semaphore:
            body: dict[str, Any] = {
                "model": self.model_name,
                "messages": self._messages_for(prompt, system, images),
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if rf is not None:
                body["response_format"] = rf
            body.update(self._extra_request_body(self.model_name))
            if thinking is not None:
                thinking_field = self._thinking_request_field(thinking)
                if thinking_field is not None:
                    body.update(thinking_field)
            try:
                content = await self._post_and_unwrap(body)
            except LLMTruncationError as truncation:
                # A length-capped EMPTY answer under response_format is the same failure the
                # re-issue below exists for, wearing a different finish_reason: the model burned
                # the whole budget on a response_format it cannot honour and returned nothing.
                # Raising truncation first made that branch unreachable, so a caller whose
                # truncation handler only doubles max_tokens re-issued the identical broken shape
                # and paid twice (2026-09-03 audit F38). A truncation carrying partial text is a
                # genuine budget cutoff and still propagates untouched.
                if rf is None or truncation.partial_text:
                    raise
                self.last_json_mode_fallback = True
                logger.warning(
                    "%s/%s returned an EMPTY completion cut off by max_tokens under response_format "
                    "- re-issuing once without it before reporting truncation",
                    self._provider_name, self.model_name,
                )
                content = await self._post_and_unwrap({k: v for k, v in body.items() if k != "response_format"})
                if not content:
                    raise
            if content is None and rf is not None:
                # Measured live 2026-09-02 on OpenRouter ``z-ai/glm-4.7-flash``: the catalogue lists
                # ``response_format`` as supported, the call returns ``finish_reason="stop"``, ``content=None``,
                # 52 reasoning tokens (with ``reasoning.exclude`` requested) - and the same prompt without
                # ``response_format`` answers in a fenced JSON block. 58 of 60 benchmark questions were lost to
                # this before it was caught, each reported by the caller as "unparsable JSON" rather than as an
                # empty completion. The catalogue check cannot see it, so the fallback keys on the response.
                self.last_json_mode_fallback = True
                logger.warning(
                    "%s/%s returned an empty completion under response_format - re-issuing once without it",
                    self._provider_name, self.model_name,
                )
                content = await self._post_and_unwrap({k: v for k, v in body.items() if k != "response_format"})
            if content is None:
                reasoning = (self._last_usage or {}).get("reasoning_tokens")
                sent = "dropped" if self.last_json_mode_fallback else ("sent" if rf is not None else "not sent")
                raise LLMProviderError(
                    f"{self._provider_name}/{self.model_name} returned an empty completion "
                    f"(finish_reason={self._last_finish_reason!r}, reasoning_tokens={reasoning}, response_format={sent})"
                )
            return content

    def _body_after_rejected_request(self, _body: dict[str, Any], _status: int, _detail: str) -> dict[str, Any] | None:
        """A replacement body for a request this upstream refused over a PARAMETER, or ``None`` to raise.

        The catalogue says which parameters a model accepts, never which VALUES of them its resolved endpoint
        will take, so this is keyed on the rejection itself. Default: no repair, the error propagates - the
        parameters are named for the subclasses that read them.
        """
        return None

    async def _post_and_unwrap(self, body: dict[str, Any], repairing: bool = False) -> str | None:
        """One POST plus envelope unwrapping. ``None`` means the model returned no text and no tool calls -
        an empty completion, which the caller decides how to treat; a tool-call-only reply returns ``""``.

        ``repairing`` marks the single re-issue `_body_after_rejected_request` is allowed, so a repaired body
        that is itself refused raises instead of looping.
        """
        resp = await self._client.post("/chat/completions", json=body, timeout=self._timeout_for(body))

        # Snapshot rate-limit headers before any status check, so a
        # 429/5xx error response's headers are captured too — that's
        # exactly when they matter most for backoff/quota decisions.
        self._capture_rate_limit_headers(resp.headers)

        # Provider-specific status handling (e.g. DeepSeek 402)
        self._handle_special_status(resp)

        if resp.status_code in _NON_RETRYABLE_STATUSES:
            try:
                err_body = resp.json()
                detail = err_body.get("error", {}).get("message", resp.text) if isinstance(err_body, dict) else str(err_body)
            except (ValueError, _JSONDecodeError):
                detail = resp.text
            repaired = self._body_after_rejected_request(body, resp.status_code, detail)
            if repaired is not None and not repairing:
                logger.warning(
                    "%s/%s rejected a request parameter (HTTP %s) - re-issuing once with it adjusted: %s",
                    self._provider_name, self.model_name, resp.status_code, detail[:160],
                )
                return await self._post_and_unwrap(repaired, repairing=True)
            raise LLMProviderError(f"{self._provider_name} API error {resp.status_code}: {detail}")
        resp.raise_for_status()
        data = parse_response_envelope(resp, self._provider_name)

        # Token usage tracking
        usage = data.get("usage", {})
        if usage:
            self._record_usage(usage)

        self._track_provider_specific_response(data)

        choices = data.get("choices", [])
        if not choices:
            raise LLMProviderError(f"{self._provider_name} returned no choices")

        self._last_finish_reason = choices[0].get("finish_reason", "unknown")
        message = choices[0].get("message") or {}
        # Capture function-calling output before unwrapping content -- these
        # silently disappeared previously; pyutilz returned only the bare
        # text (often empty when the model chose tool_calls path).
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            self.last_tool_calls = tool_calls
        else:
            self.last_tool_calls = []
        # xAI live-search citations + OpenAI annotations (which OR also
        # uses for web-search) live on the message.
        citations = message.get("citations")
        if isinstance(citations, list):
            self.last_citations = citations
        else:
            self.last_citations = []
        content = message.get("content")
        if content is None and self.last_tool_calls:
            # Tool-call-only response (no assistant text). Return empty
            # string but keep tool_calls accessible via the attribute.
            return ""
        # ORDER MATTERS: a length-capped answer with NO text is still a truncation, and reporting it as an
        # empty completion sent the caller looking for a broken model rather than a spent budget. Measured
        # 2026-09-02: z-ai/glm-4.7-flash spent all 15,775 output tokens on reasoning and returned nothing.
        if self._last_finish_reason == "length":
            # Regression fix (2026-07-21 audit): LLMTruncationError was fully specified
            # (finish_reason field, "caller should double max_tokens and re-issue" contract)
            # but never actually raised anywhere -- callers catching it to auto-retry with a
            # bigger budget never saw it fire, even on a genuine max_tokens cutoff.
            raise LLMTruncationError(
                f"{self._provider_name} response truncated by max_tokens (finish_reason='length')",
                finish_reason=self._last_finish_reason,
                # The partial content, which used to be dropped here. A caller that catches this to
                # re-issue with a bigger budget is one caller; a caller that catches it to keep what was
                # already paid for is another, and it had nothing to keep.
                partial_text=content or "",
            )
        if not content:
            return None
        return content  # type: ignore[no-any-return]  # openai-compatible JSON payload: the message content is read out of an untyped dict

    async def generate_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 0,
        # FIFTH, matching `LLMProvider.generate_json`: after this override's own extra parameters it
        # would be a Liskov violation, and the same position would mean different things per provider.
        images: list[str] | None = None,
        force_json_mode: bool = True,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate structured JSON output.

        ``force_json_mode=True`` (default) sends ``response_format={"type":
        "json_object"}`` to the provider — strictest, safest for parsing.
        Set ``False`` for attention-check / honeypot scenarios where the
        LLM must be able to emit non-JSON sentinels like ``[REFUSE]``:
        falls back to prompt-only JSON steering plus ``extract_json``.
        """
        return await self._generate_json_via(prompt, system, temperature, max_tokens, json_mode=force_json_mode, json_schema=json_schema, images=images)

    # 2026-08-02 near-duplicate-function-body finding: generate_batch/process_request used to be
    # duplicated here near-verbatim from LLMProvider.generate_batch, EXCEPT the duplicate's except
    # branch dropped the `self._classify_batch_exception(e)` call the base version has. No
    # OpenAICompatibleProvider subclass overrides that hook today, so the drop was latent, but it
    # silently broke the extension point for any future one (anthropic/deepseek/openai/xai
    # providers) that adds one -- their override would simply never fire, since this override
    # shadowed the only generate_batch that calls it. Deleted; this class now inherits the base
    # implementation, restoring the hook and removing the duplication in one fix.

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD (cache miss pricing)."""
        input_cost = (input_tokens / 1_000_000) * self._input_cost_per_1m(self.model_name)
        output_cost = (output_tokens / 1_000_000) * self._output_cost_per_1m(self.model_name)
        return input_cost + output_cost

    def get_session_cost(self) -> dict[str, Any]:
        """Return cumulative token usage and cost breakdown for this session."""
        cache_miss = self.total_prompt_tokens - self.total_cache_hit_tokens
        input_cost = (cache_miss / 1_000_000) * self._input_cost_per_1m(self.model_name) + (
            self.total_cache_hit_tokens / 1_000_000
        ) * self._cache_hit_cost_per_1m(self.model_name)
        billed_output = self._compute_billed_output(self.total_completion_tokens, self.total_reasoning_tokens)
        output_cost = (billed_output / 1_000_000) * self._output_cost_per_1m(self.model_name)
        return {
            "calls": self._call_count,
            "prompt_tokens": self.total_prompt_tokens,
            "cache_hit_tokens": self.total_cache_hit_tokens,
            "cache_miss_tokens": cache_miss,
            "completion_tokens": self.total_completion_tokens,
            "reasoning_tokens": self.total_reasoning_tokens,
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "total_cost_usd": input_cost + output_cost,
        }

    async def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken (encoding resolved per-model, e.g. o200k_base for
        gpt-4o/o1, not always cl100k_base) or len//4 fallback. An APPROXIMATION, not exact --
        see token_counter.count_tokens's own docstring."""
        from pyutilz.llm.token_counter import count_tokens
        return count_tokens(text, model=self.model_name)

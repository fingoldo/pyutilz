"""Anthropic Claude LLM provider."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, ClassVar

import anthropic
from tenacity import retry, retry_if_exception, retry_if_exception_type

from pyutilz.llm.config import get_llm_settings
from pyutilz.llm._messages import build_anthropic_content
from pyutilz.llm._retry import INFINITE_RETRY_KWARGS
from pyutilz.llm.base import LLMProvider, PerCallAttr, normalize_thinking
from pyutilz.llm._thinking import MIN_THINKING_BUDGET, THINKING_BUDGETS  # re-exported: callers import them from here
from pyutilz.llm._thinking import CLAUDE_EFFORTS, claude_effort
from pyutilz.llm._claude_models import CLAUDE_MODELS, UNKNOWN_CLAUDE_MODEL, ClaudeModelSpec, claude_model_spec
from pyutilz.llm.exceptions import LLMProviderError, LLMRefusalError, LLMTruncationError

logger = logging.getLogger(__name__)


# Models whose API refuses a `temperature` argument. Populated at runtime by the retry in
# `generate_json`, never hardcoded -- see the note at that call site. Process-local by design: it is
# a cache of an API fact, not configuration, and it must not outlive a deployment that upgrades.
_MODELS_REJECTING_TEMPERATURE: set[str] = set()


def _is_temperature_rejection(exc: Exception) -> bool:
    """Is this the specific 400 that says the model will not accept `temperature`?

    Matched narrowly on BOTH the status and the message. A broad `except BadRequestError` here
    would swallow every other 400 -- an over-long prompt, a malformed tool block -- and retry it
    identically, turning one clear error into two and a confusing log.
    """
    if getattr(exc, "status_code", None) != 400:
        return False
    return "temperature" in str(exc).lower() and ("deprecated" in str(exc).lower() or "not supported" in str(exc).lower() or "unsupported" in str(exc).lower())


# The budget table lives in _thinking, where the Claude Code provider reads it without importing the anthropic SDK.

# Models whose thinking mode was LEARNED from a 400 to differ from what the model table says: model -> True when the
# model turned out to need adaptive thinking, False when it turned out to need the manual budget form. Process-local
# for the same reason as ``_MODELS_REJECTING_TEMPERATURE``: a cache of an API fact, never configuration.
_LEARNED_THINKING_MODE: dict[str, bool] = {}


def _thinking_mode_rejection(exc: Exception) -> bool | None:
    """The thinking mode a 400 says this model needs instead: True adaptive, False manual budget, None if not that 400.

    https://platform.claude.com/docs/en/build-with-claude/extended-thinking: a model that only thinks adaptively fails
    with a message that starts ``"thinking.type.enabled" is not supported``; a budget-only model sent ``adaptive`` also
    400s (https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting).
    """
    if getattr(exc, "status_code", None) != 400:
        return None
    text = str(exc).lower()
    if "thinking.type.enabled" in text and "not supported" in text:
        return True
    if "thinking.type.adaptive" in text and ("not supported" in text or "unsupported" in text):
        return False
    return None


def uses_adaptive_thinking(model: str) -> bool:
    """Does ``model`` take ``{"type": "adaptive"}`` (True) or the manual budget form (False)?

    An empty model name is the legacy call shape (``anthropic_thinking_field(effort, max_tokens)`` with no model) and
    keeps the budget form it always produced.
    """
    if not model:
        return False
    learned = _LEARNED_THINKING_MODE.get(model)
    if learned is not None:
        return learned
    return claude_model_spec(model, provider_label="Anthropic").adaptive_thinking


def anthropic_thinking_field(thinking: bool | str | None, max_tokens: int, *, model: str = "") -> dict[str, Any] | None:
    """The ``thinking`` fragment for a Messages request, or None when reasoning is off or cannot fit.

    Free-standing because the rule is the same whoever asks: ``AnthropicProvider._thinking_request_field`` delegates
    to it, and a caller that drives the SDK directly imports it rather than reimplementing the arithmetic and
    drifting from it.

    Two shapes, chosen per model. Claude 4.6 and later think ADAPTIVELY (``{"type": "adaptive"}``), and 4.7 and later
    reject the manual ``{"type": "enabled", "budget_tokens": N}`` form with a 400, so on every current model the budget
    form failed the whole call. The depth control for adaptive thinking is ``output_config.effort``, which
    :func:`anthropic_thinking_request` adds alongside this fragment.

    Returns None for an effort the budget table does not know, rather than guessing: silently substituting a
    different budget than the caller asked for is worse than leaving reasoning off, because the cost shows up on the
    bill either way while the caller believes their setting took effect. The budget is carved OUT of ``max_tokens``,
    so one at or above it would leave no room for the answer and the API would reject the request; ``model`` only
    names the request in the warnings and picks the shape.
    """
    enabled, effort = normalize_thinking(thinking)
    if not enabled:
        return None
    if uses_adaptive_thinking(model):
        if effort is not None and effort not in CLAUDE_EFFORTS:
            claude_effort(effort, model=model)  # warns
            return None
        return {"type": "adaptive"}
    budget = THINKING_BUDGETS.get("medium" if not effort else effort)
    if budget is None:
        logger.warning(
            "Unknown thinking effort %r for %s; leaving extended thinking off. Known: %s",
            effort,
            model if model else "the request",
            sorted(THINKING_BUDGETS),
        )
        return None
    headroom = max_tokens - MIN_THINKING_BUDGET
    if headroom < MIN_THINKING_BUDGET:
        logger.warning(
            "max_tokens=%d leaves no room for an extended-thinking budget (minimum %d plus an equal allowance for the answer); leaving it off",
            max_tokens,
            MIN_THINKING_BUDGET,
        )
        return None
    return {"type": "enabled", "budget_tokens": min(budget, headroom)}


def anthropic_thinking_request(thinking: bool | str | None, max_tokens: int, *, model: str) -> dict[str, Any]:
    """Every top-level request field a ``thinking=`` value maps to: ``thinking`` and, where the model takes it,
    ``output_config.effort``. Empty when reasoning is off or cannot be sent."""
    field = anthropic_thinking_field(thinking, max_tokens, model=model)
    if field is None:
        return {}
    out: dict[str, Any] = {"thinking": field}
    _enabled, effort = normalize_thinking(thinking)
    level = claude_effort(effort, model=model)
    spec = claude_model_spec(model, provider_label="Anthropic")
    if level is not None and spec.supports_effort:
        out["output_config"] = {"effort": level}
    return out


def _schema_body(json_schema: dict[str, Any]) -> dict[str, Any]:
    """The bare JSON schema out of either shape callers pass: OpenAI's ``{"name", "strict", "schema"}`` wrapper
    (what ``OpenAICompatibleProvider.generate`` takes) or the schema itself."""
    inner = json_schema.get("schema")
    return inner if isinstance(inner, dict) and ("name" in json_schema or "strict" in json_schema) else json_schema


def _int_field(obj: Any, name: str) -> int:
    """An int usage field off an SDK object or dict; 0 when absent or not an int (mocks, older SDKs)."""
    value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider with async support and retry logic."""

    _provider_name = "Anthropic"

    # Explicit per-request timeout, mirroring OpenAICompatibleProvider._get_timeout's default, so
    # the effective ceiling is one this package controls rather than the SDK's own default. With a
    # streamed request (see `_STREAMING_THRESHOLD_TOKENS`) this bounds each read, not the whole answer.
    _request_timeout_seconds: float = 120.0

    # The largest `max_tokens` sent as one buffered request. The SDK refuses a non-streaming request it
    # expects to run past ten minutes, which it estimates from `max_tokens` (about 21,333 on the installed
    # SDK), and the docs advise streaming for long outputs. Above this the request is STREAMED and the final
    # message assembled from the stream, so a caller can ask for the model's full 128K output.
    _STREAMING_THRESHOLD_TOKENS = 21_000

    # Pricing per 1M tokens: (input, output), DERIVED from the shared Claude model table
    # (pyutilz.llm._claude_models, sourced from https://platform.claude.com/docs/en/about-claude/pricing).
    # Kept as a class attribute for callers and tests that read it; `_get_pricing` resolves through the
    # table's own matcher, which only accepts an exact ID or a snapshot suffix of one.
    _PRICING: dict[str, tuple[float, float]] = {  # noqa: RUF012 -- intentional shared class-level pricing table, not a per-instance mutable-default bug
        key: (spec.input_per_1m, spec.output_per_1m) for key, spec in CLAUDE_MODELS.items()
    }
    _DEFAULT_PRICING = (UNKNOWN_CLAUDE_MODEL.input_per_1m, UNKNOWN_CLAUDE_MODEL.output_per_1m)

    # Per-call "last successful call" state -- backed by contextvars via PerCallAttr, NOT plain
    # instance attributes. Regression fix (2026-07-21 audit round 2, HIGH): see identical
    # PerCallAttr usage + docstring in openai_compat.py / base.py -- generate_batch() fires N
    # concurrent self.generate() calls on one shared/cached provider instance, so a plain
    # attribute write from one in-flight request used to be visible to every other
    # concurrently-running request. Cumulative session totals (total_input_tokens etc.) are
    # NOT converted -- they are intentionally shared/summed across all calls.
    _last_usage: PerCallAttr = PerCallAttr(lambda: {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0})
    _last_finish_reason: PerCallAttr = PerCallAttr(lambda: None)
    last_cache_creation_input_tokens: PerCallAttr = PerCallAttr(lambda: 0)
    last_cache_read_input_tokens: PerCallAttr = PerCallAttr(lambda: 0)
    last_thinking_tokens: PerCallAttr = PerCallAttr(lambda: 0)
    last_thinking_tokens_estimated: PerCallAttr = PerCallAttr(lambda: False)
    # The refusal details the API attaches to `stop_reason == "refusal"` (`{"type", "category"}`), else None.
    last_stop_details: PerCallAttr = PerCallAttr(lambda: None)
    # Response-scoped, same reasoning as every attribute above (audit F32).
    last_rate_limits: PerCallAttr = PerCallAttr(dict)
    last_organization_id: PerCallAttr = PerCallAttr(lambda: None)

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-5",
        max_concurrent: int = 5,
        cache_ttl: str = "5m",
    ):
        """``cache_ttl``: lifetime of the system-prompt cache entry, ``"5m"`` (writes billed 1.25x input) or ``"1h"``
        (writes billed 2x input, worth it when the same system prompt is reused over more than five minutes)."""
        settings = get_llm_settings()
        self.api_key = api_key or (settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None)
        if not self.api_key:
            raise ValueError("Anthropic API key not provided. Set ANTHROPIC_API_KEY in .env or pass api_key=")
        if cache_ttl not in ("5m", "1h"):
            raise ValueError(f"cache_ttl must be '5m' or '1h', got {cache_ttl!r}")

        self.model = model
        self.cache_ttl = cache_ttl
        # max_retries=0: _retry.py's tenacity policy is the single retry authority here. The SDK
        # default (2 internal retries) multiplied every tenacity attempt, so a sustained 529
        # produced several times the upstream calls PYUTILZ_LLM_MAX_RETRIES documents. An explicit
        # timeout replaces the SDK default, matching every other provider's pinned timeout.
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key, max_retries=0, timeout=self._request_timeout_seconds)
        self._max_concurrent = max_concurrent
        # Cumulative session accounting (mirrors OpenAICompatibleProvider).
        # ``get_session_cost`` reports these across ALL calls in the session;
        # without them it would silently report calls=0 and the last call's
        # tokens only. Incremented in ``generate()`` once usage is parsed.
        self._call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_creation_input_tokens = 0
        # The part of total_cache_creation_input_tokens written to the 1-hour cache, billed at 2x input, not 1.25x.
        self.total_cache_creation_1h_input_tokens = 0
        self.total_cache_read_input_tokens = 0
        self.total_thinking_tokens = 0
        # Message Batches spend, already at the 50% batch discount: its tokens are billed at a different rate,
        # so they are costed per batch rather than folded into the synchronous totals above.
        self.total_batch_cost_usd = 0.0
        # Per-call usage/cache/thinking/finish_reason: PerCallAttr class-level descriptors
        # (declared above __init__) provide the defaults; nothing to initialize here.
        # ``last_rate_limits`` / ``last_organization_id`` are PerCallAttr descriptors declared
        # above, captured from each response's headers; nothing to initialize here.

    # Kept as class attributes for subclasses and tests that read them; the table itself lives at module level
    # beside anthropic_thinking_field, which is where the rule is implemented.
    _THINKING_BUDGETS: ClassVar[dict[str, int]] = THINKING_BUDGETS
    _MIN_THINKING_BUDGET = MIN_THINKING_BUDGET

    @property
    def _spec(self) -> ClaudeModelSpec:
        """This model's row in the shared Claude table (an unknown model gets the documented fallback and a warning)."""
        return claude_model_spec(getattr(self, "model", "") or "", provider_label="Anthropic")

    def _thinking_request_field(self, thinking: bool | str | None, max_tokens: int) -> dict[str, Any] | None:
        """The ``thinking`` request fragment for this provider's model, or None when reasoning is off or cannot fit."""
        return anthropic_thinking_field(thinking, max_tokens, model=self.model)

    def _get_pricing(self) -> tuple[float, float]:
        """``(input, output)`` USD per 1M for this model, from the shared Claude table."""
        spec = self._spec
        return (spec.input_per_1m, spec.output_per_1m)

    @property
    def max_output_tokens(self) -> int:
        """Maximum output tokens of ``self.model`` (synchronous Messages API limit), from the shared Claude table."""
        return self._spec.max_output

    @property
    def context_window(self) -> int:
        """Context window of ``self.model``: 1M on Claude 4.6 and later, 200K before. It used to be 200K for every
        model, so ``fit_max_tokens_to_context`` clamped or refused prompts a current model accepts."""
        return self._spec.context_window

    def supports_json_mode(self) -> bool:
        """Anthropic Messages API has NO native JSON-mode toggle. We
        get reliable JSON via ``generate_json()`` (the steer plus
        ``extract_json``, or a strict schema where the model supports
        one), not by passing a kwarg to ``generate()``. Callers should
        branch: if False, use ``generate_json()`` instead of passing
        ``json_mode=True``."""
        return False

    def supports_json_schema(self) -> bool:
        """True where the model takes structured outputs (``output_config.format``), which CONSTRAINS generation to
        the schema: https://platform.claude.com/docs/en/build-with-claude/structured-outputs lists the models."""
        return self._spec.supports_structured_output

    @retry(  # type: ignore[call-overload]  # tenacity's retry() overloads can't be resolved through a **dict unpack; correct at runtime
        # Regression fix (2026-07-21 audit): OverloadedError (529), ServiceUnavailableError (503),
        # and DeadlineExceededError (504) are SEPARATE, SIBLING subclasses of APIStatusError in the
        # installed SDK -- NOT subclasses of InternalServerError -- so the previous tuple silently
        # never retried Anthropic's own documented "always retry with backoff" 529 overloaded
        # condition. Retrying on the status-code set directly (rather than enumerating leaf
        # classes) also survives the SDK adding new status-specific exception subclasses later.
        retry=retry_if_exception_type((
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
        )) | retry_if_exception(
            lambda e: isinstance(e, anthropic.APIStatusError) and getattr(e, "status_code", None) in {429, 500, 502, 503, 504, 529}
        ),
        **INFINITE_RETRY_KWARGS,
    )
    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = 0.7,
        max_tokens: int = 0,
        thinking: bool | str | None = False,
        images: list[str] | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        """Generate text using Claude.

        Args:
            prompt: The user message to send.
            system: Optional system prompt; omitted from the request when None.
            temperature: Sampling temperature passed straight through to the API. ``None`` sends no
                ``temperature`` field, so the API applies its own default: a caller comparing providers
                must be able to ask for "no temperature" and get it, rather than this library's 0.7.
            max_tokens: Output-token ceiling; 0 means "derive it", and any value is clamped
                to what the model's context leaves after the prompt. Above
                ``_STREAMING_THRESHOLD_TOKENS`` the request is streamed.
            images: URLs or ``data:`` URIs to show the model, in Anthropic's own block shape --
                NOT the OpenAI ``image_url`` form, which this API rejects with a 400. Absent or
                empty, the request body is byte-identical to the text-only one it has always sent.

                This parameter was missing while `LLMProvider.generate_json` already forwarded
                `images` to `generate`, so on any deployment configured for Anthropic a posting
                with a picture attachment raised ``TypeError: generate() got an unexpected keyword
                argument 'images'`` -- the whole evaluation, not just the picture. Found by a live
                integration test on 2026-09-04; no unit test could see it, because the parameter's
                absence is exactly what the mocks were modelled on.
            thinking: Reasoning toggle. ``False`` (default) keeps the
                previous behaviour exactly. ``True`` turns it on at the model's
                default depth; an effort string ("minimal"/"low"/"medium"/"high"/
                "xhigh"/"max") selects one. Adaptive-thinking models get
                ``thinking.type="adaptive"`` plus ``output_config.effort``; older ones a
                token budget. Until this parameter existed the caller-side
                ``thinking=`` flag was DROPPED for Anthropic: llm_client only
                forwards it to providers whose signature declares it, so every
                Anthropic call ran without reasoning regardless of the setting,
                including the ones whose docstrings claimed to be disabling it.
            json_schema: A JSON schema (bare, or OpenAI's ``{"name", "strict", "schema"}`` wrapper) the answer must
                conform to, sent as ``output_config.format`` on models that support structured outputs. On any
                other model it is not sent and a warning says so.

        Raises:
            LLMTruncationError: ``stop_reason`` was ``max_tokens`` or ``model_context_window_exceeded``.
            LLMRefusalError: ``stop_reason`` was ``refusal``; ``details["stop_details"]`` carries the category.
        """
        self._reset_per_call_state()
        if max_tokens <= 0:
            max_tokens = min(self.max_output_tokens, self._STREAMING_THRESHOLD_TOKENS)
        max_tokens = self.fit_max_tokens_to_context(max_tokens, prompt, system)
        async with self.semaphore:
            kwargs = self._request_kwargs(prompt, system, temperature, max_tokens, thinking, images, json_schema)
            response = await self._create_with_learned_repairs(kwargs, thinking)
            return self._consume_response(response)

    def _request_kwargs(
        self,
        prompt: str,
        system: str | None,
        temperature: float | None,
        max_tokens: int,
        thinking: bool | str | None,
        images: list[str] | None,
        json_schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """The Messages ``create`` keyword arguments for one call."""
        # A plain string when there are no images, so a text-only request body is byte-identical
        # to the one this provider has always sent. See `build_anthropic_content`: Anthropic's
        # image blocks are its own shape, and the OpenAI `image_url` part is rejected outright.
        messages = [{"role": "user", "content": build_anthropic_content(prompt, images)}]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        # `None` means DO NOT SEND the field; `0.0` is a real temperature, so this is not a truthiness test.
        if temperature is not None:
            kwargs["temperature"] = temperature
        thinking_kwargs = anthropic_thinking_request(thinking, max_tokens, model=self.model)
        if thinking_kwargs:
            kwargs.update(thinking_kwargs)
            # Anthropic rejects any temperature other than 1 while thinking is on. Callers pass a
            # low temperature for determinism (validation runners use 0.1), so honouring both is
            # impossible: override and say so, rather than letting the API 400 on a combination
            # the caller had no way to know was illegal. An omitted temperature already means
            # the API default of 1.
            if temperature is not None and temperature != 1:
                logger.debug("Thinking requires temperature=1; overriding the requested %.2f", temperature)
                kwargs["temperature"] = 1
        if json_schema is not None:
            if self.supports_json_schema():
                output_config = dict(kwargs.get("output_config") or {})
                output_config["format"] = {"type": "json_schema", "schema": _schema_body(json_schema)}
                kwargs["output_config"] = output_config
            else:
                logger.warning("%s does not support structured outputs; json_schema is not sent (shape NOT enforced)", self.model)
        if system:
            cache_control: dict[str, Any] = {"type": "ephemeral"}
            if getattr(self, "cache_ttl", "5m") == "1h":
                cache_control["ttl"] = "1h"
            kwargs["system"] = [{"type": "text", "text": system, "cache_control": cache_control}]

        # Newer models REJECT `temperature` outright -- claude-opus-5 answers
        # `400 invalid_request_error: \`temperature\` is deprecated for this model`. The
        # parameter is not merely ignored, so a caller that has always passed one (this
        # library's own default is 0.7) breaks the moment a deployment moves to such a model.
        #
        # Discovered 2026-09-03 the expensive way: a project switched its pipeline to
        # claude-opus-5, and every call would have 400'd in production. It surfaced only
        # because a suite of integration tests was made runnable the same day and hit it first.
        #
        # LEARNED, not hardcoded. A model list here would be wrong again the next time
        # Anthropic ships one: instead the first rejection for a given model records it in
        # `_MODELS_REJECTING_TEMPERATURE` and the call is retried once without the parameter.
        # Every later call for that model skips it outright, so the cost is one wasted request
        # per model per process.
        if self.model in _MODELS_REJECTING_TEMPERATURE:
            kwargs.pop("temperature", None)
        return kwargs

    async def _create_with_learned_repairs(self, kwargs: dict[str, Any], thinking: bool | str | None = False) -> Any:
        """Send the request, repairing (once each) a rejected ``temperature`` and a rejected thinking mode.

        The thinking repair is the table's safety net: a model the table has in the wrong mode, or a new one it does
        not know, answers the documented 400, the mode is learned for this process, and the request is rebuilt in
        the other shape. Cost: one rejected request per model per process, as for ``temperature``.
        """
        for _ in range(2):
            try:
                return await self._create(kwargs)
            except Exception as exc:  # noqa: PERF203 -- a bounded repair loop; the try/except IS the repair mechanism
                if "temperature" in kwargs and _is_temperature_rejection(exc):
                    logger.info("%s rejects `temperature`; retrying without it and omitting it for the rest of this process", self.model)
                    _MODELS_REJECTING_TEMPERATURE.add(self.model)
                    kwargs.pop("temperature", None)
                    continue
                wants_adaptive = _thinking_mode_rejection(exc) if "thinking" in kwargs else None
                if wants_adaptive is None or _LEARNED_THINKING_MODE.get(self.model) == wants_adaptive:
                    raise
                logger.info(
                    "%s rejects this thinking mode; using %s thinking for the rest of this process",
                    self.model, "adaptive" if wants_adaptive else "budgeted",
                )
                _LEARNED_THINKING_MODE[self.model] = wants_adaptive
                kwargs.pop("thinking", None)
                output_config = {k: v for k, v in (kwargs.pop("output_config", None) or {}).items() if k != "effort"}
                rebuilt = anthropic_thinking_request(thinking, kwargs["max_tokens"], model=self.model)
                output_config.update(rebuilt.pop("output_config", {}))
                kwargs.update(rebuilt)
                if output_config:
                    kwargs["output_config"] = output_config
        return await self._create(kwargs)

    async def _create(self, kwargs: dict[str, Any]) -> Any:
        """One Messages call, buffered or (above the streaming threshold) streamed; captures the response headers.

        ``with_raw_response`` exposes HTTP headers (rate-limit + org id) alongside the parsed body. Without it the
        SDK swallows headers. The streamed form carries them on ``stream.response``.
        """
        if kwargs.get("max_tokens", 0) > self._STREAMING_THRESHOLD_TOKENS:
            async with self.client.messages.stream(**kwargs) as stream:
                message = await stream.get_final_message()
                self._capture_response_headers(getattr(getattr(stream, "response", None), "headers", None))
                return message
        raw = await self.client.messages.with_raw_response.create(**kwargs)  # type: ignore[call-overload]  # anthropic's create() overloads can't be resolved through a **dict unpack; correct at runtime
        response = raw.parse()
        self._capture_response_headers(raw.headers)
        return response

    def _account_usage(self, usage: Any, content: Any) -> dict[str, Any]:
        """Per-call usage from an SDK ``usage`` object, with the thinking count the API reports.

        ``usage.output_tokens_details.thinking_tokens`` is the billed reasoning count
        (https://platform.claude.com/docs/en/build-with-claude/extended-thinking). The visible thinking text is a
        SUMMARY, so the chars//4 estimate this used to be undercounted real reasoning; it remains only as the
        fallback for a response without the field, flagged by ``last_thinking_tokens_estimated``.
        """
        details = usage.get("output_tokens_details") if isinstance(usage, dict) else getattr(usage, "output_tokens_details", None)
        reported = _int_field(details, "thinking_tokens") if details is not None else 0
        has_reported = details is not None and isinstance(
            details.get("thinking_tokens") if isinstance(details, dict) else getattr(details, "thinking_tokens", None), int
        )
        estimated = 0
        if not has_reported:
            for block in content or []:
                if getattr(block, "type", None) == "thinking":
                    estimated += max(1, len(getattr(block, "thinking", "") or "") // 4)
        creation = usage.get("cache_creation") if isinstance(usage, dict) else getattr(usage, "cache_creation", None)
        return {
            "input_tokens": _int_field(usage, "input_tokens"),
            "output_tokens": _int_field(usage, "output_tokens"),
            "reasoning_tokens": reported if has_reported else estimated,
            "reasoning_estimated": (not has_reported) and estimated > 0,
            "cache_creation_input_tokens": _int_field(usage, "cache_creation_input_tokens"),
            "cache_creation_1h_input_tokens": _int_field(creation, "ephemeral_1h_input_tokens") if creation is not None else 0,
            "cache_read_input_tokens": _int_field(usage, "cache_read_input_tokens"),
        }

    def _consume_response(self, response: Any) -> str:
        """Record one response's usage and turn it into the answer text, or the typed error its stop reason means."""
        self._last_finish_reason = response.stop_reason
        u = self._account_usage(response.usage, response.content)
        self.last_cache_creation_input_tokens = u["cache_creation_input_tokens"]
        self.last_cache_read_input_tokens = u["cache_read_input_tokens"]
        self.total_cache_creation_input_tokens += u["cache_creation_input_tokens"]
        self.total_cache_creation_1h_input_tokens = getattr(self, "total_cache_creation_1h_input_tokens", 0) + u["cache_creation_1h_input_tokens"]
        self.total_cache_read_input_tokens += u["cache_read_input_tokens"]
        self.last_thinking_tokens = u["reasoning_tokens"]
        self.last_thinking_tokens_estimated = u["reasoning_estimated"]
        self.total_thinking_tokens += u["reasoning_tokens"]
        if u["reasoning_estimated"]:
            logger.debug("Anthropic thinking tokens are estimated (chars//4=%d), not API-reported.", u["reasoning_tokens"])

        # Cumulative session totals (for get_session_cost).
        self._call_count += 1
        self.total_input_tokens += u["input_tokens"]
        self.total_output_tokens += u["output_tokens"]
        self._last_usage = {k: v for k, v in u.items() if k not in ("reasoning_estimated", "cache_creation_1h_input_tokens")}

        # EVERY text block, joined: a response with citations, or with text split around other blocks, carries several,
        # and returning the first alone silently truncated the answer.
        texts = [block.text for block in response.content or [] if getattr(block, "type", None) == "text" and isinstance(getattr(block, "text", None), str)]
        result_text = "".join(texts) if texts else None
        if self._last_finish_reason in ("max_tokens", "model_context_window_exceeded"):
            # Regression fix (2026-07-21 audit): LLMTruncationError was fully specified but
            # never actually raised anywhere -- see openai_compat.py's identical fix.
            # partial_text carries whatever was already generated (and paid for) so a caller
            # catching this can keep it, as exceptions.py documents the field for. A full context
            # window is the same outcome by another route (https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons:
            # "treat the response as truncated"), and it used to be returned as a complete answer.
            raise LLMTruncationError(
                f"Anthropic response truncated (stop_reason={self._last_finish_reason!r})",
                finish_reason=self._last_finish_reason,
                partial_text=result_text or "",
            )
        if self._last_finish_reason == "refusal":
            stop_details = getattr(response, "stop_details", None)
            if stop_details is not None and not isinstance(stop_details, dict):
                dump = getattr(stop_details, "model_dump", None)
                stop_details = dump() if callable(dump) else {"type": getattr(stop_details, "type", None), "category": getattr(stop_details, "category", None)}
            self.last_stop_details = stop_details if isinstance(stop_details, dict) else None
            raise LLMRefusalError(
                f"{self.model} declined to respond (stop_reason='refusal', stop_details={self.last_stop_details!r})",
                raw_text=result_text,
                details={"stop_details": self.last_stop_details, "model": self.model},
            )
        if result_text is None:
            raise LLMProviderError(f"Anthropic returned no text block (stop_reason={self._last_finish_reason!r})")
        return result_text

    # generate_json is inherited: LLMProvider's forwards images, thinking and json_schema to generate() exactly as the
    # override that used to live here did (the code audit flagged the two bodies as duplicates).

    def _capture_response_headers(self, headers: Any) -> None:
        """Snapshot rate-limit headers + org id from the latest response.

        ``check_account_limits()`` reads from this snapshot rather than
        raising NotImplementedError when at least one call has been made.
        Anthropic doesn't expose a standalone introspection endpoint for
        regular keys, so the per-call headers are the best source.
        """
        try:
            mapping = dict(headers) if headers is not None else {}
        except Exception as exc:
            logger.debug("Anthropic response-header capture failed: %s", exc)
            return
        # Lower-case the keys for case-insensitive lookup downstream.
        lower = {k.lower(): v for k, v in mapping.items()}
        # Assigned unconditionally: the snapshot describes THIS response, so a response carrying
        # no rate-limit headers must read as "none", not silently keep the previous call's window.
        self.last_rate_limits = {k: v for k, v in lower.items() if k.startswith("anthropic-ratelimit-")}
        org = lower.get("anthropic-organization-id")
        self.last_organization_id = org if isinstance(org, str) else None

    _count_tokens_fallback_warned = False

    async def count_tokens(
        self,
        text: str,
        system: str | None = None,
    ) -> int:
        """Count tokens via Anthropic's native ``messages.count_tokens`` API.

        Tiktoken — the previous fallback — uses OpenAI's tokenizer, which
        is the WRONG tokenizer for Claude (the cl100k_base mapping
        diverges from Claude's BPE for >5% of typical text). Cache-budget
        and prompt-fits-in-context calculations need the real number.

        Falls back to tiktoken on a transient failure (connection, timeout, 429, 5xx) so an outage does not block
        calling code, with a WARNING the first time. A permanent 4xx (bad key, unknown model, malformed request) is
        RAISED: it used to be swallowed at DEBUG, so a wrong model ID or a revoked key read as a plausible count.
        """
        try:
            messages = [{"role": "user", "content": text}]
            kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
            if system:
                kwargs["system"] = system
            result = await self.client.messages.count_tokens(**kwargs)
            return int(result.input_tokens)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if isinstance(status, int) and 400 <= status < 500 and status not in (408, 409, 429):
                raise
            if not type(self)._count_tokens_fallback_warned:
                type(self)._count_tokens_fallback_warned = True
                logger.warning("Anthropic count_tokens API failed (%s); falling back to the tiktoken approximation.", exc)
            from pyutilz.llm.token_counter import count_tokens
            return count_tokens(text)

    async def get_account_credits(self) -> dict:
        """Always raise ``NotImplementedError``: Anthropic exposes no public API for remaining balance on regular API keys."""
        # Anthropic publishes balance only via the web console — there is no
        # public API endpoint for regular ``sk-ant-api03-...`` keys. The
        # Admin API (``/v1/organizations/cost_report``) requires a separate
        # ``sk-ant-admin-...`` key and reports SPEND, not remaining balance.
        raise NotImplementedError(
            "Anthropic does not expose remaining balance via the public API for "
            "regular keys. Check console.anthropic.com/settings/billing. "
            "(Admin API reports spend, not balance, and needs sk-ant-admin- keys.)"
        )

    async def check_account_limits(self) -> dict:
        """Return rate-limit info from the most recent response headers.

        Anthropic has no standalone introspection endpoint — per-key limits
        are published as ``anthropic-ratelimit-*`` headers on every call.
        After at least one call, this returns the captured snapshot
        (limit / remaining / reset for requests, tokens, input-tokens,
        output-tokens). Before any call, raises NotImplementedError with
        a hint to issue a tiny request first.
        """
        rl = getattr(self, "last_rate_limits", {}) or {}
        if not rl:
            raise NotImplementedError(
                "No Anthropic rate-limit snapshot available yet -- issue at "
                "least one generate() call first; the headers are captured "
                "automatically. Or check console.anthropic.com/settings/limits."
            )
        # Normalize from raw header form into structured dict.
        out: dict[str, Any] = {"raw": dict(rl)}
        for key, value in rl.items():
            # anthropic-ratelimit-tokens-limit -> tokens_limit
            short = key.replace("anthropic-ratelimit-", "").replace("-", "_")
            out[short] = value
        org = getattr(self, "last_organization_id", None)
        if org:
            out["organization_id"] = org
        return out

    def _cost_usd(self, input_tokens: int, output_tokens: int, cache_write_5m: int, cache_write_1h: int, cache_read: int) -> tuple[float, float]:
        """``(input_cost, output_cost)`` in USD at this model's list rates.

        Anthropic's ``usage.input_tokens`` ALREADY EXCLUDES both cache token counts (total sent = input + cache
        creation + cache read), so they are added, never subtracted. Cache writes bill 1.25x input (5-minute) or 2x
        (1-hour); cache reads bill the model's own multiplier: 0.1x on most models, 0.05x on Opus 5.5, 0.025x on
        Fable/Mythos 5.1 (https://platform.claude.com/docs/en/about-claude/pricing). A flat 0.10 overstated a cached
        Fable 5.1 session's reads fourfold.
        """
        spec = self._spec
        in_rate, out_rate = spec.input_per_1m, spec.output_per_1m
        input_cost = (
            input_tokens * in_rate + cache_write_5m * in_rate * 1.25 + cache_write_1h * in_rate * 2.0 + cache_read * in_rate * spec.cache_read_multiplier
        ) / 1_000_000
        return input_cost, output_tokens * out_rate / 1_000_000

    def get_session_cost(self) -> dict[str, Any]:
        """Return cumulative usage including cache + thinking accounting (see ``_cost_usd`` for the rates)."""
        total_input = getattr(self, "total_input_tokens", 0)
        total_output = getattr(self, "total_output_tokens", 0)
        creation = getattr(self, "total_cache_creation_input_tokens", 0)
        creation_1h = min(getattr(self, "total_cache_creation_1h_input_tokens", 0), creation)
        cache_read = getattr(self, "total_cache_read_input_tokens", 0)
        input_cost, output_cost = self._cost_usd(total_input, total_output, creation - creation_1h, creation_1h, cache_read)
        batch_cost = getattr(self, "total_batch_cost_usd", 0.0)
        return {
            "calls": getattr(self, "_call_count", 0),
            "prompt_tokens": total_input,
            "completion_tokens": total_output,
            "thinking_tokens": getattr(self, "total_thinking_tokens", 0),
            "cache_creation_input_tokens": creation,
            "cache_creation_1h_input_tokens": creation_1h,
            "cache_read_input_tokens": cache_read,
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "batch_cost_usd": batch_cost,
            # Provider-agnostic spend reporting reads this key; every other provider's
            # get_session_cost returns it, and omitting it raised KeyError on Anthropic alone.
            "total_cost_usd": input_cost + output_cost + batch_cost,
        }

    async def generate_message_batch(
        self,
        requests: list[dict[str, Any]],
        poll_interval: float = 30.0,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        """Run ``requests`` through the Message Batches API: half the price of synchronous calls, asynchronous.

        Each request dict takes the ``generate`` arguments (``prompt`` required; ``system``, ``temperature``,
        ``max_tokens``, ``thinking``, ``json_schema``, ``images``) plus an optional ``id``. Returns one dict per
        request, in request order: ``{"id", "result"}`` or ``{"id", "error"}``, with ``usage`` when the API reports it.
        Blocks, polling every ``poll_interval`` seconds, until the batch has ended (batches finish within 24 hours) or
        ``timeout`` seconds pass, which raises ``TimeoutError`` naming the batch id so it can be collected later.
        Spend is added to ``total_batch_cost_usd`` at the documented 50% discount
        (https://platform.claude.com/docs/en/build-with-claude/batch-processing).
        """
        ids = [str(req.get("id", i)) for i, req in enumerate(requests)]
        if len(set(ids)) != len(ids):
            raise ValueError("generate_message_batch: request ids must be unique")
        batch_requests: list[Any] = []
        for custom_id, req in zip(ids, requests):
            requested = req.get("max_tokens")
            # 0 / absent both mean "derive it", matching generate()'s own max_tokens <= 0 rule.
            max_tokens = int(requested) if requested else min(self.max_output_tokens, self._STREAMING_THRESHOLD_TOKENS)
            params = self._request_kwargs(
                req["prompt"], req.get("system"), req.get("temperature", 0.7), max_tokens,
                req.get("thinking", False), req.get("images"), req.get("json_schema"),
            )
            batch_requests.append({"custom_id": custom_id, "params": params})
        batch = await self.client.messages.batches.create(requests=batch_requests)
        waited = 0.0
        while getattr(batch, "processing_status", None) != "ended":
            if timeout is not None and waited >= timeout:
                raise TimeoutError(f"Anthropic message batch {batch.id} still {batch.processing_status} after {waited:.0f}s")
            await asyncio.sleep(poll_interval)
            waited += poll_interval
            batch = await self.client.messages.batches.retrieve(batch.id)
        by_id: dict[str, dict[str, Any]] = {}
        async for entry in await self.client.messages.batches.results(batch.id):
            by_id[entry.custom_id] = self._batch_entry(entry)
        return [by_id.get(custom_id, {"id": custom_id, "error": "no result returned for this request"}) for custom_id in ids]

    def _batch_entry(self, entry: Any) -> dict[str, Any]:
        """One Message Batches result line as a ``generate_batch``-style dict, costing it at the batch discount."""
        result = entry.result
        rtype = getattr(result, "type", None)
        if rtype != "succeeded":
            error = getattr(result, "error", None)
            return {"id": entry.custom_id, "error": f"{rtype}: {error}" if error is not None else str(rtype)}
        message = result.message
        u = self._account_usage(message.usage, message.content)
        input_cost, output_cost = self._cost_usd(
            u["input_tokens"], u["output_tokens"], u["cache_creation_input_tokens"] - u["cache_creation_1h_input_tokens"],
            u["cache_creation_1h_input_tokens"], u["cache_read_input_tokens"],
        )
        self.total_batch_cost_usd = getattr(self, "total_batch_cost_usd", 0.0) + 0.5 * (input_cost + output_cost)
        usage = {k: v for k, v in u.items() if k != "reasoning_estimated"}
        texts = [b.text for b in message.content or [] if getattr(b, "type", None) == "text"]
        stop = getattr(message, "stop_reason", None)
        if stop in ("max_tokens", "model_context_window_exceeded", "refusal"):
            return {"id": entry.custom_id, "error": f"stop_reason={stop}", "partial_text": "".join(texts), "usage": usage}
        return {"id": entry.custom_id, "result": "".join(texts), "usage": usage, "finish_reason": stop}

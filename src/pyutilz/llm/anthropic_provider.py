"""Anthropic Claude LLM provider."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import anthropic
from tenacity import retry, retry_if_exception, retry_if_exception_type

from pyutilz.llm.config import get_llm_settings
from pyutilz.llm._retry import INFINITE_RETRY_KWARGS
from pyutilz.llm.base import LLMProvider, PerCallAttr, longest_prefix_lookup, normalize_thinking
from pyutilz.llm.exceptions import LLMProviderError, LLMTruncationError

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider with async support and retry logic."""

    _provider_name = "Anthropic"

    # Explicit per-request timeout, mirroring OpenAICompatibleProvider._get_timeout's default, so
    # the effective ceiling is one this package controls rather than the SDK's own default.
    _request_timeout_seconds: float = 120.0

    # Pricing per 1M tokens: (input, output)
    # Source: https://platform.claude.com/docs/en/about-claude/pricing
    # Verified against the official pricing table 2026-05-01.
    # Cache write 5m = 1.25x input, cache write 1h = 2x input,
    # cache read = 0.10x input (universal multipliers across all models).
    # ``_get_pricing`` prefix-matches via ``key.rsplit("-", 1)[0]``, so
    # entries WITHOUT a date suffix also match the date-suffixed
    # ``claude-opus-4-7-YYYYMMDD`` form. Use the unsuffixed canonical ID
    # for any model whose date is not pinned in the official pricing
    # table; pin the suffix only when the snapshot is the public API ID.
    _PRICING: dict[str, tuple[float, float]] = {  # noqa: RUF012 -- intentional shared class-level pricing table, not a per-instance mutable-default bug
        # Opus 4.5+: dropped from $15/$75 to $5/$25 (3x cheaper than legacy 4/4.1).
        "claude-opus-4-7": (5.00, 25.00),
        "claude-opus-4-6-20250610": (5.00, 25.00),
        "claude-opus-4-5-20250414": (5.00, 25.00),
        # Legacy Opus 4 / 4.1 retain old $15/$75 pricing.
        "claude-opus-4-1-20250805": (15.00, 75.00),
        "claude-opus-4-20250514": (15.00, 75.00),
        # Sonnet family — same $3/$15 across all 4.x variants.
        # As of 2026-05-01 latest is Sonnet 4.6 (no Sonnet 4.7 released).
        "claude-sonnet-4-6-20250610": (3.00, 15.00),
        "claude-sonnet-4-5-20250414": (3.00, 15.00),
        "claude-sonnet-4-20250514": (3.00, 15.00),
        "claude-sonnet-3-7-20250219": (3.00, 15.00),  # deprecated
        # Haiku 4.5: $1/$5; legacy 3.5 stays $0.80/$4; Haiku 3 = $0.25/$1.25.
        "claude-haiku-4-5-20251001": (1.00, 5.00),
        "claude-haiku-3-5-20241022": (0.80, 4.00),
        "claude-haiku-3-20240307": (0.25, 1.25),
        # Legacy Opus 3 (deprecated).
        "claude-opus-3-20240229": (15.00, 75.00),
    }
    _DEFAULT_PRICING = (3.00, 15.00)  # fallback = Sonnet pricing

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
    # Response-scoped, same reasoning as every attribute above (audit F32).
    last_rate_limits: PerCallAttr = PerCallAttr(dict)
    last_organization_id: PerCallAttr = PerCallAttr(lambda: None)

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_concurrent: int = 5,
    ):
        settings = get_llm_settings()
        self.api_key = api_key or (settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None)
        if not self.api_key:
            raise ValueError("Anthropic API key not provided. Set ANTHROPIC_API_KEY in .env or pass api_key=")

        self.model = model
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
        self.total_cache_read_input_tokens = 0
        self.total_thinking_tokens = 0
        # Per-call usage/cache/thinking/finish_reason: PerCallAttr class-level descriptors
        # (declared above __init__) provide the defaults; nothing to initialize here.
        # ``last_rate_limits`` / ``last_organization_id`` are PerCallAttr descriptors declared
        # above, captured from each response's headers; nothing to initialize here.

    # Source: https://platform.claude.com/docs/en/docs/about-claude/models
    #   Opus 4.6+: 128K, Opus 4/4.1: 32K, Sonnet family: 64K, Haiku family: 64K.
    # Resolved via the same longest-prefix matcher used for _PRICING (see that table's own
    # comment) -- a bare substring test ("4-6" in self.model) has exactly the failure mode that
    # matcher was written to avoid: any Opus release whose id doesn't literally contain "4-6"
    # (claude-opus-4-7-..., claude-opus-4-8-..., a differently-numbered future release) would
    # silently get the wrong (4x smaller) limit.
    _MAX_OUTPUT_TOKENS: dict[str, int] = {  # noqa: RUF012 -- intentional shared class-level lookup table, not a per-instance mutable-default bug
        "claude-opus-4-7": 128000,
        "claude-opus-4-6-20250610": 128000,
        "claude-opus-4-5-20250414": 32000,
        "claude-opus-4-1-20250805": 32000,
        "claude-opus-4-20250514": 32000,
        "claude-sonnet-4-6-20250610": 64000,
        "claude-sonnet-4-5-20250414": 64000,
        "claude-sonnet-4-20250514": 64000,
        "claude-sonnet-3-7-20250219": 64000,
        "claude-haiku-4-5-20251001": 64000,
        "claude-haiku-3-5-20241022": 64000,
        "claude-haiku-3-20240307": 64000,
        "claude-opus-3-20240229": 32000,
    }

    # Extended-thinking budgets per effort level. Anthropic takes a token
    # budget rather than an effort string, so the shared effort vocabulary
    # (minimal/low/medium/high, see pyutilz.llm.base.normalize_thinking) is
    # mapped here. ``True`` means "on, provider default" -> medium.
    _THINKING_BUDGETS: ClassVar[dict[str, int]] = {
        "minimal": 1024,
        "low": 2048,
        "medium": 4096,
        "high": 8192,
    }
    # Anthropic's minimum accepted budget.
    _MIN_THINKING_BUDGET = 1024

    def _thinking_request_field(self, thinking: bool | str, max_tokens: int) -> dict[str, Any] | None:
        """The ``thinking`` request fragment, or None when reasoning is off.

        Returns None for an effort the budget table does not know, rather than
        guessing: silently substituting a different budget than the caller asked
        for is worse than leaving reasoning off, because the cost shows up on the
        bill either way while the caller believes their setting took effect.
        """
        enabled, effort = normalize_thinking(thinking)
        if not enabled:
            return None
        budget = self._THINKING_BUDGETS.get(effort or "medium")
        if budget is None:
            logger.warning(
                "Unknown thinking effort %r for %s; leaving extended thinking off. Known: %s",
                effort, self.model, sorted(self._THINKING_BUDGETS),
            )
            return None
        # Anthropic requires max_tokens > budget_tokens: the budget is carved OUT
        # of the output allowance, so a budget at or above it leaves no room for
        # the answer and the API rejects the request.
        headroom = max_tokens - self._MIN_THINKING_BUDGET
        if headroom < self._MIN_THINKING_BUDGET:
            logger.warning(
                "max_tokens=%d leaves no room for an extended-thinking budget " "(minimum %d plus an equal allowance for the answer); leaving it off",
                max_tokens,
                self._MIN_THINKING_BUDGET,
            )
            return None
        return {"type": "enabled", "budget_tokens": min(budget, headroom)}

    @property
    def max_output_tokens(self) -> int:
        """Maximum output tokens for ``self.model``, looked up from the known per-family limits (Opus/Sonnet/Haiku)."""
        return int(longest_prefix_lookup(self.model, self._MAX_OUTPUT_TOKENS, 64000))

    @property
    def context_window(self) -> int:
        """Total context window size (input + output tokens) of the underlying Anthropic model."""
        return 200_000

    def supports_json_mode(self) -> bool:
        """Anthropic Messages API has NO native JSON-mode toggle. We
        get reliable JSON via ``generate_json()`` (assistant prefill
        with ``{`` + parser-side ``extract_json``), not by passing a
        kwarg to ``generate()``. Callers should branch: if False, use
        ``generate_json()`` instead of passing ``json_mode=True``."""
        return False

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
        temperature: float = 0.7,
        max_tokens: int = 0,
        thinking: bool | str = False,
    ) -> str:
        """Generate text using Claude.

        Args:
            prompt: The user message to send.
            system: Optional system prompt; omitted from the request when None.
            temperature: Sampling temperature passed straight through to the API.
            max_tokens: Output-token ceiling; 0 means "derive it", and any value is clamped
                to what the model's context leaves after the prompt.
            thinking: Extended-thinking toggle. ``False`` (default) keeps the
                previous behaviour exactly. ``True`` uses the medium budget; an
                effort string ("minimal"/"low"/"medium"/"high") selects one
                explicitly. Until this parameter existed the caller-side
                ``thinking=`` flag was DROPPED for Anthropic: llm_client only
                forwards it to providers whose signature declares it, so every
                Anthropic call ran without reasoning regardless of the setting,
                including the ones whose docstrings claimed to be disabling it.
        """
        self._reset_per_call_state()
        if max_tokens <= 0:
            max_tokens = min(self.max_output_tokens, 21000)
        max_tokens = self.fit_max_tokens_to_context(max_tokens, prompt, system)
        async with self.semaphore:
            messages = [{"role": "user", "content": prompt}]

            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages,
            }
            thinking_field = self._thinking_request_field(thinking, max_tokens)
            if thinking_field is not None:
                kwargs["thinking"] = thinking_field
                # Anthropic rejects any temperature other than 1 while extended
                # thinking is on. Callers pass a low temperature for determinism
                # (validation runners use 0.1), so honouring both is impossible:
                # override and say so, rather than letting the API 400 on a
                # combination the caller had no way to know was illegal.
                if temperature != 1:
                    logger.debug(
                        "Extended thinking requires temperature=1; overriding the requested %.2f",
                        temperature,
                    )
                    kwargs["temperature"] = 1
            if system:
                kwargs["system"] = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]

            # ``with_raw_response`` exposes HTTP headers (rate-limit + org id)
            # alongside the parsed body. Without it the SDK swallows headers.
            raw = await self.client.messages.with_raw_response.create(**kwargs)  # type: ignore[call-overload]  # anthropic's create() overloads can't be resolved through a **dict unpack; correct at runtime
            response = raw.parse()
            self._capture_response_headers(raw.headers)

            self._last_finish_reason = response.stop_reason

            usage = response.usage
            cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            # NOTE: the Anthropic usage object does not report thinking/reasoning tokens directly,
            # so this is an APPROXIMATION (chars // 4), not an API-reported count. It is flagged via
            # `last_thinking_tokens_estimated` and logged so callers know the reasoning-token figure
            # is a heuristic, not billed usage.
            # Named thinking_TOKENS: ``thinking`` is now the request-side toggle
            # parameter, and shadowing it here would silently rebind it.
            thinking_tokens = 0
            for block in response.content:
                if getattr(block, "type", None) == "thinking":
                    text = getattr(block, "thinking", "") or ""
                    thinking_tokens += max(1, len(text) // 4)  # rough estimate (chars // 4)

            self.last_cache_creation_input_tokens = cache_creation
            self.last_cache_read_input_tokens = cache_read
            self.total_cache_creation_input_tokens += cache_creation
            self.total_cache_read_input_tokens += cache_read
            self.last_thinking_tokens = thinking_tokens
            self.last_thinking_tokens_estimated = thinking_tokens > 0
            self.total_thinking_tokens += thinking_tokens
            if thinking_tokens > 0:
                logger.debug("Anthropic thinking tokens are estimated (chars//4=%d), not API-reported.", thinking_tokens)

            # Cumulative session totals (for get_session_cost).
            self._call_count += 1
            self.total_input_tokens += usage.input_tokens
            self.total_output_tokens += usage.output_tokens

            self._last_usage = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "reasoning_tokens": thinking_tokens,
                "cache_creation_input_tokens": cache_creation,
                "cache_read_input_tokens": cache_read,
            }

            # Pull text from the first text block (skip thinking blocks).
            result_text = None
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    result_text = block.text
                    break
            if result_text is None:
                # Fall back to the legacy single-block layout, guarded: content can be empty, and
                # block 0 can be a thinking/tool_use block with no .text -- both reachable exactly
                # when extended thinking consumed the whole budget, i.e. the max_tokens case
                # handled below, where an IndexError/AttributeError would mask the typed error.
                first = response.content[0] if response.content else None
                result_text = getattr(first, "text", None) if first is not None else None
            if self._last_finish_reason == "max_tokens":
                # Regression fix (2026-07-21 audit): LLMTruncationError was fully specified but
                # never actually raised anywhere -- see openai_compat.py's identical fix.
                # partial_text carries whatever was already generated (and paid for) so a caller
                # catching this can keep it, as exceptions.py documents the field for.
                raise LLMTruncationError(
                    "Anthropic response truncated by max_tokens (stop_reason='max_tokens')",
                    finish_reason=self._last_finish_reason,
                    partial_text=result_text or "",
                )
            if result_text is None:
                raise LLMProviderError(f"Anthropic returned no text block (stop_reason={self._last_finish_reason!r})")
            return result_text  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime

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

        Falls back to tiktoken on any API failure so a transient outage
        doesn't block calling code.
        """
        try:
            messages = [{"role": "user", "content": text}]
            kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
            if system:
                kwargs["system"] = system
            result = await self.client.messages.count_tokens(**kwargs)
            return int(result.input_tokens)
        except Exception as exc:
            logger.debug("Anthropic count_tokens API failed (%s); falling back to tiktoken.", exc)
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

    def get_session_cost(self) -> dict[str, Any]:
        """Return cumulative usage including cache + thinking accounting.

        Cache-aware cost: cache_read tokens billed at 0.10x input rate;
        cache_creation tokens billed at 1.25x (5min) or 2x (1h) input rate.
        We use 1.25x as the default (5min ephemeral) -- if you opt into
        1h cache, multiply ``cache_creation_input_tokens`` by 2 instead.
        """
        in_rate, out_rate = self._get_pricing()
        # Regression fix (2026-07-21 audit): Anthropic's ``usage.input_tokens`` field ALREADY
        # EXCLUDES both cache_creation_input_tokens AND cache_read_input_tokens (total tokens
        # sent = input_tokens + cache_creation_input_tokens + cache_read_input_tokens -- this is
        # Anthropic's own documented semantics). Subtracting the cumulative cache totals again
        # here double-subtracted them from a figure that never included them in the first place,
        # which `max(0, ...)` could clamp to 0 -- silently dropping the entire "fresh" input-cost
        # tier for any session where cumulative cache tokens exceed cumulative fresh input_tokens
        # (the common case for a heavily-cached agentic session).
        total_input = getattr(self, "total_input_tokens", 0)
        total_output = getattr(self, "total_output_tokens", 0)
        plain_input = total_input
        input_cost = (
            (plain_input / 1_000_000) * in_rate
            + (self.total_cache_creation_input_tokens / 1_000_000) * in_rate * 1.25
            + (self.total_cache_read_input_tokens / 1_000_000) * in_rate * 0.10
        )
        output_cost = (total_output / 1_000_000) * out_rate
        return {
            "calls": getattr(self, "_call_count", 0),
            "prompt_tokens": total_input,
            "completion_tokens": total_output,
            "thinking_tokens": self.total_thinking_tokens,
            "cache_creation_input_tokens": self.total_cache_creation_input_tokens,
            "cache_read_input_tokens": self.total_cache_read_input_tokens,
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            # Provider-agnostic spend reporting reads this key; every other provider's
            # get_session_cost returns it, and omitting it raised KeyError on Anthropic alone.
            "total_cost_usd": input_cost + output_cost,
        }

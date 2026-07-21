"""Anthropic Claude LLM provider."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import anthropic
from tenacity import retry, retry_if_exception, retry_if_exception_type

from pyutilz.llm.config import get_llm_settings
from pyutilz.llm._retry import INFINITE_RETRY_KWARGS
from pyutilz.llm.base import LLMProvider, longest_prefix_lookup
from pyutilz.llm.exceptions import LLMTruncationError

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider with async support and retry logic."""

    _provider_name = "Anthropic"

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
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        # Per-call usage (read by LLMClient after generate())
        self._last_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
        # Cumulative session accounting (mirrors OpenAICompatibleProvider).
        # ``get_session_cost`` reports these across ALL calls in the session;
        # without them it would silently report calls=0 and the last call's
        # tokens only. Incremented in ``generate()`` once usage is parsed.
        self._call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        # Per-call cache + extended-thinking accounting.
        # cache_creation_input_tokens — tokens written to ephemeral cache
        #   (5min default, 2x base rate; 1h opt-in, 1.25x).
        # cache_read_input_tokens — tokens read from cache (10% of input rate).
        # thinking_tokens — extended-thinking output (Opus 4 only).
        self.last_cache_creation_input_tokens = 0
        self.last_cache_read_input_tokens = 0
        self.total_cache_creation_input_tokens = 0
        self.total_cache_read_input_tokens = 0
        self.last_thinking_tokens = 0
        self.last_thinking_tokens_estimated = False
        self.total_thinking_tokens = 0
        # Rate-limit info captured from response headers on every call.
        # Populated lazily by ``check_account_limits()`` from the snapshot
        # of the most recent response.
        self.last_rate_limits: dict[str, str] = {}
        self.last_organization_id: str | None = None

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
    ) -> str:
        """Generate text using Claude."""
        if max_tokens <= 0:
            max_tokens = min(self.max_output_tokens, 21000)
        async with self.semaphore:
            messages = [{"role": "user", "content": prompt}]

            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages,
            }
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
            thinking = 0
            for block in response.content:
                if getattr(block, "type", None) == "thinking":
                    text = getattr(block, "thinking", "") or ""
                    thinking += max(1, len(text) // 4)  # rough estimate (chars // 4)

            self.last_cache_creation_input_tokens = cache_creation
            self.last_cache_read_input_tokens = cache_read
            self.total_cache_creation_input_tokens += cache_creation
            self.total_cache_read_input_tokens += cache_read
            self.last_thinking_tokens = thinking
            self.last_thinking_tokens_estimated = thinking > 0
            self.total_thinking_tokens += thinking
            if thinking > 0:
                logger.debug("Anthropic thinking tokens are estimated (chars//4=%d), not API-reported.", thinking)

            # Cumulative session totals (for get_session_cost).
            self._call_count += 1
            self.total_input_tokens += usage.input_tokens
            self.total_output_tokens += usage.output_tokens

            self._last_usage = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "reasoning_tokens": thinking,
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
                # Fall back to the legacy single-block layout.
                result_text = response.content[0].text
            if self._last_finish_reason == "max_tokens":
                # Regression fix (2026-07-21 audit): LLMTruncationError was fully specified but
                # never actually raised anywhere -- see openai_compat.py's identical fix.
                raise LLMTruncationError("Anthropic response truncated by max_tokens (stop_reason='max_tokens')", finish_reason=self._last_finish_reason)
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
        rl = {k: v for k, v in lower.items() if k.startswith("anthropic-ratelimit-")}
        if rl:
            self.last_rate_limits = rl
        org = lower.get("anthropic-organization-id")
        if isinstance(org, str):
            self.last_organization_id = org

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
        return {
            "calls": getattr(self, "_call_count", 0),
            "prompt_tokens": total_input,
            "completion_tokens": total_output,
            "thinking_tokens": self.total_thinking_tokens,
            "cache_creation_input_tokens": self.total_cache_creation_input_tokens,
            "cache_read_input_tokens": self.total_cache_read_input_tokens,
            "input_cost_usd": (plain_input / 1_000_000) * in_rate
            + (self.total_cache_creation_input_tokens / 1_000_000) * in_rate * 1.25
            + (self.total_cache_read_input_tokens / 1_000_000) * in_rate * 0.10,
            "output_cost_usd": (total_output / 1_000_000) * out_rate,
        }

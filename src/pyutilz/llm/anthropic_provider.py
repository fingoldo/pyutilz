"""Anthropic Claude LLM provider."""

import asyncio
import json
import logging
from typing import Any, AsyncIterator

import anthropic
from tenacity import retry, retry_if_exception_type

from pyutilz.llm.config import get_llm_settings
from pyutilz.llm.exceptions import LLMProviderError, JSONParsingError
from pyutilz.llm._retry import INFINITE_RETRY_KWARGS
from pyutilz.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider with async support and retry logic."""

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
    _PRICING: dict[str, tuple[float, float]] = {
        # Opus 4.5+: dropped from $15/$75 to $5/$25 (3x cheaper than legacy 4/4.1).
        "claude-opus-4-7":             (5.00, 25.00),
        "claude-opus-4-6-20250610":    (5.00, 25.00),
        "claude-opus-4-5-20250414":    (5.00, 25.00),
        # Legacy Opus 4 / 4.1 retain old $15/$75 pricing.
        "claude-opus-4-1-20250805":    (15.00, 75.00),
        "claude-opus-4-20250514":      (15.00, 75.00),
        # Sonnet family — same $3/$15 across all 4.x variants.
        # As of 2026-05-01 latest is Sonnet 4.6 (no Sonnet 4.7 released).
        "claude-sonnet-4-6-20250610":  (3.00, 15.00),
        "claude-sonnet-4-5-20250414":  (3.00, 15.00),
        "claude-sonnet-4-20250514":    (3.00, 15.00),
        "claude-sonnet-3-7-20250219":  (3.00, 15.00),  # deprecated
        # Haiku 4.5: $1/$5; legacy 3.5 stays $0.80/$4; Haiku 3 = $0.25/$1.25.
        "claude-haiku-4-5-20251001":   (1.00, 5.00),
        "claude-haiku-3-5-20241022":   (0.80, 4.00),
        "claude-haiku-3-20240307":     (0.25, 1.25),
        # Legacy Opus 3 (deprecated).
        "claude-opus-3-20240229":      (15.00, 75.00),
    }
    _DEFAULT_PRICING = (3.00, 15.00)  # fallback = Sonnet pricing

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_concurrent: int = 5,
    ):
        settings = get_llm_settings()
        self.api_key = api_key or (
            settings.anthropic_api_key.get_secret_value()
            if settings.anthropic_api_key
            else None
        )
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")

        self.model = model
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        # Per-call usage (read by LLMClient after generate())
        self._last_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
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
        self.total_thinking_tokens = 0
        # Rate-limit info captured from response headers on every call.
        # Populated lazily by ``check_account_limits()`` from the snapshot
        # of the most recent response.
        self.last_rate_limits: dict[str, str] = {}
        self.last_organization_id: str | None = None

    @property
    def max_output_tokens(self) -> int:
        # Source: https://platform.claude.com/docs/en/docs/about-claude/models
        #   Opus 4.6: 128K, Opus 4/4.1: 32K, Sonnet 4.6: 64K, Haiku 4.5: 64K
        if self.model.startswith("claude-opus"):
            return 128000 if "4-6" in self.model or "opus-4-6" in self.model else 32000
        if self.model.startswith("claude-sonnet"):
            return 64000
        if self.model.startswith("claude-haiku"):
            return 64000
        return 64000

    @property
    def context_window(self) -> int:
        return 200_000

    def supports_json_mode(self) -> bool:
        """Anthropic Messages API has NO native JSON-mode toggle. We
        get reliable JSON via ``generate_json()`` (assistant prefill
        with ``{`` + parser-side ``extract_json``), not by passing a
        kwarg to ``generate()``. Callers should branch: if False, use
        ``generate_json()`` instead of passing ``json_mode=True``."""
        return False

    @retry(
        retry=retry_if_exception_type((
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.InternalServerError,
        )),
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
            raw = await self.client.messages.with_raw_response.create(**kwargs)
            response = raw.parse()
            self._capture_response_headers(raw.headers)

            self._last_finish_reason = response.stop_reason

            usage = response.usage
            cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            thinking = 0
            for block in response.content:
                if getattr(block, "type", None) == "thinking":
                    text = getattr(block, "thinking", "") or ""
                    thinking += max(1, len(text) // 4)  # rough estimate

            self.last_cache_creation_input_tokens = cache_creation
            self.last_cache_read_input_tokens = cache_read
            self.total_cache_creation_input_tokens += cache_creation
            self.total_cache_read_input_tokens += cache_read
            self.last_thinking_tokens = thinking
            self.total_thinking_tokens += thinking

            self._last_usage = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "reasoning_tokens": thinking,
                "cache_creation_input_tokens": cache_creation,
                "cache_read_input_tokens": cache_read,
            }

            # Pull text from the first text block (skip thinking blocks).
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    return block.text
            # Fall back to the legacy single-block layout.
            return response.content[0].text

    def _capture_response_headers(self, headers: Any) -> None:
        """Snapshot rate-limit headers + org id from the latest response.

        ``check_account_limits()`` reads from this snapshot rather than
        raising NotImplementedError when at least one call has been made.
        Anthropic doesn't expose a standalone introspection endpoint for
        regular keys, so the per-call headers are the best source.
        """
        try:
            mapping = dict(headers) if headers is not None else {}
        except Exception:  # noqa: BLE001
            return
        # Lower-case the keys for case-insensitive lookup downstream.
        lower = {k.lower(): v for k, v in mapping.items()}
        rl = {k: v for k, v in lower.items() if k.startswith("anthropic-ratelimit-")}
        if rl:
            self.last_rate_limits = rl
        org = lower.get("anthropic-organization-id")
        if isinstance(org, str):
            self.last_organization_id = org

    async def generate_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 0,
    ) -> dict[str, Any]:
        """Generate structured JSON output."""
        json_system = (system or "") + "\n\nRespond with valid JSON only."
        text = await self.generate(
            prompt=prompt,
            system=json_system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self.extract_json(text, "Anthropic")

    async def generate_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        """Generate responses in batch using concurrent requests."""
        async def process_request(req: dict) -> dict:
            request_id = req.get("id", "unknown")
            try:
                result = await self.generate(
                    prompt=req["prompt"],
                    system=req.get("system"),
                    temperature=req.get("temperature", 0.7),
                    max_tokens=req.get("max_tokens", 1024),
                )
                return {"id": request_id, "result": result}
            except anthropic.RateLimitError as e:
                logger.error(f"Batch request {request_id} rate limited: {e}")
                return {"id": request_id, "error": f"Rate limited: {e}"}
            except anthropic.APIConnectionError as e:
                logger.error(f"Batch request {request_id} connection error: {e}")
                return {"id": request_id, "error": f"Connection error: {e}"}
            except anthropic.APIStatusError as e:
                logger.error(f"Batch request {request_id} API error: {e}")
                return {"id": request_id, "error": f"API error: {e}"}

        tasks = [process_request(req) for req in requests]
        for coro in asyncio.as_completed(tasks):
            yield await coro

    def _get_pricing(self) -> tuple[float, float]:
        """Return (input_cost_per_1m, output_cost_per_1m) for current model."""
        pricing = self._PRICING.get(self.model)
        if pricing:
            return pricing
        for key, val in self._PRICING.items():
            if self.model.startswith(key.rsplit("-", 1)[0]):
                return val
        return self._DEFAULT_PRICING

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate cost in USD based on model-specific pricing."""
        inp_rate, out_rate = self._get_pricing()
        input_cost = (input_tokens / 1_000_000) * inp_rate
        output_cost = (output_tokens / 1_000_000) * out_rate
        return input_cost + output_cost

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
        except Exception as exc:  # noqa: BLE001
            logger.debug("Anthropic count_tokens API failed (%s); falling back to tiktoken.", exc)
            from pyutilz.llm.token_counter import count_tokens
            return count_tokens(text)

    async def get_account_credits(self) -> dict:
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
        # Effective non-cache input tokens (paid at full input rate)
        plain_input = max(
            0,
            self._last_usage.get("input_tokens", 0)  # last_call's plain input
            - self.last_cache_creation_input_tokens
            - self.last_cache_read_input_tokens,
        )
        return {
            "calls": getattr(self, "_call_count", 0),
            "prompt_tokens": self._last_usage.get("input_tokens", 0),
            "completion_tokens": self._last_usage.get("output_tokens", 0),
            "thinking_tokens": self.total_thinking_tokens,
            "cache_creation_input_tokens": self.total_cache_creation_input_tokens,
            "cache_read_input_tokens": self.total_cache_read_input_tokens,
            "input_cost_usd": (plain_input / 1_000_000) * in_rate
            + (self.total_cache_creation_input_tokens / 1_000_000) * in_rate * 1.25
            + (self.total_cache_read_input_tokens / 1_000_000) * in_rate * 0.10,
            "output_cost_usd": (self._last_usage.get("output_tokens", 0) / 1_000_000) * out_rate,
        }

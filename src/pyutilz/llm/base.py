"""Abstract LLM provider interface."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from pyutilz.llm.exceptions import (  # noqa: F401 — re-export for backward compat
    JSONParsingError,
    LLMRefusalError,
)

logger = logging.getLogger(__name__)


def _longest_prefix_pricing(
    model: str,
    pricing_table: dict[str, tuple[float, float]],
    default: tuple[float, float],
    provider_label: str = "LLM",
) -> tuple[float, float]:
    """Return ``(input_per_1m, output_per_1m)`` for ``model`` from ``pricing_table``.

    Resolution order:
      1. Exact key match.
      2. LONGEST matching prefix — a more specific tier wins over a shorter,
         overlapping one. Dict iteration order is arbitrary, so a naive
         first-``startswith`` match can silently mis-price (e.g. a future
         ``claude-opus-4-2-YYYYMMDD`` inheriting ``claude-opus-4-7``'s cheaper
         tier just because that entry iterated first). We always compare
         prefix lengths, mirroring the algorithm used by every provider.
      3. ``default`` fallback (with a single warning).

    Two prefix forms are tried, most-precise first, so both the
    Anthropic (date-suffixed KEY, date-less model) and the Gemini
    (semantic-suffixed keys like ``...-flash`` vs ``...-flash-lite``) naming
    schemes resolve correctly:

      a. Longest FULL key that ``model`` starts with (handles date-suffixed
         model ids against date-less keys, and keeps ``-flash-lite`` from
         being captured by the shorter ``-flash`` entry).
      b. If none, longest key with its trailing ``-<segment>`` dropped
         (``key.rsplit("-", 1)[0]``) — lets a date-less canonical model id
         (``claude-opus-4-6``) match a date-suffixed key
         (``claude-opus-4-6-20250610``).
    """
    exact = pricing_table.get(model)
    if exact:
        return exact
    # (a) longest full-key prefix.
    best_val: tuple[float, float] | None = None
    best_len = -1
    for key, val in pricing_table.items():
        if model.startswith(key) and len(key) > best_len:
            best_len = len(key)
            best_val = val
    # (b) fall back to trailing-segment-trimmed prefixes.
    if best_val is None:
        for key, val in pricing_table.items():
            prefix = key.rsplit("-", 1)[0]
            if model.startswith(prefix) and len(prefix) > best_len:
                best_len = len(prefix)
                best_val = val
    if best_val is not None:
        logger.warning(
            "%s model %r not pinned in pricing table; falling back to the "
            "longest-prefix match (%s/%s per 1M). Pin its exact id to avoid "
            "silent mispricing.",
            provider_label, model, best_val[0], best_val[1],
        )
        return best_val
    return default


# ── Refusal detection ─────────────────────────────────────────────────────
# Patterns for common LLM refusal sentinels across providers (Anthropic,
# OpenAI, Gemini). Intentionally conservative — we only match phrases that
# BEGIN a refusal ("I cannot help", "I won't do that"), not content words
# that might appear inside a normal answer. False positives here silently
# degrade a valid evaluation to a fallback, so err on the strict side.
_REFUSAL_PATTERNS = [
    re.compile(r"\bI (cannot|can't|am unable|am not able) (help|assist|comply)\b", re.I),
    re.compile(r"\bI (will not|won't) (help|do that|comply)\b", re.I),
    re.compile(r"\bI'm (not able|unable) to (help|process|assist)\b", re.I),
    re.compile(r"\bI (cannot|can't) (provide|generate|create|produce) (that|this)\b", re.I),
]


def is_llm_refusal(text: str) -> bool:
    """Return True if ``text`` contains a recognizable LLM refusal sentinel.

    Used by parsers / retry loops to raise :class:`LLMRefusalError` instead
    of retrying blindly — the same prompt will refuse again.
    """
    if not text or not isinstance(text, str):
        return False
    return any(p.search(text) for p in _REFUSAL_PATTERNS)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    def max_output_tokens(self) -> int:
        """Maximum output tokens this provider/model supports."""
        return 4096  # safe fallback

    @property
    def context_window(self) -> int:
        """Maximum context window (input + output) in tokens."""
        return 200_000  # safe default for most models

    def supports_json_mode(self) -> bool:
        """Return True if this provider+model accepts a structured
        JSON-mode parameter (``response_format={"type":"json_object"}``,
        ``response_mime_type="application/json"``, or equivalent).

        Callers should consult this before sending ``json_mode=True`` as
        a kwarg to ``generate()`` — passing it to a provider that
        doesn't support it either silently no-ops or raises (depending
        on the upstream). For OpenRouter-routed models, support varies
        per model; the override consults the catalogue's
        ``supported_parameters``.

        Default: False. Subclasses that support a hard JSON-mode toggle
        override to True (or model-aware logic). Soft "system-prompt
        hint" implementations (claude-code) keep False — the hint helps
        but doesn't guarantee, and callers may want a stronger signal.
        """
        return False

    @staticmethod
    def extract_json(text: str, provider_name: str = "LLM") -> dict[str, Any]:
        """Extract and parse JSON from LLM response, handling code blocks.

        Strategy: prefer markdown-fenced JSON, then scan the body with
        ``json.JSONDecoder.raw_decode`` from each ``{`` candidate. The
        scan-based approach replaces brittle regexes — a lazy ``\\{...?\\}``
        stops at the first ``}`` even mid-object, while a greedy form
        crosses multiple objects. ``raw_decode`` knows JSON's grammar.

        Args:
            text: Raw LLM response that may contain JSON in code blocks.
            provider_name: Name of the provider for error messages.

        Returns:
            Parsed JSON dict.

        Raises:
            JSONParsingError: If JSON parsing fails.
        """

        try:
            text = text.strip()

            # 1. Markdown-fenced JSON wins. The outer fences delimit the
            #    payload unambiguously, so we extract via regex (only the
            #    wrapper is regex; the JSON itself is parsed properly).
            fence_match = re.search(
                r"```(?:json)?\s*\n?(\{.*?\}|\[.*?\])\s*\n?```",
                text,
                re.DOTALL,
            )
            if fence_match:
                return json.loads(fence_match.group(1))  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime

            # 2. Strip leading fence even without a closing fence (some
            #    streaming LLMs forget to close).
            stripped = text
            if stripped.startswith("```json"):
                stripped = stripped[7:]
            elif stripped.startswith("```"):
                stripped = stripped[3:]
            if stripped.endswith("```"):
                stripped = stripped[:-3]
            stripped = stripped.strip()

            # 3. Try the whole stripped text first (cheap path: most
            #    json_mode= responses are pure JSON).
            try:
                return json.loads(stripped)  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime
            except json.JSONDecodeError:
                pass

            # 4. Scan for the first parseable JSON object via raw_decode.
            #    Walks each ``{`` candidate, asks the decoder to consume
            #    a single JSON value, and returns the first one that
            #    parses cleanly. Robust against prose-before-JSON,
            #    JSON-with-trailing-prose, and nested-object boundaries.
            decoder = json.JSONDecoder()
            for i, ch in enumerate(stripped):
                if ch != "{":
                    continue
                try:
                    obj, _end = decoder.raw_decode(stripped, i)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    return obj

            # 5. Last resort — re-raise via the original strict parse so
            #    the JSONDecodeError handler below produces a clean error.
            return json.loads(stripped)  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime
        except json.JSONDecodeError as e:
            # Before reporting as malformed JSON, check whether the model
            # simply refused to answer — that's a distinct error class with
            # a distinct retry policy (do NOT retry refusals).
            if is_llm_refusal(text):
                logger.warning(
                    "%s refused to answer (no JSON): %.200s",
                    provider_name, text,
                )
                raise LLMRefusalError(
                    f"{provider_name} refused to produce JSON",
                    raw_text=text,
                )
            logger.error(f"Failed to parse JSON from {provider_name}: {e}\nResponse: {text}")
            raise JSONParsingError(f"Invalid JSON response from {provider_name}: {e}")

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 0,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: User prompt.
            system: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate (0 = use provider max).

        Returns:
            Generated text.
        """
        pass

    # ── shared implementation hooks ──────────────────────────────────
    # Subclasses that use the default estimate_cost / _get_pricing below
    # supply a per-1M pricing table and default via these class attrs.
    _PRICING: dict[str, tuple[float, float]] = {}
    _DEFAULT_PRICING: tuple[float, float] = (0.0, 0.0)

    def _pricing_model_id(self) -> str:
        """Return the model id used for pricing lookup.

        Providers store the current model under different attribute names
        (``self.model`` on Anthropic, ``self.model_name`` on Gemini /
        OpenAI-compat). Resolve either so the shared ``_get_pricing`` works
        for both without per-provider overrides.
        """
        return str(getattr(self, "model", None) or getattr(self, "model_name", ""))

    def _get_pricing(self) -> tuple[float, float]:
        """Return ``(input_per_1m, output_per_1m)`` for the current model.

        Default uses longest-prefix matching against ``self._PRICING`` with
        ``self._DEFAULT_PRICING`` as fallback. Override for bespoke pricing.
        """
        return _longest_prefix_pricing(
            self._pricing_model_id(),
            self._PRICING,
            self._DEFAULT_PRICING,
            provider_label=self.__class__.__name__,
        )

    @property
    def _provider_display_name(self) -> str:
        """Human-readable provider name used in JSON-parse error messages."""
        return getattr(self, "_provider_name", self.__class__.__name__)

    def _classify_batch_exception(self, exc: Exception) -> dict[str, Any] | None:
        """Return extra fields to merge into a per-request batch error dict.

        Hook for provider-specific exception classification (e.g. Gemini
        tags safety blocks with ``error_type="safety_block"``). Return
        ``None`` for the generic path. The base ``generate_batch`` always
        catches every exception per-request so one bad request never aborts
        the whole batch.
        """
        return None

    async def generate_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 0,
    ) -> dict[str, Any]:
        """Generate structured JSON output.

        Appends a "respond with valid JSON only" steer to the system prompt,
        calls ``generate``, then parses via ``extract_json``. Providers with a
        hard JSON-mode toggle (OpenAI-compat) override to pass it through.
        """
        json_system = (system or "") + "\n\nRespond with valid JSON only."
        text = await self.generate(
            prompt=prompt,
            system=json_system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self.extract_json(text, self._provider_display_name)

    async def generate_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        """Generate responses for multiple requests concurrently.

        Every per-request failure (including a malformed request dict raising
        ``KeyError`` on ``req["prompt"]``) becomes a per-request ``error``
        entry rather than aborting the whole batch. Provider-specific error
        classification is delegated to ``_classify_batch_exception``.

        Yields:
            Response dicts with ``id`` and either ``result`` or ``error``.
        """
        async def process_request(req: dict) -> dict[str, Any]:
            request_id = req.get("id", "unknown")
            try:
                result = await self.generate(
                    prompt=req["prompt"],
                    system=req.get("system"),
                    temperature=req.get("temperature", 0.7),
                    max_tokens=req.get("max_tokens", 1024),
                )
                return {"id": request_id, "result": result}
            except Exception as e:  # noqa: BLE001
                logger.error("Batch request %s failed: %s", request_id, e)
                out = {"id": request_id, "error": str(e)}
                extra = self._classify_batch_exception(e)
                if extra:
                    out.update(extra)
                return out

        # Wrap as Tasks explicitly. ``asyncio.as_completed`` over raw
        # coroutines emits a DeprecationWarning in 3.11 and breaks in 3.12+
        # — feeding Tasks instead works across versions.
        tasks = [asyncio.create_task(process_request(req)) for req in requests]
        for coro in asyncio.as_completed(tasks):
            yield await coro

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate cost in USD for token counts.

        Default multiplies token counts by the per-1M rates from
        ``_get_pricing``. Providers with cache-tier or per-token-map pricing
        (OpenAI-compat) override.
        """
        inp_rate, out_rate = self._get_pricing()
        return (input_tokens / 1_000_000) * inp_rate + (output_tokens / 1_000_000) * out_rate

    async def get_account_credits(self) -> dict[str, Any]:
        """Return account billing snapshot.

        Common keys (all optional — provider-dependent presence):
            ``balance_usd``     — remaining credit in USD (None when not exposed)
            ``total_granted``   — total ever-granted credit in USD
            ``total_used``      — lifetime spend in USD
            ``is_available``    — whether the account can issue billable calls
            ``currency``        — native currency if not USD
            ``raw``             — provider's full raw response

        Default raises :class:`NotImplementedError` — most upstream APIs
        don't publish a balance endpoint for regular (non-admin) keys.
        Concrete providers override when their API exposes it.

        Raises:
            NotImplementedError: provider doesn't expose balance via API.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not expose account balance via API. " "Check the provider's web console.")

    async def check_account_limits(self) -> dict[str, Any]:
        """Return account-level rate limit / quota / usage snapshot.

        Default raises :class:`NotImplementedError`. Concrete providers
        override when they expose this (currently only OpenRouter).

        Common keys (provider-dependent):
            ``limit_remaining`` — credits left under the cap (USD)
            ``usage_daily/weekly/monthly``
            ``rate_limit``      — requests-per-interval shape
            ``is_free_tier``
            ``raw``             — provider's full raw response

        Raises:
            NotImplementedError: provider doesn't expose this via API.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not expose account limits via API. " "Check the provider's web console.")

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for.

        Returns:
            Token count.
        """
        pass

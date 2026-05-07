"""Abstract LLM provider interface."""

from __future__ import annotations

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
        from pyutilz.llm.exceptions import JSONParsingError, LLMRefusalError

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
                return json.loads(fence_match.group(1))

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
                return json.loads(stripped)
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
            return json.loads(stripped)
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

    @abstractmethod
    async def generate_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 0,
    ) -> dict[str, Any]:
        """Generate structured JSON output.

        Args:
            prompt: User prompt requesting JSON.
            system: Optional system prompt.
            temperature: Sampling temperature (lower for structured output).
            max_tokens: Maximum tokens to generate.

        Returns:
            Parsed JSON dict.
        """
        pass

    @abstractmethod
    async def generate_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        """Generate responses for multiple requests in batch.

        Args:
            requests: List of request dicts with 'prompt', 'system', etc.

        Yields:
            Response dicts with 'id', 'result' or 'error'.
        """
        pass

    @abstractmethod
    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate cost in USD for token counts.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Estimated cost in USD.
        """
        pass

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
        raise NotImplementedError(
            f"{self.__class__.__name__} does not expose account balance via API. "
            "Check the provider's web console."
        )

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
        raise NotImplementedError(
            f"{self.__class__.__name__} does not expose account limits via API. "
            "Check the provider's web console."
        )

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for.

        Returns:
            Token count.
        """
        pass

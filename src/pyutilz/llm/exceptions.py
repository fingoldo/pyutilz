"""Exceptions for LLM providers."""

from __future__ import annotations


class LLMProviderError(Exception):
    """Error from LLM provider (Anthropic, Gemini, etc.)."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class JSONParsingError(ValueError):
    """Raised when JSON parsing fails."""

    pass


class LLMRefusalError(LLMProviderError):
    """LLM refused to answer (safety / alignment / policy).

    Distinct from a transient error: the LLM *did* respond, and the caller
    should NOT retry — another call will almost certainly refuse again.
    Downstream should translate this to a domain-specific fallback
    (e.g. ``can_do=False`` for job evaluation).
    """

    def __init__(self, message: str, raw_text: str | None = None, details: dict | None = None):
        super().__init__(message, details=details)
        self.raw_text = raw_text


class LLMSafetyBlockError(LLMRefusalError):
    """LLM response suppressed by the provider's safety filter.

    Gemini: ``candidate.finish_reason == SAFETY``. OpenAI: content policy
    rejection. Treated identically to a refusal — do not retry.
    """

    pass


class LLMTruncationError(ValueError):
    """LLM stopped because ``max_tokens`` was hit mid-generation.

    Retryable — caller should double ``max_tokens`` (capped) and re-issue.
    """

    def __init__(self, message: str, finish_reason: str | None = None):
        super().__init__(message)
        self.finish_reason = finish_reason

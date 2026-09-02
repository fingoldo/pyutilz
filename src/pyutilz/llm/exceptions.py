"""Exceptions for LLM providers."""

from __future__ import annotations


class LLMProviderError(Exception):
    """Error from LLM provider (Anthropic, Gemini, etc.)."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class JSONParsingError(LLMProviderError, ValueError):
    """Raised when JSON parsing fails.

    Inherits BOTH bases on purpose: ``ValueError`` keeps every pre-existing ``except ValueError``
    call site working, while ``LLMProviderError`` makes the domain root actually catch-all -- a
    caller wrapping an LLM call in ``except LLMProviderError`` used to have malformed model JSON
    escape past the handler as a bare ``ValueError``.
    """


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


class LLMTruncationError(LLMProviderError, ValueError):
    """LLM stopped because ``max_tokens`` was hit mid-generation.

    Retryable — caller should double ``max_tokens`` (capped) and re-issue.

    Inherits both bases for the same reason as :class:`JSONParsingError`: ``except ValueError``
    call sites keep working AND ``except LLMProviderError`` now covers it.
    """

    def __init__(self, message: str, finish_reason: str | None = None, partial_text: str = ""):
        super().__init__(message)
        self.finish_reason = finish_reason
        # The text the model DID produce before the cutoff. Carried because the caller's whole reason for
        # catching this is to salvage a paid-for call: without it, a truncation, a wrong JSON shape and a
        # refusal are indistinguishable downstream, and the only evidence that could tell them apart was
        # discarded at the raise site. Empty string, never None, so a caller can `or ""` without a guard.
        self.partial_text = partial_text or ""


class LLMUnparseableResponseError(LLMProviderError):
    """The transport succeeded but the body is not the JSON envelope the API promises.

    Distinct from `JSONParsingError`, which is about the MODEL's own output being malformed inside a
    well-formed envelope: that is the model's answer and re-issuing it usually returns the same thing.
    This is the envelope itself missing - an empty body, a truncated one, or an HTML error page from an
    intermediary - which is a transient transport-layer fault the same call will usually survive.

    Retryable, and it has to be raised as its own type to BE retryable: `resp.json()` fails with
    `json.JSONDecodeError`, a `ValueError`, which no HTTP/transport predicate matches, so an empty body
    escaped the retry decorator and surfaced as a hard failure. MEASURED 2026-08-06 against OpenRouter:
    across three runs of one live-LLM test tier a DIFFERENT test failed each time and one run was green,
    which reads as flaky tests rather than as one retryable fault.
    """

    def __init__(self, message: str, status_code: int | None = None, body_excerpt: str = ""):
        super().__init__(message, {"status_code": status_code, "body_excerpt": body_excerpt})
        self.status_code = status_code
        # Kept because "the body was not JSON" is unactionable on its own - an HTML gateway page, an empty
        # string and a truncated envelope need different responses from a human reading the log.
        self.body_excerpt = body_excerpt

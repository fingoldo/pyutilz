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

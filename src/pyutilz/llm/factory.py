"""Factory for LLM providers.

Provides `get_llm_provider()` with instance caching: the same provider+model
combination returns the same instance, avoiding expensive re-initialization
(SSL context loading, Anthropic client creation, etc.).
"""

import threading

from pyutilz.llm.config import LLMSettings, get_llm_settings
from pyutilz.llm.base import LLMProvider
from pyutilz.llm.anthropic_provider import AnthropicProvider
from pyutilz.llm.gemini_provider import GeminiProvider
from pyutilz.llm.claude_code_provider import ClaudeCodeProvider
from pyutilz.llm.deepseek_provider import DeepSeekProvider
from pyutilz.llm.xai_provider import XAIProvider

# Instance cache: (canonical_name, kwargs_key) → LLMProvider
_provider_cache: dict[tuple, LLMProvider] = {}
_provider_lock = threading.Lock()

# Canonical provider names → constructor
_PROVIDER_CONSTRUCTORS = {
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "claude-code": ClaudeCodeProvider,
    "deepseek": DeepSeekProvider,
    "xai": XAIProvider,
}

# Aliases mapping to canonical names
_ALIASES = {
    "claude": "anthropic",
    "google": "gemini",
    "cc": "claude-code",
    "ds": "deepseek",
    "grok": "xai",
}


def get_llm_provider(
    provider_name: str = "claude-code",
    settings: LLMSettings | None = None,
    **kwargs,
) -> LLMProvider:
    """Get an LLM provider instance.

    Args:
        provider_name: Name of the provider.
            - "claude-code" / "cc": Claude Code CLI (uses Max subscription!) [DEFAULT]
            - "anthropic" / "claude": Claude API (requires API credits)
            - "gemini" / "google": Google Gemini API
            - "deepseek" / "ds": DeepSeek API
            - "xai" / "grok": xAI Grok API
        settings: Optional settings instance.
        **kwargs: Additional arguments passed to the provider.

    Returns:
        LLM provider instance.

    Raises:
        ValueError: If provider name is unknown.
    """
    settings = settings or get_llm_settings()
    name = provider_name.lower()

    # Resolve aliases
    canonical = _ALIASES.get(name, name)

    if canonical not in _PROVIDER_CONSTRUCTORS:
        available = sorted(set(list(_PROVIDER_CONSTRUCTORS.keys()) + list(_ALIASES.keys())))
        raise ValueError(
            f"Unknown provider: {provider_name}. Available: {available}"
        )

    constructor = _PROVIDER_CONSTRUCTORS[canonical]

    # Pass API key for providers that need it
    if canonical == "anthropic":
        kwargs.setdefault(
            "api_key",
            settings.anthropic_api_key.get_secret_value()
            if settings.anthropic_api_key
            else None,
        )
    elif canonical == "gemini":
        kwargs.setdefault(
            "api_key",
            settings.gemini_api_key.get_secret_value()
            if settings.gemini_api_key
            else None,
        )
    elif canonical == "deepseek":
        kwargs.setdefault(
            "api_key",
            settings.deepseek_api_key.get_secret_value()
            if settings.deepseek_api_key
            else None,
        )
    elif canonical == "xai":
        kwargs.setdefault(
            "api_key",
            settings.xai_api_key.get_secret_value()
            if settings.xai_api_key
            else None,
        )

    # Cache key: provider name + all kwargs (model, api_key, etc.)
    cache_key = (canonical, tuple(sorted(kwargs.items())))
    if cache_key in _provider_cache:
        return _provider_cache[cache_key]

    with _provider_lock:
        if cache_key in _provider_cache:
            return _provider_cache[cache_key]
        instance = constructor(**kwargs)
        _provider_cache[cache_key] = instance
        return instance

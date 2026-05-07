"""Factory for LLM providers.

Provides `get_llm_provider()` with instance caching: the same provider+model
combination returns the same instance, avoiding expensive re-initialization
(SSL context loading, Anthropic client creation, etc.).
"""

import asyncio
import atexit
import logging
import threading

from pyutilz.llm.config import LLMSettings, get_llm_settings
from pyutilz.llm.base import LLMProvider

logger = logging.getLogger(__name__)

# Instance cache: (canonical_name, kwargs_key) → LLMProvider
_provider_cache: dict[tuple, LLMProvider] = {}
_provider_lock = threading.Lock()


# Canonical provider names → (module_path, class_name) for lazy import
_PROVIDER_MODULES = {
    "anthropic": ("pyutilz.llm.anthropic_provider", "AnthropicProvider"),
    "gemini": ("pyutilz.llm.gemini_provider", "GeminiProvider"),
    "claude-code": ("pyutilz.llm.claude_code_provider", "ClaudeCodeProvider"),
    "deepseek": ("pyutilz.llm.deepseek_provider", "DeepSeekProvider"),
    "xai": ("pyutilz.llm.xai_provider", "XAIProvider"),
    "openai": ("pyutilz.llm.openai_provider", "OpenAIProvider"),
    "openrouter": ("pyutilz.llm.openrouter_provider", "OpenRouterProvider"),
}

# Aliases mapping to canonical names
_ALIASES = {
    "claude": "anthropic",
    "google": "gemini",
    "cc": "claude-code",
    "ds": "deepseek",
    "grok": "xai",
    "gpt": "openai",
    "or": "openrouter",
    "router": "openrouter",
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
            - "openai" / "gpt": OpenAI API
            - "openrouter" / "or" / "router": OpenRouter (meta-provider, 200+ models)
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

    if canonical not in _PROVIDER_MODULES:
        available = sorted(set(list(_PROVIDER_MODULES.keys()) + list(_ALIASES.keys())))
        raise ValueError(
            f"Unknown provider: {provider_name}. Available: {available}"
        )

    import importlib
    mod_path, cls_name = _PROVIDER_MODULES[canonical]
    mod = importlib.import_module(mod_path)
    constructor = getattr(mod, cls_name)

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
    elif canonical == "openai":
        kwargs.setdefault(
            "api_key",
            settings.openai_api_key.get_secret_value()
            if settings.openai_api_key
            else None,
        )
    elif canonical == "openrouter":
        kwargs.setdefault(
            "api_key",
            settings.openrouter_api_key.get_secret_value()
            if settings.openrouter_api_key
            else None,
        )

    # Cache key: provider name + all kwargs (model, api_key, etc.).
    # If a kwarg value is unhashable (list/dict), bypass the cache and
    # construct fresh — matches "do the right thing" rather than crash
    # the request flow with a confusing TypeError from tuple-as-dict-key.
    try:
        cache_key = (canonical, tuple(sorted(kwargs.items())))
        hash(cache_key)
    except TypeError as exc:
        logger.warning(
            "Unhashable kwarg passed to get_llm_provider(%r): %s. "
            "Bypassing instance cache; pass tuples instead of lists / "
            "JSON strings instead of dicts to enable caching.",
            provider_name, exc,
        )
        return constructor(**kwargs)

    if cache_key in _provider_cache:
        return _provider_cache[cache_key]

    with _provider_lock:
        if cache_key in _provider_cache:
            return _provider_cache[cache_key]
        instance = constructor(**kwargs)
        _provider_cache[cache_key] = instance
        return instance


def _close_cached_providers() -> None:
    """atexit handler: close every cached provider's HTTP client.

    Without this, ``OpenAICompatibleProvider._client`` (an httpx.AsyncClient
    with a connection pool + open SSL contexts) leaks at process shutdown,
    producing ``unclosed transport`` warnings or TLS finalisation errors
    on Windows. Best-effort: failures during shutdown are logged at debug.
    """
    if not _provider_cache:
        return
    # Try to schedule async close. If no event loop running, skip async
    # cleanup — the OS will reclaim sockets at process exit anyway.
    try:
        loop = asyncio.new_event_loop()
    except Exception:  # noqa: BLE001
        return
    try:
        for provider in list(_provider_cache.values()):
            close = getattr(provider, "_close", None)
            if close is None:
                continue
            try:
                loop.run_until_complete(close())
            except Exception as exc:  # noqa: BLE001
                logger.debug("Provider close error during shutdown: %s", exc)
    finally:
        loop.close()


atexit.register(_close_cached_providers)

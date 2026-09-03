"""Factory for LLM providers.

Provides `get_llm_provider()` with instance caching: the same provider+model
combination returns the same instance, avoiding expensive re-initialization
(SSL context loading, Anthropic client creation, etc.).
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import sys
import threading
import weakref
from collections import OrderedDict

from pyutilz.llm.config import LLMSettings, get_llm_settings
from pyutilz.llm.base import LLMProvider

logger = logging.getLogger(__name__)

# Instance cache: (canonical_name, kwargs_key) → LLMProvider, bounded LRU (see get_llm_provider's
# eviction logic below). Regression fix (2026-07-21 audit round 2, MEDIUM): this used to be a
# plain unbounded dict -- every distinct (provider, kwargs) combination a long-running process
# ever saw stayed cached (and its live httpx.AsyncClient connection pool + SSL context open)
# for the rest of the process lifetime.
_provider_cache: "OrderedDict[tuple, LLMProvider]" = OrderedDict()
_provider_lock = threading.Lock()
_DEFAULT_PROVIDER_CACHE_MAX_SIZE = 128
try:
    _PROVIDER_CACHE_MAX_SIZE = int(os.environ.get("PYUTILZ_LLM_PROVIDER_CACHE_MAX_SIZE", str(_DEFAULT_PROVIDER_CACHE_MAX_SIZE)))
except ValueError:
    # A typo in this env var used to abort ``import pyutilz.llm.factory`` outright (2026-09-03
    # audit F40). Warn and use the documented default, as _retry.py already did.
    logger.warning(
        "PYUTILZ_LLM_PROVIDER_CACHE_MAX_SIZE=%r is not an integer; falling back to %d.",
        os.environ.get("PYUTILZ_LLM_PROVIDER_CACHE_MAX_SIZE"), _DEFAULT_PROVIDER_CACHE_MAX_SIZE,
    )
    _PROVIDER_CACHE_MAX_SIZE = _DEFAULT_PROVIDER_CACHE_MAX_SIZE
# Providers built with unhashable kwargs bypass the cache (below), and providers evicted from
# the LRU above (when no running event loop can schedule their async close immediately) land
# here too -- weakly tracked so the atexit handler still closes their HTTP clients instead of
# leaking silently for the rest of the process lifetime.
_uncached_providers: "weakref.WeakSet[LLMProvider]" = weakref.WeakSet()

# asyncio only holds a WEAK reference to a task created via loop.create_task()/asyncio.create_task()
# -- with nothing else referencing it, the task can be garbage-collected mid-execution before it
# ever awaits close() (see asyncio.create_task's own docs: "task must be referenced ... or it can
# get garbage collected at any time"). This set is that strong reference; each task removes itself
# via add_done_callback once it finishes.
_pending_close_tasks: "set[asyncio.Task[None]]" = set()


def _schedule_provider_close(provider: LLMProvider) -> None:
    """Best-effort close for a provider evicted from the LRU cache.

    Evicted providers still hold live resources (httpx.AsyncClient connection pools, SSL
    contexts) that leak if simply dropped. ``_close()`` is async; if an event loop is currently
    running (the common case -- eviction happens inside ``get_llm_provider()``, itself normally
    called from async application code) we schedule it as a fire-and-forget task. Without a
    running loop we fall back to tracking the provider in ``_uncached_providers`` so the atexit
    handler (``_close_cached_providers``) still closes it at process shutdown.
    """
    close = getattr(provider, "_close", None)
    if close is None:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _uncached_providers.add(provider)
        return

    async def _close_and_log() -> None:
        """Await the evicted provider's close(), logging (not raising) on failure."""
        try:
            await close()
        except Exception as exc:
            logger.debug("Evicted provider close error: %s", exc)

    task = loop.create_task(_close_and_log())
    _pending_close_tasks.add(task)
    task.add_done_callback(_pending_close_tasks.discard)


# Canonical provider names → (module_path, class_name, settings_api_key_attr) for lazy import.
# The third element is the LLMSettings attribute holding this provider's API key, or None for a
# provider that needs no key. It is DATA here rather than a hand-written branch per provider so
# that adding a provider cannot leave the key-injection cascade out of sync with this table --
# the failure that already materialised once (OpenAIProvider reached the table but not the
# package facade). pyutilz.llm.__init__ derives its lazy-import map from this table too.
_PROVIDER_MODULES: dict[str, tuple[str, str, str | None]] = {
    "anthropic": ("pyutilz.llm.anthropic_provider", "AnthropicProvider", "anthropic_api_key"),
    "gemini": ("pyutilz.llm.gemini_provider", "GeminiProvider", "gemini_api_key"),
    "claude-code": ("pyutilz.llm.claude_code_provider", "ClaudeCodeProvider", None),
    "deepseek": ("pyutilz.llm.deepseek_provider", "DeepSeekProvider", "deepseek_api_key"),
    "xai": ("pyutilz.llm.xai_provider", "XAIProvider", "xai_api_key"),
    "openai": ("pyutilz.llm.openai_provider", "OpenAIProvider", "openai_api_key"),
    "openrouter": ("pyutilz.llm.openrouter_provider", "OpenRouterProvider", "openrouter_api_key"),
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
        raise ValueError(f"Unknown provider: {provider_name}. Available: {available}")

    import importlib
    mod_path, cls_name, settings_key_attr = _PROVIDER_MODULES[canonical]
    mod = importlib.import_module(mod_path)
    constructor = getattr(mod, cls_name)

    # Pass API key for providers that need it. Key STILL defaults to None (not omitted) when the
    # settings value is unset: providers accept api_key=None and raise their own configuration
    # error, and omitting the kwarg would change the cache key shape for a keyless environment.
    if settings_key_attr is not None:
        secret = getattr(settings, settings_key_attr, None)
        kwargs.setdefault("api_key", secret.get_secret_value() if secret else None)

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
        instance = constructor(**kwargs)
        _uncached_providers.add(instance)
        return instance  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime

    with _provider_lock:
        if cache_key in _provider_cache:
            _provider_cache.move_to_end(cache_key)
            return _provider_cache[cache_key]
        instance = constructor(**kwargs)
        _provider_cache[cache_key] = instance
        if len(_provider_cache) > _PROVIDER_CACHE_MAX_SIZE:
            _, evicted = _provider_cache.popitem(last=False)
            # Eviction is keyed on cache size, not liveness: get_llm_provider() hands out shared
            # instances and keeps no refcount, so closing unconditionally could shut the httpx
            # client of a provider a coroutine obtained earlier and is still using ("Cannot send
            # a request, as the client has been closed" mid-batch). getrefcount's baseline here
            # is 2 -- the local `evicted` plus the temporary argument reference -- so anything
            # above it means someone else still holds the provider: drop the LRU entry without
            # closing and let the WeakSet + atexit handler tear it down instead.
            if sys.getrefcount(evicted) <= 2:
                _schedule_provider_close(evicted)
            else:
                _uncached_providers.add(evicted)
        return instance  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime


def _close_cached_providers() -> None:
    """atexit handler: close every cached provider's HTTP client.

    Without this, ``OpenAICompatibleProvider._client`` (an httpx.AsyncClient
    with a connection pool + open SSL contexts) leaks at process shutdown,
    producing ``unclosed transport`` warnings or TLS finalisation errors
    on Windows. Best-effort: failures during shutdown are logged at debug.
    """
    providers = list(_provider_cache.values()) + list(_uncached_providers)
    if not providers:
        return
    # Try to schedule async close. If no event loop running, skip async
    # cleanup — the OS will reclaim sockets at process exit anyway.
    try:
        loop = asyncio.new_event_loop()
    except Exception as e:
        logger.debug("Could not create event loop for provider shutdown cleanup: %s", e)
        return
    try:
        for provider in providers:
            close = getattr(provider, "_close", None)
            if close is None:
                continue
            try:
                loop.run_until_complete(close())
            except Exception as exc:
                logger.debug("Provider close error during shutdown: %s", exc)
    finally:
        loop.close()


atexit.register(_close_cached_providers)

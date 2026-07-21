"""Minimal configuration for LLM providers.

Reads API keys from environment variables / .env files.
Callers can also inject their own settings object into the factory.
"""

import os
import threading
import time
from typing import Optional

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM provider settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: Optional[SecretStr] = None
    gemini_api_key: Optional[SecretStr] = None
    deepseek_api_key: Optional[SecretStr] = None
    xai_api_key: Optional[SecretStr] = None
    openai_api_key: Optional[SecretStr] = None
    openrouter_api_key: Optional[SecretStr] = None


# Regression fix (2026-07-21 audit round 2, LOW): a plain ``@lru_cache`` cached the *one*
# ``LLMSettings()`` instance for the entire process lifetime -- an API-key rotation (a new
# ``.env`` value or environment variable) never took effect without a full process restart.
# A short TTL removes the "stuck forever" failure mode while still avoiding a
# ``LLMSettings()``/``.env`` re-read on every single call (this is looked up once per
# ``get_llm_provider()`` call, including cache HITS -- see factory.py).
_SETTINGS_TTL_SECONDS = float(os.environ.get("PYUTILZ_LLM_SETTINGS_TTL_SECONDS", "60"))
_settings_lock = threading.Lock()
_cached_settings: Optional[LLMSettings] = None
_cached_settings_at: float = float("-inf")


def get_llm_settings() -> LLMSettings:
    """Get the LLM settings instance, refreshed at most once per ``_SETTINGS_TTL_SECONDS``
    (default 60s, override via ``PYUTILZ_LLM_SETTINGS_TTL_SECONDS``).

    Call :func:`clear_llm_settings_cache` to force an immediate reload (e.g. right after an
    operator-initiated API-key rotation) instead of waiting out the TTL.
    """
    global _cached_settings, _cached_settings_at
    now = time.monotonic()
    with _settings_lock:
        if _cached_settings is not None and (now - _cached_settings_at) < _SETTINGS_TTL_SECONDS:
            return _cached_settings
        _cached_settings = LLMSettings()
        _cached_settings_at = now
        return _cached_settings


def clear_llm_settings_cache() -> None:
    """Force the next :func:`get_llm_settings` call to re-read environment variables/``.env``.

    Use after rotating an API key so the new value takes effect immediately rather than waiting
    out ``_SETTINGS_TTL_SECONDS``.
    """
    global _cached_settings, _cached_settings_at
    with _settings_lock:
        _cached_settings = None
        _cached_settings_at = float("-inf")

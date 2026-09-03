"""Minimal configuration for LLM providers.

Reads API keys from environment variables, and from a ``.env`` file only when one is named
explicitly (``get_llm_settings(env_file=...)`` or ``PYUTILZ_LLM_ENV_FILE``) -- never from whatever
``.env`` happens to sit in the process's current working directory.
Callers can also inject their own settings object into the factory.
"""

import logging
import os
import threading
import time
from typing import Optional

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class LLMSettings(BaseSettings):
    """LLM provider settings loaded from environment variables."""

    # No default env_file: see get_llm_settings() for why a bare relative ".env" (resolved against
    # whatever directory the process happens to be in) is not a safe place to read API keys from.
    # Callers who want one pass it explicitly, or set PYUTILZ_LLM_ENV_FILE.
    model_config = SettingsConfigDict(
        env_file=None,
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
_DEFAULT_SETTINGS_TTL_SECONDS = 60.0
try:
    _SETTINGS_TTL_SECONDS = float(os.environ.get("PYUTILZ_LLM_SETTINGS_TTL_SECONDS", str(_DEFAULT_SETTINGS_TTL_SECONDS)))
except ValueError:
    # A typo in this env var used to abort ``import pyutilz.llm.config`` outright, with a
    # traceback pointing at a module-level float() rather than at the operator's environment
    # (2026-09-03 audit F40). Warn and use the documented default, as _retry.py already did.
    logger.warning(
        "PYUTILZ_LLM_SETTINGS_TTL_SECONDS=%r is not a number; falling back to %.0fs.",
        os.environ.get("PYUTILZ_LLM_SETTINGS_TTL_SECONDS"), _DEFAULT_SETTINGS_TTL_SECONDS,
    )
    _SETTINGS_TTL_SECONDS = _DEFAULT_SETTINGS_TTL_SECONDS
_settings_lock = threading.Lock()
_cached_settings: Optional[LLMSettings] = None
_cached_settings_at: float = float("-inf")
# Sentinel distinguishing "caller said nothing" (consult PYUTILZ_LLM_ENV_FILE) from an explicit
# ``env_file=None`` (environment variables only) -- None is a meaningful value here, so it cannot
# double as the default.
_UNSET_ENV_FILE = "<unset>"
_cached_env_file: Optional[str] = _UNSET_ENV_FILE


def get_llm_settings(env_file: Optional[str] = _UNSET_ENV_FILE) -> LLMSettings:
    """Get the LLM settings instance, refreshed at most once per ``_SETTINGS_TTL_SECONDS``
    (default 60s, override via ``PYUTILZ_LLM_SETTINGS_TTL_SECONDS``).

    ``env_file`` selects where API keys may be read from besides the process environment:

    * omitted (the default) -- the ``.env`` named by ``PYUTILZ_LLM_ENV_FILE`` if that variable is
      set, else no ``.env`` at all;
    * an explicit path -- read that file;
    * ``None`` -- environment variables only.

    Historically this was a bare relative ``".env"``, which pydantic-settings resolves against the
    process's CURRENT WORKING DIRECTORY at instantiation time. Running a pyutilz-based tool from a
    directory the user does not control (a cloned third-party repo, a shared scratch dir, an
    extracted archive) silently adopted that directory's ``.env`` for any key not already set in the
    real environment -- every subsequent LLM call billed to, and its prompts sent under, a key chosen
    by whoever wrote the file, with no log line naming it. An explicit, operator-chosen anchor
    replaces the implicit cwd one, and the path actually read is logged at INFO.

    Call :func:`clear_llm_settings_cache` to force an immediate reload (e.g. right after an
    operator-initiated API-key rotation) instead of waiting out the TTL. A call with an ``env_file``
    different from the cached one bypasses the TTL and reloads.
    """
    if env_file is _UNSET_ENV_FILE:
        env_file = os.environ.get("PYUTILZ_LLM_ENV_FILE") or None
    global _cached_settings, _cached_settings_at, _cached_env_file
    now = time.monotonic()
    with _settings_lock:
        if _cached_settings is not None and _cached_env_file == env_file and (now - _cached_settings_at) < _SETTINGS_TTL_SECONDS:
            return _cached_settings
        if env_file is not None and os.path.isfile(env_file):
            logger.info("LLM settings: reading .env from %s", os.path.abspath(env_file))
        _cached_env_file = env_file
        # _env_file overrides model_config's env_file for this instantiation; pydantic-settings
        # ignores a non-existent path, and process env vars keep precedence over it either way.
        # The ignore below: `_env_file` is pydantic-settings' documented per-instantiation
        # override of model_config's env_file, but BaseSettings.__init__ is typed with only the
        # model's own fields, so mypy sees it as an unexpected keyword.
        _cached_settings = LLMSettings(_env_file=env_file)  # type: ignore[call-arg]
        _cached_settings_at = now
        return _cached_settings


def clear_llm_settings_cache() -> None:
    """Force the next :func:`get_llm_settings` call to re-read environment variables/``.env``.

    Use after rotating an API key so the new value takes effect immediately rather than waiting
    out ``_SETTINGS_TTL_SECONDS``.
    """
    global _cached_settings, _cached_settings_at, _cached_env_file
    with _settings_lock:
        _cached_settings = None
        _cached_settings_at = float("-inf")
        _cached_env_file = _UNSET_ENV_FILE

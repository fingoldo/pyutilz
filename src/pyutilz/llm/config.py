"""Minimal configuration for LLM providers.

Reads API keys from environment variables / .env files.
Callers can also inject their own settings object into the factory.
"""

from functools import lru_cache
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


@lru_cache
def get_llm_settings() -> LLMSettings:
    """Get cached LLM settings instance."""
    return LLMSettings()

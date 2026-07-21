"""Tests for pyutilz.llm.config."""

import importlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

import pyutilz.llm.config as config_module


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """Every test starts from a clean cache so tests don't bleed into each other."""
    config_module.clear_llm_settings_cache()
    yield
    config_module.clear_llm_settings_cache()


class TestSettingsTTLEnvVar:
    """Regression (2026-07-21 audit round 2, LOW): ``@lru_cache`` used to cache the *one*
    ``LLMSettings()`` instance forever -- an API-key rotation never took effect without a
    process restart. ``_SETTINGS_TTL_SECONDS`` is read from ``PYUTILZ_LLM_SETTINGS_TTL_SECONDS``
    at import time, mirroring ``_retry.py``'s ``PYUTILZ_LLM_MAX_RETRIES`` pattern."""

    def _reload(self):
        return importlib.reload(config_module)

    def test_default_ttl_is_60(self, monkeypatch):
        monkeypatch.delenv("PYUTILZ_LLM_SETTINGS_TTL_SECONDS", raising=False)
        reloaded = self._reload()
        assert reloaded._SETTINGS_TTL_SECONDS == 60.0

    def test_custom_ttl_env_var_respected(self, monkeypatch):
        monkeypatch.setenv("PYUTILZ_LLM_SETTINGS_TTL_SECONDS", "5")
        reloaded = self._reload()
        assert reloaded._SETTINGS_TTL_SECONDS == 5.0

    def teardown_method(self):
        # Restore the un-monkeypatched module state for subsequent test files.
        importlib.reload(config_module)


class TestGetLlmSettingsCaching:
    def test_repeated_calls_within_ttl_return_same_instance(self):
        a = config_module.get_llm_settings()
        b = config_module.get_llm_settings()
        assert a is b

    def test_call_after_ttl_expiry_returns_new_instance(self):
        first = config_module.get_llm_settings()

        # Simulate the TTL having elapsed without actually sleeping.
        with patch.object(config_module.time, "monotonic", return_value=config_module._cached_settings_at + config_module._SETTINGS_TTL_SECONDS + 1.0):
            second = config_module.get_llm_settings()

        assert second is not first

    def test_call_before_ttl_expiry_returns_cached_instance(self):
        first = config_module.get_llm_settings()

        with patch.object(config_module.time, "monotonic", return_value=config_module._cached_settings_at + config_module._SETTINGS_TTL_SECONDS - 1.0):
            second = config_module.get_llm_settings()

        assert second is first


class TestClearLlmSettingsCache:
    def test_forces_reload_even_within_ttl(self):
        """A caller rotating an API key shouldn't have to wait out the TTL."""
        first = config_module.get_llm_settings()
        config_module.clear_llm_settings_cache()
        second = config_module.get_llm_settings()
        assert second is not first

    def test_clearing_an_already_empty_cache_does_not_raise(self):
        config_module.clear_llm_settings_cache()
        config_module.clear_llm_settings_cache()  # must not raise


class TestConcurrentAccess:
    def test_concurrent_callers_within_ttl_get_the_same_instance(self):
        config_module.get_llm_settings()  # warm the cache first

        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = [ex.submit(config_module.get_llm_settings) for _ in range(10)]
            results = [f.result() for f in as_completed(futures)]

        first = results[0]
        assert all(r is first for r in results)

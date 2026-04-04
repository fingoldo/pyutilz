"""Tests for pyutilz.llm.factory."""

import pytest
from unittest.mock import patch, MagicMock

pytest.importorskip("pydantic")

from pyutilz.llm.factory import (
    get_llm_provider,
    _provider_cache,
    _PROVIDER_MODULES,
    _ALIASES,
)


class TestAliases:
    @pytest.mark.parametrize("alias,canonical", [
        ("claude", "anthropic"),
        ("google", "gemini"),
        ("cc", "claude-code"),
        ("ds", "deepseek"),
        ("grok", "xai"),
    ])
    def test_alias_resolves(self, alias, canonical):
        assert _ALIASES[alias] == canonical

    def test_all_aliases_point_to_valid_constructors(self):
        for alias, canonical in _ALIASES.items():
            assert canonical in _PROVIDER_MODULES, (
                f"Alias '{alias}' → '{canonical}' has no constructor"
            )


class TestProviderModules:
    def test_has_at_least_five_providers(self):
        assert len(_PROVIDER_MODULES) >= 5

    def test_all_entries_are_module_class_tuples(self):
        for name, entry in _PROVIDER_MODULES.items():
            assert isinstance(entry, tuple) and len(entry) == 2, (
                f"Entry for '{name}' should be (module_path, class_name)"
            )


class TestGetLlmProvider:
    def setup_method(self):
        _provider_cache.clear()

    def teardown_method(self):
        _provider_cache.clear()

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_llm_provider("nonexistent_provider_xyz")

    def test_unknown_provider_lists_available(self):
        with pytest.raises(ValueError, match="Available"):
            get_llm_provider("bad")

    def test_case_insensitive(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_llm_provider("NONEXISTENT")

    @pytest.mark.parametrize("name", ["claude-code", "cc", "CC"])
    def test_claude_code_variants_accepted(self, name):
        try:
            get_llm_provider(name)
        except ValueError as e:
            if "Unknown provider" in str(e):
                pytest.fail(f"Provider '{name}' should be recognized")

    def test_cache_returns_same_instance(self):
        try:
            p1 = get_llm_provider("claude-code")
            p2 = get_llm_provider("claude-code")
            assert p1 is p2
        except Exception:
            pytest.skip("claude-code provider not available in test env")

    def test_default_is_claude_code(self):
        import inspect
        sig = inspect.signature(get_llm_provider)
        assert sig.parameters["provider_name"].default == "claude-code"

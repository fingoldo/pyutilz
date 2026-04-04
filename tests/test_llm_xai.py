"""Tests for xAI (Grok) provider."""

import pytest

from pyutilz.llm.xai_provider import XAIProvider


class TestXAIConfig:
    def test_timeout_reasoning(self):
        p = XAIProvider.__new__(XAIProvider)
        assert p._get_timeout("grok-4-1-fast-reasoning") == 1200.0

    def test_timeout_non_reasoning(self):
        p = XAIProvider.__new__(XAIProvider)
        assert p._get_timeout("grok-4-1-fast-non-reasoning") == 1200.0

    def test_timeout_code_model(self):
        p = XAIProvider.__new__(XAIProvider)
        assert p._get_timeout("grok-code-fast-1") == 240.0

    def test_billed_output_includes_reasoning(self):
        p = XAIProvider.__new__(XAIProvider)
        assert p._compute_billed_output(100, 50) == 150

    def test_pricing_fast(self):
        p = XAIProvider.__new__(XAIProvider)
        p.model_name = "grok-4-1-fast-reasoning"
        assert p._input_cost_per_1m("grok-4-1-fast-reasoning") == 0.20
        assert p._output_cost_per_1m("grok-4-1-fast-reasoning") == 0.50

    def test_pricing_grok4(self):
        p = XAIProvider.__new__(XAIProvider)
        p.model_name = "grok-4-0709"
        assert p._input_cost_per_1m("grok-4-0709") == 3.00
        assert p._output_cost_per_1m("grok-4-0709") == 15.00

    def test_cache_hit_cost(self):
        p = XAIProvider.__new__(XAIProvider)
        assert p._cache_hit_cost_per_1m("grok-4-1-fast-reasoning") == 0.05
        assert p._cache_hit_cost_per_1m("grok-4-0709") == 0.75

    def test_max_tokens(self):
        p = XAIProvider.__new__(XAIProvider)
        p.model_name = "grok-4-1-fast-reasoning"
        assert p.max_output_tokens == 30000

    def test_max_tokens_unknown_model(self):
        p = XAIProvider.__new__(XAIProvider)
        p.model_name = "future-model"
        assert p.max_output_tokens == 30000

    def test_provider_name(self):
        assert XAIProvider._provider_name == "xAI"

    def test_base_url(self):
        assert XAIProvider._base_url == "https://api.x.ai/v1"

    def test_missing_api_key_raises(self):
        from unittest.mock import patch, MagicMock
        mock_settings = MagicMock()
        mock_settings.xai_api_key = None
        with patch("pyutilz.llm.xai_provider.get_llm_settings", return_value=mock_settings):
            with pytest.raises(ValueError, match="API key not provided"):
                XAIProvider(api_key=None)

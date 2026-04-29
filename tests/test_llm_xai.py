"""Tests for xAI (Grok) provider."""

import pytest

pytest.importorskip("httpx")

from pyutilz.llm.xai_provider import XAIProvider


class TestXAIConfig:
    def test_timeout_reasoning(self):
        p = XAIProvider.__new__(XAIProvider)
        assert p._get_timeout("grok-4-1-fast-reasoning") == 1200.0

    def test_timeout_non_reasoning(self):
        # Regression: "non-reasoning" must NOT match "reasoning" substring.
        # Pre-fix bug: this returned 1200.0 due to naive substring check.
        p = XAIProvider.__new__(XAIProvider)
        assert p._get_timeout("grok-4-1-fast-non-reasoning") == 240.0

    def test_timeout_grok4(self):
        # Premium reasoning model — long timeout
        p = XAIProvider.__new__(XAIProvider)
        assert p._get_timeout("grok-4") == 1200.0
        assert p._get_timeout("grok-4-0709") == 1200.0

    def test_timeout_code_model(self):
        p = XAIProvider.__new__(XAIProvider)
        assert p._get_timeout("grok-code-fast-1") == 240.0

    def test_timeout_grok3_mini(self):
        p = XAIProvider.__new__(XAIProvider)
        assert p._get_timeout("grok-3-mini") == 240.0

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

    def test_pricing_grok4_alias(self):
        # grok-4 should be aliased to same pricing as grok-4-0709
        p = XAIProvider.__new__(XAIProvider)
        assert p._input_cost_per_1m("grok-4") == 3.00
        assert p._output_cost_per_1m("grok-4") == 15.00

    def test_pricing_grok420_beta(self):
        # Regression: prior code had $2000/$6000 (1000x error). Should be $2/$6.
        p = XAIProvider.__new__(XAIProvider)
        for variant in (
            "grok-4.20-beta",
            "grok-4.20-multi-agent-beta-0309",
            "grok-4.20-beta-0309-reasoning",
            "grok-4.20-beta-0309-non-reasoning",
        ):
            assert p._input_cost_per_1m(variant) == 2.00, variant
            assert p._output_cost_per_1m(variant) == 6.00, variant
            assert p._cache_hit_cost_per_1m(variant) == 0.20, variant

    def test_cache_hit_cost(self):
        p = XAIProvider.__new__(XAIProvider)
        assert p._cache_hit_cost_per_1m("grok-4-1-fast-reasoning") == 0.05
        assert p._cache_hit_cost_per_1m("grok-4-0709") == 0.75
        assert p._cache_hit_cost_per_1m("grok-4") == 0.75
        assert p._cache_hit_cost_per_1m("grok-3-mini") == 0.07
        assert p._cache_hit_cost_per_1m("grok-code-fast-1") == 0.02

    def test_context_window_grok4(self):
        p = XAIProvider.__new__(XAIProvider)
        p.model_name = "grok-4-0709"
        assert p.context_window == 256_000

    def test_context_window_default_2m(self):
        # Fast/cheap models (grok-4-fast etc.) get the 2M default
        p = XAIProvider.__new__(XAIProvider)
        p.model_name = "grok-4-1-fast-reasoning"
        assert p.context_window == 2_000_000

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

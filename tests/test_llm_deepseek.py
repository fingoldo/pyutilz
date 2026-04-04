"""Tests for DeepSeek provider."""

import pytest

pytest.importorskip("httpx")

from pyutilz.llm.exceptions import LLMProviderError
from pyutilz.llm.deepseek_provider import DeepSeekProvider


class TestDeepSeekConfig:
    def test_timeout_reasoner(self):
        p = DeepSeekProvider.__new__(DeepSeekProvider)
        assert p._get_timeout("deepseek-reasoner") == 300.0

    def test_timeout_chat(self):
        p = DeepSeekProvider.__new__(DeepSeekProvider)
        assert p._get_timeout("deepseek-chat") == 120.0

    def test_billed_output_is_completion_only(self):
        p = DeepSeekProvider.__new__(DeepSeekProvider)
        assert p._compute_billed_output(100, 50) == 100

    def test_pricing_chat(self):
        p = DeepSeekProvider.__new__(DeepSeekProvider)
        p.model_name = "deepseek-chat"
        assert p._input_cost_per_1m("deepseek-chat") == 0.28
        assert p._output_cost_per_1m("deepseek-chat") == 0.42
        assert p._cache_hit_cost_per_1m("deepseek-chat") == 0.028

    def test_pricing_reasoner(self):
        p = DeepSeekProvider.__new__(DeepSeekProvider)
        p.model_name = "deepseek-reasoner"
        assert p._input_cost_per_1m("deepseek-reasoner") == 0.28
        assert p._output_cost_per_1m("deepseek-reasoner") == 0.42
        assert p._cache_hit_cost_per_1m("deepseek-reasoner") == 0.028

    def test_max_tokens_chat(self):
        p = DeepSeekProvider.__new__(DeepSeekProvider)
        p.model_name = "deepseek-chat"
        assert p.max_output_tokens == 8192

    def test_max_tokens_reasoner(self):
        p = DeepSeekProvider.__new__(DeepSeekProvider)
        p.model_name = "deepseek-reasoner"
        assert p.max_output_tokens == 65536

    def test_handle_special_status_402_warns(self):
        import httpx
        p = DeepSeekProvider.__new__(DeepSeekProvider)
        resp = httpx.Response(
            status_code=402,
            request=httpx.Request("POST", "https://api.deepseek.com/chat/completions"),
        )
        p._handle_special_status(resp)

    def test_handle_special_status_200_noop(self):
        import httpx
        p = DeepSeekProvider.__new__(DeepSeekProvider)
        resp = httpx.Response(
            status_code=200,
            request=httpx.Request("POST", "https://api.deepseek.com/chat/completions"),
        )
        p._handle_special_status(resp)

    def test_provider_name(self):
        assert DeepSeekProvider._provider_name == "DeepSeek"

    def test_base_url(self):
        assert DeepSeekProvider._base_url == "https://api.deepseek.com"

    def test_missing_api_key_raises(self):
        from unittest.mock import patch, MagicMock
        mock_settings = MagicMock()
        mock_settings.deepseek_api_key = None
        with patch("pyutilz.llm.deepseek_provider.get_llm_settings", return_value=mock_settings):
            with pytest.raises(ValueError, match="API key not provided"):
                DeepSeekProvider(api_key=None)

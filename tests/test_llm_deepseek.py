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

    def test_pricing_v4_flash(self):
        p = DeepSeekProvider.__new__(DeepSeekProvider)
        assert p._input_cost_per_1m("deepseek-v4-flash") == 0.14
        assert p._output_cost_per_1m("deepseek-v4-flash") == 0.28
        assert p._cache_hit_cost_per_1m("deepseek-v4-flash") == 0.0028

    def test_pricing_v4_pro(self):
        p = DeepSeekProvider.__new__(DeepSeekProvider)
        assert p._input_cost_per_1m("deepseek-v4-pro") == 1.74
        assert p._output_cost_per_1m("deepseek-v4-pro") == 3.48
        assert p._cache_hit_cost_per_1m("deepseek-v4-pro") == 0.0145

    def test_pricing_unknown_falls_back_to_v4_flash(self):
        p = DeepSeekProvider.__new__(DeepSeekProvider)
        # Unknown model should fall back to cheapest current default (v4-flash)
        assert p._input_cost_per_1m("future-model") == 0.14
        assert p._output_cost_per_1m("future-model") == 0.28

    def test_default_model_is_v4_flash(self):
        # Default model should be the cheapest current option
        from inspect import signature
        default = signature(DeepSeekProvider.__init__).parameters["model"].default
        assert default == "deepseek-v4-flash"

    def test_extra_request_body_empty_by_default(self):
        # No auto-injection; callers opt in via thinking= parameter on generate()
        p = DeepSeekProvider.__new__(DeepSeekProvider)
        for model in ("deepseek-v4-flash", "deepseek-v4-pro",
                      "deepseek-chat", "deepseek-reasoner"):
            assert p._extra_request_body(model) == {}, model

    def test_thinking_field_enabled_for_v4_flash(self):
        p = DeepSeekProvider.__new__(DeepSeekProvider)
        p.model_name = "deepseek-v4-flash"
        assert p._thinking_request_field(True) == {"thinking": {"type": "enabled"}}

    def test_thinking_field_disabled_for_v4_flash(self):
        p = DeepSeekProvider.__new__(DeepSeekProvider)
        p.model_name = "deepseek-v4-flash"
        assert p._thinking_request_field(False) == {"thinking": {"type": "disabled"}}

    def test_thinking_field_for_v4_pro(self):
        p = DeepSeekProvider.__new__(DeepSeekProvider)
        p.model_name = "deepseek-v4-pro"
        assert p._thinking_request_field(False) == {"thinking": {"type": "disabled"}}

    def test_thinking_field_none_for_legacy_aliases(self):
        # Legacy chat/reasoner are fixed-mode server-side; provider rejects
        # the thinking field, so we return None to skip emitting it.
        p = DeepSeekProvider.__new__(DeepSeekProvider)
        for legacy in ("deepseek-chat", "deepseek-reasoner"):
            p.model_name = legacy
            assert p._thinking_request_field(True) is None, legacy
            assert p._thinking_request_field(False) is None, legacy

    def test_timeout_v4_flash_short(self):
        p = DeepSeekProvider.__new__(DeepSeekProvider)
        assert p._get_timeout("deepseek-v4-flash") == 120.0

    def test_timeout_v4_pro_long(self):
        # v4-pro has thinking-mode capability → longer timeout
        p = DeepSeekProvider.__new__(DeepSeekProvider)
        assert p._get_timeout("deepseek-v4-pro") == 300.0

    def test_max_tokens_v4_flash(self):
        p = DeepSeekProvider.__new__(DeepSeekProvider)
        p.model_name = "deepseek-v4-flash"
        assert p.max_output_tokens == 384_000

    def test_max_tokens_v4_pro(self):
        p = DeepSeekProvider.__new__(DeepSeekProvider)
        p.model_name = "deepseek-v4-pro"
        assert p.max_output_tokens == 384_000

    def test_context_window_v4_flash(self):
        p = DeepSeekProvider.__new__(DeepSeekProvider)
        p.model_name = "deepseek-v4-flash"
        assert p.context_window == 1_000_000

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

from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import patch
import pytest


def test_anthropic_missing_key_names_pass_api_key():
    from pyutilz.llm.anthropic_provider import AnthropicProvider

    with patch("pyutilz.llm.anthropic_provider.get_llm_settings", return_value=SimpleNamespace(anthropic_api_key=None)):
        with pytest.raises(ValueError, match=r"pass api_key="):
            AnthropicProvider(api_key=None)

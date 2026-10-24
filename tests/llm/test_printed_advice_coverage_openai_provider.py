from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest


def test_openai_missing_key_names_pass_api_key():
    from pyutilz.llm.openai_provider import OpenAIProvider

    mock_settings = MagicMock()
    mock_settings.openai_api_key = None
    with patch("pyutilz.llm.openai_provider.get_llm_settings", return_value=mock_settings):
        with pytest.raises(ValueError, match=r"pass api_key="):
            OpenAIProvider(api_key=None)

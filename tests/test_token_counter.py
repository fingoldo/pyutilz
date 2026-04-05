from unittest.mock import patch, MagicMock
import pytest


def _import_fresh():
    """Import count_tokens after patching module globals."""
    from pyutilz.llm.token_counter import count_tokens
    return count_tokens


class TestCountTokensFallback:
    """Test the len//4 fallback path by forcing _HAS_TIKTOKEN=False."""

    def _call_fallback(self, text):
        import pyutilz.llm.token_counter as mod
        orig = mod._HAS_TIKTOKEN
        try:
            mod._HAS_TIKTOKEN = False
            return mod.count_tokens(text)
        finally:
            mod._HAS_TIKTOKEN = orig

    def test_empty_string(self):
        assert self._call_fallback("") == 0

    def test_normal_text(self):
        text = "hello world test"
        assert self._call_fallback(text) == len(text) // 4

    def test_returns_int(self):
        assert isinstance(self._call_fallback("some text here"), int)

    def test_fallback_approximation_reasonable(self):
        text = "The quick brown fox jumps over the lazy dog"
        result = self._call_fallback(text)
        word_count = len(text.split())
        assert result == len(text) // 4
        assert 0.5 * word_count <= result <= 3 * word_count


class TestCountTokensTiktoken:
    def test_tiktoken_path(self):
        fake_encoding = MagicMock()
        fake_encoding.encode.return_value = [1, 2, 3, 4, 5]

        import pyutilz.llm.token_counter as mod
        orig_has, orig_enc = mod._HAS_TIKTOKEN, getattr(mod, "_ENCODING", None)
        try:
            mod._HAS_TIKTOKEN = True
            mod._ENCODING = fake_encoding
            assert mod.count_tokens("hello world") == 5
            fake_encoding.encode.assert_called_once_with("hello world")
        finally:
            mod._HAS_TIKTOKEN = orig_has
            mod._ENCODING = orig_enc

    def test_tiktoken_empty_string(self):
        fake_encoding = MagicMock()
        fake_encoding.encode.return_value = []

        import pyutilz.llm.token_counter as mod
        orig_has, orig_enc = mod._HAS_TIKTOKEN, getattr(mod, "_ENCODING", None)
        try:
            mod._HAS_TIKTOKEN = True
            mod._ENCODING = fake_encoding
            assert mod.count_tokens("") == 0
        finally:
            mod._HAS_TIKTOKEN = orig_has
            mod._ENCODING = orig_enc

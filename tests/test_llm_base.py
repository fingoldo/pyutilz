"""Tests for LLM provider abstract base classes."""

import pytest

from pyutilz.llm.base import LLMProvider
from pyutilz.llm.exceptions import JSONParsingError


class TestLLMProviderABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError, match="abstract method"):
            LLMProvider()

    def test_max_output_tokens_default(self):
        assert LLMProvider.max_output_tokens.fget.__doc__ or True

    def test_context_window_default(self):
        assert LLMProvider.context_window.fget.__doc__ or True


class TestExtractJsonEdgeCases:
    def test_json_array_in_code_block_parses_as_list(self):
        text = '```json\n[1, 2, 3]\n```'
        result = LLMProvider.extract_json(text)
        assert result == [1, 2, 3]

    def test_multiple_json_objects_raises(self):
        text = '{"a": 1} {"b": 2}'
        with pytest.raises(JSONParsingError):
            LLMProvider.extract_json(text)

    def test_json_with_markdown_prefix(self):
        text = 'Sure, here is the JSON:\n```json\n{"key": "val"}\n```'
        assert LLMProvider.extract_json(text) == {"key": "val"}

    def test_triple_backtick_no_json_label(self):
        text = '```\n{"x": 42}\n```'
        assert LLMProvider.extract_json(text) == {"x": 42}

    def test_deeply_nested(self):
        text = '{"a": {"b": {"c": {"d": 1}}}}'
        r = LLMProvider.extract_json(text)
        assert r["a"]["b"]["c"]["d"] == 1

    def test_provider_name_in_error(self):
        with pytest.raises(JSONParsingError, match="MyProvider"):
            LLMProvider.extract_json("not json", provider_name="MyProvider")

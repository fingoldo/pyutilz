"""Behaviour of ``pyutilz.llm.openai_tokens``: the default encoding delegates to the shared counter, a non-default
encoding needs tiktoken and says how to install it, the chat-message framing adds 4 tokens per message and 2 to prime
the reply, a ``name`` field removes the role token, and a non-string value is counted as its text. tiktoken is replaced
by a whitespace-splitting fake, so the counts are exact and no network or model file is needed."""

from __future__ import annotations

import pytest

from pyutilz.llm import openai_tokens


class _FakeEncoding:
    def encode(self, text: str) -> list[str]:
        return text.split()

class _FakeTiktoken:
    """Knows no model name (forcing the cl100k fallback) and returns the whitespace encoder for every encoding."""

    @staticmethod
    def encoding_for_model(model: str):
        raise KeyError(model)

    @staticmethod
    def get_encoding(name: str):
        return _FakeEncoding()

@pytest.fixture
def fake_tiktoken(monkeypatch):
    monkeypatch.setattr(openai_tokens, "tiktoken", _FakeTiktoken)
    return _FakeTiktoken

class TestStringCount:
    def test_the_default_encoding_uses_the_shared_counter(self, monkeypatch):
        monkeypatch.setattr(openai_tokens, "count_tokens", lambda s: 42)
        assert openai_tokens.num_tokens_from_string("anything", "cl100k_base") == 42

    def test_a_non_default_encoding_is_counted_by_tiktoken(self, fake_tiktoken):
        assert openai_tokens.num_tokens_from_string("one two three", "o200k_base") == 3

    def test_a_non_default_encoding_without_tiktoken_says_how_to_install_it(self, monkeypatch):
        monkeypatch.setattr(openai_tokens, "tiktoken", None)
        with pytest.raises(ImportError, match="pip install pyutilz"):
            openai_tokens.num_tokens_from_string("x", "o200k_base")

class TestMessageFraming:
    def test_one_message_costs_its_framing_plus_the_reply_priming(self, fake_tiktoken):
        # 4 framing + 1 ("user") + 1 ("hi") + 2 priming
        assert openai_tokens.num_tokens_from_messages([{"role": "user", "content": "hi"}]) == 8

    def test_each_message_adds_its_own_framing(self, fake_tiktoken):
        two = [{"role": "user", "content": "hi"}, {"role": "user", "content": "hi"}]
        assert openai_tokens.num_tokens_from_messages(two) == 8 + 6

    def test_a_name_field_replaces_the_role_token(self, fake_tiktoken):
        # 4 framing + 1 ("user") + 1 ("bob") - 1 (name drops the role) + 1 ("x") + 2 priming
        assert openai_tokens.num_tokens_from_messages([{"role": "user", "name": "bob", "content": "x"}]) == 8

    def test_a_non_string_value_is_counted_as_its_text(self, fake_tiktoken):
        counted = openai_tokens.num_tokens_from_messages([{"role": "tool", "content": {"a": 1}}])
        assert counted == 4 + 1 + len(str({"a": 1}).split()) + 2

    def test_an_empty_message_list_is_only_the_priming(self, fake_tiktoken):
        assert openai_tokens.num_tokens_from_messages([]) == 2

    def test_messages_without_tiktoken_say_how_to_install_it(self, monkeypatch):
        monkeypatch.setattr(openai_tokens, "tiktoken", None)
        with pytest.raises(ImportError, match="pip install pyutilz"):
            openai_tokens.num_tokens_from_messages([{"role": "user", "content": "hi"}])

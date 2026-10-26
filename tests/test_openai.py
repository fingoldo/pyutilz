"""Tests for pyutilz.core.openai: the deprecated token-counting alias that resolves lazily to
pyutilz.llm.openai_tokens (audit finding #6: non-str message values must be coerced, not crash)."""

import pytest

tiktoken = pytest.importorskip("tiktoken")

from pyutilz.core.openai import num_tokens_from_messages, num_tokens_from_string


class TestNumTokensFromMessages:
    def test_basic_string_messages(self):
        msgs = [{"role": "user", "content": "hello world"}]
        n = num_tokens_from_messages(msgs)
        assert isinstance(n, int)
        assert n > 0

    def test_non_str_value_does_not_crash(self):
        """Audit #6: a non-str value (e.g. structured tool-call payload) must
        be coerced to str, not crash encoding.encode."""
        msgs = [{"role": "user", "content": {"nested": [1, 2, 3]}}]
        n = num_tokens_from_messages(msgs)
        assert isinstance(n, int)
        assert n > 0

    def test_name_key_overhead(self):
        msgs = [{"role": "user", "name": "alice", "content": "hi"}]
        n = num_tokens_from_messages(msgs)
        assert n > 0


class TestNumTokensFromString:
    def test_cl100k_delegates_and_counts(self):
        from pyutilz.llm.token_counter import count_tokens
        text = "the quick brown fox jumps over the lazy dog"
        assert num_tokens_from_string(text, "cl100k_base") == count_tokens(text)

    def test_other_encoding_direct(self):
        n = num_tokens_from_string("hello", "p50k_base")
        assert isinstance(n, int)
        assert n > 0

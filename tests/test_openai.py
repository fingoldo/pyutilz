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


class TestLazyAlias:
    def test_importing_the_alias_does_not_load_the_llm_package(self):
        import subprocess
        import sys
        from pathlib import Path

        import pyutilz

        src = str(Path(pyutilz.__file__).resolve().parents[1])
        code = "import sys, pyutilz.core.openai; print('pyutilz.llm.openai_tokens' in sys.modules)"
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": src}, check=True)
        assert result.stdout.strip() == "False"

    def test_an_unknown_name_is_refused_with_the_module_and_name(self):
        import pyutilz.core.openai as alias

        with pytest.raises(AttributeError) as info:
            alias.not_a_real_name
        assert str(info.value) == "module 'pyutilz.core.openai' has no attribute 'not_a_real_name'"

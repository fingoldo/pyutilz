"""tiktoken import-failure, warning and cache-size pins for llm.token_counter (mutation survivors)."""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import pytest

from pyutilz.llm import token_counter as tc


def _exec_module_with_failing_tiktoken(monkeypatch, env: dict | None = None) -> dict:
    """Run the module's source in a fresh namespace against a stub tiktoken whose get_encoding raises."""
    stub = types.ModuleType("tiktoken")

    def _fail(name):
        raise RuntimeError("boom")

    stub.get_encoding = _fail
    monkeypatch.setitem(sys.modules, "tiktoken", stub)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    source = Path(tc.__file__).read_text(encoding="utf-8")
    namespace: dict = {"__name__": "token_counter_probe", "__file__": tc.__file__}
    exec(compile(source, tc.__file__, "exec"), namespace)
    return namespace


class TestFallbackWhenTiktokenFails:
    def test_the_fallback_flag_is_cleared_and_counting_is_quarter_length(self, monkeypatch):
        namespace = _exec_module_with_failing_tiktoken(monkeypatch)
        assert namespace["_HAS_TIKTOKEN"] is False
        assert namespace["count_tokens"]("abcdefgh") == 2

    def test_the_failure_is_logged_with_the_exception_type_and_message(self, monkeypatch, caplog):
        caplog.set_level(logging.WARNING)
        _exec_module_with_failing_tiktoken(monkeypatch)
        assert "tiktoken unavailable (RuntimeError: boom) -- using len//4 fallback for token counting" in caplog.text


class TestCacheSizeEnvironment:
    def test_the_cache_bound_is_read_from_the_named_environment_variable(self, monkeypatch):
        namespace = _exec_module_with_failing_tiktoken(monkeypatch, env={"PYUTILZ_TOKEN_ENCODING_CACHE_MAX_SIZE": "3"})
        assert namespace["_ENCODING_CACHE_MAX_SIZE"] == 3


class TestRealTiktokenCount:
    def test_counting_uses_the_cl100k_base_encoding_when_tiktoken_is_present(self):
        tiktoken = pytest.importorskip("tiktoken")
        text = "The quick brown fox jumps over the lazy dog, 12345."
        expected = len(tiktoken.get_encoding("cl100k_base").encode(text))
        assert tc.count_tokens(text) == expected

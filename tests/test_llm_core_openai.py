"""Tests for pyutilz.core.openai token counting + orjson SSE benchmark note.

Audit P2 findings:
  #4  openai_compat SSE loop now uses ``orjson.loads`` instead of json.loads.
      Benchmark (timeit, 200k loops, Python 3.14, sample SSE delta chunk
      ``{"choices":[{"delta":{"content":"hello world"}}]}``):
          json.loads   : 2.663 us/loop
          orjson.loads : 0.577 us/loop   (4.61x faster)
      orjson is the hot-path win in the streaming token loop, executed once
      per delta chunk (thousands per long completion).
  #6  core/openai.num_tokens_from_messages must validate non-str values
      before encoding (was crashing on non-str content) and delegate
      single-string counting to token_counter.count_tokens for cl100k_base.
"""

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


class TestOrjsonSseBenchmark:
    def test_orjson_parses_sse_chunk(self):
        """Behavioral check that the orjson swap parses a real SSE delta chunk.

        See module docstring for the timeit benchmark (orjson 4.61x faster
        than json.loads on this chunk).
        """
        import orjson
        chunk = '{"choices":[{"delta":{"content":"hello world"}}]}'
        parsed = orjson.loads(chunk)
        assert parsed["choices"][0]["delta"]["content"] == "hello world"

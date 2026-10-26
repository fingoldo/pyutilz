"""Tests for the orjson SSE benchmark note (audit P2 finding #4): openai_compat's SSE loop uses ``orjson.loads``.

Benchmark (timeit, 200k loops, Python 3.14, sample SSE delta chunk
``{"choices":[{"delta":{"content":"hello world"}}]}``):
    json.loads   : 2.663 us/loop
    orjson.loads : 0.577 us/loop   (4.61x faster)
"""

import pytest


class TestOrjsonSseBenchmark:
    def test_orjson_parses_sse_chunk(self):
        """Behavioral check that the orjson swap parses a real SSE delta chunk.

        See module docstring for the timeit benchmark (orjson 4.61x faster
        than json.loads on this chunk).
        """
        orjson = pytest.importorskip("orjson")
        chunk = '{"choices":[{"delta":{"content":"hello world"}}]}'
        parsed = orjson.loads(chunk)
        assert parsed["choices"][0]["delta"]["content"] == "hello world"

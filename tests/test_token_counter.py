import os
import subprocess
import sys
import textwrap
from unittest.mock import MagicMock
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
        orig_has, orig_enc = mod._HAS_TIKTOKEN, mod._DEFAULT_ENCODING
        try:
            mod._HAS_TIKTOKEN = True
            mod._DEFAULT_ENCODING = fake_encoding
            assert mod.count_tokens("hello world") == 5
            fake_encoding.encode.assert_called_once_with("hello world")
        finally:
            mod._HAS_TIKTOKEN = orig_has
            mod._DEFAULT_ENCODING = orig_enc

    def test_tiktoken_empty_string(self):
        fake_encoding = MagicMock()
        fake_encoding.encode.return_value = []

        import pyutilz.llm.token_counter as mod
        orig_has, orig_enc = mod._HAS_TIKTOKEN, mod._DEFAULT_ENCODING
        try:
            mod._HAS_TIKTOKEN = True
            mod._DEFAULT_ENCODING = fake_encoding
            assert mod.count_tokens("") == 0
        finally:
            mod._HAS_TIKTOKEN = orig_has
            mod._DEFAULT_ENCODING = orig_enc


def test_module_import_degrades_to_fallback_on_non_import_error():
    """Regression sensor: a non-ImportError failure from tiktoken.get_encoding() (network/proxy/
    interpreter-SSL issues -- e.g. the Python 3.8 stdlib SSLContext.verify_mode recursion bug hit
    via urllib3 while tiktoken fetches its BPE file over the network) must degrade the module to
    the len//4 fallback, not crash the import. Pre-fix, ``except ImportError`` only caught "tiktoken
    not installed"; any other exception from the network-dependent get_encoding() call propagated
    and crashed `import pyutilz.llm.token_counter` entirely -- breaking every test in this file,
    including the ones that only exercise the no-tiktoken fallback path (observed in CI as
    RecursionError on Python 3.8, 2026-07-09).

    Run in a subprocess so patching tiktoken.get_encoding to raise happens before the module (whose
    encoding-load side effect only runs on first import) is ever imported in this process.
    """
    script = textwrap.dedent("""
        import sys
        from unittest.mock import patch

        import tiktoken

        with patch.object(tiktoken, "get_encoding", side_effect=RecursionError("simulated 3.8 ssl bug")):
            import pyutilz.llm.token_counter as mod

        assert mod._HAS_TIKTOKEN is False, "should degrade to fallback, not propagate the exception"
        assert mod._DEFAULT_ENCODING is None
        assert mod.count_tokens("hello world test") == len("hello world test") // 4
        print("OK")
        """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "OK" in result.stdout


class TestEncodingCacheBounded:
    """Regression (meta-test-driven finding, proactive-cache-audit pass): ``_encoding_cache`` used
    to be a plain unbounded dict -- one entry per distinct model-name string ever resolved, kept
    forever with no eviction. A long-running process resolving many distinct models over its
    lifetime (e.g. iterating an OpenRouter catalogue with 200+ entries) would grow this cache
    without bound. Fixed with the same OrderedDict + LRU-eviction pattern used elsewhere this
    round (llm.factory._provider_cache)."""

    def _reset(self, mod):
        mod._encoding_cache.clear()

    def test_cache_evicts_least_recently_used_beyond_max_size(self):
        import pyutilz.llm.token_counter as mod

        pytest.importorskip("tiktoken")
        self._reset(mod)
        orig_max = mod._ENCODING_CACHE_MAX_SIZE
        try:
            mod._ENCODING_CACHE_MAX_SIZE = 2
            mod._encoding_for_model("gpt-4o")
            mod._encoding_for_model("gpt-4o-mini")
            assert list(mod._encoding_cache.keys()) == ["gpt-4o", "gpt-4o-mini"]

            # A third distinct model pushes the cache over its bound -- the LEAST recently
            # used entry ("gpt-4o", not touched since insertion) must be evicted, not the
            # most-recently-inserted one.
            mod._encoding_for_model("gpt-3.5-turbo")
            assert len(mod._encoding_cache) == 2
            assert "gpt-4o" not in mod._encoding_cache, "least-recently-used entry should have been evicted"
            assert "gpt-3.5-turbo" in mod._encoding_cache
        finally:
            mod._ENCODING_CACHE_MAX_SIZE = orig_max
            self._reset(mod)

    def test_repeated_lookup_moves_entry_to_most_recently_used(self):
        import pyutilz.llm.token_counter as mod

        pytest.importorskip("tiktoken")
        self._reset(mod)
        orig_max = mod._ENCODING_CACHE_MAX_SIZE
        try:
            mod._ENCODING_CACHE_MAX_SIZE = 2
            mod._encoding_for_model("gpt-4o")
            mod._encoding_for_model("gpt-4o-mini")
            # Re-touch "gpt-4o" so it becomes the most-recently-used entry instead.
            mod._encoding_for_model("gpt-4o")
            mod._encoding_for_model("gpt-3.5-turbo")

            assert "gpt-4o" in mod._encoding_cache, "recently re-touched entry should survive eviction"
            assert "gpt-4o-mini" not in mod._encoding_cache, "the now-least-recently-used entry should be evicted instead"
        finally:
            mod._ENCODING_CACHE_MAX_SIZE = orig_max
            self._reset(mod)


def _run_module_script(body: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("PYUTILZ_TOKEN_ENCODING_CACHE_MAX_SIZE", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run([sys.executable, "-c", textwrap.dedent(body)], capture_output=True, text=True, timeout=60, env=env)


def test_the_default_encoding_is_cl100k_base_and_it_is_loaded_at_import():
    """Run in a subprocess: the encoding is loaded at import, so the name is observed before the module is cached (openai_provider-style pin, token_counter 29)."""
    result = _run_module_script(
        """
        import tiktoken
        from unittest.mock import patch

        seen = []

        def fake(name):
            seen.append(name)
            return object()

        with patch.object(tiktoken, "get_encoding", side_effect=fake):
            import pyutilz.llm.token_counter as mod

        assert seen == ["cl100k_base"], seen
        assert mod._HAS_TIKTOKEN is True
        print("OK")
        """
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"


def test_a_missing_encoding_logs_the_exact_fallback_warning_on_stderr():
    result = _run_module_script(
        """
        import tiktoken
        from unittest.mock import patch

        with patch.object(tiktoken, "get_encoding", side_effect=RuntimeError("boom")):
            import pyutilz.llm.token_counter as mod

        assert mod._HAS_TIKTOKEN is False
        print("OK")
        """
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "tiktoken unavailable (RuntimeError: boom) -- using len//4 fallback for token counting" in result.stderr


def test_the_encoding_cache_size_defaults_to_256_and_reads_its_environment_variable():
    default = _run_module_script("import pyutilz.llm.token_counter as mod\nprint(mod._ENCODING_CACHE_MAX_SIZE)\n")
    assert default.returncode == 0 and default.stdout.strip() == "256"
    sized = _run_module_script(
        "import pyutilz.llm.token_counter as mod\nprint(mod._ENCODING_CACHE_MAX_SIZE)\n",
        env_extra={"PYUTILZ_TOKEN_ENCODING_CACHE_MAX_SIZE": "7"},
    )
    assert sized.returncode == 0 and sized.stdout.strip() == "7"

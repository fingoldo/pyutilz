import subprocess
import sys
import textwrap
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

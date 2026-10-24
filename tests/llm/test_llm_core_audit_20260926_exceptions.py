from __future__ import annotations


class TestStreamInterruptedReexport:
    def test_one_class_under_every_import_path(self):
        import pyutilz.llm as llm
        from pyutilz.llm import _openai_compat_http, exceptions, openai_compat

        cls = exceptions.LLMStreamInterruptedError
        assert llm.LLMStreamInterruptedError is cls
        assert _openai_compat_http.LLMStreamInterruptedError is cls
        assert openai_compat.LLMStreamInterruptedError is cls
        assert issubclass(cls, exceptions.LLMProviderError)
        err = cls("boom", code="server_error", partial_text="ab", retryable=False)
        assert (err.code, err.partial_text, err.retryable) == ("server_error", "ab", False)

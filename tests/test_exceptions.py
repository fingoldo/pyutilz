"""Behaviour of the typed exception hierarchies in ``pyutilz.database.exceptions``, ``pyutilz.llm.exceptions``
and ``pyutilz.web.exceptions``: each domain's root catches every subclass, each subclass keeps its builtin base so
existing ``except ValueError`` / ``except RuntimeError`` handlers still match, and the attributes callers rely on
(``details``, ``partial_text``, ``retryable`` and the rest) carry the values passed at the raise site."""

from __future__ import annotations

import pytest

from pyutilz.database.exceptions import DatabaseConnectionError, DatabaseError, SQLValidationError
from pyutilz.llm.exceptions import (
    ClaudeCodeToolUseError,
    JSONParsingError,
    LLMProviderError,
    LLMRefusalError,
    LLMSafetyBlockError,
    LLMStreamInterruptedError,
    LLMTruncationError,
    LLMUnparseableResponseError,
)
from pyutilz.web.exceptions import ProxyConfigurationError, ProxyFetchError, UnsafeURLError, WebError


class TestDatabaseHierarchy:
    @pytest.mark.parametrize("cls", [DatabaseConnectionError, SQLValidationError])
    def test_the_domain_root_catches_every_subclass(self, cls):
        with pytest.raises(DatabaseError):
            raise cls("boom")

    def test_the_connection_error_keeps_its_runtime_error_base(self):
        with pytest.raises(RuntimeError):
            raise DatabaseConnectionError("not connected")

    def test_the_sql_validation_error_keeps_its_value_error_base(self):
        with pytest.raises(ValueError):
            raise SQLValidationError("bad identifier")


class TestLLMHierarchy:
    @pytest.mark.parametrize(
        "exc",
        [
            JSONParsingError("x"),
            LLMRefusalError("x"),
            LLMSafetyBlockError("x"),
            LLMTruncationError("x"),
            LLMUnparseableResponseError("x"),
            ClaudeCodeToolUseError("x"),
            LLMStreamInterruptedError("x"),
        ],
    )
    def test_the_provider_root_catches_every_subclass(self, exc):
        with pytest.raises(LLMProviderError):
            raise exc

    def test_a_safety_block_is_a_refusal_so_it_is_not_retried(self):
        assert isinstance(LLMSafetyBlockError("x"), LLMRefusalError)

    def test_malformed_model_json_still_matches_a_value_error_handler(self):
        with pytest.raises(ValueError):
            raise JSONParsingError("x")

    def test_a_truncation_still_matches_a_value_error_handler(self):
        with pytest.raises(ValueError):
            raise LLMTruncationError("x")

    def test_an_escaped_tool_use_is_a_runtime_error_not_an_os_error(self):
        err = ClaudeCodeToolUseError("tool call")
        assert isinstance(err, RuntimeError)
        assert not isinstance(err, OSError)

    def test_the_details_default_to_an_empty_dict(self):
        assert LLMProviderError("x").details == {}

    def test_the_details_passed_at_raise_are_kept(self):
        assert LLMProviderError("x", details={"code": 1}).details == {"code": 1}

    def test_a_refusal_keeps_its_raw_text(self):
        assert LLMRefusalError("refused", raw_text="I cannot").raw_text == "I cannot"

    def test_a_truncation_keeps_the_text_produced_before_the_cutoff(self):
        err = LLMTruncationError("cut", finish_reason="length", partial_text="half an answ")
        assert (err.finish_reason, err.partial_text) == ("length", "half an answ")

    def test_a_missing_partial_text_becomes_an_empty_string_not_none(self):
        assert LLMTruncationError("cut", partial_text=None).partial_text == ""  # type: ignore[arg-type]

    def test_an_unparseable_body_records_its_status_and_excerpt(self):
        err = LLMUnparseableResponseError("html page", status_code=502, body_excerpt="<html>")
        assert (err.status_code, err.body_excerpt) == (502, "<html>")
        assert err.details == {"status_code": 502, "body_excerpt": "<html>"}

    def test_a_stream_interruption_records_its_code_and_retryability(self):
        err = LLMStreamInterruptedError("upstream", code="server_error", partial_text="pa", retryable=False)
        assert (err.code, err.partial_text, err.retryable) == ("server_error", "pa", False)

    def test_a_stream_interruption_is_retryable_by_default(self):
        assert LLMStreamInterruptedError("upstream").retryable is True


class TestWebHierarchy:
    @pytest.mark.parametrize("cls", [ProxyConfigurationError, ProxyFetchError, UnsafeURLError])
    def test_the_web_root_catches_every_subclass(self, cls):
        with pytest.raises(WebError):
            raise cls("boom")

    def test_a_proxy_configuration_error_keeps_its_runtime_error_base(self):
        with pytest.raises(RuntimeError):
            raise ProxyConfigurationError("missing key")

    def test_a_rejected_scheme_is_a_value_error_and_not_retryable_as_os_error(self):
        err = UnsafeURLError("file://")
        assert isinstance(err, ValueError)
        assert not isinstance(err, OSError)

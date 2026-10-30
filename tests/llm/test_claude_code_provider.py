"""Behaviour of ``pyutilz.llm.claude_code_provider``: the default provider starts with zeroed cost accounting and reports its
calls as free (subscription billing), the assistant-message reader joins text blocks with newlines, returns nothing for an
empty message, and refuses a tool-use block outright, and the temporary system-prompt file is removed when present and
ignored when absent. The SDK is optional and is not installed here, so the text block type is injected for the reader
tests."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from pyutilz.llm import claude_code_provider as ccp
from pyutilz.llm.exceptions import ClaudeCodeToolUseError


class _TextBlock:
    def __init__(self, text):
        self.text = text


class ToolUseBlock:
    def __init__(self, name):
        self.name = name


@pytest.fixture
def provider():
    return ccp.ClaudeCodeProvider()


class TestDefaults:
    def test_a_new_provider_has_zeroed_session_cost(self, provider):
        cost = provider.get_session_cost()
        assert cost["calls"] == 0 and cost["successful_calls"] == 0
        assert cost["total_cost_usd"] == 0.0

    def test_calls_are_free_under_subscription_billing(self, provider):
        assert provider.estimate_cost(1_000_000, 1_000_000) == 0.0

    def test_the_context_and_output_limits_are_positive(self, provider):
        assert provider.max_output_tokens > 0
        assert provider.context_window > 0


class TestAssistantMessageText:
    def test_text_blocks_are_joined_with_newlines(self, provider, monkeypatch):
        monkeypatch.setattr(ccp, "TextBlock", _TextBlock, raising=False)
        msg = SimpleNamespace(content=[_TextBlock("first"), _TextBlock("second")])
        assert provider._assistant_message_text(msg) == "first\nsecond"

    def test_a_message_without_content_gives_none(self, provider):
        assert provider._assistant_message_text(SimpleNamespace(content=[])) is None
        assert provider._assistant_message_text(SimpleNamespace()) is None

    def test_a_tool_use_block_is_refused_outright(self, provider, monkeypatch):
        monkeypatch.setattr(ccp, "TextBlock", _TextBlock, raising=False)
        msg = SimpleNamespace(content=[_TextBlock("hello"), ToolUseBlock("bash")])
        with pytest.raises(ClaudeCodeToolUseError, match="bash"):
            provider._assistant_message_text(msg)


class TestSystemPromptFile:
    def test_an_existing_file_is_removed(self, tmp_path):
        path = tmp_path / "system.txt"
        path.write_text("steer", encoding="utf-8")
        ccp._remove_system_prompt_file(str(path))
        assert not os.path.exists(path)

    def test_a_missing_file_or_no_path_is_ignored(self, tmp_path):
        absent = tmp_path / "absent.txt"
        assert ccp._remove_system_prompt_file(str(absent)) is None
        assert not absent.exists()
        assert ccp._remove_system_prompt_file(None) is None


class TestSdkVersionGate:
    def test_the_supported_sdk_versions_are_the_pinned_range(self):
        pytest.importorskip("claude_code_sdk")
        assert ccp._SUPPORTED_SDK_VERSIONS == frozenset({"0.0.25", "0.0.26", "0.0.27", "0.0.28", "0.0.29", "0.0.30"})

    def test_the_patches_are_applied_for_a_supported_sdk(self):
        pytest.importorskip("claude_code_sdk")
        assert ccp._CCSDK_VERSION in ccp._SUPPORTED_SDK_VERSIONS
        assert ccp._PATCHES_APPLIED is True


class TestPatchedMessageParser:
    def test_an_unparseable_sdk_message_is_dropped_with_a_warning(self, caplog):
        import logging

        mp = pytest.importorskip("claude_code_sdk._internal.message_parser")
        with caplog.at_level(logging.WARNING, logger=ccp.logger.name):
            result = mp.parse_message({"type": "definitely-unknown-type-xyz"})
        assert result is None
        assert any("SDK message parse failed, message dropped" in rec.getMessage() for rec in caplog.records)

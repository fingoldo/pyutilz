"""SDK version gate and command-builder pins for claude_code_provider (mutation survivors)."""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from pyutilz.llm import claude_code_provider as ccp


def _stub_sdk(version: "str | None") -> types.ModuleType:
    """A stand-in ``claude_code_sdk`` exposing the names the provider imports, with an optional ``__version__``."""
    stub = types.ModuleType("claude_code_sdk")
    stub.query = object
    stub.ClaudeCodeOptions = object
    stub.ResultMessage = object
    stub.TextBlock = object
    if version is not None:
        stub.__version__ = version
    return stub


def _exec_provider_source(monkeypatch, stub: types.ModuleType) -> dict:
    """Run the provider's source as a fresh namespace against ``stub``; the real module and its patches are untouched.

    An unsupported version takes the branch that logs and returns before any SDK patch is applied, so the
    namespace has no process-wide side effects.
    """
    monkeypatch.setitem(sys.modules, "claude_code_sdk", stub)
    source = Path(ccp.__file__).read_text(encoding="utf-8")
    namespace: dict = {"__name__": "claude_code_provider_probe", "__file__": ccp.__file__}
    exec(compile(source, ccp.__file__, "exec"), namespace)
    return namespace


class TestUnsupportedSdkVersion:
    def test_an_unsupported_version_is_logged_with_its_number_and_the_supported_set(self, monkeypatch, caplog):
        caplog.set_level(logging.ERROR)
        _exec_provider_source(monkeypatch, _stub_sdk("9.9.9"))
        assert (
            "claude-code-sdk version 9.9.9 is NOT in supported set "
            "['0.0.25', '0.0.26', '0.0.27', '0.0.28', '0.0.29', '0.0.30']. "
            "Monkey-patches SKIPPED -- using unpatched SDK. "
            "Review and update patches in claude_code_provider.py for this version."
        ) in caplog.text

    def test_the_patches_are_not_applied_for_an_unsupported_version(self, monkeypatch):
        namespace = _exec_provider_source(monkeypatch, _stub_sdk("9.9.9"))
        assert namespace["_PATCHES_APPLIED"] is False

    def test_a_sdk_without_a_version_attribute_is_reported_as_unknown(self, monkeypatch, caplog):
        caplog.set_level(logging.ERROR)
        _exec_provider_source(monkeypatch, _stub_sdk(None))
        assert "claude-code-sdk version unknown is NOT in supported set" in caplog.text

    def test_the_mismatch_is_logged_at_error_level(self, monkeypatch, caplog):
        caplog.set_level(logging.ERROR)
        _exec_provider_source(monkeypatch, _stub_sdk("9.9.9"))
        assert any(r.levelno == logging.ERROR and "9.9.9" in r.getMessage() for r in caplog.records)


class TestPatchedCommandBuilderGate:
    def test_a_streaming_command_is_never_rewritten_even_when_it_carries_print(self, monkeypatch):
        monkeypatch.setattr(ccp, "_orig_build", lambda self: ["claude", "--print", "x"])
        transport = SimpleNamespace(_is_streaming=True)
        assert ccp._patched_build(transport) == ["claude", "--print", "x"]

    def test_a_non_streaming_command_without_print_is_left_alone(self, monkeypatch):
        monkeypatch.setattr(ccp, "_orig_build", lambda self: ["claude", "--verbose"])
        transport = SimpleNamespace(_is_streaming=False)
        assert ccp._patched_build(transport) == ["claude", "--verbose"]

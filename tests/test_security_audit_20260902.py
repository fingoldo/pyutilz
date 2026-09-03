"""Regression tests for the 2026-09-02 security audit (audits/2026-09-02/02-security.md, F01-F07).

Everything here is behavioral and fully offline: no network socket is opened, no `claude` CLI is
spawned (the argv is inspected, never executed), and no database is contacted.
"""

from __future__ import annotations

import io
import os
import stat
import sys
import urllib.error
import urllib.request
from contextlib import contextmanager
from unittest.mock import patch

import pytest

# --------------------------------------------------------------------------------------------
# F01 -- suggest_json_optimization's `path` must be validated before it reaches raw SQL.
# --------------------------------------------------------------------------------------------


class TestF01JsonPathValidation:
    def test_injection_path_rejected(self):
        db = pytest.importorskip("pyutilz.database.db")
        with patch.object(db, "validate_sql_identifier", lambda *_a, **_kw: None), patch.object(db, "safe_execute") as exec_mock:
            with pytest.raises(ValueError):
                db.suggest_json_optimization(table="t", table_field="data", path="'a'; drop table users; --", fields=["x"])
        exec_mock.assert_not_called()

    def test_benign_path_accepted(self):
        db = pytest.importorskip("pyutilz.database.db")
        assert db._JSON_PATH_RE.match("'a'->'b'")
        assert db._JSON_PATH_RE.match("0")
        assert not db._JSON_PATH_RE.match("'a';drop table t")


# --------------------------------------------------------------------------------------------
# F02 -- Claude Code must not load the invoking user's ambient MCP servers, and an escaped
# tool-use block must be a hard failure rather than a log line.
# --------------------------------------------------------------------------------------------


def _fake_cli_proc_factory(captured: dict):
    """Return a ``subprocess.Popen`` stand-in that records argv and replays one successful stream-json result event."""
    import json as _json

    class _FakeProc:
        args = ["claude"]
        returncode = 0

        def __init__(self, cmd, **_kwargs):
            captured["cmd"] = list(cmd)
            self.stdin = io.StringIO()
            self.stdout = io.StringIO(_json.dumps({"type": "result", "subtype": "success", "result": "ok"}) + "\n")
            self.stderr = io.StringIO("")

        def kill(self):
            pass

        def wait(self, timeout=None):
            return 0

    return _FakeProc


def _fake_sdk_query(captured: dict, messages):
    """Return a ``cc_query`` stand-in that records the ``ClaudeCodeOptions`` it was handed and replays ``messages``."""

    def _query(prompt=None, options=None):
        captured["options"] = options

        async def _gen():
            for m in messages:
                yield m

        return _gen()

    return _query


class _StubAssistantMessage:
    """Stands in for the SDK's ``AssistantMessage``; the provider dispatches on the class NAME."""

    def __init__(self, content):
        self.content = content


_StubAssistantMessage.__name__ = "AssistantMessage"


class _StubToolUseBlock:
    """Stands in for the SDK's ``ToolUseBlock``; the provider dispatches on the class NAME."""

    def __init__(self, name):
        self.name = name


_StubToolUseBlock.__name__ = "ToolUseBlock"


class TestF02ClaudeCodeMcpIsolation:
    def test_cli_argv_carries_strict_mcp_config(self):
        import asyncio

        from pyutilz.llm import claude_code_provider as ccp

        provider = ccp.ClaudeCodeProvider(model="opus", timeout=30)
        provider._claude_path = "claude-not-executed"
        captured: dict = {}
        with patch.object(ccp.subprocess, "Popen", _fake_cli_proc_factory(captured)):
            assert asyncio.run(provider._generate_cli("hi")) == "ok"

        cmd = captured["cmd"]
        assert "--strict-mcp-config" in cmd, cmd
        # the flag only isolates MCP when no --mcp-config supplies servers of its own
        assert "--mcp-config" not in cmd, cmd

    def test_sdk_extra_args_carry_strict_mcp_config(self):
        import asyncio

        pytest.importorskip("claude_code_sdk")
        from pyutilz.llm import claude_code_provider as ccp

        provider = ccp.ClaudeCodeProvider(model="opus", timeout=30)
        captured: dict = {}
        messages = [_StubAssistantMessage([ccp.TextBlock(text="ok")])]
        with patch.object(ccp, "cc_query", _fake_sdk_query(captured, messages)):
            assert asyncio.run(provider._generate_sdk("hi")) == "ok"

        extra_args = captured["options"].extra_args
        assert "strict-mcp-config" in extra_args
        # a None value is how the SDK spells a valueless CLI flag
        assert extra_args["strict-mcp-config"] is None
        assert extra_args["tools"] == ""

    def test_tool_use_error_is_not_retryable_exception_type(self):
        from pyutilz.llm.claude_code_provider import ClaudeCodeToolUseError

        # The provider's transient-failure retry arm catches (ConnectionError, TimeoutError, OSError,
        # subprocess.TimeoutExpired); a security event must not be swallowed by it.
        assert issubclass(ClaudeCodeToolUseError, RuntimeError)
        assert not issubclass(ClaudeCodeToolUseError, OSError)

    def test_tool_use_block_raises(self, caplog):
        import asyncio

        pytest.importorskip("claude_code_sdk")
        from pyutilz.llm import claude_code_provider as ccp

        provider = ccp.ClaudeCodeProvider(model="opus", timeout=30)
        captured: dict = {}
        messages = [_StubAssistantMessage([ccp.TextBlock(text="before"), _StubToolUseBlock("Bash"), ccp.TextBlock(text="after")])]
        with patch.object(ccp, "cc_query", _fake_sdk_query(captured, messages)):
            caplog.set_level("WARNING")
            with pytest.raises(ccp.ClaudeCodeToolUseError) as excinfo:
                asyncio.run(provider._generate_sdk("hi"))

        assert "Bash" in str(excinfo.value)
        # the old behaviour was a bare logger.warning(... "(blocked)") that let the turn continue
        assert "attempted tool use" not in caplog.text

    def test_text_only_stream_is_not_treated_as_tool_use(self):
        import asyncio

        pytest.importorskip("claude_code_sdk")
        from pyutilz.llm import claude_code_provider as ccp

        provider = ccp.ClaudeCodeProvider(model="opus", timeout=30)
        captured: dict = {}
        messages = [_StubAssistantMessage([ccp.TextBlock(text="plain answer")])]
        with patch.object(ccp, "cc_query", _fake_sdk_query(captured, messages)):
            assert asyncio.run(provider._generate_sdk("hi")) == "plain answer"


# --------------------------------------------------------------------------------------------
# F03 -- a server-supplied Retry-After must be clamped.
# --------------------------------------------------------------------------------------------


def _http_error(code: int, retry_after: str | None):
    headers = {"Retry-After": retry_after} if retry_after is not None else {}

    @contextmanager
    def _open(*_args, **_kwargs):
        raise urllib.error.HTTPError("https://example.org/x", code, "throttled", headers, None)  # type: ignore[arg-type]
        yield  # pragma: no cover

    return _open


class TestF03RetryAfterClamp:
    def test_absurd_retry_after_is_not_slept(self, tmp_path):
        from pyutilz.web.cached_client import MAX_RETRY_AFTER_SECONDS, CachedHttpClient

        client = CachedHttpClient(cache_dir=tmp_path, min_interval=0.0)
        slept: list[float] = []
        with patch("pyutilz.web.cached_client.urlopen_checked", _http_error(429, "999999999")), patch("pyutilz.web.cached_client.time.sleep", slept.append):
            assert client.get_json("https://example.org/x", "tag", retries=2) is None
        assert slept, "the retry path must still back off"
        assert max(slept) <= MAX_RETRY_AFTER_SECONDS

    def test_reasonable_retry_after_is_honored(self, tmp_path):
        from pyutilz.web.cached_client import CachedHttpClient

        client = CachedHttpClient(cache_dir=tmp_path, min_interval=0.0)
        slept: list[float] = []
        with patch("pyutilz.web.cached_client.urlopen_checked", _http_error(429, "7")), patch("pyutilz.web.cached_client.time.sleep", slept.append):
            client.get_json("https://example.org/x", "tag", retries=2)
        assert 7.0 in slept


# --------------------------------------------------------------------------------------------
# F04 -- a traversing cache `tag` must be refused, not mkdir'd outside cache_dir.
# --------------------------------------------------------------------------------------------


class TestF04TagTraversal:
    @pytest.mark.parametrize("tag", ["../../evil", "..", ".", "a/b", "a\\b", "/abs", "", "C:evil"])
    def test_unsafe_tags_rejected(self, tmp_path, tag):
        from pyutilz.web.cached_client import CachedHttpClient

        client = CachedHttpClient(cache_dir=tmp_path)
        with pytest.raises(ValueError):
            client._cache_path("https://example.org/x", tag)

    def test_no_directory_created_outside_cache_dir(self, tmp_path):
        from pyutilz.web.cached_client import CachedHttpClient

        root = tmp_path / "cache"
        root.mkdir()
        client = CachedHttpClient(cache_dir=root)
        with pytest.raises(ValueError):
            client.get_json("https://example.org/x", "../../pwned")
        assert not (tmp_path.parent / "pwned").exists()

    def test_normal_tags_still_work(self, tmp_path):
        from pyutilz.web.cached_client import CachedHttpClient

        client = CachedHttpClient(cache_dir=tmp_path)
        for tag in ("vocab", "sim_brazil", "v1.2", "a-b_c"):
            assert client._cache_path("https://example.org/x", tag).parent.name == tag


# --------------------------------------------------------------------------------------------
# F05 -- the FileMaker auth response body must never be logged.
# --------------------------------------------------------------------------------------------


class TestF05FilemakerTokenLeak:
    def test_malformed_token_response_body_not_logged(self, caplog):
        fm = pytest.importorskip("pyutilz.core.filemaker")

        secret = "SECRET-BEARER-TOKEN-VALUE"  # pragma: allowlist secret
        payload = {"response": {"sessionToken": secret}, "messages": [{"code": "0"}]}

        class _Resp:
            status_code = 200

            @staticmethod
            def json():
                return payload

        with patch.object(fm, "filemaker_url", "https://fm.example.org/api"), patch.object(fm.web, "connect", lambda **_kw: None), patch.object(
            fm.web, "get_url", lambda *_a, **_kw: _Resp()
        ), patch.object(fm, "sleep", lambda *_a: None):
            caplog.set_level("WARNING")
            token = fm.get_session_token(username="u", password="p", max_retries=1, sleep_int_seconds=0)

        assert token is None
        joined = caplog.text
        assert secret not in joined
        assert "keys=['messages', 'response']" in joined  # the SHAPE (top-level keys) is still reported for diagnosis


# --------------------------------------------------------------------------------------------
# F06 -- the scheme allow-list must apply to redirect hops too.
# --------------------------------------------------------------------------------------------


class TestF06RedirectSchemeCheck:
    def test_ftp_redirect_is_refused(self):
        from pyutilz.web.exceptions import UnsafeURLError
        from pyutilz.web.url_guard import _CheckedRedirectHandler

        handler = _CheckedRedirectHandler()
        req = urllib.request.Request("https://example.org/start")
        with pytest.raises(UnsafeURLError):
            handler.redirect_request(req, io.BytesIO(b""), 302, "Found", {}, "ftp://internal-host/secret")

    def test_http_redirect_still_allowed(self):
        from pyutilz.web.url_guard import _CheckedRedirectHandler

        handler = _CheckedRedirectHandler()
        req = urllib.request.Request("https://example.org/start")
        new_req = handler.redirect_request(req, io.BytesIO(b""), 302, "Found", {}, "https://example.org/next")
        assert new_req is not None and new_req.full_url == "https://example.org/next"

    def test_opener_used_by_urlopen_checked_installs_the_handler(self):
        from pyutilz.web import url_guard

        captured = {}

        class _FakeOpener:
            def open(self, *_a, **_kw):
                return "opened"

        def _fake_build_opener(*handlers):
            captured["handlers"] = handlers
            return _FakeOpener()

        with patch.object(url_guard.urllib.request, "build_opener", _fake_build_opener):
            assert url_guard.urlopen_checked("https://example.org/x") == "opened"
        assert any(isinstance(h, url_guard._CheckedRedirectHandler) for h in captured["handlers"])


# --------------------------------------------------------------------------------------------
# F07 -- payload/cache files must not be created world-readable.
# --------------------------------------------------------------------------------------------

_POSIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits are not modelled on Windows")


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


@contextmanager
def _record_os_open():
    """Record every ``os.open`` call made inside the block as ``(path, flags, mode)``, delegating to the real call."""
    calls: list[tuple] = []
    real_open = os.open

    def _spy(path, flags, mode=0o777, *args, **kwargs):
        calls.append((path, flags, mode))
        return real_open(path, flags, mode, *args, **kwargs)

    with patch("os.open", _spy):
        yield calls


def _creating_modes(calls) -> list[int]:
    """Modes of the O_CREAT calls among recorded ``os.open`` calls."""
    return [mode for _path, flags, mode in calls if flags & os.O_CREAT]


class TestF07FilePermissions:
    def test_atomic_write_bytes_creates_with_explicit_mode(self, tmp_path):
        from pyutilz.core.serialization import atomic_write_bytes

        target = tmp_path / "payload.bin"
        with _record_os_open() as calls:
            atomic_write_bytes(str(target), lambda f: f.write(b"x"))

        creating = [(flags, mode) for _p, flags, mode in calls if flags & os.O_CREAT]
        assert creating, "atomic_write_bytes must create its temp file through os.open with an explicit mode"
        assert all(flags & os.O_EXCL for flags, _m in creating), creating
        assert all(mode & 0o077 == 0 for _f, mode in creating), creating
        assert target.read_bytes() == b"x"

    def test_safe_dump_and_disk_cache_create_with_explicit_mode(self, tmp_path):
        from pyutilz.core.disk_cache import DiskCache
        from pyutilz.core.safe_pickle import safe_dump

        target = tmp_path / "obj.pkl"
        with _record_os_open() as calls:
            safe_dump({"a": 1}, str(target))
        modes = _creating_modes(calls)
        assert modes, "safe_dump must create its temp file through os.open with an explicit mode"
        assert all(mode & 0o077 == 0 for mode in modes), modes

        cache_dir = tmp_path / "cache"
        cache = DiskCache(cache_dir=cache_dir)
        with _record_os_open() as calls:
            cache.put("deadbeef", {"a": 1})
        modes = _creating_modes(calls)
        assert modes, "DiskCache.put must create its temp file through os.open with an explicit mode"
        assert all(mode & 0o077 == 0 for mode in modes), modes

    @_POSIX_ONLY
    def test_atomic_write_bytes_result_is_owner_only(self, tmp_path):
        from pyutilz.core.serialization import atomic_write_bytes

        target = tmp_path / "payload.bin"
        atomic_write_bytes(str(target), lambda f: f.write(b"x"))
        assert _mode(target) & 0o077 == 0

    @_POSIX_ONLY
    def test_safe_dump_result_is_owner_only(self, tmp_path):
        from pyutilz.core.safe_pickle import safe_dump

        target = tmp_path / "obj.pkl"
        safe_dump({"a": 1}, str(target))
        assert _mode(target) & 0o077 == 0

    @_POSIX_ONLY
    def test_disk_cache_entry_is_owner_only(self, tmp_path):
        from pyutilz.core.disk_cache import DiskCache

        cache = DiskCache(cache_dir=tmp_path)
        cache.put("deadbeef", {"a": 1})
        entries = list(tmp_path.glob("*.pkl"))
        assert entries
        assert _mode(entries[0]) & 0o077 == 0

"""Regression tests for the 2026-09-03 security audit (audits/2026-09-03/02-security.md).

Every test here is fully offline: no network, no real browser or chromedriver, no database. The
browser tests call the extension's JS builder and the sandbox predicate directly, the HTTP
tests patch ``urlopen_checked``/build a ``HTTPRedirectHandler`` directly, and the logging test
exercises the ``logged`` decorator with ``db_path=None`` so nothing is persisted.
"""

from __future__ import annotations

import ast
import io
import json
import sys
import zlib
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# F01 -- Chrome sandbox is no longer disabled unconditionally
# ---------------------------------------------------------------------------


class TestSandboxNotDisabledUnconditionally:
    def test_non_root_keeps_the_sandbox(self):
        from pyutilz.web import browser

        with patch.object(browser, "no_sandbox", None), patch.object(browser.os, "geteuid", lambda: 1000, create=True):
            assert browser._should_disable_sandbox() is False

    def test_root_still_gets_no_sandbox(self):
        """The containerised-as-root case Chrome genuinely refuses to start in keeps working."""
        from pyutilz.web import browser

        with patch.object(browser, "no_sandbox", None), patch.object(browser.os, "geteuid", lambda: 0, create=True):
            assert browser._should_disable_sandbox() is True

    def test_explicit_opt_in_and_opt_out_win(self):
        from pyutilz.web import browser

        with patch.object(browser, "no_sandbox", True), patch.object(browser.os, "geteuid", lambda: 1000, create=True):
            assert browser._should_disable_sandbox() is True
        with patch.object(browser, "no_sandbox", False), patch.object(browser.os, "geteuid", lambda: 0, create=True):
            assert browser._should_disable_sandbox() is False

    def test_source_has_no_unconditional_no_sandbox_argument(self):
        """The flag must be reachable only through the guard, not from a second unguarded call."""
        from pyutilz.web import browser

        source = Path(browser.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        unguarded = [
            node for node in ast.walk(tree) if isinstance(node, ast.Call) and any(isinstance(a, ast.Constant) and a.value == "--no-sandbox" for a in node.args)
        ]
        assert len(unguarded) == 1, "exactly one --no-sandbox call site, inside the _should_disable_sandbox() guard"


# ---------------------------------------------------------------------------
# F02 -- proxy credentials are escaped before reaching the extension's JS
# ---------------------------------------------------------------------------


class TestProxyCredentialEscaping:
    """Drives the extension's JS builder directly -- no browser, no chromedriver, no selenium import."""

    def _js(self, password="p", host="proxy.example", port="8080", user="u"):
        from pyutilz.web.browser import _build_proxy_extension_js

        return _build_proxy_extension_js({"PROXY_HOST": host, "PROXY_PORT": port, "PROXY_USER": user, "PROXY_PASS": password})

    def test_quote_in_password_does_not_break_the_script(self):
        password = 'p"a' + chr(92) + "ss"
        script = self._js(password=password)
        # json.dumps emits an escaped literal, so the raw quote never appears unescaped.
        assert json.dumps(password) in script
        assert 'password: "p"a' not in script

    def test_injection_payload_stays_inside_the_string_literal(self):
        payload = 'x"};alert(1);var y={"a":"'
        script = self._js(password=payload)
        assert json.dumps(payload) in script
        assert "alert(1)" not in script.replace(json.dumps(payload), "")

    def test_non_numeric_port_fails_loudly(self):
        with pytest.raises(ValueError):
            self._js(port="80; drop")

    def test_numeric_port_is_emitted_unquoted(self):
        assert "port: 8080" in self._js()


# ---------------------------------------------------------------------------
# F03 -- credential headers are dropped on a cross-host redirect
# ---------------------------------------------------------------------------


class TestRedirectHeaderStripping:
    def _redirect(self, from_url, to_url):
        import urllib.request

        from pyutilz.web.url_guard import _CheckedRedirectHandler

        req = urllib.request.Request(from_url, headers={"Authorization": "Bearer secret-token", "Cookie": "sid=1", "Accept": "application/json"})
        handler = _CheckedRedirectHandler()
        return handler.redirect_request(req, None, 302, "Found", {}, to_url)

    def test_cross_host_hop_drops_authorization_and_cookie(self):
        new_req = self._redirect("https://api.example.com/v1", "https://attacker.example/steal")
        names = {k.lower() for k in new_req.headers}
        assert "authorization" not in names
        assert "cookie" not in names
        assert "accept" in names, "non-credential headers must survive"

    def test_same_host_hop_keeps_them(self):
        new_req = self._redirect("https://api.example.com/v1", "https://api.example.com/v2")
        names = {k.lower() for k in new_req.headers}
        assert "authorization" in names
        assert "cookie" in names


# ---------------------------------------------------------------------------
# F04 -- logged() redacts parameters and exception text
# ---------------------------------------------------------------------------


class TestLoggedRedaction:
    def test_credential_kwargs_and_error_text_are_redacted(self):
        from pyutilz.dev.logginglib import logged

        captured = {}

        @logged()
        def do_work(api_key=None, dsn=None, rows=None, results_log=None):
            captured["log"] = results_log
            raise RuntimeError("could not connect to postgresql://admin:s3cret@db.example.com/prod")

        with pytest.raises(RuntimeError):
            do_work(api_key="sk-live-abcdefghijklmnop", dsn="postgresql://u:p@h/db", rows=17)

        log = captured["log"]
        params = log["parameters"]
        assert "sk-live-abcdefghijklmnop" not in str(params)
        assert "p@h" not in str(params)
        assert params["rows"] == 17, "non-credential numbers stay queryable"
        assert "s3cret" not in log["results"]["error"]
        assert "<dsn-redacted>" in log["results"]["error"]


# ---------------------------------------------------------------------------
# F05 -- redact_secrets covers quoted/JSON keys
# ---------------------------------------------------------------------------


class TestRedactSecretsQuotedKeys:
    @pytest.mark.parametrize(
        "text, leaked",
        [
            ('{"password": "s3cret", "api_key": "sk-abc"}', ["s3cret", "sk-abc"]),
            ('{"error":"bad credentials","api_key":"sk-live-xyz"}', ["sk-live-xyz"]),
            ("AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI", ["wJalrXUtnFEMI"]),
            ("GITHUB_TOKEN_FOR_CI=abc123def", ["abc123def"]),
            ("'password': 'p1'", ["p1"]),
        ],
    )
    def test_secret_never_survives(self, text, leaked):
        from pyutilz.text.secrets_scrub import redact_secrets

        out = redact_secrets(text)
        for secret in leaked:
            assert secret not in out, out

    def test_ordinary_prose_is_untouched(self):
        from pyutilz.text.secrets_scrub import redact_secrets

        assert redact_secrets("Unexpected token in response, and oauth flows continue") == "Unexpected token in response, and oauth flows continue"


# ---------------------------------------------------------------------------
# F06 -- _safe_parse survives a RecursionError
# ---------------------------------------------------------------------------


class TestSafeParseRecursionError:
    def test_deeply_nested_file_is_skipped_not_fatal(self, tmp_path):
        from pyutilz.dev.code_audit._base import _safe_parse, clear_parse_cache

        deep = tmp_path / "generated.py"
        deep.write_text("x = " + " + ".join(["a"] * 20000) + "\n", encoding="utf-8")
        clear_parse_cache()
        assert _safe_parse(deep) is None

    def test_a_valid_file_still_parses(self, tmp_path):
        from pyutilz.dev.code_audit._base import _safe_parse, clear_parse_cache

        ok = tmp_path / "ok.py"
        ok.write_text("x = 1\n", encoding="utf-8")
        clear_parse_cache()
        tree = _safe_parse(ok)
        # "not None" alone would also pass for an empty or unrelated tree, i.e. for a recursion
        # guard that had started degrading ordinary files -- assert the actual parse result.
        assert isinstance(tree, ast.Module)
        assert [type(node).__name__ for node in tree.body] == ["Assign"]
        assert tree.body[0].targets[0].id == "x"
        assert tree.body[0].value.value == 1


# ---------------------------------------------------------------------------
# F07 -- CachedHttpClient caps the response size
# ---------------------------------------------------------------------------


class _CappedResponse(io.BytesIO):
    def __init__(self, body, headers=None):
        super().__init__(body)
        self.headers = headers or {}


def _fake_open(resp):
    from contextlib import contextmanager

    @contextmanager
    def _open(*_args, **_kwargs):
        yield resp

    return _open


class TestResponseSizeCap:
    def test_oversized_body_is_rejected(self, tmp_path):
        from pyutilz.web.cached_client import CachedHttpClient

        client = CachedHttpClient(cache_dir=tmp_path, max_bytes=1024)
        body = b'{"a": "' + b"x" * 5000 + b'"}'
        with patch("pyutilz.web.cached_client.urlopen_checked", _fake_open(_CappedResponse(body))):
            assert client.get_json("https://example.org/big", "tag") is None
        # The client's documented negative caching still writes an entry; what must never happen is
        # the over-cap body itself landing in it.
        for entry in tmp_path.rglob("*.json"):
            assert b"xxxx" not in entry.read_bytes()

    def test_declared_content_length_is_refused_up_front(self, tmp_path):
        from pyutilz.web.cached_client import CachedHttpClient

        client = CachedHttpClient(cache_dir=tmp_path, max_bytes=1024)
        resp = _CappedResponse(b"{}", headers={"Content-Length": "999999999"})
        with patch("pyutilz.web.cached_client.urlopen_checked", _fake_open(resp)):
            assert client.get_json("https://example.org/big", "tag") is None

    def test_normal_body_still_works(self, tmp_path):
        from pyutilz.web.cached_client import CachedHttpClient

        client = CachedHttpClient(cache_dir=tmp_path, max_bytes=1024)
        with patch("pyutilz.web.cached_client.urlopen_checked", _fake_open(_CappedResponse(b'{"a": 1}'))):
            assert client.get_json("https://example.org/small", "tag") == {"a": 1}


# ---------------------------------------------------------------------------
# F08 -- probe binaries are resolved to an absolute path
# ---------------------------------------------------------------------------


class TestBinaryResolution:
    def test_resolve_binary_returns_an_absolute_path(self):
        from pyutilz.system.system.probing import _resolve_binary

        resolved = _resolve_binary(Path(sys.executable).stem)
        assert Path(resolved).is_absolute()

    def test_missing_binary_raises_oserror(self):
        from pyutilz.system.system.probing import _BinaryNotFoundError, _resolve_binary

        with pytest.raises(_BinaryNotFoundError):
            _resolve_binary("pyutilz-no-such-binary-xyz")
        assert issubclass(_BinaryNotFoundError, OSError), "call sites already tolerate OSError"

    @pytest.mark.parametrize(
        "module_name, bare_names",
        [
            ("pyutilz.system.system.probing", ("lscpu", "vm_stat", "pmset", "powercfg", "nvidia-smi", "sudo", "dmidecode")),
            ("pyutilz.system.system.sysinfo", ("wmic", "cat", "getprop", "ioreg", "grep", "nvcc")),
            ("pyutilz.system.system.fsutils", ("pylspci",)),
        ],
    )
    def test_no_bare_executable_name_reaches_subprocess(self, module_name, bare_names):
        """A bare name lets Windows' CreateProcess search the CURRENT DIRECTORY before PATH."""
        import importlib

        module = importlib.import_module(module_name)
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in {"run", "check_output", "check_call", "call", "Popen"}:
                continue
            if not node.args:
                continue
            argv = node.args[0]
            elements = argv.elts if isinstance(argv, ast.List) else [argv]
            for element in elements[:1]:
                assert not (
                    isinstance(element, ast.Constant) and any(name in str(element.value) for name in bare_names)
                ), f"{module_name}:{node.lineno} spawns a bare executable name; use _resolve_binary()"


# ---------------------------------------------------------------------------
# F11 -- .env is never picked up from the current working directory
# ---------------------------------------------------------------------------


class TestLLMEnvFile:
    def test_cwd_dotenv_is_ignored(self, tmp_path, monkeypatch):
        pytest.importorskip("pydantic_settings")
        from pyutilz.llm import config

        (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-planted-key\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("PYUTILZ_LLM_ENV_FILE", raising=False)
        config.clear_llm_settings_cache()
        assert config.get_llm_settings().openai_api_key is None
        config.clear_llm_settings_cache()

    def test_explicit_env_file_is_honoured(self, tmp_path, monkeypatch):
        pytest.importorskip("pydantic_settings")
        from pyutilz.llm import config

        env_path = tmp_path / "chosen.env"
        env_path.write_text("OPENAI_API_KEY=sk-chosen-key\n", encoding="utf-8")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config.clear_llm_settings_cache()
        settings = config.get_llm_settings(env_file=str(env_path))
        assert settings.openai_api_key is not None
        assert settings.openai_api_key.get_secret_value() == "sk-chosen-key"
        config.clear_llm_settings_cache()


# ---------------------------------------------------------------------------
# F12 -- unserialize bounds zlib expansion
# ---------------------------------------------------------------------------


class TestDecompressionCap:
    def test_zlib_bomb_is_refused(self):
        from pyutilz.core.serialization import unserialize

        bomb = zlib.compress(b"\0" * (8 * 1024 * 1024), 9)
        assert len(bomb) < 100 * 1024, "the point of the test is a small input with a large expansion"
        with pytest.raises(ValueError, match="max_output_bytes"):
            unserialize(bomb, max_output_bytes=1024)

    def test_ordinary_payload_round_trips(self, tmp_path):
        from pyutilz.core.serialization import serialize, unserialize

        fname = str(tmp_path / "obj.dat")
        payload = {"a": list(range(1000))}
        serialize(payload, fname)
        assert unserialize(fname) == payload

    def test_opt_out_restores_unbounded_read(self):
        from pyutilz.core.serialization import _decompress_capped

        raw = b"y" * (1024 * 1024)
        assert _decompress_capped(zlib.compress(raw), None) == raw


# ---------------------------------------------------------------------------
# F13 -- the CLI backend carries the SDK's tool-use tripwire
# ---------------------------------------------------------------------------


class _FakeStream(io.StringIO):
    def close(self):  # the provider closes stdin; keep the buffer readable afterwards
        pass


class _FakeProc:
    """A stand-in for the ``claude`` CLI emitting a scripted stream-json transcript."""

    def __init__(self, events):
        self.args = ["claude"]
        self.stdin = _FakeStream()
        self.stdout = io.StringIO("".join(json.dumps(e) + "\n" for e in events))
        self.stderr = io.StringIO("")
        self.returncode = 0

    def kill(self):
        pass

    def wait(self, timeout=None):
        return 0


class TestCliToolUseTripwire:
    def _run(self, events):
        import asyncio

        from pyutilz.llm import claude_code_provider

        provider = claude_code_provider.ClaudeCodeProvider.__new__(claude_code_provider.ClaudeCodeProvider)
        provider.model = "test-model"
        provider.timeout = 5
        # Set _max_concurrent and let base.LazySemaphore build the Semaphore on first access,
        # inside the running loop. Constructing asyncio.Semaphore(1) here instead defeated exactly
        # the lazy-bind the descriptor exists for: on 3.8/3.9 Semaphore.__init__ eagerly calls
        # get_event_loop(), which raises "There is no current event loop" in a sync test body.
        provider._max_concurrent = 1
        provider._claude_path = "claude"
        provider._reset_per_call_state = lambda: None
        with patch.object(claude_code_provider.subprocess, "Popen", lambda *a, **k: _FakeProc(events)):
            return asyncio.run(provider._generate_cli("hi"))

    def test_tool_use_block_raises_on_the_cli_path(self):
        """Both backends must fail identically when a tool-use block appears."""
        from pyutilz.llm.exceptions import ClaudeCodeToolUseError

        events = [
            {"type": "system", "subtype": "init"},
            {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash"}]}},
            {"type": "result", "subtype": "success", "result": "done"},
        ]
        with pytest.raises(ClaudeCodeToolUseError):
            self._run(events)

    def test_ordinary_text_assistant_event_is_fine(self):
        events = [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "hello"}]}},
            {"type": "result", "subtype": "success", "result": "hello"},
        ]
        assert self._run(events) == "hello"

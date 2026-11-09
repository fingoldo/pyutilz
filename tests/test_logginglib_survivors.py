"""External-IP resolution and log filename pins for dev.logginglib (mutation survivors, first group)."""

from __future__ import annotations

import os
from pathlib import Path

import pyutilz.web as web

from pyutilz.dev import logginglib as ll


class _Log:
    def __init__(self):
        self.warning_calls: list = []

    def warning(self, *args):
        self.warning_calls.append(args)


def _reset(monkeypatch):
    monkeypatch.setattr(ll, "EXTERNAL_IP", None)
    monkeypatch.setattr(ll, "_EXTERNAL_IP_RESOLVED", False)


class TestNodeExternalIp:
    def test_the_resolved_address_is_returned(self, monkeypatch):
        _reset(monkeypatch)
        monkeypatch.setattr(web, "get_external_ip", lambda: "203.0.113.7")
        assert ll.get_node_external_ip() == "203.0.113.7"

    def test_resolution_happens_at_most_once_per_process(self, monkeypatch):
        _reset(monkeypatch)
        calls: list = []

        def fake():
            calls.append(1)
            return "203.0.113.7"

        monkeypatch.setattr(web, "get_external_ip", fake)
        ll.get_node_external_ip()
        ll.get_node_external_ip()
        assert len(calls) == 1

    def test_a_failed_lookup_gives_none_and_warns(self, monkeypatch):
        _reset(monkeypatch)
        log = _Log()
        monkeypatch.setattr(ll, "logger", log)

        def boom():
            raise OSError("no route")

        monkeypatch.setattr(web, "get_external_ip", boom)
        assert ll.get_node_external_ip() is None
        assert log.warning_calls == [("Could not resolve this node's external IP for logging: %s", log.warning_calls[0][1])]
        assert "no route" in str(log.warning_calls[0][1])

    def test_a_failed_lookup_is_not_retried(self, monkeypatch):
        _reset(monkeypatch)
        monkeypatch.setattr(ll, "logger", _Log())
        calls: list = []

        def boom():
            calls.append(1)
            raise OSError("down")

        monkeypatch.setattr(web, "get_external_ip", boom)
        ll.get_node_external_ip()
        ll.get_node_external_ip()
        assert len(calls) == 1


class TestLogFilename:
    def test_the_trailing_py_becomes_log(self):
        assert ll._log_filename("run.py") == "run.log"

    def test_only_the_trailing_extension_is_replaced(self):
        assert ll._log_filename("run.pytest_smoke.py") == "run.pytest_smoke.log"

    def test_a_name_without_py_gets_log_appended(self):
        assert ll._log_filename("data") == "data.log"

    def test_a_py_prefix_that_is_not_an_extension_is_kept(self):
        assert ll._log_filename("a.pyx") == "a.pyx.log"


class TestInitLoggingDefaults:
    def test_the_documented_defaults_are_pinned(self):
        import inspect

        params = inspect.signature(ll.init_logging).parameters
        assert params["default_caller_name"].default == "app.py"
        assert params["log_to_file"].default is True
        assert params["maxBytes"].default == 5_000_000
        assert "%(asctime)s - %(levelname)s - %(processName)s[%(process)d]-%(threadName)s[%(thread)d]" in params["format"].default


class TestFreshProcessState:
    def test_the_external_ip_is_not_resolved_at_import(self):
        import subprocess
        import sys

        code = "import pyutilz.dev.logginglib as ll; " "print(ll._EXTERNAL_IP_RESOLVED, ll.EXTERNAL_IP)"
        src_root = str(Path(ll.__file__).resolve().parents[2])
        env = dict(os.environ, PYTHONPATH=src_root)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=110, check=True, env=env).stdout.strip()
        assert out == "False None"

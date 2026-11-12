"""Shared fixtures and fakes for the OpenAI-compatible provider tests (split out of test_openai_compat.py)."""

from __future__ import annotations
from pyutilz.dev import logginglib as ll


class _Log:
    def __init__(self):
        self.warning_calls: list = []

    def warning(self, *args):
        self.warning_calls.append(args)


def _reset(monkeypatch):
    monkeypatch.setattr(ll, "EXTERNAL_IP", None)
    monkeypatch.setattr(ll, "_EXTERNAL_IP_RESOLVED", False)


def _parent_for_log(alpha, beta=2, **kwargs):
    return ll.initialize_function_log()


def _parent_explicit(alpha, beta=2, **kwargs):
    return ll.initialize_function_log(explicit_only=True)


def _parent_kw_for_error(alpha, **kwargs):
    return ll.initialize_function_log()


def _parent_p_for_error(alpha):
    return ll.initialize_function_log()


class _ExcLog:
    def __init__(self):
        self.exceptions: list = []

    def exception(self, e):
        self.exceptions.append(e)


class _InfoLog:
    def __init__(self):
        self.info_calls: list = []

    def info(self, *args):
        self.info_calls.append(args)


def _fresh_log(function="f"):
    from datetime import datetime, timezone

    return {
        "module": "m.py",
        "function": function,
        "parameters": {"a": 1},
        "results": {"timing": {"started_at": datetime.now(timezone.utc)}},
    }


class _FakeRedis:
    def __init__(self, fail: bool = False):
        self.pushed: list = []
        self.trimmed: list = []
        self.fail = fail

    def lpush(self, key, msg):
        if self.fail:
            raise ConnectionError("redis down")
        self.pushed.append((key, msg))

    def ltrim(self, key, start, stop):
        self.trimmed.append((key, start, stop))


def _record(msg="hello"):
    import logging

    return logging.LogRecord("t", logging.INFO, __file__, 1, msg, None, None)


class _Clock:
    def __init__(self, now: float = 0.0):
        self.now = now

    def monotonic(self) -> float:
        return self.now


class _ThrottleRecorder:
    def __init__(self):
        self.calls: list = []

    def log(self, level, msg, *args, exc_info=False):
        self.calls.append((level, msg, args, exc_info))

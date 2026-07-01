"""Regression tests for pyutilz.database (db.py, deltalakes.py, redislib.py).

Each test targets a specific fix and is written to fail on the pre-fix source.
"""

import os
import tempfile

import pytest


# ---------------------------------------------------------------------------
# db.validate_sql_identifier  (regex hoisted to module level; behavior preserved)
# ---------------------------------------------------------------------------

from pyutilz.database.db import validate_sql_identifier, construct_templates_and_values


def test_validate_sql_identifier_valid():
    assert validate_sql_identifier("my_table") == "my_table"
    assert validate_sql_identifier("_col1") == "_col1"


def test_validate_sql_identifier_rejects_injection():
    for bad in ["1abc", "a-b", "drop table x", "a;b", "a b", ""]:
        with pytest.raises(ValueError):
            validate_sql_identifier(bad)
    with pytest.raises(ValueError):
        validate_sql_identifier(123)  # non-string


def test_construct_templates_jsonize_sorts_keys():
    """orjson with OPT_SORT_KEYS -> stable serialization for hashing/dedup."""
    values, templates = construct_templates_and_values(
        mode="insert",
        fields=["payload"],
        replace_values={"payload": {"b": 1, "a": 2}},
        source={},
        jsonize=True,
    )
    assert values == ['{"a":2,"b":1}']
    assert templates == ["%s"]


# ---------------------------------------------------------------------------
# deltalakes.safe_delta_write  (#6: no hardcoded /tmp on Windows; non-local re-raises)
# ---------------------------------------------------------------------------

from pyutilz.database import deltalakes


def test_safe_delta_write_uses_platform_tempdir(monkeypatch, tmp_path):
    """Lock file must live under tempfile.gettempdir(), not a hardcoded '/tmp'.

    Pre-fix this raised on Windows because '/tmp' does not exist.
    """
    captured = {}

    class FakeLock:
        def __init__(self, path):
            captured["path"] = path

        def acquire(self, timeout=None):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(deltalakes, "FileLock", FakeLock)

    result = deltalakes.safe_delta_write(str(tmp_path / "mytable"), lambda: "ok")
    assert result == "ok"
    assert captured["path"].startswith(tempfile.gettempdir())
    assert "/tmp" not in captured["path"] or tempfile.gettempdir() == "/tmp"


def test_safe_delta_write_nonlocal_reraises():
    """#6: the non-local branch must re-raise (previously it swallowed the exception)."""

    def boom():
        raise ValueError("delta failed")

    with pytest.raises(ValueError, match="delta failed"):
        deltalakes.safe_delta_write("s3://bucket/table", boom)


# ---------------------------------------------------------------------------
# redislib.rexecute  (#7: no busy-loop on permanent errors; rc-None guard)
# ---------------------------------------------------------------------------

import pyutilz.database.redislib as redislib


def test_rexecute_raises_when_not_connected(monkeypatch):
    monkeypatch.setattr(redislib, "rc", None)
    with pytest.raises(RuntimeError, match="not established"):
        redislib.rexecute("get", "key")


def test_rexecute_permanent_error_does_not_busy_loop(monkeypatch):
    """A non-ConnectionError must propagate immediately, not spin forever."""

    call_count = {"n": 0}

    class FakeRC:
        def get(self, *a, **k):
            call_count["n"] += 1
            raise KeyError("permanent")

    monkeypatch.setattr(redislib, "rc", FakeRC())
    with pytest.raises(KeyError):
        redislib.rexecute("get", "somekey")
    assert call_count["n"] == 1  # not retried


def test_rexecute_transient_then_success(monkeypatch):
    from redis.exceptions import ConnectionError as RedisConnError

    state = {"n": 0}

    class FakeRC:
        def get(self, *a, **k):
            state["n"] += 1
            if state["n"] < 3:
                raise RedisConnError("temporary")
            return "value"

    monkeypatch.setattr(redislib, "rc", FakeRC())
    # sleep uses random backoff; keep it fast by monkeypatching sleep.
    monkeypatch.setattr(redislib, "sleep", lambda *_: None)
    assert redislib.rexecute("get", "k") == "value"
    assert state["n"] == 3


def test_rexecute_unknown_method_raises(monkeypatch):
    class FakeRC:
        pass

    monkeypatch.setattr(redislib, "rc", FakeRC())
    with pytest.raises(AttributeError):
        redislib.rexecute("no_such_method")


def test_rclose_resets_connection(monkeypatch):
    closed = {"v": False}

    class FakeRC:
        def close(self):
            closed["v"] = True

    monkeypatch.setattr(redislib, "rc", FakeRC())
    redislib.rclose()
    assert closed["v"] is True
    assert redislib.rc is None

"""Behaviour of ``pyutilz.database.redislib``: connecting stores the global client and closes the one it replaces,
the database argument must be an index, ``rclose`` drops the client, and ``rexecute`` retries only transient
connection errors (bounded by ``max_retries`` when given) while authentication and unknown-command failures are
raised at once. A fake client stands in for redis-py, so no server is needed."""

from __future__ import annotations

import pytest

from pyutilz.database import redislib
from pyutilz.database.exceptions import DatabaseConnectionError

_FAKE_PASSWORD = "pwd"  # pragma: allowlist secret


class _AuthError(Exception):
    pass


class _ConnError(Exception):
    pass


class _FakeClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        self.fail_times = 0
        self.calls = 0

    def close(self):
        self.closed = True

    def get(self, key):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise _ConnError("down")
        return f"value:{key}"


class _FakeRedisModule:
    Redis = _FakeClient


@pytest.fixture
def fake_redis(monkeypatch):
    monkeypatch.setattr(redislib, "redis", _FakeRedisModule)
    monkeypatch.setattr(redislib, "RedisAuthenticationError", _AuthError)
    monkeypatch.setattr(redislib, "RedisConnectionError", _ConnError)
    monkeypatch.setattr(redislib, "sleep", lambda _seconds: None)
    monkeypatch.setattr(redislib, "rc", None)
    yield
    monkeypatch.setattr(redislib, "rc", None)


class TestConnect:
    def test_the_new_client_is_stored_and_returned(self, fake_redis):
        client = redislib.rconnect("host", 6379, 2, _FAKE_PASSWORD)
        assert redislib.rc is client
        assert client.kwargs == {"host": "host", "port": 6379, "db": 2, "password": _FAKE_PASSWORD, "decode_responses": True}

    def test_a_name_instead_of_an_index_is_refused_before_any_state_changes(self, fake_redis):
        previous = redislib.rconnect("host", 6379, 0, _FAKE_PASSWORD)
        with pytest.raises(ValueError, match="database INDEX"):
            redislib.rconnect("host", 6379, "sessions", _FAKE_PASSWORD)
        assert redislib.rc is previous
        assert previous.closed is False

    def test_a_numeric_string_index_is_accepted(self, fake_redis):
        client = redislib.rconnect("host", 6379, "3", _FAKE_PASSWORD)
        assert client.kwargs["db"] == 3

    def test_reconnecting_closes_the_previous_client(self, fake_redis):
        first = redislib.rconnect("host", 6379, 0, _FAKE_PASSWORD)
        second = redislib.rconnect("host", 6379, 1, _FAKE_PASSWORD)
        assert first.closed is True
        assert second.closed is False


class TestClose:
    def test_rclose_closes_and_forgets_the_client(self, fake_redis):
        client = redislib.rconnect("host", 6379, 0, _FAKE_PASSWORD)
        redislib.rclose()
        assert client.closed is True
        assert redislib.rc is None

    def test_rclose_without_a_connection_does_nothing(self, fake_redis):
        redislib.rclose()
        assert redislib.rc is None


class TestExecute:
    def test_without_a_connection_the_caller_is_told_to_connect_first(self, fake_redis):
        with pytest.raises(DatabaseConnectionError, match="rconnect"):
            redislib.rexecute("get", "key")

    def test_the_command_result_is_returned(self, fake_redis):
        redislib.rconnect("host", 6379, 0, _FAKE_PASSWORD)
        assert redislib.rexecute("get", "key") == "value:key"

    def test_a_transient_failure_is_retried_until_it_succeeds(self, fake_redis):
        client = redislib.rconnect("host", 6379, 0, _FAKE_PASSWORD)
        client.fail_times = 2
        assert redislib.rexecute("get", "key") == "value:key"
        assert client.calls == 3

    def test_max_retries_bounds_the_transient_retries(self, fake_redis):
        client = redislib.rconnect("host", 6379, 0, _FAKE_PASSWORD)
        client.fail_times = 10
        with pytest.raises(_ConnError):
            redislib.rexecute("get", "key", max_retries=3)
        assert client.calls == 3

    def test_an_authentication_failure_is_not_retried(self, fake_redis, monkeypatch):
        client = redislib.rconnect("host", 6379, 0, _FAKE_PASSWORD)

        def refuse(*_args, **_kwargs):
            client.calls += 1
            raise _AuthError("wrong password")

        monkeypatch.setattr(client, "get", refuse)
        with pytest.raises(_AuthError):
            redislib.rexecute("get", "key")
        assert client.calls == 1

    def test_an_unknown_command_name_is_raised_at_once(self, fake_redis):
        redislib.rconnect("host", 6379, 0, _FAKE_PASSWORD)
        with pytest.raises(AttributeError):
            redislib.rexecute("no_such_command")

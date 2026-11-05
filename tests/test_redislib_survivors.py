"""Logging, backoff and forwarding pins for database.redislib (mutation survivors)."""

from __future__ import annotations

import logging

import pytest

from pyutilz.database import redislib as rl


class _FakeConnectionError(Exception):
    pass


class _FakeAuthenticationError(_FakeConnectionError):
    """Mirrors redis-py, where AuthenticationError is a subclass of ConnectionError."""


class _Client:
    def __init__(self, behaviour=None):
        self.behaviour = behaviour or []
        self.calls = []
        self.closed = 0

    def ping(self, **kwargs):
        self.calls.append(kwargs)
        if self.behaviour:
            outcome = self.behaviour.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return "PONG"

    def close(self):
        self.closed += 1


@pytest.fixture
def wired(monkeypatch):
    sleeps = []
    monkeypatch.setattr(rl, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(rl, "random", lambda: 0.5)
    monkeypatch.setattr(rl, "RedisConnectionError", _FakeConnectionError)
    monkeypatch.setattr(rl, "RedisAuthenticationError", _FakeAuthenticationError)
    monkeypatch.setattr(rl, "rc", None)
    return sleeps


def _install(monkeypatch, client):
    monkeypatch.setattr(rl, "rc", client)
    return client


class TestConnect:
    def test_the_connection_defaults_to_decoded_responses(self, monkeypatch, wired):
        captured = {}

        class _Redis:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def close(self):
                pass

        monkeypatch.setattr(rl, "redis", type("M", (), {"Redis": _Redis}))
        rl.rconnect("h", 6379, "0", "pw")
        assert captured["decode_responses"] is True

    def test_a_non_index_database_name_names_the_index_rule(self, wired):
        with pytest.raises(ValueError, match="must be a Redis database INDEX"):
            rl.rconnect("h", 6379, "sessions", "pw")

    def test_a_failing_close_of_the_previous_connection_is_logged(self, monkeypatch, wired, caplog):
        caplog.set_level(logging.ERROR)
        old = _Client()
        old.close = lambda: (_ for _ in ()).throw(RuntimeError("close broke"))
        monkeypatch.setattr(rl, "rc", old)

        class _Redis:
            def __init__(self, **kwargs):
                pass

        monkeypatch.setattr(rl, "redis", type("M", (), {"Redis": _Redis}))
        rl.rconnect("h", 6379, "0", "pw")
        assert "close broke" in caplog.text


class TestClose:
    def test_a_failing_close_is_logged_and_the_global_is_cleared(self, monkeypatch, wired, caplog):
        caplog.set_level(logging.ERROR)
        bad = _Client()
        bad.close = lambda: (_ for _ in ()).throw(RuntimeError("close broke"))
        monkeypatch.setattr(rl, "rc", bad)
        rl.rclose()
        assert "close broke" in caplog.text
        assert rl.rc is None


class TestRexecute:
    def test_no_connection_is_refused(self, wired):
        with pytest.raises(Exception, match="Redis connection is not established"):
            rl.rexecute("ping")

    def test_keyword_arguments_are_forwarded(self, monkeypatch, wired):
        client = _install(monkeypatch, _Client())
        rl.rexecute("ping", x=1)
        assert client.calls == [{"x": 1}]

    def test_an_unknown_method_is_logged_and_raised(self, monkeypatch, wired, caplog):
        caplog.set_level(logging.ERROR)
        _install(monkeypatch, _Client())
        with pytest.raises(AttributeError):
            rl.rexecute("no_such_method")
        assert "no_such_method" in caplog.text

    def test_an_authentication_failure_is_logged_as_permanent_and_not_retried(self, monkeypatch, wired, caplog):
        caplog.set_level(logging.ERROR)
        client = _install(monkeypatch, _Client(behaviour=[_FakeAuthenticationError("bad password")]))
        with pytest.raises(_FakeAuthenticationError):
            rl.rexecute("ping")
        assert "Redis authentication failed -- not retrying (permanent error)" in caplog.text
        assert len(client.calls) == 1
        assert wired == []

    def test_a_generic_failure_is_logged_and_propagated(self, monkeypatch, wired, caplog):
        caplog.set_level(logging.ERROR)
        _install(monkeypatch, _Client(behaviour=[KeyError("boom")]))
        with pytest.raises(KeyError):
            rl.rexecute("ping")
        assert "boom" in caplog.text

    def test_a_transient_failure_is_retried_after_a_jittered_pause(self, monkeypatch, wired):
        client = _install(monkeypatch, _Client(behaviour=[_FakeConnectionError("blip"), "OK"]))
        assert rl.rexecute("ping") == "OK"
        assert len(client.calls) == 2
        assert wired == [pytest.approx(0.5)]

    def test_only_the_first_transient_failure_logs_a_traceback(self, monkeypatch, wired, caplog):
        caplog.set_level(logging.DEBUG)
        _install(monkeypatch, _Client(behaviour=[_FakeConnectionError("one"), _FakeConnectionError("two"), "OK"]))
        rl.rexecute("ping")
        errors = [r for r in caplog.records if r.levelno == logging.ERROR and r.exc_info]
        debugs = [r for r in caplog.records if "still failing (attempt 2)" in r.getMessage()]
        assert len(errors) == 1
        assert len(debugs) == 1

    def test_giving_up_after_the_retry_budget_is_logged(self, monkeypatch, wired, caplog):
        caplog.set_level(logging.ERROR)
        _install(monkeypatch, _Client(behaviour=[_FakeConnectionError("x"), _FakeConnectionError("y")]))
        with pytest.raises(_FakeConnectionError):
            rl.rexecute("ping", max_retries=2)
        assert "rexecute: giving up after 2 attempts" in caplog.text

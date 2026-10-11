"""Behaviour of the lazily-built tenacity policy in ``pyutilz.llm._retry``."""

import pytest
from tenacity import stop_after_attempt, wait_fixed

from pyutilz.llm._retry import _LazyRetry, lazy_retry


def _policy(attempts: int) -> dict:
    return dict(stop=stop_after_attempt(attempts), wait=wait_fixed(0), reraise=True)


def test_policy_is_built_on_first_call_and_reused():
    built = []

    def factory() -> dict:
        built.append(1)
        return _policy(3)

    class Client:
        @lazy_retry(factory)
        def call(self, x: int) -> int:
            return x * 2

    assert built == []  # neither the class body nor decoration builds the policy
    client = Client()
    assert client.call(2) == 4
    assert client.call(3) == 6
    assert built == [1]


def test_retries_until_the_call_succeeds():
    attempts = []

    @lazy_retry(lambda: _policy(5))
    def flaky(x: int) -> str:
        attempts.append(x)
        if len(attempts) < 3:
            raise ValueError("transient")
        return "ok"

    assert flaky(1) == "ok"
    assert len(attempts) == 3


def test_gives_up_after_the_policy_attempt_cap():
    attempts = []

    @lazy_retry(lambda: _policy(2))
    def always_fails() -> None:
        attempts.append(1)
        raise ValueError("permanent")

    with pytest.raises(ValueError):
        always_fails()
    assert len(attempts) == 2


def test_class_access_returns_the_descriptor_and_instance_access_binds():
    class Client:
        @lazy_retry(lambda: _policy(1))
        def method(self) -> str:
            return "bound"

    assert isinstance(Client.__dict__["method"], _LazyRetry)
    assert Client.method is Client.__dict__["method"]
    assert Client().method() == "bound"


def test_retry_property_exposes_the_tenacity_retrying_object():
    class Client:
        @lazy_retry(lambda: _policy(7))
        def method(self) -> int:
            return 1

    assert Client().method.retry.stop.max_attempt_number == 7


def test_cached_attempt_cap_picks_up_a_changed_setting_after_cache_clear(monkeypatch):
    from tenacity import stop_never

    from pyutilz.llm import _retry

    try:
        _retry.retry_wait.cache_clear()
        monkeypatch.setattr(_retry, "MAX_RETRY_ATTEMPTS", 4)
        _retry._attempt_cap.cache_clear()
        assert _retry._attempt_cap().max_attempt_number == 4

        monkeypatch.setattr(_retry, "MAX_RETRY_ATTEMPTS", 0)
        _retry._attempt_cap.cache_clear()
        assert _retry._attempt_cap() is stop_never
    finally:
        _retry.retry_wait.cache_clear()
        _retry._attempt_cap.cache_clear()

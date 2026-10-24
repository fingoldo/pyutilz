from __future__ import annotations
from types import SimpleNamespace
import pytest


class ReadTimeout(Exception):  # noqa: N818 -- named like httpx.ReadTimeout on purpose
    """Named like httpx's, which is how the stop policy recognises it."""


def _state(exc: BaseException, elapsed: float, attempt: int) -> SimpleNamespace:
    return SimpleNamespace(outcome=SimpleNamespace(exception=lambda: exc), seconds_since_start=elapsed, attempt_number=attempt)


class TestCallDeadline:
    def test_deadline_stops_retries(self, monkeypatch):
        from pyutilz.llm import _retry

        monkeypatch.setattr(_retry, "MAX_CALL_SECONDS", 100.0)
        assert _retry._stop_policy(_state(ValueError("x"), 99.0, 2)) is False
        assert _retry._stop_policy(_state(ValueError("x"), 100.0, 2)) is True

    def test_zero_disables_the_deadline(self, monkeypatch):
        from pyutilz.llm import _retry

        monkeypatch.setattr(_retry, "MAX_CALL_SECONDS", 0.0)
        assert _retry._stop_policy(_state(ValueError("x"), 10_000_000.0, 2)) is False

    def test_default_deadline_is_bounded_and_fits_two_long_attempts(self):
        from pyutilz.llm import _retry
        from pyutilz.llm._timeouts import DerivedTimeoutMixin

        assert _retry._DEFAULT_MAX_CALL_SECONDS >= 2 * DerivedTimeoutMixin._max_derived_timeout_s
        assert _retry._DEFAULT_MAX_CALL_SECONDS <= 4 * 3600

    def test_consecutive_read_timeouts_stop_and_other_errors_reset_the_streak(self, monkeypatch):
        from pyutilz.llm import _retry

        monkeypatch.setattr(_retry, "MAX_CONSECUTIVE_TIMEOUTS", 3)
        monkeypatch.setattr(_retry, "MAX_CALL_SECONDS", 0.0)
        state = _state(ReadTimeout(), 1.0, 1)
        results = []
        for attempt, exc in enumerate([ReadTimeout(), ReadTimeout(), ValueError("429"), ReadTimeout(), ReadTimeout(), ReadTimeout()], 1):
            state.outcome = SimpleNamespace(exception=lambda e=exc: e)
            state.attempt_number = attempt
            results.append(_retry._stop_policy(state))
        assert results == [False, False, False, False, False, True]

    def test_connect_timeout_is_not_a_read_timeout(self):
        from pyutilz.llm._retry import _is_read_timeout

        class ConnectTimeout(Exception):  # noqa: N818 -- named like httpx.ConnectTimeout on purpose
            pass

        assert _is_read_timeout(ReadTimeout()) is True
        assert _is_read_timeout(TimeoutError()) is True
        assert _is_read_timeout(ConnectTimeout()) is False
        assert _is_read_timeout(None) is False

    def test_env_parsing(self, monkeypatch, caplog):
        from pyutilz.llm._retry import _env_number

        monkeypatch.setenv("PYUTILZ_TEST_NUM", "12.5")
        assert _env_number("PYUTILZ_TEST_NUM", 7.0) == 12.5
        monkeypatch.setenv("PYUTILZ_TEST_NUM", "abc")
        assert _env_number("PYUTILZ_TEST_NUM", 7.0) == 7.0
        monkeypatch.setenv("PYUTILZ_TEST_NUM", "-1")
        assert _env_number("PYUTILZ_TEST_NUM", 7.0) == 7.0
        monkeypatch.delenv("PYUTILZ_TEST_NUM")
        assert _env_number("PYUTILZ_TEST_NUM", 7.0) == 7.0

    def test_tenacity_actually_stops_on_the_deadline(self, monkeypatch):
        tenacity = pytest.importorskip("tenacity")
        from pyutilz.llm import _retry

        monkeypatch.setattr(_retry, "MAX_CALL_SECONDS", 0.05)
        attempts = []

        @tenacity.retry(wait=tenacity.wait_fixed(0.03), stop=_retry._STOP, reraise=True)
        def flaky() -> None:
            attempts.append(1)
            raise ValueError("transient")

        with pytest.raises(ValueError):
            flaky()
        assert 1 < len(attempts) < 10

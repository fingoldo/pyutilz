"""A heartbeat that fails transiently is redelivered from a background thread until the process ends.

Found 2026-09-30: a long-lived scraper finished its full scan, the heartbeat send hit
``Failed to resolve 'cronitor.link'`` (a DNS blip), was logged once and forgotten, and Cronitor raised a
"Missed Event" for a healthy job. These tests drive the real retry thread with delays shrunk to milliseconds.
"""

from __future__ import annotations

import logging
import random
import threading
from unittest.mock import MagicMock, patch

import pytest

from pyutilz.system import monitoring
from pyutilz.system.monitoring import job_completed

CRONITOR = dict(provider="cronitor.io", api_key="KEY", job_id="MON")  # pragma: allowlist secret


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retries ON (the suite-wide fixture switched them off), with delays of a few milliseconds."""
    monkeypatch.setattr(monitoring, "_RETRY_ENABLED", True)
    monkeypatch.setattr(monitoring, "RETRY_INITIAL_DELAY_SEC", 0.01)
    monkeypatch.setattr(monitoring, "RETRY_MAX_DELAY_SEC", 0.05)


def _resp(status: int) -> MagicMock:
    return MagicMock(status_code=status, text="x")


class TestARetryIsDelivered:
    @patch("pyutilz.system.monitoring.requests")
    def test_a_network_error_is_retried_until_it_goes_through(self, mock_req: MagicMock) -> None:
        mock_req.post.side_effect = [ConnectionError("getaddrinfo failed"), ConnectionError("still down"), _resp(200)]
        job_completed(data="ids=0", **CRONITOR)  # type: ignore[arg-type]
        assert monitoring._RETRIER.wait_idle(5.0)
        assert mock_req.post.call_count == 3

    @patch("pyutilz.system.monitoring.requests")
    def test_the_first_attempt_is_inline_and_a_success_queues_nothing(self, mock_req: MagicMock) -> None:
        mock_req.post.return_value = _resp(200)
        job_completed(**CRONITOR)  # type: ignore[arg-type]
        assert mock_req.post.call_count == 1 and monitoring._RETRIER.pending_count() == 0

    @patch("pyutilz.system.monitoring.requests")
    def test_a_server_error_is_retried(self, mock_req: MagicMock) -> None:
        mock_req.post.side_effect = [_resp(503), _resp(200)]
        job_completed(**CRONITOR)  # type: ignore[arg-type]
        assert monitoring._RETRIER.wait_idle(5.0) and mock_req.post.call_count == 2

    @patch("pyutilz.system.monitoring.requests")
    def test_a_rate_limit_is_retried_silently_on_the_first_attempt(self, mock_req: MagicMock) -> None:
        mock_req.post.side_effect = [_resp(429), _resp(200)]
        with patch("pyutilz.system.monitoring.logger") as log:
            job_completed(**CRONITOR)  # type: ignore[arg-type]
            assert monitoring._RETRIER.wait_idle(5.0)
        log.warning.assert_not_called()
        assert mock_req.post.call_count == 2

    @patch("pyutilz.system.monitoring.requests")
    def test_blocking_false_retries_too(self, mock_req: MagicMock) -> None:
        mock_req.post.side_effect = [ConnectionError("down"), _resp(200)]
        with patch("pyutilz.system.monitoring._TIMEOUT_EXECUTOR.submit") as submit:
            job_completed(blocking=False, **CRONITOR)  # type: ignore[arg-type]
            submit.call_args[0][0]()  # the executor's worker runs the send
        assert monitoring._RETRIER.wait_idle(5.0) and mock_req.post.call_count == 2


class TestARefusalIsNotRetried:
    @pytest.mark.parametrize("status", [400, 403, 404])
    @patch("pyutilz.system.monitoring.requests")
    def test_a_verdict_a_retry_cannot_change(self, mock_req: MagicMock, status: int) -> None:
        mock_req.post.return_value = _resp(status)
        job_completed(**CRONITOR)  # type: ignore[arg-type]
        assert monitoring._RETRIER.pending_count() == 0 and mock_req.post.call_count == 1

    @patch("pyutilz.system.monitoring.requests")
    def test_retry_false_is_the_old_single_shot(self, mock_req: MagicMock) -> None:
        mock_req.post.side_effect = ConnectionError("down")
        job_completed(retry=False, **CRONITOR)  # type: ignore[arg-type]
        assert monitoring._RETRIER.pending_count() == 0 and mock_req.post.call_count == 1

    @patch("pyutilz.system.monitoring.requests", None)
    def test_a_missing_requests_package_is_not_retried(self) -> None:
        job_completed(**CRONITOR)  # type: ignore[arg-type]
        assert monitoring._RETRIER.pending_count() == 0


class TestOneHeartbeatPerStream:
    @patch("pyutilz.system.monitoring.requests")
    def test_a_newer_failure_replaces_the_older_pending_one(self, mock_req: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(monitoring, "RETRY_INITIAL_DELAY_SEC", 30.0)  # keep both pending while we look
        monkeypatch.setattr(monitoring, "RETRY_MAX_DELAY_SEC", 30.0)
        mock_req.post.side_effect = ConnectionError("down")
        job_completed(data="scan 1", **CRONITOR)  # type: ignore[arg-type]
        job_completed(data="scan 2", **CRONITOR)  # type: ignore[arg-type]
        assert monitoring._RETRIER.pending_count() == 1
        with monitoring._RETRIER._cv:
            (item,) = monitoring._RETRIER._pending.values()
        assert item.params is not None and item.params["msg"] == "scan 2"

    @patch("pyutilz.system.monitoring.requests")
    def test_a_later_success_cancels_the_pending_older_one(self, mock_req: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(monitoring, "RETRY_INITIAL_DELAY_SEC", 30.0)
        monkeypatch.setattr(monitoring, "RETRY_MAX_DELAY_SEC", 30.0)
        mock_req.post.side_effect = [ConnectionError("down"), _resp(200)]
        job_completed(data="scan 1", **CRONITOR)  # type: ignore[arg-type]
        assert monitoring._RETRIER.pending_count() == 1
        job_completed(data="scan 2", **CRONITOR)  # type: ignore[arg-type]
        assert monitoring._RETRIER.pending_count() == 0, "the monitor already heard from a later run"

    @patch("pyutilz.system.monitoring.requests")
    def test_different_monitors_do_not_cancel_each_other(self, mock_req: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(monitoring, "RETRY_INITIAL_DELAY_SEC", 30.0)
        monkeypatch.setattr(monitoring, "RETRY_MAX_DELAY_SEC", 30.0)
        mock_req.post.side_effect = ConnectionError("down")
        job_completed(provider="cronitor.io", api_key="KEY", job_id="A")  # pragma: allowlist secret
        job_completed(provider="cronitor.io", api_key="KEY", job_id="B")  # pragma: allowlist secret
        assert monitoring._RETRIER.pending_count() == 2

    @patch("pyutilz.system.monitoring.requests")
    def test_the_callers_dict_is_copied(self, mock_req: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(monitoring, "RETRY_INITIAL_DELAY_SEC", 30.0)
        monkeypatch.setattr(monitoring, "RETRY_MAX_DELAY_SEC", 30.0)
        mock_req.post.side_effect = ConnectionError("down")
        body = {"n": 1}
        job_completed(provider="cronitor.io", api_key="KEY", job_id="J", data=body)  # pragma: allowlist secret
        body["n"] = 2
        with monitoring._RETRIER._cv:
            (item,) = monitoring._RETRIER._pending.values()
        assert item.data == {"n": 1}


class TestBackoff:
    def test_it_doubles_and_is_capped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(monitoring, "RETRY_INITIAL_DELAY_SEC", 5.0)
        monkeypatch.setattr(monitoring, "RETRY_MAX_DELAY_SEC", 300.0)
        monkeypatch.setattr(random, "uniform", lambda a, b: 1.0)  # the jitter source the module draws from
        assert [monitoring._backoff_delay(n) for n in (1, 2, 3, 4, 5, 6, 7, 8, 20)] == [5, 10, 20, 40, 80, 160, 300, 300, 300]

    def test_jitter_stays_within_twenty_percent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(monitoring, "RETRY_INITIAL_DELAY_SEC", 100.0)
        monkeypatch.setattr(monitoring, "RETRY_MAX_DELAY_SEC", 1000.0)
        assert all(80.0 <= monitoring._backoff_delay(1) <= 120.0 for _ in range(200))


class TestGivingUpAndExit:
    @patch("pyutilz.system.monitoring.requests")
    def test_a_heartbeat_older_than_the_max_age_is_dropped_with_one_warning(self, mock_req: MagicMock, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        mock_req.post.side_effect = ConnectionError("down")
        monkeypatch.setattr(monitoring, "RETRY_MAX_AGE_SEC", 0.05)
        with caplog.at_level(logging.WARNING, logger="pyutilz.system.monitoring"):
            job_completed(**CRONITOR)  # type: ignore[arg-type]
            assert monitoring._RETRIER.wait_idle(5.0)
        assert [r for r in caplog.records if "Giving up on heartbeat" in r.getMessage()], "the drop must be visible"

    @patch("pyutilz.system.monitoring.requests")
    def test_the_exit_flush_makes_one_last_attempt(self, mock_req: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(monitoring, "RETRY_INITIAL_DELAY_SEC", 30.0)
        monkeypatch.setattr(monitoring, "RETRY_MAX_DELAY_SEC", 30.0)
        mock_req.post.side_effect = [ConnectionError("down"), _resp(200)]
        job_completed(**CRONITOR)  # type: ignore[arg-type]
        assert monitoring._RETRIER.pending_count() == 1
        monitoring._RETRIER.flush(budget=5.0)
        assert monitoring._RETRIER.pending_count() == 0 and mock_req.post.call_count == 2

    def test_a_process_that_exits_right_after_a_failed_ping_still_delivers_it(self) -> None:
        """The behaviour the atexit hook exists for, in a real interpreter: the retry thread is a daemon and would
        die with the process, so the last attempt has to happen during shutdown."""
        import subprocess
        import sys
        import textwrap

        child = textwrap.dedent(
            """
            import sys
            from pyutilz.system import monitoring

            calls = []

            class Resp:
                status_code = 200
                text = ""

            class R:
                @staticmethod
                def post(url, data=None, params=None, timeout=None):
                    calls.append(url)
                    if len(calls) == 1:
                        raise ConnectionError("getaddrinfo failed")
                    print("DELIVERED", len(calls))
                    sys.stdout.flush()
                    return Resp()

            monitoring.requests = R
            monitoring.RETRY_INITIAL_DELAY_SEC = 3600.0
            monitoring.job_completed(job_id="MON", provider="cronitor.io", api_key="KEY")  # pragma: allowlist secret
            """
        )
        out = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=60)
        assert out.returncode == 0, out.stderr
        assert "DELIVERED 2" in out.stdout, (out.stdout, out.stderr)

    @patch("pyutilz.system.monitoring.requests")
    def test_the_retry_thread_is_a_daemon_and_ends_when_idle(self, mock_req: MagicMock) -> None:
        mock_req.post.side_effect = [ConnectionError("down"), _resp(200)]
        job_completed(**CRONITOR)  # type: ignore[arg-type]
        thread = monitoring._RETRIER._thread
        assert thread is not None and thread.daemon
        assert monitoring._RETRIER.wait_idle(5.0)
        thread.join(timeout=5.0)
        assert not thread.is_alive(), "no thread is kept alive when nothing is pending"
        assert threading.active_count() >= 1

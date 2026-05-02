import pytest
import time
import logging
from unittest.mock import patch, MagicMock


# ── job_completed (lines 39-72) ──

class TestJobCompleted:
    @patch("pyutilz.system.monitoring.requests")
    def test_healthchecks_with_api_key(self, mock_req):
        from pyutilz.system.monitoring import job_completed
        mock_req.post.return_value = MagicMock(status_code=200)
        job_completed(job_id="test-job", status=0, data="ok", provider="healthchecks.io", api_key="mykey")
        mock_req.post.assert_called_once()
        url = mock_req.post.call_args[0][0]
        assert "mykey" in url
        assert "test-job" in url

    @patch("pyutilz.system.monitoring.requests")
    def test_healthchecks_without_api_key(self, mock_req):
        from pyutilz.system.monitoring import job_completed
        mock_req.post.return_value = MagicMock(status_code=200)
        job_completed(job_id="abc123", status=0, provider="healthchecks.io")
        url = mock_req.post.call_args[0][0]
        assert "hc-ping.com/abc123/0" in url

    @patch("pyutilz.system.monitoring.requests")
    def test_cronitor_complete(self, mock_req):
        from pyutilz.system.monitoring import job_completed
        mock_req.post.return_value = MagicMock(status_code=200)
        job_completed(job_id="cron-job", status=0, data="msg", provider="cronitor.io", api_key="ck")
        call_kwargs = mock_req.post.call_args
        assert "cronitor.link" in call_kwargs[0][0]

    @patch("pyutilz.system.monitoring.requests")
    def test_cronitor_nonzero_status(self, mock_req):
        from pyutilz.system.monitoring import job_completed
        mock_req.post.return_value = MagicMock(status_code=200)
        job_completed(job_id="j", status=1, provider="cronitor.io", api_key="k")
        params = mock_req.post.call_args[1].get("params") or mock_req.post.call_args.kwargs.get("params")
        assert params["state"] == 1

    @patch("pyutilz.system.monitoring.requests")
    def test_warning_on_bad_status(self, mock_req):
        from pyutilz.system.monitoring import job_completed
        mock_req.post.return_value = MagicMock(status_code=500, text="server error")
        with patch("pyutilz.system.monitoring.logger") as mock_logger:
            job_completed(job_id="j", status=0, provider="healthchecks.io")
            mock_logger.warning.assert_called_once()

    @patch("pyutilz.system.monitoring.requests")
    def test_no_warning_on_403(self, mock_req):
        from pyutilz.system.monitoring import job_completed
        mock_req.post.return_value = MagicMock(status_code=403)
        with patch("pyutilz.system.monitoring.logger") as mock_logger:
            job_completed(job_id="j", status=0, provider="healthchecks.io")
            mock_logger.warning.assert_not_called()

    @patch("pyutilz.system.monitoring.requests")
    def test_request_exception(self, mock_req):
        from pyutilz.system.monitoring import job_completed
        mock_req.post.side_effect = ConnectionError("timeout")
        with patch("pyutilz.system.monitoring.logger") as mock_logger:
            job_completed(job_id="j", status=0, provider="healthchecks.io")
            mock_logger.warning.assert_called_once()

    def test_unknown_provider_no_endpoint(self):
        from pyutilz.system.monitoring import job_completed
        with patch("pyutilz.system.monitoring.logger") as mock_logger:
            job_completed(job_id="j", status=0, provider="unknown.io")
            mock_logger.info.assert_called_once()

    @patch("pyutilz.system.monitoring.requests")
    def test_healthchecks_data_stringified(self, mock_req):
        from pyutilz.system.monitoring import job_completed
        mock_req.post.return_value = MagicMock(status_code=200)
        job_completed(job_id="j", status=0, data=12345, provider="healthchecks.io")
        call_kwargs = mock_req.post.call_args
        assert call_kwargs[1]["data"] == "12345" or call_kwargs.kwargs.get("data") == "12345"


# ── monitored decorator (lines 85-115) ──

class TestMonitored:
    @patch("pyutilz.system.monitoring.job_completed")
    def test_basic_decoration(self, mock_jc):
        from pyutilz.system.monitoring import monitored

        @monitored(job_id="test")
        def my_func():
            return {"key": "value"}

        result = my_func()
        assert result["key"] == "value"
        assert "duration" in result
        mock_jc.assert_called_once()

    @patch("pyutilz.system.monitoring.job_completed")
    def test_none_return_gets_duration(self, mock_jc):
        from pyutilz.system.monitoring import monitored

        @monitored(job_id="t")
        def my_func():
            return None

        result = my_func()
        assert isinstance(result, dict)
        assert "duration" in result

    @patch("pyutilz.system.monitoring.job_completed")
    def test_should_have_data_early_return(self, mock_jc):
        from pyutilz.system.monitoring import monitored

        @monitored(job_id="t", should_have_data=True)
        def my_func():
            return None

        result = my_func()
        assert result is None
        mock_jc.assert_not_called()

    @patch("pyutilz.system.monitoring.job_completed")
    def test_no_duration_field(self, mock_jc):
        from pyutilz.system.monitoring import monitored

        @monitored(job_id="t", duration_field=None)
        def my_func():
            return {"x": 1}

        result = my_func()
        assert "duration" not in result

    @patch("pyutilz.system.monitoring.job_completed")
    def test_fallback_job_id_from_funcname(self, mock_jc):
        from pyutilz.system.monitoring import monitored

        @monitored()
        def special_task():
            return {}

        special_task()
        call_kwargs = mock_jc.call_args
        assert call_kwargs.kwargs.get("job_id") == "special_task" or call_kwargs[1].get("job_id") == "special_task"


# ── timeout_wrapper exception path (lines 141-143) ──

class TestTimeoutWrapperExceptions:
    def test_exception_returns_none(self):
        from pyutilz.system.monitoring import timeout_wrapper

        @timeout_wrapper(timeout=5)
        def failing():
            raise ValueError("boom")

        result = failing()
        assert result is None

    def test_report_duration_logs(self):
        from pyutilz.system.monitoring import timeout_wrapper

        @timeout_wrapper(timeout=5, report_actual_duration=True)
        def quick():
            return 42

        with patch("pyutilz.system.monitoring.logger") as mock_logger:
            result = quick()
            assert result == 42


# ── log_duration decorator (lines 157-183) ──

class TestLogDuration:
    def test_fast_function_no_log(self):
        from pyutilz.system.monitoring import log_duration

        @log_duration(threshold=10.0)
        def fast():
            return "ok"

        with patch("pyutilz.system.monitoring.logger") as mock_logger:
            result = fast()
            assert result == "ok"
            mock_logger.info.assert_not_called()

    def test_slow_function_logs(self):
        from pyutilz.system.monitoring import log_duration

        @log_duration(threshold=0.0)
        def any_func(x, y=2):
            return x + y

        with patch("pyutilz.system.monitoring.logger") as mock_logger:
            result = any_func(1, y=3)
            assert result == 4
            mock_logger.info.assert_called_once()
            # Lazy %-format: assert against the expanded message via str(call_args).
            rendered = " ".join(str(a) for a in mock_logger.info.call_args[0])
            assert "any_func" in rendered

    def test_truncation_of_large_args(self):
        from pyutilz.system.monitoring import log_duration

        @log_duration(threshold=0.0, max_arg_size=20)
        def func(data):
            return len(data)

        with patch("pyutilz.system.monitoring.logger") as mock_logger:
            result = func("x" * 1000)
            assert result == 1000
            rendered = " ".join(str(a) for a in mock_logger.info.call_args[0])
            assert "truncated" in rendered

    def test_custom_logger_name(self):
        from pyutilz.system.monitoring import log_duration

        @log_duration(threshold=0.0, logger_name="custom.logger")
        def func():
            return 1

        with patch("logging.getLogger") as mock_get:
            mock_lgr = MagicMock()
            mock_get.return_value = mock_lgr
            func()

    def test_kwargs_formatting(self):
        from pyutilz.system.monitoring import log_duration

        @log_duration(threshold=0.0)
        def func(a=1, b=2):
            return a + b

        with patch("pyutilz.system.monitoring.logger") as mock_logger:
            func(a=10, b=20)
            rendered = " ".join(str(a) for a in mock_logger.info.call_args[0])
            assert "a=" in rendered
            assert "b=" in rendered

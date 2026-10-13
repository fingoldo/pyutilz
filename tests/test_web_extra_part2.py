"""Continuation of test_web_extra: split out to keep each test module under the 1k-LOC gate."""
from unittest.mock import Mock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_module_globals():
    """Reset web module globals before each test."""
    from pyutilz.web import web as mod
    mod.init_vars()
    mod.delay = 1
    mod.max_ip_queries = 0
    mod.last_used_dict = None
    mod.min_idle_interval_minutes = None
    mod.failed_dict = None
    mod.min_failed_idle_interval_minutes = None
    mod.cur_max_ip_queries = -1
    mod.proxy_server = None
    mod.was_blocked = False
    yield


# ===== init_vars =====

class TestHandleBlocking:
    @patch("pyutilz.web.web.get_new_session")
    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.set_proxy_last_use_time")
    def test_with_proxies(self, mock_splut, mock_sleep, mock_gns):
        from pyutilz.web import web as mod
        mod.proxies = {"https": "http://user:pass@1.2.3.4:8080"}  # pragma: allowlist secret
        mod.handle_blocking("test")
        mock_splut.assert_called_once()
        mock_gns.assert_called_once()

    @patch("pyutilz.web.web.get_new_session")
    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.set_proxy_last_use_time")
    def test_without_proxies(self, mock_splut, mock_sleep, mock_gns):
        from pyutilz.web import web as mod
        mod.proxies = None
        mod.handle_blocking("test")
        mock_gns.assert_called_once()


# ===== get_url =====

class TestGetUrl:
    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.get_new_session")
    def test_success_200(self, mock_gns, mock_sleep):
        from pyutilz.web import web as mod
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.text = "ok"
        mock_sess = Mock()
        mock_sess.get.return_value = mock_resp
        mod.sess = mock_sess
        mod.delay = 0
        r = mod.get_url("http://example.com", b_use_proxy=False, b_use_session=True, b_random_ua=False)
        assert r.status_code == 200

    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.get_new_session")
    def test_creates_session_when_none(self, mock_gns, mock_sleep):
        from pyutilz.web import web as mod
        mod.sess = None
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.text = ""
        # After get_new_session is called, sess should be set
        def side_effect(**kwargs):
            s = Mock()
            s.get.return_value = mock_resp
            mod.sess = s
        mock_gns.side_effect = side_effect
        mod.delay = 0
        r = mod.get_url("http://x.com", b_use_proxy=False, b_random_ua=False)
        mock_gns.assert_called()
        assert r.status_code == 200

    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.get_new_session")
    def test_exit_status_breaks(self, mock_gns, mock_sleep):
        from pyutilz.web import web as mod
        mock_resp = Mock()
        mock_resp.status_code = 404
        mock_resp.text = "not found"
        mock_sess = Mock()
        mock_sess.get.return_value = mock_resp
        mod.sess = mock_sess
        mod.delay = 0
        r = mod.get_url("http://x.com", exit_statuses=(404,), b_use_proxy=False, b_random_ua=False)
        assert r.status_code == 404

    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.get_new_session")
    def test_session_expired_breaks(self, mock_gns, mock_sleep):
        from pyutilz.web import web as mod
        mock_resp = Mock()
        mock_resp.status_code = 401
        mock_resp.text = "expired"
        mock_sess = Mock()
        mock_sess.get.return_value = mock_resp
        mod.sess = mock_sess
        mod.delay = 0
        r = mod.get_url("http://x.com", session_expired_statuses=(401,), b_use_proxy=False, b_random_ua=False)
        assert r.status_code == 401

    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.handle_blocking")
    @patch("pyutilz.web.web.get_new_session")
    def test_blocking_status_quit(self, mock_gns, mock_hb, mock_sleep):
        from pyutilz.web import web as mod
        mock_resp = Mock()
        mock_resp.status_code = 403
        mock_resp.text = "blocked"
        mock_sess = Mock()
        mock_sess.get.return_value = mock_resp
        mod.sess = mock_sess
        mod.delay = 0
        mod.get_url("http://x.com", blocking_statuses=(403,), quit_on_blocking=True, b_use_proxy=False, b_random_ua=False, max_retries=3)
        assert mod.was_blocked is True

    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.get_new_session")
    def test_ratelimited_no_proxy(self, mock_gns, mock_sleep):
        from pyutilz.web import web as mod
        responses = [Mock(status_code=429, text="slow down"), Mock(status_code=200, text="ok")]
        mock_sess = Mock()
        mock_sess.get.side_effect = responses
        mod.sess = mock_sess
        mod.delay = 0
        mod.proxy_server = None
        r = mod.get_url("http://x.com", b_use_proxy=False, b_random_ua=False, max_retries=3, ratelimited_sleep_interval=0)
        assert r.status_code == 200

    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.get_new_smartproxy", return_value={"http": "p", "https": "p"})
    @patch("pyutilz.web.web.get_new_session")
    def test_ratelimited_with_proxy(self, mock_gns, mock_gnsp, mock_sleep):
        from pyutilz.web import web as mod
        responses = [Mock(status_code=429, text="slow"), Mock(status_code=200, text="ok")]
        mock_sess = Mock()
        mock_sess.get.side_effect = responses
        mod.sess = mock_sess
        mod.delay = 0
        mod.proxy_server = "srv"
        mod.proxy_user = "u"
        mod.proxy_pass = "p"
        mod.proxy_min_port = 1
        mod.proxy_max_port = 2
        mod.proxy_port = 1
        mod.proxy_type = "http"
        r = mod.get_url("http://x.com", b_use_proxy=True, b_random_ua=False, max_retries=3, ratelimited_proxy_sleep_interval=0)
        assert r.status_code == 200

    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.get_new_session")
    def test_unknown_error_breaks_no_blocking_statuses(self, mock_gns, mock_sleep):
        from pyutilz.web import web as mod
        mock_resp = Mock()
        mock_resp.status_code = 500
        mock_resp.text = "error"
        mock_sess = Mock()
        mock_sess.get.return_value = mock_resp
        mod.sess = mock_sess
        mod.delay = 0
        r = mod.get_url("http://x.com", b_use_proxy=False, b_random_ua=False)
        assert r.status_code == 500

    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.get_new_session")
    def test_retry_status_retries(self, mock_gns, mock_sleep):
        from pyutilz.web import web as mod
        responses = [Mock(status_code=503, text="retry"), Mock(status_code=200, text="ok")]
        mock_sess = Mock()
        mock_sess.get.side_effect = responses
        mod.sess = mock_sess
        mod.delay = 0
        r = mod.get_url("http://x.com", retry_statuses=(503,), b_use_proxy=False, b_random_ua=False, max_retries=3)
        assert r.status_code == 200

    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.get_new_session")
    def test_blocking_error_in_text(self, mock_gns, mock_sleep):
        from pyutilz.web import web as mod
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.text = "captcha required"
        mock_sess = Mock()
        mock_sess.get.return_value = mock_resp
        mod.sess = mock_sess
        mod.delay = 0
        with patch("pyutilz.web.web.handle_blocking"):
            mod.get_url("http://x.com", blocking_errors=("captcha",), quit_on_blocking=True, b_use_proxy=False, b_random_ua=False)
        assert mod.was_blocked is True

    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.get_new_session")
    def test_no_blocking_error_succeeds(self, mock_gns, mock_sleep):
        from pyutilz.web import web as mod
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.text = "all good"
        mock_sess = Mock()
        mock_sess.get.return_value = mock_resp
        mod.sess = mock_sess
        mod.delay = 0
        mod.get_url("http://x.com", blocking_errors=("captcha",), b_use_proxy=False, b_random_ua=False)
        assert mod.was_blocked is False

    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.get_new_session")
    def test_proxy_error_gets_new_proxy(self, mock_gns, mock_sleep):
        from pyutilz.web import web as mod
        mock_sess = Mock()
        mock_sess.get.side_effect = [Exception("proxy connection error"), Mock(status_code=200, text="ok")]
        mod.sess = mock_sess
        mod.delay = 0
        mod.proxy_server = "srv"
        mod.proxy_user = "u"
        mod.proxy_pass = "p"
        mod.proxy_min_port = 1
        mod.proxy_max_port = 2
        mod.proxy_port = 1
        mod.proxy_type = "http"
        with patch("pyutilz.web.web.get_new_smartproxy", return_value={"http": "p", "https": "p"}):
            r = mod.get_url("http://x.com", b_use_proxy=True, b_random_ua=False, max_retries=3)
        assert r.status_code == 200

    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.get_new_session")
    def test_non_proxy_exception_no_new_proxy(self, mock_gns, mock_sleep):
        from pyutilz.web import web as mod
        mock_sess = Mock()
        mock_sess.get.side_effect = [Exception("something weird"), Mock(status_code=200, text="ok")]
        mod.sess = mock_sess
        mod.delay = 0
        mod.proxy_server = None
        r = mod.get_url("http://x.com", b_use_proxy=False, b_random_ua=False, max_retries=3)
        assert r.status_code == 200

    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.get_new_session")
    def test_all_retries_exhausted(self, mock_gns, mock_sleep):
        from pyutilz.web import web as mod
        mock_sess = Mock()
        mock_sess.get.side_effect = Exception("timed out forever")
        mod.sess = mock_sess
        mod.delay = 0
        r = mod.get_url("http://x.com", b_use_proxy=False, b_random_ua=False, max_retries=2)
        assert r is None

    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.get_new_session")
    def test_uses_requests_when_no_session(self, mock_gns, mock_sleep):
        from pyutilz.web import web as mod
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.text = "ok"
        mod.sess = Mock()  # not None so no new session created
        mod.delay = 0
        with patch("pyutilz.web.web.requests") as mock_requests:
            mock_requests.get.return_value = mock_resp
            r = mod.get_url("http://x.com", b_use_session=False, b_use_proxy=False, b_random_ua=False)
        assert r.status_code == 200

    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.get_new_session")
    def test_post_verb(self, mock_gns, mock_sleep):
        from pyutilz.web import web as mod
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.text = "ok"
        mock_sess = Mock()
        mock_sess.post.return_value = mock_resp
        mod.sess = mock_sess
        mod.delay = 0
        r = mod.get_url("http://x.com", verb="post", b_use_proxy=False, b_random_ua=False)
        assert r.status_code == 200

    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.get_new_session")
    def test_custom_headers(self, mock_gns, mock_sleep):
        from pyutilz.web import web as mod
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.text = ""
        mock_sess = Mock()
        mock_sess.get.return_value = mock_resp
        mod.sess = mock_sess
        mod.delay = 0
        mod.get_url("http://x.com", custom_headers={"X-Custom": "v"}, b_use_proxy=False, b_random_ua=False)
        call_kwargs = mock_sess.get.call_args[1]
        assert "x-custom" in call_kwargs["headers"]

    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.get_new_session")
    def test_inject_headers(self, mock_gns, mock_sleep):
        from pyutilz.web import web as mod
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.text = ""
        mock_sess = Mock()
        mock_sess.get.return_value = mock_resp
        mod.sess = mock_sess
        mod.headers = {"existing": "h"}
        mod.delay = 0
        mod.get_url("http://x.com", inject_headers={"added": "v"}, b_use_proxy=False, b_random_ua=False)
        call_kwargs = mock_sess.get.call_args[1]
        assert "added" in call_kwargs["headers"]
        assert "existing" in call_kwargs["headers"]

    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.get_new_session")
    def test_inject_headers_when_none(self, mock_gns, mock_sleep):
        from pyutilz.web import web as mod
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.text = ""
        mock_sess = Mock()
        mock_sess.get.return_value = mock_resp
        mod.sess = mock_sess
        mod.headers = None
        mod.delay = 0
        mod.get_url("http://x.com", custom_headers=None, inject_headers={"added": "v"}, b_use_proxy=False, b_random_ua=False)
        call_kwargs = mock_sess.get.call_args[1]
        assert "added" in call_kwargs["headers"]

    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.get_new_session")
    def test_max_ip_queries_triggers_new_session(self, mock_gns, mock_sleep):
        from pyutilz.web import web as mod
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.text = ""
        mock_sess = Mock()
        mock_sess.get.return_value = mock_resp
        mod.sess = mock_sess
        mod.max_ip_queries = 5
        mod.num_ip_queries = 100
        mod.cur_max_ip_queries = 5
        mod.delay = 0
        def side_effect(**kwargs):
            s = Mock()
            s.get.return_value = mock_resp
            mod.sess = s
        mock_gns.side_effect = side_effect
        mod.get_url("http://x.com", b_use_proxy=False, b_random_ua=False)
        mock_gns.assert_called()

    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.get_new_session")
    def test_verbose_logging(self, mock_gns, mock_sleep):
        from pyutilz.web import web as mod
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.text = ""
        mock_sess = Mock()
        mock_sess.get.return_value = mock_resp
        mod.sess = mock_sess
        mod.delay = 0
        # Should not raise
        r = mod.get_url("http://x.com", verbose=True, b_use_proxy=False, b_random_ua=False)
        assert r.status_code == 200

    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.get_new_session")
    def test_delay_causes_sleep(self, mock_gns, mock_sleep):
        from pyutilz.web import web as mod
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.text = ""
        mock_sess = Mock()
        mock_sess.get.return_value = mock_resp
        mod.sess = mock_sess
        mod.delay = 5
        mod.get_url("http://x.com", b_use_proxy=False, b_random_ua=False)
        mock_sleep.assert_called()

    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.get_new_session")
    def test_partial_content_206(self, mock_gns, mock_sleep):
        from pyutilz.web import web as mod
        mock_resp = Mock()
        mock_resp.status_code = 206
        mock_resp.text = "partial"
        mock_sess = Mock()
        mock_sess.get.return_value = mock_resp
        mod.sess = mock_sess
        mod.delay = 0
        r = mod.get_url("http://x.com", b_use_proxy=False, b_random_ua=False)
        assert r.status_code == 206

    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.get_new_session")
    def test_keeps_retrying_with_blocking_statuses(self, mock_gns, mock_sleep):
        """When blocking_statuses are set, unknown errors keep retrying."""
        from pyutilz.web import web as mod
        responses = [Mock(status_code=500, text="err"), Mock(status_code=200, text="ok")]
        mock_sess = Mock()
        mock_sess.get.side_effect = responses
        mod.sess = mock_sess
        mod.delay = 0
        r = mod.get_url("http://x.com", blocking_statuses=(403,), b_use_proxy=False, b_random_ua=False, max_retries=3)
        assert r.status_code == 200

    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.get_new_session")
    def test_fetch_exception_logged_even_when_verbose_false(self, mock_gns, mock_sleep, caplog):
        """Regression (2026-07-21 audit round 2, HIGH): the per-attempt exception was previously
        logged ONLY when verbose=True (the non-default); with the default verbose=False every
        network exception across all max_retries attempts was completely silent, leaving only a
        generic "Could not get url" warning with no information about why. Now always logged."""
        import logging

        from pyutilz.web import web as mod
        mock_sess = Mock()
        mock_sess.get.side_effect = ConnectionError("a specific dns failure reason")
        mod.sess = mock_sess
        mod.delay = 0
        with caplog.at_level(logging.WARNING, logger="pyutilz.web.web"):
            mod.get_url("http://x.com", b_use_proxy=False, b_random_ua=False, max_retries=1, verbose=False)
        assert any("a specific dns failure reason" in r.message for r in caplog.records)


# ===== download_to_file =====

class TestDownloadToFile:
    @patch("pyutilz.web.web.requests.get")
    def test_success(self, mock_get, tmp_path):
        from pyutilz.web import web as mod
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.iter_content.return_value = [b"data"]
        mock_get.return_value = mock_resp
        out = tmp_path / "out.bin"
        mod.download_to_file("http://x.com/f", str(out))
        assert out.read_bytes() == b"data"

    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.requests.get")
    def test_request_exception_on_all_attempts_returns_none(self, mock_get, mock_sleep, tmp_path):
        """When every requests.get() attempt raises, `request` stays None and the
        function returns without ever attempting to iter_content/write."""
        from pyutilz.web import web as mod
        mock_get.side_effect = Exception("fail")
        out = tmp_path / "out.bin"
        result = mod.download_to_file("http://x.com/f", str(out), max_attempts=3)
        # Reframed for the 2026-09-03 F46 fix: the function now returns False on total failure
        # (it returned None for success AND failure alike, so a caller could not branch on it).
        assert result is False
        assert not out.exists()

    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.requests.get")
    def test_write_exception_retries(self, mock_get, mock_sleep, tmp_path):
        from pyutilz.web import web as mod
        mock_resp = Mock()
        mock_resp.status_code = 200
        call_count = [0]
        def iter_content(size):
            call_count[0] += 1
            if call_count[0] == 1:
                raise OSError("disk full")
            return [b"ok"]
        mock_resp.iter_content = iter_content
        mock_get.return_value = mock_resp
        out = tmp_path / "out.bin"
        mod.download_to_file("http://x.com/f", str(out), max_attempts=3)
        assert out.read_bytes() == b"ok"
        # Each attempt now re-issues the HTTP request (a requests body is a single-use stream, so
        # retrying the write loop on the consumed response left a 0-byte file), and each response
        # it obtains is closed exactly once.
        assert mock_get.call_count == 2
        assert mock_resp.close.call_count == mock_get.call_count

    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.requests.get")
    def test_response_closed_on_success(self, mock_get, mock_sleep, tmp_path):
        from pyutilz.web import web as mod
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.iter_content.return_value = [b"data"]
        mock_get.return_value = mock_resp
        out = tmp_path / "out.bin"
        mod.download_to_file("http://x.com/f", str(out))
        mock_resp.close.assert_called_once()

    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.requests.get")
    def test_response_closed_even_when_write_raises_unhandled(self, mock_get, mock_sleep, tmp_path):
        """The response must be closed even if an exception propagates all the
        way out of the write loop (e.g. a bug unrelated to the download retry
        logic, or open() itself failing on every attempt)."""
        from pyutilz.web import web as mod
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.iter_content.side_effect = RuntimeError("boom")
        mock_get.return_value = mock_resp

        # Force every retry attempt to also fail so the exception surfaces
        # instead of being swallowed by the retry loop's own except/else.
        out = tmp_path / "out.bin"
        mod.download_to_file("http://x.com/f", str(out), max_attempts=1)
        mock_resp.close.assert_called_once()

    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.requests.get")
    def test_truncated_file_removed_after_all_write_attempts_fail(self, mock_get, mock_sleep, tmp_path):
        """Regression test: `open(filename, "wb")` truncates/creates the target file on every
        write attempt. If ALL max_attempts writes fail, the truncated/partial file was
        previously left on disk while the function still returned None -- identical to a
        successful download. The file must be removed so "no file on disk" reliably signals
        failure here too, matching the connection-failure path's tested contract
        (test_request_exception_on_all_attempts_returns_none)."""
        from pyutilz.web import web as mod
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.iter_content.side_effect = RuntimeError("boom")
        mock_get.return_value = mock_resp

        out = tmp_path / "out.bin"
        result = mod.download_to_file("http://x.com/f", str(out), max_attempts=2)
        # Reframed for the 2026-09-03 F46 fix: total failure is now reported as False, not None.
        assert result is False
        assert not out.exists()

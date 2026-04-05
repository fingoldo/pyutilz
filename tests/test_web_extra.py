"""Extended tests for pyutilz/web/web.py — covers uncovered lines with mocked network calls."""

import http
import ssl
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Helpers to import module-level symbols after mocking ensure_installed
# ---------------------------------------------------------------------------

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

class TestInitVars:
    def test_clears_session(self):
        from pyutilz.web import web as mod
        mod.sess = "something"
        mod.init_vars()
        assert mod.sess is None

    def test_resets_num_ip_queries(self):
        from pyutilz.web import web as mod
        mod.num_ip_queries = 99
        mod.init_vars()
        assert mod.num_ip_queries == 0

    def test_resets_headers(self):
        from pyutilz.web import web as mod
        mod.headers = {"x": "y"}
        mod.init_vars()
        assert mod.headers == {}

    def test_resets_proxies(self):
        from pyutilz.web import web as mod
        mod.proxies = {"http": "x"}
        mod.init_vars()
        assert mod.proxies is None

    def test_resets_timeout(self):
        from pyutilz.web import web as mod
        mod.timeout = 99
        mod.init_vars()
        assert mod.timeout == 10


# ===== get_external_ip =====

class TestGetExternalIp:
    @patch("pyutilz.web.web.urllib.request.urlopen")
    def test_returns_ipv4(self, mock_urlopen):
        from pyutilz.web import web as mod
        resp = Mock()
        resp.status = http.HTTPStatus.OK
        resp.read.return_value = b"1.2.3.4\n"
        mock_urlopen.return_value = resp
        assert mod.get_external_ip() == "1.2.3.4"

    @patch("pyutilz.web.web.urllib.request.urlopen")
    def test_returns_ipv6(self, mock_urlopen):
        from pyutilz.web import web as mod
        resp = Mock()
        resp.status = http.HTTPStatus.OK
        resp.read.return_value = b"::1"
        mock_urlopen.return_value = resp
        assert mod.get_external_ip() == "::1"

    @patch("pyutilz.web.web.urllib.request.urlopen")
    def test_weird_ip_skipped(self, mock_urlopen):
        from pyutilz.web import web as mod
        resp = Mock()
        resp.status = http.HTTPStatus.OK
        resp.read.return_value = b"weirdstuff"
        mock_urlopen.return_value = resp
        assert mod.get_external_ip() is None

    @patch("pyutilz.web.web.urllib.request.urlopen", side_effect=ssl.SSLCertVerificationError("cert"))
    def test_ssl_error_skipped(self, mock_urlopen):
        from pyutilz.web import web as mod
        assert mod.get_external_ip() is None

    @patch("pyutilz.web.web.urllib.request.urlopen", side_effect=RuntimeError("fail"))
    def test_generic_exception(self, mock_urlopen):
        from pyutilz.web import web as mod
        assert mod.get_external_ip() is None


# ===== get_ipinfo =====

class TestGetIpinfo:
    @patch("pyutilz.web.web.urllib.request.urlopen")
    def test_urllib_ok(self, mock_urlopen):
        from pyutilz.web import web as mod
        resp = Mock()
        resp.status = http.HTTPStatus.OK
        resp.read.return_value = b'{"ip":"1.2.3.4"}'
        mock_urlopen.return_value = resp
        assert mod.get_ipinfo(use_urllib=True) == {"ip": "1.2.3.4"}

    @patch("pyutilz.web.web.urllib.request.urlopen")
    def test_urllib_non_ok(self, mock_urlopen):
        from pyutilz.web import web as mod
        resp = Mock()
        resp.status = http.HTTPStatus.FORBIDDEN
        mock_urlopen.return_value = resp
        assert mod.get_ipinfo(use_urllib=True) == {}

    @patch("pyutilz.web.web.urllib.request.urlopen", side_effect=Exception("fail"))
    def test_urllib_exception(self, mock_urlopen):
        from pyutilz.web import web as mod
        assert mod.get_ipinfo(use_urllib=True) is None

    @patch("pyutilz.web.web.get_url")
    def test_requests_path_ok(self, mock_get_url):
        from pyutilz.web import web as mod
        mock_resp = Mock()
        mock_resp.json.return_value = {"ip": "5.6.7.8"}
        mock_get_url.return_value = mock_resp
        assert mod.get_ipinfo(use_urllib=False) == {"ip": "5.6.7.8"}

    @patch("pyutilz.web.web.get_url")
    def test_requests_path_json_error(self, mock_get_url):
        from pyutilz.web import web as mod
        mock_resp = Mock()
        mock_resp.json.side_effect = ValueError("bad json")
        mock_get_url.return_value = mock_resp
        assert mod.get_ipinfo(use_urllib=False) is None


# ===== connect =====

class TestConnect:
    def test_sets_all_globals(self):
        from pyutilz.web import web as mod
        mod.connect(
            m_proxy_user="u", m_proxy_pass="p", m_proxy_server="s",
            m_proxy_min_port="10", m_proxy_max_port="20",
            m_template_headers={"h": "v"}, m_timeout=30,
            m_proxy_port="15", m_proxy_type="socks5",
        )
        assert mod.proxy_user == "u"
        assert mod.proxy_pass == "p"
        assert mod.proxy_server == "s"
        assert mod.proxy_min_port == "10"
        assert mod.proxy_max_port == "20"
        assert mod.template_headers == {"h": "v"}
        assert mod.timeout == 30
        assert mod.proxy_port == "15"
        assert mod.proxy_type == "socks5"

    def test_calls_init_vars(self):
        from pyutilz.web import web as mod
        mod.sess = "old"
        mod.connect()
        assert mod.sess is None


# ===== set_proxy =====

class TestSetProxy:
    @patch("pyutilz.web.web.get_new_smartproxy", return_value={"http": "x", "https": "x"})
    def test_sets_globals_and_proxies(self, mock_gnsp):
        from pyutilz.web import web as mod
        mod.set_proxy(m_proxy_user="u", m_proxy_pass="p", m_proxy_server="s",
                       m_proxy_min_port="1", m_proxy_max_port="2",
                       m_proxy_port="1", m_proxy_type="socks5")
        assert mod.proxy_user == "u"
        assert mod.proxies == {"http": "x", "https": "x"}
        mock_gnsp.assert_called_once()


# ===== set_params =====

class TestSetParams:
    def test_sets_all(self):
        from pyutilz.web import web as mod
        d1, d2 = {}, {}
        mod.set_params(m_delay=5, m_max_ip_queries=100, m_last_used_dict=d1,
                       m_min_idle_interval_minutes=10, m_failed_dict=d2,
                       m_min_failed_idle_interval_minutes=60)
        assert mod.delay == 5
        assert mod.max_ip_queries == 100
        assert mod.last_used_dict is d1
        assert mod.min_idle_interval_minutes == 10
        assert mod.failed_dict is d2
        assert mod.min_failed_idle_interval_minutes == 60


# ===== set_proxy_last_use_time =====

class TestSetProxyLastUseTime:
    def test_adds_entry(self):
        from pyutilz.web import web as mod
        d = {}
        mod.set_proxy_last_use_time(d, {"http": "a"})
        assert len(d) == 1
        assert isinstance(list(d.values())[0], datetime)

    def test_none_dict_noop(self):
        from pyutilz.web import web as mod
        mod.set_proxy_last_use_time(None, {"http": "a"})  # no error

    def test_non_dict_noop(self):
        from pyutilz.web import web as mod
        mod.set_proxy_last_use_time("notadict", {"http": "a"})  # no error


# ===== make_proxies_dict =====

class TestMakeProxiesDict:
    def test_with_auth(self):
        from pyutilz.web import web as mod
        r = mod.make_proxies_dict("user", "pass", "srv", 8080, "https")
        assert r == {"http": "https://user:pass@srv:8080", "https": "https://user:pass@srv:8080"}

    def test_without_auth(self):
        from pyutilz.web import web as mod
        r = mod.make_proxies_dict(None, None, "srv", 8080, "http")
        assert r == {"http": "http://srv:8080", "https": "http://srv:8080"}

    def test_without_user_only(self):
        from pyutilz.web import web as mod
        r = mod.make_proxies_dict(None, "pass", "srv", 8080)
        assert "srv:8080" in r["http"]

    def test_proxy_type_socks(self):
        from pyutilz.web import web as mod
        r = mod.make_proxies_dict("u", "p", "s", 1080, "socks5")
        assert r["http"].startswith("socks5://")


# ===== get_new_smartproxy =====

class TestGetNewSmartproxy:
    def test_returns_proxies_immediately_no_dicts(self):
        from pyutilz.web import web as mod
        r = mod.get_new_smartproxy("u", "p", "srv", 20001, 20002,
                                   last_used_dict={}, failed_dict={})
        assert "http" in r and "https" in r

    def test_fixed_port(self):
        from pyutilz.web import web as mod
        r = mod.get_new_smartproxy("u", "p", "srv", proxy_port=9999)
        assert "9999" in r["http"]

    def test_skips_recently_used_then_returns(self):
        from pyutilz.web import web as mod
        from joblib import hash as jl_hash
        # Build a proxy dict for fixed port and mark it as recently used
        proxies = mod.make_proxies_dict("u", "p", "srv", 5555, "http")
        key = jl_hash(proxies)
        last_used = {key: datetime.utcnow()}
        # With fixed port and recent use, it would loop forever.
        # Use random port range so it eventually gets a different port.
        r = mod.get_new_smartproxy("u", "p", "srv", 5000, 6000,
                                   last_used_dict=last_used,
                                   min_idle_interval_minutes=999,
                                   failed_dict={})
        assert "http" in r

    def test_failed_dict_blocks_port(self):
        from pyutilz.web import web as mod
        from joblib import hash as jl_hash
        proxies = mod.make_proxies_dict("u", "p", "srv", 5555, "http")
        key = jl_hash(proxies)
        failed = {key: datetime.utcnow()}
        # Random port range so it eventually gets through
        r = mod.get_new_smartproxy("u", "p", "srv", 5000, 6000,
                                   last_used_dict={},
                                   min_idle_interval_minutes=999,
                                   failed_dict=failed)
        assert "http" in r


# ===== is_rotating_proxy =====

class TestIsRotatingProxy:
    def test_true_case(self):
        from pyutilz.web import web as mod
        r = mod.is_rotating_proxy({
            "PROXY_HOST": "gate.dc.smartproxy.com",
            "PROXY_MIN_PORT": 20000,
            "PROXY_MAX_PORT": 20000,
        })
        assert r is True

    def test_wrong_host(self):
        from pyutilz.web import web as mod
        r = mod.is_rotating_proxy({
            "PROXY_HOST": "other.com",
            "PROXY_MIN_PORT": 20000,
            "PROXY_MAX_PORT": 20000,
        })
        assert r is None

    def test_wrong_min_port(self):
        from pyutilz.web import web as mod
        r = mod.is_rotating_proxy({
            "PROXY_HOST": "gate.dc.smartproxy.com",
            "PROXY_MIN_PORT": 10000,
            "PROXY_MAX_PORT": 20000,
        })
        assert r is None

    def test_wrong_max_port(self):
        from pyutilz.web import web as mod
        r = mod.is_rotating_proxy({
            "PROXY_HOST": "gate.dc.smartproxy.com",
            "PROXY_MIN_PORT": 20000,
            "PROXY_MAX_PORT": 30000,
        })
        assert r is None

    def test_empty_dict(self):
        from pyutilz.web import web as mod
        assert mod.is_rotating_proxy({}) is None


# ===== report_params =====

class TestReportParams:
    def test_logs_without_error(self):
        from pyutilz.web import web as mod
        mod.report_params("http://x", None, None, None, None, None, 10)


# ===== get_new_session =====

class TestGetNewSession:
    def test_creates_session(self):
        from pyutilz.web import web as mod
        mod.get_new_session(b_random_ua=False, b_use_proxy=False)
        assert mod.sess is not None
        assert mod.num_ip_queries == 0

    def test_random_ua_sets_user_agent(self):
        from pyutilz.web import web as mod
        mod.template_headers = None
        mod.get_new_session(b_random_ua=True, b_use_proxy=False)
        assert "user-agent" in mod.headers

    def test_template_headers_preserved(self):
        from pyutilz.web import web as mod
        mod.template_headers = {"x-custom": "val"}
        mod.get_new_session(b_random_ua=True, b_use_proxy=False)
        assert mod.headers.get("x-custom") == "val"
        assert "user-agent" in mod.headers

    @patch("pyutilz.web.web.get_new_smartproxy", return_value={"http": "p", "https": "p"})
    def test_with_proxy(self, mock_gnsp):
        from pyutilz.web import web as mod
        mod.proxy_server = "srv"
        mod.proxy_user = "u"
        mod.proxy_pass = "p"
        mod.proxy_min_port = 1
        mod.proxy_max_port = 2
        mod.proxy_port = 1
        mod.proxy_type = "http"
        mod.get_new_session(b_random_ua=False, b_use_proxy=True)
        mock_gnsp.assert_called_once()
        assert mod.proxies == {"http": "p", "https": "p"}

    def test_no_proxy_server_skips_proxy(self):
        from pyutilz.web import web as mod
        mod.proxy_server = None
        mod.get_new_session(b_random_ua=False, b_use_proxy=True)
        assert mod.proxies is None


# ===== handle_blocking =====

class TestHandleBlocking:
    @patch("pyutilz.web.web.get_new_session")
    @patch("pyutilz.web.web.sleep")
    @patch("pyutilz.web.web.set_proxy_last_use_time")
    def test_with_proxies(self, mock_splut, mock_sleep, mock_gns):
        from pyutilz.web import web as mod
        mod.proxies = {"https": "http://user:pass@1.2.3.4:8080"}
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
        r = mod.get_url("http://x.com", blocking_statuses=(403,), quit_on_blocking=True,
                        b_use_proxy=False, b_random_ua=False, max_retries=3)
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
        r = mod.get_url("http://x.com", b_use_proxy=False, b_random_ua=False, max_retries=3,
                        ratelimited_sleep_interval=0)
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
        r = mod.get_url("http://x.com", b_use_proxy=True, b_random_ua=False, max_retries=3,
                        ratelimited_proxy_sleep_interval=0)
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
            r = mod.get_url("http://x.com", blocking_errors=("captcha",), quit_on_blocking=True,
                            b_use_proxy=False, b_random_ua=False)
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
        r = mod.get_url("http://x.com", blocking_errors=("captcha",), b_use_proxy=False, b_random_ua=False)
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
        r = mod.get_url("http://x.com", custom_headers={"X-Custom": "v"},
                        b_use_proxy=False, b_random_ua=False)
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
        r = mod.get_url("http://x.com", inject_headers={"added": "v"},
                        b_use_proxy=False, b_random_ua=False)
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
        r = mod.get_url("http://x.com", custom_headers=None, inject_headers={"added": "v"},
                        b_use_proxy=False, b_random_ua=False)
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
        r = mod.get_url("http://x.com", b_use_proxy=False, b_random_ua=False)
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
        r = mod.get_url("http://x.com", b_use_proxy=False, b_random_ua=False)
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
    def test_request_exception_on_first_attempt_raises(self, mock_get, mock_sleep, tmp_path):
        """Source has a bug: `request` is unbound on first exception. Verify it raises."""
        from pyutilz.web import web as mod
        mock_get.side_effect = Exception("fail")
        out = tmp_path / "out.bin"
        with pytest.raises(UnboundLocalError):
            mod.download_to_file("http://x.com/f", str(out), max_attempts=3)

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
                raise IOError("disk full")
            return [b"ok"]
        mock_resp.iter_content = iter_content
        mock_get.return_value = mock_resp
        out = tmp_path / "out.bin"
        mod.download_to_file("http://x.com/f", str(out), max_attempts=3)
        assert out.read_bytes() == b"ok"

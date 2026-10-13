"""Extended tests for pyutilz/web/web.py — covers uncovered lines with mocked network calls."""

import http
import ssl
from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, patch
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
    @patch("pyutilz.web.web.ipinfo._direct_urlopen")
    def test_returns_ipv4(self, mock_urlopen):
        from pyutilz.web import web as mod
        resp = Mock()
        resp.status = http.HTTPStatus.OK
        resp.read.return_value = b"1.2.3.4\n"
        mock_urlopen.return_value = resp
        assert mod.get_external_ip() == "1.2.3.4"

    @patch("pyutilz.web.web.ipinfo._direct_urlopen")
    def test_returns_ipv6(self, mock_urlopen):
        from pyutilz.web import web as mod
        resp = Mock()
        resp.status = http.HTTPStatus.OK
        resp.read.return_value = b"::1"
        mock_urlopen.return_value = resp
        assert mod.get_external_ip() == "::1"

    @patch("pyutilz.web.web.ipinfo._direct_urlopen")
    def test_weird_ip_skipped(self, mock_urlopen):
        from pyutilz.web import web as mod
        resp = Mock()
        resp.status = http.HTTPStatus.OK
        resp.read.return_value = b"weirdstuff"
        mock_urlopen.return_value = resp
        assert mod.get_external_ip() is None

    @patch("pyutilz.web.web.ipinfo._direct_urlopen", side_effect=ssl.SSLCertVerificationError("cert"))
    def test_ssl_error_skipped(self, mock_urlopen):
        from pyutilz.web import web as mod
        assert mod.get_external_ip() is None

    @patch("pyutilz.web.web.ipinfo._direct_urlopen", side_effect=RuntimeError("fail"))
    def test_generic_exception(self, mock_urlopen):
        from pyutilz.web import web as mod
        assert mod.get_external_ip() is None

    @patch("pyutilz.web.web.ipinfo._direct_urlopen")
    def test_explicit_timeout_zero_is_not_silently_rewritten(self, mock_urlopen):
        """Regression: `timeout=timeout or 10` used to silently rewrite an explicit
        module-level `timeout = 0` to 10 -- `0 or 10` evaluates to `10` regardless of
        whether the caller meant it. Fixed to `timeout if timeout is not None else 10`."""
        from pyutilz.web import web as mod
        resp = Mock()
        resp.status = http.HTTPStatus.OK
        resp.read.return_value = b"1.2.3.4\n"
        mock_urlopen.return_value = resp
        orig_timeout = mod.timeout
        try:
            mod.timeout = 0
            mod.get_external_ip()
            assert mock_urlopen.call_args.kwargs["timeout"] == 0
        finally:
            mod.timeout = orig_timeout


# ===== get_ipinfo =====

class TestGetIpinfo:
    @patch("pyutilz.web.web.ipinfo._direct_urlopen")
    def test_explicit_timeout_zero_is_not_silently_rewritten(self, mock_urlopen):
        """Regression: `timeout=timeout or 10` used to silently rewrite an explicit
        module-level `timeout = 0` to 10 -- fixed to `timeout if timeout is not None else 10`."""
        from pyutilz.web import web as mod
        resp = Mock()
        resp.status = http.HTTPStatus.OK
        resp.read.return_value = b'{"ip":"1.2.3.4"}'
        mock_urlopen.return_value = resp
        orig_timeout = mod.timeout
        try:
            mod.timeout = 0
            mod.get_ipinfo(use_urllib=True)
            assert mock_urlopen.call_args.kwargs["timeout"] == 0
        finally:
            mod.timeout = orig_timeout

    @patch("pyutilz.web.web.ipinfo._direct_urlopen")
    def test_urllib_ok(self, mock_urlopen):
        from pyutilz.web import web as mod
        resp = Mock()
        resp.status = http.HTTPStatus.OK
        resp.read.return_value = b'{"ip":"1.2.3.4"}'
        mock_urlopen.return_value = resp
        assert mod.get_ipinfo(use_urllib=True) == {"ip": "1.2.3.4"}

    @patch("pyutilz.web.web.ipinfo._direct_urlopen")
    def test_urllib_non_ok(self, mock_urlopen):
        """Regression (2026-07-21 audit round 2, LOW): a non-200 response used to return `{}`
        while an exception returned None -- two different failure values for the same "the
        request failed" outcome. Both must now return None, matching get_external_ip."""
        from pyutilz.web import web as mod
        resp = Mock()
        resp.status = http.HTTPStatus.FORBIDDEN
        mock_urlopen.return_value = resp
        assert mod.get_ipinfo(use_urllib=True) is None

    @patch("pyutilz.web.web.ipinfo._direct_urlopen", side_effect=Exception("fail"))
    def test_urllib_exception(self, mock_urlopen):
        from pyutilz.web import web as mod
        assert mod.get_ipinfo(use_urllib=True) is None

    def test_urllib_non_ok_and_exception_use_the_same_failure_convention(self):
        """Both use_urllib=True failure modes (non-200, exception) must return the identical
        sentinel (None) -- matching get_external_ip's single-convention contract and this
        function's own use_urllib=False branch, which already returns None uniformly."""
        from pyutilz.web import web as mod

        with patch("pyutilz.web.web.ipinfo._direct_urlopen") as mock_urlopen:
            resp = Mock()
            resp.status = http.HTTPStatus.FORBIDDEN
            mock_urlopen.return_value = resp
            non_ok_result = mod.get_ipinfo(use_urllib=True)

        with patch("pyutilz.web.web.ipinfo._direct_urlopen", side_effect=Exception("fail")):
            exception_result = mod.get_ipinfo(use_urllib=True)

        assert non_ok_result is exception_result is None

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
        mod.set_params(
            m_delay=5, m_max_ip_queries=100, m_last_used_dict=d1, m_min_idle_interval_minutes=10, m_failed_dict=d2, m_min_failed_idle_interval_minutes=60
        )
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
        assert isinstance(next(iter(d.values())), datetime)

    def test_none_dict_noop(self):
        """A None store is skipped entirely: nothing is returned and no timestamp is recorded anywhere."""
        from pyutilz.web import web as mod
        assert mod.set_proxy_last_use_time(None, {"http": "a"}) is None

    def test_non_dict_noop(self):
        """A non-dict store is left untouched -- the guard is ``isinstance(dict)``, not ``try/except``.

        Uses a list (a MUTABLE non-dict) so "did not write" is observable, not merely "did not raise".
        """
        from pyutilz.web import web as mod
        not_a_dict: list = []
        assert mod.set_proxy_last_use_time(not_a_dict, {"http": "a"}) is None
        assert not_a_dict == [], "a non-dict store must not be written into"
        assert mod.set_proxy_last_use_time("notadict", {"http": "a"}) is None


# ===== make_proxies_dict =====

class TestMakeProxiesDict:
    def test_with_auth(self):
        from pyutilz.web import web as mod
        r = mod.make_proxies_dict("user", "pass", "srv", 8080, "https")
        assert r == {"http": "https://user:pass@srv:8080", "https": "https://user:pass@srv:8080"}  # pragma: allowlist secret

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

        r = mod.get_new_smartproxy("u", "p", "srv", 20001, 20002, last_used_dict={}, failed_dict={})
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
        last_used = {key: datetime.now(timezone.utc).replace(tzinfo=None)}
        # With fixed port and recent use, it would loop forever.
        # Use random port range so it eventually gets a different port.
        r = mod.get_new_smartproxy("u", "p", "srv", 5000, 6000, last_used_dict=last_used, min_idle_interval_minutes=999, failed_dict={})
        assert "http" in r

    def test_failed_dict_blocks_port(self):
        from pyutilz.web import web as mod
        from joblib import hash as jl_hash
        proxies = mod.make_proxies_dict("u", "p", "srv", 5555, "http")
        key = jl_hash(proxies)
        failed = {key: datetime.now(timezone.utc).replace(tzinfo=None)}
        # Random port range so it eventually gets through
        r = mod.get_new_smartproxy("u", "p", "srv", 5000, 6000, last_used_dict={}, min_idle_interval_minutes=999, failed_dict=failed)
        assert "http" in r

    def test_max_wait_seconds_raises_timeout_instead_of_hanging_forever(self):
        """Regression test: a fixed single-port pool where the only candidate is permanently
        "touched" (e.g. marked failed 24h ago per the default min_failed_idle_interval_minutes)
        previously blocked get_new_smartproxy() forever with zero way to detect/bound the hang.
        """
        from pyutilz.web import web as mod
        from joblib import hash as jl_hash

        proxies = mod.make_proxies_dict("u", "p", "srv", 9999, "http")
        key = jl_hash(proxies)
        failed = {key: datetime.now(timezone.utc).replace(tzinfo=None)}
        with patch("pyutilz.web.web.sleep"):
            with pytest.raises(TimeoutError, match="no eligible proxy found"):
                mod.get_new_smartproxy(
                    "u", "p", "srv", proxy_port=9999, failed_dict=failed,
                    min_idle_interval_minutes=999, warn_after_n_failures=1,
                    max_wait_seconds=0,
                )

    def test_unconditional_logging_on_stuck_wait_even_without_verbose(self, caplog):
        """The "could not get an untouched proxy" message must fire regardless of verbose= --
        previously it was gated on verbose=True, and every real call site in this module invokes
        get_new_smartproxy() with the default verbose=False, making a stuck wait completely
        undiagnosable from logs alone."""
        from pyutilz.web import web as mod
        from joblib import hash as jl_hash
        import logging

        proxies = mod.make_proxies_dict("u", "p", "srv", 9999, "http")
        key = jl_hash(proxies)
        failed = {key: datetime.now(timezone.utc).replace(tzinfo=None)}
        with patch("pyutilz.web.web.sleep"), caplog.at_level(logging.INFO, logger="pyutilz.web.web"):
            with pytest.raises(TimeoutError):
                mod.get_new_smartproxy(
                    "u", "p", "srv", proxy_port=9999, failed_dict=failed,
                    min_idle_interval_minutes=999, warn_after_n_failures=1,
                    max_wait_seconds=0, verbose=False,
                )
        assert any("Could not get an untouched proxy" in r.message for r in caplog.records)


# ===== report_params / handle_blocking credential redaction =====

class TestCredentialRedaction:
    def test_report_params_redacts_proxy_password(self, caplog):
        import logging
        from pyutilz.web import web as mod

        proxies = {"http": "http://scraperuser:s3cr3t@proxy.host:8080", "https": "http://scraperuser:s3cr3t@proxy.host:8080"}  # pragma: allowlist secret
        with caplog.at_level(logging.INFO, logger="pyutilz.web.web"):
            mod.report_params("https://example.com", proxies, None, None, None, {"authorization": "Bearer sk-secret"}, 10)
        logged = " ".join(r.message for r in caplog.records)
        assert "s3cr3t" not in logged
        assert "sk-secret" not in logged
        assert "scraperuser" not in logged
        assert "proxy.host:8080" in logged  # host:port kept for diagnostic value

    def test_report_params_handles_none_proxies(self, caplog):
        import logging
        from pyutilz.web import web as mod

        with caplog.at_level(logging.INFO, logger="pyutilz.web.web"):
            mod.report_params("https://example.com", None, None, None, None, None, 10)
        # The redaction helpers must pass None through and still produce the diagnostic line.
        logged = " ".join(r.getMessage() for r in caplog.records)
        assert "https://example.com" in logged
        assert "proxies=None" in logged, f"expected a None-proxies line, got: {logged!r}"

    def test_handle_blocking_no_indexerror_on_unauthenticated_proxy(self, caplog):
        """Regression test: proxies["https"].split("@")[1] raised IndexError for an
        unauthenticated proxy (make_proxies_dict() builds an "@"-free URL when
        proxy_user/proxy_pass are falsy)."""
        import logging
        from pyutilz.web import web as mod

        mod.proxies = {"https": "http://proxy.host:8080", "http": "http://proxy.host:8080"}
        mod.delay = None  # skip the sleep
        with patch("pyutilz.web.web.set_proxy_last_use_time"), patch("pyutilz.web.web.get_new_session"):
            with caplog.at_level(logging.WARNING, logger="pyutilz.web.web"):
                mod.handle_blocking("example.com")  # must not raise IndexError
        assert any("proxy.host:8080" in r.message for r in caplog.records)

    def test_handle_blocking_redacts_authenticated_proxy(self, caplog):
        import logging
        from pyutilz.web import web as mod

        mod.proxies = {
            "https": "http://user:s3cr3t@proxy.host:8080",  # pragma: allowlist secret
            "http": "http://user:s3cr3t@proxy.host:8080",  # pragma: allowlist secret
        }
        mod.delay = None
        with patch("pyutilz.web.web.set_proxy_last_use_time"), patch("pyutilz.web.web.get_new_session"):
            with caplog.at_level(logging.WARNING, logger="pyutilz.web.web"):
                mod.handle_blocking("example.com")
        logged = " ".join(r.message for r in caplog.records)
        assert "s3cr3t" not in logged
        assert "proxy.host:8080" in logged


# ===== download_in_parallel None-check ordering =====

class TestDownloadInParallelNoneCheck:
    def test_none_response_does_not_crash_the_whole_batch(self, monkeypatch):
        """Regression test: `if len(resp.history) > 0` was dereferenced BEFORE the
        `if resp is not None` check, so a single failed URL (grequests.map() puts None into the
        results for a request that raised) crashed the entire batch with AttributeError instead
        of just recording that one URL as errored."""
        import sys
        import types

        from pyutilz.web import web as mod

        fake_grequests = types.ModuleType("grequests")
        ok_resp = MagicMock(status_code=http.HTTPStatus.OK, history=[])
        fake_grequests.get = MagicMock(return_value="request_obj")
        fake_grequests.map = MagicMock(return_value=[None, ok_resp])
        monkeypatch.setitem(sys.modules, "grequests", fake_grequests)
        processed = []
        errored = mod.download_in_parallel(["http://bad.example", "http://ok.example"], lambda resp, url: processed.append(url))
        assert errored == ["http://bad.example"]
        assert processed == ["http://ok.example"]


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
    def test_logs_without_error(self, caplog):
        """report_params emits exactly one INFO record carrying the url and the timeout."""
        import logging
        from pyutilz.web import web as mod
        with caplog.at_level(logging.INFO, logger="pyutilz.web.web"):
            mod.report_params("http://x", None, None, None, None, None, 10)
        records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(records) == 1, f"expected one INFO record, got {len(records)}"
        assert "url=http://x" in records[0].getMessage()
        assert "timeout=10" in records[0].getMessage()


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

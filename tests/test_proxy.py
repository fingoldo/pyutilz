"""Tests for pyutilz.web.proxy subpackage."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from pyutilz.web.proxy import (
    DecodoProvider,
    DecodoSubscription,
    DecodoTrafficReport,
    DecodoTrafficRow,
    IP_CHECK_URLS,
    PortHealthTracker,
    ProxyConfig,
    check_ip_matches_real,
    get_ip,
    parse_ip_response,
    verify_proxy_ip,
)
from pyutilz.web.proxy.decodo import _parse_traffic_response, _fmt_gb


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return ProxyConfig(user="u", password="p", host="proxy.example.com", base_port=10000, port_range=500)


@pytest.fixture
def provider(cfg):
    return DecodoProvider(cfg, api_key="test-key-123")


@pytest.fixture
def tracker():
    return PortHealthTracker(window=600.0, min_requests=5, ban_rate_multiplier=2.0, ban_duration=5.0)


def _build_outlier(tracker, bad_port=99, good_ports=(1, 2, 3), n=10):
    """Helper: give good_ports 0% error rate, bad_port ~100% error rate, each with n requests."""
    for p in good_ports:
        for _ in range(n):
            tracker.report_success(p)
    for _ in range(n):
        tracker.report_error(bad_port)


# ── PortHealthTracker ───────────────────────────────────────────────────────

class TestPortHealthTracker:
    def test_no_bans_initially(self, tracker):
        assert tracker.stats()["banned_count"] == 0

    def test_single_error_no_ban(self, tracker):
        tracker.report_error(42)
        assert not tracker.is_banned(42)

    def test_errors_alone_dont_ban_without_peers(self, tracker):
        """With only one port tracked, no peer comparison is possible — no ban."""
        for _ in range(20):
            tracker.report_error(42)
        assert not tracker.is_banned(42)

    def test_outlier_port_gets_banned(self, tracker):
        """Port with ~100% errors banned when peers have ~0%."""
        _build_outlier(tracker, bad_port=99, good_ports=(1, 2, 3), n=10)
        assert tracker.is_banned(99)
        assert tracker.stats()["banned_count"] == 1

    def test_uniform_errors_no_ban(self, tracker):
        """When all ports have similar error rates, none gets banned."""
        for p in (1, 2, 3):
            for _ in range(5):
                tracker.report_error(p)
                tracker.report_success(p)
        assert tracker.stats()["banned_count"] == 0

    def test_below_min_requests_no_ban(self):
        """Ports with fewer than min_requests don't participate in comparison."""
        t = PortHealthTracker(min_requests=30, ban_rate_multiplier=2.0, ban_duration=5.0)
        # Port 1: 29 errors (below min_requests=30), port 2: 30 successes
        for _ in range(29):
            t.report_error(1)
        for _ in range(30):
            t.report_success(2)
        assert not t.is_banned(1)

    def test_ban_expires(self, tracker):
        with tracker._lock:
            tracker._banned[99] = time.monotonic() - 1.0  # already expired
        assert not tracker.is_banned(99)

    def test_pick_port_avoids_banned(self, tracker):
        _build_outlier(tracker, bad_port=1, good_ports=(2, 3, 4), n=10)
        assert tracker.is_banned(1)
        ports = {tracker.pick_port(500) for _ in range(50)}
        assert len(ports) > 1

    def test_pick_port_fallback_when_most_banned(self, tracker):
        """When >90% ports banned, falls back to random."""
        with tracker._lock:
            for i in range(1, 11):
                tracker._banned[i] = time.monotonic() + 300
        port = tracker.pick_port(10)
        assert 1 <= port <= 10

    def test_reset(self, tracker):
        _build_outlier(tracker, bad_port=1, good_ports=(2, 3), n=10)
        tracker.reset()
        assert tracker.stats()["banned_count"] == 0
        assert not tracker.is_banned(1)

    def test_stats_includes_avg_error_rate(self, tracker):
        for p in (1, 2):
            for _ in range(10):
                tracker.report_success(p)
        for _ in range(10):
            tracker.report_error(3)
        s = tracker.stats()
        assert "avg_error_rate" in s
        assert "qualified_ports" in s


# ── ProxyConfig & DecodoProvider ────────────────────────────────────────────

class TestDecodoProvider:
    def test_proxy_url_fixed_offset(self, provider):
        url = provider.proxy_url(port_offset=5)
        assert "proxy.example.com:10005" in url
        assert "u:p@" in url
        assert url.startswith("http://")

    def test_proxy_url_zero_offset(self, provider):
        assert ":10000" in provider.proxy_url(port_offset=0)

    def test_proxy_url_random(self, provider):
        urls = {provider.proxy_url() for _ in range(50)}
        ports = {int(u.rsplit(":", 1)[-1]) for u in urls}
        assert all(10001 <= p <= 10500 for p in ports)

    def test_proxies_dict(self, provider):
        d = provider.proxies(port_offset=42)
        assert d["http"] == d["https"]
        assert ":10042" in d["http"]

    def test_name(self, provider):
        assert provider.name == "DecodoProvider"

    def test_report_error_delegates(self, provider):
        """Outlier port gets banned when peers are healthy."""
        for p in (1, 2, 3):
            for _ in range(30):
                provider.report_success(p)
        for _ in range(30):
            provider.report_error(7)
        assert provider.health.is_banned(7)

    def test_report_success_delegates(self, provider):
        """Uniform error rates across ports — no bans."""
        for p in (7, 8, 9):
            for _ in range(30):
                provider.report_error(p)
        # All ports have same rate → none banned
        assert not provider.health.is_banned(7)


class TestDecodoProviderFromEnv:
    def test_from_env_success(self):
        env = {
            "PROXY_USER": "u", "PROXY_PASS": "p",
            "PROXY_HOST": "h", "PROXY_PORT": "10000",
            "DECODO_API_KEY": "key123",
        }
        with patch.dict("os.environ", env, clear=False):
            prov = DecodoProvider.from_env()
            assert prov.config.user == "u"
            assert prov.config.base_port == 10000
            assert prov.api_key == "key123"

    def test_from_env_missing_vars(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(EnvironmentError, match="Missing required"):
                DecodoProvider.from_env()

    def test_from_env_bad_port(self):
        env = {"PROXY_USER": "u", "PROXY_PASS": "p", "PROXY_HOST": "h", "PROXY_PORT": "abc"}
        with patch.dict("os.environ", env, clear=False):
            with pytest.raises(ValueError, match="must be an integer"):
                DecodoProvider.from_env()

    def test_from_env_custom_range(self):
        env = {
            "PROXY_USER": "u", "PROXY_PASS": "p",
            "PROXY_HOST": "h", "PROXY_PORT": "10000",
            "PROXY_PORT_RANGE": "1000",
        }
        with patch.dict("os.environ", env, clear=False):
            prov = DecodoProvider.from_env()
            assert prov.config.port_range == 1000


# ── parse_ip_response ──────────────────────────────────────────────────────

class TestParseIpResponse:
    @pytest.mark.parametrize("text,expected", [
        ('{"origin": "1.2.3.4"}', "1.2.3.4"),
        ('{"ip": "5.6.7.8"}', "5.6.7.8"),
        ("9.10.11.12\n", "9.10.11.12"),
        ("  1.2.3.4  ", "1.2.3.4"),
        ('{"origin": "1.2.3.4, 5.6.7.8"}', "1.2.3.4"),
        ("1.2.3.4, 5.6.7.8\n", "1.2.3.4"),
        ("{broken json", "{broken json"),
        ("{}", "{}"),
        ('{"foo": "bar"}', '{"foo": "bar"}'),
        ("2001:db8::1", "2001:db8::1"),
        ('{"ip": "::1"}', "::1"),
    ])
    def test_formats(self, text, expected):
        assert parse_ip_response(text) == expected


# ── get_ip ──────────────────────────────────────────────────────────────────

class TestGetIp:
    def test_first_url_succeeds(self):
        mock_lib = MagicMock()
        mock_lib.get.return_value = MagicMock(text='{"origin": "9.9.9.9"}')
        assert get_ip(mock_lib) == "9.9.9.9"
        assert mock_lib.get.call_count == 1

    def test_fallback_on_failure(self):
        mock_lib = MagicMock()
        mock_lib.get.side_effect = [ConnectionError("timeout"), MagicMock(text="8.8.8.8")]
        assert get_ip(mock_lib) == "8.8.8.8"

    def test_all_fail(self):
        mock_lib = MagicMock()
        mock_lib.get.side_effect = ConnectionError("down")
        assert get_ip(mock_lib) == "?"

    def test_proxies_kwarg(self):
        mock_lib = MagicMock()
        mock_lib.get.return_value = MagicMock(text="1.2.3.4")
        prx = {"http": "http://proxy:8080"}
        get_ip(mock_lib, prx=prx)
        _, kwargs = mock_lib.get.call_args
        assert kwargs["proxies"] == prx


# ── check_ip_matches_real ──────────────────────────────────────────────────

class TestCheckIpMatchesReal:
    @pytest.mark.parametrize("proxy_ip,real_ip,expected", [
        ("?", "1.2.3.4", False),
        ("1.2.3.4", "1.2.3.4", False),
        ("5.6.7.8", "1.2.3.4", True),
        ("?", "?", False),
        ("5.6.7.8", "?", True),
    ])
    def test_scenarios(self, proxy_ip, real_ip, expected):
        assert check_ip_matches_real(proxy_ip, real_ip, "test") is expected

    def test_exit_on_fail_leak(self):
        with pytest.raises(SystemExit):
            check_ip_matches_real("1.1.1.1", "1.1.1.1", "test", exit_on_fail=True)

    def test_exit_on_fail_unreachable(self):
        with pytest.raises(SystemExit):
            check_ip_matches_real("?", "1.1.1.1", "test", exit_on_fail=True)


# ── verify_proxy_ip ────────────────────────────────────────────────────────

class TestVerifyProxyIp:
    def test_requires_session_or_real_ip(self):
        with pytest.raises(ValueError, match="Provide either"):
            verify_proxy_ip(MagicMock())

    def test_with_real_ip(self):
        proxy_sess = MagicMock()
        proxy_sess.get.return_value = MagicMock(text='{"ip": "2.2.2.2"}')
        assert verify_proxy_ip(proxy_sess, real_ip="1.1.1.1", exit_on_fail=False) is True

    def test_leak_detected(self):
        proxy_sess = MagicMock()
        proxy_sess.get.return_value = MagicMock(text='{"ip": "1.1.1.1"}')
        assert verify_proxy_ip(proxy_sess, real_ip="1.1.1.1", exit_on_fail=False) is False


# ── IP_CHECK_URLS ──────────────────────────────────────────────────────────

class TestIpCheckUrls:
    def test_has_multiple(self):
        assert len(IP_CHECK_URLS) >= 2

    def test_all_https(self):
        for url in IP_CHECK_URLS:
            assert url.startswith("https://")


# ── Decodo API data classes ────────────────────────────────────────────────

class TestDecodoSubscription:
    def test_from_api(self):
        raw = {
            "service_type": "residential_proxies",
            "traffic_limit": "50",
            "traffic_per_period": "12.5",
            "valid_from": "2024-01-01",
            "valid_until": "2024-12-31",
            "users_limit": 3,
            "ip_address_limit": 2,
        }
        sub = DecodoSubscription.from_api(raw)
        assert sub.service_type == "residential_proxies"
        assert sub.traffic_limit_gb == 50.0
        assert sub.traffic_used_gb == 12.5
        assert sub.remaining_gb == 37.5
        assert 24.9 < sub.usage_pct < 25.1
        assert "residential_proxies" in sub.summary()
        assert "37.50 GB" in sub.summary()

    def test_from_api_missing_fields(self):
        sub = DecodoSubscription.from_api({})
        assert sub.service_type == "unknown"
        assert sub.traffic_limit_gb == 0.0
        assert sub.usage_pct == 0.0


class TestDecodoTrafficReport:
    def test_parse_traffic_response_list(self):
        data = [
            {"grouping_key": "2024-01-01", "requests": 1000, "totals": 1073741824},
            {"grouping_key": "2024-01-02", "requests": 2000, "totals": 2147483648},
        ]
        report = _parse_traffic_response(data, "day")
        assert len(report.rows) == 2
        assert report.total_requests == 3000
        assert report.rows[0].traffic_gb == pytest.approx(1.0)
        assert report.rows[1].traffic_gb == pytest.approx(2.0)
        assert "TOTAL" in report.summary()

    def test_parse_traffic_response_dict(self):
        data = {"data": [{"grouping_key": "upwork.com", "requests": 500, "traffic": 1048576}]}
        report = _parse_traffic_response(data, "target")
        assert len(report.rows) == 1
        assert report.rows[0].group_key == "upwork.com"

    def test_parse_traffic_response_empty(self):
        report = _parse_traffic_response({}, "day")
        assert report.rows == []
        assert report.total_requests == 0


class TestFmtGb:
    @pytest.mark.parametrize("val,expected", [
        (1.5, "1.50 GB"),
        (0.5, "512.0 MB"),
        (0.001, "1.0 MB"),
        (10.0, "10.00 GB"),
    ])
    def test_formatting(self, val, expected):
        assert _fmt_gb(val) == expected


# ── Decodo API methods ──────────────────────────────────────────────────────

class TestDecodoApiMethods:
    def test_api_headers_no_key(self, cfg):
        prov = DecodoProvider(cfg, api_key="")
        with pytest.raises(RuntimeError, match="DECODO_API_KEY"):
            prov._api_headers()

    def test_api_headers_with_key(self, provider):
        h = provider._api_headers()
        assert h["Authorization"] == "test-key-123"

    @patch("requests.get")
    def test_get_subscriptions(self, mock_get, provider):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"service_type": "residential_proxies", "traffic_limit": "50",
             "traffic_per_period": "10", "valid_from": "2024-01-01",
             "valid_until": "2024-12-31", "users_limit": 3, "ip_address_limit": 2}
        ]
        mock_get.return_value = mock_resp
        subs = provider.get_subscriptions()
        assert len(subs) == 1
        assert subs[0].traffic_limit_gb == 50.0

    @patch("requests.post")
    def test_get_traffic(self, mock_post, provider):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"grouping_key": "2024-01-01", "requests": 100, "totals": 1048576}
        ]
        mock_post.return_value = mock_resp
        report = provider.get_traffic(days=7)
        assert report.total_requests == 100
        assert len(report.rows) == 1

    def test_get_traffic_no_dates(self, provider):
        with pytest.raises(ValueError, match="days>0 or explicit start/end"):
            provider.get_traffic()


# ── curl_session / requests_session ─────────────────────────────────────────

class TestSessionHelpers:
    def test_curl_session(self, provider):
        mock_session = MagicMock()
        mock_cr = MagicMock()
        mock_cr.Session.return_value = mock_session
        with patch.dict("sys.modules", {
            "curl_cffi": MagicMock(requests=mock_cr),
            "curl_cffi.requests": mock_cr,
        }):
            from pyutilz.web.proxy import curl_session
            with curl_session(provider, port_offset=5) as s:
                assert s is mock_session
            mock_session.close.assert_called_once()

    def test_requests_session(self, provider):
        from pyutilz.web.proxy import requests_session
        with requests_session(provider, port_offset=3) as s:
            assert ":10003" in s.proxies["http"]

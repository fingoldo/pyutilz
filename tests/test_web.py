"""
Test suite for web.py
Tests cover web utility functions for HTTP operations.
"""

import pytest
from unittest.mock import patch


class TestWebUtilities:
    """Test web utility functions"""

    def test_make_proxies_dict(self):
        """Test creating proxies dictionary"""
        from pyutilz.web import make_proxies_dict

        result = make_proxies_dict(proxy_user="user", proxy_pass="pass", proxy_server="proxy.com", proxy_port=8080, proxy_type="https")

        assert isinstance(result, dict)
        assert "https" in result or "http" in result

    def test_is_rotating_proxy_true(self):
        """Test detecting rotating proxy"""
        from pyutilz.web import is_rotating_proxy

        proxy_server = {"PROXY_HOST": "gate.dc.smartproxy.com", "PROXY_MIN_PORT": 20000, "PROXY_MAX_PORT": 20000}
        result = is_rotating_proxy(proxy_server)

        assert result is True

    def test_is_rotating_proxy_false(self):
        """Test non-rotating proxy"""
        from pyutilz.web import is_rotating_proxy

        proxy_server = {"PROXY_HOST": "gate.dc.smartproxy.com", "PROXY_MIN_PORT": 20001, "PROXY_MAX_PORT": 37960}
        result = is_rotating_proxy(proxy_server)

        assert result is None

    def test_set_proxy_last_use_time(self):
        """Test setting proxy last use time"""
        from datetime import datetime, timedelta, timezone

        from pyutilz.web import set_proxy_last_use_time
        from joblib import hash as joblib_hash

        last_used = {}
        proxies = {"http": "http://proxy.com:8080"}

        set_proxy_last_use_time(last_used, proxies)

        key = joblib_hash(proxies)
        assert key in last_used
        assert isinstance(last_used[key], datetime)
        assert datetime.now(timezone.utc).replace(tzinfo=None) - last_used[key] < timedelta(seconds=5)


class TestWebConstants:
    """Test web module initialization"""

    def test_init_vars(self):
        """Test module variable initialization"""
        from pyutilz.web import init_vars

        # Should not crash
        init_vars()


class TestWebReporting:
    """Test web reporting utilities"""

    def test_report_params(self, caplog):
        """Test parameter reporting logs the expected INFO-level record."""
        import logging
        from pyutilz.web import report_params

        with caplog.at_level(logging.INFO, logger="pyutilz.web.web"):
            report_params(url="http://example.com", proxies=None, params=None, data=None, json=None, headers_to_use=None, timeout=30)

        assert any("http://example.com" in r.message for r in caplog.records)


class TestEnsureHttpScheme:
    """Regression tests for the file:// URL-scheme guard on urlopen call sites.

    Bug: get_ipinfo(url=...) and get_external_ip() passed their url/source
    straight to urllib.request.urlopen with no scheme check, so a caller
    passing a file:///etc/passwd-style URL would get local file contents
    back (bandit B310)."""

    def test_allows_http_and_https(self):
        from pyutilz.web.web import _ensure_http_scheme

        assert _ensure_http_scheme("http://example.com") == "http://example.com"
        assert _ensure_http_scheme("https://example.com/path?q=1") == "https://example.com/path?q=1"

    def test_rejects_file_scheme(self):
        from pyutilz.web.web import _ensure_http_scheme

        with pytest.raises(ValueError, match="file"):
            _ensure_http_scheme("file:///etc/passwd")

    def test_rejects_other_custom_schemes(self):
        from pyutilz.web.web import _ensure_http_scheme

        for scheme in ("ftp://example.com", "gopher://example.com", "data:text/plain,hi"):
            with pytest.raises(ValueError):
                _ensure_http_scheme(scheme)

    def test_get_ipinfo_rejects_file_url_before_reading_it(self):
        from pyutilz.web.web import get_ipinfo

        with patch("pyutilz.web.web.ipinfo._direct_urlopen") as mock_urlopen:
            result = get_ipinfo(use_urllib=True, url="file:///etc/passwd")
            mock_urlopen.assert_not_called()
        assert result is None

    def test_get_external_ip_rejects_non_http_provider(self):
        from pyutilz.web import web

        original_providers = web.IP_PROVIDERS
        try:
            web.IP_PROVIDERS = ["file:///etc/passwd"]
            with patch("pyutilz.web.web.ipinfo._direct_urlopen") as mock_urlopen:
                web.get_external_ip()
                mock_urlopen.assert_not_called()
        finally:
            web.IP_PROVIDERS = original_providers


def test_get_country_by_ip_returns_the_first_provider_that_states_a_country(monkeypatch):
    """The fallback chain is the point: providers rate-limit and 403 constantly, so a miss on the first
    must move to the next rather than becoming the answer."""
    from pyutilz.web import web

    calls: list[str] = []

    def fake_get_ipinfo(use_urllib: bool = False, url: str = "") -> object:
        calls.append(url)
        if "ipapi.co" in url:
            return None  # the shape a 429 produces through get_ipinfo
        return {"countryCode": "de", "country": " Germany ", "continent": "Europe", "query": "1.2.3.4"}

    monkeypatch.setattr(web, "get_ipinfo", fake_get_ipinfo)

    located = web.get_country_by_ip("1.2.3.4")

    assert located is not None
    assert located.country_code == "DE"  # normalised to ISO uppercase
    assert located.country_name == "Germany"  # whitespace stripped, provider casing kept
    assert located.provider == "ip-api.com"
    assert len(calls) == 2, "the first provider's miss must not end the search"


def test_get_country_by_ip_ignores_a_provider_that_answers_without_a_country(monkeypatch):
    """An error body ({"success": false, ...}) is a 200 with no country - taking it would produce an
    IpGeolocation with empty fields, which reads as an established answer."""
    from pyutilz.web import web

    monkeypatch.setattr(web, "get_ipinfo", lambda use_urllib=False, url="": {"success": False, "message": "quota"})

    assert web.get_country_by_ip("1.2.3.4") is None


def test_get_country_by_ip_substitutes_the_address_into_every_provider_url(monkeypatch):
    """`ip=None` must query the CALLER's own address - an empty path on every provider here - rather than
    sending the literal string "None"."""
    from pyutilz.web import web

    seen: list[str] = []
    monkeypatch.setattr(web, "get_ipinfo", lambda use_urllib=False, url="": seen.append(url) or None)

    assert web.get_country_by_ip(None) is None
    assert seen, "no provider URL was attempted at all"
    assert all("None" not in url for url in seen)
    assert any(url.rstrip("/").endswith("json") or url.endswith("/") or "?" in url for url in seen)

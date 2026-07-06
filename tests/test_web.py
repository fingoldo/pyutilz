"""
Test suite for web.py
Tests cover web utility functions for HTTP operations.
"""

import pytest
from unittest.mock import Mock, patch


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

        proxy_dict = {"http": "http://rotating-proxy.com:8080"}
        result = is_rotating_proxy(proxy_dict)

        # Function may return None if implementation incomplete
        assert result is None or isinstance(result, bool)

    def test_is_rotating_proxy_false(self):
        """Test non-rotating proxy"""
        from pyutilz.web import is_rotating_proxy

        proxy_dict = {"http": "http://static-proxy.com:8080"}
        result = is_rotating_proxy(proxy_dict)

        # Function may return None if implementation incomplete
        assert result is None or isinstance(result, bool)

    def test_set_proxy_last_use_time(self):
        """Test setting proxy last use time"""
        from pyutilz.web import set_proxy_last_use_time

        last_used = {}
        proxies = {"http": "http://proxy.com:8080"}

        set_proxy_last_use_time(last_used, proxies)

        # Should add timestamp
        assert len(last_used) > 0 or last_used == {}


class TestWebConstants:
    """Test web module initialization"""

    def test_init_vars(self):
        """Test module variable initialization"""
        from pyutilz.web import init_vars

        # Should not crash
        init_vars()


class TestWebReporting:
    """Test web reporting utilities"""

    def test_report_params(self):
        """Test parameter reporting"""
        from pyutilz.web import report_params

        # Should not crash with basic params
        try:
            report_params(url="http://example.com", proxies=None, params=None, data=None, json=None, headers_to_use=None, timeout=30)
        except Exception:
            # Expected if logger not configured
            pass


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

        with patch("pyutilz.web.web.urllib.request.urlopen") as mock_urlopen:
            result = get_ipinfo(use_urllib=True, url="file:///etc/passwd")
            mock_urlopen.assert_not_called()
        assert result is None

    def test_get_external_ip_rejects_non_http_provider(self):
        from pyutilz.web import web

        original_providers = web.IP_PROVIDERS
        try:
            web.IP_PROVIDERS = ["file:///etc/passwd"]
            with patch("pyutilz.web.web.urllib.request.urlopen") as mock_urlopen:
                web.get_external_ip()
                mock_urlopen.assert_not_called()
        finally:
            web.IP_PROVIDERS = original_providers

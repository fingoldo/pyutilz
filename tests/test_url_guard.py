"""Behaviour of ``pyutilz.web.url_guard``: the scheme and host check on a URL, the same check applied to every
redirect hop with credential headers dropped on a cross-host hop, and the proxy route chosen by ``urlopen_checked``.
Nothing here opens a socket: the opener is replaced by a mock where a request would otherwise be sent."""

from __future__ import annotations

import urllib.error
import urllib.request
from unittest.mock import patch

import pytest

from pyutilz.web import url_guard
from pyutilz.web.url_guard import (
    ALLOWED_SCHEMES,
    UnsafeURLError,
    _CheckedRedirectHandler,
    require_http_url,
    urlopen_checked,
)


class TestRequireHttpUrl:
    def test_http_and_https_are_accepted_and_returned_unchanged(self):
        assert require_http_url("http://example.org/x") == "http://example.org/x"
        assert require_http_url("https://example.org/x") == "https://example.org/x"

    def test_the_scheme_is_matched_case_insensitively(self):
        assert require_http_url("HTTPS://Example.org/x") == "HTTPS://Example.org/x"

    def test_surrounding_whitespace_is_ignored_when_checking_but_kept_in_the_result(self):
        assert require_http_url("  https://example.org/x  ") == "  https://example.org/x  "

    @pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.org/x", "gopher://example.org/x"])
    def test_a_scheme_outside_the_allow_list_is_refused_and_named(self, url):
        with pytest.raises(UnsafeURLError) as info:
            require_http_url(url)
        scheme = url.split(":", 1)[0]
        assert str(info.value) == f"refusing to open non-HTTP(S) URL (scheme {scheme!r}): {url!r}"

    def test_a_scheme_relative_url_has_no_scheme_and_is_refused(self):
        with pytest.raises(UnsafeURLError) as info:
            require_http_url("//example.org/x")
        assert "(scheme ''):" in str(info.value)

    def test_a_url_without_a_host_is_refused_with_its_own_message(self):
        with pytest.raises(UnsafeURLError) as info:
            require_http_url("https://")
        assert str(info.value) == "refusing to open URL with no host: 'https://'"

    def test_a_caller_allow_list_replaces_the_default(self):
        assert require_http_url("ftp://example.org/x", frozenset({"ftp"})) == "ftp://example.org/x"
        with pytest.raises(UnsafeURLError):
            require_http_url("https://example.org/x", frozenset({"ftp"}))

    def test_the_default_allow_list_is_http_and_https_only(self):
        assert ALLOWED_SCHEMES == frozenset({"http", "https"})

    def test_the_refusal_is_the_one_exported_from_the_historic_path(self):
        from pyutilz.web.exceptions import UnsafeURLError as Canonical

        assert url_guard.UnsafeURLError is Canonical


class TestRedirectHops:
    def test_a_redirect_to_an_allowed_scheme_is_followed(self):
        handler = _CheckedRedirectHandler()
        request = urllib.request.Request("https://a.example/x")
        new_request = handler.redirect_request(request, None, 302, "Found", {}, "https://a.example/y")
        assert new_request.full_url == "https://a.example/y"

    def test_a_redirect_to_a_disallowed_scheme_is_refused(self):
        handler = _CheckedRedirectHandler()
        request = urllib.request.Request("https://a.example/x")
        with pytest.raises(UnsafeURLError):
            handler.redirect_request(request, None, 302, "Found", {}, "ftp://internal.example/secret")

    def test_a_status_that_is_not_a_redirect_raises_with_the_server_message(self):
        handler = _CheckedRedirectHandler()
        request = urllib.request.Request("https://a.example/x")
        with pytest.raises(urllib.error.HTTPError) as info:
            handler.redirect_request(request, None, 404, "Nope", {"X": "1"}, "https://a.example/y")
        assert info.value.msg == "Nope"

    def test_a_cross_host_redirect_drops_the_credential_headers(self):
        handler = _CheckedRedirectHandler()
        request = urllib.request.Request("https://a.example/x", headers={"Authorization": "Bearer t", "Cookie": "session=1"})
        new_request = handler.redirect_request(request, None, 302, "Found", {}, "https://b.example/y")
        assert new_request.get_header("Authorization") is None
        assert new_request.get_header("Cookie") is None

    def test_a_same_host_redirect_keeps_the_credential_headers(self):
        handler = _CheckedRedirectHandler()
        request = urllib.request.Request("https://a.example/x", headers={"Authorization": "Bearer t"})
        new_request = handler.redirect_request(request, None, 302, "Found", {}, "https://a.example/y")
        assert new_request.get_header("Authorization") == "Bearer t"

    def test_the_host_comparison_ignores_case(self):
        handler = _CheckedRedirectHandler()
        request = urllib.request.Request("https://A.example/x", headers={"Authorization": "Bearer t"})
        new_request = handler.redirect_request(request, None, 302, "Found", {}, "https://a.example/y")
        assert new_request.get_header("Authorization") == "Bearer t"

    def test_a_non_credential_header_survives_a_cross_host_redirect(self):
        handler = _CheckedRedirectHandler()
        request = urllib.request.Request("https://a.example/x", headers={"Accept": "application/json"})
        new_request = handler.redirect_request(request, None, 302, "Found", {}, "https://b.example/y")
        assert new_request.get_header("Accept") == "application/json"

    def test_the_sensitive_header_names_are_the_four_credential_headers(self):
        assert _CheckedRedirectHandler.SENSITIVE_HEADERS == frozenset({"authorization", "proxy-authorization", "cookie", "cookie2"})


class TestUrlopenChecked:
    def test_a_disallowed_url_is_refused_before_any_opener_is_built(self):
        with patch("urllib.request.build_opener") as build:
            with pytest.raises(UnsafeURLError):
                urlopen_checked("file:///etc/passwd")
        build.assert_not_called()

    def test_a_disallowed_request_object_is_refused_by_its_full_url(self):
        with patch("urllib.request.build_opener") as build:
            with pytest.raises(UnsafeURLError):
                urlopen_checked(urllib.request.Request("ftp://example.org/x"))
        build.assert_not_called()

    def test_the_default_timeout_is_thirty_seconds(self):
        with patch("urllib.request.build_opener") as build:
            urlopen_checked("https://example.org/x")
        build.return_value.open.assert_called_once_with("https://example.org/x", timeout=30)

    def test_the_caller_timeout_is_passed_through(self):
        with patch("urllib.request.build_opener") as build:
            urlopen_checked("https://example.org/x", timeout=5)
        build.return_value.open.assert_called_once_with("https://example.org/x", timeout=5)

    def test_by_default_the_environment_proxy_handler_is_left_in_place(self):
        with patch("urllib.request.build_opener") as build:
            urlopen_checked("https://example.org/x")
        handlers = build.call_args.args
        assert len(handlers) == 1
        assert isinstance(handlers[0], _CheckedRedirectHandler)

    def test_an_empty_proxy_mapping_is_a_first_handler_that_suppresses_the_environment(self):
        with patch("urllib.request.build_opener") as build:
            urlopen_checked("https://example.org/x", proxies={})
        handlers = build.call_args.args
        assert isinstance(handlers[0], urllib.request.ProxyHandler)
        assert handlers[0].proxies == {}
        assert isinstance(handlers[1], _CheckedRedirectHandler)

    def test_a_proxy_mapping_is_used_exactly(self):
        with patch("urllib.request.build_opener") as build:
            urlopen_checked("https://example.org/x", proxies={"https": "http://proxy.example:3128"})
        assert build.call_args.args[0].proxies == {"https": "http://proxy.example:3128"}

    def test_the_redirect_check_uses_the_caller_allow_list(self):
        with patch("urllib.request.build_opener") as build:
            urlopen_checked("https://example.org/x", allowed_schemes=frozenset({"https"}))
        redirect = build.call_args.args[-1]
        assert redirect.allowed_schemes == frozenset({"https"})

    def test_the_returned_handle_is_the_opener_result(self):
        with patch("urllib.request.build_opener") as build:
            handle = urlopen_checked("https://example.org/x")
        assert handle is build.return_value.open.return_value

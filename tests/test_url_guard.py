"""Tests for pyutilz.web.url_guard.

Each test asserts the DEFENCE, not a linter's silence: a `file:///etc/passwd` (or ftp://, or a
scheme-relative URL) must never reach `urlopen`.
"""

from __future__ import annotations

import pytest

from pyutilz.web import url_guard

REFUSED_URLS = [
    "file:///etc/passwd",
    "file://C:/Windows/win.ini",
    "ftp://example.org/x",
    "gopher://example.org/",
    "data:text/plain;base64,aGk=",
    "jar:file:///tmp/x.jar!/y",
    "HTTPX://example.org/",  # not an allowed scheme despite the prefix
    "//example.org/no-scheme",
    "https://",  # scheme fine, no host at all
]


@pytest.mark.parametrize("url", REFUSED_URLS)
def test_non_http_schemes_are_refused(url: str):
    """A URL that is not http/https never reaches urlopen -- the refusal happens before any socket exists."""
    with pytest.raises(url_guard.UnsafeURLError):
        url_guard.require_http_url(url)


@pytest.mark.parametrize("url", ["http://example.org/x", "https://example.org/x?a=b", "  https://example.org/x  ", "HTTPS://EXAMPLE.ORG/"])
def test_http_and_https_are_allowed(url: str):
    """The guard must not be so tight that it breaks a real fetcher; case and surrounding space are fine."""
    assert url_guard.require_http_url(url) == url


def test_urlopen_checked_refuses_before_opening(tmp_path):
    """urlopen_checked raises on a file:// URL rather than reading the file -- the actual vulnerability.

    Written against a REAL file that a successful attack would disclose, so the test fails loudly if
    the guard is removed instead of passing on a technicality.
    """
    secret = tmp_path / "secret.txt"
    secret.write_text("PRIVATE", encoding="utf-8")
    with pytest.raises(url_guard.UnsafeURLError):
        url_guard.urlopen_checked(secret.as_uri())


def test_unsafe_url_error_is_not_an_oserror():
    """A caller retrying transient network faults on OSError must not retry a refused scheme too."""
    assert not issubclass(url_guard.UnsafeURLError, OSError)
    assert issubclass(url_guard.UnsafeURLError, ValueError)


def test_custom_allowed_schemes():
    assert url_guard.require_http_url("ftp://example.org/x", allowed_schemes=frozenset({"ftp"})) == "ftp://example.org/x"
    with pytest.raises(url_guard.UnsafeURLError):
        url_guard.require_http_url("http://example.org/x", allowed_schemes=frozenset({"ftp"}))


def test_urlopen_checked_accepts_a_request_object():
    """A prepared urllib.request.Request (not just a bare string URL) is accepted."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request("http://127.0.0.1:1/definitely-closed-port")
    # The guard passes (http scheme); the actual connection then fails, which is fine -- we're only
    # verifying the scheme check doesn't reject a Request object outright.
    with pytest.raises((urllib.error.URLError, ConnectionError, OSError)):
        url_guard.urlopen_checked(req, timeout=1)

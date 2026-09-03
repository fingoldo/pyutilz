"""Regression tests for the web/cloud findings of the 2026-09-03 domain audit
(09-domain-db-web-cloud-llm-text.md): F02, F03, F21, F22, F23, F42, F43, F44, F45, F46, F47.

Everything here is offline: no real socket, proxy, S3 bucket or website is ever touched.
"""

import http
import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, Mock, patch

import pytest

from pyutilz.web import web as webmod
from pyutilz.web.web import ipinfo as ipinfomod

# ---------------------------------------------------------------------------------------------
# F02 -- get_url()'s exception handler must not raise UnboundLocalError when get_new_session fails
# ---------------------------------------------------------------------------------------------


def test_f02_get_new_session_failure_does_not_raise_unbound_local(monkeypatch):
    """A proxy-gateway outage makes get_new_session() raise BEFORE the snapshot block runs; the
    except handler's proxy branch then read proxy_server_snapshot, which was never bound on that
    iteration -- an UnboundLocalError escaping get_url() and bypassing both the retry loop and the
    documented Optional[Response] return contract."""
    monkeypatch.setattr(webmod, "sess", None, raising=False)

    def boom(*args, **kwargs):
        raise ConnectionError("Cannot connect to proxy gateway")

    monkeypatch.setattr(webmod, "get_new_session", boom, raising=False)
    monkeypatch.setattr(webmod, "sleep", lambda *a, **kw: None, raising=False)

    res = webmod.get_url("http://example.com/x", b_use_proxy=True, max_retries=2, verbose=False)
    assert res is None


# ---------------------------------------------------------------------------------------------
# F03 -- get_new_smartproxy() must refresh now_time so a cooldown can expire while it waits
# ---------------------------------------------------------------------------------------------


def test_f03_cooldown_expires_while_the_loop_sleeps(monkeypatch):
    """now_time was captured once before `while True:`, so the wall time spent sleeping was
    invisible to the eligibility test and a single-candidate pool blocked forever even after the
    cooldown genuinely elapsed."""
    from joblib import hash as joblib_hash

    from pyutilz.web.web import proxy_pool

    clock = {"now": datetime(2026, 9, 3, 12, 0, 0)}

    class _FrozenDatetime:
        @staticmethod
        def now(tz=None):
            return clock["now"].replace(tzinfo=timezone.utc)

    monkeypatch.setattr(proxy_pool, "datetime", _FrozenDatetime)

    def fake_sleep(seconds):
        clock["now"] = clock["now"] + timedelta(seconds=seconds)

    monkeypatch.setattr(webmod, "sleep", fake_sleep, raising=False)

    proxies = webmod.make_proxies_dict("u", "p", "host", 20001, "http")
    failed_dict = {joblib_hash(proxies): clock["now"] - timedelta(seconds=30)}

    got = webmod.get_new_smartproxy(
        "u",
        "p",
        "host",
        20001,
        20001,
        failed_dict=failed_dict,
        min_failed_idle_interval_minutes=1,
        last_used_dict={},
        min_idle_interval_minutes=0,
        delay=10,
        warn_after_n_failures=1,
        max_wait_seconds=600,
    )
    assert got == proxies


# ---------------------------------------------------------------------------------------------
# F21 / F46 -- download_to_file(): non-2xx is a failure, and success/failure are distinguishable
# ---------------------------------------------------------------------------------------------


def _resp(status, chunks=(b"payload",)):
    r = Mock()
    r.status_code = status
    r.iter_content = Mock(return_value=list(chunks))
    return r


def test_f21_error_page_is_not_written_as_the_file(tmp_path):
    """A 500 body used to be streamed to disk and reported exactly like a successful download."""
    out = tmp_path / "model.bin"
    with patch.object(webmod, "sleep", lambda *a, **kw: None), patch.object(
        webmod.requests, "get", side_effect=lambda *a, **kw: _resp(500, [b"<html>Internal Server Error</html>"])
    ):
        result = webmod.download_to_file("http://x/f", str(out), max_attempts=2)
    assert result is False
    assert not out.exists()


def test_f21_transient_502_then_200_retries_and_succeeds(tmp_path):
    out = tmp_path / "model.bin"
    seq = [_resp(502, [b"<html>bad gateway</html>"]), _resp(200, [b"real-bytes"])]
    with patch.object(webmod, "sleep", lambda *a, **kw: None), patch.object(webmod.requests, "get", side_effect=lambda *a, **kw: seq.pop(0)):
        result = webmod.download_to_file("http://x/f", str(out), max_attempts=3)
    assert result is True
    assert out.read_bytes() == b"real-bytes"


def test_f46_success_and_failure_return_distinguishable_values(tmp_path):
    ok = tmp_path / "ok.bin"
    bad = tmp_path / "bad.bin"
    with patch.object(webmod, "sleep", lambda *a, **kw: None), patch.object(webmod.requests, "get", side_effect=lambda *a, **kw: _resp(200)):
        ok_result = webmod.download_to_file("http://x/f", str(ok))
    with patch.object(webmod, "sleep", lambda *a, **kw: None), patch.object(webmod.requests, "get", side_effect=Exception("connreset")):
        bad_result = webmod.download_to_file("http://x/f", str(bad), max_attempts=2)
    assert ok_result is True
    assert bad_result is False


def test_f46_exit_code_status_returns_false(tmp_path):
    out = tmp_path / "gone.bin"
    with patch.object(webmod, "sleep", lambda *a, **kw: None), patch.object(webmod.requests, "get", side_effect=lambda *a, **kw: _resp(404, [b"nope"])):
        assert webmod.download_to_file("http://x/f", str(out), exit_codes=(404,), max_attempts=2) is False
    assert not out.exists()


def test_f46_skip_because_exists_reports_success(tmp_path):
    out = tmp_path / "have.bin"
    out.write_bytes(b"already here")
    with patch.object(webmod.requests, "get", side_effect=AssertionError("must not issue a request")):
        assert webmod.download_to_file("http://x/f", str(out), rewrite_existing=False) is True
    assert out.read_bytes() == b"already here"


# ---------------------------------------------------------------------------------------------
# F22 -- get_ip()/parse_ip_response() must not accept an HTML error page as the exit IP
# ---------------------------------------------------------------------------------------------


def test_f22_html_error_page_is_not_an_ip():
    from pyutilz.web.proxy.ip_check import check_ip_matches_real, parse_ip_response

    page = "<html><head><title>503 Service Unavailable</title></head></html>"
    assert parse_ip_response(page) == "?"
    assert check_ip_matches_real(parse_ip_response(page), "8.8.8.8", "requests") is False


def test_f22_get_ip_skips_non_200_responses():
    from pyutilz.web.proxy import ip_check

    class _Client:
        def __init__(self):
            self.calls = []

        def get(self, url, **kwargs):
            self.calls.append(url)
            r = Mock()
            r.status_code = 503
            r.text = "<html>503 Service Unavailable</html>"
            return r

    client = _Client()
    assert ip_check.get_ip(client) == "?"
    assert len(client.calls) == len(ip_check.IP_CHECK_URLS)


def test_f22_plain_text_and_json_ips_still_parsed():
    from pyutilz.web.proxy.ip_check import parse_ip_response

    assert parse_ip_response("9.10.11.12\n") == "9.10.11.12"
    assert parse_ip_response('{"origin": "1.2.3.4, 5.6.7.8"}') == "1.2.3.4"


def test_f22_get_ip_returns_the_ip_on_a_healthy_200():
    from pyutilz.web.proxy import ip_check

    class _Client:
        def get(self, url, **kwargs):
            r = Mock()
            r.status_code = 200
            r.text = "1.2.3.4"
            return r

    assert ip_check.get_ip(_Client()) == "1.2.3.4"


# ---------------------------------------------------------------------------------------------
# F23 -- the ipinfo fetch paths must re-apply the http(s) allow-list to every redirect hop
# ---------------------------------------------------------------------------------------------


def test_f23_direct_fetch_goes_through_the_checked_opener():
    """_direct_urlopen must build an opener carrying _CheckedRedirectHandler, so a 302 to
    ftp://internal-host/... is refused instead of followed and JSON-parsed."""
    from pyutilz.web import url_guard

    captured = {}

    def spy_build_opener(*handlers):
        captured["handlers"] = handlers
        opener = MagicMock()
        opener.open.return_value = "opened"
        return opener

    with patch.object(url_guard.urllib.request, "build_opener", spy_build_opener):
        assert ipinfomod._direct_urlopen("https://example.org/ip", timeout=5) == "opened"
    assert any(isinstance(h, url_guard._CheckedRedirectHandler) for h in captured["handlers"])


def test_f23_direct_fetch_refuses_a_non_http_scheme(tmp_path):
    from pyutilz.web.exceptions import UnsafeURLError

    secret = tmp_path / "secret.txt"
    secret.write_text("top secret")
    with pytest.raises(UnsafeURLError):
        ipinfomod._direct_urlopen(secret.as_uri(), timeout=1)


def test_f23_proxy_opener_installs_the_checked_redirect_handler():
    from pyutilz.web import url_guard

    opener = ipinfomod._proxy_opener("u", "p", "proxy.example", 8080)
    assert opener is not None
    assert any(isinstance(h, url_guard._CheckedRedirectHandler) for h in opener.handlers)


def test_f23_hostless_url_is_refused():
    """_ensure_http_scheme now delegates to require_http_url, which also rejects a URL with no
    host (scheme-relative //evil/); UnsafeURLError is a ValueError so existing callers are
    unaffected."""
    from pyutilz.web.web import _ensure_http_scheme

    with pytest.raises(ValueError):
        _ensure_http_scheme("//evil.example/x")
    with pytest.raises(ValueError):
        _ensure_http_scheme("file:///etc/passwd")
    assert _ensure_http_scheme("https://example.com/x") == "https://example.com/x"


# ---------------------------------------------------------------------------------------------
# F42 / F43 -- browser.py resource lifecycle and shared header dict
# ---------------------------------------------------------------------------------------------


def test_f43_headers_is_not_the_same_object_as_basic_headers():
    from pyutilz.web import browser

    assert browser.headers is not browser.basic_headers
    original = dict(browser.basic_headers)
    saved = dict(browser.headers)
    try:
        browser.headers["authorization"] = "Bearer leak"
        assert "authorization" not in browser.basic_headers
        assert browser.basic_headers == original
    finally:
        browser.headers.clear()
        browser.headers.update(saved)


def test_f42_window_already_closed_quits_the_driver(monkeypatch):
    """`browser = None` on the session-refresh failure path orphaned a chromedriver process per
    restart; the handle must be quit via close_browser() instead."""
    from pyutilz.web import browser as browsermod

    dead = MagicMock()
    dead.refresh.side_effect = Exception("window was already closed")
    fresh = MagicMock()

    monkeypatch.setattr(browsermod, "browser", dead, raising=False)
    monkeypatch.setattr(browsermod, "home_page", "http://example.com", raising=False)
    monkeypatch.setattr(browsermod, "start_selenium", lambda *a, **kw: fresh, raising=False)
    # Stop the function right after the restart branch under test -- everything past it needs a
    # real page.
    monkeypatch.setattr(browsermod, "browser_get", MagicMock(side_effect=Exception("stop here")), raising=False)

    try:
        browsermod.LoginAndGetCookies("user", "pwd")
    except Exception:
        pass
    dead.quit.assert_called_once()


# ---------------------------------------------------------------------------------------------
# F44 -- get_external_ip() must not shuffle the shared IP_PROVIDERS global in place
# ---------------------------------------------------------------------------------------------


def test_f44_ip_providers_global_is_not_permuted(monkeypatch):
    providers = ["https://a.example", "https://b.example", "https://c.example", "https://d.example", "https://e.example"]
    monkeypatch.setattr(webmod, "IP_PROVIDERS", providers, raising=False)
    before = list(providers)

    resp = Mock()
    resp.status = http.HTTPStatus.OK
    resp.read.return_value = b"1.2.3.4"
    with patch.object(ipinfomod, "_direct_urlopen", return_value=resp):
        for _ in range(30):
            assert webmod.get_external_ip() == "1.2.3.4"
    assert providers == before
    assert webmod.IP_PROVIDERS is providers


# ---------------------------------------------------------------------------------------------
# F45 -- session helpers: default timeout + the resolved port offset is reachable
# ---------------------------------------------------------------------------------------------


class _StubProvider:
    """Minimal ProxyProvider stand-in: no network, no environment, no health-tracker state."""

    def __init__(self):
        self.picked = 0

    def pick_port(self):
        self.picked += 1
        return 7

    def proxy_url(self, port_offset=None):
        return "http://u:p@host:%d" % (10000 + (port_offset if port_offset is not None else 0))

    def proxies(self, port_offset=None):
        url = self.proxy_url(port_offset)
        return {"http": url, "https": url}


def test_f45_requests_session_exposes_the_auto_picked_offset():
    """With port_offset=None the port was picked inside provider.proxies() and never surfaced, so
    no outcome could be attributed to it via report_error()/report_success()."""
    from pyutilz.web.proxy import requests_session

    provider = _StubProvider()
    with requests_session(provider) as s:
        assert s.proxy_port_offset == 7
        assert ":10007" in s.proxies["http"]
    assert provider.picked == 1


def test_f45_requests_session_injects_a_default_timeout():
    import requests as _requests

    from pyutilz.web.proxy import requests_session

    calls = []
    # Patched BEFORE the session is built: requests_session wraps the session's bound .request,
    # which resolves to whatever Session.request is at construction time.
    with patch.object(_requests.Session, "request", lambda self, method, url, **kw: calls.append(kw.get("timeout"))):
        with requests_session(_StubProvider(), port_offset=3, timeout=9) as s:
            s.request("GET", "http://example.invalid/")
            s.request("GET", "http://example.invalid/", timeout=2)
    # Default injected on the first call; an explicit per-call timeout still wins on the second.
    assert calls == [9, 2]


def test_f45_requests_session_timeout_none_restores_blocking_behaviour():
    import requests as _requests

    from pyutilz.web.proxy import requests_session

    calls = []
    with patch.object(_requests.Session, "request", lambda self, method, url, **kw: calls.append(kw.get("timeout"))):
        with requests_session(_StubProvider(), port_offset=3, timeout=None) as s:
            s.request("GET", "http://example.invalid/")
    assert calls == [None]


def test_f45_curl_session_passes_timeout_and_exposes_offset():
    provider = _StubProvider()
    made = {}

    class _Sess:
        def __init__(self, **kwargs):
            made.update(kwargs)
            self.closed = False

        def close(self):
            self.closed = True

    mock_cr = MagicMock()
    mock_cr.Session = _Sess
    with patch.dict(sys.modules, {"curl_cffi": MagicMock(requests=mock_cr), "curl_cffi.requests": mock_cr}):
        from pyutilz.web.proxy import curl_session

        with curl_session(provider, timeout=15) as s:
            assert s.proxy_port_offset == 7
            assert made["timeout"] == 15
            assert made["proxy"] == "http://u:p@host:10007"
        assert s.closed is True


# ---------------------------------------------------------------------------------------------
# F47 -- get_from_s3_or_cache() must not busy-loop when the archive does not yield the object
# ---------------------------------------------------------------------------------------------


def test_f47_unpack_that_does_not_produce_the_object_raises(tmp_path, monkeypatch):
    """The archive is extracted into temp_dir, which need not contain local_object_path; nothing
    checked, so the enclosing `while not exists(...)` loop re-downloaded/unpacked/deleted forever
    against S3 with no backoff."""
    import shutil

    from pyutilz.cloud import cloud

    monkeypatch.setattr(cloud, "S3_BUCKET_NAME", "bucket", raising=False)
    monkeypatch.setattr(cloud, "sleep", lambda *a, **kw: None, raising=False)
    monkeypatch.setattr(cloud, "s3", MagicMock(), raising=False)

    local = tmp_path / "model.bin"
    temp_dir = tmp_path / "unpacked"
    temp_dir.mkdir()
    zip_path = str(local) + ".zip"
    with open(zip_path, "wb") as fh:
        fh.write(b"not really a zip, but unpack_archive is stubbed")

    unpacks = []

    def fake_unpack(archive, target):
        unpacks.append(archive)
        # Extracts somewhere that is NOT local_object_path.
        with open(os.path.join(target, "inner.bin"), "wb") as fh:
            fh.write(b"payload")

    monkeypatch.setattr(shutil, "unpack_archive", fake_unpack)

    with pytest.raises(FileNotFoundError):
        cloud.get_from_s3_or_cache(str(local), "objects/model.bin.zip", str(temp_dir))
    assert len(unpacks) == 1
    assert not local.exists()

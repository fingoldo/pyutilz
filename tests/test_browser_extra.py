"""Regression tests for pyutilz.web.browser (2026-07-21 audit fixes)."""

from unittest.mock import MagicMock, call

import pytest

pytest.importorskip("selenium")
pytest.importorskip("undetected_chromedriver")

from pyutilz.web import browser as browser_mod


def test_close_browser_calls_quit_not_close(monkeypatch):
    fake_browser = MagicMock()
    monkeypatch.setattr(browser_mod, "browser", fake_browser)

    browser_mod.close_browser()

    fake_browser.quit.assert_called_once()
    fake_browser.close.assert_not_called()
    assert browser_mod.browser is None


def test_browser_get_actually_retries_on_transient_error(monkeypatch):
    fake_browser = MagicMock()
    # First call raises the transient "loading status" error; second call (the retry) succeeds.
    fake_browser.get.side_effect = [Exception("cannot determine loading status"), None]
    monkeypatch.setattr(browser_mod, "browser", fake_browser)
    monkeypatch.setattr(browser_mod, "sleep", lambda *_: None)

    browser_mod.browser_get("https://example.com")

    assert fake_browser.get.call_count == 2
    fake_browser.get.assert_has_calls([call("https://example.com"), call("https://example.com")])


def test_browser_get_reraises_non_transient_errors(monkeypatch):
    fake_browser = MagicMock()
    fake_browser.get.side_effect = RuntimeError("boom")
    monkeypatch.setattr(browser_mod, "browser", fake_browser)

    with pytest.raises(RuntimeError, match="boom"):
        browser_mod.browser_get("https://example.com")
    assert fake_browser.get.call_count == 1


def test_browser_get_raises_when_browser_not_initialized(monkeypatch):
    monkeypatch.setattr(browser_mod, "browser", None)
    with pytest.raises(ValueError, match="not initialized"):
        browser_mod.browser_get("https://example.com")


def test_ensure_session_is_valid_none_interval_forces_update(monkeypatch):
    from datetime import datetime, timezone

    monkeypatch.setattr(browser_mod, "last_session_updated_at", datetime.now(timezone.utc))
    called = {}
    monkeypatch.setattr(browser_mod, "LoginAndGetCookies", lambda: called.setdefault("called", True))

    # Previously: TypeError, '>=' not supported between 'float' and 'NoneType'.
    browser_mod.ensure_session_is_valid(interval_minutes=None)

    assert called.get("called") is True


def test_ensure_session_is_valid_none_interval_first_call(monkeypatch):
    monkeypatch.setattr(browser_mod, "last_session_updated_at", None)
    called = {}
    monkeypatch.setattr(browser_mod, "LoginAndGetCookies", lambda: called.setdefault("called", True))
    browser_mod.ensure_session_is_valid(interval_minutes=None)
    assert called.get("called") is True


class TestFindElementNoLegacyFallback:
    """Selenium 4.0+ (this project's own floor) removed find_element_by_xpath/by_name/
    by_tag_name entirely -- the old except-fallback called a nonexistent method, masking the
    real NoSuchElementException with an unrelated AttributeError."""

    def test_find_element_by_xpath_calls_find_element_once(self):
        fake_browser = MagicMock()
        browser_mod.find_element_by_xpath(fake_browser, "//div")
        fake_browser.find_element.assert_called_once()
        fake_browser.find_element_by_xpath.assert_not_called()

    def test_find_element_by_xpath_propagates_real_exception(self):
        fake_browser = MagicMock()
        fake_browser.find_element.side_effect = RuntimeError("no such element")
        with pytest.raises(RuntimeError, match="no such element"):
            browser_mod.find_element_by_xpath(fake_browser, "//div")

    def test_find_element_by_name_calls_find_element_once(self):
        fake_browser = MagicMock()
        browser_mod.find_element_by_name(fake_browser, "email")
        fake_browser.find_element.assert_called_once()
        fake_browser.find_element_by_name.assert_not_called()

    def test_find_element_by_tag_name_calls_find_element_once(self):
        fake_browser = MagicMock()
        browser_mod.find_element_by_tag_name(fake_browser, "div")
        fake_browser.find_element.assert_called_once()
        fake_browser.find_element_by_tag_name.assert_not_called()


class TestLoginAndGetCookiesRestartRetry:
    """restart_on_no_cookie=True previously (a) silently dropped seconds_to_sleep_on_error/
    restart_on_no_cookie on every recursive retry level, and (b) had no depth limit."""

    def _make_missing_cookie_browser(self):
        fake_browser = MagicMock()
        fake_browser.title = "Sign-In"
        fake_browser.get_cookie.return_value = None  # required cookie never present
        return fake_browser

    def test_restart_retry_preserves_caller_overrides_and_eventually_raises(self, monkeypatch):
        monkeypatch.setattr(browser_mod, "browser", self._make_missing_cookie_browser())
        monkeypatch.setattr(browser_mod, "home_page", "https://example.com")
        monkeypatch.setattr(browser_mod, "target", "example.com")
        monkeypatch.setattr(browser_mod, "logout_signs", ())  # skip the login flow entirely
        monkeypatch.setattr(browser_mod, "successful_login_signs", ("Sign-In",))
        monkeypatch.setattr(browser_mod, "required_cookies", ("session_id",))
        monkeypatch.setattr(browser_mod, "browser_get", lambda *_: None)
        monkeypatch.setattr(browser_mod, "sleep", lambda *_: None)
        monkeypatch.setattr(browser_mod.pythonlib, "imitate_delay", lambda *a, **k: None)
        monkeypatch.setattr(browser_mod, "start_selenium", lambda: self._make_missing_cookie_browser())

        with pytest.raises(RuntimeError, match="exceeded 3 restart attempts"):
            browser_mod.LoginAndGetCookies(
                seconds_to_sleep_on_error=0,
                restart_on_no_cookie=True,
                _max_restart_attempts=3,
            )

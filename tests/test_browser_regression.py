"""Regression tests for pyutilz.web.browser.

Covers the bool-contract bug where LoginAndGetCookies() had bare `return`
statements on error paths, implicitly returning None instead of False.
"""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("selenium")
pytest.importorskip("undetected_chromedriver")

from pyutilz.web import browser as browser_mod


def test_login_and_get_cookies_returns_false_not_none(monkeypatch):
    """When browser_get() raises a generic (non-'not reachable') exception,
    LoginAndGetCookies() must return exactly False, not None."""

    # Skip the "start selenium" branch by pre-seeding a fake browser instance.
    monkeypatch.setattr(browser_mod, "browser", MagicMock())
    monkeypatch.setattr(browser_mod, "home_page", "https://example.com")
    monkeypatch.setattr(browser_mod, "target", "example.com")

    def _raise(*args, **kwargs):
        raise RuntimeError("some unrelated failure")

    monkeypatch.setattr(browser_mod, "browser_get", _raise)

    result = browser_mod.LoginAndGetCookies()

    assert result is False
    assert isinstance(result, bool)

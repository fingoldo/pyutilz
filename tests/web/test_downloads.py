"""Behaviour of ``pyutilz.web.web.downloads._check_redirect_chain``: a chain that stays within the redirect limit is accepted,
a longer one is refused, and a request that started on https is refused when any hop drops to http. Responses are simple
stand-ins carrying a history and a URL, so no request is made."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pyutilz.web.web import downloads


def _resp(url, history=()):
    return SimpleNamespace(url=url, history=[SimpleNamespace(url=h) for h in history])


def test_a_direct_response_is_accepted():
    assert downloads._check_redirect_chain(_resp("https://example.com/a"), "https://example.com/a") is None


def test_a_chain_longer_than_the_limit_is_refused():
    history = [f"https://example.com/{i}" for i in range(downloads.MAX_REDIRECTS + 1)]
    with pytest.raises(downloads.RedirectRefusedError, match="redirects"):
        downloads._check_redirect_chain(_resp("https://example.com/end", history), "https://example.com/start")


def test_a_drop_from_https_to_http_is_refused():
    with pytest.raises(downloads.RedirectRefusedError, match="https to http"):
        downloads._check_redirect_chain(_resp("http://example.com/plain", ["https://example.com/start"]), "https://example.com/start")


def test_a_response_without_a_url_is_not_a_redirect_chain():
    assert downloads._check_redirect_chain(SimpleNamespace(), "https://example.com/start") is None

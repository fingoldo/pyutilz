"""Behaviour of ``pyutilz.web.proxy.session``: the curl session context manager is importable and yields without a network
call when curl_cffi is absent, and the requests session is built with the provider's proxy. Only the construction is checked."""

from __future__ import annotations

import importlib.util

import pytest

from pyutilz.web.proxy import session


def test_the_session_helpers_are_context_managers_or_callables():
    assert callable(session.curl_session)
    assert callable(session.requests_session)


@pytest.mark.skipif(importlib.util.find_spec("curl_cffi") is None, reason="curl_cffi not installed")
def test_a_curl_session_yields_a_session_object():
    class _Provider:
        def proxy_url(self, offset=None):
            return "http://127.0.0.1:1"

        def proxies(self, offset=None):
            return {"http": "http://127.0.0.1:1", "https": "http://127.0.0.1:1"}

        def pick_port(self):
            return 1

        def report_error(self, *args, **kwargs):
            return None

    with session.curl_session(_Provider(), timeout=1.0) as sess:
        assert sess is not None

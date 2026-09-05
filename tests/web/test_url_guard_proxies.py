"""`urlopen_checked(proxies={})` makes "direct" a property of the CALL, not of the machine.

The caller this was added for downloads job attachments and documents them as going direct, no
proxy and no credential. Its residential pool is metered by the gigabyte and the attachments are the
largest thing it fetches, so the route is a billing decision -- and until now the code could not
express it: `build_opener` installs a `ProxyHandler` that reads `http_proxy`/`https_proxy` from the
environment, so the guarantee held only as long as nobody exported one.

Asserted on the OPENER's handler chain rather than by making a request: the question is which
handlers urllib will use, and a live request would answer a different question (whether the network
works) while needing a network.
"""

from __future__ import annotations

import urllib.request
from unittest.mock import patch

import pytest

from pyutilz.web.url_guard import urlopen_checked


def _handlers_for(**kwargs) -> list:
    """The handler chain `urlopen_checked` builds, without opening anything."""
    captured: list = []

    class _Opener:
        def open(self, target, timeout=None):
            return None

    def _build(*handlers):
        captured.extend(handlers)
        return _Opener()

    with patch("urllib.request.build_opener", side_effect=_build):
        urlopen_checked("https://example.com/file.pdf", **kwargs)
    return captured


def _proxy_handlers(handlers) -> list:
    return [h for h in handlers if isinstance(h, urllib.request.ProxyHandler)]


class TestTheRouteIsTheCallersToDecide:
    def test_an_empty_mapping_installs_a_proxy_handler_that_proxies_nothing(self):
        proxies = _proxy_handlers(_handlers_for(proxies={}))
        assert len(proxies) == 1, "an explicit handler is what suppresses build_opener's own"
        assert proxies[0].proxies == {}

    def test_a_mapping_is_passed_through_verbatim(self):
        mapping = {"https": "http://127.0.0.1:8888"}
        assert _proxy_handlers(_handlers_for(proxies=mapping))[0].proxies == mapping

    def test_the_default_leaves_urllib_alone(self):
        """Unchanged behaviour for every existing caller: no handler of ours, so `build_opener`
        adds its own environment-reading one."""
        assert _proxy_handlers(_handlers_for()) == []

    def test_the_redirect_guard_is_still_in_the_chain(self):
        """The proxy argument must not cost the scheme check the function exists for."""
        for kwargs in ({}, {"proxies": {}}):
            names = [type(h).__name__ for h in _handlers_for(**kwargs)]
            assert "_CheckedRedirectHandler" in names, kwargs

    def test_the_proxy_handler_comes_first(self):
        """Order is load-bearing: `build_opener` only skips its default when it sees one supplied."""
        handlers = _handlers_for(proxies={})
        assert isinstance(handlers[0], urllib.request.ProxyHandler)


class TestTheSchemeCheckStillRunsFirst:
    def test_a_file_url_is_refused_before_any_opener_is_built(self):
        """Whatever the proxy setting, a `file:` URI must not turn a download into a local read."""
        from pyutilz.web.url_guard import UnsafeURLError

        with pytest.raises(UnsafeURLError):
            urlopen_checked("file:///etc/passwd", proxies={})

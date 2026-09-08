"""A mail.ru public folder is a web page, and the URL you would guess downloads that page.

`pyutilz.cloud.mailru` exists for one non-obvious fact: `https://cloud.mail.ru/public/<a>/<b>/file` does not
serve `file`. Every test here is offline - the dispatcher and folder endpoints are stubbed - because what is
worth pinning is the URL shape and the paging, not mail.ru's uptime.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from pyutilz.cloud import mailru

FOLDER = "https://cloud.mail.ru/public/oxLS/4ZnDN8UJw"


@pytest.fixture(autouse=True)
def _forget_the_cached_host() -> None:
    """`download_host` is process-cached, so a stub from one test would otherwise leak into the next."""
    mailru.download_host.cache_clear()


def _stub_json(monkeypatch: pytest.MonkeyPatch, responses: Dict[str, Dict[str, Any]]) -> List[str]:
    """Replace the module's one JSON GET, recording the URLs asked for."""
    asked: List[str] = []

    def _fake(url: str, timeout: int = 60) -> Dict[str, Any]:
        asked.append(url)
        for fragment, payload in responses.items():
            if fragment in url:
                return payload
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(mailru, "_get_json", _fake)
    return asked


def test_the_download_url_goes_through_the_dispatcher_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point: the published link is not the base URL the file hangs off."""
    _stub_json(monkeypatch, {"dispatcher": {"body": {"weblink_get": [{"url": "https://cloclo54.cloud.mail.ru/public/TOKEN/g/no/"}]}}})
    assert mailru.public_file_url(FOLDER, "data.tar.gz") == "https://cloclo54.cloud.mail.ru/public/TOKEN/g/no/oxLS/4ZnDN8UJw/data.tar.gz"


def test_the_host_is_resolved_once_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """A restore fetches many objects; each one paying a dispatcher round trip would be a needless request
    against someone else's service, and the value does not change mid-run."""
    asked = _stub_json(monkeypatch, {"dispatcher": {"body": {"weblink_get": [{"url": "https://host/x"}]}}})
    for _ in range(5):
        mailru.public_file_url(FOLDER, "a.tar.gz")
    assert len(asked) == 1


def test_a_dispatcher_with_no_host_raises_rather_than_building_a_broken_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silently producing `None/weblink/name` would fail much later, as a 404 nobody can place."""
    _stub_json(monkeypatch, {"dispatcher": {"body": {"weblink_get": []}}})
    with pytest.raises(RuntimeError, match="no weblink_get host"):
        mailru.public_file_url(FOLDER, "a.tar.gz")


def test_a_url_that_is_not_a_public_folder_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Callers dispatch on `is_public_folder_url`; handing this an ordinary URL is a bug at the call site."""
    assert not mailru.is_public_folder_url("https://bucket.s3.amazonaws.com/prefix")
    with pytest.raises(ValueError, match=r"not a mail\.ru public folder"):
        mailru.weblink_of("https://bucket.s3.amazonaws.com/prefix")


def test_listing_pages_through_the_whole_folder(monkeypatch: pytest.MonkeyPatch) -> None:
    """The API caps a page at 500 entries, so a folder of 600 files listed as 500 would read as complete -
    the exact shape of wrongness this is meant to answer ("is my mirror all there?")."""
    first = [{"name": f"f{i}", "type": "file"} for i in range(500)]
    second = [{"name": f"g{i}", "type": "file"} for i in range(100)]

    def _fake(url: str, timeout: int = 60) -> Dict[str, Any]:
        page = second if "offset=500" in url else first
        return {"body": {"list": page, "count": {"files": 600, "folders": 0}}}

    monkeypatch.setattr(mailru, "_get_json", _fake)
    entries = mailru.list_public_folder(FOLDER)
    assert len(entries) == 600
    assert entries[0]["name"] == "f0" and entries[-1]["name"] == "g99"


def test_listing_stops_on_an_empty_page_even_if_the_count_disagrees(monkeypatch: pytest.MonkeyPatch) -> None:
    """A count that overstates what the folder returns must not spin forever."""
    monkeypatch.setattr(mailru, "_get_json", lambda url, timeout=60: {"body": {"list": [], "count": {"files": 999, "folders": 0}}})
    assert mailru.list_public_folder(FOLDER) == []


def test_the_agent_header_is_set_because_the_cdn_refuses_requests_without_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Measured against the live CDN: no agent, 403. Pinned because it looks removable.

    Also pins that the request goes through `urlopen_checked` rather than a bare `urlopen` - the download
    redirects once to a signed CDN URL, and the guard is what validates that hop's scheme too.
    """
    captured: Dict[str, Any] = {}
    _stub_json(monkeypatch, {"dispatcher": {"body": {"weblink_get": [{"url": "https://host/x"}]}}})

    def _fake_urlopen(request: Any, timeout: int = 0) -> str:
        captured["headers"] = dict(request.headers)
        return "response"

    # The module that CALLS, not `urllib` itself: `mailru` binds `urlopen_checked` at import time, so
    # patching the guard's own module would leave this call on the real one.
    monkeypatch.setattr(mailru, "urlopen_checked", _fake_urlopen)
    assert mailru.open_public_file(FOLDER, "a.tar.gz") == "response"
    assert captured["headers"].get("User-agent") == mailru.USER_AGENT


def test_the_module_is_reachable_from_the_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """`from pyutilz.cloud import mailru` is the documented entry point, so the package must bind it."""
    from pyutilz import cloud

    assert cloud.mailru is mailru
    assert json  # the import above is used by the stubs' own payload shape

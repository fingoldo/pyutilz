"""A mail.ru public folder is a web page, and the URL you would guess downloads that page.

`pyutilz.cloud.mailru` exists for one non-obvious fact: `https://cloud.mail.ru/public/<a>/<b>/file` does not
serve `file`. Every test here is offline - the dispatcher and folder endpoints are stubbed - because what is
worth pinning is the URL shape and the paging, not mail.ru's uptime.
"""

from __future__ import annotations

from pyutilz.core import jsonlib
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

    def _fake(url: str, timeout: int = 60) -> Dict[str, Any]:  # unused-ok: fake keeps the real API signature (timeout accepted, nothing to wait on in-memory)
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

    def _fake(url: str, timeout: int = 60) -> Dict[str, Any]:  # unused-ok: fake keeps the real API signature (timeout accepted, nothing to wait on in-memory)
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

    def _fake_urlopen(request: Any, timeout: int = 0) -> str:  # unused-ok: fake keeps the real API signature (timeout accepted, nothing to wait on in-memory)
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
    assert jsonlib  # the import above is used by the stubs' own payload shape


class TestRequestsPinnedExactly:
    """Each survivor of the mutation sweep was a constant or an argument whose exact value the tests never read:
    the endpoints, the agent, the page size, the timeouts and the paging stop condition."""

    def test_the_dispatcher_and_folder_endpoints_are_the_documented_urls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        asked = _stub_json(monkeypatch, {"dispatcher": {"body": {"weblink_get": [{"url": "https://host/x"}]}}, "folder": {"body": {"list": []}}})
        mailru.download_host()
        mailru.list_public_folder(FOLDER)
        assert asked[0] == "https://cloud.mail.ru/api/v2/dispatcher"
        assert asked[1] == "https://cloud.mail.ru/api/v2/folder?weblink=oxLS/4ZnDN8UJw&limit=500&offset=0"

    def test_the_user_agent_is_the_exact_value_the_cdn_accepts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sent: List[Any] = []

        class _Body:
            def __enter__(self) -> "_Body":
                return self

            def __exit__(self, *_exc: Any) -> bool:
                return False

            def read(self) -> bytes:
                return b"{}"

        def _fake_urlopen(request: Any, timeout: int = 0) -> _Body:  # unused-ok: fake keeps the real API signature
            sent.append(request)
            return _Body()

        monkeypatch.setattr(mailru, "urlopen_checked", _fake_urlopen)
        mailru._get_json("https://example.test/")
        assert sent[0].get_header("User-agent") == "Mozilla/5.0"

    def test_the_folder_request_carries_the_documented_page_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        asked = _stub_json(monkeypatch, {"folder": {"body": {"list": [{"name": "a"}], "count": {"files": 1}}}})
        mailru.list_public_folder(FOLDER)
        assert "limit=500&" in asked[0] and mailru._PAGE_LIMIT == 500

    def test_the_default_timeouts_are_one_minute_for_json_and_ten_for_file_bodies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        timeouts: List[int] = []

        def _fake_get(url: str, timeout: int = 0) -> Dict[str, Any]:  # unused-ok: fake keeps the real API signature
            timeouts.append(timeout)
            return {"body": {"list": []}}

        monkeypatch.setattr(mailru, "_get_json", _fake_get)
        mailru.list_public_folder(FOLDER)
        assert timeouts == [60]

        seen: List[int] = []
        monkeypatch.setattr(mailru, "urlopen_checked", lambda request, timeout=0: seen.append(timeout))
        _stub_json(monkeypatch, {"dispatcher": {"body": {"weblink_get": [{"url": "https://host/x"}]}}})
        mailru.open_public_file(FOLDER, "a.tar.gz")
        assert seen == [600]

    def test_the_resolution_is_logged_at_debug_with_the_host(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        _stub_json(monkeypatch, {"dispatcher": {"body": {"weblink_get": [{"url": "https://cloclo54.cloud.mail.ru/x/"}]}}})
        with caplog.at_level("DEBUG", logger="pyutilz.cloud.mailru"):
            mailru.download_host()
        assert [rec.getMessage() for rec in caplog.records if rec.levelname == "DEBUG"] == [
            "mail.ru public-download host resolved to https://cloclo54.cloud.mail.ru/x"
        ]

    def test_a_public_link_with_the_prefix_twice_keeps_everything_after_the_first_occurrence(self) -> None:
        """Split on the first prefix only: the weblink is the text after it, even when the path repeats the prefix."""
        assert mailru.weblink_of("https://cloud.mail.ru/public/a/b/cloud.mail.ru/public/c") == "a/b/cloud.mail.ru/public/c"

    def test_a_trailing_slash_on_the_folder_url_does_not_end_up_in_the_weblink(self) -> None:
        assert mailru.weblink_of("https://cloud.mail.ru/public/oxLS/4ZnDN8UJw/") == "oxLS/4ZnDN8UJw"


class TestPagingStopCondition:
    def test_a_folder_whose_count_is_exactly_one_page_is_listed_with_a_single_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Files and folders are summed for the stop condition; when the page is exactly the total, no second request."""
        asked = _stub_json(monkeypatch, {"folder": {"body": {"list": [{"name": f"f{i}"} for i in range(3)], "count": {"files": 3}}}})
        assert len(mailru.list_public_folder(FOLDER)) == 3
        assert len(asked) == 1

    def test_the_folder_total_is_files_plus_folders_not_their_difference(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Five files and two folders are seven entries: the listing must keep paging after five, not stop."""
        calls: List[int] = []

        def _fake(url: str, timeout: int = 60) -> Dict[str, Any]:  # unused-ok: fake keeps the real API signature
            calls.append(1)
            page = [{"name": f"f{i}"} for i in range(5)] if len(calls) == 1 else [{"name": "g0"}, {"name": "g1"}]
            return {"body": {"list": page, "count": {"files": 5, "folders": 2}}}

        monkeypatch.setattr(mailru, "_get_json", _fake)
        assert len(mailru.list_public_folder(FOLDER)) == 7
        assert len(calls) == 2

    def test_a_json_call_without_a_timeout_argument_uses_one_minute(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: List[int] = []

        class _Body:
            def __enter__(self) -> "_Body":
                return self

            def __exit__(self, *_exc: Any) -> bool:
                return False

            def read(self) -> bytes:
                return b"{}"

        monkeypatch.setattr(mailru, "urlopen_checked", lambda request, timeout=0: seen.append(timeout) or _Body())
        mailru._get_json("https://example.test/")
        assert seen == [60]

    def test_the_host_cache_holds_one_entry_for_the_process(self) -> None:
        """One host per process: the cache is bounded at a single entry, so a rotated host replaces the old one."""
        assert mailru.download_host.cache_info().maxsize == 1

    def test_a_count_with_only_folders_still_stops_at_the_folder_total(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing files count is zero, so three folders and three entries stop after one request."""
        asked = _stub_json(monkeypatch, {"folder": {"body": {"list": [{"name": f"d{i}"} for i in range(3)], "count": {"folders": 3}}}})
        assert len(mailru.list_public_folder(FOLDER)) == 3
        assert len(asked) == 1

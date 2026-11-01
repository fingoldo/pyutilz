"""Tests for pyutilz.web.cached_client.CachedHttpClient.

Mocks pyutilz.web.cached_client.urlopen_checked directly (not the stdlib urllib) so every test
runs offline and deterministically -- no real network access, no flakiness from a live service.
"""

from __future__ import annotations

import hashlib
import io
import logging
import pickle
import urllib.error
from pyutilz.core import jsonlib
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from pyutilz.web.cached_client import CachedHttpClient


def _fake_response(body: bytes):
    @contextmanager
    def _open(*_args, **_kwargs):
        yield io.BytesIO(body)

    return _open


class TestGetJson:
    def test_cache_miss_fetches_and_caches(self, tmp_path):
        client = CachedHttpClient(cache_dir=tmp_path)
        with patch("pyutilz.web.cached_client.urlopen_checked", _fake_response(b'{"a": 1}')):
            result = client.get_json("https://example.org/x", "tag")
        assert result == {"a": 1}
        cached = list(tmp_path.rglob("*.json"))
        assert len(cached) == 1
        assert jsonlib.loads(cached[0].read_text())["data"] == {"a": 1}

    def test_cache_hit_never_calls_urlopen(self, tmp_path):
        client = CachedHttpClient(cache_dir=tmp_path)
        with patch("pyutilz.web.cached_client.urlopen_checked", _fake_response(b'{"a": 1}')):
            client.get_json("https://example.org/x", "tag")
        with patch("pyutilz.web.cached_client.urlopen_checked") as mocked:
            result = client.get_json("https://example.org/x", "tag")
        mocked.assert_not_called()
        assert result == {"a": 1}

    def test_negative_result_cached_by_default(self, tmp_path):
        """A 404 (permanent failure) is cached too, so re-running doesn't re-pay the round trip."""
        client = CachedHttpClient(cache_dir=tmp_path)

        def _raise(*_args, **_kwargs):
            raise urllib.error.HTTPError("https://example.org/x", 404, "not found", None, None)

        with patch("pyutilz.web.cached_client.urlopen_checked", side_effect=_raise):
            result = client.get_json("https://example.org/x", "tag")
        assert result is None
        cached = list(tmp_path.rglob("*.json"))
        assert len(cached) == 1
        with patch("pyutilz.web.cached_client.urlopen_checked") as mocked:
            result2 = client.get_json("https://example.org/x", "tag")
        mocked.assert_not_called()
        assert result2 is None

    def test_retries_on_429_then_succeeds(self, tmp_path):
        client = CachedHttpClient(cache_dir=tmp_path)
        calls = {"n": 0}

        @contextmanager
        def _open(*_args, **_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise urllib.error.HTTPError("https://example.org/x", 429, "rate limited", None, None)
            yield io.BytesIO(b'{"ok": true}')

        with patch("pyutilz.web.cached_client.urlopen_checked", _open), patch("time.sleep"):
            result = client.get_json("https://example.org/x", "tag", retries=3)
        assert result == {"ok": True}
        assert calls["n"] == 2

    def test_offline_mode_skips_network_returns_none_on_miss(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MY_OFFLINE", "1")
        client = CachedHttpClient(cache_dir=tmp_path, offline_env_var="MY_OFFLINE")
        with patch("pyutilz.web.cached_client.urlopen_checked") as mocked:
            result = client.get_json("https://example.org/x", "tag")
        mocked.assert_not_called()
        assert result is None

    def test_offline_mode_still_serves_existing_cache(self, tmp_path, monkeypatch):
        client = CachedHttpClient(cache_dir=tmp_path, offline_env_var="MY_OFFLINE")
        with patch("pyutilz.web.cached_client.urlopen_checked", _fake_response(b'{"a": 1}')):
            client.get_json("https://example.org/x", "tag")
        monkeypatch.setenv("MY_OFFLINE", "1")
        with patch("pyutilz.web.cached_client.urlopen_checked") as mocked:
            result = client.get_json("https://example.org/x", "tag")
        mocked.assert_not_called()
        assert result == {"a": 1}


class TestGetText:
    def test_cache_miss_fetches_and_caches(self, tmp_path):
        client = CachedHttpClient(cache_dir=tmp_path)
        with patch("pyutilz.web.cached_client.urlopen_checked", _fake_response(b"<xml>hi</xml>")):
            result = client.get_text("https://example.org/x", "tag")
        assert result == "<xml>hi</xml>"

    def test_failure_not_cached_by_default(self, tmp_path):
        """Unlike get_json, a text-fetch failure is NOT cached by default -- an outage must not
        look like a durable answer."""
        client = CachedHttpClient(cache_dir=tmp_path)

        def _raise(*_args, **_kwargs):
            raise urllib.error.HTTPError("https://example.org/x", 500, "boom", None, None)

        with patch("pyutilz.web.cached_client.urlopen_checked", side_effect=_raise), patch("time.sleep"):
            result = client.get_text("https://example.org/x", "tag", retries=1)
        assert result is None
        assert list(tmp_path.rglob("*.json")) == []


class TestThrottle:
    def test_per_host_throttle_delays_second_call_to_same_host(self, tmp_path):
        # 60 s, not 50 ms: a slow runner spent more than 50 ms between stamping _last_call and _throttle
        # reading the clock, so no wait was left and nothing slept. The fake sleep then lets the interval
        # "elapse" by forgetting the host, since _throttle loops until a real wait would have passed.
        client = CachedHttpClient(cache_dir=tmp_path, min_interval=60.0)
        client._last_call["example.org"] = __import__("time").monotonic()
        slept = []

        def fake_sleep(seconds):
            slept.append(seconds)
            client._last_call.pop("example.org", None)

        with patch("time.sleep", side_effect=fake_sleep):
            client._throttle("https://example.org/y")
        assert len(slept) == 1 and 59.0 < slept[0] <= 60.0

    def test_different_hosts_do_not_throttle_each_other(self, tmp_path):
        client = CachedHttpClient(cache_dir=tmp_path, min_interval=10.0)
        client._last_call["a.example.org"] = __import__("time").monotonic()
        with patch("time.sleep") as mocked:
            client._throttle("https://b.example.org/y")
        mocked.assert_not_called()


def test_corrupt_cache_entry_is_refetched_not_trusted(tmp_path):
    client = CachedHttpClient(cache_dir=tmp_path)
    path = client._cache_path("https://example.org/x", "tag")
    path.parent.mkdir(parents=True)
    path.write_text("not valid json{{{", encoding="utf-8")
    with patch("pyutilz.web.cached_client.urlopen_checked", _fake_response(b'{"a": 1}')):
        result = client.get_json("https://example.org/x", "tag")
    assert result == {"a": 1}


def test_instance_survives_pickle_roundtrip(tmp_path):
    """The threading.Lock backing the per-host throttle must not break pickling -- a caller that
    caches or ships a CachedHttpClient instance (joblib fan-out, a process pool) would otherwise
    hit a TypeError the first time, far from wherever the lock was created."""
    client = CachedHttpClient(cache_dir=tmp_path, min_interval=0.5)
    client._last_call["example.org"] = 123.0
    restored = pickle.loads(pickle.dumps(client))  # deserialize-ok: round-trips a pickle this test just produced in-process; the bytes never leave the test, no untrusted input
    assert restored.cache_dir == tmp_path
    assert restored.min_interval == 0.5
    assert restored._last_call == {}  # per-host timestamps are meaningless across a pickle boundary
    with patch("pyutilz.web.cached_client.urlopen_checked", _fake_response(b'{"a": 1}')):
        assert restored.get_json("https://example.org/z", "tag") == {"a": 1}


def test_unsafe_scheme_is_rejected_before_any_socket(tmp_path):
    """The url_guard integration: a file:// URL raises UnsafeURLError rather than being silently
    swallowed into a "no result" -- an unsafe-scheme call is a caller bug, not a transient failure,
    and must not be indistinguishable from a normal cache miss."""
    from pyutilz.web.url_guard import UnsafeURLError

    client = CachedHttpClient(cache_dir=tmp_path)
    with pytest.raises(UnsafeURLError):
        client.get_json("file:///etc/passwd", "tag")


URL = "https://example.org/x"


class _FakeResp:
    """A response body served in pieces: ``read(n)`` returns at most ``step`` bytes when a step is set."""

    def __init__(self, body: bytes, headers=None, step=None):
        self._buf = io.BytesIO(body)
        self.headers = headers
        self._step = step
        self.read_sizes = []

    def read(self, n=-1):
        self.read_sizes.append(n)
        if self._step is not None and (n is None or n < 0 or n > self._step):
            n = self._step
        return self._buf.read(n)


def _open_all(*items):
    """Stands in for ``urlopen_checked``: serves ``items`` in order (the last one repeats) and records each call."""
    calls = {"requests": [], "kwargs": []}

    def _open(req, **kwargs):
        calls["requests"].append(req)
        calls["kwargs"].append(kwargs)
        item = items[min(len(calls["requests"]), len(items)) - 1]
        if isinstance(item, BaseException):
            raise item

        @contextmanager
        def _cm():
            yield item

        return _cm()

    return _open, calls


def _http_error(code, retry_after=None):
    headers = {"Retry-After": retry_after} if retry_after is not None else None
    return urllib.error.HTTPError(URL, code, "msg", headers, None)


class TestRetryPolicy:
    @pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
    def test_each_retryable_status_is_retried(self, tmp_path, code):
        opener, calls = _open_all(_http_error(code), _FakeResp(b'{"ok": 1}'))
        client = CachedHttpClient(cache_dir=tmp_path, min_interval=0)
        with patch("pyutilz.web.cached_client.urlopen_checked", opener), patch("time.sleep"):
            assert client.get_json(URL, "tag") == {"ok": 1}
        assert len(calls["requests"]) == 2

    @pytest.mark.parametrize("code", [400, 404, 501])
    def test_other_statuses_are_not_retried(self, tmp_path, code):
        opener, calls = _open_all(_http_error(code), _FakeResp(b'{"ok": 1}'))
        client = CachedHttpClient(cache_dir=tmp_path, min_interval=0)
        with patch("pyutilz.web.cached_client.urlopen_checked", opener), patch("time.sleep") as sleep:
            assert client.get_json(URL, "tag") is None
        assert len(calls["requests"]) == 1
        sleep.assert_not_called()

    def test_the_attempts_stop_at_the_retry_count_with_a_backoff_between_them(self, tmp_path):
        opener, calls = _open_all(_http_error(503))
        client = CachedHttpClient(cache_dir=tmp_path, min_interval=0)
        with patch("pyutilz.web.cached_client.urlopen_checked", opener), patch("time.sleep") as sleep:
            assert client.get_json(URL, "tag", retries=3) is None
        assert len(calls["requests"]) == 3
        assert [c.args[0] for c in sleep.call_args_list] == [1.0, 2.0]

    def test_the_default_retry_count_is_three_for_json_and_text(self, tmp_path):
        opener, calls = _open_all(_http_error(503))
        client = CachedHttpClient(cache_dir=tmp_path, min_interval=0)
        with patch("pyutilz.web.cached_client.urlopen_checked", opener), patch("time.sleep"):
            client.get_json(URL, "tag")
            client.get_text(URL, "tag")
        assert len(calls["requests"]) == 6

    def test_the_default_timeout_is_thirty_seconds(self, tmp_path):
        opener, calls = _open_all(_FakeResp(b"{}"))
        client = CachedHttpClient(cache_dir=tmp_path, min_interval=0)
        with patch("pyutilz.web.cached_client.urlopen_checked", opener):
            client.get_json(URL, "tag")
            client.get_text(URL, "other")
        assert [kw["timeout"] for kw in calls["kwargs"]] == [30, 30]

    def test_a_retry_after_header_sets_the_delay(self, tmp_path):
        opener, _ = _open_all(_http_error(429, "7"), _FakeResp(b'{"ok": 1}'))
        client = CachedHttpClient(cache_dir=tmp_path, min_interval=0)
        with patch("pyutilz.web.cached_client.urlopen_checked", opener), patch("time.sleep") as sleep:
            assert client.get_json(URL, "tag") == {"ok": 1}
        assert [c.args[0] for c in sleep.call_args_list] == [7.0]

    def test_a_retry_after_at_the_cap_is_honoured(self, tmp_path):
        opener, _ = _open_all(_http_error(429, "120"), _FakeResp(b'{"ok": 1}'))
        client = CachedHttpClient(cache_dir=tmp_path, min_interval=0)
        with patch("pyutilz.web.cached_client.urlopen_checked", opener), patch("time.sleep") as sleep:
            client.get_json(URL, "tag")
        assert [c.args[0] for c in sleep.call_args_list] == [120.0]

    def test_a_retry_after_over_the_cap_falls_back_to_local_backoff_and_says_so(self, tmp_path, caplog):
        opener, _ = _open_all(_http_error(429, "121"), _FakeResp(b'{"ok": 1}'))
        client = CachedHttpClient(cache_dir=tmp_path, min_interval=0)
        with patch("pyutilz.web.cached_client.urlopen_checked", opener), patch("time.sleep") as sleep:
            with caplog.at_level(logging.WARNING):
                client.get_json(URL, "tag")
        assert [c.args[0] for c in sleep.call_args_list] == [1.0]
        assert "cached_client: server asked for Retry-After=121s (> cap 120.0s) on https://example.org/x; using local backoff 1s instead" in [
            r.getMessage() for r in caplog.records
        ]

    def test_a_retry_after_that_is_not_a_number_is_ignored(self, tmp_path):
        opener, _ = _open_all(_http_error(429, "soon"), _FakeResp(b'{"ok": 1}'))
        client = CachedHttpClient(cache_dir=tmp_path, min_interval=0)
        with patch("pyutilz.web.cached_client.urlopen_checked", opener), patch("time.sleep") as sleep:
            client.get_json(URL, "tag")
        assert [c.args[0] for c in sleep.call_args_list] == [1.0]

    def test_a_network_error_is_retried_with_a_power_of_two_backoff(self, tmp_path):
        opener, calls = _open_all(urllib.error.URLError("boom"))
        client = CachedHttpClient(cache_dir=tmp_path, min_interval=0)
        with patch("pyutilz.web.cached_client.urlopen_checked", opener), patch("time.sleep") as sleep:
            assert client.get_json(URL, "tag", retries=3) is None
        assert len(calls["requests"]) == 3
        assert [c.args[0] for c in sleep.call_args_list] == [1, 2]

    def test_a_network_error_that_recovers_returns_the_body(self, tmp_path):
        opener, calls = _open_all(TimeoutError("slow"), _FakeResp(b'{"ok": 1}'))
        client = CachedHttpClient(cache_dir=tmp_path, min_interval=0)
        with patch("pyutilz.web.cached_client.urlopen_checked", opener), patch("time.sleep"):
            assert client.get_json(URL, "tag") == {"ok": 1}
        assert len(calls["requests"]) == 2

    def test_a_permanent_http_failure_is_logged_at_debug(self, tmp_path, caplog):
        opener, _ = _open_all(_http_error(404))
        client = CachedHttpClient(cache_dir=tmp_path, min_interval=0)
        with patch("pyutilz.web.cached_client.urlopen_checked", opener), caplog.at_level(logging.DEBUG):
            client.get_json(URL, "tag")
        assert "cached_client: HTTP 404 for https://example.org/x" in [r.getMessage() for r in caplog.records]

    def test_a_network_error_given_up_on_is_logged_at_debug(self, tmp_path, caplog):
        opener, _ = _open_all(TimeoutError("boom"))
        client = CachedHttpClient(cache_dir=tmp_path, min_interval=0)
        with patch("pyutilz.web.cached_client.urlopen_checked", opener), patch("time.sleep"), caplog.at_level(logging.DEBUG):
            client.get_json(URL, "tag", retries=1)
        assert "cached_client: boom for https://example.org/x" in [r.getMessage() for r in caplog.records]


class TestThrottleAndDefaults:
    def test_the_throttle_waits_before_a_second_request_to_the_same_host(self, tmp_path):
        opener, _ = _open_all(_FakeResp(b"a"), _FakeResp(b"b"))
        client = CachedHttpClient(cache_dir=tmp_path, min_interval=0.5)
        with patch("pyutilz.web.cached_client.urlopen_checked", opener), patch("time.sleep") as sleep:
            client.get_text("https://example.org/a", "tag")
            client.get_text("https://example.org/b", "tag")
        waits = [c.args[0] for c in sleep.call_args_list]
        assert waits
        assert 0 < waits[0] <= 0.5

    def test_the_default_minimum_interval_is_twelve_hundredths_of_a_second(self, tmp_path):
        assert CachedHttpClient(cache_dir=tmp_path).min_interval == 0.12

    def test_the_default_user_agent_is_sent(self, tmp_path):
        opener, calls = _open_all(_FakeResp(b"{}"))
        with patch("pyutilz.web.cached_client.urlopen_checked", opener):
            CachedHttpClient(cache_dir=tmp_path, min_interval=0).get_json(URL, "tag")
        assert calls["requests"][0].get_header("User-agent") == "pyutilz-cached-client/0.1"

    def test_json_requests_ask_for_json(self, tmp_path):
        opener, calls = _open_all(_FakeResp(b"{}"))
        with patch("pyutilz.web.cached_client.urlopen_checked", opener):
            CachedHttpClient(cache_dir=tmp_path, min_interval=0).get_json(URL, "tag")
        assert calls["requests"][0].get_header("Accept") == "application/json"

    def test_the_client_methods_take_their_options_by_keyword_only(self, tmp_path):
        client = CachedHttpClient(cache_dir=tmp_path, min_interval=0)
        with pytest.raises(TypeError):
            client.get_json(URL, "tag", {"x": "y"})
        with pytest.raises(TypeError):
            client.get_text(URL, "tag", {"x": "y"})


class TestCacheTagsAndNames:
    @pytest.mark.parametrize("tag", ["", ".", "..", "a/b", "../x", "a b", "/abs"])
    def test_an_unsafe_tag_is_refused_before_any_request(self, tmp_path, tag):
        opener, calls = _open_all(_FakeResp(b"{}"))
        with patch("pyutilz.web.cached_client.urlopen_checked", opener):
            with pytest.raises(ValueError, match="unsafe cache tag"):
                CachedHttpClient(cache_dir=tmp_path).get_json(URL, tag)
        assert calls["requests"] == []

    def test_the_refusal_states_the_allowed_grammar(self, tmp_path):
        with pytest.raises(ValueError) as info:
            CachedHttpClient(cache_dir=tmp_path).get_json(URL, "a/b")
        assert "expected [A-Za-z0-9_.-]+ (it is used as a directory name under cache_dir)" in str(info.value)

    def test_a_dotted_tag_with_dashes_and_underscores_is_accepted(self, tmp_path):
        with patch("pyutilz.web.cached_client.urlopen_checked", _fake_response(b'{"a": 1}')):
            assert CachedHttpClient(cache_dir=tmp_path, min_interval=0).get_json(URL, "v1.2-x_y") == {"a": 1}

    def test_a_json_entry_is_named_by_the_url_digest_under_the_tag(self, tmp_path):
        with patch("pyutilz.web.cached_client.urlopen_checked", _fake_response(b'{"a": 1}')):
            CachedHttpClient(cache_dir=tmp_path, min_interval=0).get_json(URL, "tag")
        digest = hashlib.sha256(URL.encode("utf-8")).hexdigest()[:20]
        assert (tmp_path / "tag" / f"{digest}.json").exists()

    def test_a_text_entry_is_keyed_apart_from_a_json_entry(self, tmp_path):
        with patch("pyutilz.web.cached_client.urlopen_checked", _fake_response(b"plain")):
            CachedHttpClient(cache_dir=tmp_path, min_interval=0).get_text(URL, "tag")
        digest = hashlib.sha256(f"text:{URL}".encode()).hexdigest()[:20]
        assert (tmp_path / "tag" / f"{digest}.json").exists()

    def test_a_missing_cache_root_is_created_on_write(self, tmp_path):
        root = tmp_path / "missing" / "root"
        with patch("pyutilz.web.cached_client.urlopen_checked", _fake_response(b'{"a": 1}')):
            CachedHttpClient(cache_dir=root, min_interval=0).get_json(URL, "tag")
        assert len(list(root.rglob("*.json"))) == 1

    def test_a_json_entry_records_its_url_and_data(self, tmp_path):
        with patch("pyutilz.web.cached_client.urlopen_checked", _fake_response(b'{"a": 1}')):
            CachedHttpClient(cache_dir=tmp_path, min_interval=0).get_json(URL, "tag")
        entry = jsonlib.loads(next(tmp_path.rglob("*.json")).read_text(encoding="utf-8"))
        assert entry == {"url": URL, "data": {"a": 1}}

    def test_a_text_entry_records_its_url_and_text(self, tmp_path):
        with patch("pyutilz.web.cached_client.urlopen_checked", _fake_response(b"plain")):
            CachedHttpClient(cache_dir=tmp_path, min_interval=0).get_text(URL, "tag")
        entry = jsonlib.loads(next(tmp_path.rglob("*.json")).read_text(encoding="utf-8"))
        assert entry == {"url": URL, "text": "plain"}


class TestFailureCaching:
    def test_a_failed_json_fetch_is_not_cached_when_failures_are_not_cached(self, tmp_path):
        opener, _ = _open_all(_http_error(404))
        with patch("pyutilz.web.cached_client.urlopen_checked", opener):
            assert CachedHttpClient(cache_dir=tmp_path, min_interval=0).get_json(URL, "tag", cache_failures=False) is None
        assert list(tmp_path.rglob("*.json")) == []

    def test_a_successful_json_fetch_is_cached_even_when_failures_are_not_cached(self, tmp_path):
        with patch("pyutilz.web.cached_client.urlopen_checked", _fake_response(b'{"a": 1}')):
            CachedHttpClient(cache_dir=tmp_path, min_interval=0).get_json(URL, "tag", cache_failures=False)
        assert len(list(tmp_path.rglob("*.json"))) == 1

    def test_a_failed_text_fetch_is_not_cached_by_default(self, tmp_path):
        opener, _ = _open_all(_http_error(404))
        with patch("pyutilz.web.cached_client.urlopen_checked", opener):
            assert CachedHttpClient(cache_dir=tmp_path, min_interval=0).get_text(URL, "tag") is None
        assert list(tmp_path.rglob("*.json")) == []

    def test_a_successful_text_fetch_is_cached(self, tmp_path):
        with patch("pyutilz.web.cached_client.urlopen_checked", _fake_response(b"plain")):
            CachedHttpClient(cache_dir=tmp_path, min_interval=0).get_text(URL, "tag")
        assert len(list(tmp_path.rglob("*.json"))) == 1


class TestDecodingAndCacheReads:
    def test_an_invalid_utf8_text_body_is_decoded_with_replacement(self, tmp_path):
        body = "Привет".encode() + b"\xff"
        with patch("pyutilz.web.cached_client.urlopen_checked", _fake_response(body)):
            assert CachedHttpClient(cache_dir=tmp_path, min_interval=0).get_text(URL, "tag") == "Привет�"

    def test_an_invalid_utf8_json_body_is_a_permanent_failure_not_an_exception(self, tmp_path):
        with patch("pyutilz.web.cached_client.urlopen_checked", _fake_response(b"\xff{}")):
            assert CachedHttpClient(cache_dir=tmp_path, min_interval=0).get_json(URL, "tag") is None

    def test_invalid_json_is_logged_at_debug_with_the_tag_and_url(self, tmp_path, caplog):
        with patch("pyutilz.web.cached_client.urlopen_checked", _fake_response(b"{bad")), caplog.at_level(logging.DEBUG):
            CachedHttpClient(cache_dir=tmp_path, min_interval=0).get_json(URL, "tag")
        messages = [r.getMessage() for r in caplog.records if "invalid JSON" in r.getMessage()]
        assert len(messages) == 1
        assert messages[0].startswith("cached_client tag: invalid JSON for https://example.org/x: ")

    def test_a_cached_text_is_served_without_the_network(self, tmp_path):
        with patch("pyutilz.web.cached_client.urlopen_checked", _fake_response(b"plain")):
            CachedHttpClient(cache_dir=tmp_path, min_interval=0).get_text(URL, "tag")
        with patch("pyutilz.web.cached_client.urlopen_checked") as mocked:
            assert CachedHttpClient(cache_dir=tmp_path, min_interval=0).get_text(URL, "tag") == "plain"
        mocked.assert_not_called()

    def test_a_cached_non_ascii_text_is_read_back_as_utf8(self, tmp_path):
        with patch("pyutilz.web.cached_client.urlopen_checked", _fake_response("Привет".encode())):
            CachedHttpClient(cache_dir=tmp_path, min_interval=0).get_text(URL, "tag")
        with patch("pyutilz.web.cached_client.urlopen_checked") as mocked:
            assert CachedHttpClient(cache_dir=tmp_path, min_interval=0).get_text(URL, "tag") == "Привет"
        mocked.assert_not_called()

    def test_an_offline_json_miss_is_logged_at_debug(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setenv("MY_OFFLINE", "1")
        client = CachedHttpClient(cache_dir=tmp_path, offline_env_var="MY_OFFLINE")
        with caplog.at_level(logging.DEBUG):
            assert client.get_json(URL, "tag") is None
        assert "cached_client tag: offline, no cache entry for https://example.org/x" in [r.getMessage() for r in caplog.records]

    def test_each_later_read_asks_for_the_remaining_budget_only(self, tmp_path):
        resp = _FakeResp(b"x" * 8, step=3)
        opener, _ = _open_all(resp)
        with patch("pyutilz.web.cached_client.urlopen_checked", opener):
            assert CachedHttpClient(cache_dir=tmp_path, min_interval=0, max_bytes=10).get_text(URL, "tag") == "x" * 8
        assert resp.read_sizes == [11, 8, 5, 3]

    def test_an_offline_text_miss_is_logged_at_debug(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setenv("MY_OFFLINE", "1")
        client = CachedHttpClient(cache_dir=tmp_path, offline_env_var="MY_OFFLINE")
        with caplog.at_level(logging.DEBUG):
            assert client.get_text(URL, "tag") is None
        assert "cached_client tag: offline, no cache entry for https://example.org/x" in [r.getMessage() for r in caplog.records]


class TestBodyCap:
    def test_the_default_cap_is_sixty_four_mebibytes(self, tmp_path):
        assert CachedHttpClient(cache_dir=tmp_path).max_bytes == 64 * 1024 * 1024

    def test_a_body_exactly_at_the_cap_is_kept(self, tmp_path):
        opener, _ = _open_all(_FakeResp(b"x" * 10))
        with patch("pyutilz.web.cached_client.urlopen_checked", opener):
            assert CachedHttpClient(cache_dir=tmp_path, min_interval=0, max_bytes=10).get_text(URL, "tag") == "x" * 10

    def test_a_body_one_byte_over_the_cap_is_discarded(self, tmp_path, caplog):
        opener, _ = _open_all(_FakeResp(b"x" * 11))
        with patch("pyutilz.web.cached_client.urlopen_checked", opener), caplog.at_level(logging.WARNING):
            assert CachedHttpClient(cache_dir=tmp_path, min_interval=0, max_bytes=10).get_text(URL, "tag") is None
        assert "cached_client: response from https://example.org/x exceeds max_bytes=10; discarding" in [r.getMessage() for r in caplog.records]

    def test_a_body_that_only_exceeds_the_cap_in_chunks_is_discarded(self, tmp_path):
        opener, _ = _open_all(_FakeResp(b"x" * 20, step=5))
        with patch("pyutilz.web.cached_client.urlopen_checked", opener):
            assert CachedHttpClient(cache_dir=tmp_path, min_interval=0, max_bytes=10).get_text(URL, "tag") is None

    def test_no_cap_reads_the_whole_body_in_one_call(self, tmp_path):
        resp = _FakeResp(b"x" * 20)
        opener, _ = _open_all(resp)
        with patch("pyutilz.web.cached_client.urlopen_checked", opener):
            assert CachedHttpClient(cache_dir=tmp_path, min_interval=0, max_bytes=None).get_text(URL, "tag") == "x" * 20
        assert resp.read_sizes == [-1]

    def test_the_streamed_reads_never_ask_for_more_than_one_byte_past_the_cap(self, tmp_path):
        resp = _FakeResp(b"x" * 20)
        opener, _ = _open_all(resp)
        with patch("pyutilz.web.cached_client.urlopen_checked", opener):
            CachedHttpClient(cache_dir=tmp_path, min_interval=0, max_bytes=10).get_text(URL, "tag")
        assert resp.read_sizes[0] == 11
        assert max(resp.read_sizes) <= 11

    def test_a_large_body_is_streamed_in_256_kibibyte_reads(self, tmp_path):
        body = b"x" * 300_000
        resp = _FakeResp(body)
        opener, _ = _open_all(resp)
        with patch("pyutilz.web.cached_client.urlopen_checked", opener):
            assert CachedHttpClient(cache_dir=tmp_path, min_interval=0).get_text(URL, "tag") == body.decode("utf-8")
        assert resp.read_sizes[0] == 256 * 1024

    def test_a_declared_length_over_the_cap_is_refused_with_its_numbers_named(self, tmp_path, caplog):
        resp = _FakeResp(b"x" * 11, headers={"Content-Length": "11"})
        opener, _ = _open_all(resp)
        with patch("pyutilz.web.cached_client.urlopen_checked", opener), caplog.at_level(logging.WARNING):
            assert CachedHttpClient(cache_dir=tmp_path, min_interval=0, max_bytes=10).get_text(URL, "tag") is None
        assert "cached_client: https://example.org/x declares Content-Length=11 > max_bytes=10; refusing" in [r.getMessage() for r in caplog.records]
        assert resp.read_sizes == []

    def test_a_declared_length_exactly_at_the_cap_is_not_refused(self, tmp_path):
        opener, _ = _open_all(_FakeResp(b"x" * 10, headers={"Content-Length": "10"}))
        with patch("pyutilz.web.cached_client.urlopen_checked", opener):
            assert CachedHttpClient(cache_dir=tmp_path, min_interval=0, max_bytes=10).get_text(URL, "tag") == "x" * 10

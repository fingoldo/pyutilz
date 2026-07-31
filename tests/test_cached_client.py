"""Tests for pyutilz.web.cached_client.CachedHttpClient.

Mocks pyutilz.web.cached_client.urlopen_checked directly (not the stdlib urllib) so every test
runs offline and deterministically -- no real network access, no flakiness from a live service.
"""

from __future__ import annotations

import io
import json
import pickle
import urllib.error
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
        assert json.loads(cached[0].read_text())["data"] == {"a": 1}

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
        client = CachedHttpClient(cache_dir=tmp_path, min_interval=0.05)
        client._last_call["example.org"] = __import__("time").monotonic()
        slept = []
        with patch("time.sleep", side_effect=slept.append):
            client._throttle("https://example.org/y")
        assert slept and slept[0] > 0

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
    restored = pickle.loads(pickle.dumps(client))
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

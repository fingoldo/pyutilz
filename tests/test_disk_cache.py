"""Unit tests for pyutilz.core.disk_cache."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import pytest

from pyutilz.core import disk_cache as dc
from pyutilz.core.disk_cache import (
    DiskCache,
    compose_key,
    hash_array_summary,
    hash_object,
)

# -------------------- hashing determinism --------------------


def test_hash_array_summary_deterministic_same_input():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((100, 8))
    assert hash_array_summary(arr) == hash_array_summary(arr.copy())


def test_hash_array_summary_changes_on_value_perturbation():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((100, 8))
    arr_b = arr.copy()
    arr_b[0, 0] += 1e-3  # changes head bytes + col0 sum/min/max
    assert hash_array_summary(arr) != hash_array_summary(arr_b)


def test_hash_array_summary_changes_on_shape_change():
    arr = np.zeros((10, 5))
    assert hash_array_summary(arr) != hash_array_summary(arr.reshape(5, 10))


def test_hash_array_summary_changes_on_dtype_change():
    arr = np.zeros((10, 5), dtype=np.float32)
    assert hash_array_summary(arr) != hash_array_summary(arr.astype(np.float64))


def test_hash_array_summary_catches_middle_row_change_via_col_sum():
    rng = np.random.default_rng(1)
    arr = rng.standard_normal((1000, 4))  # rows past summary head/tail
    arr_b = arr.copy()
    arr_b[500, 0] += 7.0  # middle row -> col0 sum changes
    assert hash_array_summary(arr) != hash_array_summary(arr_b)


def test_hash_array_summary_empty_array():
    a = np.zeros((0, 5))
    b = np.zeros((0, 5))
    assert hash_array_summary(a) == hash_array_summary(b)
    c = np.zeros((0, 7))
    assert hash_array_summary(a) != hash_array_summary(c)


def test_hash_array_summary_1d():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 4.0])
    assert hash_array_summary(a) != hash_array_summary(b)


def test_hash_object_dict_order_invariant():
    d1 = {"a": 1, "b": 2, "c": 3}
    d2 = {"c": 3, "a": 1, "b": 2}
    assert hash_object(d1) == hash_object(d2)


def test_hash_object_distinguishes_values():
    assert hash_object({"a": 1}) != hash_object({"a": 2})
    assert hash_object([1, 2, 3]) != hash_object([1, 3, 2])
    assert hash_object(None) != hash_object(0)
    assert hash_object(True) != hash_object(1)


def test_hash_object_handles_numpy_scalar():
    assert hash_object(np.int64(5)) == hash_object(5)


def test_hash_object_with_nested_array_uses_summary():
    """An ndarray nested inside a dict should be hashed via the summary path."""
    a = np.zeros((100, 5))
    b = a.copy()
    b[0, 0] = 1.0
    assert hash_object({"x": a}) != hash_object({"x": b})


def test_compose_key_stable():
    k1 = compose_key("abc", "def", "ghi")
    k2 = compose_key("abc", "def", "ghi")
    assert k1 == k2
    assert k1 != compose_key("abc", "def", "ghij")
    assert len(k1) == len(compose_key("a", "b"))


def test_compose_key_rejects_empty():
    with pytest.raises(ValueError):
        compose_key()


def test_compose_key_separator_safe():
    """Parts of different cardinality must not collide via concatenation."""
    assert compose_key("abc", "def") != compose_key("abcdef")
    assert compose_key("a", "bc") != compose_key("ab", "c")


# -------------------- DiskCache round-trip + LRU --------------------


def test_disk_cache_put_get_roundtrip(tmp_path: Path):
    cache = DiskCache(tmp_path)
    arr = np.arange(1000).reshape(100, 10)
    cache.put("k1", arr)
    out = cache.get("k1")
    assert out is not None
    np.testing.assert_array_equal(out, arr)
    assert cache.hits == 1
    assert cache.misses == 0


def test_disk_cache_miss_returns_none(tmp_path: Path):
    cache = DiskCache(tmp_path)
    assert cache.get("absent") is None
    assert cache.misses == 1
    assert cache.hits == 0


def test_disk_cache_pickle_complex_payload(tmp_path: Path):
    cache = DiskCache(tmp_path)
    payload = {
        "phi": np.random.default_rng(0).standard_normal((50, 4)),
        "base": np.array([0.5] * 50),
        "meta": {"n_splits": 5, "seed": 42},
    }
    cache.put("k", payload)
    out = cache.get("k")
    assert set(out) == {"phi", "base", "meta"}
    np.testing.assert_array_equal(out["phi"], payload["phi"])
    np.testing.assert_array_equal(out["base"], payload["base"])
    assert out["meta"] == payload["meta"]


def test_disk_cache_eviction_under_cap(tmp_path: Path):
    """Putting many entries above the cap should evict oldest by mtime."""
    cache = DiskCache(tmp_path, max_size_bytes=25_000)
    blob = np.zeros(1000, dtype=np.float64)  # ~8KB raw + pickle overhead
    for i in range(10):
        cache.put(f"k{i}", blob)
    assert cache.total_size() <= cache.max_size_bytes * 1.5  # eviction is best-effort
    assert cache.evictions > 0
    assert cache.get("k9") is not None  # newest entry must survive


def test_disk_cache_corrupt_entry_treated_as_miss(tmp_path: Path):
    cache = DiskCache(tmp_path)
    cache.put("k1", np.arange(10))
    p = tmp_path / "k1.pkl"
    p.write_bytes(b"this is not pickle")
    out = cache.get("k1")
    assert out is None
    assert not p.exists()


def test_disk_cache_atomic_write_no_partial(tmp_path: Path):
    """Simulate a crash mid-write: the orphan tmp_ file must not be served on get."""
    cache = DiskCache(tmp_path)
    (tmp_path / "tmp_abc123.pkl").write_bytes(b"junk")
    cache.put("k1", np.array([1, 2, 3]))
    out = cache.get("k1")
    np.testing.assert_array_equal(out, np.array([1, 2, 3]))
    assert cache.get("tmp_abc123") is None
    listed = [p.name for p in tmp_path.iterdir() if p.name.endswith(".pkl") and not p.name.startswith("tmp_")]
    assert "k1.pkl" in listed


def test_disk_cache_concurrent_same_key(tmp_path: Path):
    """Two threads writing the same key produce a valid readable entry."""
    cache = DiskCache(tmp_path)
    payload = np.arange(500)
    errors = []

    def worker():
        try:
            for _ in range(5):
                cache.put("shared", payload)
        except Exception as exc:  # pragma: no cover - test fails if hit
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    out = cache.get("shared")
    np.testing.assert_array_equal(out, payload)


def test_disk_cache_concurrent_same_key_distinct_payloads_never_corrupt(tmp_path: Path):
    """Regression sensor: concurrent put() for the same key must never leave a sidecar that
    doesn't match the payload actually on disk. Uses a DISTINCT large payload per thread so a race
    is observable -- without the per-key lock, one thread's write_sidecar() call can land after
    another thread's later os.replace(), leaving a mismatched sidecar digest."""
    cache = DiskCache(tmp_path)
    payloads = {i: np.full(200_000, i, dtype=np.int64) for i in range(6)}
    errors = []

    def worker(i):
        try:
            for _ in range(8):
                cache.put("shared", payloads[i])
        except Exception as exc:  # pragma: no cover - test fails if hit
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in payloads]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors

    out = cache.get("shared")
    assert out is not None, "entry should not have been treated as corrupt/missing"
    matches = [i for i, p in payloads.items() if np.array_equal(out, p)]
    assert len(matches) == 1, f"result did not match exactly one written payload: {out[:5] if out is not None else None}"


def test_disk_cache_clear(tmp_path: Path):
    cache = DiskCache(tmp_path)
    cache.put("a", np.array([1]))
    cache.put("b", np.array([2]))
    cache.clear()
    assert cache.get("a") is None
    assert cache.get("b") is None


def test_disk_cache_get_touches_mtime(tmp_path: Path):
    """Cache hits should refresh mtime so LRU eviction considers them recent."""
    cache = DiskCache(tmp_path)
    cache.put("old", np.array([0]))
    old_path = tmp_path / "old.pkl"
    backdated = old_path.stat().st_mtime - 600.0
    os.utime(old_path, (backdated, backdated))
    cache.put("new", np.array([1]))
    cache.get("old")  # touch
    refreshed = old_path.stat().st_mtime
    assert refreshed > backdated


def test_disk_cache_picklable(tmp_path: Path):
    """A DiskCache instance itself must survive a pickle round-trip (threading.Lock excluded)."""
    import pickle

    cache = DiskCache(tmp_path)
    cache.put("k", np.array([1, 2, 3]))
    restored = pickle.loads(pickle.dumps(cache))  # deserialize-ok: round-trips a pickle this test just produced in-process; the bytes never leave the test, no untrusted input
    out = restored.get("k")
    np.testing.assert_array_equal(out, np.array([1, 2, 3]))
    restored.put("k2", np.array([4, 5]))
    np.testing.assert_array_equal(restored.get("k2"), np.array([4, 5]))


def test_disk_cache_key_path_traversal_rejected(tmp_path: Path):
    cache = DiskCache(tmp_path)
    # A traversing key is now rejected by the separator guard before the resolve() check -- both
    # are the same contract ("a key is a content digest, not a path"), so either message is correct.
    with pytest.raises(ValueError, match=r"resolves outside cache_dir|path separator"):
        cache._key_path("../evil")


class TestGoldenDigests:
    """Cache keys are hashes of byte streams. These digests pin the exact bytes: every tag, length prefix and struct
    format feeds into them, so a change to any of those changes the key. A deliberate format change must bump
    _HASH_VERSION and update these values; an accidental one fails here instead of silently orphaning every entry."""

    @pytest.mark.parametrize(
        "obj, digest",
        [
            (0, "ba756e9d5d01ae47e452fd71f3bd79e6"),  # pragma: allowlist secret
            (1, "3e03cd21918489481bda793c0a4114f7"),  # pragma: allowlist secret
            (-5, "5e243fff9fa35ebad4ad7a2960b97406"),  # pragma: allowlist secret
            (2**70, "b595a929222acb75b629f21955ea2dba"),  # pragma: allowlist secret
            (1.5, "f85c08fafb85eca89c579cc42222399e"),  # pragma: allowlist secret
            (True, "baf924df059083c2f50b9c2b114d8a57"),  # pragma: allowlist secret
            (False, "79568dc83fd182163d35c142c4fa40ad"),  # pragma: allowlist secret
            ("abc", "5b5e4ea3878f73497a5db5f2257b3c34"),  # pragma: allowlist secret
            (b"xyz", "dbbec2c4d231f199d95de5d220aa6639"),  # pragma: allowlist secret
            ([1, 2], "7d9019fbe4f09b35168c019c3e94f03c"),  # pragma: allowlist secret
            ((1, 2), "c4916b127d88d834795aaaec067f2549"),  # pragma: allowlist secret
            ({1, 2}, "c2b43f2d060519a0475f355f1851d63f"),  # pragma: allowlist secret
            ({"a": 1}, "c265e0b104df2d51f2c87cabc62370ea"),  # pragma: allowlist secret
            (None, "d659c35a486deebe44ad33a50858a8d9"),  # pragma: allowlist secret
        ],
    )
    def test_hash_object_digest_is_pinned(self, obj, digest):
        assert dc.hash_object(obj) == digest

    def test_float64_array_digest_is_pinned(self):
        assert dc.hash_object(np.arange(12, dtype=np.float64).reshape(3, 4)) == "efcaabd03a495487d5e9d8a759f39497"  # pragma: allowlist secret

    def test_int32_array_digest_is_pinned(self):
        assert dc.hash_object(np.arange(6, dtype=np.int32).reshape(2, 3)) == "b90e9eddf0d06930bd8d0a131e73a59b"  # pragma: allowlist secret

    def test_array_summary_digest_is_pinned(self):
        arr = np.arange(40, dtype=np.float64).reshape(10, 4)
        assert dc.hash_array_summary(arr, n_summary_rows=2) == "a3e1236eb40a147841c3dd812400c288"  # pragma: allowlist secret

    @pytest.mark.parametrize("parts, digest", [(("a", "b"), "6a93c5780b9861cdd8d2761a748d9591"), (("x",), "1ff47a876f42e33dda291cecd644c99d")])  # pragma: allowlist secret
    def test_compose_key_digest_is_pinned(self, parts, digest):
        assert dc.compose_key(*parts) == digest


class TestCorruptionAndSidecarPaths:
    def test_garbage_bytes_are_caught_by_the_sidecar_check_and_both_files_are_removed(self, tmp_path, caplog):
        import logging

        cache = DiskCache(tmp_path)
        cache.put("k1", np.arange(10))
        payload = tmp_path / "k1.pkl"
        sidecar = tmp_path / "k1.pkl.sha256"
        payload.write_bytes(b"this is not pickle")
        with caplog.at_level(logging.WARNING, logger="pyutilz.core.disk_cache"):
            assert cache.get("k1") is None
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any(m.startswith(f"DiskCache: sidecar verification failed for {payload}: ") and m.endswith("; removing") for m in messages)
        assert not payload.exists()
        assert not sidecar.exists()
        assert cache.misses == 1

    def test_a_tampered_payload_fails_the_sidecar_check_and_is_dropped(self, tmp_path, caplog):
        import logging

        cache = DiskCache(tmp_path)
        cache.put("k1", np.arange(10))
        payload = tmp_path / "k1.pkl"
        sidecar = tmp_path / "k1.pkl.sha256"
        assert sidecar.exists(), "put must write the sidecar the verification reads"
        original = payload.read_bytes()
        payload.write_bytes(original[:-1] + bytes([original[-1] ^ 0xFF]))
        with caplog.at_level(logging.WARNING, logger="pyutilz.core.disk_cache"):
            assert cache.get("k1") is None
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any(m.startswith(f"DiskCache: sidecar verification failed for {payload}: ") and m.endswith("; removing") for m in messages)
        assert not payload.exists()
        assert not sidecar.exists()

    def test_a_hit_logs_its_key_at_debug(self, tmp_path, caplog):
        import logging

        cache = DiskCache(tmp_path)
        cache.put("k1", np.arange(3))
        with caplog.at_level(logging.DEBUG, logger="pyutilz.core.disk_cache"):
            cache.get("k1")
        assert [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG] == ["DiskCache: hit key=k1"]

    def test_a_hit_refreshes_the_sidecar_mtime_so_it_is_not_evicted_before_its_payload(self, tmp_path):
        cache = DiskCache(tmp_path)
        cache.put("k1", np.array([1, 2, 3]))
        sidecar = tmp_path / "k1.pkl.sha256"
        backdated = sidecar.stat().st_mtime - 600.0
        os.utime(sidecar, (backdated, backdated))
        cache.get("k1")
        assert sidecar.stat().st_mtime > backdated


class TestExactMessagesAndPutFailures:
    def test_compose_key_names_the_empty_input(self):
        with pytest.raises(ValueError) as info:
            dc.compose_key()
        assert str(info.value) == "compose_key requires at least one part"

    def test_a_key_with_a_path_separator_is_rejected_with_its_text(self, tmp_path):
        cache = DiskCache(tmp_path)
        with pytest.raises(ValueError) as info:
            cache.put("a/b", 1)
        assert str(info.value) == "DiskCache: key 'a/b' must not be empty or contain a path separator."

    def test_an_unpicklable_value_fails_the_put_without_leaving_a_temporary_file(self, tmp_path, caplog):
        import logging

        cache = DiskCache(tmp_path)
        with caplog.at_level(logging.WARNING, logger="pyutilz.core.disk_cache"):
            assert cache.put("k1", lambda: None) is False
        assert cache.put_failures == 1
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(messages) == 1 and messages[0].startswith("DiskCache: put failed for key=k1: ")
        assert list(tmp_path.glob("tmp_*")) == []
        assert not (tmp_path / "k1.pkl").exists()

    def test_a_sidecar_write_failure_is_logged_and_reported_as_not_stored(self, tmp_path, monkeypatch, caplog):
        import logging

        cache = DiskCache(tmp_path)

        def _no_sidecar(path):
            raise OSError("disk full")

        monkeypatch.setattr(dc, "write_sidecar", _no_sidecar)
        with caplog.at_level(logging.WARNING, logger="pyutilz.core.disk_cache"):
            assert cache.put("k1", np.arange(3)) is False
        assert cache.put_failures == 1
        assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING] == [
            f"DiskCache: sidecar write failed for {tmp_path / 'k1.pkl'}: disk full"
        ]
        assert list(tmp_path.glob("tmp_*")) == []

    def test_a_successful_put_returns_true_and_counts_no_failure(self, tmp_path):
        cache = DiskCache(tmp_path)
        assert cache.put("k1", np.arange(3)) is True
        assert cache.put_failures == 0


class TestEvictionAndScan:
    def _three_entries(self, tmp_path):
        """Three entries, the oldest backdated first, so eviction order is known."""
        cache = DiskCache(tmp_path)
        for i, key in enumerate(["old", "mid", "new"]):
            cache.put(key, np.zeros(5000, dtype=np.float64))
            payload = tmp_path / f"{key}.pkl"
            stamp = 1_000_000.0 + i * 100.0
            os.utime(payload, (stamp, stamp))
        return cache

    def test_eviction_removes_the_oldest_payloads_first_until_under_the_cap(self, tmp_path):
        cache = self._three_entries(tmp_path)
        cache.max_size_bytes = cache.total_size() // 2
        cache._evict_if_needed()
        remaining = sorted(p.name for p in tmp_path.glob("*.pkl"))
        assert remaining == ["new.pkl"]
        assert cache.evictions == 2

    def test_an_evicted_entry_takes_its_sidecar_with_it(self, tmp_path):
        cache = self._three_entries(tmp_path)
        cache.max_size_bytes = cache.total_size() // 2
        cache._evict_if_needed()
        assert not (tmp_path / "old.pkl.sha256").exists()
        assert not (tmp_path / "mid.pkl.sha256").exists()
        assert (tmp_path / "new.pkl.sha256").exists()

    def test_the_protected_entry_survives_even_when_it_is_the_oldest(self, tmp_path):
        cache = self._three_entries(tmp_path)
        cache.max_size_bytes = cache.total_size() // 2
        cache._evict_if_needed(protect=tmp_path / "old.pkl")
        assert (tmp_path / "old.pkl").exists()
        assert not (tmp_path / "mid.pkl").exists()

    def test_an_orphan_sidecar_is_swept_during_the_size_check(self, tmp_path):
        cache = DiskCache(tmp_path)
        orphan = tmp_path / "ghost.pkl.sha256"
        orphan.write_bytes(b"x" * 40)
        cache._evict_if_needed()
        assert not orphan.exists()

    def test_the_scan_ignores_in_flight_temporary_writes(self, tmp_path):
        (tmp_path / "tmp_inflight.pkl").write_bytes(b"x" * 99)
        (tmp_path / "real.pkl").write_bytes(b"x" * 7)
        payloads, sidecars, mtimes = dc._scan_cache_entries(tmp_path)
        assert list(payloads) == [tmp_path / "real.pkl"]
        assert payloads[tmp_path / "real.pkl"] == 7

    def test_a_sidecar_is_keyed_by_the_payload_it_belongs_to(self, tmp_path):
        (tmp_path / "k.pkl").write_bytes(b"x" * 3)
        (tmp_path / "k.pkl.sha256").write_bytes(b"y" * 64)
        payloads, sidecars, mtimes = dc._scan_cache_entries(tmp_path)
        assert sidecars == {tmp_path / "k.pkl": 64}

    def test_an_unlistable_cache_directory_is_reported_as_unavailable_with_a_warning(self, tmp_path, monkeypatch, caplog):
        import logging

        def _fail(path):
            raise OSError("no such directory")

        monkeypatch.setattr(dc.os, "scandir", _fail)
        with caplog.at_level(logging.WARNING, logger="pyutilz.core.disk_cache"):
            assert dc._scan_cache_entries(tmp_path / "gone") is None
        assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING] == [
            "disk_cache: cache entry scan failed, reporting it as unavailable: OSError: no such directory"
        ]


class TestGoldenDigestsForRemainingTags:
    """The frozenset tag, the repr fallback, object arrays, empty arrays and the nested-array tag each feed their own bytes."""

    def test_a_frozenset_has_its_own_tag_and_is_pinned(self):
        assert dc.hash_object(frozenset({1, 2})) == "79874b0c2b2b9bc19a958cbbf17dbf99"  # pragma: allowlist secret

    def test_an_unhandled_type_falls_back_to_its_repr_and_is_pinned(self):
        assert dc.hash_object(1 + 2j) == "f58aa4cd4a151a8ac5bf2c2b11c0d9ab"  # pragma: allowlist secret

    def test_an_object_array_hashes_its_values_not_its_addresses(self):
        assert dc.hash_array_summary(np.array(["a", 1, None], dtype=object)) == "c7a18c1783cc79d16e2194da7a60702b"  # pragma: allowlist secret

    def test_an_empty_array_hashes_only_its_header(self):
        assert dc.hash_array_summary(np.zeros((0, 3))) == "55c2cbf88b75389a3548866dac4c2dd9"  # pragma: allowlist secret

    def test_an_array_nested_in_a_container_contributes_its_fixed_width_digest(self):
        assert dc.hash_object([np.arange(4)]) == "6c13206f396020b203d3e407d0aee20c"  # pragma: allowlist secret

    def test_empty_bytes_are_length_prefixed_and_pinned(self):
        assert dc.hash_object(b"") == "8e93b0f4c91ab797943fd990c8b04b44"  # pragma: allowlist secret


class TestKeyLocks:
    def test_a_held_key_counts_one_reference_and_is_dropped_on_exit(self, tmp_path):
        cache = DiskCache(tmp_path)
        with cache._get_key_lock("k"):
            assert cache._key_locks["k"].refcount == 1
        assert "k" not in cache._key_locks

    def test_a_second_thread_on_the_same_key_waits_for_the_first(self, tmp_path):
        cache = DiskCache(tmp_path)
        first_in = threading.Event()
        release = threading.Event()
        second_entered = threading.Event()

        def _first():
            with cache._get_key_lock("k"):
                first_in.set()
                release.wait(5)

        def _second():
            first_in.wait(5)
            with cache._get_key_lock("k"):
                second_entered.set()

        t1 = threading.Thread(target=_first)
        t2 = threading.Thread(target=_second)
        t1.start()
        t2.start()
        first_in.wait(5)
        # The second caller is registered (refcount 2) but blocked behind the first.
        for _ in range(100):
            if cache._key_locks.get("k") is not None and cache._key_locks["k"].refcount == 2:
                break
            threading.Event().wait(0.01)
        assert not second_entered.is_set()
        assert cache._key_locks["k"].refcount == 2
        release.set()
        t1.join(5)
        t2.join(5)
        assert second_entered.is_set()
        assert cache._key_locks == {}

    def test_distinct_keys_use_distinct_locks(self, tmp_path):
        cache = DiskCache(tmp_path)
        with cache._get_key_lock("a"):
            with cache._get_key_lock("b"):
                assert cache._key_locks["a"].lock is not cache._key_locks["b"].lock
        assert cache._key_locks == {}

    def test_pickling_drops_the_locks_and_the_cache_rebuilds_them_empty(self, tmp_path):
        import pickle

        cache = DiskCache(tmp_path)
        cache.put("k1", np.arange(3))
        clone = pickle.loads(pickle.dumps(cache))
        assert clone._key_locks == {}
        assert clone.get("k1") is not None


class TestCountersAccumulate:
    def test_hits_and_misses_add_up_across_repeated_calls(self, tmp_path):
        cache = DiskCache(tmp_path)
        cache.put("a", 1)
        cache.get("a")
        cache.get("a")
        cache.get("missing")
        cache.get("missing")
        cache.get("missing")
        assert cache.hits == 2
        assert cache.misses == 3

    def test_put_failures_add_up_across_repeated_failures(self, tmp_path):
        cache = DiskCache(tmp_path)
        assert cache.put("a", lambda: None) is False
        assert cache.put("b", lambda: None) is False
        assert cache.put("c", lambda: None) is False
        assert cache.put_failures == 3

    def test_evictions_add_up_across_a_forced_eviction_pass(self, tmp_path):
        cache = DiskCache(tmp_path)
        for i in range(4):
            cache.put(f"k{i}", np.zeros(4000, dtype=np.float64))
            payload = tmp_path / f"k{i}.pkl"
            os.utime(payload, (1_000_000.0 + i, 1_000_000.0 + i))
        cache.max_size_bytes = cache.total_size() // 4
        cache._evict_if_needed()
        assert cache.evictions == 3

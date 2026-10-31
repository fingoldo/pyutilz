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

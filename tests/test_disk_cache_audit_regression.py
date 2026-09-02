"""Regression tests for the 2026-09-02 core/dev/system domain audit (disk_cache findings)."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from pyutilz.core.disk_cache import DiskCache, hash_object


def _names(d) -> list:
    return sorted(p.name for p in Path(d).iterdir())


def test_hot_entry_survives_eviction_with_its_sidecar(tmp_path):
    """F01: get() must keep the whole entry (payload + sidecar) recent, not just the payload."""
    c = DiskCache(tmp_path, max_size_bytes=10**9)
    blob = b"x" * 30_000
    for key in ("cold1", "cold2", "hot"):
        c.put(key, blob)
    # "hot" was put first in LRU terms: its sidecar is the oldest file in the directory, while the
    # cold entries are newer. Only a get() that refreshes the whole entry can save it.
    os.utime(tmp_path / "hot.pkl.sha256", (1_000_000, 1_000_000))
    os.utime(tmp_path / "hot.pkl", (1_000_001, 1_000_001))
    for i, key in enumerate(("cold1", "cold2")):
        os.utime(tmp_path / (key + ".pkl"), (1_000_010 + i * 2, 1_000_010 + i * 2))
        os.utime(tmp_path / (key + ".pkl.sha256"), (1_000_011 + i * 2, 1_000_011 + i * 2))

    assert c.get("hot") == blob  # must refresh the entry as a unit
    c.max_size_bytes = 90_000
    c._evict_if_needed()

    assert (tmp_path / "hot.pkl").exists()
    assert (tmp_path / "hot.pkl.sha256").exists()
    assert c.get("hot") == blob


def test_eviction_removes_payload_and_sidecar_together(tmp_path):
    """F31: eviction must not leave orphan sidecars occupying the size cap."""
    c = DiskCache(tmp_path, max_size_bytes=10**9)
    blob = b"y" * 30_000
    for key in ("a", "b", "c"):
        c.put(key, blob)
    c.max_size_bytes = 40_000
    c._evict_if_needed()

    names = _names(tmp_path)
    for name in names:
        if name.endswith(".sha256"):
            assert name[: -len(".sha256")] in names, "orphan sidecar " + name


def test_orphan_sidecar_is_swept(tmp_path):
    """F31: a sidecar whose payload is gone is unreachable, so it must not hold budget forever."""
    c = DiskCache(tmp_path, max_size_bytes=10**9)
    c.put("gone", b"z" * 1000)
    (tmp_path / "gone.pkl").unlink()
    c._evict_if_needed()
    assert not (tmp_path / "gone.pkl.sha256").exists()


def test_put_of_unpicklable_value_leaves_no_orphan_tmp_file(tmp_path):
    """F02: TypeError from pickling must not escape put() nor strand a tmp_ file."""
    c = DiskCache(tmp_path, max_size_bytes=10**9)
    c.put("lock", threading.Lock())  # must not raise
    assert _names(tmp_path) == []
    assert c.total_size() == 0
    assert c.put_failures == 1


def test_put_failure_is_logged_at_warning(tmp_path, caplog):
    """F30: a genuine write failure must be visible, not DEBUG-only."""
    c = DiskCache(tmp_path, max_size_bytes=10**9)
    with caplog.at_level("WARNING"):
        c.put("lock", threading.Lock())
    assert any("put failed" in r.getMessage() for r in caplog.records)


def test_hash_object_frozenset_stable_across_hash_seeds():
    """F06: a frozenset key must hash identically in every process."""
    import pyutilz

    src_root = str(Path(pyutilz.__file__).parent.parent)
    snippet = (
        "import sys; sys.path.insert(0, " + repr(src_root) + ")\n"
        "from pyutilz.core.disk_cache import hash_object\n"
        "print(hash_object(frozenset(['alpha', 'beta', 'gamma', 'delta'])))\n"
    )
    digests = set()
    for seed in ("0", "1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        out = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True, env=env, check=True)
        digests.add(out.stdout.strip())
    assert len(digests) == 1, digests


def test_hash_object_frozenset_differs_from_set():
    """A frozenset and a set of the same items are distinct types and must not collide."""
    assert hash_object(frozenset([1, 2])) != hash_object({1, 2})

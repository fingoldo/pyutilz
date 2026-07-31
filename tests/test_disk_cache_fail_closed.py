"""DiskCache.get must fail CLOSED on a cache file with no .sha256 sidecar.

A payload planted in the cache dir with no sidecar (e.g. a corrupted write, or tampering) must be
refused by default rather than unpickled silently; the only opt-in is
pyutilz.core.safe_pickle's PYUTILZ_ALLOW_UNVERIFIED_PICKLE env var.
"""

import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data

from pyutilz.core.disk_cache import DiskCache
from pyutilz.core.safe_pickle import DEFAULT_ALLOW_UNVERIFIED_ENV_VAR


def _plant_unverified_entry(cache: DiskCache, key: str, value) -> None:
    """Write a cache payload WITHOUT its .sha256 sidecar (simulates a planted/untrusted file)."""
    path = cache._key_path(key)
    with open(path, "wb") as f:
        pickle.dump(value, f)
    sidecar = path.parent / (path.name + ".sha256")
    if sidecar.exists():
        sidecar.unlink()


def test_missing_sidecar_refused_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv(DEFAULT_ALLOW_UNVERIFIED_ENV_VAR, raising=False)
    cache = DiskCache(tmp_path)
    _plant_unverified_entry(cache, "k", {"planted": 1})

    # Fail-closed: refused, reported as a miss, no unpickling of the un-sidecar'd payload.
    assert cache.get("k") is None


def test_missing_sidecar_allowed_with_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv(DEFAULT_ALLOW_UNVERIFIED_ENV_VAR, "1")
    cache = DiskCache(tmp_path)
    _plant_unverified_entry(cache, "k", {"planted": 1})

    assert cache.get("k") == {"planted": 1}


def test_legit_put_roundtrips(tmp_path, monkeypatch):
    monkeypatch.delenv(DEFAULT_ALLOW_UNVERIFIED_ENV_VAR, raising=False)
    cache = DiskCache(tmp_path)
    cache.put("k", {"ok": 2})
    assert cache.get("k") == {"ok": 2}

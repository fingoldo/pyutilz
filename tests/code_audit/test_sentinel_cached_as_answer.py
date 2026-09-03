"""Scanner tests for sentinel_cached_as_answer, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit.sentinel_cached_as_answer import scan_sentinel_cached_as_answer

from ._helpers import _write

# ---- sentinel_cached_as_answer -------------------------------------------


def test_sentinel_cached_as_answer_flags_the_canonical_case(tmp_path: Path):
    """One transient failure pins the key to None for the lifetime of the process."""
    _write(
        tmp_path,
        "out.py",
        """
_cache = {}

def lookup(key):
    if key not in _cache:
        try:
            _cache[key] = fetch(key)
        except Exception:
            _cache[key] = None
    return _cache[key]
""",
    )
    findings = scan_sentinel_cached_as_answer(tmp_path)
    assert len(findings) == 1, findings
    assert "None" in findings[0].detail


def test_sentinel_cached_as_answer_flags_an_empty_container(tmp_path: Path):
    """`{}` cached on a build failure is the same defect wearing a different sentinel."""
    _write(
        tmp_path,
        "out.py",
        """
_map_cache = {}

def maps(src):
    try:
        _map_cache[src] = build(src)
    except Exception:
        _map_cache[src] = {}
    return _map_cache[src]
""",
    )
    assert len(scan_sentinel_cached_as_answer(tmp_path)) == 1


def test_sentinel_cached_as_answer_ignores_a_real_value(tmp_path: Path):
    """A handler that caches a genuine fallback is not caching a failure."""
    _write(
        tmp_path,
        "out.py",
        """
_cache = {}

def lookup(key):
    try:
        _cache[key] = fetch(key)
    except Exception:
        _cache[key] = DEFAULT_FOR[key]
    return _cache[key]
""",
    )
    assert scan_sentinel_cached_as_answer(tmp_path) == []


def test_sentinel_cached_as_answer_ignores_a_plain_local(tmp_path: Path):
    """Assigning None to something that is not a cache costs nothing after the call returns."""
    _write(
        tmp_path,
        "out.py",
        """
def lookup(key, results):
    try:
        results[key] = fetch(key)
    except Exception:
        results[key] = None
    return results[key]
""",
    )
    assert scan_sentinel_cached_as_answer(tmp_path) == []


def test_sentinel_cached_as_answer_ignores_a_write_outside_a_handler(tmp_path: Path):
    """Caching None on a path that did not fail says nothing about a swallowed error."""
    _write(
        tmp_path,
        "out.py",
        """
_cache = {}

def reset(key):
    _cache[key] = None
""",
    )
    assert scan_sentinel_cached_as_answer(tmp_path) == []


# ---- F83/F174: sentinel cached as answer --------------------------------------------


def test_sentinel_cached_as_answer_sees_the_fallback_arm(tmp_path: Path):
    _write(tmp_path, "a.py", """
cache = {}


def g(k):
    try:
        v = lookup(k)
    except Exception:
        v = None
    if v is None:
        cache[k] = None
    return v
""")
    assert len(scan_sentinel_cached_as_answer(tmp_path)) == 1


def test_sentinel_cached_as_answer_sees_a_returned_cache_write(tmp_path: Path):
    _write(tmp_path, "a.py", """
def g(k, cache):
    try:
        return lookup(k)
    except Exception:
        return cache.set(k, None)
""")
    assert len(scan_sentinel_cached_as_answer(tmp_path)) == 1


def test_sentinel_cached_as_answer_ignores_a_successful_cache_write(tmp_path: Path):
    _write(tmp_path, "a.py", """
cache = {}


def g(k):
    try:
        cache[k] = lookup(k)
    except Exception:
        raise
""")
    assert scan_sentinel_cached_as_answer(tmp_path) == []

"""Regression test: two threads in the same process calling safe_dump() to the SAME
path concurrently must never interleave into a corrupt/truncated file.

Before the fix, the atomic-write temp filename was disambiguated only by
``os.getpid()``, so two threads in one process raced on the identical temp file
(``<path>.tmp.<pid>``) -- one thread's ``open(tmp, "wb")`` could truncate the other's
in-flight write, corrupting the payload that then got ``os.replace``'d onto ``path``.
The fix adds ``threading.get_ident()`` to the temp filename so each thread gets its
own temp file and only a real, complete pickle is ever renamed onto ``path``.
"""
from __future__ import annotations

import pickle
import threading
import time

import pytest

from pyutilz.core.safe_pickle import safe_dump, safe_load


def _make_big_payload(tag: str, n: int = 200_000) -> dict:
    # Large enough that a naive interleaved write (two threads sharing one temp
    # file handle) has a realistic chance of producing a truncated/mixed file.
    return {"tag": tag, "data": [tag] * n}


@pytest.mark.parametrize("run", range(5))
def test_safe_dump_concurrent_same_path_never_corrupts(tmp_path, run):
    path = str(tmp_path / f"shared_{run}.pkl")
    payload_a = _make_big_payload("A")
    payload_b = _make_big_payload("B")

    errors: list[BaseException] = []

    def _writer(obj: dict) -> None:
        try:
            safe_dump(obj, path)
        except BaseException as exc:  # noqa: BLE001 - capture from thread for main-thread assertion
            errors.append(exc)

    t1 = threading.Thread(target=_writer, args=(payload_a,))
    t2 = threading.Thread(target=_writer, args=(payload_b,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors, f"safe_dump raised in a thread: {errors}"

    # The file on disk must be fully valid (unpicklable) and must match exactly
    # one of the two payloads written -- never a corrupted/truncated mix.
    result = safe_load(path)
    assert result in (payload_a, payload_b)

    # Also confirm raw pickle.load succeeds (belt-and-suspenders vs. safe_load's
    # own verification masking a subtly-corrupt-but-hash-matching file).
    with open(path, "rb") as f:
        raw = pickle.load(f)
    assert raw == result


def test_path_lock_entries_are_reclaimed_after_use(tmp_path):
    """Regression (meta-test-driven finding, proactive-cache-audit pass): ``_path_locks`` used to
    be a plain unbounded dict -- one Lock object per distinct path ever safe_dump/safe_load'd,
    kept forever with no eviction. A long-running process touching many distinct paths over its
    lifetime would grow this dict without bound. Fixed via reference-counted cleanup (removing an
    entry once no thread still holds/waits on it) rather than a plain LRU bound, since evicting a
    LOCK a thread is still blocked on would let a later caller for the same path get a different,
    unlocked object -- silently breaking the mutual-exclusion guarantee. Asserts the internal
    dict is back to empty after every writer has finished, across many distinct paths."""
    from pyutilz.core import safe_pickle as sp_module

    paths = [str(tmp_path / f"reclaim_{i}.pkl") for i in range(20)]
    for i, path in enumerate(paths):
        safe_dump({"i": i}, path)

    assert sp_module._path_locks == {}, f"_path_locks should be empty once no writer is in flight, found {len(sp_module._path_locks)} stale entries"

    # And every payload actually landed correctly (the refcounted rewrite didn't break the
    # underlying atomic-write-plus-sidecar contract).
    for i, path in enumerate(paths):
        assert safe_load(path) == {"i": i}


def test_path_lock_still_serializes_concurrent_writers_to_the_same_path(tmp_path):
    """Regression companion to test_safe_dump_concurrent_same_path_never_corrupts: forces a REAL
    interleaving window (via threading.Event handshakes, not just thread.start()/join()) across
    the exact boundary where the refcounted rewrite could have introduced a bug -- a second
    caller for the SAME path arriving while the first is still inside its critical section must
    still block on the SAME lock object, not silently get a fresh, unlocked one."""
    import threading

    from pyutilz.core import safe_pickle as sp_module

    path = str(tmp_path / "interleaved.pkl")
    entered_first = threading.Event()
    release_first = threading.Event()
    observed_concurrent_entry = threading.Event()

    def first():
        with sp_module._get_path_lock(path):
            entered_first.set()
            release_first.wait(timeout=5)

    def second():
        assert entered_first.wait(timeout=5), "first() never signalled entry -- deadlock in handshake"
        # If the lock were NOT properly serializing (e.g. a fresh, unlocked entry got handed
        # out because the first caller's entry was wrongly evicted), this would enter immediately.
        with sp_module._get_path_lock(path):
            if not release_first.is_set():
                observed_concurrent_entry.set()

    t1 = threading.Thread(target=first)
    t2 = threading.Thread(target=second)
    t1.start()
    assert entered_first.wait(timeout=5)
    t2.start()
    time.sleep(0.05)  # give second() a chance to (wrongly) enter if the lock were broken
    release_first.set()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not observed_concurrent_entry.is_set(), "second() entered the critical section while first() still held it -- refcounted lock rewrite broke mutual exclusion"
    assert sp_module._path_locks == {}, "no stale entries should remain after both callers finish"

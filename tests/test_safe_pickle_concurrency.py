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

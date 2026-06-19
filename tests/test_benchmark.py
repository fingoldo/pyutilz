"""Tests for the realistic kernel-tuning benchmark helper
(``pyutilz.performance.kernel_tuning.benchmark``)."""
from __future__ import annotations

import threading
import time

import pytest

from pyutilz.performance.kernel_tuning import benchmark_backends, time_backend


def test_time_backend_returns_finite_ms():
    calls = []
    t = time_backend(lambda v: calls.append(v), lambda: (1,), concurrency=1, n_iters=3, warmup=1)
    assert t >= 0.0 and t != float("inf")


def test_fresh_inputs_per_call_mints_new_inputs_each_call():
    made = []

    def make():
        made.append(1)
        return (len(made),)

    time_backend(lambda v: None, make, concurrency=1, n_iters=4, warmup=2, fresh_inputs_per_call=True)
    # 2 warmup + 4 timed = 6 factory calls; every call got a distinct input.
    assert len(made) == 6


def test_reuse_builds_inputs_once():
    made = []
    time_backend(
        lambda v: None,
        lambda: made.append(1) or (0,),
        concurrency=1, n_iters=5, warmup=0, fresh_inputs_per_call=False,
    )
    # one shared input set built once (no warmup) -> exactly 1 factory call.
    assert len(made) == 1


def test_concurrency_runs_all_threads():
    seen_threads = set()
    lock = threading.Lock()

    def fn(_):
        with lock:
            seen_threads.add(threading.get_ident())
        time.sleep(0.002)

    time_backend(fn, lambda: (0,), concurrency=3, n_iters=2, warmup=0)
    assert len(seen_threads) == 3  # each of the 3 worker threads actually ran


def test_slower_backend_measures_higher():
    fast = time_backend(lambda _: None, lambda: (0,), n_iters=5, warmup=1)
    slow = time_backend(lambda _: time.sleep(0.01), lambda: (0,), n_iters=5, warmup=1)
    assert slow > fast


def test_benchmark_backends_returns_per_backend_medians():
    res = benchmark_backends(
        {"noop": lambda _: None, "sleep": lambda _: time.sleep(0.01)},
        lambda: (0,),
        n_iters=4, warmup=1,
    )
    assert set(res) == {"noop", "sleep"}
    assert res["sleep"] > res["noop"]
    assert min(res, key=res.get) == "noop"


def test_n_iters_must_be_positive():
    with pytest.raises(ValueError):
        time_backend(lambda _: None, lambda: (0,), n_iters=0)

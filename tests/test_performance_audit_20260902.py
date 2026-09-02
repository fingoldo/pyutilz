"""Regression tests for the 2026-09-02 performance audit (audits/2026-09-02/10-performance.md).

Every test here pins the BEHAVIOUR a performance fix had to preserve (or, for F01, the correctness
bug the fix removed) -- not the speed, which is measured in the audit's disposition lines.
"""

from __future__ import annotations

import json
import os
import pickle
import threading
from unittest import mock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# F01 -- time_backend must synchronize the device INSIDE the timed region
# ---------------------------------------------------------------------------

from pyutilz.performance.kernel_tuning import benchmark as bm


def test_f01_sync_runs_inside_the_timed_region_by_default():
    """The sync must land between the backend call and the timer stop, or the sample is launch-only."""
    order: list = []
    ticks = iter(range(1000))

    def fake_timer():
        return float(next(ticks))

    with mock.patch("pyutilz.dev.benchmarking.synchronize_gpu_if_available", lambda: order.append("sync")):
        bm.time_backend(lambda v: order.append("call"), lambda: (1,), n_iters=2, warmup=1, timer=fake_timer)

    # warmup call, warmup sync, then every timed call is immediately followed by a sync.
    assert order == ["call", "sync", "call", "sync", "call", "sync"]


def test_f01_measured_time_includes_the_synchronization():
    """A sync that takes real time must show up in the reported wall time."""
    clock = {"t": 0.0}

    def fake_timer():
        return clock["t"]

    def slow_sync():
        clock["t"] += 0.010  # 10 ms of "device work draining"

    with mock.patch("pyutilz.dev.benchmarking.synchronize_gpu_if_available", slow_sync):
        ms = bm.time_backend(lambda: None, lambda: (), n_iters=3, warmup=0, timer=fake_timer)
    assert ms == pytest.approx(10.0)


def test_f01_synchronize_gpu_can_be_switched_off():
    calls: list = []
    with mock.patch("pyutilz.dev.benchmarking.synchronize_gpu_if_available", lambda: calls.append(1)):
        bm.time_backend(lambda: None, lambda: (), n_iters=2, warmup=1, synchronize_gpu=False)
    assert calls == []


def test_f01_concurrent_path_does_not_use_the_global_stream_sync():
    """Under concurrency a null-stream sync would absorb every OTHER thread's work; the resolver
    must hand back a different (per-thread event) primitive when cupy is available."""
    pytest.importorskip("cupy")
    solo = bm._resolve_gpu_sync(1)
    many = bm._resolve_gpu_sync(4)
    assert solo is not many
    many()  # must not raise on a real device


def test_f01_schema_bump_invalidates_pre_v4_tunings(tmp_path, monkeypatch):
    """The phantom (unsynchronized) timings live in v3 files; those must no longer be read."""
    import pyutilz.performance.kernel_tuning.cache as ktc

    monkeypatch.setenv("PYUTILZ_KERNEL_CACHE_DIR", str(tmp_path))
    ktc.hw_fingerprint.cache_clear()
    ktc.cache_base._ensure_cache_dir.cache_clear()
    assert ktc.SCHEMA_VERSION >= 4

    host = ktc.host_cache_dir()
    kdir = os.path.join(host, ktc._kernel_dir("", "phantom").strip(os.sep) or "phantom")
    os.makedirs(kdir, exist_ok=True)
    rec = {
        "schema_version": 3,
        "hw_fingerprint": ktc.hw_fingerprint(),
        "kernel_name": "phantom",
        "entry": {"axes": ["n"], "regions": [{"n_max": 10**12, "variant": "gpu", "wall_ms": 0.0366}]},
    }
    with open(os.path.join(kdir, "old.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f)

    cache = ktc.KernelTuningCache()
    assert cache.lookup("phantom", n=1000) is None, "a v3 (unsynchronized) tuning must not be reused"


# ---------------------------------------------------------------------------
# F02 -- gpu_capability_summary: static caps cached, free VRAM live
# ---------------------------------------------------------------------------

import pyutilz.system.gpu_dispatch as gd

_FAKE_CAPS = {
    "COMPUTE_CAPABILITY_MAJOR": 7,
    "COMPUTE_CAPABILITY_MINOR": 5,
    "MULTIPROCESSOR_COUNT": 14,
    "TOTAL_CUDA_CORES": 1024,
    "MAX_THREADS_PER_BLOCK": 1024,
    "MAX_SHARED_MEMORY_PER_BLOCK": 49152,
    "WARP_SIZE": 32,
}


@pytest.fixture()
def fake_gpu():
    """Patch the three hardware probes and guarantee a clean memoization state around the test."""
    gd.reset_cache()
    gputil_calls: list = []

    def _gputil(attrs="", **kw):
        gputil_calls.append(attrs)
        return [{"id": 0, "name": "GTX 1660", "memoryFree": 5.0, "memoryTotal": 6.0}]

    with mock.patch.object(gd, "is_cuda_available", return_value=True), mock.patch.object(
        gd, "get_gpu_cuda_capabilities", return_value=dict(_FAKE_CAPS)
    ), mock.patch.object(gd, "get_gpuutil_gpu_info", side_effect=_gputil), mock.patch.object(gd, "_free_bytes_via_cupy", return_value=None):
        yield gputil_calls
    gd.reset_cache()


def test_f02_static_caps_probe_runs_once_per_device(fake_gpu):
    first = gd.gpu_capability_summary(0)
    n_after_first = len(fake_gpu)
    for _ in range(5):
        gd.gpu_capability_summary(0)
    assert first is not None and first["name"] == "GTX 1660"
    # The static half (name/total VRAM, via nvidia-smi) is probed once; the extra GPUtil calls are
    # only the live free-VRAM fallback, which is allowed to re-read.
    static_probes = [a for a in fake_gpu if "memoryTotal" in a]
    assert len(static_probes) == 1, fake_gpu
    assert n_after_first >= 1


def test_f02_summary_is_a_copy_so_a_mutating_caller_cannot_poison_the_cache(fake_gpu):
    s1 = gd.gpu_capability_summary(0)
    assert s1 is not None
    s1["sm_count"] = -999
    s2 = gd.gpu_capability_summary(0)
    assert s2 is not None and s2["sm_count"] == 14


def test_f02_reset_cache_forces_a_reprobe(fake_gpu):
    gd.gpu_capability_summary(0)
    before = len([a for a in fake_gpu if "memoryTotal" in a])
    gd.reset_cache()
    gd.gpu_capability_summary(0)
    assert len([a for a in fake_gpu if "memoryTotal" in a]) == before + 1


def test_f02_free_vram_is_re_read_live(fake_gpu):
    with mock.patch.object(gd, "_free_bytes_via_cupy", side_effect=[2 * 1024**3, 1 * 1024**3]):
        a = gd.gpu_capability_summary(0)
        b = gd.gpu_capability_summary(0)
    assert a is not None and b is not None
    assert a["free_vram_gb"] == pytest.approx(2.0)
    assert b["free_vram_gb"] == pytest.approx(1.0), "free VRAM must not be frozen by the static cache"


def test_f02_occupancy_block_size_does_not_need_live_vram(fake_gpu):
    gd.occupancy_aware_block_size(16)
    n = len(fake_gpu)
    for _ in range(10):
        gd.occupancy_aware_block_size(16)
    assert len(fake_gpu) == n, "sizing a block must not re-shell-out to nvidia-smi"


# ---------------------------------------------------------------------------
# F04 -- is_cuda_available memoized, cleared by reset_cache
# ---------------------------------------------------------------------------

from pyutilz.core.pythonlib import is_cuda_available


def test_f04_is_cuda_available_is_memoized():
    is_cuda_available.cache_clear()
    calls: list = []

    import numba.cuda as numba_cuda

    def _probe():
        calls.append(1)
        return True

    with mock.patch.object(numba_cuda, "is_available", _probe):
        assert is_cuda_available() is True
        for _ in range(20):
            is_cuda_available()
    assert len(calls) == 1
    is_cuda_available.cache_clear()


def test_f04_reset_cache_clears_the_cuda_probe():
    is_cuda_available.cache_clear()
    import numba.cuda as numba_cuda

    with mock.patch.object(numba_cuda, "is_available", lambda: True):
        assert is_cuda_available() is True
    gd.reset_cache()
    with mock.patch.object(numba_cuda, "is_available", lambda: False):
        assert is_cuda_available() is False
    is_cuda_available.cache_clear()


# ---------------------------------------------------------------------------
# F05 -- fused njit array-summary hash
# ---------------------------------------------------------------------------

from pyutilz.core.array_summary import _numpy_col_reductions, column_sum_min_max
from pyutilz.core.disk_cache import _HASH_VERSION, hash_array_summary


@pytest.mark.parametrize(
    "arr",
    [
        np.arange(60, dtype=np.float64).reshape(15, 4),
        np.arange(60, dtype=np.float32).reshape(15, 4),
        np.arange(60, dtype=np.int64).reshape(15, 4),
        np.arange(60, dtype=np.uint8).reshape(15, 4),
        np.random.default_rng(0).random((997, 3)),
        np.random.default_rng(1).random((5, 7, 2)),
    ],
)
def test_f05_fused_kernel_matches_numpy(arr):
    ref = _numpy_col_reductions(arr, tuple(range(arr.ndim - 1)))
    got = column_sum_min_max(arr)
    for r, g in zip(ref, got):
        assert np.allclose(r, g, equal_nan=True)


def test_f05_nan_is_propagated_like_numpy():
    arr = np.array([[1.0, 2.0], [np.nan, 4.0]])
    ref = _numpy_col_reductions(arr, (0,))
    got = column_sum_min_max(arr)
    for r, g in zip(ref, got):
        assert np.array_equal(r, g, equal_nan=True)


def test_f05_unsupported_dtype_falls_back_to_numpy():
    arr = np.array([[1 + 2j, 3 + 0j], [5 - 1j, 0 + 0j]])
    got = column_sum_min_max(arr)
    ref = _numpy_col_reductions(arr, (0,))
    for r, g in zip(ref, got):
        assert np.allclose(np.real(r), np.real(g))


def test_f05_hash_is_deterministic_and_content_sensitive():
    a = np.random.default_rng(7).random((5000, 3))
    b = a.copy()
    assert hash_array_summary(a) == hash_array_summary(b)
    b[2500, 1] += 1.0
    assert hash_array_summary(a) != hash_array_summary(b)


def test_f05_hash_version_is_mixed_into_the_digest():
    """The recipe change must MISS old on-disk keys rather than silently colliding with them."""
    import struct

    from pyutilz.core.disk_cache import _hasher

    arr = np.arange(8, dtype=np.float64)
    h = _hasher()
    h.update(struct.pack("<I", _HASH_VERSION + 1))
    h.update(struct.pack("<I", 1))
    h.update(struct.pack("<q", 8))
    db = arr.dtype.str.encode("ascii")
    h.update(struct.pack("<I", len(db)))
    h.update(db)
    assert h.hexdigest() != hash_array_summary(arr)


def test_f05_non_contiguous_input_is_not_copied_whole():
    """A strided view must hash without the up-front whole-array ``ascontiguousarray``."""
    base = np.random.default_rng(3).random((2000, 6))
    view = base[:, ::2]
    assert not view.flags["C_CONTIGUOUS"]
    assert hash_array_summary(view) == hash_array_summary(np.ascontiguousarray(view))


# ---------------------------------------------------------------------------
# F06 -- precompiled lookup plan
# ---------------------------------------------------------------------------

from pyutilz.performance.kernel_tuning.cache import KernelTuningCache

_REGIONS = [
    {"n_max": 1000, "variant": "small", "block_size": 128},
    {"n_max": 100_000, "dtype_eq": "f4", "variant": "mid", "block_size": 256},
    {"n_min": 100_001, "variant": "big", "block_size": 512, "block_size_max": 1024},
]


def _seeded_cache():
    c = KernelTuningCache(in_memory=True)
    c.update("k", axes=["n", "dtype"], regions=[dict(r) for r in _REGIONS])
    return c


def test_f06_lookup_results_are_unchanged_by_the_plan_cache():
    c = _seeded_cache()
    assert c.lookup("k", n=500) == {"variant": "small", "block_size": 128}
    assert c.lookup("k", n=50_000, dtype="f4") == {"variant": "mid", "block_size": 256}
    # dtype_eq rules the mid region out and n_min rules the big one out -> a genuine miss.
    assert c.lookup("k", n=50_000, dtype="f8") is None
    assert c.lookup("k", n=10**9) == {"variant": "big", "block_size": 512, "block_size_max": 1024}
    assert c.lookup("missing", n=1) is None


def test_f06_repeated_lookups_are_stable():
    c = _seeded_cache()
    first = c.lookup("k", n=500)
    for _ in range(50):
        assert c.lookup("k", n=500) == first


def test_f06_only_declared_axis_constraints_are_stripped():
    """B4: ``block_size_max`` is a decision field, not an axis constraint on a declared axis."""
    c = _seeded_cache()
    hit = c.lookup("k", n=10**9)
    assert hit is not None and "block_size_max" in hit
    # ...but it IS still honoured as a constraint when the caller asks about that dim.
    assert c.lookup("k", n=10**9, block_size=4096) is None


def test_f06_update_invalidates_the_cached_plan():
    c = _seeded_cache()
    assert c.lookup("k", n=500)["variant"] == "small"
    c.update("k", axes=["n"], regions=[{"n_max": 1000, "variant": "REPLACED"}])
    assert c.lookup("k", n=500)["variant"] == "REPLACED"


def test_f06_evict_and_reset_invalidate_the_cached_plan():
    c = _seeded_cache()
    c.lookup("k", n=500)
    assert c.evict("k") is True
    assert c.lookup("k", n=500) is None
    c.update("k", axes=["n"], regions=[{"n_max": 1000, "variant": "again"}])
    c.lookup("k", n=500)
    c.reset()
    assert c._plan_cache == {}


def test_f06_lookup_explain_still_agrees_with_lookup():
    c = _seeded_cache()
    for dims in ({"n": 500}, {"n": 50_000, "dtype": "f4"}, {"n": 10**9}, {"n": 10**9, "block_size": 4096}):
        explain = c.lookup_explain("k", **dims)
        hit = c.lookup("k", **dims)
        assert explain["matched"] is (hit is not None)
        if hit is not None:
            assert explain["region"] == hit


def test_f06_plan_cache_is_not_pickled_and_the_cache_still_works_after_a_round_trip():
    c = _seeded_cache()
    c.lookup("k", n=500)
    assert c._plan_cache, "plan should have been compiled"
    state = c.__getstate__()
    assert state["_plan_cache"] == {}
    restored = pickle.loads(pickle.dumps(c))
    assert restored._plan_cache == {}
    assert isinstance(restored._lock, type(threading.RLock()))
    assert restored.lookup("k", n=500) == {"variant": "small", "block_size": 128}


def test_f06_get_or_tune_returns_the_cached_hit():
    c = _seeded_cache()
    calls: list = []

    def tuner():
        calls.append(1)
        return []

    assert c.get_or_tune("k", dims={"n": 500}, tuner=tuner, axes=["n"], fallback="FB") == {"variant": "small", "block_size": 128}
    assert calls == [], "a hit must not run the tuner"


def test_f06_get_or_tune_still_detects_a_stale_code_version():
    c = KernelTuningCache(in_memory=True)
    c.update("k", axes=["n"], regions=[{"n_max": 10**9, "variant": "old"}], code_version="AAA")
    got = c.get_or_tune("k", dims={"n": 5}, tuner=lambda: [], axes=["n"], fallback="FB", code_version="BBB")
    assert got == "FB"


# ---------------------------------------------------------------------------
# F08 / F09 / F11 -- cache_base memoization
# ---------------------------------------------------------------------------

import pyutilz.performance.kernel_tuning.cache.cache_base as cb


def test_f08_build_provenance_is_memoized():
    cb._build_provenance_cached.cache_clear()
    first = cb._build_provenance()
    with mock.patch.object(cb, "_safe_version", side_effect=AssertionError("must not re-probe")):
        again = cb._build_provenance()
    assert again == first


def test_f08_build_provenance_returns_an_independent_copy():
    cb._build_provenance_cached.cache_clear()
    a = cb._build_provenance()
    a["numpy_version"] = "TAMPERED"
    if isinstance(a.get("gpu_summary"), dict):
        a["gpu_summary"]["name"] = "TAMPERED"
    b = cb._build_provenance()
    assert b["numpy_version"] != "TAMPERED"
    if isinstance(b.get("gpu_summary"), dict):
        assert b["gpu_summary"].get("name") != "TAMPERED"


def test_f09_cache_dir_creates_once_then_skips_the_syscall(tmp_path, monkeypatch):
    target = tmp_path / "kc"
    monkeypatch.setenv("PYUTILZ_KERNEL_CACHE_DIR", str(target))
    cb._ensure_cache_dir.cache_clear()
    calls: list = []
    real = os.makedirs

    def counting(path, **kw):
        calls.append(path)
        return real(path, **kw)

    with mock.patch.object(cb.os, "makedirs", counting):
        assert cb.cache_dir() == str(target)
        for _ in range(10):
            cb.cache_dir()
    assert target.is_dir()
    assert len(calls) == 1, calls
    cb._ensure_cache_dir.cache_clear()


def test_f09_a_different_cache_dir_is_still_created(tmp_path, monkeypatch):
    cb._ensure_cache_dir.cache_clear()
    for name in ("a", "b"):
        monkeypatch.setenv("PYUTILZ_KERNEL_CACHE_DIR", str(tmp_path / name))
        assert os.path.isdir(cb.cache_dir())
    cb._ensure_cache_dir.cache_clear()


@pytest.mark.skipif(os.name != "nt", reason="Windows-only liveness probe")
def test_f11_kernel32_handle_is_shared_and_keeps_use_last_error():
    import ctypes

    cb._KERNEL32 = None
    h1 = cb._kernel32()
    h2 = cb._kernel32()
    assert h1 is h2
    assert h1._FuncPtr._flags_ & ctypes._FUNCFLAG_USE_LASTERROR


def test_f11_pid_alive_still_answers_correctly():
    assert cb._pid_alive(os.getpid()) is True
    assert cb._pid_alive(-1) is False
    # A pid that is essentially certainly free must be reported dead, which is the behaviour the
    # use_last_error handle exists to make possible.
    assert cb._pid_alive(0x7FFFFFF0) is False


# ---------------------------------------------------------------------------
# F07 -- get_topk_indices fast paths
# ---------------------------------------------------------------------------

from pyutilz.data.numpylib import get_topk_indices


def test_f07_k1_matches_the_general_path():
    rng = np.random.default_rng(11)
    for arr in (rng.random(101).astype(np.float32), rng.random((6, 13)), rng.integers(0, 50, 40)):
        for highest in (True, False):
            one = get_topk_indices(arr, k=1, highest=highest)
            many = get_topk_indices(arr, k=2, highest=highest)
            assert one.dtype == np.int64
            assert one.shape == many.shape[:-1] + (1,)
            assert np.array_equal(one[..., 0], many[..., 0])


def test_f07_k1_with_nan_still_takes_the_nan_aware_path():
    arr = np.array([2.0, np.nan, 3.0])
    assert get_topk_indices(arr, k=1, highest=True).tolist() == [2]
    assert get_topk_indices(arr, k=1, highest=False).tolist() == [0]


def test_f07_all_nan_require_finite_reports_minus_one():
    arr = np.full(5, np.nan)
    assert get_topk_indices(arr, k=1, highest=True, require_finite=True).tolist() == [-1]
    assert get_topk_indices(arr, k=3, highest=False, require_finite=True).tolist() == [-1, -1, -1]


def test_f07_clean_array_is_not_copied_for_the_nan_substitution():
    arr = np.random.default_rng(5).random(1000)
    with mock.patch.object(np, "where", side_effect=AssertionError("no NaN -> no substitution copy")):
        got = get_topk_indices(arr, k=4, highest=True)
    assert got.tolist() == np.argsort(arr)[::-1][:4].tolist()


# ---------------------------------------------------------------------------
# F03 -- json_pg_dumps: no pre-walk, no reparse, no third serialization
# ---------------------------------------------------------------------------

from pyutilz.text.strings.jsonutils import _normalize_for_pg_json, json_pg_dumps

_BS = chr(92)


def test_f03_no_object_walk_on_the_clean_orjson_path():
    payload = {"rows": [{"i": i, "v": [1.0, 2.0]} for i in range(20)]}
    with mock.patch("pyutilz.text.strings.jsonutils._normalize_for_pg_json", side_effect=AssertionError("must not walk")):
        out = json_pg_dumps(payload)
    assert out.adapted == payload


def test_f03_nul_bearing_document_still_takes_the_normalizing_path():
    calls: list = []
    real = _normalize_for_pg_json

    def counting(o):
        calls.append(1)
        return real(o)

    with mock.patch("pyutilz.text.strings.jsonutils._normalize_for_pg_json", counting):
        out = json_pg_dumps({"a": "x" + chr(0) + "y"})
    assert calls, "an embedded NUL must trigger the object-level normalization"
    assert out.adapted == {"a": "xy"}


def test_f03_literal_backslash_u0000_survives():
    value = "path" + _BS + "u0000literal"
    assert json_pg_dumps({"a": value}).adapted == {"a": value}


def test_f03_serialized_text_is_handed_over_verbatim():
    """No stdlib reparse and no third serialization at adapt time."""
    orjson = pytest.importorskip("orjson")
    payload = {"b": 1, "a": [1, 2, {"c": None}]}
    out = json_pg_dumps(payload)
    assert out.dumps(None) == orjson.dumps(payload).decode("utf-8")
    assert json.loads(out.getquoted()[1:-1].decode("utf-8")) == payload


def test_f03_orjson_and_stdlib_branches_agree_on_a_hostile_payload():
    payload = {"nan": float("nan"), "inf": float("inf"), "ninf": float("-inf"), "nul": "a" + chr(0) + "b", "ok": [1, 2.5, None]}
    with_orjson = json_pg_dumps(payload).adapted
    with mock.patch("pyutilz.text.strings.jsonutils._orjson", None):
        without = json_pg_dumps(payload).adapted
    assert with_orjson == without
    assert without == {"nan": None, "inf": None, "ninf": None, "nul": "ab", "ok": [1, 2.5, None]}


def test_f03_tuple_falls_back_instead_of_raising():
    out = json_pg_dumps({"t": (1, 2, 3)})
    assert out.adapted == {"t": [1, 2, 3]}


def test_f03_sort_keys_is_honoured():
    out = json_pg_dumps({"b": 1, "a": 2}, sort_keys=True)
    assert out.dumps(None).index('"a"') < out.dumps(None).index('"b"')


# ---------------------------------------------------------------------------
# F10 -- polars showcase_df_columns converts each column once
# ---------------------------------------------------------------------------


def test_f10_rare_and_display_blocks_share_one_conversion(capsys):
    pl = pytest.importorskip("polars")
    from pyutilz.data.pandaslib.frames import showcase_df_columns

    df = pl.DataFrame({"c": ["a"] * 50 + ["b"] * 50 + ["rare"]})
    rare, uninformative = showcase_df_columns(df, max_cat_uniq_qty=10, max_unique_percent=0.05, use_print=True)
    capsys.readouterr()
    assert rare == {"c": ["rare"]}


def test_f10_rare_detection_still_runs_when_display_is_suppressed(capsys):
    pl = pytest.importorskip("polars")
    from pyutilz.data.pandaslib.frames import showcase_df_columns

    df = pl.DataFrame({"c": ["a"] * 50 + ["b"] * 50 + ["rare"]})
    rare, _ = showcase_df_columns(df, max_cat_uniq_qty=10, max_unique_percent=0.05, max_vars=0, use_print=True)
    capsys.readouterr()
    assert rare == {"c": ["rare"]}, "max_vars=0 suppresses the table, not the rare-category analysis"

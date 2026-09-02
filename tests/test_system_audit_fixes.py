"""Regression tests for the 2026-09-02 core/dev/system domain audit -- system/ findings."""

from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from pyutilz.system.config import TomlLiveConfig
from pyutilz.system.hardware_monitor import UtilizationMonitor
from pyutilz.system.parallel import (
    applyfunc_parallel,
    mem_map_array,
    split_list_into_chunks_indices,
    split_list_into_nchunks_indices,
)
from pyutilz.system.resilience import CircuitBreaker, CircuitOpenError, DeadLetterQueue
from pyutilz.system.single_flight_cache import SingleFlightCache

# -------------------- F07: CircuitBreaker consecutive-failure semantics ------


def test_circuit_stays_closed_under_alternating_failures():
    """Never two failures in a row must never trip a breaker documented as consecutive-based."""
    cb = CircuitBreaker("t", failure_threshold=5, half_open_max_calls=10)
    state = {"i": 0}

    @cb.protect
    def flaky():
        state["i"] += 1
        if state["i"] % 2:
            raise RuntimeError("transient")
        return "ok"

    for _ in range(40):
        try:
            flaky()
        except RuntimeError:
            pass
    assert cb.state.is_open is False
    assert cb.state.failure_count <= 1


def test_circuit_opens_on_genuinely_consecutive_failures():
    cb = CircuitBreaker("t2", failure_threshold=3)

    @cb.protect
    def always_bad():
        raise RuntimeError("down")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            always_bad()
    assert cb.state.is_open is True
    with pytest.raises(CircuitOpenError):
        always_bad()


# -------------------- F24: DeadLetterQueue bound --------------------


def test_dead_letter_queue_rejects_a_non_positive_bound():
    with pytest.raises(ValueError):
        DeadLetterQueue(max_size=0)
    with pytest.raises(ValueError):
        DeadLetterQueue(max_size=-2)


def test_dead_letter_queue_honours_its_bound():
    q = DeadLetterQueue(max_size=2)
    for i in range(5):
        q.add("op", {"i": i}, RuntimeError("x"))
    assert q.size() == 2


# -------------------- F14/F23/F22: parallel splitting boundaries -------------


def test_split_list_into_nchunks_indices_more_chunks_than_items():
    """No empty bands: every yielded band must be non-degenerate."""
    bands = list(split_list_into_nchunks_indices([1, 2, 3], 5))
    assert bands == [(0, 1), (1, 2), (2, 3)]


def test_split_list_into_nchunks_indices_rejects_zero():
    with pytest.raises(ValueError):
        list(split_list_into_nchunks_indices([1, 2, 3], 0))


def test_split_list_into_nchunks_indices_even_partition_unchanged():
    assert list(split_list_into_nchunks_indices(list(range(10)), 3)) == [(0, 3), (3, 6), (6, 10)]


def test_split_list_into_chunks_indices_clamps_zero_chunk_size():
    """Matches the value-returning sibling, which clamps chunk_size 0 to 1."""
    assert list(split_list_into_chunks_indices([1, 2, 3], 0)) == [(0, 1), (1, 2), (2, 3)]


def test_applyfunc_parallel_on_an_empty_iterable():
    assert applyfunc_parallel([], lambda x: x, return_dataframe=False) == []
    assert applyfunc_parallel([], lambda x: x, return_dataframe=True).empty


# -------------------- F13: mem_map_array temp-dir reuse --------------------


def test_mem_map_array_reuses_one_temp_directory():
    from pyutilz.system import parallel

    arr = np.arange(16, dtype=np.float32)
    before = len(parallel._TEMP_DIRS)
    for _ in range(3):
        mem_map_array(arr, "reuse_probe")
    assert len(parallel._TEMP_DIRS) - before <= 1


# -------------------- F11/F18/F19: TomlLiveConfig --------------------


def _write(p, text):
    p.write_text(text, encoding="utf-8")


def test_config_survives_an_unreadable_file(tmp_path, monkeypatch):
    """PermissionError during an editor's save window must not crash the pipeline."""
    p = tmp_path / "cfg.toml"
    _write(p, "[limits]\nmax_retries = 3\n")
    cfg = TomlLiveConfig(p, check_interval=0.0)
    assert cfg.get("limits", "max_retries") == 3

    def boom(self):
        raise PermissionError(32, "sharing violation")

    monkeypatch.setattr(Path, "read_bytes", boom)
    later = time.time() + 10
    os.utime(p, (later, later))
    assert cfg.get("limits", "max_retries") == 3  # previous value kept, no raise


def test_config_warns_when_truncating_a_float_to_int(tmp_path, caplog):
    p = tmp_path / "cfg.toml"
    _write(p, "[http]\ntimeout_sec = 0.5\n")
    cfg = TomlLiveConfig(p, check_interval=0.0)
    with caplog.at_level("WARNING"):
        assert cfg.get("http", "timeout_sec") == 0
    assert any("truncated to int" in r.getMessage() for r in caplog.records)
    assert cfg.get("http", "timeout_sec", type_=float) == 0.5


def test_get_section_returns_a_copy_and_never_mutates_defaults(tmp_path):
    p = tmp_path / "cfg.toml"
    _write(p, "[db]\nhost = 'a'\n")
    defaults = {"limits": {"max_workers": 4}}
    cfg = TomlLiveConfig(p, defaults=defaults, check_interval=0.0)

    sect = cfg.get_section("limits")
    sect["max_workers"] = 32
    assert defaults["limits"]["max_workers"] == 4
    assert cfg.get_section("limits")["max_workers"] == 4

    live = cfg.get_section("db")
    live["host"] = "override"
    assert cfg.get_section("db")["host"] == "a"


def test_data_property_returns_a_copy(tmp_path):
    p = tmp_path / "cfg.toml"
    _write(p, "[db]\nhost = 'a'\n")
    cfg = TomlLiveConfig(p, check_interval=0.0)
    d = cfg.data
    d["injected"] = 1
    assert "injected" not in cfg.data


def test_empty_section_in_file_is_honoured_over_defaults(tmp_path):
    p = tmp_path / "cfg.toml"
    _write(p, "[db]\n")
    cfg = TomlLiveConfig(p, defaults={"db": {"host": "fallback"}}, check_interval=0.0)
    assert cfg.get_section("db") == {}


# -------------------- F05/F20: UtilizationMonitor lifecycle --------------------


def test_monitor_survives_a_failing_sample(monkeypatch):
    m = UtilizationMonitor(sleep_interval_seconds=0.01)
    calls = {"n": 0}
    real = m._collect_sample

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("bad nvidia-smi field")
        return real()

    monkeypatch.setattr(m, "_collect_sample", flaky)
    m.start()
    time.sleep(1.5)
    m.stop(timeout=10)
    assert calls["n"] >= 2, calls  # the thread survived the failing sample
    assert m.n_sampling_errors == 1
    assert m.get_average_utilization()["n_sampling_errors"] == 1


def test_monitor_is_restartable_as_a_context_manager():
    m = UtilizationMonitor(sleep_interval_seconds=0.01)
    with m:
        time.sleep(0.5)
    with m:  # must not raise "threads can only be started once"
        time.sleep(0.5)
    assert m.get_average_utilization()["n_samples"] > 0


def test_monitor_samples_immediately_for_a_short_run():
    """The first sample used to land only one full interval after start()."""
    m = UtilizationMonitor(sleep_interval_seconds=30)
    m.start()
    time.sleep(1.5)
    m.stop(timeout=10)
    assert m.get_average_utilization()["n_samples"] >= 1


# -------------------- F16/F17: SingleFlightCache --------------------


def test_single_flight_cache_works_on_a_non_main_thread_event_loop():
    """One event loop on a dedicated worker thread satisfies the documented contract."""
    sfc = SingleFlightCache()
    cache = {}
    result = {}

    async def main():
        async def fetch():
            return 42

        result["v"] = await sfc.get_or_fetch(cache, "k", fetch)

    t = threading.Thread(target=lambda: asyncio.run(main()))
    t.start()
    t.join()
    assert result["v"] == 42


def test_single_flight_cache_rejects_a_second_event_loop():
    sfc = SingleFlightCache()

    async def use():
        async def fetch():
            return 1

        return await sfc.get_or_fetch({}, "k", fetch)

    asyncio.run(use())
    with pytest.raises(RuntimeError):
        asyncio.run(use())


class _ExpireBetweenLookups(dict):
    """Reports a key as present, then raises KeyError on the very next __getitem__."""

    def __contains__(self, key):
        return True

    def __getitem__(self, key):
        raise KeyError(key)


def test_single_flight_cache_uses_one_lookup_per_probe():
    """A TTL boundary between `in` and `[]` must not surface as KeyError."""
    sfc = SingleFlightCache()
    cache = _ExpireBetweenLookups()

    async def main():
        async def fetch():
            return "fresh"

        return await sfc.get_or_fetch(cache, "k", fetch)

    assert asyncio.run(main()) == "fresh"


# -------------------- F15: register_scraper failed node insert --------------------


def test_register_scraper_raises_when_the_nodes_table_yields_no_id(monkeypatch):
    """Returning None left the caller with an unregistered, never-heartbeating scraper."""
    import pyutilz.system.distributed as distributed_module

    with distributed_module._identity_lock:
        distributed_module._container.node_id = None

    monkeypatch.setattr(distributed_module.system, "get_system_info", lambda only_stats=False: {"host_name": "h"})
    monkeypatch.setattr(distributed_module.db, "db_command", lambda *a, **kw: None)
    monkeypatch.setattr(distributed_module.web, "get_external_ip", lambda: "1.2.3.4")

    with pytest.raises(RuntimeError, match="NOT registered"):
        distributed_module.register_scraper("worker-3", version="1.0", app_name="app", ip="1.2.3.4")


# -------------------- F36/F37: prefect scheduling helpers --------------------


def _prefect_module():
    """Import the helper module, stubbing the optional `prefect` client package if it is absent.

    Neither helper under test touches the client, so a stub keeps these behavioural checks running
    on hosts without the (heavy) prefect install instead of silently skipping.
    """
    import sys
    import types

    if "prefect" not in sys.modules:
        try:
            import prefect  # noqa: F401
        except ImportError:
            sys.modules["prefect"] = types.ModuleType("prefect")
    from pyutilz.system.scheduling import prefect as prefect_helpers

    return prefect_helpers


def test_get_running_flows_applies_both_label_filters(monkeypatch):
    prefect_helpers = _prefect_module()
    flows = [
        {"id": "f1", "flow_runs": [{"id": "r1", "labels": ["gpu", "dev"]}]},
        {"id": "f2", "flow_runs": [{"id": "r2", "labels": ["gpu", "ml", "production"]}]},
    ]
    monkeypatch.setattr(prefect_helpers, "get_flows_and_runs", lambda status=None, **kw: flows)
    got = prefect_helpers.get_running_flows(anyof_labels={"gpu"}, allof_labels={"ml", "production"})
    assert [f["id"] for f in got] == ["f2"]


def test_wait_for_absense_of_tasks_sleeps_at_most_max_retries_times(monkeypatch):
    """Guards the documented ceiling: max_retries sleeps, one final check after the last one."""
    prefect_helpers = _prefect_module()
    sleeps = []
    monkeypatch.setattr(prefect_helpers, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(prefect_helpers, "get_running_flows", lambda **kw: [{"id": "f"}])
    assert prefect_helpers.wait_for_absense_of_tasks(max_retries=3, sleep_seconds=1) is False
    assert sleeps == [1, 1, 1]

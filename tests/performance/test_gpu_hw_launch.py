"""Unit coverage for ``pyutilz.performance.gpu_hw_launch``'s analytic occupancy / launch-config derivation.

X_TEST_COVERAGE_QUALITY-6 fix (mrmr_audit_2026-07-22): this module had zero test references anywhere in
the suite. Exercises the pure-Python occupancy math (``_max_active_blocks_per_sm``,
``occupancy_block_candidates``, ``fill_grid_1d``) against synthetic device-property dicts, plus
``device_props``'s no-cupy fallback -- no real CUDA device is required for any of this.
"""

from __future__ import annotations

from pyutilz.performance import gpu_hw_launch as hw_launch

# A modest, realistic device (a mid-range consumer card's rough shape).
_PROPS = {
    "sm_count": 20,
    "max_threads_per_block": 1024,
    "max_threads_per_sm": 2048,
    "max_blocks_per_sm": 16,
    "warp": 32,
    "shared_per_block": 48 * 1024,
    "regs_per_block": 65536,
    "regs_per_sm": 65536,
    "shared_per_sm": 96 * 1024,
}


def teardown_function():
    """Never leak the module-level device-property cache into another test."""
    hw_launch._DEV_PROPS = None


def test_device_props_empty_dict_when_cupy_unavailable(monkeypatch):
    """When ``import cupy`` fails (no CUDA build / no device), ``device_props`` degrades to an empty
    dict rather than raising -- callers treat that as 'no HW info' and keep their historical default."""
    hw_launch._DEV_PROPS = None
    real_import = __import__

    def _fake_import(name, *args, **kwargs):
        """Raise on ``import cupy``, delegate every other import to the real one."""
        if name == "cupy":
            raise ImportError("no cupy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)
    assert hw_launch.device_props() == {}


def test_device_props_cached_across_calls(monkeypatch):
    """The device-property query only runs once per process; a second call returns the same cached dict
    object without re-querying (keyed nothing -- single current device)."""
    hw_launch._DEV_PROPS = None
    hw_launch._DEV_PROPS = dict(_PROPS)
    first = hw_launch.device_props()
    second = hw_launch.device_props()
    assert first is second


def test_max_active_blocks_infeasible_block_size_returns_zero():
    """A block larger than the device's per-block thread cap is infeasible -- 0 active blocks/SM, not a
    negative or nonsensical value."""
    assert hw_launch._max_active_blocks_per_sm(2048, 32, 0, 0, _PROPS) == 0


def test_max_active_blocks_infeasible_shared_mem_returns_zero():
    """Static+dynamic shared memory exceeding the per-block shared cap is infeasible regardless of block
    size or register pressure."""
    assert hw_launch._max_active_blocks_per_sm(128, 8, 64 * 1024, 0, _PROPS) == 0


def test_max_active_blocks_decreases_with_register_pressure():
    """Heavier per-thread register use must never INCREASE the analytic max-active-blocks/SM for the same
    block size -- more registers/thread can only tighten (or leave unchanged) the SM's block budget."""
    light = hw_launch._max_active_blocks_per_sm(128, 16, 0, 0, _PROPS)
    heavy = hw_launch._max_active_blocks_per_sm(128, 128, 0, 0, _PROPS)
    assert heavy <= light
    assert light > 0


def test_occupancy_block_candidates_empty_without_device_props(monkeypatch):
    """No HW info (empty ``device_props``) means the caller must keep its own historical sweep/default --
    this function returns an empty list rather than guessing."""
    monkeypatch.setattr(hw_launch, "device_props", lambda: {})
    assert hw_launch.occupancy_block_candidates() == []


def test_occupancy_block_candidates_are_warp_multiples_and_hw_valid(monkeypatch):
    """Every returned candidate is a strict warp-multiple within the device's per-block thread cap, and
    each independently satisfies the requested minimum active-blocks/SM."""
    monkeypatch.setattr(hw_launch, "device_props", lambda: _PROPS)
    candidates = hw_launch.occupancy_block_candidates(regs_per_thread=32, min_active_blocks=2)
    assert candidates
    for block in candidates:
        assert block % _PROPS["warp"] == 0
        assert block <= _PROPS["max_threads_per_block"]
        assert hw_launch._max_active_blocks_per_sm(block, 32, 0, 0, _PROPS) >= 2


def test_occupancy_block_candidates_respects_block_cap(monkeypatch):
    """An explicit ``block_cap`` narrower than the device's own limit must never be exceeded by any
    returned candidate."""
    monkeypatch.setattr(hw_launch, "device_props", lambda: _PROPS)
    candidates = hw_launch.occupancy_block_candidates(regs_per_thread=8, min_active_blocks=1, block_cap=256)
    assert candidates
    assert max(candidates) <= 256


def test_occupancy_block_candidates_falls_back_when_min_active_unreachable(monkeypatch):
    """When register/shared pressure is so severe that NO block size reaches ``min_active_blocks``, the
    function still returns the merely-feasible (>=1 active block) warp-multiples rather than an empty
    list -- the sweep must always have HW-valid options to try."""
    monkeypatch.setattr(hw_launch, "device_props", lambda: _PROPS)
    candidates = hw_launch.occupancy_block_candidates(regs_per_thread=32, min_active_blocks=999)
    assert candidates
    for block in candidates:
        assert hw_launch._max_active_blocks_per_sm(block, 32, 0, 0, _PROPS) >= 1


def test_fill_grid_1d_covers_small_work_without_device_props(monkeypatch):
    """With no HW info, the grid is simply enough blocks to cover the work (``ceil(n_work / block)``) --
    the historical behavior, unchanged when occupancy data is unavailable."""
    monkeypatch.setattr(hw_launch, "device_props", lambda: {})
    assert hw_launch.fill_grid_1d(1000, 128) == 8


def test_fill_grid_1d_fills_all_sms_for_small_work_on_big_device(monkeypatch):
    """For a large device and a tiny amount of work, the grid must still be big enough to occupy every SM
    (>= sm_count * max_active_blocks_per_SM), not just cover the (small) work itself."""
    monkeypatch.setattr(hw_launch, "device_props", lambda: _PROPS)
    grid = hw_launch.fill_grid_1d(10, 128, regs_per_thread=16)
    mab = hw_launch._max_active_blocks_per_sm(128, 16, 0, 0, _PROPS)
    assert grid >= _PROPS["sm_count"] * max(mab, 1)


def test_fill_grid_1d_dominated_by_work_when_it_exceeds_fill(monkeypatch):
    """For work far larger than the SM-filling minimum, the grid must be dominated by the actual coverage
    requirement, not clamped down to the (smaller) fill floor."""
    monkeypatch.setattr(hw_launch, "device_props", lambda: _PROPS)
    n_work = 10_000_000
    block = 128
    grid = hw_launch.fill_grid_1d(n_work, block, regs_per_thread=16)
    cover = (n_work + block - 1) // block
    assert grid == cover


def test_fill_grid_1d_invalid_block_returns_one():
    """A non-positive block size is a degenerate caller error -- return the safe minimum grid of 1 rather
    than raising or dividing by zero."""
    assert hw_launch.fill_grid_1d(1000, 0) == 1

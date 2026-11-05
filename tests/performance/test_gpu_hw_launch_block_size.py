"""``block_size_for`` and ``launch_1d`` derive the block and the grid from the device limits and the kernel's own registers and shared memory."""

from __future__ import annotations

import pytest

from pyutilz.performance import gpu_hw_launch as hw

# a 20-SM card with the usual per-SM limits
_PROPS = {
    "sm_count": 20,
    "max_threads_per_block": 1024,
    "max_threads_per_sm": 2048,
    "max_blocks_per_sm": 32,
    "warp": 32,
    "shared_per_block": 48 * 1024,
    "regs_per_block": 65536,
    "regs_per_sm": 65536,
    "shared_per_sm": 96 * 1024,
    "reg_alloc_unit": 256,
}


class _Kernel:
    """The two attributes of a compiled cupy kernel the occupancy formula reads."""

    def __init__(self, regs: int, shared: int = 0) -> None:
        self.num_regs = regs
        self.shared_size_bytes = shared


@pytest.fixture(autouse=True)
def _device(monkeypatch):
    monkeypatch.setattr(hw, "_DEV_PROPS", dict(_PROPS))
    monkeypatch.setattr(hw, "_DEV_PROPS_ID", None)
    monkeypatch.setattr(hw, "_current_device_id", lambda: None)


def test_a_light_kernel_gets_the_preferred_width_among_those_that_fill_the_sm():
    """With 32 registers 64..1024 all keep 2048 threads resident; the one closest to 256 is taken."""
    assert hw.block_size_for(_Kernel(32)) == 256


def test_a_register_heavy_kernel_gets_a_block_that_still_fits_the_sm():
    """At 128 registers per thread 8192 registers per warp-block limit the resident threads; the chosen width is a warp multiple that reaches two active blocks."""
    block = hw.block_size_for(_Kernel(128))
    assert block % 32 == 0 and hw._max_active_blocks_per_sm(block, 128, 0, 0, _PROPS) >= 2


def test_a_kernel_with_big_static_shared_memory_is_limited_by_it():
    """40 KB of static shared memory per block allows two blocks per SM at most; wider blocks put more threads on the SM, so the widest that keeps two blocks wins."""
    assert hw.block_size_for(_Kernel(32, 40 * 1024)) == 1024


def test_max_width_and_work_items_bound_the_choice():
    """A kernel written for at most 128 threads, or with 100 items of work, is never given more."""
    assert hw.block_size_for(_Kernel(32), max_width=128) == 128
    assert hw.block_size_for(_Kernel(32), work_items=100) == 128


def test_without_device_information_the_default_is_returned(monkeypatch):
    """No cupy / no device: the caller's default, so the old behaviour is unchanged."""
    monkeypatch.setattr(hw, "_DEV_PROPS", {})
    assert hw.block_size_for(_Kernel(32), default=192) == 192


def test_launch_1d_covers_the_work_and_fills_every_sm():
    """The grid covers n_work and is at least SM count x active blocks per SM."""
    (grid,), (block,) = hw.launch_1d(_Kernel(32), 10_000_000)
    assert block == 256 and grid == (10_000_000 + 255) // 256
    (small_grid,), (small_block,) = hw.launch_1d(_Kernel(32), 1000)
    assert small_block * small_grid >= 1000
    assert small_grid >= _PROPS["sm_count"]

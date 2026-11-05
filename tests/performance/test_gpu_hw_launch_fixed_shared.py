"""Launch geometry for kernels with FIXED shared memory per block, on synthetic device descriptions (no CUDA device needed)."""

from pyutilz.performance import gpu_hw_launch as hw

_PASCAL = {
    "sm_count": 6, "max_threads_per_block": 1024, "max_threads_per_sm": 2048, "max_blocks_per_sm": 32, "warp": 32,
    "shared_per_block": 49152, "regs_per_block": 65536, "regs_per_sm": 65536, "shared_per_sm": 98304, "reserved_shared_per_block": 0, "reg_alloc_unit": 256,
}
_AMPERE = {
    "sm_count": 108, "max_threads_per_block": 1024, "max_threads_per_sm": 2048, "max_blocks_per_sm": 32, "warp": 32,
    "shared_per_block": 49152, "regs_per_block": 65536, "regs_per_sm": 65536, "shared_per_sm": 167936, "reserved_shared_per_block": 1024, "reg_alloc_unit": 256,
}


def _use(monkeypatch, props):
    """Pretend the current device is ``props``."""
    monkeypatch.setattr(hw, "device_props", lambda: props)


def test_registers_are_charged_per_warp_in_allocation_units(monkeypatch):
    """53 registers per thread cost 56 (1792 per warp): 8 warps need 14336, so 4 blocks fit 65536, not the 4.6 of a per-thread count."""
    assert hw._max_active_blocks_per_sm(256, 53, 0, 0, _PASCAL) == 4
    assert hw._max_active_blocks_per_sm(256, 32, 0, 0, _PASCAL) == 8  # 32 registers: 1024 per warp, 8192 per block, 8 blocks (and 8 by threads)


def test_the_driver_reserve_counts_against_the_sm(monkeypatch):
    """With 1 KB reserved per block, 41 KB blocks leave room for 3 blocks on a 164 KB SM, not 4."""
    assert hw._max_active_blocks_per_sm(128, 32, 0, 40960, _AMPERE) == 4
    assert hw._max_active_blocks_per_sm(128, 32, 0, 40960, {**_AMPERE, "reserved_shared_per_block": 0}) == 4
    assert hw._max_active_blocks_per_sm(128, 32, 0, 41600, _AMPERE) == 3  # 41600 + 1024 = 42624; 167936 // 42624 = 3
    assert hw._max_active_blocks_per_sm(128, 32, 0, 41600, {**_AMPERE, "reserved_shared_per_block": 0}) == 4


def test_fixed_shared_width_follows_the_shared_table(monkeypatch):
    """A 25 KB table (n = 100000 bitset) holds 3 blocks on a Pascal SM: the chosen width keeps more threads resident than 128 or 256; a 2 KB table is limited by threads only."""
    _use(monkeypatch, _PASCAL)
    wide = hw.fixed_shared_block_size(25000, regs_per_thread=32)
    assert wide >= 512
    assert hw._max_active_blocks_per_sm(wide, 32, 0, 25000, _PASCAL) * wide > 3 * 256
    small = hw.fixed_shared_block_size(2048, regs_per_thread=32)
    assert small * hw._max_active_blocks_per_sm(small, 32, 0, 2048, _PASCAL) == 2048  # full occupancy reached
    assert hw.fixed_shared_block_size(25000, regs_per_thread=32, max_width=256) == 256


def test_fixed_shared_width_respects_the_work_and_the_limits(monkeypatch):
    """A block never exceeds the work one block can use, a table over the per-block limit has no feasible width, and no device means 0 (keep the constant)."""
    _use(monkeypatch, _PASCAL)
    assert hw.fixed_shared_block_size(4096, work_items=100) <= 128
    assert hw.fixed_shared_block_size(60000) == 0
    _use(monkeypatch, {})
    assert hw.fixed_shared_block_size(4096) == 0


def test_properties_are_requeried_when_the_current_device_changes(monkeypatch):
    """The cache holds one device's limits: a second, different current device gets its own."""
    first = dict(_PASCAL)
    monkeypatch.setattr(hw, "_DEV_PROPS", first)
    monkeypatch.setattr(hw, "_DEV_PROPS_ID", 0)
    monkeypatch.setattr(hw, "_current_device_id", lambda: 0)
    assert hw.device_props() is first
    calls = []

    def _fake_uncached():
        """Stands in for the driver query of device 1."""
        calls.append(1)
        hw._DEV_PROPS, hw._DEV_PROPS_ID = dict(_AMPERE), 1
        return hw._DEV_PROPS

    monkeypatch.setattr(hw, "_device_props_uncached", _fake_uncached)
    monkeypatch.setattr(hw, "_current_device_id", lambda: 1)
    assert hw.device_props()["sm_count"] == 108 and calls == [1]

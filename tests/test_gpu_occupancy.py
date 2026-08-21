"""`occupancy_aware_block_size` must maximise resident threads from stated device limits, not guess.

These run without a GPU: the function takes a `caps` dict, so the whole occupancy calculation is testable
against synthetic devices - which is the point, since the bug it prevents is a block width tuned on one
card being wrong on another.
"""

from __future__ import annotations

import pytest

from pyutilz.system.gpu_dispatch import CC_MAX_BLOCKS_PER_SM, occupancy_aware_block_size, query_cuda_device_attribute

# An Ada laptop part, as reported live by numba/cupy: 1536 threads, 100 KB shared and 24 blocks per SM.
ADA = {
    "warp_size": 32,
    "max_threads_per_block": 1024,
    "max_shared_mem_per_block": 49152,
    "max_threads_per_sm": 1536,
    "max_shared_mem_per_sm": 102400,
    "max_blocks_per_sm": 24,
    # The driver reserves this per block on top of whatever the kernel asks for; an occupancy calculation
    # that ignores it overcounts resident blocks (measured: 23 against the 17 the hardware really holds).
    "reserved_shared_mem_per_block": 1024,
}


def _is_power_of_two(value: int) -> bool:
    """Whether the width is safe for the classic halving tree reduction."""
    return value > 0 and value & (value - 1) == 0


def _resident_threads(caps: dict, threads: int, bytes_per_thread: int) -> int:
    """The occupancy the caller would actually get, computed independently of the function under test."""
    shared = bytes_per_thread * threads
    occupied = shared + caps.get("reserved_shared_mem_per_block", 0) if shared else 0
    by_shared = caps["max_shared_mem_per_sm"] // occupied if occupied else caps["max_blocks_per_sm"]
    return min(caps["max_blocks_per_sm"], caps["max_threads_per_sm"] // threads, by_shared) * threads


@pytest.mark.parametrize("bytes_per_thread", [0, 4, 8, 16, 32, 64, 128, 136, 256, 512])
def test_no_other_warp_multiple_holds_more_threads_resident(bytes_per_thread):
    """The real contract: whatever it returns, nothing else fits more threads on an SM."""
    threads, shared = occupancy_aware_block_size(bytes_per_thread, caps=ADA, power_of_two=False)
    assert threads % ADA["warp_size"] == 0
    assert shared == bytes_per_thread * threads
    assert shared <= ADA["max_shared_mem_per_block"]

    best = _resident_threads(ADA, threads, bytes_per_thread)
    candidate = ADA["warp_size"]
    while candidate <= ADA["max_threads_per_block"]:
        if bytes_per_thread * candidate <= ADA["max_shared_mem_per_block"]:
            assert _resident_threads(ADA, candidate, bytes_per_thread) <= best
        candidate += ADA["warp_size"]


def test_a_cheap_kernel_fills_the_sm_completely():
    """With shared memory not binding, the answer must reach the device's own threads-per-SM ceiling."""
    threads, _shared = occupancy_aware_block_size(8, caps=ADA)
    assert _resident_threads(ADA, threads, 8) == ADA["max_threads_per_sm"]


def test_a_greedier_kernel_gets_a_narrower_block_not_a_refusal():
    """Shared memory per thread rising must narrow the block, never return something that cannot launch."""
    wide, _ = occupancy_aware_block_size(8, caps=ADA)
    narrow, narrow_shared = occupancy_aware_block_size(1024, caps=ADA)
    assert narrow < wide
    assert narrow >= ADA["warp_size"]
    assert narrow_shared <= ADA["max_shared_mem_per_block"]


def test_two_devices_with_different_limits_get_different_answers():
    """The reason this exists: one card's tuned constant is another card's wrong answer.

    Stated as "there is SOME workload where they differ", not "they differ at one chosen size" - two cards
    genuinely do agree on plenty of widths, and an assertion that forbade that would be pinning a
    coincidence. An earlier version picked 128 bytes/thread and broke the moment the power-of-two constraint
    landed and made both devices answer 256.
    """
    turing = {**ADA, "max_threads_per_sm": 1024, "max_shared_mem_per_sm": 65536, "max_blocks_per_sm": 16}
    assert any(occupancy_aware_block_size(b, caps=ADA) != occupancy_aware_block_size(b, caps=turing) for b in (8, 16, 32, 64, 128, 256, 512))


def test_missing_limits_fall_back_to_one_warp_rather_than_dividing_by_zero():
    """A CPU-only host hands in an empty dict; the caller still needs a launchable width back."""
    threads, shared = occupancy_aware_block_size(16, caps={})
    assert threads == 32
    assert shared == 16 * 32


def test_the_blocks_per_sm_table_covers_every_cc_the_shared_mem_table_does():
    """Both tables are keyed by compute capability and consulted together - a gap in one is a silent default."""
    from pyutilz.system.gpu_dispatch import CC_SHARED_MEM_BUDGET

    assert set(CC_SHARED_MEM_BUDGET) <= set(CC_MAX_BLOCKS_PER_SM)


@pytest.mark.parametrize("bytes_per_thread", [0, 4, 8, 16, 32, 64, 128, 136, 256, 512])
def test_the_default_is_a_power_of_two_because_tree_reductions_need_one(bytes_per_thread):
    """`for (s = blockDim.x / 2; s > 0; s >>= 1)` drops elements at any other width - silently, and wrongly.

    Caught by a real caller's tests, not by inspection: a warp-multiple answer of 160 threads made a fused
    reduction kernel disagree with its own CPU reference.
    """
    threads, _shared = occupancy_aware_block_size(bytes_per_thread, caps=ADA)
    assert _is_power_of_two(threads)


@pytest.mark.parametrize("bytes_per_thread", [0, 8, 128, 136, 512])
def test_no_larger_power_of_two_holds_more_threads_resident(bytes_per_thread):
    """The same optimality contract as the unconstrained case, over the powers of two alone."""
    threads, _shared = occupancy_aware_block_size(bytes_per_thread, caps=ADA)
    best = _resident_threads(ADA, threads, bytes_per_thread)
    candidate = ADA["warp_size"]
    while candidate <= ADA["max_threads_per_block"]:
        if bytes_per_thread * candidate <= ADA["max_shared_mem_per_block"]:
            assert _resident_threads(ADA, candidate, bytes_per_thread) <= best
        candidate *= 2


def test_relaxing_the_power_of_two_constraint_can_only_help_occupancy():
    """A wider search space cannot return a worse answer - pins that the two modes stay consistent."""
    for bytes_per_thread in (8, 16, 64, 128, 256):
        strict, _ = occupancy_aware_block_size(bytes_per_thread, caps=ADA)
        relaxed, _ = occupancy_aware_block_size(bytes_per_thread, caps=ADA, power_of_two=False)
        assert _resident_threads(ADA, relaxed, bytes_per_thread) >= _resident_threads(ADA, strict, bytes_per_thread)


def test_the_driver_reservation_lowers_the_resident_block_count():
    """Pins that the reservation is actually consulted - ignoring it silently overcounts occupancy."""
    without = {**ADA, "reserved_shared_mem_per_block": 0}
    assert _resident_threads(without, 32, 136) > _resident_threads(ADA, 32, 136)
    assert occupancy_aware_block_size(136, caps=ADA) != occupancy_aware_block_size(136, caps=without)


def test_querying_a_driver_attribute_numba_does_not_define():
    """`MAX_BLOCKS_PER_MULTIPROCESSOR` has no `cudadrv.enums` entry, which is why this helper exists.

    Skipped rather than failed without a device: the point is that it needs neither cupy nor a hardcoded
    table, not that CI has a GPU.
    """
    value = query_cuda_device_attribute("MAX_BLOCKS_PER_MULTIPROCESSOR")
    if value is None:
        pytest.skip("no CUDA device on this host")
    assert value > 0
    assert query_cuda_device_attribute(106) == value  # by numeric code too
    assert query_cuda_device_attribute("CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR") == value


def test_an_unknown_attribute_name_returns_none_rather_than_raising():
    """A capability probe must degrade, never break the caller that was only tuning a launch."""
    assert query_cuda_device_attribute("NO_SUCH_ATTRIBUTE_WHATSOEVER") is None

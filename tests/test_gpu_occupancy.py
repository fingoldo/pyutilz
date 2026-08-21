"""`occupancy_aware_block_size` must maximise resident threads from stated device limits, not guess.

These run without a GPU: the function takes a `caps` dict, so the whole occupancy calculation is testable
against synthetic devices - which is the point, since the bug it prevents is a block width tuned on one
card being wrong on another.
"""

from __future__ import annotations

import pytest

from pyutilz.system.gpu_dispatch import CC_MAX_BLOCKS_PER_SM, occupancy_aware_block_size

# An Ada laptop part, as reported live by numba/cupy: 1536 threads, 100 KB shared and 24 blocks per SM.
ADA = {
    "warp_size": 32,
    "max_threads_per_block": 1024,
    "max_shared_mem_per_block": 49152,
    "max_threads_per_sm": 1536,
    "max_shared_mem_per_sm": 102400,
    "max_blocks_per_sm": 24,
}


def _resident_threads(caps: dict, threads: int, bytes_per_thread: int) -> int:
    """The occupancy the caller would actually get, computed independently of the function under test."""
    shared = bytes_per_thread * threads
    by_shared = caps["max_shared_mem_per_sm"] // shared if shared else caps["max_blocks_per_sm"]
    return min(caps["max_blocks_per_sm"], caps["max_threads_per_sm"] // threads, by_shared) * threads


@pytest.mark.parametrize("bytes_per_thread", [0, 4, 8, 16, 32, 64, 128, 136, 256, 512])
def test_no_other_warp_multiple_holds_more_threads_resident(bytes_per_thread):
    """The real contract: whatever it returns, nothing else fits more threads on an SM."""
    threads, shared = occupancy_aware_block_size(bytes_per_thread, caps=ADA)
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
    """The reason this exists: one card's tuned constant is another card's wrong answer."""
    turing = {**ADA, "max_threads_per_sm": 1024, "max_shared_mem_per_sm": 65536, "max_blocks_per_sm": 16}
    assert occupancy_aware_block_size(128, caps=ADA) != occupancy_aware_block_size(128, caps=turing)


def test_missing_limits_fall_back_to_one_warp_rather_than_dividing_by_zero():
    """A CPU-only host hands in an empty dict; the caller still needs a launchable width back."""
    threads, shared = occupancy_aware_block_size(16, caps={})
    assert threads == 32
    assert shared == 16 * 32


def test_the_blocks_per_sm_table_covers_every_cc_the_shared_mem_table_does():
    """Both tables are keyed by compute capability and consulted together - a gap in one is a silent default."""
    from pyutilz.system.gpu_dispatch import CC_SHARED_MEM_BUDGET

    assert set(CC_SHARED_MEM_BUDGET) <= set(CC_MAX_BLOCKS_PER_SM)

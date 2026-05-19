"""Smoke tests for pyutilz.system.gpu_dispatch.

These tests run on CPU-only hosts: we mock CUDA out where needed so the
module's degradation paths are exercised in CI without a GPU.
"""
from __future__ import annotations

import contextlib
from unittest import mock

import pytest

from pyutilz.system import gpu_dispatch as gd


# ---------------------------------------------------------------------------
# CPU-only fallbacks
# ---------------------------------------------------------------------------

class TestCPUOnlyHost:
    """All public helpers must degrade gracefully when CUDA is absent."""

    @pytest.fixture(autouse=True)
    def _no_cuda(self):
        # Reset the lru_cache so prior tests cannot leak a cached device id.
        gd.reset_cache()
        with mock.patch.object(gd, "is_cuda_available", return_value=False):
            yield
        gd.reset_cache()

    def test_select_best_gpu_returns_none(self):
        assert gd.select_best_gpu() is None
        assert gd.select_best_gpu("vram") is None
        assert gd.select_best_gpu("compute") is None
        assert gd.select_best_gpu("idle") is None

    def test_select_best_gpu_rejects_unknown_strategy(self):
        # Re-enable CUDA detection so we hit the strategy switch, not the
        # CUDA-absent early-return.
        with mock.patch.object(gd, "is_cuda_available", return_value=True), \
             mock.patch.object(
                 gd,
                 "get_gpuutil_gpu_info",
                 return_value=[{"id": 0, "memoryFree": 1.0, "load": 1.0}],
             ):
            gd.reset_cache()
            with pytest.raises(ValueError):
                gd.select_best_gpu("nonsense-strategy")

    def test_gpu_capability_summary_returns_none(self):
        assert gd.gpu_capability_summary() is None
        assert gd.gpu_capability_summary(device_id=3) is None

    def test_dispatch_cpu_vs_gpu_always_cpu(self):
        assert gd.dispatch_cpu_vs_gpu(10) == "cpu"
        assert gd.dispatch_cpu_vs_gpu(10_000_000) == "cpu"
        assert gd.dispatch_cpu_vs_gpu(10_000_000, prefer_gpu=False) == "cpu"

    def test_cuda_memory_guard_raises_without_cuda(self):
        with pytest.raises(RuntimeError):
            with gd.cuda_memory_guard(1024):
                pass

    def test_optimal_tpb_independent_of_cuda(self):
        # The heuristic is pure -- works fine even without CUDA.
        assert gd.optimal_threads_per_block(7, 5) == 256
        assert gd.optimal_threads_per_block(6, 1) == 128

    def test_shared_mem_independent_of_cuda(self):
        # Pure table lookup -- works fine on CPU-only.
        # cc 7.5 (Turing) per-block default is 48 KB; the 64 KB ceiling is opt-in only.
        assert gd.get_shared_mem_budget_per_block(7, 5) == 49152
        assert gd.get_shared_mem_budget_per_block(7, 5, allow_opt_in=True) == 65536


# ---------------------------------------------------------------------------
# Shared-memory budget table covers cc 3.0 .. 12.0 without raising
# ---------------------------------------------------------------------------

class TestSharedMemBudget:
    @pytest.mark.parametrize(
        "cc",
        [(3, 0), (3, 5), (3, 7), (5, 0), (5, 2), (6, 0), (6, 1), (6, 2),
         (7, 0), (7, 5), (8, 0), (8, 6), (8, 9), (9, 0)],
    )
    def test_known_cc_returns_positive_budget(self, cc):
        major, minor = cc
        default_b = gd.get_shared_mem_budget_per_block(major, minor)
        opt_in_b = gd.get_shared_mem_budget_per_block(major, minor, allow_opt_in=True)
        assert default_b > 0
        assert opt_in_b >= default_b
        # Default static shared-mem is 48 KB on every shipped architecture.
        assert default_b == 49152
        # Per-block opt-in must not exceed 228 KB (Hopper ceiling).
        assert opt_in_b <= 233472

    @pytest.mark.parametrize(
        "cc,expected_opt_in",
        [
            ((7, 0),  98304),   # Volta V100: 96 KB
            ((7, 2),  98304),   # Xavier
            ((7, 5),  65536),   # Turing: 64 KB
            ((8, 0), 166912),   # A100: 163 KB (164 KB per-SM - 1 KB reserved)
            ((8, 6), 101376),   # Ampere consumer: 99 KB
            ((8, 7), 166912),   # Orin: 163 KB
            ((8, 9), 101376),   # Ada Lovelace: 99 KB
            ((9, 0), 232448),   # Hopper: 227 KB
        ],
    )
    def test_opt_in_specific_values_per_cc(self, cc, expected_opt_in):
        """Pin exact per-block opt-in budgets for every cc where opt-in differs
        from default. Catches regressions if someone confuses per-SM with
        per-block numbers (per-SM is always higher; per-block has a 1 KB
        runtime reserve on cc >= 8.0)."""
        major, minor = cc
        opt_in = gd.get_shared_mem_budget_per_block(major, minor, allow_opt_in=True)
        assert opt_in == expected_opt_in

    def test_unknown_cc_falls_back_safely(self):
        # cc 99.99 is obviously not in the table; we should still get a sane
        # positive budget (the 48 KB conservative default).
        budget = gd.get_shared_mem_budget_per_block(99, 99)
        assert budget == 49152

    def test_forward_compat_within_known_major(self):
        # cc 8.99 should pick up the cc-8.x default (every shipped cc shares 48 KB
        # default static; this test only verifies the fallback finds *something*
        # known rather than the unknown-cc safe default).
        budget = gd.get_shared_mem_budget_per_block(8, 99)
        # Must equal the highest-known cc 8.x default (8.9 currently).
        highest_minor_8 = max(k[1] for k in gd.CC_SHARED_MEM_BUDGET if k[0] == 8)
        assert budget == gd.CC_SHARED_MEM_BUDGET[(8, highest_minor_8)][0]

    def test_opt_in_higher_than_default_for_a100(self):
        # A100 (cc 8.0) is the canonical case where opt-in unlocks 163 KB per-block.
        default_b = gd.get_shared_mem_budget_per_block(8, 0)
        opt_in_b = gd.get_shared_mem_budget_per_block(8, 0, allow_opt_in=True)
        assert opt_in_b > default_b
        assert opt_in_b == 166912  # 163 KB exact

    def test_pre_volta_has_no_opt_in_path(self):
        """On cc <= 6.x there is NO per-block opt-in beyond 48 KB regardless of
        what nvidia-smi reports for per-SM capacity. The 96 KB Pascal-consumer
        figure is per-SM, not per-block."""
        for cc in [(5, 0), (5, 2), (6, 0), (6, 1), (6, 2)]:
            default_b = gd.get_shared_mem_budget_per_block(*cc)
            opt_in_b = gd.get_shared_mem_budget_per_block(*cc, allow_opt_in=True)
            assert default_b == 49152, f"cc {cc} default must be 48 KB"
            assert opt_in_b == 49152, f"cc {cc} opt-in must equal default (no opt-in path pre-Volta)"


# ---------------------------------------------------------------------------
# Threads-per-block heuristic
# ---------------------------------------------------------------------------

class TestOptimalThreadsPerBlock:
    @pytest.mark.parametrize(
        "cc_major,cc_minor",
        [(3, 0), (5, 0), (6, 1), (7, 0), (7, 5), (8, 0), (8, 6), (9, 0), (12, 0)],
    )
    def test_returns_multiple_of_warp_in_typical_range(self, cc_major, cc_minor):
        tpb = gd.optimal_threads_per_block(cc_major, cc_minor)
        assert tpb % gd.WARP_SIZE == 0
        assert gd.WARP_SIZE <= tpb <= 1024

    def test_pascal_default_is_128(self):
        assert gd.optimal_threads_per_block(6, 1) == 128

    def test_volta_and_later_default_is_256(self):
        assert gd.optimal_threads_per_block(7, 0) == 256
        assert gd.optimal_threads_per_block(8, 6) == 256
        assert gd.optimal_threads_per_block(9, 0) == 256

    def test_max_required_clamps_result(self):
        # If the kernel only has 40 work items per block, we should not
        # allocate 128 threads.
        tpb = gd.optimal_threads_per_block(6, 1, max_required=40)
        # Warp-aligned clamp: 40 -> round up to 64.
        assert tpb == 64

    def test_max_required_below_warp_floored_to_warp(self):
        tpb = gd.optimal_threads_per_block(7, 5, max_required=10)
        assert tpb == gd.WARP_SIZE

    def test_no_warp_alignment_when_requested(self):
        tpb = gd.optimal_threads_per_block(6, 1, max_required=100, multiple_of_warp=False)
        assert tpb == 100


# ---------------------------------------------------------------------------
# select_best_gpu strategy logic (mocked GPU listings, CUDA "available")
# ---------------------------------------------------------------------------

class TestSelectBestGpuStrategies:
    @pytest.fixture(autouse=True)
    def _patch_env(self):
        gd.reset_cache()
        fake_gpus = [
            {"id": 0, "memoryFree": 1.0, "memoryTotal": 4.0, "load": 80.0,
             "name": "GTX 1050 Ti", "uuid": "u0"},
            {"id": 1, "memoryFree": 16.0, "memoryTotal": 24.0, "load": 5.0,
             "name": "RTX 3090", "uuid": "u1"},
        ]
        fake_caps = {
            0: {"COMPUTE_CAPABILITY_MAJOR": 6, "COMPUTE_CAPABILITY_MINOR": 1},
            1: {"COMPUTE_CAPABILITY_MAJOR": 8, "COMPUTE_CAPABILITY_MINOR": 6},
        }

        def _caps(device_id: int = 0):
            return fake_caps.get(int(device_id))

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(gd, "is_cuda_available", return_value=True))
            stack.enter_context(mock.patch.object(gd, "get_gpuutil_gpu_info", return_value=fake_gpus))
            stack.enter_context(mock.patch.object(gd, "get_gpu_cuda_capabilities", side_effect=_caps))
            yield
        gd.reset_cache()

    def test_vram_picks_largest_free(self):
        assert gd.select_best_gpu("vram") == 1

    def test_idle_picks_lowest_load(self):
        assert gd.select_best_gpu("idle") == 1

    def test_compute_picks_highest_cc(self):
        assert gd.select_best_gpu("compute") == 1

    def test_auto_picks_combined_score(self):
        # GTX 1050 Ti: 1.0 * 6.1 = 6.1 ; RTX 3090: 16.0 * 8.6 = 137.6 -> dev 1
        assert gd.select_best_gpu("auto") == 1


# ---------------------------------------------------------------------------
# gpu_capability_summary happy path
# ---------------------------------------------------------------------------

class TestGpuCapabilitySummaryHappyPath:
    def test_merges_caps_and_gputil(self):
        fake_caps = {
            "COMPUTE_CAPABILITY_MAJOR": 7,
            "COMPUTE_CAPABILITY_MINOR": 5,
            "MULTIPROCESSOR_COUNT": 14,
            "TOTAL_CUDA_CORES": 1024,
            "MAX_THREADS_PER_BLOCK": 1024,
            "MAX_SHARED_MEMORY_PER_BLOCK": 49152,
            "WARP_SIZE": 32,
        }
        fake_gpus = [{"id": 0, "name": "GTX 1660", "memoryFree": 5.0, "memoryTotal": 6.0}]
        with mock.patch.object(gd, "is_cuda_available", return_value=True), \
             mock.patch.object(gd, "get_gpu_cuda_capabilities", return_value=fake_caps), \
             mock.patch.object(gd, "get_gpuutil_gpu_info", return_value=fake_gpus):
            s = gd.gpu_capability_summary(device_id=0)
        assert s is not None
        assert s["cc_major"] == 7 and s["cc_minor"] == 5
        assert s["sm_count"] == 14
        assert s["total_cuda_cores"] == 1024
        assert s["name"] == "GTX 1660"
        assert s["free_vram_gb"] == 5.0
        assert s["total_vram_gb"] == 6.0


# ---------------------------------------------------------------------------
# dispatch_cpu_vs_gpu truth table
# ---------------------------------------------------------------------------

class TestDispatchCpuVsGpu:
    def test_small_workload_picks_cpu_even_with_cuda(self):
        with mock.patch.object(gd, "is_cuda_available", return_value=True):
            assert gd.dispatch_cpu_vs_gpu(1_000) == "cpu"

    def test_large_workload_picks_gpu_when_cuda_present(self):
        with mock.patch.object(gd, "is_cuda_available", return_value=True):
            assert gd.dispatch_cpu_vs_gpu(10_000_000) == "gpu"

    def test_prefer_gpu_false_forces_cpu(self):
        with mock.patch.object(gd, "is_cuda_available", return_value=True):
            assert gd.dispatch_cpu_vs_gpu(10_000_000, prefer_gpu=False) == "cpu"

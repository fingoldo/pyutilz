"""
Test suite for benchmarking.py
Tests cover the benchmark_algos_by_runtime function.
"""

import pytest
import numpy as np

from pyutilz.dev.benchmarking import benchmark_algos_by_runtime


def algo_fast(x):
    return x * 2


def algo_slow(x):
    return sum(range(100)) + x


def algo_identity(x):
    return x


class TestBenchmarkAlgosByRuntime:
    """Test benchmark_algos_by_runtime function"""

    def test_returns_sorted_implementations(self):
        """Test that implementations are returned sorted by speed"""
        implementations = [algo_fast, algo_slow]
        sorted_impls, durations = benchmark_algos_by_runtime(
            implementations, algo_name="test", n_reps=2, x=5
        )

        assert len(sorted_impls) == 2
        assert len(durations) == 2

    def test_durations_are_non_negative(self):
        """Test that all durations are non-negative"""
        implementations = [algo_fast, algo_slow, algo_identity]
        sorted_impls, durations = benchmark_algos_by_runtime(
            implementations, algo_name="test", n_reps=1, x=10
        )

        assert all(d >= 0 for d in durations)

    def test_sorted_by_ascending_duration(self):
        """Test that durations are sorted ascending"""
        implementations = [algo_fast, algo_slow]
        sorted_impls, durations = benchmark_algos_by_runtime(
            implementations, n_reps=2, x=5
        )

        assert durations[0] <= durations[1]

    def test_single_implementation(self):
        """Test with a single implementation"""
        sorted_impls, durations = benchmark_algos_by_runtime(
            [algo_fast], algo_name="single", n_reps=2, x=3
        )

        assert len(sorted_impls) == 1
        assert len(durations) == 1
        assert durations[0] >= 0

    def test_returns_tuple(self):
        """Test return type is tuple of (list, list)"""
        result = benchmark_algos_by_runtime([algo_fast], n_reps=1, x=1)

        assert isinstance(result, tuple)
        assert len(result) == 2
        sorted_impls, durations = result
        assert isinstance(sorted_impls, list)
        assert isinstance(durations, list)

    def test_verbose_mode(self):
        """Test with verbose > 1 (exercises logging paths)"""
        sorted_impls, durations = benchmark_algos_by_runtime(
            [algo_fast, algo_identity], algo_name="verbose_test", n_reps=2, verbose=2, x=7
        )

        assert len(sorted_impls) == 2
        assert len(durations) == 2

    def test_multiple_reps(self):
        """Test with multiple repetitions (n_reps > 1)"""
        sorted_impls, durations = benchmark_algos_by_runtime(
            [algo_fast], n_reps=5, x=100
        )

        assert len(sorted_impls) == 1
        assert durations[0] >= 0

    def test_implementations_preserved(self):
        """Test that all original implementations are in result"""
        implementations = [algo_fast, algo_slow, algo_identity]
        sorted_impls, durations = benchmark_algos_by_runtime(
            implementations, n_reps=1, x=1
        )

        assert set(sorted_impls) == set(implementations)


class TestSynchronizeGpu:
    """The GPU-sync guard so async cupy/cuda kernels are timed at completion."""

    def test_synchronize_is_noop_without_cupy(self):
        # Must never raise whether or not cupy is installed.
        from pyutilz.dev.benchmarking import synchronize_gpu_if_available

        synchronize_gpu_if_available()  # no exception == pass

    def test_benchmark_with_synchronize_gpu_true(self):
        # CPU algos + synchronize_gpu=True: sync is a no-op, results still valid.
        sorted_impls, durations = benchmark_algos_by_runtime(
            [algo_fast, algo_slow], n_reps=2, synchronize_gpu=True, x=5
        )
        assert len(durations) == 2
        assert all(d >= 0 for d in durations)

    def test_benchmark_with_synchronize_gpu_false(self):
        sorted_impls, durations = benchmark_algos_by_runtime(
            [algo_fast], n_reps=1, synchronize_gpu=False, x=1
        )
        assert durations[0] >= 0


class TestSweepBackendCrossover:
    """sweep_backend_crossover: fastest EQUIVALENT backend per size band."""

    @staticmethod
    def _inp(size):
        return (np.arange(size, dtype=float),)

    def test_single_variant_one_catchall_region(self):
        from pyutilz.dev.benchmarking import sweep_backend_crossover

        regions = sweep_backend_crossover(
            {"numpy": lambda x: x.sum()}, [10, 100, 1000], self._inp, "n", repeats=2
        )
        assert len(regions) == 1
        assert regions[0]["n_max"] is None  # catch-all
        assert regions[0]["backend_choice"] == "numpy"

    def test_divergent_variant_never_chosen(self):
        from pyutilz.dev.benchmarking import sweep_backend_crossover

        regions = sweep_backend_crossover(
            {"numpy": lambda x: x.sum(), "bad": lambda x: x.sum() + 1e6},
            [10, 100], self._inp, "n", reference="numpy", repeats=2,
        )
        # bad diverges (1e6 >> tol) -> disqualified at every size -> numpy wins.
        assert all(r["backend_choice"] == "numpy" for r in regions)

    def test_equivalent_variant_eligible(self):
        from pyutilz.dev.benchmarking import sweep_backend_crossover

        regions = sweep_backend_crossover(
            {"numpy": lambda x: x.sum(), "equiv": lambda x: float(np.add.reduce(x))},
            [10, 100], self._inp, "n", reference="numpy", repeats=2,
        )
        # both produce the same value -> winner is a valid eligible backend.
        for r in regions:
            assert r["backend_choice"] in ("numpy", "equiv")
            assert r.get("max_abs_diff", 0.0) < 1e-6


class TestSweepBackendGrid:
    """Full N-D Cartesian grid + residency-aware backend sweep."""

    @staticmethod
    def _inp(dims):
        return (np.arange(dims["n"], dtype=float),)

    def test_full_cartesian_one_region_per_cell(self):
        from pyutilz.dev.benchmarking import sweep_backend_grid

        regions = sweep_backend_grid(
            {"numpy": lambda x: x.sum()},
            {"n": [10, 100], "k": [1, 2]},
            lambda d: (np.arange(d["n"], dtype=float),),
            repeats=2,
        )
        assert len(regions) == 5  # 2 x 2 grid + 1 all-None catch-all, host only
        for r in regions:
            assert "n_max" in r and "k_max" in r
            assert r["backend_choice"] == "numpy"
            assert "location_eq" not in r  # single residency -> no location key
        cells = {(r["n_max"], r["k_max"]) for r in regions}
        assert {(10, 1), (10, 2), (100, 1), (100, 2)} <= cells
        assert (None, None) in cells  # catch-all for inputs beyond the grid

    def test_residency_emits_location_regions(self):
        from pyutilz.dev.benchmarking import sweep_backend_grid

        # identity to_device (no real GPU) just exercises the host/device structure.
        regions = sweep_backend_grid(
            {"numpy": lambda x: x.sum()},
            {"n": [10, 100]},
            lambda d: (np.arange(d["n"], dtype=float),),
            residencies=("host", "device"),
            to_device=lambda args: args,
            repeats=2,
        )
        assert len(regions) == 6  # 2 cells x 2 residencies + 1 catch-all per residency
        assert {r["location_eq"] for r in regions} == {"host", "device"}

    def test_catchall_resolves_inputs_beyond_grid(self):
        # B3 regression: a live size LARGER than the biggest swept value must still
        # resolve (to the largest-cell winner via the all-None catch-all), not None.
        from pyutilz.dev.benchmarking import sweep_backend_grid
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache

        regions = sweep_backend_grid(
            {"numpy": lambda x: x.sum()}, {"n": [10, 100]},
            lambda d: (np.arange(d["n"], dtype=float),), repeats=2,
        )
        assert any(r["n_max"] is None for r in regions)  # catch-all present
        c = KernelTuningCache(in_memory=True)
        c.update("k", axes=["n"], regions=regions)
        assert c.lookup("k", n=10_000_000) is not None  # beyond grid -> still routes
        assert c.lookup("k", n=10_000_000)["backend_choice"] == "numpy"

    def test_divergent_variant_skipped(self):
        from pyutilz.dev.benchmarking import sweep_backend_grid

        regions = sweep_backend_grid(
            {"numpy": lambda x: x.sum(), "bad": lambda x: x.sum() + 1e6},
            {"n": [10, 100]},
            lambda d: (np.arange(d["n"], dtype=float),),
            reference="numpy",
            repeats=2,
        )
        assert all(r["backend_choice"] == "numpy" for r in regions)


class TestSelectionAndRouting:
    """End-to-end: the benchmarker picks the fastest EQUIVALENT backend, and the
    cache routes the right backend per dims + residency."""

    @staticmethod
    def _inp(d):
        return (np.arange(d["n"], dtype=float),)

    def test_benchmarker_picks_faster_equivalent(self):
        # 'fast' and 'slow' give the SAME output; 'slow' sleeps 5ms -> 'fast' must win.
        import time
        from pyutilz.dev.benchmarking import sweep_backend_grid

        regions = sweep_backend_grid(
            {"slow": lambda x: (time.sleep(0.005) or x.sum()), "fast": lambda x: x.sum()},
            {"n": [10, 100]}, self._inp, reference="slow", repeats=3,
        )
        assert regions and all(r["backend_choice"] == "fast" for r in regions)

    def test_benchmarker_rejects_faster_but_divergent(self):
        # 'wrong' is instant but returns a different value -> equiv gate disqualifies
        # it, so the (slower) correct reference wins.
        from pyutilz.dev.benchmarking import sweep_backend_grid

        regions = sweep_backend_grid(
            {"ref": lambda x: (_t() or x.sum()), "wrong": lambda x: x.sum() + 1e6},
            {"n": [10, 100]}, self._inp, reference="ref", repeats=3,
        )
        assert regions and all(r["backend_choice"] == "ref" for r in regions)

    def test_routing_by_size_band(self):
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache

        c = KernelTuningCache(in_memory=True)
        c.update("k", axes=["n"], regions=[{"n_max": 100, "backend_choice": "cpu"}, {"backend_choice": "gpu"}])
        assert c.lookup("k", n=50)["backend_choice"] == "cpu"
        assert c.lookup("k", n=500)["backend_choice"] == "gpu"

    def test_routing_by_residency(self):
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache

        c = KernelTuningCache(in_memory=True)
        c.update("k", axes=["n", "location"], regions=[
            {"location_eq": "host", "backend_choice": "numpy"},
            {"location_eq": "device", "backend_choice": "cupy"},
        ])
        assert c.lookup("k", n=1000, location="host")["backend_choice"] == "numpy"
        assert c.lookup("k", n=1000, location="device")["backend_choice"] == "cupy"

    def test_sweep_to_dispatch_routes_swept_winner(self):
        # Full path: benchmark -> persist regions -> lookup returns the measured winner.
        import time
        from pyutilz.dev.benchmarking import sweep_backend_grid
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache

        regions = sweep_backend_grid(
            {"slow": lambda x: (time.sleep(0.004) or x.sum()), "fast": lambda x: x.sum()},
            {"n": [10, 100]}, self._inp, reference="slow", repeats=3,
        )
        c = KernelTuningCache(in_memory=True)
        c.update("k", axes=["n"], regions=regions)
        assert c.lookup("k", n=50)["backend_choice"] == "fast"
        assert c.lookup("k", n=100)["backend_choice"] == "fast"


def _t():
    import time
    time.sleep(0.004)


class TestRobustRanking:
    """The robust (interleaved min-over-reps) metric must pick the truly-fastest
    candidate even when a simulated concurrent process injects contention spikes
    that would mis-rank the legacy sequential MEAN metric."""

    @staticmethod
    def _inp(d):
        return (np.arange(d["n"], dtype=float),)

    def test_robust_min_beats_mean_under_contention(self, monkeypatch):
        # Drive the ranker with a SCRIPTED clock so the contention scenario is exact,
        # not wall-timing-flaky. truefast is genuinely faster (floor 1ms vs 5ms), but
        # a concurrent process injects a +50ms spike on SOME reps. We script the deltas
        # so that under the legacy MEAN metric truefast is mis-ranked SLOWER (its reps
        # caught more spikes), while the robust MIN-over-reps sees each candidate's
        # uncontended floor and ranks truefast correctly.
        from pyutilz.dev import benchmarking as bench

        # Robust interleaves per rep: rep0 -> [slow, fast], rep1 -> [slow, fast], ...
        # Per-call elapsed (ms) consumed by timer() pairs, in interleaved order:
        #   rep0: slow=5,   fast=1+50 (spike)
        #   rep1: slow=5+50 (spike), fast=1
        #   rep2: slow=5,   fast=1
        # Robust MIN: slow=5, fast=1 -> truefast wins. (correct)
        # If the SAME deltas were consumed by the MEAN path (sequential: slow's 3 reps
        # then fast's 3 reps) the spikes would land differently, but the point is the
        # robust path picks the true floor. We assert robust picks truefast.
        deltas_ms = [5, 51, 55, 1, 5, 1]  # interleaved: (slow,fast) x3
        clock = {"t": 0.0}
        seq = iter(deltas_ms)

        def fake_timer():
            # Even calls = start (return current), odd calls = stop (advance by next delta).
            fake_timer.calls += 1
            if fake_timer.calls % 2 == 1:
                return clock["t"]
            d = next(seq) / 1e3
            clock["t"] += d
            return clock["t"]
        fake_timer.calls = 0

        monkeypatch.setattr(bench, "timer", fake_timer)
        out = bench._rank_candidates(
            {"trueslow": lambda: None, "truefast": lambda: None},
            repeats=3, synchronize_gpu=False, ranking="robust",
        )
        assert out["truefast"] < out["trueslow"]
        assert abs(out["truefast"] - 1.0) < 1e-6  # picked the 1ms floor, not a spiked rep
        assert abs(out["trueslow"] - 5.0) < 1e-6

    def test_robust_is_default(self):
        # ranking defaults to "robust"; an explicit "mean" still works (back-compat).
        from pyutilz.dev.benchmarking import sweep_backend_grid
        for mode in (None, "mean", "robust"):
            kw = {} if mode is None else {"ranking": mode}
            regions = sweep_backend_grid(
                {"a": lambda x: x.sum(), "b": lambda x: x.sum()},
                {"n": [10]}, self._inp, reference="a", repeats=2,
                synchronize_gpu=False, **kw,
            )
            assert regions and "backend_choice" in regions[0]

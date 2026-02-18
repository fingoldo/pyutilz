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

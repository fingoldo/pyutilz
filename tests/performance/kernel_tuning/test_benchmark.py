"""Behaviour of ``pyutilz.performance.kernel_tuning.benchmark``: the busy threshold comes from the argument, else the
``PYUTILZ_KERNEL_SWEEP_HW_BUSY`` variable, else the default; a threshold above 1 disables the gate; a verdict is reused
within its TTL and re-probed after it; the median is reported in milliseconds (``inf`` for no samples); and the input
builder makes fresh inputs per call or repeats one shared tuple. The hardware probes are switched off, so nothing here
depends on the machine's load."""

from __future__ import annotations

import math

import pytest

from pyutilz.performance.kernel_tuning import benchmark as bm


@pytest.fixture(autouse=True)
def _reset_busy_cache(monkeypatch):
    monkeypatch.setattr(bm, "_HW_BUSY_CACHE", None)


class TestBusyThreshold:
    def test_an_explicit_threshold_wins_over_the_environment(self, monkeypatch):
        monkeypatch.setenv("PYUTILZ_KERNEL_SWEEP_HW_BUSY", "0.9")
        assert bm._busy_threshold(0.25) == 0.25

    def test_the_environment_is_used_when_no_threshold_is_given(self, monkeypatch):
        monkeypatch.setenv("PYUTILZ_KERNEL_SWEEP_HW_BUSY", "0.7")
        assert bm._busy_threshold(None) == 0.7

    def test_the_default_is_used_when_nothing_is_set(self, monkeypatch):
        monkeypatch.delenv("PYUTILZ_KERNEL_SWEEP_HW_BUSY", raising=False)
        assert bm._busy_threshold(None) == bm.DEFAULT_HW_BUSY_THRESHOLD

    def test_an_unparsable_environment_value_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv("PYUTILZ_KERNEL_SWEEP_HW_BUSY", "high")
        assert bm._busy_threshold(None) == bm.DEFAULT_HW_BUSY_THRESHOLD


class TestHardwareBusy:
    def test_a_threshold_above_one_disables_the_gate_without_probing(self, monkeypatch):
        probed = []
        monkeypatch.setattr(bm, "_HW_BUSY_CACHE", None)
        assert bm.hardware_busy(1.5, timer=lambda: probed.append(1) or 0.0) is False
        assert probed == []

    def test_with_both_probes_off_nothing_is_reported_busy(self):
        assert bm.hardware_busy(0.1, check_cpu=False, check_gpu=False) is False

    def test_a_fresh_verdict_is_reused_within_the_ttl(self, monkeypatch):
        monkeypatch.setattr(bm, "_HW_BUSY_CACHE", (100.0, 0.5, True))
        clock = [100.0 + bm._HW_BUSY_TTL / 2]
        assert bm.hardware_busy(0.5, check_cpu=False, check_gpu=False, timer=lambda: clock[0]) is True

    def test_a_verdict_for_another_threshold_is_not_reused(self, monkeypatch):
        monkeypatch.setattr(bm, "_HW_BUSY_CACHE", (100.0, 0.9, True))
        assert bm.hardware_busy(0.5, check_cpu=False, check_gpu=False, timer=lambda: 100.0) is False

    def test_an_expired_verdict_is_probed_again(self, monkeypatch):
        monkeypatch.setattr(bm, "_HW_BUSY_CACHE", (100.0, 0.5, True))
        expired = 100.0 + bm._HW_BUSY_TTL + 1.0
        assert bm.hardware_busy(0.5, check_cpu=False, check_gpu=False, timer=lambda: expired) is False


class TestTimingHelpers:
    def test_the_median_is_reported_in_milliseconds(self):
        assert bm._median_ms([0.001, 0.003, 0.002]) == pytest.approx(2.0)

    def test_no_samples_is_infinitely_slow(self):
        assert math.isinf(bm._median_ms([]))

    def test_shared_inputs_repeat_one_tuple(self):
        calls = []

        def make():
            calls.append(1)
            return (object(),)

        built = bm._prebuild_inputs(make, 3, fresh_inputs_per_call=False)
        assert len(calls) == 1
        assert built[0] is built[1] is built[2]

    def test_fresh_inputs_are_built_per_entry(self):
        calls = []

        def make():
            calls.append(1)
            return (object(),)

        built = bm._prebuild_inputs(make, 3, fresh_inputs_per_call=True)
        assert len(calls) == 3
        assert len({id(entry[0]) for entry in built}) == 3

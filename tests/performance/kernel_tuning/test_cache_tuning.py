"""Behaviour of ``pyutilz.performance.kernel_tuning.cache.cache_tuning``: the equivalence gate keeps a region whose recorded
divergence is within tolerance (or unrecorded) and rejects, with a hook report, one that diverges beyond it; an environment
override short-circuits the tuner entirely and is reported through the hooks. The in-memory cache keeps these tests off disk."""

from __future__ import annotations

from unittest import mock

from pyutilz.performance.kernel_tuning import cache as ktc
from pyutilz.performance.kernel_tuning.cache.cache_hooks import LoggerHooks
from pyutilz.performance.kernel_tuning.cache.cache_tuning import _CacheTuningMixin


class _Recorder(LoggerHooks):
    def __init__(self):
        self.overrides = []
        self.winners = []

    def env_override(self, kernel, choice):
        self.overrides.append((kernel, choice))

    def winner_chosen(self, kernel, region, reason):
        self.winners.append((kernel, region, reason))


class _Cache(_CacheTuningMixin):
    def __init__(self):
        self._in_memory = True
        self._path = None


class TestEquivalenceGate:
    def test_a_region_within_tolerance_is_kept(self):
        regions = [{"max_abs_diff": 1e-6, "block": 8}]
        assert _Cache()._apply_equiv_gate("matmul", regions, 1e-3) == regions

    def test_a_region_without_a_recorded_divergence_is_kept_unvalidated(self):
        regions = [{"block": 8}]
        assert _Cache()._apply_equiv_gate("matmul", regions, 1e-3) == regions

    def test_a_region_beyond_tolerance_is_dropped_and_reported(self):
        recorder = _Recorder()
        bad = {"max_abs_diff": 1.0, "block": 8}
        good = {"max_abs_diff": 1e-6, "block": 4}
        kept = _Cache()._apply_equiv_gate("matmul", [bad, good], 1e-3, hooks=recorder)
        assert kept == [good]
        assert recorder.winners and recorder.winners[0][0] == "matmul"
        assert recorder.winners[0][2].startswith("rejected: diverges")

    def test_a_rejected_region_is_dropped_even_without_a_hook(self):
        bad = {"max_abs_diff": 1.0, "block": 8}
        assert _Cache()._apply_equiv_gate("matmul", [bad], 1e-3) == []


class TestEnvOverride:
    def test_a_forced_environment_value_wins_without_running_the_tuner(self, monkeypatch):
        ktc._TUNED_THIS_PROCESS.clear()
        recorder = _Recorder()
        tuner = mock.Mock(side_effect=AssertionError("the tuner must not run"))
        monkeypatch.setenv("PYUTILZ_TEST_FORCE_MATMUL", " naive ")
        cache = ktc.KernelTuningCache(in_memory=True)
        out = cache.get_or_tune("matmul", dims={"n": 4}, tuner=tuner, axes=["n"], fallback="fallback", env_key="PYUTILZ_TEST_FORCE_MATMUL", hooks=recorder)
        assert out == "naive"
        assert recorder.overrides == [("matmul", "naive")]
        tuner.assert_not_called()

    def test_an_empty_environment_value_does_not_override(self, monkeypatch):
        ktc._TUNED_THIS_PROCESS.clear()
        recorder = _Recorder()
        monkeypatch.setenv("PYUTILZ_TEST_FORCE_MATMUL", "   ")
        cache = ktc.KernelTuningCache(in_memory=True)
        cache.get_or_tune("matmul", dims={"n": 4}, tuner=lambda: [], axes=["n"], fallback="fallback", env_key="PYUTILZ_TEST_FORCE_MATMUL", hooks=recorder)
        assert recorder.overrides == []


class TestRejectionReport:
    def test_a_rejected_region_is_reported_as_a_warning_with_its_numbers(self, caplog):
        bad = {"max_abs_diff": 1.0, "block": 8}
        with caplog.at_level("WARNING"):
            _Cache()._apply_equiv_gate("matmul", [bad], 1e-3)
        messages = [rec.getMessage() for rec in caplog.records if rec.levelname == "WARNING"]
        assert len(messages) == 1
        assert messages[0].startswith("kernel_tuning_cache: matmul region ")
        assert "REJECTED -- max_abs_diff=1.000e+00 > equiv_tol=1.000e-03" in messages[0]
        assert messages[0].endswith("not persisting it.")

    def test_the_hook_receives_the_divergence_as_its_reason(self):
        recorder = _Recorder()
        bad = {"max_abs_diff": 1.0, "block": 8}
        _Cache()._apply_equiv_gate("matmul", [bad], 1e-3, hooks=recorder)
        assert recorder.winners == [("matmul", bad, "rejected: diverges 1.000e+00>1.000e-03")]

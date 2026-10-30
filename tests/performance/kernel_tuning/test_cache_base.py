"""Behaviour of the pure helpers in ``pyutilz.performance.kernel_tuning.cache.cache_base``: filename-safe slugs and tokens,
the power-of-two VRAM class (both MiB and GiB scales), the sweep and async-sweep timing knobs read from the environment with
their defaults, the per-kernel directory name, and the provenance comparison that decides whether a saved tuning is still
valid. Nothing here probes hardware."""

from __future__ import annotations

import logging
import os
import sys
from types import SimpleNamespace

import pytest

from pyutilz.performance.kernel_tuning.cache import cache_base


class TestSlugAndToken:
    def test_a_device_name_drops_its_vendor_noise_and_is_lowercased(self):
        assert cache_base._slug("NVIDIA GeForce RTX 3080 (R) GPU @ 1.7GHz") == "nvidia-geforce-rtx-3080"

    def test_unsafe_characters_are_removed(self):
        assert cache_base._slug("a/b:c*d") == "abcd"

    def test_an_empty_name_is_unknown(self):
        assert cache_base._slug("   ") == "unknown"

    def test_the_slug_is_truncated(self):
        assert len(cache_base._slug("x" * 100, maxlen=10)) == 10

    def test_a_token_keeps_letters_digits_and_dots_only(self):
        assert cache_base._token("cuda-12.4 rc") == "cuda12.4rc"

    def test_a_missing_token_is_x(self):
        assert cache_base._token(None) == "x"


class TestVramClass:
    @pytest.mark.parametrize(
        "total, expected",
        [
            (8, "vram8g"),  # already GiB
            (5, "vram8g"),  # rounded up to the next power of two
            (8192, "vram8g"),  # MiB scale
            (6000, "vram8g"),  # MiB, rounded up
        ],
    )
    def test_the_class_is_the_next_power_of_two_gib(self, total, expected):
        assert cache_base._vram_class(total) == expected

    @pytest.mark.parametrize("total", [0, -1, "nope", None])
    def test_an_unusable_size_is_unknown(self, total):
        assert cache_base._vram_class(total) == "vramx"


class TestSweepTimingKnobs:
    def test_the_sweep_budget_defaults_when_unset(self, monkeypatch):
        monkeypatch.delenv("PYUTILZ_KERNEL_SWEEP_BUDGET_SEC", raising=False)
        assert cache_base._sweep_budget_seconds() == cache_base._DEFAULT_SWEEP_BUDGET_SECONDS

    def test_the_sweep_budget_is_read_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("PYUTILZ_KERNEL_SWEEP_BUDGET_SEC", "45")
        assert cache_base._sweep_budget_seconds() == 45.0

    def test_a_budget_below_one_second_is_clamped_up(self, monkeypatch):
        monkeypatch.setenv("PYUTILZ_KERNEL_SWEEP_BUDGET_SEC", "0.1")
        assert cache_base._sweep_budget_seconds() == 1.0

    def test_an_unparsable_budget_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv("PYUTILZ_KERNEL_SWEEP_BUDGET_SEC", "soon")
        assert cache_base._sweep_budget_seconds() == cache_base._DEFAULT_SWEEP_BUDGET_SECONDS

    def test_the_start_delay_defaults_to_ten_seconds(self, monkeypatch):
        monkeypatch.delenv("PYUTILZ_KERNEL_SWEEP_START_DELAY", raising=False)
        assert cache_base._async_sweep_start_delay() == 10.0

    def test_a_negative_start_delay_is_clamped_to_zero(self, monkeypatch):
        monkeypatch.setenv("PYUTILZ_KERNEL_SWEEP_START_DELAY", "-5")
        assert cache_base._async_sweep_start_delay() == 0.0

    def test_the_idle_wait_defaults_to_two_minutes_and_reads_the_environment(self, monkeypatch):
        monkeypatch.delenv("PYUTILZ_KERNEL_SWEEP_IDLE_MAX_WAIT", raising=False)
        assert cache_base._async_sweep_idle_max_wait() == 120.0
        monkeypatch.setenv("PYUTILZ_KERNEL_SWEEP_IDLE_MAX_WAIT", "0")
        assert cache_base._async_sweep_idle_max_wait() == 0.0


class TestKernelDir:
    def test_the_directory_is_under_the_host_dir_and_named_for_the_kernel(self, tmp_path):
        directory = cache_base._kernel_dir(str(tmp_path), "matmul")
        assert os.path.dirname(directory) == str(tmp_path)
        assert os.path.basename(directory).startswith("matmul-")

    def test_two_kernel_names_never_share_a_directory(self, tmp_path):
        assert cache_base._kernel_dir(str(tmp_path), "a b") != cache_base._kernel_dir(str(tmp_path), "a_b")

    def test_the_directory_is_stable_for_the_same_name(self, tmp_path):
        assert cache_base._kernel_dir(str(tmp_path), "matmul") == cache_base._kernel_dir(str(tmp_path), "matmul")


class TestProvenanceChanged:
    _BASE = {
        "cuda_driver_version": "12.4",
        "cuda_runtime_version": "12.4",
        "cupy_version": "13.0",
        "numba_version": "0.60",
        "numpy_version": "2.0",
        "python_version": "3.12",
        "gpu_summary": {"cc_major": 8, "cc_minor": 6, "name": "RTX 3080"},
    }

    def test_no_data_on_either_side_is_not_a_change(self):
        assert cache_base.provenance_changed(None, self._BASE) is False
        assert cache_base.provenance_changed(self._BASE, None) is False

    def test_identical_provenance_is_not_a_change(self):
        assert cache_base.provenance_changed(dict(self._BASE), dict(self._BASE)) is False

    def test_a_driver_bump_is_a_change(self):
        assert cache_base.provenance_changed(self._BASE, {**self._BASE, "cuda_driver_version": "12.5"}) is True

    def test_a_python_minor_upgrade_is_a_change(self):
        assert cache_base.provenance_changed(self._BASE, {**self._BASE, "python_version": "3.13"}) is True

    def test_a_different_gpu_is_a_change(self):
        other = {**self._BASE, "gpu_summary": {"cc_major": 8, "cc_minor": 6, "name": "RTX 4090"}}
        assert cache_base.provenance_changed(self._BASE, other) is True

    def test_a_gpu_name_one_side_could_not_read_is_not_a_change(self):
        unknown = {**self._BASE, "gpu_summary": {"cc_major": 8, "cc_minor": 6, "name": None}}
        assert cache_base.provenance_changed(self._BASE, unknown) is False


class TestProvenanceFieldsEachMatter:
    @pytest.mark.parametrize("field", ["cuda_runtime_version", "cupy_version", "numba_version", "numpy_version"])
    def test_a_library_version_bump_is_a_change(self, field):
        old = dict(TestProvenanceChanged._BASE)
        new = {**old, field: "99.9"}
        assert cache_base.provenance_changed(old, new) is True

    @pytest.mark.parametrize("gpu_field", ["cc_major", "cc_minor"])
    def test_a_compute_capability_change_is_a_change(self, gpu_field):
        old = dict(TestProvenanceChanged._BASE)
        changed = dict(old["gpu_summary"])
        changed[gpu_field] = 99
        assert cache_base.provenance_changed(old, {**old, "gpu_summary": changed}) is True


class TestSlugDetails:
    def test_the_default_length_is_forty_characters(self):
        assert len(cache_base._slug("x" * 60)) == 40

    def test_a_cpu_word_glued_to_letters_is_kept(self):
        assert cache_base._slug("XCPU") == "xcpu"

    def test_leading_and_trailing_dots_and_underscores_are_stripped(self):
        assert cache_base._slug("__abc__") == "abc"


class TestDistributionVersions:
    def test_a_missing_distribution_is_none_and_logs_its_name(self, caplog):
        with caplog.at_level(logging.WARNING, logger=cache_base.logger.name):
            assert cache_base._one_dist_version("definitely-missing-dist-xyz") is None
        messages = [rec.getMessage() for rec in caplog.records]
        assert any(m.startswith("distribution definitely-missing-dist-xyz version unavailable (") for m in messages)

    def test_the_first_installed_name_wins_over_a_missing_one(self):
        assert cache_base._dist_version("definitely-missing-dist-xyz", "pytest") == cache_base._one_dist_version("pytest")

    def test_no_installed_name_gives_x(self):
        assert cache_base._dist_version("definitely-missing-dist-xyz") == "x"

    def test_the_package_stamp_names_numba_then_cupy(self):
        assert cache_base._package_stamp().startswith(f"nb{cache_base._dist_version('numba')}_cupy")


class TestNumbaThreads:
    def test_the_numba_thread_count_is_read_from_its_config(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "numba", SimpleNamespace(config=SimpleNamespace(NUMBA_NUM_THREADS=7)))
        assert cache_base._numba_threads() == 7

    def test_a_missing_numba_thread_setting_falls_back_to_the_cpu_count(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "numba", SimpleNamespace(config=SimpleNamespace()))
        monkeypatch.setattr(cache_base.os, "cpu_count", lambda: 6)
        assert cache_base._numba_threads() == 6

    def test_an_absent_numba_falls_back_to_the_cpu_count_and_logs_it(self, monkeypatch, caplog):
        monkeypatch.setitem(sys.modules, "numba", None)
        monkeypatch.setattr(cache_base.os, "cpu_count", lambda: 6)
        with caplog.at_level(logging.DEBUG, logger=cache_base.logger.name):
            assert cache_base._numba_threads() == 6
        assert any(rec.getMessage().startswith("numba thread count unavailable (") for rec in caplog.records)


class TestVramClassBoundaries:
    def test_a_fractional_gib_rounds_up_to_one(self):
        assert cache_base._vram_class(0.5) == "vram1g"

    def test_exactly_256_is_read_as_gib(self):
        assert cache_base._vram_class(256) == "vram256g"

    def test_257_is_read_as_mib(self):
        assert cache_base._vram_class(257) == "vram1g"

    def test_just_over_eight_gib_in_mib_rounds_up_to_sixteen(self):
        assert cache_base._vram_class(8192.5) == "vram16g"

    def test_a_non_numeric_size_is_logged_with_its_exception_class(self, caplog):
        with caplog.at_level(logging.WARNING, logger=cache_base.logger.name):
            assert cache_base._vram_class("nope") == "vramx"
        assert any("total VRAM is not numeric, class reported as unknown: " in rec.getMessage() and "ValueError" in rec.getMessage() for rec in caplog.records)

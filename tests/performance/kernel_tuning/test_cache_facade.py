"""Sensor for the ``cache`` subpackage split (cache.py -> cache/ package).

Guards the split's public + test-referenced surface and, critically, the
monkeypatch-visibility contract: the HW-fingerprint probes are patched on the
FACADE package by every kernel-tuning test, and ``hw_fingerprint`` (defined in a
submodule) must honor a facade patch. If this file goes red the split has broken
either re-export completeness or the facade late-resolution wiring.
"""
from __future__ import annotations

import os

import pytest

from pyutilz.performance.kernel_tuning import cache as ktc
from pyutilz.performance.kernel_tuning.cache import cache_base as base
from pyutilz.performance.kernel_tuning.cache import cache_class
from pyutilz.performance.kernel_tuning.cache import cache_hooks as hooks
from pyutilz.performance.kernel_tuning.cache import region_matching


def test_facade_reexports_public_and_test_referenced_symbols():
    # Public API + the internals the concurrency / unit tests reach for as
    # ``ktc.<name>``. A missing one means a test would AttributeError.
    for name in (
        # public
        "SCHEMA_VERSION", "KernelTuningCache", "TuningHooks", "LoggerHooks",
        "cache_dir", "cache_path", "host_cache_dir", "register_default_cache",
        "hw_fingerprint",
        # test-referenced internals / monkeypatch targets
        "_cpu_model_slug", "_gpu_slug_and_cc", "_gpu_summary_cached",
        "gpu_capability_summary", "_pid_alive", "_kernel_dir", "_build_provenance",
        "provenance_changed", "_TUNED_THIS_PROCESS", "_DEFAULT_HOOKS",
        "_DEFAULT_INSTANCE", "_DEFAULT_CACHE", "_AXIS_SUFFIXES", "_region_matches",
        "_async_sweep_hw_busy", "_async_sweep_start_delay", "threading",
    ):
        assert hasattr(ktc, name), f"facade missing {name!r}"


def test_facade_symbols_are_the_same_objects_as_submodules():
    # Re-export identity: the facade names ARE the submodule definitions (no copy).
    assert ktc.KernelTuningCache is cache_class.KernelTuningCache
    assert ktc.register_default_cache is cache_class.register_default_cache
    assert ktc.hw_fingerprint is base.hw_fingerprint
    assert ktc._cpu_model_slug is base._cpu_model_slug
    assert ktc._gpu_slug_and_cc is base._gpu_slug_and_cc
    assert ktc._pid_alive is base._pid_alive
    assert ktc._DEFAULT_HOOKS is hooks._DEFAULT_HOOKS
    assert ktc._region_matches is region_matching._region_matches
    # _TUNED_THIS_PROCESS is a shared mutable set -- same object so tests'
    # ``.clear()`` on the facade is seen by the class.
    assert ktc._TUNED_THIS_PROCESS is base._TUNED_THIS_PROCESS


def test_facade_monkeypatch_of_cpu_probe_is_seen_by_hw_fingerprint(tmp_path, monkeypatch):
    # THE hazard the split had to preserve: hw_fingerprint lives in ``base`` but a
    # test patches ``cache._cpu_model_slug`` on the facade; hw_fingerprint must
    # resolve the patched probe.
    monkeypatch.setenv("PYUTILZ_KERNEL_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("PYUTILZ_HW_FP_REFRESH", "1")  # bypass the on-disk short-circuit
    monkeypatch.setattr(ktc, "_cpu_model_slug", lambda: "sensorcpu")
    monkeypatch.setattr(ktc, "_gpu_slug_and_cc", lambda: ("no-gpu", ""))
    ktc.hw_fingerprint.cache_clear()
    try:
        fp = ktc.hw_fingerprint()
        assert fp == "cpu_sensorcpu_no-gpu", fp
    finally:
        ktc.hw_fingerprint.cache_clear()


def test_facade_monkeypatch_of_gpu_capability_is_seen(tmp_path, monkeypatch):
    # Same contract for the GPU capability probe patched on the facade.
    from unittest import mock
    monkeypatch.setenv("PYUTILZ_KERNEL_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("PYUTILZ_HW_FP_REFRESH", "1")
    monkeypatch.setattr(ktc, "_cpu_model_slug", lambda: "sensorcpu")
    try:
        with mock.patch.object(ktc, "gpu_capability_summary", return_value=None):
            ktc._gpu_summary_cached.cache_clear()
            ktc.hw_fingerprint.cache_clear()
            fp = ktc.hw_fingerprint()
        assert "no-gpu" in fp, fp
    finally:
        ktc._gpu_summary_cached.cache_clear()
        ktc.hw_fingerprint.cache_clear()

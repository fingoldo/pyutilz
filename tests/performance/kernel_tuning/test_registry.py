"""Tests for the kernel tuner registry (TunerSpec, @kernel_tuner, discovery)."""
import types
import sys

import pytest

from pyutilz.performance.kernel_tuning.registry import (
    TunerSpec,
    kernel_tuner,
    get_registry,
    discover_tuners,
    retune_all,
    _REGISTRY,
)


@pytest.fixture(autouse=True)
def _clear_registry():
    """Each test starts with an empty registry and leaves it empty."""
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()


def _np(): ...


def _nb(): ...


def test_tunerspec_fields_and_defaults():
    spec = TunerSpec(
        kernel_name="k1",
        variant_fns=(_np, _nb),
        tuner=lambda *a: {},
        axes={"ndim_eq": [2, 3]},
        fallback=_np,
    )
    assert spec.kernel_name == "k1"
    assert spec.variant_fns == (_np, _nb)
    assert spec.extra_fns == ()
    assert spec.salt == 0
    assert spec.env_key is None
    assert spec.gpu_capable is False
    assert spec.equiv_tol is None


def test_kernel_tuner_registers():
    kernel_tuner(
        kernel_name="joint_hist_2d",
        variant_fns=(_np,),
        tuner=lambda: [],
        axes={"ndim_eq": [2]},
        fallback=_np,
    )
    reg = get_registry()
    assert "joint_hist_2d" in reg  # keyed by the globally-unique kernel_name
    assert reg["joint_hist_2d"].kernel_name == "joint_hist_2d"


def test_kernel_tuner_returns_spec():
    spec = kernel_tuner(kernel_name="k_ret", variant_fns=(_np,), tuner=lambda: [], axes={}, fallback=_np)
    assert spec.kernel_name == "k_ret"
    assert get_registry()["k_ret"] is spec


def test_duplicate_registration_raises():
    kernel_tuner(kernel_name="dup", variant_fns=(_np,), tuner=lambda: [], axes={}, fallback=_np)
    with pytest.raises(ValueError, match="Duplicate kernel_tuner"):
        kernel_tuner(kernel_name="dup", variant_fns=(_np,), tuner=lambda: [], axes={}, fallback=_np)


def test_get_registry_returns_copy():
    kernel_tuner(kernel_name="kc", variant_fns=(_np,), tuner=lambda: [], axes={}, fallback=_np)
    reg = get_registry()
    reg.clear()  # mutating the copy must not affect the global registry
    assert len(get_registry()) == 1


def test_discover_tuners_accumulates_not_clears(monkeypatch):
    # discover_tuners must NOT clear the registry: registration fires at module
    # import, and Python's import cache means an already-imported module never
    # re-runs its kernel_tuner(...) -- clearing would permanently lose it. So a
    # spec registered before discovery survives a walk of a spec-free package.
    kernel_tuner(kernel_name="stray", variant_fns=(_np,), tuner=lambda: [], axes={}, fallback=_np)
    assert "stray" in get_registry()

    found = discover_tuners(package="json", warn_on_import_fail=False)
    assert "stray" in found  # preserved, not wiped
    assert "stray" in get_registry()


def test_discover_tuners_unknown_package_preserves_registry():
    kernel_tuner(kernel_name="keep", variant_fns=(_np,), tuner=lambda: [], axes={}, fallback=_np)
    found = discover_tuners(package="no_such_package_xyz", warn_on_import_fail=False)
    assert "keep" in found  # an unimportable package returns the existing registry, not {}


def test_retune_all_no_specs_returns_empty():
    # Discovering a spec-free package -> retune_all returns {}.
    result = retune_all(package="json")
    assert result == {}


def test_run_spec_tuning_populates_cache():
    from pyutilz.performance.kernel_tuning.registry import _run_spec_tuning
    from pyutilz.performance.kernel_tuning.cache import KernelTuningCache

    cache = KernelTuningCache(in_memory=True)
    spec = TunerSpec(
        kernel_name="fake_k",
        variant_fns=(_np, _nb),
        tuner=lambda: [{"n_max": 100, "backend_choice": "numpy"}, {"backend_choice": "numba"}],
        axes={"n": [100, 1000]},
        fallback={"backend_choice": "numpy"},
    )
    n = _run_spec_tuning(cache, spec, code_version="cv1", device_id=None, force=False, hooks=None)
    assert n == 2
    assert cache.has("fake_k")
    # force=True re-evicts then re-tunes -> still 2 regions
    n2 = _run_spec_tuning(cache, spec, code_version="cv1", device_id=None, force=True, hooks=None)
    assert n2 == 2


def test_group_gpus_by_model(monkeypatch):
    """_group_gpus_by_model groups devices by name + compute capability."""

    import pyutilz.performance.kernel_tuning.registry as reg

    class _G:
        def __init__(self, gid, name, cc):
            self.id, self.name, self.compute_capability = gid, name, cc

    fake = types.ModuleType("GPUtil")
    fake.getGPUs = lambda: [_G(0, "NVIDIA RTX 4090", (8, 9)), _G(1, "NVIDIA RTX 4090", (8, 9)), _G(2, "Tesla V100", (7, 0))]
    monkeypatch.setitem(sys.modules, "GPUtil", fake)
    groups = reg._group_gpus_by_model()
    assert len(groups) == 2  # the two identical 4090s collapse into one model
    assert sorted(len(v) for v in groups.values()) == [1, 2]
    assert [0, 1] in [sorted(v) for v in groups.values()]


def test_pick_least_loaded_device(monkeypatch):
    """_pick_least_loaded_device returns the lowest-load available GPU; None if all busy."""

    import pyutilz.performance.kernel_tuning.registry as reg

    class _G:
        def __init__(self, gid, load):
            self.id, self.load = gid, load

    fake = types.ModuleType("GPUtil")
    fake.getGPUs = lambda: [_G(0, 0.7), _G(1, 0.2), _G(2, 0.5)]
    monkeypatch.setitem(sys.modules, "GPUtil", fake)
    assert reg._pick_least_loaded_device([0, 1, 2], idle_wait_tries=1, idle_wait_sec=0.0) == 1
    assert reg._pick_least_loaded_device([0, 2], idle_wait_tries=1, idle_wait_sec=0.0) == 2  # subset
    fake.getGPUs = lambda: [_G(0, 0.95), _G(1, 0.9)]  # all > 0.8 -> busy
    assert reg._pick_least_loaded_device([0, 1], idle_wait_tries=1, idle_wait_sec=0.0) is None


def test_spec_choose_returns_fallback_on_empty_cache(monkeypatch, tmp_path):
    """spec.choose() -> the fallback backend when the cache is empty + tuner is a no-op."""
    monkeypatch.setenv("PYUTILZ_KERNEL_CACHE_DIR", str(tmp_path))
    spec = kernel_tuner(kernel_name="zzz_choose", variant_fns=(_np,), tuner=lambda: [],
                        axes={"n": [10]}, fallback={"backend_choice": "cpu"})
    assert spec.choose(n=5) == "cpu"
    assert spec.choose(n=5) == "cpu"  # memoized


def test_spec_choose_callable_fallback(monkeypatch, tmp_path):
    """A callable fallback (dims -> str) gives the dynamic heuristic via choose()."""
    monkeypatch.setenv("PYUTILZ_KERNEL_CACHE_DIR", str(tmp_path))
    spec = kernel_tuner(kernel_name="zzz_choose2", variant_fns=(_np,), tuner=lambda: [],
                        axes={"n": [10]}, fallback=lambda n: "gpu" if n >= 100 else "cpu")
    assert spec.choose(n=5) == "cpu"
    assert spec.choose(n=500) == "gpu"

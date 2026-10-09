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


def test_duplicate_registration_is_idempotent():
    # Re-registering the same kernel_name overwrites (last wins) instead of raising: a module re-import
    # (importlib.reload, a test dropping + re-importing the consumer subgraph, or two import paths to the same
    # module) re-runs the decorator, and crashing there would break every consumer imported afterwards.
    kernel_tuner(kernel_name="dup", variant_fns=(_np,), tuner=lambda: [], axes={}, fallback=_np)
    second = kernel_tuner(kernel_name="dup", variant_fns=(_np,), tuner=lambda: [], axes={}, fallback=_np)
    assert get_registry()["dup"] is second
    assert len(get_registry()) == 1


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


def test_run_spec_tuning_with_a_dims_callable_fallback_and_an_empty_sweep_does_not_raise():
    """Real specs carry a fallback that is a callable OF the dims (n_samples, ...). Offline tuning has no dims, so an empty sweep used to call it bare and die with TypeError."""
    from pyutilz.performance.kernel_tuning.registry import _run_spec_tuning
    from pyutilz.performance.kernel_tuning.cache import KernelTuningCache

    cache = KernelTuningCache(in_memory=True)

    def fallback(n_samples):
        return {"backend_choice": "numpy" if n_samples < 1000 else "numba"}

    spec = TunerSpec(kernel_name="needs_dims", variant_fns=(_np, _nb), tuner=lambda: [], axes={"n": [100, 1000]}, fallback=fallback)
    assert _run_spec_tuning(cache, spec, code_version="cv1", device_id=None, force=True, hooks=None) == 0


def test_a_sweep_that_raises_is_reported_at_warning_not_swallowed_at_debug(caplog):
    """'The sweep raised' and 'the sweep found nothing' both persisted zero regions; only the second is silent now."""
    from pyutilz.performance.kernel_tuning.cache import KernelTuningCache

    def boom():
        raise RuntimeError("no GPU variants could be built")

    cache = KernelTuningCache(in_memory=True)
    with caplog.at_level("WARNING"):
        cache.get_or_tune("raising_kernel", dims={}, tuner=boom, axes=["n"], fallback={"backend_choice": "numpy"}, once_per_process=False, async_sweep=False)
    messages = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("raising_kernel" in m and "RuntimeError" in m and "no GPU variants could be built" in m for m in messages), messages


def test_group_gpus_by_model(monkeypatch):
    """_group_gpus_by_model groups devices by name + compute capability, taking the capability from CUDA (the fake GPUs carry only what GPUtil.GPU really has)."""

    import pyutilz.performance.kernel_tuning.registry as reg

    class _G:  # the attributes of a real GPUtil.GPU that matter here: no compute_capability
        def __init__(self, gid, name):
            self.id, self.name = gid, name

    capability = {0: (8, 9), 1: (8, 9), 2: (7, 0)}
    fake = types.ModuleType("GPUtil")
    fake.getGPUs = lambda: [_G(0, "NVIDIA RTX 4090"), _G(1, "NVIDIA RTX 4090"), _G(2, "Tesla V100")]
    monkeypatch.setitem(sys.modules, "GPUtil", fake)
    monkeypatch.setattr(reg, "_gpu_compute_capability", lambda device_id: capability[device_id])
    groups = reg._group_gpus_by_model()
    assert len(groups) == 2  # the two identical 4090s collapse into one model
    assert sorted(len(v) for v in groups.values()) == [1, 2]
    assert [0, 1] in [sorted(v) for v in groups.values()]
    assert set(groups) == {"NVIDIA_89", "Tesla_70"}


def test_same_name_different_capability_are_different_models(monkeypatch):
    """Two cards with the same marketing name but another compute capability must not share tuned parameters."""

    import pyutilz.performance.kernel_tuning.registry as reg

    class _G:
        def __init__(self, gid, name):
            self.id, self.name = gid, name

    fake = types.ModuleType("GPUtil")
    fake.getGPUs = lambda: [_G(0, "NVIDIA GeForce X"), _G(1, "NVIDIA GeForce X")]
    monkeypatch.setitem(sys.modules, "GPUtil", fake)
    monkeypatch.setattr(reg, "_gpu_compute_capability", lambda device_id: (6, 1) if device_id == 0 else (7, 5))
    assert len(reg._group_gpus_by_model()) == 2


def test_gpu_compute_capability_parses_cupy_and_falls_back_to_numba(monkeypatch):
    """cupy reports the capability as a digit string ("61", "100"); without cupy the numba route is used; with neither the answer is (0, 0), not an exception."""

    import pyutilz.performance.kernel_tuning.registry as reg

    class _Dev:
        def __init__(self, cc):
            self.compute_capability = cc

    fake_cupy = types.ModuleType("cupy")
    fake_cupy.cuda = types.SimpleNamespace(Device=lambda i: _Dev({0: "61", 1: "100", 2: "89"}[i]))
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    assert reg._gpu_compute_capability(0) == (6, 1)
    assert reg._gpu_compute_capability(1) == (10, 0)
    assert reg._gpu_compute_capability(2) == (8, 9)

    monkeypatch.setitem(sys.modules, "cupy", None)  # import cupy -> ImportError
    fake_probing = types.ModuleType("pyutilz.system.system")
    fake_probing.get_gpu_cuda_capabilities = lambda device_id=0: {"COMPUTE_CAPABILITY_MAJOR": 7, "COMPUTE_CAPABILITY_MINOR": 5}
    monkeypatch.setitem(sys.modules, "pyutilz.system.system", fake_probing)
    assert reg._gpu_compute_capability(0) == (7, 5)

    fake_probing.get_gpu_cuda_capabilities = lambda device_id=0: None  # numba unavailable
    assert reg._gpu_compute_capability(0) == (0, 0)


def test_gpu_compute_capability_matches_the_installed_gpu_when_cuda_is_present():
    """On a real CUDA host the helper agrees with the numba probe the rest of the package uses (skipped without a GPU)."""

    import pytest

    cuda = pytest.importorskip("numba.cuda")
    if not cuda.is_available():
        pytest.skip("no CUDA device")
    import pyutilz.performance.kernel_tuning.registry as reg
    from pyutilz.system.system import get_gpu_cuda_capabilities

    caps = get_gpu_cuda_capabilities(device_id=0) or {}
    expected = (int(caps.get("COMPUTE_CAPABILITY_MAJOR", 0)), int(caps.get("COMPUTE_CAPABILITY_MINOR", 0)))
    assert expected != (0, 0)
    assert reg._gpu_compute_capability(0) == expected


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
    spec = kernel_tuner(kernel_name="zzz_choose", variant_fns=(_np,), tuner=lambda: [], axes={"n": [10]}, fallback={"backend_choice": "cpu"})
    assert spec.choose(n=5) == "cpu"
    assert spec.choose(n=5) == "cpu"  # memoized


def test_spec_choose_callable_fallback(monkeypatch, tmp_path):
    """A callable fallback (dims -> str) gives the dynamic heuristic via choose()."""
    monkeypatch.setenv("PYUTILZ_KERNEL_CACHE_DIR", str(tmp_path))
    spec = kernel_tuner(kernel_name="zzz_choose2", variant_fns=(_np,), tuner=lambda: [], axes={"n": [10]}, fallback=lambda n: "gpu" if n >= 100 else "cpu")
    assert spec.choose(n=5) == "cpu"
    assert spec.choose(n=500) == "gpu"


def test_cache_public_code_version_stale():
    """The cache exposes a PUBLIC code_version_stale (registry no longer reaches into the private one)."""
    from pyutilz.performance.kernel_tuning.cache import KernelTuningCache

    cache = KernelTuningCache(in_memory=True)
    assert hasattr(cache, "code_version_stale")
    spec = TunerSpec(kernel_name="cv_k", variant_fns=(_np,), tuner=lambda: [{"backend_choice": "numpy"}], axes={}, fallback={"backend_choice": "numpy"})
    from pyutilz.performance.kernel_tuning.registry import _run_spec_tuning
    _run_spec_tuning(cache, spec, code_version="cvA", device_id=None, force=False, hooks=None)
    # Same code_version -> not stale; a different one -> stale; None -> never stale.
    assert cache.code_version_stale("cv_k", "cvA") is False
    assert cache.code_version_stale("cv_k", "cvB") is True
    assert cache.code_version_stale("cv_k", None) is False
    # Public method delegates to the private one identically.
    assert cache.code_version_stale("cv_k", "cvB") == cache._code_version_stale("cv_k", "cvB")


class _FakeTunedCache:
    """Deterministically reports every kernel as already-tuned -- lets ``choose()`` memoize on
    the FIRST call for every distinct ``dims``, sidestepping the real cache's background-sweep
    timing (which the existing ``test_spec_choose_returns_fallback_on_empty_cache`` shows isn't
    guaranteed to settle within one synchronous call)."""

    def get_or_tune(self, *args, **kwargs):
        return {"backend_choice": "cpu"}

    def has(self, *args, **kwargs):
        return True

    def code_version_stale(self, *args, **kwargs):
        return False


class TestChoiceCacheBoundedLRU:
    """Regression (2026-07-21 audit round 2, LOW): ``_choice_cache`` used to be a plain
    unbounded dict -- a long-running service seeing a continuous stream of distinct dims
    combinations grew one permanent entry per combination for the life of the process. It's
    now an OrderedDict-backed LRU bounded by ``_CHOICE_CACHE_MAX_SIZE``."""

    def _patch_fake_tuned_cache(self, monkeypatch):
        import pyutilz.performance.kernel_tuning.cache as cache_mod
        monkeypatch.setattr(cache_mod.KernelTuningCache, "load_or_create", classmethod(lambda cls: _FakeTunedCache()))

    def _spec(self, kernel_name: str) -> TunerSpec:
        return TunerSpec(kernel_name=kernel_name, variant_fns=(_np,), tuner=lambda: [], axes={"n": [10]}, fallback={"backend_choice": "cpu"})

    def test_cache_bounded_by_max_size(self, monkeypatch):
        self._patch_fake_tuned_cache(monkeypatch)
        spec = self._spec("bounded_direct")
        spec._CHOICE_CACHE_MAX_SIZE = 3
        for n in range(10):
            spec.choose(n=n)
        assert len(spec._choice_cache) == 3

    def test_lru_order_evicts_least_recently_used(self, monkeypatch):
        self._patch_fake_tuned_cache(monkeypatch)
        spec = self._spec("lru_direct")
        spec._CHOICE_CACHE_MAX_SIZE = 2

        spec.choose(n=1)
        spec.choose(n=2)
        # Touch n=1 again -- it becomes most-recently-used, so the NEXT insertion should
        # evict n=2 (least-recently-used), not n=1.
        spec.choose(n=1)
        spec.choose(n=3)

        cached_ns = {dict(k).get("n") for k in spec._choice_cache}
        assert 1 in cached_ns
        assert 2 not in cached_ns
        assert 3 in cached_ns

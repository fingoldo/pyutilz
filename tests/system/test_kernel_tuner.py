"""Tests for the kernel tuner registry (TunerSpec, @kernel_tuner, discovery)."""
import types
import sys

import pytest

from pyutilz.system.kernel_tuner import (
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


def test_discover_tuners_clears_then_repopulates(monkeypatch):
    # Register a stray spec; discover_tuners() on a trivial package should
    # clear it (fresh state), then repopulate from the walked package.
    kernel_tuner(kernel_name="stray", variant_fns=(_np,), tuner=lambda: [], axes={}, fallback=_np)
    assert len(get_registry()) == 1

    # Walk a package with no kernel_tuner specs (use a stdlib package).
    found = discover_tuners(package="json", warn_on_import_fail=False)
    # json has no @kernel_tuner specs -> registry cleared, nothing added.
    assert found == {}
    assert len(get_registry()) == 0


def test_discover_tuners_unknown_package_returns_empty():
    found = discover_tuners(package="no_such_package_xyz", warn_on_import_fail=False)
    assert found == {}


def test_retune_all_no_specs_returns_empty():
    # Discovering a spec-free package -> retune_all returns {}.
    result = retune_all(package="json")
    assert result == {}


def test_run_spec_tuning_populates_cache():
    from pyutilz.system.kernel_tuner import _run_spec_tuning
    from pyutilz.system.kernel_tuning_cache import KernelTuningCache

    cache = KernelTuningCache(in_memory=True)
    spec = TunerSpec(
        kernel_name="fake_k",
        variant_fns=(_np, _nb),
        tuner=lambda: [{"n_max": 100, "backend_choice": "numpy"}, {"backend_choice": "numba"}],
        axes={"n": [100, 1000]},
        fallback={"backend_choice": "numpy"},
    )
    n = _run_spec_tuning(cache, spec, code_version="cv1", device_id=None, force=False,
                         idle_wait_tries=1, idle_wait_sec=0.0, hooks=None)
    assert n == 2
    assert cache.has("fake_k")
    # force=True re-evicts then re-tunes -> still 2 regions
    n2 = _run_spec_tuning(cache, spec, code_version="cv1", device_id=None, force=True,
                          idle_wait_tries=1, idle_wait_sec=0.0, hooks=None)
    assert n2 == 2

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


def test_kernel_tuner_decorator_registers():
    @kernel_tuner(
        kernel_name="joint_hist_2d",
        variant_fns=(_np,),
        tuner=lambda *a: {},
        axes={"ndim_eq": [2]},
        fallback=_np,
    )
    def _gpu_variant():
        return _nb

    reg = get_registry()
    # Key is (module, kernel_name); module is this test module.
    keys = [k for k in reg if k[1] == "joint_hist_2d"]
    assert len(keys) == 1
    assert reg[keys[0]].kernel_name == "joint_hist_2d"
    # Decorator returns the original function unmodified.
    assert _gpu_variant() is _nb


def test_kernel_tuner_decorator_does_not_call_fn():
    called = []

    @kernel_tuner(
        kernel_name="k_nocall",
        variant_fns=(_np,),
        tuner=lambda *a: {},
        axes={},
        fallback=_np,
    )
    def _marker():
        called.append(1)
        return 42

    # Registration must NOT invoke the decorated function.
    assert called == []


def test_duplicate_registration_raises():
    def make():
        @kernel_tuner(
            kernel_name="dup",
            variant_fns=(_np,),
            tuner=lambda *a: {},
            axes={},
            fallback=_np,
        )
        def _f():
            pass

    make()
    with pytest.raises(ValueError, match="Duplicate kernel_tuner"):
        make()


def test_get_registry_returns_copy():
    @kernel_tuner(
        kernel_name="kc",
        variant_fns=(_np,),
        tuner=lambda *a: {},
        axes={},
        fallback=_np,
    )
    def _f():
        pass

    reg = get_registry()
    reg.clear()  # mutating the copy must not affect the global registry
    assert len(get_registry()) == 1


def test_discover_tuners_clears_then_repopulates(monkeypatch):
    # Register a stray spec; discover_tuners() on a trivial package should
    # clear it (fresh state), then repopulate from the walked package.
    @kernel_tuner(
        kernel_name="stray",
        variant_fns=(_np,),
        tuner=lambda *a: {},
        axes={},
        fallback=_np,
    )
    def _f():
        pass

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

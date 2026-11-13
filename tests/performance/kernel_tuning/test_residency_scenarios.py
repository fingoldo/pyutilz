"""Residency scenarios of the kernel-tuning entries: declared ``(data_in_residency, data_out_residency)`` pairs, one cache entry each, picked by ``choose``."""
from __future__ import annotations

import numpy as np
import pytest

from pyutilz.performance.kernel_tuning import registry as reg
from pyutilz.performance.kernel_tuning.residency import array_location, normalize_scenarios, operands_residency, scenario_suffix


class _Device:
    """An array-like that claims to live on the device (the cuda array interface)."""

    shape = (3,)
    __cuda_array_interface__ = {"shape": (3,), "typestr": "<f4", "version": 2}


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    """Each test registers into an empty registry."""
    monkeypatch.setattr(reg, "_REGISTRY", {})


def _spec(residencies, tuner=None, name="res_kernel"):
    """A spec whose fallback names the scenario it was resolved for (the tuner never runs: ``choose`` is checked through the fallback)."""
    return reg.kernel_tuner(
        kernel_name=name,
        variant_fns=(),
        tuner=tuner or (lambda data_in_residency="host", data_out_residency="host": []),
        axes={"n_samples": [1000]},
        fallback=lambda n_samples: "fallback",
        residencies=residencies,
    )


def test_normalize_validates_deduplicates_and_defaults():
    """A bad placement raises, duplicates collapse in order, an empty declaration is host-in / host-out."""
    assert normalize_scenarios([]) == (("host", "host"),)
    assert normalize_scenarios([("device", "device"), ("host", "host"), ("device", "device")]) == (("device", "device"), ("host", "host"))
    with pytest.raises(ValueError):
        normalize_scenarios([("host", "gpu")])


def test_suffix_keeps_the_default_entry_name():
    """The default scenario keeps the plain kernel name (an existing cache entry stays valid); the others get ``__<in>_<out>``."""
    assert scenario_suffix(("host", "host")) == ""
    assert scenario_suffix(("device", "host")) == "__device_host"


def test_operands_residency_reads_cuda_array_interface_and_ignores_scalars():
    """All array operands on the device: device; a numpy operand among them: host; scalars and None are ignored."""
    assert operands_residency([_Device(), _Device(), 5, None]) == "device"
    assert operands_residency([_Device(), np.zeros(2)]) == "host"
    assert operands_residency([]) == "host"
    assert array_location(_Device()) == "device" and array_location(np.zeros(1)) == "host"


def test_a_single_scenario_spec_is_unchanged_and_ignores_the_keys():
    """The default declaration: the spec is its own only scenario, its tuner stays zero-arg, the residency keys are consumed and do not reach the fallback."""
    spec = reg.kernel_tuner(kernel_name="plain", variant_fns=(), tuner=lambda: [], axes={"n_samples": [1000]}, fallback=lambda n_samples: "host")
    assert spec.scenario_specs() == [spec]
    assert spec.choose(n_samples=10, data_in_residency="device", data_out_residency="device") == "host"


def test_each_declared_scenario_is_its_own_entry_with_a_bound_tuner():
    """Two declared scenarios: two specs named ``<name>`` and ``<name>__device_device``, each tuner called with its own placement."""
    seen = []
    spec = _spec([("host", "host"), ("device", "device")], tuner=lambda data_in_residency, data_out_residency: seen.append((data_in_residency, data_out_residency)) or [])
    specs = spec.scenario_specs()
    assert [s.kernel_name for s in specs] == ["res_kernel", "res_kernel__device_device"]
    for s in specs:
        s.tuner()
    assert seen == [("host", "host"), ("device", "device")]


def test_scenario_for_picks_the_declared_pair_else_the_first():
    """An undeclared placement is served by the first declared scenario (a resident-only entry serves everyone with its resident verdict)."""
    both = _spec([("host", "host"), ("device", "device")], name="both")
    assert both.scenario_for("device", "device") == ("device", "device")
    assert both.scenario_for("device", "host") == ("host", "host")
    resident_only = _spec([("device", "device")], name="resident_only")
    assert resident_only.scenario_for("host", "host") == ("device", "device")


def test_choose_routes_by_keys_and_by_operands(monkeypatch):
    """``choose`` consumes the residency keys (and ``operands=``), hands the remaining dims to the scenario entry that serves the placement."""
    spec = _spec([("host", "host"), ("device", "device")])
    routed = []
    for s in spec.scenario_specs():
        monkeypatch.setattr(s, "_choose_plain", lambda dims, _s=s: routed.append((_s.kernel_name, dict(dims))) or "ok")
    spec.choose(n_samples=5)
    spec.choose(n_samples=6, data_in_residency="device", data_out_residency="device")
    spec.choose(n_samples=7, operands=(_Device(), 3), data_out_residency="device")
    spec.choose(n_samples=8, operands=(np.zeros(2),))
    assert routed == [
        ("res_kernel", {"n_samples": 5}),
        ("res_kernel__device_device", {"n_samples": 6}),
        ("res_kernel__device_device", {"n_samples": 7}),
        ("res_kernel", {"n_samples": 8}),
    ]


def test_several_scenarios_need_a_tuner_that_takes_the_placement():
    """A zero-arg tuner cannot time two placements, so registering it with two scenarios is refused."""
    with pytest.raises(TypeError):
        reg.kernel_tuner(kernel_name="bad", variant_fns=(), tuner=lambda: [], axes={"n_samples": [1]}, fallback="host", residencies=[("host", "host"), ("device", "device")])


def test_tune_spec_tunes_every_scenario(monkeypatch):
    """``tune_spec`` runs the sweep once per declared scenario and sums the regions."""
    ran = []
    monkeypatch.setattr(reg, "KernelTuningCache", lambda: object())
    monkeypatch.setattr(reg, "compute_code_version", lambda *a, **k: "v1")
    monkeypatch.setattr(reg, "_run_spec_tuning", lambda cache, spec, code_version, device_id, force, hooks, skip_existing=True: ran.append(spec.kernel_name) or 2)
    spec = _spec([("host", "host"), ("host", "device"), ("device", "device")], name="multi")
    assert reg.tune_spec(spec, force=False) == 6
    assert ran == ["multi", "multi__host_device", "multi__device_device"]

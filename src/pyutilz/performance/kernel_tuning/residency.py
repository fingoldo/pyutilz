"""Data residency vocabulary of the kernel-tuning entries.

A host-versus-device sweep times a callable end to end, so its verdict holds only for the placement of the data it was timed with. An entry therefore declares the placements its
callers can have, as ``(data_in_residency, data_out_residency)`` pairs over ``("host", "device")``, is swept once per declared pair, and is looked up by the pair the caller is in.
The two keys are recognised by ``TunerSpec.choose`` itself: pass ``data_in_residency=`` / ``data_out_residency=`` explicitly, or pass ``operands=(...)`` and the input side is read
from them (``operands_residency``); a caller that says nothing is host-in / host-out.
"""
from __future__ import annotations

from typing import Any, Iterable

__all__ = [
    "RESIDENCIES",
    "DEFAULT_SCENARIO",
    "IN_KEY",
    "OUT_KEY",
    "OPERANDS_KEY",
    "array_location",
    "operands_residency",
    "normalize_scenarios",
    "scenario_suffix",
]

RESIDENCIES = ("host", "device")
DEFAULT_SCENARIO = ("host", "host")
IN_KEY = "data_in_residency"
OUT_KEY = "data_out_residency"
OPERANDS_KEY = "operands"


def array_location(x: Any) -> str:
    """Residency of an array-like: ``"device"`` if GPU-resident (a cupy array or anything exposing ``__cuda_array_interface__``, numba device arrays included), else ``"host"``."""
    return "device" if hasattr(x, "__cuda_array_interface__") else "host"


def operands_residency(operands: Iterable[Any]) -> str:
    """``"device"`` when every array operand is on the device, else ``"host"``.

    A mix is ``"host"``: the device backend then pays the upload of the host operand, which is the host-in cost. Operands that are not arrays (scalars, strings, None) are ignored."""
    flags = []
    for op in operands:
        if op is None or isinstance(op, (str, bytes, int, float, bool)):
            continue
        if hasattr(op, "__cuda_array_interface__"):
            flags.append(True)
        elif hasattr(op, "shape"):
            flags.append(False)
    return "device" if flags and all(flags) else "host"


def normalize_scenarios(scenarios: Iterable[Any]) -> tuple:
    """The declared scenarios as a tuple of validated ``(in, out)`` pairs, de-duplicated in order; an empty declaration is the default host-in / host-out scenario."""
    out = []
    for sc in scenarios:
        pair = (str(sc[0]), str(sc[1]))
        if pair[0] not in RESIDENCIES or pair[1] not in RESIDENCIES:
            raise ValueError(f"residency scenario must be a pair over {RESIDENCIES}, got {tuple(sc)!r}")
        if pair not in out:
            out.append(pair)
    return tuple(out) if out else (DEFAULT_SCENARIO,)


def scenario_suffix(scenario: tuple) -> str:
    """The suffix that names a scenario's cache entry: empty for the default host-in / host-out (so an existing single-scenario entry keeps its key), else ``__<in>_<out>``."""
    return "" if tuple(scenario) == DEFAULT_SCENARIO else f"__{scenario[0]}_{scenario[1]}"

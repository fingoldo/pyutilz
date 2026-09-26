"""Region <-> dims matching for the kernel-tuning cache lookup (pure, no HW/disk deps)."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

# Region keys ending with one of these suffixes are interpreted as axis
# CONSTRAINTS by the matcher; everything else in a region dict is opaque
# decision payload. ``lookup`` strips exactly these suffixes from its return.
_AXIS_SUFFIXES = ("_max", "_min", "_eq")

# Integer op codes for a COMPILED constraint (see ``KernelTuningCache._lookup_plan``). The hot
# lookup compares small ints instead of re-deriving three f-string keys per axis per region.
_OP_MAX = 0
_OP_MIN = 1
_OP_EQ = 2

# Suffix -> op code, in the order a region key must be tested. "_max"/"_min"/"_eq" are mutually
# exclusive endings, so the first match wins.
_SUFFIX_OPS = (("_max", _OP_MAX), ("_min", _OP_MIN), ("_eq", _OP_EQ))


def _normalize_eq(value: Any) -> Any:
    """Tuples become lists, recursively. JSON storage turns every tuple in a region into a list, so an ``_eq``
    constraint on a tuple value could never match again once reloaded from disk unless both sides are normalized."""
    if isinstance(value, (tuple, list)):
        return [_normalize_eq(v) for v in value]
    return value


def _is_nan(value: Any) -> bool:
    """True for a float NaN (python float or numpy floating scalar), which compares False against every bound."""
    return isinstance(value, (float, np.floating)) and bool(math.isnan(value))


def _region_match_reason(region: dict, dims: dict) -> tuple:
    """``(ok, reason)`` for one region against a dims dict; the single implementation behind ``_region_matches`` and
    ``lookup_explain``, and the reference the compiled ``lookup`` plan must agree with.

    A region matches iff, for every requested dim, the region's constraints on that axis hold:
      * ``<axis>_max``: dim <= max   (numeric upper cap)
      * ``<axis>_min``: dim >= min   (numeric lower cap)
      * ``<axis>_eq`` : dim == value (categorical / exact; tuples and lists compare equal element-wise)
    A constraint key absent or None is unconstrained; a dim with no constraint key in the region is ignored.
    A NaN dim matches NO constrained region: every comparison with NaN is False, so ``nan > cap`` used to let it
    through every ``_max``/``_min`` check.
    """
    for axis_name, axis_value in dims.items():
        cap = region.get(f"{axis_name}_max")
        lo = region.get(f"{axis_name}_min")
        eq = region.get(f"{axis_name}_eq")
        if (cap is not None or lo is not None) and _is_nan(axis_value):
            return False, f"{axis_name} is NaN"
        if cap is not None and axis_value > cap:
            return False, f"{axis_name}={axis_value} > {axis_name}_max={cap}"
        if lo is not None and axis_value < lo:
            return False, f"{axis_name}={axis_value} < {axis_name}_min={lo}"
        if eq is not None and _normalize_eq(axis_value) != _normalize_eq(eq):
            return False, f"{axis_name}={axis_value!r} != {axis_name}_eq={eq!r}"
    return True, "all constraints satisfied"


def _region_matches(region: dict, dims: dict) -> bool:
    """True iff ``region`` matches ``dims``; see ``_region_match_reason``."""
    return bool(_region_match_reason(region, dims)[0])

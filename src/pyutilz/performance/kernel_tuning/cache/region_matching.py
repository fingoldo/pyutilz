"""Region <-> dims matching for the kernel-tuning cache lookup (pure, no HW/disk deps)."""
from __future__ import annotations

# Region keys ending with one of these suffixes are interpreted as axis
# CONSTRAINTS by the matcher; everything else in a region dict is opaque
# decision payload. ``lookup`` strips exactly these suffixes from its return.
_AXIS_SUFFIXES = ("_max", "_min", "_eq")


def _region_matches(region: dict, dims: dict) -> bool:
    """A region matches a dims dict iff, for every requested dim, the region's
    constraints on that axis hold:
      * ``<axis>_max``: dim <= max   (numeric upper cap)
      * ``<axis>_min``: dim >= min   (numeric lower cap)
      * ``<axis>_eq`` : dim == value (categorical / exact -- dtype, ndim, location)
    A constraint key absent or None is unconstrained; a dim with no constraint
    key in the region is ignored (the region applies to any value of it)."""
    for axis_name, axis_value in dims.items():
        cap = region.get(f"{axis_name}_max")
        if cap is not None and axis_value > cap:
            return False
        lo = region.get(f"{axis_name}_min")
        if lo is not None and axis_value < lo:
            return False
        eq = region.get(f"{axis_name}_eq")
        if eq is not None and axis_value != eq:
            return False
    return True


def _region_match_reason(region: dict, dims: dict) -> tuple:
    """Like ``_region_matches`` but returns ``(ok, reason)`` -- the first failing
    constraint -- for ``lookup_explain``."""
    for axis_name, axis_value in dims.items():
        cap = region.get(f"{axis_name}_max")
        if cap is not None and axis_value > cap:
            return False, f"{axis_name}={axis_value} > {axis_name}_max={cap}"
        lo = region.get(f"{axis_name}_min")
        if lo is not None and axis_value < lo:
            return False, f"{axis_name}={axis_value} < {axis_name}_min={lo}"
        eq = region.get(f"{axis_name}_eq")
        if eq is not None and axis_value != eq:
            return False, f"{axis_name}={axis_value!r} != {axis_name}_eq={eq!r}"
    return True, "all constraints satisfied"

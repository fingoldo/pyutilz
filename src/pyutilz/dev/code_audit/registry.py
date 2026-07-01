"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

from ._base import Finding
from .mutable_defaults import scan_mutable_defaults
from .closures import scan_late_binding_closures
from .default_via_or import scan_default_via_or_trap
from .broad_except import scan_broad_except_swallows
from .nan_equality import scan_nan_equality
from .mutation_during_iteration import scan_mutation_during_iteration


# --- registry -----------------------------------------------------------


SCANNERS: dict[str, Callable[[Path], list[Finding]]] = {
    "mutable_default": scan_mutable_defaults,
    "late_binding_closure": scan_late_binding_closures,
    "default_via_or": scan_default_via_or_trap,
    "broad_except_swallow": scan_broad_except_swallows,
    "nan_equality": scan_nan_equality,
    "mutation_during_iteration": scan_mutation_during_iteration,
}


def run_all(root: Path, checks: Optional[Iterable[str]] = None) -> list[Finding]:
    """Run every (or selected) scanner against ``root`` and return all
    findings in encounter order. Sort by (severity, check, file, line)
    for stable rendering at the call site."""
    selected = list(SCANNERS) if checks is None else list(checks)
    out: list[Finding] = []
    for name in selected:
        if name not in SCANNERS:
            raise ValueError(
                f"unknown check {name!r}; available: {sorted(SCANNERS)}"
            )
        out.extend(SCANNERS[name](root))
    sev_order = {"P0": 0, "P1": 1, "P2": 2, "Low": 3}
    out.sort(key=lambda f: (sev_order.get(f.severity, 99), f.check, f.file, f.line))
    return out

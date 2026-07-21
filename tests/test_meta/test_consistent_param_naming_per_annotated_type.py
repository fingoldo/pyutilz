"""Meta-test proposed after the 2026-07-21 audit round 2 meta-test review: within one module,
public top-level functions whose FIRST parameter carries the SAME concrete (dotted/attribute)
type annotation should spell that parameter the same way.

Class this catches: this automates the hand-curated pins already established this round --
``pandaslib.frames``'s ``df``/``X``/``ref_df`` split for the identical "a pandas DataFrame"
concept (fixed: renamed to ``df`` everywhere), and ``system.parallel``'s ``n_jobs``/``n_cores``/
``nworkers`` split for "degree of parallelism" (fixed: standardized on ``n_jobs``). Running this
check surfaced one more real, previously-unnoticed instance: ``pandaslib.dtypes``'s
``get_categorical_columns_indices(ds: pd.DataFrame)`` -- every OTHER function in that module
spells the identical role ``df`` -- now renamed (no caller used the keyword form, confirmed via
grep, so this was a safe, non-breaking rename).

Mechanism: per module, group public top-level functions by the exact string form of their FIRST
parameter's annotation -- restricted to dotted/attribute annotations (``pd.DataFrame``,
``pl.DataFrame``), deliberately excluding bare builtin annotations (``int``, ``str``) which are
far too generic a role to imply "same concept, must share a name" (a function's first ``int``
parameter could mean a row count, a port number, or a retry budget -- nothing in common). If 2+
distinct parameter names occur for the same annotation group, flag.

FP risk (confirmed empirically before finalizing): the SAME concrete type can legitimately play
DIFFERENT ROLES within one module -- ``polarslib.py``'s ``entropy_for_column``/``mi_for_column``
both take ``bins: pl.DataFrame`` (a pre-binned/discretized frame, a distinct concept from "the raw
input frame" that every sibling function in the same file calls ``df``); ``numpylib.py``'s
``smart_ratios(a, b, ...)`` (two ratio operands, not "give me an array" the way
``get_topk_indices(arr)`` is); ``stats.normality``'s public ``normality_verdict(sample, ...)``
deliberately uses a clearer, domain-specific name than its internal ``dagostino_k2(x)``/
``anderson_darling_normal(x)`` helpers' terser statistical convention. Baseline-JSON snapshot
(same idiom as ``test_no_bare_except.py``), not a hand-curated whitelist -- these three are
reviewed and safe, but a NEW group hitting this shape needs the same "is this really one role,
just spelled two ways, or two genuinely different roles that happen to share a type?" review
before being grandfathered in.
"""

from __future__ import annotations

import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

import pytest

import pyutilz

_REPO_ROOT = Path(pyutilz.__file__).resolve().parents[2]
_SRC_DIR = Path(pyutilz.__file__).resolve().parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_param_naming_baseline.json"


def _refresh_requested() -> bool:
    return "--refresh-param-naming-baseline" in sys.argv


def _dotted_annotation(ann: ast.expr | None) -> str | None:
    if ann is None:
        return None
    if isinstance(ann, ast.Attribute):
        parts = []
        cur: ast.expr = ann
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            return ".".join(reversed(parts))
        return None
    if isinstance(ann, ast.Subscript):
        return _dotted_annotation(ann.value)
    return None


def _build_findings() -> set[str]:
    out: set[str] = set()
    for py in _SRC_DIR.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            src = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        rel = py.relative_to(_REPO_ROOT).as_posix()
        groups: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name.startswith("_"):
                continue
            params = list(node.args.posonlyargs) + list(node.args.args)
            if not params:
                continue
            first = params[0]
            if first.arg == "self":
                if len(params) < 2:
                    continue
                first = params[1]
            ann = _dotted_annotation(first.annotation)
            if ann is None:
                continue
            groups[ann].add((node.name, first.arg))
        for ann, items in groups.items():
            param_names = {p for _, p in items}
            if len(param_names) < 2:
                continue
            for funcname, paramname in sorted(items):
                out.add(f"{rel} {ann}::{funcname}({paramname})")
    return out


def test_no_new_param_naming_inconsistency_per_annotated_type():
    current = _build_findings()

    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(json.dumps(sorted(current), indent=2), encoding="utf-8")
        pytest.skip(f"param-naming baseline refreshed at {_BASELINE_PATH.name} ({len(current)} finding(s))")

    baseline = set(json.loads(_BASELINE_PATH.read_text(encoding="utf-8")))
    new = sorted(current - baseline)
    fixed = sorted(baseline - current)

    if fixed:
        sys.stderr.write(
            f"\n[test_no_new_param_naming_inconsistency_per_annotated_type] {len(fixed)} site(s) DRAINED:\n  "
            + "\n  ".join(fixed[:15])
            + (f"\n  ... and {len(fixed) - 15} more" if len(fixed) > 15 else "")
            + "\n  Refresh: pytest ... --refresh-param-naming-baseline\n"
        )

    if new:
        pytest.fail(
            f"{len(new)} function(s) whose first parameter shares a concrete annotated type with "
            f"a sibling in the same module but spells it differently. Either rename to match the "
            f"module's established spelling for that concept (see "
            f"pandaslib.dtypes.get_categorical_columns_indices, renamed ds->df), OR if this "
            f"parameter genuinely plays a DIFFERENT role despite sharing a type (see "
            f"polarslib.py's entropy_for_column/mi_for_column ``bins``), refresh the baseline "
            f"after reviewing:\n  " + "\n  ".join(new[:30]) + (f"\n  ... and {len(new) - 30} more" if len(new) > 30 else "")
        )

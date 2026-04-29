"""E2 — meta-test that the set of public symbols WITHOUT type
annotations doesn't grow.

A function counts as "annotated" iff it has a return-type annotation
AND every non-``self``/``cls`` parameter is annotated. ``*args`` /
``**kwargs`` slots are exempted (they're rarely usefully typed in
library code).

Snapshot-style — same shape as the docstring coverage test (E1).
First run captures the baseline; later runs allow shrinkage but
fail on growth. Refresh:

  pytest tests/test_meta/test_public_annotations.py --refresh-annotation-baseline

Catches contributors who skip annotations on new symbols (most public
APIs in pyutilz are annotated; growth would be an erosion).
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

import pyutilz

PYUTILZ_DIR = Path(pyutilz.__file__).resolve().parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_annotation_baseline.json"

_EXEMPT_FILES = {"version.py", "__init__.py", "__main__.py"}
_EXEMPT_PATH_FRAGMENTS = ("__pycache__", "tests")


def _refresh_requested() -> bool:
    return "--refresh-annotation-baseline" in sys.argv


def _is_fully_annotated(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """A function is fully annotated when:
      * it has a non-None ``returns`` annotation, AND
      * every regular parameter (excluding self/cls/*args/**kwargs) has
        an annotation.

    Classes themselves don't carry signature-level annotations and are
    treated as annotated.
    """
    if fn.returns is None:
        return False
    args = fn.args
    regulars = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    for i, arg in enumerate(regulars):
        # Skip implicit self/cls on the first positional slot of methods.
        # We can't tell here whether ``fn`` is a method without parent
        # context, but ``self`` / ``cls`` rarely have annotations and
        # exempting them everywhere is the right behaviour.
        if i == 0 and arg.arg in ("self", "cls"):
            continue
        if arg.annotation is None:
            return False
    return True


def _is_class_node(node) -> bool:
    return isinstance(node, ast.ClassDef)


def _public_top_level_nodes(path: Path):
    try:
        src = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                yield node


def _build_missing_set() -> set[str]:
    bare: set[str] = set()
    for py in PYUTILZ_DIR.rglob("*.py"):
        if py.name in _EXEMPT_FILES:
            continue
        if any(frag in py.parts for frag in _EXEMPT_PATH_FRAGMENTS):
            continue
        if py.name.endswith(".py.old"):
            continue
        rel = py.relative_to(PYUTILZ_DIR).as_posix()
        for node in _public_top_level_nodes(py):
            if _is_class_node(node):
                continue
            if not _is_fully_annotated(node):
                bare.add(f"{rel}::{node.name}")
    return bare


def test_no_new_unannotated_public_functions():
    current = _build_missing_set()

    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(
            json.dumps(sorted(current), indent=2),
            encoding="utf-8",
        )
        pytest.skip(
            f"annotation baseline refreshed at {_BASELINE_PATH.name} "
            f"({len(current)} unannotated function(s))"
        )

    baseline = set(json.loads(_BASELINE_PATH.read_text(encoding="utf-8")))
    new = sorted(current - baseline)
    fixed = sorted(baseline - current)

    if fixed:
        sys.stderr.write(
            f"\n[test_no_new_unannotated_public_functions] "
            f"{len(fixed)} function(s) DRAINED — now fully annotated:\n  "
            + "\n  ".join(fixed[:15])
            + (f"\n  ... and {len(fixed) - 15} more" if len(fixed) > 15 else "")
            + f"\n  Refresh baseline to lock in: pytest "
            f"tests/test_meta/test_public_annotations.py "
            f"--refresh-annotation-baseline\n"
        )

    if new:
        pytest.fail(
            f"{len(new)} new public function(s) without complete type "
            f"annotations. Add types to params + return, OR refresh the "
            f"baseline if intentional:\n  "
            + "\n  ".join(new[:30])
            + (f"\n  ... and {len(new) - 30} more" if len(new) > 30 else "")
        )

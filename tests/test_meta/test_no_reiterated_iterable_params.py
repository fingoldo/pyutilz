"""Meta-test proposed after the 2026-07-21 audit round 2 meta-test review: a parameter annotated
``Iterable[...]`` (not ``list``/``Sequence``, which guarantee re-iterability) must not be
consumed by more than one ``for``/comprehension in the function body without first being
materialized into a concrete collection.

Class this catches: ``database.db.sqlite.insert_sqllite_data``'s ``columns: Iterable`` parameter
used to be iterated three separate times (validate, build placeholders, build the columns
string) -- a genuine one-shot iterator/generator, which the ``Iterable`` type hint explicitly
invites a caller to pass, was exhausted after the first pass, silently producing an empty
``INSERT INTO t () VALUES ()`` with no exception (caught by a broad ``except Exception`` further
down and reported as "0 rows inserted," indistinguishable from a genuine failure). Fixed with
``columns = list(columns)`` as the function's first line.

Mechanism: for every function parameter whose annotation resolves to (bare or subscripted, real
or string forward-ref) ``Iterable`` -- deliberately NOT ``list``/``Sequence``/``tuple``, which
already guarantee re-iterability -- count how many ``for x in <param>`` statements or
comprehension ``for`` clauses iterate directly over the parameter name. If 2+ such sites exist and
no ``<param> = list(<param>)``-shaped (or ``tuple(...)``) rebinding of the SAME name appears
anywhere in the function body, flag it: a real one-shot iterator passed for that parameter would
silently produce nothing/wrong results on the second+ pass, not raise. Verified clean (zero
findings) against the current codebase before finalizing -- ``insert_sqllite_data``'s fix already
materializes ``columns`` up front, which this check correctly recognizes as a rebind and excludes.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import pyutilz

_REPO_ROOT = Path(pyutilz.__file__).resolve().parents[2]
_SRC_DIR = Path(pyutilz.__file__).resolve().parent


def _is_iterable_annotation(ann: ast.expr | None) -> bool:
    if ann is None:
        return False
    if isinstance(ann, ast.Name):
        return ann.id == "Iterable"
    if isinstance(ann, ast.Attribute):
        return ann.attr == "Iterable"
    if isinstance(ann, ast.Subscript):
        return _is_iterable_annotation(ann.value)
    if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
        return "Iterable" in ann.value and "Sequence" not in ann.value
    return False


def _find_reiterated_params(root: Path) -> list[str]:
    out: list[str] = []
    for py in root.rglob("*.py"):
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
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            params = list(fn.args.posonlyargs) + list(fn.args.args) + list(fn.args.kwonlyargs)
            for p in params:
                if not _is_iterable_annotation(p.annotation):
                    continue
                name = p.arg
                iter_count = 0
                rebound = False
                for stmt in ast.walk(fn):
                    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == name:
                        rebound = True
                        break
                    if isinstance(stmt, ast.For) and isinstance(stmt.iter, ast.Name) and stmt.iter.id == name:
                        iter_count += 1
                    if isinstance(stmt, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                        for gen in stmt.generators:
                            if isinstance(gen.iter, ast.Name) and gen.iter.id == name:
                                iter_count += 1
                if iter_count >= 2 and not rebound:
                    out.append(f"{rel}:{fn.lineno} {fn.name}({name})")
    return out


def test_no_iterable_param_consumed_more_than_once_without_materializing():
    found = _find_reiterated_params(_SRC_DIR)
    if found:
        pytest.fail(
            f"{len(found)} function(s) with a parameter annotated Iterable[...] consumed by 2+ "
            f"for-loops/comprehensions with no materializing rebind. A one-shot iterator/"
            f"generator (which the Iterable hint explicitly invites a caller to pass) is EXHAUSTED "
            f"after the first pass -- the second+ loop silently sees nothing, no exception raised. "
            f"Add `<param> = list(<param>)` as the first line that needs multiple passes (see "
            f"database.db.sqlite.insert_sqllite_data), or narrow the annotation to "
            f"list/Sequence/tuple if a re-iterable input is actually required:\n  " + "\n  ".join(found)
        )

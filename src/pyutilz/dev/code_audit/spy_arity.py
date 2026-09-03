"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _dotted_module_path, _iter_py_files, _line_text, _module_aliases, _read_src_lines, _safe_parse


def _is_test_path(rel: str) -> bool:
    """True if ``rel`` (a posix-style relative path) lives under a ``tests``/``test`` directory
    or matches the ``test_*``/``*_test.py`` filename convention."""
    parts = rel.replace("\\", "/").split("/")
    return any(p == "tests" or p == "test" or p.startswith("test_") or p.endswith("_test.py") for p in parts)


def _positional_arity(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> float:
    """Max number of positional args this function can accept (``inf`` if it has ``*args``)."""
    a = func_node.args
    if a.vararg is not None:
        return float("inf")
    return len(a.posonlyargs) + len(a.args)


def _call_positional_count(call: ast.Call) -> int:
    """Number of plain positional args in a call (a ``*expr`` unpack counts as unbounded, so
    treat it as arity-satisfying and skip -- can't statically know its length)."""
    if any(isinstance(a, ast.Starred) for a in call.args):
        return -1  # sentinel: unknown, caller should skip this call site
    return len(call.args)


def _short_name(target: str) -> str:
    """Last dotted component of a patch-target string, e.g. ``'job_details_shared.build_job_pub_rows'`` -> ``'build_job_pub_rows'``."""
    return target.rsplit(".", 1)[-1]


def _collect_prod_call_max(root: Path, exclude_dirs: frozenset[str]) -> dict[str, int]:
    """Pass 1: production functions -> max positional args seen at any real call site.

    Keyed by the call site's QUALIFIED name (``pkg.mod.build_rows``), not by the bare short name.
    Keying on the short name let an unrelated ``Other.build_rows(self, a, b, c)`` in another class
    raise the maximum for a module-level ``prod.build_rows``, producing exactly the false positive
    this scanner's docstring promises it will not produce. A call whose receiver cannot be resolved
    to a module (any ordinary method call) is not recorded at all -- a false negative, the safe
    failure mode here.
    """
    prod_call_max: dict[str, int] = {}
    for py in _iter_py_files(root, exclude_dirs):
        rel = py.relative_to(root).as_posix()
        if _is_test_path(rel):
            continue
        tree = _safe_parse(py)
        if tree is None:
            continue
        own_module = _dotted_module_path(rel)
        aliases = _module_aliases(tree)
        module_level_defs = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                # A bare call names something defined in (or imported into) THIS module.
                if node.func.id in module_level_defs:
                    qualified = f"{own_module}.{node.func.id}"
                elif node.func.id in aliases:
                    qualified = aliases[node.func.id]
                else:
                    continue
            elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id in aliases:
                qualified = f"{aliases[node.func.value.id]}.{node.func.attr}"
            else:
                continue
            n = _call_positional_count(node)
            if n < 0:
                continue
            if n > prod_call_max.get(qualified, -1):
                prod_call_max[qualified] = n
    return prod_call_max


def _real_max_for(target: str, prod_call_max: dict[str, int]) -> "int | None":
    """The production maximum for a patch target, matched on a dotted-path SUFFIX.

    The scan root is rarely the import root (``src/pkg/mod.py`` is imported as ``pkg.mod``), so an
    exact match would make the scanner a no-op on an ordinary layout.
    """
    hits = [count for key, count in prod_call_max.items() if key == target or key.endswith("." + target)]
    return max(hits) if hits else None


def _is_patch_call(node: ast.Call) -> bool:
    """Matches bare ``patch(...)`` (after ``from unittest.mock import patch``) and
    ``mock.patch(...)``/``unittest.mock.patch(...)`` (attribute form). Deliberately does NOT
    handle ``patch.object(target, "attr", side_effect=...)`` -- its target isn't a string
    literal, so the short-name match in the caller wouldn't resolve to anything anyway."""
    return (isinstance(node.func, ast.Attribute) and node.func.attr == "patch") or (isinstance(node.func, ast.Name) and node.func.id == "patch")


def _find_spy_findings_in_test_file(
    tree: ast.Module,
    rel: str,
    src_lines: list[str],
    prod_call_max: dict[str, int],
) -> list[Finding]:
    """Pass 2: test-local spy/recorder defs referenced as ``patch(..., side_effect=<name>)``,
    flagged when their positional arity has fallen behind ``prod_call_max``."""
    findings: list[Finding] = []
    local_defs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {
        node.name: node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _is_patch_call(node)):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
            continue
        target = node.args[0].value
        side_effect = next((kw.value for kw in node.keywords if kw.arg == "side_effect"), None)
        if not isinstance(side_effect, ast.Name):
            continue
        spy = local_defs.get(side_effect.id)
        if spy is None:
            continue
        short = _short_name(target)
        real_max = _real_max_for(target, prod_call_max)
        if real_max is None:
            continue
        spy_max = _positional_arity(spy)
        if real_max > spy_max:
            findings.append(Finding(
                check="stale_test_spy_arity",
                severity="P1",
                file=rel,
                line=spy.lineno,
                snippet=_line_text(src_lines, spy.lineno),
                detail=(
                    f"spy/recorder '{side_effect.id}' (patched in for '{target}') accepts at most "
                    f"{spy_max if spy_max != float('inf') else 'unbounded'} positional arg(s), but a real "
                    f"call site for '{short}' passes {real_max} -- the spy's arity has fallen behind the "
                    f"production signature it mocks. Add the missing parameter(s) (with a default) to the spy."
                ),
            ))
    return findings


def scan_stale_test_spy_arity(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find test "spy"/"recorder" functions passed as ``patch(..., side_effect=<local def>)``
    whose own positional-arg arity has fallen behind the real production function they mock.

    A production function's signature growing a new parameter (with a default, so real callers
    stay backward compatible) is safe for every caller EXCEPT a test's own hand-written spy that
    mirrors the old signature to record call args -- ``MagicMock`` tolerates any arity, but a
    plain ``def spy(a, b, c): ...`` raises ``TypeError: takes N positional arguments but M were
    given`` the moment production code actually passes the new argument. This is the exact class
    of failure that bit ``build_job_pub_rows`` gaining a ``client_memo`` parameter mid-audit
    (2026-07-21 full-audit gap-pass, production_scrapers): 3 test recorder functions across 2
    files broke because their fixed-arity ``def`` no longer matched the calls production code
    actually made.

    Algorithm: (1) for every production function definition, find the MAX number of positional
    args passed at any real (non-test) call site; (2) for every test-local ``def`` referenced as
    a ``patch(<target ending in the same short name>, side_effect=<that def>)`` value, compute its
    own max acceptable positional-arg count; (3) flag when the real call sites exceed what the
    spy can accept. Best-effort, name-based matching (no cross-module import resolution) -- may
    miss same-named-but-unrelated functions in large codebases; false negatives are the safe
    failure mode here, not false positives.

    Severity: P1 (this test then silently accepts a smaller call signature than production
    actually uses -- if the recorder asserts on captured args, it does so incorrectly or not at
    all for the missing ones; if production is patched with it in a real test run, that test run
    raises TypeError the instant the new positional value is passed, which is confusing to debug
    days after the signature change that caused it).
    """
    prod_call_max = _collect_prod_call_max(root, exclude_dirs)

    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        rel = py.relative_to(root).as_posix()
        if not _is_test_path(rel):
            continue
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = _read_src_lines(py)
        findings.extend(_find_spy_findings_in_test_file(tree, rel, src_lines, prod_call_max))
    return findings

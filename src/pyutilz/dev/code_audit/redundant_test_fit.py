"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _safe_parse, _line_text

# --- redundant test computation (expensive fit re-run across test functions) -----
#
# A common test-suite authoring pattern: a module-level helper builds a
# deterministic synthetic dataset and/or fits a model (``_build_data(seed)``,
# ``_fit_selector(X, y, seed)``), and several ``test_*`` functions each
# independently call the SAME helper with the SAME literal arguments to derive
# a different assertion from the same underlying result. Since the call is
# deterministic (same function, same args -> same return value), every
# occurrence after the first is pure wasted computation -- confirmed in the
# wild as a 7-14x wall-clock cost on real MRMR biz_value test suites (mlframe,
# 2026-07-16) where a single ~15s-90s model fit was independently re-run by
# 3-10 sibling test functions in the same file for identical inputs.
#
# This scanner only looks at underscore-prefixed, module-level helper
# functions (the standard "_build_x / _fit_x local test helper" naming
# convention) so it never flags calls to third-party/stdlib functions where
# repeated identical calls are normal and cheap (``pd.DataFrame(...)``,
# ``np.zeros(...)``). A helper already decorated with ``@cache``/
# ``@lru_cache`` is skipped -- it has already been fixed.


def _is_cached(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True if ``func_node`` carries a ``@cache`` / ``@lru_cache`` decorator (already fixed)."""
    for dec in func_node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        name = target.attr if isinstance(target, ast.Attribute) else target.id if isinstance(target, ast.Name) else None
        if name in ("cache", "lru_cache"):
            return True
    return False


def _enclosing_test_functions(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Every ``def test_*`` / ``async def test_*`` at module or class level (pytest discovery convention)."""
    return [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")]


def _call_signature(call: ast.Call) -> str | None:
    """Normalised ``func_name(args)`` signature for a call to a bare local name, or None if not
    a simple ``_helper(...)`` call (attribute-form calls like ``obj._helper(...)`` are skipped --
    those are method calls, not the module-level helper pattern this scanner targets)."""
    if not isinstance(call.func, ast.Name) or not call.func.id.startswith("_"):
        return None
    _unparse = getattr(ast, "unparse", None)  # ast.unparse needs python>=3.9; degrade to a no-op scan on 3.8
    if _unparse is None:
        return None
    try:
        args_repr = ", ".join(_unparse(a) for a in call.args)
        kwargs_repr = ", ".join(f"{kw.arg}={_unparse(kw.value)}" for kw in call.keywords)
    except (ValueError, RecursionError):
        # ast.unparse raises ValueError on a handful of unsupported/exotic node shapes; a
        # pathologically deep expression could hit RecursionError. Neither is a scanner bug --
        # skip just this one call site rather than crash the whole scan.
        return None
    parts = ", ".join(p for p in (args_repr, kwargs_repr) if p)
    return f"{call.func.id}({parts})"


def scan_redundant_test_fit_calls(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find identical ``_helper(same, literal, args)`` calls made from 2+ DIFFERENT ``test_*``
    functions in the same file -- a strong signal that a deterministic, expensive computation
    (synthetic-data build, model fit) is being redundantly recomputed instead of shared via a
    module-level ``@cache``.

    Only underscore-prefixed, module-level (non-method) helper calls are considered; a helper
    already decorated with ``@cache``/``@lru_cache`` is skipped (already fixed). Files are
    restricted to the pytest test-discovery naming convention (``test_*.py`` / ``*_test.py``) --
    this pattern is meaningless outside test suites.

    Severity: Low (a performance/efficiency signal, not a correctness bug -- the duplicate
    computation is wasteful but produces the SAME deterministic result each time).
    """
    findings: list[Finding] = []
    for py in root.rglob("*.py"):
        if not py.is_file():
            continue
        if any(part in exclude_dirs for part in py.parts):
            continue
        if not (py.name.startswith("test_") or py.name.endswith("_test.py")):
            continue
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()

        cached_names = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_cached(node)}

        # signature -> [(test_qualname, lineno), ...]
        occurrences: dict[str, list[tuple[str, int]]] = {}
        for test_fn in _enclosing_test_functions(tree):
            for node in ast.walk(test_fn):
                if not isinstance(node, ast.Call):
                    continue
                sig = _call_signature(node)
                if sig is None:
                    continue
                func_name = sig.split("(", 1)[0]
                if func_name in cached_names:
                    continue
                occurrences.setdefault(sig, []).append((test_fn.name, node.lineno))

        for sig, hits in occurrences.items():
            distinct_tests = {name for name, _ in hits}
            if len(distinct_tests) < 2:
                continue
            _last_test, last_line = hits[-1]
            findings.append(
                Finding(
                    check="redundant_test_fit_call",
                    severity="Low",
                    file=rel,
                    line=last_line,
                    snippet=_line_text(src_lines, last_line),
                    detail=(
                        f"identical call `{sig}` made from {len(distinct_tests)} different test "
                        f"functions ({', '.join(sorted(distinct_tests))}) -- deterministic, so every "
                        f"call after the first recomputes the same result. Consider a module-level "
                        f"@cache-decorated wrapper shared across the test functions."
                    ),
                )
            )
    return findings

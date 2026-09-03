"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _is_excluded, _line_text, _read_src_lines, _safe_parse

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


# Calls a data factory may make while still being a data factory: constructing or mutating a literal
# container costs nothing, so their presence says nothing about expensiveness either way.
_CHEAP_CALLEES = frozenset(
    {"dict", "list", "set", "tuple", "str", "int", "float", "bool", "len", "range", "sorted", "update", "append", "extend", "copy", "add", "setdefault", "get", "format", "join"}
)


def _is_literal_data_factory(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True if this helper only assembles literal data - no computation an author would want shared.

    The pattern this scanner exists for is an EXPENSIVE deterministic call (a model fit, a synthetic-corpus
    build) re-run by sibling tests. A helper that fills in a dict literal and returns it is the opposite:
    microseconds, and re-running it is the POINT. Its return value is a fresh mutable object that each test
    then edits, so the remedy the finding recommends - `@cache`, or one shared fixture - would hand every
    test the same dict and let one test's mutation reach another. Flagging these does not merely add noise;
    acting on the advice introduces a bug, which is why the exemption belongs in the scanner rather than in
    each caller's baseline.
    """
    builds_a_literal_container = False
    for node in ast.walk(func_node):
        if isinstance(node, ast.Dict) and node.keys:
            builds_a_literal_container = True
        elif isinstance(node, (ast.List, ast.Set, ast.Tuple)) and node.elts:
            builds_a_literal_container = True
        elif isinstance(node, ast.Call):
            callee = node.func
            name = callee.attr if isinstance(callee, ast.Attribute) else callee.id if isinstance(callee, ast.Name) else None
            if name not in _CHEAP_CALLEES:
                return False
    # Constructing the container inline is what separates a data factory from a helper that merely LOOKS
    # cheap in a fixture: `def _build_data(seed): return seed` has no calls either, and it is exactly the
    # stub an expensive builder is reduced to in a test of this scanner. Requiring a literal keeps the
    # exemption to helpers whose whole output is data written out in the source.
    return builds_a_literal_container


def _iteration_bound_names(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Names bound by a `for` statement or a comprehension inside this function.

    A call whose argument is one of these takes a DIFFERENT value on every iteration, so two functions
    that each loop over their own data produce the same unparsed signature - `_resolves(x)` - while sharing
    no call at all. The check is for a deterministic repeat of ONE call, and its own docstring says "same
    literal args"; a loop variable is the opposite of a literal. Requiring literals outright would be too
    strong, because a module-level constant passed by name IS the same value in both callers and that is a
    genuine duplicate; binding-by-iteration is the precise distinction.
    """
    bound: set[str] = set()
    for node in ast.walk(func_node):
        targets = []
        if isinstance(node, (ast.For, ast.AsyncFor)):
            targets = [node.target]
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            targets = [gen.target for gen in node.generators]
        for target in targets:
            bound |= {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}
    return bound


def _enclosing_test_functions(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Every ``def test_*`` / ``async def test_*`` at module or class level (pytest discovery convention)."""
    return [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")]


def _call_signature(call: ast.Call) -> str | None:
    """Normalised ``func_name(args)`` signature for a call to a bare local name, or None if not
    a simple ``_helper(...)`` call (attribute-form calls like ``obj._helper(...)`` are skipped --
    those are method calls, not the module-level helper pattern this scanner targets)."""
    if not isinstance(call.func, ast.Name) or not call.func.id.startswith("_"):
        return None
    # ast.unparse needs python>=3.9. On the supported 3.8 floor fall back to ast.dump, which is a
    # drop-in here: the signature is only ever compared for EQUALITY, and ast.dump() omits
    # positions by default, so two syntactically identical calls still normalise to one key.
    # Returning None there instead made the whole scanner a silent no-op on one CI leg.
    _unparse = getattr(ast, "unparse", None) or (lambda node: ast.dump(node))
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
        if _is_excluded(py, root, exclude_dirs):
            continue
        if not (py.name.startswith("test_") or py.name.endswith("_test.py")):
            continue
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = _read_src_lines(py)
        rel = py.relative_to(root).as_posix()

        cached_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (_is_cached(node) or _is_literal_data_factory(node))
        }

        # signature -> [(test_qualname, lineno), ...]
        occurrences: dict[str, list[tuple[str, int]]] = {}
        for test_fn in _enclosing_test_functions(tree):
            loop_bound = _iteration_bound_names(test_fn)
            for node in ast.walk(test_fn):
                if not isinstance(node, ast.Call):
                    continue
                sig = _call_signature(node)
                if sig is None:
                    continue
                if any(isinstance(n, ast.Name) and n.id in loop_bound for arg in (*node.args, *(kw.value for kw in node.keywords)) for n in ast.walk(arg)):
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

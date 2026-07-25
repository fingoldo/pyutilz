"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

# --- default-via-or trap (wave 14) --------------------------------------


def _is_trivial_default(value: ast.AST) -> bool:
    """A trivial default is one whose substitution cannot corrupt a
    legitimately falsy input:

    - ``None`` / empty ``{}`` / ``[]`` / ``set()`` literals -- nothing
      meaningful to clobber.
    - Any FALSY constant (``0``, ``0.0``, ``""``, ``False``, ``b""``) --
      the trap only bites when the fallback DIFFERS from the value being
      replaced; ``count or 0`` maps a legitimate 0 to ... 0, so the
      result is observably identical whether the trap "fires" or not.
      (Confirmed at scale in a 2026-07 triage: every ``or 0`` / ``or ""``
      finding reviewed was a no-op None-coercion, not a bug.)
    """
    if isinstance(value, ast.Constant) and not value.value:
        return True
    if isinstance(value, (ast.Dict, ast.List, ast.Set)) and not getattr(value, "elts", None) and not getattr(value, "keys", None):
        return True
    return False


def _get_call_key(node: ast.AST) -> str | None:
    """The string key of a ``<obj>.get("key")`` / ``<obj>.get("key", d)``
    call, or None when ``node`` isn't that shape."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        return node.args[0].value
    return None


def _is_alias_key_fallback(lhs: ast.AST, rhs: ast.AST) -> bool:
    """True for ``d.get("notes") or d.get("note")``-style dual-key reads
    where the two keys are obvious ALIASES of one field (one is a
    substring of the other, e.g. ``note``/``notes``,
    ``type``/``prosody_type``). This is a pervasive schema-drift compat
    idiom, not a defaulting trap: an empty value under one alias has no
    meaning distinct from absence, and the intent is precisely "try the
    other spelling". Keys that are NOT substring-related (e.g.
    ``effective_cost_usd`` vs ``actual_cost_usd`` -- two different
    fields with different meanings, a confirmed real bug) stay flagged.
    """
    lhs_key = _get_call_key(lhs)
    rhs_key = _get_call_key(rhs)
    if lhs_key is None or rhs_key is None or lhs_key == rhs_key:
        return False
    a, b = lhs_key.lower(), rhs_key.lower()
    return a in b or b in a


def _is_constructor_call(node: ast.AST) -> bool:
    """True when ``node`` is a call to a CamelCase-named callable
    (``HalvingSchedule()``, ``Path("x")``, ``AsyncAnthropic()``) -- by
    Python convention a class constructor. The matching LHS is then
    almost always an ``X | None`` parameter, and instances without a
    custom ``__bool__``/``__len__`` are always truthy, so only ``None``
    can trigger the fallback. Lowercase callables (``float("inf")``,
    ``compute_default()``) keep their P2 -- their returns CAN be falsy.
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else "")
    return bool(name) and name[0].isupper()


# Documented-safe LHS callables whose ``or DEFAULT`` cannot mask caller intent
# because the callable's only falsy return is ``None`` (or an unknowable
# "no CPU detected" value). Suppress the trap warning when the LHS is one of
# these patterns.
#
# ``os.cpu_count()``:    returns int or None on unknown systems; 0 is not in
#                        the documented return set, so ``or 1`` only catches
#                        the None case.
# ``psutil.cpu_count()``: same contract; documented to return None on systems
#                        that can't determine cpu count.
# ``np.std(...)``, ``np.var(...)``, ``np.var()``, ``arr.std()``: returns 0.0
#                        ONLY when all values equal, in which case treating
#                        the result as "use the fallback denominator" is the
#                        intentional divide-by-zero guard, not a trap.
# ``len(...)``:          returns int (always >= 0); ``len(x) or N`` is the
#                        common "empty-collection fallback" idiom.
_DOCUMENTED_SAFE_LHS_FUNCS = frozenset({
    "cpu_count",  # os / psutil / multiprocessing
    "std", "var", "nanstd", "nanvar",  # numpy stats: 0.0 only when all-equal
    "len",  # always int >= 0; ``or N`` is empty-fallback idiom
})


_TRANSPARENT_WRAPPER_FUNCS = frozenset({
    # Coercions that don't change the "is the value falsy when source was None
    # or 0 / 0.0" semantics: if the inner call returns None, the wrapper
    # raises or returns 0/0.0 - either way the trap analysis is the same as
    # the inner call. We unwrap to check the inner.
    "int", "float", "bool",
    # Reducers that take multiple args; we treat the LAST arg as the
    # candidate (common idiom: `max(1, os.cpu_count() or 1)`).
    "max", "min", "abs",
})


def _unwrap_lhs(lhs: ast.AST) -> ast.AST:
    """Strip transparent wrapper calls (int / float / max / min / abs) and
    return the innermost AST node. Used by ``_lhs_is_documented_safe`` so
    ``float(np.std(arr)) or 1.0`` still recognises ``np.std`` underneath."""
    while isinstance(lhs, ast.Call):
        func = lhs.func
        name = func.id if isinstance(func, ast.Name) else None
        if name in _TRANSPARENT_WRAPPER_FUNCS:
            # For unary wrappers (int/float/bool/abs), the first positional
            # arg is the value. For max/min, the LAST positional arg is
            # commonly the candidate. Try the last positional first; if it
            # isn't a recognised pattern, fall back to the first.
            if name in {"max", "min"} and len(lhs.args) >= 2:
                lhs = lhs.args[-1]
                continue
            if lhs.args:
                lhs = lhs.args[0]
                continue
        break
    return lhs


def _lhs_is_documented_safe(lhs: ast.AST) -> bool:
    """True when ``lhs`` is a call to a callable whose only falsy return is
    None (or an intentional guard). Suppresses default-via-or noise for the
    ~30 `os.cpu_count() or 1` shapes that aren't actually traps. Drills
    through transparent wrappers (``int``, ``float``, ``max``, ``min``,
    ``abs``) so ``int(os.cpu_count() or 1)`` and
    ``max(1, np.std(arr) or 1.0)`` both resolve correctly."""
    lhs = _unwrap_lhs(lhs)
    if not isinstance(lhs, ast.Call):
        return False
    func = lhs.func
    if isinstance(func, ast.Name) and func.id in _DOCUMENTED_SAFE_LHS_FUNCS:
        return True
    if isinstance(func, ast.Attribute) and func.attr in _DOCUMENTED_SAFE_LHS_FUNCS:
        return True
    return False


# --- boolean-context exclusion ------------------------------------------
#
# ``expr1 or expr2`` used as an ``if``/``while``/``assert``/ternary TEST, or
# as a comprehension filter clause, is ordinary logical-OR control flow
# (e.g. ``if not line or line.startswith("#"): continue``) -- it never
# produces a "default value" that could clobber a caller-supplied falsy
# input, because the BoolOp's result is consumed only as a bool, not
# assigned/returned/passed as data. Confirmed via a large-scale manual
# triage (2026-07) across a downstream consumer's codebase: this shape
# accounted for the overwhelming majority of this scanner's findings and
# every single one triaged was a false positive.
_BOOLEAN_CONTEXT_FIELDS: frozenset[tuple[type, str]] = frozenset({
    (ast.If, "test"),
    (ast.While, "test"),
    (ast.Assert, "test"),
    (ast.IfExp, "test"),
    (ast.comprehension, "ifs"),
})


def _build_parent_field_map(tree: ast.AST) -> dict[int, tuple[ast.AST, str]]:
    """Map ``id(child) -> (parent_node, field_name)`` for every AST node,
    so a BoolOp can look up which field of which parent it occupies."""
    parent_map: dict[int, tuple[ast.AST, str]] = {}
    for node in ast.walk(tree):
        for field_name, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                parent_map[id(value)] = (node, field_name)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        parent_map[id(item)] = (node, field_name)
    return parent_map


def _is_in_boolean_context(node: ast.AST, parent_map: dict[int, tuple[ast.AST, str]]) -> bool:
    """Walk up through nested BoolOp/``not`` wrappers (e.g. ``(a or b) and
    c``, ``not (a or b)``) to the first "real" parent field, and check
    whether that field is a pure boolean-test position."""
    current: ast.AST = node
    while True:
        entry = parent_map.get(id(current))
        if entry is None:
            return False
        parent, field = entry
        if any(isinstance(parent, ptype) and field == pfield for ptype, pfield in _BOOLEAN_CONTEXT_FIELDS):
            return True
        if isinstance(parent, ast.BoolOp) or (isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not)):
            current = parent
            continue
        return False


_BOOLEAN_VALUED_CALL_NAMES = frozenset({"isinstance", "issubclass", "hasattr", "callable", "all", "any"})


def _is_boolean_valued(node: ast.AST) -> bool:
    """True when ``node`` can only ever evaluate to an actual ``bool`` (never an arbitrary falsy
    "empty" value like ``0``/``""``/``[]``) -- a comparison, a boolean-returning builtin call, a
    ``not`` unary op, or a BoolOp whose own operands are all themselves boolean-valued. When BOTH
    sides of an ``or`` are shapes like this, the whole expression can never be a "default value"
    trap regardless of where it's used (assigned, returned, passed as an argument) -- there is no
    falsy-but-meaningful non-bool value either side could ever produce. Confirmed as a real
    false-positive class: ``return isinstance(op, ast.IsNot) or isinstance(op, ast.NotEq)`` and
    ``return not (hi_a < lo_b or hi_b < lo_a)`` were both flagged despite being pure boolean logic,
    not a default-substitution shape at all -- the existing boolean-CONTEXT exclusion only covers
    if/while/assert/ternary TEST positions, missing a boolean expression used as a return value or
    assigned to a bool-typed variable."""
    if isinstance(node, ast.Compare):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _BOOLEAN_VALUED_CALL_NAMES:
        return True
    if isinstance(node, ast.BoolOp):
        return all(_is_boolean_valued(v) for v in node.values)
    return False


def scan_default_via_or_trap(root: Path,
                             exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
                             ) -> list[Finding]:
    """Find ``x = arg or DEFAULT`` where ``DEFAULT`` is a non-trivial
    literal/call that would silently clobber falsy-but-valid values
    (``0``, ``""``, ``[]`` non-empty container, etc.) the caller passed.

    Heuristic: AST ``BoolOp(Or, [a, b])`` where ``b`` is an integer
    literal != 0, a non-empty string, or a function call. False positives
    are common for ``label or "default"`` strings -> classified Low.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        parent_map = _build_parent_field_map(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.Or):
                continue
            if len(node.values) != 2:
                continue
            # Skip ordinary logical-OR control flow (if/while/assert/ternary
            # tests, comprehension filters) -- the result is consumed only
            # as a bool, never assigned/returned/passed as a "default value".
            if _is_in_boolean_context(node, parent_map):
                continue
            if _is_boolean_valued(node):
                continue
            rhs = node.values[-1]
            # Skip when RHS is itself "trivial" (None/empty/falsy).
            if _is_trivial_default(rhs):
                continue
            # Skip documented-safe LHS callables (cpu_count, std/var, len).
            lhs = node.values[0]
            if _lhs_is_documented_safe(lhs):
                continue
            # Skip alias-key dual reads: d.get("notes") or d.get("note").
            if _is_alias_key_fallback(lhs, rhs):
                continue
            # Also skip when the LHS is wrapped in a non-mutating expression
            # whose first inner Call is documented-safe (e.g. `int(os.cpu_count() or 1)`,
            # `max(1, os.cpu_count() or 1)`). The BoolOp here is the `or`
            # node, so check whether its left operand is wrapped in such a
            # call-chain back up the tree.
            sev = "Low"
            detail = "default-via-or trap candidate"
            if isinstance(rhs, ast.Constant) and isinstance(rhs.value, int) and rhs.value != 0:
                sev = "P1"
                detail = (
                    f"`or {rhs.value}`: caller passing the legitimate sentinel "
                    f"0 is silently rewritten to {rhs.value}. Use "
                    f"`x if x is not None else {rhs.value}` for None-only "
                    f"defaulting."
                )
            elif isinstance(rhs, ast.Constant) and isinstance(rhs.value, float) and rhs.value != 0.0:
                sev = "P1"
                detail = f"`or {rhs.value}`: caller passing 0.0 is silently rewritten."
            elif isinstance(rhs, ast.Constant) and isinstance(rhs.value, str) and rhs.value:
                sev = "Low"
                detail = f"`or {rhs.value!r}`: caller passing '' is rewritten. Often intentional."
            elif isinstance(rhs, ast.Call) and _is_constructor_call(rhs):
                sev = "Low"
                detail = (
                    "`or ClassName(...)`: constructor default -- LHS is "
                    "almost certainly an `X | None` parameter and instances "
                    "are always truthy, so only None triggers the fallback. "
                    "Verify the class has no custom __bool__/__len__."
                )
            elif isinstance(rhs, ast.Call):
                sev = "P2"
                detail = (
                    "`or <call>(...)`: callable RHS runs the default-compute "
                    "branch when caller passed a legitimate falsy value "
                    "(empty list/df/array). Confirm semantics."
                )
            findings.append(Finding(
                check="default_via_or",
                severity=sev,
                file=rel,
                line=node.lineno,
                snippet=_line_text(src_lines, node.lineno),
                detail=detail,
            ))
    return findings

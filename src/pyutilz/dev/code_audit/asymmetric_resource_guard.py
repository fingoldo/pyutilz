"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

# Context-manager call names that guard a multi-statement/server-side-cursor
# operation (e.g. ``async with conn.transaction():``). Projects with their
# own naming convention pass their own set via ``guard_call_names``.
DEFAULT_GUARD_CALL_NAMES: frozenset[str] = frozenset({"transaction", "atomic", "begin"})

# Bare-name/attribute guard context managers that don't read as a call
# (e.g. ``async with self._lock:``). Matched against the LAST dotted
# component of the context-expression.
DEFAULT_GUARD_ATTR_NAMES: frozenset[str] = frozenset({"lock", "_lock", "mutex", "_mutex"})

# Only an attribute-chain call (``x.y(...)``, never a bare ``f(...)``) whose
# LAST dotted component matches one of these suffixes counts as a
# "guardable operation" candidate. Without this filter, the scanner treats
# every call as a candidate -- including plain builtins (``len(x)``,
# ``sum(x)``) and unrelated attribute calls (``self._log.warning(...)``,
# ``self._path.stat()``) that happen to occur both inside and outside some
# OTHER method's lock/transaction for reasons unconnected to that guard,
# producing overwhelming noise on a first real-codebase run. The default
# set targets the DB-cursor/persistence-write shape the motivating bug
# was; pass a project-specific ``guardable_op_suffixes`` to broaden/narrow.
DEFAULT_GUARDABLE_OP_SUFFIXES: frozenset[str] = frozenset({
    "execute", "executemany", "cursor", "fetch", "fetchrow", "fetchval", "fetchmany", "fetchall",
    "query", "transaction", "commit", "rollback", "write", "delete", "insert", "update",
})


def _call_shape(node: ast.expr) -> str | None:
    """Render a Call's callee as a dotted string (``"conn.cursor"``,
    ``"self._db.execute"``), or None if the callee isn't a plain
    Name/Attribute chain (e.g. a subscript or another call's result)."""
    parts: list[str] = []
    cur: ast.expr = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    else:
        return None
    return ".".join(reversed(parts))


def _is_guard_context_expr(expr: ast.expr, guard_call_names: frozenset[str], guard_attr_names: frozenset[str]) -> bool:
    """True if ``expr`` (a ``with``/``async with`` item's context expression)
    is a guarding call (``.transaction()``) or a guarding bare name/attribute
    (``self._lock``)."""
    if isinstance(expr, ast.Call):
        shape = _call_shape(expr.func)
        return shape is not None and shape.rsplit(".", 1)[-1] in guard_call_names
    shape = _call_shape(expr)
    return shape is not None and shape.rsplit(".", 1)[-1] in guard_attr_names


class _GuardedCallVisitor(ast.NodeVisitor):
    """Walks one method body tracking whether each matched Call sits inside
    an active guarding ``with``/``async with`` block. Records
    ``(shape, guarded)`` for every Call whose callee resolves to a dotted
    name via ``_call_shape`` -- unresolvable callees (e.g. the result of
    another call) are silently skipped, matching this scanner's
    best-effort, no-false-crash design."""

    def __init__(self, guard_call_names: frozenset[str], guard_attr_names: frozenset[str], guardable_op_suffixes: frozenset[str]) -> None:
        self.guard_call_names = guard_call_names
        self.guard_attr_names = guard_attr_names
        self.guardable_op_suffixes = guardable_op_suffixes
        self.guard_depth = 0
        self.hits: list[tuple[str, bool, ast.Call]] = []

    def _visit_with(self, node: ast.With | ast.AsyncWith) -> None:
        """Shared body for ``visit_With``/``visit_AsyncWith``: bump the guard
        depth for the duration of the block if any of its items guard."""
        guards_here = any(self._is_guard_item(item.context_expr) for item in node.items)
        if guards_here:
            self.guard_depth += 1
        self.generic_visit(node)
        if guards_here:
            self.guard_depth -= 1

    def _is_guard_item(self, expr: ast.expr) -> bool:
        """Instance-bound wrapper around ``_is_guard_context_expr`` using
        this visitor's configured guard-name sets."""
        return _is_guard_context_expr(expr, self.guard_call_names, self.guard_attr_names)

    def visit_With(self, node: ast.With) -> None:
        """ast.NodeVisitor dispatch target for a ``with`` block."""
        self._visit_with(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        """ast.NodeVisitor dispatch target for an ``async with`` block."""
        self._visit_with(node)

    def visit_Call(self, node: ast.Call) -> None:
        """ast.NodeVisitor dispatch target for a Call; records a hit for any
        candidate operation-shape call, guarded or not."""
        shape = _call_shape(node.func)
        if shape is not None and "." in shape and shape.rsplit(".", 1)[-1] in self.guardable_op_suffixes:
            self.hits.append((shape, self.guard_depth > 0, node))
        self.generic_visit(node)


def scan_asymmetric_resource_guard(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
    guard_call_names: frozenset[str] = DEFAULT_GUARD_CALL_NAMES,
    guard_attr_names: frozenset[str] = DEFAULT_GUARD_ATTR_NAMES,
    guardable_op_suffixes: frozenset[str] = DEFAULT_GUARDABLE_OP_SUFFIXES,
) -> list[Finding]:
    """Within a single class, flag an operation-shape (a dotted call like
    ``conn.cursor`` or ``self._db.execute``) that's wrapped in a guarding
    ``with``/``async with`` block (``.transaction()``/``.atomic()``/
    ``.begin()``, or a bare ``self._lock``-shaped context manager) in at
    least one method, but performed UNGUARDED in at least one sibling
    method of the SAME class.

    Motivating shape: a class's ``prefetch_resume_cache()`` called
    ``conn.cursor(sql, *params)`` without wrapping it in
    ``async with conn.transaction():``, guaranteeing
    ``asyncpg.exceptions.NoActiveSQLTransactionError`` on every real call --
    while ``query_rows()``, defined 50 lines later in the SAME class,
    performed the IDENTICAL ``conn.cursor()`` call correctly wrapped. A
    class that already demonstrates the correct guard pattern in one
    method but omits it in a sibling is the single strongest signal this
    scanner can act on without understanding what the operation actually
    does -- the class's own code is the spec.

    Only attribute-chain calls (``x.y(...)``, never a bare ``f(...)``)
    whose last dotted component matches ``guardable_op_suffixes``
    (default: DB-cursor/persistence-write shaped names) count as a
    candidate operation -- an unfiltered first pass flagged plain
    builtins (``len(x)``, ``sum(x)``) and unrelated attribute calls
    (``self._log.warning(...)``) that merely happened to occur both
    inside and outside some OTHER method's unrelated guard, at a
    prohibitive false-positive rate.

    Deliberately narrow, real false-positive risk: a genuinely read-only,
    single-statement operation may not need a guard even where a sibling
    multi-statement operation does -- callers adopting this check should
    treat findings as review prompts (a baseline/grandfather harness, not
    a hard gate), same posture as ``shielded_resource_release_race``.

    Severity: P0 (the motivating case is a guaranteed runtime crash on
    every real invocation, not a subtle correctness drift).
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()

        for cls in ast.walk(tree):
            if not isinstance(cls, ast.ClassDef):
                continue
            methods = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            shape_uses: dict[str, list[tuple[str, bool, ast.Call]]] = {}
            for method in methods:
                visitor = _GuardedCallVisitor(guard_call_names, guard_attr_names, guardable_op_suffixes)
                visitor.visit(method)
                for shape, guarded, call_node in visitor.hits:
                    shape_uses.setdefault(shape, []).append((method.name, guarded, call_node))

            for shape, uses in shape_uses.items():
                guarded_methods = sorted({m for m, g, _ in uses if g})
                unguarded = [(m, node) for m, g, node in uses if not g]
                if not (guarded_methods and unguarded):
                    continue
                unguarded_methods = sorted({m for m, _ in unguarded})
                first_unguarded_node = unguarded[0][1]
                findings.append(Finding(
                    check="asymmetric_resource_guard",
                    severity="P0",
                    file=rel,
                    line=first_unguarded_node.lineno,
                    snippet=_line_text(src_lines, first_unguarded_node.lineno),
                    detail=(
                        f"`{cls.name}`: operation `{shape}` is guard-wrapped in {guarded_methods} "
                        f"but NOT in {unguarded_methods} -- verify the unguarded site(s) don't need "
                        "the same protection (a server-side cursor / multi-statement operation "
                        "typically requires an active transaction; a single read-only statement "
                        "usually doesn't)."
                    ),
                ))
    return findings

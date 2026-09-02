"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path

from ._base import _DEFAULT_EXCLUDE_DIRS, Finding, _iter_py_files, _line_text, _safe_parse

# --- a runtime guard whose answer is already fixed at import time ------------------------------
#
# `if MAX_RETRIES > 0:` reads like a runtime decision, and reviewers treat it as one. When
# `MAX_RETRIES` is a module-level literal the branch is decided before the program runs: one arm is
# dead code that no test can reach, and the guard documents a flexibility the module does not have.
#
# The audited codebase had exactly this: a `if SOMETHING_ENABLED:` around a whole recovery path,
# with the constant `False` twelve lines above it. The recovery had never run, and the guard was
# what made that invisible -- the code READ as conditionally enabled.
#
# Scoped hard, because the loose version of this rule is worthless:
#
# * only module-level names bound EXACTLY ONCE, to a literal. A name reassigned anywhere (including
#   under `if TYPE_CHECKING`, in a function via `global`, or by a second module-level assignment)
#   is not a constant and is skipped.
# * only names never rebound by an `import *`-shaped construct or a `globals()` write.
# * a name that some function assigns via `global` is excluded outright -- that IS the runtime
#   switch this rule would otherwise misread.
#
# What survives is the shape that is genuinely decided: a literal constant, read in a guard, that
# nothing in the module can change.

_COMPARISONS = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot)


def _all_bindings(tree: ast.Module) -> tuple[dict[str, int], set[str]]:
    """(how many times each name is bound ANYWHERE, every function-parameter name).

    Depth matters, and getting it wrong is what made the first version of this rule report
    thirteen hits with not one real among them. All three failure shapes were bindings the
    module-level statement list does not contain:

    * `spacy = None` at module level, reassigned to the real module inside a `try:` -- the
      canonical optional-import probe, and the assignment that matters is nested.
    * `undetectable = False` at module level, shadowed by a function PARAMETER of the same name
      that the reported guard actually reads.
    * `_warned = False` at module level, set inside a function under `global`.
    """
    counts: dict[str, int] = {}
    params: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            counts[node.id] = counts.get(node.id, 0) + 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Counted HERE, not in a later branch for all three definition kinds: this `elif`
            # already claims both function nodes, so that branch only ever saw a ClassDef and a
            # name bound by both `X = False` and `def X()` read as a constant.
            counts[node.name] = counts.get(node.name, 0) + 1
            args = node.args
            for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]:
                if arg is not None:
                    params.add(arg.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound = (alias.asname or alias.name).split(".")[-1]
                counts[bound] = counts.get(bound, 0) + 1
        elif isinstance(node, ast.ClassDef):
            counts[node.name] = counts.get(node.name, 0) + 1
    return counts, params


def _externally_written_names(trees: dict[str, ast.Module]) -> set[str]:
    """Names any file in the tree writes from OUTSIDE the module that defines them.

    A module-level name in Python is never truly fixed: another file can rebind it by attribute
    (`kernels._RADIX_THREADS_OVERRIDE = 256`), through the module's `globals()` dict
    (`g.setdefault("_CUDA_AVAILABLE", ...)`), or with setattr/monkeypatch. None of those are
    visible to a walk of the defining module, which is why this rule has to be package-wide --
    scoped to one module it reported four constants that a sibling file sets on every run.
    """
    written: set[str] = set()
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
                written.add(node.attr)
            elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
                index = node.slice
                if isinstance(index, ast.Constant) and isinstance(index.value, str):
                    written.add(index.value)
            elif isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name in {"setattr", "delattr", "setdefault", "update", "monkeypatch"}:
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            written.add(arg.value)
                for kw in node.keywords:
                    if kw.arg:
                        written.add(kw.arg)
    return written


def _module_constants(tree: ast.Module) -> dict[str, object]:
    """Module-level names bound exactly once, to a literal, and shadowed by nothing."""
    counts, params = _all_bindings(tree)
    constants: dict[str, object] = {}
    for stmt in tree.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(stmt, ast.Assign):
            targets, value = list(stmt.targets), stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            targets, value = [stmt.target], stmt.value
        for target in targets:
            if not isinstance(target, ast.Name) or not isinstance(value, ast.Constant):
                continue
            # Underscore-private only. A PUBLIC module-level name is a configuration knob any
            # importer can set -- `browser.undetectable = True` before calling `start_selenium()`
            # is exactly how this package's own selenium module is driven -- and an attribute
            # write from another file is invisible to a single-module AST walk. Reported
            # unrestricted, that one pattern supplied eight of this rule's first eight hits.
            if not target.id.startswith("_"):
                continue
            if counts.get(target.id, 0) != 1 or target.id in params:
                continue
            constants[target.id] = value.value

    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            for name in node.names:
                constants.pop(name, None)
    return constants


def _decided_by(test: ast.expr, constants: dict[str, object]) -> str | None:
    """The constant name that already decides this condition, if one does."""
    if isinstance(test, ast.Name) and test.id in constants:
        return test.id
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return _decided_by(test.operand, constants)
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], _COMPARISONS):
        left, right = test.left, test.comparators[0]
        # One side a known constant, the other a literal: nothing about this can vary.
        for a, b in ((left, right), (right, left)):
            if isinstance(a, ast.Name) and a.id in constants and isinstance(b, ast.Constant):
                return a.id
    return None


def scan_guard_decidable_from_constants(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find an `if` whose answer is fixed at import time by a module-level literal.

    The guard reads as a runtime decision and reviewers treat it as one, but one arm is dead code
    no test can reach. That is how a recovery path guarded by a constant `False` stayed invisible:
    the code read as conditionally enabled, and had never run once.

    Only names bound exactly once at module level, to a literal, and rebindable by nothing -- not
    by a `global`, not by an import, not by a second assignment -- are treated as constants.
    """
    findings: list[Finding] = []
    trees: dict[str, ast.Module] = {}
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is not None:
            trees[py.relative_to(root).as_posix()] = tree
    externally_written = _externally_written_names(trees)

    for py in _iter_py_files(root, exclude_dirs):
        rel = py.relative_to(root).as_posix()
        tree = trees.get(rel)
        if tree is None:
            continue
        constants = {n: v for n, v in _module_constants(tree).items() if n not in externally_written}
        if not constants:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()

        for node in ast.walk(tree):
            if not isinstance(node, (ast.If, ast.IfExp)):
                continue
            name = _decided_by(node.test, constants)
            if name is None:
                continue
            # No exemption for `if TYPE_CHECKING:` or `__debug__`, which an earlier draft carried:
            # the private-name restriction above already makes both unreachable here, and a guard
            # that cannot fire reads as protection this rule does not actually have.
            findings.append(
                Finding(
                    check="guard_decidable_from_constants",
                    severity="P2",
                    file=rel,
                    line=node.lineno,
                    snippet=_line_text(src_lines, node.lineno),
                    detail=(
                        f"this condition is already decided at import time: `{name}` is bound once "
                        f"in this module to the literal {constants[name]!r} and nothing can rebind "
                        "it. One arm is dead code no test can reach, while the guard reads as a "
                        "runtime decision. Remove the branch, or make the constant genuinely "
                        "configurable."
                    ),
                )
            )
    return findings

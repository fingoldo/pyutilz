"""AST-based code-audit scanners for recurring Python production-bug classes.

Library + CLI for the kinds of static checks we kept rebuilding from
throwaway scripts inside subagent runs. Promoted into pyutilz so they
stop living in D:/Temp and can run against any project (mlframe,
pyutilz itself, downstream applications) with a single import or
``python -m pyutilz.dev.code_audit ./src``.

Implemented checks (each a top-level scanner function returning a
list[Finding]):

- ``scan_mutable_defaults``: ``def f(x=[])`` / ``=={}`` / ``=set()`` /
  ``=dict()`` / ``=list()``. Flagged at ``severity="P0"`` only when the
  parameter is observably mutated in the body (``append``, ``extend``,
  ``setitem``, ``add``, ``update``, ``pop``, ``clear``); otherwise
  ``Low`` (idiomatic-but-questionable). Catches the classic
  "options dict default accumulates state across calls" bug.

- ``scan_late_binding_closures``: ``lambda`` / nested ``def`` inside a
  ``for`` loop that references the loop variable AND escapes the
  iteration (stored in a list/dict, passed as callback, registered).
  Classic "all closures see the FINAL loop value" bug.

- ``scan_default_via_or_trap``: ``x = arg or DEFAULT`` shape where the
  RHS default is a non-trivial literal that would silently clobber
  ``0`` / ``""`` / ``[]`` / ``{}`` if the caller passed one of those
  as a legitimate sentinel. Detected via AST ``BoolOp(Or, ...)`` with
  a constant/call/name on the right. Heuristic; not 100% recall.

- ``scan_broad_except_swallows``: ``except Exception:`` / ``except:``
  followed by a body of ``pass`` / ``continue`` / a bare ``return``
  WITHOUT a ``logger.warning``/``logger.error`` call. Catches the
  "silent data drop" pattern (a column failed to encode, the row got
  skipped, the operator sees fewer features and no log line).

Each scanner is a pure function: ``(root_path: Path) -> list[Finding]``.
The CLI ``__main__`` block wraps them with argparse and emits markdown
or JSON.

Severity tags follow the rest of the audit suite:
    P0 - silent data corruption / cross-call leak at top-level boundary
    P1 - silent leak within a suite / session / per-target loop
    P2 - degraded diagnostics / metrics-reporting bias
    Low - cosmetic / idiomatic-but-questionable

Per ``feedback_no_padding_parametric_pins``: scanners never manufacture
findings. If the codebase is clean, the result list is empty.

Per ``feedback_save_useful_scripts_in_package``: this file lives here
(in the package) precisely so subagent runs stop dropping ad-hoc AST
scanners in D:/Temp. The CLI exposes the same checks the agents would
otherwise hand-write.
"""
from __future__ import annotations

import ast
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional


# --- public types --------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """One audit hit.

    ``check`` is the scanner name (``"mutable_default"``,
    ``"late_binding_closure"``, ``"default_via_or"``,
    ``"broad_except_swallow"``).

    ``severity`` follows the project audit convention (P0/P1/P2/Low).

    ``file`` is relative to the audit root; ``line`` is the 1-based
    line number of the offending node. ``snippet`` is the source line
    text (one line, stripped) for context. ``detail`` is a short
    free-text reason.
    """

    check: str
    severity: str
    file: str
    line: int
    snippet: str
    detail: str

    def as_md_row(self) -> str:
        snip = self.snippet.replace("|", "\\|")
        return (
            f"| {self.severity} | {self.check} | {self.file}:{self.line} | "
            f"`{snip}` | {self.detail} |"
        )


# --- helpers -------------------------------------------------------------


_PY_EXTS = (".py",)
_DEFAULT_EXCLUDE_DIRS = frozenset({
    "__pycache__", ".git", ".venv", "venv", "env", ".env", ".tox",
    "build", "dist", "node_modules", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".idea", ".vscode", "htmlcov",
})


def _iter_py_files(root: Path, exclude_dirs: frozenset[str]) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.suffix not in _PY_EXTS or not p.is_file():
            continue
        # Skip if any parent name matches an excluded dir.
        if any(part in exclude_dirs for part in p.parts):
            continue
        yield p


def _safe_parse(path: Path) -> Optional[ast.Module]:
    try:
        src = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        return ast.parse(src, filename=str(path))
    except SyntaxError:
        return None


def _line_text(src_lines: list[str], lineno: int) -> str:
    if 1 <= lineno <= len(src_lines):
        return src_lines[lineno - 1].strip()
    return ""


def _arg_names(func_node: ast.AST) -> list[str]:
    """Extract every parameter name from a FunctionDef / AsyncFunctionDef."""
    args: list[str] = []
    if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return args
    a = func_node.args
    args.extend(arg.arg for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs))
    if a.vararg:
        args.append(a.vararg.arg)
    if a.kwarg:
        args.append(a.kwarg.arg)
    return args


# --- Class A: mutable defaults ------------------------------------------


_MUTATING_METHODS = frozenset({
    "append", "extend", "insert", "pop", "clear", "remove", "sort",
    "reverse", "setdefault", "update", "add", "discard", "popitem",
    "__setitem__", "__delitem__", "__iadd__", "__ior__",
})

_MUTABLE_LITERAL_TYPES = (ast.List, ast.Dict, ast.Set)
_MUTABLE_CALLS = frozenset({"list", "dict", "set"})


def _is_mutable_default(default: ast.AST) -> Optional[str]:
    """Return a short label if ``default`` is a mutable literal/call,
    else None. ``label`` is one of ``"list"``, ``"dict"``, ``"set"``."""
    if isinstance(default, ast.List):
        return "list"
    if isinstance(default, ast.Dict):
        return "dict"
    if isinstance(default, ast.Set):
        return "set"
    if isinstance(default, ast.Call) and isinstance(default.func, ast.Name):
        if default.func.id in _MUTABLE_CALLS and not default.args and not default.keywords:
            return default.func.id
    return None


def _param_is_mutated(func: ast.FunctionDef | ast.AsyncFunctionDef,
                     param_name: str) -> bool:
    """Heuristic: does the function body apply a mutating operation to
    the named parameter? Walks calls of shape ``param.append(...)``,
    subscript-assigns ``param[k] = v``, augmented assigns ``param += x``."""
    for node in ast.walk(func):
        # param.<mutating_method>(...)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func
            if (
                isinstance(attr.value, ast.Name)
                and attr.value.id == param_name
                and attr.attr in _MUTATING_METHODS
            ):
                return True
        # param[k] = v  /  del param[k]
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.Delete)):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
            for t in targets:
                if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == param_name:
                    return True
                if isinstance(t, ast.Name) and t.id == param_name and isinstance(node, ast.AugAssign):
                    return True
    return False


def scan_mutable_defaults(root: Path,
                          exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
                          ) -> list[Finding]:
    """Find ``def f(x=[]) / ={} / =set() / =list() / =dict() / =set()`` defaults.

    Severity rule: parameter mutated in body -> P0 (state leaks across
    callers); not mutated -> Low (idiomatic-but-questionable, no
    observable bug today but reader trap and future-bug bait).
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = node.args
            # Pair every positional default with its corresponding name.
            pos_args = (*args.posonlyargs, *args.args)
            n_defaults = len(args.defaults)
            if n_defaults:
                paired = list(zip(pos_args[-n_defaults:], args.defaults))
            else:
                paired = []
            paired.extend(
                (k, d) for k, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None
            )
            for arg_node, default in paired:
                label = _is_mutable_default(default)
                if label is None:
                    continue
                mutated = _param_is_mutated(node, arg_node.arg)
                sev = "P0" if mutated else "Low"
                detail = (
                    f"def {node.name}(..., {arg_node.arg}={label}()): "
                    + ("MUTATED in body -> shared state leaks across callers"
                       if mutated else
                       "default is mutable but never mutated in body (Low)")
                )
                findings.append(Finding(
                    check="mutable_default",
                    severity=sev,
                    file=rel,
                    line=node.lineno,
                    snippet=_line_text(src_lines, node.lineno),
                    detail=detail,
                ))
    return findings


# --- Class B: late-binding closures over loop vars ----------------------


def _names_referenced(tree: ast.AST) -> set[str]:
    """Set of ``Name(id=...)`` strings loaded inside ``tree`` (not stored)."""
    used: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            used.add(n.id)
    return used


def _loop_target_names(node: ast.For | ast.AsyncFor) -> set[str]:
    """Names bound by a for-loop target (handles tuple unpacking)."""
    names: set[str] = set()
    def _walk(t: ast.AST) -> None:
        if isinstance(t, ast.Name):
            names.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for el in t.elts:
                _walk(el)
        elif isinstance(t, ast.Starred):
            _walk(t.value)
    _walk(node.target)
    return names


_SYNCHRONOUS_CONSUMERS = frozenset({
    # builtins that exhaust their callable arg before returning.
    "sorted", "min", "max", "filter", "map", "any", "all", "sum", "next",
    "reduce", "groupby", "takewhile", "dropwhile",
})

_STORING_METHODS = frozenset({
    "append", "extend", "insert", "setdefault", "__setitem__",
    "register", "add_callback", "schedule", "submit", "apply_async",
})


def _ancestor_chain(target: ast.AST, root: ast.stmt) -> list[ast.AST]:
    """Return ancestors of ``target`` inside ``root``, root-first.

    Empty list if ``target`` is not within ``root``.
    """
    chain: list[ast.AST] = []
    found = [False]

    def _walk(node: ast.AST, stack: list[ast.AST]) -> None:
        if found[0]:
            return
        if node is target:
            chain.extend(stack)
            found[0] = True
            return
        for child in ast.iter_child_nodes(node):
            _walk(child, stack + [node])

    _walk(root, [])
    return chain


def _closure_escapes_iteration(closure_node: ast.AST,
                               loop_body: list[ast.stmt]) -> bool:
    """Conservative check: does ``closure_node`` look like it escapes
    this iteration of the loop body?

    Walk up the ancestor chain from ``closure_node`` toward the
    enclosing statement. The closure escapes iff the FIRST enclosing
    Call passes it to a known storage method (``append``, ``register``,
    ``functools.partial``, etc.), OR there is no enclosing Call at all
    and the enclosing statement is an Assign / Return / Yield.

    Importantly: if the FIRST enclosing Call is a synchronous consumer
    (``sorted``, ``min``, ``max``, ``filter``, ``map``, ``functools.reduce``,
    or any callable passed as ``key=``), the closure is exhausted inside
    that call before iteration ends -> NOT an escape, regardless of how
    the call result is stored downstream.
    """
    for stmt in loop_body:
        chain = _ancestor_chain(closure_node, stmt)
        if not chain:
            continue
        # Find the closest enclosing Call (walk up from leaf).
        enclosing_call: Optional[ast.Call] = None
        kw_name: Optional[str] = None
        for idx in range(len(chain) - 1, -1, -1):
            node = chain[idx]
            if isinstance(node, ast.Call):
                enclosing_call = node
                # If our previous-step parent was a keyword, capture its name.
                if idx + 1 < len(chain):
                    parent_of_child = chain[idx + 1]
                    if isinstance(parent_of_child, ast.keyword):
                        kw_name = parent_of_child.arg
                break
        if enclosing_call is not None:
            # Synchronous reducer with the closure as a key=... or first arg?
            func = enclosing_call.func
            func_name = None
            if isinstance(func, ast.Name):
                func_name = func.id
            elif isinstance(func, ast.Attribute):
                func_name = func.attr
            if kw_name == "key":
                return False  # ``key=lambda ...`` is exhausted by the call
            if func_name in _SYNCHRONOUS_CONSUMERS:
                return False
            if func_name in _STORING_METHODS:
                return True
            # functools.partial(lambda, ...) stores the closure.
            if func_name == "partial":
                return True
            # Default: a Call wrapping the lambda whose semantics we
            # don't know. Conservatively NOT an escape (avoid noise).
            return False
        # No enclosing Call: look at the enclosing statement type.
        if isinstance(stmt, ast.Assign):
            return True
        if isinstance(stmt, (ast.Return, ast.Yield, ast.YieldFrom)):
            return True
        if isinstance(stmt, ast.Expr):
            return False
        return False
    return False


def scan_late_binding_closures(root: Path,
                               exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
                               ) -> list[Finding]:
    """Find ``lambda`` / nested ``def`` inside a ``for`` loop that
    references the loop variable AND escapes the iteration.

    Defaults shielding (``lambda x=x: ...``) is detected and excluded:
    if the closure has the loop var bound as its own positional
    default, the late-binding hazard is neutralized.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.For, ast.AsyncFor)):
                continue
            loop_vars = _loop_target_names(node)
            if not loop_vars:
                continue
            for sub in ast.walk(node):
                if sub is node:
                    continue
                if isinstance(sub, ast.Lambda):
                    own_defaults = {a.arg for a in (*sub.args.posonlyargs, *sub.args.args, *sub.args.kwonlyargs)}
                    referenced = _names_referenced(sub.body)
                    captured = (referenced & loop_vars) - own_defaults
                    if not captured:
                        continue
                    if not _closure_escapes_iteration(sub, node.body):
                        continue
                    findings.append(Finding(
                        check="late_binding_closure",
                        severity="P1",
                        file=rel,
                        line=sub.lineno,
                        snippet=_line_text(src_lines, sub.lineno),
                        detail=(
                            f"lambda inside for-loop captures loop var(s) "
                            f"{sorted(captured)!r}; closure escapes iteration "
                            f"(stored/yielded). Bind as default: "
                            f"`lambda x=x: ...`."
                        ),
                    ))
                elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    own_args = set(_arg_names(sub))
                    referenced = set()
                    for inner in sub.body:
                        referenced |= _names_referenced(inner)
                    captured = (referenced & loop_vars) - own_args
                    if not captured:
                        continue
                    # Heuristic for escape: the nested def is decorated or assigned.
                    if not (sub.decorator_list or _closure_escapes_iteration(sub, node.body)):
                        continue
                    findings.append(Finding(
                        check="late_binding_closure",
                        severity="P1",
                        file=rel,
                        line=sub.lineno,
                        snippet=_line_text(src_lines, sub.lineno),
                        detail=(
                            f"nested def {sub.name!r} inside for-loop "
                            f"captures loop var(s) {sorted(captured)!r}; "
                            f"closure escapes iteration."
                        ),
                    ))
    return findings


# --- default-via-or trap (wave 14) --------------------------------------


def _is_trivial_default(value: ast.AST) -> bool:
    """A trivial default is ``None`` / ``True`` / ``False`` / ``{}`` /
    ``[]`` -- callers can't pass a falsy variant that would conflict
    with the intent of these defaults, so ``or DEFAULT`` is safe."""
    if isinstance(value, ast.Constant) and value.value is None:
        return True
    if isinstance(value, (ast.Dict, ast.List, ast.Set)) and not getattr(value, "elts", None) and not getattr(value, "keys", None):
        return True
    return False


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
        for node in ast.walk(tree):
            if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.Or):
                continue
            if len(node.values) != 2:
                continue
            rhs = node.values[-1]
            # Skip when RHS is itself "trivial" (None/empty/falsy).
            if _is_trivial_default(rhs):
                continue
            # Skip documented-safe LHS callables (cpu_count, std/var, len).
            lhs = node.values[0]
            if _lhs_is_documented_safe(lhs):
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


# --- broad-except swallow (wave 16) -------------------------------------


_LOG_METHODS = frozenset({"warning", "error", "critical", "exception", "warn"})


def _has_log_call(stmts: list[ast.stmt]) -> bool:
    """True if ``stmts`` contains a logger.warning/error/critical/exception call."""
    for s in stmts:
        for n in ast.walk(s):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                if n.func.attr in _LOG_METHODS:
                    return True
    return False


def _has_raise(stmts: list[ast.stmt]) -> bool:
    for s in stmts:
        for n in ast.walk(s):
            if isinstance(n, ast.Raise):
                return True
    return False


def _is_silent_swallow(handler: ast.ExceptHandler) -> bool:
    """True if the handler body is silent: only pass/continue/return/break
    AND has no log call AND has no re-raise."""
    if _has_log_call(handler.body):
        return False
    if _has_raise(handler.body):
        return False
    # If body contains a meaningful side-effect (assign, function call other
    # than logger), it's NOT a silent swallow.
    for s in handler.body:
        if isinstance(s, (ast.Pass, ast.Continue, ast.Break)):
            continue
        if isinstance(s, ast.Return):
            continue
        # Augment: bare ``logger.debug(...)`` should also classify as silent
        # at WARN level (debug logs are off by default).
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call):
            call = s.value
            if isinstance(call.func, ast.Attribute) and call.func.attr == "debug":
                continue
            return False  # other side-effecting call -> not silent
        return False
    return True


def _is_broad_except(handler: ast.ExceptHandler) -> bool:
    """True if the except clause catches Exception or is bare."""
    if handler.type is None:
        return True
    if isinstance(handler.type, ast.Name) and handler.type.id in {"Exception", "BaseException"}:
        return True
    if isinstance(handler.type, ast.Tuple):
        return any(
            isinstance(el, ast.Name) and el.id in {"Exception", "BaseException"}
            for el in handler.type.elts
        )
    return False


# Best-effort filesystem / process operations whose failure is legitimately
# swallowed in production (the file may not exist, the dir may already be
# cleaned up, the process may have already exited). Suppress the broad-except
# warning when the try body is a single call to one of these.
_BEST_EFFORT_OPS = frozenset({
    # filesystem
    "chmod", "unlink", "remove", "rmdir", "makedirs", "mkdir", "rmtree",
    "close", "flush",
    # process
    "kill", "terminate", "wait",
    # logging / metrics teardown
    "stop", "shutdown", "join",
})


def _try_body_is_imports_only(try_body: list[ast.stmt]) -> bool:
    """True if every statement in the try body is an Import / ImportFrom
    (optional-dep guard pattern). These broad-except blocks are legitimate
    -- the swallow is the entire point of the import guard."""
    if not try_body:
        return False
    for s in try_body:
        if not isinstance(s, (ast.Import, ast.ImportFrom)):
            return False
    return True


def _try_body_is_best_effort_op(try_body: list[ast.stmt]) -> bool:
    """True if the try body is a single call (or attribute assignment) to a
    documented best-effort op (``os.chmod``, ``proc.kill``, ``file.close``,
    etc.). These swallows are intentional and don't degrade observability."""
    if len(try_body) != 1:
        return False
    s = try_body[0]
    # ``os.chmod(path, mode)`` etc.
    if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call):
        call = s.value
        if isinstance(call.func, ast.Attribute) and call.func.attr in _BEST_EFFORT_OPS:
            return True
        if isinstance(call.func, ast.Name) and call.func.id in _BEST_EFFORT_OPS:
            return True
    return False


def scan_broad_except_swallows(root: Path,
                               exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
                               ) -> list[Finding]:
    """Find broad ``except Exception:`` / ``except:`` handlers whose body
    silently passes/continues/returns without any logger.warning/error
    call or re-raise. These are the classic silent-data-drop sites.

    Severity P1 by default; downgrade to Low for import-guard shapes
    detected via the ``ImportError`` / ``ModuleNotFoundError`` exception
    type (those should already short-circuit via the broad-except check,
    but defensive narrowing helps).
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            # Suppress whole Try if the body is one of the documented-safe
            # shapes (import guard / single best-effort op). The handler
            # body's silent swallow is intentional in these cases.
            if _try_body_is_imports_only(node.body):
                continue
            if _try_body_is_best_effort_op(node.body):
                continue
            for handler in node.handlers:
                if not _is_broad_except(handler):
                    continue
                if not _is_silent_swallow(handler):
                    continue
                kind = "bare except" if handler.type is None else "except Exception"
                findings.append(Finding(
                    check="broad_except_swallow",
                    severity="P1",
                    file=rel,
                    line=handler.lineno,
                    snippet=_line_text(src_lines, handler.lineno),
                    detail=(
                        f"{kind} with silent body (pass/continue/return, "
                        f"no logger.warning/error, no re-raise). "
                        f"Operators get no signal when this fires."
                    ),
                ))
    return findings


# --- nan-equality trap --------------------------------------------------


def _is_nan_constant(node: ast.AST) -> bool:
    """True if ``node`` is a literal NaN constant in any of the common forms.

    Catches:
    - ``float("nan")`` / ``float('NaN')``
    - ``np.nan`` / ``numpy.nan`` / ``math.nan``
    - ``float("inf")`` is NOT NaN; not flagged.
    """
    # ``float("nan")`` form
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "float"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
        and node.args[0].value.strip().lower() == "nan"
    ):
        return True
    # ``np.nan`` / ``numpy.nan`` / ``math.nan`` form
    if isinstance(node, ast.Attribute) and node.attr == "nan":
        return True
    # ``math.inf`` / ``np.inf`` are NOT NaN; let those through.
    return False


def scan_nan_equality(root: Path,
                     exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
                     ) -> list[Finding]:
    """Find ``x == NaN`` / ``x != NaN`` shapes.

    NaN doesn't equal anything in IEEE 754 (not even itself), so these
    comparisons ALWAYS yield False. The correct check is
    ``math.isnan(x)`` / ``np.isnan(x)`` / ``pd.isna(x)``.

    Severity: P0. The comparison evaluates False unconditionally, so any
    code-path guarded on detecting NaN never fires.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            # Walk every (op, comparator) pair plus the LHS to cover the
            # 'x == NaN' AND 'NaN == x' AND 'x != NaN' shapes.
            operands = [node.left] + list(node.comparators)
            ops = node.ops
            for i, op in enumerate(ops):
                if not isinstance(op, (ast.Eq, ast.NotEq)):
                    continue
                lhs = operands[i]
                rhs = operands[i + 1]
                nan_side = None
                if _is_nan_constant(lhs):
                    nan_side = "left"
                elif _is_nan_constant(rhs):
                    nan_side = "right"
                if nan_side is None:
                    continue
                findings.append(Finding(
                    check="nan_equality",
                    severity="P0",
                    file=rel,
                    line=node.lineno,
                    snippet=_line_text(src_lines, node.lineno),
                    detail=(
                        f"`{type(op).__name__}` comparison against a NaN "
                        f"literal on the {nan_side}. NaN does not equal "
                        f"anything in IEEE 754 -> the result is always "
                        f"False (or True for NotEq). Use `math.isnan(x)` "
                        f"/ `np.isnan(x)` / `pd.isna(x)` instead."
                    ),
                ))
    return findings


# --- mutation-during-iteration ------------------------------------------


_MUTATING_DICT_METHODS = frozenset({
    "pop", "popitem", "clear", "update", "setdefault",
})
_MUTATING_LIST_METHODS = frozenset({
    "append", "extend", "insert", "pop", "remove", "clear",
})
_MUTATING_SET_METHODS = frozenset({
    "add", "discard", "remove", "pop", "clear", "update",
    "difference_update", "intersection_update", "symmetric_difference_update",
})


def _iter_target_chain(node: ast.AST) -> Optional[str]:
    """Return a dotted-name chain for the iterated collection, or None
    for complex expressions (call results, comprehensions, etc.) that
    we can't statically associate with a mutation site.

    Examples:
        ``self._cache`` -> ``"self._cache"``
        ``d`` -> ``"d"``
        ``d.copy()`` -> None (a fresh object; mutation on `d` doesn't affect it)
        ``list(d)`` -> None (same; explicit defensive copy)
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _iter_target_chain(node.value)
        if base is None:
            return None
        return f"{base}.{node.attr}"
    # Calls / subscripts: only ``.items() / .keys() / .values()`` calls on a
    # known chain are treated as live views; everything else is opaque.
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr in {"items", "keys", "values"}:
            return _iter_target_chain(node.func.value)
    return None


def scan_mutation_during_iteration(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find ``for x in d: ... d[k] = v / del d[k] / d.append(...)`` shapes.

    Catches dict / list / set mutation while a live view (``d.keys()``,
    ``d.items()``, ``d.values()``, or the collection directly) is being
    iterated. Defensive-copy patterns (``for k in list(d):``,
    ``d.copy().items()``) are correctly NOT flagged because the iterated
    chain resolves to None.

    Severity: P0. Dict / set mutation raises RuntimeError loudly at the
    next iteration; list mutation silently skips elements (correctness
    bug). The audit from wave 32 (2026-05-20) confirmed this is a
    high-leverage scanner -- mlframe came back clean but the scanner
    pins the discipline for future contributors.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.For, ast.AsyncFor)):
                continue
            iter_chain = _iter_target_chain(node.iter)
            if iter_chain is None:
                continue  # Iterated chain wasn't a live view we can track.
            for child in ast.walk(node):
                if child is node:
                    continue
                # del d[...]
                if isinstance(child, ast.Delete):
                    for tgt in child.targets:
                        if isinstance(tgt, ast.Subscript):
                            sub_chain = _iter_target_chain(tgt.value)
                            if sub_chain == iter_chain:
                                findings.append(Finding(
                                    check="mutation_during_iteration",
                                    severity="P0",
                                    file=rel,
                                    line=child.lineno,
                                    snippet=_line_text(src_lines, child.lineno),
                                    detail=(
                                        f"``del {iter_chain}[...]`` while iterating "
                                        f"the same collection at line {node.lineno}. "
                                        f"Dict / set mutation raises RuntimeError; "
                                        f"list mutation silently skips elements. "
                                        f"Iterate over ``list({iter_chain})`` or "
                                        f"``{iter_chain}.copy()`` instead."
                                    ),
                                ))
                # d[k] = v  -- only when adding a new key (size change)
                # The safe case is reassignment of an existing key, which
                # is size-preserving and not raised by CPython. We can't
                # statically tell new vs existing, so flag conservatively
                # at severity P1 (one rung lower than del / .pop).
                if isinstance(child, ast.Assign):
                    for tgt in child.targets:
                        if isinstance(tgt, ast.Subscript):
                            sub_chain = _iter_target_chain(tgt.value)
                            if sub_chain == iter_chain:
                                findings.append(Finding(
                                    check="mutation_during_iteration",
                                    severity="P1",
                                    file=rel,
                                    line=child.lineno,
                                    snippet=_line_text(src_lines, child.lineno),
                                    detail=(
                                        f"``{iter_chain}[...] = ...`` while iterating "
                                        f"the same collection at line {node.lineno}. "
                                        f"Reassigning an EXISTING key is size-preserving "
                                        f"and safe (CPython); ADDING a new key changes "
                                        f"size and raises RuntimeError on dict / set. "
                                        f"Statically indistinguishable -- inspect."
                                    ),
                                ))
                # d.append(...) / d.pop(...) / etc.
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                    method = child.func.attr
                    receiver_chain = _iter_target_chain(child.func.value)
                    if receiver_chain == iter_chain and method in (
                        _MUTATING_DICT_METHODS
                        | _MUTATING_LIST_METHODS
                        | _MUTATING_SET_METHODS
                    ):
                        findings.append(Finding(
                            check="mutation_during_iteration",
                            severity="P0",
                            file=rel,
                            line=child.lineno,
                            snippet=_line_text(src_lines, child.lineno),
                            detail=(
                                f"``{iter_chain}.{method}(...)`` while iterating "
                                f"the same collection at line {node.lineno}. "
                                f"Iterate over a copy: ``for x in list({iter_chain}):`` "
                                f"or ``{iter_chain}.copy()``."
                            ),
                        ))
    return findings


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


# --- CLI ----------------------------------------------------------------


def _render_markdown(findings: list[Finding]) -> str:
    if not findings:
        return "_No findings._\n"
    lines = [
        "| Sev | Check | File:Line | Snippet | Detail |",
        "|---|---|---|---|---|",
    ]
    lines.extend(f.as_md_row() for f in findings)
    return "\n".join(lines) + "\n"


def _render_json(findings: list[Finding]) -> str:
    return json.dumps([asdict(f) for f in findings], indent=2, sort_keys=True)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for ``python -m pyutilz.dev.code_audit``.

    Parses argv (uses ``sys.argv[1:]`` when ``None``) into a ``root``
    directory, an optional list of ``--check`` scanner names, an output
    format (markdown or JSON), and a minimum severity threshold. Runs
    the selected scanners against ``root``, prints the rendered findings
    to stdout, and returns an exit code: ``1`` when any P0 or P1 finding
    is present (so CI can gate on the result), ``0`` otherwise.

    Returns the exit code rather than calling ``sys.exit`` directly so
    the function is testable from a process.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m pyutilz.dev.code_audit",
        description=(
            "AST audit: mutable defaults, late-binding closures, "
            "default-via-or trap, silent broad-except swallows. "
            "Designed to be run against any Python source tree, "
            "not just pyutilz."
        ),
    )
    parser.add_argument("root", type=Path, help="source-tree root to scan (e.g. ./src)")
    parser.add_argument(
        "--check",
        action="append",
        choices=sorted(SCANNERS),
        help=(
            "scanner(s) to run; repeat for multiple. Default: run all. "
            "Available: " + ", ".join(sorted(SCANNERS))
        ),
    )
    parser.add_argument(
        "--format", choices=("markdown", "json"), default="markdown",
        help="output format (default markdown).",
    )
    parser.add_argument(
        "--exclude-dir", action="append", default=None,
        help=(
            "directory name to exclude (matched against any path part). "
            "Repeat. Adds to the default set of build/cache/venv dirs."
        ),
    )
    parser.add_argument(
        "--min-severity", choices=("P0", "P1", "P2", "Low"), default="Low",
        help="filter out findings below this severity (default Low: show all).",
    )
    args = parser.parse_args(argv)

    root: Path = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"root must be a directory: {root}")

    exclude_dirs = frozenset(_DEFAULT_EXCLUDE_DIRS | set(args.exclude_dir or ()))

    findings = run_all(root, checks=args.check)
    sev_order = {"P0": 0, "P1": 1, "P2": 2, "Low": 3}
    cutoff = sev_order[args.min_severity]
    findings = [f for f in findings if sev_order.get(f.severity, 99) <= cutoff]

    out = _render_json(findings) if args.format == "json" else _render_markdown(findings)
    sys.stdout.write(out)
    # exit code: non-zero only when P0/P1 found, so CI can gate on it.
    return 1 if any(f.severity in {"P0", "P1"} for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())

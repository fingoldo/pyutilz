"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

# asyncio coordination primitives whose whole purpose is being SHARED across
# concurrent callers -- one created fresh per call is a private, useless copy.
DEFAULT_PRIMITIVE_NAMES: frozenset[str] = frozenset({"Lock", "Event", "Semaphore", "BoundedSemaphore", "Condition"})


def _is_asyncio_primitive_call(node: ast.AST, primitive_names: frozenset[str]) -> bool:
    """True if ``node`` is a call shaped like ``asyncio.Lock()`` / ``asyncio.Semaphore(n)``."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return isinstance(func, ast.Attribute) and func.attr in primitive_names and isinstance(func.value, ast.Name) and func.value.id == "asyncio"


def _is_persistent_target(target: ast.expr) -> bool:
    """True for an assignment target that persists PAST the current function
    call -- an object attribute (``self._lock``, ``cls.sem``) or a subscript
    into one (``instance.__dict__[self._name]``, a lazy-descriptor memoization
    pattern; ``self._cache[key]``). A subscript into a plain local Name
    (``local_dict[key] = ...`` where ``local_dict`` is itself function-scoped)
    is NOT persistent and stays unflagged-by-this-exemption -- only recurse
    through Attribute/Subscript chains, never through a bare Name base."""
    if isinstance(target, ast.Attribute):
        return True
    if isinstance(target, ast.Subscript):
        return _is_persistent_target(target.value)
    return False


def _attribute_assigned_primitive_calls(func: ast.AST) -> set[ast.AST]:
    """Primitive-constructor call nodes that eventually reach a persistent
    target (see ``_is_persistent_target``) ANYWHERE inside ``func`` --
    either directly (``self._lock = asyncio.Lock()``) or via one level of
    local-variable indirection (``value = asyncio.Semaphore(...); ...;
    instance.__dict__[key] = value``, the lazy-descriptor memoization
    shape). Such an assignment persists on the object past the current
    call, exactly the safe "create once, share via the instance" pattern
    (typically in ``__init__``, or a lazy-descriptor's ``__get__``), so
    these are never flagged regardless of which method they appear in."""
    direct: set[ast.AST] = set()
    local_var_of_call: dict[str, ast.AST] = {}
    persisted_var_names: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if any(_is_persistent_target(t) for t in targets):
            direct.add(value)
            if isinstance(value, ast.Name):
                persisted_var_names.add(value.id)
        for t in targets:
            if isinstance(t, ast.Name):
                local_var_of_call[t.id] = value
    aliased = {local_var_of_call[name] for name in persisted_var_names if name in local_var_of_call}
    return direct | aliased


def scan_async_primitive_reinit_per_call(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
    primitive_names: frozenset[str] = DEFAULT_PRIMITIVE_NAMES,
) -> list[Finding]:
    """Find ``asyncio.Lock()``/``Event()``/``Semaphore()``/``Condition()``
    instantiated INSIDE a function or method body, rather than at module or
    class scope.

    These primitives exist specifically to coordinate MULTIPLE concurrent
    callers -- a fresh instance created on every call of the function gives
    each invocation its own private, unshared copy. Callers still run
    concurrently and still "wait" on their own primitive, so nothing crashes
    and no exception is raised; the coordination semantics (mutual exclusion,
    wait-for-signal, bounded concurrency) simply never engage. This is the
    kind of bug that ships and passes tests run one-at-a-time, then silently
    fails to gate anything the moment two callers overlap in production.

    The fix is to create the primitive once at module scope, as a class
    attribute set in ``__init__``, or via a module-level ``functools.cache``
    /``lru_cache``-wrapped factory -- anywhere that guarantees every caller
    gets the SAME object.

    Deliberately narrow: only flags the primitive-constructor call appearing
    directly inside a ``def``/``async def`` body (as an assignment target,
    a default-arg expression, or a bare expression) -- it does not attempt
    to determine whether the enclosing function is actually ever called
    concurrently, since that requires call-graph analysis this scanner
    doesn't do. A function-scoped primitive is a smell whether or not this
    scanner can prove the specific call site is dangerous.

    Severity: P1 (silent coordination no-op under concurrency, not an
    immediate crash -- surfaces only under load).
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()

        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # Only the function's OWN body -- a nested def's primitive calls are
            # reported when THAT nested def is visited on its own turn of the
            # outer ast.walk, so don't double-report by walking into it here.
            nested_defs = {n for n in ast.walk(func) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not func}
            attr_assigned = _attribute_assigned_primitive_calls(func)
            for node in ast.walk(func):
                if any(node in ast.walk(nested) for nested in nested_defs):
                    continue
                if not _is_asyncio_primitive_call(node, primitive_names):
                    continue
                if node in attr_assigned:
                    continue
                assert isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                findings.append(Finding(
                    check="async_primitive_reinit_per_call",
                    severity="P1",
                    file=rel,
                    line=node.lineno,
                    snippet=_line_text(src_lines, node.lineno),
                    detail=(
                        f"asyncio.{node.func.attr}() created inside {func.name}()'s body -- every call gets its "
                        "own private instance, so concurrent callers never actually coordinate through it. Create "
                        "the primitive once (module scope, an __init__-set instance attribute, or a cached "
                        "factory) so every caller shares the SAME object."
                    ),
                ))
    return findings

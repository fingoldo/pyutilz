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


def _is_persistent_target(
    target: ast.expr,
    global_names: frozenset[str],
    module_names: frozenset[str] = frozenset(),
    *,
    via_subscript: bool = False,
) -> bool:
    """True for an assignment target that persists PAST the current function
    call -- an object attribute (``self._lock``, ``cls.sem``), a subscript
    into one (``instance.__dict__[self._name]``, a lazy-descriptor memoization
    pattern; ``self._cache[key]``), or a bare Name declared ``global`` in this
    function (the ``global _sem; if _sem is None: _sem = asyncio.Lock()``
    lazy-module-singleton idiom -- the module-level binding is exactly the
    safe "one shared instance" case, not a fresh-per-call local). A subscript
    into a plain function-LOCAL Name (``local_dict[key] = ...`` where
    ``local_dict`` is itself function-scoped) is NOT persistent and stays
    unflagged-by-this-exemption; a subscript into a MODULE-level container
    is, because such a container is shared by every caller."""
    if isinstance(target, ast.Attribute):
        return True
    if isinstance(target, ast.Subscript):
        return _is_persistent_target(target.value, global_names, module_names, via_subscript=True)
    if isinstance(target, ast.Name):
        # A module-level container mutated in place (`_inflight[key] = asyncio.Event()`,
        # the single-flight/per-key coordination idiom) needs no `global` declaration,
        # because rebinding never happens -- only subscript assignment. Such a primitive
        # is published into shared state for other callers to find, which is the exact
        # OPPOSITE of the private-per-call copy this scanner exists to catch.
        if target.id in global_names:
            return True
        # Only for in-place mutation of an existing container, never for a bare
        # rebinding: a LOCAL named the same as a module-level global shadows it,
        # so `sem = asyncio.Semaphore()` must stay flagged even if some module
        # attribute happens to share the name.
        return via_subscript and target.id in module_names
    return False


def _attribute_assigned_primitive_calls(func: ast.AST, module_names: frozenset[str] = frozenset()) -> set[ast.AST]:
    """Primitive-constructor call nodes that eventually reach a persistent
    target (see ``_is_persistent_target``) ANYWHERE inside ``func`` --
    either directly (``self._lock = asyncio.Lock()``, ``global _sem; _sem =
    asyncio.Lock()``) or via one level of local-variable indirection
    (``value = asyncio.Semaphore(...); ...; instance.__dict__[key] =
    value``, the lazy-descriptor memoization shape). Such an assignment
    persists past the current call, exactly the safe "create once, share"
    pattern (``__init__``, a lazy-descriptor's ``__get__``, or a
    global-declared module-level lazy singleton), so these are never
    flagged regardless of which method they appear in."""
    global_names = frozenset(name for node in ast.walk(func) if isinstance(node, ast.Global) for name in node.names)
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
        if any(_is_persistent_target(t, global_names, module_names) for t in targets):
            direct.add(value)
            if isinstance(value, ast.Name):
                persisted_var_names.add(value.id)
        for t in targets:
            if isinstance(t, ast.Name):
                local_var_of_call[t.id] = value
    aliased = {local_var_of_call[name] for name in persisted_var_names if name in local_var_of_call}
    return direct | aliased


def _module_level_names(tree: ast.Module) -> frozenset[str]:
    """Names bound at module scope: assignments, imports, and def/class names.

    Used to tell ``_inflight[key] = asyncio.Event()`` (publishing into a shared
    module-level registry) apart from ``local_map[key] = asyncio.Event()``.
    """
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            continue
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname if a.asname is not None else a.name.split(".")[0] for a in node.names)
            continue
        else:
            continue
        for t in targets:
            if isinstance(t, ast.Name):
                names.add(t.id)
    return frozenset(names)


def _fanout_limiter_calls(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    nested_defs: set[ast.FunctionDef | ast.AsyncFunctionDef],
) -> set[ast.AST]:
    """Primitive-constructor calls assigned to a local that a closure defined in
    the SAME invocation then reads -- the bounded-fan-out idiom::

        sem = asyncio.Semaphore(limit)

        async def _run(factory):
            async with sem:
                return await factory()

        return await asyncio.gather(*[_run(f) for f in factories])

    Here the primitive is meant to bound the tasks THIS call spawns, not to
    coordinate separate callers, so a fresh instance per call is correct and
    deliberate. Cross-caller coordination through a closure-captured local is
    not expressible anyway -- each call builds its own closures too -- so this
    exemption cannot mask a genuine shared-primitive bug.
    """
    closure_reads = {n.id for nested in nested_defs for n in ast.walk(nested) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    # Same idiom, written with a helper instead of a closure:
    #     sem = asyncio.Semaphore(n); await _run_pipeline_for(session, sem, ...)
    # The primitive is created here and handed to the work THIS call starts.
    # Tradeoff, stated plainly: a primitive created per call and passed into a
    # function that stores it somewhere shared would also be exempted. That
    # shape is rare and, unlike fan-out, is not expressible as a local idiom --
    # a genuinely shared primitive is built at module or instance scope, not
    # constructed fresh at each call site and handed off. Everything that uses
    # the primitive DIRECTLY in this body (`async with sem:`) stays flagged.
    handed_off = {
        n.id
        for node in ast.walk(func)
        if isinstance(node, ast.Call)
        for n in [*node.args, *(kw.value for kw in node.keywords)]
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    consumed_elsewhere = closure_reads | handed_off
    if not consumed_elsewhere:
        return set()
    exempt: set[ast.AST] = set()
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in consumed_elsewhere:
            exempt.add(node.value)
    return exempt

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

    Three shapes are deliberately NOT flagged, because in each the primitive
    is either shared on purpose or scoped to the fan-out it exists to bound:
    assignment to a persistent target (``self._lock``, a ``global``-declared
    module singleton), publication into a MODULE-level container
    (``_inflight[key] = asyncio.Event()`` -- the single-flight idiom), and a
    local consumed by work this same call starts -- captured by a closure, or
    handed to a helper as an argument (the bounded-gather idiom; see
    ``_fanout_limiter_calls``).

    Otherwise deliberately narrow: only flags the primitive-constructor call
    appearing directly inside a ``def``/``async def`` body (as an assignment target,
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
        module_names = _module_level_names(tree)

        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # Only the function's OWN body -- a nested def's primitive calls are
            # reported when THAT nested def is visited on its own turn of the
            # outer ast.walk, so don't double-report by walking into it here.
            nested_defs = {n for n in ast.walk(func) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not func}
            attr_assigned = _attribute_assigned_primitive_calls(func, module_names)
            fanout_exempt = _fanout_limiter_calls(func, nested_defs)
            for node in ast.walk(func):
                if any(node in ast.walk(nested) for nested in nested_defs):
                    continue
                if not _is_asyncio_primitive_call(node, primitive_names):
                    continue
                if node in attr_assigned or node in fanout_exempt:
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

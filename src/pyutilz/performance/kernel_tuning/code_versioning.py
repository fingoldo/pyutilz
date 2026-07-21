"""Deterministic code-version hashing for ``kernel_tuning_cache``.

A kernel's cached tuning is keyed partly by a ``code_version`` so a tuning is
invalidated **only** when the function's logic changes -- not on comment /
whitespace / docstring / decorator edits. ``KernelTuningCache`` checks it
alongside ``hw_fingerprint`` + provenance.

Recipe (per fn): unwrap ``functools.wraps`` (``__wrapped__``) then numba's
``Dispatcher.py_func`` -> ``inspect.getsource`` -> ``ast.parse`` -> locate the
``FunctionDef`` -> strip ``decorator_list`` + a leading docstring ->
``ast.unparse`` (canonical source text; far more stable across CPython minors
than ``ast.dump``'s node repr). Hash ``variant_fns`` + a developer-curated
``extra_fns`` (shared helpers / env-toggled kernels whose source the variants
don't textually contain) + a per-spec integer ``salt``.

The ``salt`` is the escape hatch for semantic dependencies the source hashing
cannot see: module-level constants a sweep reads, numba ``@njit`` directive
changes (``parallel`` / ``fastmath`` -- stripped with the decorator), or a data
generator the sweep uses. Bump it manually when such a dep changes.

Never raises: if source is unavailable (C-extension, REPL-defined, lost source)
it falls back to a ``module+qualname+signature`` identity string.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import logging
import textwrap
from functools import lru_cache
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _unwrap(fn: Callable) -> Callable:
    """Unwrap ``functools.wraps`` decorators and numba Dispatcher to the
    original Python function whose source we want to hash."""
    seen: set[int] = set()
    while hasattr(fn, "__wrapped__") and id(fn) not in seen:
        seen.add(id(fn))
        fn = fn.__wrapped__  # type: ignore[attr-defined]
    # numba @njit / @cuda.jit Dispatcher keeps the original Python function here.
    if hasattr(fn, "py_func"):
        fn = fn.py_func  # type: ignore[attr-defined]
    return fn


def _canonical(node: ast.AST) -> str:
    """Canonical text for an AST node -- ``ast.unparse`` (3.9+), else ``ast.dump``."""
    unparse = getattr(ast, "unparse", None)
    if unparse is not None:
        return unparse(node)  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime
    return ast.dump(node, include_attributes=False)  # 3.8 fallback (less version-stable)


def _normalized_source(fn: Callable) -> str:
    """Comment/whitespace/docstring/decorator-insensitive canonical source of ``fn``.

    Falls back to an identity string if the source cannot be retrieved or parsed."""
    original = _unwrap(fn)
    name = getattr(original, "__name__", None)
    try:
        src = textwrap.dedent(inspect.getsource(original))
    except (OSError, TypeError) as e:
        logger.debug("code_versioning: no source for %r (%s); identity fallback", fn, e)
        try:
            sig = str(inspect.signature(original))
        except (ValueError, TypeError):
            sig = "<no-sig>"
        return f"__nosrc__:{getattr(original, '__module__', '?')}.{getattr(original, '__qualname__', name)}{sig}"
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        logger.debug("code_versioning: parse failed for %r (%s); hashing raw source", fn, e)
        return src
    target = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            target = node
            break
    if target is None:  # lambda / unusual def -- hash the whole parsed module canonically
        return _canonical(tree)
    # Keep decorator CALL-ARGS but not the decorator's identity/import path (B5):
    # a numba directive change (@njit(parallel=True) -> fastmath=True) alters
    # codegen AND the optimal tuning, so it must invalidate the cache; a bare
    # decorator rename / import move must not. We fold the sorted arg text into
    # the hash and then drop the decorator nodes themselves.
    deco_args: list = []
    for deco in target.decorator_list:
        if isinstance(deco, ast.Call):
            deco_args += [_canonical(a) for a in deco.args]
            deco_args += [f"{kw.arg}={_canonical(kw.value)}" for kw in deco.keywords]
    target.decorator_list = []
    if target.body and isinstance(target.body[0], ast.Expr) and isinstance(target.body[0].value, ast.Constant) and isinstance(target.body[0].value.value, str):
        target.body = target.body[1:] or [ast.Pass()]
    target.name = "_k_"  # name-normalize: a pure rename / module move is not a logic change
    canonical = _canonical(target)
    if deco_args:
        canonical += "\n# deco-args: " + ",".join(sorted(deco_args))
    return canonical


def _code_identity_key(fn: Callable) -> Any:
    """Return a key that changes when ``fn``'s bytecode changes, even though ``fn`` may keep the
    SAME object identity -- e.g. IPython's ``%autoreload 2`` patches ``fn.__code__`` in place
    rather than rebinding the name to a new function object. CPython code objects compare/hash
    STRUCTURALLY (by bytecode/consts/names/etc.), unlike plain function objects (``id()``-based),
    so keying on the code object itself gives correct content-based cache invalidation "for
    free" -- no need to separately hash source text just to build the key. Falls back to
    ``id(fn)`` for callables with no ``__code__`` (e.g. a C-extension callable).
    """
    unwrapped = _unwrap(fn)
    code = getattr(unwrapped, "__code__", None)
    return code if code is not None else id(unwrapped)


@lru_cache(maxsize=4096)
def _normalized_source_by_code(fn: Callable, code_key: Any) -> str:
    """Memoized :func:`_normalized_source`, LRU-keyed on ``(fn, code_key)``.

    ``code_key`` (see :func:`_code_identity_key`) is what actually makes this content-aware: if
    ``fn``'s bytecode is swapped in place, ``code_key`` changes, the ``(fn, code_key)`` tuple no
    longer matches any cached entry, and this recomputes -- unlike the old design (a plain
    ``lru_cache`` directly on ``compute_code_version``, keyed by the function objects
    themselves), which stayed identity-blind to exactly that swap. See the HIGH audit finding
    (2026-07-21 audit round 2) this fixes.
    """
    return _normalized_source(fn)


def compute_code_version(*variant_fns: Callable, extra_fns: tuple = (), salt: int = 0) -> str:
    """Deterministic SHA-256 over the normalized source of all ``variant_fns`` +
    ``extra_fns`` + ``salt``.

    Order-independent (parts are sorted by module+qualname) so passing the
    variants in any order yields the same hash. A change to ANY listed function's
    logic -- or a ``salt`` bump -- changes the result; comment / format / docstring
    edits do not.

    Per-function normalized source is memoized (the AST parse + unparse is ~1.3 ms/call) via
    :func:`_normalized_source_by_code`, content-keyed rather than identity-keyed (see that
    function's docstring) -- this function itself is NOT ``lru_cache``d, only the per-function
    step is, so an in-place code swap on any one variant is picked up on the next call without
    needing to invalidate anything keyed by the full ``variant_fns`` tuple.
    """
    # Logic-only: hash the name-normalized sources, sorted, so order / rename /
    # module-move do not change the result -- only a body-logic edit or salt does.
    sources = sorted(_normalized_source_by_code(fn, _code_identity_key(fn)) for fn in list(variant_fns) + list(extra_fns))
    h = hashlib.sha256()
    for src in sources:
        h.update(src.encode())
        h.update(b"\x00")
    h.update(f"salt={int(salt)}".encode())
    return h.hexdigest()


__all__ = ["compute_code_version"]

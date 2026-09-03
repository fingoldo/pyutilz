"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _line_text, _read_src_lines, _safe_parse

# --- unpicklable live-resource state without __getstate__ ----------------

# Constructor calls that produce a live, process-bound, unpicklable object when assigned to
# ``self.<attr>``. Matched on the call's final attribute/name component so both ``threading.Lock()``
# and a bare ``Lock()`` (imported via ``from threading import Lock``) are caught.
_UNPICKLABLE_CTORS = frozenset({
    "Lock", "RLock", "BoundedSemaphore",
    "Thread", "Process", "SimpleQueue", "JoinableQueue",
    "Popen",
})

# Names that ALSO name ordinary picklable domain objects -- `catboost.Pool(X, y)` is the obvious
# one in this codebase's ML domain, and a domain `Event`/`Queue`/`Condition` dataclass is routine.
# These need a known concurrency base (or a `from threading import ...` trace), the same discipline
# `_UNPICKLABLE_DOTTED_CTORS` already applies to the ambiguous `Stream`/`Event` spellings.
_AMBIGUOUS_CTORS = frozenset({"Pool", "Queue", "Event", "Condition", "Barrier", "Semaphore"})
_RESOURCE_MODULES = frozenset({"threading", "multiprocessing", "asyncio", "subprocess", "queue", "mp"})

# Attribute-access chains (e.g. ``torch.cuda.Stream()``) whose LAST component alone is ambiguous
# (``Stream`` could be anything) -- matched by the fully dotted call name instead.
_UNPICKLABLE_DOTTED_CTORS = frozenset({
    "torch.cuda.Stream",
    "torch.cuda.Event",
    "cuda.Stream",
})

# Builtin ``open(...)`` calls assigned to ``self.<attr>`` are the same "live resource held past
# construction" shape -- a file handle can't survive pickling either.
_OPEN_BUILTIN = "open"


def _dotted_call_name(call: ast.Call) -> str:
    """Return the dotted name of ``call.func`` (e.g. ``"torch.cuda.Stream"``), or "" if it isn't a
    plain Name/Attribute chain (e.g. a subscript or another call)."""
    parts: list[str] = []
    node: ast.AST = call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    else:
        return ""
    return ".".join(reversed(parts))


def _last_component(dotted: str) -> str:
    """Return the final ``.``-separated component of a dotted name."""
    return dotted.rsplit(".", 1)[-1]


def _resource_imported_names(tree: ast.Module) -> set[str]:
    """Names bound by ``from threading import Lock`` / ``from multiprocessing import Pool`` etc."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in _RESOURCE_MODULES:
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def _is_unpicklable_ctor_call(call: ast.Call, resource_names: "set[str] | None" = None) -> bool:
    """True if ``call`` constructs a live, process-bound resource (lock, thread, open file, CUDA
    stream/event) that can't survive a plain pickle round-trip."""
    dotted = _dotted_call_name(call)
    if not dotted:
        return False
    if dotted in _UNPICKLABLE_DOTTED_CTORS:
        return True
    last = _last_component(dotted)
    if last in _UNPICKLABLE_CTORS:
        return True
    if last in _AMBIGUOUS_CTORS:
        base = dotted.rsplit(".", 2)[0] if "." in dotted else ""
        if base in _RESOURCE_MODULES:
            return True
        return dotted == last and last in (resource_names or set())
    if dotted == _OPEN_BUILTIN:
        return True
    return False


def _self_attr_target(target: ast.AST) -> "str | None":
    """Return ``attr`` if ``target`` is ``self.<attr>`` (an ``ast.Attribute`` whose value is the
    ``self`` Name), else None. Only the conventional first-parameter name ``self`` is matched --
    a differently-named instance parameter is out of scope for this heuristic scanner."""
    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
        return target.attr
    return None


def _find_unpicklable_self_assignments(init_body: list[ast.stmt], resource_names: "set[str] | None" = None) -> list[tuple[str, ast.stmt]]:
    """Walk an ``__init__`` body for ``self.<attr> = <unpicklable ctor call>`` assignments (top-level
    Assign/AnnAssign only -- doesn't descend into nested function/lambda scopes, matching this
    scanner's other heuristics' conservative-false-negative bias). Returns ``[(attr, node), ...]``."""
    hits: list[tuple[str, ast.stmt]] = []
    for stmt in init_body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            attr = _self_attr_target(stmt.targets[0])
            value = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            attr = _self_attr_target(stmt.target)
            value = stmt.value
        else:
            continue
        if attr is None or not isinstance(value, ast.Call):
            continue
        if _is_unpicklable_ctor_call(value, resource_names):
            hits.append((attr, stmt))
    return hits


def _class_defines_getstate(class_node: ast.ClassDef) -> bool:
    """True if ``class_node`` (only -- not its base classes, which this static scanner can't
    resolve across module boundaries) defines ``__getstate__``."""
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "__getstate__" for n in class_node.body)


def scan_unpicklable_resource_state(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find classes whose ``__init__`` assigns a live, process-bound, unpicklable resource
    (``threading.Lock``/``RLock``/``Event``/etc., a ``Thread``/``Process``/``Pool``, an ``open()``
    file handle, or a CUDA stream/event) directly to ``self.<attr>`` with no ``__getstate__``
    defined on the class to drop it before a pickle round-trip.

    A plain ``pickle.dumps(instance)`` on such a class raises ``TypeError: cannot pickle
    '_thread.lock' object`` (or the CUDA/file-handle equivalent) the moment anything upstream
    tries to cache, checkpoint, or ship the instance across a process boundary -- often long after
    the class was written, when a caller neither knows nor should need to know about the internal
    lock. The fix is ``__getstate__``/``__setstate__`` dropping/recreating the live attribute
    (see e.g. mlframe's ``FeatureCache`` and ``training/neural/ranker.py``'s trainer_/CUDA-tensor
    exclusion for the established pattern this scanner generalizes).

    False-negative-biased by design (matches this package's other heuristic scanners): does not
    resolve inherited ``__getstate__`` across module boundaries, does not follow attribute
    assignment outside ``__init__``, and only recognizes the constructor-call shapes listed above
    -- a class assigning an unpicklable resource via a differently-named factory function is not
    flagged. Severity P2: real but usually latent until something actually pickles the instance.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = _read_src_lines(py)
        rel = py.relative_to(root).as_posix()
        resource_names = _resource_imported_names(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            init_fn = next(
                (n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "__init__"),
                None,
            )
            if init_fn is None:
                continue
            hits = _find_unpicklable_self_assignments(init_fn.body, resource_names)
            if not hits:
                continue
            if _class_defines_getstate(node):
                continue
            for attr, assign_node in hits:
                findings.append(Finding(
                    check="unpicklable_resource_state",
                    severity="P2",
                    file=rel,
                    line=assign_node.lineno,
                    snippet=_line_text(src_lines, assign_node.lineno),
                    detail=(
                        f"class {node.name!r} assigns self.{attr} from an unpicklable live resource "
                        f"constructor with no __getstate__ defined on the class; pickle.dumps(instance) "
                        f"will raise TypeError the first time anything caches/ships this instance."
                    ),
                ))
    return findings

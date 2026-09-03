"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _line_text, _read_src_lines, _safe_parse

# --- per-call state on a shared instance ---------------------------------

# A name mentioned in a `with` header that makes the block a critical section. Matched on `_`-separated
# SEGMENTS of the identifiers in the context expression, so `self._lock`, `_provider_lock`,
# `asyncio.Lock()` and `threading.RLock()` count while `blocking`, `blocklist`, `unlocked` and
# `clock` -- which a substring match accepted, silently suppressing real findings -- do not.
_LOCK_HINTS = frozenset({"lock", "rlock", "semaphore", "mutex"})

# Attribute-name shapes that are per-CALL by construction: whatever the last call left behind, read
# back later by a summary accessor. These are the names the defect keeps reappearing under.
_PERCALL_PREFIXES = ("last_", "_last_", "current_", "_current_")


def _mentions_lock(node: ast.AST) -> bool:
    """Whether a ``with``/``async with`` context expression names something lock-shaped."""
    for inner in ast.walk(node):
        name = inner.id if isinstance(inner, ast.Name) else (inner.attr if isinstance(inner, ast.Attribute) else None)
        if name is None:
            continue
        if any(segment.lower() in _LOCK_HINTS for segment in name.split("_")):
            return True
    return False


def _self_attr_stores(func: ast.AST, self_name: str) -> dict[str, tuple[int, bool]]:
    """Map ``self.<attr>`` STORE sites in ``func`` to (line, guarded-by-a-lock).

    Guardedness is lexical: the assignment sits inside a ``with``/``async with`` whose header names
    something lock-shaped. That under-reports a lock taken by a caller, which is why this scanner
    warns rather than blocks.
    """
    stores: dict[str, tuple[int, bool]] = {}

    def walk(node: ast.AST, locked: bool) -> None:
        """Recurses the tree, carrying whether the current node sits inside a lock-shaped ``with`` block."""
        for child in ast.iter_child_nodes(node):
            child_locked = locked
            if isinstance(child, (ast.With, ast.AsyncWith)) and not locked:
                child_locked = any(_mentions_lock(item.context_expr) for item in child.items)
            if isinstance(child, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                for target in targets:
                    if isinstance(child, ast.AnnAssign) and child.value is None:
                        continue  # `self.x: int` declares the attribute's type; it stores nothing.
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == self_name:
                        # First write wins for the reported line; guardedness is OR'd, so a single
                        # unguarded write among several locked ones still reports.
                        line, was_locked = stores.get(target.attr, (child.lineno, True))
                        stores[target.attr] = (min(line, child.lineno), was_locked and child_locked)
            # A nested def/lambda has its own `self` binding only if it rebinds the name; treat the
            # enclosing self as still visible, which is what a closure over `self` actually does.
            walk(child, child_locked)

    walk(func, False)
    return stores


def _self_attr_loads(func: ast.AST, self_name: str) -> set[str]:
    """Every ``self.<attr>`` read in ``func``."""
    return {
        node.attr
        for node in ast.walk(func)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == self_name and isinstance(node.ctx, ast.Load)
    }


def _first_arg_name(func: ast.AST) -> str:
    """Returns the name the method uses for its instance, falling back to ``self`` when there is no first argument."""
    if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)) and func.args.args:
        return func.args.args[0].arg
    return "self"


def _collect_shared_class_names(trees: dict[Path, ast.Module]) -> set[str]:
    """Class names whose instances are SHARED rather than constructed per use.

    Three signals, all read off the tree itself rather than assumed:

    1. an instance is stored into a module-level container (``_cache[key] = Foo(...)``);
    2. a factory decorated with ``lru_cache``/``cache`` constructs one;
    3. the class NAME appears as a string constant in a module-level container -- the registry
       shape, where the class is resolved by name and the instance handed out by a cached factory
       (``llm.factory._PROVIDER_MODULES`` plus its ``_provider_cache`` is exactly this).

    Signal 3 is what makes the check see provider classes at all: nothing ever writes
    ``_provider_cache[key] = OpenRouterProvider(...)`` literally.
    """
    shared: set[str] = set()
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Subscript) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                        shared.add(node.value.func.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                decorators = " ".join(ast.dump(d) for d in node.decorator_list)
                if "lru_cache" in decorators or "'cache'" in decorators or '"cache"' in decorators:
                    for inner in ast.walk(node):
                        if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
                            shared.add(inner.func.id)
    # Signal 3 is deliberately restricted to MODULE-LEVEL containers (the registry shape). Walking the
    # whole tree instead let any capitalised identifier-shaped string in any local list -- e.g.
    # `labels = ["Worker", "Other"]` in an unrelated function -- mark a class as shared.
    for tree in trees.values():
        for stmt in tree.body:
            if not isinstance(stmt, (ast.Assign, ast.AnnAssign)) or stmt.value is None:
                continue
            for container in ast.walk(stmt.value):
                if not isinstance(container, (ast.Dict, ast.List, ast.Tuple, ast.Set)):
                    continue
                for element in ast.walk(container):
                    if isinstance(element, ast.Constant) and isinstance(element.value, str) and element.value.isidentifier() and element.value[:1].isupper():
                        shared.add(element.value)
    return shared


def scan_per_call_state_on_shared_instance(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find per-call state written onto a SHARED instance during an in-flight ``async def``.

    The shape: a class whose instances are cached and handed to many callers stores what the last
    call produced (``last_usage``, ``_last_json_schema_applied``, ``current_model``) on ``self``, and
    a separate accessor reads it back. Two concurrent calls on the same instance interleave, so the
    accessor reports the OTHER request's numbers, and no exception is ever raised - the values are
    simply attributed to the wrong call. Metadata that describes ONE request belongs in that
    request's own result, not on the instance.

    Reported when all of the following hold:

    1. the attribute is assigned inside an ``async def`` of the class;
    2. it is read in a DIFFERENT method (a summary accessor), or its name is per-call-shaped
       (``last_*`` / ``current_*``) - either is enough, since a write-only per-call attribute is
       still misattributed the moment anything reads it;
    3. no lock-shaped ``with``/``async with`` guards the assignment;
    4. the class's instances are shared (see :func:`_collect_shared_class_names`).

    Severity: P2, and the check is opt-in rather than part of the default sweep. Lexical lock
    detection under-reports a lock held by the caller, and a provider that is single-caller by
    contract is a legitimate exception - so this produces a triage list to read, not a verdict.
    """
    findings: list[Finding] = []
    trees: dict[Path, ast.Module] = {}
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is not None:
            trees[py] = tree

    shared_names = _collect_shared_class_names(trees)

    # A shared subclass makes its in-tree bases shared too: the attributes usually live on the base
    # (OpenAICompatibleProvider), while only the concrete subclass is ever named in a registry.
    for _ in range(4):  # bounded transitive closure; provider hierarchies here are 2-3 deep
        grew = False
        for tree in trees.values():
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name in shared_names:
                    for base in node.bases:
                        name = base.id if isinstance(base, ast.Name) else (base.attr if isinstance(base, ast.Attribute) else None)
                        if name and name not in shared_names:
                            shared_names.add(name)
                            grew = True
        if not grew:
            break

    for py, tree in trees.items():
        src_lines = _read_src_lines(py)
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name not in shared_names:
                continue
            methods = [m for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
            async_methods = [m for m in methods if isinstance(m, ast.AsyncFunctionDef)]
            if not async_methods:
                continue

            reads_elsewhere: dict[str, set[str]] = {}
            for method in methods:
                for attr in _self_attr_loads(method, _first_arg_name(method)):
                    reads_elsewhere.setdefault(attr, set()).add(method.name)

            for method in async_methods:
                for attr, (line, guarded) in _self_attr_stores(method, _first_arg_name(method)).items():
                    if guarded:
                        continue
                    readers = reads_elsewhere.get(attr, set()) - {method.name}
                    per_call_shaped = attr.startswith(_PERCALL_PREFIXES)
                    if not readers and not per_call_shaped:
                        continue
                    reader_text = f"read back by {sorted(readers)[0]}()" if readers else "per-call-shaped name"
                    findings.append(
                        Finding(
                            check="per_call_state_on_shared_instance",
                            severity="P2",
                            file=rel,
                            line=line,
                            snippet=_line_text(src_lines, line),
                            detail=(
                                f"`self.{attr}` is written during the in-flight `async def "
                                f"{method.name}()` of `{node.name}`, whose instances are shared "
                                f"between callers, with no lock held ({reader_text}). Two concurrent "
                                f"calls interleave and the value is attributed to the wrong request. "
                                f"Return it in that call's own result (or snapshot it per call) "
                                f"instead of storing it on the instance."
                            ),
                        )
                    )
    return findings

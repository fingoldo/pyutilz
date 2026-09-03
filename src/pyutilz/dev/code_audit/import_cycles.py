"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path
from typing import FrozenSet

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _safe_parse, _is_excluded

# --- internal import-graph cycle detection ----------------------------------


def _module_name_from_path(path: Path, root: Path, package_name: str) -> str:
    """``<root>/llm/factory.py`` -> ``<package_name>.llm.factory``."""
    rel = path.relative_to(root)
    parts = list(rel.parts)
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][: -len(".py")]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join([package_name, *parts])


def _absolute_import_targets(node: ast.AST, package_name: str) -> "set[str]":
    """In-package modules an ABSOLUTE import statement pulls in (``import pkg.x`` / ``from pkg.x import y``).

    ``from pkg.x import y`` loads the SUBMODULE ``pkg.x.y`` when one exists, so both the module and
    every ``pkg.x.<alias>`` candidate are emitted; the caller prunes candidates that are not real
    modules. Returns an empty set for anything that is not an in-package absolute import.
    """
    out: "set[str]" = set()
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name.startswith(package_name):
                out.add(alias.name)
    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and node.module.startswith(package_name):
        out.add(node.module)
        for alias in node.names:
            out.add(f"{node.module}.{alias.name}")
    return out


def _relative_import_targets(node: ast.ImportFrom, current_parts: "list[str]", package_name: str, is_package: bool) -> "set[str]":
    """In-package modules a RELATIVE ``from . import`` / ``from .mod import`` statement pulls in.

    A package's ``__init__.py`` IS its package, so ``from .a import f`` there resolves against the
    package itself -- one fewer component to strip than for a plain module. And bare
    ``from . import other`` imports the SUBMODULES, not the base package; emitting an edge to the
    base package fabricates a cycle through the parent that does not exist.
    """
    out: "set[str]" = set()
    strip = node.level - 1 if is_package else node.level
    base_parts = current_parts[: len(current_parts) - strip] if strip else list(current_parts)
    if node.module:
        base_parts.append(node.module)
        if base_parts and base_parts[0] == package_name:
            out.add(".".join(base_parts))
            for alias in node.names:
                out.add(".".join([*base_parts, alias.name]))
    elif base_parts and base_parts[0] == package_name:
        for alias in node.names:
            out.add(".".join([*base_parts, alias.name]))
    return out


def _internal_imports(tree: ast.AST, current: str, package_name: str, is_package: bool = False) -> set[str]:
    """Yield fully-qualified names ``current`` imports from inside the same package, considering
    ONLY top-level imports -- lazy imports inside function bodies don't participate in the
    module-load dependency graph (they fire after both modules have finished loading), so a
    "cycle" that only closes via lazy imports isn't a runtime ImportError waiting to happen.
    """
    out: set[str] = set()
    current_parts = current.split(".")
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ImportFrom) and node.level != 0:
            out |= _relative_import_targets(node, current_parts, package_name, is_package)
        elif isinstance(node, ast.If):
            # `if TYPE_CHECKING: ... import X` is not at runtime, already excluded by iterating
            # top-level statements without walking into other If bodies -- but `try: import
            # optdep / except ...` IS top-level and contributes, so descend into If bodies here.
            for sub in ast.walk(node):
                out |= _absolute_import_targets(sub, package_name)
        else:
            out |= _absolute_import_targets(node, package_name)
    return out


def _package_roots(root: Path) -> list[Path]:
    """The importable package directories under ``root``.

    ``root`` is documented by the CLI as "source-tree root to scan (e.g. ./src)", which is NOT
    itself a package -- taking ``root.name`` as the package name there ("src") matched no import
    target at all and made the whole check a silent no-op.
    """
    if (root / "__init__.py").is_file():
        return [root]
    subs: list[Path] = []
    try:
        # `__init__.py` is not required: PEP 420 namespace packages are importable too.
        subs.extend(d for d in sorted(root.iterdir()) if d.is_dir() and next(d.rglob("*.py"), None) is not None)
    except OSError:
        pass
    if not subs or next(root.glob("*.py"), None) is not None:
        subs.append(root)
    return subs


def _build_graph(root: Path, package_name: str, exclude_dirs: FrozenSet[str], display_root: Path | None = None) -> tuple[dict, dict]:
    """``({module_name: imported internal module names}, {module_name: file path})``."""
    graph: dict = defaultdict(set)
    files: dict = {}
    base = display_root if display_root is not None else root
    for py in root.rglob("*.py"):
        if _is_excluded(py, root, exclude_dirs):
            continue
        tree = _safe_parse(py)
        if tree is None:
            continue
        mod_name = _module_name_from_path(py, root, package_name)
        graph[mod_name].update(_internal_imports(tree, mod_name, package_name, py.name == "__init__.py"))
        files[mod_name] = py.relative_to(base).as_posix()
    return graph, files


def _strongly_connected_components(graph: dict) -> list:
    """Tarjan's SCC. Returns a list of components (each >= 1 node); cycles are SCCs with > 1 node."""
    index_counter = [0]
    stack: list = []
    lowlinks: dict = {}
    index: dict = {}
    on_stack: dict = {}
    result: list = []

    def strongconnect(v: str) -> None:
        """Tarjan's DFS step: assign ``v`` an index/lowlink and pop its completed SCC off the stack."""
        index[v] = index_counter[0]
        lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True
        for w in graph.get(v, ()):
            if w not in index:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif on_stack.get(w, False):
                lowlinks[v] = min(lowlinks[v], index[w])
        if lowlinks[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                comp.append(w)
                if w == v:
                    break
            result.append(comp)

    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(10_000)
    try:
        for v in list(graph):
            if v not in index:
                strongconnect(v)
    finally:
        sys.setrecursionlimit(old_limit)
    return result


def scan_import_cycles(
    root: Path,
    exclude_dirs: FrozenSet[str] = _DEFAULT_EXCLUDE_DIRS,
    *,
    package_name: str = "",
    deferred_cycles: FrozenSet[str] = frozenset(),
) -> list[Finding]:
    """Find multi-node cycles in the package's internal top-level import graph.

    A circular import can lurk for years because Python's import machinery resolves cycles at
    module-load if the offending name is accessed lazily -- then a refactor moves a top-level
    access of one of the cycle members and ``ImportError: cannot import name X from partially
    initialized module Y`` ships to users. Builds a dependency graph by AST-walking every .py file
    for top-level ``import``/``from`` statements (lazy imports inside function bodies are
    excluded -- they don't participate in the module-load cycle), restricted to imports whose
    target is inside the package itself, then runs Tarjan's SCC to find any cycle of size > 1.

    ``package_name`` defaults to ``root.name`` when not given (the usual convention: ``root`` is
    the package's own source directory, e.g. ``src/mypackage``, so its directory name IS the
    importable package name).

    ``deferred_cycles``: a set of ``" -> ".join(sorted(component))`` strings for cycles the project
    has reviewed and confirmed benign (e.g. a sibling-file monolith split where both halves
    top-level-import each other but the cycle resolves at runtime because the parent binds its
    exports before importing the sibling) -- these are not flagged. Each project maintains its own
    list; nothing is baked into this scanner.
    """
    if package_name:
        package_dirs = [(root, package_name)]
    else:
        package_dirs = [(directory, directory.name) for directory in _package_roots(root)]
    graph: dict = defaultdict(set)
    module_files: dict = {}
    for directory, name in package_dirs:
        sub_graph, sub_files = _build_graph(directory, name, exclude_dirs, root)
        for module, targets in sub_graph.items():
            graph[module].update(targets)
        module_files.update(sub_files)
    # Keep only edges that name a module actually present in the scan; `from pkg import CONSTANT`
    # contributes the candidate `pkg.CONSTANT`, which is a name inside a module, not a module.
    for module in list(graph):
        graph[module] = {target for target in graph[module] if target in module_files and target != module}
    if not graph:
        return []
    sccs = _strongly_connected_components(graph)
    findings: list[Finding] = []
    for comp in sccs:
        if len(comp) <= 1:
            continue
        # Tarjan's DFS visits each SCC's edges via a plain `set` (see _build_graph /
        # strongconnect above), whose iteration order depends on Python's per-process string
        # hash randomization -- so the SAME cycle in the SAME code can pop off the stack in a
        # different member order across runs. Sorting here pins a single deterministic
        # representative (both for `file=` and for `cycle_display`), so the Finding's
        # ``check::file:line`` key -- and therefore any baseline built from it -- is stable
        # regardless of PYTHONHASHSEED. The SCC membership itself (which files participate) is
        # already order-independent; only the display/key representative needed pinning.
        comp = sorted(comp)
        comp_key = " -> ".join(comp)
        if comp_key in deferred_cycles:
            continue
        cycle_display = " -> ".join([*comp, comp[0]])
        findings.append(Finding(
            check="import_cycle",
            severity="P1",
            # The real path, from the graph itself: a package member's file is `<name>/__init__.py`,
            # which a dotted-name-to-path rewrite cannot produce.
            file=module_files.get(comp[0], comp[0].replace(".", "/") + ".py"),
            line=1,
            snippet=cycle_display,
            detail=(
                f"import cycle of {len(comp)} module(s) in the internal import graph: "
                f"{cycle_display}. Resolves at runtime only if every member happens to bind its "
                f"top-level exports before the cycle's back-edge executes -- a later refactor can "
                f"silently break this into `ImportError: cannot import name X from partially "
                f"initialized module Y`."
            ),
        ))
    return findings

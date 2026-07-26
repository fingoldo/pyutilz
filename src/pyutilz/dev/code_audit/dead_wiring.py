"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path

from ._base import _DEFAULT_EXCLUDE_DIRS, Finding, _iter_py_files, _line_text, _safe_parse

# --- dead wiring: built, measured, documented, never called ---------------------------------------
#
# Confirmed repeatedly in the 2026-07-26 Autopsia audits: a deterministic modality mapper measured at
# 0.775 against a 0.425 majority baseline with zero callers; an ordinal-reasoning module written to
# retire a thresholding rule its own API never stopped using; a routing tier invoked only from tests;
# a whole hierarchy read surface over a 900k-concept vocabulary install that nothing queried.
#
# A plain "is this name mentioned anywhere else" grep is not enough for two reasons. The routing tier
# hid behind an IMPORT of its module, so an import-graph check saw a live edge over a callable nothing
# invoked. And a public function called only by another dead public function is dead too. So this is a
# reachability computation: seed with everything a caller outside the audited tree, a module body, a
# decorator or a `main()` can reach, propagate through call edges, and report the complement.


def _names_mentioned(node: ast.AST) -> set[str]:
    """Every identifier a subtree mentions: `Name` ids, attribute tails, and string constants.

    String constants count because dispatch tables and `getattr(mod, "screen")` reference by name;
    including them trades a few false negatives for far fewer false positives.
    """
    out: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            out.add(child.id)
        elif isinstance(child, ast.Attribute):
            out.add(child.attr)
        elif isinstance(child, ast.ImportFrom):
            out.update(a.name for a in child.names)
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            out.add(child.value)
    return out


def scan_dead_public_callables(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
    consumer_roots: tuple[Path, ...] = (),
    entry_point_names: frozenset[str] = frozenset({"main"}),
) -> list[Finding]:
    """Find public top-level callables under ``root`` that nothing outside a test suite can reach.

    ``root`` is the audited production tree. ``consumer_roots`` are additional trees that count as
    call sites but are not themselves audited (demo scripts, a separate ``bench/`` tree, an
    application that imports the library) -- pass them so a callable used only by a demo is not
    reported. Note what is deliberately NOT a consumer root: the test suite. A callable exercised
    only by its own tests is exactly the defect this scanner exists to name.

    Not flagged, because each is genuinely invoked by something this analysis cannot see:
    decorated functions (a FastAPI route, a pytest fixture, a registry decorator), anything named
    in ``entry_point_names``, and anything referenced at module scope (``__all__``, an argparse
    dispatch table, a default argument).

    Severity: P2. The code is not wrong; it is unreachable, which is worse than wrong because it
    accrues maintenance, documentation and measurement while changing nothing.
    """
    audited = sorted(_iter_py_files(root, exclude_dirs))
    consumers = list(audited)
    for extra in consumer_roots:
        consumers.extend(sorted(_iter_py_files(Path(extra), exclude_dirs)))

    trees: dict[Path, ast.Module] = {}
    for path in {*audited, *consumers}:
        tree = _safe_parse(path)
        if tree is not None:
            trees[path] = tree

    funcs: dict[tuple[Path, str], ast.FunctionDef | ast.AsyncFunctionDef] = {}
    module_scope: dict[Path, set[str]] = {}
    for path in audited:
        tree = trees.get(path)
        if tree is None:
            continue
        scope: set[str] = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs[(path, node.name)] = node
            else:
                scope |= _names_mentioned(node)
        module_scope[path] = scope

    external = {path: _names_mentioned(tree) for path, tree in trees.items()}

    seed: set[tuple[Path, str]] = set()
    for (path, name), node in funcs.items():
        if name in entry_point_names or getattr(node, "decorator_list", []):
            seed.add((path, name))
        elif name in module_scope.get(path, ()):
            seed.add((path, name))
        elif any(name in names for other, names in external.items() if other != path):
            seed.add((path, name))

    by_name: dict[str, list[tuple[Path, str]]] = {}
    for key in funcs:
        by_name.setdefault(key[1], []).append(key)

    live: set[tuple[Path, str]] = set()
    stack = list(seed)
    while stack:
        key = stack.pop()
        if key in live:
            continue
        live.add(key)
        for mentioned in _names_mentioned(funcs[key]):
            stack.extend(target for target in by_name.get(mentioned, ()) if target not in live)

    findings: list[Finding] = []
    for (path, name), node in sorted(funcs.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        if (path, name) in live or name.startswith("_"):
            continue
        src_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        findings.append(
            Finding(
                check="dead_public_callable",
                severity="P2",
                file=path.relative_to(root).as_posix(),
                line=node.lineno,
                snippet=_line_text(src_lines, node.lineno),
                detail=(
                    f"public callable {name!r} is not reachable from any module body, entry point, decorator or "
                    f"consumer outside the test suite -- it is built, documented and dead. Wire it or delete it."
                ),
            )
        )
    return findings

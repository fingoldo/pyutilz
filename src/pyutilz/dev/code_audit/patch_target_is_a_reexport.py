"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path

from ._base import _DEFAULT_EXCLUDE_DIRS, Finding, _iter_py_files, _line_text, _safe_parse

# --- a patch aimed at a re-export, where the real call site never looks -------------------------
#
# `patch("facade.fetch")` rebinds the name in `facade`'s namespace. That works only if the code
# under test reaches `fetch` THROUGH `facade` at call time. When `facade` merely re-exports it
# (`from _impl import fetch`) and the real caller lives in `_impl` and calls `fetch()` directly,
# the patch rebinds a name nobody reads: the production function still runs, the mock records no
# calls, and every assertion about it is vacuous.
#
# What it costs depends on what the test then does, and the rule does not overstate it. A test that
# asserts ON THE MOCK (`m.assert_called_once()`, or an expected return value) fails loudly, and is
# merely confusing. The expensive case is the defensive patch -- one applied to keep a test off the
# network or the database -- where the assertions are about the output: the production function
# runs, reaches the real dependency, and the test passes anyway. Verified by construction: the
# worked example in the tests returns the production value, not the mocked one.
#
# A facade module is precisely where this happens, and a facade is exactly what a package with a
# monolith split into siblings and re-exports will have.
#
# MEASURED REACH, so silence is not mistaken for coverage. On the scraper codebase this rule comes
# from: 1,407 string patch targets, 823 of them naming a re-exported name, 328 where the facade
# does not use the name itself -- and zero where the implementation calls it directly, because
# every implementation module there reaches back through its own facade (`_facade.fetch()`). Those
# patches all land. That is the answer, not a gap: an earlier version that counted the
# facade-routed call as a direct one reported fifteen correct tests as vacuous.
#
# Reported when all three hold:
#
# * the patch target names a module in this tree that binds the name by `from ... import name`,
#   not by defining it.
# * the module the name is imported FROM also calls it.
# * the facade module itself does not call it -- if the facade calls it too, the patch does reach
#   that call, and whether the test means the other one is not decidable from here.

_PATCHERS = frozenset({"patch", "patch.object"})


def _patch_targets(tree: ast.Module) -> list[tuple[str, int]]:
    """(dotted target, line) for every `patch("a.b.c")` in this module."""
    targets: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name != "patch":
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str) and "." in first.value:
            targets.append((first.value, node.lineno))
    return targets


def _resolve(module: str | None, level: int, package: str) -> str:
    """Turn a possibly-relative `from . import x` module into an absolute dotted path.

    Relative imports are the normal spelling inside a package, and leaving them unresolved was why
    this rule found nothing at all: 328 candidate patches in one codebase all named an origin like
    `_core`, which resolves to no top-level module, so every one was dropped at the last step.
    """
    if not level:
        return module or ""
    base = package.split(".")
    # `from . import x` inside `pkg.mod` is `pkg`; each extra dot climbs one more level.
    prefix = base[: len(base) - level + 1] if len(base) >= level else []
    return ".".join([*prefix, module]) if module else ".".join(prefix)


def _reexports(tree: ast.Module, package: str) -> dict[str, str]:
    """Names this module binds by `from X import name` rather than defining, mapped to X."""
    defined = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    imported: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        origin = _resolve(node.module, node.level, package)
        if not origin:
            continue
        for alias in node.names:
            bound = alias.asname or alias.name
            if bound != "*" and bound not in defined:
                imported[bound] = origin
    return imported


def _bare_calls(tree: ast.Module) -> set[str]:
    """Names this module calls DIRECTLY -- `fetch()`, never `facade.fetch()`.

    The distinction is the whole rule, and getting it wrong inverted the answer. An implementation
    module routinely calls back through its own facade (`_facade._load_tracked_active_from_db(db)`),
    and a patch on the facade DOES reach that call. Counting attribute calls here reported fifteen
    correct tests in one codebase as vacuous.
    """
    return {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}


def _any_calls(tree: ast.Module) -> set[str]:
    """Names this module calls at all, bare or through an attribute.

    Used for the FACADE side, where either spelling means the patch lands on something real.
    """
    called: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            called.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            called.add(node.func.attr)
    return called


def _module_name(rel: str) -> str:
    """The dotted module path a repo-relative file corresponds to."""
    stem = rel[:-3] if rel.endswith(".py") else rel
    if stem.endswith("/__init__"):
        stem = stem[: -len("/__init__")]
    return stem.replace("/", ".")


def scan_patch_target_is_a_reexport(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find a test that patches a name on a facade the real caller never looks through.

    `patch("facade.fetch")` rebinds the name in `facade`'s namespace, which reaches the code under
    test only if it resolves `fetch` through `facade` at call time. When `facade` re-exports it
    from `_impl` and the caller lives in `_impl` calling `fetch()` directly, the patch rebinds a
    name nobody reads: the production function runs, the mock records nothing, and every assertion
    about it is vacuous.

    Silent when the facade calls the name too -- the patch does reach that call, and which one the
    test means is not decidable from here.
    """
    findings: list[Finding] = []
    modules: dict[str, ast.Module] = {}
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is not None:
            modules[_module_name(py.relative_to(root).as_posix())] = tree

    for py in _iter_py_files(root, exclude_dirs):
        name = py.name
        if not (name.startswith("test_") or name.endswith("_test.py")):
            continue
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()

        for target, line in _patch_targets(tree):
            module_path, _, attribute = target.rpartition(".")
            facade = modules.get(module_path)
            if facade is None:
                continue
            origin = _reexports(facade, module_path).get(attribute)
            if origin is None:
                continue
            # The facade reaching the name itself means the patch does land somewhere real.
            if attribute in _any_calls(facade):
                continue
            source = modules.get(origin)
            if source is None or attribute not in _bare_calls(source):
                continue
            findings.append(
                Finding(
                    check="patch_target_is_a_reexport",
                    severity="P1",
                    file=rel,
                    line=line,
                    snippet=_line_text(src_lines, line),
                    detail=(
                        f"`{module_path}` does not define `{attribute}` -- it re-exports it from "
                        f"`{origin}`, which is also where it is called. Patching it here rebinds a "
                        "name nothing reads at call time: the production function still runs, the "
                        f"mock records nothing, and every assertion about it holds vacuously. Patch "
                        f"`{origin}.{attribute}` instead."
                    ),
                )
            )
    return findings

"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs.

DELIBERATE PAIR, not an accident -- `patch_target_is_a_reexport.py` targets the same defect class
and is the one `run_all()` selects by default. Both are kept, and they differ in exactly one place:
their false-positive suppression rule. THIS one is the narrower, older half: it reports the
ambiguity when it cannot see the call site's binding style, so it is the stricter reading and can
speak about a case its sibling stays silent on. Registered but listed in `registry.OPT_IN_ONLY`,
so upgrading pyutilz does not suddenly report every site twice; name it in `checks=` to run it.
"""

from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _line_text, _read_src_lines, _safe_parse

# The three idioms that rebind a name on a module object for the duration of a test.
_PATCH_CALLS = frozenset({"setattr", "patch", "object"})


def _reexported_names(tree: ast.Module) -> dict[str, str]:
    """``{bound_name: module_it_came_from}`` for every module-scope ``from X import name`` in a file.

    This is what makes a module a FACADE for that name: it holds its own reference, bound at import
    time, and the module that DEFINES the name is somewhere else. Star imports are skipped (the bound
    names are not knowable statically) and so is ``import x``/``import x as y``, which binds a module
    rather than a name re-exported from one.
    """
    out: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            out[alias.asname if alias.asname is not None else alias.name] = node.module
    return out


def _patched_targets(tree: ast.Module) -> list[tuple[str, str, int]]:
    """``(module_expr, attribute_name, lineno)`` for every monkeypatch-shaped call in a test file.

    Covers ``monkeypatch.setattr(mod, "name", ...)``, ``mock.patch.object(mod, "name")`` and the bare
    ``setattr(mod, "name", ...)`` form. Only the two-positional-argument shape is read: the
    string-target form (``patch("pkg.mod.name")``) names its module in the string itself and is
    resolved by a different rule, so reporting it here would be guesswork.
    """
    out: list[tuple[str, str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) < 2:
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name not in _PATCH_CALLS:
            continue
        target, attr = node.args[0], node.args[1]
        if not isinstance(attr, ast.Constant) or not isinstance(attr.value, str):
            continue
        module_expr = target.id if isinstance(target, ast.Name) else (target.attr if isinstance(target, ast.Attribute) else "")
        if module_expr:
            out.append((module_expr, attr.value, node.lineno))
    return out


def _module_aliases(tree: ast.Module) -> dict[str, str]:
    """``{local_name: dotted_module}`` for ``import a.b as c`` and ``from a import b`` module imports.

    A test almost never patches a module under its own dotted path - it binds a short alias first
    (``import autopsia.reason.clinical as clinical``), so the alias has to be resolved back to the
    real module before its file can be found.
    """
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out[alias.asname if alias.asname is not None else alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                out[alias.asname if alias.asname is not None else alias.name] = f"{node.module}.{alias.name}"
    return out


def _references_outside_imports(tree: ast.Module, name: str) -> bool:
    """Whether the module USES ``name`` anywhere beyond the import that bound it.

    This is the precision filter that separates a pure FACADE from a real caller. A module that
    re-exports a name and also calls it is a legitimate patch target - the stub changes what that
    module itself does. A module that only re-exports it cannot be affected by the patch at all, so
    the patch was aimed at somebody else's binding and is silently doing nothing.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name:
            return True
        if isinstance(node, ast.Attribute) and node.attr == name:
            return True
    return False


def scan_reexport_patch_target(
    root: Path | str,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
    package_roots: tuple[Path | str, ...] = (),
) -> list[Finding]:
    """A test that patches a name on the module which RE-EXPORTS it rather than the one that CALLS it.

    A monkeypatch rebinds a name in ONE module's namespace. When the module under test did
    ``from x import f`` at its own module scope, it already holds its own reference - so patching
    ``x.f`` leaves that module running the original, and the test passes while proving nothing.
    That silent pass is the expensive half: a red test gets fixed, a green one that no longer
    exercises its mechanism gets trusted.

    The shape this finds: a test patches ``M.name``, and ``M`` itself obtained ``name`` from another
    module via ``from OTHER import name``. ``M`` is then a facade for that name, and whichever module
    holds the CALL is the correct target - which is ``OTHER`` when the call site did its own
    module-scope import.

    NOT reported: a lazy ``from ... import`` inside a function body, where the name resolves at call
    time and patching the defining module IS correct. This scanner cannot see the call site's binding
    style, so it reports the ambiguity rather than a verdict, and says so in ``detail``.

    ``package_roots`` are the source trees to resolve a patched module's own file in. With none given
    the scan still runs but can only report modules it finds under ``root`` itself.
    """
    root = Path(root)
    roots = [Path(p) for p in package_roots] if package_roots else [root]

    def _module_file(dotted: str) -> Path | None:
        """The source file for a dotted module name, searched under each configured package root."""
        rel = Path(*dotted.split("."))
        for base in roots:
            for candidate in (base / rel.with_suffix(".py"), base.parent / rel.with_suffix(".py")):
                if candidate.is_file():
                    return candidate
        return None

    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        patched = _patched_targets(tree)
        if not patched:
            continue
        src_lines = _read_src_lines(py)
        aliases = _module_aliases(tree)
        rel = py.relative_to(root).as_posix()

        for module_expr, attr, lineno in patched:
            dotted = aliases.get(module_expr)
            if not dotted:
                continue
            target_file = _module_file(dotted)
            if target_file is None:
                continue
            target_tree = _safe_parse(target_file)
            if target_tree is None:
                continue
            origin = _reexported_names(target_tree).get(attr)
            if origin is None:
                continue  # the patched module defines the name itself - the target is unambiguous
            if _references_outside_imports(target_tree, attr):
                continue  # it re-exports the name AND calls it, so patching it here changes its own behaviour
            findings.append(
                Finding(
                    check="reexport_patch_target",
                    severity="P2",
                    file=rel,
                    line=lineno,
                    snippet=_line_text(src_lines, lineno),
                    detail=(
                        f"patches '{attr}' on '{dotted}', which does not define it - it re-exports it from "
                        f"'{origin}' via a module-scope import. A monkeypatch rebinds ONE module's namespace, so "
                        f"any caller that did its own `from {origin} import {attr}` keeps the original and the "
                        f"stub is ignored, leaving the test green while exercising nothing. Patch the module that "
                        f"holds the CALL. If the call site imports '{attr}' lazily inside a function body, the "
                        f"current target is correct - say so in a comment so this is not re-flagged."
                    ),
                )
            )
    return findings

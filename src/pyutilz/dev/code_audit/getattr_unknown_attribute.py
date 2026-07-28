"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path

from ._base import _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _line_text, _safe_parse, Finding


def _class_attribute_names(root: Path, exclude_dirs: frozenset[str]) -> set[str]:
    """Every name that is a class attribute ANYWHERE in the tree.

    Deliberately the union across the whole project rather than the attributes of the specific object being
    read. Resolving `getattr(a.b.c, "x")` to a type needs cross-module inference, which is where this kind of
    check normally dies; the union needs none and is still decisive, because a name that is not an attribute
    of ANY class here cannot be an attribute of the one at hand.

    Collected from everywhere a name is used AS an attribute or as a module global, because each of those is
    direct evidence that the name exists on something:
      * annotated class-body fields (`x: int`) - dataclasses, and the common case;
      * plain class-body assignments and method/property names;
      * ANY attribute assignment, `anything.x = ...`, not only `self.x`. A `threading.local()` and a plain
        namespace object gain their attributes exactly this way, and a rule that ignored them would report
        every such read as a miss;
      * module-level bindings, since `getattr(some_module, "NAME", default)` is a legitimate pattern and the
        name is a global rather than an attribute of any class;
      * any UNGUARDED attribute access, `x.name`. This is the widening that makes the rule usable at all on
        code that works with objects it does not define: a file that reads `node.lineno` directly is
        asserting that the attribute exists, and it would crash on the first call if it did not. Without it
        the rule fires on every read of a stdlib or third-party field - measured at 49 hits on this package
        alone, all of them `ast` node attributes.

    The defect the rule exists for survives this widening precisely BECAUSE it is the defect: a name that is
    only ever reached through a defaulted `getattr` is a name nobody was ever confident enough to touch
    directly, and that is the signature of a field that does not exist.

    Deliberately generous. The cost of a false positive here is a reader sent to inspect correct code; the
    cost of a false negative is a panel that silently prints nothing. But generosity is bounded by what is
    written in the tree, and the defect this rule exists for - a name that appears NOWHERE as an attribute -
    survives all four widenings.
    """
    names: set[str] = set()
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        for item in tree.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                names.add(item.target.id)
            elif isinstance(item, ast.Assign):
                names.update(t.id for t in item.targets if isinstance(t, ast.Name))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        names.add(item.target.id)
                    elif isinstance(item, ast.Assign):
                        names.update(t.id for t in item.targets if isinstance(t, ast.Name))
                    elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        names.add(item.name)
            elif isinstance(node, ast.Assign):
                names.update(t.attr for t in node.targets if isinstance(t, ast.Attribute))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Attribute):
                names.add(node.target.attr)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
    return names


def scan_getattr_unknown_attribute(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
    extra_known: frozenset[str] = frozenset(),
) -> list[Finding]:
    """Find ``getattr(obj, "name", default)`` where ``name`` is an attribute of no class in this tree.

    Such a call can only ever return the default. It does not raise, it does not warn, and with the common
    ``getattr(obj, "rows", None) or []`` spelling it produces an empty result that reads as "there was
    nothing" rather than "nobody asked". A renamed field, a printer written against an older shape, or a
    plain typo all land here and stay silent for as long as nobody checks the value against another source.

    **This is not hypothetical.** The rule exists because a project's flagship demonstration printed two of
    its headline panels as empty for want of `out.plan.steps` and `out.flows.lines` - neither of which had
    ever existed on those dataclasses, whose real fields were `ask`/`obtain`/`refused`/`notes` and
    `hypotheses`/`mechanisms`/`findings`/`flows`. The work behind both panels was computed and paid for on
    every run. Strategy was then argued from the blank output for several days.

    Only the THREE-argument form is reported. Two-argument `getattr(obj, "name")` raises `AttributeError`
    on a miss, which is loud and needs no scanner; it is the default that converts the mistake into silence.

    ``extra_known`` is for names a project reads on objects it does not define - a third-party class, or a
    duck-typed protocol. Passing one is a statement that the attribute exists somewhere out of tree, so it
    belongs in the caller's parameters rather than in this rule.

    Severity: P1 - silently wrong output, and by construction invisible to any test that asserts on the
    same expression.
    """
    known = _class_attribute_names(root, exclude_dirs) | set(extra_known)
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"):
                continue
            # Three arguments exactly: a two-argument getattr raises on a miss and is not silent.
            if len(node.args) != 3 or node.keywords:
                continue
            attribute = node.args[1]
            if not (isinstance(attribute, ast.Constant) and isinstance(attribute.value, str)):
                continue  # a computed name is a different pattern and this rule cannot judge it
            name = attribute.value
            if name in known or name.startswith("__"):
                continue
            findings.append(
                Finding(
                    check="getattr_unknown_attribute",
                    severity="P1",
                    file=rel,
                    line=node.lineno,
                    snippet=_line_text(src_lines, node.lineno),
                    detail=(
                        f"getattr(..., {name!r}, default) always returns the default: {name!r} is an "
                        f"attribute of no class defined in this tree. Read the real field, or pass the name "
                        f"in `extra_known` if it belongs to a class defined outside it."
                    ),
                )
            )
    return findings

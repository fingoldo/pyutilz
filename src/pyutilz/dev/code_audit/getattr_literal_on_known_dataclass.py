"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path

from ._base import _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _line_text, _safe_parse, Finding


def _dataclass_fields(tree: ast.Module) -> dict[str, set[str]]:
    """``{class name -> its declared field names}`` for every ``@dataclass``-decorated class in one module.

    Only annotated class-body assignments count as fields (``x: int``), which is what a dataclass actually
    turns into a constructor parameter / instance attribute - a plain ``x = 3`` inside the class body is a
    class-level constant, not a field, and including it would let a genuine miss on the real fields hide
    behind an unrelated class constant of the same name.
    """
    out: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(_is_dataclass_decorator(dec) for dec in node.decorator_list):
            continue
        fields = {item.target.id for item in node.body if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)}
        out[node.name] = fields
    return out


def _is_dataclass_decorator(dec: ast.expr) -> bool:
    """Whether a decorator expression is ``@dataclass`` or ``@dataclass(...)``, qualified or not."""
    target = dec.func if isinstance(dec, ast.Call) else dec
    if isinstance(target, ast.Name):
        return target.id == "dataclass"
    if isinstance(target, ast.Attribute):
        return target.attr == "dataclass"
    return False


def _local_dataclass_types(func: ast.AST, known: dict[str, set[str]]) -> dict[str, str]:
    """``{local variable name -> dataclass class name}`` inferred within ONE function body.

    Two sources, both intentionally conservative (a missed binding only costs a missed finding, never a
    false one): a parameter annotated with a known dataclass's name, and a plain assignment ``x = ClassName(...)``
    calling a known dataclass's constructor directly. Anything else - a function return, an attribute chain,
    a rebind through a different constructor - is left untracked rather than guessed at.
    """
    if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return {}
    types: dict[str, str] = {}
    all_args = (*func.args.posonlyargs, *func.args.args, *func.args.kwonlyargs)
    for arg in all_args:
        if isinstance(arg.annotation, ast.Name) and arg.annotation.id in known:
            types[arg.arg] = arg.annotation.id
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            cls_name = node.value.func.id
            if cls_name in known:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        types[target.id] = cls_name
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and isinstance(node.annotation, ast.Name):
            if node.annotation.id in known:
                types[node.target.id] = node.annotation.id
    return types


def scan_getattr_literal_on_known_dataclass(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find ``getattr(x, "literal", default)`` where ``x`` is a KNOWN dataclass instance whose real fields
    do not include ``"literal"``.

    Narrower and stronger than ``scan_getattr_unknown_attribute``: that rule only catches a name that is an
    attribute of NOTHING anywhere in the tree, so it misses the sharper case of a name that IS a real field
    on some OTHER class but not on the one actually being read - which is exactly the shape of a printer
    written against a stale or wrong dataclass shape (``out.plan.steps`` when ``Sheet`` has never had a
    ``steps`` field, but some unrelated class happens to).

    ``x``'s type is inferred locally and conservatively per function - a parameter annotated with a known
    dataclass's name, or a variable just assigned from that dataclass's own constructor call. Anything this
    cannot resolve is left alone rather than guessed at, by design: the false-positive cost of guessing wrong
    here is a reader sent to inspect correct code for nothing, and duck-typing across an intentional boundary
    (an object built one way and read as if built another) is the one legitimate escape hatch this leaves
    unflagged.

    A literal is also excluded when it is assigned as an attribute ANYWHERE in the tree
    (``anything.literal = ...``), tree-wide and untyped - a dataclass instance that gains fields dynamically
    (a sibling module doing ``state.cache = ...`` on an instance passed in) is real, direct evidence the name
    is a legitimate attribute, even though it is invisible to ``_dataclass_fields``'s literal-annotation walk.
    Measured: without this widening, three genuinely wired dynamic-attribute reads on one project's dataclass
    (a batch-precompute cache and two per-call scratch flags, each set via ``instance.name = value`` from a
    sibling module) reported as this class's flagship defect while being correct, live code.

    Only the THREE-argument form is reported, same reasoning as ``scan_getattr_unknown_attribute``: two-arg
    ``getattr(obj, "name")`` raises on a miss and needs no scanner.

    Severity: P1 - the same class of silent, test-invisible defect ``scan_getattr_unknown_attribute``
    documents, found by a project that shipped `out.plan.steps`/`out.flows.lines` against dataclasses whose
    real fields were `ask`/`obtain`/`refused`/`notes` and `hypotheses`/`mechanisms`/`findings`/`flows` - two
    panels of the same flagship demo printed empty while the work behind them was computed and paid for.
    """
    trees: dict[Path, ast.Module] = {}
    known: dict[str, set[str]] = {}
    dynamic_attrs: set[str] = set()
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        trees[py] = tree
        known.update(_dataclass_fields(tree))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                dynamic_attrs.update(t.attr for t in node.targets if isinstance(t, ast.Attribute))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Attribute):
                dynamic_attrs.add(node.target.attr)

    findings: list[Finding] = []
    for py, tree in trees.items():
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            local_types = _local_dataclass_types(func, known)
            if not local_types:
                continue
            for node in ast.walk(func):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"):
                    continue
                if len(node.args) != 3 or node.keywords:
                    continue
                obj_arg, name_arg = node.args[0], node.args[1]
                if not isinstance(obj_arg, ast.Name) or obj_arg.id not in local_types:
                    continue
                if not (isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str)):
                    continue
                cls_name = local_types[obj_arg.id]
                literal = name_arg.value
                if literal.startswith("__") or literal in known[cls_name] or literal in dynamic_attrs:
                    continue
                findings.append(
                    Finding(
                        check="getattr_literal_on_known_dataclass",
                        severity="P1",
                        file=rel,
                        line=node.lineno,
                        snippet=_line_text(src_lines, node.lineno),
                        detail=(
                            f"getattr({obj_arg.id}, {literal!r}, default): {obj_arg.id} is a {cls_name!r} "
                            f"instance and {cls_name!r} has no field named {literal!r} ({sorted(known[cls_name])!r} "
                            "are its real fields) - this call can only ever return the default."
                        ),
                    )
                )
    return findings

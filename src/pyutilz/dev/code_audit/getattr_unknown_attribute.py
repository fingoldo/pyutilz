"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _line_text, _read_src_lines, _safe_parse


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
        alone, all of them `ast` node attributes;
      * every ``__init__`` parameter name. The common sklearn-`BaseEstimator`-style constructor binds each
        parameter onto ``self`` via a bulk ``for k, v in defaults.items(): setattr(self, k, v)`` (or an
        equivalent loop) rather than one literal `self.x = x` per field - real, direct evidence the name is
        an attribute, but invisible to a literal-assignment walk since neither `self.<name>` nor a string
        constant naming it appears anywhere in the source. Measured: a single large sklearn-style estimator
        with ~500 constructor parameters bound this way produced ~500 false `getattr_unknown_attribute` hits
        for its own fields before this widening;
      * module-level ``def``/``class``/``from X import Y`` bindings, not just module-level ``Assign``. A
        very common facade-patchability pattern is ``getattr(some_module, "func_name", func_name)`` --
        looking a name up on a live module object (so a test's ``monkeypatch.setattr(module, "func_name",
        ...)`` is honored) with the LOCAL, in-tree function/import as the fallback. The module-level-bindings
        widening above already claims to cover this ("since `getattr(some_module, "NAME", default)` is a
        legitimate pattern"), but the implementation only walked module-level `Assign`/`AnnAssign`, missing
        `def`/`class`/`import` bindings entirely -- confirmed in the wild dogfooding this scanner on
        pyutilz's own source (2026-08-04): ``cache_base.py``'s ``getattr(_facade, "_cpu_model_slug",
        _cpu_model_slug)`` (a module-level function) and ``getattr(_facade, "gpu_capability_summary",
        gpu_capability_summary)`` (a module-level `from ... import` name) both false-positived because
        neither kind of module-level binding was in ``names``.

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
            elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(item.name)
            elif isinstance(item, ast.Import):
                names.update((a.asname or a.name.split(".")[0]) for a in item.names)
            elif isinstance(item, ast.ImportFrom):
                names.update((a.asname or a.name) for a in item.names if a.name != "*")
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        names.add(item.target.id)
                    elif isinstance(item, ast.Assign):
                        names.update(t.id for t in item.targets if isinstance(t, ast.Name))
                    elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        names.add(item.name)
                        if item.name == "__init__":
                            _args = item.args
                            names.update(a.arg for a in (*_args.posonlyargs, *_args.args, *_args.kwonlyargs) if a.arg != "self")
            elif isinstance(node, ast.Assign):
                names.update(t.attr for t in node.targets if isinstance(t, ast.Attribute))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Attribute):
                names.add(node.target.attr)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
    return names


def _dynamic_attribute_classes(tree: ast.Module) -> set[str]:
    """Classes in one module that bind attributes onto ``self`` DYNAMICALLY -- a
    ``setattr(self, k, v)`` with a non-literal name, or ``self.__dict__.update(...)``.

    A name bound this way appears nowhere in the source as an attribute, so no walk can see it and
    every ``getattr(self, "x", d)`` on such a class reads as a miss. Measured on a downstream repo
    (2026-09-03): a config object whose fields are injected by a bulk ``setattr`` loop produced a
    long tail of false ``getattr_unknown_attribute`` hits for its own, live fields.
    """
    dynamic: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == "setattr" and len(inner.args) >= 2:
                target, name_arg = inner.args[0], inner.args[1]
                if isinstance(target, ast.Name) and target.id == "self" and not isinstance(name_arg, ast.Constant):
                    dynamic.add(node.name)
            elif isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) and inner.func.attr == "update":
                base = inner.func.value
                if isinstance(base, ast.Attribute) and base.attr == "__dict__" and isinstance(base.value, ast.Name) and base.value.id == "self":
                    dynamic.add(node.name)
    return dynamic


def _in_tree_class_names(root: Path, exclude_dirs: frozenset[str]) -> tuple[set[str], set[str]]:
    """``(class names defined in the tree, of those the ones with dynamic attribute binding)``."""
    classes: set[str] = set()
    dynamic: set[str] = set()
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        classes.update(node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        dynamic |= _dynamic_attribute_classes(tree)
    return classes, dynamic


def _receivers_defined_in_tree(tree: ast.Module, classes: set[str], dynamic: set[str]) -> set[int]:
    """``{id(receiver_node)}`` for every ``getattr`` receiver this module can PROVE is an instance
    of a class defined in the scanned tree, or a module of the scanned tree itself.

    Deliberately conservative -- exactly three bindings are recognised, each of them written in the
    file being scanned:

    * ``self`` inside a class defined here (and not one that binds its attributes dynamically);
    * a parameter or variable ANNOTATED with an in-tree class name (``sheet: Sheet``);
    * a local assigned straight from an in-tree class's constructor (``sheet = Sheet(...)``).

    Everything else is left alone. That is the whole repair of 2026-09-03: without it the rule's
    premise ("an attribute of no class in this tree") is applied to receivers this tree does not
    own -- an ``ast`` node, a pandas dtype, a CUDA error object, a foreign model tagged by a
    sibling library -- for which the premise says nothing at all. 13 of 13 sampled findings across
    two fresh repos were exactly that, with no true positive observed.
    """
    resolved: set[int] = set()
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        local: set[str] = set()
        args = func.args
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            if isinstance(arg.annotation, ast.Name) and arg.annotation.id in classes and arg.annotation.id not in dynamic:
                local.add(arg.arg)
        for node in ast.walk(func):
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and isinstance(node.annotation, ast.Name):
                if node.annotation.id in classes and node.annotation.id not in dynamic:
                    local.add(node.target.id)
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                if node.value.func.id in classes and node.value.func.id not in dynamic:
                    local.update(t.id for t in node.targets if isinstance(t, ast.Name))
        for node in ast.walk(func):
            if isinstance(node, ast.Name) and node.id in local:
                resolved.add(id(node))
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef) or cls.name in dynamic:
            continue
        for node in ast.walk(cls):
            if isinstance(node, ast.Name) and node.id == "self":
                resolved.add(id(node))
    return resolved


def _module_receiver_names(tree: ast.Module) -> set[str]:
    """Names bound to a module object of THIS tree: the ``_facade = sys.modules[__name__]`` /
    ``import <own package> as _facade`` self-reference idiom. Reading an unknown name off one is
    the same provable miss as reading it off an in-tree class, since the module's globals are all
    written here."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Subscript):
            base = node.value.value
            if isinstance(base, ast.Attribute) and base.attr == "modules":
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def scan_getattr_unknown_attribute(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
    extra_known: frozenset[str] = frozenset(),
    require_known_receiver: bool = True,
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

    The RECEIVER must be an object this tree defines (``require_known_receiver``, on by default):
    ``self`` in an in-tree class, a name annotated with an in-tree class, a variable assigned from
    an in-tree constructor, or a self-referencing module object. The premise "this name is an
    attribute of no class in this tree" is only decisive about a receiver the tree owns; applied
    to an ``ast`` node, a pandas dtype or a foreign model object it says nothing, and a
    2026-09-03 scan of two fresh repos measured 13 of 13 sampled findings as exactly that, with no
    true positive observed. Pass ``require_known_receiver=False`` for the pre-2026-09-03,
    tree-wide-name-union-only behaviour -- it is a broader net at a measured cost of near-total
    noise on code that touches stdlib or third-party objects, which is all real code.

    ``extra_known`` is for names a project reads on objects it does not define - a third-party class, or a
    duck-typed protocol. Passing one is a statement that the attribute exists somewhere out of tree, so it
    belongs in the caller's parameters rather than in this rule.

    Severity: P1 - silently wrong output, and by construction invisible to any test that asserts on the
    same expression.
    """
    known = _class_attribute_names(root, exclude_dirs) | set(extra_known)
    classes, dynamic = _in_tree_class_names(root, exclude_dirs) if require_known_receiver else (set(), set())
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = _read_src_lines(py)
        rel = py.relative_to(root).as_posix()
        resolved = _receivers_defined_in_tree(tree, classes, dynamic) if require_known_receiver else set()
        module_receivers = _module_receiver_names(tree) if require_known_receiver else set()
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
            if require_known_receiver:
                receiver = node.args[0]
                if id(receiver) not in resolved and not (isinstance(receiver, ast.Name) and receiver.id in module_receivers):
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

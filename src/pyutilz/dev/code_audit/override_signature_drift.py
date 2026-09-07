"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ._base import _DEFAULT_EXCLUDE_DIRS, Finding, _iter_py_files, _line_text, _safe_parse

# --- a base method left behind by the overrides that were all edited ---------------------------
#
# A parameter is added to a method on three of the five subclasses that override it, and not to the
# concrete method on the base they share. Every subclass that did NOT override it keeps inheriting
# the old signature, so the call the public API now advertises raises `TypeError` -- but only on
# whichever subclass a given deployment happens to instantiate, which is why it reaches CI rather
# than the author's box. This is the pyutilz `generate_json(thinking=...)` incident: `thinking`
# reached the claude-code, gemini and openai-compat providers and not `LLMProvider`, and the
# signature-snapshot test refreshed itself around the edit without a word.
#
# The rule keys on AGREEMENT AMONG SIBLINGS, not on a single override widening its base. Measured
# over pyutilz, py-ci-shared and mlframe, "an override accepts a parameter its base does not" fires
# on ordinary specialisation -- one subclass that alone supports a feature -- and the sibling
# requirement is what separates the two: when two independent subclasses reach for the same
# parameter name on the same method, the base is the thing that is behind, not the subclasses.

_MIN_SIBLINGS = 2


def _decorator_names(func: ast.AST) -> Set[str]:
    """Every dotted component of every decorator on ``func``, as a flat set of names."""
    found: Set[str] = set()
    for dec in getattr(func, "decorator_list", []):
        node: ast.AST = dec.func if isinstance(dec, ast.Call) else dec
        while isinstance(node, ast.Attribute):
            found.add(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            found.add(node.id)
    return found


def _is_stub(func: "ast.FunctionDef | ast.AsyncFunctionDef") -> bool:
    """Whether ``func`` only declares a contract: abstract, or a body of ``pass``/``...``/``raise NotImplementedError``.

    Nothing inherits such a signature at run time -- every subclass must supply its own -- so a
    subclass parameter the declaration lacks cannot produce the ``TypeError`` this rule is about.
    """
    if "abstractmethod" in _decorator_names(func):
        return True
    body = [stmt for stmt in func.body if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))]
    if not body:
        return True
    if len(body) != 1:
        return False
    only = body[0]
    if isinstance(only, ast.Pass):
        return True
    return isinstance(only, ast.Raise) and "NotImplementedError" in ast.dump(only)


def _declared_params(func: "ast.FunctionDef | ast.AsyncFunctionDef") -> Set[str]:
    """Parameter names ``func`` accepts by name, without ``self``/``cls`` and without ``*args``/``**kwargs``."""
    args = func.args
    positional = list(getattr(args, "posonlyargs", [])) + args.args
    return {arg.arg for arg in positional[1:]} | {arg.arg for arg in args.kwonlyargs}


def _accepts_anything(func: "ast.FunctionDef | ast.AsyncFunctionDef") -> bool:
    """Whether ``func`` has a ``**kwargs``, which makes any keyword call on it legal."""
    return func.args.kwarg is not None


def _placeholder_variants(func: "ast.FunctionDef | ast.AsyncFunctionDef") -> Set[str]:
    """Underscore-stripped forms of the base's ``_unused``-style parameter names.

    A base hook that ignores its arguments names them ``_body``/``_status``; the override that uses
    them drops the underscore. That rename is the convention working as intended, not drift.
    """
    return {name.lstrip("_") for name in _declared_params(func) if name.startswith("_")}


def _methods(klass: ast.ClassDef) -> "Dict[str, ast.FunctionDef | ast.AsyncFunctionDef]":
    """The directly defined methods of ``klass``, by name, excluding dunders and ``@overload`` stubs."""
    found: "Dict[str, ast.FunctionDef | ast.AsyncFunctionDef]" = {}
    for node in klass.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("__") and node.name.endswith("__"):
            continue
        if "overload" in _decorator_names(node):
            continue
        found[node.name] = node
    return found


def _base_names(klass: ast.ClassDef) -> List[str]:
    """The names of ``klass``'s declared bases, keeping only the final component of a dotted one."""
    names: List[str] = []
    for base in klass.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
    return names


def _index_classes(root: Path, exclude_dirs: "frozenset[str]") -> "Dict[str, Optional[Tuple[str, ast.ClassDef]]]":
    """Map every class name under ``root`` to its ``(relative file, node)``, or to ``None`` when the name is ambiguous.

    A name defined by two different modules cannot be resolved to one base without an import graph,
    and guessing would attribute a subclass's parameters to a class that never had them.
    """
    index: "Dict[str, Optional[Tuple[str, ast.ClassDef]]]" = {}
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                index[node.name] = None if node.name in index else (rel, node)
    return index


def _ancestors(
    klass: ast.ClassDef,
    index: "Dict[str, Optional[Tuple[str, ast.ClassDef]]]",
    seen: Optional[Set[str]] = None,
) -> "List[Tuple[str, str, ast.ClassDef]]":
    """Resolvable ancestors of ``klass`` in declaration order, breadth first, as ``(name, file, node)``."""
    seen = set() if seen is None else seen
    out: "List[Tuple[str, str, ast.ClassDef]]" = []
    for name in _base_names(klass):
        entry = index.get(name)
        if entry is None or name in seen:
            continue
        seen.add(name)
        out.append((name, entry[0], entry[1]))
        out.extend(_ancestors(entry[1], index, seen))
    return out


def scan_override_signature_drift(
    root: Path,
    exclude_dirs: "frozenset[str]" = _DEFAULT_EXCLUDE_DIRS,
) -> List[Finding]:
    """Find a concrete base method that lacks a parameter two or more of its overrides declare.

    A parameter added to several subclasses' overrides and not to the base they share leaves every
    subclass that inherits the base implementation raising ``TypeError`` on a call the API now
    advertises -- and only on the subclasses a given deployment happens to use, so it surfaces in
    CI rather than at the keyboard. A signature-snapshot test does not see it: the snapshot records
    change, not compatibility, and refreshing it after the edit makes it silent.

    Agreement among siblings is the signal. One override that widens its base is ordinary
    specialisation -- measured across three codebases, that form is almost entirely a subclass
    alone supporting a feature -- while two independent subclasses reaching for the same parameter
    name on the same method means the base is the one left behind.

    Exempt: ``@abstractmethod`` and stub-bodied base methods (nothing inherits that signature at
    run time), a base with ``**kwargs`` (any keyword call on it is already legal), ``@overload``
    stubs, dunders, class names defined more than once under ``root``, and a base parameter that
    differs from the override's only by a leading underscore.
    """
    index = _index_classes(root, exclude_dirs)
    # (base class name, method name) -> parameter -> the subclasses declaring it.
    widened: "Dict[Tuple[str, str], Dict[str, Set[str]]]" = {}

    for name, entry in index.items():
        if entry is None:
            continue
        klass = entry[1]
        ancestors = _ancestors(klass, index)
        if not ancestors:
            continue
        for method_name, method in _methods(klass).items():
            base = next(((bn, bnode) for bn, _, bnode in ancestors if method_name in _methods(bnode)), None)
            if base is None:
                continue
            base_name, base_node = base
            base_method = _methods(base_node)[method_name]
            if _is_stub(base_method) or _accepts_anything(base_method):
                continue
            extra = _declared_params(method) - _declared_params(base_method) - _placeholder_variants(base_method)
            for param in extra:
                widened.setdefault((base_name, method_name), {}).setdefault(param, set()).add(name)

    findings: List[Finding] = []
    for (base_name, method_name), params in sorted(widened.items()):
        entry = index[base_name]
        if entry is None:
            continue
        rel, base_node = entry
        base_method = _methods(base_node)[method_name]
        src_lines = (root / rel).read_text(encoding="utf-8", errors="replace").splitlines()
        for param, subclasses in sorted(params.items()):
            if len(subclasses) < _MIN_SIBLINGS:
                continue
            findings.append(
                Finding(
                    check="override_signature_drift",
                    severity="P1",
                    file=rel,
                    line=base_method.lineno,
                    snippet=_line_text(src_lines, base_method.lineno),
                    detail=(
                        f"`{base_name}.{method_name}` does not accept `{param}`, which "
                        f"{sorted(subclasses)} all declare on their overrides. Every subclass that "
                        "inherits this implementation raises TypeError on a call the other "
                        "subclasses accept."
                    ),
                )
            )
    return findings

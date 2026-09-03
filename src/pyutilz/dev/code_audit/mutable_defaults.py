"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator, Optional

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _arg_names, _iter_py_files, _line_text, _read_src_lines, _safe_parse

# --- Class A: mutable defaults ------------------------------------------


_MUTATING_METHODS = frozenset({
    "append", "extend", "insert", "pop", "clear", "remove", "sort",
    "reverse", "setdefault", "update", "add", "discard", "popitem",
    "__setitem__", "__delitem__", "__iadd__", "__ior__",
})

_MUTABLE_LITERAL_TYPES = (ast.List, ast.Dict, ast.Set)
_MUTABLE_CALLS = frozenset({"list", "dict", "set"})
# collections factories that likewise build one shared mutable object at def time.
_MUTABLE_FACTORIES = {"defaultdict": "dict", "OrderedDict": "dict", "Counter": "dict", "deque": "list"}


def _is_mutable_default(default: ast.AST) -> Optional[str]:
    """Return a short label if ``default`` is a mutable literal/call,
    else None. ``label`` is one of ``"list"``, ``"dict"``, ``"set"``."""
    if isinstance(default, ast.List):
        return "list"
    if isinstance(default, ast.Dict):
        return "dict"
    if isinstance(default, ast.Set):
        return "set"
    if isinstance(default, ast.Call):
        # Arguments do NOT make the call safe: ``dict(a=1)`` builds a fresh mutable object once,
        # at def time, exactly like ``{}`` does.
        name = default.func.id if isinstance(default.func, ast.Name) else (default.func.attr if isinstance(default.func, ast.Attribute) else None)
        if name in _MUTABLE_CALLS:
            return name
        if name in _MUTABLE_FACTORIES:
            return _MUTABLE_FACTORIES[name]
    return None


def _shadows_param(node: ast.AST, param_name: str) -> bool:
    """True if a nested function/lambda redeclares ``param_name`` as its own parameter,
    shadowing the outer binding for its entire body."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return False
    args = node.args
    all_names = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
    if args.vararg is not None:
        all_names.add(args.vararg.arg)
    if args.kwarg is not None:
        all_names.add(args.kwarg.arg)
    return param_name in all_names


def _walk_skipping_shadowed_scopes(func: ast.AST, param_name: str) -> Iterator[ast.AST]:
    """Like ast.walk(func), but stops descending into nested function/lambda scopes that
    redeclare param_name as their own parameter (a shadow of the outer binding)."""
    todo = list(ast.iter_child_nodes(func))
    while todo:
        node = todo.pop()
        yield node
        if _shadows_param(node, param_name):
            continue
        todo.extend(ast.iter_child_nodes(node))


def _param_is_mutated(func: ast.FunctionDef | ast.AsyncFunctionDef, param_name: str) -> bool:
    """Heuristic: does the function body apply a mutating operation to
    the named parameter? Walks calls of shape ``param.append(...)``,
    subscript-assigns ``param[k] = v``, augmented assigns ``param += x``.
    Does not descend into nested functions/lambdas that shadow ``param_name`` with their
    own parameter of the same name -- those mutate an independently-scoped binding."""
    for node in _walk_skipping_shadowed_scopes(func, param_name):
        # param.<mutating_method>(...)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func
            if isinstance(attr.value, ast.Name) and attr.value.id == param_name and attr.attr in _MUTATING_METHODS:
                return True
        # param[k] = v  /  del param[k]
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.Delete)):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
            for t in targets:
                if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == param_name:
                    return True
                if isinstance(t, ast.Name) and t.id == param_name and isinstance(node, ast.AugAssign):
                    return True
    return False


_KNOWN_IMMUTABLE_SCALAR_NAMES = frozenset({"int", "float", "str", "bytes", "bool", "complex"})


def _param_annotations(func: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Optional[ast.expr]]:
    """param_name -> its annotation AST node (None if unannotated)."""
    a = func.args
    out: dict[str, Optional[ast.expr]] = {}
    for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs):
        out[arg.arg] = arg.annotation
    return out


def _param_default_looks_like_immutable_scalar(func: ast.FunctionDef | ast.AsyncFunctionDef, param_name: str) -> bool:
    """True if ``param_name`` is unannotated but its OWN default value is an int/float/str/bool/
    complex/None literal -- a caller overriding a ``title=""`` / ``count=0`` default with a
    completely different (mutable) type would be a pre-existing contract violation on the
    function's own terms, not something this scanner needs to defend against. Confirmed real
    false-positive shape: ``def f(title="", ...): title_line = title; title_line += f"..."`` --
    ``title_line`` can only ever hold a str (its own default proves the contract), so the
    ``+=`` always rebinds, never mutates in place, regardless of the missing annotation."""
    a = func.args
    positional = (*a.posonlyargs, *a.args)
    defaults_by_name: dict[str, ast.expr] = dict(zip((p.arg for p in positional[len(positional) - len(a.defaults) :]), a.defaults))
    for kw, kw_default in zip(a.kwonlyargs, a.kw_defaults):
        if kw_default is not None:  # a kwonly arg with no default has a None placeholder here
            defaults_by_name[kw.arg] = kw_default
    default = defaults_by_name.get(param_name)
    if default is None:
        return False
    # A `None` default (`items: list = None`) is the universal "optional/unset" sentinel and
    # says nothing about the real type -- only a CONCRETE scalar literal default is a genuine
    # type signal (a caller wouldn't override `count=0`/`title=""` with a container type).
    return isinstance(default, ast.Constant) and default.value is not None and isinstance(default.value, (int, float, str, bytes, bool, complex))


def _is_known_immutable_scalar_annotation(annotation: Optional[ast.expr]) -> bool:
    """True for a bare ``int``/``float``/``str``/``bytes``/``bool``/``complex``
    annotation, optionally ``X | None`` -- deliberately narrow (only these
    simple shapes) rather than attempting to resolve arbitrary ``Optional[...]``/
    subscripted/string-quoted annotations. ``+=``/``-=`` etc. on any of these
    types always rebinds (creates a new object via ``__add__``/``__sub__``,
    never ``__iadd__``/in-place mutation), so aliasing one is never the
    parameter-mutation-leak shape this scanner targets -- unlike a bare,
    unannotated name (where the type is unknown and the conservative default
    of flagging AugAssign stays in effect) or a container-typed one.
    """
    if annotation is None:
        return False
    if isinstance(annotation, ast.Name):
        return annotation.id in _KNOWN_IMMUTABLE_SCALAR_NAMES
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        return _is_known_immutable_scalar_annotation(annotation.left) or _is_known_immutable_scalar_annotation(annotation.right)
    # audit-2026-07-22 false-positive fix: `typing.Optional[X]` is a Subscript node, not the
    # `X | None` BinOp shape -- the SAME immutable-scalar guarantee applies to either spelling
    # (needed for Python < 3.10 compatibility, where `X | None` isn't valid at runtime without
    # `from __future__ import annotations`), so both must be recognized for this exemption to
    # actually cover typing.Optional-style code.
    if isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name) and annotation.value.id == "Optional":
        _slice = annotation.slice
        # Python < 3.9 wraps Subscript.slice in ast.Index (removed as a distinct node in 3.9,
        # where Subscript.slice holds the inner expression directly) -- unwrap it so
        # `Optional[str]` is recognized identically on every supported Python version.
        if isinstance(_slice, ast.Index):  # type: ignore[attr-defined]  # ast.Index only exists pre-3.9; guarded by isinstance
            _slice = _slice.value  # type: ignore[attr-defined]  # ast.Index only exists pre-3.9, so the checker sees no .value on 3.9+
        return _is_known_immutable_scalar_annotation(_slice)
    return False


_SHARED_BUFFER_SUFFIXES = ("_shared", "_buf", "_buffer", "_out", "_output")
_SHARED_BUFFER_NAMES = frozenset({"out", "buf", "buffer", "output", "dest", "destination"})


def _looks_like_a_shared_output_buffer(param_name: str) -> bool:
    """Whether a parameter's NAME declares it to be a caller-owned buffer the callee is meant to
    write into (``out``, ``dest``, ``scores_buf``, ``final_transformed_vals_shared``).

    For such a parameter, writing through the caller's object is the entire contract -- it is why
    the caller preallocated and passed it -- so the "silent cross-call corruption" this rule reports
    is neither silent nor corruption. Measured on a downstream repo (2026-09-03): 4 of the 11 P0
    findings were one deliberately reused preallocated column buffer whose slots are fully
    overwritten before use, and 3 more were the same ``out``-buffer shape in tests.
    """
    return param_name in _SHARED_BUFFER_NAMES or param_name.endswith(_SHARED_BUFFER_SUFFIXES)


def _branch_paths(func: ast.AST) -> dict[int, tuple[tuple[int, int], ...]]:
    """``{id(node) -> its branch path}``, a tuple of ``(id(If/Try), arm index)`` for each
    conditional arm the node sits inside.

    Two nodes whose paths disagree on the ARM of a shared ``If``/``Try`` can never both execute, so
    an alias established in one and a mutation performed in the other are not the same object's
    story. The pre-existing source-line ordering only defended against a REBIND in a sibling branch;
    it could still pair an alias in the ``if`` arm with a mutation in the ``else`` arm. Measured on
    a downstream repo (2026-09-03): ``if run_tags is None: run_tags = tags`` / ``else:
    run_tags.update(tags)`` was reported as a P0 caller-mutation leak, where the two lines are
    mutually exclusive by construction.
    """
    paths: dict[int, tuple[tuple[int, int], ...]] = {}

    def visit(node: ast.AST, path: tuple[tuple[int, int], ...]) -> None:
        """Walks the tree, tagging every statement with the branch arms it sits under."""
        paths[id(node)] = path
        arms: list[list] = []
        if isinstance(node, ast.If):
            visit(node.test, path)
            arms = [node.body, node.orelse]
        elif isinstance(node, ast.Try):
            arms = [node.body, *[h.body for h in node.handlers], node.orelse, node.finalbody]
        if arms:
            for index, arm in enumerate(arms):
                for stmt in arm:
                    visit(stmt, (*path, (id(node), index)))
            return
        for child in ast.iter_child_nodes(node):
            visit(child, path)

    visit(func, ())
    return paths


def _branches_are_exclusive(left: tuple[tuple[int, int], ...], right: tuple[tuple[int, int], ...]) -> bool:
    """Whether two branch paths sit in different arms of the same conditional."""
    for a, b in zip(left, right):
        if a == b:
            continue
        return a[0] == b[0]
    return False


def _augassign_target_is_a_scalar_counter(func: ast.AST, local_name: str, node: ast.AugAssign) -> bool:
    """Whether ``local += <numeric literal>`` is provably arithmetic on a NUMBER rather than an
    in-place mutation of a container the caller still holds.

    ``+=`` on an ``int``/``float`` rebinds the name (``x = x + 1``); on a list or a numpy array it
    mutates in place, so the numeric literal alone is not enough (``arr += 1`` is a real leak). The
    second half of the evidence is that the name is used as a SUBSCRIPT INDEX or as the operand of a
    comparison driving a ``while``/``if`` -- neither of which a numpy array can do (a truth-value
    test on an array raises). Together they are the ``i = l; while i < r: ... buf[i] ...; i += 1``
    hand-rolled loop counter, 3 of the 11 false P0s measured on a downstream repo (2026-09-03), all
    of them inside ``@njit`` kernels where that spelling is mandatory.
    """
    if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)) and not isinstance(node.value.value, bool)):
        return False
    for other in ast.walk(func):
        if isinstance(other, ast.Subscript):
            index = other.slice
            if isinstance(index, ast.Name) and index.id == local_name:
                return True
        elif isinstance(other, (ast.While, ast.If)) and isinstance(other.test, ast.Compare):
            operands = [other.test.left, *other.test.comparators]
            if any(isinstance(o, ast.Name) and o.id == local_name for o in operands):
                return True
        elif isinstance(other, ast.Call) and isinstance(other.func, ast.Name) and other.func.id == "range":
            if any(isinstance(a, ast.Name) and a.id == local_name for a in other.args):
                return True
    return False


def _find_param_aliasing_mutation(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[tuple[str, str, int]]:
    """Find ``local = param`` (bare Name-to-Name rebind, no ``.copy()``/``list()``/``dict()``/
    ``[*...]`` wrapping) followed later in the SAME function by an in-place mutation of ``local``
    (``+=``, ``.append``/``.extend``/``.update``/etc., or subscript-assign) -- the mutation then
    silently reaches the CALLER's object through ``param``, since no copy was ever made.

    Returns a list of ``(param_name, local_name, mutation_lineno)`` for each such site found.

    Nodes are processed in SOURCE-LINE order (not ``ast.walk``'s breadth-first order): mutually
    exclusive ``if``/``elif``/``else`` branches routinely reassign the same local name (e.g.
    ``res = obj`` in one branch, ``res = {}`` in a sibling branch, then ``res[key] = ...`` inside
    that sibling branch's own body) -- BFS order can pair a mutation from one branch against an
    alias established in a DIFFERENT, mutually-exclusive branch, a false positive found in the
    wild (``text/strings/jsonutils.py``) during the first real run of this scanner.

    An ``AugAssign`` (``+=``/``-=``/etc.) on a local aliasing a parameter
    annotated as a known-immutable scalar (``int``/``float``/``str``/
    ``bytes``/``bool``/``complex``, optionally ``| None``) is never flagged:
    unlike a mutable container's ``__iadd__``, augmented assignment on an
    immutable type always REBINDS the local name to a new object (``x = x -
    y``), never mutating the original -- confirmed false positive found in
    the wild (a `remaining -= slice_sec` loop over a `total: float`
    parameter) during the first real run of this scanner.
    """
    param_names = {name for name in _arg_names(func) if not _looks_like_a_shared_output_buffer(name)}
    param_annotations = _param_annotations(func)
    paths = _branch_paths(func)
    aliases: dict[str, str] = {}  # local_name -> param_name
    alias_paths: dict[str, tuple[tuple[int, int], ...]] = {}  # local_name -> the branch path the alias was established in
    hits: list[tuple[str, str, int]] = []
    nodes = sorted(ast.walk(func), key=lambda n: getattr(n, "lineno", 0))
    for node in nodes:
        node_path = paths.get(id(node), ())
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            local_name = node.targets[0].id
            if isinstance(node.value, ast.Name) and node.value.id in param_names:
                aliases[local_name] = node.value.id
                alias_paths[local_name] = node_path
                continue
            elif local_name in aliases:
                del aliases[local_name]  # reassigned to something else -- no longer a bare alias
                continue
        if not aliases:
            continue
        target_name = None
        lineno = 0
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id in aliases and node.func.attr in _MUTATING_METHODS:
                target_name = node.func.value.id
                lineno = node.lineno
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id in aliases:
            aliased_param = aliases[node.target.id]
            if (
                not _is_known_immutable_scalar_annotation(param_annotations.get(aliased_param))
                and not _param_default_looks_like_immutable_scalar(func, aliased_param)
                and not _augassign_target_is_a_scalar_counter(func, node.target.id, node)
            ):
                target_name = node.target.id
                lineno = node.lineno
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id in aliases:
                    target_name = t.value.id
                    lineno = node.lineno
        if target_name is not None and not _branches_are_exclusive(alias_paths.get(target_name, ()), node_path):
            hits.append((aliases[target_name], target_name, lineno))
    return hits


def scan_parameter_aliasing_mutation(root: Path,
                                     exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
                                     ) -> list[Finding]:
    """Find ``local = param`` (bare rebind, no ``.copy()``) followed by an in-place mutation of
    ``local`` -- the mutation silently reaches the caller's object through the un-copied
    parameter reference.

    Confirmed real bug in the 2026-07-21 audit: ``database/db/upsert.py``'s
    ``returning_fields = history_fields; returning_fields += [hash_field]`` used
    ``list.__iadd__`` (in-place) to mutate the CALLER's own ``history_fields`` list, leaking
    ``hash_field`` into a history-table column list the caller never asked for it in.

    Three shapes are deliberately NOT reported, each measured as a false P0 on a fresh repo
    (2026-09-03, 11 findings, 11 false): a hand-rolled scalar loop counter (``i = l`` ... ``i += 1``
    with ``i`` used as an index or a loop bound -- ``+=`` on a number rebinds, it does not mutate);
    an alias and a mutation sitting in mutually exclusive arms of the same ``if``/``try``; and a
    parameter whose NAME declares it a caller-owned output buffer (``out``, ``*_shared``,
    ``*_buffer``), where writing through to the caller is the contract rather than the leak.

    Severity: P0 (silent cross-call state corruption, same class as a mutable-default leak).
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = _read_src_lines(py)
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for param_name, local_name, lineno in _find_param_aliasing_mutation(node):
                findings.append(Finding(
                    check="parameter_aliasing_mutation",
                    severity="P0",
                    file=rel,
                    line=lineno,
                    snippet=_line_text(src_lines, lineno),
                    detail=(
                        f"`{local_name} = {param_name}` (bare alias, no `.copy()`) then mutated "
                        f"in place -- this silently mutates the CALLER's `{param_name}` object "
                        "too, since no copy was ever made."
                    ),
                ))
    return findings


def scan_mutable_defaults(root: Path,
                          exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
                          ) -> list[Finding]:
    """Find ``def f(x=[]) / ={} / =set() / =list() / =dict() / =set()`` defaults.

    Severity rule: parameter mutated in body -> P0 (state leaks across
    callers); not mutated -> Low (idiomatic-but-questionable, no
    observable bug today but reader trap and future-bug bait).
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = _read_src_lines(py)
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = node.args
            # Pair every positional default with its corresponding name.
            pos_args = (*args.posonlyargs, *args.args)
            n_defaults = len(args.defaults)
            if n_defaults:
                paired = list(zip(pos_args[-n_defaults:], args.defaults))
            else:
                paired = []
            paired.extend((k, d) for k, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None)
            for arg_node, default in paired:
                label = _is_mutable_default(default)
                if label is None:
                    continue
                mutated = _param_is_mutated(node, arg_node.arg)
                sev = "P0" if mutated else "Low"
                detail = f"def {node.name}(..., {arg_node.arg}={label}()): " + (
                    "MUTATED in body -> shared state leaks across callers" if mutated else "default is mutable but never mutated in body (Low)"
                )
                findings.append(Finding(
                    check="mutable_default",
                    severity=sev,
                    file=rel,
                    line=node.lineno,
                    snippet=_line_text(src_lines, node.lineno),
                    detail=detail,
                ))
    return findings

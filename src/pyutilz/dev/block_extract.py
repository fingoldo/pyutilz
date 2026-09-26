"""Extract a run of statements of a function into a helper function, with exact data flow.

:mod:`pyutilz.dev.freevar_analysis` answers "which names does this line range read from outside?". Lifting the range
into a helper needs more than that: which names it must hand BACK (assigned in the range, read later), whether it can be
lifted at all (an early ``return``, a ``break`` out of an enclosing loop), which module-level names the helper's module
must import, and - inside a loop - that the rest of the loop body both feeds the range and reads what it writes on the
next iteration. This module plans and applies such an extraction.

Name events are recorded in EVALUATION order, scope-aware:

* an assignment's value before its targets (``x = [... x ...]`` reads ``x`` first), ``x += 1`` as a read then a write,
  a ``for`` loop's iterable before its target;
* comprehensions, lambdas and nested ``def``s contribute only the names they read from the enclosing scope;
* ``except ... as e`` binds ``e`` for its handler only (Python deletes it afterwards), so neither the binding nor the
  reads of ``e`` inside that handler are data flow of the enclosing function.

A plan's ``inputs`` are the names the range reads before binding them that exist earlier in the function (parameters
included); ``outputs`` are the names it binds that the rest of the function reads. An output that already existed is
also an input (the range may bind it on some paths only); a NEW name bound only on some paths and read later cannot be
returned safely, so such a range is refused, as is one containing ``return``/``yield``/``await`` at its own level, a
``break``/``continue`` leaving it, or ``global``/``nonlocal``.

**State mode** (:func:`apply_extraction` ``state_names``): long-lived locals that many stages share can live on one
state object instead of being threaded through 20-40 parameters. Those names are rewritten to ``<state>.<name>`` in the
moved code at the TOKEN level, so comments and formatting survive (``ast.unparse`` would drop every comment).

Typical use::

    plan = plan_extraction("core.py", "fit", 120, 188)
    if not plan.problems:
        apply_extraction(plan, "_setup_inputs", "stages/_setup.py", "pkg.stages._setup", core_module="pkg.core")
"""

from __future__ import annotations

import ast
import builtins
import io
import re
import textwrap
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Union

# ``match`` statements exist from 3.10; on older interpreters there is no such node to meet.
_MATCH: "tuple[type, ...]" = (ast.Match,) if hasattr(ast, "Match") else ()


def _write(path: Path, text: str, *, newline: str) -> None:
    """``Path.write_text(..., newline=)`` without the 3.10+ keyword: the same bytes on every supported Python."""
    with open(path, "w", encoding="utf-8", newline=newline) as fh:
        fh.write(text)

__all__ = ["ExtractionPlan", "ExtractionResult", "rename_in_function", "name_events", "plan_extraction", "apply_extraction", "import_lines", "absolutise_relative_imports"]

_LOOPS = (ast.For, ast.AsyncFor, ast.While)
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


class _Events(ast.NodeVisitor):
    """Name loads/stores in evaluation order: ``(line, kind, name)`` with kind ``"load"``/``"store"``."""

    def __init__(self) -> None:
        """Start with no events."""
        self.events: list = []

    def _add(self, line: int, kind: str, name: str) -> None:
        """Record one event."""
        self.events.append((line, kind, name))

    def visit_Name(self, node: ast.Name) -> None:
        """A read or a write of a plain name."""
        self._add(node.lineno, "load" if isinstance(node.ctx, ast.Load) else "store", node.id)

    def _assign(self, node) -> None:
        """Assignment: the value (and annotation) is evaluated before the targets are bound."""
        if getattr(node, "value", None) is not None:
            self.visit(node.value)
        if isinstance(node, ast.AnnAssign) and node.annotation is not None:
            self.visit(node.annotation)
        for t in node.targets if isinstance(node, ast.Assign) else [node.target]:
            self.visit(t)

    visit_Assign = visit_AnnAssign = _assign  # noqa: N815 - ast.NodeVisitor dispatch names

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        """Walrus: value first, then the binding."""
        self.visit(node.value)
        self.visit(node.target)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """``x += v``: ``v``, then a read and a write of ``x``."""
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self._add(node.lineno, "load", node.target.id)
            self._add(node.lineno, "store", node.target.id)
        else:
            self.visit(node.target)

    def _for(self, node) -> None:
        """Loop: the iterable before the target, then the body and ``else``."""
        self.visit(node.iter)
        self.visit(node.target)
        for st in node.body + node.orelse:
            self.visit(st)

    visit_For = visit_AsyncFor = _for  # noqa: N815

    def visit_With(self, node) -> None:
        """Each context expression before its ``as`` target, then the body."""
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self.visit(item.optional_vars)
        for st in node.body:
            self.visit(st)

    visit_AsyncWith = visit_With  # noqa: N815

    def _free_of(self, nodes: Iterable[ast.AST], bound: set) -> None:
        """Record the loads of ``nodes`` (a nested scope) that are not bound inside it."""
        inner = _Events()
        for n in nodes:
            inner.visit(n)
        local = set(bound) | {name for _, k, name in inner.events if k == "store"}
        for line, k, name in inner.events:
            if k == "load" and name not in local:
                self._add(line, "load", name)

    @staticmethod
    def _params(args: ast.arguments) -> set:
        """Every parameter name of a function or lambda signature."""
        every = [*args.posonlyargs, *args.args, *args.kwonlyargs, *([args.vararg] if args.vararg else []), *([args.kwarg] if args.kwarg else [])]
        return {a.arg for a in every}

    def visit_FunctionDef(self, node) -> None:
        """Nested def: decorators and defaults are read here, the name is bound, the body contributes only its free reads."""
        for d in node.decorator_list + node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
            self.visit(d)
        self._add(node.lineno, "store", node.name)
        self._free_of(node.body, self._params(node.args))

    visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Nested class: decorators, bases and keywords are read here, the name is bound, the body contributes its free reads."""
        for d in node.decorator_list + node.bases + [k.value for k in node.keywords]:
            self.visit(d)
        self._add(node.lineno, "store", node.name)
        self._free_of(node.body, set())

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Lambda: defaults are read here, the body contributes its free reads."""
        for d in node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
            self.visit(d)
        self._free_of([node.body], self._params(node.args))

    def _comprehension(self, node) -> None:
        """Comprehension: the first iterable is read in the enclosing scope, the rest only through its free reads."""
        self.visit(node.generators[0].iter)  # evaluated in the enclosing scope
        inner_nodes: list = []
        for i, gen in enumerate(node.generators):
            inner_nodes.append(gen.target)
            if i:
                inner_nodes.append(gen.iter)
            inner_nodes.extend(gen.ifs)
        inner_nodes.extend(getattr(node, a) for a in ("elt", "key", "value") if hasattr(node, a))
        self._free_of(inner_nodes, set())

    visit_ListComp = visit_SetComp = visit_GeneratorExp = visit_DictComp = _comprehension  # noqa: N815

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """Handler: its ``as`` name and the reads of it belong to the handler, not to the enclosing function."""
        if node.type is not None:
            self.visit(node.type)
        inner = _Events()
        for st in node.body:
            inner.visit(st)
        for ev in inner.events:
            if node.name and ev[2] == node.name:
                continue
            self.events.append(ev)

    def visit_Import(self, node) -> None:
        """An import binds its (first-component or ``as``) name."""
        for alias in node.names:
            self._add(node.lineno, "store", (alias.asname or alias.name).split(".")[0])

    visit_ImportFrom = visit_Import  # noqa: N815

    def visit_Delete(self, node: ast.Delete) -> None:
        """``del name`` counts as a write (the binding changes)."""
        for t in node.targets:
            if isinstance(t, ast.Name):
                self._add(node.lineno, "store", t.id)
            else:
                self.visit(t)

    def visit_Global(self, node) -> None:
        """``global``/``nonlocal`` bind nothing themselves (a block containing one is refused elsewhere)."""
        pass

    visit_Nonlocal = visit_Global  # noqa: N815


def name_events(nodes: Iterable[ast.AST]) -> list:
    """Evaluation-ordered, scope-aware ``(line, kind, name)`` events of ``nodes`` (see the module docstring)."""
    v = _Events()
    for n in nodes:
        v.visit(n)
    return v.events


@dataclass
class ExtractionResult:
    """What :func:`apply_extraction` wrote: the call that replaced the statements, and the helper's source."""

    call: str
    helper_source: str


@dataclass
class ExtractionPlan:
    """What lifting ``[start, end]`` of ``func`` into a helper needs; ``problems`` non-empty means it cannot be lifted."""

    path: Path
    func: str
    start: int
    end: int
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    module_names: list = field(default_factory=list)
    lazy_imports: list = field(default_factory=list)
    problems: list = field(default_factory=list)
    in_loop: bool = False


def _find_function(tree: ast.Module, func: str):
    """The (first) function or async function named ``func`` anywhere in ``tree``."""
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == func:
            return n
    raise ValueError(f"no function {func!r}")


def _container(fn, start: int, end: int):
    """The statement list holding exactly the statements from ``start`` to ``end``, and whether a loop encloses it."""
    found: list = []

    def walk(node, in_loop: bool) -> None:
        """Find the body whose overlapped statements all lie inside the range; track whether a loop encloses it."""
        for fld in ("body", "orelse", "finalbody"):
            stmts = getattr(node, fld, None)
            if isinstance(stmts, list) and stmts and isinstance(stmts[0], ast.stmt):
                # the range may start/end on comment lines: every statement it overlaps must lie fully inside it
                overlap = [s for s in stmts if s.lineno <= end and s.end_lineno >= start]
                if overlap and all(start <= s.lineno and s.end_lineno <= end for s in overlap):
                    found.append((stmts, in_loop))
                    return
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _SCOPES) and child is not fn:
                continue
            if not found:
                walk(child, in_loop or (isinstance(child, _LOOPS) and child is not fn))

    walk(fn, False)
    return found[0] if found else (None, False)


def _control_problems(block: list) -> list:
    """Why ``block`` cannot become a helper: an early return/yield/await, a break/continue leaving it, global/nonlocal."""
    problems: list = []

    def walk(node, in_loop: bool) -> None:
        """Record control-flow problems below ``node``; ``in_loop``: a loop inside the block encloses it."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _SCOPES):
                continue
            if isinstance(child, (ast.Return, ast.Yield, ast.YieldFrom, ast.Await)):
                problems.append(f"{type(child).__name__} at line {child.lineno}")
            if isinstance(child, (ast.Break, ast.Continue)) and not in_loop:
                problems.append(f"{type(child).__name__} at line {child.lineno} leaves the block")
            if isinstance(child, (ast.Global, ast.Nonlocal)):
                problems.append(f"{type(child).__name__} at line {child.lineno}")
            walk(child, in_loop or isinstance(child, _LOOPS))

    for st in block:
        if isinstance(st, _SCOPES):  # a nested def/class moves whole; its own returns are its business
            continue
        if isinstance(st, ast.Return):
            problems.append(f"Return at line {st.lineno}")
        walk(st, isinstance(st, _LOOPS))
    return problems


def _definitely_bound(stmts: list) -> set:
    """Names every path through ``stmts`` binds: plain statements, both arms of an if/else, a with-body."""
    out: set = set()
    for st in stmts:
        if isinstance(st, ast.If):
            out |= _definitely_bound(st.body) & _definitely_bound(st.orelse) if st.orelse else set()
        elif isinstance(st, (ast.With, ast.AsyncWith)):
            out |= _definitely_bound(st.body)
        elif isinstance(st, (ast.For, ast.AsyncFor, ast.While, ast.Try, *_MATCH)):
            continue
        else:
            out |= {n for _, k, n in name_events([st]) if k == "store"}
    return out


def _never_reads_prior_binding(later: list, name: str) -> bool:
    """True when the statements after the block cannot read the value the block bound to ``name``: a top-level statement
    rebinds it before any read, or every read of it sits inside a ``for`` loop whose own target rebinds it."""
    for st in later:
        mentions = [e for e in name_events([st]) if e[2] == name]
        if not mentions:
            continue
        if not isinstance(st, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith, *_MATCH)) and mentions[0][1] == "store":
            return True  # rebound unconditionally before this statement reads it
        break
    rebinding_loops = [n for st in later for n in ast.walk(st) if isinstance(n, (ast.For, ast.AsyncFor))
                       and any(isinstance(t, ast.Name) and t.id == name for t in ast.walk(n.target))]
    # the body only: a loop's ``else`` also runs after zero iterations, when the name still holds the block's value
    inside = {id(x) for loop in rebinding_loops for body_st in loop.body for x in ast.walk(body_st)}
    reads = [x for st in later for x in ast.walk(st) if isinstance(x, ast.Name) and x.id == name and isinstance(x.ctx, ast.Load)]
    if not reads:
        return True
    return all(id(r) in inside for r in reads)


def _module_names(tree: ast.Module) -> set:
    """Every name bound at the top level of ``tree`` (assignments, imports, defs, classes)."""
    names: set = set()
    for node in tree.body:
        names |= {n for _, k, n in name_events([node]) if k == "store"}
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def plan_extraction(path: Union[str, Path], func: str, start: int, end: int) -> ExtractionPlan:
    """Plan lifting the statements of ``func`` spanning lines ``[start, end]`` (one body, whole statements) into a helper."""
    path = Path(path)
    src_text = path.read_text(encoding="utf-8")
    tree = ast.parse(src_text, filename=str(path))
    fn = _find_function(tree, func)
    stmts, in_loop = _container(fn, start, end)
    plan = ExtractionPlan(path=path, func=func, start=start, end=end, in_loop=in_loop)
    if stmts is None:
        plan.problems.append(f"lines {start}-{end} are not whole statements of one body of {func}")
        return plan
    block = [s for s in stmts if s.lineno >= start and s.end_lineno <= end]
    params = _Events._params(fn.args)
    events = name_events(fn.body)
    block_ev = name_events(block)
    outside = [e for e in events if not (start <= e[0] <= end)]
    before = outside if in_loop else [e for e in outside if e[0] < start]
    after = outside if in_loop else [e for e in outside if e[0] > end]
    defined_before = params | {n for _, k, n in before if k == "store"}
    bound: set = set()
    for _, kind, name in block_ev:
        if kind == "load" and name not in bound and name in defined_before and name not in plan.inputs:
            plan.inputs.append(name)
        if kind == "store":
            bound.add(name)
    read_after = {n for _, k, n in after if k == "load"}
    plan.outputs = sorted(n for n in bound if n in read_after)
    for n in plan.outputs:
        if n in defined_before and n not in plan.inputs:
            plan.inputs.append(n)
    # a NEW name bound only on some paths cannot be returned safely - unless the rest of the function never reads the
    # block's binding (it rebinds first, or only reads it inside loops that rebind it)
    top = _definitely_bound(block)
    later = [s for s in fn.body if s.lineno > end] if not in_loop else []
    unread = {n for n in plan.outputs if n not in defined_before and n not in top and later and _never_reads_prior_binding(later, n)}
    plan.outputs = [n for n in plan.outputs if n not in unread]
    risky = [n for n in plan.outputs if n not in defined_before and n not in top]
    if risky:
        plan.problems.append(f"new names bound only on some paths and read later: {risky}")
    plan.problems += _control_problems(block)
    module = _module_names(tree)
    fn_bound = {n for _, k, n in events if k == "store"} | params
    loads = {n for _, k, n in block_ev if k == "load"}
    plan.module_names = sorted(n for n in loads - bound - set(plan.inputs) if n in module and not hasattr(builtins, n))
    # An input the function binds ONLY through a function-level (lazy) import is re-imported inside the helper, lazily,
    # instead of becoming a parameter: the lazy import usually exists to break a cycle, and a helper should get its own.
    imports: dict = {}
    other_binding: set = set()
    guarded: set = set()  # imports inside ``try:`` (optional dependencies) stay parameters: re-importing them unguarded could raise
    for node in ast.walk(fn):
        if isinstance(node, ast.Try):
            for sub in node.body:
                guarded |= {id(n) for n in ast.walk(sub) if isinstance(n, (ast.Import, ast.ImportFrom))}
    for st in ast.walk(fn):
        if isinstance(st, (ast.Import, ast.ImportFrom)) and id(st) not in guarded:
            for alias in st.names:
                imports.setdefault((alias.asname or alias.name).split(".")[0], ast.get_source_segment(src_text, st))
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            other_binding.add(node.id)
    for n in list(plan.inputs):
        if n in imports and n not in other_binding and n not in params:
            plan.inputs.remove(n)
            if imports[n] not in plan.lazy_imports:
                plan.lazy_imports.append(imports[n])
    unknown = sorted(n for n in loads - bound - set(plan.inputs) if n in fn_bound and n not in module and n not in imports)
    if unknown:
        plan.problems.append(f"reads names of {func} not available at the block: {unknown}")
    return plan


def absolutise_relative_imports(text: str, module: str) -> str:
    """Rewrite ``from .x import y`` in code taken from ``module`` to the absolute module it meant there."""
    pkg = module.split(".")[:-1]

    def fix(m: "re.Match") -> str:
        """The absolute form of one relative ``from ... import``."""
        dots, mod = m.group(2), m.group(3) or ""
        parts = pkg[: len(pkg) - (len(dots) - 1)]
        return str(m.group(1)) + "from " + ".".join(parts + ([mod] if mod else [])) + " import"

    return re.sub(r"^(\s*)from (\.+)([\w.]*)? import", fix, text, flags=re.M)


def import_lines(path: Union[str, Path], module: str, names: Iterable[str]) -> list:
    """One import line per name, from where ``module`` (the file at ``path``) gets it: the same import statement made
    absolute for an imported name, ``from module import name`` for a name the module defines itself."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    pkg = module.split(".")[:-1]
    origin: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                bound = a.asname or a.name.split(".")[0]
                origin[bound] = f"import {a.name} as {a.asname}" if a.asname else f"import {a.name.split('.')[0]}"
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = ".".join(pkg[: len(pkg) - (node.level - 1)] + ([node.module] if node.module else []))
            for a in node.names:
                origin[a.asname or a.name] = f"from {base} import {a.name}" + (f" as {a.asname}" if a.asname else "")
    return [origin.get(n, f"from {module} import {n}") for n in names]


def rename_in_function(path: Union[str, Path], func: str, mapping: dict) -> int:
    """Rewrite every use of the local names in ``mapping`` inside ``func`` to the mapped expression (e.g.
    ``{"_hybrid_orth_pre_recipes": "recipes.hybrid_orth"}``), at token level so comments and layout survive.

    Refuses (``ValueError``) when a name is a parameter of ``func``, is read by a nested scope (a closure would still
    see the old name), appears inside an f-string, or ``func`` calls ``locals``/``vars``/``eval``/``exec``. After the
    rewrite the function is re-parsed and must contain no ``Name`` node for any mapped name. Returns the number of
    occurrences rewritten.
    """
    path = Path(path)
    raw = path.read_bytes().decode("utf-8")
    nl = "\r\n" if "\r\n" in raw else "\n"
    src = raw.replace("\r\n", "\n")
    fn = _find_function(ast.parse(src), func)
    names = set(mapping)
    if names & _Events._params(fn.args):
        raise ValueError(f"parameters cannot be renamed: {sorted(names & _Events._params(fn.args))}")
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("locals", "vars", "eval", "exec"):
            raise ValueError(f"{func} calls {node.func.id}() at line {node.lineno}; renaming locals would change what it sees")
        if isinstance(node, ast.JoinedStr) and any(isinstance(s, ast.Name) and s.id in names for s in ast.walk(node)):
            raise ValueError(f"a renamed name is used inside an f-string at line {node.lineno}")
        if isinstance(node, _SCOPES) and node is not fn and any(isinstance(s, ast.Name) and s.id in names for s in ast.walk(node)):
            raise ValueError(f"a renamed name is used by a nested scope at line {node.lineno}")
    lines = src.splitlines(keepends=True)
    start, end = fn.body[0].lineno, fn.end_lineno
    body = "".join(lines[start - 1 : end])
    edits = _name_token_positions(body, names)
    for row, col, name in sorted(edits, reverse=True):
        ln = lines[start - 1 + row - 1]
        lines[start - 1 + row - 1] = ln[:col] + mapping[name] + ln[col + len(name) :]
    new_src = "".join(lines)
    fn2 = _find_function(ast.parse(new_src), func)
    left = sorted({s.id for s in ast.walk(fn2) if isinstance(s, ast.Name) and s.id in names})
    if left:
        raise ValueError(f"rename left uses of {left}")
    _write(path, new_src.replace("\n", nl), newline="")
    return len(edits)


def _name_token_positions(text: str, names: set) -> list:
    """``(row, col, name)`` of every NAME token in ``text`` that is one of ``names`` used as a variable (not an attribute
    ``obj.name`` and not a keyword argument ``f(name=...)``)."""
    toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    out = []
    depth = 0
    skip = (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT)
    for i, tok in enumerate(toks):
        if tok.type == tokenize.OP and tok.string in "([{":
            depth += 1
        elif tok.type == tokenize.OP and tok.string in ")]}":
            depth -= 1
        if tok.type != tokenize.NAME or tok.string not in names:
            continue
        prev = next((t for t in reversed(toks[:i]) if t.type not in skip), None)
        nxt = next((t for t in toks[i + 1 :] if t.type not in skip), None)
        if prev is not None and prev.string == ".":
            continue
        if depth > 0 and nxt is not None and nxt.string == "=" and prev is not None and prev.string in ("(", ","):
            continue
        out.append((tok.start[0], tok.start[1], tok.string))
    return out


def _to_state(text: str, names: set, state: str) -> str:
    """``name`` -> ``state.name`` for the given names, token by token (comments and layout untouched). Attribute names
    (``obj.name``) and keyword-argument names (``f(name=...)``) are left alone."""
    lines = text.splitlines(keepends=True)
    for row, col, _name in sorted(_name_token_positions(text, names), reverse=True):
        lines[row - 1] = lines[row - 1][:col] + f"{state}." + lines[row - 1][col:]
    return "".join(lines)


def apply_extraction(
    plan: ExtractionPlan,
    helper: str,
    target_path: Union[str, Path],
    target_module: str,
    *,
    core_module: str,
    state_names: Optional[Iterable[str]] = None,
    state_param: str = "st",
    doc: Optional[str] = None,
) -> "ExtractionResult":
    """Move the planned statements into ``helper`` at the end of ``target_path`` and call it in their place.

    ``state_names``: plan inputs/outputs that live on a state object passed as ``state_param`` (read and written as
    ``st.name`` in the helper, neither passed nor returned); the call site passes the state object under the same name.
    Returns the call inserted and the helper's source. Raises ``ValueError`` on a plan with problems.
    """
    if plan.problems:
        raise ValueError("; ".join(plan.problems))
    src = plan.path.read_text(encoding="utf-8")
    raw = plan.path.read_bytes().decode("utf-8")
    nl = "\r\n" if "\r\n" in raw else "\n"
    lines = src.splitlines(keepends=True)
    text = absolutise_relative_imports("".join(lines[plan.start - 1 : plan.end]), core_module)
    body = textwrap.dedent(text)
    if plan.lazy_imports:
        lazy = "".join(absolutise_relative_imports(imp, core_module) + chr(10) for imp in plan.lazy_imports)
        body = lazy + chr(10) + body
    state = set(state_names or ())
    ins = [n for n in plan.inputs if n not in state]
    ins = [n for n in ("self", "cls") if n in ins] + [n for n in ins if n not in ("self", "cls")]  # the receiver reads first
    outs = [n for n in plan.outputs if n not in state]
    uses_state = bool(state & (set(plan.inputs) | set(plan.outputs) | {n for _, k, n in name_events(ast.parse(body).body) if k == "store"}))
    if uses_state:
        body = _to_state(body, state, state_param)
    params = ([state_param] if uses_state else []) + ins
    doc = doc if doc is not None else f"Stage of ``{plan.func}``."
    ret = ("    return " + ", ".join(outs) + "\n") if outs else ""
    helper_src = f"\n\ndef {helper}({', '.join(params)}):\n    \"\"\"{doc}\"\"\"\n" + textwrap.indent(body, "    ") + ret
    target_path = Path(target_path)
    begin, finish = "# --- imports (managed) ---\n", "# --- end imports ---\n"
    if target_path.exists():
        tgt = target_path.read_text(encoding="utf-8")
    else:
        tgt = f'"""Stages of ``{plan.func}``."""\n\nfrom __future__ import annotations\n\n{begin}{finish}'
    i, j = tgt.index(begin) + len(begin), tgt.index(finish)
    have = {ln for ln in tgt[i:j].split("\n") if ln.strip()} | set(import_lines(plan.path, core_module, plan.module_names))
    ordered = sorted(have, key=lambda s: (not s.startswith("import "), s))
    tgt = tgt[:i] + "".join(ln + "\n" for ln in ordered) + tgt[j:]
    _write(target_path, tgt.rstrip("\n") + "\n" + helper_src, newline="\n")
    first = lines[plan.start - 1]
    indent = first[: len(first) - len(first.lstrip())]
    call = f"{helper}({', '.join(params)})"
    if outs:
        call = ", ".join(outs) + " = " + call
    new = [*lines[: plan.start - 1], f"{indent}from {target_module} import {helper}\n", indent + call + "\n", *lines[plan.end :]]
    _write(plan.path, "".join(new).replace("\r\n", "\n").replace("\n", nl), newline="")
    return ExtractionResult(call=indent + call, helper_source=helper_src.lstrip())

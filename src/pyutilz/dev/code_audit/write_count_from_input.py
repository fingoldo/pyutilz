"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Union

from ._base import (
    Finding,
    _DEFAULT_EXCLUDE_DIRS,
    _iter_py_files,
    _line_text,
    _module_sql_constants,
    _read_src_lines,
    _safe_parse,
    _sql_text,
    _string_constants_in_body,
)

# --- a written-rows counter taken from the rows given to an insert that can skip some -----------

_FuncDef = Union[ast.FunctionDef, ast.AsyncFunctionDef]

# An insert that can write fewer rows than it is handed: ON CONFLICT DO NOTHING, ON CONFLICT DO UPDATE ... WHERE
# (a row failing the WHERE is left alone), and the SQLite and MySQL ignore forms. DO UPDATE with no WHERE writes
# every row it is given, so it is deliberately absent.
_SKIPPING_INSERT = re.compile(
    r"\bON\s+CONFLICT\b[^;]*?\bDO\s+NOTHING\b" r"|\bON\s+CONFLICT\b[^;]*?\bDO\s+UPDATE\b[^;]*?\bWHERE\b" r"|\bINSERT\s+OR\s+IGNORE\b|\bINSERT\s+IGNORE\b",
    re.IGNORECASE | re.DOTALL,
)
# A counter is named for what it counts: the assigned name, attribute, or subscripted container.
_COUNTER_WORDS = ("stat", "count", "imported", "inserted", "written", "saved", "loaded", "upserted", "total")
# ...unless its name or key says it counts the input, which is then exactly what len() gives.
_INPUT_WORDS = ("read", "seen", "found", "collected", "parsed", "scanned", "candidate", "received", "input")


def _docless(body: list[ast.stmt]) -> list[ast.stmt]:
    """*body* without a leading docstring, which may describe an insert without running one."""
    first = body[0] if body else None
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        return body[1:]
    return body


def _is_len(node: ast.AST) -> bool:
    """True for a ``len(...)`` call."""
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "len"


def _reads(node: ast.AST, names: set[str]) -> bool:
    """True if *node* reads one of *names* other than through ``len()``: the rows themselves, not their count."""
    if _is_len(node):
        return False
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in names:
        return True
    return any(_reads(child, names) for child in ast.iter_child_nodes(node))


def _derived(stmts: list[ast.stmt], names: set[str]) -> set[str]:
    """*names* plus every name bound from an expression reading one of them (``batch = rows[i:i + n]``,
    ``for row in rows``), to a fixed point."""
    bindings: list[tuple[str, ast.AST]] = []
    for stmt in stmts:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Assign):
                bindings += [(t.id, node.value) for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)) and isinstance(node.target, ast.Name):
                bindings.append((node.target.id, node.iter))
    out = set(names)
    grew = True
    while grew:
        grew = False
        for name, value in bindings:
            if name not in out and _reads(value, out):
                out.add(name)
                grew = True
    return out


def _receiving_calls(stmts: list[ast.stmt], names: set[str]) -> list[ast.Call]:
    """Calls, other than ``len()``, handed one of *names* as an argument."""
    return [
        node
        for stmt in stmts
        for node in ast.walk(stmt)
        if isinstance(node, ast.Call) and not _is_len(node) and any(_reads(arg, names) for arg in [*node.args, *(k.value for k in node.keywords)])
    ]


def _skips_rows(node: ast.AST, constants: dict[str, str]) -> bool:
    """True if *node* carries the SQL of an insert that can write fewer rows than it is handed."""
    for sub in ast.walk(node):
        if isinstance(sub, (ast.Constant, ast.JoinedStr, ast.Name, ast.Attribute)):
            text = _sql_text(sub, constants)
            if text and _SKIPPING_INSERT.search(text):
                return True
    return False


def _helper_of(call: ast.Call, helpers: dict[str, _FuncDef]) -> _FuncDef | None:
    """The same-module function a call reaches: ``helper(...)``, or ``self.helper(...)`` / ``cls.helper(...)``."""
    func = call.func
    if isinstance(func, ast.Name):
        return helpers.get(func.id)
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in ("self", "cls"):
        return helpers.get(func.attr)
    return None


def _writes_lossily(
    stmts: list[ast.stmt],
    names: set[str],
    constants: dict[str, str],
    helpers: dict[str, _FuncDef],
    hops: int,
) -> bool:
    """True if a call handed rows from *names* is an insert that can skip some, directly or *hops* helpers down."""
    names = _derived(stmts, names)
    for call in _receiving_calls(stmts, names):
        if _skips_rows(call, constants):
            return True
        helper = _helper_of(call, helpers) if hops else None
        if helper is None:
            continue
        params = [a.arg for a in [*helper.args.posonlyargs, *helper.args.args]]
        if isinstance(call.func, ast.Attribute) and params[:1] in (["self"], ["cls"]):
            params = params[1:]
        bound = {params[i] for i, arg in enumerate(call.args) if i < len(params) and _reads(arg, names)}
        bound |= {k.arg for k in call.keywords if k.arg and _reads(k.value, names)}
        body = _docless(helper.body)
        if bound and _writes_lossily(body, bound, {**constants, **_string_constants_in_body(body)}, helpers, hops - 1):
            return True
    return False


def _counter_label(target: ast.expr) -> str:
    """What a counter assignment writes to, lower-cased: ``stats`` plus its key for ``self.stats["x"]``."""
    key = ""
    if isinstance(target, ast.Subscript):
        index = target.slice
        if isinstance(index, ast.Constant) and isinstance(index.value, str):
            key = index.value
        elif isinstance(index, ast.JoinedStr):
            key = "".join(v.value for v in index.values if isinstance(v, ast.Constant) and isinstance(v.value, str))
        target = target.value
    name = target.id if isinstance(target, ast.Name) else target.attr if isinstance(target, ast.Attribute) else ""
    return f"{name} {key}".lower()


def _is_write_counter(target: ast.expr) -> bool:
    """True if *target* is named as a count of rows written rather than rows read."""
    label = _counter_label(target)
    if any(w in label for w in _INPUT_WORDS):
        return False
    if any(w in label for w in _COUNTER_WORDS):
        return True
    words = label.split()
    return bool(words) and words[0].startswith("n_")


def _len_calls(value: ast.expr) -> list[ast.Call]:
    """The ``len()`` calls *value* is made of, reached through arithmetic only.

    ``len(rows)`` and ``done + len(rows)`` are counts of the rows. ``await self._record_written(k, t, len(rows), before)``
    is not: the assigned number comes from the call, and ``len`` is one of the things it was told. So the walk stops at
    any other call rather than descending into its arguments.
    """
    if _is_len(value):
        return [value] if isinstance(value, ast.Call) else []
    if isinstance(value, ast.BinOp):
        return _len_calls(value.left) + _len_calls(value.right)
    if isinstance(value, ast.UnaryOp):
        return _len_calls(value.operand)
    if isinstance(value, (ast.Await, ast.Starred)):
        return _len_calls(value.value)
    return []


def _measured_names(value: ast.expr) -> list[str]:
    """The name *value* is the length of, when it is one length and nothing else.

    Two lengths mean a comparison rather than a count of what was written: glossum's
    ``stats["not_in_oewn"] = len(candidates) - len(existing)`` is how many candidates the table lacked, which is
    true whatever the insert wrote.
    """
    calls = _len_calls(value)
    if len(calls) != 1:
        return []
    args = calls[0].args
    return [args[0].id] if len(args) == 1 and isinstance(args[0], ast.Name) else []


def _counter_assignments(stmts: list[ast.stmt]) -> list[tuple[ast.stmt, ast.expr, str]]:
    """``(statement, target, measured name)`` for each ``counter = len(x)`` / ``counter += len(x)``."""
    out: list[tuple[ast.stmt, ast.expr, str]] = []
    targets: list[ast.expr]
    for stmt in stmts:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add):
                targets, value = [node.target], node.value
            else:
                continue
            for target in targets:
                if _is_write_counter(target):
                    out += [(node, target, name) for name in _measured_names(value)]
    return out


def _module_functions(tree: ast.Module) -> dict[str, _FuncDef]:
    """Module functions and class methods by name, for the one-hop helper follow. The first definition wins."""
    out: dict[str, _FuncDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.setdefault(node.name, node)
    return out


def scan_write_counted_from_input(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find a rows-written counter set from ``len()`` of the rows handed to an insert that can skip some.

    ``self.stats["synsets"] = len(synsets)`` after ``INSERT ... ON CONFLICT (id) DO NOTHING`` reports every row
    the stage was given, and a re-run over data already imported reports the whole corpus again while writing
    nothing. The report and the database disagree and nothing says so. Real case (glossum 2026-09-01 audit,
    08-M15): every stage of the OEWN importer counted this way.

    The population is an assignment or ``+=`` to a counter (a name, attribute or subscripted container named
    for a count: ``stats``, ``count``, ``imported``, ``inserted``, ``written``, ``saved``, ``total``, ``n_*``...)
    whose value holds ``len(x)``, where ``x``, or a name bound from it (a slice, a loop variable, a
    comprehension), is handed to a call whose own arguments carry skipping SQL: ``ON CONFLICT DO NOTHING``,
    ``ON CONFLICT DO UPDATE ... WHERE``, ``INSERT OR IGNORE`` or ``INSERT IGNORE``. The SQL may be a literal,
    an f-string, a local or module constant, or a clause passed to a helper. When the call is a function of the
    same module, the rows are followed one hop into it by parameter.

    Not reported: ``DO UPDATE`` with no WHERE (every row is written), a counter whose name or key says it counts
    the input (``..._read``, ``seen``, ``found``, ``collected``...), and a count taken from ``rowcount`` or a
    RETURNING result. Known blind spot: rows copied into a staging table and then moved with
    ``INSERT ... SELECT ... ON CONFLICT DO NOTHING``, since the rows never meet the skipping statement.

    Severity: P2 -- a wrong progress figure, not lost data; but it hides a re-run that did nothing.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        try:
            raw = py.read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            continue
        if "len(" not in raw or not ("conflict" in raw or "ignore" in raw):
            continue
        tree = _safe_parse(py)
        if tree is None:
            continue
        module_constants = _module_sql_constants(tree)
        helpers = _module_functions(tree)
        # Read once per file rather than lazily: _read_src_lines returns an empty list on a read failure, not
        # None, so a "have I read it yet" guard could not tell the two apart.
        src_lines = _read_src_lines(py)
        rel = py.relative_to(root).as_posix()
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = _docless(fn.body)
            # Nested functions are their own scope; each is visited by the outer walk.
            own = [s for s in body if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
            assignments = _counter_assignments(own)
            if not assignments:
                continue
            constants = {**module_constants, **_string_constants_in_body(body)}
            for stmt, target, name in assignments:
                if not _writes_lossily(own, {name}, constants, helpers, hops=1):
                    continue
                findings.append(Finding(
                    check="write_counted_from_input",
                    severity="P2",
                    file=rel,
                    line=stmt.lineno,
                    snippet=_line_text(src_lines, stmt.lineno),
                    detail=(
                        f"`{ast.unparse(target)}` counts len({name}), the rows handed to an insert that can skip "
                        f"rows (ON CONFLICT DO NOTHING / DO UPDATE ... WHERE / INSERT IGNORE); a re-run reports them "
                        f"all again while writing none. Count what the statement wrote (rowcount, RETURNING, or the "
                        f"table before and after), or name the counter for what it holds (`..._read`)."
                    ),
                ))
    return findings

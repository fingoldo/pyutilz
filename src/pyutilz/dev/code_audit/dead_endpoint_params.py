"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _line_text, _read_src_lines, _safe_parse

# --- an HTTP endpoint parameter the handler never reads ------------------------------------------

# Decorator attributes that register a FastAPI / Starlette-style route: `@router.get(...)`, `@app.post(...)`.
_ROUTE_METHODS = frozenset({"get", "post", "put", "delete", "patch", "head", "options", "trace", "api_route", "websocket"})
# Parameter sources that bind a value from the request. `Depends` / `Security` are deliberately absent: a
# dependency is often wanted only for its side effect (an auth check, a session), so an unused one is not dead.
_REQUEST_SOURCES = frozenset({"Query", "Path", "Header", "Cookie", "Body", "Form", "File"})


def _call_name(node: ast.expr) -> str:
    """The trailing identifier of a call's function (``fastapi.Query(...)`` -> ``Query``), else ``""``."""
    if not isinstance(node, ast.Call):
        return ""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _is_route(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True if a decorator of *fn* is a call like ``router.get(...)`` / ``app.post(...)``."""
    for deco in fn.decorator_list:
        if isinstance(deco, ast.Call) and isinstance(deco.func, ast.Attribute) and deco.func.attr in _ROUTE_METHODS:
            return True
    return False


def _annotated_source(annotation: ast.expr | None) -> str:
    """The request source named in ``Annotated[T, Query(...)]`` metadata, else ``""``."""
    if not (isinstance(annotation, ast.Subscript) and isinstance(annotation.slice, ast.Tuple)):
        return ""
    head = annotation.value
    head_name = head.id if isinstance(head, ast.Name) else head.attr if isinstance(head, ast.Attribute) else ""
    if head_name != "Annotated":
        return ""
    return next((_call_name(m) for m in annotation.slice.elts[1:] if _call_name(m) in _REQUEST_SOURCES), "")


def _request_bound_params(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[tuple[ast.arg, str]]:
    """``(parameter, source)`` for each parameter bound from the request by a default or an ``Annotated`` marker."""
    args = fn.args
    positional = args.posonlyargs + args.args
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    pairs = list(zip(positional, defaults)) + list(zip(args.kwonlyargs, args.kw_defaults))
    out = []
    for arg, default in pairs:
        if arg.arg.startswith("_"):
            continue
        source = _call_name(default) if default is not None and _call_name(default) in _REQUEST_SOURCES else _annotated_source(arg.annotation)
        if source:
            out.append((arg, source))
    return out


def _loaded_names(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Every name read in *fn*'s body, nested functions included. The docstring is a string, not a name."""
    return {n.id for stmt in fn.body for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def scan_dead_endpoint_parameters(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find an HTTP endpoint parameter bound from the request that the handler never reads.

    A route handler declares ``strategy: str = Query(default="random", description="...")``: the parameter
    appears in the OpenAPI schema, clients send it, and nothing in the handler uses it. The endpoint's
    documented behaviour and its real behaviour have diverged, silently -- ``dead_cli_flag`` for the HTTP
    surface. Real case (glossum 2026-09-01 audit, 05-F3 / 11-F4): ``strategy: Literal["random", "easy", "hard"]``
    on ``/batch/learning`` was accepted and documented, and the only other place its name appeared was the
    handler's docstring.

    The population is a parameter of a route-decorated function (``@router.get``, ``@app.post``,
    ``api_route``, ``websocket``...) whose value comes from the request: a ``Query`` / ``Path`` / ``Header`` /
    ``Cookie`` / ``Body`` / ``Form`` / ``File`` default, or the same marker in ``Annotated[...]``. It is
    reported when no ``Name`` load of it occurs anywhere in the body; walking names rather than text is what
    keeps a docstring mention from counting.

    Not in the population: ``Depends`` / ``Security`` parameters (a dependency may be wanted only for its side
    effect), names starting with ``_`` (the conventional "deliberately unused"), and ``request: Request`` /
    ``response: Response``, which are annotation-typed rather than request-bound. Known blind spot: a value
    read only through ``locals()`` or a decorator is reported as dead.

    Severity: P2 -- a documented request parameter that silently does nothing is a behaviour-vs-contract gap,
    not a crash.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines: list[str] | None = None
        rel = py.relative_to(root).as_posix()
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or not _is_route(fn):
                continue
            params = _request_bound_params(fn)
            if not params:
                continue
            loaded = _loaded_names(fn)
            for arg, source in params:
                if arg.arg in loaded:
                    continue
                if src_lines is None:
                    src_lines = _read_src_lines(py)
                findings.append(Finding(
                    check="dead_endpoint_parameter",
                    severity="P2",
                    file=rel,
                    line=arg.lineno,
                    snippet=_line_text(src_lines, arg.lineno),
                    detail=(
                        f"`{fn.name}` takes `{arg.arg}` from the request ({source}) but never reads it: clients see it "
                        f"in the API schema and send it, and it changes nothing. Use it, or remove it from the endpoint."
                    ),
                ))
    return findings

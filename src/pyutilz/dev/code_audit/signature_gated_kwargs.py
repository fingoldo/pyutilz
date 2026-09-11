"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _line_text, _read_src_lines, _safe_parse

# --- a caller's argument dropped because a signature did not name it ----------------------------

# What the negative branch has to do for the drop to be visible: say something, or refuse.
_REPORTING_CALLS = frozenset({"warn", "warning", "error", "exception", "critical", "log", "print", "fail", "raise_for_status"})


def _signature_parameter_names(node: ast.AST) -> bool:
    """True if *node* is ``inspect.signature(...).parameters``, or ``set``/``list``/``dict`` of it."""
    if isinstance(node, ast.Call):
        return any(_signature_parameter_names(arg) for arg in node.args)
    if isinstance(node, ast.Attribute) and node.attr == "parameters":
        inner = node.value
        return isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) and inner.func.attr == "signature"
    return False


def _parameter_set_names(fn: ast.AST) -> set[str]:
    """Local names bound to a signature's parameter names inside *fn*."""
    out: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and _signature_parameter_names(node.value):
            out |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.Try):
            # `try: params = set(signature(x).parameters)  except (ValueError, TypeError): params = set()`
            for handler in node.handlers:
                for sub in ast.walk(handler):
                    if isinstance(sub, ast.Assign):
                        out |= {t.id for t in sub.targets if isinstance(t, ast.Name) and t.id in out}
    return out


def _gated_parameter(test: ast.expr, parameter_sets: set[str]) -> str:
    """The literal name a membership test gates on, for ``"thinking" in provider_params``, else ``""``."""
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.In)):
        return ""
    left, right = test.left, test.comparators[0]
    if not (isinstance(left, ast.Constant) and isinstance(left.value, str)):
        return ""
    if isinstance(right, ast.Name) and right.id in parameter_sets:
        return left.value
    if _signature_parameter_names(right):
        return left.value
    return ""


def _passes_the_value_on(body: list[ast.stmt], name: str) -> bool:
    """True if *body* forwards the gated argument: ``kwargs["thinking"] = thinking`` or ``kwargs.update(...)``."""
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Subscript) for t in node.targets):
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("update", "setdefault", "append"):
            return True
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return True
    return False


def _says_something(body: list[ast.stmt]) -> bool:
    """True if *body* reports the drop: a log/warn/print call, or a raise."""
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Raise):
            return True
        if isinstance(node, ast.Call):
            func = node.func
            named = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
            if named in _REPORTING_CALLS:
                return True
    return False


def scan_signature_gated_kwarg_drops(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find an argument forwarded only when a callee's signature names it, and dropped in silence when it does not.

    ``if "thinking" in inspect.signature(provider.generate).parameters: kwargs["thinking"] = thinking`` sends the
    caller's request where it fits and discards it everywhere else, with nothing logged: the call runs at the
    callee's own default and the caller is told nothing. Real case (glossum 2026-09-01 audit, proposal #6):
    ``temperature`` and ``thinking`` vanished for two providers this way, and the gate also read
    ``OpenRouterProvider.generate(self, *args, **kwargs)`` as accepting nothing at all, because a signature that
    forwards ``**kwargs`` to its base names no parameters of its own. A silent branch hides both mistakes.

    The population is an ``if`` whose test is ``"<literal>" in <signature parameter names>`` (the names bound from
    ``inspect.signature(...).parameters``, directly or through a ``try``/``except`` fallback), whose body forwards
    the value into a call's arguments, and whose ``else`` is absent or empty. Reported once per such branch.

    Not reported: a gate whose negative branch logs, warns, prints or raises -- the drop is then visible, which is
    the fix; and a membership test against anything that is not a signature's parameter names.

    Severity: P2 -- the call still runs, but it runs with different arguments than the caller asked for.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = _read_src_lines(py)
        rel = py.relative_to(root).as_posix()
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            parameter_sets = _parameter_set_names(fn)
            for node in ast.walk(fn):
                if not isinstance(node, ast.If) or node.orelse:
                    continue
                name = _gated_parameter(node.test, parameter_sets)
                if not name or not _passes_the_value_on(node.body, name):
                    continue
                findings.append(Finding(
                    check="signature_gated_kwarg_drop",
                    severity="P2",
                    file=rel,
                    line=node.lineno,
                    snippet=_line_text(src_lines, node.lineno),
                    detail=(
                        f"`{name}` is forwarded only when the callee's signature names it, and dropped in silence "
                        f"when it does not: the call then runs with the callee's default and the caller is never "
                        f"told. Report the drop in the negative branch. Note that a signature taking `**kwargs` "
                        f"names no parameters of its own, so a forwarding subclass reads as accepting nothing."
                    ),
                ))
    return findings

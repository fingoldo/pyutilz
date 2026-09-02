"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path

from ._base import _DEFAULT_EXCLUDE_DIRS, Finding, _iter_py_files, _line_text, _safe_parse

# --- a constructor argument another method overwrites from configuration -----------------------
#
# `__init__` stores a parameter on `self`; another method of the same class later reassigns that
# same attribute from `cfg()`, the environment, or a module global. The constructor argument is
# therefore ADVISORY -- it holds until the first call that refreshes it.
#
# That is often the intended design (a hot-reloading rate limiter is exactly this), which is why
# this is reported as informational rather than as a defect. What makes it worth reporting at all
# is the test-side consequence, confirmed in the upwork scrapers:
#
#     tb = _TokenBucket(5.0)
#     ...
#     assert tb._tokens < 1.0
#
# `acquire()` self-refreshes the rate from live config on its first call, so the 5.0 was replaced
# by the deployed 50 before anything was measured. Both assertions were measuring the DEPLOYMENT,
# not the object -- and they passed, on a machine whose config happened to make them pass.
#
# This is `CLAUDE.md`'s "a fix in a default is not a fix if a caller overrides it" seen from the
# mirror side: here the object overrode the test.
#
# A repo-local meta-test can intersect this with "a test constructs this class with an explicit
# value for that parameter and asserts on it", which is where the precision comes from. This
# scanner supplies the production half.

_CONFIG_READS = {"cfg", "config", "getenv", "environ", "get_config", "settings", "get_setting"}


def _reads_config(node: ast.AST) -> bool:
    """Does this subtree read configuration or the environment?"""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            target = sub.func
            names = set()
            while isinstance(target, ast.Attribute):
                names.add(target.attr)
                target = target.value
            if isinstance(target, ast.Name):
                names.add(target.id)
            if isinstance(target, ast.Call) and isinstance(target.func, ast.Name):
                names.add(target.func.id)  # the `cfg().get(...)` spelling
            if names & _CONFIG_READS:
                return True
        if isinstance(sub, ast.Attribute) and sub.attr in _CONFIG_READS:
            return True
    return False


def _self_attr_targets(stmt: ast.AST) -> set[str]:
    """`self.x` assignment targets in this statement."""
    targets = stmt.targets if isinstance(stmt, ast.Assign) else [getattr(stmt, "target", None)]
    out: set[str] = set()
    for t in targets:
        if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
            out.add(t.attr)
    return out


def _param_backed_attributes(init: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, str]:
    """`self.x = param` bindings in `__init__`, as {attribute: parameter}."""
    params = {a.arg for a in init.args.args} | {a.arg for a in init.args.kwonlyargs}
    bound: dict[str, str] = {}
    for stmt in ast.walk(init):
        if not isinstance(stmt, ast.Assign):
            continue
        if isinstance(stmt.value, ast.Name) and stmt.value.id in params:
            for attr in _self_attr_targets(stmt):
                bound[attr] = stmt.value.id
    return bound


def scan_constructor_param_overwritten(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find constructor arguments that another method later overwrites from configuration.

    The argument is advisory: it holds until the first call that refreshes it. That is frequently
    the intended design, so this is INFORMATIONAL -- it is reported so that a test passing an
    explicit value for such a parameter can be recognised as measuring the deployment rather than
    the object, which is how two assertions in the upwork scrapers came to measure a config file.

    A repo-local meta-test that intersects this with "a test constructs the class with that
    parameter and asserts on the value" is where the precision lives; this is the production half.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()

        for cls in ast.walk(tree):
            if not isinstance(cls, ast.ClassDef):
                continue
            methods = [m for m in cls.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
            init = next((m for m in methods if m.name == "__init__"), None)
            if init is None:
                continue
            bound = _param_backed_attributes(init)
            if not bound:
                continue

            # Which method assigns which constructor-bound attribute, and which methods read
            # configuration. The two are frequently NOT the same method: `_TokenBucket`, the
            # case this rule was written for, reads `cfg()` in `_refresh_rate` and assigns
            # `self._rate` inside `update_rate(rate)` one call further down. Requiring both
            # in one statement missed its own worked example.
            assigners: dict[str, list[tuple[str, int]]] = {}
            for method in methods:
                if method is init:
                    continue
                for stmt in ast.walk(method):
                    if not isinstance(stmt, (ast.Assign, ast.AugAssign)):
                        continue
                    for attr in _self_attr_targets(stmt) & set(bound):
                        assigners.setdefault(attr, []).append((method.name, stmt.lineno))

            for method in methods:
                if method is init:
                    continue
                if not _reads_config(method):
                    continue
                # Attributes this method assigns directly, plus those assigned by same-class
                # methods it calls -- one hop, which is what `_refresh_rate -> update_rate`
                # needs and is where the chain stops being legible anyway.
                reachable = {
                    n.func.attr
                    for n in ast.walk(method)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == "self"
                } | {method.name}
                for attr, sites in sorted(assigners.items()):
                    site = next(((m, ln) for m, ln in sites if m in reachable), None)
                    if site is None:
                        continue
                    via, stmt_line = site
                    through = "" if via == method.name else f" (via `{via}`)"
                    findings.append(
                        Finding(
                            check="constructor_param_overwritten",
                            severity="Low",
                            file=rel,
                            line=stmt_line,
                            snippet=_line_text(src_lines, stmt_line),
                            detail=(
                                f"`{cls.name}.__init__` stores its `{bound[attr]}` parameter as "
                                f"`self.{attr}`, and `{method.name}` reassigns it from "
                                f"configuration{through}. The constructor argument is advisory -- it "
                                "holds only until this runs. A test that passes one and asserts on it "
                                "is measuring the deployment, not the object."
                            ),
                        )
                    )
    return findings

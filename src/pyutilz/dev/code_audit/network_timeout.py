"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _line_text, _read_src_lines, _safe_parse

# --- network call with no timeout= ----------------------------------------
#
# Class of bug found repeatedly in the 2026-07-21 web/cloud audit (the single most
# High-finding-dense domain in the whole audit): web.py's urllib.request.urlopen() calls in
# get_external_ip()/get_ipinfo() had no timeout, and download_in_parallel()'s grequests.get()
# call had none either -- a single unresponsive URL could hang the whole batch/process
# indefinitely on a real network failure (not a hypothetical -- both were flagged High).


# (module, attr) pairs recognized as network calls that accept timeout= as a kwarg.
_NETWORK_CALL_ATTRS = frozenset({
    "get", "post", "put", "delete", "patch", "head", "options", "request",
    "urlopen",
})
_NETWORK_MODULE_HINTS = frozenset({"requests", "grequests", "httpx", "urllib", "urlopen"})


def _call_root_name(func: ast.AST) -> str:
    """Best-effort dotted-root name for a Call's func (e.g. ``requests.get`` -> ``"requests"``)."""
    node = func
    while isinstance(node, ast.Attribute):
        node = node.value
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _timeout_preconfigured_vars(tree: ast.AST) -> "set[str]":
    """Local variable names assigned from a call that itself set ``timeout=``.

    Same-file heuristic (not type inference): ``session = httpx.Client(timeout=10)`` makes every
    ``session.get(...)`` pre-configured, so flagging those would be pure noise.
    """
    out: "set[str]" = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if any(kw.arg == "timeout" for kw in node.value.keywords):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        out.add(t.id)
    return out


def _directly_imported_network_calls(tree: ast.AST) -> "dict[str, str]":
    """{local name -> network call attr} for ``from urllib.request import urlopen``-style imports.

    The directly imported spelling is at least as common as the dotted one, and a bare ``Name``
    callee has no receiver to key on, so resolve it back to the module it came from.
    """
    out: "dict[str, str]" = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if node.module.split(".")[0] in _NETWORK_MODULE_HINTS:
                for alias in node.names:
                    if alias.name in _NETWORK_CALL_ATTRS:
                        out[alias.asname or alias.name] = alias.name
    return out


def _resolve_network_callee(node: ast.Call, directly_imported: "dict[str, str]") -> "tuple[str, str] | None":
    """``(root name, call attr)`` if this Call looks like a network call, else ``None``."""
    if isinstance(node.func, ast.Name):
        if node.func.id not in directly_imported:
            return None
        return node.func.id, directly_imported[node.func.id]
    if isinstance(node.func, ast.Attribute):
        if node.func.attr not in _NETWORK_CALL_ATTRS:
            return None
        return _call_root_name(node.func), node.func.attr
    return None


def _is_flagged_network_call(node: ast.Call, root_name: str, attr: str, preconfigured: "set[str]") -> bool:
    """Whether a resolved network callee should be reported (no ``timeout=``, real receiver)."""
    if not isinstance(node.func, ast.Name):
        # Neither a known network-library root name nor a locally pre-configured session/client
        # variable -- likely an unrelated .get()/.post() on some other object (e.g. dict.get());
        # skip to avoid noise.
        if root_name not in _NETWORK_MODULE_HINTS and root_name not in preconfigured and attr not in ("urlopen",):
            return False
    if root_name in preconfigured:
        return False
    if any(kw.arg == "timeout" for kw in node.keywords):
        return False
    if any(kw.arg is None for kw in node.keywords):  # **kwargs -- can't tell, skip
        return False
    return True


def scan_missing_network_timeout(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find HTTP/socket calls (``requests.get/post/...``, ``urllib.request.urlopen``,
    ``grequests.get/post``, ``httpx.get/post``) with no ``timeout=`` keyword -- a call that can
    hang forever on a real network failure (unresponsive server, black-holed connection) instead
    of raising after a bounded wait.

    Session/client objects constructed with their own ``timeout=`` (e.g. ``httpx.Client(timeout=...)``)
    are not flagged when called via that object's variable name in the same file (best-effort
    same-file heuristic, not full type inference).

    Severity: P2 (a real, if narrow, hang risk -- not a crash, so lower than a correctness bug).
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = _read_src_lines(py)
        rel = py.relative_to(root).as_posix()

        preconfigured = _timeout_preconfigured_vars(tree)
        directly_imported = _directly_imported_network_calls(tree)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            resolved = _resolve_network_callee(node, directly_imported)
            if resolved is None:
                continue
            root_name, attr = resolved
            if not _is_flagged_network_call(node, root_name, attr, preconfigured):
                continue
            findings.append(Finding(
                check="missing_network_timeout",
                severity="P2",
                file=rel,
                line=node.lineno,
                snippet=_line_text(src_lines, node.lineno),
                detail=(
                    f"`{root_name}.{attr}(...)` has no `timeout=` -- can hang " if not isinstance(node.func, ast.Name) else f"`{root_name}(...)` has no `timeout=` -- can hang "
                    "indefinitely on an unresponsive server/black-holed connection instead of "
                    "raising after a bounded wait."
                ),
            ))
    return findings

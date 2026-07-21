"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

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
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()

        # Same-file heuristic: variable names assigned from a call that itself set timeout=
        # (e.g. `session = httpx.Client(timeout=10)`) are treated as pre-configured.
        timeout_preconfigured_vars: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                if any(kw.arg == "timeout" for kw in node.value.keywords):
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            timeout_preconfigured_vars.add(t.id)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in _NETWORK_CALL_ATTRS:
                continue
            root_name = _call_root_name(node.func)
            if root_name not in _NETWORK_MODULE_HINTS and root_name not in timeout_preconfigured_vars:
                # Neither a known network-library root name nor a locally pre-configured
                # session/client variable -- likely an unrelated .get()/.post() on some other
                # object (e.g. dict.get()); skip to avoid noise.
                if node.func.attr not in ("urlopen",):
                    continue
            if root_name in timeout_preconfigured_vars:
                continue
            has_timeout = any(kw.arg == "timeout" for kw in node.keywords)
            has_star_kwargs = any(kw.arg is None for kw in node.keywords)  # **kwargs -- can't tell, skip
            if has_timeout or has_star_kwargs:
                continue
            findings.append(Finding(
                check="missing_network_timeout",
                severity="P2",
                file=rel,
                line=node.lineno,
                snippet=_line_text(src_lines, node.lineno),
                detail=(
                    f"`{root_name}.{node.func.attr}(...)` has no `timeout=` -- can hang "
                    "indefinitely on an unresponsive server/black-holed connection instead of "
                    "raising after a bounded wait."
                ),
            ))
    return findings

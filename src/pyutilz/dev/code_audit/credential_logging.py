"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
import re
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

# --- credential-shaped value reaching a logger call ------------------------
#
# Class of bug found repeatedly in the 2026-07-21 security audit: web.py's report_params()
# logged full proxies/headers at INFO with NO redaction gate, even though the very same file's
# handle_blocking() (a few lines away) explicitly does `.split("@")[1]` before logging a proxy
# URL -- proof the author knew the risk in one place but not the other (Critical). logginglib.py's
# `special_vars` were excluded from one log dict key but copied unredacted into another
# ("relocation, not redaction" -- Medium).

_CREDENTIAL_NAME_RE = re.compile(r"(?i)\b(proxy|proxies|password|passwd|pass|auth|token|credential|secret|api_key|apikey|cookie)\b")
_REDACT_HINT_RE = re.compile(r"(?i)redact|mask|sanitiz|\.split\(.@.\)|scrub")


def _looks_credential_shaped(node: ast.AST) -> bool:
    """True if ``node`` (a Name/Attribute/keyword value expression) has a name matching the
    credential-shaped vocabulary."""
    if isinstance(node, ast.Name):
        return bool(_CREDENTIAL_NAME_RE.search(node.id))
    if isinstance(node, ast.Attribute):
        return bool(_CREDENTIAL_NAME_RE.search(node.attr))
    return False


_LOG_METHODS = frozenset({"debug", "info", "warning", "warn", "error", "exception", "critical", "log"})


def scan_credential_shaped_log_args(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find ``logger.<level>(...)`` calls whose arguments include a variable/attribute whose
    NAME matches a credential-shaped vocabulary (proxy, password, token, auth, secret, api_key,
    cookie, credential), with no redaction hint (``.split("@")``, a name/call containing
    redact/mask/sanitize/scrub) anywhere in the same source line.

    Deliberately name-based (not value/type-based -- statically impossible in general) and
    therefore prone to false positives on legitimately-safe values whose name happens to match
    (e.g. ``token_type``) -- use the baseline mechanism liberally.

    Severity: P2 (a real, if noisy, security-adjacent signal -- credential-shaped values in log
    output are exactly the shape of two confirmed findings in the 2026-07-21 audit).
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _LOG_METHODS):
                continue
            if not (isinstance(node.func.value, ast.Name) and node.func.value.id in ("logger", "log", "logging")):
                continue
            offending = [a for a in node.args if _looks_credential_shaped(a)]
            offending += [kw.value for kw in node.keywords if _looks_credential_shaped(kw.value)]
            if not offending:
                continue
            line_text = _line_text(src_lines, node.lineno)
            # Redaction-hint check spans a small window around the call (helper redaction often
            # happens on the line(s) just before the log call, assigning a redacted copy).
            window = "\n".join(src_lines[max(0, node.lineno - 4) : node.lineno])
            if _REDACT_HINT_RE.search(window):
                continue
            credential_names: set[str] = set()
            for a in offending:
                assert isinstance(a, (ast.Name, ast.Attribute))  # nosec B101 - _looks_credential_shaped only accepts Name/Attribute nodes; narrows the type for the access below, not a security check
                credential_names.add(a.id if isinstance(a, ast.Name) else a.attr)
            names = ", ".join(sorted(credential_names))
            findings.append(Finding(
                check="credential_shaped_log_arg",
                severity="P2",
                file=rel,
                line=node.lineno,
                snippet=line_text,
                detail=(
                    f"logger call passes credential-shaped value(s) ({names}) with no redaction "
                    "hint nearby (.split('@'), redact/mask/sanitize) -- confirm this doesn't leak "
                    "a plaintext credential into logs."
                ),
            ))
    return findings

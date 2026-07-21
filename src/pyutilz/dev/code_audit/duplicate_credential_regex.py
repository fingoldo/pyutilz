"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
import re
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

# Default canonical-module candidates: a project designates its own single
# source of truth for credential-shaped regexes (e.g. "secrets_scrub.py") via
# the ``canonical_module_rel_paths`` kwarg. No default file name is assumed
# repo-wide -- every project names this module differently (or doesn't have
# one yet, in which case every hit is worth reviewing).
DEFAULT_CREDENTIAL_KEYWORDS_RE = re.compile(
    r"password|passwd|api[_-]?key|secret|credential|authorization|bearer|\btoken\b",
    re.IGNORECASE,
)


def scan_duplicate_credential_regex(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
    canonical_module_rel_paths: frozenset[str] = frozenset(),
    credential_keywords_re: re.Pattern[str] = DEFAULT_CREDENTIAL_KEYWORDS_RE,
) -> list[Finding]:
    """Find ``re.compile(...)`` calls whose pattern string contains a
    credential-shaped keyword (password/token/secret/api_key/credential/
    authorization/bearer), OUTSIDE the project's designated canonical
    scrubber module(s).

    The same "redact a secret-shaped substring before logging/persisting
    it" problem tends to get solved independently, and non-identically, in
    multiple places once a codebase has more than one logging/persistence
    call site touching potentially-sensitive data -- coverage then silently
    drifts between the copies (one gets a new pattern added, the others
    don't). A single canonical module, with every OTHER file importing from
    it instead of defining its own ``re.compile(...)``, is the fix; this
    scanner is the tripwire against the drift starting over.

    ``canonical_module_rel_paths`` (relative to ``root``, e.g.
    ``frozenset({"secrets_scrub.py"})``) is the project's own designated
    module(s) -- calls inside those are never flagged (that's WHERE new
    patterns belong). A project with no canonical module yet gets every
    credential-shaped regex flagged, surfacing the duplication this check
    exists to prevent so one can be designated.

    A regex matching a credential-shaped keyword for a DIFFERENT purpose
    than redaction (e.g. a prompt-injection/data-exfiltration ATTEMPT
    DETECTOR scanning untrusted text for social-engineering phrases like
    "please share your api key") is a legitimate false positive -- grandfather
    it via the baseline, same as any other reviewed finding in this package's
    convention, rather than loosening the keyword list.

    Severity: P2 (redaction-coverage drift risk, not an immediate leak --
    each individual copy may still work today).
    """
    canonical_paths = {(root / rel).resolve() for rel in canonical_module_rel_paths}
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        if py.resolve() in canonical_paths:
            continue
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "compile"):
                continue
            if not (isinstance(node.func.value, ast.Name) and node.func.value.id == "re"):
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
                continue
            pattern = node.args[0].value
            if not credential_keywords_re.search(pattern):
                continue
            findings.append(Finding(
                check="duplicate_credential_regex",
                severity="P2",
                file=rel,
                line=node.lineno,
                snippet=_line_text(src_lines, node.lineno),
                detail=(
                    f"re.compile({pattern!r}) matches a credential-shaped keyword outside the "
                    "designated canonical scrubber module -- move it there (union it into an "
                    "existing pattern, or add a new one) and import from there instead, or "
                    "grandfather this exact finding if it's a reviewed false positive (a "
                    "different-purpose regex that happens to match a credential-shaped keyword)."
                ),
            ))
    return findings

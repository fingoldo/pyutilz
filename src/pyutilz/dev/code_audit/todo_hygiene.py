"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS

from ..meta_test_utils import ATTRIBUTION_RE, scan_todo_markers

# --- un-attributed TODO/FIXME/XXX/HACK markers ------------------------------


def scan_todo_hygiene(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find ``TODO``/``FIXME``/``XXX``/``HACK`` comments with no attribution.

    Un-attributed debt markers accumulate as anonymous wishlist items nobody owns -- a 2-year-old
    "TODO: handle the empty list case" the original author has long since forgotten about, that
    later surfaces as a P0 outage. An attribution is either an assignee in parens
    (``TODO(name): ...``), an ISO date (``TODO 2026-04-28: ...``), or an ``@mention``.

    Thin wrapper over ``pyutilz.dev.meta_test_utils.scan_todo_markers`` (the shared marker
    scanner), producing baseline-drift-compatible ``Finding`` objects for the un-attributed subset.
    """
    findings: list[Finding] = []
    for path, lineno, kw, line in scan_todo_markers(root, extra_excludes=tuple(exclude_dirs)):
        if ATTRIBUTION_RE.search(line):
            continue
        rel = path.relative_to(root).as_posix()
        findings.append(Finding(
            check="todo_hygiene",
            severity="Low",
            file=rel,
            line=lineno,
            snippet=line[:160],
            detail=(
                f"un-attributed {kw} comment. Add an assignee in parens (`{kw}(name): ...`), an "
                f"ISO date (`{kw} 2026-04-28: ...`), or an `@mention`."
            ),
        ))
    return findings

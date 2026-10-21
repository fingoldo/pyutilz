"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs.

A test that asserts on the TEXT of the code under test instead of running it. The rule lives in py-ci-shared
(``py_ci_shared.source_text_claims``), which judges every spelling of the practice and reports every site this scanner did;
this module is the pyutilz face of it, reached lazily because py-ci-shared is a development tool here, not a runtime dependency.

Shared-gate counterpart: py_ci_shared.source_text_claims is the one implementation of this rule.
"""

from __future__ import annotations

from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _read_src_lines, is_line_exempt

_CHECK = "source_text_assertion"


_READ_SPELLINGS = ("getsourcelines", "getsourcefile", "getsource", "unparse", "read_text", "read_bytes", "readlines", "read")


def _how(lines: list[str], line: int) -> str:
    """The read spelling nearest above the assertion: the source the claim is about, named as the code reads it."""
    for text in reversed(lines[: max(line - 1, 0)]):
        for spelling in _READ_SPELLINGS:
            if spelling + "(" in text:
                return spelling
    return "a source-text read"


def scan_source_text_assertions(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Test assertions made against the source text of the code under test, across the ``test_*.py`` files under *root*.

    A file the shared detector cannot read or parse is skipped, as every pyutilz scanner skips a file it cannot parse.
    """
    from py_ci_shared.source_text_claims import find_source_text_claims  # dev tool, imported only when the scanner runs
    from py_ci_shared._core.errors import SourceError  # the detector's own error for a file it cannot read

    out: list[Finding] = []
    for path in sorted(root.rglob("*.py")):
        parts = path.relative_to(root).parts
        if any(part in exclude_dirs for part in parts[:-1]):
            continue
        if not (path.name.startswith("test_") or any(part in ("tests", "test") for part in parts[:-1])):
            continue
        try:
            claims = find_source_text_claims(path)
        except SourceError:
            continue
        lines = _read_src_lines(path)
        rel = path.relative_to(root).as_posix()
        for claim in claims:
            if is_line_exempt(lines, claim.line, _CHECK):
                continue
            text = lines[claim.line - 1].strip() if 0 < claim.line <= len(lines) else ""
            out.append(
                Finding(
                    check=_CHECK,
                    severity="P2",
                    file=rel,
                    line=claim.line,
                    snippet=text[:160],
                    detail=(
                        f"assertion is made against SOURCE TEXT (via {_how(lines, claim.line)}) rather than against behaviour -- "
                        "it passes whether or not the inspected code ever runs. Call the code and assert on what it returns or does; "
                        "if the artefact under test really is a text, keep the assertion and mark the line audit-exempt with a reason."
                    ),
                )
            )
    return out

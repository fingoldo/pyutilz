"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from ._base import _DEFAULT_EXCLUDE_DIRS, Finding, _iter_py_files, _line_text, _safe_parse

# --- an invariant asserted in prose and nowhere else -----------------------------------------------
#
# Confirmed instances (Autopsia, 2026-07-26): a model documented as "deliberately not one of the
# models under test" was sitting in the tournament's own candidate list; a module whose docstring
# states the hard rule "never decompose a posterior with a different model than produced it" was
# violated by its own caller two files away; a helper documented as emitting cause->finding rows
# emitted them inverted.
#
# What makes this class expensive is that the prose is the only specification anyone reads. A reader
# who trusts the docstring writes correct calling code against an incorrect implementation, and the
# error surfaces far from either.
#
# The phrase list is short and absolute on purpose. "always" and "must" appear in ordinary
# explanatory prose constantly; "never", "must not" and "deliberately not" are load-bearing claims a
# reader is entitled to rely on, and they are rare enough that requiring a test per claim is cheap.

_CLAIM_RE = re.compile(
    r"\b(never|must not|may not|cannot ever|deliberately not|is not one of|guaranteed to|invariant:)\b",
    re.IGNORECASE,
)


def scan_unenforced_docstring_invariants(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
    test_roots: tuple[Path, ...] = (),
) -> list[Finding]:
    """Find symbols whose docstring states an absolute invariant that no test names.

    ``test_roots`` are the trees searched for enforcement. A symbol is considered enforced when its
    name appears anywhere in them -- a deliberately weak bar, because proving that a given test
    checks a given sentence is not decidable, and the defect being targeted is the claim NOTHING
    references at all.

    Only functions and classes are scanned, not module docstrings: a module-level essay describing
    design rationale is not a per-symbol contract, and including it made the check fire on prose.

    Severity: P2. The claim may well be true today; the finding is that nothing will notice when it
    stops being true.
    """
    if not test_roots:
        return []

    test_corpus = ""
    for troot in test_roots:
        troot = Path(troot)
        if not troot.exists():
            continue
        for py in _iter_py_files(troot, exclude_dirs):
            test_corpus += py.read_text(encoding="utf-8", errors="replace")

    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            doc = ast.get_docstring(node) or ""
            match = _CLAIM_RE.search(doc)
            if not match or node.name in test_corpus:
                continue
            sentence = next((s.strip() for s in re.split(r"(?<=[.;])\s", doc) if _CLAIM_RE.search(s)), doc[:120])
            findings.append(
                Finding(
                    check="unenforced_docstring_invariant",
                    severity="P2",
                    file=rel,
                    line=node.lineno,
                    snippet=_line_text(src_lines, node.lineno),
                    detail=(f'{node.name!r} documents an absolute invariant no test names: "{sentence[:160]}". Assert it, or soften the wording to what the code actually promises.'),
                )
            )
    return findings

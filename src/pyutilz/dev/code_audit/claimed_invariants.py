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


def _callers_within(tree: ast.AST) -> dict[str, set[str]]:
    """``{callee_name -> {names of functions/methods in this module that call it}}``, one level of
    indirection only (a caller's own caller is not chased further - see the docstring on the
    transitive-enforcement check below for why one hop is the deliberate stopping point)."""
    callers: dict[str, set[str]] = {}
    for outer in ast.walk(tree):
        if not isinstance(outer, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(outer):
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
                callers.setdefault(inner.func.id, set()).add(outer.name)
    return callers


def scan_unenforced_docstring_invariants(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
    test_roots: tuple[Path, ...] = (),
) -> list[Finding]:
    """Find symbols whose docstring states an absolute invariant that no test names.

    ``test_roots`` are the trees searched for enforcement. A symbol is considered enforced when its
    OWN name appears anywhere in them, OR when a function in the SAME MODULE that calls it (one hop
    only - a private helper's own direct caller, not an arbitrary chain) has its name appear there --
    a deliberately weak bar either way, because proving that a given test checks a given sentence is
    not decidable, and the defect being targeted is the claim NOTHING references at all.

    The one-hop caller extension (2026-08-03) closes the most common false-positive shape measured in
    practice: a PRIVATE helper (`_mondo_name_index`, `kb/gbd_prevalence.py`) whose invariant ("EXACT
    name match only, never fuzzy") is exercised by real tests that call the PUBLIC function it lives
    inside (`build()`) rather than the private helper directly - the normal way a private symbol gets
    test coverage at all. Chasing more than one hop was deliberately rejected: two hops away starts
    matching call chains a reader would not recognise as "this test covers that claim" on inspection,
    which is the same false-confidence failure mode this whole check exists to prevent one level up.

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
        callers_of = _callers_within(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            doc = ast.get_docstring(node) or ""
            match = _CLAIM_RE.search(doc)
            if not match:
                continue
            if node.name in test_corpus:
                continue
            if any(caller in test_corpus for caller in callers_of.get(node.name, ())):
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

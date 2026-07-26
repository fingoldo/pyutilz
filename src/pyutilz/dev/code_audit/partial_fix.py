"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path

from ._base import _DEFAULT_EXCLUDE_DIRS, Finding, _iter_py_files, _line_text, _safe_parse

# --- two shapes that only show up when you look ACROSS functions ----------------------------------
#
# Both were named by the Autopsia round-2 audits as the most frequently repeated defect shapes, and
# neither is visible while reading one function at a time.
#
# PARTIAL FIX -- a defect closed on one leg of a fork and left open on its siblings. The audits found
# it four separate times: the empty-input hole was closed in `score_grounding` and left open in
# `score_junk` and `score_forbidden`; macro-vs-micro averaging was fixed for the modality metric and
# left alone for the headline recall; a page-resolution fix landed in the extraction layer and was
# re-broken one function later at the row layer. Sibling functions sharing a prefix are a family with
# a shared contract, and a guard that most of the family has and one member lacks is a fix that
# stopped one call short.
#
# INCONSISTENT FILTER -- the same exclusion rule applied by some consumers of a collection and not
# others. Found three times: post-mortem events excluded by the ordinal ranker and the can't-miss
# list but not by the Monte-Carlo ranker or the validation endpoint; a reliability parameter passed
# to two of four decomposition entry points; a licence redaction applied to two of four row types.
# This one cannot be inferred - the project must name the pairs - but once named it is exact.

# Only shapes that actually TEST emptiness. `len(p)` alone is not one of them - `len(rows) * 3` is
# arithmetic, and treating it as a guard made every member of a family look guarded and the check silent.
_GUARD_TEMPLATES = ("if not {p}", "not {p}:", "if {p}:", "bool({p})", "{p} and ", "and {p}", "len({p}) ==", "len({p}) >", "len({p}) <")


def scan_partial_guard_across_siblings(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
    min_family: int = 3,
    min_share: float = 0.6,
) -> list[Finding]:
    """Find a family of sibling functions where most members guard their first argument and one does not.

    A family is the set of top-level functions in one module sharing a ``prefix_`` and taking the
    same first parameter name -- ``score_triples`` / ``score_junk`` / ``score_grounding`` over
    ``pred_triples``. Requiring at least ``min_family`` members and a ``min_share`` majority keeps
    the check off two-function coincidences; the majority IS the evidence that the guard is part of
    the family's contract rather than one member's local need.

    Severity: P2. Each member is individually plausible; the finding is the asymmetry.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src = py.read_text(encoding="utf-8", errors="replace")
        src_lines = src.splitlines()
        rel = py.relative_to(root).as_posix()

        families: dict[tuple[str, str], list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or "_" not in node.name.strip("_"):
                continue  # a leading underscore is privacy, not a family prefix: `_a`/`_b` are not siblings
            params = node.args.args
            if not params:
                continue
            families.setdefault((node.name.lstrip("_").split("_", 1)[0], params[0].arg), []).append(node)

        for (prefix, param), members in sorted(families.items()):
            if len(members) < min_family:
                continue
            guards = tuple(t.format(p=param) for t in _GUARD_TEMPLATES)
            guarded: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
            unguarded: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
            for member in members:
                body = ast.get_source_segment(src, member) or ""
                (guarded if any(g in body for g in guards) else unguarded).append(member)
            if not unguarded or len(guarded) / len(members) < min_share:
                continue
            findings.extend(
                Finding(
                    check="partial_guard_across_siblings",
                    severity="P2",
                    file=rel,
                    line=member.lineno,
                    snippet=_line_text(src_lines, member.lineno),
                    detail=(
                        f"{member.name!r} does not guard {param!r} while {len(guarded)} of {len(members)} sibling "
                        f"{prefix}_* functions do -- a fix applied to the family and not to this member."
                    ),
                )
                for member in unguarded
            )
    return findings


def scan_inconsistent_filter(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
    filter_pairs: tuple[tuple[str, str], ...] = (),
) -> list[Finding]:
    """Find consumers of a collection that skip an exclusion rule its other consumers apply.

    ``filter_pairs`` is ``((collection_accessor, required_filter), ...)`` by name -- for example
    ``(("causes", "postmortem_events"),)`` to require that every function reading the cause list also
    consults the post-mortem exclusion. Returns nothing when no pairs are configured, so the check
    costs nothing until a project declares an invariant it wants held across consumers.

    A function is exempt when it DEFINES the filter, and when it is the accessor itself.

    Severity: P1. Two consumers of one collection returning different answers is the shape that
    produces a report and a metric that disagree while both look correct in isolation.
    """
    if not filter_pairs:
        return []
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src = py.read_text(encoding="utf-8", errors="replace")
        src_lines = src.splitlines()
        rel = py.relative_to(root).as_posix()
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = ast.get_source_segment(src, fn) or ""
            for accessor, required in filter_pairs:
                if fn.name in (accessor, required):
                    continue
                if f"{accessor}(" in body and f"{required}(" not in body:
                    findings.append(
                        Finding(
                            check="inconsistent_filter",
                            severity="P1",
                            file=rel,
                            line=fn.lineno,
                            snippet=_line_text(src_lines, fn.lineno),
                            detail=(f"{fn.name}() consumes {accessor}() without applying {required}(), which the collection's other consumers apply -- the two answers will differ."),
                        )
                    )
    return findings

"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from ._base import _DEFAULT_EXCLUDE_DIRS, Finding, _line_text, _safe_parse

# --- domain vocabulary inside code declared domain-neutral ----------------------------------------
#
# THE SHAPE. A codebase grows one product, then a second product is meant to reuse its machinery. The
# reusable half is usually named long before it is extracted -- "the provenance envelope", "the quote
# grounding", "the corroboration rule" -- and the extraction is postponed, correctly, because a wide
# speculative refactor with no second consumer rots. What is NOT safe to postpone is knowing whether
# that half is still separable. Entanglement is what makes the later split expensive, and it arrives
# one identifier at a time, invisibly, in code nobody thought they were making domain-specific.
#
# THE RULE. The caller names a BOUNDARY: symbols (a class, a function, a method) that are claimed to
# be domain-neutral, and a VOCABULARY of terms that belong to the current domain. Any vocabulary term
# appearing anywhere inside a boundary symbol's source span -- identifier, string literal, docstring
# or comment -- is a finding.
#
# WHY THE SOURCE SPAN AND NOT THE AST. A domain leak shows up in a docstring and a comment at least as
# often as in an identifier, and those are exactly where the reader's model of "what this code is
# about" is formed. The AST locates the symbol; a line scan over its span decides.
#
# WHY SYMBOLS AND NOT MODULES. In a codebase that has not been split yet, the neutral concept and the
# domain concept usually live in the SAME file -- that is what "not extracted yet" means. A
# module-granularity rule would report every such file as violated and say nothing about whether the
# specific thing to be extracted is clean. Symbol granularity states the boundary at the resolution
# the eventual extraction will actually cut on.
#
# WHAT THIS IS NOT. It does not prove separability: a symbol can be domain-entangled through a
# CALLEE whose name is innocent. It is a ratchet on the visible half, and its baseline is a
# measurement of how entangled the boundary is today.


@dataclass(frozen=True)
class BoundarySymbol:
    """One symbol claimed to be domain-neutral.

    ``module`` is a path relative to the scan root (``"pkg/graph.py"``). ``symbol`` is a top-level
    name (``"Provenance"``, ``"merge_parallel_edges"``) or a dotted member (``"CausalGraph.validate"``).
    ``note`` is the reader-facing reason this symbol is inside the boundary; it is carried into the
    finding so a report says WHY the leak matters here.
    """

    module: str
    symbol: str
    note: str = ""


def _iter_named(tree: ast.Module) -> dict[str, ast.AST]:
    """Map every top-level and one-level-nested definition to its node, keyed by name and ``Owner.name``."""
    found: dict[str, ast.AST] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        found[node.name] = node
        if isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    found[f"{node.name}.{member.name}"] = member
    return found


def _span(node: ast.AST) -> tuple[int, int]:
    """The 1-based inclusive line range a definition occupies, decorators included."""
    start = getattr(node, "lineno", 1)
    for dec in getattr(node, "decorator_list", []) or []:
        start = min(start, getattr(dec, "lineno", start))
    return start, getattr(node, "end_lineno", start) or start


def _term_pattern(terms: Iterable[str]) -> Optional[re.Pattern[str]]:
    """A case-insensitive alternation matching any term on a word boundary, or None when empty."""
    ordered = sorted({t.strip() for t in terms if t and t.strip()}, key=len, reverse=True)
    if not ordered:
        return None
    return re.compile(r"(?<![A-Za-z])(" + "|".join(re.escape(t) for t in ordered) + r")(?![A-Za-z])", re.IGNORECASE)


def scan_domain_vocabulary_leak(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
    *,
    boundary: Sequence[BoundarySymbol] = (),
    vocabulary: Iterable[str] = (),
    allowed: Iterable[str] = (),
    severity: str = "P2",
) -> list[Finding]:
    """Find domain-specific vocabulary inside symbols the project declares domain-neutral.

    Silent until configured: with no ``boundary`` or no ``vocabulary`` it returns nothing, because a
    boundary is a claim only the project can make.

    Emits two checks. ``domain_vocabulary_leak`` is one finding per (symbol, term): the term appears
    somewhere in that symbol's source span, so the symbol cannot be lifted into a shared library
    without carrying the current domain with it. ``boundary_symbol_missing`` fires when a declared
    symbol no longer exists at the declared path -- without it a rename would silently empty the
    boundary and the check would pass by having nothing left to look at.

    Args:
        root: directory the scan and every ``BoundarySymbol.module`` path are relative to.
        exclude_dirs: unused here (every file is named explicitly); accepted for registry uniformity.
        boundary: the symbols claimed to be domain-neutral.
        vocabulary: terms belonging to the current domain, matched case-insensitively on word boundaries.
        allowed: terms exempted, for the case where a vocabulary term has an unrelated general meaning
            in this codebase. Matched the same way and applied to the MATCHED TEXT, not the line.
        severity: severity tag for emitted findings.
    """
    pattern = _term_pattern(vocabulary)
    if pattern is None or not boundary:
        return []
    exempt = {t.strip().lower() for t in allowed if t and t.strip()}
    findings: list[Finding] = []
    by_module: dict[str, list[BoundarySymbol]] = {}
    for item in boundary:
        by_module.setdefault(item.module, []).append(item)

    for module in sorted(by_module):
        path = root / module
        tree = _safe_parse(path) if path.is_file() else None
        src_lines = path.read_text(encoding="utf-8", errors="replace").splitlines() if path.is_file() else []
        named = _iter_named(tree) if tree is not None else {}
        for item in sorted(by_module[module], key=lambda b: b.symbol):
            node = named.get(item.symbol)
            if node is None:
                findings.append(
                    Finding(
                        check="boundary_symbol_missing",
                        severity="P1",
                        file=module,
                        line=1,
                        snippet="",
                        detail=(
                            f"declared boundary symbol {item.symbol!r} was not found -- the boundary manifest is stale, "
                            f"and a boundary that names nothing passes by vacuity."
                        ),
                    )
                )
                continue
            start, end = _span(node)
            # term -> (first line it appears on, how many lines mention it)
            hits: dict[str, tuple[int, int]] = {}
            for lineno in range(start, min(end, len(src_lines)) + 1):
                for match in pattern.finditer(src_lines[lineno - 1]):
                    term = match.group(1).lower()
                    if term in exempt:
                        continue
                    first, count = hits.get(term, (lineno, 0))
                    hits[term] = (first, count + 1)
            for term in sorted(hits):
                first, count = hits[term]
                where = f"{item.module}::{item.symbol}"
                why = f" ({item.note})" if item.note else ""
                findings.append(
                    Finding(
                        check="domain_vocabulary_leak",
                        severity=severity,
                        file=module,
                        line=first,
                        snippet=_line_text(src_lines, first),
                        # The symbol and the TERM lead the detail, and everything variable follows the
                        # ` -- ` separator, both deliberately. A ratchet keys on the truncated head: it
                        # must tell two leaks in one symbol apart (or a second term arriving in an
                        # already-recorded symbol is silently absorbed), and it must NOT change when the
                        # occurrence count does (or removing one of four mentions - an improvement - fails
                        # the build as a brand-new violation).
                        detail=(
                            f"{item.symbol} names the domain term {term!r} -- {count} occurrence(s); "
                            f"{where}{why} is declared domain-neutral, so lifting it into a shared library "
                            f"would carry the domain with it."
                        ),
                    )
                )
    return findings

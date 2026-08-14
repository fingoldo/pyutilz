"""Free-variable analysis for planning safe function/method extraction (monolith splits).

When carving a block of code out of a large function into its own sibling function/module, the
question that decides correctness is: which names does the block reference that it does NOT itself
define? Get that list wrong (miss one) and the extracted function raises ``NameError``/``UnboundLocalError``
at runtime on whatever code path first reaches the missed name -- often not caught by a quick smoke
import, since name lookup inside a function body is lazy.

This module answers that question directly from the AST instead of by eyeballing a 1000+ line
block: given a source file and a 1-indexed inclusive line range, it reports every name the range
LOADS that isn't bound (assigned, or bound as a ``def``/``async def``) within that same range --
the free variables the extracted function must receive as parameters (or resolve via ``self``,
module-level import, etc).

A companion check catches a sharper trap the naive Load/Store diff misses: a name that IS assigned
somewhere in the range but whose FIRST occurrence (by source position) is a Load. That name is
reassigned later, so a plain Store-anywhere-in-range filter reports it as "not free" -- but the block
still needs an INCOMING value for its first use. This shows up constantly in accumulator-style code
(``selected = [...]``, then repeated ``selected = [x for x in selected if ...]`` filters): the naive
check misses it entirely; ``find_names_needing_incoming_value`` catches it.

Known limitation: comprehension/generator-expression loop variables (``[x for x in xs]``) get a
``Store`` context for ``x`` in the AST just like a real assignment, and a comprehension's own scope
is invisible to this line-range-only walk -- ``find_names_needing_incoming_value`` will report such
loop variables as false positives ("needs incoming value") when their first appearance in the range
happens to be inside the comprehension's ``for`` clause rather than its own binding. Skim the report:
single/double-letter names (``x``, ``_i``, ``nm``) reused across multiple comprehensions are almost
always this false-positive class, not a real dependency -- cross-check against the source before
adding one to a new function's signature. Also does not track ``import ... as X`` bindings as
"defines X locally" in every code path (rare in practice for this use case, since a re-import inside
the extracted range is exactly the kind of self-contained code this tool is meant to flag as needing
no external binding).

Typical workflow when splitting a large function:
    1. Read the block of code you want to extract; note its start/end line numbers.
    2. ``python -m pyutilz.dev.freevar_analysis <file.py> <start> <end>`` (or call
       :func:`find_free_names` / :func:`find_names_needing_incoming_value` directly).
    3. Build the new function's signature from the free-name report, threading each one through as an
       explicit keyword argument (never re-import fresh inside the new function purely to avoid
       threading a parameter -- that risks a real behavioural difference, e.g. reintroducing an import
       cycle the original code broke deliberately via a lazy import).
    4. Move the block verbatim (same indentation depth if the new function sits at the same nesting
       level as the code it was extracted from -- a top-level function's body moving into another
       top-level function's body needs NO extra indentation, only the ``def``/docstring wrapper).
    5. Run ``ruff check`` (unrestricted, no narrow ``--select``) on both the new file and the file the
       block was removed from -- it reliably catches now-dead imports in the old file and genuinely
       undefined names in the new one (``F821``) that a hand read missed.
    6. Run the real test suite (or at least the tests covering the exact mechanism the block
       implements) before trusting the split; a clean ``ruff`` pass proves the names resolve, not
       that the behaviour is unchanged.
"""

from __future__ import annotations

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

# ----------------------------------------------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------------------------------------------


@dataclass
class IncomingNameUse:
    """One name that is assigned somewhere in the analysed range but whose first use there is a read.

    Attributes
    ----------
    name
        The identifier.
    first_load_lineno
        1-indexed source line of its first (read-before-write, within the range) occurrence.
    """

    name: str
    first_load_lineno: int


@dataclass
class FreeVarReport:
    """Result of :func:`analyze_range`: everything a line range of a Python file depends on from
    outside itself.

    Attributes
    ----------
    free_names
        Names Loaded within the range that are never Stored (assigned, or bound as a
        ``def``/``async def``) anywhere in the range -- unconditionally external dependencies.
    needs_incoming_value
        Names that ARE assigned somewhere in the range, but whose first occurrence (by source
        position) is a Load -- the range reassigns them later but still needs a starting value.
        Includes comprehension-loop-variable false positives; see the module docstring.
    """

    free_names: list = field(default_factory=list)
    needs_incoming_value: list = field(default_factory=list)

    def all_dependency_names(self) -> "list[str]":
        """Every name (from both categories) that must be resolvable when the range is lifted into
        its own function -- the raw candidate list for that function's parameter signature."""
        return sorted({n for n in self.free_names} | {u.name for u in self.needs_incoming_value})


def analyze_range(path: Union[str, Path], start_line: int, end_line: int) -> FreeVarReport:
    """Compute the :class:`FreeVarReport` for the 1-indexed inclusive line range ``[start_line, end_line]``
    of the Python source file at ``path``.

    Parses the WHOLE file (not just the sliced range) so nested scopes, decorators, and multi-line
    expressions overlapping the range boundary still resolve to correct AST line numbers -- slicing the
    source text first and re-parsing it in isolation would break on any range that doesn't happen to
    start/end on a clean statement boundary with no external syntactic context (e.g. a range starting
    mid-``if``-block).
    """
    src = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))

    # name -> list of (lineno, col_offset, is_store) occurrences within the range, in source order.
    occurrences: dict = {}

    class _Visitor(ast.NodeVisitor):
        """Records every ``Name`` occurrence within ``[start_line, end_line]`` into ``occurrences``, keyed by identifier, as ``(lineno, col_offset, is_store)`` tuples in visitation order."""

        def visit_Name(self, node: ast.Name) -> None:
            """Record one ``ast.Name`` node's position and Store/Load-ness if it falls inside the analysed line range."""
            if start_line <= node.lineno <= end_line:
                is_store = isinstance(node.ctx, (ast.Store, ast.Del))
                occurrences.setdefault(node.id, []).append((node.lineno, node.col_offset, is_store))
            self.generic_visit(node)

        def _visit_def(self, node) -> None:
            """Record a ``def``/``async def``/``class`` statement's own name as a Store occurrence when its header line falls inside the analysed range."""
            if start_line <= node.lineno <= end_line:
                occurrences.setdefault(node.name, []).append((node.lineno, node.col_offset, True))
            self.generic_visit(node)

        visit_FunctionDef = _visit_def  # noqa: N815 - ast.NodeVisitor's own dispatch-method naming convention
        visit_AsyncFunctionDef = _visit_def  # noqa: N815
        visit_ClassDef = _visit_def  # noqa: N815

    _Visitor().visit(tree)

    free_names: list = []
    needs_incoming: list = []
    for name, occs in occurrences.items():
        occs.sort(key=lambda o: (o[0], o[1]))
        has_store = any(is_store for (_, _, is_store) in occs)
        first_is_store = occs[0][2]
        if not has_store:
            free_names.append(name)
        elif not first_is_store:
            needs_incoming.append(IncomingNameUse(name=name, first_load_lineno=occs[0][0]))

    return FreeVarReport(
        free_names=sorted(free_names),
        needs_incoming_value=sorted(needs_incoming, key=lambda u: u.name),
    )


def format_report(report: FreeVarReport, path: Union[str, Path], start_line: int, end_line: int) -> str:
    """Human-readable rendering of a :class:`FreeVarReport`, matching this module's CLI output."""
    lines = [f"=== {path}:{start_line}-{end_line} ==="]
    lines.append(f"Free (external) names referenced: {len(report.free_names)}")
    lines.extend(f"  {n}" for n in report.free_names)
    lines.append("")
    lines.append(
        f"Names STORED somewhere in range but whose FIRST use is a LOAD "
        f"(need an incoming value; comprehension-loop-variable false positives possible -- see module docstring): "
        f"{len(report.needs_incoming_value)}"
    )
    lines.extend(f"  {u.name}  (first load at line {u.first_load_lineno})" for u in report.needs_incoming_value)
    return "\n".join(lines)


def _main(argv: "list[str] | None" = None) -> int:
    """CLI entry point: ``python -m pyutilz.dev.freevar_analysis <file.py> <start_line> <end_line>``."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3:
        print("usage: python -m pyutilz.dev.freevar_analysis <file.py> <start_line> <end_line>", file=sys.stderr)  # noqa: T201 - CLI usage message
        return 2
    path, start_s, end_s = args
    report = analyze_range(path, int(start_s), int(end_s))
    print(format_report(report, path, int(start_s), int(end_s)))  # noqa: T201 - CLI's actual output
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

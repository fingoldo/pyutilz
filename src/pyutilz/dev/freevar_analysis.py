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

For the OTHER half of a monolith split -- moving whole top-level definitions out to a sibling module rather
than carving a block out of one function -- see :func:`split_out_module`. It reuses the analysis above as its
safety gate: a moved range that still reads a module-level name staying behind is exactly the ``NameError``
on-an-untaken-branch this module exists to prevent, so such a move is refused rather than written.

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
import builtins
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


# ----------------------------------------------------------------------------------------------------------------------------
# Module splitting
# ----------------------------------------------------------------------------------------------------------------------------


def _end_line(node: ast.AST) -> int:
    """A parsed node's last line. ``end_lineno`` is Optional on the node types only because the AST classes
    declare it so; every node produced by ``ast.parse`` carries one. Falling back explicitly rather than with
    ``or`` keeps a line number of 0 from being silently rewritten into the start line."""
    end = getattr(node, "end_lineno", None)
    return int(end) if end is not None else int(node.lineno)  # type: ignore[attr-defined]


def _top_level_span(tree: ast.Module, name: str) -> "tuple[int, int]":
    """The 1-indexed inclusive line range of the top-level statement defining ``name``, decorators included."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            return min([node.lineno, *(d.lineno for d in node.decorator_list)]), _end_line(node)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node.lineno, _end_line(node)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return node.lineno, _end_line(node)
    raise ValueError(f"{name!r} is not a top-level definition in this module")


def _top_level_bodies(src: str) -> "dict[str, str]":
    """Each top-level definition's own source text, keyed by the name it binds.

    This is the object :func:`split_out_module`'s identity check compares before and after a move. Comparing
    TEXT rather than a re-dumped AST is deliberate: an ``ast.unparse`` round trip normalises formatting and
    would hide exactly the accidental edit the check exists to catch.
    """
    lines = src.splitlines()
    out: "dict[str, str]" = {}
    for node in ast.parse(src).body:
        name = getattr(node, "name", None)
        if name is None and isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name is None and isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        if name is not None:
            out[name] = "\n".join(lines[node.lineno - 1 : _end_line(node)])
    return out


def split_out_module(source: Union[str, Path], target: Union[str, Path], first: str, last: str, *, apply: bool = True) -> "list[str]":
    """Move the contiguous run of top-level definitions from ``first`` to ``last`` into a sibling ``target``.

    The common remedy for a module past a project's size limit is "sibling module + named re-export", and the
    risk in doing it by hand is silent drift: a body edited mid-move, a comment carrying the measurement that
    justified the code left behind, or a module-level constant the moved code still reads. All three are
    addressed structurally here rather than by review.

    * CONTIGUOUS, by line range, so every comment and blank line BETWEEN the moved definitions travels with
      them. A scattered set of names cannot express that, and in a well-commented module the prose between
      two functions is often the part worth keeping most.
    * The moved range's external reads are checked with :func:`analyze_range`, then intersected with the
      module's OWN top-level bindings. Anything left is a module-level name the moved code reads that would
      stay behind, and the move is REFUSED rather than written -- that is the ``NameError`` on an untaken
      branch this module exists to prevent. Imports travel with the move, so they never count; parameters and
      locals of the moved definitions are not module-level names, so they never count either.
    * Every moved body is compared byte-for-byte, before and after; any difference refuses the write.

    ``source`` keeps a ``from .<target> import name as name`` block in place of the moved range, so existing
    imports keep resolving; the explicit ``as`` form is what makes those names re-exported rather than merely
    imported. Returns the moved names. With ``apply=False`` nothing is written and every check still runs.
    """
    source, target = Path(source), Path(target)
    # `open` rather than `Path.read_text(newline=...)`: that keyword is 3.13+, and newline="" is what keeps
    # the file's own line endings visible instead of being translated on the way in.
    with open(source, encoding="utf-8", newline="") as fh:
        raw = fh.read()
    newline = "\r\n" if "\r\n" in raw else "\n"
    flat = raw.replace("\r\n", "\n")
    lines = flat.split("\n")
    tree = ast.parse(flat)

    start = _top_level_span(tree, first)[0]
    while start - 1 >= 1 and lines[start - 2].lstrip().startswith("#"):
        start -= 1
    end = _top_level_span(tree, last)[1]
    if start > end:
        raise ValueError(f"{first!r} must appear before {last!r} in {source}")

    moved_src = "\n".join(lines[start - 1 : end])
    moved_names = list(_top_level_bodies(moved_src))
    if not moved_names:
        raise ValueError("the selected range defines no top-level name")

    import_spans = [(n.lineno, _end_line(n)) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    import_block = "\n".join("\n".join(lines[a - 1 : b]) for a, b in import_spans)
    # `asname if asname else name`, not `asname or name`: an empty alias is not a thing the parser produces,
    # but spelling the None case is what this repo's own default-via-or rule asks for.
    imported = {a.asname if a.asname else a.name.split(".")[0] for n in tree.body if isinstance(n, ast.Import) for a in n.names}
    imported |= {a.asname if a.asname else a.name for n in tree.body if isinstance(n, ast.ImportFrom) for a in n.names}

    # `analyze_range` answers "what does this range read from outside itself", which for a range of whole
    # top-level definitions includes their own PARAMETERS - external to the block, but bound by the `def`.
    # The question that actually decides this move is narrower: does the range read a name bound at MODULE
    # level that is staying behind? Intersecting with the module's own top-level bindings answers exactly
    # that, and parameters/locals fall out on their own because they are not module-level names.
    module_level = set(imported)
    for node in tree.body:
        name = getattr(node, "name", None)
        if name is not None:
            module_level.add(name)
        elif isinstance(node, ast.Assign):
            module_level.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            module_level.add(node.target.id)

    report = analyze_range(source, start, end)
    left_behind = sorted(
        n for n in report.all_dependency_names() if n in module_level and n not in imported and n not in moved_names and not hasattr(builtins, n)
    )
    if left_behind:
        raise ValueError(
            f"refusing to split {source}: the moved range reads {len(left_behind)} module-level name(s) that would stay behind "
            f"({', '.join(left_behind[:8])}). Widen the range to include them, or import them back explicitly."
        )

    header = (
        f'"""Split out of `{source.as_posix()}` to keep both modules under the size limit.\n\n'
        "Moved verbatim: every body was compared byte-for-byte against the pre-split file, and the moved range\n"
        "was checked to read no module-level name left behind - see\n"
        "`pyutilz.dev.freevar_analysis.split_out_module`.\n\n"
        f'See `{source.as_posix()}` for the surrounding context these definitions were written in.\n"""\n\n'
        "from __future__ import annotations\n\n"
    )
    new_src = header + import_block + "\n\n\n" + moved_src + "\n"
    reexport = (
        f"# Split out to `{target.stem}.py` for module size; re-exported by name so existing imports keep working.\n"
        f"from .{target.stem} import (\n" + "".join(f"    {n} as {n},\n" for n in sorted(moved_names)) + ")\n"
    )
    rest_src = "\n".join([*lines[: start - 1], reexport.rstrip("\n"), *lines[end:]])

    before = _top_level_bodies(flat)
    after = {**_top_level_bodies(rest_src), **_top_level_bodies(new_src)}
    drifted = [n for n in moved_names if before[n] != after.get(n)]
    if drifted:
        raise ValueError(f"refusing to split {source}: {len(drifted)} body/bodies changed during the move ({', '.join(drifted[:5])})")

    if apply:
        with open(target, "w", encoding="utf-8", newline="") as fh:
            fh.write(new_src.replace("\n", newline))
        with open(source, "w", encoding="utf-8", newline="") as fh:
            fh.write(rest_src.replace("\n", newline))
    return moved_names


def _main(argv: "list[str] | None" = None) -> int:
    """CLI entry point.

    ``python -m pyutilz.dev.freevar_analysis <file.py> <start_line> <end_line>`` reports free variables;
    ``python -m pyutilz.dev.freevar_analysis --split <source.py> <target.py> <first> <last> [--dry-run]``
    moves a contiguous run of top-level definitions out to a sibling module.
    """
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == "--split":
        dry = "--dry-run" in args
        rest = [a for a in args[1:] if a != "--dry-run"]
        if len(rest) != 4:
            print("usage: python -m pyutilz.dev.freevar_analysis --split <source.py> <target.py> <first> <last> [--dry-run]", file=sys.stderr)  # noqa: T201 - CLI usage message
            return 2
        try:
            moved = split_out_module(rest[0], rest[1], rest[2], rest[3], apply=not dry)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)  # noqa: T201 - CLI's actual output
            return 1
        print(f"{'would move' if dry else 'moved'} {len(moved)} top-level name(s) to {rest[1]}; every body verified byte-identical")  # noqa: T201 - CLI's actual output
        for name in moved:
            print(f"    {name}")  # noqa: T201 - CLI's actual output
        return 0
    if len(args) != 3:
        print("usage: python -m pyutilz.dev.freevar_analysis <file.py> <start_line> <end_line>", file=sys.stderr)  # noqa: T201 - CLI usage message
        return 2
    path, start_s, end_s = args
    report = analyze_range(path, int(start_s), int(end_s))
    print(format_report(report, path, int(start_s), int(end_s)))  # noqa: T201 - CLI's actual output
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

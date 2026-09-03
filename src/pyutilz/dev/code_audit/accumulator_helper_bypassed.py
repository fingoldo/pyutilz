"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _arg_names, _iter_py_files, _line_text, _read_src_lines, _safe_parse, _subscript_index

# --- a shared accumulator that some call sites write around ------------------------------------
#
# A module grows a helper -- `_inc_stat(name)`, `record(event)`, `_bump(counter)` -- that owns one
# structure and does something on the way in: registers the key, clamps the value, takes a lock,
# fires a callback. Most call sites use it. One or two write to the structure directly, and those
# skip whatever the helper does.
#
# In the audited codebase this was a production `KeyError`: `_inc_stat("overlapping_axis_skipped")`
# on a key the reset function never declared. The direct writers were the ones that had never gone
# through the helper's registration, and a green test suite said nothing because only one test
# built its stats from the real initialiser.
#
# Three conditions, each added because the version without it was wrong on real code:
#
# * the structure is SHARED -- an attribute or a module-level name, never a local. Accepting a
#   local made every scanner in this very package look like a bypass, because each appends to its
#   own `findings` list: 56 hits across three repositories, not one of them real.
# * the structure READS as an accumulator. Without that the rule named `_get_infonet_model` the
#   owner of `path` and a numba kernel cache the owner of `atomic`.
# * the helper writes the accumulator using ITS OWN PARAMETER as the key. That is what separates
#   `_inc_stat(name)` from an ordinary site writing `stats["dup"] += 1`. Without it the rule
#   picked whichever writer it saw first and called it the owner, reporting two functions that
#   both write `self.stats` with literal keys, where no helper exists at all.
#
# Setup functions (`__init__`, `reset`, `_reset_stats`) are never bypasses: building the structure
# is not writing around the accumulator that maintains it.

# Matched as whole `_`-separated SEGMENTS of the function name, never as substrings: as substrings
# `load`/`copy`/`init` exempted `upload_batch`, `download_page`, `reload_rows`, `payload_scan` and
# `recopy` -- ordinary methods, every one of them a possible bypass site.
_SETUP_NAMES = frozenset({"init", "reset", "clear", "setup", "load", "restore", "from", "new", "copy", "snapshot", "getstate", "setstate"})
_MUTATING_METHODS = frozenset({"append", "add", "extend", "update", "insert"})
_ACCUMULATOR_HINTS = ("stat", "count", "counter", "metric", "tally", "total", "histogram", "skipped", "errors", "failures", "seen")


def _mutated_structures(node: ast.AST, module_level: frozenset[str]) -> set[str]:
    """The SHARED structures this subtree writes to, by subscript or by a mutating method."""
    written: set[str] = set()
    for sub in ast.walk(node):
        target: ast.expr | None = None
        if isinstance(sub, ast.Subscript) and isinstance(sub.ctx, (ast.Store, ast.Del)):
            target = sub.value
        elif isinstance(sub, ast.AugAssign) and isinstance(sub.target, ast.Subscript):
            target = sub.target.value
        elif isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr in _MUTATING_METHODS:
            target = sub.func.value
        if isinstance(target, ast.Attribute):
            written.add(target.attr)
        elif isinstance(target, ast.Name) and target.id in module_level:
            written.add(target.id)
    return written


def _is_setup(name: str) -> bool:
    """Whether this function's job is to build the structure rather than accumulate into it."""
    return bool(_SETUP_NAMES & set(name.lower().split("_")))


def _module_level_names(tree: ast.Module) -> frozenset[str]:
    """Names assigned at module level -- the only bare names shared between functions."""
    return frozenset(target.id for stmt in tree.body if isinstance(stmt, ast.Assign) for target in stmt.targets if isinstance(target, ast.Name))


def _called_names(tree: ast.Module) -> set[str]:
    """Every name called anywhere in this module. A function nobody calls is not the way in."""
    called = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    called |= {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    return called


def _keys_by_parameter(func: ast.AST, structure: str) -> bool:
    """Whether this function writes the accumulator using one of its own parameters as the key."""
    if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    params = set(_arg_names(func))
    for sub in ast.walk(func):
        subscript: ast.Subscript | None = None
        if isinstance(sub, ast.Subscript) and isinstance(sub.ctx, ast.Store):
            subscript = sub
        elif isinstance(sub, ast.AugAssign) and isinstance(sub.target, ast.Subscript):
            subscript = sub.target
        if subscript is None:
            continue
        owner = subscript.value
        name = owner.attr if isinstance(owner, ast.Attribute) else getattr(owner, "id", "")
        if name != structure:
            continue
        index = _subscript_index(subscript)
        if isinstance(index, ast.Name) and index.id in params:
            return True
    return False


def _accumulates(node: ast.AST, structure: str) -> bool:
    """Whether this node ACCUMULATES into the structure, rather than assigning a fresh value.

    The helper's operation is `+=`. `self.stats["root_total_count"] = count` stores a measurement
    that was just computed; routing it through `_inc_stat` would be wrong, not safer. All four of
    this rule's surviving hits in one codebase were that shape, and narrowing to accumulation is
    the difference between a rule about a real bypass and a rule about any write at all.
    """
    if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Subscript):
        owner = node.target.value
        name = owner.attr if isinstance(owner, ast.Attribute) else getattr(owner, "id", "")
        return name == structure
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript):
        owner = node.targets[0].value
        name = owner.attr if isinstance(owner, ast.Attribute) else getattr(owner, "id", "")
        if name != structure:
            return False
        # `stats[k] = stats.get(k, 0) + 1` is `+=` spelled out, and is the same bypass.
        return any(
            isinstance(sub, ast.Attribute) and sub.attr == "get" and isinstance(sub.value, (ast.Attribute, ast.Name)) and (sub.value.attr if isinstance(sub.value, ast.Attribute) else sub.value.id) == structure
            for sub in ast.walk(node.value)
        )
    return False


def _is_test_file(rel: str) -> bool:
    """Whether this path is a test module, whose direct writes are fixture setup."""
    parts = rel.split("/")
    return any(part in {"tests", "test"} for part in parts[:-1]) or parts[-1].startswith("test_") or parts[-1].endswith("_test.py")


def _lines_under_a_lock(func: ast.AST) -> set[int]:
    """Line numbers inside a `with <something>lock:` block in this function.

    The helper this rule points at is typically nothing BUT a lock plus the write, so a direct
    write that already holds the lock has skipped nothing that matters. Without this the rule
    reported fifteen sites in one codebase, most of them inside `with self._ids_lock:`.
    """
    protected: set[int] = set()
    for node in ast.walk(func):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        # Only the context expressions themselves (`with self._ids_lock:`), and matched on whole
        # `_`-separated segments: scanning the whole subtree for the substring "lock" let
        # `with self._block_reader:`, `with blocking_section():` and `with clock_timer:` silence
        # genuine bypasses.
        names: set[str] = set()
        for item in node.items:
            for sub in ast.walk(item.context_expr):
                if isinstance(sub, ast.Attribute):
                    names.add(sub.attr)
                elif isinstance(sub, ast.Name):
                    names.add(sub.id)
        if not any({"lock", "mutex"} & set(name.lower().split("_")) for name in names):
            continue
        for stmt in node.body:
            for sub in ast.walk(stmt):
                line = getattr(sub, "lineno", None)
                if line is not None:
                    protected.add(line)
    return protected


def _owners_of_accumulators(parsed: list[tuple[str, ast.Module, list[str]]], called: set[str]) -> dict[str, list[tuple[str, str]]]:
    """Every function that keys an accumulator by one of its own parameters, per structure.

    A list, not one winner: a module routinely carries a batched sibling (`_inc_stat` and
    `_inc_stats`), and treating a second parameter-keyed writer as ambiguous discarded both.
    """
    owners: dict[str, list[tuple[str, str]]] = {}
    for rel, tree, _lines in parsed:
        module_level = _module_level_names(tree)
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if _is_setup(func.name) or func.name not in called:
                continue
            written = _mutated_structures(func, module_level)
            if len(written) != 1:
                continue
            structure = next(iter(written))
            if not any(hint in structure.lower() for hint in _ACCUMULATOR_HINTS):
                continue
            if _keys_by_parameter(func, structure):
                owners.setdefault(structure, []).append((func.name, rel))
    return owners


def _bypasses_in(
    rel: str,
    tree: ast.Module,
    src_lines: list[str],
    owners: dict[str, list[tuple[str, str]]],
) -> list[Finding]:
    """Every site in this module that accumulates into an owned structure without its helper."""
    findings: list[Finding] = []
    module_level = _module_level_names(tree)
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)) or _is_setup(func.name):
            continue
        protected = _lines_under_a_lock(func)
        for structure in sorted(_mutated_structures(func, module_level)):
            entry = owners.get(structure)
            # (file, name), not the bare name: a same-named helper in an unrelated module must not
            # suppress a genuine bypass here.
            if not entry or any(name == func.name and owner_file == rel for name, owner_file in entry):
                continue
            site = _first_bypass(func, structure, protected)
            if site is None:
                continue
            named = ", ".join(f"`{name}`" for name, _file in entry)
            owns, they, are, keys = ("own", "they", "are", "key") if len(entry) > 1 else ("owns", "it", "is", "keys")
            findings.append(
                Finding(
                    check="accumulator_helper_bypassed",
                    severity="P2",
                    file=rel,
                    line=site,
                    snippet=_line_text(src_lines, site),
                    detail=(
                        f"`{func.name}` accumulates into `{structure}` directly, while {named} "
                        f"({entry[0][1]}) {owns} it -- {they} {are} what {keys} the accumulator by an "
                        "argument. Whatever they do on the way in (registering the key, clamping, "
                        "locking) this write skips, and an unregistered key reached production as "
                        "a KeyError exactly this way. Route it through the helper, or say why "
                        "this site cannot."
                    ),
                )
            )
    return findings


def _first_bypass(func: ast.AST, structure: str, protected: set[int]) -> int | None:
    """The line of the first unprotected accumulation into ``structure`` in this function."""
    for sub in ast.walk(func):
        if not isinstance(sub, (ast.Assign, ast.AugAssign)) or not _accumulates(sub, structure):
            continue
        if sub.lineno not in protected:
            return sub.lineno
    return None


def scan_accumulator_helper_bypassed(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find a shared accumulator with a dedicated helper that other sites write around.

    The helper does something on the way in -- registers the key, clamps the value, takes a lock --
    and a direct write skips all of it. That is how `_inc_stat("overlapping_axis_skipped")` reached
    production as a `KeyError`: the key had never been declared, because the writers that would
    have declared it bypassed the helper.

    Package-wide in two passes, because a mixin splits the accumulator from its writers across
    files: `_inc_stat` in one mixin and the direct `self.stats[...] += ...` writes in its sibling.
    A per-file rule saw a helper with no bypasses and bypasses with no helper, and so reported
    nothing on the very defect it was written for.
    """
    parsed: list[tuple[str, ast.Module, list[str]]] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        parsed.append((py.relative_to(root).as_posix(), tree, _read_src_lines(py)))

    # Every name called ANYWHERE in the tree. Per-file was wrong: `_inc_stat` is defined in a mixin
    # and called only from its siblings, so its own file never mentions it and the helper was
    # skipped as uncalled -- on the exact defect this rule was written for.
    called: set[str] = set()
    for _rel, tree, _lines in parsed:
        called |= _called_names(tree)

    owners = _owners_of_accumulators(parsed, called)
    if not owners:
        return []

    findings: list[Finding] = []
    for rel, tree, src_lines in parsed:
        # A test arranging `c.stats["count_queries"] = 0` before asserting on it is building a
        # fixture, not bypassing an accumulator -- five of this rule's nine surviving hits were
        # exactly that, and reporting them would teach readers to skim the rest.
        if _is_test_file(rel):
            continue
        findings.extend(_bypasses_in(rel, tree, src_lines, owners))
    return findings

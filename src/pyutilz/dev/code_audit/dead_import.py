"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
import re
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text


def _alias_own_lineno(alias: ast.alias, node: "ast.Import | ast.ImportFrom", src_lines: list[str], claimed: set[int]) -> int:
    """The alias's own 1-based line number.

    ``ast.alias`` only gained ``lineno``/``col_offset`` in Python 3.10 (bpo-39235) -- on 3.8/3.9
    (this package's own supported floor, see pyproject.toml ``requires-python``), every alias in
    a multi-line ``from x import (a, b, c)`` block falls back to the SAME ``node.lineno`` with no
    fix possible via attribute access alone. Confirmed in CI on 3.8 (2026-08-04): that collision
    is exactly the bug the alias-own-line fix (see the caller's docstring) exists to prevent, so
    the fallback must not just silently reproduce it. Scan the statement's own source lines (node
    is Import/ImportFrom, which HAS had end_lineno since 3.8) for the first not-yet-claimed line
    whose text contains the alias's own name as a whole identifier -- ``claimed`` lets sibling
    aliases in the same block each get a distinct line even when one name is a substring of
    another (e.g. ``foo`` / ``foo_bar``), by ruling out lines already matched to an earlier alias.
    """
    own_lineno = getattr(alias, "lineno", None)
    if own_lineno is not None:
        return int(own_lineno)
    name = alias.name.split(".")[0]
    pattern = re.compile(r"\b" + re.escape(name) + r"\b")
    end_lineno = getattr(node, "end_lineno", node.lineno)
    for lineno in range(node.lineno, end_lineno + 1):
        if lineno in claimed:
            continue
        if pattern.search(_line_text(src_lines, lineno)):
            claimed.add(lineno)
            return lineno
    return node.lineno


def _imported_bindings(tree: ast.Module, src_lines: list[str]) -> list[tuple[str, int, str]]:
    """``(bound_name, lineno, display_source)`` for every top-level (module-scope) ``import
    X``/``from X import Y [as Z]`` binding. Star imports and ``__all__``-listed names are excluded
    entirely (a star import can't be usage-checked by name; ``__all__`` membership already IS the
    usage). Wildcard aliasing (``as _``) is skipped -- a leading-underscore alias is a conventional
    "explicitly discard" marker, not a name meant to be referenced. ``from __future__ import ...``
    is skipped entirely -- a compiler directive, not a name ever meant to be referenced, so a
    "never referenced" check is a category error for it, not a real finding. A line already
    carrying a ``# noqa`` comment (any code) is skipped too -- the codebase has already reviewed
    and explicitly exempted that import; re-flagging it is pure noise, not a new signal. Checked
    on BOTH the statement's own line and each alias's own line, since a multi-line
    ``from x import (\n    a,  # noqa\n)`` block's per-name noqa lands on the alias line, not
    the statement line.
    """
    out: list[tuple[str, int, str]] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            if "# noqa" in _line_text(src_lines, node.lineno):
                continue
            claimed: set[int] = set()
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                if bound.startswith("_"):
                    continue
                # A multi-line import block's own alias may carry its own per-name rationale
                # comment on a different physical line than node.lineno -- check both.
                alias_lineno = _alias_own_lineno(alias, node, src_lines, claimed)
                if "# noqa" in _line_text(src_lines, alias_lineno):
                    continue
                out.append((bound, alias_lineno, f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else "")))
        elif isinstance(node, ast.ImportFrom):
            if node.module is None or node.module == "__future__":
                continue
            if "# noqa" in _line_text(src_lines, node.lineno):
                continue
            claimed = set()
            for alias in node.names:
                if alias.name == "*":
                    continue
                bound = alias.asname or alias.name
                if bound.startswith("_"):
                    continue
                # A multi-line parenthesized import block puts a per-name rationale comment on
                # the ALIAS's own line, not the statement line -- check both, since this is the
                # codebase's actual convention for per-name-reviewed re-exports.
                alias_lineno = _alias_own_lineno(alias, node, src_lines, claimed)
                if "# noqa" in _line_text(src_lines, alias_lineno):
                    continue
                # Report the ALIAS's own line (not the enclosing statement's node.lineno):
                # a multi-line `from x import (a, b, c)` block has every alias share the SAME
                # node.lineno, so two independently-dead names in one such block previously
                # reported IDENTICAL (check, file, line) keys -- the baseline-diff harness keys
                # findings on exactly that tuple (see py_ci_shared.code_audit_meta._key), so the
                # second finding silently overwrote/masked the first in the results dict instead
                # of being a second, independently-reviewable finding. Confirmed in the wild
                # (2026-08-04): removing one dead name from such a block (via `# noqa`) caused a
                # SECOND, previously-hidden dead import in the same block to suddenly "appear" --
                # it was there all along, just masked by the key collision.
                out.append((bound, alias_lineno, f"from {node.module} import {alias.name}" + (f" as {alias.asname}" if alias.asname else "")))
    return out


def _names_referenced_in_file(tree: ast.Module, skip_import_lines: set[int]) -> set[str]:
    """Every bare ``Name`` reference in ``tree``, excluding the import statements themselves
    (``skip_import_lines``) so an import doesn't count as its own usage."""
    referenced: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.lineno not in skip_import_lines:
            referenced.add(node.id)
    # `__all__ = ["name", ...]` string-literal entries count as usage (an explicit re-export).
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            if isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
                for elt in node.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        referenced.add(elt.value)
    return referenced


def _attribute_names_in_corpus(root: Path, exclude_dirs: frozenset[str]) -> set[str]:
    """Every ``.attr`` name accessed anywhere (``obj.attr``) across the whole corpus -- the
    cheap, repo-wide "is this name EVER accessed as an attribute" signal used to avoid flagging a
    facade re-export (``from module_b import some_name`` in module_a, only ever consumed
    elsewhere as ``module_a.some_name``) as a dead import. Deliberately not import-alias-resolved
    (no attempt to confirm the attribute access is on THIS SPECIFIC module) -- a cheap, safe-by-
    construction (favors false negatives, never manufactures a finding) whole-corpus scan."""
    names: set[str] = set()
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                names.add(node.attr)
    return names


def _from_import_names_by_file(root: Path, exclude_dirs: frozenset[str]) -> dict[Path, set[str]]:
    """For every file, the set of names it pulls in via ``from <anything> import name`` (star
    imports excluded). Used to detect a name re-exported from a facade ``__init__.py`` and
    consumed by a DIFFERENT file doing ``from that_facade import name`` -- a shape invisible to
    ``_attribute_names_in_corpus`` (the name never appears as ``X.name`` anywhere, only as the
    ``from`` import's own target list) and invisible to the per-file bare-name check (the
    consuming reference lives in a different file entirely). Confirmed as a real false-positive
    class: mlframe package ``__init__.py`` facades re-exporting via ``from .submodule import
    Name`` for downstream ``from mlframe.pkg import Name`` consumers were flagged wholesale."""
    by_file: dict[Path, set[str]] = {}
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name != "*":
                        names.add(alias.name)
        by_file[py] = names
    return by_file


def _is_consumed_by_another_files_from_import(name: str, this_file: Path, from_import_names_by_file: dict[Path, set[str]]) -> bool:
    """True if some file OTHER than ``this_file`` does ``from <anything> import name``. Excludes
    ``this_file`` itself so the import binding being checked can't trivially "consume itself" by
    virtue of being a ``from`` import in the first place."""
    return any(name in names for other, names in from_import_names_by_file.items() if other != this_file)


def scan_possibly_dead_import(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find a module-level ``import X``/``from X import Y`` binding that's never referenced as a
    bare name anywhere else in its own file, AND never accessed as ``<anything>.Y`` anywhere in
    the whole scanned corpus -- a genuinely unused import, not a facade re-export consumed
    indirectly elsewhere.

    A naive "grep the bare name within this one file" dead-import check has a real false-positive
    class: a name imported purely to be re-exported through a facade module is legitimately
    unused WITHIN its own file, but IS consumed elsewhere via ``facade_module.name`` attribute
    access -- indistinguishable from a bare-name-only scan. Confirmed in the wild during the
    2026-07-21 full-audit (production_scrapers): 9 of 12 audit-suspected "dead" re-exports in
    ``bruteforce_cipher.py`` turned out to be live, consumed via ``import bruteforce_cipher as b;
    b.<name>`` in test files -- only found by hand-grepping every test file for that indirect
    attribute-access pattern, which is exactly what this scanner automates.

    Whole-corpus attribute-access matching is NOT import-alias-resolved (doesn't confirm the
    ``.name`` access is actually on THIS module) -- a deliberately conservative, cheap heuristic:
    a same-named attribute ANYWHERE in the corpus suppresses the finding. False negatives (a
    genuinely dead import sharing its name with an unrelated attribute elsewhere) are the safe
    failure mode; per ``feedback_no_padding_parametric_pins``, this scanner never manufactures a
    finding through guesswork.

    Severity: Low (a real but low-urgency hygiene finding -- an unused import costs nothing at
    runtime; the risk it guards against is reader confusion and import-time side-effect surprise,
    not a correctness bug).
    """
    findings: list[Finding] = []
    corpus_attr_names = _attribute_names_in_corpus(root, exclude_dirs)
    from_import_names_by_file = _from_import_names_by_file(root, exclude_dirs)

    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        bindings = _imported_bindings(tree, src_lines)
        if not bindings:
            continue
        skip_lines = {lineno for _, lineno, _ in bindings}
        referenced = _names_referenced_in_file(tree, skip_lines)
        rel = py.relative_to(root).as_posix()

        for name, lineno, display in bindings:
            if name in referenced:
                continue
            if name in corpus_attr_names:
                continue  # possibly a facade re-export consumed as <module>.name elsewhere
            if _is_consumed_by_another_files_from_import(name, py, from_import_names_by_file):
                continue  # a facade re-export consumed via `from this_module import name` elsewhere
            findings.append(Finding(
                check="possibly_dead_import",
                severity="Low",
                file=rel,
                line=lineno,
                snippet=_line_text(src_lines, lineno),
                detail=(
                    f"'{display}' binds '{name}', never referenced as a bare name in this file and never "
                    f"accessed as '<anything>.{name}' anywhere in the scanned corpus -- looks genuinely "
                    f"unused, not a facade re-export. Remove it, or if it IS consumed indirectly in a way "
                    f"this scan can't see (e.g. via getattr(), a string-keyed dispatch table, or an "
                    f"external consumer outside this scan root), leave a comment explaining why."
                ),
            ))
    return findings

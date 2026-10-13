"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import walk_tree, Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _line_text, _read_src_lines, _safe_parse

# --- custom exception class defined but never raised -----------------------
#
# Confirmed real dead-contract in the 2026-07-21 audit: `LLMTruncationError` in
# `llm/exceptions.py` was fully specified (docstring: "Retryable -- caller should double
# max_tokens and re-issue", a `finish_reason` field to support that), but zero `raise
# LLMTruncationError(...)` call sites existed anywhere in the 7 provider implementations --
# callers who wrote `except LLMTruncationError: double_budget_and_retry()` (the exact pattern
# the docstring recommends) never saw it fire, even on a genuine max-tokens cutoff.
#
# Repo-wide (not single-file) by construction: an exception class's raise sites are commonly in
# a DIFFERENT file (or several) than its definition.


def _exception_base_names(cls: ast.ClassDef) -> set[str]:
    """Return the names of ``cls``'s direct base classes (Name and Attribute forms)."""
    names = set()
    for base in cls.bases:
        if isinstance(base, ast.Name):
            names.add(base.id)
        elif isinstance(base, ast.Attribute):
            names.add(base.attr)
    return names


# Built-in/stdlib exception base names -- a class inheriting (possibly transitively, but this
# scanner only checks direct bases to stay simple) from one of these OR from another class
# already known to be an exception is considered an exception class.
_KNOWN_EXCEPTION_BASES = frozenset({
    "Exception", "BaseException", "ValueError", "TypeError", "RuntimeError", "KeyError",
    "AttributeError", "OSError", "IOError", "LookupError", "ArithmeticError", "NotImplementedError",
    "StopIteration", "ImportError", "NameError",
})


def _parse_trees(root: Path, exclude_dirs: frozenset[str]) -> tuple[list[tuple[Path, ast.Module]], dict[str, list[str]]]:
    """Parse every scanned file, returning ``(path, tree)`` pairs and each file's source lines keyed by relative path."""
    trees: list[tuple[Path, ast.Module]] = []
    per_file_lines: dict[str, list[str]] = {}
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        trees.append((py, tree))
        per_file_lines[py.relative_to(root).as_posix()] = _read_src_lines(py)
    return trees, per_file_lines


def _collect_class_defs(root: Path, trees: list[tuple[Path, ast.Module]]) -> list[tuple[ast.ClassDef, str]]:
    """Return every ``ClassDef`` in the scanned trees paired with its relative file path."""
    class_defs: list[tuple[ast.ClassDef, str]] = []
    for py, tree in trees:
        rel = py.relative_to(root).as_posix()
        for node in walk_tree(tree):
            if isinstance(node, ast.ClassDef):
                class_defs.append((node, rel))  # noqa: PERF401 -- isinstance-narrowed append; a list.extend genexpr loses mypy's narrowing here
    return class_defs


def _resolve_exception_class_names(class_defs: list[tuple[ast.ClassDef, str]]) -> set[str]:
    """Return the known builtin exception names plus every class in the tree that (transitively) subclasses one."""
    exception_class_names: set[str] = set(_KNOWN_EXCEPTION_BASES)
    # Iterate twice to catch exception classes that subclass ANOTHER custom exception class defined elsewhere in the tree.
    for _pass in range(2):  # second pass catches subclasses-of-subclasses defined in this repo
        for node, _rel in class_defs:
            if node.name in exception_class_names:
                continue
            if _exception_base_names(node) & exception_class_names:
                exception_class_names.add(node.name)
    return exception_class_names


def _collect_exception_classes(
    class_defs: list[tuple[ast.ClassDef, str]],
    exception_class_names: set[str],
    per_file_lines: dict[str, list[str]],
) -> dict[tuple[str, str], tuple[str, int, list[str]]]:
    """Map each custom exception class, keyed by ``(file, name)``, to its file, line number and file source lines."""
    # Keyed by (file, name), NOT by bare name: two modules each defining their own `DupError`
    # (ordinary in provider/plugin layouts, which this scanner's own header cites) collapsed onto
    # one entry, so only the last one scanned could ever be reported.
    all_classes: dict[tuple[str, str], tuple[str, int, list[str]]] = {}
    for node, rel in class_defs:
        if node.name in _KNOWN_EXCEPTION_BASES:
            continue
        if _exception_base_names(node) & exception_class_names:
            all_classes[(rel, node.name)] = (rel, node.lineno, per_file_lines[rel])
    return all_classes


def _collect_raised_names(trees: list[tuple[Path, ast.Module]]) -> set[str]:
    """Return the name of every exception raised anywhere in the scanned trees."""
    raised_names: set[str] = set()
    for _py, tree in trees:
        for node in walk_tree(tree):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            exc = node.exc
            if isinstance(exc, ast.Call):
                exc = exc.func
            if isinstance(exc, ast.Name):
                raised_names.add(exc.id)
            elif isinstance(exc, ast.Attribute):
                raised_names.add(exc.attr)
    return raised_names


def _covered_via_subclass(
    all_classes: dict[tuple[str, str], tuple[str, int, list[str]]],
    class_defs: list[tuple[ast.ClassDef, str]],
    raised_names: set[str],
) -> set[str]:
    """Return the class names whose in-tree subclass is raised (directly or through another covered subclass)."""
    # A base class whose contract fires whenever ANY of its in-tree
    # subclasses is raised -- `except BaseError:` still catches a raised
    # `SpecificError(BaseError)` -- so an ancestor should not be flagged just
    # because nothing raises IT directly. Confirmed real pattern (2026-08-03
    # glossum unraised_exception_class audit): `GlossumError` is a pure base
    # (only `GenerationError`/`ValidationError`/etc subclasses carry meaning),
    # and `ProviderError` is only ever raised via its own subclass
    # `LLMProviderError` -- both false-flagged despite their documented
    # error-signaling contract firing correctly through the subclass. Direct
    # bases only (mirrors `_exception_base_names`'s one-level scope above);
    # transitivity still works because this loop is a fixed-point iteration.
    base_to_subclasses: dict[str, set[str]] = {}
    for node, _rel in class_defs:
        for base_name in _exception_base_names(node):
            base_to_subclasses.setdefault(base_name, set()).add(node.name)

    covered_via_subclass: set[str] = set()
    changed = True
    while changed:
        changed = False
        for _rel_key, name in all_classes:
            if name in covered_via_subclass:
                continue
            subclasses = base_to_subclasses.get(name, set())
            if any(sub in raised_names or sub in covered_via_subclass for sub in subclasses):
                covered_via_subclass.add(name)
                changed = True
    return covered_via_subclass


def scan_unraised_exceptions(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find custom exception classes (direct or one-level-transitive subclasses of a known
    builtin exception, or of another custom exception class ALSO defined in the scanned tree)
    that are never ``raise``d anywhere in the scanned tree.

    Repo-wide: collects every ``ClassDef``+every ``Raise`` across ALL scanned files before
    computing the difference, since a class's raise sites are commonly in a different file than
    its definition (provider/plugin architectures especially).

    Severity: P2 (a fully-specified error-signaling contract that silently never fires --
    callers relying on ``except SpecificError:`` for the documented behavior never see it).
    """
    trees, per_file_lines = _parse_trees(root, exclude_dirs)
    class_defs = _collect_class_defs(root, trees)
    exception_class_names = _resolve_exception_class_names(class_defs)
    all_classes = _collect_exception_classes(class_defs, exception_class_names, per_file_lines)
    raised_names = _collect_raised_names(trees)
    covered_via_subclass = _covered_via_subclass(all_classes, class_defs, raised_names)

    findings: list[Finding] = []
    for (_key_rel, name), (rel, lineno, src_lines) in all_classes.items():
        if name in raised_names or name in covered_via_subclass:
            continue
        findings.append(Finding(
            check="unraised_exception_class",
            severity="P2",
            file=rel,
            line=lineno,
            snippet=_line_text(src_lines, lineno),
            detail=(
                f"`class {name}` is a custom exception, but `raise {name}(...)` appears nowhere "
                "in the scanned tree -- its documented error-signaling contract never fires. "
                "Either wire it in where the condition it describes actually occurs, or remove it "
                "if it's not meant to be load-bearing yet."
            ),
        ))
    return findings

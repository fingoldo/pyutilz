"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

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

    Severity: Medium (a fully-specified error-signaling contract that silently never fires --
    callers relying on ``except SpecificError:`` for the documented behavior never see it).
    """
    all_classes: dict[str, tuple[str, int, list[str]]] = {}  # name -> (file, line, src_line)
    exception_class_names: set[str] = set(_KNOWN_EXCEPTION_BASES)
    raised_names: set[str] = set()
    per_file_lines: dict[str, list[str]] = {}

    trees: list[tuple[Path, ast.Module]] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        trees.append((py, tree))
        per_file_lines[py.relative_to(root).as_posix()] = py.read_text(encoding="utf-8", errors="replace").splitlines()

    # Pass 1: collect every ClassDef (name -> is it exception-like) -- iterate twice to catch
    # exception classes that subclass ANOTHER custom exception class defined elsewhere in the tree.
    class_defs: list[tuple[ast.ClassDef, str]] = []
    for py, tree in trees:
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_defs.append((node, rel))  # noqa: PERF401 -- isinstance-narrowed append; a list.extend genexpr loses mypy's narrowing here
    for _pass in range(2):  # second pass catches subclasses-of-subclasses defined in this repo
        for node, _rel in class_defs:
            if node.name in exception_class_names:
                continue
            if _exception_base_names(node) & exception_class_names:
                exception_class_names.add(node.name)

    for node, rel in class_defs:
        if node.name in _KNOWN_EXCEPTION_BASES:
            continue
        if _exception_base_names(node) & exception_class_names:
            all_classes[node.name] = (rel, node.lineno, per_file_lines[rel])

    # Pass 2: collect every raised name across the whole tree.
    for _py, tree in trees:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            exc = node.exc
            if isinstance(exc, ast.Call):
                exc = exc.func
            if isinstance(exc, ast.Name):
                raised_names.add(exc.id)
            elif isinstance(exc, ast.Attribute):
                raised_names.add(exc.attr)

    findings: list[Finding] = []
    for name, (rel, lineno, src_lines) in all_classes.items():
        if name in raised_names:
            continue
        findings.append(Finding(
            check="unraised_exception_class",
            severity="Medium",
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

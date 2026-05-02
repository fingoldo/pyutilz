"""Reusable building blocks for package-level meta-tests.

A *meta-test* polices a structural property of a Python package — config
field consumption parity, alias integrity, dead-helper detection,
public-API stability — without exercising runtime behaviour. Most of
them share the same plumbing:

  * Walk every production ``.py`` under the package, excluding tests
    and caches.
  * Concatenate file contents into a single corpus string for fast
    grep-style scans.
  * Discover top-level public symbols via the AST.
  * Capture a signature shape for comparing public APIs across runs.
  * Scan ``TODO`` / ``FIXME`` markers and check attribution.
  * Count entries in `_USER_DEFERRED_*` whitelists for drift tracking.

This module factors those helpers out so every project that wants
meta-tests imports them rather than re-pasting ~400 LOC of boilerplate.
None of these functions depend on Pydantic, pytest fixtures, or any
particular framework — they accept a ``package_dir: Path`` and operate
on filesystem state.

Used by:
  * pyutilz's own ``tests/test_meta/`` (PT-1..PT-9)
  * mlframe's ``tests/test_meta/`` (Ext #0..#7, MT-1..MT-7)
  * downstream projects that want the same suite without copy-paste

The accompanying drift tracker (``count_user_deferred_entries``) reads
``_USER_DEFERRED_*`` set/dict literals out of meta-test files via the
AST so the count survives reformatting.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
from pathlib import Path
from typing import Iterable, Optional

__all__ = [
    "consumer_corpus",
    "enumerate_test_files",
    "public_top_level_symbols",
    "strip_lineno",
    "capture_signature",
    "capture_module_surface",
    "scan_todo_markers",
    "count_user_deferred_entries",
    "snake_case_variants_of",
    "MARKER_LINE_RE",
    "ATTRIBUTION_RE",
]

# ---------------------------------------------------------------------------
# Corpus utilities
# ---------------------------------------------------------------------------

# Path fragments excluded from "production corpus" scans by default.
# Project-specific tests can pass extra fragments via the `extra_excludes`
# kwarg (e.g. mlframe excludes ``legacy/`` whose contents are pre-sunset).
_DEFAULT_EXCLUDED_FRAGMENTS = ("test", "tests", "__pycache__")


def consumer_corpus(
    package_dir: Path,
    *,
    extra_excludes: Iterable[str] = (),
    exclude_files: Iterable[Path] = (),
) -> str:
    """Concatenate every production ``.py`` under ``package_dir``.

    Parameters
    ----------
    package_dir
        The directory to walk recursively. For pyutilz that's
        ``Path(pyutilz.__file__).parent``; for mlframe (a flat-layout
        package), the same expression — the package IS the repo root.
    extra_excludes
        Additional path fragments to skip (e.g. ``"legacy"``,
        ``"profiling"``, ``"explore"``).
    exclude_files
        Additional file paths to skip outright. Use this to exclude the
        config / module under audit from its own corpus (a class
        referenced only inside its own definition is still "unused").

    Returns
    -------
    str
        Newline-separated concatenation of every readable .py file.

    Notes
    -----
    The function silently skips files that fail to read (OSError) or
    contain non-utf-8 sequences (UnicodeDecodeError). Either case
    means "no signal here", not "test broken".
    """
    excluded_fragments = set(_DEFAULT_EXCLUDED_FRAGMENTS) | set(extra_excludes)
    excluded_resolved = {p.resolve() for p in exclude_files}
    chunks: list[str] = []
    for py in package_dir.rglob("*.py"):
        if py.resolve() in excluded_resolved:
            continue
        if any(frag in py.parts for frag in excluded_fragments):
            continue
        try:
            chunks.append(py.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue
    return "\n".join(chunks)


# ---------------------------------------------------------------------------
# Test-file discovery
# ---------------------------------------------------------------------------


def enumerate_test_files(tests_dir: Path) -> set[str]:
    """All ``test_*.py`` basenames (no extension) under ``tests_dir``,
    recursively. Used for parity tests that match a source module name
    against existing test files."""
    out: set[str] = set()
    if not tests_dir.exists():
        return out
    for py in tests_dir.rglob("test_*.py"):
        out.add(py.stem)
    return out


# ---------------------------------------------------------------------------
# AST-based symbol discovery
# ---------------------------------------------------------------------------


def public_top_level_symbols(path: Path) -> list[tuple[str, int]]:
    """Top-level ``def``/``async def``/``class`` defined in a .py file
    whose name does NOT start with an underscore.

    Returns ``[(name, lineno), ...]``. Returns ``[]`` for files that
    fail to read or fail to parse — same "no signal" convention as
    ``consumer_corpus``.
    """
    try:
        src = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    out: list[tuple[str, int]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                out.append((node.name, node.lineno))
    return out


def strip_lineno(entry: str) -> str:
    """``"path/file.py:42::name"`` → ``"path/file.py::name"`` with
    forward-slash normalisation. Used by tests that grandfather
    findings via a set keyed on path-and-name (so renumbering after
    nearby edits doesn't break the whitelist)."""
    return re.sub(r":\d+::", "::", entry).replace("\\", "/")


# ---------------------------------------------------------------------------
# Signature / class shape capture (for API-stability snapshots)
# ---------------------------------------------------------------------------


def capture_signature(obj) -> str:
    """Stringify a callable's signature in a form stable across
    refactors that don't change semantics.

    Captures parameter name, kind (positional / keyword / var), and
    has-default. Annotations are intentionally omitted because they
    churn under harmless type-narrowing refactors. Failures (LocalProxy,
    builtins without inspectable signatures, etc.) yield
    ``"<no-signature>"``.
    """
    try:
        sig = inspect.signature(obj)
    except (ValueError, TypeError, RuntimeError):
        return "<no-signature>"
    params = []
    for name, p in sig.parameters.items():
        kind_short = {
            inspect.Parameter.POSITIONAL_ONLY: "pos",
            inspect.Parameter.POSITIONAL_OR_KEYWORD: "any",
            inspect.Parameter.VAR_POSITIONAL: "*args",
            inspect.Parameter.KEYWORD_ONLY: "kw",
            inspect.Parameter.VAR_KEYWORD: "**kw",
        }.get(p.kind, str(p.kind))
        has_default = p.default is not inspect.Parameter.empty
        params.append(f"{name}:{kind_short}:{int(has_default)}")
    return "(" + ", ".join(params) + ")"


def capture_module_surface(mod) -> dict[str, str]:
    """Return ``{public_name: kind_string}`` for every public symbol
    on a module (skips dunders + private). Used by API-stability
    snapshot tests.

    Filters out symbols imported from outside pyutilz (typing aliases,
    stdlib helpers like ``os.path.join``) so the snapshot is stable
    across Python versions. Python 3.11 promoted ``typing.Any`` to a
    real class while 3.10 keeps it as ``_SpecialForm``; capturing such
    re-exports makes the snapshot version-sensitive without testing any
    pyutilz API.

    Robust against import-time-bound symbols (Flask LocalProxy etc.)
    — they're reported as ``"<inaccessible-at-import-time>"`` rather
    than crashing the snapshot.
    """
    mod_name = getattr(mod, "__name__", "")
    out: dict[str, str] = {}
    for name in dir(mod):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(mod, name)
        except Exception:
            out[name] = "<inaccessible-at-import-time>"
            continue
        if inspect.ismodule(obj):
            continue
        # Skip symbols re-exported from outside pyutilz; their behaviour
        # (Any class-vs-callable, signature param names) varies across
        # Python versions and isn't part of our public API.
        owner = getattr(obj, "__module__", None)
        if owner and not owner.startswith("pyutilz"):
            # ``value:`` symbols (None / int / str module-level
            # constants) have no ``__module__`` attribute on the value
            # itself; ``getattr(None, '__module__', None)`` is None →
            # they fall through to the ``else`` branch below. Only
            # drop class/callable re-exports.
            if inspect.isclass(obj) or callable(obj):
                continue
        if inspect.isclass(obj):
            out[name] = f"class:{getattr(obj, '__module__', '?')}.{obj.__name__}"
        elif callable(obj):
            out[name] = "callable" + capture_signature(obj)
        else:
            out[name] = f"value:{type(obj).__name__}"
    return out


# ---------------------------------------------------------------------------
# TODO / FIXME / XXX / HACK marker scanning
# ---------------------------------------------------------------------------


MARKER_LINE_RE = re.compile(
    r"#\s*(?P<kw>TODO|FIXME|XXX|HACK)\b(?P<rest>.*)$",
    re.IGNORECASE,
)

# An attribution is one of:
#   * (assignee) — anything in parens immediately after the keyword
#   * 2026-04-28 — ISO date anywhere in the comment
#   * @assignee  — at-mention anywhere in the comment
ATTRIBUTION_RE = re.compile(
    r"(\([^)]+\)|\d{4}-\d{2}-\d{2}|@\w+)"
)


def scan_todo_markers(
    package_dir: Path,
    *,
    extra_excludes: Iterable[str] = (),
) -> list[tuple[Path, int, str, str]]:
    """Walk every production .py file under ``package_dir`` and yield
    every ``# TODO`` / ``# FIXME`` / ``# XXX`` / ``# HACK`` line.

    Returns ``[(path, lineno, marker_kw_uppercase, full_line_stripped)]``.
    """
    excluded_fragments = set(_DEFAULT_EXCLUDED_FRAGMENTS) | set(extra_excludes)
    out: list[tuple[Path, int, str, str]] = []
    for py in package_dir.rglob("*.py"):
        if any(frag in py.parts for frag in excluded_fragments):
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            m = MARKER_LINE_RE.search(line)
            if not m:
                continue
            kw = m.group("kw").upper()
            out.append((py, lineno, kw, line.strip()))
    return out


# ---------------------------------------------------------------------------
# Drift tracking — count `_USER_DEFERRED_*` whitelist entries
# ---------------------------------------------------------------------------


_DEFERRED_PREFIXES = ("_USER_DEFERRED", "_GRANDFATHERED")


def count_user_deferred_entries(
    test_meta_dir: Path,
    *,
    extra_prefixes: Iterable[str] = (),
) -> dict[str, int]:
    """Count entries in every ``_USER_DEFERRED_*`` (or other listed
    prefix) set/dict literal across every meta-test file in
    ``test_meta_dir``.

    The drift tracker compares this dict to a stored baseline; growth
    means tech debt is accumulating without anyone noticing.

    Returns
    -------
    dict[str, int]
        ``{f"<test_file_stem>::<var_name>": count}`` — e.g.
        ``{"test_dead_helpers::_USER_DEFERRED_DEAD_HELPERS": 9}``.
    """
    prefixes = tuple(_DEFERRED_PREFIXES) + tuple(extra_prefixes)
    out: dict[str, int] = {}
    if not test_meta_dir.exists():
        return out
    for py in sorted(test_meta_dir.rglob("test_*.py")):
        try:
            src = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            # Both ``X = {...}`` (Assign) and ``X: set[str] = {...}``
            # (AnnAssign) need to be matched — modern code typically
            # uses the annotated form for whitelists.
            if isinstance(node, ast.Assign):
                targets = node.targets
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target] if node.target else []
                value = node.value
            else:
                continue
            if value is None:  # type: ignore[unreachable]
                continue
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                if not target.id.startswith(prefixes):
                    continue
                key = f"{py.stem}::{target.id}"
                if isinstance(value, (ast.Set, ast.List, ast.Tuple)):
                    out[key] = len(value.elts)
                elif isinstance(value, ast.Dict):
                    out[key] = len(value.keys)
                elif isinstance(value, ast.Call):
                    # ``set()`` / ``dict()`` empty constructor.
                    out[key] = 0
    return out


# ---------------------------------------------------------------------------
# Class-name → variable-name heuristics (for the model_dump splat detector)
# ---------------------------------------------------------------------------


def snake_case_variants_of(cls_name: str) -> set[str]:
    """Heuristic snake-case names a Pydantic instance of ``cls_name``
    is likely bound to in calling code.

    Examples::

        "TrainingBehaviorConfig" → {"training_behavior_config",
                                    "training_behavior",
                                    "behavior_config"}
        "ModelHyperparamsConfig" → {"model_hyperparams_config",
                                    "model_hyperparams",
                                    "hyperparams_config"}

    Used by the ``model_dump()``-splat detector across multiple
    projects so the regex isn't re-implemented per repo.
    """
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", cls_name).lower()
    short = snake.replace("_config", "")
    candidates = {snake, short}
    parts = short.split("_")
    if parts:
        candidates.add(parts[-1] + "_config")
    return candidates


# ---------------------------------------------------------------------------
# Module-import probing (used by alias-integrity tests)
# ---------------------------------------------------------------------------


def safe_import(module_path: str) -> Optional[object]:
    """``importlib.import_module`` that returns ``None`` instead of
    raising ``ImportError``. Lets meta-tests aggregate failures rather
    than aborting on the first broken module."""
    try:
        return importlib.import_module(module_path)
    except ImportError:
        return None

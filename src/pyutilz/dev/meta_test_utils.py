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
import dataclasses
import importlib
import inspect
import re
import types
import typing
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Optional


def _rglob_test_files(root: Path) -> "typing.Iterator[Path]":
    """``root.rglob("test_*.py")``, skipping `_DEFAULT_EXCLUDE_DIRS` - most pointedly `.claude`, which holds
    Claude Code's agent worktrees, each a COMPLETE checkout of the repo being scanned. Without this, a
    caller passed the repo ROOT (rather than a narrow `tests/` subtree) walks every worktree's own test
    suite too: measured on autopsia with 30 worktrees present, `unbacked_audit_dispositions` concatenated
    7,989 duplicate test files into its search corpus instead of the real 295, at ~810s instead of a few.

    The `code_audit._base` import is DEFERRED inside the function rather than at module top-level: this
    module (`meta_test_utils`) is itself imported BY `pyutilz.dev.code_audit.__init__` (for
    `todo_hygiene.py`'s `scan_todo_markers`), so a top-level import back into `code_audit` is circular -
    `ImportError: cannot import name 'ATTRIBUTION_RE' from partially initialized module`, measured live.
    Deferring costs nothing here since this helper is never on a hot per-call path itself.
    """
    from pyutilz.dev.code_audit._base import _DEFAULT_EXCLUDE_DIRS

    for p in root.rglob("test_*.py"):
        if not any(part in _DEFAULT_EXCLUDE_DIRS for part in p.relative_to(root).parts[:-1]):
            yield p


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
    "safe_import",
    "sentinel_for_type",
    "optional_scalar_fields",
    "assert_fields_roundtrip",
    "MARKER_LINE_RE",
    "ATTRIBUTION_RE",
    "findings_ratchet",
    "unbacked_audit_dispositions",
    "clear_repo_scan_caches",
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
    for py in _rglob_test_files(tests_dir):
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


_SAFE_DEFAULT_TYPES = (bool, int, float, str, type(None))


def _safe_default_repr(default: object) -> str:
    """Return a stable ``repr()`` for a parameter default, for the "safe to serialize" types
    (bool/int/float/str/None, and tuples thereof) -- else a fixed placeholder.

    Arbitrary objects/sentinels (a module-level ``_UNSET = object()``, a mutable default, a
    class instance) often ``repr()`` with a memory address or other unstable identity, which
    would make the API snapshot flap on every run even though the actual default never changed.
    Restricting to literal-like types keeps the capture deterministic while still catching the
    common, highest-risk case: a bool/int/float/str/None default silently flipping.
    """
    if isinstance(default, _SAFE_DEFAULT_TYPES):
        return repr(default)
    if isinstance(default, tuple) and all(isinstance(v, _SAFE_DEFAULT_TYPES) for v in default):
        return repr(default)
    return "<unstable-default>"


def capture_signature(obj: object) -> str:
    """Stringify a callable's signature in a form stable across
    refactors that don't change semantics.

    Captures parameter name, kind (positional / keyword / var), has-default, and -- for
    "safe to serialize" default types (bool/int/float/str/None/tuple-of-those) -- the default
    value itself (see ``_safe_default_repr``). Regression fix (2026-07-21 audit round 2, HIGH):
    the has-default flag alone can't distinguish ``def f(verbose=False)`` from
    ``def f(verbose=True)`` -- both captured as the byte-identical ``"(verbose:any:1)"``, so a
    silent default-value flip (a real, common backward-compat break) went completely undetected
    by this snapshot. Annotations are intentionally omitted because they churn under harmless
    type-narrowing refactors. Failures (LocalProxy, builtins without inspectable signatures,
    etc.) yield ``"<no-signature>"``.
    """
    try:
        sig = inspect.signature(obj)  # type: ignore[arg-type]  # obj is intentionally untyped object; non-callables are caught below and yield "<no-signature>"
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
        default_repr = _safe_default_repr(p.default) if has_default else ""
        params.append(f"{name}:{kind_short}:{int(has_default)}:{default_repr}")
    return "(" + ", ".join(params) + ")"


_TYPING_ABSENT = object()


def capture_module_surface(mod: object) -> dict[str, str]:
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
    out: dict[str, str] = {}
    # Names the module itself declares as lazily-populated caches (``__lazy_globals__``). Whether
    # one still holds its ``None`` placeholder or the real object depends on whether some earlier
    # test in the same process happened to trigger the lazy init, so their captured kind is pinned
    # to "value" regardless -- see the comment on the plain-value branch below for the full
    # rationale. Without this, ``webtext.nlp`` flipped from "value" to a spaCy Language
    # ``callable(text, ...)`` (a CALLABLE, so the value-branch narrowing did not cover it) on
    # whichever CI leg both installs spaCy and runs a sentencizing test first (2026-09-03).
    lazy_globals = frozenset(getattr(mod, "__lazy_globals__", ()))
    for name in dir(mod):
        if name.startswith("_"):
            continue
        if name in lazy_globals:
            out[name] = "value"
            continue
        try:
            obj = getattr(mod, name)
        except Exception:
            out[name] = "<inaccessible-at-import-time>"
            continue
        if inspect.ismodule(obj):
            continue
        # Skip CLASSES re-exported from outside pyutilz; their identity
        # (typing.Any class-vs-_SpecialForm) flips across Python versions
        # and isn't part of our public API. Don't filter callables /
        # instances — we WANT to track ``logger = logging.getLogger(...)``,
        # ``client = SomeClient()``, etc. as part of the surface even
        # though their ``__module__`` is outside pyutilz.
        # A typing re-export identified by IDENTITY, not by __module__: `TYPE_CHECKING` is a plain
        # bool, so it carries no module and slipped past the `owner == "typing"` filter below into
        # the public snapshot of any module doing `from typing import TYPE_CHECKING`.
        if getattr(typing, name, _TYPING_ABSENT) is obj:
            continue
        owner = getattr(obj, "__module__", None)
        if inspect.isclass(obj):
            if owner and not owner.startswith("pyutilz"):
                continue
        # Skip ``typing.*`` symbols entirely — typing has flipped
        # ``Union``/``Optional``/``Any``/``Tuple`` between
        # ``_SpecialForm`` instances and real classes across Python
        # 3.10/3.11/3.14, producing phantom diffs.
        elif owner == "typing":
            continue
        # Skip callables (functions / methods) imported from stdlib —
        # their parameter names drift across Python versions
        # (``os.path.join(path, *paths)`` -> ``join(a, *p)``).
        elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
            if owner and not owner.startswith("pyutilz"):
                continue
        if inspect.isclass(obj):
            out[name] = f"class:{getattr(obj, '__module__', '?')}.{obj.__name__}"
        elif callable(obj):
            out[name] = "callable" + capture_signature(obj)
        else:
            # Plain values are captured WITHOUT their concrete type. Several pyutilz modules use a
            # `_singleton = None` module global that a lazy-init function replaces with a real
            # instance on first use (e.g. webtext.inflect_engine, webtext.nlp) -- whether that has
            # happened yet depends on pytest's execution order within the process, so pinning
            # "value:NoneType" vs "value:engine" produces an order-dependent flake, not a real API
            # change. The test's own stated purpose is catching symbol REMOVAL/RENAME, not a
            # value's type drifting, so dropping the type here is a deliberate narrowing of scope,
            # not a loss of a guarantee this test ever meant to provide.
            out[name] = "value"
    return out


# ---------------------------------------------------------------------------
# Debt-marker scanning (TODO-style annotations)
# ---------------------------------------------------------------------------


MARKER_LINE_RE = re.compile(
    r"#\s*(?P<kw>TODO|FIXME|XXX|HACK)\b(?P<rest>.*)$",
    re.IGNORECASE,
)

# An attribution is one of:
#   * (assignee) — anything in parens immediately after the keyword
#   * 2026-04-28 — ISO date anywhere in the comment
#   * @assignee  — at-mention anywhere in the comment
ATTRIBUTION_RE = re.compile(r"(\([^)]+\)|\d{4}-\d{2}-\d{2}|@\w+)")


def scan_todo_markers(
    package_dir: Path,
    *,
    extra_excludes: Iterable[str] = (),
) -> list[tuple[Path, int, str, str]]:
    """Walk every production .py file under ``package_dir`` and yield
    every debt-marker comment line (TODO / FIXME / XXX / HACK).

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
    for py in sorted(_rglob_test_files(test_meta_dir)):
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
            value: Optional[ast.expr]
            if isinstance(node, ast.Assign):
                targets = node.targets
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target] if node.target else []
                value = node.value
            else:
                continue
            if value is None:
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
                elif isinstance(value, ast.Call) and not value.args and not value.keywords:
                    # An EMPTY ``set()`` / ``dict()`` constructor only. ``set(_LEGACY_ENTRIES)`` is
                    # also an ast.Call, and recording it as 0 made 40 tracked entries read as none --
                    # the drift tracker reported shrinking debt in exactly the case where it grew.
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
    # endswith-guarded slice, not an unanchored replace: "MyConfigManagerConfig" -> "my_manager"
    # dropped the INNER "_config" too, so the plausible real binding "my_config_manager" was never
    # generated and a valid symbol got reported as missing.
    short = snake[: -len("_config")] if snake.endswith("_config") else snake
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


# ---------------------------------------------------------------------------
# Field-harvesting round-trip harness (verdict / DTO parser parity)
# ---------------------------------------------------------------------------
#
# Recurring bug class: a dataclass declares an optional field the producer
# (an LLM prompt, an API response schema, an upstream config) is meant to
# populate, but the field is never actually extracted by whatever parses the
# raw payload into that dataclass -- a missing key in a hand-written
# field-name map, an allowlist that forgot the new field, a post-hook that
# claims some fields but not this one. The dataclass, the schema, and the
# call site can all look complete on review; only feeding a payload through
# the real parser and checking the field survived catches the gap.
#
# These three helpers factor out the MECHANISM (pick a sentinel value for a
# field's type; compare a batch of expected sentinels against actual
# attribute values) so a project only has to supply the payload-shape glue
# (what JSON key(s) a field maps to, which parser function to call) that IS
# inherently specific to its own producer/parser contract.


def sentinel_for_type(tp: Any) -> Optional[object]:
    """Best-effort sentinel value for a resolved type annotation.

    Handles ``bool`` / ``str`` / ``float`` / ``int``, each optionally
    wrapped in ``X | None`` / ``Optional[X]`` (the ``Union`` is unwrapped
    to find the first recognised member). Returns ``None`` for anything
    else (``list[...]``, ``dict[...]``, a nested dataclass, ``Literal``,
    an unresolved ``TypeVar``, ...) -- callers should read that as "skip
    this field, provide your own value" rather than "the sentinel is
    ``None``" (``bool``/``str``/``float``/``int`` never legitimately
    produce a ``None`` sentinel here, so the return value is unambiguous).

    The ``bool`` sentinel returned here is ``True``, which is also the commonest default value;
    :func:`optional_scalar_fields` therefore flips it to ``not field.default`` where it knows the
    default, so the probe can distinguish "populated" from "dropped".

    ``bool`` is checked before ``int`` deliberately: ``bool`` is a
    subclass of ``int`` in Python, but ``isinstance``/type-identity checks
    here compare the annotation object itself (``tp is bool``), not an
    instance, so no such precedence bug applies -- the ordering just
    keeps the common case first for readability.
    """
    # `X | None` (PEP 604) produces a `types.UnionType`, not `typing.Union` -- distinct origins on
    # python<3.14 (3.14 unified the two representations, which masked this exact gap when tested only
    # under 3.14: `typing.get_origin(bool | None) is typing.Union` is True there but False on 3.10-3.13).
    # `types.UnionType` itself doesn't exist before python 3.10 (no `X | None` syntax to produce it then).
    _union_type = getattr(types, "UnionType", None)
    origin = typing.get_origin(tp)
    if origin is typing.Union or (_union_type is not None and origin is _union_type):
        candidates = [a for a in typing.get_args(tp) if a is not type(None)]
    else:
        candidates = [tp]
    for cand in candidates:
        if cand is bool:
            return True
        if cand is str:
            return "__sentinel_str__"
        if cand is float:
            return 0.5
        if cand is int:
            return 7
    return None


def optional_scalar_fields(cls: type, skip: Iterable[str] = ()) -> dict[str, object]:
    """``{field_name: sentinel_value}`` for every ``dataclasses.field`` of
    ``cls`` whose resolved type is a scalar (``bool``/``str``/``float``/
    ``int``, optionally ``| None``), excluding names in ``skip``.

    Resolves annotations via ``typing.get_type_hints`` rather than reading
    ``field.type`` directly -- under ``from __future__ import annotations``
    (PEP 563) ``field.type`` is a bare string, not the actual type object,
    and ``get_type_hints`` is what correctly evaluates it against the
    defining module's globals either way.

    Fields of non-scalar type (``list[str] | None``, a nested dataclass,
    etc.) are silently omitted -- the caller decides how (or whether) to
    populate those; this helper only covers the shapes it can generate an
    unambiguous, comparable sentinel for.
    """
    hints = typing.get_type_hints(cls)
    out: dict[str, object] = {}
    for f in dataclasses.fields(cls):
        if f.name in skip:
            continue
        tp = hints.get(f.name)
        if tp is None:
            continue
        value = sentinel_for_type(tp)
        if value is None:
            continue
        if isinstance(value, bool) and isinstance(f.default, bool):
            # The bool sentinel is True, which is also the commonest default: with the comparison
            # being ``actual != expected``, a parser that DROPPED the field entirely still matched
            # its True default and passed as intact -- precisely the "declared field never
            # extracted" bug this harness exists to catch. Probe with the NON-default value.
            value = not f.default
        out[f.name] = value
    return out


def assert_fields_roundtrip(
    sentinels: dict[str, object],
    get_actual: Callable[[str], object],
) -> list[str]:
    """For each ``(field_name, sentinel)`` in ``sentinels``, call
    ``get_actual(field_name)`` (typically ``lambda name: getattr(parsed_obj,
    name)``) and compare against the sentinel.

    Returns the list of field names whose actual value did NOT match the
    expected sentinel -- an empty list means every field made the full
    round trip (payload -> parser -> dataclass instance) intact. Pure
    comparison helper; building the payload and invoking the parser is the
    caller's job, since that's the part that's genuinely project-specific.
    """
    mismatches: list[str] = []
    for name, expected in sentinels.items():
        actual = get_actual(name)
        if actual != expected:
            mismatches.append(name)
    return mismatches


# ---------------------------------------------------------------------------
# Ratchets and audit-document parity
# ---------------------------------------------------------------------------


def findings_ratchet(findings: Iterable[str], baseline_path: Path) -> tuple[list[str], list[str]]:
    """Diff ``findings`` against a committed JSON baseline, returning ``(new, drained)``.

    The point of a ratchet, rather than a plain assertion, is that a rule with real debt behind it
    fails on the day it is written and gets deleted the week after. The baseline records the debt
    that existed when the rule landed; ``new`` is what must fail the build; ``drained`` is what the
    project has since fixed and can prune from the baseline. Pair it with a separate strict-xfail
    test asserting the baseline is EMPTY, so the debt stays visible instead of being absorbed.

    A missing baseline file is treated as an empty baseline, so the first run reports everything.
    """
    import json

    known: set[str] = set()
    if baseline_path.exists():
        known = set(json.loads(baseline_path.read_text(encoding="utf-8")))
    current = set(findings)
    return sorted(current - known), sorted(known - current)


# CASE-SENSITIVE on purpose. A disposition is written in the fixed vocabulary RESOLVED / DOCUMENTED /
# FUTURE / REJECTED, always upper case, so matching case-insensitively claimed ordinary prose and field
# names instead: measured on autopsia 2026-09-03, the IGNORECASE form claimed 163 table rows the
# upper-case one does not - among them "...the parameters were resolved at age 70..." and a row whose
# only hit was the field name `resolved[].observation_kind`. Every row it stops claiming was checked:
# the nine whose own cell could read as a verdict are table HEADERS or data columns, never a
# disposition, so nothing real stops being checked.
_DISPOSITION_RE = re.compile(r"\bRESOLVED\b")
_CITATION_RE = re.compile(r"`([^`]+)`")
# A `| Disposition | Count |` legend row - the word beside a bare number, nothing else. Such a row counts
# dispositions, it does not make one, so demanding an artefact of it asks the summary to cite the rows it is
# summarising. Deliberately the narrowest shape that describes only a tally: measured over autopsia's whole
# audit corpus (~1,900 disposition rows, 2026-09-03), exactly ONE row matches, the one this exists for. The
# looser rule tried first - skip any row whose FIRST cell is the verdict - would have stopped checking 52
# rows, "PARTIALLY RESOLVED" and "(a) RESOLVED" verdicts among them.
_TALLY_VERDICT_RE = re.compile(r"^[*_ ]*[A-Z][A-Z-]+[*_ ]*$")
_TALLY_COUNT_RE = re.compile(r"^[*_ ]*\d+[*_ ]*$")
# A citation naming a symbol the way prose names one: `is_all_population()`, `_http.get_text`. Anchored
# and narrow - a dotted identifier with an optional call suffix and nothing else - so a real name written
# naturally is recognised without degenerating into "any row mentioning any identifier passes", which
# would make the rule fire on nothing.
_QUALIFIED_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*(?:\(\))?")


def clear_repo_scan_caches() -> None:
    """Drop the cached repository filename/symbol scans.

    ``_repo_filenames`` / ``_repo_symbols`` are ``lru_cache``d on the ROOT PATH ALONE and never
    invalidated, so within one pytest session a test that writes a file and a later test that scans
    the same root see a stale set -- making the result depend on test ORDER. Any test (or tool) that
    mutates the tree between scans must call this.
    """
    _repo_filenames.cache_clear()
    _repo_symbols.cache_clear()


@lru_cache(maxsize=8)
def _repo_filenames(repo_root: Path) -> frozenset[str]:
    """Every filename in the repository, cached on the root path -- see :func:`clear_repo_scan_caches`."""
    skip = {".git", "__pycache__", ".mypy_cache", ".pytest_cache", "node_modules", ".venv"}
    return frozenset(p.name for p in repo_root.rglob("*") if p.is_file() and not skip & set(p.parts))


@lru_cache(maxsize=8)
def _repo_symbols(repo_root: Path) -> frozenset[str]:
    """Names the source itself defines: functions, classes, module constants, and the keys of the data rows.

    An audit row citing `resolve_symptom` or `prior_basis` names something as real as a path - a function
    and a stored field. Reading only paths made 362 such rows read as citing nothing.
    """
    import ast

    skip = {".git", "__pycache__", ".mypy_cache", ".pytest_cache", "node_modules", ".venv"}
    names: set[str] = set()
    for py in repo_root.rglob("*.py"):
        if skip & set(py.parts):
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, ValueError, OSError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.isidentifier():
                # Field names of the rows this repository writes, which audit rows cite as often as functions.
                names.add(node.value)
    return frozenset(names)


def _is_a_commit_this_repo_has(cite: str, repo_root: Path) -> bool:
    """A short SHA is a citation of the change itself, and the strongest one an audit row can carry."""
    import subprocess  # nosec B404 - reads git metadata of the repository under test, no external input

    if not (7 <= len(cite) <= 40) or not all(c in "0123456789abcdef" for c in cite.lower()):
        return False
    try:
        done = subprocess.run(  # nosec B603 B607 - fixed argv, no shell, `cite` validated as hex above
            ["git", "cat-file", "-e", f"{cite}^{{commit}}"], cwd=repo_root, capture_output=True, timeout=15, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def _names_a_repo_symbol(citation: str, repo_root: Path) -> bool:
    """Whether a citation names a symbol this repository defines, written as prose writes one.

    `is_all_population()` and `Observation.is_all_population()` name the same function an exact-match test
    misses over a call suffix and an attribute prefix; `_http.get_text` and `TB_notif_num` are the same
    shape. Measured on autopsia 2026-09-03: 25 rows citing a real, findable symbol were reported as citing
    nothing for this reason alone. The match stays anchored to a whole dotted name, so a citation that
    merely CONTAINS an identifier is still reported - a rule that accepted any row mentioning any name
    would never fire.
    """
    text = citation.strip()
    if not _QUALIFIED_NAME_RE.fullmatch(text):
        return False
    # Not str.removesuffix: that is Python 3.9+ (PEP 616) and this package supports 3.8, where the
    # call raises AttributeError on the first bare qualified-name citation.
    name = text[:-2] if text.endswith("()") else text
    symbols = _repo_symbols(repo_root)
    return name in symbols or name.rsplit(".", 1)[-1] in symbols


def unbacked_audit_dispositions(audit_dir: Path, repo_root: Path, test_name_prefix: str = "test_") -> list[str]:
    """Find audit rows marked RESOLVED that cite nothing which exists.

    An audit disposition table is the record of what was fixed, and it is read months later by
    somebody deciding what still needs work. A row reading RESOLVED that names no artefact -- or
    names a file that has since moved, or a test that was never written -- converts the audit from a
    record into a claim. The instance behind this rule: a round-1 table marked a finding resolved and
    round 2 measured the two code paths still disagreeing.

    A row passes when at least one backtick-quoted citation names an artefact that EXISTS: a path under
    ``repo_root``, a FILENAME that exists somewhere under it, a function starting with ``test_name_prefix``
    defined anywhere in the repository, or a COMMIT this repository actually has.

    The last three were the docstring's own bar and not the implementation's, which resolved
    ``repo_root / cite`` and nothing else. Measured on autopsia 2026-09-02: 799 rows read as unbacked, and
    the citations they were failing on were ordinary ones - ``corpus.py:128`` (a basename, not a path from
    the root), ``review_batch_1_report.md`` (a real file two directories down), ``f01b54e6`` (a real commit).
    A rule that reports 799 false alarms is not read at all, which costs more than the rule buys.

    Returns one string per unbacked row.
    """
    unbacked: list[str] = []
    test_corpus = ""
    for py in sorted(_rglob_test_files(repo_root)):
        test_corpus += py.read_text(encoding="utf-8", errors="replace")
    by_name = _repo_filenames(repo_root)

    for doc in sorted(audit_dir.rglob("*.md")):
        for lineno, line in enumerate(doc.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if not line.lstrip().startswith("|") or not _DISPOSITION_RE.search(line):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) == 2 and _TALLY_VERDICT_RE.match(cells[0]) and _TALLY_COUNT_RE.match(cells[1]):
                continue
            citations = _CITATION_RE.findall(line)
            backed = False
            for cite in citations:
                cite = cite.strip().split(":")[0].split("::")[0]
                if cite.startswith(test_name_prefix) and f"def {cite}" in test_corpus:
                    backed = True
                elif cite and (repo_root / cite).exists():
                    backed = True
                elif cite and PurePosixPath(cite).name in by_name:
                    backed = True
                elif _is_a_commit_this_repo_has(cite, repo_root):
                    backed = True
                elif cite in _repo_symbols(repo_root):
                    backed = True
                elif _names_a_repo_symbol(cite, repo_root):
                    backed = True
            if not backed:
                cited = ", ".join(citations) if citations else "nothing"
                unbacked.append(f"{doc.relative_to(repo_root).as_posix()}:{lineno} RESOLVED cites {cited}")
    return unbacked

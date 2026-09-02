"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path

from ._base import _DEFAULT_EXCLUDE_DIRS, Finding, _iter_py_files, _line_text, _safe_parse

# --- a counter written but never initialised by the reset that owns its dict --------------------
#
# An object holds a per-cycle counter dict, a reset method re-initialises it between cycles, and
# one key is written somewhere that the reset never declares. Two things follow, and both have
# happened in the audited codebase:
#
# * The key is created lazily on first write, so it is cumulative since PROCESS START while every
#   sibling beside it is per-cycle -- and it is logged and persisted next to them as though it
#   meant the same thing. `reconcile_skipped_small` was the one counter `_reset_stats` never reset
#   on a crawler object that lives for the whole process.
# * If the increment goes through a helper that does `self.stats[key] += n`, an unregistered key
#   is a KeyError in production rather than a silent zero. That is the wave-17 `worker_crashes`
#   regression on the same dict, and it happened again while this rule was being written: a new
#   `overlapping_axis_skipped` counter was added, incremented through `_inc_stat`, and never
#   declared -- caught only because one test happened to build its stats from the real initialiser.
#
# Three independent incidents on one dict is what makes this mechanical rather than a review note.
# A comment in that file even records the migration from lazy `.get(k, 0) + n` counters to
# pre-initialised ones: the sweep was done by hand once, and it missed a key.

# Attribute names that look like a per-cycle counter dict.
_COUNTER_ATTRS = {"stats", "counters", "metrics", "totals", "counts"}

# Helpers that increment a key by name -- the spelling that turns a missing key into a KeyError.
_INC_HELPERS = {"_inc_stat", "_inc_stats", "inc_stat", "bump", "_bump"}


def _dict_literal_keys(node: ast.AST) -> set[str]:
    """String keys of a dict literal assigned here."""
    if isinstance(node, ast.Dict):
        return {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    return set()


def _is_counter_attribute(node: ast.AST) -> str | None:
    """`self.stats` -> "stats", for the attribute names that look like counter dicts."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self" and node.attr in _COUNTER_ATTRS:
        return node.attr
    return None


def _declared_by(method: ast.AST) -> dict[str, set[str]]:
    """{attribute: keys} this method initialises, from `self.stats = {...}` literals."""
    declared: dict[str, set[str]] = {}
    for stmt in ast.walk(method):
        if not isinstance(stmt, ast.Assign):
            continue
        for target in stmt.targets:
            attr = _is_counter_attribute(target)
            if attr:
                declared.setdefault(attr, set()).update(_dict_literal_keys(stmt.value))
    return declared


def _written_keys(cls: ast.ClassDef, attr: str) -> dict[str, int]:
    """{key: line} written to `self.<attr>` anywhere in this class, by any spelling."""
    written: dict[str, int] = {}

    def _record(key: ast.AST, line: int) -> None:
        """Note a string-literal key write; a computed key cannot be checked statically."""
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            written.setdefault(key.value, line)

    # ACCUMULATING writes only. `self.stats["k"] = value` overwrites completely every cycle and
    # is therefore safe without a declaration -- the four such keys in the audited crawler were
    # this rule's only false positives. `+=`, `.get(k, 0) + n`, `.setdefault(k, ...)` and an
    # increment helper all ACCUMULATE, which is where "cumulative since process start while its
    # siblings are per-cycle" comes from.
    for node in ast.walk(cls):
        if isinstance(node, ast.AugAssign):
            target = node.target
            if isinstance(target, ast.Subscript) and _is_counter_attribute(target.value) == attr:
                _record(target.slice, node.lineno)
        # self.stats.get("k", 0) / self.stats.setdefault("k", 0)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("get", "setdefault") and _is_counter_attribute(node.func.value) == attr:
                if node.args:
                    _record(node.args[0], node.lineno)
            # self._inc_stat("k")
            elif node.func.attr in _INC_HELPERS and node.args:
                _record(node.args[0], node.lineno)
    return written


def scan_stats_key_coverage(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find counter keys written to a per-cycle dict that no reset method initialises.

    Such a key is cumulative since process start while every sibling beside it is per-cycle,
    and it is logged and persisted next to them as though it meant the same thing. Where the
    increment goes through a helper doing `self.stats[key] += n`, it is a `KeyError` in
    production instead of a silent zero.

    DECLARATIONS AND WRITES ARE MATCHED ACROSS THE PACKAGE, not within one class, because the
    real shape is a MIXIN: `StatsMixin._reset_stats` declares the dict and `SplitMixin`
    increments it, and the two are one object only at runtime. Scoped per class, this rule
    missed the very bug it was written from.

    The cost of that widening is false NEGATIVES -- a key declared by an unrelated class that
    happens to use the same attribute name masks a genuine omission. It cannot produce a false
    positive, which is the right trade for a ratchet: a check that cries wolf gets refreshed
    without being read.

    Only string-literal keys are tracked; a computed key cannot be checked against a
    declaration statically.
    """
    declared: dict[str, set[str]] = {}
    declaring_method: dict[str, str] = {}
    writes: list[tuple[str, str, str, int, str]] = []  # (attr, key, file, line, class)

    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        rel = py.relative_to(root).as_posix()

        for cls in ast.walk(tree):
            if not isinstance(cls, ast.ClassDef):
                continue
            for method in cls.body:
                if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for attr, keys in _declared_by(method).items():
                    if not keys:
                        continue
                    declared.setdefault(attr, set()).update(keys)
                    declaring_method.setdefault(attr, f"{cls.name}.{method.name}")

            for attr in _COUNTER_ATTRS:
                for key, line in _written_keys(cls, attr).items():
                    writes.append((attr, key, rel, line, cls.name))

    findings: list[Finding] = []
    sources: dict[str, list[str]] = {}
    for py in _iter_py_files(root, exclude_dirs):
        rel = py.relative_to(root).as_posix()
        if any(w[2] == rel for w in writes):
            sources[rel] = py.read_text(encoding="utf-8", errors="replace").splitlines()

    for attr, key, rel, line, cls_name in sorted(writes, key=lambda w: (w[2], w[3])):
        known = declared.get(attr)
        if known is None or key in known:
            continue
        findings.append(
            Finding(
                check="stats_key_coverage",
                severity="P2",
                file=rel,
                line=line,
                snippet=_line_text(sources.get(rel, []), line),
                detail=(
                    f"`{cls_name}` writes `self.{attr}[{key!r}]`, which "
                    f"`{declaring_method[attr]}` never initialises. The key is created on first "
                    "write, so it is cumulative since process start while every sibling is "
                    "per-cycle -- and if the write goes through an increment helper, it is a "
                    "KeyError rather than a zero."
                ),
            )
        )
    return findings

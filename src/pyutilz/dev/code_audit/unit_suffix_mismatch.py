"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path

from ._base import _DEFAULT_EXCLUDE_DIRS, Finding, _iter_py_files, _line_text, _safe_parse, _subscript_index

# --- a quantity stored under one unit and read from another ------------------------------------
#
# `work_s = totals["minutes"]` -- a value produced in one unit, consumed under a name declaring a
# different one, with no conversion between them. The consumer is then confidently wrong, and
# nothing downstream can tell.
#
# Confirmed in one audited codebase: a `duration_s` column measured cycle wall-clock while the
# actual work time existed one JSONB level away as `extra.minutes`; the disposition for the fix
# reads "The number already existed as `extra.minutes`; what it lacked was a name a reader would
# reach for and a matching unit." A sibling finding in the same round: `proxy_bytes_received`
# counted DECOMPRESSED bytes into a column read as a billing figure.
#
# The rule is deliberately narrow. It fires only when both sides carry an explicit unit token and
# those tokens belong to different FAMILIES, and only when no multiplicative literal sits between
# them -- `work_s = totals["minutes"] * 60` is the correct form and must stay silent.

# Unit tokens by family. A mismatch WITHIN a family (seconds vs minutes) is the interesting case;
# across families (seconds vs bytes) is almost always a coincidence of naming, so both are
# reported but the message names the family.
_UNIT_FAMILIES: dict[str, str] = {
    "s": "time",
    "sec": "time",
    "secs": "time",
    "second": "time",
    "seconds": "time",
    "ms": "time",
    "msec": "time",
    "millis": "time",
    "min": "time",
    "mins": "time",
    "minute": "time",
    "minutes": "time",
    "h": "time",
    "hour": "time",
    "hours": "time",
    "day": "time",
    "days": "time",
    "bytes": "size",
    "kb": "size",
    "mb": "size",
    "gb": "size",
    "pct": "ratio",
    "percent": "ratio",
    "ratio": "ratio",
}

# Tokens that mean the same amount, so pairing them is not a mismatch.
_SYNONYMS: dict[str, str] = {
    "sec": "seconds",
    "secs": "seconds",
    "second": "seconds",
    "s": "seconds",
    "msec": "ms",
    "millis": "ms",
    "min": "minutes",
    "mins": "minutes",
    "minute": "minutes",
    "h": "hours",
    "hour": "hours",
    "day": "days",
    "percent": "pct",
}


def _unit_of(name: str) -> tuple[str, str] | None:
    """(canonical unit, family) for a trailing unit token in *name*, or None."""
    token = name.rstrip("_").rsplit("_", 1)[-1].lower()
    family = _UNIT_FAMILIES.get(token)
    if family is None:
        return None
    return _SYNONYMS.get(token, token), family


def _subscript_key(node: ast.Subscript) -> str | None:
    """The literal string key of `d["key"]`, on every supported python.

    Until 3.8, the parser wrapped a plain subscript in ``ast.Index`` (``Subscript(slice=Index(value=
    Constant('key')))``); 3.9 removed that wrapper and put the expression on ``.slice`` directly
    (bpo-34822). Reading ``.slice`` alone therefore saw an ``Index`` node on 3.8 and matched nothing,
    which silently disabled this whole rule for dict reads on that version.
    """
    key = _subscript_index(node)
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        return key.value
    return None


def _source_name(node: ast.AST) -> str | None:
    """The name a value is read FROM: `x`, `obj.attr`, or `d["key"]`."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _subscript_key(node)
    return None


def _target_names(node: ast.AST) -> list[str]:
    """The name a value is assigned TO: `x`, `obj.attr`, or `d["key"]`."""
    out: list[str] = []
    if isinstance(node, ast.Name):
        out.append(node.id)
    elif isinstance(node, ast.Attribute):
        out.append(node.attr)
    elif isinstance(node, ast.Subscript):
        key_name = _subscript_key(node)
        if key_name is not None:
            out.append(key_name)
    return out


def scan_unit_suffix_mismatch(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find assignments that move a value between incompatible units with no conversion.

    `work_s = totals["minutes"]` names one unit and reads another. Reported only when BOTH sides
    carry an explicit unit token, the tokens differ after canonicalising synonyms (`_secs` and
    `_seconds` are the same unit), and the right-hand side is a BARE read -- no arithmetic.
    `work_s = totals["minutes"] * 60` is the correct form and is silent.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()

        pairs: list[tuple[str, ast.AST, int]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                pairs.extend((name, node.value, node.lineno) for target in node.targets for name in _target_names(target))
            elif isinstance(node, ast.Call):
                pairs.extend((kw.arg, kw.value, node.lineno) for kw in node.keywords if kw.arg)

        for target_name, value, line in pairs:
            target_unit = _unit_of(target_name)
            if target_unit is None:
                continue
            # A BARE read only: any arithmetic is a conversion until proven otherwise, and
            # assuming otherwise would flag every correct conversion in the tree.
            source = _source_name(value)
            if source is None:
                continue
            source_unit = _unit_of(source)
            if source_unit is None or source_unit[0] == target_unit[0]:
                continue

            same_family = source_unit[1] == target_unit[1]
            findings.append(
                Finding(
                    check="unit_suffix_mismatch",
                    severity="P2" if same_family else "Low",
                    file=rel,
                    line=line,
                    snippet=_line_text(src_lines, line),
                    detail=(
                        f"`{target_name}` declares {target_unit[0]} and is assigned `{source}`, "
                        f"which declares {source_unit[0]}, with no conversion between them"
                        + (
                            " -- the same quantity under two units, which is the shape that put " "wall-clock seconds in a column read as work time."
                            if same_family
                            else " (different unit families, so this may be a naming coincidence)."
                        )
                    ),
                )
            )
    return findings

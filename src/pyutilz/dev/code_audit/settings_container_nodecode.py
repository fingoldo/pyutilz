"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _line_text, _read_src_lines, _safe_parse

# --- a settings container field whose before-validator never sees the environment ----------------

# Annotations pydantic-settings treats as "complex": its env source JSON-decodes these before any validator.
_CONTAINERS = frozenset({"list", "set", "frozenset", "dict", "tuple", "List", "Set", "FrozenSet", "Dict", "Tuple", "Sequence", "Mapping"})


def _name_of(node: ast.expr) -> str:
    """The trailing identifier of a Name or Attribute (``typing.List`` -> ``List``), else ``""``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _split_annotated(ann: ast.expr) -> tuple[ast.expr, list[ast.expr]]:
    """``Annotated[X, m1, m2]`` -> (X, [m1, m2]); anything else -> (ann, [])."""
    if isinstance(ann, ast.Subscript) and _name_of(ann.value) == "Annotated" and isinstance(ann.slice, ast.Tuple) and ann.slice.elts:
        return ann.slice.elts[0], list(ann.slice.elts[1:])
    return ann, []


def _is_container(ann: ast.expr) -> bool:
    """A container annotation, or a union / Optional that holds one (``list[str] | None``, ``Optional[set[str]]``)."""
    if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
        return _is_container(ann.left) or _is_container(ann.right)
    if isinstance(ann, ast.Subscript):
        head = _name_of(ann.value)
        if head in _CONTAINERS:
            return True
        if head in ("Optional", "Union"):
            parts = ann.slice.elts if isinstance(ann.slice, ast.Tuple) else [ann.slice]
            return any(_is_container(p) for p in parts)
        if head == "Annotated":
            return _is_container(_split_annotated(ann)[0])
        return False
    return _name_of(ann) in _CONTAINERS


def _has_nodecode(ann: ast.expr) -> bool:
    """True if the annotation, or a union member of it, carries ``NoDecode`` in its ``Annotated`` metadata."""
    if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
        return _has_nodecode(ann.left) or _has_nodecode(ann.right)
    _, meta = _split_annotated(ann)
    return any(_name_of(m) == "NoDecode" or (isinstance(m, ast.Call) and _name_of(m.func) == "NoDecode") for m in meta)


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    """The value passed as keyword *name* to *call*, or None when it is not passed."""
    return next((kw.value for kw in call.keywords if kw.arg == name), None)


def _is_true(node: ast.expr | None) -> bool:
    """True only for the literal ``True``: ``exclude=flag`` or a missing keyword is not a decision the source shows."""
    return isinstance(node, ast.Constant) and node.value is True


def _before_validated_fields(cls: ast.ClassDef) -> set[str]:
    """Field names a ``@field_validator(..., mode="before")`` (or v1 ``@validator(..., pre=True)``) in *cls* names."""
    names: set[str] = set()
    for item in cls.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for deco in item.decorator_list:
            if not isinstance(deco, ast.Call):
                continue
            kind = _name_of(deco.func)
            mode = _keyword(deco, "mode")
            before = (kind == "field_validator" and isinstance(mode, ast.Constant) and mode.value in ("before", "plain", "wrap")) or (
                kind == "validator" and _is_true(_keyword(deco, "pre"))
            )
            if before:
                names.update(a.value for a in deco.args if isinstance(a, ast.Constant) and isinstance(a.value, str))
    return names


def _decoding_disabled(cls: ast.ClassDef) -> bool:
    """``model_config = SettingsConfigDict(..., enable_decoding=False)`` turns the decoding off for every field."""
    for item in cls.body:
        targets: list[ast.expr]
        value: ast.expr
        if isinstance(item, ast.Assign):
            targets, value = item.targets, item.value
        elif isinstance(item, ast.AnnAssign) and item.value is not None:
            targets, value = [item.target], item.value
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id == "model_config" for t in targets):
            continue
        if isinstance(value, ast.Call):
            flag = _keyword(value, "enable_decoding")
            if isinstance(flag, ast.Constant) and flag.value is False:
                return True
        if isinstance(value, ast.Dict):
            for k, v in zip(value.keys, value.values):
                if isinstance(k, ast.Constant) and k.value == "enable_decoding" and isinstance(v, ast.Constant) and v.value is False:
                    return True
    return False


def _settings_classes(tree: ast.Module) -> list[ast.ClassDef]:
    """Classes deriving from ``BaseSettings``, directly or through another settings class in the same module."""
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    settings: dict[str, ast.ClassDef] = {}
    changed = True
    while changed:
        changed = False
        for cls in classes:
            if cls.name in settings:
                continue
            if any(_name_of(b) == "BaseSettings" or _name_of(b) in settings for b in cls.bases):
                settings[cls.name] = cls
                changed = True
    return list(settings.values())


def scan_settings_container_field_needs_nodecode(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find a container-typed ``BaseSettings`` field that a before-validator parses, with no ``NoDecode``.

    pydantic-settings' ``EnvSettingsSource`` JSON-decodes a "complex" field (list, set, dict, tuple...)
    BEFORE any validator runs. A ``mode="before"`` validator written to accept ``a,b,c`` is therefore
    unreachable from the environment: ``FIELD=a,b`` fails JSON decoding with a ``SettingsError`` and the
    validator never sees it. It still runs for direct construction, ``Settings(field="a,b")``, which is
    exactly why a unit test of the validator passes and the fix reads as done. ``Annotated[list[str],
    NoDecode]`` hands the raw string to the validator.

    Real case (glossum 2026-09-01 audit 12-H1): ``CORS_ORIGINS=http://localhost:3000,http://localhost:8080``
    from the project's own ``.env.example`` failed startup, and the fix that had been cited as precedent,
    a comma-splitting before-validator on ``supported_languages: set[str]``, had never run either.

    Not flagged: a class whose ``model_config`` sets ``enable_decoding=False``, a field declared with
    ``Field(exclude=True)``, a validator in ``mode="after"``, and any class that is not a settings class
    (a plain ``BaseModel`` field is never read from the environment).

    Severity: P1 -- the code reads as fixed and is not.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines: list[str] | None = None
        rel = py.relative_to(root).as_posix()
        for cls in _settings_classes(tree):
            if _decoding_disabled(cls):
                continue
            validated = _before_validated_fields(cls)
            if not validated:
                continue
            for item in cls.body:
                if not (isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)):
                    continue
                name = item.target.id
                if name not in validated or not _is_container(item.annotation) or _has_nodecode(item.annotation):
                    continue
                if isinstance(item.value, ast.Call) and _name_of(item.value.func) == "Field" and _is_true(_keyword(item.value, "exclude")):
                    continue
                if src_lines is None:
                    src_lines = _read_src_lines(py)
                findings.append(Finding(
                    check="settings_container_field_needs_nodecode",
                    severity="P1",
                    file=rel,
                    line=item.lineno,
                    snippet=_line_text(src_lines, item.lineno),
                    detail=(
                        f"`{cls.name}.{name}` is a container field parsed by a before-validator, but pydantic-settings "
                        f"JSON-decodes it from the environment before that validator runs, so `{name.upper()}=a,b` "
                        f"raises SettingsError and the validator only ever runs for direct construction. Annotate it "
                        f"`Annotated[..., NoDecode]`."
                    ),
                ))
    return findings

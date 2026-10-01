"""Typed, strict pydantic models generated from a callable's signature, plus a drift check that keeps a committed copy honest.

The problem this solves: a configuration object that forwards ``dict`` bags (``mrmr_kwargs={...}``) to a constructor with
hundreds of parameters only finds a typo, a wrong type or an out-of-range value when the estimator is finally built - often
deep inside a long run. Rewriting every parameter by hand into a typed config fixes that but duplicates the signature, and the
two drift the moment a parameter is added. Generating the model from the signature removes the duplication; a drift check
(``signature_drift``) turns "the signature moved on" into a failing test instead of a silent gap.

Three entry points:

* :func:`model_from_signature` builds the model at runtime: one field per parameter (name, annotation, default taken from the
  signature, required when the parameter has no default), ``extra="forbid"`` so an unknown key raises at instantiation, and
  ``frozen=True``. ``overrides`` tightens chosen fields (``Literal`` sets, ``Field(gt=0)``) without touching the target.
* :func:`render_model_source` writes the same model as Python source, for a module that is committed and imported cheaply
  (the generated file does not import the target, so building a config never pays the target's import cost).
* :func:`signature_drift` compares a model with the current signature and lists every difference (new/removed parameter,
  changed default, changed annotation, required <-> optional) so a meta-test can assert the list is empty.

Command line (regenerate a committed module)::

    python -m pyutilz.dev.signature_models pkg.module:Class -o pkg/_class_config.py --name ClassConfig --exclude estimator

Python 3.8 compatible: annotations are resolved with ``typing.get_type_hints`` and no PEP 604 syntax is emitted.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import inspect
import importlib
import re
import typing
from pathlib import Path
import sys
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, cast

from pydantic import BaseModel, ConfigDict, Field, create_model

__all__ = ["model_from_signature", "render_model_source", "signature_drift", "signature_parameters", "SignatureParameter", "main"]

_SKIP_KINDS = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


class SignatureParameter:
    """One parameter of the target signature: its name, resolved annotation (``Any`` when absent) and default (``inspect.Parameter.empty`` when required)."""

    __slots__ = ("annotated", "annotation", "default", "name")

    def __init__(self, name: str, annotation: Any, default: Any, annotated: bool) -> None:
        """Store the parameter facts; ``annotated`` records whether the source declared an annotation at all."""
        self.name = name
        self.annotation = annotation
        self.default = default
        self.annotated = annotated

    @property
    def required(self) -> bool:
        """True when the signature gives the parameter no default."""
        return self.default is inspect.Parameter.empty


def _callable_of(target: Callable[..., Any]) -> Callable[..., Any]:
    """The function whose signature describes the arguments: a class's ``__init__``, otherwise the callable itself."""
    return target.__init__ if inspect.isclass(target) else target  # type: ignore[misc]


def signature_parameters(target: Callable[..., Any], *, exclude: Iterable[str] = ()) -> List[SignatureParameter]:
    """Parameters of ``target`` (a class or a function) in declaration order, without ``self``/``cls``, ``*args``/``**kwargs`` or ``exclude``.

    Annotations are resolved with ``typing.get_type_hints`` so string annotations (``from __future__ import annotations``)
    become real types; when resolution fails the raw annotation is kept, and a parameter without any annotation is typed ``Any``.
    """
    fn = _callable_of(target)
    skip = set(exclude) | {"self", "cls"}
    try:
        hints = typing.get_type_hints(fn)
    except Exception:  # an unresolvable forward reference must not make the whole signature unusable
        hints = {}
    out: List[SignatureParameter] = []
    for name, param in inspect.signature(fn).parameters.items():
        if name in skip or param.kind in _SKIP_KINDS:
            continue
        raw = hints.get(name, param.annotation)
        annotated = raw is not inspect.Parameter.empty
        out.append(SignatureParameter(name, raw if annotated else Any, param.default, annotated))
    return out


def _field_definitions(params: Sequence[SignatureParameter], overrides: Mapping[str, Any], unannotated: str) -> Dict[str, Tuple[Any, Any]]:
    """``create_model`` field definitions: an override wins over the signature annotation; defaults come from the signature."""
    definitions: Dict[str, Tuple[Any, Any]] = {}
    for p in params:
        if not p.annotated and p.name not in overrides and unannotated == "error":
            raise TypeError(f"parameter {p.name!r} has no annotation; annotate it or pass an override")
        ann: Any = p.annotation
        default: Any = ... if p.required else p.default
        if p.name in overrides:
            override = overrides[p.name]
            if isinstance(override, tuple):
                ann, extra = override
                default = Field(... if p.required else p.default, **extra) if isinstance(extra, dict) else extra
            else:
                ann = override
        definitions[p.name] = (ann, default)
    return definitions


def model_from_signature(
    target: Callable[..., Any],
    *,
    name: Optional[str] = None,
    exclude: Iterable[str] = (),
    overrides: Optional[Mapping[str, Any]] = None,
    base: Optional[Type[BaseModel]] = None,
    frozen: bool = True,
    unannotated: str = "any",
) -> Type[BaseModel]:
    """Build a strict pydantic model with one field per parameter of ``target``.

    ``extra="forbid"``: a key the signature does not have raises at instantiation. ``arbitrary_types_allowed`` lets a parameter be
    annotated with an estimator or any other non-pydantic class. ``overrides`` maps a parameter name to a replacement annotation, or
    to ``(annotation, {"gt": 0, ...})`` to attach ``Field`` constraints. ``unannotated="error"`` refuses a parameter with no annotation
    instead of typing it ``Any``. The resulting class records ``__signature_source__`` and ``__signature_overrides__`` for the drift check.
    """
    overrides = dict(overrides or {})
    params = signature_parameters(target, exclude=exclude)
    config = ConfigDict(extra="forbid", frozen=frozen, arbitrary_types_allowed=True)
    fields = _field_definitions(params, overrides, unannotated)
    kwargs: Dict[str, Any] = {"__config__": config} if base is None else {"__base__": base}
    model = create_model(name if name is not None else f"{getattr(target, '__name__', 'Signature')}Params", **kwargs, **fields)  # type: ignore[call-overload]
    model.__signature_source__ = f"{getattr(target, '__module__', '?')}:{getattr(target, '__qualname__', '?')}"  # type: ignore[attr-defined]
    model.__signature_overrides__ = tuple(sorted(overrides))  # type: ignore[attr-defined]
    return cast(Type[BaseModel], model)


def _same_default(a: Any, b: Any) -> bool:
    """Equality of two defaults that never raises (arrays and exotic objects fall back to identity)."""
    try:
        return bool(a == b)
    except Exception:
        return a is b


def signature_drift(
    model: Type[BaseModel],
    target: Callable[..., Any],
    *,
    exclude: Iterable[str] = (),
    check_annotations: bool = True,
) -> List[str]:
    """Differences between ``model`` and the current signature of ``target``; an empty list means they agree.

    Reported: a parameter missing from the model, a field the signature no longer has, a changed default, a parameter that became
    required or optional, and (unless ``check_annotations=False``) a changed annotation. A field the model deliberately tightened
    (listed in ``model.__signature_overrides__``) is exempt from the annotation comparison, but its default is still checked.
    """
    params = {p.name: p for p in signature_parameters(target, exclude=exclude)}
    fields = model.model_fields
    skip_default = set(getattr(model, "__signature_skip_default__", ()))
    exempt = set(getattr(model, "__signature_overrides__", ())) | skip_default
    problems: List[str] = []
    problems.extend(f"parameter {name!r} is in the signature but not in {model.__name__}" for name in params if name not in fields)
    problems.extend(f"field {name!r} of {model.__name__} is no longer a parameter" for name in fields if name not in params)
    for name, p in params.items():
        field = fields.get(name)
        if field is None:
            continue
        if p.required != field.is_required():
            problems.append(f"{name!r}: required in the signature={p.required}, in the model={field.is_required()}")
        elif not p.required and name not in skip_default and not _same_default(p.default, field.get_default(call_default_factory=True)):
            problems.append(f"{name!r}: default {p.default!r} in the signature, {field.get_default(call_default_factory=True)!r} in the model")
        if check_annotations and name not in exempt and p.annotated and field.annotation != p.annotation:
            problems.append(f"{name!r}: annotation {p.annotation!r} in the signature, {field.annotation!r} in the model")
    return problems


_TYPING_NAMES = frozenset(typing.__all__)


def _annotation_source(annotation: Any, imports: "set[str]", allowed_modules: Optional[Sequence[str]] = None) -> str:
    """Python source for ``annotation``; names it needs are added to ``imports`` (``typing`` names and ``import pkg`` modules).

    ``typing`` constructs and builtins render as written; a class renders as its dotted path with the module imported. Anything
    that cannot be rendered as an expression (a local class, a lambda) degrades to ``Any`` so the generated file always imports. With
    ``allowed_modules`` an annotation that names a class from any other module also degrades to ``Any``: importing such a module
    from the generated file would drag the target's package (and its import cost) in, which a lightweight config must not do.
    """
    if annotation is Any or annotation is inspect.Parameter.empty:
        imports.add("typing.Any")
        return "Any"
    if annotation is None or annotation is type(None):
        return "None"
    text = typing._type_repr(annotation) if hasattr(typing, "_type_repr") else repr(annotation)  # type: ignore[attr-defined]
    text = text.replace("typing.", "").replace("NoneType", "None")
    if "<" in text or "lambda" in text or "<locals>" in text:
        imports.add("typing.Any")
        return "Any"
    scan = re.sub(r"'[^']*'|\"[^\"]*\"", "", text)  # a Literal's string values are not names to import
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", scan))
    if allowed_modules is not None:
        for token in tokens:
            if "." in token and not any(token.startswith(prefix + ".") for prefix in allowed_modules):
                imports.add("typing.Any")
                return "Any"
    for token in tokens:
        if "." in token:
            imports.add(f"module:{token.rsplit('.', 1)[0]}")
        elif token in _TYPING_NAMES:
            imports.add(f"typing.{token}")
    return text


_UNPARSED: Any = object()


def _is_literal_default(value: Any) -> bool:
    """True when ``repr(value)`` is valid Python source that evaluates back to an equal ``repr`` (so it can be written into a generated module)."""
    text = repr(value)
    parsed: Any = _UNPARSED
    with contextlib.suppress(ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        parsed = ast.literal_eval(text)
    return parsed is not _UNPARSED and repr(parsed) == text


def render_model_source(
    target: Callable[..., Any],
    class_name: str,
    *,
    exclude: Iterable[str] = (),
    overrides: Optional[Mapping[str, Any]] = None,
    docstring: Optional[str] = None,
    regenerate_hint: Optional[str] = None,
    allowed_modules: Optional[Sequence[str]] = None,
) -> str:
    """Source of a module that defines ``class_name`` as a strict pydantic model mirroring ``target``'s signature.

    ``allowed_modules`` (module-name prefixes, e.g. ``("collections", "numpy")``) keeps the generated file import-light: an annotation
    naming a class from any other module is written as ``Any``. ``overrides`` values are source strings (``'Literal["a", "b"]'`` or ``'Field(0.9, gt=0, le=1)'``) written verbatim: a string that
    starts with ``Field(`` replaces the default, anything else replaces the annotation. The module does not import ``target``.
    """
    overrides = dict(overrides or {})
    imports: "set[str]" = {"pydantic.BaseModel", "pydantic.ConfigDict"}
    lines: List[str] = []
    skip_default: List[str] = []
    for p in signature_parameters(target, exclude=exclude):
        ann = _annotation_source(p.annotation, imports, allowed_modules)
        default = "" if p.required else f" = {p.default!r}"
        if not p.required and not _is_literal_default(p.default):
            # A default that has no source form (a numpy dtype, an enum member) cannot be written into the module; the field accepts anything
            # and defaults to None, and the drift check leaves it out of the default/annotation comparison.
            imports.add("typing.Any")
            ann, default = "Any", " = None"
            skip_default.append(p.name)
        if not p.required and isinstance(p.default, (list, dict, set)):
            imports.add("pydantic.Field")
            default = f" = Field(default_factory=lambda: {p.default!r})"
        if p.name in overrides:
            text = overrides[p.name]
            if str(text).startswith("Field("):
                imports.add("pydantic.Field")
                default = f" = {text}"
            else:
                ann = str(text)
        lines.append(f"    {p.name}: {ann}{default}")
    override_text = " ".join(str(v) for v in overrides.values())
    imports.update(f"typing.{name}" for name in _TYPING_NAMES if re.search(rf"\b{name}\[", override_text))
    typing_names = sorted(i.split(".", 1)[1] for i in imports if i.startswith("typing."))
    pydantic_names = sorted(i.split(".", 1)[1] for i in imports if i.startswith("pydantic."))
    modules = sorted(i.split(":", 1)[1] for i in imports if i.startswith("module:"))
    source_ref = f"{getattr(target, '__module__', '?')}:{getattr(target, '__qualname__', '?')}"
    hint = regenerate_hint if regenerate_hint is not None else f"python -m pyutilz.dev.signature_models {source_ref} --name {class_name}"
    out = [
        f'"""{docstring if docstring is not None else f"Strict parameters of `{source_ref}`."}',
        "",
        "GENERATED from the signature; do not edit by hand. A meta-test compares it with the live signature",
        f"(`pyutilz.dev.signature_models.signature_drift`). Regenerate: `{hint}`.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
    ]
    if typing_names:
        out.append(f"from typing import {', '.join(typing_names)}")
    out.extend(f"import {m}" for m in modules)
    out.append(f"from pydantic import {', '.join(pydantic_names)}")
    out.extend(
        [
            "",
            "",
            f"class {class_name}(BaseModel):",
            f'    """Parameters of `{source_ref}`; an unknown name or a wrong type raises when this is instantiated."""',
            "",
            '    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)',
            f"    __signature_overrides__ = {tuple(sorted(overrides))!r}",
            f"    __signature_skip_default__ = {tuple(skip_default)!r}",
            "",
            *(lines if lines else ["    pass"]),
            "",
        ]
    )
    return "\n".join(out)


def _load_target(spec: str) -> Callable[..., Any]:
    """Resolve ``package.module:Qualified.Name`` to the object it names."""
    module_name, _, qual = spec.partition(":")
    if not qual:
        raise SystemExit(f"target must look like 'package.module:Name', got {spec!r}")
    obj: Any = importlib.import_module(module_name)
    for part in qual.split("."):
        obj = getattr(obj, part)
    return obj  # type: ignore[no-any-return]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: print or write the generated model module for ``package.module:Name``."""
    parser = argparse.ArgumentParser(prog="python -m pyutilz.dev.signature_models", description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("target", help="package.module:Name of the class or function whose signature to mirror")
    parser.add_argument("--name", default=None, help="name of the generated class (default: <Target>Config)")
    parser.add_argument("--exclude", nargs="*", default=[], help="parameters to leave out of the model")
    parser.add_argument("-o", "--out", default=None, help="write to this file instead of stdout")
    parser.add_argument("--check", action="store_true", help="with -o: exit 1 if the file on disk differs from what would be generated")
    args = parser.parse_args(argv)
    target = _load_target(args.target)
    name = args.name if args.name is not None else f"{getattr(target, '__name__', 'Target')}Config"
    source = render_model_source(target, name, exclude=args.exclude)
    if args.out is None:
        sys.stdout.write(source)
        return 0
    path = Path(args.out)
    if args.check:
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if current.replace("\r\n", "\n") != source:
            sys.stdout.write(f"{path} is out of date; regenerate it without --check\n")
            return 1
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

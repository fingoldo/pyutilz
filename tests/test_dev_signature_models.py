"""``pyutilz.dev.signature_models``: strict models generated from a signature, their rendered source, and the drift check."""

from __future__ import annotations

import importlib.util
import sys
import typing
from typing import Any, Dict, List, Optional

import pytest
from pydantic import ValidationError

from pyutilz.dev.signature_models import main, model_from_signature, render_model_source, signature_drift, signature_parameters


class _Selector:
    """A stand-in estimator with an annotated, a defaulted and an unannotated parameter."""

    def __init__(self, n_features: int, alpha: float = 0.5, mode: str = "fast", names: Optional[List[str]] = None, extra_opts: Dict[str, int] = {}, loose=None, **kwargs: Any) -> None:  # noqa: B006
        """Keep the arguments; only the signature matters here."""


def test_the_model_mirrors_names_defaults_and_requiredness():
    """One field per parameter; a parameter without a default is required; *args/**kwargs and self are not fields."""
    model = model_from_signature(_Selector, name="SelectorConfig")
    assert list(model.model_fields) == ["n_features", "alpha", "mode", "names", "extra_opts", "loose"]
    assert model.model_fields["n_features"].is_required()
    cfg = model(n_features=3)
    assert (cfg.alpha, cfg.mode, cfg.names, cfg.extra_opts, cfg.loose) == (0.5, "fast", None, {}, None)


def test_an_unknown_parameter_raises_at_instantiation():
    """The point of the exercise: a typo is an error now, not a swallowed key."""
    model = model_from_signature(_Selector)
    with pytest.raises(ValidationError, match="not_a_param"):
        model(n_features=3, not_a_param=0.1)


def test_a_wrong_type_raises_at_instantiation():
    """Annotations are enforced, so a string where a float belongs fails immediately."""
    model = model_from_signature(_Selector)
    with pytest.raises(ValidationError):
        model(n_features=3, alpha="high")


def test_the_model_is_frozen():
    """A validated config cannot be mutated into an invalid one afterwards."""
    cfg = model_from_signature(_Selector)(n_features=3)
    with pytest.raises(ValidationError):
        cfg.alpha = 0.9


def test_overrides_tighten_a_field_without_touching_the_target():
    """A Literal override rejects other strings; a constrained override rejects an out-of-range number; the default is kept."""
    from typing import Literal

    model = model_from_signature(_Selector, overrides={"mode": Literal["fast", "slow"], "alpha": (float, {"gt": 0, "le": 1})})
    assert model(n_features=1).mode == "fast"
    with pytest.raises(ValidationError):
        model(n_features=1, mode="medium")
    with pytest.raises(ValidationError):
        model(n_features=1, alpha=1.5)


def test_exclude_leaves_a_parameter_out_and_extra_is_then_rejected():
    """An excluded parameter (one the caller must not set) is rejected like any unknown key."""
    model = model_from_signature(_Selector, exclude=["n_features"])
    assert "n_features" not in model.model_fields
    with pytest.raises(ValidationError):
        model(n_features=3)


def test_unannotated_error_mode_refuses_an_unannotated_parameter():
    """Strict callers can insist that every parameter carries an annotation."""
    with pytest.raises(TypeError, match="loose"):
        model_from_signature(_Selector, unannotated="error")


def test_drift_is_empty_when_nothing_changed_and_names_every_difference_when_it_did():
    """A model built from an older signature reports the added parameter, the changed default and the removed one."""

    def old(a: int, b: float = 1.0, gone: str = "x") -> None:
        """Previous signature."""

    def new(a: int, b: float = 2.0, added: bool = False) -> None:
        """Current signature."""

    model = model_from_signature(old)
    assert signature_drift(model, old) == []
    problems = signature_drift(model, new)
    assert any("'added'" in p and "not in" in p for p in problems)
    assert any("'gone'" in p and "no longer" in p for p in problems)
    assert any("'b'" in p and "default" in p for p in problems)


def test_drift_catches_a_parameter_that_became_required_and_a_changed_annotation():
    """Required-ness and annotation changes are drift too; an overridden field is exempt from the annotation check only."""

    def old(a: int = 1, b: int = 2) -> None:
        """Previous signature."""

    def new(a: int, b: str = 2) -> None:  # type: ignore[assignment]
        """Current signature."""

    model = model_from_signature(old)
    problems = signature_drift(model, new)
    assert any("'a'" in p and "required" in p for p in problems)
    assert any("'b'" in p and "annotation" in p for p in problems)
    tightened = model_from_signature(old, overrides={"b": (int, {"ge": 0})})
    assert not any("annotation" in p for p in signature_drift(tightened, new))


def test_rendered_source_imports_and_behaves_like_the_runtime_model(tmp_path, monkeypatch):
    """The committed form: render, import without the target, get the same fields, and see no drift against the target."""
    source = render_model_source(_Selector, "SelectorConfig", overrides={"mode": 'Literal["fast", "slow"]', "alpha": "Field(0.5, gt=0, le=1)"})
    path = tmp_path / "selector_config.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("selector_config_generated", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "selector_config_generated", module)
    spec.loader.exec_module(module)
    model = module.SelectorConfig
    assert list(model.model_fields) == [p.name for p in signature_parameters(_Selector)]
    assert model(n_features=2).mode == "fast"
    with pytest.raises(ValidationError):
        model(n_features=2, mode="medium")
    with pytest.raises(ValidationError):
        model(n_features=2, nope=1)
    assert signature_drift(model, _Selector) == []


def test_cli_writes_and_checks_the_generated_file(tmp_path, capsys):
    """``-o`` writes the module; ``--check`` passes on an identical file and fails once the file is stale."""
    out = tmp_path / "gen.py"
    target = f"{__name__}:_Selector"
    assert main([target, "--name", "SelectorConfig", "-o", str(out)]) == 0
    assert "class SelectorConfig" in out.read_text(encoding="utf-8")
    assert main([target, "--name", "SelectorConfig", "-o", str(out), "--check"]) == 0
    out.write_text(out.read_text(encoding="utf-8") + "\n# stale\n", encoding="utf-8")
    assert main([target, "--name", "SelectorConfig", "-o", str(out), "--check"]) == 1
    assert "out of date" in capsys.readouterr().out


class _Color(__import__("enum").Enum):
    """An enum used as an annotation and a default (no source form for the default)."""

    RED = "red"


class _Heavy:
    """A target whose parameters have a non-literal default and an enum annotation."""

    def __init__(self, dtype: type = __import__("numpy").int32, color: _Color = _Color.RED, mode: "typing.Literal['a.b', 'c']" = "c") -> None:
        """Keep the arguments; only the signature matters here."""


def test_a_default_without_a_source_form_is_rendered_as_any_and_left_out_of_the_drift_check(tmp_path, monkeypatch):
    """dtype/enum defaults cannot be written into a module: the field accepts anything, defaults to None and is not reported as drift."""
    path = tmp_path / "heavy_config.py"
    path.write_text(render_model_source(_Heavy, "HeavyConfig"), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("heavy_config_generated", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "heavy_config_generated", module)
    spec.loader.exec_module(module)
    assert module.HeavyConfig().dtype is None
    assert module.HeavyConfig(dtype=int).dtype is int
    assert signature_drift(module.HeavyConfig, _Heavy) == []
    with pytest.raises(ValidationError):
        module.HeavyConfig(mode="z")


class _Elsewhere:
    """A class from a module the generated file must not import."""


def _target_with_foreign_annotation(a: _Elsewhere, b: int = 1) -> None:
    """A target with one annotation naming a foreign class."""


def test_allowed_modules_degrades_a_foreign_class_annotation_to_any():
    """With ``allowed_modules`` the generated file never imports the target's package just to spell an annotation."""
    source = render_model_source(_target_with_foreign_annotation, "ForeignConfig", allowed_modules=("collections",))
    assert "a: Any" in source
    assert "import tests.test_dev_signature_models" not in source and "import test_dev_signature_models" not in source
    unrestricted = render_model_source(_target_with_foreign_annotation, "ForeignConfig")
    assert "a: tests.test_dev_signature_models._Elsewhere" in unrestricted and "import tests.test_dev_signature_models" in unrestricted


def test_typing_names_used_only_in_an_override_are_imported():
    """An override that spells ``Union[...]`` needs ``Union`` imported even when no signature annotation used it."""
    source = render_model_source(_Selector, "SelectorConfig", overrides={"alpha": 'Union[float, Literal["auto"]]'})
    assert "Union" in source.split("from pydantic")[0] and "Literal" in source.split("from pydantic")[0]

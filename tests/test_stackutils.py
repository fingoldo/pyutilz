"""Behaviour of ``pyutilz.core.pythonlib.stackutils``: soft field access, the parameter capture and restore
round trip through an object, and the caller's arguments read from the call stack."""

from __future__ import annotations

from pyutilz.core.pythonlib.stackutils import (
    get_or_warn,
    get_parent_func_args,
    load_object_params_into_func,
    store_params_in_object,
)


class _Holder:
    pass


class TestGetOrWarn:
    def test_returns_the_present_field(self):
        assert get_or_warn({"a": 1}, "a", "target") == 1

    def test_a_missing_field_returns_none_and_names_the_target_in_the_warning(self, caplog):
        with caplog.at_level("WARNING"):
            assert get_or_warn({}, "b", "target") is None
        assert any("target" in rec.getMessage() for rec in caplog.records)


class TestParameterRoundTrip:
    def test_store_then_load_restores_the_same_mapping(self):
        obj = _Holder()
        store_params_in_object(obj, {"x": 1, "y": "z"})
        assert load_object_params_into_func(obj, {}) == {"x": 1, "y": "z"}

    def test_the_default_postfix_marks_the_attributes(self):
        obj = _Holder()
        store_params_in_object(obj, {"x": 1})
        assert getattr(obj, "x_param_") == 1

    def test_load_writes_into_the_given_locals_as_well(self):
        obj = _Holder()
        store_params_in_object(obj, {"x": 1})
        scope: dict = {}
        load_object_params_into_func(obj, scope)
        assert scope == {"x": 1}

    def test_a_none_object_yields_an_empty_mapping(self):
        assert load_object_params_into_func(None, {}) == {}


class TestParentFuncArgs:
    def test_the_calling_function_arguments_are_read_from_the_stack(self):
        def callee(self, a=1, b=2):
            return get_parent_func_args()

        class Caller:
            def method(self, q=3):
                return callee(self, 5)

        assert Caller().method() == {"a": 5, "b": 2}

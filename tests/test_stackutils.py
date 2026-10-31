"""Behaviour of ``pyutilz.core.pythonlib.stackutils``: soft field access, the parameter capture and restore
round trip through an object, and the caller's arguments read from the call stack."""

from __future__ import annotations

from pyutilz.core.pythonlib import stackutils
from pyutilz.core.pythonlib.stackutils import (
    get_or_warn,
    lookup_in_stack,
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


class TestGetOrWarnMessage:
    def test_the_warning_names_the_field_then_the_target_then_the_object(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger=stackutils.logger.name):
            get_or_warn({"x": 1}, "b", "target")
        assert [r.getMessage() for r in caplog.records] == ["No b field in target {'x': 1}"]


_PROBE_SOURCE = "def probe(name):\n    return lookup_in_stack(name)\n"


class TestLookupInStack:
    def test_a_truthy_global_of_the_immediate_caller_is_found(self):
        # The probe lives in its own namespace, so no outer frame defines the name.
        ns = {"lookup_in_stack": lookup_in_stack, "_PROBE": 7}
        exec(_PROBE_SOURCE, ns)
        assert ns["probe"]("_PROBE") == 7

    def test_a_falsy_global_is_skipped_and_gives_none(self):
        ns = {"lookup_in_stack": lookup_in_stack, "_FALSY": 0}
        exec(_PROBE_SOURCE, ns)
        assert ns["probe"]("_FALSY") is None

    def test_a_missing_name_gives_none(self):
        assert lookup_in_stack("_NO_SUCH_NAME_ANYWHERE_123") is None

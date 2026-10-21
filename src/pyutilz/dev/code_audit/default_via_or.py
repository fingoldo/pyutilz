"""(internal) default_via_or now lives in py_ci_shared.code_audit.default_via_or; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.default_via_or import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _BOOLEAN_CONTEXT_FIELDS,
    _BOOLEAN_VALUED_CALL_NAMES,
    _BOOLEAN_VALUED_METHOD_NAMES,
    _DOCUMENTED_SAFE_LHS_FUNCS,
    _PATH_TYPE_NAMES,
    _RE_MATCH_FUNCS,
    _TRANSPARENT_WRAPPER_FUNCS,
    _annotation_is_path,
    _assigned_names_and_values,
    _bool_declared_params,
    _build_parent_field_map,
    _classify,
    _enclosing_symbols,
    _expr_repr,
    _get_call_key,
    _get_getattr_call,
    _is_alias_key_fallback,
    _is_alternative_source_read,
    _is_boolean_valued,
    _is_constructor_call,
    _is_env_get_call,
    _is_getattr_alias_fallback,
    _is_in_boolean_context,
    _is_predicate_name,
    _is_regex_group_alternation,
    _is_regex_pattern_alternation,
    _is_same_key_on_other_receiver,
    _is_text_fallback,
    _is_trivial_default,
    _is_wrapped_in_bool_call,
    _lhs_default_is_also_the_or_fallback,
    _lhs_is_documented_safe,
    _module_boolean_names,
    _module_path_names,
    _pair_is_exempt,
    _site,
    _unwrap_lhs,
    logger,
    scan_default_via_or_trap,
)

from py_ci_shared.code_audit import default_via_or as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl

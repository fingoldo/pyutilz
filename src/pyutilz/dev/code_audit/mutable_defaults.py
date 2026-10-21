"""(internal) mutable_defaults now lives in py_ci_shared.code_audit.mutable_defaults; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.mutable_defaults import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _KNOWN_IMMUTABLE_SCALAR_NAMES,
    _MUTABLE_CALLS,
    _MUTABLE_FACTORIES,
    _MUTABLE_LITERAL_TYPES,
    _MUTATING_METHODS,
    _SHARED_BUFFER_NAMES,
    _SHARED_BUFFER_SUFFIXES,
    _augassign_target_is_a_scalar_counter,
    _branch_paths,
    _branches_are_exclusive,
    _find_param_aliasing_mutation,
    _is_known_immutable_scalar_annotation,
    _is_mutable_default,
    _looks_like_a_shared_output_buffer,
    _mutated_alias,
    _param_annotations,
    _param_default_looks_like_immutable_scalar,
    _param_is_mutated,
    _shadows_param,
    _track_param_alias,
    _walk_skipping_shadowed_scopes,
    scan_mutable_defaults,
    scan_parameter_aliasing_mutation,
)

from py_ci_shared.code_audit import mutable_defaults as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl

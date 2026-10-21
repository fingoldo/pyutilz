"""(internal) The code-audit base now lives in py_ci_shared.code_audit._base. This module re-exports every name under the
old path, so the detectors' relative imports keep working until the consumers move to py_ci_shared.code_audit."""

from py_ci_shared.code_audit._base import (  # noqa: F401 - re-exported for the detectors in this package
    CACHING_DECORATOR_NAMES,
    Finding,
    SEVERITIES,
    UNKNOWN_SEVERITY_RANK,
    _CACHE_LOCK,
    _DEFAULT_EXCLUDE_DIRS,
    _EXEMPT_RE,
    _PARSE_CACHE,
    _PARSE_CACHE_MAX_ENTRIES,
    _PY_EXTS,
    _SEVERITY_RANK,
    _SQL_FROM,
    _SQL_INTERPOLATION,
    _SRC_LINES_CACHE,
    _TEST_DIR_NAMES,
    _arg_names,
    _dotted_module_path,
    _is_excluded,
    _iter_py_files,
    _line_text,
    _module_aliases,
    _module_sql_constants,
    _own_nodes,
    _read_src_lines,
    _safe_parse,
    _sql_table_of,
    _sql_text,
    _string_constants_in_body,
    _subscript_index,
    clear_parse_cache,
    decorator_name,
    dotted_name,
    is_cached,
    is_line_exempt,
    is_locals_or_globals_call,
    is_test_file,
    logger,
    module_level_names,
    severity_rank,
    split_src_lines,
    walk_tree,
)

import sys as _sys

from py_ci_shared.code_audit import _base as _impl

# Runtime identity: a test that rebinds a module constant here must reach the code that reads it, so the old path IS the new module.
_sys.modules[__name__] = _impl

"""(internal) dead_import now lives in py_ci_shared.code_audit.dead_import; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.dead_import import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _alias_own_lineno,
    _attribute_names_in_corpus,
    _corpus_import_index,
    _from_import_bindings,
    _from_import_names_by_file,
    _imported_bindings,
    _is_consumed_by_another_files_from_import,
    _names_referenced_in_file,
    _plain_import_bindings,
    scan_possibly_dead_import,
)

from py_ci_shared.code_audit import dead_import as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl

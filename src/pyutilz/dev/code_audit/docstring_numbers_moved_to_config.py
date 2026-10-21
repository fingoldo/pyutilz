"""(internal) docstring_numbers_moved_to_config now lives in py_ci_shared.code_audit.docstring_numbers_moved_to_config; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.docstring_numbers_moved_to_config import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _ABBREVIATION_RE,
    _CONFIG_READERS,
    _FILENAME_RE,
    _NAMES_A_SOURCE_RE,
    _NUMBER_RE,
    _REFERENCE_RE,
    _THRESHOLD_PROSE_RE,
    _YEAR_RE,
    _body_numbers,
    _docstring_numbers,
    _reads_configuration,
    scan_docstring_numbers_moved_to_config,
)

from py_ci_shared.code_audit import docstring_numbers_moved_to_config as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl

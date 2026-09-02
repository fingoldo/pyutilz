"""Closes gaps in python itself.

Works on general matters such as types, reading settings, etc.

Historically a single flat module; split into a subpackage of cohesive submodules once it grew
past the repo's 1000-LOC limit. This ``__init__`` re-exports the entire original surface (public
AND private names, module-level imports included) so that ``from pyutilz.core.pythonlib import X``
and the lazy ``pyutilz.pythonlib`` alias keep resolving unchanged, and so that patch targets such
as ``pyutilz.core.pythonlib.subprocess.check_call`` keep pointing at the very same objects.

Submodules:
  _common     - logger plus every stdlib/third-party name the flat module bound at top level
  packages    - ensure_installed
  objects     - object/dict/sequence operations and ordering helpers
  numerics    - numeric parsing and digit-level inspection
  datetimes   - time/date/timestamp conversion and formatting
  stackutils  - call-stack introspection and parameter capture/restore
  filesystem  - paths, filenames, object dump/load, shelve, stream redirection
  hardware    - Jupyter/CUDA/CPU-flag probing
"""

# pylint: disable=wrong-import-order,wrong-import-position,unidiomatic-typecheck,pointless-string-statement

from ._common import (
    Any,
    Callable,
    Iterable,
    Literal,
    MappingABC,
    Optional,
    Sequence,
    Set,
    SizedABC,
    Tuple,
    Union,
    abspath,
    contextlib,
    contextmanager,
    date,
    datetime,
    errno,
    exists,
    importlib,
    inspect,
    joblib,
    join,
    logger,
    logging,
    lru_cache,
    njit,
    numbers,
    os,
    portalocker,
    re,
    shelve,
    subprocess,
    sys,
    time,
    timezone,
)

from .packages import (
    ensure_installed,
)

from .objects import (
    _GET_ATTR_UNSET,
    anyof_elements_in_string,
    batch,
    ensure_dict_elem,
    ensure_list_set_tuple,
    filter_elements_by_type,
    flatten_keys_to_dict,
    flatten_keys_to_set,
    get_attr,
    keys_changed_enough,
    list_is_non_decreasing,
    list_is_non_increasing,
    populate_object_from_dict,
    prefix_dict_elems,
    show_methods,
    sort_dict_by_key,
    sort_dict_by_value,
    unpack_counter,
)

from .numerics import (
    count_trailing_zeros,
    float_distinct_digits_percent,
    integer_digits,
    is_float,
    to_float,
)

from .datetimes import (
    age,
    datetime_to_unix_ts,
    datetime_to_utc_timestamp,
    get_utc_unix_ts_milliseconds,
    get_utc_unix_ts_seconds,
    imitate_delay,
    read_timezoned_ts,
    utc_to_local,
    utc_ts_2_locstr,
    weekofmonth,
)

from .stackutils import (
    get_or_warn,
    get_parent_func_args,
    load_object_params_into_func,
    lookup_in_stack,
    store_params_in_object,
)

from .filesystem import (
    HashableDict,
    ObjectsAndFilesProcessor,
    ObjectsDumper,
    ObjectsLoader,
    ensure_valid_filename,
    get_human_readable_set_size,
    get_partitioned_filepath,
    load_file,
    open_safe_shelve,
    suppress_stdout_stderr,
)

from .hardware import (
    _ensure_cuda_home_from_pip,
    check_cpu_flag,
    is_cuda_available,
    is_jupyter_notebook,
)

# ----------------------------------------------------------------------------------------------------------------------------
# Hide the submodule names so ``dir(pyutilz.core.pythonlib)`` matches the pre-split flat module
# exactly. Explicit ``from . import`` here (rather than relying on the implicit submodule binding
# the ``from .X import name`` statements above already produce) so mypy sees a real binding to
# delete instead of needing an unreliable ignore comment.
# ----------------------------------------------------------------------------------------------------------------------------

from . import _common, packages, objects, numerics, datetimes, stackutils, filesystem, hardware

# Deleting these bindings would make ``dir()`` depend on import ORDER rather than on a fixed
# property of the module: any later access through the import system (a
# ``mock.patch("<pkg>.<submodule>.name")`` target, a plain ``import <pkg>.<submodule>``
# anywhere in the process) re-sets the deleted attribute as a side effect, while code that
# runs before that sees an AttributeError. A ``__dir__`` hook (PEP 562) reports the same
# curated surface without ever removing the real attribute, so every access path stays
# reliable no matter what has been imported first -- the idiom pyutilz.system.system uses.
_SUBMODULE_NAMES = frozenset({"_common", "packages", "objects", "numerics", "datetimes", "stackutils", "filesystem", "hardware"})


def __dir__() -> list:
    """Reports the facade's curated surface, hiding the submodule names the re-exports came from."""
    return sorted(n for n in globals() if n not in _SUBMODULE_NAMES)

# Explicit public surface: makes this facade's re-export intent self-documenting and gives the
# self-scan code-audit possibly_dead_import check an authoritative usage signal for names only
# ever consumed by external repos (``from pyutilz.core.pythonlib import X``). The two private
# names (``_GET_ATTR_UNSET``, ``_ensure_cuda_home_from_pip``) stay importable as attributes but
# are deliberately absent here, matching the flat module's own notion of its public surface.
__all__ = [
    "Any",
    "Callable",
    "HashableDict",
    "Iterable",
    "Literal",
    "MappingABC",
    "ObjectsAndFilesProcessor",
    "ObjectsDumper",
    "ObjectsLoader",
    "Optional",
    "Sequence",
    "Set",
    "SizedABC",
    "Tuple",
    "Union",
    "abspath",
    "age",
    "anyof_elements_in_string",
    "batch",
    "check_cpu_flag",
    "contextlib",
    "contextmanager",
    "count_trailing_zeros",
    "date",
    "datetime",
    "datetime_to_unix_ts",
    "datetime_to_utc_timestamp",
    "ensure_dict_elem",
    "ensure_installed",
    "ensure_list_set_tuple",
    "ensure_valid_filename",
    "errno",
    "exists",
    "filter_elements_by_type",
    "flatten_keys_to_dict",
    "flatten_keys_to_set",
    "float_distinct_digits_percent",
    "get_attr",
    "get_human_readable_set_size",
    "get_or_warn",
    "get_parent_func_args",
    "get_partitioned_filepath",
    "get_utc_unix_ts_milliseconds",
    "get_utc_unix_ts_seconds",
    "imitate_delay",
    "importlib",
    "inspect",
    "integer_digits",
    "is_cuda_available",
    "is_float",
    "is_jupyter_notebook",
    "joblib",
    "join",
    "keys_changed_enough",
    "list_is_non_decreasing",
    "list_is_non_increasing",
    "load_file",
    "load_object_params_into_func",
    "logger",
    "logging",
    "lookup_in_stack",
    "lru_cache",
    "njit",
    "numbers",
    "open_safe_shelve",
    "os",
    "populate_object_from_dict",
    "portalocker",
    "prefix_dict_elems",
    "re",
    "read_timezoned_ts",
    "shelve",
    "show_methods",
    "sort_dict_by_key",
    "sort_dict_by_value",
    "store_params_in_object",
    "subprocess",
    "suppress_stdout_stderr",
    "sys",
    "time",
    "timezone",
    "to_float",
    "unpack_counter",
    "utc_to_local",
    "utc_ts_2_locstr",
    "weekofmonth",
]

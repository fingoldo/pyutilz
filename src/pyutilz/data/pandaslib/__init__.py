"""``pyutilz.data.pandaslib`` -- pandas/polars dataframe utilities.

Historically a single flat module; split into a subpackage of cohesive
submodules (``dtypes``, ``frames``, ``io_ops``, ``benchmarks``) once it grew
past the 1000-LOC limit. This ``__init__`` re-exports the entire original
public surface so that ``from pyutilz.data.pandaslib import X`` and the lazy
``from pyutilz.pandaslib import X`` alias keep resolving unchanged.
"""

# ----------------------------------------------------------------------------------------------------------------------------
# Shared module-level names (imports/constants) preserved from the flat module
# ----------------------------------------------------------------------------------------------------------------------------

from ._common import (
    logging,
    logger,
    ensure_installed,
    Union,
    Optional,
    Sequence,
    Dict,
    List,
    Tuple,
    gc,
    io,
    os,
    warnings,
    np,
    pd,
    pl,
    find_between,
    defaultdict,
    dataset,
    to_float,
    ctypes,
    Array,
    HAS_IPYTHON,
    tqdmu,
    join,
    sep,
    glob,
    basename,
    dirname,
    splitext,
    exists,
    getsize,
    ensure_dir_exists,
    timer,
    chain,
    tempfile,
    shutil,
)

# IPython display helpers are only present when IPython is installed; mirror the
# original module which bound these names conditionally.
if HAS_IPYTHON:
    from ._common import display, Markdown, Latex

# ----------------------------------------------------------------------------------------------------------------------------
# Public functions/classes, re-exported from the cohesive submodules
# ----------------------------------------------------------------------------------------------------------------------------

from .dtypes import (
    set_df_columns_types,
    get_categorical_columns_indices,
    get_columns_of_type,
    optimize_dtypes,
    group_columns_by_dtype,
    classify_column_types,
    get_df_memory_consumption,
    ensure_dataframe_float32_convertability,
    convert_float64_to_float32,
)

from .frames import (
    nullify_standard_values,
    prefixize_columns,
    showcase_df_columns,
    FeatureNamer,
    share_dataframe,
    remove_stale_columns,
    get_non_stale_columns,
    get_suspiciously_constant_columns,
    remove_constant_columns,
)

from .io_ops import (
    load_df,
    concat_and_flush_df_list,
    read_stats_from_multiple_files,
    read_parquet_with_pyarrow,
)

from .benchmarks import (
    measure_read_write_performance,
    pack_benchmark_results,
    benchmark_dataframe_parquet_compression,
    benchmark_dataframe_pickle_compression,
    benchmark_dataframe_hdf_compression,
    benchmark_dataframe_csv_compression,
    benchmark_dataframe_orc_compression,
    benchmark_dataframe_feather_compression,
    benchmark_dataframe_compression,
)

# ----------------------------------------------------------------------------------------------------------------------------
# Hide the submodule names so ``dir(pyutilz.data.pandaslib)`` matches the
# pre-split flat module exactly (only public symbols, no submodule attributes).
# Explicit `from . import` here (rather than relying on the implicit submodule
# binding the `from .X import name` statements above already produce) so mypy
# sees a real binding to delete instead of needing an unreliable ignore comment.
# ----------------------------------------------------------------------------------------------------------------------------

from . import dtypes, frames, io_ops, benchmarks

# Deleting these bindings would make ``dir()`` depend on import ORDER rather than on a fixed
# property of the module: any later access through the import system (a
# ``mock.patch("<pkg>.<submodule>.name")`` target, a plain ``import <pkg>.<submodule>``
# anywhere in the process) re-sets the deleted attribute as a side effect, while code that
# runs before that sees an AttributeError. A ``__dir__`` hook (PEP 562) reports the same
# curated surface without ever removing the real attribute, so every access path stays
# reliable no matter what has been imported first -- the idiom pyutilz.system.system uses.
_SUBMODULE_NAMES = frozenset({"dtypes", "frames", "io_ops", "benchmarks"})


def __dir__() -> list:
    """Reports the facade's curated surface, hiding the submodule names the re-exports came from."""
    return sorted(n for n in globals() if n not in _SUBMODULE_NAMES)

# Explicit public surface: makes this facade's re-export intent self-documenting (rather than
# relying on nothing but "this is __init__.py") and gives the self-scan code-audit
# possibly_dead_import check an authoritative usage signal for names only ever consumed by
# external repos (``from pyutilz.data.pandaslib import X``), which its own-repo corpus scan
# can't see.
__all__ = [
    "logging",
    "logger",
    "ensure_installed",
    "Union",
    "Optional",
    "Sequence",
    "Dict",
    "List",
    "Tuple",
    "gc",
    "io",
    "os",
    "warnings",
    "np",
    "pd",
    "pl",
    "find_between",
    "defaultdict",
    "dataset",
    "to_float",
    "ctypes",
    "Array",
    "HAS_IPYTHON",
    "tqdmu",
    "join",
    "sep",
    "glob",
    "basename",
    "dirname",
    "splitext",
    "exists",
    "getsize",
    "ensure_dir_exists",
    "timer",
    "chain",
    "tempfile",
    "shutil",
    "set_df_columns_types",
    "get_categorical_columns_indices",
    "get_columns_of_type",
    "optimize_dtypes",
    "group_columns_by_dtype",
    "classify_column_types",
    "get_df_memory_consumption",
    "ensure_dataframe_float32_convertability",
    "convert_float64_to_float32",
    "nullify_standard_values",
    "prefixize_columns",
    "showcase_df_columns",
    "FeatureNamer",
    "share_dataframe",
    "remove_stale_columns",
    "get_non_stale_columns",
    "get_suspiciously_constant_columns",
    "remove_constant_columns",
    "load_df",
    "concat_and_flush_df_list",
    "read_stats_from_multiple_files",
    "read_parquet_with_pyarrow",
    "measure_read_write_performance",
    "pack_benchmark_results",
    "benchmark_dataframe_parquet_compression",
    "benchmark_dataframe_pickle_compression",
    "benchmark_dataframe_hdf_compression",
    "benchmark_dataframe_csv_compression",
    "benchmark_dataframe_orc_compression",
    "benchmark_dataframe_feather_compression",
    "benchmark_dataframe_compression",
]
if HAS_IPYTHON:
    __all__ += ["display", "Markdown", "Latex"]

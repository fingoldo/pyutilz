"""``pyutilz.data.polarslib`` -- common functions for working with Polars.

Historically a single flat module; split into a subpackage of cohesive
submodules (``columns``, ``aggregations``, ``binning``, ``frames``) once it grew
past the 1000-LOC limit. This ``__init__`` re-exports the entire original
public surface so that ``from pyutilz.data.polarslib import X`` and the lazy
``from pyutilz.polarslib import X`` alias keep resolving unchanged.
"""

# ----------------------------------------------------------------------------------------------------------------------------
# Shared module-level names (imports/constants/logger) preserved from the flat module
# ----------------------------------------------------------------------------------------------------------------------------

from ._common import (
    logging,
    logger,
    os,
    pl,
    cs,
    Any,
    Dict,
    Iterable,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    np,
    textwrap,
    Counter,
    clean_ram,
    is_cuda_available,
    check_cpu_flag,
    POLARS_DEFAULT_NUMAGGS,
    POLARS_DEFAULT_QUANTILES,
)

# ----------------------------------------------------------------------------------------------------------------------------
# Public functions/classes, re-exported from the cohesive submodules
# ----------------------------------------------------------------------------------------------------------------------------

from .columns import (
    _cols_matching,
    find_nan_cols,
    find_infinite_cols,
    clean_numeric,
    _PlFrameT,
    polars_castable_int_dtypes,
    cast_f64_to_f32,
    apply_agg_func_safe,
)

from .aggregations import (
    compute_concentrations,
    add_weighted_aggregates,
    build_aggregate_features_polars,
    create_ts_features_polars,
)

from .binning import (
    _group_freqs,
    _shannon_entropy,
    entropy_for_column,
    mi_for_column,
    _BIN_DTYPE_MAX,
    bin_numerical_columns,
)

from .frames import (
    drop_constant_columns,
    remove_constant_columns,
    polars_df_info,
)

# ----------------------------------------------------------------------------------------------------------------------------
# Hide the submodule names so ``dir(pyutilz.data.polarslib)`` matches the
# pre-split flat module exactly (only the original symbols, no submodule attributes).
# Explicit `from . import` here (rather than relying on the implicit submodule
# binding the `from .X import name` statements above already produce) so mypy
# sees a real binding to delete instead of needing an unreliable ignore comment.
# ----------------------------------------------------------------------------------------------------------------------------

from . import columns, aggregations, binning, frames

del columns, aggregations, binning, frames

# Explicit public surface: makes this facade's re-export intent self-documenting (rather than
# relying on nothing but "this is __init__.py") and gives the self-scan code-audit
# possibly_dead_import check an authoritative usage signal for names only ever consumed by
# external repos (``from pyutilz.data.polarslib import X``), which its own-repo corpus scan
# can't see.
__all__ = [
    "logging",
    "logger",
    "os",
    "pl",
    "cs",
    "Any",
    "Dict",
    "Iterable",
    "Literal",
    "Optional",
    "Tuple",
    "TypeVar",
    "np",
    "textwrap",
    "Counter",
    "clean_ram",
    "is_cuda_available",
    "check_cpu_flag",
    "POLARS_DEFAULT_NUMAGGS",
    "POLARS_DEFAULT_QUANTILES",
    "find_nan_cols",
    "find_infinite_cols",
    "clean_numeric",
    "polars_castable_int_dtypes",
    "cast_f64_to_f32",
    "apply_agg_func_safe",
    "compute_concentrations",
    "add_weighted_aggregates",
    "build_aggregate_features_polars",
    "create_ts_features_polars",
    "entropy_for_column",
    "mi_for_column",
    "bin_numerical_columns",
    "drop_constant_columns",
    "remove_constant_columns",
    "polars_df_info",
]

"""Column-level polars helpers: NaN/infinite column detection, non-finite cleaning and float32 downcasting.

Split out of the historical flat ``pyutilz.data.polarslib`` module; re-exported
from the package ``__init__`` to preserve the public import surface.
"""

from ._common import (
    TypeVar,
    cs,
    logger,
    pl,
)

# PROJECT IDIOM for a re-export package's submodules (see also pyutilz/data/pandaslib/frames.py and
# pyutilz/text/strings/_logproxy.py, which apply the same rule):
#   `import <parent> as _facade`   -- ALLOWED, and load-bearing.
#   `from <parent> import <name>`  -- FORBIDDEN at module top level.
# A re-export package's __init__ imports its submodules, so a submodule importing the parent back is
# a genuine cycle. Plain `import x` binds the PARTIALLY-INITIALISED sys.modules entry and defers every
# attribute lookup to call time, so it survives; `from x import name` needs the name to exist at import
# time and raises "cannot import name ... (most likely due to a circular import)". Deferring the lookup
# is also what makes the name patchable: a test patching `pyutilz.data.polarslib.clean_ram` is seen here,
# where a from-import would have snapshotted the original function.
import pyutilz.data.polarslib as _facade  # patchable-name indirection for clean_ram/is_cuda_available/check_cpu_flag

# ----------------------------------------------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------------------------------------------


def _cols_matching(df: pl.DataFrame, predicate: pl.Expr) -> pl.DataFrame:
    """Return a DataFrame keeping only the columns for which ``predicate`` (a single-row-per-column
    boolean aggregate expression, e.g. ``cs.numeric().is_infinite().any()``) evaluates True.

    2026-08-02 near-duplicate-function-body finding: find_nan_cols and find_infinite_cols
    independently duplicated this evaluate-then-filter-columns logic; extracted so a future
    caller (a third "find X cols" helper) doesn't paste a third copy.
    """
    meta = df.select(predicate)
    true_cols = meta.row(0)
    return df.select([col for col, val in zip(meta.columns, true_cols) if val is True])


def find_nan_cols(df: pl.DataFrame) -> pl.DataFrame:
    """Return a DataFrame keeping only the numeric columns that contain at least one NaN or null value.

    Polars distinguishes float NaN from missing/null (unlike pandas, where isna()/isnull() catch
    both), so both ``is_nan()`` and ``is_null()`` are checked -- an all-null column would otherwise
    be invisible to a caller expecting pandas-style semantics.
    """
    return _cols_matching(df, cs.numeric().is_nan().fill_null(False).any() | cs.numeric().is_null().any())


def find_infinite_cols(df: pl.DataFrame) -> pl.DataFrame:
    """Return a DataFrame keeping only the numeric columns that contain at least one infinite value."""
    return _cols_matching(df, cs.numeric().is_infinite().any())


def clean_numeric(expr: pl.Expr, nans_filler: float = 0.0, fill_nulls: bool = False) -> pl.Expr:
    """Replace non-finite floats (inf, -inf, NaN) with ``nans_filler``.

    Uses ``is_finite()`` rather than ``.replace([inf, -inf, NaN], ...)`` because polars
    ``replace`` matches via float equality, and ``NaN != NaN`` so the NaN branch never fires.
    Without this fix, downstream ``.cast(int)`` after a groupby-agg that produced NaN/inf
    (zero-weight groups in weighted-mean, single-row variance, etc.) raises
    ``InvalidOperationError: conversion from f64 to i64 failed for [inf, -inf, NaN]``.

    Nulls are PRESERVED by default. ``null.is_finite()`` is null and polars treats a null
    ``when`` predicate as false, so a naive ``when(is_finite)`` also rewrites missing values into
    ``nans_filler`` -- conflating "not computable" (a std/corr over a single-row group returns
    null, not NaN) with a genuinely measured value, which no downstream missingness audit can
    then recover. Pass ``fill_nulls=True`` to opt into the old behavior of filling them too.
    """
    filler = pl.lit(float(nans_filler))
    if fill_nulls:
        return pl.when(expr.is_finite()).then(expr).otherwise(filler)
    return pl.when(expr.is_null()).then(expr).when(expr.is_finite()).then(expr).otherwise(filler)


_PlFrameT = TypeVar("_PlFrameT", pl.DataFrame, pl.LazyFrame)


def polars_castable_int_dtypes() -> list:
    """Integer polars dtypes that the float32 downcast helpers operate on.

    Shared by ``cast_f64_to_f32`` and pandaslib's ``ensure_dataframe_float32_convertability`` so the
    two documented mirrors can't drift: Int128 only exists in newer polars, and hardcoding it in one
    of them raised AttributeError on exactly the older versions the hasattr guard exists for.
    """
    int_types = [pl.Int32, pl.UInt32, pl.Int64, pl.UInt64]
    if hasattr(pl, "Int128"):
        int_types.append(pl.Int128)
    return int_types


def cast_f64_to_f32(df: _PlFrameT) -> _PlFrameT:
    """Downcast Float64 AND the common integer dtypes to Float32 (not Int32) to shrink memory usage.

    Float32 has a 24-bit mantissa, so integers are only representable exactly up to 2**24
    (16,777,216); any Int64/UInt64/UInt32/Int128 column with values beyond that magnitude loses
    precision silently. This mirrors the sibling ``ensure_dataframe_float32_convertability``
    (pandaslib/dtypes.py), which does the same int->float32 cast for LightGBM compatibility.
    If you need integers preserved exactly, downcast to a smaller *integer* dtype instead.
    """
    # Int128 was added in polars 0.19.0, make it optional for older versions
    int_types = _facade.polars_castable_int_dtypes()
    if isinstance(df, pl.DataFrame):
        int_cols = df.select(pl.col(*int_types)).columns
        if int_cols:
            overflow = df.select([pl.col(c).abs().max().gt(2**24).alias(c) for c in int_cols]).row(0)
            lossy = [c for c, is_lossy in zip(int_cols, overflow) if is_lossy]
            if lossy:
                logger.warning("cast_f64_to_f32: integer column(s) %s have values beyond 2**24 -- Float32 cast will lose exact-integer precision.", lossy)
    return df.with_columns(pl.col(*int_types, pl.Float64).cast(pl.Float32))


def apply_agg_func_safe(expr: pl.Expr, func_name: str, nans_filler: float = 0.0) -> pl.Expr:
    """Apply :func:`clean_numeric` to ``expr`` when ``func_name`` is a stat prone to NaN (skew/kurtosis on near-constant data), otherwise return ``expr`` unchanged."""
    if func_name in ["skew", "kurtosis"]:
        return clean_numeric(expr, nans_filler=nans_filler)
    else:
        return expr

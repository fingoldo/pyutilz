"""Whole-frame polars helpers: constant-column dropping and a pandas-``.info()``-style summary.

Split out of the historical flat ``pyutilz.data.polarslib`` module; re-exported
from the package ``__init__`` to preserve the public import surface.
"""

from ._common import (
    Counter,
    cs,
    logger,
    pl,
    textwrap,
)


def drop_constant_columns(df: pl.DataFrame, max_log_text_width: int = 300, verbose: bool = False) -> pl.DataFrame:
    """Drop numeric columns whose min and max are equal (or missing/NaN), i.e. columns with no informative variation.

    Unlike ``pandaslib.frames.remove_constant_columns`` (the conceptually equivalent pandas
    function, which mutates its input DataFrame in place and returns None), polars frames are
    immutable -- this function returns a NEW DataFrame and does NOT touch the caller's ``df``.
    You must capture the return value (``df = drop_constant_columns(df)``); porting a
    `remove_constant_columns(df)`-style call-and-discard from the pandas idiom silently leaves
    the constant columns in place with no error or warning.

    Also importable as :func:`remove_constant_columns` (a plain alias, same function) for
    grep/discoverability against its pandas sibling's name.
    """
    # ----------------------------------------------------------------------------------------------------------------------------
    # Inits
    # ----------------------------------------------------------------------------------------------------------------------------

    all_num_cols = cs.numeric()

    # ----------------------------------------------------------------------------------------------------------------------------
    # Stats
    # ----------------------------------------------------------------------------------------------------------------------------

    stats_expr = [
        all_num_cols.min().name.suffix("_min"),
        all_num_cols.max().name.suffix("_max"),
    ]

    stats = df.lazy().select(stats_expr).collect().row(0, named=True)

    # ----------------------------------------------------------------------------------------------------------------------------
    # Deciding
    # ----------------------------------------------------------------------------------------------------------------------------

    dead_columns = []
    for col in cs.expand_selector(df.head(), all_num_cols):
        min_val, max_val = stats.get(f"{col}_min"), stats.get(f"{col}_max")
        # Exact equality, NOT np.allclose: allclose is a RELATIVE test (rtol=1e-5), so it declared
        # "constant" any column whose spread is under 1e-5 of its own magnitude -- large monotone
        # ids, epoch-microsecond timestamps and prices around 1e9 were silently dropped.
        if (min_val is None or max_val is None) or min_val == max_val:
            dead_columns.append(col)
    if dead_columns:
        if verbose:
            logger.warning(
                "Dropping %s columns with no change: %s",
                format(len(dead_columns), "_"),
                textwrap.shorten(", ".join(dead_columns), width=max_log_text_width),
            )
        df = df.drop(dead_columns)

    return df


remove_constant_columns = drop_constant_columns  # discoverability alias, see drop_constant_columns's own docstring


def polars_df_info(df: pl.DataFrame) -> str:
    """Build a pandas-``.info()``-style multi-line summary string for a polars DataFrame (shape, columns, dtype counts, estimated memory usage)."""
    lines = []
    lines.append(f"{type(df)}")
    # "Rows:", not pandas' "RangeIndex: N entries, 0 to N-1": polars has no index at all, so borrowing
    # that line described a structure the frame does not have and invited readers to reason about
    # index alignment / reindexing semantics that simply do not exist here.
    lines.append(f"Rows: {df.height}")
    if df.width > 0:
        first_col = df.columns[0]
        last_col = df.columns[-1]
        lines.append(f"Columns: {df.width} entries, {first_col} to {last_col}")
    else:
        lines.append("Columns: 0 entries")
    dtype_counts = Counter(str(dtype) for dtype in df.dtypes)
    dtype_str = ", ".join(f"{dtype}({count})" for dtype, count in sorted(dtype_counts.items()))
    lines.append(f"dtypes: {dtype_str}")
    # Auto-scale like pandas' .info() does: a hardcoded "GB" reported "0.0+ GB" for every frame
    # under ~50 MB, i.e. for essentially all interactive use.
    size = float(df.estimated_size(unit="b"))
    for unit in ("bytes", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            break
        size /= 1024.0
    lines.append(f"memory usage: {size:.1f}+ {unit}")
    return "\n".join(lines)

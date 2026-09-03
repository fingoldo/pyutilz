"""Discretization and information-theoretic feature scoring in polars: uniform binning with
Tukey-fence outlier clipping, Shannon entropy and mutual information over binned columns.

Split out of the historical flat ``pyutilz.data.polarslib`` module; re-exported
from the package ``__init__`` to preserve the public import surface.
"""

from ._common import (
    Any,
    Dict,
    Optional,
    Tuple,
    cs,
    logger,
    np,
    pl,
    textwrap,
    concat_horizontal_ragged,
)
from .columns import clean_numeric

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
# FS in polars
# ----------------------------------------------------------------------------------------------------------------------------


def _group_freqs(bins: pl.DataFrame, cols, drop_nulls: bool = False) -> Any:
    """Empirical frequency array (sums to 1) for a group-by of ``cols`` (a column name or a list
    of column names) over ``bins``.

    2026-08-02 near-duplicate-function-body finding: entropy_for_column and mi_for_column
    independently duplicated this group_by-then-normalize step (single-column for the marginal,
    two-column for the joint); extracted alongside ``_shannon_entropy`` below.

    With ``drop_nulls=False`` (the default, preserving historical behavior) polars ``group_by``
    emits null as its own group, so missingness contributes real probability mass to the entropy.
    Pass ``drop_nulls=True`` to restrict the estimate to fully-observed rows instead.
    """
    if drop_nulls:
        bins = bins.drop_nulls(subset=cols)
    if len(bins) == 0:
        return np.empty(0, dtype=np.float64)
    return bins.group_by(cols).agg(pl.len())["len"].to_numpy() / len(bins)


def _shannon_entropy(freqs: Any) -> float:
    """Shannon entropy (in nats) of a discrete frequency distribution that sums to 1."""
    return -np.sum(freqs * np.log(freqs))  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime


def entropy_for_column(bins: pl.DataFrame, col: str, drop_nulls: bool = False) -> float:
    """Compute the Shannon entropy (in nats) of the discrete (already-binned) values in ``bins[col]``.

    By default null is counted as its own bin category and so adds probability mass of its own
    (``bin_numerical_columns`` only guarantees null-free bins when ``fill_nulls=True``); pass
    ``drop_nulls=True`` to measure the observed values only.
    """
    return _shannon_entropy(_group_freqs(bins, col, drop_nulls=drop_nulls))


def mi_for_column(bins: pl.DataFrame, entropies: dict, col: str, target_col: str, drop_nulls: bool = False) -> float:
    """Compute the mutual information between binned columns ``col`` and ``target_col``, using precomputed marginal entropies
    from ``entropies`` and the joint entropy of the two columns (mi = H(col) + H(target) - H(col, target)).

    ``drop_nulls`` mirrors :func:`entropy_for_column`: with the default False, missingness is a
    category of its own and a feature whose only signal is its null pattern scores as informative.
    Pass True to score observed rows only -- note that in that mode the precomputed ``entropies``
    are IGNORED and both marginals are recomputed locally, because the identity
    ``mi = H(x) + H(y) - H(x, y)`` only holds when all three terms come from the same sample.
    Per-column marginals dropped nulls per column, so H(x), H(y) and H(x, y) were estimated on
    three different row populations and the result could exceed min(H(x), H(y)) -- impossible for
    a mutual information, and it ranked columns with complementary missingness highest.
    """
    if drop_nulls:
        complete = bins.drop_nulls(subset=[col, target_col])
        h_col = _shannon_entropy(_group_freqs(complete, col))
        h_target = _shannon_entropy(_group_freqs(complete, target_col))
        joint_entropy = _shannon_entropy(_group_freqs(complete, [col, target_col]))
    else:
        h_col, h_target = float(entropies[col]), float(entropies[target_col])
        joint_entropy = _shannon_entropy(_group_freqs(bins, [col, target_col]))
    return h_target + h_col - joint_entropy


_BIN_DTYPE_MAX: Dict[Any, int] = {
    pl.Int8: 127,
    pl.Int16: 32767,
    pl.Int32: 2147483647,
    pl.Int64: 9223372036854775807,
    pl.UInt8: 255,
    pl.UInt16: 65535,
    pl.UInt32: 4294967295,
    pl.UInt64: 18446744073709551615,
}


def bin_numerical_columns(
    df: pl.DataFrame,
    target_columns: list,
    binned_targets: Optional[pl.DataFrame] = None,
    clean_features: bool = True,
    clean_targets: bool = True,
    num_bins: int = 10,
    bin_dtype: Any = pl.Int8,
    exclude_columns: Optional[list] = None,
    min_nuniques_to_clip: int = 10,
    tukey_fences_multiplier: float = 3.0,
    fill_nulls: bool = True,
    fill_nans: bool = True,
    max_log_text_width: int = 300,
    verbose: bool = False,
) -> Tuple[pl.DataFrame, Optional[pl.DataFrame], dict, list, dict]:
    """Computes min, max, and quantiles of all numerical columns in one go.
    Decides which are outliers and adds clipping.
    Converts values into integer uniform bin ids.
    Suggest for dropping columns that do not change.
    """
    if exclude_columns is None:
        exclude_columns = []

    needed_max = num_bins - 1
    if bin_dtype in _BIN_DTYPE_MAX and needed_max > _BIN_DTYPE_MAX[bin_dtype]:
        for candidate in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
            if needed_max <= _BIN_DTYPE_MAX[candidate]:
                if verbose:
                    logger.warning("bin_dtype=%s can't hold num_bins-1=%s; auto-widening to %s", bin_dtype, needed_max, candidate)
                bin_dtype = candidate
                break
        else:
            raise ValueError(f"num_bins={num_bins} exceeds the range even of Int64 bin_dtype")

    # ----------------------------------------------------------------------------------------------------------------------------
    # Inits
    # ----------------------------------------------------------------------------------------------------------------------------

    columns_to_drop = []

    # Get existing columns from dataframe schema (works for both DataFrame and LazyFrame)
    df_columns = set(df.collect_schema().names()) if hasattr(df, "collect_schema") else set(df.columns)

    # Filter target_columns to only include columns that exist in the dataframe
    existing_target_columns = [col for col in target_columns if col in df_columns]
    if len(existing_target_columns) < len(target_columns):
        missing = set(target_columns) - set(existing_target_columns)
        if verbose:
            logger.warning("Ignoring %s target columns not found in dataframe: %s", len(missing), missing)

    all_num_cols = cs.numeric()
    if exclude_columns:
        existing_exclude = [col for col in exclude_columns if col in df_columns]
        if existing_exclude:
            all_num_cols = all_num_cols - cs.by_name(existing_exclude)
    if binned_targets is not None and existing_target_columns:
        all_num_cols = all_num_cols - cs.by_name(existing_target_columns)

    _facade.clean_ram()

    # ----------------------------------------------------------------------------------------------------------------------------
    # Compute stats for every column
    # ----------------------------------------------------------------------------------------------------------------------------

    if verbose > 1:
        logger.info("Computing Min/Max/Quantiles...")

    stats_expr = [
        all_num_cols.min().name.suffix("_min"),
        all_num_cols.max().name.suffix("_max"),
    ]
    if clean_features or (clean_targets and binned_targets is None and existing_target_columns):
        if clean_features:
            quantile_cols = all_num_cols
        else:
            quantile_cols = cs.by_name(existing_target_columns)
        stats_expr.extend(
            [
                quantile_cols.quantile(0.25).name.suffix("_q1"),
                quantile_cols.quantile(0.75).name.suffix("_q3"),
            ]
        )
    stats_df = df.lazy().select(stats_expr).collect()

    stats: dict
    if len(stats_df) > 0:
        stats = stats_df.row(0, named=True)
    else:
        stats = {}
    orig_stats = stats.copy()

    _facade.clean_ram()

    # ----------------------------------------------------------------------------------------------------------------------------
    # Features with no change (min==max) are reported & dropped.
    # ----------------------------------------------------------------------------------------------------------------------------

    dead_columns = []
    for col in cs.expand_selector(df.head(), all_num_cols):
        min_val, max_val = stats.get(f"{col}_min"), stats.get(f"{col}_max")
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
        columns_to_drop.extend(dead_columns)

        _facade.clean_ram()

    # ----------------------------------------------------------------------------------------------------------------------------
    # Outliers are clipped & reported.
    # ----------------------------------------------------------------------------------------------------------------------------

    public_clips = {}
    clips = {}
    if clean_features or clean_targets:
        for col in cs.expand_selector(df.head(), all_num_cols):
            if not clean_targets:
                if col in existing_target_columns:
                    continue
            if not clean_features:
                if clean_targets and col not in existing_target_columns:
                    continue

            q1, q3 = stats.get(f"{col}_q1"), stats.get(f"{col}_q3")
            min_val, max_val = stats.get(f"{col}_min"), stats.get(f"{col}_max")
            assert (
                q1 is not None and q3 is not None and min_val is not None and max_val is not None
            ), f"stats missing quantile/min/max entries for column {col!r} -- stats_expr and target_cols disagree"

            iqr = q3 - q1

            lower_fence = q1 - tukey_fences_multiplier * iqr
            upper_fence = q3 + tukey_fences_multiplier * iqr

            if upper_fence > lower_fence or (np.isneginf(min_val) or np.isinf(max_val)):
                is_outlier = False
                lower_bound = min_val
                upper_bound = max_val
                if max_val > upper_fence:
                    stats[f"{col}_max"] = upper_fence
                    upper_bound = upper_fence
                    is_outlier = True
                if min_val < lower_fence:
                    stats[f"{col}_min"] = lower_fence
                    lower_bound = lower_fence
                    is_outlier = True
                if is_outlier:
                    public_clips[col] = dict(lower_bound=lower_bound, upper_bound=upper_bound)
                    clips[col] = pl.col(col).clip(lower_bound=lower_bound, upper_bound=upper_bound)

    if clips:
        skipped_clips = []
        if min_nuniques_to_clip:
            # do not apply clipping if # of unique values is too low (under 10)
            n_uniques_dict = df.lazy().select(pl.col(clips.keys()).n_unique()).collect().row(0, named=True)
            for col, nuniques in n_uniques_dict.items():
                if nuniques < min_nuniques_to_clip:
                    for field in "min max".split():
                        stats[f"{col}_{field}"] = orig_stats[f"{col}_{field}"]
                    skipped_clips.append(col)
                    del public_clips[col]
                    del clips[col]
        if verbose:
            if clips:
                logger.warning(
                    "Clipping %s columns with outliers: %s",
                    format(len(clips), "_"),
                    textwrap.shorten(", ".join(clips.keys()), width=max_log_text_width),
                )
            if skipped_clips:
                logger.warning(
                    "Clipping of %s columns skipped due to nuniques<%s: %s",
                    format(len(skipped_clips), "_"),
                    format(min_nuniques_to_clip, "_"),
                    textwrap.shorten(", ".join(skipped_clips), width=max_log_text_width),
                )

    # ----------------------------------------------------------------------------------------------------------------------------
    # Binning performed.
    # ----------------------------------------------------------------------------------------------------------------------------

    if verbose > 1:
        logger.info("Binning columns...")

    # collect_schema().len() rather than .width: on a LazyFrame the latter emits a PerformanceWarning
    # for resolving the schema it resolves anyway.
    if df.collect_schema().len() == 0:
        # Every column was constant (a 1-row frame is constant by construction) or the frame came in
        # empty. Nothing left to bin: the null_count/selector reads below would collect to a 0-row
        # 0-column frame and .row(0) would raise a raw polars OutOfBoundsError the caller can't
        # distinguish from a polars bug.
        return pl.DataFrame(), binned_targets, public_clips, columns_to_drop, stats

    dead_columns = []
    bin_expressions = []

    if fill_nulls:
        cols_with_nulls = [key for key, value in df.lazy().select(pl.all().null_count()).collect().row(0, named=True).items() if value > 0]
    # Needed whether or not fill_nans is set: only float columns can hold NaN at all, so they are also
    # the only ones the NaN handling below may touch. cs.numeric() includes Decimal, for which every
    # NaN predicate (is_nan/is_not_nan, and therefore fill_nan) raises InvalidOperationError -- a
    # Decimal column is what pl.read_database returns for a SQL NUMERIC, i.e. an entirely ordinary input.
    cols_with_floats = cs.expand_selector(df.head(), all_num_cols & cs.float())

    for col in cs.expand_selector(df.head(), all_num_cols):
        if binned_targets is not None:
            if col in existing_target_columns:
                continue

        # Calculate bin edges based on min and max values
        min_val = stats.get(f"{col}_min")
        max_val = stats.get(f"{col}_max")
        assert min_val is not None and max_val is not None, f"stats missing min/max entries for column {col!r}"

        if min_val == max_val:
            dead_columns.append(col)
        else:

            # Define the binning expression
            bin_width = (max_val - min_val) / num_bins
            col_expr = clips.get(col, pl.col(col))
            if fill_nulls and (col in cols_with_nulls):
                col_expr = col_expr.fill_null(min_val)
            if fill_nans and (col in cols_with_floats):
                col_expr = clean_numeric(col_expr, nans_filler=min_val)

            binned_col = ((col_expr - min_val) / bin_width).floor()
            if col in cols_with_floats:
                # Bin 0 is where the column's true minimum lands, so folding NaN into it makes a missing
                # value indistinguishable from the smallest real observation -- and it used to happen even
                # with fill_nans=False, which is exactly what that flag asks not to happen. With filling on,
                # clean_numeric above has already mapped NaN to min_val (hence bin 0) so this is a no-op
                # agreeing with that path; with filling off, NaN becomes null, matching what fill_nulls=False
                # yields for a missing value and what .cast() needs to not raise on NaN.
                binned_col = binned_col.fill_nan(0 if fill_nans else None)
            binned_col = binned_col.clip(0, num_bins - 1).cast(bin_dtype)

            bin_expressions.append(binned_col)

    if dead_columns:
        if verbose:
            logger.warning(
                "Dropping %s columns with no change: %s",
                format(len(dead_columns), "_"),
                textwrap.shorten(", ".join(dead_columns), width=max_log_text_width),
            )
        df = df.drop(dead_columns)
        columns_to_drop.extend(dead_columns)

    # Apply all binning expressions in parallel
    bins = df.lazy().select(bin_expressions).collect()
    _facade.clean_ram()

    if binned_targets is not None:
        bins = concat_horizontal_ragged([bins, binned_targets], rechunk=True)
    elif existing_target_columns:
        binned_targets = bins.select(cs.by_name(existing_target_columns)).clone()

    return bins, binned_targets, public_clips, columns_to_drop, stats

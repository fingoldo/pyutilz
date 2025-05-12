"""Common functions for working with Polars"""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *
import numpy as np, pandas as pd
import polars as pl, polars.selectors as cs

from functools import partial

import psutil
import textwrap
from collections import defaultdict
from pyutilz.system import clean_ram

# ----------------------------------------------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------------------------------------------


def clean_numeric(expr: pl.Expr, nans_filler: float = 0.0) -> pl.Expr:
    return expr.replace([float("inf"), -float("inf"), float("nan")], nans_filler)


def cast_f64_to_f32(df: pl.DataFrame) -> pl.Expr:
    return df.with_columns(pl.col(pl.Float64).cast(pl.Float32))


def entropy_for_column(bins: pl.DataFrame, col: str) -> float:
    marginal_freqs = bins.group_by(col).agg(pl.len())["len"].to_numpy() / len(bins)
    return -np.sum(marginal_freqs * np.log(marginal_freqs))


def mi_for_column(bins: pl.DataFrame, entropies: dict, col: str, target_col: str) -> float:
    joint_freqs = bins.group_by([col, target_col]).agg(pl.len())["len"].to_numpy() / len(bins)
    joint_entropy = -np.sum(joint_freqs * np.log(joint_freqs))
    mi = entropies[target_col] + entropies[col] - joint_entropy
    return mi


# ----------------------------------------------------------------------------------------------------------------------------
# FS in polars
# ----------------------------------------------------------------------------------------------------------------------------


def bin_numerical_columns(
    df: pl.DataFrame,
    target_columns: list,
    binned_targets: pl.DataFrame = None,
    clean_features: bool = True,
    clean_targets: bool = True,
    num_bins: int = 10,
    bin_dtype: object = pl.Int8,
    exclude_columns: list = [],
    min_nuniques_to_clip: int = 10,
    tukey_fences_multiplier: float = 3.0,
    fill_nulls: bool = True,
    fill_nans: bool = True,
    max_log_text_width: int = 300,
    verbose: int = 1,
) -> pl.DataFrame:
    """Computes min, max, and quantiles of all numerical columns in one go.
    Decides which are outliers and adds clipping.
    Converts values into integer uniform bin ids.
    Suggest for dropping columns that do not change.
    """

    # ----------------------------------------------------------------------------------------------------------------------------
    # Inits
    # ----------------------------------------------------------------------------------------------------------------------------

    columns_to_drop = []

    all_num_cols = cs.numeric()
    if exclude_columns:
        all_num_cols = all_num_cols - cs.by_name(exclude_columns)
    if binned_targets is not None:
        all_num_cols = all_num_cols - cs.by_name(target_columns)

    clean_ram()

    # ----------------------------------------------------------------------------------------------------------------------------
    # Compute stats for every column
    # ----------------------------------------------------------------------------------------------------------------------------

    if verbose > 1:
        logger.info("Computing Min/Max/Quantiles...")

    stats_expr = [
        all_num_cols.min().name.suffix("_min"),
        all_num_cols.max().name.suffix("_max"),
    ]
    if clean_features or (clean_targets and binned_targets is None):
        if clean_features:
            quantile_cols = all_num_cols
        else:
            quantile_cols = cs.by_name(target_columns)
        stats_expr.extend(
            [
                quantile_cols.quantile(0.25).name.suffix("_q1"),
                quantile_cols.quantile(0.75).name.suffix("_q3"),
            ]
        )
    stats = df.select(stats_expr).collect().row(0, named=True)
    orig_stats = stats.copy()

    clean_ram()

    # ----------------------------------------------------------------------------------------------------------------------------
    # Features with no change (min==max) are reported & dropped.
    # ----------------------------------------------------------------------------------------------------------------------------

    dead_columns = []
    for col in cs.expand_selector(df, all_num_cols):
        min_val, max_val = stats.get(f"{col}_min"), stats.get(f"{col}_max")
        if (min_val is None and max_val is None) or np.allclose(min_val, max_val):
            dead_columns.append(col)
    if dead_columns:
        if verbose:
            logger.warning(f"Dropping {len(dead_columns):_} columns with no change: {textwrap.shorten(', '.join(dead_columns), width=max_log_text_width)}")
        df = df.drop(dead_columns)
        columns_to_drop.extend(dead_columns)

        clean_ram()

    # ----------------------------------------------------------------------------------------------------------------------------
    # Outliers are clipped & reported.
    # ----------------------------------------------------------------------------------------------------------------------------

    public_clips = {}
    clips = {}
    if clean_features or clean_targets:
        for col in cs.expand_selector(df, all_num_cols):
            if not clean_targets:
                if col in target_columns:
                    continue
            if not clean_features:
                if clean_targets and not (col in target_columns):
                    continue

            q1, q3 = stats.get(f"{col}_q1"), stats.get(f"{col}_q3")
            min_val, max_val = stats.get(f"{col}_min"), stats.get(f"{col}_max")

            iqr = q3 - q1

            lower_fence = q1 - tukey_fences_multiplier * iqr
            upper_fence = q3 + tukey_fences_multiplier * iqr

            if upper_fence > lower_fence:
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
            n_uniques_dict = df.select(pl.col(clips.keys()).n_unique()).collect().row(0, named=True)
            for col, nuniques in n_uniques_dict.items():
                if nuniques < min_nuniques_to_clip:
                    for field in "min max".split():
                        stats[f"{col}_{field}"] = orig_stats[f"{col}_{field}"]
                    skipped_clips.append(col)
                    del public_clips[col]
                    del clips[col]
        if verbose:
            if clips:
                logger.warning(f"Clipping {len(clips):_} columns with outliers: {textwrap.shorten(', '.join(clips.keys()), width=max_log_text_width)}")
            if skipped_clips:
                logger.warning(
                    f"Clipping of {len(skipped_clips):_} columns skipped due to nuniques<{min_nuniques_to_clip:_}: {textwrap.shorten(', '.join(skipped_clips), width=max_log_text_width)}"
                )

    # ----------------------------------------------------------------------------------------------------------------------------
    # Binning performed.
    # ----------------------------------------------------------------------------------------------------------------------------

    if verbose > 1:
        logger.info("Binning columns...")

    dead_columns = []
    bin_expressions = []

    if fill_nulls:
        cols_with_nulls = [key for key, value in df.select(pl.all().null_count()).collect().row(0, named=True).items() if value > 0]
    if fill_nans:
        cols_with_floats = cs.expand_selector(df, all_num_cols & cs.float())

    for col in cs.expand_selector(df, all_num_cols):
        if binned_targets is not None:
            if col in target_columns:
                continue

        # Calculate bin edges based on min and max values
        min_val = stats.get(f"{col}_min")
        max_val = stats.get(f"{col}_max")

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

            binned_col = ((col_expr - min_val) / bin_width).floor().clip(0, num_bins - 1).cast(bin_dtype)

            bin_expressions.append(binned_col)

    if dead_columns:
        if verbose:
            logger.warning(f"Dropping {len(dead_columns):_} columns with no change: {textwrap.shorten(', '.join(dead_columns), width=max_log_text_width)}")
        df = df.drop(dead_columns)
        columns_to_drop.extend(dead_columns)

    # Apply all binning expressions in parallel
    bins = df.select(bin_expressions).collect()
    clean_ram()

    if binned_targets is not None:
        bins = pl.concat([bins, binned_targets], how="horizontal", rechunk=True)
    else:
        binned_targets = bins.select(cs.by_name(target_columns)).clone()

    return bins, binned_targets, public_clips, columns_to_drop, stats


def drop_constant_columns(df: pl.DataFrame, max_log_text_width: int = 300, verbose: int = 1) -> pl.DataFrame:
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

    stats = df.select(stats_expr).collect().row(0, named=True)

    # ----------------------------------------------------------------------------------------------------------------------------
    # Deciding
    # ----------------------------------------------------------------------------------------------------------------------------

    dead_columns = []
    for col in cs.expand_selector(df, all_num_cols):
        min_val, max_val = stats.get(f"{col}_min"), stats.get(f"{col}_max")
        if (min_val is None and max_val is None) or np.allclose(min_val, max_val):
            dead_columns.append(col)
    if dead_columns:
        if verbose:
            logger.warning(f"Dropping {len(dead_columns):_} columns with no change: {textwrap.shorten(', '.join(dead_columns), width=max_log_text_width)}")
        df = df.drop(dead_columns)

    return df

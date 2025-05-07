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
import numpy as np
import polars as pl, polars.selectors as cs

# ----------------------------------------------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------------------------------------------


def clean_numeric(expr: pl.Expr, nans_filler: float = 0.0) -> pl.Expr:
    return expr.replace([float("inf"), -float("inf"), float("nan")], nans_filler)


def cast_f64_to_f32(df: pl.DataFrame) -> pl.Expr:
    return df.with_columns(pl.col(pl.Float64).cast(pl.Float32))


def drop_unrelated_features(
    df: pl.DataFrame,
    entropies: dict = None,
    binned_targets: pl.DataFrame = None,
    target_columns_prefix: str = "target_",
    clean_targets: bool = False,
    num_bins: int = 10,
    num_reps: int = 1,
    min_mi: float = 1e-5,
    exclude_columns: list = [],
    min_nuniques_to_clip: int = 50,
    tukey_fences_multiplier: float = 3.0,
    max_log_text_width: int = 300,
    verbose: int = 1,
) -> pl.DataFrame:
    """Drop features that have no direct relationship to at least one of the targets.
    Mutual Information (MI) is used to estimate presence of a relationship.

    Columns from exclude_columns are exempt of this check, so put here what you arelady have checked is relevant.

    Categorical features support rare categories merging.

    Numerical features:
        often can have outliers, so we provide an option of clipping outliers.

        First stats like min, max, quantiles are computed for each numerical feature.
        Features with no change (min==max) are dropped.
        For the rest, Tukey fences are computed, outliers are reported & clipped (windzorized).

        Next, each num column is binned into num_bins bins using borders=np.linspace(min_val, max_val, num_bins + 1),
            where min_val and max_val account for clipping.

        Marginal frequencies, and then entropies are computed for each column.

        for each of targets:
            Marginal frequencies, and then joint entropies & MIs are computed for each combination of explanatory column and target.
            Columns with zero MI are considered irrelevant to the target.
            for each of n_reps:
                columns bins are randomly permuted
                Marginal frequencies, and then joint entropies & shuffled MIs are computed for each combination of explanatory column and target.
            as long as shuffled MI for any column exceeds original MI, column is considered irrelevant to the target.

        Columns not relevant to any of the targets are reported & dropped.

    """

    import textwrap
    from pyutilz.system import tqdmu
    from collections import defaultdict

    columns_to_drop = []

    all_num_cols = cs.numeric()
    if exclude_columns:
        all_num_cols = all_num_cols - cs.by_name(exclude_columns)

    # ----------------------------------------------------------------------------------------------------------------------------
    # MinMax computed. Features with no change (min==max) are reported & dropped.
    # ----------------------------------------------------------------------------------------------------------------------------

    min_max_dict = df.select(all_num_cols.min().name.suffix("_min"), all_num_cols.max().name.suffix("_max")).row(0, named=True)
    orig_min_max_dict = min_max_dict.copy()

    dead_columns = []
    for col in cs.expand_selector(df, all_num_cols):
        min_val, max_val = min_max_dict.get(f"{col}_min"), min_max_dict.get(f"{col}_max")
        if (min_val is None and max_val is None) or np.allclose(min_val, max_val):
            dead_columns.append(col)
    if dead_columns:
        if verbose:
            logger.warning(
                f"Dropping {len(dead_columns):_} columns with no change: {textwrap.shorten(', '.join(dead_columns), width=max_log_text_width, placeholder='...')}"
            )
        df = df.drop(dead_columns)
        columns_to_drop.extend(dead_columns)

    # ----------------------------------------------------------------------------------------------------------------------------
    # Quantiles computed. Outliers are clipped & reported.
    # ----------------------------------------------------------------------------------------------------------------------------

    quantiles_dict = df.select(all_num_cols.quantile(0.25).name.suffix("_q1"), all_num_cols.quantile(0.75).name.suffix("_q3")).row(0, named=True)

    dead_columns = []
    public_clips = {}
    clips = {}
    for col in cs.expand_selector(df, all_num_cols):
        if not clean_targets:
            if col.startswith(target_columns_prefix):
                continue
        q1, q3 = quantiles_dict.get(f"{col}_q1"), quantiles_dict.get(f"{col}_q3")
        min_val, max_val = min_max_dict.get(f"{col}_min"), min_max_dict.get(f"{col}_max")

        iqr = q3 - q1

        lower_fence = q1 - tukey_fences_multiplier * iqr
        upper_fence = q3 + tukey_fences_multiplier * iqr

        # to avoid InvalidOperationError: conversion from `f64` to `u32` failed in column 'literal' for 1 out of 1 values: [-18.0000000] later
        # lower_fence = type(min_val)(lower_fence)
        # upper_fence = type(min_val)(upper_fence)

        if upper_fence > lower_fence:
            is_outlier = False
            if max_val > upper_fence:
                min_max_dict[f"{col}_max"] = upper_fence
                is_outlier = True
            if min_val < lower_fence:
                min_max_dict[f"{col}_min"] = lower_fence
                is_outlier = True
            if is_outlier:
                public_clips[col] = dict(lower_bound=lower_fence, upper_bound=upper_fence)
                clips[col] = pl.col(col).clip(lower_bound=lower_fence, upper_bound=upper_fence)

    if clips:
        # do not apply clipping if # of unique values is too low (under 10)
        n_uniques_dict = df.select(pl.col(clips.keys()).n_unique()).row(0, named=True)
        skipped_clips = []
        for col, nuniques in n_uniques_dict.items():
            if nuniques < min_nuniques_to_clip:
                for field in "min max".split():
                    min_max_dict[f"{col}_{field}"] = orig_min_max_dict[f"{col}_{field}"]
                skipped_clips.append(col)
                del public_clips[col]
                del clips[col]
        if verbose:
            if clips:
                logger.warning(
                    f"Clipping {len(clips):_} columns with outliers: {textwrap.shorten(', '.join(clips.keys()), width=max_log_text_width, placeholder='...')}"
                )
            if skipped_clips:
                logger.warning(
                    f"Clipping of {len(skipped_clips):_} columns skipped due to nuniques<{min_nuniques_to_clip:_}: {textwrap.shorten(', '.join(skipped_clips), width=max_log_text_width, placeholder='...')}"
                )
        if clips:
            df = df.with_columns(clips.values())

    # ----------------------------------------------------------------------------------------------------------------------------
    # Binning performed.
    # ----------------------------------------------------------------------------------------------------------------------------

    bin_labels = list(map(str, np.arange(num_bins + 2, dtype=np.int32)))  # Create labels for the bins
    bin_expressions = []

    for col in cs.expand_selector(df, all_num_cols):
        if binned_targets is not None:
            if col.startswith(target_columns_prefix):
                continue
        # Calculate bin edges based on min and max values
        min_val = min_max_dict.get(f"{col}_min")
        max_val = min_max_dict.get(f"{col}_max")

        if min_val == max_val:
            dead_columns.append(col)
        else:
            bin_edges = np.linspace(min_val, max_val, num_bins + 1)

            if len(set(bin_edges)) != len(bin_edges):
                raise ValueError

            # Define the binning expression
            binned_col = pl.col(col).cut(breaks=bin_edges, labels=bin_labels)
            bin_expressions.append(binned_col)

    if dead_columns:
        if verbose:
            logger.warning(
                f"Dropping {len(dead_columns):_} columns with no change: {textwrap.shorten(', '.join(dead_columns), width=max_log_text_width, placeholder='...')}"
            )
        df = df.drop(dead_columns)
        columns_to_drop.extend(dead_columns)

    # Apply all binning expressions in parallel
    bins = df.select(bin_expressions)  # .select(pl.all().value_counts())

    if binned_targets is not None:
        bins = pl.concat([bins, binned_targets], how="horizontal")
    else:
        binned_targets = bins.select(cs.starts_with(target_columns_prefix)).clone()

    # ----------------------------------------------------------------------------------------------------------------------------
    # Compute original marginal freqs for each column:
    # ----------------------------------------------------------------------------------------------------------------------------

    if entropies is None:
        entropies = {}
    for col in tqdmu(cs.expand_selector(bins, cs.all()), desc="computing entropies", leave=False):
        marginal_freqs = bins.group_by(col).agg(pl.len())["len"].to_numpy() / len(bins)
        entropies[col] = -np.sum(marginal_freqs * np.log(marginal_freqs))

    # ----------------------------------------------------------------------------------------------------------------------------
    # For each of the targets, compute joint freqs and then MI for each of the "normal" columns:
    # ----------------------------------------------------------------------------------------------------------------------------

    mutual_informations = {}
    target_cols = cs.expand_selector(bins, cs.starts_with(target_columns_prefix))

    for target_id, target_col in enumerate(tqdmu(target_cols, desc="checking MIs with targets", leave=False)):
        for i in tqdmu(range(num_reps), desc="bootstrap", leave=False):
            if i > 0:
                bins = bins.with_columns(pl.col(target_col).shuffle())
            for col in tqdmu(cs.expand_selector(bins, cs.all()), desc="var", leave=False):
                if col.startswith(target_columns_prefix):
                    continue

                if i == 0 or mutual_informations[(col, target_col)] > min_mi:  # only compute when it makes sense
                    joint_freqs = bins.group_by([col, target_col]).agg(pl.len())["len"].to_numpy() / len(bins)
                    joint_entropy = -np.sum(joint_freqs * np.log(joint_freqs))
                    mi = entropies[target_col] + entropies[col] - joint_entropy

                    if i == 0:
                        # save original MI of col vs target
                        mutual_informations[(col, target_col)] = mi
                    else:
                        # if permuted MI is same good or better than original - zero out original. Feature to be dropped.
                        if mi >= mutual_informations[(col, target_col)]:
                            mutual_informations[(col, target_col)] = 0
                            if verbose > 1:
                                logger.warning(f"After permutation, MI of var={col} with target {target_col} remained high: {mi}")

    # ----------------------------------------------------------------------------------------------------------------------------
    # Sum up MIs per column (over targets), decide what features have no influence, report & drop them.
    # ----------------------------------------------------------------------------------------------------------------------------

    cols_total_mis = defaultdict(float)

    for (col, target_col), value in mutual_informations.items():
        cols_total_mis[col] += value

    dead_columns = []
    for col, total_mi in cols_total_mis.items():
        if total_mi / len(target_cols) < min_mi:
            dead_columns.append(col)

    if dead_columns:
        if verbose:
            logger.warning(
                f"Dropping {len(dead_columns):_} columns with no direct impact on any target: {textwrap.shorten(', '.join(dead_columns), width=max_log_text_width, placeholder='...')}"
            )
        df = df.drop(dead_columns)
        columns_to_drop.extend(dead_columns)

    return df, binned_targets, public_clips, columns_to_drop, entropies, mutual_informations

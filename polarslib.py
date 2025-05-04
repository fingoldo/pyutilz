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


def drop_unrelated_features(df: pl.DataFrame, targets: list, num_bins: int = 10, exclude_columns: list = []) -> pl.DataFrame:
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

    min_max_dict = df.select(cs.numeric().min().name.suffix("_min"), cs.numeric().max().name.suffix("_max")).row(0, named=True)

    bin_expressions = []

    bin_labels = list(map(str, np.arange(num_bins + 2, dtype=np.int32)))

    for col in cs.expand_selector(df, cs.numeric()):
        # Calculate bin edges based on min and max values
        min_val = min_max_dict.get(f"{col}_min")
        max_val = min_max_dict.get(f"{col}_max")
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)

        # Create labels for the bins

        # Define the binning expression
        binned_col = pl.col(col).cut(breaks=bin_edges, labels=bin_labels)
        bin_expressions.append(binned_col)

    # Apply all binning expressions in parallel
    return df.select(["ticker"] + bin_expressions)  # .select(pl.all().value_counts())

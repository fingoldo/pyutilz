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
from pyutilz.system import tqdmu
from collections import defaultdict
from pyutilz.system import clean_ram
from concurrent.futures import ThreadPoolExecutor

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
# Polars
# ----------------------------------------------------------------------------------------------------------------------------


def find_unrelated_features(
    df: pl.DataFrame,
    target_columns_prefix: str = "target_",
    binned_targets: pl.DataFrame = None,
    clean_targets: bool = False,
    num_bins: int = 10,
    exclude_columns: list = [],
    min_nuniques_to_clip: int = 50,
    tukey_fences_multiplier: float = 3.0,
    max_log_text_width: int = 300,
    verbose: int = 1,
) -> pl.DataFrame:
    """DropFinds features that have no direct relationship to at least one of the targets.
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
    return


def bin_numerical_columns(
    df: pl.DataFrame,
    target_columns_prefix: str = "target_",
    binned_targets: pl.DataFrame = None,
    clean_features: bool = False,
    clean_targets: bool = False,
    num_bins: int = 10,
    bin_dtype: object = pl.Int8,
    exclude_columns: list = [],
    min_nuniques_to_clip: int = 20,
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
        all_num_cols = all_num_cols - cs.starts_with(target_columns_prefix)

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
            quantile_cols = cs.starts_with(target_columns_prefix)
        stats_expr.extend(
            [
                quantile_cols.quantile(0.25).name.suffix("_q1"),
                quantile_cols.quantile(0.75).name.suffix("_q3"),
            ]
        )
    stats = df.select(stats_expr).collect().row(0, named=True)
    orig_stats = stats.copy()

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

    # ----------------------------------------------------------------------------------------------------------------------------
    # Outliers are clipped & reported.
    # ----------------------------------------------------------------------------------------------------------------------------

    public_clips = {}
    clips = {}
    if clean_features or clean_targets:
        for col in cs.expand_selector(df, all_num_cols):
            if not clean_targets:
                if col.startswith(target_columns_prefix):
                    continue
            if not clean_features:
                if clean_targets and not col.startswith(target_columns_prefix):
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
            if col.startswith(target_columns_prefix):
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

    if binned_targets is not None:
        bins = pl.concat([bins, binned_targets], how="horizontal", rechunk=True)
    else:
        binned_targets = bins.select(cs.starts_with(target_columns_prefix)).clone()

    return bins, binned_targets, public_clips, columns_to_drop, stats


def compute_columns_mi_polars(
    bins: pl.DataFrame,
    target_columns_prefix: str = "target_",
    entropies: dict = None,
    noise_mutual_informations: dict = None,
    num_reps: int = 1,
    min_mi_prevalence: float = 10,
    entropy_computing_workers: int = None,
    mi_computing_workers: int = None,
    leave_progressbar: bool = False,
    max_log_text_width: int = 300,
    verbose: int = 1,
):
    columns_to_drop = []

    if entropies is None:
        entropies = {}

    target_cols = cs.expand_selector(bins, cs.starts_with(target_columns_prefix))

    # ----------------------------------------------------------------------------------------------------------------------------
    # Compute MIs of random noise, per target, for reference.
    # ----------------------------------------------------------------------------------------------------------------------------

    if noise_mutual_informations is None:

        if verbose > 1:
            logger.info("Computing MIs of random noise...")

        random_col_name = "random_col"
        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
        noise_bins = bins.select(cs.starts_with(target_columns_prefix))
        noise_bins = noise_bins.insert_column(
            0, pl.Series(name=random_col_name, values=np.random.rand(len(noise_bins))).cut(breaks=bin_edges, labels=bin_labels)
        )

        noise_mutual_informations = {}
        noise_entropies = {}
        noise_entropies[random_col_name] = entropy_for_column(noise_bins, random_col_name)
        for target_col in tqdmu(target_cols, desc="noise MIs with targets", leave=leave_progressbar):
            noise_entropies[target_col] = entropies[target_col]
            noise_mutual_informations[target_col] = mi_for_column(noise_bins, noise_entropies, random_col_name, target_col)

    # ----------------------------------------------------------------------------------------------------------------------------
    # For each of the targets, compute joint freqs and then MI for each of the "normal" columns:
    # ----------------------------------------------------------------------------------------------------------------------------

    if verbose > 1:
        logger.info("Computing MIs of candidate features...")

    mutual_informations = {}

    for target_id, target_col in enumerate(tqdmu(target_cols, desc="features MIs with targets", leave=leave_progressbar)):
        for i in tqdmu(range(num_reps), desc="bootstrap", leave=leave_progressbar):

            # Shuffle target, if it's time to
            if i > 0:
                bins = bins.with_columns(pl.col(target_col).shuffle())

            mi_results = {}
            cols_to_compute_mis = []

            # Who needs computing, actually?

            for col in tqdmu(cs.expand_selector(bins, cs.all()), desc="var", leave=leave_progressbar):
                if col.startswith(target_columns_prefix):
                    continue
                if (
                    i == 0 or mutual_informations[(col, target_col)] > noise_mutual_informations[target_col] * min_mi_prevalence
                ):  # only compute when it makes sense
                    cols_to_compute_mis.append(col)

            # Computing part
            w = partial(tqdmu, desc="computing MIs", leave=leave_progressbar)
            tasks = cols_to_compute_mis
            if mi_computing_workers <= 1:
                for col in w(tasks):
                    # print(col)
                    mi_results[col] = mi_for_column(bins, entropies, col, target_col)
            else:

                # Use ThreadPoolExecutor to run tasks in parallel
                with ThreadPoolExecutor(max_workers=mi_computing_workers) as executor:
                    # Submit tasks for each column
                    futures = {executor.submit(mi_for_column, bins, entropies, col, target_col): col for col in tasks}

                    # Collect results as they complete
                    for future in w(futures):
                        col = futures[future]
                        mi_results[col] = future.result()

            # Decision making part

            for col, mi in mi_results.items():
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

    cols_total_mis = defaultdict(int)

    for (col, target_col), mi in mutual_informations.items():
        if mi > noise_mutual_informations[target_col] * min_mi_prevalence:
            cols_total_mis[col] += 1
        else:
            cols_total_mis[col] += 0

    dead_columns = []
    for col, total_mi in cols_total_mis.items():
        if total_mi == 0:  # not related to any target
            dead_columns.append(col)

    if dead_columns:
        if verbose:
            logger.warning(
                f"Dropping {len(dead_columns):_} columns with no direct impact on any target: {textwrap.shorten(', '.join(dead_columns), width=max_log_text_width)}"
            )
        df = df.drop(dead_columns)
        columns_to_drop.extend(dead_columns)

    if verbose > 1:
        logger.info(f"Done. {len(columns_to_drop):_} columns_to_drop: {textwrap.shorten(', '.join(columns_to_drop), width=max_log_text_width)}.")

    del bins
    clean_ram()

    return columns_to_drop, entropies, noise_mutual_informations, mutual_informations


def run_efs(df, exclude_columns, entropies, noise_mutual_informations, binned_targets, efs_params) -> tuple:
    binned_targets, public_clips, columns_to_drop, entropies, noise_mutual_informations, mutual_informations = find_unrelated_features(
        df, entropies=entropies, noise_mutual_informations=noise_mutual_informations, binned_targets=binned_targets, **efs_params
    )

    df = df.drop(columns_to_drop)
    exclude_columns.update(set(df.columns))
    features_mis = pd.Series(mutual_informations).sort_values(ascending=False)

    return df, exclude_columns, entropies, noise_mutual_informations, binned_targets, features_mis


# ----------------------------------------------------------------------------------------------------------------------------
# PySR
# ----------------------------------------------------------------------------------------------------------------------------


def ru_pysr_fe(df: pl.DataFrame, nsamples: int = 100_000, target_columns_prefix: str = "target_", timeout_mins: int = 5, fill_nans: bool = True):

    from pysr import PySRRegressor

    clean_ram()

    model = PySRRegressor(
        turbo=True,
        timeout_in_seconds=timeout_mins * 60,
        maxsize=10,
        niterations=10,  # < Increase me for better results
        binary_operators=["+", "*"],
        unary_operators=[
            "cos",
            "exp",
            "log",
            "sin",
            "inv(x) = 1/x",
            # ^ Custom operator (julia syntax)
        ],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        elementwise_loss="loss(prediction, target) = abs(prediction - target)",
        # ^ Custom loss function (julia syntax)
    )

    # Build a mapping from old → new names
    rename_map = {col: col.replace("=", "_").replace(".", "_") for col in df.columns}

    tmp_df = df.sample(nsamples) if nsamples else df
    expr = cs.numeric() - cs.starts_with(target_columns_prefix)
    if fill_nans:
        expr = expr.fill_null(0).fill_nan(0)

    model.fit(tmp_df.select(expr).rename(rename_map), tmp_df.select(cs.starts_with(target_columns_prefix)))

    del tmp_df
    clean_ram()

    return model


# ----------------------------------------------------------------------------------------------------------------------------
# Numba
# ----------------------------------------------------------------------------------------------------------------------------

import numpy as np
from numba import njit, prange

USE_FASTMATH: bool = True

# ----------------------------------------------------------------------------------------------------------------------------
# GROK
# ----------------------------------------------------------------------------------------------------------------------------


@njit(fastmath=USE_FASTMATH)
def grok_compute_joint_hist(a: np.ndarray, b: np.ndarray, n_bins: int, dtype: object = np.int64):
    hist = np.zeros((n_bins, n_bins), dtype=dtype)
    for i in range(len(a)):
        hist[a[i], b[i]] += 1
    return hist


@njit(fastmath=USE_FASTMATH)
def grok_mutual_information_old(a: np.ndarray, b: np.ndarray, n_bins: int = 15, hist_dtype: object = np.int64):
    joint_hist = grok_compute_joint_hist(a=a, b=b, n_bins=n_bins, dtype=hist_dtype)
    a_hist = np.sum(joint_hist, axis=1)
    b_hist = np.sum(joint_hist, axis=0)
    n_samples = len(a)
    mi = 0.0
    for x in range(n_bins):
        for y in range(n_bins):
            if joint_hist[x, y] > 0:
                p_joint = joint_hist[x, y] / n_samples
                p_a = a_hist[x] / n_samples
                p_b = b_hist[y] / n_samples
                mi += p_joint * np.log(p_joint / (p_a * p_b))
    return mi


@njit(fastmath=USE_FASTMATH)
def grok_mutual_information(a: np.ndarray, b: np.ndarray, inv_n_samples: float, log_n_samples: float, n_bins: int = 15, hist_dtype: object = np.int64):
    joint_hist = grok_compute_joint_hist(a=a, b=b, n_bins=n_bins, dtype=hist_dtype)
    a_hist = np.sum(joint_hist, axis=1)
    b_hist = np.sum(joint_hist, axis=0)
    mi = 0.0
    for x in range(n_bins):
        for y in range(n_bins):
            if joint_hist[x, y] > 0:
                joint_count = joint_hist[x, y]
                p_joint = joint_count * inv_n_samples
                log_term = np.log(joint_count) - np.log(a_hist[x]) - np.log(b_hist[y]) + log_n_samples
                mi += p_joint * log_term
    return mi


@njit(parallel=True)
def grok_compute_mutual_information(
    data: np.ndarray, target_indices: np.ndarray | list[int], n_bins: int = 15, hist_dtype=np.int64, out_dtype=np.float64
) -> np.ndarray:
    """
    MI of every specified target column against every column in `data`.

    Parameters
    ----------
    data            : int8 ndarray, shape (n_samples, n_cols), already binned 0-14
    target_indices  : iterable of int column indices
    n_bins          : number of discrete bins (default 15)

    Returns
    -------
    mi_matrix       : float64 ndarray, shape (n_targets, n_cols)
                      Row k = MI(target_indices[k], all columns)
    """

    n_samples, n_columns = data.shape
    K = len(target_indices)
    mi_results = np.zeros((K, n_columns), dtype=out_dtype)

    inv_n_samples = 1.0 / n_samples
    log_n_samples = np.log(n_samples)

    for t in range(K):
        target = target_indices[t]
        target_col = data[:, target]
        for j in prange(n_columns):
            if j != target:
                mi_results[t, j] = grok_mutual_information(
                    target_col, data[:, j], n_bins=n_bins, inv_n_samples=inv_n_samples, log_n_samples=log_n_samples, hist_dtype=hist_dtype
                )
            else:
                mi_results[t, j] = np.nan
    return mi_results


# ----------------------------------------------------------------------------------------------------------------------------
# ChatGPT
# ----------------------------------------------------------------------------------------------------------------------------


# Single-pair MI (15 discrete bins, natural-log base)
@njit(fastmath=USE_FASTMATH)
def _chatgpt_mi_pair(x: np.ndarray, y: np.ndarray, n_bins: int = 15, hist_dtype=np.int64) -> float:
    """Mutual information between two 1-D int8 vectors already binned to 0..n_bins-1."""

    # 1) joint counts

    joint = np.zeros((n_bins, n_bins), dtype=hist_dtype)
    for k in range(x.size):
        joint[x[k], y[k]] += 1

    # 2) marginals

    row = np.zeros(n_bins, dtype=hist_dtype)  # P(x)
    col = np.zeros(n_bins, dtype=hist_dtype)  # P(y)
    for i in range(n_bins):
        for j in range(n_bins):
            c = joint[i, j]
            row[i] += c
            col[j] += c

    # 3) MI

    N = x.size
    mi = 0.0
    for i in range(n_bins):
        if row[i] == 0:
            continue
        p_i = row[i] / N
        for j in range(n_bins):
            c = joint[i, j]
            if c == 0 or col[j] == 0:
                continue
            p_ij = c / N
            p_j = col[j] / N
            mi += p_ij * np.log(p_ij / (p_i * p_j))
    return mi


# All features vs. one target (parallel over the wide axis)
@njit(parallel=True, fastmath=USE_FASTMATH)
def _chatgpt_mi_one_target(
    data: np.ndarray, target_idx: int, n_bins: int = 15, hist_dtype=np.int64, out_dtype=np.float64
) -> np.ndarray:  # shape (n_samples, n_cols), int8
    """Vector of MI(target, every feature)."""
    n_rows, n_cols = data.shape
    y = data[:, target_idx]
    out = np.empty(n_cols, dtype=out_dtype)

    # Parallel loop across *features* – this is the expensive axis.
    for c in prange(n_cols):
        out[c] = _chatgpt_mi_pair(data[:, c], y, n_bins, hist_dtype=hist_dtype)

    return out


# Public API: many targets vs. all features
def chatgpt_compute_mutual_information(
    data: np.ndarray, target_indices: np.ndarray | list[int], n_bins: int = 15, hist_dtype=np.int64, out_dtype=np.float64
) -> np.ndarray:
    """
    MI of every specified target column against every column in `data`.

    Parameters
    ----------
    data            : int8 ndarray, shape (n_samples, n_cols), already binned 0-14
    target_indices  : iterable of int column indices
    n_bins          : number of discrete bins (default 15)

    Returns
    -------
    mi_matrix       : float64 ndarray, shape (n_targets, n_cols)
                      Row k = MI(target_indices[k], all columns)
    """
    # Safety – make sure the array is C-contiguous int8 for maximum speed.
    if data.dtype != np.int8 or not data.flags.c_contiguous:
        data = np.ascontiguousarray(data, dtype=np.int8)

    targets = np.asarray(target_indices, dtype=np.int64)
    out = np.empty((targets.size, data.shape[1]), dtype=out_dtype)

    # Few targets, many features ⇒ parallel inside _mi_one_target
    for k, t in enumerate(targets):
        out[k, :] = _chatgpt_mi_one_target(data=data, target_idx=int(t), n_bins=n_bins, hist_dtype=hist_dtype, out_dtype=out_dtype)

    return out


# ----------------------------------------------------------------------------------------------------------------------------
# DeepSeek
# ----------------------------------------------------------------------------------------------------------------------------


@njit(parallel=True, fastmath=USE_FASTMATH)
def deepseek_compute_mutual_information(
    data: np.ndarray, target_indices: np.ndarray | list[int], n_bins: int = 15, hist_dtype=np.int64, out_dtype=np.float64
) -> np.ndarray:
    """
    MI of every specified target column against every column in `data`.

    Parameters
    ----------
    data            : int8 ndarray, shape (n_samples, n_cols), already binned 0-14
    target_indices  : iterable of int column indices
    n_bins          : number of discrete bins (default 15)

    Returns
    -------
    mi_matrix       : float64 ndarray, shape (n_targets, n_cols)
                      Row k = MI(target_indices[k], all columns)
    """

    n_samples, n_columns = data.shape
    n_targets = len(target_indices)

    # Precompute marginals and sum_N_log_N for each column
    marginals = np.zeros((n_columns, n_bins), dtype=hist_dtype)
    sum_N_log_N = np.zeros(n_columns, dtype=out_dtype)

    for col in prange(n_columns):
        counts = np.zeros(n_bins, dtype=hist_dtype)
        for i in range(n_samples):
            val = data[i, col]
            counts[val] += 1
        marginals[col] = counts
        s = 0.0
        for b in range(n_bins):
            c = counts[b]
            if c > 0:
                s += c * np.log(c)
        sum_N_log_N[col] = s

    N = n_samples
    N_log_N = N * np.log(N) if N > 0 else 0.0

    mi_results = np.zeros((n_targets, n_columns), dtype=out_dtype)
    n_total_pairs = n_targets * n_columns

    for pair_idx in prange(n_total_pairs):
        t_idx = pair_idx // n_columns
        feature_col = pair_idx % n_columns
        target_col = target_indices[t_idx]

        joint = np.zeros((n_bins, n_bins), dtype=hist_dtype)
        for i in range(n_samples):
            y_val = data[i, target_col]
            x_val = data[i, feature_col]
            joint[y_val, x_val] += 1

        sum_Nxy_log_Nxy = 0.0
        for y_bin in range(n_bins):
            for x_bin in range(n_bins):
                n = joint[y_bin, x_bin]
                if n > 0:
                    sum_Nxy_log_Nxy += n * np.log(n)

        sum_Ny_log_Ny = sum_N_log_N[target_col]
        sum_Nx_log_Nx = sum_N_log_N[feature_col]

        mi = (sum_Nxy_log_Nxy - sum_Ny_log_Ny - sum_Nx_log_Nx + N_log_N) / N
        mi_results[t_idx, feature_col] = mi

    return mi_results

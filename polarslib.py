"""Common functions for working with Polars"""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

# Set jemalloc config early
import os

os.environ["_RJEM_MALLOC_CONF"] = "muzzy_decay_ms:0"  # prevents memory leak in polars
import polars as pl, polars.selectors as cs

from typing import *
import numpy as np, pandas as pd

from functools import partial

import psutil
import textwrap
from collections import defaultdict, Counter
from pyutilz.system import clean_ram
from mlframe.utils import is_cuda_available, check_cpu_flag

# ----------------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------------

POLARS_DEFAULT_NUMAGGS: list = (
    "first last min max mean std arg_max arg_min skew kurtosis entropy n_unique".split()
)  # replace by approx_n_unique? # median excluded
POLARS_DEFAULT_QUANTILES: list = [0.1, 0.25, 0.5, 0.75, 0.9]

# ----------------------------------------------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------------------------------------------


def find_nan_cols(df: pl.DataFrame) -> pl.DataFrame:
    meta = df.select(cs.numeric().is_nan().any())
    true_cols = meta.row(0)
    df.select([col for col, val in zip(meta.columns, true_cols) if val is True])


def find_infinite_cols(df: pl.DataFrame) -> pl.DataFrame:
    meta = df.select(cs.numeric().is_infinite().any())
    true_cols = meta.row(0)
    df.select([col for col, val in zip(meta.columns, true_cols) if val is True])


def clean_numeric(expr: pl.Expr, nans_filler: float = 0.0) -> pl.Expr:
    return expr.replace([float("inf"), -float("inf"), float("nan")], nans_filler)
    # return pl.when(expr.is_infinite()).then(expr).otherwise(pl.lit(nans_filler))


def cast_f64_to_f32(df: pl.DataFrame) -> pl.Expr:
    return df.with_columns(pl.col(pl.Int32, pl.Uint32, pl.Int64, pl.UInt64, pl.Int128, pl.Float64).cast(pl.Float32))


def apply_agg_func_safe(expr: pl.Expr, func_name: str, nans_filler: float = 0.0) -> pl.Expr:
    if func_name in ["skew", "kurtosis"]:
        return clean_numeric(expr, nans_filler=nans_filler)
    else:
        return expr


def compute_concentrations(
    groupby_object: object,
    entity: str,
    entity_name: str,
    by: str,
    groupby_columns: tuple,
    top_n: int = 2,
    return_ids: bool = True,
    return_values: bool = True,
    sort_by_concentration: bool = True,
    add_mean_concentration: bool = True,
    dtype: object = pl.Float64,
    fields_remap: dict = None,
) -> pl.DataFrame:
    """Computes within a group_by (dynamic or rolling), for example, concentrations of customers by total volume of their sales.
    groupby_columns must include both group_byand index_column arguments pased to group_by_dynamic.

    Ensuring sortedness after top_k_by is messy (requires zipping lists etc), so using sort+head instead."""

    assert return_ids or return_values

    if not fields_remap:
        fields_remap = {}

    label = f"{entity_name}-by-{fields_remap.get(by,by)}"

    columns_to_unnest = []
    unnest_rules = []

    exprs = []
    if return_ids:
        exprs.append(pl.col(entity_name).alias(f"{label}"))
        columns_to_unnest.extend(
            [
                pl.col(f"{label}").list.to_struct(
                    n_field_strategy="max_width",
                    upper_bound=top_n,
                    fields=[f"{label}_top{i+1}" for i in range(top_n)],
                ),  # Convert list to struct
            ]
        )
        unnest_rules.append(f"{label}")
    if return_values:
        exprs.append(pl.col("rel_total_by").alias(f"{label}_r{fields_remap.get(by,by)}"))

        if add_mean_concentration:
            columns_to_unnest.append(pl.col(f"{label}_r{fields_remap.get(by,by)}").list.mean().cast(dtype).alias(f"{label}_top{top_n}_avg_conc"))

        columns_to_unnest.append(
            pl.col(f"{label}_r{fields_remap.get(by,by)}").list.to_struct(
                n_field_strategy="max_width",
                upper_bound=top_n,
                fields=[f"{label}_top{i+1}_conc" for i in range(top_n)],
            )
        )

        unnest_rules.append(f"{label}_r{fields_remap.get(by,by)}")

    df = (
        groupby_object.agg(entity, by)
        .explode(entity_name, by)
        .group_by(*groupby_columns, entity_name)
        .agg(total_by=pl.sum(by))
        .with_columns(rel_total_by=pl.col("total_by") / pl.sum("total_by").over(groupby_columns))
        # .group_by(groupby_columns).agg(pl.col(entity,rel_total_by).top_k_by(rel_total_by,top_n))
        .sort(*groupby_columns, "rel_total_by" if sort_by_concentration else entity_name, descending=([False] * len(groupby_columns) + [sort_by_concentration]))
        .group_by(groupby_columns, maintain_order=True)
        .head(top_n)
        .group_by(groupby_columns, maintain_order=True)
        .agg(exprs)
    )

    if columns_to_unnest:
        df = df.with_columns(columns_to_unnest).unnest(unnest_rules)

    return df


# ----------------------------------------------------------------------------------------------------------------------------
# FE in polars
# ----------------------------------------------------------------------------------------------------------------------------


def add_weighted_aggregates(columns_selector: object, weighting_columns: Iterable, fpref: str = "", fields_remap: dict = None) -> list:
    """Computes weighted aggregates."""
    if not fields_remap:
        fields_remap = {}
    wcols = []
    if weighting_columns:
        for wcol in weighting_columns:
            all_other_num_cols = columns_selector - cs.by_name(wcol)
            weighted_mean = ((all_other_num_cols * pl.col(wcol)).sum() / pl.col(wcol).sum()).name.suffix(f"_{fpref}wmeanby_{fields_remap.get(wcol,wcol)}")
            wcols.append(weighted_mean)
            # !TODO causes error for now
            # weighted_std = ((pl.col(wcol) * (all_other_num_cols - weighted_mean) ** 2).sum() / pl.col(wcol).sum()).sqrt().name.suffix(f"_wstdby_{wcol}")
            # wcols.append(weighted_std)
    return wcols


def build_aggregate_features_polars(
    df: pl.DataFrame,
    #
    boolean_fields: list = None,
    numerical_fields: list = None,
    categorical_fields: list = None,
    ts_diff_fields: list = None,
    exclude_fields: list = None,
    weighting_fields: list = None,
    #
    subgroups: dict = None,
    numaggs: list = None,
    quantiles: list = None,
    tds_quantiles: list = None,
    #
    engine: str = "cpu",
    dtype: object = pl.Float64,
    fields_remap: dict = None,
    nans_filler: float = 0.0,
    concentration_top_n: int = 3,
    concentrations_params: dict = None,
    add_peaks_stats: bool = True,
    custom_expressions: list = None,
    #
    othersvals_at_extremums: bool = False,
    othersvals_basic_fields: list = None,
    othersvals_other_fields: list = None,
    othersvals_excluded_fields: list = None,
    #
    ewm_spans: list = None,
    ewm_timestamp: str = None,
    ewm_time_half_lifes: list = None,
    ewm_basic_funcs: list = None,
    ewm_final_funcs: list = None,
    # polars-ds features (work on AVX2+ only!)
    pds_params: dict = None,
    pds_numaggs: list = None,
    corr_methods: list = None,
    corr_fields: dict = None,
    pds_fields: list = None,
    linreg_fields: list = None,
    linreg_timestamp_field: str = None,
    use_parametrized_pds_features: bool = True,
) -> tuple:

    # ----------------------------------------------------------------------------------------------------------------------------
    # Checks
    # ----------------------------------------------------------------------------------------------------------------------------

    assert engine in ("cpu", "gpu")

    if engine == "gpu" and not is_cuda_available():
        # logger.warning(f"GPU FE path chosen, but Cuda seems to be unavailble on this system!")
        pass

    # ----------------------------------------------------------------------------------------------------------------------------
    # Inits
    # ----------------------------------------------------------------------------------------------------------------------------

    if check_cpu_flag("avx2"):
        import polars_ds as pds

    # Params

    if fields_remap is None:
        fields_remap = {}

    if not concentrations_params:
        concentrations_params = {}

    if numaggs is None:
        numaggs = POLARS_DEFAULT_NUMAGGS

    if quantiles is None:
        quantiles = POLARS_DEFAULT_QUANTILES

    if tds_quantiles is None:
        tds_quantiles = POLARS_DEFAULT_QUANTILES

    if not ewm_spans:
        ewm_spans: list = []
    if not ewm_time_half_lifes:
        ewm_time_half_lifes: list = []
    if not ewm_basic_funcs:
        ewm_basic_funcs = "ewm_mean ewm_std".split()
    if not ewm_final_funcs:
        ewm_final_funcs = "mean".split()

    if pds_numaggs is None:
        pds_numaggs = "hmean query_abs_energy query_cid_ce query_mean_abs_change".split()  # query_permute_entropy # query_first_digit_cnt
    if corr_methods is None:
        corr_methods = ["pearson", "spearman", "xi", "kendall", "bicor"]

    if pds_params is None:
        pds_params = dict(lag=1, n_maxima=1, n_lags=1)

    # Fields

    if boolean_fields is None:
        boolean_fields = cs.expand_selector(df.head(), cs.by_dtype(pl.Boolean))
    if ts_diff_fields is None:
        ts_diff_fields = cs.expand_selector(df.head(), cs.by_dtype(pl.Datetime))
    if numerical_fields is None:
        numerical_fields = cs.expand_selector(df.head(), cs.numeric())
    if categorical_fields is None:
        categorical_fields = list(cs.expand_selector(df.head(), cs.by_dtype(pl.Categorical, pl.Utf8)))

    if exclude_fields:
        if boolean_fields:
            boolean_fields = set(boolean_fields) - set(exclude_fields)
        if ts_diff_fields:
            ts_diff_fields = set(ts_diff_fields) - set(exclude_fields)
        if numerical_fields:
            numerical_fields = set(numerical_fields) - set(exclude_fields)
        if categorical_fields:
            categorical_fields = list(set(categorical_fields) - set(exclude_fields))

    if pds_fields is None:
        pds_fields = numerical_fields
    if linreg_fields is None:
        linreg_fields = numerical_fields

    if not subgroups:
        subgroups = {"": [""]}  # {"action": ["buy", "sell"]}

    # Counters

    feature_expressions, columns_to_unnest, unnest_rules = [], [], []

    if custom_expressions:
        feature_expressions = [expr for expr in custom_expressions]

    ts_numaggs = [el for el in numaggs if el not in ("first",)]

    # ----------------------------------------------------------------------------------------------------------------------------
    # Actual building
    # ----------------------------------------------------------------------------------------------------------------------------

    num_no_filter: int = 0
    orig_categorical_fields = categorical_fields.copy()
    for filter_field, filter_values in subgroups.items():

        assert isinstance(filter_values, list)

        if not filter_field:
            num_no_filter += 1
            assert num_no_filter <= 1
        else:
            categorical_fields = orig_categorical_fields.copy()
            if filter_field in categorical_fields:
                categorical_fields.remove(filter_field)

        for filter_value in filter_values:

            def af(expr) -> pl.expr:
                return expr if not filter_field else expr.filter(pl.col(filter_field) == filter_value)

            fpref = "" if not filter_field else f"{filter_field}_{filter_value}_"

            feature_expressions.append(af(cs.first()).len().alias(f"{fpref}nrecs"))

            if boolean_fields:
                # Means for boolean columns
                feature_expressions.extend(
                    [
                        getattr(af(pl.col(field)), func)().alias(f"{fpref}{fields_remap.get(field,field)}_{func}")
                        for field in boolean_fields
                        for func in ["mean"]
                    ]
                )
                # +lziv
                if check_cpu_flag("avx2"):
                    feature_expressions.extend(
                        [
                            pds.query_lempel_ziv(af(pl.col(field)), as_ratio=True).alias(f"{fpref}{fields_remap.get(field,field)}_lziv")
                            for field in boolean_fields
                        ]
                    )

            if add_peaks_stats:
                # Means
                feature_expressions.extend(
                    [
                        getattr(af(pl.col(field)), func)().mean().cast(dtype).alias(f"{fpref}{fields_remap.get(field,field)}_{func}_mean")
                        for field in numerical_fields
                        for func in ["peak_min", "peak_max"]
                    ]
                )
                # +lziv
                if check_cpu_flag("avx2"):
                    feature_expressions.extend(
                        [
                            pds.query_lempel_ziv(getattr(af(pl.col(field)), func)(), as_ratio=True).alias(f"{fpref}{fields_remap.get(field,field)}_{func}_lziv")
                            for field in numerical_fields
                            for func in ["peak_min", "peak_max"]
                        ]
                    )

            # Numaggs over numerical columns
            feature_expressions.extend(
                [getattr(af(pl.col(field)), func)().alias(f"{fpref}{fields_remap.get(field,field)}_{func}") for field in numerical_fields for func in numaggs]
            )

            # Quantiles
            feature_expressions.extend(
                [af(pl.col(field)).quantile(q).alias(f"{fpref}{fields_remap.get(field,field)}_quantile_{q}") for field in numerical_fields for q in quantiles]
            )

            # Weighting
            if weighting_fields:
                wcols = add_weighted_aggregates(
                    columns_selector=(cs.numeric() - cs.by_name(exclude_fields)), weighting_columns=weighting_fields, fpref=fpref, fields_remap=fields_remap
                )
                feature_expressions.extend(wcols)

            if othersvals_at_extremums:
                for col in numerical_fields:
                    if not othersvals_basic_fields or col in othersvals_basic_fields:
                        if othersvals_other_fields:
                            other_columns = cs.by_name(othersvals_other_fields) - cs.by_name(col)
                        else:
                            other_columns = cs.all() - cs.by_name(col)

                        if exclude_fields:
                            other_columns = other_columns - cs.by_name(exclude_fields)

                        if othersvals_excluded_fields:
                            other_columns = other_columns - cs.by_name(othersvals_excluded_fields)

                        if filter_field:
                            other_columns = other_columns - cs.by_name(filter_field)

                        feature_expressions.append(other_columns.get(pl.col(col).arg_max().alias("arg_max")).name.suffix(f"_{fpref}at_{col}_max"))
                        feature_expressions.append(other_columns.get(pl.col(col).arg_min().alias("arg_min")).name.suffix(f"_{fpref}at_{col}_min"))

            # Exponentially weighted mean/std
            feature_expressions.extend(
                [
                    getattr(getattr(af(pl.col(field)), func)(span=span), agg_func)().alias(
                        f"{fpref}{fields_remap.get(field,field)}_{func}_span={span}_{agg_func}"
                    )
                    for field in numerical_fields
                    for func in ewm_basic_funcs
                    for span in ewm_spans
                    for agg_func in ewm_final_funcs
                ]
            )

            if ewm_timestamp and ewm_time_half_lifes:  # careful: causes a bug sometimes (lengths mismatch)
                feature_expressions.extend(
                    [
                        getattr(af(pl.col(field)).ewm_mean_by(by=ewm_timestamp, half_life=half_life), agg_func)().alias(
                            f"{fpref}{fields_remap.get(field,field)}_ewm_ts_hl={half_life}_{agg_func}"
                        )
                        for field in numerical_fields
                        for half_life in ewm_time_half_lifes
                        for agg_func in ewm_final_funcs
                    ]
                )

            # Categorical stats. For gpu mode, categoricals need to be converted to String upfront.
            feature_expressions.extend([af(pl.col(field)).n_unique().alias(f"{fpref}{fields_remap.get(field,field)}_nunique") for field in categorical_fields])

            if engine != "gpu":
                # Modes require special treatment
                feature_expressions.extend(
                    [
                        getattr(af(pl.col(field)).mode(), func)().alias(f"{fpref}{fields_remap.get(field,field)}_mode_{func}")
                        for field in numerical_fields
                        for func in ["min", "max", "mean"]
                    ]
                )

                if concentration_top_n > 0:
                    for field in categorical_fields:
                        field_concentration_top_n = concentrations_params.get(field, concentration_top_n)
                        # Validate field_concentration_top_n
                        if not isinstance(field_concentration_top_n, int) or field_concentration_top_n <= 0:
                            print(f"Skipping field {field}: invalid top_n={field_concentration_top_n}")
                            continue
                        alias = f"{fpref}{fields_remap.get(field,field)}_top{field_concentration_top_n}"
                        feature_expressions.append(
                            af(pl.col(field)).value_counts(sort=True, normalize=True).head(field_concentration_top_n).struct.field("proportion").alias(alias)
                        )

                        columns_to_unnest.extend(
                            [
                                pl.col(alias).list.mean().cast(dtype).alias(f"{alias}_avg_conc"),
                                pl.col(alias).list.to_struct(
                                    n_field_strategy="max_width",
                                    upper_bound=field_concentration_top_n,
                                    fields=[f"{fpref}{fields_remap.get(field,field)}_top{i+1}_conc" for i in range(field_concentration_top_n)],
                                ),
                            ]
                        )
                        unnest_rules.append(alias)

                # Time diffs: numaggs
                feature_expressions.extend(
                    [
                        getattr(apply_agg_func_safe(af(pl.col(field)).diff().dt.total_seconds() / 60, func_name=func, nans_filler=nans_filler), func)()
                        .cast(dtype)
                        .alias(f"{fpref}{fields_remap.get(field,field)}_tsd_{func}")
                        for field in ts_diff_fields
                        for func in ts_numaggs
                    ]
                )
                if tds_quantiles:
                    # Time diffs: Quantiles
                    feature_expressions.extend(
                        [
                            (af(pl.col(field)).diff().dt.total_seconds() / 60).quantile(q).alias(f"{fpref}{fields_remap.get(field,field)}_tsd_quantile={q}")
                            for field in ts_diff_fields
                            for q in tds_quantiles
                        ]
                    )

                if check_cpu_flag("avx2"):

                    # simple stats with no params
                    for field in pds_fields:
                        feature_expressions.extend(
                            [
                                getattr(pds, func)(af(pl.col(field))).alias(f"{fpref}{fields_remap.get(field,field)}_{func.replace('query_','')}")
                                for func in pds_numaggs
                            ]
                        )
                        if "query_first_digit_cnt" in pds_numaggs:
                            NDIGITS = 9
                            alias = f"{fpref}{fields_remap.get(field,field)}_{'query_first_digit_cnt'.replace('query_','')}"
                            columns_to_unnest.append(
                                pl.col(alias).list.to_struct(
                                    n_field_strategy="max_width", upper_bound=NDIGITS, fields=[f"{alias}_{i+1}" for i in range(NDIGITS)]
                                )
                            )
                            unnest_rules.append(alias)

                    # stats with params
                    if use_parametrized_pds_features:
                        lag, n_maxima, n_lags = pds_params.get("lag", 1), pds_params.get("n_maxima", 1), pds_params.get("n_lags", 1)
                        for field in pds_fields:
                            feature_expressions.extend(
                                [
                                    pds.query_c3_stats(af(pl.col(field)), lag=lag).cast(dtype).alias(f"{fpref}{fields_remap.get(field,field)}_c3_stats-{lag}"),
                                    pds.query_mean_n_abs_max(af(pl.col(field)), n_maxima=n_maxima).alias(
                                        f"{fpref}{fields_remap.get(field,field)}_mean_n_abs_max-{n_maxima}"
                                    ),
                                    pds.query_time_reversal_asymmetry_stats(af(pl.col(field)), n_lags=n_lags).alias(
                                        f"{fpref}{fields_remap.get(field,field)}_tras-{n_lags}"
                                    ),
                                ]
                            )

                    # Corrs
                    if corr_fields:
                        for corr_x, corr_y in corr_fields.items():
                            for corr_method in corr_methods:
                                feature_expressions.append(
                                    pds.corr(corr_x, corr_y, method=corr_method).cast(dtype).alias(f"{fpref}corr_{corr_x}-{corr_y}-{corr_method}")
                                )

                    # Linregs
                    for field in linreg_fields:
                        alias = f"{fpref}{fields_remap.get(field,field)}_linreg"
                        feature_expressions.append(pds.simple_lin_reg(pl.int_range(pl.len()), target=pl.col(field), add_bias=True).alias(alias))
                        columns_to_unnest.extend(
                            [
                                pl.col(alias).list.to_struct(
                                    n_field_strategy="max_width",
                                    upper_bound=2,
                                    fields=[f"{alias}_k", f"{alias}_b"],
                                )
                            ]
                        )
                        unnest_rules.append(alias)
                        if linreg_timestamp_field:
                            alias = f"{fpref}{fields_remap.get(field,field)}_linregby_{linreg_timestamp_field}"
                            feature_expressions.append(
                                pds.simple_lin_reg(
                                    (pl.col(linreg_timestamp_field) - pl.col(linreg_timestamp_field).min()).dt.total_seconds(),
                                    target=pl.col(field),
                                    add_bias=True,
                                ).alias(alias)
                            )
                            columns_to_unnest.extend(
                                [
                                    pl.col(alias).list.to_struct(
                                        n_field_strategy="max_width",
                                        upper_bound=2,
                                        fields=[f"{alias}_k", f"{alias}_b"],
                                    )
                                ]
                            )
                            unnest_rules.append(alias)

    return feature_expressions, columns_to_unnest, unnest_rules


def create_ts_features_polars(
    df: pl.DataFrame,
    index_column: str,
    period: str,
    every: str = None,
    offset: str = None,
    closed: str = None,
    label: str = "left",
    group_by: str = None,
    rolling: bool = False,
    include_boundaries: bool = False,
    clean_memory: bool = True,
    dtype: object = pl.Float64,
    engine: str = "cpu",
    **kwargs,
) -> pl.DataFrame:
    """
    Recipe for integer window rolling:
        create_rolling_features(df.with_columns(row_idx=pl.col("tokenAddress").cum_count().over("tokenAddress")),period="24i",index_column="row_idx",...).drop("row_idx")
    """
    if not closed:
        closed = "left" if not rolling else "right"

    if clean_memory:
        clean_ram()

    if group_by:
        additional_exclude = [group_by] if isinstance(group_by, str) else group_by
        if kwargs.get("exclude_fields"):
            kwargs["exclude_fields"] = list(kwargs.get("exclude_fields")) + additional_exclude
        else:
            kwargs["exclude_fields"] = additional_exclude

    expressions, columns_to_unnest, unnest_rules = build_aggregate_features_polars(df, dtype=dtype, **kwargs)

    if rolling:
        res = df.rolling(index_column=index_column, period=period, offset=offset, closed=closed, group_by=group_by).agg(expressions)
    else:
        res = df.group_by_dynamic(
            index_column=index_column,
            every=every,
            period=period,
            offset=offset,
            closed=closed,
            label=label,
            group_by=group_by,
            include_boundaries=include_boundaries,
        ).agg(expressions)

    # ----------------------------------------------------------------------------------------------------------------------------
    # Unnest remaining arrays in one go
    # ----------------------------------------------------------------------------------------------------------------------------

    if columns_to_unnest:
        res = res.with_columns(columns_to_unnest).unnest(unnest_rules)

    fragile_cols = (
        cs.contains("_skew")
        | cs.contains("_kurtosis")
        | cs.contains("_entropy")
        | cs.contains("_c3_stats")
        | cs.contains("_cid_ce")
        | cs.contains("corr_")
        | cs.contains("_linreg")
        | cs.contains("_lempel_ziv")
    )
    res = res.with_columns(
        clean_numeric(
            fragile_cols.cast(dtype=dtype),
        )
    )

    if dtype == pl.Float32:
        res = cast_f64_to_f32(res)

    if clean_memory:
        clean_ram()

    logger.info(f"create_ts_features_polars using {engine} engine, {pl.thread_pool_size():_} threads...")
    res = res.collect(engine=engine)
    logger.info(f"Done.")

    return res


# ----------------------------------------------------------------------------------------------------------------------------
# FS in polars
# ----------------------------------------------------------------------------------------------------------------------------


def entropy_for_column(bins: pl.DataFrame, col: str) -> float:
    marginal_freqs = bins.group_by(col).agg(pl.len())["len"].to_numpy() / len(bins)
    return -np.sum(marginal_freqs * np.log(marginal_freqs))


def mi_for_column(bins: pl.DataFrame, entropies: dict, col: str, target_col: str) -> float:
    joint_freqs = bins.group_by([col, target_col]).agg(pl.len())["len"].to_numpy() / len(bins)
    joint_entropy = -np.sum(joint_freqs * np.log(joint_freqs))
    mi = entropies[target_col] + entropies[col] - joint_entropy
    return mi


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
    stats = df.select(stats_expr).collect()

    if len(stats) > 0:
        stats = stats.row(0, named=True)
    else:
        stats = {}
    orig_stats = stats.copy()

    clean_ram()

    # ----------------------------------------------------------------------------------------------------------------------------
    # Features with no change (min==max) are reported & dropped.
    # ----------------------------------------------------------------------------------------------------------------------------

    dead_columns = []
    for col in cs.expand_selector(df.head(), all_num_cols):
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
        for col in cs.expand_selector(df.head(), all_num_cols):
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
        cols_with_floats = cs.expand_selector(df.head(), all_num_cols & cs.float())

    for col in cs.expand_selector(df.head(), all_num_cols):
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

            binned_col = ((col_expr - min_val) / bin_width).floor().fill_nan(0).clip(0, num_bins - 1).cast(bin_dtype)

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
    for col in cs.expand_selector(df.head(), all_num_cols):
        min_val, max_val = stats.get(f"{col}_min"), stats.get(f"{col}_max")
        if (min_val is None and max_val is None) or np.allclose(min_val, max_val):
            dead_columns.append(col)
    if dead_columns:
        if verbose:
            logger.warning(f"Dropping {len(dead_columns):_} columns with no change: {textwrap.shorten(', '.join(dead_columns), width=max_log_text_width)}")
        df = df.drop(dead_columns)

    return df


def polars_df_info(df: pl.DataFrame) -> str:
    lines = []
    lines.append(f"{type(df)}")
    lines.append(f"RangeIndex: {df.height} entries, 0 to {df.height - 1 if df.height > 0 else 0}")
    if df.width > 0:
        first_col = df.columns[0]
        last_col = df.columns[-1]
        lines.append(f"Columns: {df.width} entries, {first_col} to {last_col}")
    else:
        lines.append("Columns: 0 entries")
    dtype_counts = Counter(str(dtype) for dtype in df.dtypes)
    dtype_str = ", ".join(f"{dtype}({count})" for dtype, count in sorted(dtype_counts.items()))
    lines.append(f"dtypes: {dtype_str}")
    size_kb = df.estimated_size(unit="gb")
    lines.append(f"memory usage: {size_kb:.1f}+ GB")
    return "\n".join(lines)

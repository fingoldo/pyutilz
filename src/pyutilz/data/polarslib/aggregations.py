"""Polars aggregate/time-series feature engineering: concentrations, weighted and EWM aggregates,
polars-ds statistics, and the ``group_by_dynamic``/``rolling`` feature builder.

Split out of the historical flat ``pyutilz.data.polarslib`` module; re-exported
from the package ``__init__`` to preserve the public import surface.
"""

from ._common import (
    Any,
    Iterable,
    Optional,
    POLARS_DEFAULT_NUMAGGS,
    POLARS_DEFAULT_QUANTILES,
    cs,
    logger,
    pl,
    explode_keeping_empty_as_null,
)
from typing import Callable, List, Literal, Tuple
from .columns import apply_agg_func_safe, cast_f64_to_f32, clean_numeric

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


def compute_concentrations(
    groupby_object: Any,
    entity: str,
    entity_name: str,
    by: str,
    groupby_columns: tuple,
    top_n: int = 2,
    return_ids: bool = True,
    return_values: bool = True,
    sort_by_concentration: bool = True,
    add_mean_concentration: bool = True,
    dtype: Any = pl.Float64,
    fields_remap: Optional[dict] = None,
) -> pl.DataFrame:
    """Computes within a group_by (dynamic or rolling), for example, concentrations of customers by total volume of their sales.
    groupby_columns must include both group_byand index_column arguments passed to group_by_dynamic.

    Ensuring sortedness after top_k_by is messy (requires zipping lists etc), so using sort+head instead."""

    assert return_ids or return_values  # nosec B101 - internal API-misuse guard: caller must request at least one output kind, not a security boundary

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
                fields=[f"{label}_top{i+1}_conc" for i in range(top_n)],
            )
        )

        unnest_rules.append(f"{label}_r{fields_remap.get(by,by)}")

    df = (
        explode_keeping_empty_as_null(groupby_object.agg(entity, by), entity_name, by)
        .group_by(*groupby_columns, entity_name)
        .agg(total_by=pl.sum(by))
        # A group whose total is 0 (or null) has no share to rank: 0/0 would be NaN and inf/NaN sort keys, so it gets 0.0.
        .with_columns(
            rel_total_by=pl.when(pl.sum("total_by").over(groupby_columns) != 0)
            .then(pl.col("total_by") / pl.sum("total_by").over(groupby_columns))
            .otherwise(0.0)
        )
        # A single group_by(...).agg(top_k_by(rel_total_by, top_n)) could replace the sort + head below, but head() on a
        # maintain_order group_by is what keeps the per-group rows in the requested order.
        .sort(*groupby_columns, "rel_total_by" if sort_by_concentration else entity_name, descending=([False] * len(groupby_columns) + [sort_by_concentration]))
        .group_by(groupby_columns, maintain_order=True)
        .head(top_n)
        .group_by(groupby_columns, maintain_order=True)
        .agg(exprs)
    )

    if columns_to_unnest:
        df = df.with_columns(columns_to_unnest).unnest(unnest_rules)

    return df  # type: ignore[no-any-return]  # polars: the group_by/agg/unnest chain above loses the concrete frame type through cs.* selectors


# ----------------------------------------------------------------------------------------------------------------------------
# FE in polars
# ----------------------------------------------------------------------------------------------------------------------------


def add_weighted_aggregates(
    columns_selector: "cs.Selector",  # quoted: older polars has no cs.Selector, and this annotation is evaluated at import
    weighting_columns: Iterable,
    fpref: str = "",
    fields_remap: Optional[dict] = None,
    nans_filler: float = 0.0,
    expr_filter: Optional[Callable[[Any], Any]] = None,
) -> List[pl.Expr]:
    """Computes weighted aggregates.

    A zero-sum-weight group (e.g. hedged buy/sell volumes, net-zero flow) makes the weighted-mean
    division produce Inf/NaN; ``nans_filler`` is applied via :func:`clean_numeric` before the
    ``_wmeanby_`` suffix is attached, so this never leaks unguarded Inf/NaN into the returned
    expressions the way an un-cleaned division would.

    ``expr_filter``, when given, is applied to BOTH the weighted-value numerator and the weight
    denominator. ``fpref`` only names the result: a caller that restricts a subgroup by prefixing
    the alias but leaves the expression unfiltered gets the whole-group number under every
    subgroup's name, i.e. one constant column presented as several distinct features.
    """
    if not fields_remap:
        fields_remap = {}
    afilter: Callable[[Any], Any] = expr_filter if expr_filter is not None else (lambda expr: expr)
    wcols: List[pl.Expr] = []
    if weighting_columns:
        for wcol in weighting_columns:
            all_other_num_cols = columns_selector - cs.by_name(wcol)
            raw_weighted_mean = afilter(all_other_num_cols * pl.col(wcol)).sum() / afilter(pl.col(wcol)).sum()
            weighted_mean = clean_numeric(raw_weighted_mean, nans_filler=nans_filler).name.suffix(f"_{fpref}wmeanby_{fields_remap.get(wcol,wcol)}")
            wcols.append(weighted_mean)
            # A weighted standard deviation (square root of the weight-averaged squared deviation from weighted_mean, suffixed
            # _wstdby_<wcol>) is not emitted: the multi-column polars expression for it raised an error when it was tried.
    return wcols


def _or_default(value: Any, default: Any) -> Any:
    """Return ``value`` unless it is None, in which case return ``default``."""
    return default if value is None else value


def _expand_default_fields(
    df: pl.DataFrame,
    boolean_fields: Optional[list],
    ts_diff_fields: Optional[list],
    numerical_fields: Optional[list],
    categorical_fields: Optional[list],
) -> tuple:
    """Fill each unspecified field group with the columns of the matching dtype found in ``df``."""
    return (
        list(cs.expand_selector(df.head(), cs.by_dtype(pl.Boolean))) if boolean_fields is None else boolean_fields,
        list(cs.expand_selector(df.head(), cs.by_dtype(pl.Datetime))) if ts_diff_fields is None else ts_diff_fields,
        list(cs.expand_selector(df.head(), cs.numeric())) if numerical_fields is None else numerical_fields,
        list(cs.expand_selector(df.head(), cs.by_dtype(pl.Categorical, pl.Utf8))) if categorical_fields is None else categorical_fields,
    )


def _subgroup_categoricals(orig_categorical_fields: list, filter_field: str) -> list:
    """Return a copy of the categorical fields without the subgroup's own filter field."""
    categorical_fields = orig_categorical_fields.copy()
    if filter_field in categorical_fields:
        categorical_fields.remove(filter_field)
    return categorical_fields


def _without(fields: Optional[list], excluded: list) -> Optional[list]:
    """Return ``fields`` minus ``excluded`` as a list; a falsy ``fields`` is returned unchanged."""
    return list(set(fields) - set(excluded)) if fields else fields


def _import_polars_ds_for_avx2() -> None:
    """Import polars_ds when the CPU supports AVX2 (the pds-backed features are AVX2-only)."""
    if _facade.check_cpu_flag("avx2"):
        import polars_ds  # noqa: F401 -- import-time side effect: a missing polars_ds must surface here, as before


def _warn_if_cuda_unavailable(engine: str) -> None:
    """Warn when the GPU feature-engineering engine is requested but CUDA is not available."""
    if engine == "gpu" and not _facade.is_cuda_available():
        logger.warning("GPU FE path chosen, but CUDA seems to be unavailable on this system")


def _apply_exclusions(
    boolean_fields: Optional[list],
    ts_diff_fields: Optional[list],
    numerical_fields: Optional[list],
    categorical_fields: Optional[list],
    exclude_fields: Optional[list],
) -> tuple:
    """Drop the excluded fields from each field group; without exclusions the groups are returned unchanged."""
    if not exclude_fields:
        return boolean_fields, ts_diff_fields, numerical_fields, categorical_fields
    return (
        _without(boolean_fields, exclude_fields),
        _without(ts_diff_fields, exclude_fields),
        _without(numerical_fields, exclude_fields),
        _without(categorical_fields, exclude_fields),
    )


def _boolean_feature_exprs(af: Callable, fpref: str, boolean_fields: list, fields_remap: dict, dtype: Any) -> list:
    """Mean expressions for the boolean columns, plus their Lempel-Ziv ratio on AVX2 CPUs."""
    exprs: list = []
    if boolean_fields:
        exprs.extend(
            [getattr(af(pl.col(field)), func)().alias(f"{fpref}{fields_remap.get(field,field)}_{func}") for field in boolean_fields for func in ["mean"]]
        )
        # +lziv
        if _facade.check_cpu_flag("avx2"):
            import polars_ds as pds

            exprs.extend(
                [pds.query_lempel_ziv(af(pl.col(field)), as_ratio=True).cast(dtype).alias(f"{fpref}{fields_remap.get(field,field)}_lziv") for field in boolean_fields]
            )
    return exprs


def _peak_feature_exprs(af: Callable, fpref: str, numerical_fields: list, fields_remap: dict, dtype: Any) -> list:
    """Mean of the peak-min/peak-max statistics, plus their Lempel-Ziv ratio on AVX2 CPUs."""
    # Means
    exprs = [
        getattr(af(pl.col(field)), func)().mean().cast(dtype).alias(f"{fpref}{fields_remap.get(field,field)}_{func}_mean")
        for field in numerical_fields
        for func in ["peak_min", "peak_max"]
    ]
    # +lziv
    if _facade.check_cpu_flag("avx2"):
        import polars_ds as pds

        exprs.extend(
            [
                pds.query_lempel_ziv(getattr(af(pl.col(field)), func)(), as_ratio=True).cast(dtype).alias(f"{fpref}{fields_remap.get(field,field)}_{func}_lziv")
                for field in numerical_fields
                for func in ["peak_min", "peak_max"]
            ]
        )
    return exprs


def _numagg_quantile_exprs(af: Callable, fpref: str, numerical_fields: list, numaggs: list, quantiles: list, fields_remap: dict) -> list:
    """Numerical aggregations and quantiles over the numerical columns."""
    # Numaggs over numerical columns
    exprs = [getattr(af(pl.col(field)), func)().alias(f"{fpref}{fields_remap.get(field,field)}_{func}") for field in numerical_fields for func in numaggs]
    # Quantiles
    exprs.extend([af(pl.col(field)).quantile(q).alias(f"{fpref}{fields_remap.get(field,field)}_quantile_{q}") for field in numerical_fields for q in quantiles])
    return exprs


def _othersvals_exprs(
    af: Callable,
    fpref: str,
    numerical_fields: list,
    filter_field: Optional[str],
    exclude_fields: Optional[list],
    othersvals_basic_fields: Optional[list],
    othersvals_other_fields: Optional[list],
    othersvals_excluded_fields: Optional[list],
) -> list:
    """Values of the other columns at each numerical column's arg-max and arg-min."""
    exprs: list = []
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

            # Both sides must be restricted to the active subgroup: an arg_max computed over
            # the whole group indexes a row the filtered `other_columns` no longer contains.
            exprs.append(af(other_columns).get(af(pl.col(col)).arg_max().alias("arg_max")).name.suffix(f"_{fpref}at_{col}_max"))
            exprs.append(af(other_columns).get(af(pl.col(col)).arg_min().alias("arg_min")).name.suffix(f"_{fpref}at_{col}_min"))
    return exprs


def _ewm_feature_exprs(
    af: Callable,
    fpref: str,
    numerical_fields: list,
    ewm_basic_funcs: list,
    ewm_spans: list,
    ewm_final_funcs: list,
    ewm_timestamp: Optional[str],
    ewm_time_half_lifes: list,
    fields_remap: dict,
) -> list:
    """Exponentially weighted mean/std over spans, plus time-weighted EWM when a timestamp and half-lives are given."""
    # Exponentially weighted mean/std
    exprs = [
        getattr(getattr(af(pl.col(field)), func)(span=span), agg_func)().alias(f"{fpref}{fields_remap.get(field,field)}_{func}_span={span}_{agg_func}")
        for field in numerical_fields
        for func in ewm_basic_funcs
        for span in ewm_spans
        for agg_func in ewm_final_funcs
    ]

    if ewm_timestamp and ewm_time_half_lifes:  # careful: causes a bug sometimes (lengths mismatch)
        exprs.extend(
            [
                getattr(af(pl.col(field)).ewm_mean_by(by=ewm_timestamp, half_life=half_life), agg_func)().alias(
                    f"{fpref}{fields_remap.get(field,field)}_ewm_ts_hl={half_life}_{agg_func}"
                )
                for field in numerical_fields
                for half_life in ewm_time_half_lifes
                for agg_func in ewm_final_funcs
            ]
        )
    return exprs


def _nunique_exprs(af: Callable, fpref: str, categorical_fields: list, fields_remap: dict) -> list:
    """Number of distinct values of each categorical column."""
    return [af(pl.col(field)).n_unique().alias(f"{fpref}{fields_remap.get(field,field)}_nunique") for field in categorical_fields]


def _mode_exprs(af: Callable, fpref: str, numerical_fields: list, fields_remap: dict) -> list:
    """Min, max and mean of the mode of each numerical column."""
    # Modes require special treatment
    return [
        getattr(af(pl.col(field)).mode(), func)().alias(f"{fpref}{fields_remap.get(field,field)}_mode_{func}")
        for field in numerical_fields
        for func in ["min", "max", "mean"]
    ]


def _concentration_exprs(
    af: Callable,
    fpref: str,
    categorical_fields: list,
    fields_remap: dict,
    dtype: Any,
    concentration_top_n: int,
    concentrations_params: dict,
) -> tuple:
    """Top-N value proportions of each categorical column, with the unnest rules that split them into columns."""
    exprs: list = []
    columns_to_unnest: list = []
    unnest_rules: list = []
    if concentration_top_n > 0:
        for field in categorical_fields:
            field_concentration_top_n = concentrations_params.get(field, concentration_top_n)
            # Validate field_concentration_top_n
            if not isinstance(field_concentration_top_n, int) or field_concentration_top_n <= 0:
                logger.warning("Skipping field %s: invalid top_n=%s", field, field_concentration_top_n)  # audit-exempt: unthrottled_hot_loop_log -- per invalid field of a caller-supplied field list, bounded by its length
                continue
            alias = f"{fpref}{fields_remap.get(field,field)}_top{field_concentration_top_n}"
            exprs.append(af(pl.col(field)).value_counts(sort=True, normalize=True).head(field_concentration_top_n).struct.field("proportion").alias(alias))

            columns_to_unnest.extend(
                [
                    pl.col(alias).list.mean().cast(dtype).alias(f"{alias}_avg_conc"),
                    pl.col(alias).list.to_struct(
                        fields=[f"{fpref}{fields_remap.get(field,field)}_top{i+1}_conc" for i in range(field_concentration_top_n)],
                    ),
                ]
            )
            unnest_rules.append(alias)
    return exprs, columns_to_unnest, unnest_rules


def _tsd_exprs(
    af: Callable,
    fpref: str,
    ts_diff_fields: list,
    ts_numaggs: list,
    nans_filler: float,
    dtype: Any,
    fields_remap: dict,
    tds_quantiles: list,
) -> list:
    """Aggregations and quantiles of the per-row time differences (in minutes) of each datetime column."""
    # Time diffs: numaggs
    exprs = [
        getattr(apply_agg_func_safe(af(pl.col(field)).diff().dt.total_seconds() / 60, func_name=func, nans_filler=nans_filler), func)()
        .cast(dtype)
        .alias(f"{fpref}{fields_remap.get(field,field)}_tsd_{func}")
        for field in ts_diff_fields
        for func in ts_numaggs
    ]
    if tds_quantiles:
        # Time diffs: Quantiles
        exprs.extend(
            [
                (af(pl.col(field)).diff().dt.total_seconds() / 60).quantile(q).alias(f"{fpref}{fields_remap.get(field,field)}_tsd_quantile={q}")
                for field in ts_diff_fields
                for q in tds_quantiles
            ]
        )
    return exprs


def _pds_stat_exprs(
    af: Callable,
    fpref: str,
    pds_fields: list,
    pds_numaggs: list,
    fields_remap: dict,
    dtype: Any,
    pds_params: dict,
    use_parametrized_pds_features: bool,
) -> tuple:
    """polars-ds simple stats per field and, when enabled, the parametrized stats; returns expressions and unnest rules."""
    import polars_ds as pds

    exprs: list = []
    columns_to_unnest: list = []
    unnest_rules: list = []
    # simple stats with no params
    for field in pds_fields:
        exprs.extend(
            [getattr(pds, func)(af(pl.col(field))).alias(f"{fpref}{fields_remap.get(field,field)}_{func.replace('query_','')}") for func in pds_numaggs]
        )
        if "query_first_digit_cnt" in pds_numaggs:
            NDIGITS = 9
            alias = f"{fpref}{fields_remap.get(field,field)}_{'query_first_digit_cnt'.replace('query_','')}"
            columns_to_unnest.append(pl.col(alias).list.to_struct(fields=[f"{alias}_{i+1}" for i in range(NDIGITS)]))
            unnest_rules.append(alias)

    # stats with params
    if use_parametrized_pds_features:
        lag, n_maxima, n_lags = pds_params.get("lag", 1), pds_params.get("n_maxima", 1), pds_params.get("n_lags", 1)
        for field in pds_fields:
            exprs.extend(
                [
                    pds.query_c3_stats(af(pl.col(field)), lag=lag).cast(dtype).alias(f"{fpref}{fields_remap.get(field,field)}_c3_stats-{lag}"),
                    pds.query_mean_n_abs_max(af(pl.col(field)), n_maxima=n_maxima).alias(f"{fpref}{fields_remap.get(field,field)}_mean_n_abs_max-{n_maxima}"),
                    pds.query_time_reversal_asymmetry_stats(af(pl.col(field)), n_lags=n_lags).alias(f"{fpref}{fields_remap.get(field,field)}_tras-{n_lags}"),
                ]
            )
    return exprs, columns_to_unnest, unnest_rules


def _corr_exprs(af: Callable, fpref: str, corr_fields: Optional[dict], corr_methods: list, dtype: Any) -> list:
    """Pairwise correlations between the configured field pairs, for every correlation method."""
    import polars_ds as pds

    # Expressions, not bare column names: a name carries no filter, so the
    # correlation would be the whole-group one under a subgroup alias.
    return [
        pds.corr(af(pl.col(corr_x)), af(pl.col(corr_y)), method=corr_method).cast(dtype).alias(f"{fpref}corr_{corr_x}-{corr_y}-{corr_method}")
        for corr_x, corr_y in (corr_fields or {}).items()
        for corr_method in corr_methods
    ]


def _linreg_exprs(
    af: Callable,
    fpref: str,
    linreg_fields: list,
    fields_remap: dict,
    linreg_timestamp_field: Optional[str],
) -> tuple:
    """Linear regressions of each field on its row index (and, optionally, on elapsed seconds of a timestamp field)."""
    import polars_ds as pds

    exprs: list = []
    columns_to_unnest: list = []
    unnest_rules: list = []
    for field in linreg_fields:
        alias = f"{fpref}{fields_remap.get(field,field)}_linreg"
        # pl.len() and the target both have to see the subgroup only, otherwise the fit is
        # the whole-group one and gets emitted identically under every subgroup's alias.
        exprs.append(pds.simple_lin_reg(pl.int_range(af(pl.col(field)).len()), target=af(pl.col(field)), add_bias=True).alias(alias))
        columns_to_unnest.extend(
            [
                pl.col(alias).list.to_struct(
                    fields=[f"{alias}_k", f"{alias}_b"],
                )
            ]
        )
        unnest_rules.append(alias)
        if linreg_timestamp_field:
            alias = f"{fpref}{fields_remap.get(field,field)}_linregby_{linreg_timestamp_field}"
            exprs.append(
                pds.simple_lin_reg(
                    (af(pl.col(linreg_timestamp_field)) - af(pl.col(linreg_timestamp_field)).min()).dt.total_seconds(),
                    target=af(pl.col(field)),
                    add_bias=True,
                ).alias(alias)
            )
            columns_to_unnest.extend(
                [
                    pl.col(alias).list.to_struct(
                        fields=[f"{alias}_k", f"{alias}_b"],
                    )
                ]
            )
            unnest_rules.append(alias)
    return exprs, columns_to_unnest, unnest_rules


def _cpu_only_exprs(
    af: Callable,
    fpref: str,
    numerical_fields: list,
    categorical_fields: list,
    ts_diff_fields: list,
    pds_fields: list,
    linreg_fields: list,
    fields_remap: dict,
    dtype: Any,
    nans_filler: float,
    ts_numaggs: list,
    tds_quantiles: list,
    pds_numaggs: list,
    pds_params: dict,
    use_parametrized_pds_features: bool,
    corr_methods: list,
    corr_fields: Optional[dict],
    linreg_timestamp_field: Optional[str],
    concentration_top_n: int,
    concentrations_params: dict,
) -> tuple:
    """Features that are computed only on the CPU engine (modes, concentrations, time diffs, polars-ds stats)."""
    exprs = _mode_exprs(af, fpref, numerical_fields, fields_remap)
    conc_exprs, columns_to_unnest, unnest_rules = _concentration_exprs(
        af, fpref, categorical_fields, fields_remap, dtype, concentration_top_n, concentrations_params
    )
    exprs.extend(conc_exprs)
    exprs.extend(_tsd_exprs(af, fpref, ts_diff_fields, ts_numaggs, nans_filler, dtype, fields_remap, tds_quantiles))

    if _facade.check_cpu_flag("avx2"):
        pds_exprs, pds_unnest, pds_rules = _pds_stat_exprs(af, fpref, pds_fields, pds_numaggs, fields_remap, dtype, pds_params, use_parametrized_pds_features)
        exprs.extend(pds_exprs)
        columns_to_unnest.extend(pds_unnest)
        unnest_rules.extend(pds_rules)
        exprs.extend(_corr_exprs(af, fpref, corr_fields, corr_methods, dtype))
        lin_exprs, lin_unnest, lin_rules = _linreg_exprs(af, fpref, linreg_fields, fields_remap, linreg_timestamp_field)
        exprs.extend(lin_exprs)
        columns_to_unnest.extend(lin_unnest)
        unnest_rules.extend(lin_rules)
    return exprs, columns_to_unnest, unnest_rules


def _ewm_defaults(
    ewm_spans: Optional[list], ewm_time_half_lifes: Optional[list], ewm_basic_funcs: Optional[list], ewm_final_funcs: Optional[list]
) -> Tuple[list, list, list, list]:
    """Replace each falsy EWM list argument with its default (empty spans/half-lifes, mean/std and mean funcs)."""
    if not ewm_spans:
        ewm_spans = []
    if not ewm_time_half_lifes:
        ewm_time_half_lifes = []
    if not ewm_basic_funcs:
        ewm_basic_funcs = "ewm_mean ewm_std".split()
    if not ewm_final_funcs:
        ewm_final_funcs = "mean".split()
    return ewm_spans, ewm_time_half_lifes, ewm_basic_funcs, ewm_final_funcs


def _filter_value_feature_exprs(
    *,
    af: Callable[[pl.Expr], pl.Expr],
    fpref: str,
    filter_field: str,
    boolean_fields: list,
    numerical_fields: list,
    categorical_fields: list,
    ts_diff_fields: list,
    pds_fields: list,
    linreg_fields: list,
    pds_numaggs: list,
    pds_params: dict,
    numaggs: list,
    quantiles: list,
    ts_numaggs: list,
    tds_quantiles: list,
    fields_remap: dict,
    dtype: Any,
    nans_filler: float,
    engine: str,
    add_peaks_stats: bool,
    weighting_fields: Optional[list],
    exclude_fields: Optional[list],
    othersvals_at_extremums: bool,
    othersvals_basic_fields: Optional[list],
    othersvals_other_fields: Optional[list],
    othersvals_excluded_fields: Optional[list],
    ewm_spans: list,
    ewm_timestamp: Optional[str],
    ewm_time_half_lifes: list,
    ewm_basic_funcs: list,
    ewm_final_funcs: list,
    use_parametrized_pds_features: bool,
    corr_methods: list,
    corr_fields: Optional[dict],
    linreg_timestamp_field: Optional[str],
    concentration_top_n: int,
    concentrations_params: dict,
) -> Tuple[list, list, list]:
    """Feature expressions, unnest columns and unnest rules for one subgroup filter value (``af`` applies its filter)."""
    exprs: list = []
    unnest: list = []
    rules: list = []
    exprs.append(af(cs.first()).len().alias(f"{fpref}nrecs"))
    exprs.extend(_boolean_feature_exprs(af, fpref, boolean_fields, fields_remap, dtype))

    if add_peaks_stats:
        exprs.extend(_peak_feature_exprs(af, fpref, numerical_fields, fields_remap, dtype))

    exprs.extend(_numagg_quantile_exprs(af, fpref, numerical_fields, numaggs, quantiles, fields_remap))

    # Weighting
    if weighting_fields:
        wcols = add_weighted_aggregates(
            columns_selector=(cs.numeric() - cs.by_name(exclude_fields or [])),
            weighting_columns=weighting_fields,
            fpref=fpref,
            fields_remap=fields_remap,
            expr_filter=af,
        )
        exprs.extend(wcols)

    if othersvals_at_extremums:
        exprs.extend(
            _othersvals_exprs(
                af,
                fpref,
                numerical_fields,
                filter_field,
                exclude_fields,
                othersvals_basic_fields,
                othersvals_other_fields,
                othersvals_excluded_fields,
            )
        )

    exprs.extend(
        _ewm_feature_exprs(
            af,
            fpref,
            numerical_fields,
            ewm_basic_funcs,
            ewm_spans,
            ewm_final_funcs,
            ewm_timestamp,
            ewm_time_half_lifes,
            fields_remap,
        )
    )

    # Categorical stats. For gpu mode, categoricals need to be converted to String upfront.
    exprs.extend(_nunique_exprs(af, fpref, categorical_fields, fields_remap))

    if engine != "gpu":
        cpu_exprs, cpu_unnest, cpu_rules = _cpu_only_exprs(
            af,
            fpref,
            numerical_fields,
            categorical_fields,
            ts_diff_fields,
            pds_fields,
            linreg_fields,
            fields_remap,
            dtype,
            nans_filler,
            ts_numaggs,
            tds_quantiles,
            pds_numaggs,
            pds_params,
            use_parametrized_pds_features,
            corr_methods,
            corr_fields,
            linreg_timestamp_field,
            concentration_top_n,
            concentrations_params,
        )
        exprs.extend(cpu_exprs)
        unnest.extend(cpu_unnest)
        rules.extend(cpu_rules)
    return exprs, unnest, rules


def build_aggregate_features_polars(
    df: pl.DataFrame,
    #
    boolean_fields: Optional[list] = None,
    numerical_fields: Optional[list] = None,
    categorical_fields: Optional[list] = None,
    ts_diff_fields: Optional[list] = None,
    exclude_fields: Optional[list] = None,
    weighting_fields: Optional[list] = None,
    #
    subgroups: Optional[dict] = None,
    numaggs: Optional[list] = None,
    quantiles: Optional[list] = None,
    tds_quantiles: Optional[list] = None,
    #
    engine: str = "cpu",
    dtype: Any = pl.Float64,
    fields_remap: Optional[dict] = None,
    nans_filler: float = 0.0,
    concentration_top_n: int = 3,
    concentrations_params: Optional[dict] = None,
    add_peaks_stats: bool = True,
    custom_expressions: Optional[list] = None,
    #
    othersvals_at_extremums: bool = False,
    othersvals_basic_fields: Optional[list] = None,
    othersvals_other_fields: Optional[list] = None,
    othersvals_excluded_fields: Optional[list] = None,
    #
    ewm_spans: Optional[list] = None,
    ewm_timestamp: Optional[str] = None,
    ewm_time_half_lifes: Optional[list] = None,
    ewm_basic_funcs: Optional[list] = None,
    ewm_final_funcs: Optional[list] = None,
    # polars-ds features (work on AVX2+ only!)
    pds_params: Optional[dict] = None,
    pds_numaggs: Optional[list] = None,
    corr_methods: Optional[list] = None,
    corr_fields: Optional[dict] = None,
    pds_fields: Optional[list] = None,
    linreg_fields: Optional[list] = None,
    linreg_timestamp_field: Optional[str] = None,
    use_parametrized_pds_features: bool = True,
) -> tuple:
    """Build the list of polars aggregate-feature expressions used by ``create_ts_features_polars`` (numaggs, quantiles,
    weighted/EWM aggregates, categorical concentrations, peaks/lziv, polars-ds stats, correlations, linregs, etc.),
    optionally split by ``subgroups`` filters.

    Args:
        df: Polars DataFrame the feature expressions are built against; used to expand default field lists.
        boolean_fields: Boolean columns to aggregate. None selects all Boolean columns.
        numerical_fields: Numeric columns for the numeric aggregates. None selects all numeric columns.
        categorical_fields: Categorical/string columns for n-unique and concentration features. None selects them by dtype.
        ts_diff_fields: Datetime columns whose successive time differences are aggregated. None selects all Datetime columns.
        exclude_fields: Columns removed from every field group, and from the othersvals and weighting selectors.
        weighting_fields: Columns used as weights for the weighted-mean aggregates.
        subgroups: Mapping of filter field to a list of values. One feature set is built per value, filtered to that value.
            An empty filter field means no filter.
        numaggs: Basic aggregations (for example mean, std) applied to each numerical field.
        quantiles: Quantiles computed for each numerical field.
        tds_quantiles: Quantiles of the time-difference fields.
        engine: ``"cpu"`` or ``"gpu"``. The GPU engine skips the CPU-only feature group.
        dtype: Dtype the aggregated values are cast to.
        fields_remap: Mapping that renames fields inside the generated column names.
        nans_filler: Value substituted for NaN/Inf results of the weighted and time-difference aggregates.
        concentration_top_n: Number of top categories for the concentration features. 0 disables them.
        concentrations_params: Per-field overrides of ``concentration_top_n``, keyed by field name.
        add_peaks_stats: Whether to add peak statistics for the numerical fields.
        custom_expressions: Extra polars expressions appended to the feature list.
        othersvals_at_extremums: Whether to add the values of other fields observed at each field's extremum.
        othersvals_basic_fields: Fields for which other-field values are taken. None means all.
        othersvals_other_fields: Fields whose values are taken at the extremums. None means all other columns.
        othersvals_excluded_fields: Fields never taken as other-field values.
        ewm_spans: Spans for the exponentially weighted aggregates.
        ewm_timestamp: Timestamp column for the time-weighted EWM. Used together with ``ewm_time_half_lifes``.
        ewm_time_half_lifes: Half-lives for the time-weighted EWM.
        ewm_basic_funcs: EWM statistics computed per span. Defaults to ``ewm_mean`` and ``ewm_std``.
        ewm_final_funcs: Aggregations applied to each EWM series. Defaults to ``mean``.
        pds_params: Parameters for the polars-ds statistics (``lag``, ``n_maxima``, ``n_lags``).
        pds_numaggs: polars-ds statistics computed per field.
        corr_methods: Correlation methods applied to each pair in ``corr_fields``.
        corr_fields: Mapping of x field name to y field name for pairwise correlations.
        pds_fields: Numerical fields for the polars-ds statistics. None means ``numerical_fields``.
        linreg_fields: Numerical fields for the linear-regression features. None means ``numerical_fields``.
        linreg_timestamp_field: Optional time column for time-based linear regression.
        use_parametrized_pds_features: Whether to add the parametrized polars-ds features.

    Returns:
        A tuple ``(feature_expressions, columns_to_unnest, unnest_rules)`` to be passed to a ``group_by(...).agg(...)``
        call followed by ``.with_columns(columns_to_unnest).unnest(unnest_rules)``.
    """

    # ----------------------------------------------------------------------------------------------------------------------------
    # Checks
    # ----------------------------------------------------------------------------------------------------------------------------

    if engine not in ("cpu", "gpu"):  # developer-supplied engine-selection parameter; API-misuse guard, not a security boundary
        raise AssertionError(f"engine must be 'cpu' or 'gpu', got {engine!r}")

    _warn_if_cuda_unavailable(engine)

    # ----------------------------------------------------------------------------------------------------------------------------
    # Inits
    # ----------------------------------------------------------------------------------------------------------------------------

    _import_polars_ds_for_avx2()

    # Params

    fields_remap = _or_default(fields_remap, {})
    concentrations_params = concentrations_params or {}
    numaggs = _or_default(numaggs, POLARS_DEFAULT_NUMAGGS)
    quantiles = _or_default(quantiles, POLARS_DEFAULT_QUANTILES)
    tds_quantiles = _or_default(tds_quantiles, POLARS_DEFAULT_QUANTILES)

    ewm_spans, ewm_time_half_lifes, ewm_basic_funcs, ewm_final_funcs = _ewm_defaults(ewm_spans, ewm_time_half_lifes, ewm_basic_funcs, ewm_final_funcs)

    pds_numaggs = _or_default(pds_numaggs, "hmean query_abs_energy query_cid_ce query_mean_abs_change".split())  # query_permute_entropy # query_first_digit_cnt
    corr_methods = _or_default(corr_methods, ["pearson", "spearman", "xi", "kendall", "bicor"])
    pds_params = _or_default(pds_params, dict(lag=1, n_maxima=1, n_lags=1))

    # Fields

    boolean_fields, ts_diff_fields, numerical_fields, categorical_fields = _expand_default_fields(
        df, boolean_fields, ts_diff_fields, numerical_fields, categorical_fields
    )

    boolean_fields, ts_diff_fields, numerical_fields, categorical_fields = _apply_exclusions(
        boolean_fields, ts_diff_fields, numerical_fields, categorical_fields, exclude_fields
    )

    pds_fields = _or_default(pds_fields, numerical_fields)
    linreg_fields = _or_default(linreg_fields, numerical_fields)

    if not subgroups:
        subgroups = {"": [""]}  # {"action": ["buy", "sell"]}

    # Counters

    feature_expressions = list(custom_expressions) if custom_expressions else []
    columns_to_unnest, unnest_rules = [], []

    ts_numaggs = [el for el in numaggs if el not in ("first",)]

    # ----------------------------------------------------------------------------------------------------------------------------
    # Actual building
    # ----------------------------------------------------------------------------------------------------------------------------

    num_no_filter: int = 0
    orig_categorical_fields = categorical_fields.copy()
    for filter_field, filter_values in subgroups.items():
        # mypy loses the None-narrowing established in the "Fields"/"Params" blocks above once
        # inside this loop; re-assert here (all are genuinely guaranteed non-None by that point).
        assert boolean_fields is not None and numerical_fields is not None and categorical_fields is not None and ts_diff_fields is not None
        assert numaggs is not None and quantiles is not None
        assert ewm_spans is not None and ewm_basic_funcs is not None and ewm_final_funcs is not None
        assert pds_fields is not None and pds_numaggs is not None and linreg_fields is not None

        if not isinstance(filter_values, list):  # shape of the caller-supplied subgroups dict values; not a security boundary
            raise AssertionError(f"subgroup filter values must be a list, got {type(filter_values).__name__}")

        if not filter_field:
            num_no_filter += 1
            if num_no_filter > 1:  # at most one "no filter" (falsy filter_field) subgroup entry is expected; not a security boundary
                raise AssertionError(f"at most one subgroup may have no filter field, found {num_no_filter}")
        categorical_fields = _subgroup_categoricals(orig_categorical_fields, filter_field) if filter_field else categorical_fields

        for filter_value in filter_values:

            def af(expr: pl.Expr, _filter_field=filter_field, _filter_value=filter_value) -> pl.Expr:
                """Wrap ``expr`` with a ``.filter(col == value)`` for the current subgroup, or leave it unfiltered if there is no active filter field."""
                return expr if not _filter_field else expr.filter(pl.col(_filter_field) == _filter_value)

            fpref = "" if not filter_field else f"{filter_field}_{filter_value}_"
            exprs, unnest, rules = _filter_value_feature_exprs(
                af=af,
                fpref=fpref,
                filter_field=filter_field,
                boolean_fields=boolean_fields,
                numerical_fields=numerical_fields,
                categorical_fields=categorical_fields,
                ts_diff_fields=ts_diff_fields,
                pds_fields=pds_fields,
                linreg_fields=linreg_fields,
                pds_numaggs=pds_numaggs,
                pds_params=pds_params,
                numaggs=numaggs,
                quantiles=quantiles,
                ts_numaggs=ts_numaggs,
                tds_quantiles=tds_quantiles,
                fields_remap=fields_remap,
                dtype=dtype,
                nans_filler=nans_filler,
                engine=engine,
                add_peaks_stats=add_peaks_stats,
                weighting_fields=weighting_fields,
                exclude_fields=exclude_fields,
                othersvals_at_extremums=othersvals_at_extremums,
                othersvals_basic_fields=othersvals_basic_fields,
                othersvals_other_fields=othersvals_other_fields,
                othersvals_excluded_fields=othersvals_excluded_fields,
                ewm_spans=ewm_spans,
                ewm_timestamp=ewm_timestamp,
                ewm_time_half_lifes=ewm_time_half_lifes,
                ewm_basic_funcs=ewm_basic_funcs,
                ewm_final_funcs=ewm_final_funcs,
                use_parametrized_pds_features=use_parametrized_pds_features,
                corr_methods=corr_methods,
                corr_fields=corr_fields,
                linreg_timestamp_field=linreg_timestamp_field,
                concentration_top_n=concentration_top_n,
                concentrations_params=concentrations_params,
            )
            feature_expressions.extend(exprs)
            columns_to_unnest.extend(unnest)
            unnest_rules.extend(rules)

    return feature_expressions, columns_to_unnest, unnest_rules


def create_ts_features_polars(
    df: pl.DataFrame,
    index_column: str,
    period: str,
    every: Optional[str] = None,
    offset: Optional[str] = None,
    closed: Optional[Literal["left", "right", "both", "none"]] = None,
    label: Literal["left", "right", "datapoint"] = "left",
    group_by: Optional[str] = None,
    rolling: bool = False,
    include_boundaries: bool = False,
    clean_memory: bool = True,
    dtype: Any = pl.Float64,
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
        _facade.clean_ram()

    if group_by:
        additional_exclude = [group_by] if isinstance(group_by, str) else group_by
        existing_exclude_fields = kwargs.get("exclude_fields")
        if existing_exclude_fields:
            kwargs["exclude_fields"] = list(existing_exclude_fields) + additional_exclude
        else:
            kwargs["exclude_fields"] = additional_exclude

    expressions, columns_to_unnest, unnest_rules = build_aggregate_features_polars(df, dtype=dtype, **kwargs)

    if rolling:
        res = df.lazy().rolling(index_column=index_column, period=period, offset=offset, closed=closed, group_by=group_by).agg(expressions)
    else:
        res = (
            df.lazy()
            .group_by_dynamic(
                index_column=index_column,
                every=every if every is not None else period,
                period=period,
                offset=offset,
                closed=closed,
                label=label,
                group_by=group_by,
                include_boundaries=include_boundaries,
            )
            .agg(expressions)
        )

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
        _facade.clean_ram()

    logger.info("create_ts_features_polars using %s engine, %s threads...", engine, pl.thread_pool_size())
    res = res.collect(engine=engine)  # type: ignore[call-overload]  # "cpu" (this function's own documented default) works at runtime but isn't in polars' EngineType Literal
    logger.info("Done.")

    return res  # type: ignore[no-any-return]  # polars: LazyFrame.collect() through the call-overload ignore above yields Any

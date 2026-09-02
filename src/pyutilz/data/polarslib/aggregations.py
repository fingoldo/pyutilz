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
from typing import Literal
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
                upper_bound=top_n,
                fields=[f"{label}_top{i+1}_conc" for i in range(top_n)],
            )
        )

        unnest_rules.append(f"{label}_r{fields_remap.get(by,by)}")

    df = (
        explode_keeping_empty_as_null(groupby_object.agg(entity, by), entity_name, by)
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

    return df  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime


# ----------------------------------------------------------------------------------------------------------------------------
# FE in polars
# ----------------------------------------------------------------------------------------------------------------------------


def add_weighted_aggregates(
    columns_selector: object, weighting_columns: Iterable, fpref: str = "", fields_remap: Optional[dict] = None, nans_filler: float = 0.0
) -> list:
    """Computes weighted aggregates.

    A zero-sum-weight group (e.g. hedged buy/sell volumes, net-zero flow) makes the weighted-mean
    division produce Inf/NaN; ``nans_filler`` is applied via :func:`clean_numeric` before the
    ``_wmeanby_`` suffix is attached, so this never leaks unguarded Inf/NaN into the returned
    expressions the way an un-cleaned division would.
    """
    if not fields_remap:
        fields_remap = {}
    wcols = []
    if weighting_columns:
        for wcol in weighting_columns:
            all_other_num_cols: Any = columns_selector - cs.by_name(wcol)
            raw_weighted_mean = (all_other_num_cols * pl.col(wcol)).sum() / pl.col(wcol).sum()
            weighted_mean = clean_numeric(raw_weighted_mean, nans_filler=nans_filler).name.suffix(f"_{fpref}wmeanby_{fields_remap.get(wcol,wcol)}")
            wcols.append(weighted_mean)
            # !TODO causes error for now
            # weighted_std = ((pl.col(wcol) * (all_other_num_cols - weighted_mean) ** 2).sum() / pl.col(wcol).sum()).sqrt().name.suffix(f"_wstdby_{wcol}")
            # wcols.append(weighted_std)
    return wcols


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

    Returns:
        A tuple ``(feature_expressions, columns_to_unnest, unnest_rules)`` to be passed to a ``group_by(...).agg(...)``
        call followed by ``.with_columns(columns_to_unnest).unnest(unnest_rules)``.
    """

    # ----------------------------------------------------------------------------------------------------------------------------
    # Checks
    # ----------------------------------------------------------------------------------------------------------------------------

    assert engine in ("cpu", "gpu")  # nosec B101 - internal API-misuse guard on a developer-supplied engine-selection parameter, not a security boundary

    if engine == "gpu" and not _facade.is_cuda_available():
        # logger.warning(f"GPU FE path chosen, but Cuda seems to be unavailable on this system!")
        pass

    # ----------------------------------------------------------------------------------------------------------------------------
    # Inits
    # ----------------------------------------------------------------------------------------------------------------------------

    if _facade.check_cpu_flag("avx2"):
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
        ewm_spans = []
    if not ewm_time_half_lifes:
        ewm_time_half_lifes = []
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
        boolean_fields = list(cs.expand_selector(df.head(), cs.by_dtype(pl.Boolean)))
    if ts_diff_fields is None:
        ts_diff_fields = list(cs.expand_selector(df.head(), cs.by_dtype(pl.Datetime)))
    if numerical_fields is None:
        numerical_fields = list(cs.expand_selector(df.head(), cs.numeric()))
    if categorical_fields is None:
        categorical_fields = list(cs.expand_selector(df.head(), cs.by_dtype(pl.Categorical, pl.Utf8)))

    if exclude_fields:
        if boolean_fields:
            boolean_fields = list(set(boolean_fields) - set(exclude_fields))
        if ts_diff_fields:
            ts_diff_fields = list(set(ts_diff_fields) - set(exclude_fields))
        if numerical_fields:
            numerical_fields = list(set(numerical_fields) - set(exclude_fields))
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
        # mypy loses the None-narrowing established in the "Fields"/"Params" blocks above once
        # inside this loop; re-assert here (all are genuinely guaranteed non-None by that point).
        assert boolean_fields is not None and numerical_fields is not None and categorical_fields is not None and ts_diff_fields is not None
        assert numaggs is not None and quantiles is not None
        assert ewm_spans is not None and ewm_basic_funcs is not None and ewm_final_funcs is not None
        assert pds_fields is not None and pds_numaggs is not None and linreg_fields is not None

        assert isinstance(
            filter_values, list
        )  # nosec B101 - internal invariant on the shape of the caller-supplied subgroups dict values, not a security boundary

        if not filter_field:
            num_no_filter += 1
            assert (
                num_no_filter <= 1
            )  # nosec B101 - internal invariant: at most one "no filter" (falsy filter_field) entry is expected in subgroups, not a security boundary
        else:
            categorical_fields = orig_categorical_fields.copy()
            if filter_field in categorical_fields:
                categorical_fields.remove(filter_field)

        for filter_value in filter_values:

            def af(expr, _filter_field=filter_field, _filter_value=filter_value) -> pl.Expr:
                """Wrap ``expr`` with a ``.filter(col == value)`` for the current subgroup, or leave it unfiltered if there is no active filter field."""
                return expr if not _filter_field else expr.filter(pl.col(_filter_field) == _filter_value)  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime

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
                if _facade.check_cpu_flag("avx2"):
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
                if _facade.check_cpu_flag("avx2"):
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
                    columns_selector=(cs.numeric() - cs.by_name(exclude_fields or [])),
                    weighting_columns=weighting_fields,
                    fpref=fpref,
                    fields_remap=fields_remap,
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
                            logger.warning("Skipping field %s: invalid top_n=%s", field, field_concentration_top_n)
                            continue
                        alias = f"{fpref}{fields_remap.get(field,field)}_top{field_concentration_top_n}"
                        feature_expressions.append(
                            af(pl.col(field)).value_counts(sort=True, normalize=True).head(field_concentration_top_n).struct.field("proportion").alias(alias)
                        )

                        columns_to_unnest.extend(
                            [
                                pl.col(alias).list.mean().cast(dtype).alias(f"{alias}_avg_conc"),
                                pl.col(alias).list.to_struct(
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

                if _facade.check_cpu_flag("avx2"):

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
                            columns_to_unnest.append(pl.col(alias).list.to_struct(upper_bound=NDIGITS, fields=[f"{alias}_{i+1}" for i in range(NDIGITS)]))
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

    return res  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime

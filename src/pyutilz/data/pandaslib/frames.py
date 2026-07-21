"""Column/row/frame manipulation helpers (renaming, value showcasing, sharing,
constant/stale column removal, feature naming).

Split out of the historical flat ``pyutilz.data.pandaslib`` module; re-exported
from the package ``__init__`` to preserve the public import surface.
"""

from ._common import (
    Any,
    Dict,
    Optional,
    Sequence,
    Union,
    np,
    pd,
    pl,
    ctypes,
    Array,
    tqdmu,
    logger,
)

import pyutilz.data.pandaslib as _facade  # patchable-name indirection for HAS_IPYTHON/display/Markdown


def nullify_standard_values(
    df: pd.DataFrame, field: str, min_records: int = 300, persons_field: Optional[str] = None, min_persons: int = 0, placeholder=np.nan, verbose: bool = False
):
    """
    Replaces (inplace) all standard responses with NaN to indicate it's not custom
    """
    tmp = df[field].value_counts(dropna=False)
    standard_values = tmp[tmp > min_records].index.values
    if persons_field:
        # Use groupby for O(N) instead of O(N × M) performance
        person_counts = df[df[field].isin(standard_values)].groupby(field)[persons_field].nunique()
        top_values = person_counts[person_counts > min_persons].index.tolist()

        if verbose and top_values:
            print(f"Field {field}")  # noqa: T201 -- verbose=True is an explicit stdout-display contract (see test_verbose_output), not a log-volume toggle
            for val in top_values:
                qty = person_counts[val]
                print(f"\t: value {val} is not custom, as used by {qty} persons")  # noqa: T201 -- same verbose=True stdout contract
    else:
        top_values = standard_values
    df.loc[df[field].isin(top_values), field] = placeholder


def prefixize_columns(df: pd.DataFrame, prefix: str, special_prefixes: Optional[dict] = None, sep="_", exclusions: Optional[Sequence] = None, inplace: bool = True):
    """
    Prefix every column of a pandas dataframe (except clearly formulated exclusions) with some arbitrary prefix string - to identify variable's source
    """
    if special_prefixes is None:
        special_prefixes = {}
    if exclusions is None:
        exclusions = ()
    # Build column mapping once instead of duplicating 3 times
    columns = {col: special_prefixes.get(col, prefix) + sep + col if col not in exclusions else col for col in df.columns}
    if inplace:
        df.rename(columns=columns, inplace=True)
        return columns
    else:
        return df.rename(columns=columns, inplace=False)


def showcase_df_columns(
    df: Union[pd.DataFrame, pl.DataFrame],
    cols: Optional[list] = None,
    excluded_cols: Optional[list] = None,
    max_vars: Optional[int] = None,
    dropna: bool = False,
    use_markdown: bool = True,
    use_print: bool = True,
    max_cat_uniq_qty: int = 50,
    max_unique_percent: float = 0.001,
) -> tuple:
    """
    Show distribution of values for each dataframe column. Works with both pandas and polars DataFrames.

    Also detects low-variability features useful for ML feature selection:
    - rare_categories: for columns with <= max_cat_uniq_qty unique values, any value whose
      share of total rows is <= max_unique_percent is flagged as rare.
    - uninformative_features: columns where, after dropping all rare values, only 1 unique
      value remains. The dict value is the fraction of non-dominant rows (i.e. 1 - dominant_count/total).

    Returns:
        (rare_categories, uninformative_features) -- both dicts keyed by column name.

    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": [1, 1, 1, 2, 2, 3], "b": ["x", "x", "x", "y", "y", None]})
    >>> showcase_df_columns(df, use_markdown=False, use_print=True, dropna=True)
    A int64
    a
    1    3
    2    2
    3    1
    Name: count, dtype: int64
    B object
    b
    x    3
    y    2
    Name: count, dtype: int64
    ({}, {})

    >>> showcase_df_columns(df, cols=["a"], max_vars=1, use_markdown=False, use_print=True)
    A int64
    a
    1    3
    Name: count, dtype: int64
    ({}, {})

    >>> showcase_df_columns(df, cols=["a"], max_vars=0, use_markdown=False, use_print=True)
    A int64
    <BLANKLINE>
    ({}, {})

    >>> import polars as pl
    >>> dfp = pl.DataFrame({"a": [1, 1, 1, 2, 2, 3], "b": ["x", "x", "x", "y", "y", None]})
    >>> showcase_df_columns(dfp, use_markdown=False, use_print=True, dropna=True)
    A Int64
    a
    1    3
    2    2
    3    1
    Name: count, dtype: int64
    B String
    b
    x    3
    y    2
    Name: count, dtype: int64
    ({}, {})

    >>> showcase_df_columns(dfp, cols=["a"], max_vars=1, use_markdown=False, use_print=True)
    A Int64
    a
    1    3
    Name: count, dtype: int64
    ({}, {})

    >>> showcase_df_columns(dfp, cols=["a"], max_vars=0, use_markdown=False, use_print=True)
    A Int64
    <BLANKLINE>
    ({}, {})

    >>> showcase_df_columns(dfp, use_markdown=False, use_print=True, dropna=False)
    A Int64
    a
    1    3
    2    2
    3    1
    Name: count, dtype: int64
    B String
    b
    x       3
    y       2
    None    1
    Name: count, dtype: int64
    ({}, {})

    Rare/uninformative detection (max_unique_percent=0.34 means values with <=34% share are rare):

    >>> df2 = pd.DataFrame({"x": ["a"]*100 + ["b"]*2 + ["c"]*1})
    >>> r, u = showcase_df_columns(df2, use_markdown=False, use_print=True, max_unique_percent=0.05)
    X object
    x
    a    100
    b      2
    c      1
    Name: count, dtype: int64
    >>> sorted(r["x"])
    ['b', 'c']
    >>> "x" in u and u["x"] == 1 - 100/103
    True

    >>> dfp2 = pl.DataFrame({"x": ["a"]*100 + ["b"]*2 + ["c"]*1})
    >>> r, u = showcase_df_columns(dfp2, use_markdown=False, use_print=True, max_unique_percent=0.05)
    X String
    x
    a    100
    b      2
    c      1
    Name: count, dtype: int64
    >>> sorted(r["x"])
    ['b', 'c']
    >>> "x" in u and u["x"] == 1 - 100/103
    True
    """
    if excluded_cols is None:
        excluded_cols = []

    _is_polars = isinstance(df, pl.DataFrame)

    if cols is None or len(cols) == 0:
        cols = df.columns

    excluded_set = set(excluded_cols)
    target_cols = [c for c in cols if c not in excluded_set]

    rare_categories: Dict[Any, Any] = {}
    uninformative_features: Dict[Any, Any] = {}

    if not target_cols:
        return rare_categories, uninformative_features

    height = df.height if _is_polars else len(df)
    rare_threshold = max_unique_percent * height

    if _is_polars:
        # Build lazy value_counts queries for all columns, collect in parallel
        lazy_queries = []
        for var in target_cols:
            lq = df.lazy().select(pl.col(var))
            if dropna:
                lq = lq.drop_nulls()
            lq = lq.group_by(var).agg(pl.len().alias("count")).sort("count", descending=True)
            lazy_queries.append(lq)

        # pl.collect_all runs all queries in parallel via the Polars thread pool
        vc_results = pl.collect_all(lazy_queries)

        # Also collect n_unique in parallel for the rare-category check. Respect `dropna` here
        # too (drop_nulls() before n_unique() when dropna=True) so the eligibility gate agrees
        # with the dropna-aware value-counts computed above -- otherwise a null-inclusive count
        # can push a column one over max_cat_uniq_qty and skip it entirely, even though the
        # caller's dropna=True means they only care about the non-null distinct values.
        nuniq_queries = [
            (df.lazy().select(pl.col(var).drop_nulls().n_unique().alias("n")) if dropna else df.lazy().select(pl.col(var).n_unique().alias("n")))
            for var in target_cols
        ]
        nuniq_results = pl.collect_all(nuniq_queries)

        for var, vc, nuniq_df in zip(target_cols, vc_results, nuniq_results):
            dtype = df.schema[var]
            if use_markdown and _facade.HAS_IPYTHON:
                _facade.display(_facade.Markdown(f"**{var}** {dtype}"))
            if use_print or not _facade.HAS_IPYTHON:
                print(f"{var.upper()} {dtype}")  # noqa: T201 -- use_print is an explicit stdout-display contract, doctest-verified above

            if max_vars is not None and max_vars == 0:
                print("")  # noqa: T201 -- use_print is an explicit stdout-display contract, doctest-verified above
            elif vc.height == 0:
                stats = pd.Series([], name="count", dtype="int64")
                stats.index.name = var
                print(stats)  # noqa: T201 -- use_print is an explicit stdout-display contract, doctest-verified above
            else:
                vals = vc.get_column(var).to_list()
                counts = vc.get_column("count").to_list()
                stats = pd.Series(counts, index=vals, name="count")
                stats.index.name = var
                if max_vars is not None and max_vars > 0:
                    print(stats.head(max_vars))  # noqa: T201 -- use_print is an explicit stdout-display contract, doctest-verified above
                else:
                    print(stats)  # noqa: T201 -- use_print is an explicit stdout-display contract, doctest-verified above

            # Rare/uninformative analysis
            n_unique = nuniq_df.item(0, 0)
            if n_unique <= max_cat_uniq_qty and vc.height > 0:
                rare_mask = vc.get_column("count").to_list()
                rare_vals = vc.get_column(var).to_list()
                col_rare = [v for v, c in zip(rare_vals, rare_mask) if c <= rare_threshold]
                if col_rare:
                    rare_categories[var] = col_rare
                    non_rare_count = sum(c for c in rare_mask if c > rare_threshold)
                    non_rare_unique = sum(1 for c in rare_mask if c > rare_threshold)
                    if non_rare_unique <= 1:
                        uninformative_features[var] = 1 - non_rare_count / height if height > 0 else 0.0
    else:
        assert isinstance(df, pd.DataFrame)
        for var in target_cols:
            if use_markdown and _facade.HAS_IPYTHON:
                _facade.display(_facade.Markdown(f"**{var}** {df[var].dtype}"))
            if use_print or not _facade.HAS_IPYTHON:
                print(f"{var.upper()} {df[var].dtype}")  # noqa: T201 -- use_print is an explicit stdout-display contract, doctest-verified above
            stats = df[var].value_counts(dropna=dropna)
            if max_vars is not None:
                assert max_vars >= 0  # nosec B101 - internal invariant on a display-row-count parameter (only used to slice a printed head()), not a security boundary
                if max_vars > 0:
                    print(stats.head(max_vars))  # noqa: T201 -- use_print is an explicit stdout-display contract, doctest-verified above
                else:
                    print("")  # noqa: T201 -- use_print is an explicit stdout-display contract, doctest-verified above
            else:
                print(stats)  # noqa: T201 -- use_print is an explicit stdout-display contract, doctest-verified above

            # Rare/uninformative analysis -- gate respects `dropna` (matches the value_counts
            # computed above), so a column that's within max_cat_uniq_qty distinct non-null
            # values isn't silently skipped just because it also has nulls the caller asked to ignore.
            n_unique = df[var].nunique(dropna=dropna)
            if n_unique <= max_cat_uniq_qty and len(stats) > 0:
                full_stats = df[var].value_counts(dropna=dropna) if max_vars is not None else stats
                col_rare = full_stats[full_stats <= rare_threshold].index.tolist()
                if col_rare:
                    rare_categories[var] = col_rare
                    non_rare = full_stats[full_stats > rare_threshold]
                    if len(non_rare) <= 1:
                        non_rare_count = int(non_rare.sum()) if len(non_rare) == 1 else 0
                        uninformative_features[var] = 1 - non_rare_count / height if height > 0 else 0.0

    return rare_categories, uninformative_features


class FeatureNamer:
    """
    Used to save RAM when building dataframes from dicts with big feature names, by renaming them to small consecutive numbers
    >>>fname=FeatureNamer();fname('abc');fname('abcd');fname('abc'),fname.rev(0)
    (0, 'abc')
    """

    def __init__(self, initial_values: Optional[Sequence] = None):
        self.fnames_index = 0
        self.fnames: Dict[Any, Any] = {}
        self.revfnames: Dict[Any, Any] = {}
        if initial_values is None:
            initial_values = []
        for name in initial_values:
            self(name)

    def __call__(self, name: str):
        if name not in self.fnames:
            self.fnames[name] = self.fnames_index
            self.revfnames[self.fnames_index] = name
            self.fnames_index += 1
        return self.fnames[name]

    def rev(self, key: int):
        """Return the original feature name previously mapped to the numeric ``key``, or None if unknown."""
        return self.revfnames.get(key)


def share_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a cloned dataframe from create numpy mem views that can be shared with multiple worker processes as a global variable.
    Should not contain datetime dtype! or won't be able to fit the double dtype.
    Ram usage grows from 7x (while cloning) to 3x (while using) of the original's df, but adding more workers does not increase RAM consumption anymore!
    """
    # the origingal dataframe is df, store the columns/dtypes pairs
    df_dtypes_dict = dict(list(zip(df.columns, df.dtypes)))

    # declare a shared Array with data from df
    mparr = Array(ctypes.c_double, df.values.reshape(-1), lock=True)

    # create a new df based on the shared array
    df_shared = pd.DataFrame(np.frombuffer(mparr.get_obj()).reshape(df.shape), columns=df.columns).astype(df_dtypes_dict)  # type: ignore[call-overload]  # multiprocessing.Array's ctypes buffer is a standard np.frombuffer input; numpy's stub overloads don't cover it

    return df_shared


def get_non_stale_columns(X: pd.DataFrame) -> list:
    """
    Returns the names of ``X`` columns whose values DO change (i.e. every column except the
    stale/constant ones). Does NOT mutate ``X`` -- unlike its sibling ``remove_constant_columns``,
    this function only rebinds its local parameter, so the caller's DataFrame is untouched;
    the caller must apply the returned column list itself (``X = X[get_non_stale_columns(X)]``).
    """
    if len(X) == 0:
        return X.columns.tolist()  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime

    # nunique(dropna=False) treats NaN as its own value, so an all-NaN column (nunique==1)
    # is correctly flagged stale -- unlike ``X != X.iloc[0]``, which is always True for NaN
    # vs NaN and so never flags an all-NaN column as stale.
    stale_columns = X.nunique(dropna=False) <= 1

    num_stale = stale_columns.sum()
    if num_stale > 0:
        logger.warning(f"Found {num_stale} stale columns: {','.join(stale_columns[stale_columns].index.values.tolist())}")
        X = X.loc[:, stale_columns[~stale_columns].index.values]
        all_features_names = X.columns.tolist()
        return all_features_names  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime
    return X.columns.tolist()  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime


def remove_stale_columns(X: pd.DataFrame) -> list:
    """Deprecated alias for :func:`get_non_stale_columns` -- kept for backward compatibility.

    Despite the name (mirroring ``remove_constant_columns``, which genuinely mutates its input
    in place), this does NOT remove anything from the caller's DataFrame; it only returns the
    list of non-stale column names. Prefer :func:`get_non_stale_columns`, whose name reflects
    the actual (non-mutating) contract.
    """
    return get_non_stale_columns(X)


def get_suspiciously_constant_columns(ref_df: pd.DataFrame) -> list:
    """
    Return names of columns in ``ref_df`` that have at most one distinct value (constant or all-NaN).
    Falls back to a per-column loop, skipping columns whose values raise TypeError (e.g. unhashable), if the vectorized ``nunique()`` call fails.
    """
    try:
        susp_columns = ref_df.columns[ref_df.nunique() <= 1].tolist()
    except Exception:
        susp_columns = []
        for col in ref_df.columns:
            try:
                if ref_df[col].nunique() <= 1:
                    susp_columns.append(col)
            except TypeError:  # noqa: PERF203 -- per-iteration fault isolation is intentional (skip this column, check the rest)
                # Skip the column if a TypeError (e.g. unhashable type) occurs.
                continue
    return susp_columns  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime


def remove_constant_columns(df: pd.DataFrame, verbose: bool = False, prewarm_size: int = 10_000) -> None:
    """
    Drop constant (single-unique-value) columns from ``df`` in place.

    For large dataframes (more rows than ``prewarm_size``), constancy is first screened on a ``prewarm_size``-row
    head sample for speed, then any false positives are rejected by checking the full column before dropping.
    """
    if len(df) <= prewarm_size:
        susp_columns = get_suspiciously_constant_columns(df)
    else:
        susp_columns = get_suspiciously_constant_columns(df.head(prewarm_size))
        cols_to_drop: set = set()
        for col in tqdmu(susp_columns, desc="cnst col", leave=False):
            if df[col].nunique() > 1:
                cols_to_drop.add(col)
        if cols_to_drop:
            susp_columns = [c for c in susp_columns if c not in cols_to_drop]

    if susp_columns:
        if verbose:
            if len(susp_columns) > 20:
                logger.warning(f"Removing {len(susp_columns):_} constant columns")
            else:
                logger.warning(f"Removing constant columns {susp_columns}")

        if len(susp_columns) > 20:
            df.drop(columns=susp_columns, inplace=True)
        else:
            for var in susp_columns:
                del df[var]

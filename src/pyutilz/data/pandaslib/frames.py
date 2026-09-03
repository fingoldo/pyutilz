"""Column/row/frame manipulation helpers (renaming, value showcasing, sharing,
constant/stale column removal, feature naming).

Split out of the historical flat ``pyutilz.data.pandaslib`` module; re-exported
from the package ``__init__`` to preserve the public import surface.
"""

import warnings

from ._common import (
    Any,
    Dict,
    Optional,
    Sequence,
    Tuple,
    Union,
    np,
    pd,
    pl,
    ctypes,
    Array,
    tqdmu,
    logger,
)

# PROJECT IDIOM for a re-export package's submodules (see also pyutilz/text/strings/_logproxy.py,
# which applies the same rule through a dedicated proxy object):
#   `import <parent> as _facade`   -- ALLOWED, and load-bearing.
#   `from <parent> import <name>`  -- FORBIDDEN at module top level.
# A re-export package's __init__ imports its submodules, so a submodule importing the parent back is
# a genuine cycle. Plain `import x` binds the PARTIALLY-INITIALISED sys.modules entry and defers every
# attribute lookup to call time, so it survives; `from x import name` needs the name to exist at import
# time and raises "cannot import name ... (most likely due to a circular import)". Deferring the lookup
# is also what makes the name patchable: a test setting `pyutilz.data.pandaslib.HAS_IPYTHON` is seen here,
# where a from-import would have snapshotted the original value. tests/test_meta/test_reexport_package_idiom.py
# enforces this mechanically.
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


def prefixize_columns(
    df: pd.DataFrame, prefix: str, special_prefixes: Optional[dict] = None, sep="_", exclusions: Optional[Sequence] = None, inplace: bool = True
) -> "tuple[pd.DataFrame, dict]":
    """
    Prefix every column of a pandas dataframe (except clearly formulated exclusions) with some arbitrary prefix string - to identify variable's source.

    Returns ``(df, columns)`` -- the (possibly in-place-mutated) dataframe and the
    ``{old_col: new_col}`` rename mapping -- regardless of ``inplace``, so the return shape
    doesn't silently flip between a ``dict`` and a ``DataFrame`` depending on that flag.
    """
    if special_prefixes is None:
        special_prefixes = {}
    if exclusions is None:
        exclusions = ()
    # Build column mapping once instead of duplicating 3 times
    columns = {col: special_prefixes.get(col, prefix) + sep + col if col not in exclusions else col for col in df.columns}
    if inplace:
        df.rename(columns=columns, inplace=True)
    else:
        df = df.rename(columns=columns, inplace=False)
    return df, columns


def _rare_and_uninformative_from_lists(vals: list, counts: list, max_unique_percent: float) -> "Tuple[list, Optional[float]]":
    """``(rare values, uninformative fraction)`` for one column, from parallel value/count lists.

    A value is rare when its share of the column's OWN value_counts total is at most
    ``max_unique_percent``. The denominator is that per-column total, not the frame height: with
    ``dropna=True`` the counts exclude nulls, so dividing by the full height mixes two populations
    and reports the same "uninformative" fraction for both dropna modes.

    The second element is the fraction of non-dominant rows when at most ONE distinct value
    survives the rare filter (the column carries almost no information), else ``None``.
    """
    col_total = sum(counts)
    rare_threshold = max_unique_percent * col_total
    col_rare = [v for v, c in zip(vals, counts) if c <= rare_threshold]
    if not col_rare:
        return col_rare, None
    non_rare_unique = sum(1 for c in counts if c > rare_threshold)
    if non_rare_unique > 1:
        return col_rare, None
    non_rare_count = sum(c for c in counts if c > rare_threshold)
    return col_rare, (1 - non_rare_count / col_total if col_total > 0 else 0.0)


def _rare_and_uninformative_from_value_counts(stats, max_unique_percent: float) -> "Tuple[list, Optional[float]]":
    """``_rare_and_uninformative_from_lists`` for a pandas ``value_counts()`` Series.

    Kept as a separate entry point rather than converting the Series to python lists: the pandas
    branch never needs the lists for anything else, so the vectorised mask is strictly cheaper.
    ``stats`` is the FULL value_counts result -- the ``.head(max_vars)`` above only affects what
    gets printed and never truncates it.
    """
    col_total = int(stats.sum())
    rare_threshold = max_unique_percent * col_total
    col_rare = stats[stats <= rare_threshold].index.tolist()
    if not col_rare:
        return col_rare, None
    non_rare = stats[stats > rare_threshold]
    if len(non_rare) > 1:
        return col_rare, None
    non_rare_count = int(non_rare.sum()) if len(non_rare) == 1 else 0
    return col_rare, (1 - non_rare_count / col_total if col_total > 0 else 0.0)


def _showcase_polars_columns(
    df: "pl.DataFrame",
    target_cols: list,
    dropna: bool,
    use_markdown: bool,
    should_print: bool,
    max_vars: "Optional[int]",
    max_cat_uniq_qty: int,
    max_unique_percent: float,
) -> tuple:
    """The polars half of ``showcase_df_columns``: print each column's value counts and collect
    its rare / uninformative verdict. Returns ``(rare_categories, uninformative_features)``."""
    rare_categories: Dict[Any, Any] = {}
    uninformative_features: Dict[Any, Any] = {}
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

    for var, vc in zip(target_cols, vc_results):
        dtype = df.schema[var]
        if use_markdown and _facade.HAS_IPYTHON:
            _facade.display(_facade.Markdown(f"**{var}** {dtype}"))
        if should_print:
            print(f"{var.upper()} {dtype}")  # noqa: T201 -- use_print is an explicit stdout-display contract, doctest-verified above

        # Rare/uninformative analysis reads exactly the same two columns the display block
        # does, so convert each Arrow column to python ONCE and share it. Both blocks used to
        # call ``.to_list()`` independently -- two full Arrow-to-python conversions of the same
        # unchanged frame per column. `vc` is grouped under the same dropna treatment as
        # n_unique would be (drop_nulls() before group_by when dropna=True, null counted as
        # its own group otherwise), so vc.height already IS n_unique -- no second full-column
        # scan needed.
        n_unique = vc.height
        wants_display = n_unique > 0 and not (max_vars is not None and max_vars == 0)
        wants_rare = 0 < n_unique <= max_cat_uniq_qty
        if wants_display or wants_rare:
            vals = vc.get_column(var).to_list()
            counts = vc.get_column("count").to_list()

        if should_print:
            if max_vars is not None and max_vars == 0:
                print("")  # noqa: T201 -- use_print is an explicit stdout-display contract, doctest-verified above
            elif vc.height == 0:
                stats = pd.Series([], name="count", dtype="int64")
                stats.index.name = var
                print(stats)  # noqa: T201 -- use_print is an explicit stdout-display contract, doctest-verified above
            else:
                stats = pd.Series(counts, index=vals, name="count")
                stats.index.name = var
                if max_vars is not None and max_vars > 0:
                    print(stats.head(max_vars))  # noqa: T201 -- use_print is an explicit stdout-display contract, doctest-verified above
                else:
                    print(stats)  # noqa: T201 -- use_print is an explicit stdout-display contract, doctest-verified above

        # bench-attempt-rejected (2026-09-02): doing the rare filter in polars
        # (vc.filter(pl.col("count") <= threshold)) instead of over the python lists. Rejected on
        # reading, not timing: the display block above needs the FULL lists whenever max_vars != 0,
        # so a polars filter would ADD a pass over the frame rather than remove one. The measured
        # win here came from sharing the single conversion (16.9/17.7/13.4 ms of a 250/315/265 ms
        # call, 40 cols x 5000 uniques), which the shared vals/counts above already banks.
        if wants_rare:
            col_rare, uninformative = _rare_and_uninformative_from_lists(vals, counts, max_unique_percent)
            if col_rare:
                rare_categories[var] = col_rare
                if uninformative is not None:
                    uninformative_features[var] = uninformative
    return rare_categories, uninformative_features


def _showcase_pandas_columns(
    df: "pd.DataFrame",
    target_cols: list,
    dropna: bool,
    use_markdown: bool,
    should_print: bool,
    max_vars: "Optional[int]",
    max_cat_uniq_qty: int,
    max_unique_percent: float,
) -> tuple:
    """The pandas half of ``showcase_df_columns``: print each column's value counts and collect
    its rare / uninformative verdict. Returns ``(rare_categories, uninformative_features)``."""
    rare_categories: Dict[Any, Any] = {}
    uninformative_features: Dict[Any, Any] = {}
    for var in target_cols:
        if use_markdown and _facade.HAS_IPYTHON:
            _facade.display(_facade.Markdown(f"**{var}** {df[var].dtype}"))
        if should_print:
            print(f"{var.upper()} {df[var].dtype}")  # noqa: T201 -- use_print is an explicit stdout-display contract, doctest-verified above
        stats = df[var].value_counts(dropna=dropna)
        if max_vars is not None:
            assert max_vars >= 0  # nosec B101 - internal invariant on a display-row-count parameter (only used to slice a printed head()), not a security boundary
        if should_print:
            if max_vars is not None:
                if max_vars > 0:
                    print(stats.head(max_vars))  # noqa: T201 -- use_print is an explicit stdout-display contract, doctest-verified above
                else:
                    print("")  # noqa: T201 -- use_print is an explicit stdout-display contract, doctest-verified above
            else:
                print(stats)  # noqa: T201 -- use_print is an explicit stdout-display contract, doctest-verified above

        # Rare/uninformative analysis -- gate respects `dropna` (matches the value_counts
        # computed above), so a column that's within max_cat_uniq_qty distinct non-null
        # values isn't silently skipped just because it also has nulls the caller asked to ignore.
        # `stats` already IS the full value_counts(dropna=dropna) result (`.head(max_vars)`
        # above only affects what gets printed, never truncates `stats` itself), so both
        # `len(stats)` and re-running value_counts() are redundant full-column rescans of a
        # value already on hand.
        n_unique = len(stats)
        if n_unique <= max_cat_uniq_qty and n_unique > 0:
            col_rare, uninformative = _rare_and_uninformative_from_value_counts(stats, max_unique_percent)
            if col_rare:
                rare_categories[var] = col_rare
                if uninformative is not None:
                    uninformative_features[var] = uninformative
    return rare_categories, uninformative_features


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

    # Whether ANY of this function's stdout output happens. `use_print` used to gate only the one-line
    # dtype header, so a caller after nothing but the returned dicts (a feature-selection loop over
    # hundreds of columns) still got every value-count table -- and, with IPython present, got those
    # tables stripped of the headers that said which column each belonged to.
    should_print = use_print or not _facade.HAS_IPYTHON

    if cols is None or len(cols) == 0:
        cols = df.columns

    excluded_set = set(excluded_cols)
    target_cols = [c for c in cols if c not in excluded_set]

    rare_categories: Dict[Any, Any] = {}
    uninformative_features: Dict[Any, Any] = {}

    if not target_cols:
        return rare_categories, uninformative_features

    # Denominators are taken per column from the value_counts total actually in play, NOT from the
    # frame height: with dropna=True the counts exclude nulls, so dividing by the full height mixed
    # two populations and reported the same "uninformative" fraction for both dropna modes.

    if _is_polars:
        rare_categories, uninformative_features = _showcase_polars_columns(
            df, target_cols, dropna, use_markdown, should_print, max_vars, max_cat_uniq_qty, max_unique_percent
        )
    else:
        assert isinstance(df, pd.DataFrame)  # nosec B101 - the polars branch above is the only other DataFrame flavour this function accepts
        rare_categories, uninformative_features = _showcase_pandas_columns(
            df, target_cols, dropna, use_markdown, should_print, max_vars, max_cat_uniq_qty, max_unique_percent
        )

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


_SHARE_DATAFRAME_INT_CTYPES: Dict[str, Any] = {
    "int8": ctypes.c_int8,
    "int16": ctypes.c_int16,
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
    "uint8": ctypes.c_uint8,
    "uint16": ctypes.c_uint16,
    "uint32": ctypes.c_uint32,
    "uint64": ctypes.c_uint64,
}


def share_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a cloned dataframe from create numpy mem views that can be shared with multiple worker processes as a global variable.
    Should not contain datetime dtype! or won't be able to fit the double dtype.
    Ram usage grows from 7x (while cloning) to 3x (while using) of the original's df, but adding more workers does not increase RAM consumption anymore!

    Integer columns (int8/16/32/64, uint8/16/32/64) are routed through their own dtype-matched
    ctypes buffer rather than the shared float64 buffer below -- float64's 52-bit mantissa can't
    represent every int64 value above 2**53 exactly, so forcing integers through it silently
    corrupts large IDs/timestamps/hashes. Every other dtype (float/bool/object/...) still shares
    one float64 buffer, unchanged from before.
    """
    # the origingal dataframe is df, store the columns/dtypes pairs
    df_dtypes_dict = dict(list(zip(df.columns, df.dtypes)))

    int_cols = [c for c in df.columns if str(df[c].dtype) in _SHARE_DATAFRAME_INT_CTYPES]
    other_cols = [c for c in df.columns if c not in int_cols]

    pieces: Dict[Any, Any] = {}

    if other_cols:
        sub = df[other_cols]
        mparr = Array(ctypes.c_double, sub.values.reshape(-1), lock=True)
        sub_shared = pd.DataFrame(np.frombuffer(mparr.get_obj()).reshape(sub.shape), columns=other_cols)  # type: ignore[call-overload]  # multiprocessing.Array's ctypes buffer is a standard np.frombuffer input; numpy's stub overloads don't cover it
        for c in other_cols:
            pieces[c] = sub_shared[c]

    # Group integer columns by their EXACT dtype (not just "is it an int") -- mixing e.g. int32
    # and int64 columns in one sub-frame would make pandas' `.values` upcast them all to int64
    # before they ever reach the ctypes buffer, silently reintroducing the same class of bug.
    dtype_groups: Dict[str, list] = {}
    for c in int_cols:
        dtype_groups.setdefault(str(df[c].dtype), []).append(c)

    for dtype_name, cols in dtype_groups.items():
        ctype = _SHARE_DATAFRAME_INT_CTYPES[dtype_name]
        sub = df[cols]
        mparr = Array(ctype, sub.values.reshape(-1), lock=True)
        arr = np.frombuffer(mparr.get_obj(), dtype=np.dtype(dtype_name)).reshape(sub.shape)
        sub_shared = pd.DataFrame(arr, columns=cols)
        for c in cols:
            pieces[c] = sub_shared[c]

    # create a new df based on the shared arrays, restoring the caller's original column order
    df_shared = pd.DataFrame({c: pieces[c] for c in df.columns}).astype(df_dtypes_dict)
    # The pieces are built positionally from .values, so the frame would otherwise come back with a
    # fresh RangeIndex: a worker doing shared_df.loc[key] would raise KeyError, and arithmetic/joins
    # against a frame still carrying the original labels would align on mismatched labels (all-NaN)
    # instead of failing. Assigning the index doesn't copy the shared buffers.
    df_shared.index = df.index

    return df_shared


def get_non_stale_columns(df: pd.DataFrame) -> list:
    """
    Returns the names of ``df`` columns whose values DO change (i.e. every column except the
    stale/constant ones). Does NOT mutate ``df`` -- unlike its sibling ``remove_constant_columns``,
    this function only rebinds its local parameter, so the caller's DataFrame is untouched;
    the caller must apply the returned column list itself (``df = df[get_non_stale_columns(df)]``).

    A frame of 0 or 1 rows carries no evidence of staleness, so ALL columns are returned for both.
    (Previously only the 0-row case was special-cased, and a 1-row frame -- constant by
    construction -- came back with an empty list, e.g. for a single-row inference frame.)
    """
    if len(df) <= 1:
        return df.columns.tolist()  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime

    # nunique(dropna=False) treats NaN as its own value, so an all-NaN column (nunique==1)
    # is correctly flagged stale -- unlike ``df != df.iloc[0]``, which is always True for NaN
    # vs NaN and so never flags an all-NaN column as stale.
    stale_columns = df.nunique(dropna=False) <= 1

    num_stale = stale_columns.sum()
    if num_stale > 0:
        logger.warning(f"Found {num_stale} stale columns: {','.join(stale_columns[stale_columns].index.values.tolist())}")
        df = df.loc[:, stale_columns[~stale_columns].index.values]
        all_features_names = df.columns.tolist()
        return all_features_names  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime
    return df.columns.tolist()  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime


def remove_stale_columns(df: pd.DataFrame) -> list:
    """Deprecated alias for :func:`get_non_stale_columns` -- kept for backward compatibility.

    Despite the name (mirroring ``remove_constant_columns``, which genuinely mutates its input
    in place), this does NOT remove anything from the caller's DataFrame; it only returns the
    list of non-stale column names. Prefer :func:`get_non_stale_columns`, whose name reflects
    the actual (non-mutating) contract.
    """
    warnings.warn(
        "remove_stale_columns is deprecated and will be removed in a future release; use get_non_stale_columns instead (same non-mutating behavior, name reflects it).",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_non_stale_columns(df)


def get_suspiciously_constant_columns(df: pd.DataFrame) -> list:
    """
    Return names of columns in ``df`` that have at most one distinct value (constant or all-NaN).
    Falls back to a per-column loop, skipping columns whose values raise TypeError (e.g. unhashable), if the vectorized ``nunique()`` call fails.

    Counts NaN as a value of its own (``dropna=False``), matching the sibling
    ``get_non_stale_columns`` and this docstring: with pandas' default ``dropna=True`` a column
    holding one real observation plus nulls (a rare-event indicator) looked constant here while
    the sibling kept it, so the two screens disagreed on the very same frame.
    """
    try:
        susp_columns = df.columns[df.nunique(dropna=False) <= 1].tolist()
    except Exception:
        susp_columns = []
        for col in df.columns:
            try:
                if df[col].nunique(dropna=False) <= 1:
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
            if df[col].nunique(dropna=False) > 1:  # same null semantics as the head-sample screen above
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

"""Dtype classification, dtype optimization and memory-consumption helpers.

Split out of the historical flat ``pyutilz.data.pandaslib`` module; re-exported
from the package ``__init__`` to preserve the public import surface.
"""

import re

from ._common import (
    Any,
    Dict,
    Union,
    Optional,
    Sequence,
    np,
    pd,
    pl,
    defaultdict,
    tqdmu,
    logger,
)

# Formatting that a number would not reproduce: a leading zero before another digit ("01234", "-0501"),
# leading/trailing whitespace, or an explicit "+" sign. Anchored alternatives, so it is a cheap scan.
_IDENTIFIER_LIKE_TEXT_RE = re.compile(r"^\s|\s$|^\+|^-?0\d")


def _has_identifier_like_text(series: pd.Series) -> bool:
    """True if any non-null value of ``series`` carries text an equivalent number would not reproduce.

    Such a column is an identifier that happens to be spelled with digits (zip code, zero-padded
    account/customer id, phone number, EAN/ISBN), not a quantity: casting it to an integer is a
    lossy rewrite of the data, not the storage optimization :func:`optimize_dtypes` promises.

    Only ever consulted for a column whose numeric cast already succeeded, so the scan stays off the
    hot path for ordinary free-text columns (which the cast rejects on their first non-numeric value).
    """
    return any(_IDENTIFIER_LIKE_TEXT_RE.search(str(value)) is not None for value in series.dropna())


def set_df_columns_types(df: pd.DataFrame, types_dict: dict) -> None:
    """Cast ``df`` columns to the dtypes given in ``types_dict`` (mapping dtype -> list of column names), in place.

    Columns listed under a dtype that are not present in ``df`` are silently skipped.
    """
    df_columns = set(df.columns)
    for the_type in types_dict.keys():
        for column in types_dict[the_type]:
            if column in df_columns:
                df[column] = df[column].astype(the_type)


def get_categorical_columns_indices(df: pd.DataFrame) -> tuple:
    """Split ``df`` columns into categorical vs non-categorical, by positional index.

    Returns
    -------
    tuple
        ``(non_categorical_features_indices, categorical_features_indices, unique_categorical_values)``
        where the first two are lists of column positions and the third maps each
        categorical column's name to the list of its unique category values.
    """
    categorical_features_indices = []
    non_categorical_features_indices = []
    unique_categorical_values = dict()
    for i, (col, thetype) in enumerate(df.dtypes.items()):
        if isinstance(thetype, pd.CategoricalDtype):
            # print(i,col,thetype,type(thetype))
            categorical_features_indices.append(i)
            unique_categorical_values[col] = list(df[col].cat.categories.values)
        else:
            non_categorical_features_indices.append(i)
    return non_categorical_features_indices, categorical_features_indices, unique_categorical_values


def get_columns_of_type(df: pd.DataFrame, type_names: Sequence) -> list:
    """Return names of ``df`` columns whose dtype string contains any of ``type_names`` as a substring."""
    res = []
    for col, type_name in df.dtypes.to_dict().items():
        # str(type_name) is loop-invariant across type_names; hoisting it avoids recomputing the dtype repr once per probed type.
        type_name_str = str(type_name)
        if any(the_type in type_name_str for the_type in type_names):
            res.append(col)
    return res


def _promote_object_columns(
    df: pd.DataFrame,
    old_dtypes: dict,
    new_dtypes: dict,
    int_fields: list,
    float_fields: list,
    skip_columns: Sequence,
    max_categories: int,
    inplace: bool,
    verbose: bool,
) -> None:
    """Retype every object/string column of ``df`` in place: int64, else float64, else category.

    Mutates ``old_dtypes``/``int_fields``/``float_fields`` for a numeric promotion (the size
    reduction pass downstream reads them back) and ``new_dtypes`` for a category one. Columns with
    more than ``max_categories`` distinct values, and anything the casts reject, are left alone.
    """
    for col, the_type in old_dtypes.items():
        if "object" in the_type or "str" in the_type or "string" in the_type:
            if col in skip_columns:
                continue

            # first try to int64, then to float64, then to category
            try:
                candidate = df[col].astype(np.int64)
                # np.int64 cast silently truncates fractional floats (3.5 -> 3) instead of raising,
                # so verify the round-trip is exactly equal to the original values before accepting it.
                if not (candidate.astype(np.float64) == df[col].astype(np.float64)).all():
                    raise ValueError(f"Column {col} contains fractional values; cannot be int64")
                # That round-trip compares VALUES, so textual detail a number reproduces identically
                # (a leading zero above all) passes it and is then thrown away. Such a column is an
                # identifier spelled with digits, not a quantity; leave it to the category branch.
                if _has_identifier_like_text(df[col]):
                    raise ValueError(f"Column {col} holds identifier-like text; casting it to a number would lose formatting")
                df[col] = candidate
                old_dtypes[col] = "int64"
                int_fields.append(col)
            except Exception:
                try:
                    as_float = df[col].astype(np.float64)
                    if _has_identifier_like_text(df[col]):
                        raise ValueError(f"Column {col} holds identifier-like text; casting it to a number would lose formatting")
                    df[col] = as_float
                    old_dtypes[col] = "float64"
                    float_fields.append(col)
                except Exception:
                    try:
                        n = df[col].nunique()
                        if n <= max_categories:
                            if verbose:
                                logger.info("%s %s->category", col, the_type)

                            new_dtypes[col] = "category"
                            if inplace:
                                df[col] = df[col].astype(new_dtypes[col])

                    except Exception as e3:
                        if verbose:
                            logger.warning(f"Could not convert to category column {col}: {e3}")
                        pass  # to avoid stumbling on lists like [1]


def optimize_dtypes(
    df: pd.DataFrame,
    max_categories: Optional[int] = 100,
    reduce_size: bool = True,
    float_to_int: bool = True,
    float_to_float: bool = True,
    skip_columns: Sequence = (),
    use_uint: bool = True,  # might want to turn this off when using sqlalchemy (Unsigned 64 bit integer datatype is not supported)
    verbose: bool = False,
    inplace: bool = True,
    skip_halffloat: bool = True,
    ensure_float64_precision: bool = True,
    exact_float_roundtrip: bool = False,
) -> pd.DataFrame:
    """Compress datatypes in a pandas dataframe to save space while keeping precision.
    Optionally attempts converting floats to ints where feasible.
    Optionally converts object fields with nuniques less than max_categories to categorical.

    With ``inplace=False`` the caller's frame is never touched: the object->int64/float64 probing
    below has to materialize its candidate columns (the size-reduction pass reads them back), so
    it runs on a copy rather than writing through to the original as it used to.

    ``ensure_float64_precision`` compares mantissas rounded to float64's 15 significant decimal
    digits, i.e. it protects decimal-repr fidelity, NOT bit-exactness: two float64 values differing
    only in their last few bits look identical to it and can be collapsed by a float32 downcast.
    Pass ``exact_float_roundtrip=True`` to additionally require a bit-exact float32->float64
    round-trip (much stricter -- it also rejects ordinary decimals like 0.1, which the decimal
    heuristic deliberately accepts).
    """

    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # Inits
    # -----------------------------------------------------------------------------------------------------------------------------------------------------

    if not inplace:
        df = df.copy()

    old_dtypes: Dict[Any, str] = {}
    new_dtypes: Dict[Any, str] = {}
    int_fields: list = []
    float_fields: list = []
    for field, the_type in df.dtypes.to_dict().items():
        if field not in skip_columns:
            old_dtypes[field] = the_type.name
            if "int" in the_type.name:
                int_fields.append(field)
            elif "float" in the_type.name:
                float_fields.append(field)

    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # Every object var with too few categories must become a Category
    # -----------------------------------------------------------------------------------------------------------------------------------------------------

    if max_categories is not None:
        _promote_object_columns(df, old_dtypes, new_dtypes, int_fields, float_fields, skip_columns, max_categories, inplace, verbose)

    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # Finds minimal size suitable to hold each variable of interest without loss of coverage
    # -----------------------------------------------------------------------------------------------------------------------------------------------------

    if reduce_size:
        mantissas = {}
        uint_fields = []
        if use_uint:
            conversions = [
                (int_fields, "uint"),
                (int_fields, "int"),
            ]
        else:
            conversions = [
                (int_fields, "int"),
            ]
        if float_to_int:

            # -----------------------------------------------------------------------------------------------------------------------------------------------------
            # Checks for each float if it has no fractional digits and NaNs, and, therefore, can be made an int
            # ----------------------------------------------------------------------------------------------------------------------------------------------------

            possibly_integer = []
            for col in tqdmu(float_fields, desc="checking float2int", leave=False):
                if not df[col].isna().any():  # NAs can't be converted to int
                    fract_part, _ = np.modf(df[col])
                    if (fract_part == 0.0).all():
                        possibly_integer.append(col)
            if possibly_integer:
                if use_uint:
                    conversions.append((possibly_integer, "uint"))
                conversions.append((possibly_integer, "int"))
        if float_to_float:
            conversions.append((float_fields, "float"))
        for fields, type_name in tqdmu(conversions, desc="size reduction", leave=False):
            fields = [el for el in fields if el not in uint_fields]
            if len(fields) > 0:
                # Compute min and max in single pass for efficiency
                stats = df[fields].agg(["min", "max"])
                min_vals = stats.loc["min"]
                max_vals = stats.loc["max"]

                topvals: Any
                if type_name in ("int", "uint"):
                    powers = [8, 16, 32, 64]
                    topvals = [np.iinfo(type_name + str(p)) for p in powers]
                elif type_name == "float":
                    powers = [32, 64] if skip_halffloat else [16, 32, 64]  # no float8
                    topvals = [np.finfo(type_name + str(p)) for p in powers]

                min_max = pd.concat([min_vals, max_vals], axis=1)
                min_max.columns = ["min", "max"]

                for r in min_max.itertuples():
                    col = r.Index
                    cur_power = int(old_dtypes[col].replace("uint", "").replace("int", "").replace("float", ""))
                    for j, p in enumerate(powers):
                        if p >= cur_power:
                            if not (col in float_fields and type_name != "float"):
                                break
                        if r.max <= topvals[j].max and r.min >= topvals[j].min:
                            if ensure_float64_precision and type_name == "float":
                                # need to ensure we are not losing precision! np.array([2.205001270000e09]).astype(np.float32) must not pass here, for example.
                                if col not in mantissas:
                                    values = df[col].values
                                    # invalid="ignore" also suppresses the "invalid value encountered in
                                    # divide" RuntimeWarning from 0.0/0.0 (exact 0.0 values: log10(0)==-inf,
                                    # 10**-inf==0.0) -- a different warning category than "divide", which
                                    # errstate(divide=...) alone doesn't cover. The resulting NaN is already
                                    # masked out downstream via np.ma.array, so this only silences noise.
                                    with np.errstate(divide="ignore", invalid="ignore"):
                                        _, int_part = np.modf(np.log10(np.abs(values)))
                                        mantissa = np.round(values / 10**int_part, np.finfo(old_dtypes[col]).precision - 1)

                                    mantissas[col] = mantissa
                                else:
                                    mantissa = mantissas[col]

                                fract_part, _ = np.modf(mantissa * 10 ** (np.finfo("float" + str(p)).precision + 1))
                                fract_part, _ = np.modf(np.round(fract_part, np.finfo("float" + str(p)).precision - 1))
                                if (np.ma.array(fract_part, mask=np.isnan(fract_part)) != 0).any():  # masking so that NaNs do not count
                                    if verbose:
                                        logger.info("Column %s can't be converted to float%s due to precision loss.", col, p)
                                    break
                                if exact_float_roundtrip:
                                    # The mantissa heuristic above rounds to 15 significant decimals, so
                                    # differences living in float64's last bits are erased BEFORE it looks;
                                    # near-tied values used as ranking keys would silently become exact ties.
                                    orig = df[col].values
                                    if not np.array_equal(orig.astype("float" + str(p)).astype(np.float64), orig.astype(np.float64), equal_nan=True):
                                        if verbose:
                                            logger.info("Column %s can't be converted to float%s: float64 round-trip is not exact.", col, p)
                                        break
                            if type_name in ("uint", "int"):
                                uint_fields.append(col)  # successfully converted, so won't need to consider anymore
                            if verbose:
                                logger.info("%s [%s]->[%s%s]", col, old_dtypes[col], type_name, p)
                            new_dtypes[col] = type_name + str(p)
                            if inplace:
                                df[col] = df[col].astype(new_dtypes[col])
                            break

    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # Actual converting & reporting.
    # -----------------------------------------------------------------------------------------------------------------------------------------------------

    if len(new_dtypes) > 0 and not inplace:
        if verbose:
            logger.info("Going to use the following new dtypes: %s", new_dtypes)
        return df.astype(new_dtypes)
    else:
        return df


def group_columns_by_dtype(df: pd.DataFrame) -> dict:
    """Group ``df`` column names by their dtype name, returning a dict mapping dtype name -> set of column names."""
    groups = defaultdict(set)
    for var_name, var_type in df.dtypes.items():
        groups[var_type.name].add(var_name)

    return groups


def classify_column_types(df: Optional[pd.DataFrame] = None, col: Optional[str] = None, dtype: Any = None) -> tuple:
    """Return bunch of booleans: whether certain column is of particular dtype."""
    if dtype is None:
        assert (df is not None) and (col)  # nosec B101 - internal API-misuse guard: caller must supply either an explicit dtype or both df+col; not a security boundary
        # df[col].dtype reads the single column's dtype directly; df.dtypes[col] rebuilds the
        # whole-frame dtypes Series on every call, which is O(ncols) per call (O(ncols**2) per
        # per-column scan) and dominates the cost on wide frames.
        dtype = df[col].dtype
    type_name = dtype.name
    col_is_boolean = "bool" in type_name
    # pandas >=3.0 / future.infer_string reports string columns as 'str'/'string' (not 'object'); treat them as object-like so they are not misclassified as numeric downstream.
    col_is_object = "object" in type_name or "str" in type_name
    col_is_datetime = "datetime" in type_name
    col_is_categorical = "category" in type_name
    # Positive test, not "everything that isn't bool/object/datetime/category": by exclusion,
    # period[D], interval, timedelta64 and Sparse all fell through to "numeric" and a caller
    # routing them into .mean()/corr()/a scaler got a pandas TypeError (or an unrequested densification).
    col_is_numeric = bool(pd.api.types.is_numeric_dtype(dtype)) and not (col_is_boolean or col_is_object or col_is_datetime or col_is_categorical)

    return col_is_boolean, col_is_object, col_is_datetime, col_is_categorical, col_is_numeric


def get_df_memory_consumption(df, max_cols: int = 0, deep: bool = True) -> float:
    """
    Returns RAM occupied by a pandas or polars dataframe in bytes.

    Works for:
      - pandas.DataFrame: via df.memory_usage(deep=deep).sum()
      - polars.DataFrame: via estimated_size()

    Parameters
    ----------
    df : pandas.DataFrame | polars.DataFrame
        DataFrame to measure.
    max_cols : int, optional
        Deprecated parameter, kept for backward compatibility (ignored).
    deep : bool, default True
        pandas-only. Default True uses ``df.memory_usage(deep=True)`` --
        byte-precise accounting that recursively sizes every element of
        object columns. On frames with million-unique strings this is
        O(rows * avg_str_len) and can take minutes; callers using this
        only for coarse heuristics (GPU-RAM fit checks, capacity
        planning) should pass ``deep=False`` explicitly -- that yields
        pointer-size accounting (8 B per object-column cell), O(cols),
        milliseconds. The polars branch ignores this flag --
        ``.estimated_size()`` is already O(cols).

    Returns
    -------
    float
        Memory consumption in bytes.
    """
    if isinstance(df, pl.DataFrame):
        # polars provides direct method
        return float(df.estimated_size())

    elif isinstance(df, pd.DataFrame):
        # Use direct API instead of text parsing for better performance and reliability
        return float(df.memory_usage(deep=deep).sum())

    else:
        raise TypeError(f"Unsupported dataframe type: {type(df)}")


def ensure_dataframe_float32_convertability(
    df: Union[pd.DataFrame, pl.DataFrame],
    verbose: bool = False,
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Ensures numeric columns are convertible to float32 for compatibility with LightGBM and
    rational memory usage.

    LightGBM uses np.result_type(*df_dtypes) when inferring array dtype from Pandas input,
    which often upcasts int32/int64/float64 to float64. To prevent this and save memory,
    convert numeric columns to float32 beforehand.

    Supports both Pandas (NumPy or PyArrow backend) and Polars DataFrames. Always returns a NEW
    object and never mutates the input in place, for either backend -- the caller must capture
    the return value regardless of which backend it passed in.
    """

    if isinstance(df, pl.DataFrame):
        # Convert integer and float64-like types to float32
        # Imported here rather than at module scope: pandaslib must stay usable without paying
        # polarslib's import-time setup, and this is the only polars-specific branch in the file.
        from pyutilz.data.polarslib import polars_castable_int_dtypes

        df = df.with_columns(pl.col([*polars_castable_int_dtypes(), pl.Float64]).cast(pl.Float32))

    elif isinstance(df, pd.DataFrame):
        df = df.copy()
        arrow_backed = df.dtypes.apply(lambda dt: "pyarrow" in str(dt))

        # --- Regular (NumPy-backed) dtypes ---
        # Consolidate into single select_dtypes call for efficiency
        numeric_cols = df.select_dtypes(include=["uint32", "int32", "int64", "uint64", "float64"]).columns
        if len(numeric_cols) > 0:
            if verbose:
                logger.info("Converting %s numeric columns to float32", len(numeric_cols))
            df[numeric_cols] = df[numeric_cols].astype(np.float32)

        # --- PyArrow-backed dtypes ---
        if arrow_backed.any():
            from pandas.api.types import is_integer_dtype, is_float_dtype

            for col in df.columns[arrow_backed]:
                pa_dtype = df[col].dtype
                # Infer numeric Arrow types
                if is_integer_dtype(pa_dtype) or is_float_dtype(pa_dtype):
                    if verbose:
                        logger.info("Converting PyArrow column '%s' (%s) -> float32[pyarrow]", col, pa_dtype)
                    df[col] = df[col].astype("float32[pyarrow]")

    return df


def convert_float64_to_float32(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert float64 columns to float32.

    Note: Consider using ensure_dataframe_float32_convertability() instead,
    which handles more numeric types comprehensively.

    Returns a NEW frame and never mutates the input, matching that sibling's documented contract
    -- it used to downcast the caller's own columns in place while presenting a return-value API,
    so a frame kept as the float64 reference silently lost its float64 originals.
    """
    df = df.copy()
    float64_cols = df.select_dtypes(include=["float64"]).columns
    for col in float64_cols:
        df[col] = df[col].astype(np.float32)
    return df

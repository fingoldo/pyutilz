"""Dtype classification, dtype optimization and memory-consumption helpers.

Split out of the historical flat ``pyutilz.data.pandaslib`` module; re-exported
from the package ``__init__`` to preserve the public import surface.
"""

from ._common import (
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


def set_df_columns_types(df: object, types_dict: dict) -> None:
    df_columns = set(df.columns)
    for the_type in types_dict.keys():
        for column in types_dict[the_type]:
            if column in df_columns:
                df[column] = df[column].astype(the_type)


def get_categorical_columns_indices(ds: object) -> tuple:
    categorical_features_indices = []
    non_categorical_features_indices = []
    unique_categorical_values = dict()
    for i, (col, thetype) in enumerate(ds.dtypes.items()):
        if isinstance(thetype, pd.CategoricalDtype):
            # print(i,col,thetype,type(thetype))
            categorical_features_indices.append(i)
            unique_categorical_values[col] = list(ds[col].cat.categories.values)
        else:
            non_categorical_features_indices.append(i)
    return non_categorical_features_indices, categorical_features_indices, unique_categorical_values


def get_columns_of_type(df: object, type_names: Sequence) -> list:
    res = []
    for col, type_name in df.dtypes.to_dict().items():
        # str(type_name) is loop-invariant across type_names; hoisting it avoids recomputing the dtype repr once per probed type.
        type_name_str = str(type_name)
        for the_type in type_names:
            if the_type in type_name_str:
                res.append(col)
    return res


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
) -> pd.DataFrame:
    """Compress datatypes in a pandas dataframe to save space while keeping precision.
    Optionally attempts converting floats to ints where feasible.
    Optionally converts object fields with nuniques less than max_categories to categorical.
    """

    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # Inits
    # -----------------------------------------------------------------------------------------------------------------------------------------------------

    old_dtypes = {}
    new_dtypes = {}
    int_fields = []
    float_fields = []
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
        for col, the_type in old_dtypes.items():
            if "object" in the_type or "str" in the_type or "string" in the_type:
                if col in skip_columns:
                    continue

                # first try to int64, then to float64, then to category
                try:
                    df[col] = df[col].astype(np.int64)
                    old_dtypes[col] = "int64"
                    int_fields.append(col)
                except Exception:
                    try:
                        df[col] = df[col].astype(np.float64)
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
                                logger.warning(f"Could not convert to category column {col}: {str(e3)}")
                            pass  # to avoid stumbling on lists like [1]
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
                                    with np.errstate(divide="ignore"):
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
    groups = defaultdict(set)
    for var_name, var_type in df.dtypes.items():
        groups[var_type.name].add(var_name)

    return groups


def classify_column_types(df: pd.DataFrame = None, col: Optional[str] = None, dtype: object = None) -> tuple:
    """Return bunch of booleans: whether certain column is of particualr dtype."""
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
    col_is_numeric = not (col_is_boolean or col_is_object or col_is_datetime or col_is_categorical)

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
    verbose: int = 0,
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Ensures numeric columns are convertible to float32 for compatibility with LightGBM and
    rational memory usage.

    LightGBM uses np.result_type(*df_dtypes) when inferring array dtype from Pandas input,
    which often upcasts int32/int64/float64 to float64. To prevent this and save memory,
    convert numeric columns to float32 beforehand.

    Supports both Pandas (NumPy or PyArrow backend) and Polars DataFrames.
    """

    if isinstance(df, pl.DataFrame):
        # Convert integer and float64-like types to float32
        df = df.with_columns(pl.col([pl.UInt32, pl.Int32, pl.Int64, pl.UInt64, pl.Int128, pl.Float64]).cast(pl.Float32))

    elif isinstance(df, pd.DataFrame):
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
    """
    float64_cols = df.select_dtypes(include=["float64"]).columns
    for col in float64_cols:
        df[col] = df[col].astype(np.float32)
    return df

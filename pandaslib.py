# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from .pythonlib import ensure_installed

# ensure_installed("pandas numpy pyarrow")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import gc
import io
import os
import warnings
import numpy as np
import pandas as pd
import pyarrow as pa
from .strings import find_between
from collections import defaultdict
from pyarrow.dataset import dataset
from pyutilz.pythonlib import to_float

import ctypes
from multiprocessing import Array

from IPython.display import display, Markdown, Latex

from .system import tqdmu
from os.path import join, sep
import glob

from os.path import basename, dirname, splitext, join, exists, getsize
from pyutilz.system import ensure_dir_exists
from timeit import default_timer as timer
from itertools import chain
import tempfile
import shutil


def load_df(fpath: str, tail: int) -> pd.DataFrame:
    logger.info(f"Загружаем данные из файла {fpath}...")

    df = pd.read_pickle(fpath)
    if tail is not None:
        if tail > 0:
            logger.info(f"Ограничимся работой с последними {tail} откликами")
            df = df.tail(tail)
    return df


def set_df_columns_types(df: object, types_dict: dict) -> None:
    df_columns = set(df.columns)
    for the_type in types_dict.keys():
        for column in types_dict[the_type]:
            if column in df_columns:
                df[column] = df[column].astype(the_type)


def get_categorical_columns_indices(ds: object) -> tuple:
    i = 0
    categorical_features_indices = []
    non_categorical_features_indices = []
    unique_categorical_values = dict()
    for col, thetype in ds.dtypes.iteritems():
        if type(thetype) == pd.core.dtypes.dtypes.CategoricalDtype:
            # print(i,col,thetype,type(thetype))
            categorical_features_indices.append(i)
            unique_categorical_values[col] = list(ds[col].cat.categories.values)
        else:
            non_categorical_features_indices.append(i)
        i = i + 1
    return non_categorical_features_indices, categorical_features_indices, unique_categorical_values


def get_columns_of_type(df: object, type_names: Sequence) -> list:
    res = []
    for col, type_name in df.dtypes.to_dict().items():
        for the_type in type_names:
            if the_type in str(type_name):
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
            if "object" in the_type:
                if field in skip_columns:
                    continue

                # first try to int64, then to float64, then to category
                new_dtype = None
                try:
                    df[col] = df[col].astype(np.int64)
                    old_dtypes[col] = "int64"
                    int_fields.append(col)
                except Exception as e1:
                    try:
                        df[col] = df[col].astype(np.float64)
                        old_dtypes[col] = "float64"
                        float_fields.append(col)
                    except Exception as e2:
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
                if not (df[col].isna().any().any()):  # NAs can't be converted to int
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
                max_vals = df[fields].max()
                min_vals = df[fields].min()

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
            logger.info(f"Going to use the following new dtypes: {new_dtypes}")
        return df.astype(new_dtypes)
    else:
        return df


def nullify_standard_values(
    df: object, field: str, min_records: int = 300, persons_field: str = None, min_persons: int = 0, placeholder=np.nan, verbose: bool = False
):
    """
    Replaces (inplace) all standard responses with NaN to indicate it's not custom
    """
    tmp = df[field].value_counts(dropna=False)
    standard_values = tmp[tmp > min_records].index.values
    if persons_field:
        top_values = set()
        for val in standard_values:
            qty = df[df[field] == val][persons_field].nunique()
            if qty > min_persons:
                if verbose:
                    if len(top_values) == 0:
                        print(f"Field {field}")
                    print(f"\t: value {val} is not custom, as used by {qty} persons")
                top_values.add(val)

    else:
        top_values = standard_values
    df.loc[df[field].isin(top_values), field] = placeholder


def prefixize_columns(df: object, prefix: str, special_prefixes: dict = {}, sep="_", exclusions: Sequence = set(), inplace: bool = True):
    """
    Prefix every column of a pandas dataframe (except clearly formulated exclusions) with some arbitrary prefix string - to identify variable's source
    """
    columns = {col: special_prefixes.get(col, prefix) + sep + col if col not in exclusions else col for col in df.columns}
    if inplace:
        df.rename(columns={col: special_prefixes.get(col, prefix) + sep + col if col not in exclusions else col for col in df.columns}, inplace=True)
        return columns
    else:
        return df.rename(columns={col: special_prefixes.get(col, prefix) + sep + col if col not in exclusions else col for col in df.columns}, inplace=False)


def showcase_df_columns(
    df: object, cols: list = None, excluded_cols: list = [], max_vars: int = None, dropna: bool = False, use_markdown: bool = True, use_print: bool = True
):
    """
    Show distribution of values for each dataframe column
    """

    if cols is None or len(cols) == 0:
        cols = df.columns
    for var in cols:
        if var not in excluded_cols:
            if use_markdown:
                display(Markdown(f"**{var}** {df[var].dtype}"))
            if use_print:
                print(f"{var.upper()} {df[var].dtype}")
            stats = df[var].value_counts(dropna=dropna)
            if max_vars is not None:
                assert max_vars >= 0
                if max_vars > 0:
                    stats = stats.head(max_vars)
                else:
                    stats = ""
            print(stats)


class FeatureNamer:
    """
    Used to save RAM when building dataframes from dicts with big feature names, by renaming them to small consecutive numbers
    >>>fname=FeatureNamer();fname('abc');fname('abcd');fname('abc'),fname.rev(0)
    (0, 'abc')
    """

    def __init__(self, initial_values: Sequence = []):
        self.fnames_index = 0
        self.fnames = {}
        self.revfnames = {}
        for name in initial_values:
            self(name)

    def __call__(self, name: str):
        if name not in self.fnames:
            self.fnames[name] = self.fnames_index
            self.revfnames[self.fnames_index] = name
            self.fnames_index += 1
        return self.fnames[name]

    def rev(self, key: int):
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
    df_shared = pd.DataFrame(np.frombuffer(mparr.get_obj()).reshape(df.shape), columns=df.columns).astype(df_dtypes_dict)

    return df_shared


def remove_stale_columns(X: pd.DataFrame) -> list:
    """
    Removes columns with values that do not change
    """
    stale_columns = ~(X != X.iloc[0]).any()

    num_stale = stale_columns.sum()
    if num_stale > 0:
        logger.warning(f"Found {num_stale} stale columns: {','.join(stale_columns[stale_columns == True].index.values.tolist())}")
        X = X.loc[:, stale_columns[stale_columns == False].index.values]
        all_features_names = X.columns.tolist()
    return all_features_names


def concat_and_flush_df_list(
    lst: list, file_name: str, to_csv: bool = False, csv_cols: list = None, write_fcn: str = "to_pickle", write_extension: str = "pckl", set_index: str = None
) -> object:

    if len(lst) > 0:
        joined_df = pd.concat(lst, axis=0, ignore_index=True)
        lst.clear()
        del lst
        gc.collect()
        if to_csv:
            if cols is None:
                cols = joined_df.columns.values
                joined_df.to_csv(f"{file_name}.csv", mode="w", header=True)
            else:
                joined_df[cols].to_csv(f"{file_name}.csv", mode="a", header=False)
        else:
            if set_index:
                joined_df.set_index(set_index, inplace=True)
            getattr(joined_df, write_fcn)(f"{file_name}.{write_extension}")
            return joined_df


def read_stats_from_multiple_files(
    joint_file_name: str = "joint_features",
    folder: str = "features",
    max_files: int = 250,
    template: str = "*.pckl",
    exclude: str = None,
    read_fcn: str = "read_pickle",
    write_fcn: str = "to_pickle",
    write_extension: str = "pckl",
    delete_after: bool = False,
    sentinel_field: str = None,
    sentinel_fcn: object = None,
    set_index: str = None,
    optimize: bool = False,
    save_on_successful_optimization: bool = False,
    min_size_improvement_percent: float = 0.05,
    min_size_improvement: float = 5.0,
):

    lst = []
    fnames = []
    for i, filename in tqdmu(enumerate(glob.glob(join(folder, template)))):
        if exclude:
            if exclude in filename:
                continue

        fnames.append(filename)
        tmp_df = getattr(pd, read_fcn)(filename)

        old_size = tmp_df.memory_usage(index=True).sum() / 1024**3
        logger.info(f"Merging {filename} with {len(tmp_df):_} rows of size {old_size:.1f} Gb")

        if optimize:
            tmp_df = optimize_dtypes(tmp_df)
            gc.collect()
            new_size = tmp_df.memory_usage(index=True).sum() / 1024**3
            logger.info(f"After optimization, {filename} got size {new_size:.1f} Gb")

            if save_on_successful_optimization:
                if new_size <= old_size * (1 - min_size_improvement_percent) or old_size - new_size >= min_size_improvement:
                    logger.info(f"Re-saving file {filename} due to lower size")
                    getattr(tmp_df, write_fcn)(f"{'.'.join(filename.split('.')[:-1])}.{write_extension}")

        if sentinel_field:

            while sentinel_field in tmp_df:
                logger.warning(f"Sentinel field {sentinel_field} was already in the frame {filename}")
                sentinel_field += "1"

            fname_part = filename.split(sep)[-1]
            if sentinel_fcn:
                tmp_df[sentinel_field] = sentinel_fcn(fname_part)
            else:
                tmp_df[sentinel_field] = fname_part

        lst.append(tmp_df)
        if max_files is not None:
            if len(lst) >= max_files:
                break
        del tmp_df
    if len(lst) >= 0:
        try:
            res = concat_and_flush_df_list(lst, file_name=joint_file_name, write_fcn=write_fcn, write_extension=write_extension, set_index=set_index)
            logger.info(f"Final df size ({len(res):_} rows)")
            if delete_after:
                for i, filename in enumerate(fnames):
                    try:
                        os.remove(filename)
                    except:
                        pass
            return res
        except:
            pass


def group_columns_by_dtype(df: pd.DataFrame) -> dict:
    groups = defaultdict(set)
    for var_name, var_type in df.dtypes.iteritems():
        groups[var_type.name].add(var_name)

    return groups


def classify_column_types(df: pd.DataFrame = None, col: str = None, dtype: object = None) -> tuple:
    """Return bunch of booleans: whether certain column is of particualr dtype."""
    if dtype is None:
        assert (df is not None) and (col)
        dtype = df.dtypes[col]
    type_name = dtype.name
    col_is_boolean = "bool" in type_name
    col_is_object = "object" in type_name
    col_is_datetime = "datetime" in type_name
    col_is_categorical = "category" in type_name
    col_is_numeric = not (col_is_boolean or col_is_object or col_is_datetime or col_is_categorical)

    return col_is_boolean, col_is_object, col_is_datetime, col_is_categorical, col_is_numeric


def read_parquet_with_pyarrow(path: str, nrows: int) -> pd.DataFrame:

    if nrows:
        df = dataset(path).scanner().head(nrows).to_pandas()
    else:
        df = dataset(path).scanner().to_pandas()

    return df


def get_df_memory_consumption(df: pd.DataFrame, max_cols: int = 0) -> float:
    """Returns RAM occupied by a pandas dataframe in bytes.
    Example df.info() output:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 11546660 entries, 0 to 11546659
    Columns: 4 entries, basic>ticker to basic>ts_minute
    dtypes: category(1), int8(3)
    memory usage: 44.0 MB
    """
    mem_consumption = io.StringIO()
    df.info(memory_usage="deep", buf=mem_consumption, max_cols=max_cols)
    res = mem_consumption.getvalue()
    res = find_between(res, "memory usage: ", "\n")
    for symbol, size in [
        ("KB", 1e3),
        ("MB", 1e6),
        ("GB", 1e9),
        ("TB", 1e12),
        ("B", 1),
    ]:
        if res.endswith(symbol):
            res = to_float(res.strip(symbol).strip()) * size
            break
    return res


def get_suspiciously_constant_columns(ref_df: pd.DataFrame) -> list:
    try:
        susp_columns = ref_df.columns[ref_df.nunique() <= 1].tolist()
    except Exception as e:
        susp_columns = []
        for col in ref_df.columns:
            try:
                if ref_df[col].nunique() <= 1:
                    susp_columns.append(col)
            except TypeError:
                # Skip the column if a TypeError (e.g. unhashable type) occurs.
                continue
    return susp_columns


def remove_constant_columns(df: pd.DataFrame, verbose: bool = False, prewarm_size: int = 10_000) -> None:

    if len(df) <= prewarm_size:
        susp_columns = get_suspiciously_constant_columns(df)
    else:
        susp_columns = get_suspiciously_constant_columns(df.head(prewarm_size))
        for col in tqdmu(susp_columns.copy(), desc="cnst col", leave=False):
            if df[col].nunique() > 1:
                susp_columns.remove(col)

    if verbose and susp_columns:
        if len(susp_columns) > 20:
            logger.warning(f"Removing {len(susp_columns):_} constant columns")
            df.drop(columns=susp_columns, inplace=True)
        else:
            logger.warning(f"Removing constant columns {susp_columns}")

            for var in susp_columns:
                del df[var]


# ----------------------------------------------------------------------------------------------------------------------------
# Dataframe compression benchmarks
# ----------------------------------------------------------------------------------------------------------------------------


def measure_read_write_performance(df: pd.DataFrame, fname: str, read_method: str, read_params: dict, write_method: str, write_params: dict, nrepeats: int):
    read_times, write_times, read_sizes, write_sizes = [], [], [], []
    for _ in range(nrepeats):
        # write
        start_time = timer()
        getattr(df, write_method)(fname, **write_params)
        duration = timer() - start_time
        size = getsize(fname) / (1024**2)
        write_times.append(duration)
        write_sizes.append(size)

        # read
        gc.collect()
        start_time = timer()
        tmp = getattr(pd, read_method)(fname, **read_params)
        duration = timer() - start_time
        size = get_df_memory_consumption(tmp) / (1024**2)
        del tmp
        read_times.append(duration)
        read_sizes.append(size)
        gc.collect()
    return [np.array(arr) for arr in (read_times, write_times, read_sizes, write_sizes)]


def pack_benchmark_results(res, config, read_times, write_times, read_sizes, write_sizes):
    res.append([config, *list(chain(*[(np.mean(arr), np.std(arr)) for arr in (read_times, write_times, read_sizes, write_sizes)]))])


def benchmark_dataframe_parquet_compression(
    df: pd.DataFrame,
    temp_folder: str,
    nrepeats: int = 3,
    engines: tuple = ("fastparquet", "pyarrow"),
    max_compression_level: int = 22,
    skip_configs: tuple = ("parquet-fastparquet-brotli",),
    write_method: str = "to_parquet",
) -> list:
    res = []
    file_format = "parquet"
    if write_method == "write_parquet":
        engines = ("main",)
    for engine in tqdmu(engines, desc=f"{file_format} engine", leave=False):

        if write_method == "write_parquet":
            engine_params = {}
        else:
            engine_params = dict(engine=engine)

        for compr in tqdmu("snappy gzip brotli lz4 zstd".split(), desc=f"{file_format} compression method", leave=False):

            if f"{file_format}-{engine}-{compr}" in skip_configs:
                continue

            config = f"{file_format}-{engine}-{compr}"
            fname = join(temp_folder, rf"{config}.{file_format}")
            read_times, write_times, read_sizes, write_sizes = measure_read_write_performance(
                df=df,
                fname=fname,
                read_method="read_parquet",
                read_params=dict(**engine_params),
                write_method=write_method,
                write_params=dict(**engine_params, compression=compr),
                nrepeats=nrepeats,
            )

            pack_benchmark_results(res, config, read_times, write_times, read_sizes, write_sizes)

            for compression_level in range(1, max_compression_level + 1):
                config = f"{file_format}-{engine}-{compr}-{compression_level}"

                fname = join(temp_folder, rf"{config}.{file_format}")
                try:
                    read_times, write_times, read_sizes, write_sizes = measure_read_write_performance(
                        df=df,
                        fname=fname,
                        read_method="read_parquet",
                        read_params=dict(**engine_params),
                        write_method=write_method,
                        write_params=dict(**engine_params, compression=compr, compression_level=compression_level),
                        nrepeats=nrepeats,
                    )

                    pack_benchmark_results(res, config, read_times, write_times, read_sizes, write_sizes)
                except Exception as e:
                    logger.warning(f"Skipping config {config}")

    return pd.DataFrame(
        res,
        columns=[
            "config",
        ]
        + "mean_read_times,std_read_times,mean_write_times,std_write_times,mean_read_sizes,std_read_sizes,mean_write_sizes,std_write_sizes".split(","),
    )


def benchmark_dataframe_pickle_compression(res, temp_folder, df, nrepeats):
    file_format = "pickle"

    # for level in tqdmu(range(1, 10), desc=f"{file_format} engine", leave=False):
    for compr in tqdmu(["zip", "gzip", "bz2", "zstd", "xz", "tar"], desc=f"{file_format} compression method", leave=False):
        config = f"{file_format}-{compr}"  # -{level}

        fname = join(temp_folder, rf"{config}.{file_format}.{compr}")
        read_times, write_times, read_sizes, write_sizes = measure_read_write_performance(
            df=df,
            fname=fname,
            read_method="read_pickle",
            read_params=dict(compression={"method": compr}),
            write_method="to_pickle",
            write_params=dict(compression={"method": compr}, protocol=-1),  # "compresslevel": level
            nrepeats=nrepeats,
        )

        pack_benchmark_results(res, config, read_times, write_times, read_sizes, write_sizes)


def benchmark_dataframe_hdf_compression(res, temp_folder, df, nrepeats):
    file_format = "hdf"

    for level in tqdmu(range(1, 10), desc=f"{file_format} engine", leave=False):
        for compr in tqdmu("zlib lzo bzip2 blosc".split(), desc=f"{file_format} compression method", leave=False):
            config = f"{file_format}-{compr}"  # -{level}

            fname = join(temp_folder, rf"{config}.{file_format}.{compr}")
            read_times, write_times, read_sizes, write_sizes = measure_read_write_performance(
                df=df,
                fname=fname,
                read_method="read_hdf",
                read_params=dict(complib=compr),
                write_method="to_hdf",
                write_params=dict(complib=compr, complevel=level, key="test"),
                nrepeats=nrepeats,
            )

            pack_benchmark_results(res, config, read_times, write_times, read_sizes, write_sizes)


def benchmark_dataframe_csv_compression(res, temp_folder, df, nrepeats):
    file_format = "csv"

    for compr in tqdmu(["zip", "gzip", "bz2", "zstd", "xz", "tar"], desc=f"{file_format} compression method", leave=False):
        config = f"{file_format}-{compr}"

        fname = join(temp_folder, rf"{config}.{file_format}.{compr}")
        read_times, write_times, read_sizes, write_sizes = measure_read_write_performance(
            df=df,
            fname=fname,
            read_method="read_csv",
            read_params=dict(compression={"method": compr}),
            write_method="to_csv",
            write_params=dict(compression={"method": compr}),
            nrepeats=nrepeats,
        )

        pack_benchmark_results(res, config, read_times, write_times, read_sizes, write_sizes)


def benchmark_dataframe_orc_compression(res, temp_folder, df, nrepeats):
    file_format = "orc"

    config = f"{file_format}"

    fname = join(temp_folder, rf"{config}.{file_format}")
    read_times, write_times, read_sizes, write_sizes = measure_read_write_performance(
        df=df,
        fname=fname,
        read_method="read_orc",
        read_params=dict(),
        write_method="to_orc",
        write_params=dict(),
        nrepeats=nrepeats,
    )

    pack_benchmark_results(res, config, read_times, write_times, read_sizes, write_sizes)


def benchmark_dataframe_feather_compression(res, temp_folder, df, nrepeats):
    file_format = "feather"

    config = f"{file_format}"

    fname = join(temp_folder, rf"{config}.{file_format}")
    read_times, write_times, read_sizes, write_sizes = measure_read_write_performance(
        df=df,
        fname=fname,
        read_method="read_feather",
        read_params=dict(),
        write_method="to_feather",
        write_params=dict(),
        nrepeats=nrepeats,
    )

    pack_benchmark_results(res, config, read_times, write_times, read_sizes, write_sizes)


def benchmark_dataframe_compression(
    df: pd.DataFrame,
    head: int = 100_000,
    benchmark_dir_path=None,
    nrepeats: int = 10,
    sort_by="mean_write_size",
    return_styled: bool = True,
    should_clean_temp_folder: bool = True,
    verbose: bool = True,
):
    """Tries various formats & compressiom methods on a part of your dataframe, reports write, read data size & durations."""
    warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    if head:
        df = df.head(head).reset_index(drop=True)

    df_size = get_df_memory_consumption(df) / (1024**2)
    if verbose:
        logger.info(f"Pandas: {pd.__version__}, DF size: {df_size:_.2f}Mb, Dtypes: {df.dtypes.value_counts().to_dict()}")

    if benchmark_dir_path:
        ensure_dir_exists(benchmark_dir_path)
    temp_folder = tempfile.mkdtemp(dir=benchmark_dir_path)

    res = []

    for func in (
        benchmark_dataframe_feather_compression,
        benchmark_dataframe_orc_compression,
        benchmark_dataframe_hdf_compression,
        benchmark_dataframe_parquet_compression,
        benchmark_dataframe_pickle_compression,
        benchmark_dataframe_csv_compression,
    ):
        try:
            func(res, temp_folder, df, nrepeats)
        except Exception as e:
            logger.error(e)

    if should_clean_temp_folder:
        shutil.rmtree(temp_folder)

    res = (
        pd.DataFrame(res, columns=["config"] + list(chain(*[("mean_" + arr, "std_" + arr) for arr in "read_time write_time read_size write_size".split()])))
        .set_index("config")
        .sort_values(sort_by, ascending=True)
    )

    remove_constant_columns(res)
    if return_styled:
        try:
            res = res.style.background_gradient(axis=None, subset=["mean_write_size", "mean_write_time", "mean_read_time"])
        except Exception as e:
            logger.exception(e)

    return res


def ensure_dataframe_float32_convertability(df: pd.DataFrame) -> None:
    """Lightgbm uses np.result_type(*df_dtypes) to detect array dtype when converting from Pandas input,
    which results in float64 for int32 and above. For the rational mem usage, it makes sense to convert cols to float32 directly before training lightgbm."""
    for precise_dtype in "uint32 int32 int64 uint64 float64".split():

        tmp = df.select_dtypes(precise_dtype)
        if tmp.shape[1] > 0:
            logger.info(f"Converting {tmp.shape[1]:_} {precise_dtype} columns to float32")
            df[tmp.columns] = tmp.astype(np.float32)


def convert_float64_to_float32(df: pd.DataFrame) -> None:
    for col in df.head().select_dtypes("float64"):
        df[col] = df[col].astype(np.float32)
